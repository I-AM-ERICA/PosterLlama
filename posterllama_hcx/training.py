import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from .data import ImageCollator


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataset, batch_size=1):
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=ImageCollator(model.image_processor))
    was_training = model.training
    model.eval()
    loss_sum, token_count = 0.0, 0
    try:
        for batch in loader:
            packed = model.prepare_batch(**batch)
            output = model.llm(**packed, use_cache=False, return_dict=True)
            tokens = (packed["labels"][:, 1:] != -100).sum().item()
            loss_sum += output.loss.item() * tokens
            token_count += tokens
    finally:
        model.train(was_training)
    return loss_sum / token_count


def train_stage(model, dataset, config, output_dir, validation=None):
    """Single-device notebook training. Checkpoints contain adapter/LoRA, not optimizer state."""
    if getattr(dataset, "stage", model.stage) != model.stage:
        raise ValueError("Dataset stage does not match model stage")
    if validation is not None and getattr(validation, "stage", model.stage) != model.stage:
        raise ValueError("Validation stage does not match model stage")
    model.assert_trainable_parameters()
    seed_everything(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=ImageCollator(model.image_processor),
        generator=torch.Generator().manual_seed(config.seed))
    if not len(loader):
        raise ValueError("Training dataset is empty")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    updates = math.ceil(len(loader) / config.gradient_accumulation_steps) * config.epochs
    total_steps = min(updates, config.max_steps) if config.max_steps else updates
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio), num_training_steps=total_steps)
    # Float32 adapter / LoRA master weights, optional fp16 frozen LLM.
    use_scaler = model.device.type == "cuda" and model.llm.get_input_embeddings().weight.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    history, global_step = [], 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=total_steps, desc=f"Stage {model.stage}")
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(config.epochs):
            model.train()
            iterator = iter(loader)
            for start in range(0, len(loader), config.gradient_accumulation_steps):
                # Normalize by supervised tokens across this actual accumulation group,
                # including the last, shorter group and unequal target lengths.
                batches = [next(iterator) for _ in range(min(config.gradient_accumulation_steps, len(loader) - start))]
                counts = [sum(len(model.tokenizer.encode(t, add_special_tokens=False)) + 1 for t in b["targets"]) for b in batches]
                group_loss = 0.0
                for batch, tokens in zip(batches, counts):
                    loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite stage {model.stage} loss")
                    weight = tokens / sum(counts)
                    scaler.scale(loss * weight).backward()
                    group_loss += loss.detach().item() * weight
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if model.stage == 1:
                    model.adapter_ready = True
                entry = {"epoch": epoch + 1, "step": global_step, "loss": group_loss,
                         "learning_rate": scheduler.get_last_lr()[0]}
                history.append(entry)
                progress.update(1)
                progress.set_postfix(loss=f"{group_loss:.4f}")
                if global_step >= total_steps:
                    break
            if validation is not None:
                history[-1]["validation_loss"] = evaluate(model, validation, config.batch_size)
            model.save_checkpoint(output_dir)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if global_step >= total_steps:
                break
    finally:
        progress.close()
        optimizer.zero_grad(set_to_none=True)
    return history
