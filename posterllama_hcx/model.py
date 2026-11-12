import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ModelConfig
from .vision import PatchAdapter, load_hcx_tower


class PosterLlama(nn.Module):
    def __init__(self, vision, llm, tokenizer, image_processor, config):
        super().__init__()
        self.config = config
        self.vision = vision
        self.llm = llm
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        embedding = llm.get_input_embeddings().weight
        self.adapter = PatchAdapter(vision.hidden_size, embedding.shape[1], config.patch_group_size)
        # Keep trainable adapter in float32, especially when LLM uses fp16.
        self.adapter.to(device=embedding.device)
        self.stage = 1
        self.adapter_ready = False
        self.vision.requires_grad_(False)
        self.llm.requires_grad_(False)
        if tokenizer.eos_token_id is None or tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer must define BOS and EOS tokens")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.llm.config.use_cache = False
        if config.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.train()

    @classmethod
    def from_config(cls, config):
        if config.smoke:
            from .smoke import make_smoke_model
            return make_smoke_model(config)
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        if config.dtype == "auto":
            dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        else:
            dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                     "float16": torch.float16}[config.dtype]
        if config.load_in_4bit and not device.startswith("cuda"):
            raise ValueError("4-bit training requires CUDA + bitsandbytes; use smoke=True on CPU/Mac")
        vision, processor = load_hcx_tower(config.vision_model, config.vision_revision, dtype)
        vision.feature_layer = config.vision_feature_layer
        vision.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(config.language_model, revision=config.language_revision)
        kwargs = dict(revision=config.language_revision, torch_dtype=dtype, attn_implementation="eager")
        if config.load_in_4bit:
            kwargs.update(
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
                device_map={"": device})
        llm = AutoModelForCausalLM.from_pretrained(config.language_model, **kwargs)
        if config.load_in_4bit:
            llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=False)
        else:
            llm.to(device=device)
        return cls(vision, llm, tokenizer, processor, config)

    @property
    def device(self):
        return self.llm.get_input_embeddings().weight.device

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        # HF activates gradient checkpointing only while the decoder is in train
        # mode, including stage 1 where gradients still flow to visual inputs.
        self.llm.train(mode and (self.stage == 2 or self.config.gradient_checkpointing))
        self.adapter.train(mode and self.stage == 1)
        return self

    def start_stage2(self):
        if self.stage == 2:
            return self
        if not self.adapter_ready:
            raise RuntimeError("Train stage 1 or load its adapter checkpoint before stage 2")
        self.adapter.requires_grad_(False)
        self.llm = get_peft_model(self.llm, LoraConfig(
            task_type="CAUSAL_LM", r=self.config.lora_r, lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout, target_modules=list(self.config.lora_target_modules),
            bias="none"))
        self.stage = 2
        self.train()
        self.assert_trainable_parameters()
        return self

    def assert_trainable_parameters(self):
        names = [name for name, p in self.named_parameters() if p.requires_grad]
        valid = (lambda name: name.startswith("adapter.")) if self.stage == 1 else (
            lambda name: name.startswith("llm.") and "lora_" in name)
        if not names or not all(valid(name) for name in names):
            raise RuntimeError(f"Unexpected trainable parameters at stage {self.stage}: {names}")
        return names

    def parameter_summary(self):
        return {name: {"total": sum(p.numel() for p in module.parameters()),
                       "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
                for name, module in [("vision", self.vision), ("adapter", self.adapter), ("llm", self.llm)]}

    def _ids(self, text, limit, label):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > limit:
            raise ValueError(f"{label} has {len(ids)} tokens, limit={limit}. Increase the limit; HTML is never silently truncated.")
        return ids

    def _embed(self, ids):
        return self.llm.get_input_embeddings()(torch.tensor(ids, device=self.device, dtype=torch.long))

    def prepare_batch(self, pixel_values, prompts, targets=None):
        if len(prompts) != pixel_values.shape[0] or not prompts:
            raise ValueError("Image and prompt batch sizes must match and be nonempty")
        if targets is not None and len(targets) != len(prompts):
            raise ValueError("Target and prompt batch sizes must match")
        visual = self.adapter(self.vision(pixel_values)).to(self.llm.get_input_embeddings().weight.dtype)
        sequences, labels = [], []
        for i, prompt in enumerate(prompts):
            before = self._ids("[INST] <Img>", self.config.max_prompt_tokens, "Prompt")
            after = self._ids("</Img> " + prompt + " [/INST] ", self.config.max_prompt_tokens, "Prompt")
            if len(before) + len(after) > self.config.max_prompt_tokens:
                raise ValueError("Prompt including wrapper exceeds max_prompt_tokens")
            prefix = torch.cat((self._embed([self.tokenizer.bos_token_id] + before), visual[i], self._embed(after)))
            target_ids = []
            if targets is not None:
                if not targets[i].strip():
                    raise ValueError("Empty training target")
                target_ids = self._ids(targets[i], self.config.max_target_tokens - 1, "Target") + [self.tokenizer.eos_token_id]
            sequences.append(torch.cat((prefix, self._embed(target_ids))))
            labels.append(torch.tensor([-100] * len(prefix) + target_ids, device=self.device))
        longest = max(len(x) for x in sequences)
        context_limit = getattr(self.llm.config, "max_position_embeddings", None)
        if context_limit is not None and longest > context_limit:
            raise ValueError(f"Multimodal sequence {longest} exceeds context window {context_limit}")
        # Right padding for training, left padding for decoder-only generation.
        embeddings, attention, padded_labels = [], [], []
        for seq, label in zip(sequences, labels):
            missing = longest - len(seq)
            pad = seq.new_zeros((missing, seq.shape[-1]))
            zeros = torch.zeros(missing, device=self.device, dtype=torch.long)
            ones = torch.ones(len(seq), device=self.device, dtype=torch.long)
            ignored = torch.full((missing,), -100, device=self.device, dtype=torch.long)
            if targets is None:
                embeddings.append(torch.cat((pad, seq)))
                attention.append(torch.cat((zeros, ones)))
                padded_labels.append(torch.cat((ignored, label)))
            else:
                embeddings.append(torch.cat((seq, pad)))
                attention.append(torch.cat((ones, zeros)))
                padded_labels.append(torch.cat((label, ignored)))
        mask = torch.stack(attention)
        batch = dict(inputs_embeds=torch.stack(embeddings), attention_mask=mask,
                     position_ids=(mask.cumsum(-1) - 1).clamp_min(0))
        if targets is not None:
            batch["labels"] = torch.stack(padded_labels).long()
        return batch

    def forward(self, pixel_values, prompts, targets):
        # Do NOT put the frozen LLM under no_grad in stage 1: gradients must reach the adapter.
        return self.llm(**self.prepare_batch(pixel_values, prompts, targets), use_cache=False, return_dict=True)

    @torch.no_grad()
    def generate(self, pixel_values, prompts, max_new_tokens=512):
        was_training = self.training
        self.eval()
        try:
            batch = self.prepare_batch(pixel_values, prompts)
            # generate() derives and updates position_ids from the left-padded mask.
            batch.pop("position_ids")
            limit = getattr(self.llm.config, "max_position_embeddings", None)
            if max_new_tokens < 1 or (limit and batch["inputs_embeds"].shape[1] + max_new_tokens > limit):
                raise ValueError("Invalid max_new_tokens or generation exceeds model context window")
            output = self.llm.generate(**batch, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, bos_token_id=self.tokenizer.bos_token_id)
            # With inputs_embeds and no input_ids, HF returns generated token IDs only.
            return self.tokenizer.batch_decode(output, skip_special_tokens=True)
        finally:
            self.train(was_training)

    def save_checkpoint(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.adapter.state_dict().items()},
                  str(path / "adapter.safetensors"))
        metadata = {"format_version": 1, "stage": self.stage, "adapter_ready": self.adapter_ready,
                    "model_config": self.config.to_dict()}
        (path / "posterllama.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.stage == 2:
            self.llm.save_pretrained(path / "lora", safe_serialization=True)

    def load_stage1_adapter(self, directory):
        if self.stage != 1:
            raise RuntimeError("Load stage 1 adapter before attaching LoRA")
        path = Path(directory)
        metadata = json.loads((path / "posterllama.json").read_text())
        if metadata["stage"] != 1 or not metadata["adapter_ready"]:
            raise ValueError("A trained stage 1 checkpoint is required")
        for key in ("vision_model", "vision_revision", "language_model", "language_revision",
                    "patch_group_size", "vision_feature_layer", "smoke"):
            if metadata["model_config"][key] != self.config.to_dict()[key]:
                raise ValueError(f"Stage 1 checkpoint configuration mismatch: {key}")
        self.adapter.load_state_dict(load_file(str(path / "adapter.safetensors")), strict=True)
        self.adapter_ready = True

    @classmethod
    def from_checkpoint(cls, directory, **runtime_overrides):
        path = Path(directory)
        metadata = json.loads((path / "posterllama.json").read_text())
        allowed = {"device", "dtype", "load_in_4bit", "gradient_checkpointing"}
        if set(runtime_overrides) - allowed:
            raise ValueError(f"Only runtime overrides are allowed: {allowed}")
        config = ModelConfig(**{**metadata["model_config"], **runtime_overrides})
        model = cls.from_config(config)
        model.adapter.load_state_dict(load_file(str(path / "adapter.safetensors")), strict=True)
        model.adapter_ready = metadata["adapter_ready"]
        if metadata["stage"] == 2:
            model.adapter.requires_grad_(False)
            model.llm = PeftModel.from_pretrained(model.llm, path / "lora", is_trainable=True)
            model.stage = 2
        model.assert_trainable_parameters()
        model.eval()
        return model
