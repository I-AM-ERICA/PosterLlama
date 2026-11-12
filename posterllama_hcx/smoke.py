"""Offline structural demo: real tiny SigLIP/Llama/PEFT with RANDOM weights.

This is not a pretrained HyperCLOVA/CodeLlama model and cannot make useful layouts.
"""
import json
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import LlamaConfig, LlamaForCausalLM, SiglipVisionConfig, SiglipVisionModel

from .config import ModelConfig, TrainConfig
from .data import PosterDataset, build_layout_prompt
from .vision import FrozenVision, ImageProcessor


class ByteTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 2
    eos_token = "</s>"

    def encode(self, text, add_special_tokens=False):
        ids = [byte + 3 for byte in text.encode("utf-8")]
        return [self.bos_token_id] + ids if add_special_tokens else ids

    def batch_decode(self, batch, skip_special_tokens=True):
        return [bytes(int(i) - 3 for i in row if 3 <= int(i) <= 258).decode("utf-8", errors="replace") for row in batch]


def make_smoke_model(config=None):
    from .model import PosterLlama
    config = replace(config or ModelConfig(), smoke=True, load_in_4bit=False)
    # Recreate identical frozen weights on checkpoint reload without saving a base model.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        tower = SiglipVisionModel(SiglipVisionConfig(hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, image_size=28, patch_size=7))
        llm = LlamaForCausalLM(LlamaConfig(vocab_size=259, hidden_size=48, intermediate_size=96,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=4096, bos_token_id=1, eos_token_id=2, pad_token_id=2))
        model = PosterLlama(FrozenVision(tower, config.vision_feature_layer), llm,
                           ByteTokenizer(), ImageProcessor(28), config)
    return model


def write_demo_data(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (512, 360), "#e9edf3")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((300, 110, 460, 320), radius=20, fill="#254568")
    draw.rectangle((315, 145, 445, 200), fill="#82b4d1")
    image.save(root / "product.png")
    html = ('<html><body><svg width="512" height="360">'
            '<rect data-category="text" x="24" y="30" width="250" height="60" />'
            '<rect data-category="text" x="24" y="108" width="230" height="45" />'
            '<rect data-category="logo" x="24" y="292" width="96" height="40" />'
            '</svg></body></html>')
    caption = {"image": "product.png", "caption": "A blue product on the right, with empty space on the left."}
    layout = {"image": "product.png", "prompt": build_layout_prompt(["NEW PRODUCT", "20% OFF"]), "html": html}
    for name, record in [("stage1.jsonl", caption), ("stage2.jsonl", layout)]:
        (root / name).write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return root


def run_smoke_test(directory="outputs/smoke"):
    from .model import PosterLlama
    from .training import train_stage
    root = Path(directory)
    data = write_demo_data(root / "data")
    model = make_smoke_model(ModelConfig(smoke=True, lora_r=4, gradient_checkpointing=True))
    training = TrainConfig(max_steps=1, gradient_accumulation_steps=1)
    stage1 = train_stage(model, PosterDataset(data / "stage1.jsonl", 1), training, root / "stage1")
    model.start_stage2()
    stage2 = train_stage(model, PosterDataset(data / "stage2.jsonl", 2), training, root / "stage2")
    restored = PosterLlama.from_checkpoint(root / "stage2")
    pixels = restored.image_processor([Image.open(data / "product.png").convert("RGB")])
    generated = restored.generate(pixels, ["Generate a layout."], max_new_tokens=8)
    return {"stage1_loss": stage1[-1]["loss"], "stage2_loss": stage2[-1]["loss"],
            "parameters": restored.parameter_summary(), "random_output": generated[0],
            "checkpoint": str(root / "stage2")}
