"""Load only HCX's SigLIP tower, without instantiating its language model.

Reference: the official HCX config.json and modeling_hyperclovax.py.
The original AnyRes/C-Abstractor is intentionally replaced by PosterLlama's adapter.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel


def model_file(model_id, filename, revision):
    local = Path(model_id).expanduser()
    if local.is_dir():
        path = local / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    return Path(hf_hub_download(model_id, filename, revision=revision))


class ImageProcessor:
    """One square view with letterboxing; preserve the full poster and aspect ratio."""

    def __init__(self, size=378):
        self.size = size

    def __call__(self, images):
        values = []
        for image in images:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.pad(image, (self.size, self.size),
                                 method=Image.Resampling.BICUBIC, color=(127, 127, 127))
            # HCX's published image_mean/image_std are both [0.5, 0.5, 0.5].
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            values.append(torch.from_numpy(array).permute(2, 0, 1))
        if not values:
            raise ValueError("At least one image is required")
        return torch.stack(values)


def load_hcx_tower(model_id, revision="main", dtype=torch.float32):
    raw = json.loads(model_file(model_id, "config.json", revision).read_text())
    config = raw.get("vision_config", raw)
    if config.get("model_type") != "siglip_vision_model":
        raise ValueError("Expected HCX SEED's SiglipVisionModel or an exported SigLIP tower")
    # Native Transformers implementation; no trust_remote_code execution required.
    tower = SiglipVisionModel(SiglipVisionConfig.from_dict(config)).to(dtype=dtype)
    if "vision_config" in raw:
        index = json.loads(model_file(model_id, "model.safetensors.index.json", revision).read_text())
        prefix = "vision_model."
        selected = {k: v for k, v in index["weight_map"].items() if k.startswith(prefix)}
        if not selected:
            raise ValueError("Checkpoint index contains no vision_model weights")
        state = {}
        for shard in sorted(set(selected.values())):
            # A full shard is downloaded (currently ~5 GB), but only vision tensors
            # are materialized in RAM. The other two language shards are not needed.
            with safe_open(model_file(model_id, shard, revision), framework="pt", device="cpu") as f:
                for key, filename in selected.items():
                    if filename == shard:
                        state[key[len(prefix):]] = f.get_tensor(key)
    else:
        with safe_open(model_file(model_id, "model.safetensors", revision), framework="pt", device="cpu") as f:
            state = {key: f.get_tensor(key) for key in f.keys()}
    tower.load_state_dict(state, strict=True)
    return FrozenVision(tower), ImageProcessor(config["image_size"])


class FrozenVision(nn.Module):
    def __init__(self, tower, feature_layer=-2):
        super().__init__()
        self.tower = tower
        self.feature_layer = feature_layer
        self.hidden_size = tower.config.hidden_size
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Keep the frozen encoder deterministic even when parent.train() is called.
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixels):
        parameter = next(self.tower.parameters())
        output = self.tower(pixels.to(device=parameter.device, dtype=parameter.dtype),
                            output_hidden_states=True, return_dict=True)
        # SigLIP has NO CLS token; all 729 patches must be retained.
        return output.hidden_states[self.feature_layer]


class PatchAdapter(nn.Module):
    def __init__(self, vision_dim, text_dim, group_size=4):
        super().__init__()
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.group_size = group_size
        self.projection = nn.Linear(vision_dim * group_size, text_dim)

    def forward(self, patches):
        batch, count, dim = patches.shape
        # 729 % 4 != 0: pad three zero patches instead of losing image content.
        missing = (-count) % self.group_size
        if missing:
            patches = torch.cat((patches, patches.new_zeros(batch, missing, dim)), dim=1)
        patches = patches.reshape(batch, -1, dim * self.group_size)
        return self.projection(patches.to(self.projection.weight.dtype))
