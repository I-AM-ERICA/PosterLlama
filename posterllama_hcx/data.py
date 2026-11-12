import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

CAPTION_PROMPT = "Describe this image in detail."
PAPER_TASKS = {
    "cond_cate": "cond_cate_to_size_pos_seq_modeling",
    "cond_cate_size": "cond_cate_size_to_pos_seq_modeling",
    "cond_cate_pos": "cond_cate_pos_to_size_seq_modeling",
    "unconditional": "unconditional_seq_modeling",
    "recover": "cond_recover_mask_seq_modeling",
    "completion": "completion_seq_modeling",
    "refinement": "refinement_seq_modeling",
}


def build_layout_prompt(texts, width=512, height=360, task="Generate a content-aware poster layout."):
    if width <= 0 or height <= 0:
        raise ValueError("Canvas dimensions must be positive")
    return (
        "I want to generate layout in Poster style. " + task + "\n"
        f"Canvas: width={width}, height={height}.\n"
        "Text constraints: " + json.dumps(list(texts), ensure_ascii=False) + "\n"
        "Place text and logo boxes without covering the main product. "
        "Return only HTML containing an SVG. Coordinates are pixels in the original canvas.\n"
        f'<html><body><svg width="{width}" height="{height}">\n'
        '<rect data-category="text" x="<X>" y="<Y>" width="<W>" height="<H>" />\n'
        '<rect data-category="logo" x="<X>" y="<Y>" width="<W>" height="<H>" />\n'
        '</svg></body></html>'
    )


class PosterDataset(Dataset):
    """JSON/JSONL records, plus the original PosterLlama stage-2 <MID> schema.

    Stage 1: image, caption[, prompt]. Stage 2: image, prompt, html.
    Paths are relative to image_root (default: the manifest's parent).
    """

    def __init__(self, manifest, stage, image_root=None, task="unconditional", augment=False):
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        path = Path(manifest).expanduser()
        raw = path.read_text(encoding="utf-8")
        self.records = json.loads(raw) if path.suffix == ".json" else [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not isinstance(self.records, list) or not self.records:
            raise ValueError("Manifest must contain a nonempty list of records")
        if task not in PAPER_TASKS and task != "random":
            raise ValueError(f"Unknown paper task: {task}")
        self.image_root = Path(image_root).expanduser() if image_root else path.parent
        self.stage, self.task, self.augment = stage, task, augment

    def __len__(self):
        return len(self.records)

    def _image_path(self, record):
        name = record.get("image", record.get("name"))
        if isinstance(name, list):
            name = random.choice(name) if self.augment and name else (name[0] if name else None)
        if not isinstance(name, str) or not name:
            raise ValueError("Record needs an image path ('image' or paper 'name')")
        path = self.image_root / name
        if path.is_file():
            return path
        if "name" in record:
            for suffix in (".png", ".jpg", ".jpeg"):
                alternative = path.with_suffix(suffix)
                if alternative.is_file():
                    return alternative
        raise FileNotFoundError(f"Image not found: {path}")

    def __getitem__(self, index):
        record = self.records[index]
        if self.stage == 1:
            prompt = record.get("prompt", CAPTION_PROMPT)
            target = record.get("caption", record.get("labels"))
        elif "html" in record or "labels" in record:
            prompt = record.get("prompt", record.get("input"))
            target = record.get("html", record.get("labels"))
        else:
            task = self.task
            if task == "random":
                available = [k for k, value in PAPER_TASKS.items() if value in record]
                if not available:
                    raise ValueError("No original PosterLlama tasks in this record")
                task = random.choices(available, weights=[1 if k.startswith("cond_") else 2 for k in available])[0]
            combined = record.get(PAPER_TASKS[task], "")
            if "<MID>" not in combined:
                raise ValueError(f"Missing <MID> delimiter for task {task}, record {index}")
            prompt, target = combined.split("<MID>", 1)
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(target, str) or not target.strip():
            raise ValueError(f"Record {index} needs nonempty prompt and target text")
        with Image.open(self._image_path(record)) as image:
            image = image.convert("RGB")
        return {"image": image, "prompt": prompt.strip(), "target": target.strip()}


class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, records):
        return {"pixel_values": self.processor([r["image"] for r in records]),
                "prompts": [r["prompt"] for r in records], "targets": [r["target"] for r in records]}
