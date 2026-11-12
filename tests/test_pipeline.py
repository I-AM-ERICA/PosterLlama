import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from transformers import SiglipVisionConfig, SiglipVisionModel

from posterllama_hcx import ModelConfig, PosterDataset, PosterLlama, TrainConfig, parse_layout, render_layout
from posterllama_hcx.data import ImageCollator
from posterllama_hcx.smoke import make_smoke_model, write_demo_data
from posterllama_hcx.training import train_stage
from posterllama_hcx.vision import PatchAdapter, load_hcx_tower


@pytest.fixture
def model():
    torch.set_num_threads(2)
    return make_smoke_model(ModelConfig(smoke=True, lora_r=4, gradient_checkpointing=True))


def snapshot(module):
    return {k: p.detach().clone() for k, p in module.named_parameters()}


def unchanged(module, before):
    return all(torch.equal(before[k], p) for k, p in module.named_parameters())


def test_two_stages_and_round_trip(tmp_path, model):
    data = write_demo_data(tmp_path / "data")
    train_cfg = TrainConfig(gradient_accumulation_steps=2, max_steps=1)
    vision_before, llm_before = snapshot(model.vision), snapshot(model.llm)
    adapter_before = snapshot(model.adapter)
    history1 = train_stage(model, PosterDataset(data / "stage1.jsonl", 1), train_cfg, tmp_path / "stage1")
    assert history1[-1]["loss"] > 0
    assert unchanged(model.vision, vision_before) and unchanged(model.llm, llm_before)
    assert not unchanged(model.adapter, adapter_before)
    assert all(p.grad is None for p in model.vision.parameters())

    restored1 = PosterLlama.from_checkpoint(tmp_path / "stage1")
    assert unchanged(restored1.adapter, snapshot(model.adapter))
    fresh = make_smoke_model(model.config)
    fresh.load_stage1_adapter(tmp_path / "stage1")
    assert unchanged(fresh.adapter, snapshot(model.adapter))

    model.start_stage2()
    adapter_before, base_before = snapshot(model.adapter), snapshot(model.llm)
    history2 = train_stage(model, PosterDataset(data / "stage2.jsonl", 2), train_cfg, tmp_path / "stage2")
    assert history2[-1]["loss"] > 0
    assert unchanged(model.adapter, adapter_before) and unchanged(model.vision, vision_before)
    changed = [name for name, p in model.llm.named_parameters() if not torch.equal(p, base_before[name])]
    assert changed and all("lora_" in name for name in changed)
    assert all("lora_" in name for name in model.assert_trainable_parameters())

    restored = PosterLlama.from_checkpoint(tmp_path / "stage2")
    batch = ImageCollator(model.image_processor)([PosterDataset(data / "stage2.jsonl", 2)[0]])
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(**batch).logits, restored(**batch).logits, atol=1e-6)
    generated = restored.generate(batch["pixel_values"].repeat(2, 1, 1, 1),
                                 ["Layout.", "A longer layout instruction."], max_new_tokens=4)
    assert len(generated) == 2
    assert not restored.training


def test_target_mask_padding_eos_and_gradients(model):
    pixels = torch.randn(2, 3, 28, 28)
    batch = model.prepare_batch(pixels, ["Short.", "A longer instruction."], ["X", "XYZ"])
    labels = batch["labels"]
    for row, target in zip(labels, ["X", "XYZ"]):
        assert row[row != -100].tolist() == model.tokenizer.encode(target) + [model.tokenizer.eos_token_id]
    assert (labels[batch["attention_mask"] == 0] == -100).all()
    # EOS == PAD is deliberate: real EOS must still contribute to the loss.
    assert (labels == model.tokenizer.eos_token_id).sum() == 2
    model.llm(**batch, use_cache=False).loss.backward()
    assert model.adapter.projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.llm.parameters())
    assert not model.vision.training
    assert model.llm.training and model.llm.is_gradient_checkpointing
    generated_batch = model.prepare_batch(pixels, ["Short.", "A longer instruction."])
    assert generated_batch["attention_mask"][0, 0] == 0
    assert generated_batch["attention_mask"][:, -1].all()


def test_adapter_keeps_all_729_patches():
    adapter = PatchAdapter(1, 4, group_size=4)
    with torch.no_grad():
        adapter.projection.weight.copy_(torch.eye(4))
        adapter.projection.bias.zero_()
    patches = torch.arange(729.0).reshape(1, 729, 1)
    output = adapter(patches)
    assert output.shape == (1, 183, 4)
    assert output[0, -1].tolist() == [728, 0, 0, 0]


def test_official_weight_prefix_loading(tmp_path):
    tower = SiglipVisionModel(SiglipVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=7))
    state = {"vision_model." + k: v for k, v in tower.state_dict().items()}
    state["language_model.unused"] = torch.ones(1)
    save_file(state, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": tower.config.to_dict()}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in state}}))
    loaded, processor = load_hcx_tower(str(tmp_path))
    assert unchanged(loaded.tower, snapshot(tower))
    assert processor.size == 28
    assert loaded(torch.randn(1, 3, 28, 28)).shape == (1, 16, 16)


def test_dataset_and_render(tmp_path):
    data = write_demo_data(tmp_path)
    dataset = PosterDataset(data / "stage2.jsonl", 2)
    record = dataset[0]
    layout = parse_layout(record["target"])
    assert len(layout.boxes) == 3
    assert render_layout(record["image"], record["target"]).size == (512, 360)
    paper = {"name": ["product.jpg"], "unconditional_seq_modeling": "Generate layout.<MID>" + record["target"]}
    (tmp_path / "paper.jsonl").write_text(json.dumps(paper) + "\n")
    paper_record = PosterDataset(tmp_path / "paper.jsonl", 2)[0]
    assert paper_record["prompt"] == "Generate layout." and paper_record["target"] == record["target"]
    with pytest.raises(ValueError, match="beyond"):
        parse_layout('<svg width="10" height="10"><rect x="9" y="0" width="5" height="1" /></svg>')
    with pytest.raises(ValueError, match="complete"):
        parse_layout('<svg width="10" height="10"><rect x="0" y="0" width="5" height="1" />')


def test_reject_untrained_stage2_and_truncation(model):
    with pytest.raises(RuntimeError, match="stage 1"):
        model.start_stage2()
    model.config.max_target_tokens = 3
    with pytest.raises(ValueError, match="never silently truncated"):
        model.prepare_batch(torch.randn(1, 3, 28, 28), ["Hi"], ["Long HTML"])


def test_accumulation_matches_combined_batch(tmp_path):
    config = ModelConfig(smoke=True, lora_r=4, gradient_checkpointing=False)
    one, two = make_smoke_model(config), make_smoke_model(config)
    data = write_demo_data(tmp_path / "data")
    caption = json.loads((data / "stage1.jsonl").read_text())
    (data / "stage1.jsonl").write_text("\n".join(json.dumps({**caption, "caption": text})
        for text in ["Short.", "Longer caption.", "Last partial accumulation."]) + "\n")
    dataset = PosterDataset(data / "stage1.jsonl", 1)
    cfg = TrainConfig(batch_size=1, gradient_accumulation_steps=2, max_steps=2)
    train_stage(one, dataset, cfg, tmp_path / "one")
    train_stage(two, dataset, replace(cfg, batch_size=2, gradient_accumulation_steps=1), tmp_path / "two")
    for p, q in zip(one.adapter.parameters(), two.adapter.parameters()):
        assert torch.allclose(p, q, atol=2e-6)
