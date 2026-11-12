from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vision_model: str = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
    vision_revision: str = "main"
    language_model: str = "codellama/CodeLlama-7b-hf"
    language_revision: str = "main"
    # Sequential groups of four patches, as in the original MiniGPT4.encode_img.
    patch_group_size: int = 4
    vision_feature_layer: int = -2
    max_prompt_tokens: int = 1024
    max_target_tokens: int = 1024
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj")
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    smoke: bool = False
    seed: int = 42

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    max_steps: int | None = None
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.gradient_accumulation_steps) < 1:
            raise ValueError("epochs, batch_size, gradient_accumulation_steps must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or None")
        if not 0 <= self.warmup_ratio < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid warmup_ratio or learning_rate")
