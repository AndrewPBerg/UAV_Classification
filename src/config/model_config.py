from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

@dataclass
class PeftArgs:
    """Arguments for PEFT configuration"""
    adapter_type: Literal["lora", "ia3"] = "lora"
    r: int = 8  # LoRA rank
    alpha: int = 16  # LoRA alpha scaling
    dropout: float = 0.1
    bias: str = "none"
    target_modules: Optional[list] = None
    modules_to_save: Optional[list] = None
    init_lora_weights: bool = True

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0

@dataclass
class ModelConfig:
    """Model configuration"""
    model_size: str = "resnet18"
    num_classes: int = None
    image_size: int = 224
    project_name: str = "uav_classification"
    model_name: Optional[str] = None
    peft_args: Optional[PeftArgs] = None

@dataclass
class FeatureExtractorConfig:
    """Feature extractor configuration"""
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0

def get_default_config() -> Dict[str, Any]:
    """Returns the default configuration dictionary"""
    return {
        "data_path": "/path/to/your/data",
        "training": TrainingConfig(),
        "model": ModelConfig(),
        "feature_extractor": FeatureExtractorConfig(),
    }
