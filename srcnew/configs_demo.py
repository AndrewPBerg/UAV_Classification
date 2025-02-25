from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError
import yaml
from icecream import ic


class GeneralConfig(BaseModel):
    data_path: str # "/app/src/datasets/UAV_Dataset_31"
    num_classes: int = 31
    save_dataloader: bool = False

    model_type: str # "vit232"
    batch_size: int = 32
    seed: int = 42
    num_cuda_workers: int = 10
    pinned_memory: bool = True
    epochs: int = 10
    save_model: bool = False

    test_size: float = 0.2
    inference_size: float = 0.1
    val_size: float = 0.1

    shuffled: bool = False
    sweep_count: int = 200
    accumulation_steps: int = 2
    learning_rate: float = 0.001
    patience: int = 10
    use_wandb: bool = False
    torch_viz: bool = False

    use_kfold: bool = False
    k_folds: int = 5

class FeatureExtractionConfig(BaseModel):
    type: str = 'melspectrogram'
    sampling_rate: int = 16000
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0

class CnnConfig(BaseModel):
    hidden_units: int = 256
    feature_extraction: FeatureExtractionConfig = FeatureExtractionConfig()



def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ic(config.general)
    ic(config.cnn_config)
    



if __name__ == '__main__':
    main()









"""
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
"""