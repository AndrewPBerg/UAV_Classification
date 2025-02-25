from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml
from icecream import ic
import sys


def handle_exception(exc_type, exc_value, exc_traceback):
    """Custom exception handler that terminates the script on any exception"""
    print(f"Fatal error: {exc_type.__name__}: {exc_value}", file=sys.stderr)
    sys.exit(1)


class ModelNames(BaseModel):
    """
    pydantic model for listing available model names
    """

    model_list: List[str] = [
                             "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
                             "ast", 
                             "mert",
                             "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                             "densenet121", "densenet161", "densenet169", "densenet201",
                             "mobilenet_v3_small", "mobilenet_v3_large",
                            ]

class GeneralConfig(BaseModel):
    """
    pydantic model for general run configs

    Required Keys (will not be defaulted):
        - model_type
        - num_classes
    
    """
    class Config:
        strict = True

    data_path: str # "/app/src/datasets/UAV_Dataset_31"
    num_classes: int
    save_dataloader: bool = False

    model_type: str # "vit232"

    @field_validator('model_type')
    @classmethod
    def model_type_must_be_in_model_list(cls, v):
        if v not in ModelNames().model_list:
            raise ValueError(f'model_type must be one of {ModelNames().model_list}')
        return v
        
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
    """
    nested pydantic model for general run configs

    runs in CnnConfig

    Required Keys (will not be defaulted):
        N/A
    
    """
    class Config:
        strict = True

    type: str = 'melspectrogram' #TODO isinstance of melspectrogram and mfcc
    sampling_rate: int = 16000
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0

class CnnConfig(BaseModel):
    """
    pydantic model for general run configs

    Depends on: FeatureExtractionConfig

    Required Keys (will not be defaulted):
        N/A
    """
    hidden_units: int = 256
    feature_extraction_config: FeatureExtractionConfig = FeatureExtractionConfig()



def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create GeneralConfig instance from the dictionary
    try:
        general_config = GeneralConfig(**config["general"])
        ic("GeneralConfig instance created successfully:")
        # ic(general_config)
    except ValidationError as e:
        ic("Validation error occurred:")
        ic(e)

    # need to create FeatureExtractionConfig before CnnConfig to pass in!

    try:
        feature_extraction_config = FeatureExtractionConfig(**config["cnn_config"]["feature_extraction"])
        ic("FeatureExtractionConfig instance created successfully:")
        # ic(feature_extraction_config)
    except ValidationError as e:
        ic("Validation error occurred:")
        ic(e)
    
    try:
        cnn_config = CnnConfig(**config["cnn_config"], feature_extraction_config=feature_extraction_config)
        ic("CnnConfig instance created successfully:")
        # ic(cnn_config)
    except ValidationError as e:
        ic("Validation error occurred:")
        ic(e)




if __name__ == '__main__':
    main()









"""
@dataclass
class PeftArgs:
    Arguments for PEFT configuration
    adapter_type: Literal["lora", "ia3"] = "lora"
    r: int = 8  # LoRA rank
    alpha: int = 16  # LoRA alpha scaling
    dropout: float = 0.1
    bias: str = "none"
    target_modules: Optional[list] = None
    modules_to_save: Optional[list] = None
    init_lora_weights: bool = True

@dataclass
class TrainingConfig
    Training configuration
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0

@dataclass
class ModelConfig:
    Model configuration"
    model_size: str = "resnet18"
    num_classes: int = None
    image_size: int = 224
    project_name: str = "uav_classification"
    model_name: Optional[str] = None
    peft_args: Optional[PeftArgs] = None

@dataclass
class FeatureExtractorConfig:
    Feature extractor configuration
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0

def get_default_config() -> Dict[str, Any]:
    Returns the default configuration dictionary
    return {
        "data_path": "/path/to/your/data",
        "training": TrainingConfig(),
        "model": ModelConfig(),
        "feature_extractor": FeatureExtractorConfig(),
    }
"""