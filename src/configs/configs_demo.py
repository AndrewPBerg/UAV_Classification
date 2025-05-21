from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml
from icecream import ic
import sys
try:
    from .peft_config import * # noqa: F403
    from .peft_config import PEFTConfig  # Import the PEFTConfig type alias explicitly
    from .wandb_config import get_wandb_config, WandbConfig, SweepConfig
    from .augmentation_config import create_augmentation_configs, AugmentationConfig
except ImportError as e:
    from peft_config import * # noqa: F403
    from peft_config import PEFTConfig  # Import the PEFTConfig type alias explicitly
    from wandb_config import get_wandb_config, WandbConfig, SweepConfig
    from augmentation_config import create_augmentation_configs, AugmentationConfig

def handle_exception(exc_type, exc_value, exc_traceback):
    """Custom exception handler that terminates the script on any exception"""
    print(f"Fatal error: {exc_type.__name__}: {exc_value}", file=sys.stderr)
    sys.exit(1)


class _ModelNames(BaseModel):
    """
    pydantic model for listing available model names
    """

    model_list: List[str] = [
                             "vit-base","vit-large",
                             "deit-tiny", "deit-small", "deit-base",
                             "deit-tiny-distil", "deit-small-distil", "deit-base-distil",
                             "ast", 
                             "mert",
                             "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                             "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", 
                             "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
                             "mobilenet_v3_small", "mobilenet_v3_large",
                             "custom_cnn"
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
        if v not in _ModelNames().model_list:
            raise ValueError(f'model_type must be one of {_ModelNames().model_list}')
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

    sweep_count: int = 200
    accumulation_steps: int = 2
    learning_rate: float = 0.001
    patience: int = 10
    use_wandb: bool = False
    use_sweep: bool = False
    torch_viz: bool = False

    use_kfold: bool = False
    k_folds: int = 5

    adapter_type: str = "none-classifier"
    
    # Training monitoring settings
    early_stopping: bool = True
    checkpointing: bool = True
    monitor: str = "test_loss"  # Metric to monitor for early stopping and model checkpointing (val_loss, val_acc, etc.)
    mode: str = "min"  # "min" for loss metrics, "max" for accuracy metrics
    save_top_k: int = 1
    test_during_training: bool = True
    test_during_training_freq: int = 1  # Run test evaluation every N epochs

class _FeatureExtractionType(BaseModel):
    type: List[str] = ['melspectrogram','mfcc'] 

class FeatureExtractionConfig(BaseModel):
    """
    nested pydantic model for general run configs

    Required Keys (will not be defaulted):
        N/A
    
    """
    class Config:
        strict = True

    type: str = 'melspectrogram' 

    @field_validator('type')
    @classmethod
    def type_must_be_in_list(cls, v):
        if v not in _FeatureExtractionType().type:
            raise ValueError(f'type must be one of {_FeatureExtractionType().type}')
        return v
        
    sampling_rate: int = 16000
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0


def load_configs(config: dict) -> tuple[GeneralConfig, FeatureExtractionConfig, Optional[PEFTConfig], WandbConfig, SweepConfig, AugmentationConfig ]: # noqa: F405

    

    # Create GeneralConfig instance from the dictionary
    try:
        general_config = GeneralConfig(**config["general"])
        ic("GeneralConfig instance created successfully:")

        feature_extraction_config = FeatureExtractionConfig(**config["feature_extraction"])
        ic("FeatureExtractionConfig instance created successfully:")

        peft_config = get_peft_config(config) # noqa: F405
        ic("PeftConfig instance created successfully:")

        wandb_config, sweep_config = get_wandb_config(config)
        ic("WandbConfig instance created successfully:")
        ic("SweepConfig instance created successfully:")

        augmentation_config = create_augmentation_configs(config)
        ic("AugmentationConfig instance created successfully:")

        return general_config, feature_extraction_config, peft_config, wandb_config, sweep_config, augmentation_config
    except ValidationError as e:
        ic("Validation error occurred: ")
        ic(e)
    
    except ValueError as e:
        ic("ValueError occurred: ")
        ic(e)
    except KeyError as e:
        ic("Key error occurred, defaulting to sweeps case: ", e)
        general_config = GeneralConfig(**config)
        ic("GeneralConfig instance created successfully:")

        feature_extraction_config = FeatureExtractionConfig(**config)
        ic("FeatureExtractionConfig instance created successfully:")

        peft_config = get_peft_config(config) # noqa: F405
        ic("PeftConfig instance created successfully:")

        wandb_config, sweep_config = get_wandb_config(config)
        ic("WandbConfig instance created successfully:")
        ic("SweepConfig instance created successfully:")

        augmentation_config = create_augmentation_configs(config)
        ic("AugmentationConfig instance created successfully:")

        return general_config, feature_extraction_config, peft_config, wandb_config, sweep_config, augmentation_config

def wandb_config_dict(general_config, feature_extraction_config, peft_config, wandb_config, augmentation_config):
    """
    What dis do:
    - takes in all the configs and returns a correctly formatted
    dictionary of all the configs for the Weights & Biases Platform.
    
    """
    res = {}
    res['wandb_config'] = dict(wandb_config)
    res['general_config'] = dict(general_config)
    res['peft_config'] = dict(peft_config.to_dict())
    res['feature_extraction_config'] = dict(feature_extraction_config)
    res['augmentation_config'] = dict(augmentation_config)
    
    return res

def main():
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    (
        general_config,
        feature_extraction_config,
        peft_config,
        wandb_config,
        sweep_config,
        augmentation_config
    ) = load_configs(config)

    ic(wandb_config_dict(general_config, feature_extraction_config, peft_config, wandb_config, augmentation_config))

if __name__ == '__main__':
    main()