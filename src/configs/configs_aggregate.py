from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml
from icecream import ic
import sys
try:
    from .peft_config import * # noqa: F403
    from .peft_config import PEFTConfig, get_peft_config  # Import the PEFTConfig type alias and get_peft_config function explicitly
    from .wandb_config import get_wandb_config, WandbConfig, SweepConfig
    from .augmentation_config import create_augmentation_configs, AugmentationConfig
    from .dataset_config import get_dataset_config, DatasetConfig
    from .optim_config import get_optimizer_config, OptimizerConfig
except ImportError as e:
    from peft_config import * # noqa: F403
    from peft_config import PEFTConfig, get_peft_config  # Import the PEFTConfig type alias and get_peft_config function explicitly
    from wandb_config import get_wandb_config, WandbConfig, SweepConfig
    from augmentation_config import create_augmentation_configs, AugmentationConfig
    from dataset_config import get_dataset_config, DatasetConfig
    from optim_config import get_optimizer_config, OptimizerConfig

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
    
    Note: data_path and num_classes are now handled by DatasetConfig
    """
    class Config:
        strict = True

    # Core model configuration
    model_type: str # "efficientnet_b1"

    @field_validator('model_type')
    @classmethod
    def model_type_must_be_in_model_list(cls, v):
        if v not in _ModelNames().model_list:
            raise ValueError(f'model_type must be one of {_ModelNames().model_list}')
        return v
    
    # Training configuration
    save_dataloader: bool = False
    batch_size: int = 32
    seed: int = 42
    num_cuda_workers: int = 10
    pinned_memory: bool = True
    epochs: int = 10
    save_model: bool = False

    # Data splitting configuration (kept here for backward compatibility)
    test_size: float = 0.2
    inference_size: float = 0.1
    val_size: float = 0.1

    # Hyperparameter and experiment configuration
    sweep_count: int = 200
    accumulation_steps: int = 2
    patience: int = 10
    use_wandb: bool = False
    use_sweep: bool = False
    torch_viz: bool = False

    # Cross-validation configuration
    use_kfold: bool = False
    k_folds: int = 5

    # Adapter configuration
    adapter_type: str = "none-classifier"
    
    # Training monitoring settings
    early_stopping: bool = True
    checkpointing: bool = True
    monitor: str = "test_loss"  # Metric to monitor for early stopping and model checkpointing (val_loss, val_acc, etc.)
    mode: str = "min"  # "min" for loss metrics, "max" for accuracy metrics
    save_top_k: int = 1
    test_during_training: bool = True
    test_during_training_freq: int = 1  # Run test evaluation every N epochs
    
    # Distributed training configuration
    distributed_training: bool = False
    num_gpus: int = 1
    strategy: str = "ddp"
    
    @field_validator('num_gpus')
    @classmethod
    def num_gpus_must_be_positive(cls, v):
        if v < 1:
            raise ValueError('num_gpus must be at least 1')
        return v
    
    @field_validator('strategy')
    @classmethod
    def strategy_must_be_valid(cls, v):
        valid_strategies = ["ddp", "ddp_spawn", "ddp2", "dp", "fsdp"]
        if v not in valid_strategies:
            raise ValueError(f'strategy must be one of {valid_strategies}')
        return v

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


def load_configs(config: dict) -> tuple[GeneralConfig, FeatureExtractionConfig, DatasetConfig, Optional[PEFTConfig], WandbConfig, SweepConfig, AugmentationConfig, OptimizerConfig]: # noqa: F405
    """
    Load and validate all configuration objects from the config dictionary.
    
    Returns:
        Tuple containing all configuration objects in order:
        (general_config, feature_extraction_config, dataset_config, peft_config, wandb_config, sweep_config, augmentation_config, optimizer_config)
    """

    # Create configuration instances from the dictionary
    try:
        general_config = GeneralConfig(**config["general"])
        ic("GeneralConfig instance created successfully:")

        feature_extraction_config = FeatureExtractionConfig(**config["feature_extraction"])
        ic("FeatureExtractionConfig instance created successfully:")

        dataset_config = get_dataset_config(config)
        ic("DatasetConfig instance created successfully:")

        peft_config = get_peft_config(config) # noqa: F405
        ic("PeftConfig instance created successfully:")

        wandb_config, sweep_config = get_wandb_config(config)
        ic("WandbConfig instance created successfully:")
        ic("SweepConfig instance created successfully:")

        augmentation_config = create_augmentation_configs(config)
        ic("AugmentationConfig instance created successfully:")

        optimizer_config = get_optimizer_config(config)
        ic("OptimizerConfig instance created successfully:")

        return general_config, feature_extraction_config, dataset_config, peft_config, wandb_config, sweep_config, augmentation_config, optimizer_config
        
    except ValidationError as e:
        ic("Validation error occurred: ")
        ic(e)
        raise e
    
    except ValueError as e:
        ic("ValueError occurred: ")
        ic(e)
        raise e
        
    except KeyError as e:
        ic("Key error occurred, attempting fallback for sweeps case: ", e)
        try:
            # Fallback for sweep configurations that might have flattened structure
            general_config = GeneralConfig(**config)
            ic("GeneralConfig instance created successfully (fallback):")

            feature_extraction_config = FeatureExtractionConfig(**config)
            ic("FeatureExtractionConfig instance created successfully (fallback):")

            dataset_config = get_dataset_config(config)
            ic("DatasetConfig instance created successfully (fallback):")

            peft_config = get_peft_config(config) # noqa: F405
            ic("PeftConfig instance created successfully (fallback):")

            wandb_config, sweep_config = get_wandb_config(config)
            ic("WandbConfig instance created successfully (fallback):")
            ic("SweepConfig instance created successfully (fallback):")

            augmentation_config = create_augmentation_configs(config)
            ic("AugmentationConfig instance created successfully (fallback):")

            optimizer_config = get_optimizer_config(config)
            ic("OptimizerConfig instance created successfully (fallback):")

            return general_config, feature_extraction_config, dataset_config, peft_config, wandb_config, sweep_config, augmentation_config, optimizer_config
            
        except Exception as fallback_error:
            ic("Fallback also failed:", fallback_error)
            raise e  # Raise the original KeyError

def wandb_config_dict(general_config, feature_extraction_config, dataset_config, peft_config, wandb_config, augmentation_config, optimizer_config):
    """
    What dis do:
    - takes in all the configs and returns a correctly formatted
    dictionary of all the configs for the Weights & Biases Platform.
    
    """
    res = {}
    
    try:
        res['wandb_config'] = wandb_config.model_dump() if hasattr(wandb_config, 'model_dump') else dict(wandb_config)
    except Exception as e:
        print(f"Warning: Could not serialize wandb_config: {e}")
        res['wandb_config'] = {}
    
    try:
        res['general_config'] = general_config.model_dump() if hasattr(general_config, 'model_dump') else dict(general_config)
    except Exception as e:
        print(f"Warning: Could not serialize general_config: {e}")
        res['general_config'] = {}
    
    try:
        res['dataset_config'] = dataset_config.model_dump() if hasattr(dataset_config, 'model_dump') else dict(dataset_config)
    except Exception as e:
        print(f"Warning: Could not serialize dataset_config: {e}")
        res['dataset_config'] = {}
    
    try:
        if hasattr(peft_config, 'to_dict'):
            res['peft_config'] = peft_config.to_dict()
        elif hasattr(peft_config, 'model_dump'):
            res['peft_config'] = peft_config.model_dump()
        elif hasattr(peft_config, '__dict__'):
            res['peft_config'] = dict(peft_config.__dict__)
        else:
            res['peft_config'] = str(peft_config)
    except Exception as e:
        print(f"Warning: Could not serialize peft_config: {e}")
        res['peft_config'] = {}
    
    try:
        res['feature_extraction_config'] = feature_extraction_config.model_dump() if hasattr(feature_extraction_config, 'model_dump') else dict(feature_extraction_config)
    except Exception as e:
        print(f"Warning: Could not serialize feature_extraction_config: {e}")
        res['feature_extraction_config'] = {}
    
    try:
        aug_dict = augmentation_config.model_dump() if hasattr(augmentation_config, 'model_dump') else dict(augmentation_config)
        # Remove complex nested objects that might cause WandB serialization issues
        if 'aug_configs' in aug_dict:
            del aug_dict['aug_configs']
        res['augmentation_config'] = aug_dict
    except Exception as e:
        print(f"Warning: Could not serialize augmentation_config: {e}")
        res['augmentation_config'] = {}
    
    try:
        res['optimizer_config'] = optimizer_config.model_dump() if hasattr(optimizer_config, 'model_dump') else dict(optimizer_config)
    except Exception as e:
        print(f"Warning: Could not serialize optimizer_config: {e}")
        res['optimizer_config'] = {}
    
    return res

def main():
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    (
        general_config,
        feature_extraction_config,
        dataset_config,
        peft_config,
        wandb_config,
        sweep_config,
        augmentation_config,
        optimizer_config
    ) = load_configs(config)

    ic(wandb_config_dict(general_config, feature_extraction_config, dataset_config, peft_config, wandb_config, augmentation_config, optimizer_config))

if __name__ == '__main__':
    main() 