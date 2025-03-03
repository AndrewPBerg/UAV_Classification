from typing import Optional, Literal, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml
from icecream import ic
import sys
try:
    from .peft_config import * # noqa: F403
    from .wandb_config import get_wandb_config, WandbConfig, SweepConfig
    from .augmentation_config import create_augmentation_configs, AugmentationConfig
except ImportError as e:
    from peft_config import * # noqa: F403
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


def load_configs(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Load configurations from the given dictionary into pydantic models.
    
    Args:
        config: Dictionary containing the configuration.
        
    Returns:
        Tuple of (general_config, model_config, train_config, aug_config, wandb_config, sweep_config).
    """
    try:
        # Ensure all required sections exist
        if 'general' not in config:
            config['general'] = {
                'data_path': 'data/processed',
                'num_classes': 10,
                'model_type': 'ast',
                'seed': 42
            }
        
        if 'model' not in config:
            config['model'] = {
                'adapter_type': 'default',
                'trainable_layers': 'all',
                'freeze_feature_extractor': False
            }
        
        if 'train' not in config:
            config['train'] = {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001
            }
        
        if 'augmentation' not in config:
            config['augmentation'] = {
                'enabled': False
            }
        
        if 'wandb' not in config:
            config['wandb'] = {
                'project': 'default_project',
                'entity': 'default_entity',
                'tags': [],
                'notes': '',
                'group': '',
                'job_type': '',
                'save_code': False,
                'log_model': False
            }
        
        if 'sweep' not in config:
            config['sweep'] = {
                'method': 'random',
                'metric': {'name': 'val_acc', 'goal': 'maximize'},
                'parameters': {}
            }
        
        # Create pydantic models
        general_config = GeneralConfig(**config['general'])
        print(ic('GeneralConfig instance created successfully:'))
        
        model_config = FeatureExtractionConfig(**config['model'])
        print(ic('FeatureExtractionConfig instance created successfully:'))
        
        peft_config = PeftConfig(**config['model'])
        print(ic('PeftConfig instance created successfully:'))
        
        wandb_config = WandbConfig(**config['wandb'])
        print(ic('WandbConfig instance created successfully:'))
        
        sweep_config = SweepConfig(**config['sweep'])
        print(ic('SweepConfig instance created successfully:'))
        
        # Initialize aug_config before any potential exceptions
        aug_config = AugmentationConfig(**config['augmentation'])
        print(ic('AugmentationConfig instance created successfully:'))
        
        return general_config, model_config, peft_config, aug_config, wandb_config, sweep_config
    except Exception as e:
        print(ic("Error in load_configs: ", e))
        # Re-raise the exception to be handled by the caller
        raise

def wandb_config_dict(general_config, feature_extraction_config, peft_config, wandb_config):
    """
    What dis do:
    - takes in all the configs and returns a correctly formatted
    dictionary of all the configs for the Weights & Biases Platform.
    
    """
    res = {}
    res['wandb_config'] = dict(wandb_config)
    res['general_config'] = dict(general_config)
    res['peft_config'] = dict(peft_config)
    res['feature_extraction_config'] = dict(feature_extraction_config)
    
    return res

def main():
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    (
        general_config,
        feature_extraction_config,
        peft_config,
        augmentation_config,
        wandb_config,
        sweep_config
    ) = load_configs(config)

    ic(wandb_config_dict(general_config, feature_extraction_config, peft_config, wandb_config))

if __name__ == '__main__':
    main()