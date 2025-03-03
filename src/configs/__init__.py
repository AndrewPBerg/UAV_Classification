"""
Configuration module for UAV Classification project.

This module contains various configuration classes and utilities for the project.
"""

# Import main configuration classes from configs_demo
from .configs_demo import (
    GeneralConfig,
    FeatureExtractionConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    AugmentationConfig,
    get_config
)

# Import augmentation configurations
from .augmentation_config import (
    AugmentationConfig as AugConfig,
    SinDistortionConfig,
    TanhDistortionConfig
)

# Import PEFT configurations
from .peft_config import (
    LoraConfig,
    IA3Config,
    AdaLoraConfig,
    OFTConfig,
    FourierConfig,
    LayernormConfig,
    get_peft_config
)

# Import Wandb configurations
from .wandb_config import (
    WandbConfig,
    SweepConfig,
    get_wandb_config
)

__all__ = [
    # configs_demo exports
    'GeneralConfig',
    'FeatureExtractionConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'AugmentationConfig',
    'get_config',
    
    # augmentation_config exports
    'AugConfig',
    'SinDistortionConfig',
    'TanhDistortionConfig',
    
    # peft_config exports
    'LoraConfig',
    'IA3Config',
    'AdaLoraConfig',
    'OFTConfig',
    'FourierConfig',
    'LayernormConfig',
    'get_peft_config',
    
    # wandb_config exports
    'WandbConfig',
    'SweepConfig',
    'get_wandb_config'
] 