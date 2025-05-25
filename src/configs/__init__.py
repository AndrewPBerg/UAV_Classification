"""
Configuration module for UAV Classification project.

This module contains various configuration classes and utilities for the project.
"""

# Import main configuration classes from configs_aggregate
from .configs_aggregate import (
    GeneralConfig,
    FeatureExtractionConfig,
    wandb_config_dict,
    load_configs
)

# Import augmentation configurations
from .augmentation_config import (
    AugmentationConfig as AugConfig,
    SinDistortionConfig,
    TanhDistortionConfig
)

# Import dataset configurations
from .dataset_config import (
    DatasetConfig,
    ESC50Config,
    UAVConfig,
    create_dataset_config,
    get_dataset_config
)

# Import PEFT configurations
from .peft_config import (
    get_peft_config,
    NoneClassifierConfig,
    NoneFullConfig,
    VALID_PEFT_TYPES,
    PEFTConfig,
    BatchNormConfig,
    SSFConfig,
    BitFitConfig
)

# Import native PEFT configs directly from peft library with correct paths
try:
    from peft.config import (
        LoraConfig,
        IA3Config,
        AdaLoraConfig,
        OFTConfig,
        HRAConfig,
        LNTuningConfig
    )
    from peft.utils.peft_types import TaskType
except ImportError:
    # Fallback imports if the above don't work
    try:
        from peft import (
            LoraConfig,
            IA3Config,
            AdaLoraConfig,
            OFTConfig,
            HRAConfig,
            LNTuningConfig,
            TaskType
        )
    except ImportError as e:
        print(f"Warning: Could not import PEFT configs: {e}")
        # Create dummy classes to prevent import errors
        class DummyConfig:
            pass
        LoraConfig = DummyConfig
        IA3Config = DummyConfig
        AdaLoraConfig = DummyConfig
        OFTConfig = DummyConfig
        HRAConfig = DummyConfig
        LNTuningConfig = DummyConfig
        TaskType = DummyConfig

# Import Wandb configurations
from .wandb_config import (
    WandbConfig,
    SweepConfig,
    get_wandb_config
)

__all__ = [
    # configs_aggregate exports
    'GeneralConfig',
    'FeatureExtractionConfig',
    'wandb_config_dict',
    'load_configs',
    
    # augmentation_config exports
    'AugConfig',
    'SinDistortionConfig',
    'TanhDistortionConfig',
    
    # dataset_config exports
    'DatasetConfig',
    'ESC50Config',
    'UAVConfig',
    'create_dataset_config',
    'get_dataset_config',
    
    # peft_config exports
    'LoraConfig',
    'IA3Config',
    'AdaLoraConfig',
    'OFTConfig',
    'HRAConfig',
    'LNTuningConfig',
    'TaskType',
    'NoneClassifierConfig',
    'NoneFullConfig',
    'VALID_PEFT_TYPES',
    'get_peft_config',
    'PEFTConfig',
    'BatchNormConfig',
    'SSFConfig',
    'BitFitConfig',
    
    # wandb_config exports
    'WandbConfig',
    'SweepConfig',
    'get_wandb_config'
] 