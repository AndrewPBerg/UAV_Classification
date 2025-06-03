"""
DataModule Factory for routing to the correct datamodule based on dataset type.
"""

import sys
from pathlib import Path
from typing import Optional, Any
from icecream import ic

# Add the project root to sys.path to find esc50 package
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Pydantic configs
from configs import AugConfig as AugmentationConfig
from configs import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig
from configs.dataset_config import DatasetConfig, UAVConfig, ESC50Config, ESC10Config, UrbanSound8KConfig, AudioMNISTConfig

# Import datamodules
from .UAV_datamodule import UAVDataModule, create_uav_datamodule

# Try to import ESC50 datamodule with error handling
try:
    from src.esc50.esc50_datamodule import ESC50DataModule, create_esc50_datamodule
except ImportError as e:
    ic(f"Warning: Could not import ESC50DataModule: {e}")
    ESC50DataModule = None
    create_esc50_datamodule = None

# Try to import ESC10 datamodule with error handling
try:
    from src.esc10.esc10_datamodule import ESC10DataModule, create_esc10_datamodule
except ImportError as e:
    ic(f"Warning: Could not import ESC10DataModule: {e}")
    ESC10DataModule = None
    create_esc10_datamodule = None

# Try to import UrbanSound8K datamodule with error handling
try:
    from src.urbansound8k.urbansound8k_datamodule import UrbanSound8KDataModule, create_urbansound8k_datamodule
except ImportError as e:
    ic(f"Warning: Could not import UrbanSound8KDataModule: {e}")
    UrbanSound8KDataModule = None
    create_urbansound8k_datamodule = None

# Try to import AudioMNIST datamodule with error handling
try:
    from src.audioMNIST.audiomnist_datamodule import AudioMNISTDataModule, create_audiomnist_datamodule
except ImportError as e:
    ic(f"Warning: Could not import AudioMNISTDataModule: {e}")
    AudioMNISTDataModule = None
    create_audiomnist_datamodule = None


def create_datamodule(
    general_config: GeneralConfig,
    feature_extraction_config: FeatureExtractionConfig,
    dataset_config: DatasetConfig,
    augmentation_config: Optional[AugmentationConfig] = None,
    feature_extractor: Optional[Any] = None,
    num_channels: int = 1,
    wandb_config: Optional[WandbConfig] = None,
    sweep_config: Optional[SweepConfig] = None,
    **kwargs
):
    """
    Factory function to create the appropriate datamodule based on dataset type.
    
    Args:
        general_config: General configuration
        feature_extraction_config: Feature extraction configuration
        dataset_config: Dataset configuration (UAVConfig, ESC50Config, ESC10Config, or UrbanSound8KConfig)
        augmentation_config: Augmentation configuration
        feature_extractor: Optional pre-created feature extractor
        num_channels: Number of audio channels
        wandb_config: Optional WandB configuration for logging
        sweep_config: Optional sweep configuration for hyperparameter tuning
        **kwargs: Additional arguments
        
    Returns:
        Appropriate datamodule instance (UAVDataModule, ESC50DataModule, ESC10DataModule, or UrbanSound8KDataModule)
        
    Raises:
        ValueError: If dataset type is not supported
        ImportError: If required datamodule cannot be imported
    """
    
    dataset_type = dataset_config.dataset_type
    ic(f"Creating datamodule for dataset type: {dataset_type}")
    
    # Common arguments for all datamodules
    common_args = {
        "general_config": general_config,
        "feature_extraction_config": feature_extraction_config,
        "augmentation_config": augmentation_config,
        "feature_extractor": feature_extractor,
        "num_channels": num_channels,
        "wandb_config": wandb_config,
        "sweep_config": sweep_config,
        **kwargs
    }
    
    if dataset_type == "uav":
        if not isinstance(dataset_config, UAVConfig):
            # Convert to UAVConfig if needed
            dataset_config = UAVConfig(**dataset_config.model_dump())
        
        ic("Creating UAV datamodule")
        return UAVDataModule(
            uav_config=dataset_config,
            **common_args
        )
        
    elif dataset_type == "esc50":
        if ESC50DataModule is None:
            raise ImportError(
                "ESC50DataModule could not be imported. "
                "Make sure the esc50 directory is accessible and contains the required files."
            )
        
        if not isinstance(dataset_config, ESC50Config):
            # Convert to ESC50Config if needed
            dataset_config = ESC50Config(**dataset_config.model_dump())
        
        ic("Creating ESC-50 datamodule")
        return ESC50DataModule(
            esc50_config=dataset_config,
            **common_args
        )
        
    elif dataset_type == "esc10":
        if ESC10DataModule is None:
            raise ImportError(
                "ESC10DataModule could not be imported. "
                "Make sure the esc10 directory is accessible and contains the required files."
            )
        
        if not isinstance(dataset_config, ESC10Config):
            # Convert to ESC10Config if needed
            dataset_config = ESC10Config(**dataset_config.model_dump())
        
        ic("Creating ESC-10 datamodule")
        return ESC10DataModule(
            esc10_config=dataset_config,
            **common_args
        )
        
    elif dataset_type == "urbansound8k":
        if UrbanSound8KDataModule is None:
            raise ImportError(
                "UrbanSound8KDataModule could not be imported. "
                "Make sure the urbansound8k directory is accessible and contains the required files."
            )
        
        if not isinstance(dataset_config, UrbanSound8KConfig):
            # Convert to UrbanSound8KConfig if needed
            dataset_config = UrbanSound8KConfig(**dataset_config.model_dump())
        
        ic("Creating UrbanSound8K datamodule")
        return UrbanSound8KDataModule(
            urbansound8k_config=dataset_config,
            **common_args
        )
        
    elif dataset_type == "audiomnist":
        if AudioMNISTDataModule is None:
            raise ImportError(
                "AudioMNISTDataModule could not be imported. "
                "Make sure the audiomnist directory is accessible and contains the required files."
            )
        
        if not isinstance(dataset_config, AudioMNISTConfig):
            # Convert to AudioMNISTConfig if needed
            dataset_config = AudioMNISTConfig(**dataset_config.model_dump())
        
        ic("Creating AudioMNIST datamodule")
        return AudioMNISTDataModule(
            audiomnist_config=dataset_config,
            **common_args
        )
        
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Supported types are: 'uav', 'esc50', 'esc10', 'urbansound8k', 'audiomnist'"
        )


def get_datamodule_class(dataset_type: str):
    """
    Get the datamodule class for a given dataset type.
    
    Args:
        dataset_type: Type of dataset ('uav', 'esc50', 'esc10', 'urbansound8k', or 'audiomnist')
        
    Returns:
        Datamodule class
        
    Raises:
        ValueError: If dataset type is not supported
        ImportError: If required datamodule cannot be imported
    """
    if dataset_type == "uav":
        return UAVDataModule
    elif dataset_type == "esc50":
        if ESC50DataModule is None:
            raise ImportError(
                "ESC50DataModule could not be imported. "
                "Make sure the esc50 directory is accessible and contains the required files."
            )
        return ESC50DataModule
    elif dataset_type == "esc10":
        if ESC10DataModule is None:
            raise ImportError(
                "ESC10DataModule could not be imported. "
                "Make sure the esc10 directory is accessible and contains the required files."
            )
        return ESC10DataModule
    elif dataset_type == "urbansound8k":
        if UrbanSound8KDataModule is None:
            raise ImportError(
                "UrbanSound8KDataModule could not be imported. "
                "Make sure the urbansound8k directory is accessible and contains the required files."
            )
        return UrbanSound8KDataModule
    elif dataset_type == "audiomnist":
        if AudioMNISTDataModule is None:
            raise ImportError(
                "AudioMNISTDataModule could not be imported. "
                "Make sure the audiomnist directory is accessible and contains the required files."
            )
        return AudioMNISTDataModule
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Supported types are: 'uav', 'esc50', 'esc10', 'urbansound8k', 'audiomnist'"
        )


def get_supported_dataset_types():
    """
    Get list of supported dataset types.
    
    Returns:
        List of supported dataset type strings
    """
    supported_types = ["uav"]
    
    if ESC50DataModule is not None:
        supported_types.append("esc50")
    
    if ESC10DataModule is not None:
        supported_types.append("esc10")
    
    if UrbanSound8KDataModule is not None:
        supported_types.append("urbansound8k")
    
    if AudioMNISTDataModule is not None:
        supported_types.append("audiomnist")
    
    return supported_types


def validate_dataset_config(dataset_config: DatasetConfig):
    """
    Validate that the dataset configuration is compatible with available datamodules.
    
    Args:
        dataset_config: Dataset configuration to validate
        
    Raises:
        ValueError: If dataset type is not supported
        ImportError: If required datamodule cannot be imported
    """
    dataset_type = dataset_config.dataset_type
    supported_types = get_supported_dataset_types()
    
    if dataset_type not in supported_types:
        raise ValueError(
            f"Dataset type '{dataset_type}' is not supported. "
            f"Supported types are: {supported_types}"
        )
    
    # Additional validation for ESC-50
    if dataset_type == "esc50" and ESC50DataModule is None:
        raise ImportError(
            "ESC-50 dataset type is configured but ESC50DataModule could not be imported. "
            "Make sure the esc50 directory is accessible and contains the required files."
        )
    
    # Additional validation for ESC-10
    if dataset_type == "esc10" and ESC10DataModule is None:
        raise ImportError(
            "ESC-10 dataset type is configured but ESC10DataModule could not be imported. "
            "Make sure the esc10 directory is accessible and contains the required files."
        )
    
    # Additional validation for UrbanSound8K
    if dataset_type == "urbansound8k" and UrbanSound8KDataModule is None:
        raise ImportError(
            "UrbanSound8K dataset type is configured but UrbanSound8KDataModule could not be imported. "
            "Make sure the urbansound8k directory is accessible and contains the required files."
        )
    
    # Additional validation for AudioMNIST
    if dataset_type == "audiomnist" and AudioMNISTDataModule is None:
        raise ImportError(
            "AudioMNIST dataset type is configured but AudioMNISTDataModule could not be imported. "
            "Make sure the audiomnist directory is accessible and contains the required files."
        )


# Backward compatibility aliases
AudioDataModule = UAVDataModule  # The original AudioDataModule is now UAVDataModule


def create_audio_datamodule(*args, **kwargs):
    """
    Backward compatibility function.
    Creates a UAV datamodule (the original AudioDataModule).
    """
    ic("Warning: create_audio_datamodule is deprecated. Use create_datamodule instead.")
    return create_uav_datamodule(*args, **kwargs)


# Example usage and testing
def example_usage():
    """Example of how to use the datamodule factory"""
    from configs import load_configs
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Load configs
    (general_config, feature_extraction_config, dataset_config, 
     peft_config, wandb_config, sweep_config, augmentation_config) = load_configs(config)
    
    print("="*50)
    print("DataModule Factory Example")
    print("="*50)
    
    # Validate dataset config
    try:
        validate_dataset_config(dataset_config)
        ic(f"Dataset config validation passed for type: {dataset_config.dataset_type}")
    except (ValueError, ImportError) as e:
        ic(f"Dataset config validation failed: {e}")
        return
    
    # Create data module using factory
    try:
        data_module = create_datamodule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            dataset_config=dataset_config,
            augmentation_config=augmentation_config
        )
        ic(f"Created datamodule: {type(data_module).__name__}")
        
        # Setup data module
        data_module.setup()
        ic("Setup datamodule")
        
        # Get class information
        classes, class_to_idx, idx_to_class = data_module.get_class_info()
        print(f"Dataset type: {dataset_config.dataset_type}")
        print(f"Number of classes: {len(classes)}")
        print(f"First few classes: {classes[:5] if len(classes) > 5 else classes}")
        
        print("✅ DataModule factory test completed successfully!")
        
    except Exception as e:
        ic(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_usage() 