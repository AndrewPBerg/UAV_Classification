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
from configs.dataset_config import DatasetConfig, UAVConfig, ESC50Config

# Import datamodules
from .UAV_datamodule import UAVDataModule, create_uav_datamodule

# Try to import ESC50 datamodule with error handling
try:
    from src.esc50.esc50_datamodule import ESC50DataModule, create_esc50_datamodule
except ImportError as e:
    ic(f"Warning: Could not import ESC50DataModule: {e}")
    raise ImportError(f"Could not import ESC50DataModule: {e}")


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
        dataset_config: Dataset configuration (UAVConfig or ESC50Config)
        augmentation_config: Augmentation configuration
        feature_extractor: Optional pre-created feature extractor
        num_channels: Number of audio channels
        wandb_config: Optional WandB configuration for logging
        sweep_config: Optional sweep configuration for hyperparameter tuning
        **kwargs: Additional arguments
        
    Returns:
        Appropriate datamodule instance (UAVDataModule or ESC50DataModule)
        
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
        
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Supported types are: 'uav', 'esc50'"
        )


def get_datamodule_class(dataset_type: str):
    """
    Get the datamodule class for a given dataset type.
    
    Args:
        dataset_type: Type of dataset ('uav' or 'esc50')
        
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
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Supported types are: 'uav', 'esc50'"
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


def main():
    example_usage()


if __name__ == "__main__":
    main() 