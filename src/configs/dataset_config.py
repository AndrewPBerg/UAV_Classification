from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator
from pathlib import Path

class DatasetConfig(BaseModel):
    """
    Pydantic model for dataset configuration.
    Allows seamless switching between different datasets.
    """
    class Config:
        strict = True

    # Dataset type selection
    dataset_type: Literal["uav", "esc50"] = Field(
        description="Type of dataset to use"
    )
    
    # Common dataset parameters
    data_path: str = Field(
        description="Path to the dataset directory"
    )
    
    # Dataset-specific parameters
    num_classes: Optional[int] = Field(
        default=None,
        description="Number of classes in the dataset. If None, will be auto-detected."
    )
    
    # Audio processing parameters
    target_sr: int = Field(
        default=16000,
        description="Target sampling rate for audio files"
    )
    
    target_duration: int = Field(
        default=5,
        description="Target duration in seconds for audio clips"
    )
    
    # File format
    file_extension: str = Field(
        default=".wav",
        description="Audio file extension to look for"
    )

    @field_validator('dataset_type')
    @classmethod
    def validate_dataset_type(cls, v):
        valid_types = ["uav", "esc50"]
        if v not in valid_types:
            raise ValueError(f'dataset_type must be one of {valid_types}')
        return v

    @field_validator('data_path')
    @classmethod
    def validate_data_path(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f'data_path {v} does not exist')
        return v

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset-specific information."""
        if self.dataset_type == "uav":
            return {
                "expected_classes": 31,  # Default for UAV dataset
                "description": "UAV Audio Classification Dataset",
                "default_duration": 5,
                "default_sr": 16000
            }
        elif self.dataset_type == "esc50":
            return {
                "expected_classes": 50,
                "description": "ESC-50 Environmental Sound Classification Dataset", 
                "default_duration": 5,
                "default_sr": 44100  # ESC-50 original sampling rate
            }
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def auto_detect_num_classes(self) -> int:
        """Auto-detect number of classes from the dataset directory."""
        data_path = Path(self.data_path)
        
        # Count subdirectories (each represents a class)
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        num_classes = len(class_dirs)
        
        if num_classes == 0:
            raise ValueError(f"No class directories found in {self.data_path}")
            
        return num_classes

    def get_num_classes(self) -> int:
        """Get the number of classes, auto-detecting if not specified."""
        if self.num_classes is not None:
            return self.num_classes
        return self.auto_detect_num_classes()

class ESC50Config(DatasetConfig):
    """Specific configuration for ESC-50 dataset."""
    
    dataset_type: Literal["esc50"] = "esc50"
    target_sr: int = 16000  # Resample from 44.1kHz to 16kHz for consistency
    target_duration: int = 5
    file_extension: str = ".wav"
    
    @field_validator('target_duration')
    @classmethod
    def validate_target_duration(cls, v):
        if v != 5:
            raise ValueError("target_duration must be 5 for ESC50Config")
        return v
    
    
    # ESC-50 specific parameters
    use_esc10_subset: bool = Field(
        default=False,
        description="Whether to use only the ESC-10 subset (10 classes)"
    )
    @field_validator('use_esc10_subset')
    @classmethod
    def validate_use_esc10_subset(cls, v):
        if v:
            raise ValueError("Not implemented, use_esc10_subset must be false for ESC50Config")
        return v
    
    fold_based_split: bool = Field(
        default=True,
        description="Whether to use the predefined fold-based splits from ESC-50"
    )

class UAVConfig(DatasetConfig):
    """Specific configuration for UAV dataset."""
    
    dataset_type: Literal["uav"] = "uav"
    target_sr: int = 16000
    target_duration: int = 5
    file_extension: str = ".wav"

def create_dataset_config(config_dict: Dict[str, Any]) -> DatasetConfig:
    """
    Factory function to create the appropriate dataset configuration.
    
    Args:
        config_dict: Dictionary containing dataset configuration
        
    Returns:
        DatasetConfig: Appropriate dataset configuration instance
    """
    dataset_type = config_dict.get("dataset_type", "uav")
    
    if dataset_type == "esc50":
        return ESC50Config(**config_dict)
    elif dataset_type == "uav":
        return UAVConfig(**config_dict)
    else:
        # Fallback to generic DatasetConfig
        return DatasetConfig(**config_dict)

def get_dataset_config(config: Dict[str, Any]) -> DatasetConfig:
    """
    Extract and create dataset configuration from the main config.
    
    Args:
        config: Main configuration dictionary
        
    Returns:
        DatasetConfig: Dataset configuration instance
    """
    # Extract dataset config from general config or dedicated dataset section
    if "dataset" in config:
        dataset_config_dict = config["dataset"]
    else:
        # Fallback: extract from general config
        general_config = config.get("general", {})
        dataset_config_dict = {
            "dataset_type": general_config.get("dataset_type", "uav"),
            "data_path": general_config.get("data_path"),
            "num_classes": general_config.get("num_classes"),
        }
    
    return create_dataset_config(dataset_config_dict) 