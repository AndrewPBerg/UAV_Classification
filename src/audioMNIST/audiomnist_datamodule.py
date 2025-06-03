import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any, Callable
from torch.utils.data import Dataset, DataLoader
from time import time as timer
from icecream import ic
import json

# Import from the main codebase
sys.path.append(str(Path(__file__).parent.parent))

from helper.util import UAVDataset, wandb_login
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, ViTImageProcessor

# Import Pydantic configs
from configs import AugConfig as AugmentationConfig
from configs import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig
from configs.dataset_config import DatasetConfig, AudioMNISTConfig

# Import AudioMNIST specific dataset and functions
from audioMNIST.audiomnist_dataset import AudioMNISTDataset, create_audiomnist_speaker_splits, create_audiomnist_random_splits


class AudioMNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AudioMNIST audio data.
    
    Handles AudioMNIST specific features:
    - 10 digit classes (0-9)
    - Speaker-independent or speaker-dependent splits
    - Supports both random and speaker-based splitting strategies
    """
    
    def __init__(
        self,
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        audiomnist_config: AudioMNISTConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
        feature_extractor: Optional[Any] = None,
        num_channels: int = 1,
        wandb_config: Optional[WandbConfig] = None,
        sweep_config: Optional[SweepConfig] = None
    ):
        """
        Initialize the AudioMNISTDataModule using Pydantic config models.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            audiomnist_config: AudioMNIST specific configuration
            augmentation_config: Augmentation configuration
            feature_extractor: Optional pre-created feature extractor
            num_channels: Number of audio channels
            wandb_config: Optional WandB configuration for logging
            sweep_config: Optional sweep configuration for hyperparameter tuning
        """
        super().__init__()
        
        # Store configs
        self.general_config = general_config
        self.feature_extraction_config = feature_extraction_config
        self.audiomnist_config = audiomnist_config
        self.augmentation_config = augmentation_config or AugmentationConfig(
            augmentations_per_sample=0, 
            augmentations=[],
            aug_configs={}
        )
        self.wandb_config = wandb_config
        self.sweep_config = sweep_config
        
        # Unpack general config
        self.data_path = audiomnist_config.data_path
        self.batch_size = general_config.batch_size
        self.num_workers = general_config.num_cuda_workers
        self.seed = general_config.seed
        self.pin_memory = general_config.pinned_memory
        self.use_sweep = general_config.use_sweep
        self.sweep_count = general_config.sweep_count
        
        # AudioMNIST specific parameters
        self.use_speaker_splits = audiomnist_config.use_speaker_splits
        self.test_speakers = audiomnist_config.test_speakers
        
        # Unpack feature extraction config
        self.target_sr = feature_extraction_config.sampling_rate
        self.target_duration = audiomnist_config.target_duration
        self.num_channels = num_channels
        
        # Get number of classes (should be 10 for AudioMNIST)
        self.num_classes = 10  # AudioMNIST always has 10 digit classes (0-9)
        
        # Create feature extractor if not provided
        if feature_extractor is None:
            if feature_extraction_config.type == 'melspectrogram':
                self.feature_extractor = MelSpectrogramFeatureExtractor(
                    sampling_rate=feature_extraction_config.sampling_rate,
                    n_mels=feature_extraction_config.n_mels,
                    n_fft=feature_extraction_config.n_fft,
                    hop_length=feature_extraction_config.hop_length,
                    power=feature_extraction_config.power
                )
            elif feature_extraction_config.type == 'mfcc':
                self.feature_extractor = MFCCFeatureExtractor(
                    sampling_rate=feature_extraction_config.sampling_rate,
                    n_mfcc=feature_extraction_config.n_mfcc,
                    n_mels=feature_extraction_config.n_mels,
                    n_fft=feature_extraction_config.n_fft,
                    hop_length=feature_extraction_config.hop_length
                )
            else:
                raise ValueError(f"Unsupported feature extraction type: {feature_extraction_config.type}")
        else:
            self.feature_extractor = feature_extractor
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate AudioMNIST specific configuration."""
        # AudioMNIST should have exactly 10 classes
        if self.num_classes != 10:
            ic(f"Warning: AudioMNIST should have 10 classes, but num_classes is set to {self.num_classes}. Using 10 classes.")
            self.num_classes = 10
        
        # Validate speaker splits configuration
        if self.use_speaker_splits and self.test_speakers:
            if not isinstance(self.test_speakers, list):
                raise ValueError("test_speakers must be a list of speaker names")
        
    def prepare_data(self):
        """
        Prepare data for training.
        This method is called only once and on 1 GPU.
        """
        # Check if data path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"AudioMNIST data path {self.data_path} does not exist")
            
        # Check if there are audio files
        all_paths = list(Path(self.data_path).glob("*/*.wav"))
        if len(all_paths) == 0:
            raise ValueError(f"No .wav files found in {self.data_path}")
        
        # Check if we have the expected 10 digit directories
        digit_dirs = [d for d in Path(self.data_path).iterdir() if d.is_dir() and d.name.isdigit()]
        if len(digit_dirs) != 10:
            ic(f"Warning: Expected 10 digit directories, found {len(digit_dirs)}: {[d.name for d in digit_dirs]}")
        
        ic(f"AudioMNIST dataset prepared: {len(all_paths)} files in {len(digit_dirs)} digit directories")
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation.
        
        Args:
            stage: Current stage ('fit', 'validate', or 'predict')
        """
        start_time = timer()
        
        if self.use_speaker_splits:
            # Setup for speaker-independent splits
            self._setup_speaker_splits()
        else:
            # Setup for random splits (speaker-dependent)
            self._setup_random_splits()
            
        end_time = timer()
        dataset_init_time = end_time - start_time
        
        ic(f"AudioMNIST datasets created in {dataset_init_time:.2f} seconds")
    
    def _setup_speaker_splits(self):
        """
        Setup datasets for speaker-independent splits.
        """
        ic("Setting up AudioMNIST speaker-independent splits")
        
        # Create speaker-based splits
        train_dataset, val_dataset = create_audiomnist_speaker_splits(
            data_path=self.data_path,
            feature_extractor=self.feature_extractor,
            config=self.audiomnist_config,
            test_speakers=self.test_speakers,
            val_ratio=0.2,  # 20% of remaining speakers for validation
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            aug_config=self.augmentation_config
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        print(f"Speaker-based splits - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
        # Print speaker information
        train_speakers = self.train_dataset.get_available_speakers()
        val_speakers = self.val_dataset.get_available_speakers()
        
        print(f"Train speakers ({len(train_speakers)}): {train_speakers}")
        print(f"Val speakers ({len(val_speakers)}): {val_speakers}")
        
    def _setup_random_splits(self):
        """
        Setup datasets for random splits (speaker-dependent).
        """
        ic("Setting up AudioMNIST random splits (speaker-dependent)")
        
        # Create random splits
        train_dataset, val_dataset = create_audiomnist_random_splits(
            data_path=self.data_path,
            feature_extractor=self.feature_extractor,
            config=self.audiomnist_config,
            train_ratio=0.8,  # 80% for training
            val_ratio=0.2,    # 20% for validation
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            aug_config=self.augmentation_config,
            random_seed=self.seed
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        print(f"Random splits - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
    def train_dataloader(self):
        """
        Get training dataloader.
        
        Returns:
            Training dataloader
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def val_dataloader(self):
        """
        Get validation dataloader.
        
        Returns:
            Validation dataloader
        """
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def test_dataloader(self):
        """
        Get test dataloader (same as validation dataloader for AudioMNIST).
        
        Returns:
            Test dataloader (copy of validation dataloader)
        """
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def predict_dataloader(self):
        """
        Get inference dataloader (same as test dataloader for AudioMNIST).
        
        Returns:
            Inference dataloader
        """
        return self.test_dataloader()
        
    def get_class_info(self):
        """
        Get class information.
        
        Returns:
            Tuple of (list of class names, mapping of class name to index, mapping of index to class name)
        """
        if self.train_dataset:
            classes = self.train_dataset.get_classes()
            class_to_idx = self.train_dataset.get_class_dict()
            idx_to_class = self.train_dataset.get_idx_dict()
            return classes, class_to_idx, idx_to_class
        else:
            # Fallback: return digit classes
            classes = [str(i) for i in range(10)]
            class_to_idx = {str(i): i for i in range(10)}
            idx_to_class = {i: str(i) for i in range(10)}
            return classes, class_to_idx, idx_to_class
    
    def get_speaker_info(self):
        """
        Get speaker information.
        
        Returns:
            Dictionary with speaker statistics
        """
        if self.train_dataset and hasattr(self.train_dataset, 'get_speaker_statistics'):
            info = {
                "use_speaker_splits": self.use_speaker_splits,
                "train_speakers": self.train_dataset.get_available_speakers(),
                "train_speaker_stats": self.train_dataset.get_speaker_statistics(),
            }
            
            if self.val_dataset and hasattr(self.val_dataset, 'get_speaker_statistics'):
                info["val_speakers"] = self.val_dataset.get_available_speakers()
                info["val_speaker_stats"] = self.val_dataset.get_speaker_statistics()
            
            return info
        
        return {"error": "Speaker information not available"}
    
    def get_digit_info(self):
        """
        Get digit distribution information.
        
        Returns:
            Dictionary with digit statistics
        """
        if self.train_dataset and hasattr(self.train_dataset, 'get_digit_statistics'):
            info = {
                "train_digits": self.train_dataset.get_available_digits(),
                "train_digit_stats": self.train_dataset.get_digit_statistics(),
            }
            
            if self.val_dataset and hasattr(self.val_dataset, 'get_digit_statistics'):
                info["val_digits"] = self.val_dataset.get_available_digits()
                info["val_digit_stats"] = self.val_dataset.get_digit_statistics()
            
            return info
        
        return {"error": "Digit information not available"}


def create_audiomnist_datamodule(
    general_config: GeneralConfig,
    feature_extraction_config: FeatureExtractionConfig,
    audiomnist_config: AudioMNISTConfig,
    augmentation_config: Optional[AugmentationConfig] = None,
    **kwargs
) -> AudioMNISTDataModule:
    """
    Factory function to create AudioMNISTDataModule.
    
    Args:
        general_config: General configuration
        feature_extraction_config: Feature extraction configuration
        audiomnist_config: AudioMNIST specific configuration
        augmentation_config: Augmentation configuration
        **kwargs: Additional arguments
        
    Returns:
        AudioMNISTDataModule instance
    """
    return AudioMNISTDataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        audiomnist_config=audiomnist_config,
        augmentation_config=augmentation_config,
        **kwargs
    )


# Example usage
def example_usage():
    from configs.configs_aggregate import load_configs
    import yaml
    
    try:
        # Load configs from config.yaml
        print("Loading configs from config.yaml")
        with open('../configs/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Change dataset type and data path for AudioMNIST
        config['dataset']['dataset_type'] = 'audiomnist'
        config['dataset']['data_path'] = "../datasets/audiomnist_dataset"
        
        print("Loading all configs using the configs_aggregate function")
        # Load all configs using the configs_aggregate function
        (
            general_config,
            feature_extraction_config,
            dataset_config,
            peft_config,
            wandb_config,
            sweep_config,
            augmentation_config
        ) = load_configs(config)
        
        print("successfully loaded all configs")
        
        # Create data module
        data_module = create_audiomnist_datamodule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            audiomnist_config=dataset_config,  # dataset_config contains AudioMNISTConfig
            augmentation_config=augmentation_config
        )
        
        print("successfully created the data module")
        
        # Setup data module
        data_module.setup()
        print("successfully setup the data module")
        
        # Get dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        ic(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        # Print dataset info
        classes, class_to_idx, idx_to_class = data_module.get_class_info()
        speaker_info = data_module.get_speaker_info()
        digit_info = data_module.get_digit_info()
        
        print(f"Classes: {classes}")
        print(f"Number of classes: {len(classes)}")
        ic("Speaker Info:", speaker_info)
        ic("Digit Info:", digit_info)
        
        
    except Exception as e:
        ic(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    example_usage()


if __name__ == "__main__":
    main() 