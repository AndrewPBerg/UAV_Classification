
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from time import time as timer
from icecream import ic

# Import from existing code
from helper.util import AudioDataset
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, BitImageProcessor

# Import Pydantic configs
from configs.augmentation_config import AugmentationConfig
from configs.configs_demo import GeneralConfig, FeatureExtractionConfig, CnnConfig, WandbConfig, SweepConfig



"""
Core functionality of current AudioDataset?

1. load audio from datapath
2. feature extraction
3. apply & inflate w/ augmentations (naunce with static augmentations data paths):
    [not sure yet how to handle, might just force creation of new augmentation dataset with each new augmentation permutation]
4. iterable
5. getter helper methods
6. show sepctrogram of i
"""



"""
What should the new PL AudioDataset look like?

1. maintain and speed up above functionality 
2. higher readibility, maintainability, and modularity
3. ...

"""

"""
Audio dataset implementation using PyTorch Lightning
"""

class AudioDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for audio data.
    """
    def __init__(
        self,
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
        feature_extractor: Optional[Any] = None,
        num_channels: int = 1
    ):
        """
        Initialize the AudioDataModule using Pydantic config models.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            augmentation_config: Augmentation configuration
            feature_extractor: Optional pre-created feature extractor (if None, will be created based on config)
            num_channels: Number of audio channels
        """
        super().__init__()
        
        # Unpack general config
        self.data_path = general_config.data_path
        self.batch_size = general_config.batch_size
        self.num_workers = general_config.num_cuda_workers
        self.test_size = general_config.test_size
        self.val_size = general_config.val_size
        self.inference_size = general_config.inference_size
        self.seed = general_config.seed
        self.use_kfold = general_config.use_kfold
        self.k_folds = general_config.k_folds
        self.pin_memory = general_config.pinned_memory
        
        # Unpack feature extraction config
        self.target_sr = feature_extraction_config.sampling_rate
        
        # Set other parameters
        self.target_duration = 5  # Default value, could be added to config
        self.num_channels = num_channels
        self.augmentation_config = augmentation_config or AugmentationConfig(augmentations_per_sample=0, augmentations=[])
        
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
        self.test_dataset = None
        self.inference_dataset = None
        self.fold_datasets = None
        
    def prepare_data(self):
        """
        Prepare data for training.
        This method is called only once and on 1 GPU.
        """
        # Check if data path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
            
        # Check if there are audio files
        all_paths = list(Path(self.data_path).glob("*/*.wav"))
        if len(all_paths) == 0:
            raise ValueError(f"No .wav files found in {self.data_path}")
            
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, testing, and inference.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Get all audio file paths
        all_paths = list(Path(self.data_path).glob("*/*.wav"))
        
        if self.use_kfold:
            self._setup_kfold(all_paths)
        else:
            self._setup_train_val_test(all_paths)
            
    def _setup_train_val_test(self, all_paths: List[Path]):
        """
        Setup datasets for standard train/val/test split.
        
        Args:
            all_paths: List of paths to all audio files
        """
        # Split the dataset
        train_size = 1.0 - (self.val_size + self.test_size + self.inference_size)
        assert np.isclose(train_size + self.val_size + self.test_size + self.inference_size, 1.0), \
            "The sum of val_size, test_size, and inference_size should be less than 1.0"
        
        n_samples = len(all_paths)
        n_train = int(n_samples * train_size)
        n_val = int(n_samples * self.val_size)
        n_test = int(n_samples * self.test_size)
        
        # Create random state for reproducibility
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n_samples)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:n_train+n_val+n_test]
        inference_indices = indices[n_train+n_val+n_test:]
        
        # Get paths for each split
        train_paths = [str(all_paths[i]) for i in train_indices]
        val_paths = [str(all_paths[i]) for i in val_indices]
        test_paths = [str(all_paths[i]) for i in test_indices]
        inference_paths = [str(all_paths[i]) for i in inference_indices]
        
        # Create datasets
        start_time = timer()
        
        self.train_dataset = AudioDataset(
            data_path=self.data_path,
            data_paths=train_paths,
            feature_extractor=self.feature_extractor,
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels,
            config=self.augmentation_config.aug_configs
        )
        
        self.val_dataset = AudioDataset(
            data_path=self.data_path,
            data_paths=val_paths,
            feature_extractor=self.feature_extractor,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels
        )
        
        self.test_dataset = AudioDataset(
            data_path=self.data_path,
            data_paths=test_paths,
            feature_extractor=self.feature_extractor,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels
        )
        
        self.inference_dataset = AudioDataset(
            data_path=self.data_path,
            data_paths=inference_paths,
            feature_extractor=self.feature_extractor,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels
        )
        
        end_time = timer()
        dataset_init_time = end_time - start_time
        
        ic(f"Datasets created in {dataset_init_time:.2f} seconds - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, "
           f"Test: {len(self.test_dataset)}, Inference: {len(self.inference_dataset)}")
            
    def _setup_kfold(self, all_paths: List[Path]):
        """
        Setup datasets for k-fold cross validation.
        
        Args:
            all_paths: List of paths to all audio files
        """
        n_samples = len(all_paths)
        n_inference = int(n_samples * self.inference_size)
        
        # Create random state for reproducibility
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n_samples)
        
        # Split off inference set
        train_val_indices = indices[:-n_inference]
        inference_indices = indices[-n_inference:]
        
        # Create KFold object
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        
        # Create datasets for each fold
        self.fold_datasets = []
        start_time = timer()
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices)):
            # Get paths for this fold
            fold_train_paths = [str(all_paths[i]) for i in train_val_indices[train_idx]]
            fold_val_paths = [str(all_paths[i]) for i in train_val_indices[val_idx]]
            
            # Create datasets
            train_dataset = AudioDataset(
                data_path=self.data_path,
                data_paths=fold_train_paths,
                feature_extractor=self.feature_extractor,
                augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
                augmentations=self.augmentation_config.augmentations,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                num_channels=self.num_channels,
                config=self.augmentation_config.aug_configs
            )
            
            val_dataset = AudioDataset(
                data_path=self.data_path,
                data_paths=fold_val_paths,
                feature_extractor=self.feature_extractor,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                num_channels=self.num_channels
            )
            
            self.fold_datasets.append((train_dataset, val_dataset))
            ic(f"Fold {fold+1} datasets loaded")
        
        # Create inference dataset
        inference_paths = [str(all_paths[i]) for i in inference_indices]
        self.inference_dataset = AudioDataset(
            data_path=self.data_path,
            data_paths=inference_paths,
            feature_extractor=self.feature_extractor,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels
        )
        
        end_time = timer()
        dataset_init_time = end_time - start_time
        
        ic(f"K-fold datasets created in {dataset_init_time:.2f} seconds")
        ic(f"Inference dataset loaded - {len(self.inference_dataset)} samples")
        
    def train_dataloader(self):
        """
        Get training dataloader.
        
        Returns:
            Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def val_dataloader(self):
        """
        Get validation dataloader.
        
        Returns:
            Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def test_dataloader(self):
        """
        Get test dataloader.
        
        Returns:
            Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def predict_dataloader(self):
        """
        Get inference dataloader.
        
        Returns:
            Inference dataloader
        """
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def get_fold_dataloaders(self, fold_idx: int):
        """
        Get dataloaders for a specific fold.
        
        Args:
            fold_idx: Index of the fold
            
        Returns:
            Tuple of (train dataloader, validation dataloader)
        """
        if not self.use_kfold:
            raise ValueError("K-fold cross validation is not enabled")
            
        if fold_idx < 0 or fold_idx >= self.k_folds:
            raise ValueError(f"Fold index {fold_idx} out of range (0-{self.k_folds-1})")
            
        train_dataset, val_dataset = self.fold_datasets[fold_idx]
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader
        
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
        elif self.fold_datasets:
            train_dataset, _ = self.fold_datasets[0]
            classes = train_dataset.get_classes()
            class_to_idx = train_dataset.get_class_dict()
            idx_to_class = train_dataset.get_idx_dict()
            return classes, class_to_idx, idx_to_class
        else:
            raise ValueError("No datasets available")


# Example usage
def example_usage():
    """Example of how to use the AudioDataModule"""
    from configs.configs_demo import load_configs
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    general_config, feature_extraction_config, cnn_config, peft_config, wandb_config, sweep_config, augmentation_config = load_configs(config)
    
    # Create data module directly from configs
    data_module = AudioDataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        augmentation_config=augmentation_config
    )
    ic("created the audio data module")
    
    # Setup data module
    data_module.setup()
    ic("setup the data module")
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    ic("got the dataloaders")
    
    # For k-fold cross validation
    if general_config.use_kfold:
        for fold in range(general_config.k_folds):
            fold_train_loader, fold_val_loader = data_module.get_fold_dataloaders(fold)
            # Use fold_train_loader and fold_val_loader for training
    
    # Get class information
    classes, class_to_idx, idx_to_class = data_module.get_class_info()
    print(f"Classes: {classes}")
    print(f"Number of classes: {len(classes)}")


def main():
    example_usage()


if __name__ == "__main__":
    main()


