import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from time import time as timer
from icecream import ic
import json

# Import from existing code
from helper.util import AudioDataset
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, BitImageProcessor

# Import Pydantic configs
from configs.augmentation_config import AugmentationConfig
from configs.configs_demo import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig


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
        self.save_dataloader = general_config.save_dataloader
        self.num_classes = general_config.num_classes
        
        # Unpack feature extraction config
        self.target_sr = feature_extraction_config.sampling_rate
        
        # Set other parameters
        self.target_duration = 5  # Default value, could be added to config
        self.num_channels = num_channels
        self.augmentation_config = augmentation_config or AugmentationConfig(
            augmentations_per_sample=0, 
            augmentations=[],
            aug_configs={}
        )
        
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
        
        # Initialize new attributes based on GeneralConfig
        self.val_size = general_config.val_size
        self.test_size = general_config.test_size
        self.inference_size = general_config.inference_size
        
        # Calculate train_size
        self.train_size = 1.0 - (self.val_size + self.test_size + self.inference_size)
        assert np.isclose(self.train_size + self.val_size + self.test_size + self.inference_size, 1.0), \
            "The sum of val_size, test_size, and inference_size should be less than 1.0"
        
    def prepare_data(self):
        """
        Prepare data for training.
        This method is called only once and on 1 GPU.
        """
        # Skip file checking for static datasets
        if "static" in self.data_path:
            return
        
        
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
        # Check if we're trying to save dataloaders with k-fold, which is not supported
        if self.use_kfold and self.save_dataloader:
            raise ValueError("Saving dataloaders is not supported when using k-fold cross-validation. "
                             "Please set either use_kfold=False or save_dataloader=False in your configuration.")
        
        # Check if we should load from static dataset
        if "static" in self.data_path:
            # Check if we're trying to use k-fold with static dataset, which is not supported
            if self.use_kfold:
                raise ValueError("Loading from static dataset is not supported when using k-fold cross-validation. "
                                 "Please set use_kfold=False in your configuration when using a static dataset path.")
            
            # Check if we're trying to save dataloaders when loading from static dataset, which is redundant
            if self.save_dataloader:
                raise ValueError("Saving dataloaders is redundant when loading from a static dataset. "
                                 "Please set save_dataloader=False in your configuration when using a static dataset path.")
            
            ic(f"Loading from static dataset: {self.data_path}")
            self.load_dataloaders(self.data_path)
            return
            
        # Get all audio file paths
        all_paths = list(Path(self.data_path).glob("*/*.wav"))
        
        if self.use_kfold:
            self._setup_kfold(all_paths)
            # Don't save dataloaders when using k-fold
        else:
            self._setup_train_val_test(all_paths)
            
            if self.save_dataloader:
                self.save_dataloaders()
        # Save dataloaders if configured - MOVED AFTER dataset initialization
            
        # Set sizes based on the dataset split
        self.train_size = 1.0 - (self.val_size + self.test_size + self.inference_size)
        assert np.isclose(self.train_size + self.val_size + self.test_size + self.inference_size, 1.0), \
            "The sum of val_size, test_size, and inference_size should be less than 1.0"
        
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
            config=self.augmentation_config.aug_configs or {}
        )
        
        self.val_dataset = AudioDataset(
            data_path=self.data_path,
            data_paths=val_paths,
            feature_extractor=self.feature_extractor,
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels,
            config=self.augmentation_config.aug_configs or {}
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
        
        print(f"Datasets created in {dataset_init_time:.2f} seconds - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, "
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
                config=self.augmentation_config.aug_configs or {}
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
        if self.train_dataset is None:
            raise ValueError("Training dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers= True
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
            persistent_workers=True
        )
        
    def test_dataloader(self):
        """
        Get test dataloader.
        
        Returns:
            Test dataloader
        """
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
        
    def predict_dataloader(self):
        """
        Get inference dataloader.
        
        Returns:
            Inference dataloader
        """
        if self.inference_dataset is None:
            raise ValueError("Inference dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
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
            
        if self.fold_datasets is None:
            raise ValueError("Fold datasets are not initialized. Call setup() first.")
            
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

    def save_dataloaders(self, base_path: Optional[str] = None):
        """
        Save dataloaders to disk.
        
        Args:
            base_path: Base path to save dataloaders. If None, a default path will be used.
        
        Returns:
            str: Path where dataloaders were saved
        """
        if not self.save_dataloader:
            return None
            
        # Check if datasets are initialized
        if not self.use_kfold and (self.train_dataset is None or self.val_dataset is None):
            ic("Cannot save dataloaders: datasets not initialized yet")
            return None
        
        if self.use_kfold and not self.fold_datasets:
            ic("Cannot save dataloaders: fold datasets not initialized yet")
            return None
            
        ic("Saving the dataloaders, this might take a while...")
        
        # Use default path if none provided
        if base_path is None:
            base_path = '/app/src/datasets'
            # base_path = self.data_path
            
        
        # Create a distinct name based on configuration 
        distinct_name = f"/static/{self.num_classes}-augs-{self.augmentation_config.augmentations_per_sample}"
        
        # Add augmentations to the path string
        save_path = os.path.join(base_path, distinct_name.lstrip('/'))

        for aug in self.augmentation_config.augmentations:
            save_path += f"-{aug.replace(' ', '-')}"  # Remove white space from the string
            
        ic(f"Saving dataloaders to: {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save dataloaders
        if not self.use_kfold:
            # Save standard dataloaders
            train_loader = self.train_dataloader()
            val_loader = self.val_dataloader()
            test_loader = self.test_dataloader()
            inference_loader = self.predict_dataloader()
            
            torch.save(train_loader, f"{save_path}/train_dataloader.pth")
            torch.save(val_loader, f"{save_path}/val_dataloader.pth")
            torch.save(test_loader, f"{save_path}/test_dataloader.pth")
            torch.save(inference_loader, f"{save_path}/inference_dataloader.pth")
            
            # Create a serializable version of the augmentation config
            serializable_aug_config = {
                "augmentations_per_sample": self.augmentation_config.augmentations_per_sample,
                "augmentations": self.augmentation_config.augmentations,
                "aug_configs": {}
            }

            # metadata dict
            metadata = {
                "data_path": self.data_path,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "num_classes": self.num_classes,
                "train_size": self.train_size,
                "val_size": self.val_size,
                "test_size": self.test_size,
                "inference_size": self.inference_size,
                "seed": self.seed,
                "pin_memory": self.pin_memory
            }
            
            # Convert each Pydantic model in aug_configs to a dictionary
            if self.augmentation_config.aug_configs and self.augmentation_config.augmentations_per_sample > 0:
                for aug_name, aug_config in self.augmentation_config.aug_configs.items():
                    if hasattr(aug_config, "model_dump"):
                        # For newer Pydantic v2
                        serializable_aug_config["aug_configs"][aug_name] = aug_config.model_dump()
                    elif hasattr(aug_config, "dict"):
                        # For older Pydantic v1
                        serializable_aug_config["aug_configs"][aug_name] = aug_config.dict()
                    else:
                        # Fallback for non-Pydantic objects
                        serializable_aug_config["aug_configs"][aug_name] = vars(aug_config)
            
                # add the serializable augmentation config to the metadata
                metadata["augmentation_config"] = serializable_aug_config
            
            # ic(f"Metadata: {metadata}")
            with open(f"{save_path}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)  # Use indent for better formatting
        else:
            raise NotImplementedError("K-fold saving is not supported. Loaders will not be saved.")
            
        ic(f"Saved the dataloaders to: {save_path}")
        sys.exit(0)
        # return save_path
        
    def load_dataloaders(self, path: str):
        """
        Load dataloaders from disk.
        
        Args:
            path: Path to load dataloaders from
        """
        ic(f"Loading dataloaders from: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataloader path {path} does not exist")
            
        if not self.use_kfold:
            # Load standard dataloaders
            train_path = f"{path}/train_dataloader.pth"
            val_path = f"{path}/val_dataloader.pth"
            test_path = f"{path}/test_dataloader.pth"
            inference_path = f"{path}/inference_dataloader.pth"
            metadata_path = f"{path}/metadata.json"  # Path to the metadata file
            
            if not all(os.path.exists(p) for p in [train_path, val_path, test_path, inference_path, metadata_path]):
                raise FileNotFoundError(f"One or more dataloader files not found in {path}")

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # You can now access metadata['data_path'], metadata['batch_size'], etc.
            
            # Load dataloaders
            train_loader = torch.load(train_path, weights_only=False)
            val_loader = torch.load(val_path, weights_only=False)
            test_loader = torch.load(test_path, weights_only=False)
            inference_loader = torch.load(inference_path, weights_only=False)

            self.train_dataset = train_loader.dataset
            self.val_dataset = val_loader.dataset
            self.test_dataset = test_loader.dataset
            self.inference_dataset = inference_loader.dataset

            # Access instance variables
            train_samples = getattr(train_loader.dataset, "__len__", lambda: "unknown")()
            val_samples = getattr(val_loader.dataset, "__len__", lambda: "unknown")()
            test_samples = getattr(test_loader.dataset, "__len__", lambda: "unknown")()
            inference_samples = getattr(inference_loader.dataset, "__len__", lambda: "unknown")()
            augmentations_per_sample = train_loader.dataset.augmentations_per_sample  # Accessing instance variable
            
            # Sanity Data check
            
            # Comprehensive sanity checks for train data
            # Check if datasets exist and have proper length
            if train_samples <= 0:
                raise ValueError("Train dataset is empty")
            if train_samples < val_samples:
                raise ValueError("Train dataset is smaller than validation dataset")
            if not isinstance(train_samples, int):
                raise ValueError("Train samples count is not an integer")
                
            # Check if we can actually get a batch of data
            try:
                batch = next(iter(train_loader))
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise ValueError("Training batch should contain (data, labels)")
                if not torch.is_tensor(batch[0]) or not torch.is_tensor(batch[1]):
                    raise ValueError("Both data and labels should be torch tensors")
                if batch[0].dim() < 2:  # At least [batch_size, features]
                    raise ValueError("Input tensor has incorrect dimensions")
            except Exception as e:
                raise ValueError(f"Failed to load a batch from train_loader: {str(e)}")

            ic("Sanity check Passed \n\n")
            ic(f"Train loader samples: {train_samples}")
            ic(f"Val loader samples: {val_samples}")
            ic(f"Test loader samples: {test_samples}")
            ic(f"Inference loader samples: {inference_samples}")
            ic(f"Number of augmentations: {augmentations_per_sample}")

            return train_loader, val_loader, test_loader, inference_loader


# Example usage
def example_usage():
    """Example of how to use the AudioDataModule"""
    from configs.configs_demo import load_configs
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    general_config, feature_extraction_config, peft_config, wandb_config, sweep_config, augmentation_config = load_configs(config)
    print("_"*40+"\n")

    
    # Create data module directly from configs
    data_module = AudioDataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        augmentation_config=augmentation_config
    )
    ic("Created the audio data module")
    
    # Setup data module (this will also save dataloaders if save_dataloader is True)
    data_module.setup()
    ic("Setup the data module")
    
    # Get dataloaders
    try:
        if general_config.use_kfold:
            for fold in range(general_config.k_folds):
                fold_train_loader, fold_val_loader = data_module.get_fold_dataloaders(fold)
                # Use fold_train_loader and fold_val_loader for training
                train_samples = getattr(fold_train_loader.dataset, "__len__", lambda: "unknown")()
                val_samples = getattr(fold_val_loader.dataset, "__len__", lambda: "unknown")()
                ic(f"Fold {fold} train loader: {train_samples} samples")
                ic(f"Fold {fold} val loader: {val_samples} samples")
        elif "static" in general_config.data_path:
            ic("Loading from static dataloaders")
            data_module.load_dataloaders(general_config.data_path)
        else:
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            test_loader = data_module.test_dataloader()
            inference_loader = data_module.predict_dataloader()

            # Fixed: Check if dataset has __len__ before calling len()
            train_samples = getattr(train_loader.dataset, "__len__", lambda: "unknown")()
            val_samples = getattr(val_loader.dataset, "__len__", lambda: "unknown")()
            test_samples = getattr(test_loader.dataset, "__len__", lambda: "unknown")()
            inference_samples = getattr(inference_loader.dataset, "__len__", lambda: "unknown")()
            ic(f"Train loader: {train_samples} samples")
            ic(f"Val loader: {val_samples} samples")
            ic(f"Test loader: {test_samples} samples")
            ic(f"Inference loader: {inference_samples} samples")
            ic(f"number of augmentations: {train_loader.dataset.augmentations_per_sample}")
        
        # Get class information
        classes, class_to_idx, idx_to_class = data_module.get_class_info()
        print(f"Classes: {classes}")
        print(f"Number of classes: {len(classes)}")
        
    except Exception as e:
        ic(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    example_usage()


if __name__ == "__main__":
    main()


