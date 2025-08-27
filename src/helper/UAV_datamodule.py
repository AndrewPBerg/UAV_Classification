import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any, Callable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from time import time as timer
from icecream import ic
import json
import wandb
import yaml

# Import from existing code
from .util import UAVDataset, wandb_login
from .cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, BitImageProcessor

# Import Pydantic configs
from configs import AugConfig as AugmentationConfig
from configs import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig
from configs.dataset_config import UAVConfig


class UAVDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for UAV audio data.
    """
    def __init__(
        self,
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        uav_config: UAVConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
        feature_extractor: Optional[Any] = None,
        num_channels: int = 1,
        wandb_config: Optional[WandbConfig] = None,
        sweep_config: Optional[SweepConfig] = None
    ):
        """
        Initialize the UAVDataModule using Pydantic config models.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            uav_config: UAV specific configuration
            augmentation_config: Augmentation configuration
            feature_extractor: Optional pre-created feature extractor (if None, will be created based on config)
            num_channels: Number of audio channels
            wandb_config: Optional WandB configuration for logging
            sweep_config: Optional sweep configuration for hyperparameter tuning
        """
        super().__init__()
        
        # Store configs
        self.general_config = general_config
        self.feature_extraction_config = feature_extraction_config
        self.uav_config = uav_config
        self.augmentation_config = augmentation_config or AugmentationConfig(
            augmentations_per_sample=0, 
            augmentations=[],
            aug_configs={}
        )
        self.wandb_config = wandb_config
        self.sweep_config = sweep_config
        
        # Unpack general config
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
        self.use_sweep = general_config.use_sweep
        self.sweep_count = general_config.sweep_count
        
        # Unpack UAV config
        self.data_path = uav_config.data_path
        self.num_classes = uav_config.get_num_classes()
        self.target_duration = uav_config.target_duration
        
        # Unpack feature extraction config
        self.target_sr = feature_extraction_config.sampling_rate
        
        # Set other parameters
        self.num_channels = num_channels
        
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
        
        self.train_dataset = UAVDataset(
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
        
        self.val_dataset = UAVDataset(
            data_path=self.data_path,
            data_paths=val_paths,
            feature_extractor=self.feature_extractor,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels,
        )
        
        self.test_dataset = UAVDataset(
            data_path=self.data_path,
            data_paths=test_paths,
            feature_extractor=self.feature_extractor,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels
        )
        
        self.inference_dataset = UAVDataset(
            data_path=self.data_path,
            data_paths=inference_paths,
            feature_extractor=self.feature_extractor,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            num_channels=self.num_channels
        )
        
        end_time = timer()
        dataset_init_time = end_time - start_time
        
        print(f"UAV datasets created in {dataset_init_time:.2f} seconds - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, "
           f"Test: {len(self.test_dataset)}, Inference: {len(self.inference_dataset)}")
            
    def _setup_kfold(self, all_paths: List[Path]):
        """
        Setup datasets for k-fold cross validation.
        
        Args:
            all_paths: List of paths to all audio files
        """
        n_samples = len(all_paths)
        # Create random state for reproducibility
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n_samples)
        
        if self.general_config.inference_size > 0:
            n_inference = int(n_samples * self.inference_size)
            
            # Split off inference set
            train_val_indices = indices[:-n_inference]
            inference_indices = indices[-n_inference:]
    
        else:
            n_inference = 0
            train_val_indices = indices
            inference_indices = []
        
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
            train_dataset = UAVDataset(
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
            
            val_dataset = UAVDataset(
                data_path=self.data_path,
                data_paths=fold_val_paths,
                feature_extractor=self.feature_extractor,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                num_channels=self.num_channels
            )
            
            self.fold_datasets.append((train_dataset, val_dataset))
            ic(f"UAV Fold {fold+1} datasets loaded")
        
        # Create inference dataset
        if inference_indices:
            ic(f"Creating UAV Inference dataset with {len(inference_indices)} samples")
            inference_paths = [str(all_paths[i]) for i in inference_indices]
            self.inference_dataset = UAVDataset(
                data_path=self.data_path,
                data_paths=inference_paths,
                feature_extractor=self.feature_extractor,
                target_sr=self.target_sr,
                target_duration=self.target_duration,
                num_channels=self.num_channels
            )
            ic(f"UAV Inference dataset loaded - {len(self.inference_dataset)} samples")
        else:
            ic("No inference dataset created as inference_size is 0")
        
        end_time = timer()
        dataset_init_time = end_time - start_time
        
        ic(f"UAV K-fold datasets created in {dataset_init_time:.2f} seconds")
        
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
            
        ic("Saving the UAV dataloaders, this might take a while...")
        
        # Use default path if none provided
        if base_path is None:
            base_path = '/app/src/datasets'
            # base_path = self.data_path
            
        
        # Create a distinct name based on configuration 
        distinct_name = f"/static/UAV-{self.num_classes}-augs-{self.augmentation_config.augmentations_per_sample}"
        
        # Add augmentations to the path string
        save_path = os.path.join(base_path, distinct_name.lstrip('/'))

        for aug in self.augmentation_config.augmentations:
            save_path += f"-{aug.replace(' ', '-')}"  # Remove white space from the string
            
        ic(f"Saving UAV dataloaders to: {save_path}")
        
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
                "dataset_type": "uav",
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
            
        ic(f"Saved the UAV dataloaders to: {save_path}")
        sys.exit(0)
        # return save_path
        
    def load_dataloaders(self, path: str):
        """
        Load dataloaders from disk.
        
        Args:
            path: Path to load dataloaders from
        """
        ic(f"Loading UAV dataloaders from: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataloader path {path} does not exist")
            
        if not self.use_kfold:
            # Load standard dataloaders
            train_path = f"{path}/train_dataloader.pth"
            val_path = f"{path}/val_dataloader.pth"
            test_path = f"{path}/test_dataloader.pth"
            inference_path = f"{path}/inference_dataloader.pth"
            metadata_path = f"{path}/metadata.json"  # Path to the metadata file
            
            if not all(os.path.exists(p) for p in [train_path, val_path, test_path, inference_path]):
                raise FileNotFoundError(f"One or more dataloader files not found in {path}")
            
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
            if isinstance(train_samples, int) and train_samples <= 0:
                raise ValueError("Train dataset is empty")
            if isinstance(train_samples, int) and isinstance(val_samples, int) and train_samples < val_samples:
                raise ValueError("Train dataset is smaller than validation dataset")
            if not isinstance(train_samples, (int, str)):
                raise ValueError("Train samples count is not an integer or 'unknown'")
                
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

            ic("UAV Sanity check Passed \n\n")
            ic(f"UAV Train loader samples: {train_samples}")
            ic(f"UAV Val loader samples: {val_samples}")
            ic(f"UAV Test loader samples: {test_samples}")
            ic(f"UAV Inference loader samples: {inference_samples}")
            ic(f"UAV Number of augmentations: {augmentations_per_sample}")

            return train_loader, val_loader, test_loader, inference_loader
        
        else:
            raise NotImplementedError("K-fold loading is not supported. Loaders will not be loaded.")

    def run_sweep(self, model_pipeline_fn: Callable, config_path: str = 'configs/config.yaml'):
        """
        Run a wandb sweep using the provided model pipeline function.
        
        Args:
            model_pipeline_fn: Function that takes a sweep configuration and trains a model
            config_path: Path to the configuration file
        """
        if not self.use_sweep:
            ic("Sweeps are not enabled in the configuration. Set use_sweep=True to enable sweeps.")
            return
        
        if self.wandb_config is None or self.sweep_config is None:
            ic("WandB or sweep configuration is missing. Make sure both are provided.")
            return
        
        # Login to wandb
        wandb_login()
        
        # Load the full configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Get sweep configuration
        sweep_config = config.get('sweep', {})
        if not sweep_config:
            ic("Sweep configuration is empty. Check your config file.")
            return
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=self.wandb_config.project
        )
        
        # Run sweep agent
        wandb.agent(
            sweep_id,
            function=model_pipeline_fn,
            count=self.sweep_count
        )
    
    def get_mixed_params(self, sweep_config: Dict[str, Any], general_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine sweep configuration with general configuration.
        
        Args:
            sweep_config: Sweep configuration from wandb
            general_config: General configuration from YAML file
            
        Returns:
            Combined configuration dictionary
        """
        mixed_params = {}
        
        # Add all general config parameters first
        for key, value in general_config.items():
            mixed_params[key] = value
        
        # Override with sweep config parameters
        for key, value in sweep_config.items():
            mixed_params[key] = value
        
        return mixed_params
    
    def create_sweep_pipeline(self, trainer_factory: Callable, config_path: str = 'configs/config.yaml'):
        """
        Create a model pipeline function for wandb sweep.
        
        Args:
            trainer_factory: Function that creates a trainer instance
            config_path: Path to the configuration file
            
        Returns:
            Function that can be used as a sweep agent
        """
        def model_pipeline(sweep_config=None):
            """
            Pipeline function for wandb sweep agent.
            
            Args:
                sweep_config: Configuration provided by wandb sweep
            """
            # Initialize wandb run
            with wandb.init(config=sweep_config):
                # Get the sweep configuration
                config = wandb.config
                
                # Load general config
                with open(config_path, 'r') as file:
                    yaml_config = yaml.safe_load(file)
                general_config_dict = yaml_config['general']
                
                # Combine sweep config with general config
                mixed_params = self.get_mixed_params(dict(config), general_config_dict)
                
                # Create trainer with the mixed parameters
                trainer = trainer_factory(mixed_params)
                
                # Train model and get results
                results = trainer.train()
                
                # Log results to wandb
                # The train method now returns a dictionary directly
                if isinstance(results, dict):
                    wandb.log(results)
                else:
                    print(f"Warning: Unexpected results format: {type(results)}. Cannot log to wandb.")
                
                return results
        
        return model_pipeline


def create_uav_datamodule(
    general_config: GeneralConfig,
    feature_extraction_config: FeatureExtractionConfig,
    uav_config: UAVConfig,
    augmentation_config: Optional[AugmentationConfig] = None,
    **kwargs
) -> UAVDataModule:
    """
    Factory function to create UAVDataModule.
    
    Args:
        general_config: General configuration
        feature_extraction_config: Feature extraction configuration
        uav_config: UAV specific configuration
        augmentation_config: Augmentation configuration
        **kwargs: Additional arguments
        
    Returns:
        UAVDataModule instance
    """
    return UAVDataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        uav_config=uav_config,
        augmentation_config=augmentation_config,
        **kwargs
    ) 