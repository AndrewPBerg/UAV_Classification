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
import pandas as pd

# Import from the main codebase
sys.path.append(str(Path(__file__).parent.parent))

from helper.util import UAVDataset, wandb_login
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, BitImageProcessor

# Import Pydantic configs
from configs import AugConfig as AugmentationConfig
from configs import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig
from configs.dataset_config import UrbanSound8KConfig

# Import UrbanSound8K specific dataset and functions
from urbansound8k.urbansound8k_dataset import UrbanSound8KDataset, create_urbansound8k_fold_splits, create_urbansound8k_kfold_splits


class UrbanSound8KDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for UrbanSound8K audio data.
    
    Handles UrbanSound8K specific features:
    - 10-fold cross-validation using predefined UrbanSound8K folds
    - Metadata integration
    - Urban sound classification (10 classes)
    - No test/inference splits (uses k-fold cross-validation)
    """
    
    def __init__(
        self,
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        urbansound8k_config: UrbanSound8KConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
        feature_extractor: Optional[Any] = None,
        num_channels: int = 1,
        wandb_config: Optional[WandbConfig] = None,
        sweep_config: Optional[SweepConfig] = None
    ):
        """
        Initialize the UrbanSound8KDataModule using Pydantic config models.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            urbansound8k_config: UrbanSound8K specific configuration
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
        self.urbansound8k_config = urbansound8k_config
        self.augmentation_config = augmentation_config or AugmentationConfig(
            augmentations_per_sample=0, 
            augmentations=[],
            aug_configs={}
        )
        self.wandb_config = wandb_config
        self.sweep_config = sweep_config
        
        # Unpack general config
        self.data_path = urbansound8k_config.data_path
        self.batch_size = general_config.batch_size
        self.num_workers = general_config.num_cuda_workers
        self.seed = general_config.seed
        self.use_kfold = general_config.use_kfold
        self.k_folds = general_config.k_folds
        self.pin_memory = general_config.pinned_memory
        self.use_sweep = general_config.use_sweep
        self.sweep_count = general_config.sweep_count
        
        # UrbanSound8K specific parameters
        self.fold_based_split = urbansound8k_config.fold_based_split
        
        # Unpack feature extraction config
        self.target_sr = feature_extraction_config.sampling_rate
        self.target_duration = urbansound8k_config.target_duration
        self.num_channels = num_channels
        
        # Get number of classes
        self.num_classes = urbansound8k_config.get_num_classes()
        viable_classes = [10]
        if self.num_classes not in viable_classes:
            raise ValueError(f"Number of classes must be one of {viable_classes}")
        
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
        self.fold_datasets = None
        self.current_fold = 0
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate UrbanSound8K specific configuration."""
        # UrbanSound8K should always use fold-based splits
        if not self.fold_based_split:
            raise ValueError("UrbanSound8K dataset should use fold_based_split=True")
        
        # UrbanSound8K has exactly 10 folds
        if self.k_folds != 10:
            ic(f"Warning: UrbanSound8K has 10 predefined folds, but k_folds is set to {self.k_folds}. Using 10 folds.")
            self.k_folds = 10
        
        # UrbanSound8K doesn't use test/inference splits in k-fold mode
        if self.use_kfold:
            if hasattr(self.general_config, 'test_size') and self.general_config.test_size > 0:
                ic("Warning: UrbanSound8K k-fold cross-validation doesn't use separate test sets. test_size will be ignored.")
            if hasattr(self.general_config, 'inference_size') and self.general_config.inference_size > 0:
                ic("Warning: UrbanSound8K k-fold cross-validation doesn't use separate inference sets. inference_size will be ignored.")
        
    def prepare_data(self):
        """
        Prepare data for training.
        This method is called only once and on 1 GPU.
        """
        # Check if data path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"UrbanSound8K data path {self.data_path} does not exist")
            
        # Check if there are audio files
        all_paths = list(Path(self.data_path).glob("*/*.wav"))
        if len(all_paths) == 0:
            raise ValueError(f"No .wav files found in {self.data_path}")
        
        # Check if metadata exists (required for UrbanSound8K)
        data_path_obj = Path(self.data_path)
        possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
        
        meta_file_found = False
        for root in possible_roots:
            meta_file = root / "metadata" / "UrbanSound8K.csv"
            if meta_file.exists():
                meta_file_found = True
                break
        
        if not meta_file_found:
            raise FileNotFoundError(
                f"UrbanSound8K metadata file (UrbanSound8K.csv) not found in any of the expected locations: "
                f"{[str(root / 'metadata' / 'UrbanSound8K.csv') for root in possible_roots]}. "
                f"Metadata is required for UrbanSound8K fold splitting."
            )
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation using UrbanSound8K fold-based splits.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        start_time = timer()
        
        if self.use_kfold:
            # Setup for k-fold cross-validation
            self._setup_kfold()
        else:
            # Setup for single fold training using fold 1 as validation, rest as training
            self._setup_single_fold()
            
        end_time = timer()
        dataset_init_time = end_time - start_time
        
        ic(f"UrbanSound8K datasets created in {dataset_init_time:.2f} seconds")
    
    def _setup_kfold(self):
        """
        Setup datasets for k-fold cross validation using UrbanSound8K predefined folds.
        """
        ic("Setting up UrbanSound8K k-fold cross-validation")
        
        # Get all audio file paths
        data_path = self.data_path
        all_paths = list(Path(data_path).glob("*/*.wav"))
        if not all_paths:
            raise FileNotFoundError(f"No .wav files found in {data_path}")

        # Load metadata to get fold information
        data_path_obj = Path(data_path)
        possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
        
        metadata_df = None
        for root in possible_roots:
            meta_file = root / "metadata" / "UrbanSound8K.csv"
            if meta_file.exists():
                metadata_df = pd.read_csv(meta_file)
                break
        
        if metadata_df is None:
            raise FileNotFoundError("UrbanSound8K metadata file not found")

        # Initialize fold lists (UrbanSound8K has folds 1-10)
        fold_data_paths = {i: [] for i in range(1, 11)}
        
        # Process each file and sort into folds based on metadata
        for path in all_paths:
            filename = path.name
            
            # Find fold from metadata
            file_row = metadata_df[metadata_df['slice_file_name'] == filename]
            if len(file_row) > 0:
                fold_num = file_row.iloc[0]['fold']
                if fold_num < 1 or fold_num > 10:
                    raise ValueError(f"Invalid fold number {fold_num} for file {filename}")
                
                fold_data_paths[fold_num].append(path)
            else:
                ic(f"Warning: File {filename} not found in metadata, skipping")
        
        # Print fold statistics
        for fold_num in range(1, 11):
            print(f"Total files in fold_{fold_num}: {len(fold_data_paths[fold_num])}")
        
        self.fold_splits = []  # List[Dict[list[str], list[str]]]

        # Create a map of the fold splits (for each fold as validation)
        for val_fold in range(1, 11):  # UrbanSound8K has folds 1-10
            val_paths = fold_data_paths[val_fold]
            
            # Combine all other folds for training
            train_paths = []
            for fold_num in range(1, 11):
                if fold_num != val_fold:
                    train_paths.extend(fold_data_paths[fold_num])

            # Validation for UrbanSound8K (approximately 800-900 files per fold)
            if len(val_paths) < 700 or len(val_paths) > 1000:
                ic(f"Warning: Val paths for fold {val_fold} are len {len(val_paths)} (expected ~800-900)")

            if len(train_paths) < 7000 or len(train_paths) > 8000:
                ic(f"Warning: Train paths for fold {val_fold} are len {len(train_paths)} (expected ~7200-7300)")
            
            self.fold_splits.append({"train_paths": train_paths, "val_paths": val_paths})
        
    def _setup_single_fold(self):
        """
        Setup datasets for single fold training using UrbanSound8K predefined folds.
        Uses fold 1 as validation set and folds 2-10 as training set.
        """
        ic("Setting up UrbanSound8K single fold training")
        
        # Get all audio file paths
        data_path = self.data_path
        all_paths = list(Path(data_path).glob("*/*.wav"))
        if not all_paths:
            raise FileNotFoundError(f"No .wav files found in {data_path}")

        # Load metadata to get fold information
        data_path_obj = Path(data_path)
        possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
        
        metadata_df = None
        for root in possible_roots:
            meta_file = root / "metadata" / "UrbanSound8K.csv"
            if meta_file.exists():
                metadata_df = pd.read_csv(meta_file)
                break
        
        if metadata_df is None:
            raise FileNotFoundError("UrbanSound8K metadata file not found")

        # Initialize fold lists
        fold_data_paths = {i: [] for i in range(1, 11)}
        
        # Process each file and sort into folds based on metadata
        for path in all_paths:
            filename = path.name
            
            # Find fold from metadata
            file_row = metadata_df[metadata_df['slice_file_name'] == filename]
            if len(file_row) > 0:
                fold_num = file_row.iloc[0]['fold']
                fold_data_paths[fold_num].append(path)
        
        # Use fold 1 as validation, folds 2-10 as training
        val_paths = fold_data_paths[1]
        train_paths = []
        for fold_num in range(2, 11):
            train_paths.extend(fold_data_paths[fold_num])
        
        # Create datasets
        self.train_dataset = UrbanSound8KDataset(
            data_path=self.data_path,
            data_paths=train_paths,
            feature_extractor=self.feature_extractor,
            config=self.urbansound8k_config,
            target_sr=self.urbansound8k_config.target_sr,
            target_duration=self.urbansound8k_config.target_duration,
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            aug_config=self.augmentation_config,
            load_metadata=True
        )

        self.val_dataset = UrbanSound8KDataset(
            data_path=self.data_path,
            data_paths=val_paths,
            feature_extractor=self.feature_extractor,
            config=self.urbansound8k_config,
            target_sr=self.urbansound8k_config.target_sr,
            target_duration=self.urbansound8k_config.target_duration,
            augmentations_per_sample=0,  # No augmentations for validation
            augmentations=[],  # No augmentations for validation
            aug_config=self.augmentation_config,
            load_metadata=True
        )
        print(f"Single fold setup - Train files: {len(self.train_dataset)}, Val files: {len(self.val_dataset)}")

    def get_fold_dataloaders(self, fold_idx: int):
        """
        Get dataloaders for a specific fold.
        
        Args:
            fold_idx: Index of the fold (0-based)
            
        Returns:
            Tuple of (train dataloader, validation dataloader)
        """
        
        paths_obj = self.fold_splits[fold_idx]
        train_paths = paths_obj["train_paths"]
        val_paths = paths_obj["val_paths"]

        train_dataset = UrbanSound8KDataset(
            data_path=self.data_path,
            data_paths=train_paths,
            feature_extractor=self.feature_extractor,
            config=self.urbansound8k_config,
            target_sr=self.urbansound8k_config.target_sr,
            target_duration=self.urbansound8k_config.target_duration,
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            aug_config=self.augmentation_config,
            load_metadata=True
        )

        val_dataset = UrbanSound8KDataset(
            data_path=self.data_path,
            data_paths=val_paths,
            feature_extractor=self.feature_extractor,
            config=self.urbansound8k_config,
            target_sr=self.urbansound8k_config.target_sr,
            target_duration=self.urbansound8k_config.target_duration,
            augmentations_per_sample=0,  # No augmentations for validation
            augmentations=[],  # No augmentations for validation
            aug_config=self.augmentation_config,
            load_metadata=True
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return train_loader, val_loader
        
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
        UrbanSound8K doesn't use separate test sets in k-fold cross-validation.
        Returns validation dataloader for compatibility.
        """
        ic("Warning: UrbanSound8K uses k-fold cross-validation. Returning validation dataloader.")
        return self.val_dataloader()
        
    def predict_dataloader(self):
        """
        UrbanSound8K doesn't use separate inference sets in k-fold cross-validation.
        Returns validation dataloader for compatibility.
        """
        ic("Warning: UrbanSound8K uses k-fold cross-validation. Returning validation dataloader.")
        return self.val_dataloader()
        
    def get_all_fold_dataloaders(self):
        """
        Get dataloaders for all folds.
        
        Returns:
            List of (train dataloader, validation dataloader) tuples for each fold
        """
        if not self.use_kfold:
            raise ValueError("get_all_fold_dataloaders() can only be used when use_kfold=True")
            
        if self.fold_datasets is None:
            raise ValueError("Fold datasets are not initialized. Call setup() first.")
            
        fold_loaders = []
        for fold_idx in range(len(self.fold_datasets)):
            train_loader, val_loader = self.get_fold_dataloaders(fold_idx)
            fold_loaders.append((train_loader, val_loader))
            
        return fold_loaders
        
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
    
    def get_metadata_info(self):
        """
        Get UrbanSound8K metadata information.
        
        Returns:
            Dictionary with metadata statistics
        """
        if self.train_dataset and hasattr(self.train_dataset, 'metadata_df'):
            metadata_df = self.train_dataset.metadata_df
            if metadata_df is not None:
                info = {
                    "total_files": len(metadata_df),
                    "num_folds": len(metadata_df['fold'].unique()),
                    "folds": sorted(metadata_df['fold'].unique()),
                    "classes": sorted(metadata_df['class'].unique()) if 'class' in metadata_df.columns else [],
                    "dataset_name": "UrbanSound8K"
                }
                return info
        
        return {"error": "Metadata not available"}
    
    def get_fold_info(self):
        """
        Get information about current fold setup.
        
        Returns:
            Dictionary with fold information
        """
        info = {
            "use_kfold": self.use_kfold,
            "k_folds": self.k_folds,
            "current_fold": self.current_fold if self.use_kfold else None,
            "fold_based_split": self.fold_based_split
        }
        
        if self.fold_datasets:
            info["fold_sizes"] = []
            for i, (train_ds, val_ds) in enumerate(self.fold_datasets):
                train_size = getattr(train_ds, "__len__", lambda: 0)()
                val_size = getattr(val_ds, "__len__", lambda: 0)()
                info["fold_sizes"].append({
                    "fold": i + 1,
                    "train_size": train_size,
                    "val_size": val_size,
                    "total_size": train_size + val_size
                })
        
        return info


def create_urbansound8k_datamodule(
    general_config: GeneralConfig,
    feature_extraction_config: FeatureExtractionConfig,
    urbansound8k_config: UrbanSound8KConfig,
    augmentation_config: Optional[AugmentationConfig] = None,
    **kwargs
) -> UrbanSound8KDataModule:
    """
    Factory function to create UrbanSound8KDataModule.
    
    Args:
        general_config: General configuration
        feature_extraction_config: Feature extraction configuration
        urbansound8k_config: UrbanSound8K specific configuration
        augmentation_config: Augmentation configuration
        **kwargs: Additional arguments
        
    Returns:
        UrbanSound8KDataModule instance
    """
    return UrbanSound8KDataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        urbansound8k_config=urbansound8k_config,
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
        
        # Change datapath and dataset type for UrbanSound8K
        config['dataset']['dataset_type'] = 'urbansound8k'
        config['dataset']['data_path'] = "../datasets/UrbanSound8K/classes"
        config['dataset']['num_classes'] = 10
        config['dataset']['target_duration'] = 4
        config['general']['k_folds'] = 10
        
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
        data_module = create_urbansound8k_datamodule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            urbansound8k_config=dataset_config,  # dataset_config contains UrbanSound8KConfig
            augmentation_config=augmentation_config
        )
        print("successfully created the data module")
        
        # Setup data module
        data_module.setup()
        print("successfully setup the data module")
        
        # Example of getting dataloaders for first fold
        train_loader, val_loader = data_module.get_fold_dataloaders(0)
        ic(f"First fold - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Print dataset info after getting the fold_dataloaders
        info = data_module.get_class_info()
        ic("Dataset Info:", info)
        
    except Exception as e:
        ic(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    example_usage()


if __name__ == "__main__":
    main() 