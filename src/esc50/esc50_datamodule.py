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
# import pandas as pd

# Import from the main codebase
sys.path.append(str(Path(__file__).parent.parent / "src"))

from helper.util import UAVDataset, wandb_login
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, BitImageProcessor

# Import Pydantic configs
from configs import AugConfig as AugmentationConfig
from configs import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig
from configs.dataset_config import ESC50Config

# Import ESC-50 specific dataset and functions
from .esc50_dataset import ESC50Dataset, create_esc50_fold_splits, create_esc50_kfold_splits


class ESC50DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for ESC-50 audio data.
    
    Handles ESC-50 specific features:
    - Fold-based cross-validation using predefined ESC-50 folds
    - ESC-10 subset filtering
    - Metadata integration
    - No test/inference splits (uses k-fold cross-validation)
    """
    
    def __init__(
        self,
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        esc50_config: ESC50Config,
        augmentation_config: Optional[AugmentationConfig] = None,
        feature_extractor: Optional[Any] = None,
        num_channels: int = 1,
        wandb_config: Optional[WandbConfig] = None,
        sweep_config: Optional[SweepConfig] = None
    ):
        """
        Initialize the ESC50DataModule using Pydantic config models.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            esc50_config: ESC-50 specific configuration
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
        self.esc50_config = esc50_config
        self.augmentation_config = augmentation_config or AugmentationConfig(
            augmentations_per_sample=0, 
            augmentations=[],
            aug_configs={}
        )
        self.wandb_config = wandb_config
        self.sweep_config = sweep_config
        
        # Unpack general config
        self.data_path = esc50_config.data_path
        self.batch_size = general_config.batch_size
        self.num_workers = general_config.num_cuda_workers
        self.seed = general_config.seed
        self.use_kfold = general_config.use_kfold
        self.k_folds = general_config.k_folds
        self.pin_memory = general_config.pinned_memory
        self.use_sweep = general_config.use_sweep
        self.sweep_count = general_config.sweep_count
        
        # ESC-50 specific parameters
        self.use_esc10_subset = esc50_config.use_esc10_subset
        self.fold_based_split = esc50_config.fold_based_split
        
        # Unpack feature extraction config
        self.target_sr = feature_extraction_config.sampling_rate
        self.target_duration = esc50_config.target_duration
        self.num_channels = num_channels
        
        # Get number of classes
        self.num_classes = esc50_config.get_num_classes()
        viable_classes = [50,10]
        if self.num_classes not in viable_classes:
            raise ValueError(f"Number of classes must be one of {viable_classes}")
        if self.use_esc10_subset:
            self.num_classes = 10  # ESC-10 subset has 10 classes
        
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
        """Validate ESC-50 specific configuration."""
        # ESC-50 should always use fold-based splits
        if not self.fold_based_split:
            raise ValueError("ESC-50 dataset should use fold_based_split=True")
        
        # ESC-50 has exactly 5 folds
        if self.k_folds != 5:
            ic(f"Warning: ESC-50 has 5 predefined folds, but k_folds is set to {self.k_folds}. Using 5 folds.")
            self.k_folds = 5
        
        # ESC-50 doesn't use test/inference splits in k-fold mode
        if self.use_kfold:
            if hasattr(self.general_config, 'test_size') and self.general_config.test_size > 0:
                ic("Warning: ESC-50 k-fold cross-validation doesn't use separate test sets. test_size will be ignored.")
            if hasattr(self.general_config, 'inference_size') and self.general_config.inference_size > 0:
                ic("Warning: ESC-50 k-fold cross-validation doesn't use separate inference sets. inference_size will be ignored.")
        
    def prepare_data(self):
        """
        Prepare data for training.
        This method is called only once and on 1 GPU.
        """
        # Check if data path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ESC-50 data path {self.data_path} does not exist")
            
        # Check if there are audio files
        all_paths = list(Path(self.data_path).glob("*/*.wav"))
        if len(all_paths) == 0:
            raise ValueError(f"No .wav files found in {self.data_path}")
        
        # Check if metadata exists
        data_path_obj = Path(self.data_path)
        possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
        
        meta_file_found = False
        for root in possible_roots:
            meta_file = root / "meta" / "esc50.csv"
            if meta_file.exists():
                meta_file_found = True
                break
        
        if not meta_file_found:
            raise FileNotFoundError(
                f"ESC-50 metadata file (esc50.csv) not found in any of the expected locations: "
                f"{[str(root / 'meta' / 'esc50.csv') for root in possible_roots]}"
            )
            
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation using ESC-50 fold-based splits.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        start_time = timer()
        
        if self.use_kfold:
            self._setup_kfold()
        else:
            # Single fold setup (use fold 5 as validation by default)
            self._setup_single_fold(val_fold=5)
            
        end_time = timer()
        dataset_init_time = end_time - start_time
        
        ic(f"ESC-50 datasets created in {dataset_init_time:.2f} seconds")
        
    def _setup_single_fold(self, val_fold: int = 5):
        """
        Setup datasets for single fold split.
        
        Args:
            val_fold: Fold number to use for validation (1-5)
        """
        ic(f"Setting up ESC-50 single fold split with validation fold {val_fold}")
        
        train_dataset, val_dataset = create_esc50_fold_splits(
            data_path=self.data_path,
            feature_extractor=self.feature_extractor,
            config=self.esc50_config,
            val_fold=val_fold,
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            aug_config=self.augmentation_config
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        ic(f"Single fold setup complete - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
    def _setup_kfold(self):
        """
        Setup datasets for k-fold cross validation using ESC-50 predefined folds.
        """
        ic("Setting up ESC-50 k-fold cross-validation")
        
        self.fold_datasets = create_esc50_kfold_splits(
            data_path=self.data_path,
            feature_extractor=self.feature_extractor,
            config=self.esc50_config,
            k_folds=self.k_folds,
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            aug_config=self.augmentation_config
        )
        
        ic(f"K-fold setup complete - {len(self.fold_datasets)} folds created")
        
        # Set the first fold as default
        self.set_fold(0)
        
    def set_fold(self, fold_idx: int):
        """
        Set the current fold for training and validation.
        
        Args:
            fold_idx: Index of the fold (0-based)
        """
        if not self.use_kfold:
            raise ValueError("set_fold() can only be used when use_kfold=True")
            
        if self.fold_datasets is None:
            raise ValueError("Fold datasets are not initialized. Call setup() first.")
            
        if fold_idx < 0 or fold_idx >= len(self.fold_datasets):
            raise ValueError(f"Fold index {fold_idx} out of range (0-{len(self.fold_datasets)-1})")
            
        self.current_fold = fold_idx
        self.train_dataset, self.val_dataset = self.fold_datasets[fold_idx]
        
        ic(f"Set to fold {fold_idx+1}: Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
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
        ESC-50 doesn't use separate test sets in k-fold cross-validation.
        Returns validation dataloader for compatibility.
        """
        ic("Warning: ESC-50 uses k-fold cross-validation. Returning validation dataloader.")
        return self.val_dataloader()
        
    def predict_dataloader(self):
        """
        ESC-50 doesn't use separate inference sets in k-fold cross-validation.
        Returns validation dataloader for compatibility.
        """
        ic("Warning: ESC-50 uses k-fold cross-validation. Returning validation dataloader.")
        return self.val_dataloader()
        
    def get_fold_dataloaders(self, fold_idx: int):
        """
        Get dataloaders for a specific fold.
        
        Args:
            fold_idx: Index of the fold (0-based)
            
        Returns:
            Tuple of (train dataloader, validation dataloader)
        """
        if not self.use_kfold:
            raise ValueError("get_fold_dataloaders() can only be used when use_kfold=True")
            
        if self.fold_datasets is None:
            raise ValueError("Fold datasets are not initialized. Call setup() first.")
            
        if fold_idx < 0 or fold_idx >= len(self.fold_datasets):
            raise ValueError(f"Fold index {fold_idx} out of range (0-{len(self.fold_datasets)-1})")
            
        train_dataset, val_dataset = self.fold_datasets[fold_idx]
        
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
        Get ESC-50 metadata information.
        
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
                    "categories": sorted(metadata_df['category'].unique()) if 'category' in metadata_df.columns else [],
                    "esc10_files": len(metadata_df[metadata_df['esc10'] == True]) if 'esc10' in metadata_df.columns else 0,
                    "use_esc10_subset": self.use_esc10_subset
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


def create_esc50_datamodule(
    general_config: GeneralConfig,
    feature_extraction_config: FeatureExtractionConfig,
    esc50_config: ESC50Config,
    augmentation_config: Optional[AugmentationConfig] = None,
    **kwargs
) -> ESC50DataModule:
    """
    Factory function to create ESC50DataModule.
    
    Args:
        general_config: General configuration
        feature_extraction_config: Feature extraction configuration
        esc50_config: ESC-50 specific configuration
        augmentation_config: Augmentation configuration
        **kwargs: Additional arguments
        
    Returns:
        ESC50DataModule instance
    """
    return ESC50DataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        esc50_config=esc50_config,
        augmentation_config=augmentation_config,
        **kwargs
    )


# Example usage
def example_usage():
    """Example of how to use the ESC50DataModule"""
    from configs import load_configs
    from configs.dataset_config import ESC50Config
    import yaml
    
    # Load configuration
    config_path = 'config_esc50_example.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load configs (assuming load_configs is updated to handle dataset config)
    general_config, feature_extraction_config, dataset_config, peft_config, wandb_config, sweep_config, augmentation_config = load_configs(config)
    
    # Ensure we have ESC50Config
    if not isinstance(dataset_config, ESC50Config):
        esc50_config = ESC50Config(**config['dataset'])
    else:
        esc50_config = dataset_config
    
    print("="*50)
    print("ESC-50 DataModule Example")
    print("="*50)
    
    # Create data module
    data_module = ESC50DataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        esc50_config=esc50_config,
        augmentation_config=augmentation_config
    )
    ic("Created ESC-50 data module")
    
    # Setup data module
    data_module.setup()
    ic("Setup ESC-50 data module")
    
    # Get metadata and fold information
    metadata_info = data_module.get_metadata_info()
    fold_info = data_module.get_fold_info()
    
    print(f"Metadata info: {metadata_info}")
    print(f"Fold info: {fold_info}")
    
    # Test dataloaders
    try:
        if general_config.use_kfold:
            ic("Testing k-fold cross-validation")
            
            # Test all folds
            for fold_idx in range(data_module.k_folds):
                data_module.set_fold(fold_idx)
                
                train_loader = data_module.train_dataloader()
                val_loader = data_module.val_dataloader()
                
                # Get a sample batch
                train_batch = next(iter(train_loader))
                val_batch = next(iter(val_loader))
                
                ic(f"Fold {fold_idx+1}: Train batch shape: {train_batch[0].shape}, Val batch shape: {val_batch[0].shape}")
                ic(f"Fold {fold_idx+1}: Train samples: {getattr(train_loader.dataset, '__len__', lambda: 'unknown')()}, Val samples: {getattr(val_loader.dataset, '__len__', lambda: 'unknown')()}")
        else:
            ic("Testing single fold setup")
            
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            
            # Get a sample batch
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            
            ic(f"Train batch shape: {train_batch[0].shape}, Val batch shape: {val_batch[0].shape}")
            ic(f"Train samples: {getattr(train_loader.dataset, '__len__', lambda: 'unknown')()}, Val samples: {getattr(val_loader.dataset, '__len__', lambda: 'unknown')()}")
        
        # Get class information
        classes, class_to_idx, idx_to_class = data_module.get_class_info()
        print(f"Classes: {classes[:10]}...")  # Show first 10 classes
        print(f"Number of classes: {len(classes)}")
        
        print("✅ ESC-50 DataModule test completed successfully!")
        
    except Exception as e:
        ic(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    example_usage()


if __name__ == "__main__":
    main() 