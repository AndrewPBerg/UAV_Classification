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
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, BitImageProcessor

# Import Pydantic configs
from configs import AugConfig as AugmentationConfig
from configs import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig
from configs.dataset_config import ESC10Config

# Import ESC-10 specific dataset and functions
from esc10.esc10_dataset import ESC10Dataset, create_esc10_fold_splits, create_esc10_kfold_splits


class ESC10DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for ESC-10 audio data.
    
    Handles ESC-10 specific features:
    - Fold-based cross-validation using predefined ESC-10 folds (inherited from ESC-50)
    - 10 classes from the ESC-50 subset
    - Metadata integration
    - No test/inference splits (uses k-fold cross-validation)
    """
    
    def __init__(
        self,
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        esc10_config: ESC10Config,
        augmentation_config: Optional[AugmentationConfig] = None,
        feature_extractor: Optional[Any] = None,
        num_channels: int = 1,
        wandb_config: Optional[WandbConfig] = None,
        sweep_config: Optional[SweepConfig] = None,
        use_filename_based_splits: bool = True
    ):
        """
        Initialize the ESC10DataModule using Pydantic config models.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            esc10_config: ESC-10 specific configuration
            augmentation_config: Augmentation configuration
            feature_extractor: Optional pre-created feature extractor
            num_channels: Number of audio channels
            wandb_config: Optional WandB configuration for logging
            sweep_config: Optional sweep configuration for hyperparameter tuning
            use_filename_based_splits: Whether to use filename-based (faster) or metadata-based fold splitting
        """
        super().__init__()
        
        # Store configs
        self.general_config = general_config
        self.feature_extraction_config = feature_extraction_config
        self.esc10_config = esc10_config
        self.augmentation_config = augmentation_config or AugmentationConfig(
            augmentations_per_sample=0, 
            augmentations=[],
            aug_configs={}
        )
        self.wandb_config = wandb_config
        self.sweep_config = sweep_config
        self.use_filename_based_splits = use_filename_based_splits
        
        # Unpack general config
        self.data_path = esc10_config.data_path
        self.batch_size = general_config.batch_size
        self.num_workers = general_config.num_cuda_workers
        self.seed = general_config.seed
        self.use_kfold = general_config.use_kfold
        self.k_folds = general_config.k_folds
        self.pin_memory = general_config.pinned_memory
        self.use_sweep = general_config.use_sweep
        self.sweep_count = general_config.sweep_count
        
        # ESC-10 specific parameters
        self.fold_based_split = esc10_config.fold_based_split
        
        # Unpack feature extraction config
        self.target_sr = feature_extraction_config.sampling_rate
        self.target_duration = esc10_config.target_duration
        self.num_channels = num_channels
        
        # Get number of classes (ESC-10 always has 10 classes)
        self.num_classes = esc10_config.get_num_classes()
        if self.num_classes != 10:
            ic(f"Warning: ESC-10 should have 10 classes, but found {self.num_classes}")
        
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
        """Validate ESC-10 specific configuration."""
        # ESC-10 should always use fold-based splits
        if not self.fold_based_split:
            raise ValueError("ESC-10 dataset should use fold_based_split=True")
        
        # ESC-10 has exactly 5 folds (inherited from ESC-50)
        if self.k_folds != 5:
            ic(f"Warning: ESC-10 has 5 predefined folds, but k_folds is set to {self.k_folds}. Using 5 folds.")
            self.k_folds = 5
        
        # ESC-10 doesn't use test/inference splits in k-fold mode
        if self.use_kfold:
            if hasattr(self.general_config, 'test_size') and self.general_config.test_size > 0:
                ic("Warning: ESC-10 k-fold cross-validation doesn't use separate test sets. test_size will be ignored.")
            if hasattr(self.general_config, 'inference_size') and self.general_config.inference_size > 0:
                ic("Warning: ESC-10 k-fold cross-validation doesn't use separate inference sets. inference_size will be ignored.")
        
    def prepare_data(self):
        """
        Prepare data for training.
        This method is called only once and on 1 GPU.
        """
        # Check if data path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ESC-10 data path {self.data_path} does not exist")
            
        # Check if there are audio files
        all_paths = list(Path(self.data_path).glob("*/*.wav"))
        if len(all_paths) == 0:
            raise ValueError(f"No .wav files found in {self.data_path}")
        
        # Check if metadata exists (optional for ESC-10)
        need_metadata = not self.use_filename_based_splits
        
        if need_metadata:
            data_path_obj = Path(self.data_path)
            possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
            
            meta_file_found = False
            for root in possible_roots:
                meta_file = root / "meta" / "esc10.csv"
                if meta_file.exists():
                    meta_file_found = True
                    break
            
            if not meta_file_found:
                ic(f"ESC-10 metadata file (esc10.csv) not found, falling back to filename-based splits")
                self.use_filename_based_splits = True
        else:
            ic("Using filename-based splits - metadata CSV not required")
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation using ESC-10 fold-based splits.
        
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
        
        ic(f"ESC-10 datasets created in {dataset_init_time:.2f} seconds")
    
    def _setup_kfold(self):
        """
        Setup datasets for k-fold cross validation using ESC-10 predefined folds.
        """
        ic("Setting up ESC-10 k-fold cross-validation")
        
        # Get all audio file paths
        data_path = self.data_path
        all_paths = list(Path(data_path).glob("*/*.wav"))
        if not all_paths:
            raise FileNotFoundError(f"No .wav files found in {data_path}")

        fold_1 = []
        fold_2 = []
        fold_3 = []
        fold_4 = []
        fold_5 = []
        
        # Process each file and sort into folds based on filename
        for path in all_paths:
            filename = path.name
            
            # ESC-10 files have fold number as first character (1-5) inherited from ESC-50
            try:
                fold_num = int(filename[0])
                if fold_num < 1 or fold_num > 5:
                    raise ValueError(f"Invalid fold number {fold_num} for file {filename}")
            except (ValueError, IndexError) as e:
                raise ValueError(f"Cannot extract fold number from filename {filename}: {e}")
            
            # Add complete path to appropriate fold
            if fold_num == 1:
                fold_1.append(path)
            elif fold_num == 2:
                fold_2.append(path)
            elif fold_num == 3:
                fold_3.append(path)
            elif fold_num == 4:
                fold_4.append(path)
            elif fold_num == 5:
                fold_5.append(path)
                
        print(f"Total files in fold_1: {len(fold_1)}")
        print(f"Total files in fold_2: {len(fold_2)}")
        print(f"Total files in fold_3: {len(fold_3)}")
        print(f"Total files in fold_4: {len(fold_4)}")
        print(f"Total files in fold_5: {len(fold_5)}")
        
        # create a list of the folds
        fold_data_paths = [fold_1, fold_2, fold_3, fold_4, fold_5]

        self.fold_splits = []  # List[Dict[list[str], list[str]]]

        # create a map of the fold splits
        for i in range(5):  # should always be 5 folds
            fold_copy = fold_data_paths.copy()

            val_paths = fold_copy.pop(i)
            # need to break down the rest of fold_copy's iterations into a single list
            train_paths = []
            for fold in fold_copy:
                train_paths.extend(fold)

            # simple validation specific to esc10
            # Each fold in ESC-10 should have roughly the same number of files
            # (ESC-10 has 400 files total, so ~80 per fold)
            expected_val_size = len(all_paths) // 5
            if abs(len(val_paths) - expected_val_size) > expected_val_size * 0.2:  # Allow 20% variation
                ic(f"Warning: Val paths for fold {i} are len {len(val_paths)}, expected ~{expected_val_size}")

            expected_train_size = len(all_paths) - len(val_paths)
            if abs(len(train_paths) - expected_train_size) > expected_train_size * 0.1:  # Allow 10% variation
                ic(f"Warning: Train paths for fold {i} are len {len(train_paths)}, expected ~{expected_train_size}")
            
            self.fold_splits.append({"train_paths": train_paths, "val_paths": val_paths})
        
    def _setup_single_fold(self):
        """
        Setup datasets for single fold training using ESC-10 predefined folds.
        Uses fold 1 as validation set and folds 2-5 as training set.
        """
        ic("Setting up ESC-10 single fold training")
        
        # Reuse the fold splitting logic from _setup_kfold
        data_path = self.data_path
        all_paths = list(Path(data_path).glob("*/*.wav"))
        if not all_paths:
            raise FileNotFoundError(f"No .wav files found in {data_path}")

        fold_1 = []
        fold_2 = []
        fold_3 = []
        fold_4 = []
        fold_5 = []
        
        # Process each file and sort into folds based on filename
        for path in all_paths:
            filename = path.name
            
            # ESC-10 files have fold number as first character (1-5) inherited from ESC-50
            try:
                fold_num = int(filename[0])
                if fold_num < 1 or fold_num > 5:
                    raise ValueError(f"Invalid fold number {fold_num} for file {filename}")
            except (ValueError, IndexError) as e:
                raise ValueError(f"Cannot extract fold number from filename {filename}: {e}")
            
            # Add complete path to appropriate fold
            if fold_num == 1:
                fold_1.append(path)
            elif fold_num == 2:
                fold_2.append(path)
            elif fold_num == 3:
                fold_3.append(path)
            elif fold_num == 4:
                fold_4.append(path)
            elif fold_num == 5:
                fold_5.append(path)
        
        # Use fold 1 as validation, folds 2-5 as training
        val_paths = fold_1
        train_paths = fold_2 + fold_3 + fold_4 + fold_5
        
        # Create datasets
        self.train_dataset = ESC10Dataset(
            data_path=self.data_path,
            data_paths=train_paths,
            feature_extractor=self.feature_extractor,
            config=self.esc10_config,
            target_sr=self.esc10_config.target_sr,
            target_duration=self.esc10_config.target_duration,
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            aug_config=self.augmentation_config,
        )

        self.val_dataset = ESC10Dataset(
            data_path=self.data_path,
            data_paths=val_paths,
            feature_extractor=self.feature_extractor,
            config=self.esc10_config,
            target_sr=self.esc10_config.target_sr,
            target_duration=self.esc10_config.target_duration,
            augmentations_per_sample=0,  # No augmentations for validation
            augmentations=[],  # No augmentations for validation
            aug_config=self.augmentation_config,
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

        train_dataset = ESC10Dataset(
            data_path=self.data_path,
            data_paths=train_paths,
            feature_extractor=self.feature_extractor,
            config=self.esc10_config,
            target_sr=self.esc10_config.target_sr,
            target_duration=self.esc10_config.target_duration,
            augmentations_per_sample=self.augmentation_config.augmentations_per_sample,
            augmentations=self.augmentation_config.augmentations,
            aug_config=self.augmentation_config,
        )

        val_dataset = ESC10Dataset(
            data_path=self.data_path,
            data_paths=val_paths,
            feature_extractor=self.feature_extractor,
            config=self.esc10_config,
            target_sr=self.esc10_config.target_sr,
            target_duration=self.esc10_config.target_duration,
            augmentations_per_sample=0,  # No augmentations for validation
            augmentations=[],  # No augmentations for validation
            aug_config=self.augmentation_config,
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
        ESC-10 doesn't use separate test sets in k-fold cross-validation.
        Returns validation dataloader for compatibility.
        """
        ic("Warning: ESC-10 uses k-fold cross-validation. Returning validation dataloader.")
        return self.val_dataloader()
        
    def predict_dataloader(self):
        """
        ESC-10 doesn't use separate inference sets in k-fold cross-validation.
        Returns validation dataloader for compatibility.
        """
        ic("Warning: ESC-10 uses k-fold cross-validation. Returning validation dataloader.")
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
        Get ESC-10 metadata information.
        
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
                    "dataset_type": "ESC-10"
                }
                return info
        
        return {"error": "Metadata not available", "dataset_type": "ESC-10"}
    
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
            "fold_based_split": self.fold_based_split,
            "dataset_type": "ESC-10"
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


def create_esc10_datamodule(
    general_config: GeneralConfig,
    feature_extraction_config: FeatureExtractionConfig,
    esc10_config: ESC10Config,
    augmentation_config: Optional[AugmentationConfig] = None,
    use_filename_based_splits: bool = True,
    **kwargs
) -> ESC10DataModule:
    """
    Factory function to create ESC10DataModule.
    
    Args:
        general_config: General configuration
        feature_extraction_config: Feature extraction configuration
        esc10_config: ESC-10 specific configuration
        augmentation_config: Augmentation configuration
        use_filename_based_splits: Whether to use filename-based (faster) or metadata-based fold splitting
        **kwargs: Additional arguments
        
    Returns:
        ESC10DataModule instance
    """
    return ESC10DataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        esc10_config=esc10_config,
        augmentation_config=augmentation_config,
        use_filename_based_splits=use_filename_based_splits,
        **kwargs
    )


# Example usage
def example_usage():

    from configs.configs_aggregate import load_configs
    import yaml
    try:
        # Load configs from config.yaml
        print("Loading configs from config.yaml")
        with open('configs/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Change dataset type and datapath for ESC-10
        config['dataset']['dataset_type'] = 'esc10'
        config['dataset']['data_path'] = "datasets/ESC-10-master/classes"
        # Enable k-fold for testing
        config['general']['use_kfold'] = True
        print("Loading all configs using the configs_aggregate function")
        
        # Load all configs using the configs_aggregate function
        (
            general_config,
            feature_extraction_config,
            dataset_config,
            peft_config,
            wandb_config,
            sweep_config,
            augmentation_config,
            optimizer_config
        ) = load_configs(config)
        print("successfully loaded all configs")
        
        # Create data module
        data_module = create_esc10_datamodule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            esc10_config=dataset_config,  # dataset_config contains ESC10Config
            augmentation_config=augmentation_config,
            use_filename_based_splits=True
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