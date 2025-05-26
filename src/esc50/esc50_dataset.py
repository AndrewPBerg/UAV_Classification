import os
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Tuple
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, ViTImageProcessor

# Import from the main codebase
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from helper.util import UAVDataset, find_classes
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from configs.dataset_config import ESC50Config
from configs.augmentation_config import AugmentationConfig, create_augmentation_configs

class ESC50Dataset(UAVDataset):
    """
    ESC-50 specific dataset class that extends UAVDataset.
    
    Handles ESC-50 specific features:
    - Fold-based cross-validation splits
    - ESC-10 subset filtering
    - Metadata integration
    """
    
    def __init__(self,
                 data_path: str,
                 data_paths: List[str],
                 feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor, ViTImageProcessor],
                 config: Optional[ESC50Config] = None,
                 standardize_audio_boolean: bool = True,
                 target_sr: int = 16000,
                 target_duration: int = 5,
                 augmentations_per_sample: int = 0,
                 augmentations: List[str] = [],
                 num_channels: int = 1,
                 aug_config: Optional[Union[dict, AugmentationConfig]] = None) -> None:
        
        self.esc50_config = config
        self.metadata_df = None
        self.esc50_root = None
        
        # Load ESC-50 metadata if available
        self._load_metadata(data_path)
        
        # Filter for ESC-10 subset if requested
        if self.esc50_config and self.esc50_config.use_esc10_subset:
            data_paths = self._filter_esc10_subset(data_paths)
        
        # Convert AugmentationConfig to dict if needed
        if isinstance(aug_config, AugmentationConfig):
            aug_config_dict = aug_config.aug_configs
        else:
            aug_config_dict = aug_config or {}
        
        # Initialize parent UAVDataset
        super().__init__(
            data_path=data_path,
            data_paths=data_paths,
            feature_extractor=feature_extractor,
            standardize_audio_boolean=standardize_audio_boolean,
            target_sr=target_sr,
            target_duration=target_duration,
            augmentations_per_sample=augmentations_per_sample,
            augmentations=augmentations,
            num_channels=num_channels,
            config=aug_config_dict
        )
    
    def _load_metadata(self, data_path: str) -> None:
        """Load ESC-50 metadata CSV file."""
        data_path_obj = Path(data_path)
        
        # Try to find the ESC-50 root directory and metadata
        possible_roots = [
            data_path_obj,
            data_path_obj.parent,
            data_path_obj.parent.parent
        ]
        
        for root in possible_roots:
            meta_file = root / "meta" / "esc50.csv"
            if meta_file.exists():
                self.metadata_df = pd.read_csv(meta_file)
                self.esc50_root = root
                print(f"Loaded ESC-50 metadata from: {meta_file}")
                return
        
        # Atomic validation: fail if metadata not found
        raise FileNotFoundError(
            f"ESC-50 metadata file (esc50.csv) not found in any of the expected locations: "
            f"{[str(root / 'meta' / 'esc50.csv') for root in possible_roots]}"
        )
    
    def _filter_esc10_subset(self, data_paths: List[str]) -> List[str]:
        """Filter data paths to include only ESC-10 subset."""
        # Atomic validation: fail if metadata not available when ESC-10 filtering is requested
        if self.metadata_df is None:
            raise ValueError("Cannot filter ESC-10 subset without metadata. ESC-50 metadata file not found.")
        
        # Get ESC-10 filenames from metadata
        esc10_files = set(self.metadata_df[self.metadata_df['esc10'] == True]['filename'].tolist())
        
        if not esc10_files:
            raise ValueError("No ESC-10 files found in metadata. Check metadata file integrity.")
        
        # Filter data paths
        filtered_paths = []
        for path in data_paths:
            filename = Path(path).name
            if filename in esc10_files:
                filtered_paths.append(path)
        
        if not filtered_paths:
            raise ValueError(
                f"No ESC-10 files found in the provided data paths. "
                f"Expected {len(esc10_files)} ESC-10 files but found none."
            )
        
        print(f"Filtered to ESC-10 subset: {len(filtered_paths)} files from {len(data_paths)} total files")
        return filtered_paths
    
    def get_fold_for_file(self, filename: str) -> Optional[int]:
        """Get the fold number for a given filename."""
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Cannot get fold information.")
        
        file_row = self.metadata_df[self.metadata_df['filename'] == filename]
        if len(file_row) > 0:
            return file_row.iloc[0]['fold']
        
        raise ValueError(f"File {filename} not found in metadata.")
    
    def get_metadata_for_file(self, filename: str) -> dict:
        """Get metadata for a given filename."""
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Cannot get file metadata.")
        
        file_row = self.metadata_df[self.metadata_df['filename'] == filename]
        if len(file_row) > 0:
            return file_row.iloc[0].to_dict()
        
        raise ValueError(f"File {filename} not found in metadata.")

def create_esc50_fold_splits(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    config: Optional[ESC50Config] = None,
    val_fold: int = 5,
    augmentations_per_sample: int = 0,
    augmentations: Optional[List[str]] = None,
    aug_config: Optional[Union[dict, AugmentationConfig]] = None
) -> Tuple[ESC50Dataset, ESC50Dataset]:
    """
    Create ESC-50 dataset splits based on predefined folds.
    
    For k-fold cross-validation:
    - 1 fold is used for validation
    - Remaining 4 folds are used for training
    
    Args:
        data_path: Path to ESC-50 dataset
        feature_extractor: Feature extractor to use
        config: ESC-50 configuration
        val_fold: Fold number to use for validation (1-5)
        augmentations_per_sample: Number of augmentations per sample
        augmentations: List of augmentation names
        aug_config: Augmentation configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Atomic validation: check data path exists
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Find metadata file
    meta_file = None
    possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
    
    for root in possible_roots:
        candidate = root / "meta" / "esc50.csv"
        if candidate.exists():
            meta_file = candidate
            break
    
    if meta_file is None:
        raise FileNotFoundError(
            f"ESC-50 metadata file (esc50.csv) not found in any of the expected locations: "
            f"{[str(root / 'meta' / 'esc50.csv') for root in possible_roots]}"
        )
    
    metadata_df = pd.read_csv(meta_file)
    
    # Atomic validation: check fold number is valid
    available_folds = set(metadata_df['fold'].unique())
    if val_fold not in available_folds:
        raise ValueError(f"Validation fold {val_fold} not found in metadata. Available folds: {sorted(available_folds)}")
    
    # Filter for ESC-10 if requested
    if config and config.use_esc10_subset:
        esc10_subset = metadata_df[metadata_df['esc10'] == True]
        if esc10_subset.empty:
            raise ValueError("No ESC-10 files found in metadata.")
        metadata_df = esc10_subset
    
    # Get all audio file paths
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if not all_paths:
        raise FileNotFoundError(f"No .wav files found in {data_path}")
    
    # Create fold-based splits
    train_paths = []
    val_paths = []
    
    for path in all_paths:
        filename = path.name
        file_row = metadata_df[metadata_df['filename'] == filename]
        
        if len(file_row) > 0:
            fold = file_row.iloc[0]['fold']
            
            if fold == val_fold:
                val_paths.append(str(path))
            else:
                train_paths.append(str(path))
    
    # Atomic validation: ensure both splits have data
    if not train_paths:
        raise ValueError("No training files found after fold splitting.")
    if not val_paths:
        raise ValueError("No validation files found after fold splitting.")
    
    print(f"ESC-50 fold splits - Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets
    train_dataset = ESC50Dataset(
        data_path=data_path,
        data_paths=train_paths,
        feature_extractor=feature_extractor,
        config=config,
        augmentations_per_sample=augmentations_per_sample,
        augmentations=augmentations or [],
        aug_config=aug_config
    )
    
    val_dataset = ESC50Dataset(
        data_path=data_path,
        data_paths=val_paths,
        feature_extractor=feature_extractor,
        config=config,
        aug_config=aug_config
    )
    
    return train_dataset, val_dataset

def create_esc50_kfold_splits(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    config: Optional[ESC50Config] = None,
    k_folds: int = 5,
    augmentations_per_sample: int = 0,
    augmentations: Optional[List[str]] = None,
    aug_config: Optional[Union[dict, AugmentationConfig]] = None
) -> List[Tuple[ESC50Dataset, ESC50Dataset]]:
    """
    Create ESC-50 k-fold cross-validation splits using predefined folds.
    
    For each iteration:
    - 1 fold is used for validation
    - Remaining 4 folds are used for training
    
    Args:
        data_path: Path to ESC-50 dataset
        feature_extractor: Feature extractor to use
        config: ESC-50 configuration
        k_folds: Number of folds (should be 5 for ESC-50)
        augmentations_per_sample: Number of augmentations per sample
        augmentations: List of augmentation names
        aug_config: Augmentation configuration
        
    Returns:
        List of (train_dataset, val_dataset) tuples for each fold
    """
    # Atomic validation: check data path exists
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Find metadata file
    meta_file = None
    possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
    
    for root in possible_roots:
        candidate = root / "meta" / "esc50.csv"
        if candidate.exists():
            meta_file = candidate
            break
    
    if meta_file is None:
        raise FileNotFoundError(
            f"ESC-50 metadata file (esc50.csv) not found in any of the expected locations: "
            f"{[str(root / 'meta' / 'esc50.csv') for root in possible_roots]}"
        )
    
    metadata_df = pd.read_csv(meta_file)
    
    # Atomic validation: check k_folds matches available folds
    available_folds = sorted(metadata_df['fold'].unique())
    if len(available_folds) != k_folds:
        raise ValueError(
            f"Requested {k_folds} folds but metadata contains {len(available_folds)} folds: {available_folds}"
        )
    
    # Filter for ESC-10 if requested
    if config and config.use_esc10_subset:
        esc10_subset = metadata_df[metadata_df['esc10'] == True]
        if esc10_subset.empty:
            raise ValueError("No ESC-10 files found in metadata.")
        metadata_df = esc10_subset
    
    # Get all audio file paths
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if not all_paths:
        raise FileNotFoundError(f"No .wav files found in {data_path}")
    
    fold_datasets = []
    
    # For each fold, use it as validation and the rest as training
    for val_fold in available_folds:
        train_paths = []
        val_paths = []
        
        for path in all_paths:
            filename = path.name
            file_row = metadata_df[metadata_df['filename'] == filename]
            
            if len(file_row) > 0:
                fold = file_row.iloc[0]['fold']
                
                if fold == val_fold:
                    val_paths.append(str(path))
                else:
                    train_paths.append(str(path))
        
        # Atomic validation: ensure both splits have data
        if not train_paths:
            raise ValueError(f"No training files found for fold {val_fold}.")
        if not val_paths:
            raise ValueError(f"No validation files found for fold {val_fold}.")
        
        # Create datasets for this fold
        train_dataset = ESC50Dataset(
            data_path=data_path,
            data_paths=train_paths,
            feature_extractor=feature_extractor,
            config=config,
            augmentations_per_sample=augmentations_per_sample,
            augmentations=augmentations or [],
            aug_config=aug_config
        )
        
        val_dataset = ESC50Dataset(
            data_path=data_path,
            data_paths=val_paths,
            feature_extractor=feature_extractor,
            config=config,
            aug_config=aug_config
        )
        
        fold_datasets.append((train_dataset, val_dataset))
        print(f"Fold {val_fold} (validation): Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    return fold_datasets 