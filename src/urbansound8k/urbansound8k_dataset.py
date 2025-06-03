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
sys.path.append(str(Path(__file__).parent.parent))

from helper.util import UAVDataset, find_classes
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from configs.dataset_config import UrbanSound8KConfig
from configs.augmentation_config import AugmentationConfig, create_augmentation_configs

class UrbanSound8KDataset(UAVDataset):
    """
    UrbanSound8K specific dataset class that extends UAVDataset.
    
    Handles UrbanSound8K specific features:
    - 10-fold cross-validation splits
    - Metadata integration
    - Urban sound classification (10 classes)
    """
    
    def __init__(self,
                 data_path: str,
                 data_paths: List[str],
                 feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor, ViTImageProcessor],
                 config: Optional[UrbanSound8KConfig] = None,
                 standardize_audio_boolean: bool = True,
                 target_sr: int = 16000,
                 target_duration: int = 4,  # UrbanSound8K clips are up to 4 seconds
                 augmentations_per_sample: int = 0,
                 augmentations: List[str] = [],
                 num_channels: int = 1,
                 aug_config: Optional[Union[dict, AugmentationConfig]] = None,
                 load_metadata: bool = False) -> None:
        
        self.urbansound8k_config = config
        self.metadata_df = None
        self.urbansound8k_root = None
        
        # Build fold mappings from the actual data_paths parameter
        self.file_names = []  # List[str] - filenames only (no path)
        self.assigned_fold = []  # List[int] - corresponding fold numbers
        self.filename_to_fold = {}  # Dict[str, int] - mapping for quick lookup
        self.fold_to_files = {}  # Dict[int, List[str]] - reverse mapping
        
        # Load UrbanSound8K metadata if requested
        if load_metadata:
            self._load_metadata(data_path)
        
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
        """Load UrbanSound8K metadata CSV file."""
        data_path_obj = Path(data_path)
        
        # Try to find the UrbanSound8K root directory and metadata
        possible_roots = [
            data_path_obj,
            data_path_obj.parent,
            data_path_obj.parent.parent
        ]
        
        for root in possible_roots:
            meta_file = root / "metadata" / "UrbanSound8K.csv"
            if meta_file.exists():
                self.metadata_df = pd.read_csv(meta_file)
                self.urbansound8k_root = root
                print(f"Loaded UrbanSound8K metadata from: {meta_file}")
                return
        
        # Don't fail if metadata not found - it's now optional
        print("UrbanSound8K metadata file not found - using filename-based operations only")
        self.metadata_df = None
        self.urbansound8k_root = None
    
    def get_fold_for_file(self, filename: str) -> Optional[int]:
        """Get the fold number for a given filename using metadata."""
        # Try metadata lookup first if available
        if self.metadata_df is not None:
            file_row = self.metadata_df[self.metadata_df['slice_file_name'] == filename]
            if len(file_row) > 0:
                return file_row.iloc[0]['fold']
        
        # Fallback: could not determine fold from filename alone for UrbanSound8K
        # as fold information is only in metadata
        raise ValueError(f"Cannot determine fold for file {filename} without metadata")
    
    def get_fold_files(self, fold_num: int) -> List[str]:
        """Get all file paths for a specific fold."""
        return self.fold_to_files.get(fold_num, [])
    
    def get_available_folds(self) -> List[int]:
        """Get list of available fold numbers."""
        return sorted(self.fold_to_files.keys())
    
    def get_fold_statistics(self) -> dict:
        """Get statistics about fold distribution."""
        stats = {}
        for fold, files in self.fold_to_files.items():
            stats[fold] = len(files)
        return stats
    
    def get_metadata_for_file(self, filename: str) -> dict:
        """Get metadata for a given filename."""
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Cannot get file metadata.")
        
        file_row = self.metadata_df[self.metadata_df['slice_file_name'] == filename]
        if len(file_row) > 0:
            return file_row.iloc[0].to_dict()
        
        raise ValueError(f"File {filename} not found in metadata.")

def create_urbansound8k_fold_splits(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    config: Optional[UrbanSound8KConfig] = None,
    val_fold: int = 10,
    augmentations_per_sample: int = 0,
    augmentations: Optional[List[str]] = None,
    aug_config: Optional[Union[dict, AugmentationConfig]] = None
) -> Tuple[UrbanSound8KDataset, UrbanSound8KDataset]:
    """
    Create UrbanSound8K dataset splits based on predefined folds.
    
    For k-fold cross-validation:
    - 1 fold is used for validation
    - Remaining 9 folds are used for training
    
    Args:
        data_path: Path to UrbanSound8K dataset
        feature_extractor: Feature extractor to use
        config: UrbanSound8K configuration
        val_fold: Fold number to use for validation (1-10)
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
        candidate = root / "metadata" / "UrbanSound8K.csv"
        if candidate.exists():
            meta_file = candidate
            break
    
    if meta_file is None:
        raise FileNotFoundError(
            f"UrbanSound8K metadata file (UrbanSound8K.csv) not found in any of the expected locations: "
            f"{[str(root / 'metadata' / 'UrbanSound8K.csv') for root in possible_roots]}"
        )
    
    metadata_df = pd.read_csv(meta_file)
    
    # Atomic validation: check fold number is valid
    available_folds = set(metadata_df['fold'].unique())
    if val_fold not in available_folds:
        raise ValueError(f"Validation fold {val_fold} not found in metadata. Available folds: {sorted(available_folds)}")
    
    # Get all audio file paths
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if not all_paths:
        raise FileNotFoundError(f"No .wav files found in {data_path}")
    
    # Create fold-based splits
    train_paths = []
    val_paths = []
    
    for path in all_paths:
        filename = path.name
        file_row = metadata_df[metadata_df['slice_file_name'] == filename]
        
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
    
    print(f"UrbanSound8K fold splits - Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets
    train_dataset = UrbanSound8KDataset(
        data_path=data_path,
        data_paths=train_paths,
        feature_extractor=feature_extractor,
        config=config,
        augmentations_per_sample=augmentations_per_sample,
        augmentations=augmentations or [],
        aug_config=aug_config,
        load_metadata=True
    )
    
    val_dataset = UrbanSound8KDataset(
        data_path=data_path,
        data_paths=val_paths,
        feature_extractor=feature_extractor,
        config=config,
        aug_config=aug_config,
        load_metadata=True
    )
    
    return train_dataset, val_dataset

def create_urbansound8k_kfold_splits(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    config: Optional[UrbanSound8KConfig] = None,
    k_folds: int = 10,
    augmentations_per_sample: int = 0,
    augmentations: Optional[List[str]] = None,
    aug_config: Optional[Union[dict, AugmentationConfig]] = None
) -> List[Tuple[UrbanSound8KDataset, UrbanSound8KDataset]]:
    """
    Create UrbanSound8K k-fold cross-validation splits using predefined folds.
    
    For each iteration:
    - 1 fold is used for validation
    - Remaining 9 folds are used for training
    
    Args:
        data_path: Path to UrbanSound8K dataset
        feature_extractor: Feature extractor to use
        config: UrbanSound8K configuration
        k_folds: Number of folds (should be 10 for UrbanSound8K)
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
        candidate = root / "metadata" / "UrbanSound8K.csv"
        if candidate.exists():
            meta_file = candidate
            break
    
    if meta_file is None:
        raise FileNotFoundError(
            f"UrbanSound8K metadata file (UrbanSound8K.csv) not found in any of the expected locations: "
            f"{[str(root / 'metadata' / 'UrbanSound8K.csv') for root in possible_roots]}"
        )
    
    metadata_df = pd.read_csv(meta_file)
    
    # Atomic validation: check k_folds matches available folds
    available_folds = sorted(metadata_df['fold'].unique())
    if len(available_folds) != k_folds:
        raise ValueError(
            f"Requested {k_folds} folds but metadata contains {len(available_folds)} folds: {available_folds}"
        )
    
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
            file_row = metadata_df[metadata_df['slice_file_name'] == filename]
            
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
        train_dataset = UrbanSound8KDataset(
            data_path=data_path,
            data_paths=train_paths,
            feature_extractor=feature_extractor,
            config=config,
            augmentations_per_sample=augmentations_per_sample,
            augmentations=augmentations or [],
            aug_config=aug_config,
            load_metadata=True
        )
        
        val_dataset = UrbanSound8KDataset(
            data_path=data_path,
            data_paths=val_paths,
            feature_extractor=feature_extractor,
            config=config,
            aug_config=aug_config,
            load_metadata=True
        )
        
        fold_datasets.append((train_dataset, val_dataset))
        print(f"Fold {val_fold} (validation): Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    return fold_datasets

# Example usage:
"""
# K-fold splitting for UrbanSound8K
from transformers import ASTFeatureExtractor

# Create feature extractor
feature_extractor = ASTFeatureExtractor()

# Create all 10-fold splits
fold_splits = create_urbansound8k_kfold_splits(
    data_path="path/to/urbansound8k/classes", 
    feature_extractor=feature_extractor
)

# Inspect fold statistics for each split
for i, (train_dataset, val_dataset) in enumerate(fold_splits):
    print(f"\nFold {i+1}:")
    print(f"  Train files: {len(train_dataset.file_names)}")
    print(f"  Val files: {len(val_dataset.file_names)}")
    print(f"  Train fold distribution: {train_dataset.get_fold_statistics()}")
    print(f"  Val fold distribution: {val_dataset.get_fold_statistics()}")

# Create single split (fold 10 as validation)
train_dataset, val_dataset = create_urbansound8k_fold_splits(
    data_path="path/to/urbansound8k/classes",
    feature_extractor=feature_extractor,
    val_fold=10
)

# Quick lookup examples
try:
    fold_num = train_dataset.get_fold_for_file("100032-3-0-0.wav")  # Requires metadata
except ValueError as e:
    print(f"Error: {e}")
""" 

def main():
    """Test the UrbanSound8K k-fold splitting functionality."""
    import sys
    from pathlib import Path
    
    # For testing, we'll use a mock feature extractor or create a simple one
    try:
        from transformers import ASTFeatureExtractor
        feature_extractor = ASTFeatureExtractor()
        print("✅ Using ASTFeatureExtractor")
    except ImportError:
        # Fallback to a simple mock if transformers not available
        class MockFeatureExtractor:
            def __call__(self, audio, sampling_rate=16000):
                return {"input_values": audio}
        feature_extractor = MockFeatureExtractor()
        print("⚠️  Using MockFeatureExtractor (transformers not available)")
    
    # Test data path - adjust this to your actual UrbanSound8K path
    test_data_path = "../datasets/UrbanSound8K/classes"  # Replace with actual path
    
    # Check if we have command line arguments for data path
    if len(sys.argv) > 1:
        test_data_path = sys.argv[1]
        print(f"📁 Using data path from command line: {test_data_path}")
    else:
        print(f"📁 Using default data path: {test_data_path}")
        print("💡 You can provide a custom path as: python urbansound8k_dataset.py /path/to/your/urbansound8k/classes")
    
    # Check if path exists
    if not Path(test_data_path).exists():
        print(f"❌ Data path does not exist: {test_data_path}")
        print("Please provide a valid UrbanSound8K classes directory path")
        return
    
    try:
        print("\n🧪 Testing UrbanSound8K k-fold splits...")
        
        # Test 1: Create all k-fold splits
        print("\n1️⃣ Creating all 10-fold splits...")
        fold_splits = create_urbansound8k_kfold_splits(
            data_path=test_data_path,
            feature_extractor=feature_extractor
        )
        
        print(f"✅ Successfully created {len(fold_splits)} fold splits")
        
        # Test 2: Inspect each fold
        print("\n2️⃣ Inspecting fold statistics...")
        for i, (train_dataset, val_dataset) in enumerate(fold_splits):
            fold_num = i + 1
            print(f"\n  Fold {fold_num} (validation fold {fold_num}):")
            print(f"    📚 Train files: {len(train_dataset.file_names)}")
            print(f"    📖 Val files: {len(val_dataset.file_names)}")
            
            # Show fold distributions
            train_folds = train_dataset.get_available_folds()
            val_folds = val_dataset.get_available_folds()
            print(f"    🔢 Train folds: {train_folds}")
            print(f"    🔢 Val folds: {val_folds}")
        
        # Test 3: Test single fold split
        print("\n3️⃣ Testing single fold split (fold 10 as validation)...")
        train_single, val_single = create_urbansound8k_fold_splits(
            data_path=test_data_path,
            feature_extractor=feature_extractor,
            val_fold=10
        )
        
        print(f"    📚 Single split - Train: {len(train_single.file_names)} files")
        print(f"    📖 Single split - Val: {len(val_single.file_names)} files")
        
        print("\n✅ All tests completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found error: {e}")
        print("Make sure the UrbanSound8K dataset is properly organized with .wav files in subdirectories")
        print("and that the metadata/UrbanSound8K.csv file exists")
    except ValueError as e:
        print(f"❌ Value error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 