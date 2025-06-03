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
from configs.dataset_config import ESC10Config
from configs.augmentation_config import AugmentationConfig, create_augmentation_configs

class ESC10Dataset(UAVDataset):
    """
    ESC-10 specific dataset class that extends UAVDataset.
    
    Handles ESC-10 specific features:
    - Fold-based cross-validation splits (inherited from ESC-50)
    - 10 classes from the ESC-50 subset
    - Metadata integration
    """
    
    def __init__(self,
                 data_path: str,
                 data_paths: List[str],
                 feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor, ViTImageProcessor],
                 config: Optional[ESC10Config] = None,
                 standardize_audio_boolean: bool = True,
                 target_sr: int = 16000,
                 target_duration: int = 5,
                 augmentations_per_sample: int = 0,
                 augmentations: List[str] = [],
                 num_channels: int = 1,
                 aug_config: Optional[Union[dict, AugmentationConfig]] = None,
                 load_metadata: bool = False) -> None:
        
        self.esc10_config = config
        self.metadata_df = None
        self.esc10_root = None
        
        # Build fold mappings from the actual data_paths parameter
        self.file_names = []  # List[str] - filenames only (no path)
        self.assigned_fold = []  # List[int] - corresponding fold numbers
        self.filename_to_fold = {}  # Dict[str, int] - mapping for quick lookup
        self.fold_to_files = {}  # Dict[int, List[str]] - reverse mapping
        
        # Extract fold information from filenames in data_paths
        # Comment out fold mapping code as it should be handled at datamodule level
        # for path_str in data_paths:
        #     path = Path(path_str)
        #     filename = path.name
        #     
        #     # ESC-10 files have fold number as first character (1-5) inherited from ESC-50
        #     try:
        #         fold_num = int(filename[0])
        #         if fold_num < 1 or fold_num > 5:
        #             raise ValueError(f"Invalid fold number {fold_num} for file {filename}")
        #     except (ValueError, IndexError) as e:
        #         raise ValueError(f"Cannot extract fold number from filename {filename}: {e}")
        #     
        #     self.file_names.append(filename)
        #     self.assigned_fold.append(fold_num)
        #     self.filename_to_fold[filename] = fold_num
        #     
        #     # Build reverse mapping
        #     if fold_num not in self.fold_to_files:
        #         self.fold_to_files[fold_num] = []
        #     self.fold_to_files[fold_num].append(path_str)
        # 
        # print(f"Found {len(self.file_names)} ESC-10 files across folds: {sorted(self.fold_to_files.keys())}")
        # for fold, files in self.fold_to_files.items():
        #     print(f"  Fold {fold}: {len(files)} files")

        # Load ESC-10 metadata if requested
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
        """Load ESC-10 metadata CSV file."""
        data_path_obj = Path(data_path)
        
        # Try to find the ESC-10 root directory and metadata
        possible_roots = [
            data_path_obj,
            data_path_obj.parent,
            data_path_obj.parent.parent
        ]
        
        for root in possible_roots:
            meta_file = root / "meta" / "esc10.csv"
            if meta_file.exists():
                self.metadata_df = pd.read_csv(meta_file)
                self.esc10_root = root
                print(f"Loaded ESC-10 metadata from: {meta_file}")
                return
        
        # Don't fail if metadata not found - it's now optional
        print("ESC-10 metadata file not found - using filename-based operations only")
        self.metadata_df = None
        self.esc10_root = None
    
    def get_fold_for_file(self, filename: str) -> Optional[int]:
        """Get the fold number for a given filename."""
        # First try filename-based lookup (more efficient)
        if filename in self.filename_to_fold:
            return self.filename_to_fold[filename]
        
        # Fallback to metadata if available
        if self.metadata_df is not None:
            file_row = self.metadata_df[self.metadata_df['filename'] == filename]
            if len(file_row) > 0:
                return file_row.iloc[0]['fold']
        
        # Extract directly from filename as last resort
        try:
            return int(filename[0])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot determine fold for file {filename}")
    
    def get_fold_files(self, fold_num: int) -> List[str]:
        """Get all file paths for a specific fold."""
        return self.fold_to_files.get(fold_num, [])
    
    def get_available_folds(self) -> List[int]:
        """Get list of available fold numbers."""
        return sorted(self.fold_to_files.keys()) if self.fold_to_files else []
    
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
        
        file_row = self.metadata_df[self.metadata_df['filename'] == filename]
        if len(file_row) > 0:
            return file_row.iloc[0].to_dict()
        
        raise ValueError(f"File {filename} not found in metadata.")

def create_esc10_fold_splits(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    config: Optional[ESC10Config] = None,
    val_fold: int = 5,
    augmentations_per_sample: int = 0,
    augmentations: Optional[List[str]] = None,
    aug_config: Optional[Union[dict, AugmentationConfig]] = None
) -> Tuple[ESC10Dataset, ESC10Dataset]:
    """
    Create ESC-10 dataset splits based on predefined folds.
    
    For k-fold cross-validation:
    - 1 fold is used for validation
    - Remaining 4 folds are used for training
    
    Args:
        data_path: Path to ESC-10 dataset
        feature_extractor: Feature extractor to use
        config: ESC-10 configuration
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
    
    # Find metadata file (optional for ESC-10)
    meta_file = None
    possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
    
    for root in possible_roots:
        candidate = root / "meta" / "esc10.csv"
        if candidate.exists():
            meta_file = candidate
            break
    
    if meta_file is not None:
        metadata_df = pd.read_csv(meta_file)
        
        # Atomic validation: check fold number is valid
        available_folds = set(metadata_df['fold'].unique())
        if val_fold not in available_folds:
            raise ValueError(f"Validation fold {val_fold} not found in metadata. Available folds: {sorted(available_folds)}")
    else:
        print("ESC-10 metadata not found, using filename-based fold extraction")
    
    # Get all audio file paths
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if not all_paths:
        raise FileNotFoundError(f"No .wav files found in {data_path}")
    
    # Create fold-based splits
    train_paths = []
    val_paths = []
    
    for path in all_paths:
        filename = path.name
        
        if meta_file is not None:
            # Use metadata if available
            file_row = metadata_df[metadata_df['filename'] == filename]
            if len(file_row) > 0:
                fold = file_row.iloc[0]['fold']
            else:
                # Fallback to filename extraction
                fold = int(filename[0])
        else:
            # Extract fold from filename (first character)
            try:
                fold = int(filename[0])
            except (ValueError, IndexError) as e:
                raise ValueError(f"Cannot extract fold number from filename {filename}: {e}")
        
        if fold == val_fold:
            val_paths.append(str(path))
        else:
            train_paths.append(str(path))
    
    # Atomic validation: ensure both splits have data
    if not train_paths:
        raise ValueError("No training files found after fold splitting.")
    if not val_paths:
        raise ValueError("No validation files found after fold splitting.")
    
    print(f"ESC-10 fold splits - Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets
    train_dataset = ESC10Dataset(
        data_path=data_path,
        data_paths=train_paths,
        feature_extractor=feature_extractor,
        config=config,
        augmentations_per_sample=augmentations_per_sample,
        augmentations=augmentations or [],
        aug_config=aug_config
    )
    
    val_dataset = ESC10Dataset(
        data_path=data_path,
        data_paths=val_paths,
        feature_extractor=feature_extractor,
        config=config,
        aug_config=aug_config
    )
    
    return train_dataset, val_dataset

def create_esc10_kfold_splits(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    config: Optional[ESC10Config] = None,
    k_folds: int = 5,
    augmentations_per_sample: int = 0,
    augmentations: Optional[List[str]] = None,
    aug_config: Optional[Union[dict, AugmentationConfig]] = None
) -> List[Tuple[ESC10Dataset, ESC10Dataset]]:
    """
    Create ESC-10 k-fold cross-validation splits using predefined folds.
    
    For each iteration:
    - 1 fold is used for validation
    - Remaining 4 folds are used for training
    
    Args:
        data_path: Path to ESC-10 dataset
        feature_extractor: Feature extractor to use
        config: ESC-10 configuration
        k_folds: Number of folds (should be 5 for ESC-10)
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
    
    # Find metadata file (optional for ESC-10)
    meta_file = None
    possible_roots = [data_path_obj, data_path_obj.parent, data_path_obj.parent.parent]
    
    for root in possible_roots:
        candidate = root / "meta" / "esc10.csv"
        if candidate.exists():
            meta_file = candidate
            break
    
    if meta_file is not None:
        metadata_df = pd.read_csv(meta_file)
        
        # Atomic validation: check k_folds matches available folds
        available_folds = sorted(metadata_df['fold'].unique())
        if len(available_folds) != k_folds:
            raise ValueError(
                f"Requested {k_folds} folds but metadata contains {len(available_folds)} folds: {available_folds}"
            )
    else:
        print("ESC-10 metadata not found, using filename-based fold extraction")
        # Assume 5 folds for ESC-10 (inherited from ESC-50)
        if k_folds != 5:
            raise ValueError(f"ESC-10 has exactly 5 folds, but {k_folds} was requested")
    
    # Get all audio file paths
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if not all_paths:
        raise FileNotFoundError(f"No .wav files found in {data_path}")
    
    fold_datasets = []
    
    # For each fold, use it as validation and the rest as training
    for val_fold in range(1, k_folds + 1):
        train_paths = []
        val_paths = []
        
        for path in all_paths:
            filename = path.name
            
            if meta_file is not None:
                # Use metadata if available
                file_row = metadata_df[metadata_df['filename'] == filename]
                if len(file_row) > 0:
                    fold = file_row.iloc[0]['fold']
                else:
                    # Fallback to filename extraction
                    fold = int(filename[0])
            else:
                # Extract fold from filename (first character)
                try:
                    fold = int(filename[0])
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Cannot extract fold number from filename {filename}: {e}")
            
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
        train_dataset = ESC10Dataset(
            data_path=data_path,
            data_paths=train_paths,
            feature_extractor=feature_extractor,
            config=config,
            augmentations_per_sample=augmentations_per_sample,
            augmentations=augmentations or [],
            aug_config=aug_config
        )
        
        val_dataset = ESC10Dataset(
            data_path=data_path,
            data_paths=val_paths,
            feature_extractor=feature_extractor,
            config=config,
            aug_config=aug_config
        )
        
        fold_datasets.append((train_dataset, val_dataset))
        print(f"Fold {val_fold} (validation): Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    return fold_datasets

# Example usage:
"""
# Filename-based k-fold splitting (no CSV required, faster)
from transformers import ASTFeatureExtractor

# Create feature extractor
feature_extractor = ASTFeatureExtractor()

# Create all 5-fold splits using filename-based approach
fold_splits = create_esc10_kfold_splits(
    data_path="path/to/esc10/classes", 
    feature_extractor=feature_extractor
)

# Inspect fold statistics for each split
for i, (train_dataset, val_dataset) in enumerate(fold_splits):
    print(f"\nFold {i+1}:")
    print(f"  Train files: {len(train_dataset.file_names)}")
    print(f"  Val files: {len(val_dataset.file_names)}")
    print(f"  Train fold distribution: {train_dataset.get_fold_statistics()}")
    print(f"  Val fold distribution: {val_dataset.get_fold_statistics()}")
    
    # Access individual attributes
    print(f"  Available folds in train: {train_dataset.get_available_folds()}")
    print(f"  Available folds in val: {val_dataset.get_available_folds()}")

# Create single split (fold 5 as validation)
train_dataset, val_dataset = create_esc10_fold_splits(
    data_path="path/to/esc10/classes",
    feature_extractor=feature_extractor,
    val_fold=5
)

# Quick lookup examples
fold_num = train_dataset.get_fold_for_file("1-100032-A-0.wav")  # Returns 1
fold_files = train_dataset.get_fold_files(1)  # Returns all fold 1 file paths
""" 

def main():
    """Test the ESC-10 dataset functionality."""
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
    
    # Test data path - adjust this to your actual ESC-10 path
    test_data_path = "../datasets/ESC-10-master/classes"  # Replace with actual path
    
    # Check if we have command line arguments for data path
    if len(sys.argv) > 1:
        test_data_path = sys.argv[1]
        print(f"📁 Using data path from command line: {test_data_path}")
    else:
        print(f"📁 Using default data path: {test_data_path}")
        print("💡 You can provide a custom path as: python esc10_dataset.py /path/to/your/esc10/classes")
    
    # Check if path exists
    if not Path(test_data_path).exists():
        print(f"❌ Data path does not exist: {test_data_path}")
        print("Please provide a valid ESC-10 classes directory path")
        print("You can create ESC-10 dataset by running: python download_esc10.py")
        return
    
    try:
        print("\n🧪 Testing ESC-10 k-fold splits...")
        
        # Test 1: Create all k-fold splits
        print("\n1️⃣ Creating all 5-fold splits...")
        fold_splits = create_esc10_kfold_splits(
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
            print(f"    📊 Train fold distribution: {train_dataset.get_fold_statistics()}")
            print(f"    📊 Val fold distribution: {val_dataset.get_fold_statistics()}")
            
            # Show available folds
            train_folds = train_dataset.get_available_folds()
            val_folds = val_dataset.get_available_folds()
            print(f"    🔢 Train folds: {train_folds}")
            print(f"    🔢 Val folds: {val_folds}")
        
        # Test 3: Test individual dataset functionality
        print("\n3️⃣ Testing dataset functionality...")
        train_dataset, val_dataset = fold_splits[0]  # Use first fold split
        
        if train_dataset.file_names:
            # Test fold lookup
            test_filename = train_dataset.file_names[0]
            fold_num = train_dataset.get_fold_for_file(test_filename)
            print(f"    📋 File '{test_filename}' belongs to fold {fold_num}")
            
            # Test getting files for a specific fold
            fold_files = train_dataset.get_fold_files(fold_num)
            print(f"    📁 Fold {fold_num} has {len(fold_files)} files")
            
            # Show first few files from that fold
            if fold_files:
                print(f"    📄 First 3 files in fold {fold_num}:")
                for file_path in fold_files[:3]:
                    filename = Path(file_path).name
                    print(f"      - {filename}")
        
        # Test 4: Create single split
        print("\n4️⃣ Testing single fold split (fold 5 as validation)...")
        train_single, val_single = create_esc10_fold_splits(
            data_path=test_data_path,
            feature_extractor=feature_extractor,
            val_fold=5
        )
        
        print(f"    📚 Single split - Train: {len(train_single.file_names)} files")
        print(f"    📖 Single split - Val: {len(val_single.file_names)} files")
        print(f"    📊 Train folds: {train_single.get_fold_statistics()}")
        print(f"    📊 Val folds: {val_single.get_fold_statistics()}")
        
        print("\n✅ All tests completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found error: {e}")
        print("Make sure the ESC-10 dataset is properly organized with .wav files in subdirectories")
        print("Run 'python download_esc10.py' to create the ESC-10 dataset")
    except ValueError as e:
        print(f"❌ Value error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 