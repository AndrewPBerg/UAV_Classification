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
from configs.dataset_config import AudioMNISTConfig
from configs.augmentation_config import AugmentationConfig, create_augmentation_configs

class AudioMNISTDataset(UAVDataset):
    """
    AudioMNIST specific dataset class that extends UAVDataset.
    
    Handles AudioMNIST specific features:
    - 10 digit classes (0-9)
    - Spoken digit classification
    - Speaker-based splits (optional)
    - Metadata integration for speaker information
    """
    
    def __init__(self,
                 data_path: str,
                 data_paths: List[str],
                 feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor, ViTImageProcessor],
                 config: Optional[AudioMNISTConfig] = None,
                 standardize_audio_boolean: bool = True,
                 target_sr: int = 16000,
                 target_duration: int = 1,  # AudioMNIST clips are typically ~1 second
                 augmentations_per_sample: int = 0,
                 augmentations: List[str] = [],
                 num_channels: int = 1,
                 aug_config: Optional[Union[dict, AugmentationConfig]] = None,
                 load_metadata: bool = False) -> None:
        
        self.audiomnist_config = config
        self.metadata_df = None
        self.audiomnist_root = None
        
        # Build speaker mappings from the actual data_paths parameter
        self.file_names = []  # List[str] - filenames only (no path)
        self.assigned_speaker = []  # List[str] - corresponding speaker names
        self.assigned_digit = []  # List[int] - corresponding digit labels
        self.filename_to_speaker = {}  # Dict[str, str] - mapping for quick lookup
        self.filename_to_digit = {}  # Dict[str, int] - mapping for quick lookup
        self.speaker_to_files = {}  # Dict[str, List[str]] - reverse mapping
        self.digit_to_files = {}  # Dict[int, List[str]] - reverse mapping
        
        # Extract speaker and digit information from filenames in data_paths
        for path_str in data_paths:
            path = Path(path_str)
            filename = path.name
            
            # AudioMNIST files have format: "digit_speaker_instance.wav" (e.g., "0_jackson_0.wav")
            try:
                parts = Path(filename).stem.split('_')
                if len(parts) >= 2:
                    digit = int(parts[0])
                    speaker = parts[1]
                    
                    if digit < 0 or digit > 9:
                        raise ValueError(f"Invalid digit {digit} for file {filename}")
                else:
                    raise ValueError(f"Invalid filename format {filename}")
            except (ValueError, IndexError) as e:
                # Fallback: try to extract digit from parent directory name
                parent_dir = path.parent.name
                try:
                    digit = int(parent_dir)
                    speaker = "unknown"
                    if digit < 0 or digit > 9:
                        raise ValueError(f"Invalid digit {digit} from directory {parent_dir}")
                except ValueError:
                    raise ValueError(f"Cannot extract digit and speaker from filename {filename}: {e}")
            
            self.file_names.append(filename)
            self.assigned_speaker.append(speaker)
            self.assigned_digit.append(digit)
            self.filename_to_speaker[filename] = speaker
            self.filename_to_digit[filename] = digit
            
            # Build reverse mappings
            if speaker not in self.speaker_to_files:
                self.speaker_to_files[speaker] = []
            self.speaker_to_files[speaker].append(path_str)
            
            if digit not in self.digit_to_files:
                self.digit_to_files[digit] = []
            self.digit_to_files[digit].append(path_str)
        
        print(f"Found {len(self.file_names)} files from {len(self.speaker_to_files)} speakers")
        print(f"Digits represented: {sorted(self.digit_to_files.keys())}")
        for digit, files in self.digit_to_files.items():
            print(f"  Digit {digit}: {len(files)} files")
        
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
    
    def get_speaker_for_file(self, filename: str) -> Optional[str]:
        """Get the speaker name for a given filename."""
        return self.filename_to_speaker.get(filename, None)
    
    def get_digit_for_file(self, filename: str) -> Optional[int]:
        """Get the digit label for a given filename."""
        return self.filename_to_digit.get(filename, None)
    
    def get_speaker_files(self, speaker: str) -> List[str]:
        """Get all file paths for a specific speaker."""
        return self.speaker_to_files.get(speaker, [])
    
    def get_digit_files(self, digit: int) -> List[str]:
        """Get all file paths for a specific digit."""
        return self.digit_to_files.get(digit, [])
    
    def get_available_speakers(self) -> List[str]:
        """Get list of available speakers."""
        return sorted(self.speaker_to_files.keys())
    
    def get_available_digits(self) -> List[int]:
        """Get list of available digits."""
        return sorted(self.digit_to_files.keys())
    
    def get_speaker_statistics(self) -> dict:
        """Get statistics about speaker distribution."""
        stats = {}
        for speaker, files in self.speaker_to_files.items():
            stats[speaker] = len(files)
        return stats
    
    def get_digit_statistics(self) -> dict:
        """Get statistics about digit distribution."""
        stats = {}
        for digit, files in self.digit_to_files.items():
            stats[digit] = len(files)
        return stats

def create_audiomnist_speaker_splits(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    config: Optional[AudioMNISTConfig] = None,
    test_speakers: Optional[List[str]] = None,
    val_ratio: float = 0.2,
    augmentations_per_sample: int = 0,
    augmentations: Optional[List[str]] = None,
    aug_config: Optional[Union[dict, AugmentationConfig]] = None
) -> Tuple[AudioMNISTDataset, AudioMNISTDataset]:
    """
    Create AudioMNIST dataset splits based on speakers (speaker-independent splits).
    Creates only train and validation splits.
    
    Args:
        data_path: Path to AudioMNIST dataset
        feature_extractor: Feature extractor to use
        config: AudioMNIST configuration
        test_speakers: Deprecated parameter (kept for compatibility)
        val_ratio: Ratio of speakers to use for validation
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
    
    # Get all audio file paths
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if not all_paths:
        raise FileNotFoundError(f"No .wav files found in {data_path}")
    
    # Extract speaker information from filenames
    speaker_to_files = {}
    for path in all_paths:
        filename = path.name
        try:
            parts = Path(filename).stem.split('_')
            if len(parts) >= 2:
                speaker = parts[1]
            else:
                # Fallback to unknown speaker
                speaker = "unknown"
        except (ValueError, IndexError):
            speaker = "unknown"
        
        if speaker not in speaker_to_files:
            speaker_to_files[speaker] = []
        speaker_to_files[speaker].append(str(path))
    
    available_speakers = list(speaker_to_files.keys())
    print(f"Found {len(available_speakers)} speakers: {available_speakers}")
    
    # Split speakers into train and validation (no test split)
    import random
    random.seed(42)
    num_val_speakers = max(1, int(len(available_speakers) * val_ratio))
    val_speakers = random.sample(available_speakers, num_val_speakers)
    train_speakers = [s for s in available_speakers if s not in val_speakers]
    
    print(f"Speaker splits (train/val only):")
    print(f"  Train speakers ({len(train_speakers)}): {train_speakers}")
    print(f"  Val speakers ({len(val_speakers)}): {val_speakers}")
    
    # Create file lists for each split
    train_paths = []
    val_paths = []
    
    for speaker in train_speakers:
        train_paths.extend(speaker_to_files[speaker])
    
    for speaker in val_speakers:
        val_paths.extend(speaker_to_files[speaker])
    
    # Atomic validation: ensure all splits have data
    if not train_paths:
        raise ValueError("No training files found after speaker splitting.")
    if not val_paths:
        raise ValueError("No validation files found after speaker splitting.")
    
    print(f"AudioMNIST speaker splits - Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets
    train_dataset = AudioMNISTDataset(
        data_path=data_path,
        data_paths=train_paths,
        feature_extractor=feature_extractor,
        config=config,
        augmentations_per_sample=augmentations_per_sample,
        augmentations=augmentations or [],
        aug_config=aug_config
    )
    
    val_dataset = AudioMNISTDataset(
        data_path=data_path,
        data_paths=val_paths,
        feature_extractor=feature_extractor,
        config=config,
        aug_config=aug_config
    )
    
    return train_dataset, val_dataset

def create_audiomnist_random_splits(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    config: Optional[AudioMNISTConfig] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    augmentations_per_sample: int = 0,
    augmentations: Optional[List[str]] = None,
    aug_config: Optional[Union[dict, AugmentationConfig]] = None,
    random_seed: int = 42
) -> Tuple[AudioMNISTDataset, AudioMNISTDataset]:
    """
    Create AudioMNIST dataset splits using random splitting (speaker-dependent).
    Creates only train and validation splits.
    
    Args:
        data_path: Path to AudioMNIST dataset
        feature_extractor: Feature extractor to use
        config: AudioMNIST configuration
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        augmentations_per_sample: Number of augmentations per sample
        augmentations: List of augmentation names
        aug_config: Augmentation configuration
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio}")
    
    # Atomic validation: check data path exists
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Get all audio file paths
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if not all_paths:
        raise FileNotFoundError(f"No .wav files found in {data_path}")
    
    # Randomize and split
    import random
    random.seed(random_seed)
    random.shuffle(all_paths)
    
    total_files = len(all_paths)
    train_size = int(total_files * train_ratio)
    
    train_paths = [str(p) for p in all_paths[:train_size]]
    val_paths = [str(p) for p in all_paths[train_size:]]
    
    print(f"AudioMNIST random splits - Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets
    train_dataset = AudioMNISTDataset(
        data_path=data_path,
        data_paths=train_paths,
        feature_extractor=feature_extractor,
        config=config,
        augmentations_per_sample=augmentations_per_sample,
        augmentations=augmentations or [],
        aug_config=aug_config
    )
    
    val_dataset = AudioMNISTDataset(
        data_path=data_path,
        data_paths=val_paths,
        feature_extractor=feature_extractor,
        config=config,
        aug_config=aug_config
    )
    
    return train_dataset, val_dataset

# Example usage:
"""
# Speaker-based splitting (speaker-independent)
from transformers import ASTFeatureExtractor

# Create feature extractor
feature_extractor = ASTFeatureExtractor()

# Create speaker-based splits
train_dataset, val_dataset = create_audiomnist_speaker_splits(
    data_path="path/to/audiomnist",
    feature_extractor=feature_extractor
)

# Random splitting (speaker-dependent)
train_dataset, val_dataset = create_audiomnist_random_splits(
    data_path="path/to/audiomnist",
    feature_extractor=feature_extractor,
    train_ratio=0.8,
    val_ratio=0.2
)

# Inspect dataset statistics
print(f"Train speakers: {train_dataset.get_available_speakers()}")
print(f"Train digits: {train_dataset.get_available_digits()}")
print(f"Speaker distribution: {train_dataset.get_speaker_statistics()}")
print(f"Digit distribution: {train_dataset.get_digit_statistics()}")
"""

def main():
    """Test the AudioMNIST dataset functionality."""
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
    
    # Test data path - adjust this to your actual AudioMNIST path
    test_data_path = "../datasets/audiomnist_dataset"  # Replace with actual path
    
    # Check if we have command line arguments for data path
    if len(sys.argv) > 1:
        test_data_path = sys.argv[1]
        print(f"📁 Using data path from command line: {test_data_path}")
    else:
        print(f"📁 Using default data path: {test_data_path}")
        print("💡 You can provide a custom path as: python audiomnist_dataset.py /path/to/your/audiomnist")
    
    # Check if path exists
    if not Path(test_data_path).exists():
        print(f"❌ Data path does not exist: {test_data_path}")
        print("Please provide a valid AudioMNIST directory path")
        return
    
    try:
        print("\n🧪 Testing AudioMNIST dataset functionality...")
        
        # Test 1: Create speaker-based splits
        print("\n1️⃣ Creating speaker-based splits...")
        train_dataset, val_dataset = create_audiomnist_speaker_splits(
            data_path=test_data_path,
            feature_extractor=feature_extractor
        )
        
        print(f"✅ Successfully created speaker-based splits")
        print(f"    📚 Train files: {len(train_dataset.file_names)}")
        print(f"    📖 Val files: {len(val_dataset.file_names)}")
        
        # Test 2: Inspect dataset functionality
        print("\n2️⃣ Testing dataset functionality...")
        
        if train_dataset.file_names:
            # Test speaker and digit lookup
            test_filename = train_dataset.file_names[0]
            speaker = train_dataset.get_speaker_for_file(test_filename)
            digit = train_dataset.get_digit_for_file(test_filename)
            print(f"    📋 File '{test_filename}' - Speaker: {speaker}, Digit: {digit}")
            
            # Show statistics
            speaker_stats = train_dataset.get_speaker_statistics()
            digit_stats = train_dataset.get_digit_statistics()
            print(f"    📊 Train speakers: {len(speaker_stats)} speakers")
            print(f"    📊 Train digits: {sorted(digit_stats.keys())}")
            
            # Show first few speakers and their file counts
            print(f"    📄 First 3 speakers in train set:")
            for i, (speaker, count) in enumerate(list(speaker_stats.items())[:3]):
                print(f"      - {speaker}: {count} files")
        
        # Test 3: Create random splits
        print("\n3️⃣ Testing random splits...")
        train_random, val_random = create_audiomnist_random_splits(
            data_path=test_data_path,
            feature_extractor=feature_extractor,
            train_ratio=0.8,
            val_ratio=0.2
        )
        
        print(f"    📚 Random split - Train: {len(train_random.file_names)} files")
        print(f"    📖 Random split - Val: {len(val_random.file_names)} files")
        
        print("\n✅ All tests completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found error: {e}")
        print("Make sure the AudioMNIST dataset is properly organized with .wav files in subdirectories")
    except ValueError as e:
        print(f"❌ Value error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 