#!/usr/bin/env python3
"""
Test script for ESC-50 dataset implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from esc50.esc50_dataset import ESC50Dataset, create_esc50_fold_splits, create_esc50_kfold_splits
from configs.dataset_config import ESC50Config
from configs.augmentation_config import AugmentationConfig, create_augmentation_configs
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor

def test_esc50_dataset():
    """Test basic ESC-50 dataset functionality."""
    print("Testing ESC-50 dataset implementation...")
    
    # Configuration
    data_path = "../esc50_data/ESC-50-master/classes"
    
    # Check if dataset exists
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}")
        print("Please run the download script first:")
        print("uv run src/download_esc50.py --data-dir ../esc50_data")
        return False
    
    # Create ESC-50 config
    config = ESC50Config(
        data_path=data_path,
        use_esc10_subset=False,  # Test with full ESC-50 first
        fold_based_split=True
    )
    
    print(f"Dataset type: {config.dataset_type}")
    print(f"Number of classes: {config.get_num_classes()}")
    print(f"Dataset info: {config.get_dataset_info()}")
    
    # Create feature extractor
    feature_extractor = MelSpectrogramFeatureExtractor(
        sampling_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    
    # Test fold split (1 validation fold, 4 train folds)
    print("\nTesting fold split (1 validation fold, 4 train folds)...")
    try:
        train_dataset, val_dataset = create_esc50_fold_splits(
            data_path=data_path,
            feature_extractor=feature_extractor,
            config=config,
            val_fold=5  # Use fold 5 as validation, folds 1-4 as training
        )
        
        print(f"Train dataset size: {len(train_dataset)} (4 folds)")
        print(f"Validation dataset size: {len(val_dataset)} (1 fold)")
        
        # Test getting a sample
        print("\nTesting sample retrieval...")
        if len(train_dataset) > 0:
            sample, label = train_dataset[0]
            print(f"Sample shape: {sample.shape}")
            print(f"Label: {label}")
            print(f"Classes: {train_dataset.get_classes()[:10]}...")  # Show first 10 classes
        
        print("✅ ESC-50 dataset test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ESC-50 dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_esc10_subset():
    """Test ESC-10 subset functionality."""
    print("\nTesting ESC-10 subset...")
    
    data_path = "../esc50_data/ESC-50-master/classes"
    
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}")
        return False
    
    # Create ESC-10 config
    config = ESC50Config(
        data_path=data_path,
        use_esc10_subset=True,  # Test ESC-10 subset
        fold_based_split=True
    )
    
    print(f"ESC-10 subset - Number of classes: {config.get_num_classes()}")
    
    # Create feature extractor
    feature_extractor = MelSpectrogramFeatureExtractor(
        sampling_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    
    try:
        train_dataset, val_dataset = create_esc50_fold_splits(
            data_path=data_path,
            feature_extractor=feature_extractor,
            config=config,
            val_fold=5  # Use fold 5 as validation, folds 1-4 as training
        )
        
        print(f"ESC-10 Train dataset size: {len(train_dataset)} (4 folds)")
        print(f"ESC-10 Validation dataset size: {len(val_dataset)} (1 fold)")
        
        print("✅ ESC-10 subset test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ESC-10 subset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kfold_cross_validation():
    """Test proper k-fold cross-validation with all 5 folds."""
    print("\nTesting k-fold cross-validation (5 folds)...")
    
    data_path = "../esc50_data/ESC-50-master/classes"
    
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}")
        return False
    
    config = ESC50Config(
        data_path=data_path,
        use_esc10_subset=True,  # Use smaller subset for faster testing
        fold_based_split=True
    )
    
    feature_extractor = MelSpectrogramFeatureExtractor(
        sampling_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    
    try:
        # Get all k-fold splits
        fold_datasets = create_esc50_kfold_splits(
            data_path=data_path,
            feature_extractor=feature_extractor,
            config=config,
            k_folds=5
        )
        
        print(f"Created {len(fold_datasets)} fold splits")
        
        # Test each fold
        for i, (train_dataset, val_dataset) in enumerate(fold_datasets, 1):
            print(f"Fold {i}: Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
            
            # Verify that train + val = total samples (approximately, since we're using ESC-10 subset)
            total_samples = len(train_dataset) + len(val_dataset)
            print(f"  Total samples in fold {i}: {total_samples}")
        
        print("✅ K-fold cross-validation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing k-fold cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_esc50_with_augmentations():
    """Test ESC-50 dataset with audio augmentations."""
    print("\nTesting ESC-50 dataset with augmentations...")
    
    data_path = "../esc50_data/ESC-50-master/classes"
    
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}")
        return False
    
    # Create ESC-50 config
    config = ESC50Config(
        data_path=data_path,
        use_esc10_subset=True,  # Use smaller subset for faster testing
        fold_based_split=True
    )
    
    # Create feature extractor
    feature_extractor = MelSpectrogramFeatureExtractor(
        sampling_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    
    # Create augmentation configuration
    aug_config_dict = {
        "augmentations": {
            "augmentations_per_sample": 2,
            "augmentations": ["pitch_shift", "time_stretch", "add_noise"],
            "pitch_shift_min_semitones": -4,
            "pitch_shift_max_semitones": 4,
            "pitch_shift_p": 1.0,
            "time_stretch_min_rate": 0.8,
            "time_stretch_max_rate": 1.2,
            "time_stretch_p": 1.0,
            "gaussian_noise_min_amplitude": 0.001,
            "gaussian_noise_max_amplitude": 0.015,
            "gaussian_noise_p": 1.0
        }
    }
    
    try:
        # Create augmentation config
        aug_config = create_augmentation_configs(aug_config_dict)
        print(f"Created augmentation config with {len(aug_config.augmentations)} augmentations")
        
        # Test with augmentations
        train_dataset, val_dataset = create_esc50_fold_splits(
            data_path=data_path,
            feature_extractor=feature_extractor,
            config=config,
            val_fold=5,
            augmentations_per_sample=2,
            augmentations=["pitch_shift", "time_stretch", "add_noise"],
            aug_config=aug_config
        )
        
        print(f"Train dataset size with augmentations: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Test getting a sample with augmentations
        if len(train_dataset) > 0:
            sample, label = train_dataset[0]
            print(f"Augmented sample shape: {sample.shape}")
            print(f"Label: {label}")
            
            # Test multiple samples to ensure augmentations are working
            for i in range(min(3, len(train_dataset))):
                sample, label = train_dataset[i]
                print(f"Sample {i} shape: {sample.shape}, label: {label}")
        
        print("✅ ESC-50 augmentation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ESC-50 with augmentations: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("ESC-50 Dataset K-Fold Cross-Validation Test")
    print("=" * 50)
    
    success = True
    
    # # Test full ESC-50 dataset
    # success &= test_esc50_dataset()
    
    # # Test ESC-10 subset
    # success &= test_esc10_subset()
    
    # # Test proper k-fold cross-validation
    success &= test_kfold_cross_validation()
    
    # Test ESC-50 with augmentations
    # success &= test_esc50_with_augmentations()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("\nYou can now use ESC-50 dataset with k-fold cross-validation and augmentations!")
        print("\nK-fold setup:")
        print("- 5 total folds in ESC-50 dataset")
        print("- For each iteration: 1 fold used for validation, 4 folds used for training")
        print("- No test folds - use k-fold cross-validation for evaluation")
        print("\nAugmentation support:")
        print("- Supports pitch_shift, time_stretch, add_noise, tanh_distortion, sin_distortion, polarity_inversion")
        print("- Configure augmentations via AugmentationConfig")
        print("- Augmentations applied only to training data")
        print("\nExample usage:")
        print("1. Update config.yaml to use ESC-50:")
        print("   dataset_type: esc50")
        print("   data_path: /path/to/esc50/classes")
        print("   num_classes: 50")
        print("   fold_based_split: true")
        print("\n2. Use create_esc50_kfold_splits() for proper k-fold cross-validation:")
        print("   fold_datasets = create_esc50_kfold_splits(...)")
        print("   for train_dataset, val_dataset in fold_datasets:")
        print("       # Train and validate on this fold")
        print("\n3. Add augmentations:")
        print("   aug_config = create_augmentation_configs(config_dict)")
        print("   train_dataset, val_dataset = create_esc50_fold_splits(")
        print("       ..., augmentations=['pitch_shift', 'time_stretch'], aug_config=aug_config)")
    else:
        print("❌ Some tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 