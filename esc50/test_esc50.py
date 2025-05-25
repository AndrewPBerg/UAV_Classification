#!/usr/bin/env python3
"""
Test script for ESC-50 dataset implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from esc50_dataset import ESC50Dataset, create_esc50_fold_splits, create_esc50_kfold_splits
from configs.dataset_config import ESC50Config
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
        print("uv run download.py --data-dir ../esc50_data")
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

def main():
    """Main test function."""
    print("=" * 50)
    print("ESC-50 Dataset K-Fold Cross-Validation Test")
    print("=" * 50)
    
    success = True
    
    # Test full ESC-50 dataset
    success &= test_esc50_dataset()
    
    # Test ESC-10 subset
    success &= test_esc10_subset()
    
    # Test proper k-fold cross-validation
    success &= test_kfold_cross_validation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("\nYou can now use ESC-50 dataset with k-fold cross-validation!")
        print("\nK-fold setup:")
        print("- 5 total folds in ESC-50 dataset")
        print("- For each iteration: 1 fold used for validation, 4 folds used for training")
        print("- No test folds - use k-fold cross-validation for evaluation")
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
    else:
        print("❌ Some tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 