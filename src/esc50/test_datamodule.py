#!/usr/bin/env python3
"""
Test script for ESC-50 datamodule implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from esc50.esc50_datamodule import ESC50DataModule
from configs.dataset_config import ESC50Config
from configs import GeneralConfig, FeatureExtractionConfig, AugConfig
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor

def test_esc50_datamodule():
    """Test ESC-50 datamodule functionality."""
    print("="*50)
    print("ESC-50 DataModule Test")
    print("="*50)
    
    # Configuration
    data_path = "../esc50_data/ESC-50-master/classes"
    
    # Check if dataset exists
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}")
        print("Please run the download script first:")
        print("uv run src/download_esc50.py --data-dir ../esc50_data")
        return False
    
    # Create configurations
    general_config = GeneralConfig(
        batch_size=8,
        model_type="efficientnet_b0",
        num_cuda_workers=2,
        seed=42,
        use_kfold=True,
        k_folds=5,
        pinned_memory=True,
        use_sweep=False,
        sweep_count=10
    )
    
    feature_extraction_config = FeatureExtractionConfig(
        type='melspectrogram',
        sampling_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        power=2
    )
    
    esc50_config = ESC50Config(
        data_path=data_path,
        use_esc10_subset=True,  # Use smaller subset for faster testing
        fold_based_split=True,
        target_sr=16000,
        target_duration=5
    )
    
    augmentation_config = AugConfig(
        augmentations_per_sample=1,
        augmentations=["time_stretch"],
        aug_configs={}
    )
    
    print(f"Dataset type: {esc50_config.dataset_type}")
    print(f"Number of classes: {esc50_config.get_num_classes()}")
    print(f"Use ESC-10 subset: {esc50_config.use_esc10_subset}")
    
    try:
        # Create data module
        data_module = ESC50DataModule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            esc50_config=esc50_config,
            augmentation_config=augmentation_config
        )
        print("✅ Created ESC-50 data module")
        
        # Setup data module
        data_module.setup()
        print("✅ Setup ESC-50 data module")
        
        # Get metadata and fold information
        metadata_info = data_module.get_metadata_info()
        fold_info = data_module.get_fold_info()
        
        print(f"\nMetadata info: {metadata_info}")
        print(f"Fold info: {fold_info}")
        
        # Test k-fold cross-validation
        print(f"\nTesting k-fold cross-validation ({data_module.k_folds} folds)...")
        
        for fold_idx in range(data_module.k_folds):
            data_module.set_fold(fold_idx)
            
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            
            # Get a sample batch
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            
            train_samples = getattr(train_loader.dataset, '__len__', lambda: 'unknown')()
            val_samples = getattr(val_loader.dataset, '__len__', lambda: 'unknown')()
            
            print(f"Fold {fold_idx+1}: Train batch shape: {train_batch[0].shape}, Val batch shape: {val_batch[0].shape}")
            print(f"Fold {fold_idx+1}: Train samples: {train_samples}, Val samples: {val_samples}")
            
            # Verify batch contents
            assert train_batch[0].dim() == 4, f"Expected 4D tensor, got {train_batch[0].dim()}D"
            assert val_batch[0].dim() == 4, f"Expected 4D tensor, got {val_batch[0].dim()}D"
            assert train_batch[1].dim() == 1, f"Expected 1D label tensor, got {train_batch[1].dim()}D"
            assert val_batch[1].dim() == 1, f"Expected 1D label tensor, got {val_batch[1].dim()}D"
        
        # Test class information
        classes, class_to_idx, idx_to_class = data_module.get_class_info()
        print(f"\nClasses: {classes[:5]}...")  # Show first 5 classes
        print(f"Number of classes: {len(classes)}")
        
        # Test fold dataloaders method
        print(f"\nTesting get_fold_dataloaders method...")
        train_loader_0, val_loader_0 = data_module.get_fold_dataloaders(0)
        train_samples_0 = getattr(train_loader_0.dataset, '__len__', lambda: 'unknown')()
        val_samples_0 = getattr(val_loader_0.dataset, '__len__', lambda: 'unknown')()
        print(f"Fold 1 via get_fold_dataloaders: Train: {train_samples_0}, Val: {val_samples_0}")
        
        print("\n✅ All ESC-50 DataModule tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ESC-50 datamodule: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_fold_mode():
    """Test ESC-50 datamodule in single fold mode."""
    print("\n" + "="*50)
    print("ESC-50 DataModule Single Fold Test")
    print("="*50)
    
    data_path = "../esc50_data/ESC-50-master/classes"
    
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}")
        return False
    
    # Create configurations for single fold
    general_config = GeneralConfig(
        batch_size=8,
        num_cuda_workers=2,
        model_type="efficientnet_b0",
        seed=42,
        use_kfold=False,  # Single fold mode
        k_folds=5,
        pinned_memory=True,
        use_sweep=False,
        sweep_count=10
    )
    
    feature_extraction_config = FeatureExtractionConfig(
        type='melspectrogram',
        sampling_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        power=2
    )
    
    esc50_config = ESC50Config(
        data_path=data_path,
        use_esc10_subset=True,
        fold_based_split=True,
        target_sr=16000,
        target_duration=5
    )
    
    try:
        # Create data module
        data_module = ESC50DataModule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            esc50_config=esc50_config
        )
        print("✅ Created ESC-50 data module (single fold mode)")
        
        # Setup data module
        data_module.setup()
        print("✅ Setup ESC-50 data module")
        
        # Test single fold
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Get a sample batch
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        train_samples = getattr(train_loader.dataset, '__len__', lambda: 'unknown')()
        val_samples = getattr(val_loader.dataset, '__len__', lambda: 'unknown')()
        
        print(f"Single fold: Train batch shape: {train_batch[0].shape}, Val batch shape: {val_batch[0].shape}")
        print(f"Single fold: Train samples: {train_samples}, Val samples: {val_samples}")
        
        print("✅ Single fold mode test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing single fold mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing ESC-50 DataModule Implementation")
    print("="*60)
    
    success = True
    
    # Test k-fold mode
    if not test_esc50_datamodule():
        success = False
    
    # Test single fold mode
    if not test_single_fold_mode():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 All tests passed!")
        print("\nYou can now use ESC-50 DataModule with PyTorch Lightning!")
        print("\nExample usage:")
        print("1. Update your config.yaml to use ESC-50:")
        print("   dataset_type: esc50")
        print("   data_path: /path/to/esc50/classes")
        print("   use_esc10_subset: false")
        print("   fold_based_split: true")
        print("2. Use ESC50DataModule in your training pipeline")
        print("3. For k-fold cross-validation, iterate through folds:")
        print("   for fold_idx in range(5):")
        print("       data_module.set_fold(fold_idx)")
        print("       # Train on this fold")
    else:
        print("❌ Some tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 