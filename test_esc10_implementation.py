#!/usr/bin/env python3
"""
Comprehensive test script for ESC-10 implementation.
Tests both the dataset creation and the dataset/datamodule functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_esc10_download():
    """Test ESC-10 dataset creation from ESC-50."""
    print("🧪 Testing ESC-10 dataset creation...")
    
    try:
        from esc10.download import download_esc10
        
        # Create ESC-10 dataset
        dataset_path = download_esc10()
        print(f"✅ ESC-10 dataset created successfully at: {dataset_path}")
        
        # Verify the dataset
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            raise FileNotFoundError(f"Dataset path doesn't exist: {dataset_path}")
        
        # Count categories and files
        categories = [d for d in dataset_path_obj.iterdir() if d.is_dir()]
        print(f"📊 Found {len(categories)} categories:")
        
        total_files = 0
        for cat_dir in categories:
            wav_files = list(cat_dir.glob("*.wav"))
            total_files += len(wav_files)
            print(f"  - {cat_dir.name}: {len(wav_files)} files")
        
        print(f"📈 Total files: {total_files}")
        
        # Verify expected structure
        if len(categories) != 10:
            raise ValueError(f"Expected 10 categories, found {len(categories)}")
        if total_files != 400:
            raise ValueError(f"Expected 400 files, found {total_files}")
        
        return dataset_path
        
    except Exception as e:
        print(f"❌ ESC-10 dataset creation failed: {e}")
        raise

def test_esc10_dataset(dataset_path):
    """Test ESC-10 dataset functionality."""
    print("\n🧪 Testing ESC-10 dataset functionality...")
    
    try:
        from esc10.esc10_dataset import create_esc10_kfold_splits
        from transformers import ASTFeatureExtractor
        
        # Create feature extractor
        feature_extractor = ASTFeatureExtractor()
        
        # Test k-fold splits
        fold_splits = create_esc10_kfold_splits(
            data_path=dataset_path,
            feature_extractor=feature_extractor
        )
        
        print(f"✅ Created {len(fold_splits)} k-fold splits")
        
        # Test each fold
        for i, (train_dataset, val_dataset) in enumerate(fold_splits):
            fold_num = i + 1
            train_size = len(train_dataset.file_names)
            val_size = len(val_dataset.file_names)
            
            print(f"  Fold {fold_num}: Train={train_size}, Val={val_size}")
            
            # Verify fold sizes
            if train_size != 320:
                raise ValueError(f"Expected 320 training files in fold {fold_num}, got {train_size}")
            if val_size != 80:
                raise ValueError(f"Expected 80 validation files in fold {fold_num}, got {val_size}")
        
        print("✅ All fold splits validated successfully")
        
    except Exception as e:
        print(f"❌ ESC-10 dataset test failed: {e}")
        raise

def test_esc10_datamodule(dataset_path):
    """Test ESC-10 DataModule functionality."""
    print("\n🧪 Testing ESC-10 DataModule functionality...")
    
    try:
        import yaml
        from configs.configs_aggregate import load_configs
        from esc10.esc10_datamodule import create_esc10_datamodule
        
        # Load base config
        with open('src/configs/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Modify for ESC-10
        config['dataset']['dataset_type'] = 'esc10'
        config['dataset']['data_path'] = dataset_path
        config['general']['use_kfold'] = True
        
        # Load configs
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
        
        print("✅ Configs loaded successfully")
        
        # Create datamodule
        data_module = create_esc10_datamodule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            esc10_config=dataset_config,
            augmentation_config=augmentation_config,
            use_filename_based_splits=True
        )
        
        print("✅ DataModule created successfully")
        
        # Setup datamodule
        data_module.setup()
        print("✅ DataModule setup successfully")
        
        # Test fold dataloaders
        train_loader, val_loader = data_module.get_fold_dataloaders(0)
        
        print(f"✅ Dataloaders created - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Test class info
        classes, class_to_idx, idx_to_class = data_module.get_class_info()
        
        if len(classes) != 10:
            raise ValueError(f"Expected 10 classes, got {len(classes)}")
        
        print(f"✅ Found {len(classes)} classes: {classes}")
        
    except Exception as e:
        print(f"❌ ESC-10 DataModule test failed: {e}")
        raise

def main():
    """Main test function."""
    print("🚀 Starting ESC-10 implementation tests...")
    
    try:
        # Test 1: Dataset creation
        dataset_path = test_esc10_download()
        
        # Test 2: Dataset functionality
        test_esc10_dataset(dataset_path)
        
        # Test 3: DataModule functionality
        test_esc10_datamodule(dataset_path)
        
        print("\n🎉 All tests passed! ESC-10 implementation is working correctly.")
        print(f"\n📍 ESC-10 dataset available at: {dataset_path}")
        print("📖 To use in your training, point data_path to this location and set dataset_type to 'esc10'")
        
    except Exception as e:
        print(f"\n💥 Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 