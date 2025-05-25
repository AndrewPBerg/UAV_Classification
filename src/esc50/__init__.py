"""
ESC-50 Dataset Package

This package contains ESC-50 specific dataset handling components including:
- ESC50Dataset: Custom dataset class for ESC-50 audio data
- ESC50DataModule: PyTorch Lightning data module for ESC-50
- Utility functions for ESC-50 fold-based cross-validation
"""

# Import main dataset class
from .esc50_dataset import (
    ESC50Dataset,
    create_esc50_fold_splits,
    create_esc50_kfold_splits
)

# Import data module
from .esc50_datamodule import (
    ESC50DataModule,
    create_esc50_datamodule
)

__all__ = [
    # Dataset classes and functions
    'ESC50Dataset',
    'create_esc50_fold_splits', 
    'create_esc50_kfold_splits',
    
    # Data module
    'ESC50DataModule',
    'create_esc50_datamodule'
]
