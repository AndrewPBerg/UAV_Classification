"""
Core functionality of current AudioDataset?

1. load audio from datapath
2. feature extraction
3. apply & inflate w/ augmentations (naunce with static augmentations data paths):
    [not sure yet how to handle, might just force creation of new augmentation dataset with each new augmentation permutation]
4. iterable
5. getter helper methods
6. show sepctrogram of i
"""



"""
What should the new PL AudioDataset look like?

1. maintain and speed up above functionality 
2. higher readibility, maintainability, and modularity
3. ...

"""

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
from torchaudio.transforms import Resample
from pathlib import Path
from typing import Union, Optional, List, Dict
import numpy as np

from src.helper.util import find_classes, create_augmentation_pipeline, apply_augmentations
from src.helper.feature_extractors import (
    ASTFeatureExtractor,
    SeamlessM4TFeatureExtractor,
    MelSpectrogramFeatureExtractor,
    MFCCFeatureExtractor
)

class AudioDatasetIterator(Dataset):
    """Dataset class for audio data handling."""
    
    def __init__(
        self,
        data_paths: List[str],
        feature_extractor,
        class_to_idx: Dict[str, int],
        target_sr: int,
        target_duration: int,
        num_channels: int = 1,
        augmentations_per_sample: int = 0,
        augmentations: List[str] = [],
        config: Dict = None
    ):
        self.paths = data_paths
        self.feature_extractor = feature_extractor
        self.class_to_idx = class_to_idx
        self.idx_to_class = {value: key for key, value in class_to_idx.items()}
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = target_duration * target_sr
        
        total_samples = (augmentations_per_sample + 1) * len(self.paths)
        self.audio_tensors = torch.empty(total_samples, num_channels, self.target_sr * target_duration)
        self.class_indices = []
        
        # Setup augmentations if needed
        self.augmentations_per_sample = augmentations_per_sample
        if augmentations_per_sample > 0:
            self.composed_transform = create_augmentation_pipeline(augmentations, config)
            
        # Load and process audio data
        self._load_and_process_audio()
    
    def _load_and_process_audio(self):
        """Load and process all audio files including augmentations."""
        for index, path in enumerate(self.paths):
            audio_tensor, class_idx = self._load_single_audio(path)
            self.audio_tensors[index] = audio_tensor
            self.class_indices.append(class_idx)
            
        # Apply augmentations if specified
        if self.augmentations_per_sample > 0:
            self._apply_augmentations()
    
    def _load_single_audio(self, path: str):
        """Load and preprocess a single audio file."""
        audio_tensor, sr = torchaudio.load(str(path))
        resampler = Resample(sr, self.target_sr)
        audio_tensor = resampler(audio_tensor)
        
        # Convert to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        
        # Pad or trim to target length
        if audio_tensor.shape[1] < self.target_length:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.target_length - audio_tensor.shape[1]))
        else:
            audio_tensor = audio_tensor[:, :self.target_length]
        
        class_name = Path(path).parent.name
        class_idx = self.class_to_idx[class_name]
        return audio_tensor, class_idx
    
    def _apply_augmentations(self):
        """Apply augmentations to the loaded audio samples."""
        original_samples = len(self.paths)
        for i in range(original_samples):
            for j in range(self.augmentations_per_sample):
                new_index = original_samples + i * self.augmentations_per_sample + j
                self.class_indices.append(self.class_indices[i])
                augmented_audio = apply_augmentations(
                    self.audio_tensors[i], 
                    self.composed_transform, 
                    self.target_sr
                )
                self.audio_tensors[new_index] = augmented_audio
    
    def __len__(self):
        return len(self.audio_tensors)
    
    def __getitem__(self, idx):
        audio = self.audio_tensors[idx]
        features = self.feature_extractor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt"
        )
        return features, self.class_indices[idx]

class AudioDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for audio data."""
    
    def __init__(
        self,
        data_path: str,
        feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
        target_sr: int = 16000,
        target_duration: int = 5,
        batch_size: int = 32,
        num_workers: int = 4,
        augmentations_per_sample: int = 0,
        augmentations: List[str] = [],
        train_val_test_split: List[float] = [0.7, 0.15, 0.15],
        num_channels: int = 1,
        config: Dict = None
    ):
        super().__init__()
        self.data_path = data_path
        self.feature_extractor = feature_extractor
        self.target_sr = self._get_target_sr(feature_extractor, target_sr)
        self.target_duration = target_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentations_per_sample = augmentations_per_sample
        self.augmentations = augmentations
        self.train_val_test_split = train_val_test_split
        self.num_channels = num_channels
        self.config = config
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _get_target_sr(self, feature_extractor, default_sr):
        """Get target sampling rate from feature extractor if available."""
        if isinstance(feature_extractor, (MelSpectrogramFeatureExtractor, MFCCFeatureExtractor)):
            return feature_extractor.sampling_rate
        return getattr(feature_extractor, 'sampling_rate', default_sr)
    
    def prepare_data(self):
        """Called only once and on 1 GPU."""
        # Here you could download data if needed
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Called on every GPU."""
        # Get all audio file paths and class information
        all_paths = []
        for class_dir in Path(self.data_path).iterdir():
            if class_dir.is_dir():
                all_paths.extend([str(p) for p in class_dir.glob('*.wav')])
                
        # Get class mapping
        self.classes, self.class_to_idx = find_classes(self.data_path)
        
        # Split data
        np.random.shuffle(all_paths)
        n_total = len(all_paths)
        n_train = int(self.train_val_test_split[0] * n_total)
        n_val = int(self.train_val_test_split[1] * n_total)
        
        train_paths = all_paths[:n_train]
        val_paths = all_paths[n_train:n_train + n_val]
        test_paths = all_paths[n_train + n_val:]
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = AudioDatasetIterator(
                train_paths, self.feature_extractor, self.class_to_idx,
                self.target_sr, self.target_duration, self.num_channels,
                self.augmentations_per_sample, self.augmentations, self.config
            )
            self.val_dataset = AudioDatasetIterator(
                val_paths, self.feature_extractor, self.class_to_idx,
                self.target_sr, self.target_duration, self.num_channels
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = AudioDatasetIterator(
                test_paths, self.feature_extractor, self.class_to_idx,
                self.target_sr, self.target_duration, self.num_channels
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )