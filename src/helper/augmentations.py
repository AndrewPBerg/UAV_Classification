import torch
import numpy as np
from audiomentations import (Compose, PitchShift, TimeStretch, AddGaussianNoise, PolarityInversion, TanhDistortion)
import random
from numpy.typing import NDArray
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import calculate_rms
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel

def create_augmentation_pipeline(augmentations: list[str], config: Dict[str, Any]):
    """
    Create an augmentation pipeline based on the specified augmentations and configuration.
    
    Args:
        augmentations: List of augmentation types to apply
        config: Configuration dictionary containing Pydantic models for each augmentation
        
    Returns:
        Composed augmentation transform or None if no augmentations are specified
    """
    if not augmentations or len(augmentations) == 0:
        print("No augmentations selected, returning original audio. Traceback: Augmentations.py, apply_random_augmentation()")
        return None

    transforms = []
    for aug in augmentations:
        try:
            # Get the config for this augmentation type
            aug_config = config.get(aug)
            
            if aug_config is None:
                print(f"Warning: No configuration found for {aug}. Using default values.")
                
            match aug:
                case "pitch_shift":
                    if aug_config:
                        transforms.append(PitchShift(
                            min_semitones=aug_config.min_semitones,
                            max_semitones=aug_config.max_semitones,
                            p=aug_config.p
                        ))
                    else:
                        # Use default values
                        transforms.append(PitchShift(min_semitones=-5.0, max_semitones=5.0, p=1.0))
                        
                case "time_stretch":
                    if aug_config:
                        transforms.append(TimeStretch(
                            min_rate=aug_config.min_rate,
                            max_rate=aug_config.max_rate,
                            p=aug_config.p
                        ))
                    else:
                        # Use default values
                        transforms.append(TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0))
                        
                case "tanh_distortion":
                    if aug_config:
                        transforms.append(TanhDistortion(
                            min_distortion=aug_config.min_distortion,
                            max_distortion=aug_config.max_distortion,
                            p=aug_config.p
                        ))
                    else:
                        # Use default values
                        transforms.append(TanhDistortion(min_distortion=0.01, max_distortion=0.7, p=1.0))
                        
                case "sin_distortion":
                    if aug_config:
                        transforms.append(SinDistortion(
                            min_distortion=aug_config.min_distortion,
                            max_distortion=aug_config.max_distortion,
                            p=aug_config.p
                        ))
                    else:
                        # Use default values
                        transforms.append(SinDistortion(min_distortion=0.01, max_distortion=0.7, p=1.0))
                        
                case "add_noise":
                    if aug_config:
                        transforms.append(AddGaussianNoise(
                            min_amplitude=aug_config.min_amplitude,
                            max_amplitude=aug_config.max_amplitude,
                            p=aug_config.p
                        ))
                    else:
                        # Use default values
                        transforms.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0))
                        
                case "polarity_inversion":
                    if aug_config:
                        transforms.append(PolarityInversion(p=aug_config.p))
                    else:
                        # Use default values
                        transforms.append(PolarityInversion(p=1.0))
                        
                case _:
                    print(f"Unknown augmentation: {aug}. Skipping.")
        except Exception as e:
            print(f"Error setting up augmentation {aug}: {str(e)}. Using default values.")
            # Continue with next augmentation instead of failing

    # Compose all the selected transforms
    if transforms:
        transform = Compose(transforms)
        return transform
    else:
        return None


def apply_augmentations(audio: torch.Tensor, transform, sr: int) -> torch.Tensor:
    """
    Apply augmentations to the audio tensor.
    
    Args:
        audio: Audio tensor to augment
        transform: Augmentation transform to apply
        sr: Sample rate of the audio
        
    Returns:
        Augmented audio tensor
    """
    # Apply the composed transform to the audio
    audio_numpy = np.ascontiguousarray(audio)
    augmented_audio = transform(samples=audio_numpy, sample_rate=int(sr))
    
    return torch.from_numpy(augmented_audio).float()


class SinDistortion(BaseWaveformTransform):
    """
    Apply sine distortion to the audio. This technique adds harmonics and changes
    the timbre of the sound.
    """
    supports_multichannel = True

    def __init__(
        self, min_distortion: float = 0.01, max_distortion: float = 0.7, p: float = 0.5
    ):
        """
        Initialize the SinDistortion transform.
        
        Args:
            min_distortion: Minimum amount of distortion (between 0 and 1)
            max_distortion: Maximum amount of distortion (between 0 and 1)
            p: The probability of applying this transform
        """
        super().__init__(p)
        assert 0 <= min_distortion <= 1
        assert 0 <= max_distortion <= 1
        assert min_distortion <= max_distortion
        self.min_distortion = min_distortion
        self.max_distortion = max_distortion

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            # Fix: Explicitly define the type of distortion_amount as float
            self.parameters["distortion_amount"] = float(random.uniform(
                self.min_distortion, self.max_distortion
            ))

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        # Find out how much to pre-gain the audio to get a given amount of distortion
        # Fix: Ensure distortion_amount is treated as a float
        distortion_amount = float(self.parameters["distortion_amount"])
        percentile = 100 - 99 * distortion_amount
        threshold = np.percentile(np.abs(samples), percentile)
        gain_factor = 0.5 / (threshold + 1e-6)

        # Distort the audio
        distorted_samples = np.sin(gain_factor * samples)

        # Scale the output so its loudness matches the input
        rms_before = calculate_rms(samples)
        if rms_before > 1e-9:
            rms_after = calculate_rms(distorted_samples)
            post_gain = rms_before / rms_after
            distorted_samples = post_gain * distorted_samples

        return distorted_samples