import torch
import numpy as np
from audiomentations import (Compose, PitchShift, TimeStretch, AddGaussianNoise, 
    Shift, TimeMask, Reverse, Normalize, GainTransition, PolarityInversion, Gain)

def create_augmentation_pipeline(augmentations: list[str], config: dict):
    if len(augmentations) == 0:
        print("No augmentations selected, returning original audio. Traceback: Augmentations.py, apply_random_augmentation()")
        return None

    transforms = []
    for aug in augmentations:
        if aug == "pitch_shift":
            transforms.append(PitchShift(min_semitones=config['pitch_shift_min_rate'],
                                         max_semitones=config['pitch_shift_max_rate'],
                                         p=1.0))
        elif aug == "time_stretch":
            transforms.append(TimeStretch(min_rate=config['time_stretch_min_rate'],
                                          max_rate=config['time_stretch_max_rate'],
                                          p=1.0))
        elif aug == "AddGaussianNoise":
            transforms.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0))
        elif aug == "Shift":
            transforms.append(Shift(min_shift=1.0, max_shift=2.0, shift_unit="seconds", rollover=True, fade_duration=0.01, p=1.0))
        elif aug == "TimeMask":
            transforms.append(TimeMask(min_band_part=0.0, max_band_part=0.5, p=1.0))
        elif aug == "Reverse":
            transforms.append(Reverse(p=1.0))
        elif aug == "Normalize":
            transforms.append(Normalize(apply_to="all", p=1.0))
        elif aug == "GainTransition":
            transforms.append(GainTransition(min_gain_db=-12, max_gain_db=0, p=1.0))
        elif aug == "PolarityInversion":
            transforms.append(PolarityInversion(p=1.0))
        elif aug == "Gain":
            transforms.append(Gain(min_gain_db=-12, max_gain_db=0, p=1.0))
        else:
            print(f"Unknown augmentation: {aug}. Skipping.")
        
        # Compose all the selected transforms
        transform = Compose(transforms)

        return transform



def apply_augmentations(audio: torch.Tensor, transform, sr: int) -> torch.Tensor:




    # Apply the composed transform to the audio
    audio_numpy = np.ascontiguousarray(audio)
    augmented_audio = transform(samples=audio_numpy, sample_rate=int(sr))
    
    return torch.from_numpy(augmented_audio).float()