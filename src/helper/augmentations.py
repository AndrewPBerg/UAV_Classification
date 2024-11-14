import torch
import numpy as np
from audiomentations import (Compose, PitchShift, TimeStretch, AddGaussianNoise, PolarityInversion, TanhDistortion)
import random
from numpy.typing import NDArray
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import calculate_rms

def create_augmentation_pipeline(augmentations: list[str], config: dict):
    if len(augmentations) == 0:
        print("No augmentations selected, returning original audio. Traceback: Augmentations.py, apply_random_augmentation()")
        return None

    transforms = []
    for aug in augmentations:
        match aug:
            case "pitch_shift":
                transforms.append(PitchShift(min_semitones=config['pitch_shift_min_rate'],
                                             max_semitones=config['pitch_shift_max_rate'],
                                             p=1.0))
            case "time_stretch":
                transforms.append(TimeStretch(min_rate=config['time_stretch_min_rate'],
                                              max_rate=config['time_stretch_max_rate'],
                                              p=1.0))
            case "tanh_distortion":
                transforms.append(TanhDistortion(min_distortion=config['tanh_distortion_min_rate'],
                                              max_distortion=config['tanh_distortion_max_rate'],
                                              p=1.0))
            case "sin_distortion":
                transforms.append(SinDistortion(min_distortion=config['sin_distortion_min_rate'],
                                              max_distortion=config['sin_distortion_max_rate'],
                                              p=1.0))
            case "add_noise":
                transforms.append(AddGaussianNoise(min_amplitude=config['add_noise_min_amplitude'], 
                                                   max_amplitude=config['add_noise_max_amplitude'],
                                                   p=1.0))
            case "polarity_inversion":
                transforms.append(PolarityInversion(p=1.0))
            case _:
                print(f"Unknown augmentation: {aug}. Skipping.")
        
        # Compose all the selected transforms
        transform = Compose(transforms)

        return transform



def apply_augmentations(audio: torch.Tensor, transform, sr: int) -> torch.Tensor:

    # Apply the composed transform to the audio
    audio_numpy = np.ascontiguousarray(audio)
    augmented_audio = transform(samples=audio_numpy, sample_rate=int(sr))
    
    return torch.from_numpy(augmented_audio).float()


class SinDistortion(BaseWaveformTransform):

    """
    Apply tanh (hyperbolic tangent) distortion to the audio. This technique is sometimes
    used for adding distortion to guitar recordings. The tanh() function can give a rounded
    "soft clipping" kind of distortion, and the distortion amount is proportional to the
    loudness of the input and the pre-gain. Tanh is symmetric, so the positive and
    negative parts of the signal are squashed in the same way. This transform can be
    useful as data augmentation because it adds harmonics. In other words, it changes
    the timbre of the sound.

    See this page for examples: http://gdsp.hf.ntnu.no/lessons/3/17/
    """

    supports_multichannel = True

    def __init__(
        self, min_distortion: float = 0.01, max_distortion: float = 0.7, p: float = 0.5
    ):
        """
        :param min_distortion: Minimum amount of distortion (between 0 and 1)
        :param max_distortion: Maximum amount of distortion (between 0 and 1)
        :param p: The probability of applying this transform
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
            self.parameters["distortion_amount"] = random.uniform(
                self.min_distortion, self.max_distortion
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        # Find out how much to pre-gain the audio to get a given amount of distortion
        percentile = 100 - 99 * self.parameters["distortion_amount"]
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