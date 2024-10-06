import random
import librosa
import torch

def apply_augmentations(
    audio_tensor: torch.Tensor,
    noise_factor: float = 0.005,
    target_sr: int = 44100,
    target_duration: int = 5
) -> torch.Tensor:
    """
    Applies a series of augmentations to the audio tensor.

    Args:
        audio_tensor (torch.Tensor): The original audio tensor.
        noise_factor (float, optional): Scaling factor for adding Gaussian noise. Defaults to 0.005.
        target_sr (int, optional): Target sampling rate. Defaults to 44100.
        target_duration (int, optional): Target duration in seconds. Defaults to 5.

    Returns:
        torch.Tensor: The augmented audio tensor.
    """
    augmented_audio = audio_tensor.clone()

    # # 1. Time Stretching
    # rate = random.uniform(0.8, 1.2)
    # augmented_audio = time_stretch(augmented_audio, rate)

    # # 2. Pitch Shifting
    # n_steps = random.randint(-5, 5)
    # augmented_audio = pitch_shift(augmented_audio, n_steps, target_sr)

    # 3. Adding Gaussian Noise
    augmented_audio = add_noise(augmented_audio, noise_factor)

    # # 4. Volume Control
    # gain = random.uniform(0.5, 1.5)
    # augmented_audio = volume_control(augmented_audio, gain)

    # # 5. Time Shifting
    # max_shift = int(0.1 * target_sr)  # 0.1 seconds shift
    # augmented_audio = time_shift(augmented_audio, max_shift)

    return augmented_audio

def time_stretch(audio_tensor: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Stretches the audio in time without altering the pitch.

    Args:
        audio_tensor (torch.Tensor): The audio tensor.
        rate (float): Stretch rate.

    Returns:
        torch.Tensor: Time-stretched audio tensor.
    """
    audio = audio_tensor.squeeze().numpy()
    stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
    return torch.tensor(stretched_audio).unsqueeze(0)

def pitch_shift(audio_tensor: torch.Tensor, n_steps: int, sr: int) -> torch.Tensor:
    """
    Shifts the pitch of the audio up or down.

    Args:
        audio_tensor (torch.Tensor): The audio tensor.
        n_steps (int): Number of steps to shift.
        sr (int): Sampling rate.

    Returns:
        torch.Tensor: Pitch-shifted audio tensor.
    """
    audio = audio_tensor.squeeze().numpy()
    shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return torch.tensor(shifted_audio).unsqueeze(0)

def add_noise(audio_tensor: torch.Tensor, noise_factor: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the audio tensor.

    Args:
        audio_tensor (torch.Tensor): The audio tensor.
        noise_factor (float): Noise scaling factor.

    Returns:
        torch.Tensor: Noisy audio tensor.
    """
    noise = torch.randn_like(audio_tensor) * noise_factor
    return audio_tensor + noise

def volume_control(audio_tensor: torch.Tensor, gain: float) -> torch.Tensor:
    """
    Adjusts the volume of the audio tensor.

    Args:
        audio_tensor (torch.Tensor): The audio tensor.
        gain (float): Gain factor.

    Returns:
        torch.Tensor: Volume-controlled audio tensor.
    """
    return audio_tensor * gain

def time_shift(audio_tensor: torch.Tensor, max_shift: int) -> torch.Tensor:
    """
    Shifts the audio tensor forward or backward in time.

    Args:
        audio_tensor (torch.Tensor): The audio tensor.
        max_shift (int): Maximum number of samples to shift.

    Returns:
        torch.Tensor: Time-shifted audio tensor.
    """
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(audio_tensor, shifts=shift)