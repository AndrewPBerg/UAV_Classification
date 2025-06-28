from __future__ import annotations

"""Triple-channel spectrogram extractor.

This helper encapsulates the logic for generating three log-mel spectrograms
with different window / hop sizes and stacking them to obtain a (3, 128, W)
image-like tensor that can be consumed by CNN / ViT style models.  The default
parameters match the spec provided in Task 11-1 but can be overridden via the
constructor (values are expected in *milliseconds* for window / hop sizes).
"""

from typing import List

import numpy as np
from PIL import Image
import torch
import torchaudio
import torchvision.transforms as T

__all__ = ["TripleMelExtractor"]


class TripleMelExtractor:  # pylint: disable=too-few-public-methods
    """Generate a 3-channel log-mel spectrogram.

    The three channels are computed with different (window_size, hop_size)
    tuples that are passed in *milliseconds*.
    """

    def __init__(
        self,
        *,
        window_sizes_ms: List[int] | tuple[int, int, int] = (25, 50, 100),
        hop_sizes_ms: List[int] | tuple[int, int, int] = (10, 25, 50),
        n_mels: int = 128,
        n_fft: int = 4410,
        target_width: int = 250,
        eps: float = 1.0e-6,
    ) -> None:
        if len(window_sizes_ms) != 3 or len(hop_sizes_ms) != 3:
            raise ValueError("Expected exactly three window and hop sizes (one per channel)")

        self.window_sizes_ms = list(window_sizes_ms)
        self.hop_sizes_ms = list(hop_sizes_ms)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.target_width = target_width
        self.eps = eps

        # Keeping a shared resize transform avoids reallocating each call.
        # We resize (H, W) -> (n_mels, target_width) where H == n_mels already.
        self._resize = T.Resize((n_mels, target_width), antialias=True)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def __call__(
        self,
        waveform: torch.Tensor,  # shape: (1, time) or (time,)
        *,
        sampling_rate: int,
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """Convert *waveform* into a stacked 3-channel log-mel spectrogram.

        Parameters
        ----------
        waveform:
            Input audio.  A 1-D or (1, len) Tensor in the range [-1, 1].
        sampling_rate:
            Sampling rate in Hz.
        return_tensors:
            Only "pt" (PyTorch Tensor) is supported for now.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 2 or waveform.size(0) != 1:
            raise ValueError("waveform must have shape (1, n_samples)")

        specs: list[torch.Tensor] = []
        for win_ms, hop_ms in zip(self.window_sizes_ms, self.hop_sizes_ms):
            win_length = int(round(win_ms * sampling_rate / 1000))
            hop_length = int(round(hop_ms * sampling_rate / 1000))

            # MelSpectrogram returns (n_mels, time)
            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_fft=self.n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=self.n_mels,
            )(waveform)

            # Convert to log scale and ensure numerical stability.
            spec = torch.log(spec + self.eps)

            # Resize time axis to *target_width* using PIL for simplicity.
            spec_np = spec.squeeze(0).cpu().numpy()  # shape: (n_mels, time)
            image = Image.fromarray(spec_np).convert("F")  # 32-bit float image
            image = self._resize(image)  # type: ignore[arg-type]
            spec_resized = torch.from_numpy(np.array(image, dtype=np.float32))

            specs.append(spec_resized)

        # Stack to shape (3, n_mels, target_width)
        spec_tensor = torch.stack(specs, dim=0)

        if return_tensors == "pt":
            return spec_tensor
        raise ValueError(f"Unsupported return_tensors value: {return_tensors}") 