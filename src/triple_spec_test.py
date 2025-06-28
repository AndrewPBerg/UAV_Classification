"""Quick smoke test for TripleMelExtractor.

Run with:
    uv run src/triple_spec_test.py
This creates a 1-second 440 Hz sine wave, extracts the 3-channel
log-mel spectrogram and prints the resulting tensor shape.
"""

import math

import torch

from helper.spectrogram_extractors import TripleMelExtractor


def main() -> None:
    sr = 16_000
    duration_sec = 1.0
    t = torch.linspace(0, duration_sec, int(sr * duration_sec))
    waveform = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)  # (1, n_samples)

    extractor = TripleMelExtractor()
    spec = extractor(waveform, sampling_rate=sr)

    print("Extracted spectrogram shape:", tuple(spec.shape))
    # Expected: (3, 128, 250)


if __name__ == "__main__":
    main() 