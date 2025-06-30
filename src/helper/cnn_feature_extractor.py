import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC
from dataclasses import dataclass

@dataclass
class MelSpectrogramFeatureExtractor:
    """
    Feature extractor that handles MelSpectrogram transformation.
    Designed to have a similar interface to ASTFeatureExtractor for compatibility.
    """
    sampling_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    power: float = 2.0

    def __post_init__(self):
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power
        )

    def __call__(self, audio, sampling_rate=None, return_tensors=None):
        """
        Process audio data to create mel spectrogram features.
        
        Args:
            audio: Audio data (numpy array or tensor)
            sampling_rate: Original sampling rate (not used, kept for compatibility)
            return_tensors: Format of return tensors (not used, kept for compatibility)
            
        Returns:
            Object with input_values attribute containing the mel spectrogram
        """
        # print("You are using the MelSpectrogramFeatureExtractor, which is a custom feature extractor designed to mimic the ASTFeatureExtractor interface.")
        if isinstance(audio, torch.Tensor):
            waveform = audio
        else:
            waveform = torch.from_numpy(audio)

        # Add batch dimension if not present
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Extract mel spectrogram features
        mel_spec = self.mel_transform(waveform)  # Shape: [1, n_mels, time]
        
        # Convert to decibel scale
        mel_spec = torch.log10(mel_spec + 1e-9)
        
        # Create a simple namespace to match ASTFeatureExtractor interface
        class FeatureExtractorOutput:
            def __init__(self, input_values):
                self.input_values = input_values
        
        return FeatureExtractorOutput(mel_spec)

@dataclass
class MFCCFeatureExtractor:
    """
    Feature extractor that handles MFCC transformation.
    Designed to have a similar interface to ASTFeatureExtractor for compatibility.
    """
    sampling_rate: int = 16000
    n_mfcc: int = 40
    melkwargs: dict = None
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    
    def __post_init__(self):
        if self.melkwargs is None:
            self.melkwargs = {
                'n_mels': self.n_mels,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length
            }
        
        self.mfcc_transform = MFCC(
            sample_rate=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            melkwargs=self.melkwargs
        )

    def __call__(self, audio, sampling_rate=None, return_tensors=None):
        """
        Process audio data to create MFCC features.
        
        Args:
            audio: Audio data (numpy array or tensor)
            sampling_rate: Original sampling rate (not used, kept for compatibility)
            return_tensors: Format of return tensors (not used, kept for compatibility)
            
        Returns:
            Object with input_values attribute containing the MFCC features
        """
        if isinstance(audio, torch.Tensor):
            waveform = audio
        else:
            waveform = torch.from_numpy(audio)

        # Add batch dimension if not present
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Extract MFCC features
        mfcc = self.mfcc_transform(waveform)  # Shape: [1, n_mfcc, time]
        
        # Create a simple namespace to match ASTFeatureExtractor interface
        class FeatureExtractorOutput:
            def __init__(self, input_values):
                self.input_values = input_values
        
        return FeatureExtractorOutput(mfcc)
