# Purpose of this script is to generate all of the feature extracted spectrograms from the audio files. 
# Note that the audio files are not included in this repository

import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    DIR_NAME = 'UAV_31_MEL_SPECTROGRAM_PLOTS'
    print(f"Starting to generate spectrograms in directory: {DIR_NAME}")
    # 'UAV_9_MEL_SPECTROGRAM_PLOTS'
    # Get the current file's directory and navigate to .datasets
    current_dir = Path(__file__).resolve().parent
    datasets_dir = current_dir.parent / '.datasets' / 'UAV_Dataset_31'
    print(f"Datasets directory: {datasets_dir}")

    # Function to get all audio files recursively
    def get_audio_files(root_dir: Path):
        return list(root_dir.rglob("*.wav"))

    # Get all WAV files from datasets
    audio_files = get_audio_files(datasets_dir)
    print(f"Found {len(audio_files)} audio files.")

    # Setup spectrogram parameters
    sampling_rate = 16000
    n_mels = 128
    n_fft = 1024
    hop_length = 512
    power = 2.0

    # Create mel spectrogram transform
    mel_transform = MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power
    )

    # Process each audio file
    for audio_path in audio_files:
        try:
            print(f"Processing audio file: {audio_path}")
            # Load the audio file
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Generate mel spectrogram
            mel_spec = mel_transform(waveform)
            mel_spec_db = torch.log10(mel_spec + 1e-9)
            
            # Create output directory for spectrograms
            output_dir = current_dir / DIR_NAME / audio_path.parent.name
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory created: {output_dir}")
            
            # Plot and save
            plt.figure(figsize=(12, 8))
            plt.imshow(mel_spec_db[0].numpy(), aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram - {audio_path.stem}')
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Frequency Bins')
            
            # Save the plot
            output_path = output_dir / f"{audio_path.stem}_mel_spectrogram.png"
            plt.savefig(str(output_path), format='png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved spectrogram to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")

if __name__ == '__main__':
    main()
    print("All done!")