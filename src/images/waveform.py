import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

def save_waveform_image(load_path, save_path, no_text=False):
    """
    Load a .wav file and save it as a waveform plot.
    """

    # Read the .wav file
    sample_rate, data = wavfile.read(load_path)

    # If stereo, take only one channel
    if len(data.shape) > 1:
        data = data[:, 0]

    plt.figure(figsize=(10, 4))
    plt.plot(data)
    if not no_text:
        plt.title("Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
    else:
        # Remove axis labels and ticks if no_text is True
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Remove all spines (borders)
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()

    # If save_path is a directory, create a filename.
    # Otherwise, use save_path as the full path to the file.
    if os.path.isdir(save_path):
        base_filename = os.path.basename(load_path)
        name, _ = os.path.splitext(base_filename)
        output_path = os.path.join(save_path, f"{name}_waveform.png")
    else:
        output_path = save_path
    
    # Ensure the directory for the output file exists.
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved waveform to {output_path}")


def main():
    # This call was failing because save_path was a directory. It will now work.
    save_waveform_image(
        load_path="/Users/applefella/Documents/UAV_Classification/UAV_Classification/.datasets/UAV_Dataset_31/DJI_Tello/DJI_Tello_1.wav",
        save_path="src/images",
        no_text=True
    )
    
    # This call provides a full file path and will continue to work.
    save_waveform_image(
        load_path="/Users/applefella/Documents/UAV_Classification/UAV_Classification/.datasets/UAV_Dataset_31/DJI_Tello/DJI_Tello_3.wav",
        save_path="src/images/DJI_Tello_3_waveform.png",
        no_text=True
    )


if __name__ == "__main__":
    main()
