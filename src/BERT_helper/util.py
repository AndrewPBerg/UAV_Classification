import os
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import torch
import random
import matplotlib.pyplot as plt
import librosa
from typing import Optional, Union
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor
import warnings
from .augmentations import create_augmentation_pipeline, apply_augmentations      # Assume this function exists
from torchaudio.transforms import Resample
import numpy as np

import wandb

def calculated_load_time(start, end) -> str:
    total_load_time = end - start
    hours = int(total_load_time // 3600)
    minutes = int((total_load_time % 3600) // 60)
    seconds = int(total_load_time % 60)
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    return formatted_time

def wandb_login():
    api_key = os.environ.get('WANDB_API_KEY')
    if not api_key:
        raise ValueError("WANDB_API_KEY environment variable is not set")
    wandb.login(key=api_key)
    
def count_classes(dataset):
    class_counts = {cls: 0 for cls in dataset.get_classes()}
    for _, class_idx in dataset:
        class_name = dataset.get_idx_dict()[class_idx]
        class_counts[class_name] += 1
    return class_counts


class AudioDataset(Dataset):

    def __init__(self,
                 data_path: str, 
                 data_paths: list[str],
                 feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor],
                 standardize_audio_boolean: bool=True, 
                 target_sr: int=16000,  # Changed to match Wav2Vec2 requirements
                 target_duration: int=5, 
                 augmentations_per_sample: int = 0,
                 augmentations: list[str] = [],
                 num_channels: int = 1,
                 config: dict = None) -> Dataset: # type: ignore
        self.paths = data_paths
        self.feature_extractor = feature_extractor
        self.classes, self.class_to_idx = find_classes(data_path)
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        self.sampling_rate = None  # current dataset sample rate
        self.resampler = None
        self.standardize_audio_boolean = standardize_audio_boolean
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = target_duration * target_sr
        self.augmentations_per_sample = augmentations_per_sample
        self.standardize_audio_boolean = standardize_audio_boolean
        self.config = config

        total_samples = (augmentations_per_sample + 1) * len(self.paths)
        self.audio_tensors = torch.empty(total_samples, num_channels, target_sr * target_duration)
        self.class_indices = []

        if self.augmentations_per_sample > 0:
            self.composed_transform = create_augmentation_pipeline(augmentations, self.config)

        # Load original audio samples
        for index, path in enumerate(self.paths):
            audio_to_append, class_idx = self.load_audio(path)
            self.audio_tensors[index] = audio_to_append
            self.class_indices.append(class_idx)

        # Apply augmentations 
        # TODO refactor following this guide using torch.cat once after all augmentations are generated
        # sourse https://discuss.pytorch.org/t/appending-to-a-tensor/2665/3 
        if self.augmentations_per_sample > 0:
            original_samples = len(self.paths)
            for i in range(original_samples):
                for j in range(self.augmentations_per_sample):
                    new_index = original_samples + i * self.augmentations_per_sample + j
                    self.class_indices.append(self.class_indices[i])
                    if len(augmentations) != 0:
                        augmented_audio = apply_augmentations(self.audio_tensors[i], self.composed_transform, self.target_sr)
                        self.audio_tensors[new_index] = augmented_audio
                        # self.audio_tensors.append(augmented_audio())
                    else:
                        self.audio_tensors[new_index] = self.audio_tensors[i]
                        # self.audio_tensors.append(self.audio_tensors[i])

        # if self.augmentations_per_sample > 0:
        #     outputs = []


        # Ensure audio_tensors and class_indices have the same length
        assert len(self.audio_tensors) == len(self.class_indices), "Mismatch between audio_tensors and class_indices"

    def load_audio(self, path:str):
        "load audio from path, return raw tensor"
    

        audio_tensor, sr = torchaudio.load(str(path))
        
        if self.sampling_rate is None:
            self.sampling_rate = sr

        if self.resampler is None:
            self.resampler = Resample(sr, self.target_sr)

        # Convert to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        
        # Pad or trim to target length
        if audio_tensor.shape[1] < self.target_length:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.target_length - audio_tensor.shape[1]))
        else:
            audio_tensor = audio_tensor[:, :self.target_length]
        
        class_name = Path(path).parent.name # Get the class name from the file's name
        class_idx = self.class_to_idx[class_name] # Convert class name to index
        return audio_tensor, class_idx

    
    def __len__(self) -> int:
        if self.augmentations_per_sample > 0:
            return len(self.paths) * (self.augmentations_per_sample + 1)
        else:
            return len(self.paths)
        
    def feature_extraction(self, audio_tensor):
        """Process audio tensor for model input."""
        audio_np = audio_tensor.squeeze().numpy()
        
        if isinstance(self.feature_extractor, ASTFeatureExtractor):
            # For AST feature extractor
            features = self.feature_extractor(
                audio_np,
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            return features.input_values
        else:
            # For Wav2Vec2 processor
            features = self.feature_extractor(
                audio_np,
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            return features.input_values.squeeze(0)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        features = self.feature_extraction(self.audio_tensors[index])
        
        if self.audio_tensors[index].shape[0] == 1:
            features = features.squeeze(dim=0)
        
        return features, self.class_indices[index]

    def get_classes(self) -> list[str]:
        return self.classes
    
    def get_class_dict(self) -> dict[str, int]:
        return self.class_to_idx
        
    def get_idx_dict(self) -> dict[int, str]:
        return self.idx_to_class
    
    def get_feature_extractor(self):
        return self.feature_extractor
    
    def show_spectrogram(self, audio_or_index):
        # Check if the input is an index or an audio tensor
        if isinstance(audio_or_index, int):
            audio, _ = self.__getitem__(audio_or_index)  # Get audio tensor from index
        else:
            audio = audio_or_index  # Assume it's an audio tensor

        # Convert audio features to numpy array
        spectrogram = audio.numpy()
        
        # Get the sampling rate
        if self.sampling_rate is None:
            # If sampling rate is not set, use a default value and issue a warning
            default_sr = 16000  # Choose an appropriate default sampling rate
            warnings.warn(f"Sampling rate not set. Using default value of {default_sr} Hz.", UserWarning)
            sr = default_sr
        else:
            sr = self.sampling_rate
        
        # Plot the spectrogram
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.show()

def display_random_images(dataset: AudioDataset,
                          n: int = 9,  # Set n to 9 for a 3x3 grid layout
                          display_shape: bool = False,
                          seed: Optional[int] = None):
    
    # 2. Adjust display if n is not 9 for 3x3 grid layout
    if n != 9:
        n = 9
        display_shape = False
        print("For a 3x3 grid layout, n is set to 9 and shape display is turned off.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    
    # 5. Setup plot
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))  # 3x3 grid layout
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # 6. Loop through samples and display random samples 
    idx_to_class = dataset.get_idx_dict()
    for i, targ_sample in enumerate(random_samples_idx):
        targ_spec, targ_label_idx = dataset[targ_sample][0], dataset[targ_sample][1]

        # Squeeze targ_spec for right shape for plot function
        targ_spec = targ_spec.squeeze(dim=0)
        targ_spec = targ_spec.numpy() # librosa likes numpy array > pytorch tensor
        # Plot the spectrogram on the corresponding subplot
        ax = axes[i]
        ax.imshow(librosa.power_to_db(S=targ_spec), origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(idx_to_class[targ_label_idx])
        
        # Optionally, display the shape of the spectrogram
        if display_shape:
            ax.set_xlabel(f"Shape: {targ_spec.shape}")

    plt.tight_layout()
    plt.show()

def find_classes(directory: str) -> tuple[list[str], dict[str,int]]:
    """
    Finds class names from folder

    Args:
        directory(str): target directory to load folder names from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: index, ...))
    
    Example:
    ```python
        find_classes("path_name")
        >>>(["class_1", "class_2"], {"class_1": 0, ...})
    ```
    """

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir()) # Gets names of classes by scanning classes folders
    
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.") # Code by fire 
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)} # enumerated class list ex:{class_1 : 0, ...}

    return classes, class_to_idx

def train_test_split_custom(
    data_path: str, 
    feature_extractor, # A featureExtractor object
    test_size: float = 0.2, 
    val_size: float = 0.1,
    inference_size: float = 0.1,
    seed: int = 42, 
    augmentations_per_sample: int = 3,
    augmentations: list[str] = None,
    config: dict = None # type: ignore
):
    def split_dataset(data, val_size, test_size, inference_size, random_state=None):
        train_size = 1.0 - (val_size + test_size + inference_size)
        assert np.isclose(train_size + val_size + test_size + inference_size, 1.0), \
            "The sum of val_size, test_size, and inference_size should be less than 1.0"
        
        n_samples = len(data)
        n_train = int(n_samples * train_size)
        n_val = int(n_samples * val_size)
        n_test = int(n_samples * test_size)
        
        indices = np.random.RandomState(random_state).permutation(n_samples)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:n_train+n_val+n_test]
        inference_indices = indices[n_train+n_val+n_test:]
        
        return train_indices, val_indices, test_indices, inference_indices

    all_paths = list(Path(data_path).glob("*/*.wav"))  # Get all audio file paths
    if len(all_paths) == 0:
        raise ValueError(f"No .wav files found in {data_path}. Please check the data path and file extensions.")

    # Split the dataset using our new function
    train_indices, val_indices, test_indices, inference_indices = split_dataset(
        all_paths, val_size, test_size, inference_size, random_state=seed
    )

    # Create new datasets based on the split indices
    train_paths = [all_paths[i] for i in train_indices]
    val_paths = [all_paths[i] for i in val_indices]
    test_paths = [all_paths[i] for i in test_indices]
    inference_paths = [all_paths[i] for i in inference_indices]

    # Create AudioDataset instances
    train_dataset = AudioDataset(data_path,
                                 train_paths,
                                 feature_extractor,
                                 augmentations_per_sample=augmentations_per_sample,
                                 augmentations=augmentations,
                                 config=config)
    
    val_dataset = AudioDataset(data_path, 
                               val_paths, 
                               feature_extractor,
                               augmentations_per_sample=augmentations_per_sample,
                               augmentations=augmentations,
                               config=config)
    
    test_dataset = AudioDataset(data_path, test_paths, feature_extractor, config=config)
    inference_dataset = AudioDataset(data_path, inference_paths, feature_extractor, config=config)
    
    print(f"Lengths: Train: {len(train_dataset)}, Validation: {len(val_dataset)}, "
          f"Test: {len(test_dataset)}, Inference: {len(inference_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, inference_dataset

def save_model(model: torch.nn.Module,
            target_dir: str,
            model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)
    
def load_model(model_path:str, model): #TODO type hinting for HGFace transformer
    model.load_state_dict(torch.load(model_path))
    return model

# if __name__ == "__main__":
#     main()




