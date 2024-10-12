import glob
import os
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import torch
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import librosa
from typing import Optional
from transformers import ASTFeatureExtractor
import warnings
from helper.augmentations import apply_augmentations, apply_random_augmentation  # Assume this function exists

from dotenv import dotenv_values
import wandb

def wandb_login(secrets_path:str):

    config = dotenv_values(secrets_path)
    wandb.login(key=config['WANDB_API_KEY'])
    
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
                 feature_extractor: ASTFeatureExtractor, 
                 standardize_audio_boolean: bool=True, 
                 target_sr: int=44100, 
                 target_duration: int=5, 
                 augmentations_per_sample: int = 0,
                 augmentation_probability: float = 0.5,
                 num_channels: int = 1) -> None:
        self.paths = data_paths
        self.feature_extractor = feature_extractor
        self.classes, self.class_to_idx = find_classes(data_path)
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        self.sampling_rate = None  # current dataset sample rate
        self.standardize_audio_boolean = standardize_audio_boolean
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.augmentations_per_sample = augmentations_per_sample
        self.augmentation_probability = augmentation_probability
        self.standardize_audio_boolean = standardize_audio_boolean

        total_samples = (augmentations_per_sample+1) * (len(self.paths)) # Total amount of samples in dataset * the number of augmentations apppled to data
        self.audio_tensors = torch.empty(total_samples, num_channels, target_sr * target_duration)  # expected shape for UAV == ([1,1,220500)
        self.class_indices = []
        # Load Original audio samples into the 0th dim of 0th index of audio_tensors
        for index, path in enumerate(self.paths): 
            audio_to_append, class_idx = self.load_audio(path)  
            self.audio_tensors[index] = audio_to_append
            self.class_indices.append(class_idx)

        original_audio_tensors = self.audio_tensors
        # Now produce if augmentations_per_sample > 0 produce and append augmentations
        if self.augmentations_per_sample > 0:
            for i in range(len(original_audio_tensors),len(self.audio_tensors)): # Loop through each possible augmentation index, starting at 1
                self.class_indices.append(self.class_indices[i])
                if random.random() < self.augmentation_probability: # Chance to not augment, and repeat original
                    augmented_audio = apply_random_augmentation(original_audio_tensors[i], self.target_sr)
                    self.audio_tensors[i] = augmented_audio
                else:
                    self.audio_tensors[i] = original_audio_tensors[i]
    def load_audio(self, path:str):
        "load audio from path, return raw tensor"
    

        audio_tensor, sr = torchaudio.load(str(path))
        
        # Logic to only change the sr once per dataset instance
        if self.sampling_rate is None: 
            self.sampling_rate = sr

        # Standardization option here
        class_name = Path(path).parent.name # Get the class name from the file's name
        class_idx = self.class_to_idx[class_name] # Convert class name to index
        return audio_tensor, class_idx

    
    def __len__(self) -> int:
        if self.augmentations_per_sample > 0:
            return len(self.paths) * (self.augmentations_per_sample + 1)
        else:
            return len(self.paths)
        
    def feature_extraction(self, audio_tensor):
        features = self.feature_extractor(audio_tensor.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
        return features

    def __getitem__(self, index):
        features = self.feature_extractor(self.audio_tensors[index].squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
        # features size == torch.Size([1, 1024, 128]

        if self.audio_tensors[index].shape[0] == 1: # if the number of channels in the first dim of self.audio_tensor == 1: squeeze the dim out
            features = features.squeeze(dim=0)
            # features size == torch.Size([1024, 128]
        

        return features, self.class_indices[index]

    def get_classes(self) -> list[str]:
        return self.classes
    
    def get_class_dict(self) -> dict[str, int]:
        return self.class_to_idx
        
    def get_idx_dict(self) -> dict[int, str]:
        return self.idx_to_class
    
    def get_feature_extractor(self):
        return self.feature_extractor
    
    def standardize_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Standardize audio to the specified target duration and sampling rate"""
        if self.sampling_rate is None:
            # If sampling rate is not set, use the target sampling rate and issue a warning
            warnings.warn(f"Sampling rate not set. Using target value of {self.target_sr} Hz.", UserWarning)
            current_sr = self.target_sr
        else:
            current_sr = self.sampling_rate
        
        target_length = self.target_sr * self.target_duration
        
        # Resample if necessary
        if current_sr != self.target_sr:
            audio_tensor = torchaudio.transforms.Resample(current_sr, self.target_sr)(audio_tensor)
        
        # Pad or trim to target length
        if audio_tensor.size(1) < target_length:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, target_length - audio_tensor.size(1)))
        else:
            audio_tensor = audio_tensor[:, :target_length]
        
        return audio_tensor
    
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
    feature_extractor: ASTFeatureExtractor,
    test_size: float = 0.2, 
    val_size: float = 0.1,
    inference_size: float = 0.1,
    seed: int = 42, 
    augmentations_per_sample: int = 3,
    augmentation_probability: float = 0.5
):                          
    all_paths = list(Path(data_path).glob("*/*.wav"))  # Get all audio file paths
    # print(f"all paths {all_paths}")
    all_indices = list(range(len(all_paths)))
    
    if len(all_paths) == 0:
        raise ValueError(f"No .wav files found in {data_path}. Please check the data path and file extensions.")


    # First split: separate train+val from test+inference
    train_val_indices, test_inference_indices = train_test_split(
        all_indices,
        test_size=test_size + inference_size,
        random_state=seed
    )

    # Second split: separate train from val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size / (1 - test_size - inference_size),
        random_state=seed
    )

    # Third split: separate test from inference
    test_indices, inference_indices = train_test_split(
        test_inference_indices,
        test_size=inference_size / (test_size + inference_size),
        random_state=seed
    )

    # Create new datasets based on the split indices
    train_paths = [all_paths[i] for i in train_indices]
    val_paths = [all_paths[i] for i in val_indices]
    test_paths = [all_paths[i] for i in test_indices]
    inference_paths = [all_paths[i] for i in inference_indices]

    print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}, Inference: {len(inference_paths)}")
    print(f"Total samples: {len(train_paths) + len(val_paths) + len(test_paths) + len(inference_paths)}")

    # train, test,val, inferences indices contain unique keys to the all_paths list, used in AudioDataset2 for Tensor loading
    train_dataset = AudioDataset(data_path,
                                  train_paths,
                                  feature_extractor,
                                  augmentations_per_sample=augmentations_per_sample,
                                  augmentation_probability=augmentation_probability)
    
    val_dataset = AudioDataset(data_path, 
                                val_paths, 
                                feature_extractor,
                                augmentations_per_sample=3,
                                augmentation_probability=augmentation_probability)
    
    test_dataset = AudioDataset(data_path, test_paths, feature_extractor)
    inference_dataset = AudioDataset(data_path, inference_paths, feature_extractor)
    
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
