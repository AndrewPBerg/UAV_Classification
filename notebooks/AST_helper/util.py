import os
from torch.utils.data import Dataset, Subset
from pathlib import Path
import torchaudio
import torch
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import librosa
from typing import Optional, Union, Tuple
from transformers import ASTFeatureExtractor




class AudioDataset(Dataset):
    def __init__(self, data_path: str, feature_extractor:ASTFeatureExtractor) -> None:
        self.paths = list(Path(data_path).glob("*/*.wav"))
        self.feature_extractor = feature_extractor
        self.classes, self.class_to_idx = find_classes(data_path)
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        
    def load_audio(self, index:int) -> torch.Tensor:
        "load audio from path, return raw tensor"
        audio_tensor, _= torchaudio.load(self.paths[index])
        return audio_tensor # don't return sr (sample rate) b/c all data has same sr  
    def get_classes(self) -> list[str]:
        return self.classes
        
    def get_class_dict(self) -> dict[str, int]:
        return self.class_to_idx
        
    def get_idx_dict(self) -> dict[int, str]:
        return self.idx_to_class
    
    def get_feature_extractor(self):
        return self.feature_extractor
        
    def __len__(self) -> int:
        "Override Dataset.__len__ for custom file structre"
        return len(self.paths)


    # can add conditional audio transformation if needed! (resamples, cutting, volume, etc.)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor,int]:
        "Returns one sample of data, and it's class index"

        audio_tensor= self.load_audio(index)
        class_name  = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        
        # Using the AST model's correct size for transformation instead of mel_spectrogram
        inputs = self.feature_extractor(audio_tensor.squeeze().numpy(), return_tensors="pt", sampling_rate=16000) # adjust sampling rate as needed
        inputs = inputs['input_values'][0]  # Extract input tensor
    
        return inputs, class_idx
        
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

def train_test_split_custom(dataset: AudioDataset, test_size: float = 0.2, seed: int = 42, inference_size: Optional[float]= None
                            ) -> Union[Tuple[Subset, Subset], Tuple[Subset,Subset,Subset]]:

    all_indices = list(range(len(dataset)))

    if inference_size:
        test_size += inference_size # add to test_size so test set isn't reducded (in size)
    # 1st split
    train_indices, test_indices = train_test_split(
        all_indices, 
        test_size=test_size, 
        random_state=seed
        )
    # 2nd split
    if inference_size:
        test_indices, inference_indices = train_test_split(
            test_indices,
            test_size= inference_size,
            random_state=seed
        )

        inference_subset = Subset(dataset, inference_indices)
    
    
    # Subset datasets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    
    if inference_size:
        return train_subset, test_subset, inference_subset
    else:
        return train_subset, test_subset

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
