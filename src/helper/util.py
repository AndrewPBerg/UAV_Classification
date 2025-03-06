import os
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import torch
import random
import matplotlib.pyplot as plt
import librosa
from typing import Optional, Union
from transformers import ASTFeatureExtractor, SeamlessM4TFeatureExtractor, WhisperProcessor, Wav2Vec2FeatureExtractor, BitImageProcessor, ViTImageProcessor
import warnings
from torchviz import make_dot

from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from .augmentations import create_augmentation_pipeline, apply_augmentations
from torchaudio.transforms import Resample
from typing import Union
import numpy as np
import os
from torch.cuda.amp import autocast
import wandb
from sklearn.model_selection import KFold
from icecream import ic
from time import time as timer
from dotenv import load_dotenv
from configs import AugConfig as AugmentationConfig

def generate_model_image(model: torch.nn.Module, device=None):
    # Create a random input tensor
    # Note: We removed the explicit .to(device) to let PyTorch Lightning handle device placement
    # The model will be on the correct device when called
    x = torch.randn(128, 157, requires_grad=True)
    x = x.float()
    x = x.unsqueeze(0)
    
    with autocast(enabled=True, dtype=torch.float16):
        y = model(x)
        if hasattr(y, "logits"):
            y_pred = y.logits
        else:
            y_pred = y

    dot = make_dot(y_pred.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    dot.render("images/model_graph", format="svg")  # Save the visualization as PNG
        

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return f'Total Parameters: {total_params:,}\nTrainable Parameters: {trainable_params:,}'
    return total_params, trainable_params

def get_mixed_params(sweep_config, general_config):
    # copy sweep_config to result dict
    result = sweep_config 

    for key, value in general_config.items():
        # just like LeetCode isDuplicate problem
        if key in result:
            pass
        else:
            # if not already occupied by sweep config value add the current general parameter
            result[key] = value
    
    # final dict should contain all of the config.yaml parameters
    # where sweep parameters have priority over duplicates in the general configuration
    return result

def calculated_load_time(start, end) -> str:
    total_load_time = end - start
    hours = int(total_load_time // 3600)
    minutes = int((total_load_time % 3600) // 60)
    seconds = int(total_load_time % 60)
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    return formatted_time

def wandb_login():
    load_dotenv()

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
                 feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor, ViTImageProcessor],
                 standardize_audio_boolean: bool=True, 
                 target_sr: int=16000,
                 target_duration: int=5, 
                 augmentations_per_sample: int = 0,
                 augmentations: list[str] = [],
                 num_channels: int = 1,
                 config: Optional[dict] = None) -> None:
        self.paths = data_paths
        self.feature_extractor = feature_extractor
        self.classes, self.class_to_idx = find_classes(data_path)
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        self.sampling_rate = None  # current dataset sample rate
        self.resampler = None
        self.standardize_audio_boolean = standardize_audio_boolean
        
        # Get target sampling rate from feature extractor if available
        if isinstance(feature_extractor, MelSpectrogramFeatureExtractor) or isinstance(feature_extractor, MFCCFeatureExtractor):
            self.target_sr = feature_extractor.sampling_rate
        else:
            try:
                self.target_sr = feature_extractor.sampling_rate or target_sr
            except:
                self.target_sr = target_sr
                
        self.target_duration = target_duration
        self.target_length = target_duration * self.target_sr
        self.augmentations_per_sample = augmentations_per_sample
        self.config = config or {}

        total_samples = (augmentations_per_sample + 1) * len(self.paths)
        self.audio_tensors = torch.empty(total_samples, num_channels, self.target_sr * target_duration)
        self.class_indices = []

        if self.augmentations_per_sample > 0 and augmentations:
            self.composed_transform = create_augmentation_pipeline(augmentations, self.config)
        else:
            self.composed_transform = None

        # Load original audio samples
        for index, path in enumerate(self.paths):
            audio_to_append, class_idx = self.load_audio(path)
            self.audio_tensors[index] = audio_to_append
            self.class_indices.append(class_idx)

        # Apply augmentations
        if self.augmentations_per_sample > 0:
            original_samples = len(self.paths)
            for i in range(original_samples):
                for j in range(self.augmentations_per_sample):
                    new_index = original_samples + i * self.augmentations_per_sample + j
                    self.class_indices.append(self.class_indices[i])
                    if len(augmentations) != 0:
                        augmented_audio = apply_augmentations(self.audio_tensors[i], self.composed_transform, self.target_sr)
                        self.audio_tensors[new_index] = augmented_audio
                    else:
                        self.audio_tensors[new_index] = self.audio_tensors[i]

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
        import logging
        import sys
        
        # Set up detailed logging
        logging.basicConfig(
            level=logging.CRITICAL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        logger = logging.getLogger("AUDIO_DATASET")
        
        audio_np = audio_tensor.squeeze().numpy()
        logger.debug(f"Audio tensor shape before feature extraction: {audio_tensor.shape}")
        logger.debug(f"Audio numpy shape after squeeze: {audio_np.shape}")

        try:
            # For MelSpectrogram feature extractor
            if isinstance(self.feature_extractor, MelSpectrogramFeatureExtractor):
                logger.debug("Using MelSpectrogramFeatureExtractor")
                features = self.feature_extractor(
                    audio_tensor,  # Pass tensor directly
                    sampling_rate=self.target_sr,
                    return_tensors="pt"
                )
                logger.debug(f"MelSpectrogram features shape: {features.input_values.shape}")
                return features.input_values
            # For MFCC feature extractor
            elif isinstance(self.feature_extractor, MFCCFeatureExtractor):
                logger.debug("Using MFCCFeatureExtractor")
                features = self.feature_extractor(
                    audio_tensor,  # Pass tensor directly
                    sampling_rate=self.target_sr,
                    return_tensors="pt"
                )
                logger.debug(f"MFCC features shape: {features.input_values.shape}")
                return features.input_values
            # For AST feature extractor
            elif isinstance(self.feature_extractor, ASTFeatureExtractor):
                logger.debug("Using ASTFeatureExtractor")
                
                # AST expects specific input dimensions
                # Make sure we're using the right parameters for feature extraction
                try:
                    features = self.feature_extractor(
                        audio_np,
                        sampling_rate=self.target_sr,
                        return_tensors="pt",
                        padding="max_length",  # Use max_length padding to ensure consistent dimensions
                        max_length=1024,       # Set a fixed max length that works with the model
                        truncation=True        # Truncate if needed
                    )
                    logger.debug(f"AST raw features shape: {features.input_values.shape}")
                except Exception as e:
                    logger.error(f"Error in AST feature extraction with padding: {e}")
                    # Fallback to basic extraction
                    features = self.feature_extractor(
                        audio_np,
                        sampling_rate=self.target_sr,
                        return_tensors="pt"
                    )
                    logger.debug(f"AST fallback features shape: {features.input_values.shape}")
                
                # Get the features and ensure it has the right shape
                feature_values = features.input_values.squeeze(0)
                logger.debug(f"AST features after squeeze(0): {feature_values.shape}")
                
                # Check for problematic shapes and fix
                if len(feature_values.shape) == 4:
                    logger.debug(f"Detected 4D feature tensor: {feature_values.shape}")
                    
                    # If we have [channels, height, 1, width], reshape to [channels, height, width]
                    if feature_values.shape[2] == 1:
                        feature_values = feature_values.squeeze(2)
                        logger.debug(f"Squeezed dimension 2, new shape: {feature_values.shape}")
                    # If we have a different 4D shape, try to reshape it to 3D
                    else:
                        c, h, d, w = feature_values.shape
                        try:
                            feature_values = feature_values.reshape(c, h*d, w)
                            logger.debug(f"Reshaped to 3D tensor: {feature_values.shape}")
                        except Exception as e:
                            logger.error(f"Error reshaping 4D tensor: {e}")
                
                logger.debug(f"Final AST feature shape: {feature_values.shape}")
                return feature_values
            elif isinstance(self.feature_extractor, SeamlessM4TFeatureExtractor):
                features = self.feature_extractor(
                    audio_np,
                    sampling_rate=self.target_sr,
                    return_tensors="pt",
                    padding=True
                )
                return features.input_features.squeeze(0)
            elif isinstance(self.feature_extractor, Wav2Vec2FeatureExtractor):
                features = self.feature_extractor(
                    audio_np,
                    sampling_rate=self.target_sr,
                    return_tensors="pt"
                )
                return features.input_values.squeeze(0)
            elif isinstance(self.feature_extractor, BitImageProcessor):
                features = self.feature_extractor(
                    audio_np,
                    sampling_rate=self.target_sr,
                    return_tensors="pt"
                )
                return features.input_values.squeeze(0)
            elif isinstance(self.feature_extractor, ViTImageProcessor):
                # Convert audio to spectrogram for ViT
                # We've modified the ViT model to accept grayscale input (1 channel)
                logger.debug("Using ViTImageProcessor with grayscale input")
                
                # Convert audio to mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_np, 
                    sr=self.target_sr
                )
                
                # Convert to decibels
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Normalize to [0, 1] range
                mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
                
                # Create a single-channel image as a PyTorch tensor
                # Shape: [1, H, W] - for grayscale input to our modified ViT
                h, w = mel_spec_normalized.shape
                tensor_1ch = torch.from_numpy(mel_spec_normalized).unsqueeze(0).float()  # Add channel dimension
                
                # Resize if needed to match ViT's expected input size (typically 224x224)
                if h != 224 or w != 224:
                    tensor_1ch = torch.nn.functional.interpolate(
                        tensor_1ch.unsqueeze(0),  # Add batch dimension
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dimension
                
                logger.debug(f"Final tensor shape for ViT (grayscale): {tensor_1ch.shape}")
                
                # Verify we have 1 channel
                assert tensor_1ch.shape[0] == 1, f"Expected 1 channel, got {tensor_1ch.shape[0]}"
                
                return tensor_1ch
            elif isinstance(self.feature_extractor, WhisperProcessor):
                # Whisper expects 30-second inputs
                target_length = 30 * self.target_sr
                
                # Pad the 5-second audio to 30 seconds
                if len(audio_np) < target_length:
                    padding_length = target_length - len(audio_np)
                    audio_np = np.pad(audio_np, (0, padding_length), mode='constant')
                
                features = self.feature_extractor(
                    audio_np, 
                    sampling_rate=self.target_sr,
                    return_tensors="pt",
                    padding=True
                )
                return features.input_features.squeeze(0)
            else:
                raise ValueError(f"Unsupported feature extractor type: {type(self.feature_extractor)}")

        except Exception as e:
            raise e
 
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        try:
            features = self.feature_extraction(self.audio_tensors[index])
            return features, self.class_indices[index]
        except Exception as e:
            print(f"Error in __getitem__ for index {index}: {str(e)}")
            print(f"Audio tensor shape: {self.audio_tensors[index].shape}")
            raise e

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
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    test_size: float = 0.2, 
    val_size: float = 0.1,
    inference_size: float = 0.1,
    seed: int = 42, 
    augmentations_per_sample: int = 3,
    augmentations: Optional[list[str]] = None,
    config: Optional[AugmentationConfig] = None
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

    # Create AudioDataset instances with proper config handling
    augmentations = augmentations or []
    config_dict = config.aug_configs if isinstance(config, AugmentationConfig) else {}
    
    train_dataset = AudioDataset(data_path,
                                train_paths,
                                feature_extractor,
                                augmentations_per_sample=augmentations_per_sample,
                                augmentations=augmentations,
                                config=config_dict)
    
    val_dataset = AudioDataset(data_path, 
                               val_paths, 
                               feature_extractor,
                               augmentations_per_sample=augmentations_per_sample,
                               augmentations=augmentations,
                               config=config_dict)
    
    test_dataset = AudioDataset(data_path, test_paths, feature_extractor, config=config_dict)
    inference_dataset = AudioDataset(data_path, inference_paths, feature_extractor, config=config_dict)
    
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

def k_fold_split_custom(
    data_path: str,
    feature_extractor: Union[ASTFeatureExtractor, SeamlessM4TFeatureExtractor, MelSpectrogramFeatureExtractor, MFCCFeatureExtractor],
    k_folds: int = 5,
    inference_size: float = 0.1,
    seed: int = 42,
    augmentations_per_sample: int = 3,
    augmentations: Optional[list[str]] = None,
    config: Optional[AugmentationConfig] = None
) -> list[tuple]:
    """
    Creates k-fold splits of the dataset for cross validation.
    Returns a list of tuples containing (train_dataset, val_dataset) for each fold.
    """
    # Get all paths
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if len(all_paths) == 0:
        raise ValueError(f"No .wav files found in {data_path}")

    # First separate inference set
    n_samples = len(all_paths)
    n_inference = int(n_samples * inference_size)
    
    # Create random state for reproducibility
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    
    # Split off inference set
    train_val_indices = indices[:-n_inference]
    inference_indices = indices[-n_inference:]
    
    # Create KFold object
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    # Create datasets for each fold
    fold_datasets = []
    start_time = timer()  # Start timer for dataset initialization
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices)):
        # Get paths for this fold
        fold_train_paths = [all_paths[i] for i in train_val_indices[train_idx]]
        fold_val_paths = [all_paths[i] for i in train_val_indices[val_idx]]
        
        # Create datasets
        train_dataset = AudioDataset(
            data_path,
            fold_train_paths,
            feature_extractor,
            augmentations_per_sample=augmentations_per_sample,
            augmentations=augmentations,
            config=config.aug_configs if isinstance(config, AugmentationConfig) else {}
        )
        
        val_dataset = AudioDataset(
            data_path,
            fold_val_paths,
            feature_extractor,
            config=config.aug_configs if isinstance(config, AugmentationConfig) else {}
        )
        
        fold_datasets.append((train_dataset, val_dataset))
        ic(f"Fold {fold+1} datasets loaded")  # Show progression of folds datasets loading
    
    # Create inference dataset
    inference_paths = [all_paths[i] for i in inference_indices]
    inference_dataset = AudioDataset(data_path, inference_paths, feature_extractor, config=config.aug_configs if isinstance(config, AugmentationConfig) else {})
    
    end_time = timer()  # End timer for dataset initialization
    dataset_init_time = end_time - start_time
    print(f"Dataset initialization took {dataset_init_time:.2f} seconds")
    
    # Handle config properly
    augmentations = augmentations or []
    config_dict = config.aug_configs if isinstance(config, AugmentationConfig) else {}
    
    return fold_datasets, inference_dataset



