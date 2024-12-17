import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchaudio
import numpy as np
from torch.optim import Adam
from tqdm.auto import tqdm
import random
from typing import Tuple, List, Dict
from torchaudio.transforms import MelSpectrogram, Resample
import torch.nn.functional as F
from torchinfo import summary

class AudioDataset(Dataset):
    def __init__(
        self,
        data_paths: List[Path],
        target_sr: int = 16000,
        target_duration: int = 5,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512
    ):
        self.paths = data_paths
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = target_duration * target_sr
        
        # Initialize mel spectrogram transform
        self.mel_transform = MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Get class names from parent folders
        self.classes = sorted(list(set(path.parent.name for path in data_paths)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        path = self.paths[index]
        
        # Load and preprocess audio
        audio_tensor, sr = torchaudio.load(str(path))
        
        # Convert to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.target_sr:
            resampler = Resample(sr, self.target_sr)
            audio_tensor = resampler(audio_tensor)
            
        # Pad or trim to target length
        if audio_tensor.shape[1] < self.target_length:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, self.target_length - audio_tensor.shape[1])
            )
        else:
            audio_tensor = audio_tensor[:, :self.target_length]
            
        # Get class label
        class_name = path.parent.name
        class_idx = self.class_to_idx[class_name]
        
        # Extract mel spectrogram features
        mel_spec = self.mel_transform(audio_tensor)  # Shape: [1, n_mels, time]
        
        # Convert to decibel scale
        mel_spec = torch.log10(mel_spec + 1e-9)
        
        # Reshape to [n_mels, time]
        mel_spec = mel_spec.squeeze(0)
        
        return mel_spec, class_idx
    
    def get_classes(self) -> List[str]:
        return self.classes

def create_data_splits(
    data_path: str,
    train_size: float = 0.75,
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """Create train and test splits from a directory of audio files."""
    
    # Get all .wav files
    all_paths = list(Path(data_path).glob("*/*.wav"))
    if len(all_paths) == 0:
        raise ValueError(f"No .wav files found in {data_path}")
        
    # Shuffle paths
    random.seed(seed)
    random.shuffle(all_paths)
    
    # Calculate split index
    train_end = int(len(all_paths) * train_size)
    
    # Split data
    train_paths = all_paths[:train_end]
    test_paths = all_paths[train_end:]
    
    print(f"Train samples: {len(train_paths)}, Test samples: {len(test_paths)}")
    return train_paths, test_paths

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return total_loss / len(train_loader), correct / total

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Evaluate model on given data loader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / len(data_loader), correct / total

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: str,
    epochs: int = 50,
    patience: int = 5
) -> Dict[str, List[float]]:
    """Train model with early stopping."""
    
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs)):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Test
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    return history


class TorchCNN(nn.Module):
    def __init__(self, num_classes: int = 9):
        super(TorchCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._get_conv_output_size((128, 157))  # Initialize _to_linear
        
        # Dense layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output_size(self, shape):
        # Helper function to calculate conv output size
        bs = 1
        x = torch.rand(bs, *shape)  # [batch_size, n_mels, time]
        x = x.unsqueeze(1)  # Add channel dimension [batch_size, channels, n_mels, time]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        self._to_linear = x.shape[1]
        return self._to_linear

    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 3:  # [batch_size, n_mels, time]
            x = x.unsqueeze(1)  # [batch_size, channels, n_mels, time]
            
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.flatten(1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data splits
    train_paths, test_paths = create_data_splits(
        "C:/Users/Sidewinders/Desktop/CODE/UAV_Classification_repo/.datasets/UAV_Dataset_9"
    )
    
    # Create datasets with mel spectrogram parameters
    train_dataset = AudioDataset(
        train_paths,
        target_sr=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    test_dataset = AudioDataset(
        test_paths,
        target_sr=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = TorchCNN(num_classes=len(train_dataset.get_classes())).to(device)
    
    summary(model, input_size=(32, 1, 128, 157))  # Batch size, channels, n_mels, time
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device="cuda",
        epochs=50,
        patience=5
    )
    


if __name__ == "__main__":
    main()