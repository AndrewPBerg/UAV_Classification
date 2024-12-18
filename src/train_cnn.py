import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchaudio
import numpy as np
from torch.optim import Adam
from tqdm.auto import tqdm
import random
from typing import Tuple, List, Dict, Union
from torchaudio.transforms import MelSpectrogram, Resample
import torch.nn.functional as F
from torchinfo import summary
from helper.util import k_fold_split_custom, wandb_login
from sklearn.model_selection import KFold
from timeit import default_timer as timer
from icecream import ic
from helper.fold_engine import train_fold, k_fold_cross_validation
import wandb
from typing import Dict, List, Optional
# from helper.engine import train_step, test_step


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

def CNN_k_fold_split_custom(
    data_path: str,
    k_folds: int = 5,
    inference_size: float = 0.1,
    seed: int = 42,
    target_sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512
) -> list:
    """
    Creates k-fold splits of the dataset for cross validation.
    Returns a tuple containing:
    - List of (train_dataset, val_dataset) tuples for each fold
    - Inference dataset
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
            fold_train_paths,
            target_sr=target_sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        val_dataset = AudioDataset(
            fold_val_paths,
            target_sr=target_sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        fold_datasets.append((train_dataset, val_dataset))
        ic(f"Fold {fold+1} datasets loaded")
    
    # Create inference dataset
    inference_paths = [all_paths[i] for i in inference_indices]
    inference_dataset = AudioDataset(
        inference_paths,
        target_sr=target_sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    end_time = timer()
    dataset_init_time = end_time - start_time
    print(f"Dataset initialization took {dataset_init_time:.2f} seconds")
    
    return fold_datasets



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

def train_fold(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: str,
    fold: int,
    epochs: int = 50,
    patience: int = 5,
) -> Dict[str, List]:
    """Trains a single fold of k-fold cross validation"""
    
    results = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc=f"Fold {fold+1} Training"):
        # Training
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Add channel dimension if needed
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_acc += predicted.eq(targets).sum().item() / targets.size(0)
            
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)
                    
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_acc += predicted.eq(targets).sum().item() / targets.size(0)
                
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

        # Print progress
        print(
            f"Fold {fold+1} Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Log metrics to wandb
        if wandb.run is not None:
            wandb.log({
                f"fold_{fold+1}/train_acc": train_acc,
                f"fold_{fold+1}/train_loss": train_loss,
                f"fold_{fold+1}/val_acc": val_acc,
                f"fold_{fold+1}/val_loss": val_loss,
                "epoch": epoch+1
            })

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered in fold {fold+1} at epoch {epoch+1}")
            break

    return results

def cnn_k_fold_cross_validation(
    model_fn,  # Function that creates a new model instance
    fold_datasets: list,
    optimizer_fn,  # Function that creates a new optimizer instance
    criterion: nn.Module,
    device: str,
    epochs: int,
    patience: int = 5,
) -> List[Dict]:
    """Performs k-fold cross validation training"""
    
    all_fold_results = []
    
    for fold, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f"\nTraining Fold {fold + 1}/{len(fold_datasets)}")
        
        # Create dataloaders for this fold
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            num_workers=2,
            pin_memory=True,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=32,
            num_workers=2,
            pin_memory=True,
            shuffle=False
        )
        
        # Create new model instance for this fold
        model = model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())
        
        # Train the fold
        fold_results = train_fold(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            fold=fold,
            epochs=epochs,
            patience=patience
        )
        
        all_fold_results.append(fold_results)
    # Log final metrics for this fold
        if wandb.run is not None:
            wandb.log({
                f"fold_{fold+1}_final_val_acc": fold_results["val_acc"][-1],
                f"fold_{fold+1}_final_val_loss": fold_results["val_loss"][-1],
                # f"fold_{fold+1}_final_val_f1": fold_results["val_f1"][-1]
            })
            
    # Calculate and log average metrics across folds
    avg_metrics = calculate_average_metrics(all_fold_results)
    if wandb.run is not None:
        wandb.log(avg_metrics)
            
    if wandb.run is not None:
        wandb_table = wandb.Table(columns=["Metric", "Average", "Standard Deviation"])
        for metric in ["val_acc", "val_f1", "val_loss", "val_precision", "val_recall"]:
            avg_value = avg_metrics[f"average_{metric}"]
            std_value = avg_metrics[f"std_{metric}"]
            wandb_table.add_data(metric, avg_value, std_value)
        wandb.log({"average_metrics_table": wandb_table})
    
    
    return all_fold_results

def calculate_average_metrics(all_fold_results: List[Dict]) -> Dict:
    """Calculate average metrics across all folds"""
    avg_metrics = {}
    metrics = ["val_acc", "val_loss", "val_f1", "val_precision", "val_recall"]
    
    for metric in metrics:
        values = [fold[metric][-1] for fold in all_fold_results]  # Get final value for each fold
        avg_metrics[f"average_{metric}"] = np.mean(values)
        avg_metrics[f"std_{metric}"] = np.std(values)
    
    return avg_metrics 
        

def main():
    # Configuration
    USE_KFOLD = True
    USE_WANDB = True
    # DATA_PATH ="/app/src/datasets/UAV_Dataset_31"
    DATA_PATH ="/app/src/datasets/UAV_Dataset_31"
    INFERENCE_SIZE = 0.2
    SEED = 42
    EPOCHS = 20
    PATIENCE = 5
    NUM_CLASSES = 31
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if USE_WANDB:
        wandb_login()

        wandb.init(project="CNN kfold test",
                    name="") # run name
    
    if USE_KFOLD:
        # K-fold cross validation
        fold_datasets = CNN_k_fold_split_custom(
            DATA_PATH,
            k_folds=5,
            inference_size=INFERENCE_SIZE,
            seed=SEED
        )
        
        # Define model and optimizer creation functions
        def model_fn():
            return TorchCNN(num_classes=NUM_CLASSES).to(device)
        
        def optimizer_fn(parameters):
            return Adam(parameters, lr=0.001)
        
        def scheduler_fn(optimizer):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=4
            )
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Perform k-fold cross validation
        fold_results = k_fold_cross_validation(
            model_fn=model_fn,
            fold_datasets=fold_datasets,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            loss_fn=loss_fn,
            num_classes=NUM_CLASSES,
            epochs= 20,
            batch_size= 16,
            num_workers= 4,
            pin_memory= True,
            shuffle= True,
            accumulation_steps= 2,
            device=device,
            patience=PATIENCE)
        
        
        # Calculate and log average metrics
        # if USE_WANDB:
        #     final_metrics = {}
        #     for metric in ["val_acc", "val_loss"]:
        #         values = [fold[metric][-1] for fold in fold_results]
        #         final_metrics[f"average_{metric}"] = sum(values) / len(values)
        #     wandb.log(final_metrics)
            
    else:
        # Original train-test split training
        train_paths, test_paths = create_data_splits(DATA_PATH)
        
        train_dataset = AudioDataset(train_paths)
        test_dataset = AudioDataset(test_paths)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        model = TorchCNN(num_classes=len(train_dataset.get_classes())).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=EPOCHS,
            patience=PATIENCE
        )
        
        if USE_WANDB:
            wandb.log({
                "final_test_acc": history["test_acc"][-1],
                "final_test_loss": history["test_loss"][-1]
            })

    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()