import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.amp.grad_scaler import GradScaler
from torch.optim.adam import Adam
from timeit import default_timer as timer
from tqdm.auto import tqdm
import wandb
from typing import Dict, List, Optional

from .engine import train_step, test_step

class TorchCNN(nn.Module):
    def __init__(self, num_classes: int = 9, hidden_units: int = 256):
        """
        Initialize the CNN model with configurable hidden units for the fully connected layers.
        
        Args:
            num_classes (int): Number of output classes
            hidden_units (int): Number of hidden units in the fully connected layer
        """
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
        
        # Dense layers with configurable hidden units
        self.fc1 = nn.Linear(self._to_linear, hidden_units)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def _get_conv_output_size(self, shape):
        """Helper function to calculate conv output size"""
        bs = 1
        x = torch.rand(bs, *shape)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        self._to_linear = x.shape[1]
        return self._to_linear

    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
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

def train_cnn(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: Optional[DataLoader],
    optimizer: Adam,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    epochs: int,
    device: str,
    num_classes: int,
    accumulation_steps: int = 1,
    patience: int = 5,
    delta: float = 0.01,
    scaler: Optional[GradScaler] = None
) -> Dict[str, List]:
    """
    Training function specifically designed for CNN models.
    
    Args:
        model (nn.Module): The CNN model to train
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        test_dataloader (Optional[DataLoader]): Test data loader
        optimizer (Adam): The optimizer
        scheduler (lr_scheduler): Learning rate scheduler
        loss_fn (nn.Module): Loss function
        epochs (int): Number of epochs to train
        device (str): Device to train on
        num_classes (int): Number of classes
        accumulation_steps (int): Number of steps for gradient accumulation
        patience (int): Early stopping patience
        delta (float): Minimum change in monitored quantity for early stopping
        scaler (Optional[GradScaler]): Gradient scaler for mixed precision training
    
    Returns:
        Dict[str, List]: Dictionary containing training metrics
    """
    start = timer()
    
    results = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "test_loss": [], "test_acc": [], "test_f1": [],
        "train_precision": [], "val_precision": [], "test_precision": [],
        "train_recall": [], "val_recall": [], "test_recall": []
    }
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Initialize metrics
    precision_metric_train = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_train = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_train = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)
    
    precision_metric_val = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_val = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_val = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)
    
    precision_metric_test = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_test = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_test = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)
    
    for epoch in tqdm(range(epochs), dynamic_ncols=True):
        # Training step
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            precision_metric=precision_metric_train,
            recall_metric=recall_metric_train,
            f1_metric=f1_metric_train,
            scaler=scaler
        )
        
        # Validation step
        val_loss, val_acc, val_precision, val_recall, val_f1 = test_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
            precision_metric=precision_metric_val,
            recall_metric=recall_metric_val,
            f1_metric=f1_metric_val
        )
        
        # Test step (if test_dataloader is provided)
        if test_dataloader is not None:
            test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
                precision_metric=precision_metric_test,
                recall_metric=recall_metric_test,
                f1_metric=f1_metric_test
            )
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            results["test_f1"].append(test_f1)
            results["test_precision"].append(test_precision)
            results["test_recall"].append(test_recall)
        
        # Print metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )
        
        if wandb.run is not None:
            wandb.log({
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_precision": train_precision,
                "val_precision": val_precision,
                "train_recall": train_recall,
                "val_recall": val_recall,
                "epoch": epoch+1
            })
            
            if test_dataloader is not None:
                wandb.log({
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "test_f1": test_f1,
                    "test_precision": test_precision,
                    "test_recall": test_recall
                })
        
        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["train_f1"].append(train_f1)
        results["val_f1"].append(val_f1)
        results["train_precision"].append(train_precision)
        results["val_precision"].append(val_precision)
        results["train_recall"].append(train_recall)
        results["val_recall"].append(val_recall)
        
        # Early stopping logic
        if val_loss < best_loss - delta:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.")
            break
            
        # Step the scheduler
        scheduler.step(val_loss)
        
    end = timer()
    total_train_time = end - start
    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)
    seconds = int(total_train_time % 60)
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"Train time on {device}: {formatted_time}")
    
    if wandb.run is not None:
        wandb.log({"train_time": formatted_time})
        
    return results 