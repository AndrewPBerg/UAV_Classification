import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union, Tuple
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam
from icecream import ic

from configs.configs_demo import GeneralConfig
import re


class AudioClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for audio classification.
    This replaces the functionality in engine.py and fold_engine.py.
    """
    def __init__(
        self,
        model: nn.Module,
        general_config: GeneralConfig,
        peft_config: Optional[Any] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Initialize the AudioClassifier module.
        
        Args:
            model: The PyTorch model to train
            general_config: General configuration
            peft_config: PEFT configuration (optional)
            num_classes: Number of classes (if not provided, will use general_config.num_classes)
        """
        super().__init__()
        self.model = model
        self.general_config = general_config
        self.peft_config = peft_config
        
        # Set number of classes
        self.num_classes = num_classes or general_config.num_classes
        
        # Set up loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model'])
        
        # Initialize metrics
        self._init_metrics()
        
        # Enable automatic optimization
        self.automatic_optimization = True
        
    def _init_metrics(self):
        """Initialize metrics for training, validation, and testing."""
        # Training metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")
        self.train_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted")
        self.train_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted")
        self.train_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        
        # Validation metrics
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")
        self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted")
        self.val_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted")
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        
        # Test metrics
        self.test_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")
        self.test_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted")
        self.test_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted")
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        
    def forward(self, x):
        """Forward pass through the model."""
        # Ensure input is float32
        x = x.float()
        
        # Forward pass
        outputs = self.model(x)
        
        # Handle different model output formats
        if hasattr(outputs, "logits"):
            return outputs.logits
        else:
            return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        
        # Validate target labels to ensure they are within range
        if torch.any(y < 0) or torch.any(y >= self.num_classes):
            invalid_labels = y[(y < 0) | (y >= self.num_classes)]
            raise ValueError(f"Invalid target labels found: {invalid_labels.tolist()}. Labels must be in range [0, {self.num_classes-1}]")
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Update metrics
        self.train_accuracy(y_pred_class, y)
        self.train_precision(y_pred_class, y)
        self.train_recall(y_pred_class, y)
        self.train_f1(y_pred_class, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        
        # Validate target labels to ensure they are within range
        if torch.any(y < 0) or torch.any(y >= self.num_classes):
            invalid_labels = y[(y < 0) | (y >= self.num_classes)]
            raise ValueError(f"Invalid target labels found: {invalid_labels.tolist()}. Labels must be in range [0, {self.num_classes-1}]")
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Update metrics
        self.val_accuracy(y_pred_class, y)
        self.val_precision(y_pred_class, y)
        self.val_recall(y_pred_class, y)
        self.val_f1(y_pred_class, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        
        # Validate target labels to ensure they are within range
        if torch.any(y < 0) or torch.any(y >= self.num_classes):
            invalid_labels = y[(y < 0) | (y >= self.num_classes)]
            raise ValueError(f"Invalid target labels found: {invalid_labels.tolist()}. Labels must be in range [0, {self.num_classes-1}]")
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Update metrics
        self.test_accuracy(y_pred_class, y)
        self.test_precision(y_pred_class, y)
        self.test_recall(y_pred_class, y)
        self.test_f1(y_pred_class, y)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        x, y = batch
        
        # Forward pass
        y_pred = self(x)
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        return y_pred_class, y
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Choose optimizer based on model type
        if re.match(r'^(ast|vit|mert)', self.general_config.model_type.lower()):
            optimizer = AdamW(self.model.parameters(), lr=self.general_config.learning_rate)
        else:
            optimizer = Adam(self.model.parameters(), lr=self.general_config.learning_rate)
        
        # Configure scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        } 