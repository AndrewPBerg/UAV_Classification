import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union, Tuple
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam
from icecream import ic

from configs import GeneralConfig
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
        
        # Enable automatic optimization (keeping this as true for compatibility with gradient clipping)
        self.automatic_optimization = True

        # Initialize prediction metrics dictionary to store results
        self.predict_metrics = {}
        
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
        
        # Prediction metrics
        self.predict_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")
        self.predict_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted")
        self.predict_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted")
        self.predict_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        
        # Initialize prediction metrics storage
        self.predict_batch_preds = []
        self.predict_batch_targets = []
        
    def forward(self, x):
        """Forward pass through the model."""
        
        
        # Ensure input is float32
        x = x.float()
        
        # Debug input shape
        
        
        # Check for problematic 5D input shape for AST model specifically
        # Properly detect if this is an AST model
        is_ast_model = False

        # Check if this is an AST model by looking for specific attributes
        if hasattr(self.model, 'audio_spectrogram_transformer'):
            is_ast_model = True
            
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type') and self.model.config.model_type == 'audio-spectrogram-transformer':
            is_ast_model = True

            
        
        # Handle 5D input for AST models
        if is_ast_model and len(x.shape) == 5:
          

            # If we have a 5D tensor [batch, channels, height, extra_dim, width]
            batch_size, channels, height, extra_dim, width = x.shape
            
            # Try different approaches to reshape
            if extra_dim == 1:
                # If the extra dimension is 1, we can just squeeze it out
                x = x.squeeze(3)
                
            else:
                # Otherwise reshape to combine dimensions
                x = x.reshape(batch_size, channels, height * extra_dim, width)
                
        
        
        # Forward pass
        

        outputs = self.model(x)

            
            # If we're using an AST model and get an error, try one more approach
        if is_ast_model:                    
            # Try to reshape the input to match expected dimensions
            if len(x.shape) == 4:  # [batch, channels, height, width]
                # AST expects specific input dimensions, try to adapt
                batch_size, channels, height, width = x.shape
                # Reshape to match expected input shape for AST
                # The exact reshape depends on the model's expectations
                x = x.view(batch_size, channels, height, width)
                
                outputs = self.model(x)
                    
            else:
                raise Exception("Model is not in correct AST model format")

        
        # Handle different model output formats
        if hasattr(outputs, "logits"):
            return outputs.logits
        else:
            return outputs
        
    def on_predict_start(self):
        """Called at the beginning of prediction."""
        print("Starting prediction...")
        # Reset prediction metrics storage
        self.predict_batch_preds = []
        self.predict_batch_targets = []
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        x, y = batch
        
        # Forward pass
        y_pred = self(x)
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Store predictions and targets for later metric calculation
        self.predict_batch_preds.append(y_pred_class)
        self.predict_batch_targets.append(y)
        
        # Update metrics for this batch
        self.predict_accuracy(y_pred_class, y)
        self.predict_precision(y_pred_class, y)
        self.predict_recall(y_pred_class, y)
        self.predict_f1(y_pred_class, y)
        
        # Display batch metrics in progress bar (without logging)
        batch_acc = self.predict_accuracy.compute()
        batch_f1 = self.predict_f1.compute()
        if batch_idx % 10 == 0:  # Display every 10 batches to avoid clutter
            print(f"Batch {batch_idx} - Acc: {batch_acc:.4f}, F1: {batch_f1:.4f}")
        
        return y_pred_class, y
    
    def on_predict_epoch_end(self):
        """Called at the end of the prediction epoch."""
        # Concatenate all predictions and targets
        if self.predict_batch_preds and self.predict_batch_targets:
            all_preds = torch.cat(self.predict_batch_preds)
            all_targets = torch.cat(self.predict_batch_targets)
            
            # Calculate final metrics
            final_accuracy = self.predict_accuracy.compute()
            final_precision = self.predict_precision.compute()
            final_recall = self.predict_recall.compute()
            final_f1 = self.predict_f1.compute()
            
            # Store metrics as attributes so they can be accessed later
            # Convert tensor values to Python scalars
            self.predict_metrics = {
                "predict_acc": final_accuracy.item() if isinstance(final_accuracy, torch.Tensor) else final_accuracy,
                "predict_precision": final_precision.item() if isinstance(final_precision, torch.Tensor) else final_precision,
                "predict_recall": final_recall.item() if isinstance(final_recall, torch.Tensor) else final_recall,
                "predict_f1": final_f1.item() if isinstance(final_f1, torch.Tensor) else final_f1
            }
            
            # Print final metrics
            print("\nPrediction Metrics:")
            print(f"Accuracy: {self.predict_metrics['predict_acc']:.4f}")
            print(f"Precision: {self.predict_metrics['predict_precision']:.4f}")
            print(f"Recall: {self.predict_metrics['predict_recall']:.4f}")
            print(f"F1 Score: {self.predict_metrics['predict_f1']:.4f}")
            
            # Reset metrics
            self.predict_accuracy.reset()
            self.predict_precision.reset()
            self.predict_recall.reset()
            self.predict_f1.reset()
            
            # Clear storage
            self.predict_batch_preds = []
            self.predict_batch_targets = []
    
    def training_step(self, batch, batch_idx):
        """Training step with manual optimization for proper gradient scaling."""
        # Get optimizer
        opt = self.optimizers()
        
        # Zero gradients
        opt.zero_grad()
        
        x, y = batch
        
        # Validate target labels to ensure they are within range
        if torch.any(y < 0) or torch.any(y >= self.num_classes):
            invalid_labels = y[(y < 0) | (y >= self.num_classes)]
            raise ValueError(f"Invalid target labels found: {invalid_labels.tolist()}. Labels must be in range [0, {self.num_classes-1}]")
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Manual backward pass which properly works with the gradient scaler
        self.manual_backward(loss)
        
        # Step optimizer (PyTorch Lightning handles the gradient scaling internally)
        opt.step()
        
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

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Custom optimizer step that ensures gradient scaler inf checks are properly recorded.
        This fixes the "No inf checks were recorded for this optimizer" error.
        """
        # First, ensure we have a closure for the optimizer
        if optimizer_closure is None:
            raise ValueError("optimizer_closure cannot be None")
            
        # Make sure the closure computes gradients before we continue
        optimizer_closure()
        
        # Skip optimizer step if any gradients are invalid (infinity/NaN)
        # This ensures inf checks are recorded for the optimizer
        valid_gradients = True
        for param in self.model.parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    valid_gradients = False
                    break
        
        # Only perform the optimizer step if all gradients are valid
        if valid_gradients:
            optimizer.step()
        
        # Zero gradients after stepping
        optimizer.zero_grad()

    # Alternative training_step for manual optimization with gradient clipping
    # If you want to use manual optimization, set self.automatic_optimization = False in __init__
    # and uncomment this method (comment out the current training_step method)
    """
    def training_step(self, batch, batch_idx):
        \"""Training step with manual optimization and gradient clipping.\"""
        # Get optimizer
        opt = self.optimizers()
        
        # Zero gradients
        opt.zero_grad()
        
        x, y = batch
        
        # Validate target labels to ensure they are within range
        if torch.any(y < 0) or torch.any(y >= self.num_classes):
            invalid_labels = y[(y < 0) | (y >= self.num_classes)]
            raise ValueError(f"Invalid target labels found: {invalid_labels.tolist()}. Labels must be in range [0, {self.num_classes-1}]")
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Manual backward pass which properly works with the gradient scaler
        self.manual_backward(loss)
        
        # Manual gradient clipping (since we can't use automatic gradient clipping with manual optimization)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Step optimizer (PyTorch Lightning handles the gradient scaling internally)
        opt.step()
        
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
    """