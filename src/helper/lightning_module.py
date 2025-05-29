import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union, Tuple
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy, Accuracy, Precision, Recall, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam
from icecream import ic
import os
import re

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

from configs import GeneralConfig


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
        """Initialize the classifier.
        
        Args:
            model: PyTorch model
            general_config: General configuration
            peft_config: PEFT configuration (optional)
            num_classes: Number of classes (optional)
        """
        super().__init__()
        self.model = model
        self.general_config = general_config
        self.peft_config = peft_config
        
        # Set number of classes
        self.num_classes = num_classes if num_classes is not None else general_config.num_classes
        
        # Set up loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model'])
        
        # Initialize metrics
        self._init_metrics()
        
        # Manual optimization settings
        self.automatic_optimization = False
        self.accumulation_steps = general_config.accumulation_steps
        self.current_step = 0

        # Initialize prediction metrics dictionary
        self.predict_metrics = {}
        
        # Initialize tracking variables
        self.current_train_loss = None
        self.best_val_accuracy = 0.0

    def _init_metrics(self):
        """Initialize metrics for training, validation, and testing."""
        # Get the current device
        device = self.device
        
        # Training metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted").to(device)
        self.train_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted").to(device)
        self.train_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted").to(device)
        self.train_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted").to(device)
        
        # Validation metrics
        if self.general_config.val_size > 0:
            self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted").to(device)
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted").to(device)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted").to(device)
            self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted").to(device)
        
        # Test metrics
        if self.general_config.test_size > 0:
            self.test_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted").to(device)
            self.test_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted").to(device)
            self.test_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted").to(device)
            self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted").to(device)
        
        # Prediction metrics - these will be re-initialized in on_predict_start
        # but we initialize them here as well for completeness
        if self.general_config.inference_size > 0:
            self.predict_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted").to(device)
            self.predict_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted").to(device)
            self.predict_recall = Recall(task="multiclass", num_classes=self.num_classes, average="weighted").to(device)
            self.predict_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="weighted").to(device)
        
        # Initialize prediction metrics storage
        self.predict_batch_preds = []
        self.predict_batch_targets = []
        
    def forward(self, x):
        """Forward pass through the model."""
        
        # Ensure input is float32
        x = x.float()
        
        # Debug input shape
        try:
            import logging
            logger = logging.getLogger("LIGHTNING")
            logger.debug(f"Input shape in lightning module forward: {x.shape}")
            
            # Check model type for debugging
            model_type = "unknown"
            if hasattr(self.model, 'vit'):
                model_type = "vit"
            elif hasattr(self.model, 'audio_spectrogram_transformer'):
                model_type = "ast"
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
                model_type = self.model.config.model_type
                
            logger.debug(f"Model type: {model_type}")
        except Exception as e:
            print(f"Debug logging error (non-critical): {e}")
        
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
        
        # Forward pass - simply pass the input to the model
        # The model itself now handles the correct input parameter names
        try:
            # Resize for ViT models if needed
            is_vit_model = hasattr(self.model, "config") and getattr(self.model.config, "model_type", "") == "vit"
            
            if is_vit_model and len(x.shape) == 4:
                # Check if the input dimensions match what ViT expects (224x224)
                if x.shape[2] != 224 or x.shape[3] != 224:
                    # print(f"Resizing input from {x.shape[2]}x{x.shape[3]} to 224x224 for ViT model")
                    x = torch.nn.functional.interpolate(
                        x, 
                        size=(224, 224), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Ensure channel count is correct (3 channels for ViT)
                if x.shape[1] == 1:
                    # print(f"Converting 1-channel input to 3-channel for ViT model")
                    x = x.repeat(1, 3, 1, 1)
            
            # Print shape info for debugging
            # print(f"Final input shape passed to model: {x.shape}")
            
            outputs = self.model(x)
        except Exception as e:
            print(f"Forward pass error: {e}")
            print(f"Input shape: {x.shape}, Input type: {type(x)}")
            if hasattr(self.model, 'config'):
                print(f"Model config: {self.model.config}")
            raise e
            
        # If we're using an AST model and get an error, try one more approach
        if is_ast_model:                    
            # Try to reshape the input to match expected dimensions
            if len(x.shape) == 4:  # [batch, channels, height, width]
                # AST expects specific input dimensions, try to adapt
                batch_size, channels, height, width = x.shape
                # Reshape to match expected input shape for AST
                # The exact reshape depends on the model's expectations
                x = x.view(batch_size, channels, height, width)
                
                try:
                    outputs = self.model(x)
                except Exception as e:
                    print(f"Alternate AST approach failed: {e}")
                    raise e
                    
            else:
                raise Exception("Model is not in correct AST model format")

        
        # Handle different model output formats
        if hasattr(outputs, "logits"):
            return outputs.logits
        else:
            return outputs

    def predict_step(self, batch, batch_idx):
        """Prediction step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Tuple of (predicted_classes, targets)
        """
        x, y = batch
        
        # Forward pass
        y_pred = self(x)
        
        # For transformer models that return a sequence classification output
        if hasattr(y_pred, 'logits'):
            y_pred = y_pred.logits
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Update metrics
        if self.general_config.inference_size > 0:
            self.predict_accuracy(y_pred_class, y)
            self.predict_precision(y_pred_class, y)
            self.predict_recall(y_pred_class, y)
            
            # Store predictions and targets for later use
            self.predict_batch_preds.append(y_pred_class)
            self.predict_batch_targets.append(y)
        
        # Return predicted classes and targets
        return y_pred_class, y
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Get optimizer
        opt = self.optimizers()
        
        # Only zero gradients when starting a new accumulation cycle
        if self.current_step % self.accumulation_steps == 0:
            opt.zero_grad()
        
        # Get the input and target
        x, y = batch
        
        # Validate target labels to ensure they are within range
        if torch.any(y < 0) or torch.any(y >= self.num_classes):
            invalid_labels = y[(y < 0) | (y >= self.num_classes)]
            raise ValueError(f"Invalid target labels found: {invalid_labels.tolist()}. Labels must be in range [0, {self.num_classes-1}]")
        
        # Forward pass
        y_pred = self(x)
        
        # For transformer models that return a sequence classification output
        if hasattr(y_pred, 'logits'):
            y_pred = y_pred.logits
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Store current loss for epoch-level logging
        self.current_train_loss = loss.detach().clone()
        
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass with scaled loss
        self.manual_backward(scaled_loss)
        
        # Update weights only when accumulation is complete
        if (self.current_step + 1) % self.accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
        
        # Increment step counter
        self.current_step += 1
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Update metrics
        self.train_accuracy(y_pred_class, y)
        self.train_f1(y_pred_class, y)
        self.train_precision(y_pred_class, y)
        self.train_recall(y_pred_class, y)
        
        # Log metrics - ensure they appear in progress bar and are formatted consistently
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, sync_dist=True)

        # test_loss = self.test_step(batch, batch_idx)
        
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
        
        # For transformer models that return a sequence classification output
        if hasattr(y_pred, 'logits'):
            y_pred = y_pred.logits
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Update metrics
        if self.general_config.val_size > 0:
            self.val_accuracy(y_pred_class, y)
            self.val_precision(y_pred_class, y)
            self.val_recall(y_pred_class, y)
            self.val_f1(y_pred_class, y)
        
            # Log metrics - ensure they appear in progress bar and are formatted consistently
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
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
        
        # For transformer models that return a sequence classification output
        if hasattr(y_pred, 'logits'):
            y_pred = y_pred.logits
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Get predicted classes
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Update metrics
        if self.general_config.test_size > 0:
            self.test_accuracy(y_pred_class, y)
            self.test_precision(y_pred_class, y)
            self.test_recall(y_pred_class, y)
            self.test_f1(y_pred_class, y)
            
            # Log metrics - changed to only log at epoch level
            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Choose optimizer based on model type
        if re.match(r'^(ast|vit|mert)', self.general_config.model_type.lower()):
            optimizer =  AdamW(self.model.parameters(), lr=self.general_config.learning_rate, weight_decay=0.05) # TODO imp a config feature for optimizer choices
        else:
            optimizer = Adam(self.model.parameters(), lr=self.general_config.learning_rate)
        
        # Configure scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.85, 
            patience=5,
        )
        
        # Store scheduler as an attribute so we can access it in on_epoch_end
        self.lr_scheduler = scheduler
        
        # For manual optimization, just return the optimizer
        if not self.automatic_optimization:
            return optimizer
        
        # For automatic optimization, return with scheduler config
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
        epoch: int = None,
        batch_idx: int = None,
        optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        """
        Custom optimizer step that ensures gradient scaler inf checks are properly recorded.
        This fixes the "No inf checks were recorded for this optimizer" error.
        Also implements gradient clipping to stabilize training.
        """
        # First, ensure we have a closure for the optimizer
        if optimizer_closure is None:
            raise ValueError("optimizer_closure cannot be None")
            
        # Compute loss and gradients
        loss = optimizer_closure()
        
        # Apply gradient clipping if configured
        if hasattr(self.general_config, 'gradient_clip_val') and self.general_config.gradient_clip_val > 0:
            clip_val = self.general_config.gradient_clip_val
            # Clip gradients by value
            torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=clip_val)
        
        # Skip optimizer step if any gradients are invalid (infinity/NaN)
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
        
        # Return the loss value
        return loss

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch.
        If using manual optimization, manually step the learning rate scheduler.
        """
        if not self.automatic_optimization and hasattr(self, 'lr_scheduler'):
            # Get the current validation loss
            val_loss = self.trainer.callback_metrics.get('val_loss')
            if val_loss is not None:
                # Step the scheduler with the validation loss
                self.lr_scheduler.step(val_loss)
