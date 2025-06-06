import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union, Tuple
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy, Accuracy, Precision, Recall, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau, SequentialLR, LambdaLR
from torch.optim import AdamW, Adam
from icecream import ic
import os
import re
import math

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

from configs import GeneralConfig
from configs.optim_config import OptimizerConfig


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
        optimizer_config: Optional[OptimizerConfig] = None,
    ):
        """Initialize the classifier.
        
        Args:
            model: PyTorch model
            general_config: General configuration
            peft_config: PEFT configuration (optional)
            num_classes: Number of classes (optional)
            optimizer_config: Optimizer configuration (optional)
        """
        super().__init__()
        self.model = model
        self.general_config = general_config
        self.peft_config = peft_config
        self.optimizer_config = optimizer_config or OptimizerConfig()  # Use default if not provided
        
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
        # Initialize validation metrics if we have val_size > 0 OR if we're using k-fold (which always provides separate val dataloaders)
        # OR if this is an ESC dataset that uses fold-based splits (they always provide validation data)
        should_init_val_metrics = (
            (self.general_config.val_size > 0) or 
            (hasattr(self.general_config, 'use_kfold') and self.general_config.use_kfold) or
            # Check for ESC datasets that use fold-based splits
            (hasattr(self.general_config, 'fold_based_split') and getattr(self.general_config, 'fold_based_split', False))
        )
        
        if should_init_val_metrics:
            self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted").to(device)
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted").to(device)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted").to(device)
            self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted").to(device)
        
        # Test metrics - only when test_size > 0 and NOT using k-fold (k-fold ignores test/inference)
        should_init_test_metrics = (self.general_config.test_size > 0) and not (hasattr(self.general_config, 'use_kfold') and self.general_config.use_kfold)
        
        if should_init_test_metrics:
            self.test_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted").to(device)
            self.test_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted").to(device)
            self.test_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted").to(device)
            self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted").to(device)
        
        # Prediction metrics - only when inference_size > 0 and NOT using k-fold (k-fold ignores test/inference)
        should_init_predict_metrics = (self.general_config.inference_size > 0) and not (hasattr(self.general_config, 'use_kfold') and self.general_config.use_kfold)
        
        if should_init_predict_metrics:
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
        
        # Update metrics - only when inference_size > 0 and NOT using k-fold (k-fold ignores test/inference)
        should_log_predict_metrics = (self.general_config.inference_size > 0) and not (hasattr(self.general_config, 'use_kfold') and self.general_config.use_kfold)
        
        if should_log_predict_metrics:
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
        
        # Initialize validation metrics on the fly if they don't exist
        # This handles cases where validation data exists but wasn't detected during _init_metrics
        if not hasattr(self, 'val_accuracy'):
            device = self.device
            self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted").to(device)
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average="weighted").to(device)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average="weighted").to(device)
            self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted").to(device)
            print("Initialized validation metrics on the fly - validation data detected")
        
        # Update validation metrics (always log if validation_step is called)
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
        
        # Update metrics - only when test_size > 0 and NOT using k-fold (k-fold ignores test/inference)
        should_log_test_metrics = (self.general_config.test_size > 0) and not (hasattr(self.general_config, 'use_kfold') and self.general_config.use_kfold)
        
        if should_log_test_metrics:
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
    
    def _create_warmup_scheduler(self, optimizer, target_lr: float):
        """Create a warmup scheduler based on configuration."""
        warmup_config = self.optimizer_config.warmup
        
        if warmup_config.warmup_method == "linear":
            # Linear warmup: lr multiplier increases linearly from start_lr/target_lr to 1.0
            def lr_lambda(step):
                if step < warmup_config.warmup_steps:
                    start_factor = warmup_config.warmup_start_lr / target_lr
                    lr_multiplier = start_factor + (1.0 - start_factor) * step / warmup_config.warmup_steps
                    return lr_multiplier
                return 1.0
        else:  # cosine
            # Cosine warmup: lr multiplier increases following cosine curve
            def lr_lambda(step):
                if step < warmup_config.warmup_steps:
                    start_factor = warmup_config.warmup_start_lr / target_lr
                    warmup_progress = step / warmup_config.warmup_steps
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * (1 - warmup_progress)))
                    lr_multiplier = start_factor + (1.0 - start_factor) * (1 - cosine_factor)
                    return lr_multiplier
                return 1.0
        
        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer configuration based on type
        if self.optimizer_config.optimizer_type == "adamw":
            optimizer_params = dict(self.optimizer_config.adamw)
            target_lr = optimizer_params['lr']
            optimizer = AdamW(self.model.parameters(), **optimizer_params)
        elif self.optimizer_config.optimizer_type == "adam":
            optimizer_params = dict(self.optimizer_config.adam)
            target_lr = optimizer_params['lr']
            optimizer = Adam(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_config.optimizer_type}")
        
        # Handle warmup + scheduler combinations
        final_scheduler = None
        
        if self.optimizer_config.warmup.enabled:
            print(f"LR warmup enabled: {self.optimizer_config.warmup.warmup_steps} steps, method: {self.optimizer_config.warmup.warmup_method}")
            warmup_scheduler = self._create_warmup_scheduler(optimizer, target_lr)
            
            # Check if we have a base scheduler that's compatible with SequentialLR
            if self.optimizer_config.scheduler_type == "reduce_lr_on_plateau":
                # ReduceLROnPlateau is not compatible with SequentialLR
                # We'll use only warmup for now and handle ReduceLROnPlateau manually
                final_scheduler = warmup_scheduler
                # Store the base scheduler config for manual handling later
                self.base_scheduler_config = {
                    "type": "reduce_lr_on_plateau",
                    "params": dict(self.optimizer_config.reduce_lr_on_plateau)
                }
                self.warmup_steps = self.optimizer_config.warmup.warmup_steps
                self.post_warmup_scheduler = None
            elif self.optimizer_config.scheduler_type == "step_lr":
                from torch.optim.lr_scheduler import StepLR
                scheduler_params = dict(self.optimizer_config.step_lr)
                base_scheduler = StepLR(optimizer, **scheduler_params)
                final_scheduler = SequentialLR(optimizer, [warmup_scheduler, base_scheduler], [self.optimizer_config.warmup.warmup_steps])
            elif self.optimizer_config.scheduler_type == "cosine_annealing_lr":
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler_params = dict(self.optimizer_config.cosine_annealing_lr)
                base_scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
                final_scheduler = SequentialLR(optimizer, [warmup_scheduler, base_scheduler], [self.optimizer_config.warmup.warmup_steps])
            else:
                # Only warmup, no base scheduler
                final_scheduler = warmup_scheduler
        else:
            # No warmup, just use base scheduler if specified
            if self.optimizer_config.scheduler_type == "reduce_lr_on_plateau":
                scheduler_params = dict(self.optimizer_config.reduce_lr_on_plateau)
                final_scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
            elif self.optimizer_config.scheduler_type == "step_lr":
                from torch.optim.lr_scheduler import StepLR
                scheduler_params = dict(self.optimizer_config.step_lr)
                final_scheduler = StepLR(optimizer, **scheduler_params)
            elif self.optimizer_config.scheduler_type == "cosine_annealing_lr":
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler_params = dict(self.optimizer_config.cosine_annealing_lr)
                final_scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
        
        # Store scheduler as an attribute so we can access it in on_epoch_end
        self.lr_scheduler = final_scheduler
        
        # For manual optimization, just return the optimizer
        if not self.automatic_optimization:
            return optimizer
        
        # For automatic optimization, return with scheduler config if scheduler exists
        if final_scheduler is not None:
            if self.optimizer_config.scheduler_type == "reduce_lr_on_plateau":
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": final_scheduler,
                        "monitor": "val_loss",
                        "interval": "step" if self.optimizer_config.warmup.enabled else "epoch",
                        "frequency": 1
                    }
                }
            else:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": final_scheduler,
                        "interval": "step" if self.optimizer_config.warmup.enabled else "epoch",
                        "frequency": 1
                    }
                }
        else:
            return optimizer

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
        if (self.optimizer_config.gradient_clipping_enabled and 
            self.optimizer_config.gradient_clip_val is not None and 
            self.optimizer_config.gradient_clip_val > 0):
            clip_val = self.optimizer_config.gradient_clip_val
            if self.optimizer_config.gradient_clip_algorithm == "value":
                # Clip gradients by value
                torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=clip_val)
            elif self.optimizer_config.gradient_clip_algorithm == "norm":
                # Clip gradients by norm
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_val)
        
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
        Also handle transition from warmup to ReduceLROnPlateau when needed.
        """
        if not self.automatic_optimization and hasattr(self, 'lr_scheduler'):
            # Handle warmup + ReduceLROnPlateau combination
            if (hasattr(self, 'base_scheduler_config') and 
                self.base_scheduler_config["type"] == "reduce_lr_on_plateau" and
                hasattr(self, 'warmup_steps')):
                
                # Check if we've finished warmup and need to initialize ReduceLROnPlateau
                current_step = self.global_step
                if current_step >= self.warmup_steps and self.post_warmup_scheduler is None:
                    # Initialize ReduceLROnPlateau after warmup
                    opt = self.optimizers()
                    self.post_warmup_scheduler = ReduceLROnPlateau(opt, **self.base_scheduler_config["params"])
                    print(f"LR warmup completed at step {current_step}. Switching to ReduceLROnPlateau scheduler.")
                
                # Use ReduceLROnPlateau if warmup is done
                if self.post_warmup_scheduler is not None:
                    val_loss = self.trainer.callback_metrics.get('val_loss')
                    if val_loss is not None:
                        self.post_warmup_scheduler.step(val_loss)
            else:
                # Standard scheduler stepping
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    val_loss = self.trainer.callback_metrics.get('val_loss')
                    if val_loss is not None:
                        self.lr_scheduler.step(val_loss)
                else:
                    # For other schedulers, step without arguments
                    self.lr_scheduler.step()
