import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import wandb
from pathlib import Path
from icecream import ic
from datetime import datetime

from .lightning_module import AudioClassifier
from .datamodule import AudioDataModule
from configs import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig, wandb_config_dict
from .util import wandb_login
import time


class PTLTrainer:
    """
    PyTorch Lightning trainer that handles training using a single GPU.
    This replaces the functionality in engine.py and fold_engine.py.
    """
    def __init__(
        self,
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        peft_config: Any,
        wandb_config: WandbConfig,
        sweep_config: SweepConfig,
        data_module: AudioDataModule,
        model_factory: Callable,
    ):
        """
        Initialize the PTLTrainer.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            peft_config: PEFT configuration
            wandb_config: WandB configuration
            sweep_config: Sweep configuration
            data_module: Audio data module
            model_factory: Model factory function
        """
        self.general_config = general_config
        self.feature_extraction_config = feature_extraction_config
        self.peft_config = peft_config
        self.wandb_config = wandb_config
        self.sweep_config = sweep_config
        self.data_module = data_module
        self.model_factory = model_factory
        
        # Single GPU configuration
        self.gpu_available = torch.cuda.is_available()
        
        # Set up wandb logger
        self.wandb_logger = None
        if self.general_config.use_wandb:
            wandb_login()
            self.wandb_logger = WandbLogger(
                project=self.wandb_config.project,
                name=self.wandb_config.name,
                tags=self.wandb_config.tags if self.wandb_config.tags else [],
                notes=self.wandb_config.notes,
                log_model=False,
                save_dir="wandb",
                config=wandb_config_dict(self.general_config, self.feature_extraction_config, self.peft_config, self.wandb_config),
                group=self.wandb_config.group if hasattr(self.wandb_config, 'group') and self.wandb_config.group else None,
                reinit=True  # Force reinitialize a new wandb run
            )
        
        # Set device
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
    
        
        # Set random seeds for reproducibility
        torch.manual_seed(general_config.seed)
        torch.cuda.manual_seed(general_config.seed)
        np.random.seed(general_config.seed)
    
    def _get_callbacks(self) -> List[pl.Callback]:
        """
        Get the callbacks for the trainer.
        
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping callback
        if self.general_config.patience > 0:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.general_config.patience,
                mode="min",
                verbose=True
            )
            callbacks.append(early_stopping)
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        # Custom progress bar with better metric display
        class CustomProgressBar(TQDMProgressBar):
            def get_metrics(self, trainer, model):
                # Get metrics from parent class
                items = super().get_metrics(trainer, model)
                
                # Format all metrics to 2 decimal places
                for key in list(items.keys()):
                    if key in ['v_num', 'epoch', 'step']:
                        continue
                    if isinstance(items[key], (float, int, torch.Tensor)):
                        try:
                            # Convert to float first to handle tensor values
                            value = float(items[key])
                            items[key] = f"{value:.2f}"
                        except (ValueError, TypeError):
                            # If conversion fails, keep the original value
                            pass
                
                # Reorder metrics to show in desired order
                ordered_items = {}
                
                # First show epoch and step
                if 'epoch' in items:
                    ordered_items['epoch'] = items['epoch']
                if 'step' in items:
                    ordered_items['step'] = items['step']
                
                # Then show train metrics
                for metric in ['train_loss', 'train_acc']:
                    if metric in items:
                        ordered_items[metric] = items[metric]
                
                # Then show validation metrics
                for metric in ['val_loss', 'val_acc']:
                    if metric in items:
                        ordered_items[metric] = items[metric]
                
                # Then show test metrics
                for metric in ['test_loss', 'test_acc']:
                    if metric in items:
                        ordered_items[metric] = items[metric]
                
                # Add any remaining metrics
                for key, val in items.items():
                    if key not in ordered_items:
                        ordered_items[key] = val
                
                return ordered_items
        
        # Add custom progress bar
        progress_bar = CustomProgressBar()
        callbacks.append(progress_bar)

        return callbacks
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dict[str, Any]: Test results
        """
        import torch
        
        # Set float32 matmul precision to 'high' to properly utilize Tensor Cores on CUDA devices
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            print("Set float32 matmul precision to 'high' for better performance on Tensor Core GPUs")
        
        # Check if deterministic algorithms are enabled and disable them if needed
        # This is necessary for operations like interpolation that don't have deterministic implementations
        if torch.are_deterministic_algorithms_enabled():
            print("Deterministic algorithms are enabled. Disabling for compatibility with certain operations.")
            torch.use_deterministic_algorithms(False)
        
        # Create model
        model, feature_extractor = self.model_factory(self.device)
        
        # Create a custom callback to display metrics after each epoch
        class MetricsDisplayCallback(pl.Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                metrics = trainer.callback_metrics
                
                # Extract metrics
                train_loss = metrics.get('train_loss_epoch', None)
                train_acc = metrics.get('train_acc_epoch', None)
                val_loss = metrics.get('val_loss', None)
                val_acc = metrics.get('val_acc', None)
                
                # Convert tensors to floats
                if isinstance(train_loss, torch.Tensor):
                    train_loss = train_loss.item()
                if isinstance(train_acc, torch.Tensor):
                    train_acc = train_acc.item()
                if isinstance(val_loss, torch.Tensor):
                    val_loss = val_loss.item()
                if isinstance(val_acc, torch.Tensor):
                    val_acc = val_acc.item()
                
                # Print metrics in a nice format
                print("\n" + "-"*50)
                print(f"Epoch {trainer.current_epoch} Metrics:")
                
                # Format metrics safely, handling None values
                train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
                train_acc_str = f"{train_acc:.4f}" if train_acc is not None else "N/A"
                val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
                
                print(f"Train Loss: {train_loss_str} | Train Acc: {train_acc_str}")
                print(f"Val Loss: {val_loss_str} | Val Acc: {val_acc_str}")
                print("-"*50 + "\n")
        
        # Create trainer with our custom callbacks
        callbacks = self._get_callbacks()
        callbacks.append(MetricsDisplayCallback())
        
        trainer = pl.Trainer(
            max_epochs=self.general_config.epochs,
            accelerator="gpu" if self.gpu_available else "cpu",
            devices=1,
            callbacks=callbacks,
            logger=self.wandb_logger,
            deterministic=False,
            precision=32  # Changed to 32-bit precision to avoid AMP issues
        )
        ic("trainer created")
        
        # Ensure data module is set up - call setup directly
        try:
            self.data_module.setup()
        except Exception as e:
            print(f"Warning: Error during data module setup: {str(e)}")
            print("This may be normal if the data module is already set up.")
            
        # Validate number of classes
        if not hasattr(self.data_module, 'num_classes') or self.data_module.num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {getattr(self.data_module, 'num_classes', None)}. Must be positive.")
            
        # Log number of classes for debugging
        print(f"Number of classes in data module: {self.data_module.num_classes}")
        
        # Create lightning module
        lightning_module = AudioClassifier(
            model=model,
            general_config=self.general_config,
            peft_config=self.peft_config,
            num_classes=self.data_module.num_classes
        )
        
        # Print training start message with clear formatting
        print("\n" + "="*80)
        print(f"STARTING TRAINING: {self.general_config.model_type.upper()} MODEL")
        print(f"Epochs: {self.general_config.epochs} | Batch Size: {self.general_config.batch_size} | LR: {self.general_config.learning_rate}")
        print("="*80 + "\n")
        
        # Start timer
        start_time = time.time()
        
        try:
            trainer.fit(
                model=lightning_module,
                datamodule=self.data_module
            )
        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Check if this is a CUDA assertion error related to class labels
            if "nll_loss_forward" in str(e) and "Assertion `t >= 0 && t < n_classes`" in str(e):
                print("\nERROR: Invalid class labels detected in your dataset.")
                print(f"The model expects labels in range [0, {self.data_module.num_classes-1}], but found labels outside this range.")
                print("Please check your dataset preprocessing and ensure all labels are correctly mapped.")
                return {"error": "Invalid class labels in dataset"}
            raise  # Re-raise the exception if it's not the specific error we're handling
        
        
        def _format_time(seconds):
            """Convert seconds to mm:ss format"""
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"
        
        
        end_time = time.time() - start_time
        formatted_end_time = _format_time(end_time)
        # Print total train time
        print(f"\nTotal training time: {formatted_end_time}")
        
        if self.wandb_logger:
            self.wandb_logger.experiment.log({
                "total_train_time": formatted_end_time
            })
        
        # Run test evaluation after training
        print("\n" + "="*80)
        print("RUNNING TEST EVALUATION")
        print("="*80 + "\n")
        
        test_results = trainer.test(
            model=lightning_module,
            datamodule=self.data_module
        )
        
        # Print test results in a nice table
        if test_results:
            print("\n" + "="*80)
            print("TEST RESULTS")
            print("="*80)
            
            print(f"{'Metric':<20} {'Value':<10}")
            print("-"*30)
            
            # Print test metrics in a specific order
            test_metric_order = [
                'test_loss_epoch', 'test_acc_epoch', 'test_f1_epoch', 'test_precision_epoch', 'test_recall_epoch'
            ]
            
            for metric_name in test_metric_order:
                if metric_name in test_results[0]:
                    value = test_results[0][metric_name]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    print(f"{metric_name:<20} {value:.2f}")
            
            print("="*80 + "\n")
            
            # Log final test metrics to wandb summary
            if self.wandb_logger:
                for metric_name in test_metric_order:
                    if metric_name in test_results[0]:
                        value = test_results[0][metric_name]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        # Add to wandb summary
                        self.wandb_logger.experiment.summary[f"final_{metric_name}"] = value
        
        # Run inference on the inference split if available
        inference_results = {}
        try:
            # Check if inference dataset exists and has samples
            has_inference_data = False
            try:
                inference_loader = self.data_module.predict_dataloader()
                has_inference_data = len(inference_loader.dataset) > 0
            except (ValueError, AttributeError, TypeError) as e:
                print(f"No inference data available: {str(e)}")
                has_inference_data = False
            
            if has_inference_data:
                print("\n" + "="*80)
                print("RUNNING INFERENCE EVALUATION")
                print("="*80 + "\n")
                
                inference_results = self._run_inference(trainer, lightning_module)
                
                # Print inference results in a nice table
                if inference_results:
                    print("\n" + "="*80)
                    print("INFERENCE RESULTS")
                    print("="*80)
                    
                    print(f"{'Metric':<20} {'Value':<10}")
                    print("-"*30)
                    
                    # Print inference metrics in a specific order
                    inference_metric_order = [
                        'inference_acc', 'inference_precision', 'inference_recall', 'inference_f1'
                    ]
                    
                    for metric_name in inference_metric_order:
                        if metric_name in inference_results:
                            value = inference_results[metric_name]
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            print(f"{metric_name:<20} {value:.2f}")
                    
                    print("="*80 + "\n")
                
                # Log inference results to wandb summary
                if self.wandb_logger and inference_results:
                    for key, value in inference_results.items():
                        # Log to metrics (regular logging)
                        self.wandb_logger.log_metrics({key: value})
                        # Also add to wandb summary
                        self.wandb_logger.experiment.summary[f"final_{key}"] = value
            else:
                print("No inference data available, skipping inference evaluation.")
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            print("Continuing with training results...")
        
        # Save model if enabled
        if self.general_config.save_model:
            model_path = Path("saved_models")
            model_path.mkdir(parents=True, exist_ok=True)
            model_name = f"{self.general_config.model_type}_classifier.pt"
            trainer.save_checkpoint(str(model_path / model_name))
            print(f"\nModel saved to {model_path / model_name}")
        
        # Ensure the return type is Dict[str, Any]
        results_dict: Dict[str, Any] = {} if not test_results else dict(test_results[0])
        
        # Add any additional metrics you want to track
        if hasattr(lightning_module, 'best_val_accuracy'):
            results_dict['best_val_accuracy'] = lightning_module.best_val_accuracy
        
        # Add inference results to the overall results
        if inference_results:
            for key, value in inference_results.items():
                results_dict[key] = value
        
        # Log best validation metrics to wandb summary
        if self.wandb_logger and hasattr(lightning_module, 'best_val_accuracy'):
            self.wandb_logger.experiment.summary['final_best_val_acc'] = lightning_module.best_val_accuracy
            
            # Log other best validation metrics if available
            if hasattr(lightning_module, 'best_val_f1'):
                self.wandb_logger.experiment.summary['final_best_val_f1'] = lightning_module.best_val_f1
            
            if hasattr(lightning_module, 'best_val_precision'):
                self.wandb_logger.experiment.summary['final_best_val_precision'] = lightning_module.best_val_precision
                
            if hasattr(lightning_module, 'best_val_recall'):
                self.wandb_logger.experiment.summary['final_best_val_recall'] = lightning_module.best_val_recall
        
        # Print a summary of all metrics at the end
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Model: {self.general_config.model_type}")
        print(f"Total training time: {formatted_end_time}")
        
        # Get validation accuracy
        val_acc = None
        if 'val_acc' in trainer.callback_metrics:
            val_acc = trainer.callback_metrics['val_acc']
            if isinstance(val_acc, torch.Tensor):
                val_acc = val_acc.item()
        
        # Get test accuracy
        test_acc = None
        if 'test_acc_epoch' in results_dict:
            test_acc = results_dict['test_acc_epoch']
            if isinstance(test_acc, torch.Tensor):
                test_acc = test_acc.item()
        
        # Get inference accuracy
        inf_acc = None
        if 'inference_acc' in results_dict:
            inf_acc = results_dict['inference_acc']
            if isinstance(inf_acc, torch.Tensor):
                inf_acc = inf_acc.item()
        
        # Format metrics safely, handling None values
        val_acc_str = f"{val_acc:.2f}" if val_acc is not None else "N/A"
        test_acc_str = f"{test_acc:.2f}" if test_acc is not None else "N/A"
        inf_acc_str = f"{inf_acc:.2f}" if inf_acc is not None else "N/A"
        
        print(f"Final validation accuracy: {val_acc_str}")
        print(f"Test accuracy: {test_acc_str}")
        print(f"Inference accuracy: {inf_acc_str}")
        
        print("="*80 + "\n")
        
        return results_dict
    
    def _run_inference(self, trainer: pl.Trainer, model: pl.LightningModule) -> Dict[str, Any]:
        """
        Run inference on the inference dataset.
        
        Args:
            trainer: PyTorch Lightning trainer
            model: Trained PyTorch Lightning module
            
        Returns:
            Dictionary of inference results
        """
        try:
            # Get number of classes from data module
            num_classes = self.data_module.num_classes
            
            # Get the current device
            device = model.device
            
            # Run prediction
            print("Starting inference with predict...")
            predictions = trainer.predict(
                model=model,
                datamodule=self.data_module
            )
            
            if not predictions:
                print("No predictions returned from trainer.predict()")
                return {}
            
            # Get metrics from the model's stored metrics attribute
            metrics = {}
            
            # Check if model has predict_metrics attribute (added in our fix)
            if hasattr(model, 'predict_metrics') and model.predict_metrics:
                metrics = {
                    "inference_acc": model.predict_metrics.get("predict_acc", None),
                    "inference_precision": model.predict_metrics.get("predict_precision", None),
                    "inference_recall": model.predict_metrics.get("predict_recall", None),
                    "inference_f1": model.predict_metrics.get("predict_f1", None)
                }
                
                # Log metrics to wandb if wandb logger is enabled
                if self.wandb_logger and all(metrics.values()):
                    print("Logging prediction metrics to wandb...")
                    self.wandb_logger.experiment.log({
                        "inference_accuracy": metrics["inference_acc"],
                        "inference_precision": metrics["inference_precision"],
                        "inference_recall": metrics["inference_recall"],
                        "inference_f1": metrics["inference_f1"]
                    })
            
            # If metrics are not available from model attributes, calculate them manually
            if not all(metrics.values()):
                print("Calculating inference metrics manually...")
            
            # Collect all predictions and ground truth
            all_preds = []
            all_targets = []
            
            for batch_preds in predictions:
                batch_pred_classes, batch_targets = batch_preds
                all_preds.append(batch_pred_classes)
                all_targets.append(batch_targets)
            
            # Concatenate all batches
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            
            # Ensure tensors are on the same device
            all_preds = all_preds.to(device)
            all_targets = all_targets.to(device)
            
            # Calculate metrics
            from torchmetrics.functional.classification import (
                multiclass_accuracy,
                multiclass_precision,
                multiclass_recall,
                multiclass_f1_score
            )
            
            # Calculate metrics using num_classes from data_module
            accuracy = multiclass_accuracy(
                all_preds, all_targets, num_classes=num_classes, average="weighted"
            ).item()
            
            precision = multiclass_precision(
                all_preds, all_targets, num_classes=num_classes, average="weighted"
            ).item()
            
            recall = multiclass_recall(
                all_preds, all_targets, num_classes=num_classes, average="weighted"
            ).item()
            
            f1 = multiclass_f1_score(
                all_preds, all_targets, num_classes=num_classes, average="weighted"
            ).item()
            
            # Update metrics
            metrics = {
                "inference_acc": accuracy,
                "inference_precision": precision,
                "inference_recall": recall,
                "inference_f1": f1
            }
            
            # Store metrics in model for future reference
            if not hasattr(model, 'predict_metrics'):
                model.predict_metrics = {}
            
            model.predict_metrics = {
                "predict_acc": accuracy,
                "predict_precision": precision,
                "predict_recall": recall,
                "predict_f1": f1
            }
            
            # Log manually calculated metrics to wandb if wandb logger is enabled
            if self.wandb_logger:
                print("Logging manually calculated metrics to wandb...")
                self.wandb_logger.experiment.log({
                    "inference_accuracy": metrics["inference_acc"],
                    "inference_precision": metrics["inference_precision"],
                    "inference_recall": metrics["inference_recall"],
                    "inference_f1": metrics["inference_f1"]
                })
    
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
        
        # Create confusion matrix
        try:
            # We still need to collect predictions to create the confusion matrix
            all_preds = []
            all_targets = []
            
            for batch_preds in predictions:
                batch_pred_classes, batch_targets = batch_preds
                all_preds.append(batch_pred_classes)
                all_targets.append(batch_targets)
            
            # Concatenate all batches
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            
            # Ensure tensors are on the same device
            all_preds = all_preds.to(device)
            all_targets = all_targets.to(device)
            
            # Create confusion matrix
            from torchmetrics.functional.classification import confusion_matrix
            conf_mat = confusion_matrix(
                all_preds, all_targets, num_classes=num_classes, task="multiclass"
            ).cpu().numpy()
            
            # Log confusion matrix if wandb is enabled
            if self.wandb_logger:
                import wandb
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Get class names if available
                class_names = None
                try:
                    _, _, idx_to_class = self.data_module.get_class_info()
                    class_names = [idx_to_class[i] for i in range(num_classes)]
                except (AttributeError, KeyError):
                    class_names = [str(i) for i in range(num_classes)]
                
                # Create confusion matrix plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Inference Confusion Matrix')
                
                # Log to wandb
                self.wandb_logger.experiment.log({
                    "inference_confusion_matrix": wandb.Image(plt),
                    "final_inference_accuracy": metrics["inference_acc"],
                    "final_inference_precision": metrics["inference_precision"],
                    "final_inference_recall": metrics["inference_recall"],
                    "final_inference_f1": metrics["inference_f1"]
                })
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {str(e)}")
        
        return metrics
    
    def k_fold_cross_validation(self) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation using PyTorch Lightning.
        
        Returns:
            Dictionary of aggregated results across all folds
        """
        import torch
        
        # Set float32 matmul precision to 'high' to properly utilize Tensor Cores on CUDA devices
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            print("Set float32 matmul precision to 'high' for better performance on Tensor Core GPUs")
        
        if not self.general_config.use_kfold:
            raise ValueError("K-fold cross-validation is not enabled in the configuration.")
        
        # Set up data module for k-fold
        self.data_module.setup()
        
        # Results storage
        all_fold_results = []
        best_model = None
        best_val_loss = float('inf')
        
        # Initialize WandB for all folds if enabled
        # We'll use a single WandB run for all folds
        if self.general_config.use_wandb and self.wandb_logger is None:
            wandb_login()
            self.wandb_logger = WandbLogger(
                project=self.wandb_config.project,
                name=self.wandb_config.name,
                tags=self.wandb_config.tags if self.wandb_config.tags else [],
                notes=self.wandb_config.notes,
                log_model=False,
                save_dir="wandb",
                config=wandb_config_dict(self.general_config, self.feature_extraction_config, self.peft_config, self.wandb_config),
                group=self.wandb_config.group if hasattr(self.wandb_config, 'group') and self.wandb_config.group else None,
                reinit=True  # Force reinitialize a new wandb run
            )
        
        # Print k-fold start message with clear formatting
        print("\n" + "="*80)
        print(f"STARTING {self.general_config.k_folds}-FOLD CROSS-VALIDATION: {self.general_config.model_type.upper()} MODEL")
        print(f"Epochs: {self.general_config.epochs} | Batch Size: {self.general_config.batch_size} | LR: {self.general_config.learning_rate}")
        print("="*80 + "\n")
        
        # Train on each fold
        for fold in range(self.general_config.k_folds):
            print("\n" + "="*80)
            print(f"TRAINING FOLD {fold+1}/{self.general_config.k_folds}")
            print("="*80)
            
            # Get fold dataloaders
            fold_train_loader, fold_val_loader = self.data_module.get_fold_dataloaders(fold)
            
            # Create new model instance for this fold
            model, feature_extractor = self.model_factory(self.device)
            
            # Create Lightning module
            lightning_module = AudioClassifier(
                model=model,
                general_config=self.general_config,
                peft_config=self.peft_config,
                num_classes=self.data_module.num_classes
            )
            
            # Create checkpoint directory for this fold
            checkpoint_dir = Path("checkpoints") / f"fold_{fold+1}" / datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create custom progress bar for this fold
            class CustomFoldProgressBar(TQDMProgressBar):
                def get_metrics(self, trainer, model):
                    # Get metrics from parent class
                    items = super().get_metrics(trainer, model)
                    
                    # Format all metrics to 2 decimal places
                    for key in list(items.keys()):
                        if key in ['v_num', 'epoch', 'step']:
                            continue
                        if isinstance(items[key], (float, int, torch.Tensor)):
                            try:
                                # Convert to float first to handle tensor values
                                value = float(items[key])
                                items[key] = f"{value:.2f}"
                            except (ValueError, TypeError):
                                # If conversion fails, keep the original value
                                pass
                    
                    # Reorder metrics to show in desired order
                    ordered_items = {}
                    
                    # First show epoch and step
                    if 'epoch' in items:
                        ordered_items['epoch'] = items['epoch']
                    if 'step' in items:
                        ordered_items['step'] = items['step']
                    
                    # Then show train metrics
                    for metric in ['train_loss', 'train_acc']:
                        if metric in items:
                            ordered_items[metric] = items[metric]
                    
                    # Then show validation metrics
                    for metric in ['val_loss', 'val_acc']:
                        if metric in items:
                            ordered_items[metric] = items[metric]
                    
                    # Add any remaining metrics
                    for key, val in items.items():
                        if key not in ordered_items:
                            ordered_items[key] = val
                    
                    return ordered_items
            
            # Create callbacks for this fold
            fold_callbacks = [
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="{epoch}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    verbose=True
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.general_config.patience,
                    mode="min",
                    verbose=True
                ),
                LearningRateMonitor(logging_interval="epoch"),
                CustomFoldProgressBar()
            ]
            
            # # Create trainer for this fold
            # trainer = pl.Trainer(
            #     max_epochs=self.general_config.epochs,
            #     accelerator="gpu" if self.gpu_available else "cpu",
            #     devices=1,  # Always use a single device
            #     callbacks=fold_callbacks,
            #     logger=self.wandb_logger,
            #     gradient_clip_val=1.0,
            #     accumulate_grad_batches=self.general_config.accumulation_steps,
            #     deterministic=True,
            #     precision="16-mixed" if self.gpu_available else "32"
            # )
            
            trainer = pl.Trainer(
            max_epochs=self.general_config.epochs,
            accelerator="gpu" if self.gpu_available else "cpu",
            devices=1,
            callbacks=fold_callbacks,
            logger=self.wandb_logger,
            deterministic=False,
            precision=32  # Changed to 32-bit precision to avoid AMP issues
            )
            
            
            
            # Train on this fold
            trainer.fit(
                model=lightning_module,
                train_dataloaders=fold_train_loader,
                val_dataloaders=fold_val_loader
            )
            
            # Get validation loss
            val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
            val_acc = trainer.callback_metrics.get("val_acc", 0.0)
            val_f1 = trainer.callback_metrics.get("val_f1", 0.0)
            val_precision = trainer.callback_metrics.get("val_precision", 0.0)
            val_recall = trainer.callback_metrics.get("val_recall", 0.0)
            
            # Convert tensor values to Python scalars
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            if isinstance(val_acc, torch.Tensor):
                val_acc = val_acc.item()
            if isinstance(val_f1, torch.Tensor):
                val_f1 = val_f1.item()
            if isinstance(val_precision, torch.Tensor):
                val_precision = val_precision.item()
            if isinstance(val_recall, torch.Tensor):
                val_recall = val_recall.item()
            
            # Store results
            fold_results = {
                "fold": fold + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall
            }
            
            all_fold_results.append(fold_results)
            
            # Keep track of best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = lightning_module
            
            # Print fold results in a nice table
            print("\n" + "="*80)
            print(f"FOLD {fold+1} RESULTS")
            print("="*80)
            
            print(f"{'Metric':<20} {'Value':<10}")
            print("-"*30)
            
            for metric_name, value in fold_results.items():
                if metric_name != "fold":
                    print(f"{metric_name:<20} {value:.2f}")
            
            print("="*80 + "\n")
            
            # Log final metrics for this fold to WandB
            if self.general_config.use_wandb:
                wandb.log({
                    f"fold_{fold+1}_final_val_acc": val_acc,
                    f"fold_{fold+1}_final_val_loss": val_loss,
                    f"fold_{fold+1}_final_val_f1": val_f1,
                    f"fold_{fold+1}_final_val_precision": val_precision,
                    f"fold_{fold+1}_final_val_recall": val_recall
                })
            
            # Run inference evaluation for this fold if available
            try:
                # Check if inference dataset exists and has samples
                has_inference_data = False
                try:
                    inference_loader = self.data_module.predict_dataloader()
                    has_inference_data = len(inference_loader.dataset) > 0
                except (ValueError, AttributeError, TypeError) as e:
                    print(f"No inference data available for fold {fold+1}: {str(e)}")
                    has_inference_data = False
                
                if has_inference_data:
                    print("\n" + "="*80)
                    print(f"RUNNING INFERENCE EVALUATION FOR FOLD {fold+1}")
                    print("="*80 + "\n")
                    
                    inference_results = self._run_inference(trainer, lightning_module)
                    
                    # Print inference results in a nice table
                    if inference_results:
                        print("\n" + "="*80)
                        print(f"FOLD {fold+1} INFERENCE RESULTS")
                        print("="*80)
                        
                        print(f"{'Metric':<20} {'Value':<10}")
                        print("-"*30)
                        
                        # Print inference metrics in a specific order
                        inference_metric_order = [
                            'inference_acc', 'inference_precision', 'inference_recall', 'inference_f1'
                        ]
                        
                        for metric_name in inference_metric_order:
                            if metric_name in inference_results:
                                value = inference_results[metric_name]
                                if isinstance(value, torch.Tensor):
                                    value = value.item()
                                print(f"{metric_name:<20} {value:.2f}")
                        
                        print("="*80 + "\n")
                    
                    # Add inference results to fold results
                    for key, value in inference_results.items():
                        fold_results[key] = value
                    
                    # Log inference results to wandb
                    if self.general_config.use_wandb:
                        for key, value in inference_results.items():
                            wandb.log({f"fold_{fold+1}_{key}": value})
                else:
                    print(f"No inference data available for fold {fold+1}, skipping inference evaluation.")
            except Exception as e:
                print(f"Error during inference for fold {fold+1}: {str(e)}")
        
        # Calculate average metrics across all folds
        avg_metrics = self._calculate_average_metrics(all_fold_results)
        
        # Print average metrics in a nice table
        print("\n" + "="*80)
        print(f"AVERAGE METRICS ACROSS {self.general_config.k_folds} FOLDS")
        print("="*80)
        
        print(f"{'Metric':<20} {'Value':<10}")
        print("-"*30)
        
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name:<20} {value:.2f}")
        
        print("="*80 + "\n")
        
        # Log average metrics to wandb
        if self.general_config.use_wandb:
            for key, value in avg_metrics.items():
                # Use correct format that matches the keys in avg_metrics
                if key.startswith("average_"):
                    wandb.log({key: value})
                elif key.startswith("std_"):
                    wandb.log({key: value})
        
        # Save best model if enabled
        if self.general_config.save_model and best_model is not None:
            model_path = Path("saved_models")
            model_path.mkdir(parents=True, exist_ok=True)
            model_name = f"{self.general_config.model_type}_kfold_best.pt"
            torch.save(best_model.model.state_dict(), str(model_path / model_name))
        
        # Create the results dictionary
        results = {
            "fold_results": all_fold_results,
            "avg_metrics": avg_metrics
        }
        
        # After creating the results dictionary
        # Print and log the average metrics across all folds
        for key, value in results['avg_metrics'].items():
            if isinstance(value, torch.Tensor):
                results['avg_metrics'][key] = value.item()
            
        # Print average metrics as a table
        metric_order = ['average_val_loss', 'average_val_acc', 'average_val_f1', 'average_val_precision', 'average_val_recall']
        
        # Add inference metrics to the order if they exist
        inference_metrics = [k for k in results['avg_metrics'].keys() if k.startswith('average_inference_')]
        metric_order.extend(sorted(inference_metrics))
        
        # Print the table header
        print("\n" + "="*80)
        print("AVERAGE METRICS ACROSS ALL FOLDS")
        print("="*80)
        print(f"{'Metric':<30} {'Value':<10} {'Std':<10}")
        print("-"*50)
        
        # Print the metrics in order
        for key in metric_order:
            if key in results['avg_metrics']:
                value = results['avg_metrics'][key]
                std_key = key.replace('average_', 'std_')
                std = results['avg_metrics'].get(std_key, 0.0)
                
                # Format the metric name for better readability
                pretty_name = key.replace('average_', '').replace('_', ' ').title()
                print(f"{pretty_name:<30} {value:.4f} {std:.4f}")
        
        print("="*80 + "\n")
        
        # Log average metrics to WandB
        if self.wandb_logger:
            for key, value in results['avg_metrics'].items():
                # Log all average metrics to the summary
                self.wandb_logger.experiment.summary[f"final_{key}"] = value
        
        # Log best validation metrics from the best fold
        if best_model is not None and hasattr(best_model, 'best_val_accuracy') and self.wandb_logger:
            self.wandb_logger.experiment.summary['final_best_fold_val_acc'] = best_model.best_val_accuracy
            
            # Log other best validation metrics if available
            if hasattr(best_model, 'best_val_f1'):
                self.wandb_logger.experiment.summary['final_best_fold_val_f1'] = best_model.best_val_f1
            
            if hasattr(best_model, 'best_val_precision'):
                self.wandb_logger.experiment.summary['final_best_fold_val_precision'] = best_model.best_val_precision
                
            if hasattr(best_model, 'best_val_recall'):
                self.wandb_logger.experiment.summary['final_best_fold_val_recall'] = best_model.best_val_recall
        
        # Return results dict
        return results
    
    def _calculate_average_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average metrics across all folds.
        
        Args:
            fold_results: List of dictionaries containing results for each fold
            
        Returns:
            Dictionary of average metrics
        """
        # Get all metric keys except 'fold'
        metric_keys = set()
        for result in fold_results:
            metric_keys.update(result.keys())
        
        if 'fold' in metric_keys:
            metric_keys.remove('fold')
        
        # Calculate average for each metric
        avg_metrics = {}
        for key in metric_keys:
            # Get values for this metric across all folds
            values = []
            for result in fold_results:
                if key in result:
                    value = result[key]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    values.append(value)
            
            # Calculate average and standard deviation
            if values:
                avg_value = sum(values) / len(values)
                std_value = np.std(values) if len(values) > 1 else 0.0
                
                # Store average and standard deviation - use 'average_' prefix instead of 'avg_'
                avg_metrics[f"average_{key}"] = avg_value
                avg_metrics[f"std_{key}"] = std_value
        
        return avg_metrics