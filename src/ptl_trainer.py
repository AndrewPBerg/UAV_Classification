import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import wandb
from pathlib import Path
from icecream import ic
from datetime import datetime

from lightning_module import AudioClassifier
from datamodule import AudioDataModule
from configs.configs_demo import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig, wandb_config_dict
from helper.util import wandb_login


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
        print(f"GPU available: {self.gpu_available}")
        
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
                group=self.wandb_config.group if hasattr(self.wandb_config, 'group') and self.wandb_config.group else None
            )
        
        # Set device
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
    
        
        # Set random seeds for reproducibility
        torch.manual_seed(general_config.seed)
        torch.cuda.manual_seed(general_config.seed)
        np.random.seed(general_config.seed)
    
    def _get_callbacks(self) -> List[pl.Callback]:
        """
        Get callbacks for training.
        
        Returns:
            List of PyTorch Lightning callbacks
        """
        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints") / datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Model checkpoint callback
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True
            ),
            
            # Early stopping callback
            EarlyStopping(
                monitor="val_loss",
                patience=self.general_config.patience,
                mode="min",
                verbose=True
            ),
            
            # Learning rate monitor
            LearningRateMonitor(logging_interval="epoch")
        ]
        
        return callbacks
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model using PyTorch Lightning.
        
        Returns:
            Dictionary of training results
        """
        # Create model
        model, feature_extractor = self.model_factory(self.device)
        
        # Ensure data module is set up
        if not hasattr(self.data_module, 'num_classes') or self.data_module.num_classes is None:
            self.data_module.setup()
            
        # Validate number of classes
        if self.data_module.num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {self.data_module.num_classes}. Must be positive.")
            
        # Log number of classes for debugging
        print(f"Number of classes in data module: {self.data_module.num_classes}")
        
        # Create Lightning module
        lightning_module = AudioClassifier(
            model=model,
            general_config=self.general_config,
            peft_config=self.peft_config,
            num_classes=self.data_module.num_classes
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.general_config.epochs,
            accelerator="gpu" if self.gpu_available else "cpu",
            devices=1,  # Always use a single device
            callbacks=self._get_callbacks(),
            logger=self.wandb_logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=self.general_config.accumulation_steps,
            deterministic=True,
            precision="16-mixed" if self.gpu_available else "32"
        )
        
        # Train model
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
        
        # Test model
        try:
            test_results = trainer.test(
                model=lightning_module,
                datamodule=self.data_module
            )
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            # Check if this is a CUDA assertion error related to class labels
            if "nll_loss_forward" in str(e) and "Assertion `t >= 0 && t < n_classes`" in str(e):
                print("\nERROR: Invalid class labels detected in your test dataset.")
                print(f"The model expects labels in range [0, {self.data_module.num_classes-1}], but found labels outside this range.")
                print("Please check your dataset preprocessing and ensure all labels are correctly mapped.")
                return {"error": "Invalid class labels in test dataset"}
            raise  # Re-raise the exception if it's not the specific error we're handling
        
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
                print("Running final inference evaluation...")
                inference_results = self._run_inference(trainer, lightning_module)
                
                # Log inference results
                if self.wandb_logger and inference_results:
                    for key, value in inference_results.items():
                        self.wandb_logger.log_metrics({key: value})
                    
                # Print inference results
                print("\nInference Results:")
                print("-" * 40)
                for key, value in inference_results.items():
                    if isinstance(value, (int, float)):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
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
        
        # Ensure the return type is Dict[str, Any]
        results_dict: Dict[str, Any] = {} if not test_results else dict(test_results[0])
        
        # Add any additional metrics you want to track
        if hasattr(lightning_module, 'best_val_accuracy'):
            results_dict['best_val_accuracy'] = lightning_module.best_val_accuracy
        
        # Add inference results to the overall results
        if inference_results:
            for key, value in inference_results.items():
                results_dict[key] = value
        
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
            # Run prediction
            print("Starting inference with predict...")
            predictions = trainer.predict(
                model=model,
                datamodule=self.data_module
            )
            
            if not predictions:
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
                print("Metrics not found in model attributes, calculating manually...")
            
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
            
            # Calculate metrics
            from torchmetrics.functional import (
                multiclass_accuracy,
                multiclass_precision,
                multiclass_recall,
                multiclass_f1_score
            )
            
            num_classes = self.data_module.num_classes
            
            # Calculate metrics
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
            return {}
        
        # Create confusion matrix
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
        
        # Create confusion matrix
        from torchmetrics.functional import confusion_matrix
        conf_mat = confusion_matrix(
            all_preds, all_targets, num_classes=self.data_module.num_classes
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
                class_names = [idx_to_class[i] for i in range(self.data_module.num_classes)]
            except (AttributeError, KeyError):
                class_names = [str(i) for i in range(self.data_module.num_classes)]
            
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
        
        return metrics
    
    def k_fold_cross_validation(self) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation using PyTorch Lightning.
        
        Returns:
            Dictionary of aggregated results across all folds
        """
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
                group=self.wandb_config.group if hasattr(self.wandb_config, 'group') and self.wandb_config.group else None
            )
        
        # Train on each fold
        for fold in range(self.general_config.k_folds):
            ic(f"Training fold {fold+1}/{self.general_config.k_folds}")
            
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
                LearningRateMonitor(logging_interval="epoch")
            ]
            
            # Create trainer for this fold
            trainer = pl.Trainer(
                max_epochs=self.general_config.epochs,
                accelerator="gpu" if self.gpu_available else "cpu",
                devices=1,  # Always use a single device
                callbacks=fold_callbacks,
                logger=self.wandb_logger,
                gradient_clip_val=1.0,
                accumulate_grad_batches=self.general_config.accumulation_steps,
                deterministic=True,
                precision="16-mixed" if self.gpu_available else "32"
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
                    print(f"Running inference evaluation for fold {fold+1}...")
                    inference_results = self._run_inference(trainer, lightning_module)
                    
                    # Add inference results to fold results
                    for key, value in inference_results.items():
                        fold_results[key] = value
                    
                    # Log inference results for this fold to WandB
                    if self.general_config.use_wandb:
                        fold_inference_metrics = {
                            f"fold_{fold+1}_inference_acc": inference_results.get("inference_acc", 0.0),
                            f"fold_{fold+1}_inference_f1": inference_results.get("inference_f1", 0.0),
                            f"fold_{fold+1}_inference_precision": inference_results.get("inference_precision", 0.0),
                            f"fold_{fold+1}_inference_recall": inference_results.get("inference_recall", 0.0)
                        }
                        wandb.log(fold_inference_metrics)
                    
                    # Print inference results
                    print(f"\nInference Results for Fold {fold+1}:")
                    print("-" * 40)
                    for key, value in inference_results.items():
                        if isinstance(value, (int, float)):
                            print(f"{key}: {value:.4f}")
                        else:
                            print(f"{key}: {value}")
                else:
                    print(f"No inference data available for fold {fold+1}, skipping inference evaluation.")
            except Exception as e:
                print(f"Error during inference for fold {fold+1}: {str(e)}")
                print("Continuing with next fold...")
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(all_fold_results)
        
        # Calculate average inference metrics if available
        inference_metrics_keys = ["inference_acc", "inference_f1", "inference_precision", "inference_recall"]
        avg_inference_metrics = {}
        
        for key in inference_metrics_keys:
            values = [fold_result.get(key, None) for fold_result in all_fold_results]
            values = [v for v in values if v is not None]  # Filter out None values
            
            if values:
                avg_value = sum(values) / len(values)
                std_value = np.std(values) if len(values) > 1 else 0.0
                avg_inference_metrics[f"average_{key}"] = avg_value
                avg_inference_metrics[f"std_{key}"] = std_value
        
        # Add average inference metrics to overall average metrics
        avg_metrics.update(avg_inference_metrics)
        
        # Log average metrics to WandB
        if self.general_config.use_wandb:
            # Log average metrics
            wandb.log(avg_metrics)
            
            # Create a table of fold results
            fold_table = wandb.Table(columns=["Fold", "Val Loss", "Val Acc", "Val F1", "Val Precision", "Val Recall"])
            for result in all_fold_results:
                fold_table.add_data(
                    result["fold"],
                    result["val_loss"],
                    result["val_acc"],
                    result["val_f1"],
                    result["val_precision"],
                    result["val_recall"]
                )
            
            wandb.log({"fold_results": fold_table})
        
        # Save best model if enabled
        if self.general_config.save_model and best_model is not None:
            model_path = Path("saved_models")
            model_path.mkdir(parents=True, exist_ok=True)
            model_name = f"{self.general_config.model_type}_kfold_best.pt"
            torch.save(best_model.model.state_dict(), str(model_path / model_name))
        
        return {
            "fold_results": all_fold_results,
            "avg_metrics": avg_metrics,
            "best_model": best_model
        }
    
    def _calculate_average_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average metrics across all folds.
        
        Args:
            fold_results: List of dictionaries containing results for each fold
            
        Returns:
            Dictionary of average metrics
        """
        avg_metrics = {}
        std_metrics = {}
        
        # Skip the 'fold' key
        metric_keys = [key for key in fold_results[0].keys() if key != 'fold']
        
        # Calculate average and standard deviation for each metric
        for key in metric_keys:
            values = [result[key] for result in fold_results]
            avg_metrics[f"average_{key}"] = np.mean(values)
            std_metrics[f"std_{key}"] = np.std(values)
        
        # Combine average and standard deviation metrics
        combined_metrics = {**avg_metrics, **std_metrics}
        
        return combined_metrics 