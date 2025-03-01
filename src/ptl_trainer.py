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
    PyTorch Lightning trainer that handles both regular training and k-fold cross-validation.
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
            data_module: AudioDataModule instance
            model_factory: Function that creates a model instance
        """
        self.general_config = general_config
        self.feature_extraction_config = feature_extraction_config
        self.peft_config = peft_config
        self.wandb_config = wandb_config
        self.sweep_config = sweep_config
        self.data_module = data_module
        self.model_factory = model_factory
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(general_config.seed)
        torch.cuda.manual_seed(general_config.seed)
        np.random.seed(general_config.seed)
        
        # Initialize WandB if enabled
        self.wandb_logger = None
        if general_config.use_wandb:
            wandb_login()
            ic("successfully logged in to wandb")
            self.wandb_logger = WandbLogger(
                project=wandb_config.project,
                name=wandb_config.name,
                tags=wandb_config.tags if wandb_config.tags else [],
                notes=wandb_config.notes,
                log_model=True,
                config=wandb_config_dict(general_config, feature_extraction_config, peft_config, wandb_config)
            )
    
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
    
    def train(self) -> Tuple[AudioClassifier, Dict[str, Any]]:
        """
        Train the model using PyTorch Lightning.
        
        Returns:
            Tuple of (trained model, training results)
        """
        # Create model
        model, feature_extractor = self.model_factory(self.device)
        
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
            accelerator="auto",
            devices=1,  # Always use 1 device (either 1 GPU or 1 CPU)
            callbacks=self._get_callbacks(),
            logger=self.wandb_logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=self.general_config.accumulation_steps,
            deterministic=True,
            precision="16-mixed" if torch.cuda.is_available() else "32"
        )
        
        # Train model
        trainer.fit(
            model=lightning_module,
            datamodule=self.data_module
        )
        
        # Test model
        test_results = trainer.test(
            model=lightning_module,
            datamodule=self.data_module
        )
        
        # Save model if enabled
        if self.general_config.save_model:
            model_path = Path("saved_models")
            model_path.mkdir(parents=True, exist_ok=True)
            model_name = f"{self.general_config.model_type}_classifier.pt"
            trainer.save_checkpoint(str(model_path / model_name))
        
        # Ensure the return type is Dict[str, Any]
        results_dict: Dict[str, Any] = {} if not test_results else dict(test_results[0])
        
        return lightning_module, results_dict
    
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
            
            # Create fold-specific WandB logger if enabled
            fold_logger = None
            if self.general_config.use_wandb:
                fold_logger = WandbLogger(
                    project=self.wandb_config.project,
                    name=f"{self.wandb_config.name}_fold_{fold+1}",
                    tags=(self.wandb_config.tags + [f"fold_{fold+1}"]) if self.wandb_config.tags else [f"fold_{fold+1}"],
                    notes=self.wandb_config.notes,
                    log_model=False
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
                accelerator="auto",
                devices=1,  # Always use 1 device (either 1 GPU or 1 CPU)
                callbacks=fold_callbacks,
                logger=fold_logger,
                gradient_clip_val=1.0,
                accumulate_grad_batches=self.general_config.accumulation_steps,
                deterministic=True,
                precision="16-mixed" if torch.cuda.is_available() else "32"
            )
            
            # Train on this fold
            trainer.fit(
                model=lightning_module,
                train_dataloaders=fold_train_loader,
                val_dataloaders=fold_val_loader
            )
            
            # Get validation loss
            val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
            
            # Store results
            fold_results = {
                "fold": fold + 1,
                "val_loss": val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss,
                "val_acc": trainer.callback_metrics.get("val_acc", 0.0),
                "val_f1": trainer.callback_metrics.get("val_f1", 0.0),
                "val_precision": trainer.callback_metrics.get("val_precision", 0.0),
                "val_recall": trainer.callback_metrics.get("val_recall", 0.0)
            }
            
            all_fold_results.append(fold_results)
            
            # Keep track of best model
            if fold_results["val_loss"] < best_val_loss:
                best_val_loss = fold_results["val_loss"]
                best_model = lightning_module
            
            # Close WandB run for this fold
            if fold_logger and fold_logger.experiment:
                wandb.finish()
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(all_fold_results)
        
        # Log average metrics to WandB
        if self.general_config.use_wandb:
            wandb_login()
            wandb.init(
                project=self.wandb_config.project,
                name=f"{self.wandb_config.name}_kfold_summary",
                tags=(self.wandb_config.tags + ["kfold_summary"]) if self.wandb_config.tags else ["kfold_summary"],
                notes=self.wandb_config.notes
            )
            
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
            wandb.finish()
        
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