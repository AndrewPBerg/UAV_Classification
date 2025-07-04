import os
from configs.augmentation_config import AugmentationConfig
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import wandb
from pathlib import Path
from icecream import ic
from datetime import datetime

from .lightning_module import AudioClassifier
from configs import GeneralConfig, FeatureExtractionConfig, WandbConfig, SweepConfig, wandb_config_dict
from configs.dataset_config import DatasetConfig
from configs.optim_config import OptimizerConfig
from configs.peft_scheduling_config import PEFTSchedulingConfig
from configs.loss_config import LossConfig
from .util import wandb_login
import time
from configs.ensemble_config import EnsembleConfig


class CustomProgressBar(TQDMProgressBar):
    """Custom progress bar for training that shows formatted metrics."""
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
        for metric in ['train_loss', 'train_acc', 'train_f1']:
            if metric in items:
                ordered_items[metric] = items[metric]
        
        # Then show validation metrics
        for metric in ['val_loss', 'val_acc', 'val_f1']:
            if metric in items:
                ordered_items[metric] = items[metric]
        
        # Add any remaining metrics
        for key, val in items.items():
            if key not in ordered_items:
                ordered_items[key] = val
        
        return ordered_items


class CustomFoldProgressBar(TQDMProgressBar):
    """Custom progress bar for k-fold training that shows formatted metrics."""
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
        for metric in ['train_loss', 'train_acc', 'train_f1']:
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


class PTLTrainer:
    """
    PyTorch Lightning trainer that handles training using a single GPU.
    This replaces the functionality in engine.py and fold_engine.py.
    """
    def __init__(
        self,
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        dataset_config: DatasetConfig,
        peft_config: Any,
        wandb_config: WandbConfig,
        sweep_config: SweepConfig,
        data_module: pl.LightningDataModule,
        model_factory: Callable,
        augmentation_config: AugmentationConfig,
        optimizer_config: OptimizerConfig,
        peft_scheduling_config: Optional[PEFTSchedulingConfig] = None,
        loss_config: Optional[LossConfig] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PTLTrainer.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            dataset_config: Dataset configuration
            peft_config: PEFT configuration
            wandb_config: WandB configuration
            sweep_config: Sweep configuration
            data_module: Lightning data module (UAVDataModule or ESC50DataModule)
            model_factory: Model factory function
            augmentation_config: Augmentation configuration
            optimizer_config: Optimizer configuration
            peft_scheduling_config: PEFT scheduling configuration (optional)
            loss_config: Loss configuration (optional)
            config_dict: Configuration dictionary (optional)
        """
        self.general_config = general_config
        self.feature_extraction_config = feature_extraction_config
        self.dataset_config = dataset_config
        self.peft_config = peft_config
        self.wandb_config = wandb_config
        self.sweep_config = sweep_config
        self.data_module = data_module
        self.model_factory = model_factory
        self.augmentation_config = augmentation_config
        self.optimizer_config = optimizer_config
        self.peft_scheduling_config = peft_scheduling_config
        self.loss_config = loss_config
        self.config_dict = config_dict
        
        # ------------------------------------------------------------------
        # Ensemble configuration (optional – defaults keep behaviour identical)
        # ------------------------------------------------------------------
        ensemble_cfg_raw: Dict[str, Any] = {}
        if self.config_dict is not None and "ensemble" in self.config_dict:
            ensemble_cfg_raw = self.config_dict["ensemble"] or {}

        try:
            self.ensemble_config = EnsembleConfig(**ensemble_cfg_raw)
        except Exception as e:
            print("[Ensemble] Invalid ensemble configuration – falling back to defaults. Error:", e)
            self.ensemble_config = EnsembleConfig()

        # Informative banner so the user immediately sees if ensembling is active
        if self.ensemble_config.is_active():
            print("="*80)
            print("🧩 ENSEMBLE MODE ENABLED")
            print(f"→ Ensemble size (M): {self.ensemble_config.M}")
            print(f"→ Same mini-batch across models: {self.ensemble_config.same_minibatch}")
            print("="*80)
        else:
            # Mild verbosity to confirm single-model path
            print("[Ensemble] Single-model mode (M=1) – standard pipeline.")
        
        # GPU configuration
        self.gpu_available = torch.cuda.is_available()
        self.num_gpus = general_config.num_gpus if general_config.distributed_training else 1
        self.distributed_strategy = general_config.strategy if general_config.distributed_training else None
        
        # Validate GPU configuration
        if general_config.distributed_training:
            if not self.gpu_available:
                raise ValueError("Distributed training requires CUDA to be available")
            available_gpus = torch.cuda.device_count()
            if self.num_gpus > available_gpus:
                print(f"Warning: Requested {self.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
                self.num_gpus = available_gpus
        
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
                config=wandb_config_dict(
                    self.general_config, 
                    self.feature_extraction_config, 
                    self.dataset_config,
                    self.peft_config, 
                    self.wandb_config, 
                    self.augmentation_config,
                    self.optimizer_config
                ),
                reinit=True  # Force reinitialize a new wandb run
            )
        
        # Set device
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
    
        
        # Set random seeds for reproducibility
        torch.manual_seed(general_config.seed)
        torch.cuda.manual_seed(general_config.seed)
        np.random.seed(general_config.seed)
    
    def _get_current_lr(self) -> float:
        """Get the current learning rate from optimizer config."""
        if self.optimizer_config.optimizer_type == "adamw":
            return self.optimizer_config.adamw.lr
        elif self.optimizer_config.optimizer_type == "adam":
            return self.optimizer_config.adam.lr
        elif self.optimizer_config.optimizer_type == "adamspd":
            return self.optimizer_config.adamspd.lr
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_config.optimizer_type}")
    
    def _get_callbacks(self) -> List[pl.Callback]:
        """
        Get PyTorch Lightning callbacks.
        
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Add custom progress bar
        callbacks.append(CustomProgressBar(refresh_rate=10))
        
        # Add learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        
        # Determine if validation metrics will be available
        # This should match the logic in lightning_module._init_metrics()
        will_have_val_metrics = (
            (self.general_config.val_size > 0) or 
            (hasattr(self.general_config, 'use_kfold') and self.general_config.use_kfold) or
            # Also check if datamodule has validation data (for ESC datasets)
            (hasattr(self.data_module, 'val_dataset') and self.data_module.val_dataset is not None)
        )
        
        # Choose appropriate monitor metric and mode
        if will_have_val_metrics and self.general_config.monitor.startswith('val_'):
            monitor_metric = self.general_config.monitor
            monitor_mode = self.general_config.mode
        else:
            # Fallback to training metrics if validation metrics aren't available
            if self.general_config.monitor == 'val_acc':
                monitor_metric = 'train_acc'
                monitor_mode = 'max'
            elif self.general_config.monitor == 'val_loss':
                monitor_metric = 'train_loss'
                monitor_mode = 'min'
            elif self.general_config.monitor == 'val_f1':
                monitor_metric = 'train_f1'
                monitor_mode = 'max'
            else:
                # Use the configured monitor if it's already a training metric
                monitor_metric = self.general_config.monitor
                monitor_mode = self.general_config.mode
        
        print(f"Using monitor metric: {monitor_metric} (mode: {monitor_mode})")
        
        # Add model checkpoint callback
        if self.general_config.checkpointing:
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=self.general_config.save_top_k,
                save_last=True,
                filename='{epoch}-{' + monitor_metric.replace('_', '-') + ':.2f}',
                auto_insert_metric_name=False
            )
            callbacks.append(checkpoint_callback)
        
        # Add early stopping callback
        if self.general_config.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=self.general_config.patience,
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        return callbacks
    
    def _get_num_classes(self) -> int:
        """
        Get the number of classes from the data module.
        
        Returns:
            Number of classes
        """
        if hasattr(self.data_module, 'num_classes'):
            return self.data_module.num_classes
        else:
            # Fallback to dataset config
            return self.dataset_config.get_num_classes()
    
    def _get_strategy(self):
        """
        Get the appropriate strategy for distributed training.
        
        Returns:
            Strategy configuration for PyTorch Lightning trainer
        """
        if not self.general_config.distributed_training:
            return "auto"
        
        # Check if PEFT scheduling is enabled
        peft_scheduling_enabled = (
            self.peft_scheduling_config is not None and 
            self.peft_scheduling_config.enabled
        )
        
        # If PEFT scheduling is enabled, we need to handle unused parameters
        if peft_scheduling_enabled:
            print("PEFT scheduling detected - enabling find_unused_parameters for DDP")
            if self.general_config.strategy == "ddp":
                return DDPStrategy(find_unused_parameters=True)
            elif self.general_config.strategy == "ddp_spawn":
                return DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
            else:
                # For other strategies, use the string and hope it works
                return self.distributed_strategy
        else:
            # No PEFT scheduling, use the configured strategy
            return self.distributed_strategy
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model using PyTorch Lightning.
        
        Returns:
            Dictionary of test results
        """
        # ------------------------------------------------------------------
        # If ensemble mode is ACTIVE (M > 1) – train each member separately
        # ------------------------------------------------------------------
        if self.ensemble_config.is_active():
            self.trained_models = []  # store trained LightningModules for later inference
            member_results: List[Dict[str, Any]] = []

            for m_idx in range(self.ensemble_config.M):
                print("\n" + "#"*90)
                print(f"TRAINING ENSEMBLE MEMBER {m_idx+1}/{self.ensemble_config.M}")
                print("#"*90 + "\n")

                # ------------------------------------------------------
                # Fresh callbacks & logger so state doesn't leak between runs
                # ------------------------------------------------------
                callbacks = self._get_callbacks()

                # Redirect checkpoints to member-specific folder if using the default dirpath
                for cb in callbacks:
                    if isinstance(cb, pl.callbacks.ModelCheckpoint):
                        if cb.dirpath is None:
                            cb.dirpath = os.path.join("checkpoints", f"member_{m_idx+1}")
                        else:
                            cb.dirpath = os.path.join(cb.dirpath, f"member_{m_idx+1}")

                # Create a new wandb run per member if enabled
                logger = None
                if self.general_config.use_wandb:
                    wandb_login()
                    run_name = f"{self.wandb_config.name}-m{m_idx+1}" if self.wandb_config.name else f"ensemble-m{m_idx+1}"
                    logger = WandbLogger(
                        project=self.wandb_config.project,
                        name=run_name,
                        tags=self.wandb_config.tags if self.wandb_config.tags else [],
                        notes=self.wandb_config.notes,
                        log_model=False,
                        save_dir=self.wandb_config.dir if self.wandb_config.dir else "wandb",
                        reinit=True,
                    )

                # ------------------------------------------------------
                # Diversify seed so models don't train identically unless user disables it
                # ------------------------------------------------------
                current_seed = self.general_config.seed + m_idx
                torch.manual_seed(current_seed)
                torch.cuda.manual_seed(current_seed)
                np.random.seed(current_seed)

                trainer = pl.Trainer(
                    max_epochs=self.general_config.epochs,
                    accelerator="gpu" if self.gpu_available else "cpu",
                    devices=self.num_gpus if self.gpu_available else "auto",
                    strategy=self._get_strategy(),
                    callbacks=callbacks,
                    logger=logger,
                    deterministic=False,
                    precision=32,
                )

                # Ensure data module is set up once (outside loop is already ok)
                try:
                    self.data_module.setup(stage="fit")
                except Exception:
                    pass

                num_classes = self._get_num_classes()

                model, _ = self.model_factory(self.device)

                lightning_module = AudioClassifier(
                    model=model,
                    general_config=self.general_config,
                    peft_config=self.peft_config,
                    num_classes=num_classes,
                    optimizer_config=self.optimizer_config,
                    peft_scheduling_config=self.peft_scheduling_config,
                    loss_config=self.loss_config,
                    config_dict=self.config_dict,
                )

                trainer.fit(model=lightning_module, datamodule=self.data_module)

                # Store trained model (move to CPU to free GPU memory)
                lightning_module.cpu()
                self.trained_models.append(lightning_module)

                # Optionally collect metrics from trainer.callback_metrics
                member_metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
                member_results.append(member_metrics)

                # Clean up GPU memory before next member
                torch.cuda.empty_cache()

            # After ensemble training, we can trigger inference evaluation if desired
            # ------------------------------------------------------------------
            # Ensemble evaluation on VALIDATION set when no explicit test split
            # ------------------------------------------------------------------
            val_results: Dict[str, Any] = {}
            try:
                val_loader = self.data_module.val_dataloader()
                if val_loader is not None and len(val_loader) > 0:
                    print("\n" + "="*80)
                    print("EVALUATING ENSEMBLE ON VALIDATION SET (softmax-averaged)")
                    print("="*80 + "\n")
                    val_results = self._ensemble_evaluate_on_loader(val_loader, self.trained_models)
            except Exception as e:
                print(f"[Ensemble] Validation evaluation skipped: {e}")

            # Use first trained model's trainer instance for evaluation convenience
            base_trainer = trainer  # last trainer created
            base_module = self.trained_models[0]

            inference_results = {}
            if self.data_module is not None and self.general_config.inference_size > 0:
                models_for_inf = getattr(self, 'trained_models', [])
                inference_results = self._run_ensemble_inference(base_trainer, base_module, models_for_inf or None)

            # Aggregate output
            return {"members": member_results, **val_results, **inference_results}

        # ------------------------------------------------------------------
        # SINGLE MODEL PATH (original behaviour)
        # ------------------------------------------------------------------
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.general_config.epochs,
            accelerator="gpu" if self.gpu_available else "cpu",
            devices=self.num_gpus if self.gpu_available else "auto",
            strategy=self._get_strategy(),
            callbacks=callbacks,
            logger=self.wandb_logger,
            deterministic=False,
            precision=32  # Changed to 32-bit precision to avoid AMP issues
        )
        ic("trainer created")
        
        # Ensure data module is set up - call setup directly
        try:
            self.data_module.setup(stage="fit")
        except Exception as e:
            print(f"Warning: Error during data module setup: {str(e)}")
            print("This may be normal if the data module is already set up.")
            
        # Validate number of classes
        num_classes = self._get_num_classes()
        if num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {num_classes}. Must be positive.")
            
        # Log number of classes for debugging
        print(f"Number of classes in data module: {num_classes}")
        
        # Create model
        model, feature_extractor = self.model_factory(self.device)
        
        # Create lightning module
        lightning_module = AudioClassifier(
            model=model,
            general_config=self.general_config,
            peft_config=self.peft_config,
            num_classes=num_classes,
            optimizer_config=self.optimizer_config,
            peft_scheduling_config=self.peft_scheduling_config,
            loss_config=self.loss_config,
            config_dict=self.config_dict
        )
        
        # Print training start message with clear formatting
        print("\n" + "="*80)
        print(f"STARTING TRAINING: {self.general_config.model_type.upper()} MODEL")
        print(f"Epochs: {self.general_config.epochs} | Batch Size: {self.general_config.batch_size} | LR: {self._get_current_lr()}")
        
        # Print loss configuration verification
        if self.loss_config:
            print(f"Loss Config - Type: {self.loss_config.type}, Label Smoothing: {self.loss_config.label_smoothing}")
        else:
            print("⚠️  No loss config provided - using defaults")
        
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
            self.wandb_logger.log_metrics({
                "total_train_time": formatted_end_time
            })
        
        # Run test evaluation after training

        if self.general_config.test_size > 0:
            
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
                
        else:
            print("No test split available, skipping test evaluation.")
            test_results = []
        
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
                # ------------------------------------------------------------------
                # Choose between standard and ensemble inference
                # ------------------------------------------------------------------
                if self.ensemble_config.is_active():
                    models_for_inf = getattr(self, 'trained_models', [])
                    inference_results = self._run_ensemble_inference(trainer, lightning_module, models_for_inf or None)
                else:
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
                        # # Also add to wandb summary
                        # self.wandb_logger.experiment.summary[f"final_{key}"] = value
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
        test_acc_str = f"{test_acc:.2f}" if test_acc is not None else "N/A"
        inf_acc_str = f"{inf_acc:.2f}" if inf_acc is not None else "N/A"
        
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
                    self.wandb_logger.log_metrics({
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
                self.wandb_logger.log_metrics({
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
                config=wandb_config_dict(self.general_config, self.feature_extraction_config, self.peft_config, self.wandb_config, self.augmentation_config, self.optimizer_config),
                group=self.wandb_config.group if hasattr(self.wandb_config, 'group') and self.wandb_config.group else None,
                reinit=True  # Force reinitialize a new wandb run
            )
        
        # Print k-fold start message with clear formatting
        print("\n" + "="*80)
        print(f"STARTING {self.general_config.k_folds}-FOLD CROSS-VALIDATION: {self.general_config.model_type.upper()} MODEL")
        print(f"Epochs: {self.general_config.epochs} | Batch Size: {self.general_config.batch_size} | LR: {self._get_current_lr()}")
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
                num_classes=self.data_module.num_classes,
                optimizer_config=self.optimizer_config,
                peft_scheduling_config=self.peft_scheduling_config,
                loss_config=self.loss_config,
                config_dict=self.config_dict
            )
            
            # Create checkpoint directory for this fold
            checkpoint_dir = Path("checkpoints") / f"fold_{fold+1}" / datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create custom progress bar for this fold
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
            
            trainer = pl.Trainer(
            max_epochs=self.general_config.epochs,
            accelerator="gpu" if self.gpu_available else "cpu",
            devices=self.num_gpus if self.gpu_available else "auto",
            strategy=self._get_strategy(),
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
                fold_data = {
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall
                }

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
                            fold_data[key] = value # add to fold_data dict
                else:
                    print(f"No inference data available for fold {fold+1}, skipping inference evaluation.")
            except Exception as e:
                print(f"Error during inference for fold {fold+1}: {str(e)}")

            if self.general_config.use_wandb:
                # Create a wandb Table for the fold
                fold_table = wandb.Table(data=[[key, val] for key, val in fold_data.items()], columns=["Metric", "Value"])
                wandb.log({f"fold_{fold+1}_metrics": fold_table})
        
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
        if self.wandb_logger:
            # Log average metrics, but only to the summary
            for key, value in avg_metrics.items():
                if key.startswith("average_") or key.startswith("std_"):
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

    # ------------------------------------------------------------------
    # Ensemble Inference
    # ------------------------------------------------------------------
    def _run_ensemble_inference(self, trainer: pl.Trainer, base_module: pl.LightningModule, models: Optional[List[pl.LightningModule]] = None) -> Dict[str, Any]:
        """Run inference using an ensemble of *M* models.

        This implementation focuses on *prediction-time* ensembling; all models share
        the same weights as the trained ``base_module`` (unless the user replaces or
        perturbs them externally).  The logic is as follows:

        1. **Model preparation** – create *M* deep copies of the trained model to
           avoid gradient sharing and place them on the correct device.
        2. **Data strategy**     – depending on ``same_minibatch`` either feed the
           *same* mini-batch to every model or consume *M* consecutive mini-batches,
           one per model.
        3. **Prediction fusion** – aggregate the *logits* by arithmetic mean and
           derive final class predictions via ``argmax``.
        4. **Metric computation** – identical to the single-model path.
        """

        if not self.ensemble_config.is_active():
            # Safety guard – should never happen because the caller checks this.
            return self._run_inference(trainer, base_module)

        from copy import deepcopy
        import torch.nn.functional as F

        device = base_module.device

        # ------------------------------------------------------------------
        # Build ensemble list
        #   • If user supplied trained models list → use their .model attribute
        #   • Otherwise fall back to cloning the base model
        # ------------------------------------------------------------------
        if models and len(models) >= self.ensemble_config.M:
            models = [lm.model.to(device).eval() for lm in models[: self.ensemble_config.M]]
        else:
            models = [deepcopy(base_module.model).to(device).eval() for _ in range(self.ensemble_config.M)]

        for m in models:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        # ------------------------------------------------------------------
        # Data loading
        # ------------------------------------------------------------------
        inference_loader = self.data_module.predict_dataloader()

        # Prepare metric accumulators
        all_preds = []
        all_targets = []

        loader_iter = iter(inference_loader)

        while True:
            try:
                if self.ensemble_config.same_minibatch:
                    batch = next(loader_iter)
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)

                    # ------------------------------------------------------
                    # Vectorised forward pass with torch.vmap for efficiency
                    # ------------------------------------------------------
                    try:
                        from torch.func import stack_module_state, functional_call, vmap

                        # Combine parameters & buffers across ensemble
                        params, buffers = stack_module_state(models)

                        # Build a meta copy of the model architecture once
                        from copy import deepcopy
                        base_for_vmap = deepcopy(models[0]).to("meta")

                        def _fmodel(p, b, xx):
                            return functional_call(base_for_vmap, (p, b), (xx,))

                        logits_vmap = vmap(_fmodel)(params, buffers, x)  # (M, B, C)
                        avg_logits = logits_vmap.mean(dim=0)

                    except Exception as e:
                        # Fallback to simple Python loop if vmap path fails
                        print("[Ensemble] vmap path failed – falling back to Python loop. Error:", e)
                        logits_list = [m(x) for m in models]
                        stacked_logits = torch.stack([
                            lg.logits if hasattr(lg, "logits") else lg for lg in logits_list
                        ])
                        avg_logits = stacked_logits.mean(dim=0)

                    preds = torch.argmax(F.softmax(avg_logits, dim=1), dim=1)

                    all_preds.append(preds.detach())
                    all_targets.append(y.detach())

                else:
                    # Different mini-batch per model – fetch up to M batches
                    batch_list = []
                    for _ in range(self.ensemble_config.M):
                        batch_list.append(next(loader_iter))
                    # Extract x & y for each model
                    logits_collection = []
                    # Using majority vote averaging logits over different data may
                    # not make sense, but we follow user's request and process
                    for model, (x_m, y_m) in zip(models, batch_list):
                        x_m = x_m.to(device)
                        y_m = y_m.to(device)
                        logits_m = model(x_m)
                        if hasattr(logits_m, "logits"):
                            logits_m = logits_m.logits
                        logits_collection.append(logits_m)
                        all_targets.append(y_m.detach())
                    # For heterogeneous mini-batches we cannot average logits
                    # directly.  Instead we derive predictions per model and
                    # concatenate.
                    preds_concat = [torch.argmax(F.softmax(lg, dim=1), dim=1) for lg in logits_collection]
                    all_preds.extend([p.detach() for p in preds_concat])

            except StopIteration:
                break

        # Flatten lists
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # ------------------------------------------------------------------
        # Compute metrics – re-use helper from _run_inference for consistency
        # ------------------------------------------------------------------
        # Use torchmetrics functional API for simplicity here
        from torchmetrics.functional.classification import (
            multiclass_accuracy,
            multiclass_precision,
            multiclass_recall,
            multiclass_f1_score,
        )

        num_classes = self.data_module.num_classes

        accuracy = multiclass_accuracy(all_preds, all_targets, num_classes=num_classes, average="weighted").item()
        precision = multiclass_precision(all_preds, all_targets, num_classes=num_classes, average="weighted").item()
        recall = multiclass_recall(all_preds, all_targets, num_classes=num_classes, average="weighted").item()
        f1 = multiclass_f1_score(all_preds, all_targets, num_classes=num_classes, average="weighted").item()

        metrics = {
            "inference_acc": accuracy,
            "inference_precision": precision,
            "inference_recall": recall,
            "inference_f1": f1,
        }

        print("\n" + "="*80)
        print("ENSEMBLE INFERENCE RESULTS (aggregated)")
        print("="*80)
        for m_name, m_val in metrics.items():
            print(f"{m_name:<20} {m_val:.4f}")
        print("="*80 + "\n")

        # Log to wandb if available
        if self.wandb_logger:
            self.wandb_logger.log_metrics(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Generic loader evaluation for trained ensemble (softmax averaging)
    # ------------------------------------------------------------------
    def _ensemble_evaluate_on_loader(self, loader, trained_modules: List[pl.LightningModule]) -> Dict[str, Any]:
        """Evaluate ensemble on a given DataLoader by averaging softmax outputs.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The dataloader (validation, test, etc.)
        trained_modules : list[LightningModule]
            List of fully–trained LightningModules representing the ensemble.
        Returns
        -------
        dict
            Weighted accuracy / precision / recall / f1 for the loader.
        """
        import torch.nn.functional as F
        device = self.device

        if not trained_modules:
            raise ValueError("No trained ensemble members supplied for evaluation.")

        # Ensure models are in eval mode and on correct device
        models = []
        for lm in trained_modules[: self.ensemble_config.M]:
            m = lm.model.to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            models.append(m)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                probs_stack = []
                for m in models:
                    logits = m(x)
                    if hasattr(logits, "logits"):
                        logits = logits.logits
                    probs = F.softmax(logits, dim=1)
                    probs_stack.append(probs)

                avg_probs = torch.stack(probs_stack).mean(dim=0)
                preds = avg_probs.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        from torchmetrics.functional.classification import (
            multiclass_accuracy,
            multiclass_precision,
            multiclass_recall,
            multiclass_f1_score,
        )

        num_classes = self.data_module.num_classes

        metrics = {
            "val_ensemble_acc": multiclass_accuracy(all_preds, all_targets, num_classes=num_classes, average="weighted").item(),
            "val_ensemble_precision": multiclass_precision(all_preds, all_targets, num_classes=num_classes, average="weighted").item(),
            "val_ensemble_recall": multiclass_recall(all_preds, all_targets, num_classes=num_classes, average="weighted").item(),
            "val_ensemble_f1": multiclass_f1_score(all_preds, all_targets, num_classes=num_classes, average="weighted").item(),
        }

        print("\n" + "="*80)
        print("ENSEMBLE VALIDATION METRICS")
        print("="*80)
        for k, v in metrics.items():
            print(f"{k:<25} {v:.4f}")
        print("="*80 + "\n")

        # Log to wandb if enabled
        if self.wandb_logger:
            self.wandb_logger.log_metrics(metrics)

        return metrics