import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config.model_config import TrainingConfig, ModelConfig
from models.model_factory import ModelFactory
from data.data_module import UAVDataModule
import wandb
from helper.util import wandb_login
from pytorch_lightning.loggers import WandbLogger

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_module: UAVDataModule
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_module = data_module
        
        # Initialize wandb
        wandb_login()
        
        # Create WandB logger
        self.wandb_logger = WandbLogger(
            project=model_config.project_name,
            name=model_config.model_name or f"{model_config.model_size}_uav",
            log_model=True
        )
        
        # Update model config with number of classes
        if model_config.num_classes is None:
            self.model_config.num_classes = data_module.num_classes
        
        # Create model
        self.model = ModelFactory.create_model(self.model_config)
        
    def _get_callbacks(self):
        """Creates training callbacks"""
        return [
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.training_config.early_stopping_patience,
                mode="min"
            )
        ]
    
    def train(self):
        """Trains the model"""
        trainer = pl.Trainer(
            max_epochs=self.training_config.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=self._get_callbacks(),
            logger=self.wandb_logger,
            deterministic=True,
            gradient_clip_val=self.training_config.gradient_clip_val
        )
        
        trainer.fit(
            model=self.model,
            datamodule=self.data_module
        )
        
        # Test the model
        trainer.test(
            model=self.model,
            datamodule=self.data_module
        )
