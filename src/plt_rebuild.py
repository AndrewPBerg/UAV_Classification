import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Tuple, Dict, Any
from helper.util import AudioDataset, wandb_login
import wandb

class UAVClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        model_name: str = "resnet18_uav",
        project_name: str = "uav_classification"
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize wandb
        wandb_login()
        self.wandb_logger = pl.loggers.WandbLogger(
            project=project_name,
            name=model_name,
            log_model=True
        )
        
        # Load pretrained ResNet
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the first conv layer to accept single channel input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Log hyperparameters
        self.wandb_logger.log_hyperparams(self.hparams)
        
    def forward(self, x):
        return self.backbone(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Log learning rate
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        acc = (logits.argmax(dim=1) == y).float().mean()
        preds = logits.argmax(dim=1)
        
        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        
        # Log confusion matrix (every 5 epochs)
        if self.current_epoch % 5 == 0 and batch_idx == 0:
            self.log_confusion_matrix(preds, y)
        
        return loss
    
    def log_confusion_matrix(self, preds: torch.Tensor, targets: torch.Tensor):
        """Log confusion matrix to WandB"""
        if not self.trainer.sanity_checking:
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=targets.cpu().numpy(),
                    preds=preds.cpu().numpy(),
                )
            })
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
        
    def on_train_end(self):
        """Cleanup WandB logging"""
        self.wandb_logger.experiment.finish()

def main():
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from helper.util import train_test_split_custom, MelSpectrogramFeatureExtractor
    
    # Configuration
    config = {
        "data_path": "/path/to/your/data",  # Update this with your data path
        "batch_size": 32,
        "num_workers": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_epochs": 100,
        "project_name": "uav_classification",
        "model_name": "resnet18_uav_v1",
        "feature_extractor_config": {
            "n_mels": 64,
            "n_fft": 1024,
            "hop_length": 512,
            "power": 2.0
        }
    }
    
    # Create feature extractor
    feature_extractor = MelSpectrogramFeatureExtractor(**config["feature_extractor_config"])
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = train_test_split_custom(
        data_path=config["data_path"],
        feature_extractor=feature_extractor,
        test_size=0.2,
        val_size=0.1,
        seed=42
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    # Create model
    model = UAVClassifier(
        num_classes=len(train_dataset.classes),
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_epochs=config["max_epochs"],
        project_name=config["project_name"],
        model_name=config["model_name"]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",  # Automatically detect GPU if available
        devices=1,
        callbacks=callbacks,
        logger=model.wandb_logger,
        deterministic=True
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test model
    trainer.test(
        model=model,
        dataloaders=test_loader
    )

if __name__ == "__main__":
    main()