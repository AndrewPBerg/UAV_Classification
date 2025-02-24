import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import (
    # ResNet variants
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights,
    # EfficientNet variants
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
    # MobileNet variants
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)
from transformers import ViTForImageClassification, ViTConfig
from peft import (
    get_peft_model,
    LoraConfig,
    IA3Config,
    TaskType,
    PeftModel,
    PeftConfig
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Tuple, Dict, Any, Literal
from helper.util import AudioDataset, wandb_login
import wandb
from dataclasses import dataclass

@dataclass
class PeftArgs:
    """Arguments for PEFT configuration"""
    adapter_type: Literal["lora", "ia3"] = "lora"
    r: int = 8  # LoRA rank
    alpha: int = 16  # LoRA alpha scaling
    dropout: float = 0.1
    bias: str = "none"
    target_modules: Optional[list] = None  # If None, will use default for ViT
    modules_to_save: Optional[list] = None
    init_lora_weights: bool = True

ModelSizeType = Literal[
    # ResNet variants
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    # EfficientNet variants
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    # MobileNet variants
    "mobilenet_v3_small", "mobilenet_v3_large",
    # ViT variants
    "vit_tiny", "vit_small", "vit_base", "vit_large"
]

class UAVClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_size: ModelSizeType = "resnet18",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        model_name: str = None,
        project_name: str = "uav_classification",
        image_size: int = 224,  # Required for ViT
        peft_args: Optional[PeftArgs] = None  # PEFT configuration
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize wandb
        wandb_login()
        
        # If no model name provided, create one based on architecture
        if model_name is None:
            model_name = f"{model_size}_uav"
            if peft_args:
                model_name += f"_{peft_args.adapter_type}"
            
        self.wandb_logger = pl.loggers.WandbLogger(
            project=project_name,
            name=model_name,
            log_model=True
        )
        
        # Model architecture mapping
        self.architecture_mapping = {
            # ResNet variants
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
            "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
            "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
            "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
            "resnet152": (resnet152, ResNet152_Weights.DEFAULT),
            # EfficientNet variants
            "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
            "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
            "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
            "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
            "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.DEFAULT),
            "efficientnet_b5": (efficientnet_b5, EfficientNet_B5_Weights.DEFAULT),
            "efficientnet_b6": (efficientnet_b6, EfficientNet_B6_Weights.DEFAULT),
            "efficientnet_b7": (efficientnet_b7, EfficientNet_B7_Weights.DEFAULT),
            # MobileNet variants
            "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
            "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
        }
        
        # ViT configurations
        self.vit_configs = {
            "vit_tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
            "vit_small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
            "vit_base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
            "vit_large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        }
        
        # Initialize the model based on architecture type
        if model_size in self.architecture_mapping:
            model_fn, weights = self.architecture_mapping[model_size]
            self.backbone = model_fn(weights=weights)
            
            # Modify first conv layer for single channel input
            if "resnet" in model_size:
                self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            elif "efficientnet" in model_size:
                self.backbone.features[0][0] = nn.Conv2d(1, self.backbone.features[0][0].out_channels, 
                                                       kernel_size=3, stride=2, padding=1, bias=False)
            elif "mobilenet" in model_size:
                self.backbone.features[0][0] = nn.Conv2d(1, self.backbone.features[0][0].out_channels,
                                                       kernel_size=3, stride=2, padding=1, bias=False)
            
            # Replace final classification layer
            if hasattr(self.backbone, 'fc'):
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(num_features, num_classes)
            elif hasattr(self.backbone, 'classifier'):
                if isinstance(self.backbone.classifier, nn.Sequential):
                    num_features = self.backbone.classifier[-1].in_features
                    self.backbone.classifier[-1] = nn.Linear(num_features, num_classes)
                else:
                    num_features = self.backbone.classifier.in_features
                    self.backbone.classifier = nn.Linear(num_features, num_classes)
                    
        elif "vit" in model_size:
            config = ViTConfig(
                image_size=image_size,
                patch_size=16,
                num_channels=1,
                num_labels=num_classes,
                **self.vit_configs[model_size]
            )
            self.backbone = ViTForImageClassification(config)
            self.backbone.load_pretrained_weights()  # Load pretrained weights if available
            
            # Apply PEFT if specified (only for ViT models)
            if peft_args:
                # Default target modules for ViT if not specified
                if not peft_args.target_modules:
                    peft_args.target_modules = ["query", "key", "value", "output.dense"]
                
                if peft_args.adapter_type == "lora":
                    peft_config = LoraConfig(
                        task_type=TaskType.IMAGE_CLASSIFICATION,
                        inference_mode=False,
                        r=peft_args.r,
                        lora_alpha=peft_args.alpha,
                        lora_dropout=peft_args.dropout,
                        bias=peft_args.bias,
                        target_modules=peft_args.target_modules,
                        modules_to_save=peft_args.modules_to_save,
                        init_lora_weights=peft_args.init_lora_weights
                    )
                else:  # ia3
                    peft_config = IA3Config(
                        task_type=TaskType.IMAGE_CLASSIFICATION,
                        inference_mode=False,
                        target_modules=peft_args.target_modules,
                        modules_to_save=peft_args.modules_to_save,
                        init_ia3_weights=True
                    )
                
                self.backbone = get_peft_model(self.backbone, peft_config)
                self.backbone.print_trainable_parameters()  # Log trainable parameters
        
        else:
            raise ValueError(f"Unsupported model architecture: {model_size}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Log hyperparameters
        self.wandb_logger.log_hyperparams(self.hparams)
        
    def forward(self, x):
        if isinstance(self.backbone, (ViTForImageClassification, PeftModel)):
            return self.backbone(x).logits
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
        "model_size": "resnet18",
        "feature_extractor_config": {
            "n_mels": 64,
            "n_fft": 1024,
            "hop_length": 512,
            "power": 2.0
        },
        "peft_args": {
            "adapter_type": "lora",
            "r": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none",
            "target_modules": None,
            "modules_to_save": None,
            "init_lora_weights": True
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
    peft_args = PeftArgs(**config["peft_args"])
    model = UAVClassifier(
        num_classes=len(train_dataset.classes),
        model_size=config["model_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_epochs=config["max_epochs"],
        project_name=config["project_name"],
        model_name=config["model_name"],
        peft_args=peft_args
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