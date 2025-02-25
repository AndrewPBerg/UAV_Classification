import torch.nn as nn
from typing import Tuple, Optional
from models.architectures import ModelArchitectures
from transformers import ViTForImageClassification, ViTConfig
from config.model_config import ModelConfig
from models.peft_handler import PeftHandler

class ModelFactory:
    """Factory class for creating and configuring models"""
    
    @staticmethod
    def create_model(config: ModelConfig) -> nn.Module:
        """Creates and configures a model based on the provided configuration"""
        if ModelArchitectures.is_vit_model(config.model_size):
            return ModelFactory._create_vit_model(config)
        else:
            return ModelFactory._create_cnn_model(config)
    
    @staticmethod
    def _create_vit_model(config: ModelConfig) -> nn.Module:
        """Creates and configures a ViT model"""
        vit_config = ModelArchitectures.get_vit_configs().get(config.model_size)
        if not vit_config:
            raise ValueError(f"Invalid ViT model size: {config.model_size}")
            
        model_config = ViTConfig(
            image_size=config.image_size,
            patch_size=16,
            num_channels=1,
            num_labels=config.num_classes,
            **vit_config
        )
        
        model = ViTForImageClassification(model_config)
        model.load_pretrained_weights()
        
        # Apply PEFT if specified
        return PeftHandler.apply_peft(model, config.peft_args)
    
    @staticmethod
    def _create_cnn_model(config: ModelConfig) -> nn.Module:
        """Creates and configures a CNN model (ResNet, EfficientNet, MobileNet)"""
        architecture_mapping = ModelArchitectures.get_architecture_mapping()
        if config.model_size not in architecture_mapping:
            raise ValueError(f"Invalid model size: {config.model_size}")
            
        model_fn, weights = architecture_mapping[config.model_size]
        model = model_fn(weights=weights)
        
        # Modify first conv layer for single channel input
        if "resnet" in config.model_size:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif "efficientnet" in config.model_size:
            model.features[0][0] = nn.Conv2d(1, model.features[0][0].out_channels, 
                                           kernel_size=3, stride=2, padding=1, bias=False)
        elif "mobilenet" in config.model_size:
            model.features[0][0] = nn.Conv2d(1, model.features[0][0].out_channels,
                                           kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace final classification layer
        ModelFactory._replace_classification_layer(model, config.num_classes)
        
        return model
    
    @staticmethod
    def _replace_classification_layer(model: nn.Module, num_classes: int):
        """Replaces the final classification layer with appropriate output size"""
        if hasattr(model, 'fc'):
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                num_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_features, num_classes)
            else:
                num_features = model.classifier.in_features
                model.classifier = nn.Linear(num_features, num_classes)
