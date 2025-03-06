from torchvision.models import (
    # ResNet variants
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet152, ResNet152_Weights,
    # EfficientNet variants
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
    # MobileNet variants
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights)

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
from configs.peft_config import PEFTConfig, NoneClassifierConfig, NoneFullConfig, SSFConfig, LoRACConfig
from .ssf_adapter import apply_ssf_to_model
from .lorac_adapter import apply_lorac_to_model

from configs import BatchNormConfig




    
class CNNModel:
    
    peft_type = ['lorac', 'none-full', 'none-classifier', 'ssf', 'batchnorm']
    
    cnn_models = ['resnet18','resnet50','resnet152', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b4', 'efficientnet_b7']
    @staticmethod
    def _create_resnet_model(model_type: str, num_classes: int, peft_config: Optional[PEFTConfig] = None) -> nn.Module:
        """
        Create a ResNet model.
        
        Args:
            model_type: Model type (e.g., 'resnet18')
            num_classes: Number of classes
            input_shape: Input shape (channels, height, width)
            
        Returns:
            ResNet model
        """
        # Parse model size from model type
        if "18" in model_type:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif "50" in model_type:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif "152" in model_type:
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported ResNet model type: {model_type}")
        
        # Modify first convolutional layer to accept grayscale input
        # if input_shape[0] == 1:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Replace classification head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        model = apply_peft(model, peft_config)
        
        return model
    
    @staticmethod
    def _create_mobilenet_model(model_type: str, num_classes: int, peft_config: Optional[PEFTConfig] = None) -> nn.Module:
        """
        Create a MobileNet model.
        
        Args:
            model_type: Model type (e.g., 'mobilenet_v3_small')
            num_classes: Number of classes
            input_shape: Input shape (channels, height, width)
            
        Returns:
            MobileNet model
        """
        # Parse model size from model type
        if "small" in model_type:
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        elif "large" in model_type:
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported MobileNet model type: {model_type}")
        
        # Modify first convolutional layer to accept grayscale input

        model.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Replace classification head
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
        model = apply_peft(model, peft_config)
        
        return model
    
    @staticmethod
    def _create_efficientnet_model(model_type: str, num_classes: int, peft_config: Optional[PEFTConfig] = None) -> nn.Module:
        """
        Create an EfficientNet model.
        
        Args:
            model_type: Model type (e.g., 'efficientnet_b0')
            num_classes: Number of classes
            input_shape: Input shape (channels, height, width)
            
        Returns:
            EfficientNet model
        """
        # Parse model size from model type
        if "b0" in model_type:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        elif "b4" in model_type:
            model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        elif "b7" in model_type:
            model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported EfficientNet model type: {model_type}")
        
        # Modify first convolutional layer to accept grayscale input

        model.features[0][0] = nn.Conv2d(
            1, model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=False
        )
    
        # Replace classification head
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
        model = apply_peft(model, peft_config)
        
        return model

# outside of class scope for static method use
def apply_peft(model: nn.Module, peft_config: Optional[PEFTConfig]) -> nn.Module:
    """
    Apply parameter-efficient fine-tuning to a model.
    
    Args:
        model: Model to apply PEFT to.
        peft_config: PEFT configuration.
        
    Returns:
        Model with PEFT applied.
    """
    if peft_config is None:
        return model
    
    elif isinstance(peft_config, NoneFullConfig):
        # Fine-tune all parameters
        for param in model.parameters():
            param.requires_grad = True
    
    elif isinstance(peft_config, NoneClassifierConfig):
        # Freeze all parameters except the classifier
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable training for classifier layers
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
            
            # For ResNet models, also enable training for the average pooling layer
            if hasattr(model, 'avgpool'):
                for param in model.avgpool.parameters():
                    param.requires_grad = True
    
    elif isinstance(peft_config, BatchNormConfig):
        # Only fine-tune BatchNorm layers
        for param in model.parameters():
            param.requires_grad = False
        
        target_modules = peft_config.target_modules
        # Enable training for BatchNorm layers
        for name, module in model.named_modules():
            # Check if module is a BatchNorm layer
            if any(batch_norm_type in module.__class__.__name__.lower() for batch_norm_type in target_modules):
                for param in module.parameters():
                    param.requires_grad = True
    
    elif isinstance(peft_config, SSFConfig):
        # Apply SSF adapter to the model
        
        model = apply_ssf_to_model(
            model=model,
            init_scale=peft_config.init_scale,
            init_shift=peft_config.init_shift,
            target_modules=peft_config.target_modules,
            expected_input_channels=3,  # Image data typically has 3 channels (RGB)
            verbose=True
        )
    
    elif isinstance(peft_config, LoRACConfig):
        # Apply LoRA-C to the model

        # First, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply LoRA-C to the model
        model = apply_lorac_to_model(
            model=model,
            r=peft_config.r,
            alpha=peft_config.alpha,
            dropout=peft_config.dropout,
            target_modules=peft_config.target_modules if peft_config.target_modules else None
        )
    
    else:
        raise ValueError(f"Unsupported PEFT configuration type: {type(peft_config).__name__}")
    
    return model
