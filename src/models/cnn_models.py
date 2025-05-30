from torchvision.models import (
    # ResNet variants
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
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
    mobilenet_v3_large, MobileNet_V3_Large_Weights)

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
from configs.peft_config import PEFTConfig, NoneClassifierConfig, NoneFullConfig, SSFConfig, LoRACConfig
from .ssf_adapter import apply_ssf_to_model
from .lorac_adapter import apply_lorac_to_model
import sys
from torch.nn import functional as F
from configs import BatchNormConfig,NoneFullConfig




    
class CNNModel:
    
    peft_type = ['lorac', 'none-full', 'none-classifier', 'ssf', 'batchnorm']
    
    cnn_models = ['resnet18','resnet50','resnet152', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'custom_cnn']
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
    def _create_custom_cnn_model(model_type: str, num_classes: int, peft_config: Optional[PEFTConfig] = None) -> nn.Module:
        class CustomCNN(nn.Module):
            def __init__(self, num_classes: int, hidden_units: int = 256):
                """
                Initialize the CNN model with dynamic input shape handling.
                
                Args:
                    num_classes (int): Number of output classes
                    hidden_units (int): Number of hidden units in the fully connected layer
                """
                super(CustomCNN, self).__init__()
                
                # First convolutional block
                self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(16)
                )
                
                # Second convolutional block
                self.conv2 = nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(32)
                )
                
                # Third convolutional block
                self.conv3 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(64)
                )
                
                # Adaptive pooling to handle different input sizes
                self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Output 4x4 feature maps
                
                # Calculate fixed feature size after adaptive pooling
                self.feature_size = 64 * 4 * 4  # 64 channels * 4 * 4 = 1024
                
                # Dense layers with configurable hidden units
                self.fc1 = nn.Linear(self.feature_size, hidden_units)
                self.dropout = nn.Dropout(p=0.5)
                self.fc2 = nn.Linear(hidden_units, num_classes)
                
                # Track if we've seen data yet (for debugging/logging)
                self._input_shape_logged = False

            def forward(self, x):
                # Log input shape on first forward pass for debugging
                if not self._input_shape_logged:
                    print(f"CustomCNN: Processing input with shape {x.shape}")
                    self._input_shape_logged = True
                
                # Add channel dimension if not present (batch_size, height, width) -> (batch_size, 1, height, width)
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                elif x.dim() == 2:
                    # Handle case where batch dimension might be missing
                    x = x.unsqueeze(0).unsqueeze(0)
                
                # Ensure we have the right number of channels (should be 1 for grayscale spectrograms)
                if x.size(1) != 1:
                    # If RGB or other multi-channel, convert to grayscale
                    if x.size(1) == 3:
                        x = torch.mean(x, dim=1, keepdim=True)
                    else:
                        # Take only the first channel
                        x = x[:, :1, :, :]
                
                # Convolutional layers
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                
                # Adaptive pooling to standardize feature map size
                x = self.adaptive_pool(x)
                
                # Flatten
                x = x.flatten(1)
                
                # Dense layers
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                
                return x
            
            def get_model_info(self):
                """Get information about the model architecture"""
                total_params = sum(p.numel() for p in self.parameters())
                trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
                
                return {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'feature_size_after_conv': self.feature_size,
                    'model_type': 'CustomCNN'
                }
                
            def test_input_shape(self, height: int, width: int):
                """Test the model with a given input shape to verify it works"""
                with torch.no_grad():
                    test_input = torch.randn(1, 1, height, width)
                    try:
                        output = self.forward(test_input)
                        print(f"✓ Input shape ({height}, {width}) works. Output shape: {output.shape}")
                        return True
                    except Exception as e:
                        print(f"✗ Input shape ({height}, {width}) failed: {str(e)}")
                        return False
                        
        model = CustomCNN(num_classes=num_classes)
        model = apply_peft(model, peft_config if peft_config is not None else NoneFullConfig())
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
        elif "b1" in model_type:
            model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        elif "b2" in model_type:
            model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        elif "b3" in model_type:
            model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        elif "b4" in model_type:
            model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        elif "b5" in model_type:
            model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        elif "b6" in model_type:
            model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
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
            target_modules=peft_config.target_modules
        )
        
        # Print information about applied LoRAC layers
        if hasattr(model, 'lorac_layers') and len(model.lorac_layers) > 0:
            print(f"Applied LoRAC to {len(model.lorac_layers)} layers:")
            for name in model.lorac_layers.keys():
                print(f"  - {name}")
        else:
            print("Warning: No LoRAC layers were applied to the model.")
    
    else:
        raise ValueError(f"Unsupported PEFT configuration type: {type(peft_config).__name__}")
    
    return model
