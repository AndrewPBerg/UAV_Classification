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
    mobilenet_v3_large, MobileNet_V3_Large_Weights)

import torch
import torch.nn as nn
from typing import Tuple




    
class CNNModel:
    
    peft_type = ['lorac', 'none-full', 'none-classifier', 'ssf', 'batchnorm']
    
    cnn_models = ['resnet', 'mobilenet', 'efficientnet']
    @staticmethod
    def _create_resnet_model(model_type: str, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
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
        elif "34" in model_type:
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif "50" in model_type:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif "101" in model_type:
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif "152" in model_type:
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported ResNet model type: {model_type}")
        
        # Modify first convolutional layer to accept grayscale input
        if input_shape[0] == 1:
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace classification head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        return model
    
    @staticmethod
    def _create_mobilenet_model(model_type: str, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
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
        if input_shape[0] == 1:
            model.features[0][0] = nn.Conv2d(
                1, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        
        # Replace classification head
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
        return model
    
    @staticmethod
    def _create_efficientnet_model(model_type: str, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
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
        if input_shape[0] == 1:
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
    