import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Optional, Union, Callable
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
    # ViT variants
    vit_b_16, ViT_B_16_Weights,
    vit_b_32, ViT_B_32_Weights,
    vit_l_16, ViT_L_16_Weights,
    vit_l_32, ViT_L_32_Weights,
    vit_h_14, ViT_H_14_Weights,
)
from peft import get_peft_model, LoraConfig, IA3Config, AdaLoraConfig, OFTConfig, FourierFTConfig, LNTuningConfig
from peft.utils.peft_types import TaskType
from icecream import ic
from transformers import PreTrainedModel

from configs.configs_demo import GeneralConfig, FeatureExtractionConfig
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from ast_model import ASTModel


class ModelFactory:
    """
    Factory class for creating models based on configuration.
    """
    @staticmethod
    def create_model(
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        peft_config: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, Any]:
        """
        Create a model based on configuration.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            peft_config: PEFT configuration (optional)
            device: Device to put model on (optional)
            
        Returns:
            Tuple of (model, feature_extractor)
        """
        
        torch.hub.set_dir('/app/src/model_cache')  # Set custom cache directory
        
        model_type = general_config.model_type.lower()
        num_classes = general_config.num_classes
        
        # Get feature extractor and input shape
        input_shape, feature_extractor = ModelFactory._get_feature_extractor(feature_extraction_config)
        
        # Create model based on type
        if model_type == "ast":
            # Use the new ASTModel class
            model, processor, _ = ASTModel.create_model(
                num_classes=num_classes,
                adapter_type=general_config.adapter_type,
                peft_config=peft_config,
                device=device
            )
            return model, processor
        elif model_type.startswith("vit"):
            model = ModelFactory._create_vit_model(model_type, num_classes, input_shape)
        elif model_type.startswith("resnet"):
            model = ModelFactory._create_resnet_model(model_type, num_classes, input_shape)
        elif model_type.startswith("mobilenet"):
            model = ModelFactory._create_mobilenet_model(model_type, num_classes, input_shape)
        elif model_type.startswith("efficientnet"):
            model = ModelFactory._create_efficientnet_model(model_type, num_classes, input_shape)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Apply PEFT if provided - only for AST models which are PreTrainedModels
        # Other models don't support PEFT directly
        if peft_config is not None and not isinstance(peft_config, str) and model_type == "ast":
            model = get_peft_model(model, peft_config)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
        
        # Move model to device if provided
        if device is not None and model is not None:
            model = model.to(device)
        
        return model, feature_extractor
    
    @staticmethod
    def _get_feature_extractor(
        feature_extraction_config: FeatureExtractionConfig
    ) -> Tuple[Tuple[int, int, int], Any]:
        """
        Get feature extractor and input shape based on configuration.
        
        Args:
            feature_extraction_config: Feature extraction configuration
            
        Returns:
            Tuple of (input_shape, feature_extractor)
        """
        feature_type = feature_extraction_config.type
        
        if feature_type == 'melspectrogram':
            input_shape = (1, feature_extraction_config.n_mels, 157)  # Channels, height, width
            feature_extractor = MelSpectrogramFeatureExtractor(
                sampling_rate=feature_extraction_config.sampling_rate,
                n_mels=feature_extraction_config.n_mels,
                n_fft=feature_extraction_config.n_fft,
                hop_length=feature_extraction_config.hop_length,
                power=feature_extraction_config.power
            )
        elif feature_type == 'mfcc':
            input_shape = (1, feature_extraction_config.n_mfcc, 157)  # Channels, height, width
            feature_extractor = MFCCFeatureExtractor(
                sampling_rate=feature_extraction_config.sampling_rate,
                n_mfcc=feature_extraction_config.n_mfcc,
                n_mels=feature_extraction_config.n_mels,
                n_fft=feature_extraction_config.n_fft,
                hop_length=feature_extraction_config.hop_length
            )
        else:
            raise ValueError(f"Unsupported feature extraction type: {feature_type}")
        
        return input_shape, feature_extractor
    
    @staticmethod
    def _create_vit_model(model_type: str, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
        """
        Create a ViT model.
        
        Args:
            model_type: Model type (e.g., 'vit_b_16')
            num_classes: Number of classes
            input_shape: Input shape (channels, height, width)
            
        Returns:
            ViT model
        """
        # Parse model size from model type
        if "b_16" in model_type:
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif "b_32" in model_type:
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        elif "l_16" in model_type:
            model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
        elif "l_32" in model_type:
            model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
        elif "h_14" in model_type:
            model = vit_h_14(weights=ViT_H_14_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported ViT model type: {model_type}")
        
        # Modify the model for audio input
        model.image_size = 224  # Set fixed size expected by ViT
        resize_layer = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # Store original forward method
        original_forward = model.forward
        
        # Define new forward method
        def new_forward(self, x):
            # Handle input
            x = x.float()
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            # Resize input to match ViT's expected size
            x = resize_layer(x)
            
            # Call original forward method
            return original_forward(x)
        
        # Replace forward method
        model.forward = new_forward.__get__(model, type(model))
        
        # Replace classification head
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
        return model
    
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
    
    @staticmethod
    def get_model_factory(
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        peft_config: Optional[Any] = None
    ) -> Callable[[torch.device], Tuple[nn.Module, Any]]:
        """
        Get a model factory function that creates a model based on configuration.
        
        Args:
            general_config: General configuration
            feature_extraction_config: Feature extraction configuration
            peft_config: PEFT configuration (optional)
            
        Returns:
            Function that creates a model
        """
        def factory_fn(device: torch.device) -> Tuple[nn.Module, Any]:
            return ModelFactory.create_model(
                general_config=general_config,
                feature_extraction_config=feature_extraction_config,
                peft_config=peft_config,
                device=device
            )
        
        return factory_fn 