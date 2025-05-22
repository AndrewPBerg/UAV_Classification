import torch
import torch.nn as nn
import types
from typing import Dict, Tuple, Any, Optional, Union, Callable
import os

from peft import get_peft_model, LoraConfig, IA3Config, AdaLoraConfig, OFTConfig, LNTuningConfig
from peft.utils.peft_types import TaskType
from icecream import ic
from torchinfo import summary
import wandb

import math
import logging
import sys
import torch.nn.functional as F
    

from configs import GeneralConfig, FeatureExtractionConfig, PEFTConfig
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from models.transformer_models import TransformerModel
from models.cnn_models import CNNModel
from configs.peft_config import NoneClassifierConfig, NoneFullConfig, SSFConfig


class ModelFactory:
    """
    Factory class for creating models based on configuration.
    """
    @staticmethod
    def log_model_parameters(model: nn.Module) -> Dict[str, Any]:
        """
        Calculate and log model parameter metrics.
        
        Args:
            model: The PyTorch model
            
        Returns:
            Dictionary containing parameter metrics
        """
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate percentage of trainable parameters
        trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        # Calculate memory footprint (assuming float32/4 bytes per parameter)
        memory_footprint_bytes = total_params * 4  # 4 bytes for float32
        memory_footprint_mb = memory_footprint_bytes / (1024 * 1024)
        
        # Create metrics dictionary
        metrics = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/trainable_percentage": trainable_percentage,
            "model/memory_footprint_mb": memory_footprint_mb
        }
        
        # Log to console
        print(f"Model Parameters Summary:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Trainable Parameters (%): {trainable_percentage:.2f}%")
        print(f"  Memory Footprint: {memory_footprint_mb:.2f} MB")
        
        # Log to wandb if it's initialized
        if wandb.run is not None:
            wandb.log(metrics)
            # wandb.summary.update(metrics)
            # self.wandb_logger.experiment.summary(metrics)
        
        return metrics
    
    @staticmethod
    def create_model(
        general_config: GeneralConfig,
        feature_extraction_config: FeatureExtractionConfig,
        peft_config: Optional[PEFTConfig] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, Any]:
        """Create a model based on the configurations"""
        model_type = general_config.model_type
        CACHE_DIR = './model_cache'
        torch.hub.set_dir(CACHE_DIR)
        
        # Get input shape and feature extractor
        input_shape, feature_extractor = ModelFactory._get_feature_extractor(feature_extraction_config)
        # If input shape is 4D, convert to 3D
        # Shape is [width, height, channels], transform to [channels, width, height]
        if len(input_shape) == 3:
            input_shape = (input_shape[2], input_shape[0], input_shape[1])
        
        # Get number of classes
        num_classes = general_config.num_classes
        
        # Make sure adapter type is supported by the model
        adapter_type = general_config.adapter_type if hasattr(general_config, "adapter_type") else None
        
        # Create model based on model type
        if model_type in TransformerModel.transformer_models:
            # Verify adapter_type is supported by transformer models
            if adapter_type is not None and adapter_type not in TransformerModel.peft_type and adapter_type != "none":
                raise ValueError(f"Adapter type {adapter_type} not supported by transformer models")
            
            # Create transformer model
            transformer_model = TransformerModel()
            if model_type == "ast":
                model, feature_extractor = transformer_model._create_ast_model(num_classes, CACHE_DIR, general_config, peft_config)
            elif model_type == "mert":
                model, feature_extractor = transformer_model._create_mert_model(num_classes, CACHE_DIR, general_config, peft_config)
            elif model_type.startswith("vit"):
                model, feature_extractor = transformer_model._create_vit_model(
                    num_classes=num_classes,
                    CACHE_DIR=CACHE_DIR,
                    general_config=general_config,
                    peft_config=peft_config,
                    model_type=model_type
                )
            elif model_type.startswith("deit"):
                model, feature_extractor = transformer_model._create_deit_model(
                    num_classes=num_classes,
                    CACHE_DIR=CACHE_DIR,
                    general_config=general_config,
                    peft_config=peft_config,
                    model_type=model_type
                )
            else:
                raise ValueError(f"Unsupported transformer model type: {model_type}, please use one of the following: {TransformerModel.transformer_models}")
        
        elif model_type in CNNModel.cnn_models:
            # Verify adapter_type is supported by CNN models
            if adapter_type is not None and adapter_type not in CNNModel.peft_type and adapter_type != "none":
                raise ValueError(f"Adapter type {adapter_type} not supported by CNN models, please use one of the following: {CNNModel.peft_type}")
            
            # Create CNN model
            cnn_model = CNNModel()
            if "resnet" in model_type:
                model = cnn_model._create_resnet_model(model_type, num_classes, peft_config)
            elif "mobilenet" in model_type:
                model = cnn_model._create_mobilenet_model(model_type, num_classes, peft_config)
            elif "efficientnet" in model_type:
                model = cnn_model._create_efficientnet_model(model_type, num_classes, peft_config)
            elif "custom_cnn" in model_type:
                model = cnn_model._create_custom_cnn_model(model_type, num_classes, peft_config)
            else:
                raise ValueError(f"Unsupported CNN model type: {model_type}, please use one of the following: {CNNModel.cnn_models}")
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}, please use one of the following: {TransformerModel.transformer_models + CNNModel.cnn_models}")
        
        # Move model to device if specified
        if device is not None:
            model = model.to(device)

            
        summary(model,
                col_names=["num_params","trainable"],
                col_width=20,
                row_settings=["var_names"])

        print(model)
        
        # Log model parameter metrics
        ModelFactory.log_model_parameters(model)

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