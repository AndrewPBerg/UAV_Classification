import torch
import torch.nn as nn
import types
from typing import Dict, Tuple, Any, Optional, Union, Callable
import os
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
from transformers import (
    ASTFeatureExtractor, ASTForAudioClassification, AutoModel, PreTrainedModel, AutoFeatureExtractor, Wav2Vec2FeatureExtractor
)
import math
import logging
import sys
import torch.nn.functional as F
    

from configs import GeneralConfig, FeatureExtractionConfig
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor



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
            peft_config: PEFT configuration
            device: Device to move model to (no longer used with PyTorch Lightning)
            
        Returns:
            Tuple of (model, feature_extractor)
        """
        # # Set a custom directory for PyTorch to cache downloaded models based on the operating system
        # if os.name == 'linux':  # Specifically for Linux
        #     torch.hub.set_dir('/app/src/model_cache/linux')  # Set custom cache directory for Linux
        # elif os.name == 'posix':  # This includes macOS
        #     torch.hub.set_dir('/app/src/model_cache/mac')  # Set custom cache directory for macOS
        # else:
        #     torch.hub.set_dir('C:/app/src/model_cache')  # Set custom cache directory for Windows
        
        # set model cache directory (for both transformer and torch-hub models)

        CACHE_DIR = './model_cache'
        torch.hub.set_dir(CACHE_DIR)
        
        model_type = general_config.model_type.lower()
        num_classes = general_config.num_classes
        
        ic(model_type)
        
        # handle models downloaded from huggingface
        if model_type in ["ast", "mert"]:
            
            # Create model and feature extractor based on type
            if model_type == "ast":
                # Use the new ASTModel class
                model, feature_extractor = ModelFactory._create_ast_model(num_classes, CACHE_DIR)
            elif model_type == "mert":
                model, feature_extractor = ModelFactory._create_mert_model(num_classes, CACHE_DIR)
                
            
        
        # handle models downloaded from torch-hub
        else:
            # Get feature extractor and input shape
            input_shape, feature_extractor = ModelFactory._get_feature_extractor(feature_extraction_config)
            # Create model based on type
            if model_type == "vit":
                model = ModelFactory._create_vit_model(model_type, num_classes, input_shape)
            elif model_type.startswith("resnet"):
                model = ModelFactory._create_resnet_model(model_type, num_classes, input_shape)
            elif model_type.startswith("mobilenet"):
                model = ModelFactory._create_mobilenet_model(model_type, num_classes, input_shape)
            elif model_type.startswith("efficientnet"):
                model = ModelFactory._create_efficientnet_model(model_type, num_classes, input_shape)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        
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
    def _create_mert_model(num_classes, CACHE_DIR: str) -> nn.Module:
        """
        Create a MERT model with proper input shape and device handling.
        """
        # Set up logging
        logger = logging.getLogger("MERT_MODEL")
        
        pretrained_MERT_model = "m-a-p/MERT-v1-330M"
        
        try:
            model = AutoModel.from_pretrained(pretrained_MERT_model, cache_dir=CACHE_DIR, trust_remote_code=True, local_files_only=True)
        except OSError:
            model = AutoModel.from_pretrained(pretrained_MERT_model, trust_remote_code=True, cache_dir=CACHE_DIR)
        
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_MERT_model, cache_dir=CACHE_DIR, local_files_only=True)
        except OSError:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_MERT_model, cache_dir=CACHE_DIR)
        
        # Save original forward method
        original_forward = model.forward
        
        # Define new forward method to handle shape issues and device consistency
        def new_forward(self, x):
            logger.debug(f"Input shape: {x.shape}, device: {x.device}")
            
            # Ensure model is on same device as input
            if next(self.parameters()).device != x.device:
                self = self.to(x.device)
                logger.debug(f"Moved model to device: {x.device}")
            
            # Handle 4D input [batch, channels, height, width]
            if len(x.shape) == 4:
                batch_size, channels, height, width = x.shape
                if channels == 1:
                    # For single channel input, reshape to [batch, sequence_length]
                    x = x.squeeze(1)  # Remove channel dimension
                    if height == 1:
                        x = x.squeeze(1)  # Remove height dimension if it's 1
                    else:
                        x = x.view(batch_size, -1)  # Flatten remaining dimensions
                    logger.debug(f"Reshaped 4D input to: {x.shape}")
            
            # Handle 3D input [batch, channels, sequence_length]
            elif len(x.shape) == 3:
                if x.size(1) == 1:  # If single channel
                    x = x.squeeze(1)  # Remove channel dimension
                    logger.debug(f"Squeezed 3D input to: {x.shape}")
            
            # Add classifier head if needed
            if not hasattr(self, 'classifier'):
                hidden_size = self.config.hidden_size
                self.classifier = nn.Linear(hidden_size, num_classes).to(x.device)
                logger.debug(f"Added classifier head with {hidden_size} -> {num_classes} on device {x.device}")
            else:
                # Ensure classifier is on same device as input
                self.classifier = self.classifier.to(x.device)
            
            # Forward pass through MERT
            outputs = original_forward(x)
            hidden_states = outputs[0]  # Get the last hidden state
            
            # Pool the output (mean pooling over sequence dimension)
            pooled_output = torch.mean(hidden_states, dim=1)
            
            # Ensure pooled output and classifier are on same device
            pooled_output = pooled_output.to(self.classifier.weight.device)
            
            # Pass through classifier
            logits = self.classifier(pooled_output)
            
            return logits
        
        # Replace the forward method
        model.forward = types.MethodType(new_forward, model)
        
        return model, feature_extractor
    

    
    @staticmethod
    def _create_ast_model(num_classes: int, CACHE_DIR: str) -> nn.Module:
        """
        Create an AST model.
        """
        
        # Set up detailed logging
        logging.basicConfig(
            level=logging.CRITICAL,  # This will effectively disable most logging
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        logger = logging.getLogger("AST_MODEL")
        
        pretrained_AST_model="MIT/ast-finetuned-audioset-10-10-0.4593"

        try:
            model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, attn_implementation="sdpa", cache_dir=CACHE_DIR, local_files_only=True)
        except OSError:
            model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
        
        model.config.num_labels = num_classes
        in_features = model.classifier.dense.in_features
        model.classifier.dense = nn.Linear(in_features, num_classes)
        
        # Get the expected sequence length from the position embeddings
        expected_seq_length = model.audio_spectrogram_transformer.embeddings.position_embeddings.shape[1]
        logger.debug(f"Expected sequence length from position embeddings: {expected_seq_length}")
        
        # Save original embeddings forward method
        original_embeddings_forward = model.audio_spectrogram_transformer.embeddings.forward
        
        # Define a new embeddings forward method to handle sequence length mismatch
        def new_embeddings_forward(self, input_values):
            logger.debug(f"Embeddings input shape: {input_values.shape}")
            
            # Get patch embeddings
            embeddings = self.patch_embeddings(input_values)
            logger.debug(f"Patch embeddings shape: {embeddings.shape}")
            
            # Check if sequence length matches expected length
            current_seq_length = embeddings.shape[1]
            
            if current_seq_length != expected_seq_length:
                logger.debug(f"Sequence length mismatch: got {current_seq_length}, expected {expected_seq_length}")
                
                # Use a deterministic approach instead of interpolation
                # We'll use a learned projection layer to change the sequence length
                if not hasattr(self, 'seq_adapter'):
                    # Create a sequence adapter if it doesn't exist
                    # This is a simple linear layer that projects from current_seq_length to expected_seq_length
                    self.seq_adapter = torch.nn.Linear(
                        current_seq_length, 
                        expected_seq_length
                    ).to(embeddings.device)
                    logger.debug("Created sequence adapter layer")
                
                # Apply the sequence adapter
                # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim, seq_len]
                embeddings = embeddings.transpose(1, 2)
                # Apply linear projection to change sequence length
                embeddings = self.seq_adapter(embeddings)
                # [batch_size, hidden_dim, seq_len] -> [batch_size, seq_len, hidden_dim]
                embeddings = embeddings.transpose(1, 2)
                
                logger.debug(f"Adapted embeddings shape: {embeddings.shape}")
            
            # Add position embeddings
            embeddings = embeddings + self.position_embeddings
            embeddings = self.dropout(embeddings)
            
            return embeddings
        
        # Replace the embeddings forward method
        model.audio_spectrogram_transformer.embeddings.forward = types.MethodType(
            new_embeddings_forward, 
            model.audio_spectrogram_transformer.embeddings
        )
        
        # Save original forward method
        original_forward = model.forward
        
        # Define a new forward method to handle input shape issues
        def new_forward(self, x):
            # Debug logging
            logger.debug(f"AST model input shape before processing: {x.shape}")
            
            # Check if input has 5 dimensions [batch, channels, height, extra_dim, width]
            if len(x.shape) == 5:
                logger.debug(f"Detected 5D input tensor with shape: {x.shape}")
                
                # Get the dimensions
                batch_size, channels, height, extra_dim, width = x.shape
                
                # Reshape to 4D tensor that conv2d can accept
                # We need to reshape from [batch, channels, height, extra_dim, width] to [batch, channels, height*extra_dim, width]
                try:
                    x = x.reshape(batch_size, channels, height * extra_dim, width)
                    logger.debug(f"Reshaped tensor to: {x.shape}")
                except Exception as e:
                    logger.error(f"Error reshaping tensor: {e}")
                    
                    # Alternative approach: try to squeeze out the extra dimension if it's 1
                    if extra_dim == 1:
                        try:
                            x = x.squeeze(3)  # Remove the 4th dimension (index 3)
                            logger.debug(f"Squeezed tensor to: {x.shape}")
                        except Exception as e2:
                            logger.error(f"Error squeezing tensor: {e2}")
                            
                            # Last resort: try to view the tensor differently
                            try:
                                x = x.view(batch_size, channels, height, width)
                                logger.debug(f"Viewed tensor as: {x.shape}")
                            except Exception as e3:
                                logger.error(f"Error viewing tensor: {e3}")
            
            # Final shape check before passing to original forward
            logger.debug(f"Final tensor shape before original forward: {x.shape}")
            
            try:
                return original_forward(x)
            except Exception as e:
                logger.error(f"Error in original_forward: {e}")
                logger.error(f"Input tensor shape: {x.shape}")
                raise
        
        # Replace the forward method
        model.forward = types.MethodType(new_forward, model)
        
        # Also patch the patch_embeddings projection method directly
        original_patch_embeddings_forward = model.audio_spectrogram_transformer.embeddings.patch_embeddings.forward
        
        def new_patch_embeddings_forward(self, input_values):
            logger.debug(f"Patch embeddings input shape: {input_values.shape}")
            
            # Handle 5D input directly at the patch embeddings level
            if len(input_values.shape) == 5:
                logger.debug("Fixing 5D input at patch embeddings level")
                batch_size, channels, height, extra_dim, width = input_values.shape
                
                # Try different approaches
                if extra_dim == 1:
                    # If extra dimension is 1, just squeeze it out
                    input_values = input_values.squeeze(3)
                    logger.debug(f"Squeezed to shape: {input_values.shape}")
                else:
                    # Otherwise reshape
                    try:
                        input_values = input_values.reshape(batch_size, channels, height * extra_dim, width)
                        logger.debug(f"Reshaped to: {input_values.shape}")
                    except Exception as e:
                        logger.error(f"Error reshaping in patch embeddings: {e}")
            
            # Call original method with fixed input
            try:
                result = original_patch_embeddings_forward(input_values)
                logger.debug(f"Patch embeddings output shape: {result.shape}")
                return result
            except Exception as e:
                logger.error(f"Error in original patch embeddings forward: {e}")
                logger.error(f"Input shape: {input_values.shape}")
                # Try one more approach if it fails
                if len(input_values.shape) == 4:
                    logger.debug("Attempting alternative approach for 4D input")
                    # Try to use the projection directly
                    try:
                        result = self.projection(input_values).flatten(2).transpose(1, 2)
                        logger.debug(f"Direct projection successful, shape: {result.shape}")
                        return result
                    except Exception as e2:
                        logger.error(f"Direct projection failed: {e2}")
                raise
        
        # Replace the patch embeddings forward method
        model.audio_spectrogram_transformer.embeddings.patch_embeddings.forward = types.MethodType(
            new_patch_embeddings_forward, 
            model.audio_spectrogram_transformer.embeddings.patch_embeddings
        )
        
        try:
            feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
        except OSError:
            feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
            
        return model, feature_extractor
        
    
    @staticmethod
    def _create_vit_model(model_type: str, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
        """
        Create a ViT model.
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

        # Add resize layer to match ViT's expected input size
        model.image_size = 224  # Set fixed size expected by ViT
        resize_layer = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        def new_forward(self, x):
            # Handle input
            x = x.float()
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            # Resize input to match ViT's expected size
            x = resize_layer(x)
            
            # Process through convolutional projection
            x = self.conv_proj(x)
            
            # Reshape to sequence
            batch_size = x.shape[0]
            x = x.flatten(2).transpose(1, 2)
            
            # Add class token
            class_token = self.class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)
            
            # Get position embeddings and adjust if necessary
            pos_embedding = self.encoder.pos_embedding
            if x.size(1) != pos_embedding.size(1):
                # Remove class token from position embeddings
                pos_tokens = pos_embedding[:, 1:]
                # Reshape position embeddings to square grid
                grid_size = int(math.sqrt(pos_tokens.size(1)))
                pos_tokens = pos_tokens.reshape(-1, grid_size, grid_size, pos_embedding.size(-1))
                
                # Interpolate position embeddings to match sequence length
                new_grid_size = int(math.sqrt(x.size(1) - 1))  # -1 for class token
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens.permute(0, 3, 1, 2),
                    size=(new_grid_size, new_grid_size),
                    mode='bilinear',
                    align_corners=False
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                
                # Add class token position embedding back
                new_pos_embedding = torch.cat([pos_embedding[:, :1], pos_tokens], dim=1)
                x = x + new_pos_embedding
            else:
                x = x + pos_embedding
            
            # Forward through encoder and classification head
            for block in self.encoder.layers:
                x = block(x)
            x = self.encoder.ln(x)
            x = x[:, 0]
            x = self.heads(x)
            
            return x

        # Update model components
        model.forward = types.MethodType(new_forward, model)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        hidden_dim = model.conv_proj.out_channels
        model.conv_proj = nn.Conv2d(1, hidden_dim, kernel_size=16, stride=16)

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