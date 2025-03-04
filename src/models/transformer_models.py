from torchvision.models import (

    # ViT variants
    vit_b_16, ViT_B_16_Weights,
    vit_b_32, ViT_B_32_Weights,
    vit_l_16, ViT_L_16_Weights,
    vit_l_32, ViT_L_32_Weights,
    vit_h_14, ViT_H_14_Weights,
)

from transformers import (
    ASTFeatureExtractor, ASTForAudioClassification, AutoModel, PreTrainedModel, AutoFeatureExtractor, Wav2Vec2FeatureExtractor
)
import math
import logging
import sys
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, IA3Config, AdaLoraConfig, OFTConfig, HRAConfig, TaskType
from peft.utils.peft_types import TaskType
from icecream import ic
import torch
import torch.nn as nn
import types
from typing import Dict, Tuple, Any, Optional, Union, Callable
import os
from configs import PEFTConfig, GeneralConfig, NoneClassifierConfig, NoneFullConfig
from src.configs.peft_config import (
    PEFTConfig, NoneClassifierConfig, NoneFullConfig, SSFConfig, LoraConfig as CustomLoraConfig,
    IA3Config as CustomIA3Config, AdaLoraConfig as CustomAdaLoraConfig,
    OFTConfig as CustomOFTConfig, HRAConfig as CustomHRAConfig, LNTuningConfig
)
from src.models.ssf_adapter import apply_ssf_to_model


def apply_peft(model: nn.Module, peft_config: PEFTConfig, general_config: GeneralConfig) -> nn.Module:
    """
    Apply PEFT to the model based on the peft_config and general_config
    """
    adapter_type = general_config.adapter_type
    if isinstance(peft_config, NoneClassifierConfig):
        # Only train the classification head
        # Freeze all parameters except those in the classification head
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classification head
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        
    elif isinstance(peft_config, (NoneFullConfig)):
        # turn on all the parameters
        for param in model.parameters():
            param.requires_grad = True
    
    elif isinstance(peft_config, SSFConfig):
        # Apply SSF adapter to the model
        model = apply_ssf_to_model(
            model=model,
            init_scale=peft_config.init_scale,
            init_shift=peft_config.init_shift,
            verbose=True
        )
        
    elif isinstance(peft_config, (LoraConfig, IA3Config, AdaLoraConfig, OFTConfig, HRAConfig, LNTuningConfig)):
        try:
            # Generic approach to handle ModulesToSaveWrapper issue
            # This will work for any model that uses ModulesToSaveWrapper
            # Recursively expose attributes from original_module to parent wrapper
            def expose_attributes(module, prefix=""):
                if hasattr(module, 'original_module'):
                    # For each attribute in the original module, expose it to the wrapper
                    for name, child in module.original_module.named_children():
                        # Skip if the attribute already exists
                        if not hasattr(module, name):
                            setattr(module, name, child)
                    
                    # Also expose methods and attributes that aren't modules
                    for name in dir(module.original_module):
                        if not name.startswith('_') and not hasattr(module, name):
                            try:
                                setattr(module, name, getattr(module.original_module, name))
                            except (AttributeError, TypeError):
                                # Skip if we can't set the attribute
                                pass
                
                # Recursively process child modules
                for name, child in module.named_children():
                    new_prefix = f"{prefix}.{name}" if prefix else name
                    expose_attributes(child, new_prefix)
            
            # Apply the attribute exposure to the model
            expose_attributes(model)
            
            # get peft model given peft config
            model = get_peft_model(model, peft_config)
        except Exception as e:
            # Log the error but continue with the original model
            print(f"Error applying PEFT ({type(peft_config).__name__}): {str(e)}")
            print("Falling back to non-PEFT model")
            # Turn on all parameters for training as a fallback
            for param in model.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Invalid PEFT config type: {type(peft_config)} with a {adapter_type} adapter type")
    
    return model


class TransformerModel:
    
    peft_type = ['lora', 'adalora', 'hra', 'ia3', 'oft', 'layernorm', 
                 'none-full', 'none-classifier', 'ssf', 'bitfit']
    
    transformer_models = ['ast', 'mert','vit']
    
    @staticmethod
    def _create_ast_model(num_classes: int, CACHE_DIR: str, general_config: GeneralConfig, peft_config: Optional[PEFTConfig] = None) -> nn.Module:
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
        
        # Update the classifier in a generic way that works with any structure
        def update_classifier(module, in_features=None):
            # If this is a Linear layer with the right output dimension, update it
            if isinstance(module, nn.Linear) and hasattr(module, 'out_features'):
                if module.out_features != num_classes:
                    if in_features is None:
                        in_features = module.in_features
                    return nn.Linear(in_features, num_classes)
            return module
        
        # Find and update the classifier
        if hasattr(model, 'classifier'):
            # Handle direct classifier
            if hasattr(model.classifier, 'dense') and isinstance(model.classifier.dense, nn.Linear):
                in_features = model.classifier.dense.in_features
                model.classifier.dense = nn.Linear(in_features, num_classes)
            # Handle wrapped classifier
            elif hasattr(model.classifier, 'original_module'):
                if hasattr(model.classifier.original_module, 'dense') and isinstance(model.classifier.original_module.dense, nn.Linear):
                    in_features = model.classifier.original_module.dense.in_features
                    model.classifier.original_module.dense = nn.Linear(in_features, num_classes)
        
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

        
        model = apply_peft(model, peft_config, general_config)

        
        return model, feature_extractor
        
    
    @staticmethod
    def _create_mert_model(num_classes, CACHE_DIR: str, general_config: GeneralConfig, peft_config: Optional[PEFTConfig] = None) -> nn.Module:
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
        
        
        model = apply_peft(model, peft_config, general_config)
        
        return model, feature_extractor
    
    
    @staticmethod
    def _create_vit_model(model_type: str, num_classes: int, input_shape: Tuple[int, int, int], general_config: GeneralConfig, peft_config: Optional[PEFTConfig] = None) -> nn.Module:
        """
        Create a ViT model.
        """
        # Map config model type to actual model function
        vit_models = {
            'vit_b_16': (vit_b_16, ViT_B_16_Weights.DEFAULT),
            'vit_b_32': (vit_b_32, ViT_B_32_Weights.DEFAULT),
            'vit_l_16': (vit_l_16, ViT_L_16_Weights.DEFAULT),
            'vit_l_32': (vit_l_32, ViT_L_32_Weights.DEFAULT),
            'vit_h_14': (vit_h_14, ViT_H_14_Weights.DEFAULT),
        }
        
        if model_type not in vit_models:
            raise ValueError(f"Unsupported ViT model type: {model_type}")
            
        model_fn, weights = vit_models[model_type]
        model = model_fn(weights=weights)
    
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
        
        model = apply_peft(model, peft_config, general_config)
    
        return model
    