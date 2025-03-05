# Removed torchvision.models import as we're using Hugging Face models

from transformers import (
    ASTFeatureExtractor, ASTForAudioClassification, AutoModel, PreTrainedModel, AutoFeatureExtractor, Wav2Vec2FeatureExtractor,
    # Add ViT imports from Hugging Face
    ViTForImageClassification, ViTImageProcessor
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
from configs.peft_config import (
    PEFTConfig, NoneClassifierConfig, NoneFullConfig, SSFConfig, LoraConfig as CustomLoraConfig,
    IA3Config as CustomIA3Config, AdaLoraConfig as CustomAdaLoraConfig,
    OFTConfig as CustomOFTConfig, HRAConfig as CustomHRAConfig, LNTuningConfig, BitFitConfig
)
from models.ssf_adapter import apply_ssf_to_model


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
                print(f"Unfreezing classifier parameters with shape {param.shape}")
        else:
            print("Warning: Model does not have a 'classifier' attribute. No parameters were unfrozen.")
            # Try to find classifier-like modules
            for name, module in model.named_modules():
                if 'classifier' in name.lower() or 'head' in name.lower() or 'output' in name.lower():
                    print(f"Found potential classifier module: {name}")
                    for param_name, param in module.named_parameters():
                        param.requires_grad = True
                        print(f"Unfreezing {name}.{param_name} with shape {param.shape}")
        
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
            verbose=False
        )
        
    elif isinstance(peft_config, BitFitConfig):
        # Implement BitFit: freeze all parameters except biases
        for name, param in model.named_parameters():
            param.requires_grad = False
            # Unfreeze bias terms
            if 'bias' in name:
                param.requires_grad = True
        
        # Always unfreeze classifier for the task
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
    
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

            raise e
    else:
        raise ValueError(f"Invalid PEFT config type: {type(peft_config)} with a {adapter_type} adapter type")
    
    # Print trainable parameter summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
    
    return model


class TransformerModel:
    
    peft_type = ['lora', 'adalora', 'hra', 'ia3', 'oft', 'layernorm', 
                 'none-full', 'none-classifier', 'ssf', 'bitfit']
    
    transformer_models = ['ast', 'mert', 'vit']  # Changed to just 'vit' instead of multiple variants
    
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
        def new_forward(self, x=None, input_ids=None, attention_mask=None, **kwargs):
            # Handle different input types - PEFT might pass input_ids instead of x
            if x is None and input_ids is not None:
                x = input_ids
            
            if x is None:
                raise ValueError("Either x or input_ids must be provided to the forward method")
                
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
        
        # Explicitly add a classifier head to the model
        hidden_size = model.config.hidden_size
        model.classifier = nn.Linear(hidden_size, num_classes)
        logger.info(f"Added classifier head with {hidden_size} -> {num_classes}")
        
        # Define new forward method to handle shape issues and device consistency
        def new_forward(self, x=None, input_ids=None, attention_mask=None, **kwargs):
            # Handle different input types - PEFT might pass input_ids instead of x
            if x is None and input_ids is not None:
                x = input_ids
            
            if x is None:
                raise ValueError("Either x or input_ids must be provided to the forward method")
                
            logger.debug(f"Input shape: {x.shape}, device: {x.device}")
            
            # Ensure input is float32
            x = x.float()
            
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
            
            # Ensure classifier is on same device as input
            self.classifier = self.classifier.to(x.device)
            
            # Forward pass through MERT
            outputs = original_forward(x)
            hidden_states = outputs[0]  # Get the last hidden state
            
            # Pool the output (mean pooling over sequence dimension)
            pooled_output = torch.mean(hidden_states, dim=1)
            
            # Ensure pooled output and classifier are on same device
            # Check if classifier is wrapped by SSFWrapper
            if hasattr(self.classifier, 'module') and hasattr(self.classifier.module, 'weight'):
                # If it's an SSFWrapper, use the device of the wrapped module's weight
                target_device = self.classifier.module.weight.device
            elif hasattr(self.classifier, 'weight'):
                # If it's a regular Linear layer, use its weight's device
                target_device = self.classifier.weight.device
            else:
                # Fallback to the classifier's device
                target_device = next(self.classifier.parameters()).device
                
            pooled_output = pooled_output.to(target_device)
            
            # Pass through classifier
            logits = self.classifier(pooled_output)
            
            return logits
        
        # Replace the forward method
        model.forward = types.MethodType(new_forward, model)
        
        # Apply PEFT configuration
        model = apply_peft(model, peft_config, general_config)
        
        # If using none-classifier, double-check that the classifier is trainable
        if isinstance(peft_config, NoneClassifierConfig):
            # Verify that all model parameters except classifier are frozen
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    logger.info(f"Ensuring classifier parameter {name} is trainable")
        
        return model, feature_extractor
    
    
    @staticmethod
    def _create_vit_model(model_type: str, num_classes: int, CACHE_DIR: str, input_shape: Tuple[int, int, int], general_config: GeneralConfig, peft_config: Optional[PEFTConfig] = None) -> Tuple[nn.Module, Any]:
        """
        Create a ViT model using Hugging Face's implementation.
        """
        # Use the Hugging Face ViT model
        model_name = "google/vit-large-patch16-224"
        
        try:
            # Try to load the model with local_files_only first
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
            processor = ViTImageProcessor.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
        except Exception as e:
            print(f"Failed to load model with local_files_only=True, trying to download: {e}")
            # If local loading fails, download the model
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                cache_dir=CACHE_DIR
            )
            processor = ViTImageProcessor.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR
            )
        
        # Add resize layer to match ViT's expected input size
        resize_layer = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # Save original forward method
        original_forward = model.forward
        
        # Define a new forward method to handle input shape issues
        def new_forward(self, x=None, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
            # Handle different input types - PEFT might pass input_ids instead of x
            if x is None and input_ids is not None:
                x = input_ids
            elif x is None and pixel_values is not None:
                x = pixel_values
            
            if x is None:
                raise ValueError("Either x, input_ids, or pixel_values must be provided to the forward method")
                
            # Handle input
            x = x.float()
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            # Resize input to match ViT's expected size
            x = resize_layer(x)
            
            # Convert to RGB if input is grayscale (1 channel)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            
            # Normalize the input as expected by the ViT model
            # The ViT model expects normalized images with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std
            
            # Pass through the model
            outputs = original_forward(pixel_values=x, **kwargs)
            
            return outputs
        
        # Replace the forward method
        model.forward = types.MethodType(new_forward, model)
        
        # Apply PEFT configuration
        model = apply_peft(model, peft_config, general_config)
        
        return model, processor
    