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
import traceback


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
            target_modules=peft_config.target_modules,
            expected_input_channels=1,  # Audio data typically has 1 channel
            verbose=True
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
            # Build target_modules list from reliable full module paths
            new_target_modules = []
            found_classifier = False
            
            # Collect all Linear layers with their full paths
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Only add modules with valid names (no ModulesToSaveWrapper in path)
                    if 'ModulesToSaveWrapper' not in name:
                        new_target_modules.append(name)
                        
                        # Track if we found the classifier.dense
                        if name == 'classifier.dense':
                            found_classifier = True
            
            # If classifier.dense wasn't found but we need it
            if not found_classifier and hasattr(model, 'classifier'):
                # Handle the case where classifier is a ModulesToSaveWrapper
                if hasattr(model.classifier, 'original_module') and hasattr(model.classifier.original_module, 'dense'):
                    if isinstance(model.classifier.original_module.dense, nn.Linear):
                        new_target_modules.append('classifier.original_module.dense')
                
                # Also try modules_to_save path
                if hasattr(model.classifier, 'modules_to_save'):
                    for key, module in model.classifier.modules_to_save.items():
                        if hasattr(module, 'dense') and isinstance(module.dense, nn.Linear):
                            path = f'classifier.modules_to_save.{key}.dense'
                            new_target_modules.append(path)
            
            # Override peft_config's target_modules with our new list
            peft_config.target_modules = new_target_modules
            
            # De-duplicate target modules
            seen = set()
            unique_modules = []
            for module in peft_config.target_modules:
                if module not in seen:
                    seen.add(module)
                    unique_modules.append(module)
            peft_config.target_modules = unique_modules
            
            # Create a mapping of model attribute paths
            def set_attribute_path_mapping(model):
                """Create an attribute map to help PEFT find modules directly"""
                module_map = {}
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Store the direct reference to this module
                        module_map[name] = module
                
                # Attach this map to the model for PEFT to use
                if not hasattr(model, '_peft_path_map'):
                    model._peft_path_map = module_map
            
            # Apply the path mapping to help PEFT
            set_attribute_path_mapping(model)
            
            # Monkey patch get_submodule to use our map if it exists
            original_get_submodule = nn.Module.get_submodule
            
            def patched_get_submodule(self, target):
                """Patched get_submodule that checks our path map first"""
                if hasattr(self, '_peft_path_map') and target in self._peft_path_map:
                    return self._peft_path_map[target]
                return original_get_submodule(self, target)
            
            # Apply the monkey patch
            nn.Module.get_submodule = patched_get_submodule
            
            # Get PEFT model with the updated configuration
            model = get_peft_model(model, peft_config)
            
            # Restore original method after we're done
            nn.Module.get_submodule = original_get_submodule
            
        except Exception as e:
            # In case of error, restore original get_submodule if we patched it
            if 'original_get_submodule' in locals():
                nn.Module.get_submodule = original_get_submodule
            raise e
    else:
        raise ValueError(f"Invalid PEFT config type: {type(peft_config)} with a {adapter_type} adapter type")
    
    # Print trainable parameter summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
    
    return model


def update_classifier(model, num_classes):
    """
    Update the classifier of a model to match the specified number of classes.
    Works with various model architectures by handling different classifier structures.
    """
    # Update the model config
    if hasattr(model, 'config'):
        model.config.num_labels = num_classes
    
    # Handle direct classifier
    if hasattr(model, 'classifier'):
        # Case 1: Simple classifier with dense attribute
        if hasattr(model.classifier, 'dense') and isinstance(model.classifier.dense, nn.Linear):
            in_features = model.classifier.dense.in_features
            model.classifier.dense = nn.Linear(in_features, num_classes)
            print(f"Updated classifier.dense: {in_features} -> {num_classes}")
        
        # Case 2: ModulesToSaveWrapper with original_module
        elif hasattr(model.classifier, 'original_module'):
            if hasattr(model.classifier.original_module, 'dense') and isinstance(model.classifier.original_module.dense, nn.Linear):
                in_features = model.classifier.original_module.dense.in_features
                model.classifier.original_module.dense = nn.Linear(in_features, num_classes)
                print(f"Updated classifier.original_module.dense: {in_features} -> {num_classes}")
        
        # Case 3: ModulesToSaveWrapper with modules_to_save
        elif hasattr(model.classifier, 'modules_to_save'):
            for key in model.classifier.modules_to_save:
                module = model.classifier.modules_to_save[key]
                if hasattr(module, 'dense') and isinstance(module.dense, nn.Linear):
                    in_features = module.dense.in_features
                    module.dense = nn.Linear(in_features, num_classes)
                    print(f"Updated classifier.modules_to_save.{key}.dense: {in_features} -> {num_classes}")
    
    # Handle other classifier structures
    else:
        # Look for classifier-like modules
        for name, module in model.named_modules():
            if 'classifier' in name.lower() or 'head' in name.lower() or 'output' in name.lower():
                if isinstance(module, nn.Linear) and module.out_features != num_classes:
                    in_features = module.in_features
                    parent_name = '.'.join(name.split('.')[:-1])
                    module_name = name.split('.')[-1]
                    parent = model
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, module_name, nn.Linear(in_features, num_classes))
                    print(f"Updated {name}: {in_features} -> {num_classes}")


class TransformerModel:
    
    peft_type = ['lora', 'adalora', 'hra', 'ia3', 'oft', 'layernorm', 
                 'none-full', 'none-classifier', 'ssf', 'bitfit']
    
    transformer_models = [
        'ast', 'mert', 
        'vit-base', 'vit-large',
        'deit-tiny', 'deit-small', 'deit-base',
        'deit-tiny-distil', 'deit-small-distil', 'deit-base-distil'
    ]  # Updated to include both ViT and DeiT variants
    
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
        
        # Create a logger for this function
        logger = logging.getLogger("AST_Model_Creation")
        logger.setLevel(logging.DEBUG)
        
        # Define the pretrained model to use
        pretrained_AST_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
        
        # Check if we should train from scratch
        from_scratch = getattr(general_config, 'from_scratch', False)
        
        print(f"Creating AST model with from_scratch={from_scratch}")
        
        if from_scratch:
            # For from scratch training, we need to create the model with random initialization
            # NOTE: We still use the pretrained model name here, but ONLY to get the architecture
            # configuration (num layers, hidden dims, etc.), NOT to load pretrained weights
            try:
                from transformers import ASTConfig
                # Try local files first, but don't fail if they don't exist
                try:
                    config = ASTConfig.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
                except (OSError, AttributeError, Exception):
                    # If local files don't exist or there's any other error, download
                    config = ASTConfig.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
            except Exception as e:
                print(f"Error loading AST config: {e}")
                # Fallback to downloading
                config = ASTConfig.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
            
            # Set number of labels
            config.num_labels = num_classes
            
            # Create model from config (this will use random initialization, NO pretrained weights)
            model = ASTForAudioClassification(config)
            print("AST model created from scratch with random initialization")
        else:
            # Load pretrained model with weights
            try:
                # Try local files first, but don't fail if they don't exist
                try:
                    model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
                except (OSError, AttributeError, Exception):
                    # If local files don't exist or there's any other error, download
                    model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
            except Exception as e:
                print(f"Error loading AST model: {e}")
                # Fallback to downloading
                model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
            
            # Update the classifier to match our number of classes
            update_classifier(model, num_classes)
        
        # Special handling for AST model with ModulesToSaveWrapper
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'modules_to_save'):
            # For AST models with ModulesToSaveWrapper, we need to ensure the classifier
            # is properly set up for PEFT
            print("Setting up AST model classifier for PEFT compatibility")
            
            # Create direct references to the dense layer in the classifier
            # This is crucial for PEFT to find the dense layer
            if hasattr(model.classifier.modules_to_save, 'default') and hasattr(model.classifier.modules_to_save.default, 'dense'):
                # Create a direct reference to the dense layer
                model.classifier.dense = model.classifier.modules_to_save.default.dense
                print("Created direct reference to classifier.dense for PEFT compatibility")
            
            # Also create a reference in the original_module if it exists
            if hasattr(model.classifier, 'original_module') and hasattr(model.classifier.original_module, 'dense'):
                model.dense = model.classifier.original_module.dense
                print("Created direct reference to classifier.original_module.dense for PEFT compatibility")
        
        # Define a new forward method for the embeddings to handle our input format
        def new_embeddings_forward(self, input_values):
            # Original code from AST model
            batch_size = input_values.shape[0]
            
            if input_values.dim() > 4:
                raise ValueError(f"Input has {input_values.dim()} dimensions, expected 4 dimensions")
            
            # When we have a 3D input (batch_size, channels, sequence), we need to add the height dimension
            if input_values.dim() == 3:
                # Reshape to (batch_size, channels, height=1, width=sequence)
                input_values = input_values.unsqueeze(2)
            
            # Get patch embeddings
            embeddings = self.patch_embeddings(input_values)
            
            # Check if sequence length matches expected length for position embeddings
            expected_seq_length = self.position_embeddings.shape[1]
            current_seq_length = embeddings.shape[1]
            
            if current_seq_length != expected_seq_length:
                # Use interpolation to adapt embeddings to the expected sequence length
                # First transpose to [batch_size, hidden_dim, seq_len]
                embeddings = embeddings.transpose(1, 2)
                
                # Use 1D interpolation to get the correct sequence length
                embeddings = F.interpolate(
                    embeddings, 
                    size=expected_seq_length, 
                    mode='linear', 
                    align_corners=False
                )
                
                # Transpose back to [batch_size, seq_len, hidden_dim]
                embeddings = embeddings.transpose(1, 2)
            
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
                
            # Check if input has 5 dimensions [batch, channels, height, extra_dim, width]
            if len(x.shape) == 5:
                # Get the dimensions
                batch_size, channels, height, extra_dim, width = x.shape
                
                # Reshape to 4D tensor that conv2d can accept
                try:
                    x = x.reshape(batch_size, channels, height * extra_dim, width)
                except Exception:
                    # Alternative approach: try to squeeze out the extra dimension if it's 1
                    if extra_dim == 1:
                        try:
                            x = x.squeeze(3)  # Remove the 4th dimension (index 3)
                        except Exception:
                            # Last resort: try to view the tensor differently
                            try:
                                x = x.view(batch_size, channels, height, width)
                            except Exception:
                                pass
            
            try:
                return original_forward(x)
            except Exception as e:
                raise
        
        # Replace the forward method
        model.forward = types.MethodType(new_forward, model)
        
        # Also patch the patch_embeddings projection method directly
        original_patch_embeddings_forward = model.audio_spectrogram_transformer.embeddings.patch_embeddings.forward
        
        def new_patch_embeddings_forward(self, input_values):
            # logger.debug(f"Patch embeddings input shape: {input_values.shape}")
            
            # Handle 5D input directly at the patch embeddings level
            if len(input_values.shape) == 5:
                # logger.debug("Fixing 5D input at patch embeddings level")
                batch_size, channels, height, extra_dim, width = input_values.shape
                
                # Try different approaches
                if extra_dim == 1:
                    # If extra dimension is 1, just squeeze it out
                    input_values = input_values.squeeze(3)
                    # logger.debug(f"Squeezed to shape: {input_values.shape}")
                else:
                    # Otherwise reshape
                    try:
                        input_values = input_values.reshape(batch_size, channels, height * extra_dim, width)
                        # logger.debug(f"Reshaped to: {input_values.shape}")
                    except Exception as e:
                        pass
                        # logger.error(f"Error reshaping in patch embeddings: {e}")
            
            # Call original method with fixed input
            try:
                result = original_patch_embeddings_forward(input_values)
                # logger.debug(f"Patch embeddings output shape: {result.shape}")
                return result
            except Exception as e:
                # logger.error(f"Error in original patch embeddings forward: {e}")
                # logger.error(f"Input shape: {input_values.shape}")
                # Try one more approach if it fails
                if len(input_values.shape) == 4:
                    # logger.debug("Attempting alternative approach for 4D input")
                    # Try to use the projection directly
                    try:
                        result = self.projection(input_values).flatten(2).transpose(1, 2)
                        # logger.debug(f"Direct projection successful, shape: {result.shape}")
                        return result
                    except Exception as e2:
                        pass
                        # logger.error(f"Direct projection failed: {e2}")
                raise
        
        # Replace the patch embeddings forward method
        model.audio_spectrogram_transformer.embeddings.patch_embeddings.forward = types.MethodType(
            new_patch_embeddings_forward, 
            model.audio_spectrogram_transformer.embeddings.patch_embeddings
        )
        
        # Load feature extractor (this doesn't need to change based on from_scratch)
        try:
            feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
        except (OSError, AttributeError, Exception):
            # If local files don't exist or there's any other error, download
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
        
        # Check if we should train from scratch
        from_scratch = getattr(general_config, 'from_scratch', False)
        
        print(f"Creating MERT model with from_scratch={from_scratch}")
        
        if from_scratch:
            # For from scratch training, we need to create the model with random initialization
            # NOTE: We still use the pretrained model name here, but ONLY to get the architecture
            # configuration (num layers, hidden dims, etc.), NOT to load pretrained weights
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(pretrained_MERT_model, cache_dir=CACHE_DIR, trust_remote_code=True, local_files_only=True)
            except (OSError, AttributeError, Exception):
                # If local files don't exist or there's any other error, download
                config = AutoConfig.from_pretrained(pretrained_MERT_model, trust_remote_code=True, cache_dir=CACHE_DIR)
            
            # Create model from config (this will use random initialization, NO pretrained weights)
            model = AutoModel.from_config(config, trust_remote_code=True)
            print("MERT model created from scratch with random initialization")
        else:
            # Load pretrained model with weights
            try:
                model = AutoModel.from_pretrained(pretrained_MERT_model, cache_dir=CACHE_DIR, trust_remote_code=True, local_files_only=True)
            except (OSError, AttributeError, Exception):
                # If local files don't exist or there's any other error, download
                model = AutoModel.from_pretrained(pretrained_MERT_model, trust_remote_code=True, cache_dir=CACHE_DIR)
        
        # Load feature extractor (this doesn't need to change based on from_scratch)
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_MERT_model, cache_dir=CACHE_DIR, local_files_only=True)
        except (OSError, AttributeError, Exception):
            # If local files don't exist or there's any other error, download
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
    def _create_vit_model(num_classes: int, CACHE_DIR: str, general_config: GeneralConfig, peft_config: Optional[PEFTConfig] = None, model_type: str = "base") -> Tuple[nn.Module, Any]:
        """
        Create a ViT model using Hugging Face's implementation.

        Args:
            num_classes (int): Number of output classes
            CACHE_DIR (str): Directory to cache model files
            general_config (GeneralConfig): General configuration object
            peft_config (Optional[PEFTConfig]): PEFT configuration if using parameter efficient fine-tuning
            model_variant (str): Which ViT variant to use - "base" or "large". Defaults to "base"

        TODO: Might need to change the model instead of the feature extractor. AST for example uses a 1-d Grayscale input, taking the avg. 
        of the color channels for initialization. This would have the advantage of standardizing the spectrogram data across all models.
        
        This implementation uses the standard 3-channel approach, where grayscale
        spectrograms are converted to RGB format before processing. This is handled
        in the feature_extraction method in util.py.
        
        The model automatically handles resizing of input tensors to the required 224x224 size.
        """
        # Map model types str to HuggingFace checkpoint str
        model_mapping = {
            "vit-base": "google/vit-base-patch16-224",
            "vit-large": "google/vit-large-patch16-224"
        }
        
        if model_type not in model_mapping:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(model_mapping.keys())}")
            
        model_checkpoint = model_mapping[model_type]
        
        # Check if we should train from scratch
        from_scratch = getattr(general_config, 'from_scratch', False)
        
        print(f"Attempting to load ViT model: {model_checkpoint} with {num_classes} classes and from_scratch={from_scratch}.")

        if from_scratch:
            # For from scratch training, we need to create the model with random initialization
            # NOTE: We still use the pretrained model name here, but ONLY to get the architecture
            # configuration (num layers, hidden dims, etc.), NOT to load pretrained weights
            try:
                from transformers import ViTConfig
                config = ViTConfig.from_pretrained(model_checkpoint, cache_dir=CACHE_DIR, local_files_only=True)
            except OSError:
                config = ViTConfig.from_pretrained(model_checkpoint, cache_dir=CACHE_DIR)
            
            # Set number of labels
            config.num_labels = num_classes
            
            # Create model from config (this will use random initialization, NO pretrained weights)
            model = ViTForImageClassification(config)
            print("ViT model created from scratch with random initialization")
            
        else:
            # Load pretrained model with weights
            try:
                print("Loading model with local_files_only=True...")
                model = ViTForImageClassification.from_pretrained(
                    model_checkpoint,
                    num_labels=num_classes,
                    cache_dir=CACHE_DIR,
                    local_files_only=True,
                    ignore_mismatched_sizes=True
                )
                print("Model loaded successfully with local_files_only=True.")
                
            except Exception as e:
                print(f"Failed to load model with local_files_only=True, trying to download: {e}")
                model = ViTForImageClassification.from_pretrained(
                    model_checkpoint,
                    num_labels=num_classes,
                    cache_dir=CACHE_DIR,
                    ignore_mismatched_sizes=True
                )
                print("Model downloaded successfully.")

        # Load the processor (this doesn't change based on from_scratch)
        try:
            # Load the processor with explicit resize configuration to 224x224
            processor = ViTImageProcessor.from_pretrained(
                model_checkpoint,
                cache_dir=CACHE_DIR,
                local_files_only=True,
                size={"height": 224, "width": 224},  # Force resize to 224x224
                do_resize=True,
                do_normalize=True
            )
            print("Processor loaded successfully with local_files_only=True and configured for 224x224 images.")
            
        except Exception as e:
            print(f"Failed to load processor with local_files_only=True, trying to download: {e}")
            processor = ViTImageProcessor.from_pretrained(
                model_checkpoint,
                cache_dir=CACHE_DIR,
                size={"height": 224, "width": 224},  # Force resize to 224x224
                do_resize=True,
                do_normalize=True
            )
            print("Processor downloaded successfully and configured for 224x224 images.")

        # Debugging model architecture
        # print("Model architecture:")
        # print(model)
        
        # Update model configuration to accept inputs of any size
        print("Updating model configuration to handle dynamic input sizes...")
        if hasattr(model, 'config'):
            # Save the original image_size
            original_image_size = model.config.image_size if hasattr(model.config, 'image_size') else 224
            print(f"Original image_size in config: {original_image_size}")
            
            # Explicitly set model to handle 224x224 images
            model.config.image_size = 224
            print(f"Updated image_size in config: {model.config.image_size}")
        
        # Create a custom forward method to handle different input parameter names
        original_forward = model.forward
        
        def new_forward(self, pixel_values=None, input_ids=None, inputs_embeds=None, x=None, **kwargs):
            """
            Custom forward method that handles different input parameter names and is compatible with PEFT.
            
            This method accepts all common input parameter names and routes them correctly:
            - pixel_values: Standard ViT input name
            - input_ids: Used by some PEFT models
            - inputs_embeds: Used by some transformer models
            - x: Generic input used in many custom implementations
            
            It also automatically converts 1-channel inputs to 3-channel to match model expectations.
            """
            # Add debug information
            # print(f"ViT model forward called with: pixel_values={pixel_values is not None}, "
            #   f"input_ids={input_ids is not None}, x={x is not None}, "
            #   f"inputs_embeds={inputs_embeds is not None}, kwargs={list(kwargs.keys())}")
            
            # Determine which input to use
            if pixel_values is not None:
                actual_input = pixel_values
                if hasattr(actual_input, 'shape'):
                    pass
                    # print(f"pixel_values input shape: {actual_input.shape}")
            elif input_ids is not None:
                actual_input = input_ids
                if hasattr(actual_input, 'shape'):
                    pass
                    # print(f"input_ids input shape: {actual_input.shape}")
            elif x is not None:
                actual_input = x
                if hasattr(actual_input, 'shape'):
                    pass
                    # print(f"x input shape: {actual_input.shape}")
            elif inputs_embeds is not None:
                actual_input = inputs_embeds
                if hasattr(actual_input, 'shape'):
                    pass
                    # print(f"inputs_embeds input shape: {actual_input.shape}")
            else:
                raise ValueError("No valid input provided to ViT model. Expected one of: pixel_values, input_ids, x, or inputs_embeds")
            
            # Print the overall input shape for debugging
            if hasattr(actual_input, 'shape'):
                # print(f"Selected input shape: {actual_input.shape}")
                if len(actual_input.shape) >= 4:
                    # For typical image input (B, C, H, W)
                    pass
                    # print(f"Image dimensions (Height × Width): {actual_input.shape[-2]}×{actual_input.shape[-1]}")
            
            # Check if actual_input is a tensor and has 1 channel instead of 3
            if isinstance(actual_input, torch.Tensor) and len(actual_input.shape) == 4:
                # print(f"Input tensor shape: {actual_input.shape}")

                
                # Handle channel count - convert from 1 to 3 channels if needed
                if actual_input.shape[1] == 1:
                    # print("Converting 1-channel input to 3-channel by repeating the channel")
                    actual_input = actual_input.repeat(1, 3, 1, 1)
                    # print(f"Converted input shape: {actual_input.shape}")
                
                # Handle input size - resize all inputs to 224x224 using interpolation
                if actual_input.shape[2] != 224 or actual_input.shape[3] != 224:
                    # print(f"Resizing input from {actual_input.shape[2]}x{actual_input.shape[3]} to 224x224")
                    # Use interpolate to resize the tensor to 224x224
                    actual_input = torch.nn.functional.interpolate(
                        actual_input, 
                        size=(224, 224), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    # print(f"Resized input shape: {actual_input.shape}")
            
            # Remove attention_mask from kwargs if it exists - ViT doesn't use it
            vit_kwargs = {k: v for k, v in kwargs.items() if k not in ['attention_mask']}
            
            # Add debug info to see what's being passed to the original forward
            # print(f"Passing to ViT: pixel_values={actual_input.shape if hasattr(actual_input, 'shape') else None}, kwargs={list(vit_kwargs.keys())}")
            
            try:
                return original_forward(pixel_values=actual_input, **vit_kwargs)
            except Exception as e:
                # print(f"Forward pass error: {str(e)}")
                # print(f"Input shape: {actual_input.shape if hasattr(actual_input, 'shape') else None}, Input type: {type(actual_input)}")
                
                # Get model configuration to help debug
                if hasattr(self, 'config'):
                    print(f"Model config: {self.config}")
                
                # Re-raise the exception for proper error handling
                raise
        
        # Replace the forward method
        model.forward = types.MethodType(new_forward, model)
        # print("Custom forward method added to handle different input parameter names and filter unsupported parameters")

        # Note: We're using the standard 3-channel approach where grayscale spectrograms
        # are converted to RGB in the feature_extraction method in util.py.
        # This is simpler and doesn't require modifying the model architecture.
        
        # If you want to use the 1-channel approach instead, uncomment the following code:
        """
        # 1-CHANNEL APPROACH (ALTERNATIVE)
        # This modifies the model to accept 1-channel input directly.
        
        # First, modify the configuration to expect 1 channel
        from transformers import ViTConfig
        config = ViTConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            cache_dir=CACHE_DIR
        )
        config.num_channels = 1  # Set number of channels to 1
        print(f"Modified config to use {config.num_channels} channels")
        
        # Create model with modified config
        model = ViTForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Get the original projection layer
        original_projection = model.vit.embeddings.patch_embeddings.projection
        print(f"Original projection layer: {original_projection}")
        
        # Modify the projection layer to accept 1-channel input
        model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1 for grayscale
            out_channels=original_projection.out_channels,
            kernel_size=original_projection.kernel_size,
            stride=original_projection.stride,
            padding=original_projection.padding
        )
        
        # Initialize the weights of the new projection layer
        with torch.no_grad():
            original_weights = original_projection.weight.data
            new_weights = original_weights.mean(dim=1, keepdim=True)
            model.vit.embeddings.patch_embeddings.projection.weight.data = new_weights
        
        # Double-check the configuration
        print(f"Model config num_channels: {model.config.num_channels}")
        print(f"Modified projection layer: {model.vit.embeddings.patch_embeddings.projection}")
        print("Model modified to accept 1-channel input")
        """

        # Apply PEFT configuration
        print("Applying PEFT configuration...")
        model = apply_peft(model, peft_config, general_config)
        print("PEFT configuration applied successfully.")

        return model, processor
    
    @staticmethod
    def _create_deit_model(num_classes: int, CACHE_DIR: str, general_config: GeneralConfig, peft_config: Optional[PEFTConfig] = None, model_type: str = "base") -> Tuple[nn.Module, Any]:
        """
        Create a DeiT model using Hugging Face's implementation.

        Args:
            num_classes (int): Number of output classes
            CACHE_DIR (str): Directory to cache model files
            general_config (GeneralConfig): General configuration object
            peft_config (Optional[PEFTConfig]): PEFT configuration if using parameter efficient fine-tuning
            model_type (str): Which DeiT variant to use. Options:
                - deit-tiny
                - deit-small
                - deit-base
                - deit-tiny-distil
                - deit-small-distil
                - deit-base-distil
        """
        # Map model types to HuggingFace checkpoint names
        model_mapping = {
            "deit-tiny": "facebook/deit-tiny-patch16-224",
            "deit-small": "facebook/deit-small-patch16-224",
            "deit-base": "facebook/deit-base-patch16-224",
            "deit-tiny-distil": "facebook/deit-tiny-distilled-patch16-224",
            "deit-small-distil": "facebook/deit-small-distilled-patch16-224",
            "deit-base-distil": "facebook/deit-base-distilled-patch16-224"
        }
        
        if model_type not in model_mapping:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(model_mapping.keys())}")
            
        model_checkpoint = model_mapping[model_type]
        
        # Check if we should train from scratch
        from_scratch = getattr(general_config, 'from_scratch', False)
        
        print(f"Attempting to load DeiT model: {model_checkpoint} with {num_classes} classes and from_scratch={from_scratch}.")

        if from_scratch:
            # For from scratch training, we need to create the model with random initialization
            # NOTE: We still use the pretrained model name here, but ONLY to get the architecture
            # configuration (num layers, hidden dims, etc.), NOT to load pretrained weights
            try:
                from transformers import ViTConfig
                config = ViTConfig.from_pretrained(model_checkpoint, cache_dir=CACHE_DIR, local_files_only=True)
            except OSError:
                config = ViTConfig.from_pretrained(model_checkpoint, cache_dir=CACHE_DIR)
            
            # Set number of labels
            config.num_labels = num_classes
            
            # Create model from config (this will use random initialization, NO pretrained weights)
            model = ViTForImageClassification(config)
            print("DeiT model created from scratch with random initialization")
            
        else:
            # Load pretrained model with weights
            try:
                print("Loading model with local_files_only=True...")
                model = ViTForImageClassification.from_pretrained(
                    model_checkpoint,
                    num_labels=num_classes,
                    cache_dir=CACHE_DIR,
                    local_files_only=True,
                    ignore_mismatched_sizes=True
                )
                print("Model loaded successfully with local_files_only=True.")
                
            except Exception as e:
                print(f"Failed to load model with local_files_only=True, trying to download: {e}")
                model = ViTForImageClassification.from_pretrained(
                    model_checkpoint,
                    num_labels=num_classes,
                    cache_dir=CACHE_DIR,
                    ignore_mismatched_sizes=True
                )
                print("Model downloaded successfully.")

        # Load the processor (this doesn't change based on from_scratch)
        try:
            # Load the processor with explicit resize configuration to 224x224
            processor = ViTImageProcessor.from_pretrained(
                model_checkpoint,
                cache_dir=CACHE_DIR,
                local_files_only=True,
                size={"height": 224, "width": 224},  # Force resize to 224x224
                do_resize=True,
                do_normalize=True
            )
            print("Processor loaded successfully with local_files_only=True and configured for 224x224 images.")
            
        except Exception as e:
            print(f"Failed to load processor with local_files_only=True, trying to download: {e}")
            processor = ViTImageProcessor.from_pretrained(
                model_checkpoint,
                cache_dir=CACHE_DIR,
                size={"height": 224, "width": 224},  # Force resize to 224x224
                do_resize=True,
                do_normalize=True
            )
            print("Processor downloaded successfully and configured for 224x224 images.")

        # Debugging model architecture
        # print("Model architecture:")
        # print(model)
        
        # Update model configuration to accept inputs of any size
        print("Updating model configuration to handle dynamic input sizes...")
        if hasattr(model, 'config'):
            # Save the original image_size
            original_image_size = model.config.image_size if hasattr(model.config, 'image_size') else 224
            print(f"Original image_size in config: {original_image_size}")
            
            # Explicitly set model to handle 224x224 images
            model.config.image_size = 224
            print(f"Updated image_size in config: {model.config.image_size}")
        
        # Create a custom forward method to handle different input parameter names
        original_forward = model.forward
        
        def new_forward(self, pixel_values=None, input_ids=None, inputs_embeds=None, x=None, **kwargs):
            """
            Custom forward method that handles different input parameter names and is compatible with PEFT.
            
            This method accepts all common input parameter names and routes them correctly:
            - pixel_values: Standard DeiT input name
            - input_ids: Used by some PEFT models
            - inputs_embeds: Used by some transformer models
            - x: Generic input used in many custom implementations
            
            It also automatically converts 1-channel inputs to 3-channel to match model expectations.
            """
            # Determine which input to use
            if pixel_values is not None:
                actual_input = pixel_values
            elif input_ids is not None:
                actual_input = input_ids
            elif x is not None:
                actual_input = x
            elif inputs_embeds is not None:
                actual_input = inputs_embeds
            else:
                raise ValueError("No valid input provided to DeiT model. Expected one of: pixel_values, input_ids, x, or inputs_embeds")
            
            # Check if actual_input is a tensor and has 1 channel instead of 3
            if isinstance(actual_input, torch.Tensor) and len(actual_input.shape) == 4:
                # Handle channel count - convert from 1 to 3 channels if needed
                if actual_input.shape[1] == 1:
                    actual_input = actual_input.repeat(1, 3, 1, 1)
                
                # Handle input size - resize all inputs to 224x224 using interpolation
                if actual_input.shape[2] != 224 or actual_input.shape[3] != 224:
                    actual_input = torch.nn.functional.interpolate(
                        actual_input, 
                        size=(224, 224), 
                        mode='bilinear', 
                        align_corners=False
                    )
            
            # Remove attention_mask from kwargs if it exists - DeiT doesn't use it
            deit_kwargs = {k: v for k, v in kwargs.items() if k not in ['attention_mask']}
            
            try:
                return original_forward(pixel_values=actual_input, **deit_kwargs)
            except Exception as e:
                # Get model configuration to help debug
                if hasattr(self, 'config'):
                    print(f"Model config: {self.config}")
                raise
        
        # Replace the forward method
        model.forward = types.MethodType(new_forward, model)

        # Apply PEFT configuration
        print("Applying PEFT configuration...")
        model = apply_peft(model, peft_config, general_config)
        print("PEFT configuration applied successfully.")

        return model, processor
    