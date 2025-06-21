"""
Utilities for merging and reparameterizing PEFT methods during training.

This module handles the transition between different PEFT methods, particularly
when switching from additive methods (that add parameters) to selective methods
(that only change which parameters are trainable).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging
from copy import deepcopy
from configs.peft_scheduling_config import requires_reparameterization, PEFT_CATEGORIZATION, PEFTCategory


logger = logging.getLogger(__name__)


class PEFTMerger:
    """Handles merging and reparameterization of PEFT methods during training."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize the PEFT merger.
        
        Args:
            model: The model to manage PEFT transitions for
        """
        self.model = model
        self.original_state_dict = None
        self.current_peft_method = None
        self.peft_history = []
        
    def save_base_model_state(self):
        """Save the base model state before applying any PEFT methods."""
        if self.original_state_dict is None:
            self.original_state_dict = deepcopy(self.model.state_dict())
            logger.info("Saved base model state for PEFT merging")
    
    def merge_additive_peft(self, peft_method: str) -> bool:
        """
        Merge additive PEFT parameters into the base model.
        
        Args:
            peft_method: The PEFT method to merge
            
        Returns:
            True if merge was successful, False otherwise
        """
        if not requires_reparameterization(peft_method):
            logger.info(f"PEFT method '{peft_method}' does not require merging")
            return True
        
        try:
            if peft_method in ['lora', 'adalora', 'hra', 'ia3', 'oft']:
                return self._merge_huggingface_peft()
            elif peft_method == 'ssf':
                return self._merge_ssf()
            elif peft_method == 'lorac':
                return self._merge_lorac()
            else:
                logger.warning(f"Unknown additive PEFT method: {peft_method}")
                return False
        except Exception as e:
            logger.error(f"Error merging PEFT method '{peft_method}': {e}")
            return False
    
    def _merge_huggingface_peft(self) -> bool:
        """Merge Hugging Face PEFT methods (LoRA, AdaLoRA, etc.)."""
        try:
            # Check if this is a PEFT model
            if hasattr(self.model, 'merge_and_unload'):
                logger.info("Merging Hugging Face PEFT adapter weights")
                self.model = self.model.merge_and_unload()
                return True
            elif hasattr(self.model, 'merge_adapter'):
                logger.info("Merging specific PEFT adapter")
                self.model.merge_adapter()
                return True
            else:
                logger.warning("Model does not appear to be a Hugging Face PEFT model")
                return False
        except Exception as e:
            logger.error(f"Error merging Hugging Face PEFT: {e}")
            return False
    
    def _merge_ssf(self) -> bool:
        """Merge SSF (Scale and Shift) parameters."""
        try:
            # Look for SSF wrapped modules
            merged_count = 0
            for name, module in self.model.named_modules():
                if hasattr(module, 'ssf_scale') and hasattr(module, 'ssf_shift'):
                    # Apply scale and shift to the underlying module
                    if hasattr(module, 'module') and hasattr(module.module, 'weight'):
                        # Scale the weights
                        module.module.weight.data *= module.ssf_scale.data
                        
                        # Add shift to bias if it exists, otherwise create bias
                        if hasattr(module.module, 'bias') and module.module.bias is not None:
                            module.module.bias.data += module.ssf_shift.data
                        else:
                            # Create bias parameter with shift values
                            module.module.bias = nn.Parameter(module.ssf_shift.data.clone())
                        
                        merged_count += 1
                        logger.debug(f"Merged SSF parameters for module: {name}")
            
            if merged_count > 0:
                logger.info(f"Successfully merged SSF parameters for {merged_count} modules")
                # Remove SSF wrappers and restore original modules
                self._unwrap_ssf_modules()
                return True
            else:
                logger.warning("No SSF modules found to merge")
                return False
                
        except Exception as e:
            logger.error(f"Error merging SSF parameters: {e}")
            return False
    
    def _merge_lorac(self) -> bool:
        """Merge LoRA-C parameters."""
        try:
            merged_count = 0
            if hasattr(self.model, 'lorac_layers'):
                for name, lorac_layer in self.model.lorac_layers.items():
                    # Get the original module
                    module_parts = name.split('.')
                    current_module = self.model
                    
                    # Navigate to the parent module
                    for part in module_parts[:-1]:
                        current_module = getattr(current_module, part)
                    
                    module_name = module_parts[-1]
                    original_module = getattr(current_module, module_name)
                    
                    # Merge LoRA-C weights: W = W_original + A @ B
                    if hasattr(lorac_layer, 'lora_A') and hasattr(lorac_layer, 'lora_B'):
                        delta_weight = lorac_layer.lora_A @ lorac_layer.lora_B
                        original_module.weight.data += delta_weight
                        merged_count += 1
                        logger.debug(f"Merged LoRA-C parameters for module: {name}")
                
                if merged_count > 0:
                    logger.info(f"Successfully merged LoRA-C parameters for {merged_count} modules")
                    # Clear the LoRA-C layers
                    self.model.lorac_layers.clear()
                    return True
            
            logger.warning("No LoRA-C layers found to merge")
            return False
            
        except Exception as e:
            logger.error(f"Error merging LoRA-C parameters: {e}")
            return False
    
    def _unwrap_ssf_modules(self):
        """Remove SSF wrappers and restore original modules."""
        try:
            modules_to_replace = {}
            
            # Find all SSF wrapped modules
            for name, module in self.model.named_modules():
                if hasattr(module, 'module') and hasattr(module, 'ssf_scale'):
                    modules_to_replace[name] = module.module
            
            # Replace wrapped modules with their original versions
            for name, original_module in modules_to_replace.items():
                module_parts = name.split('.')
                current_module = self.model
                
                # Navigate to the parent module
                for part in module_parts[:-1]:
                    current_module = getattr(current_module, part)
                
                # Replace the wrapped module
                setattr(current_module, module_parts[-1], original_module)
                
            logger.info(f"Unwrapped {len(modules_to_replace)} SSF modules")
            
        except Exception as e:
            logger.error(f"Error unwrapping SSF modules: {e}")
    
    def prepare_for_selective_peft(self, target_method: str):
        """
        Prepare the model for a selective PEFT method.
        
        Args:
            target_method: The selective PEFT method to prepare for
        """
        if requires_reparameterization(target_method):
            logger.warning(f"Method '{target_method}' is not a selective PEFT method")
            return
        
        # For selective methods, we just need to ensure all parameters are available
        # and then the apply_peft function will handle freezing/unfreezing
        logger.info(f"Model prepared for selective PEFT method: {target_method}")
    
    def transition_peft_method(
        self, 
        from_method: str, 
        to_method: str, 
        merge_previous: bool = True
    ) -> bool:
        """
        Handle transition between PEFT methods.
        
        Args:
            from_method: Current PEFT method
            to_method: Target PEFT method
            merge_previous: Whether to merge the previous method
            
        Returns:
            True if transition was successful, False otherwise
        """
        logger.info(f"Transitioning from '{from_method}' to '{to_method}' (merge={merge_previous})")
        
        # Save the current state if this is the first transition
        if self.original_state_dict is None:
            self.save_base_model_state()
        
        success = True
        
        # If the previous method was additive and we need to merge
        if requires_reparameterization(from_method) and merge_previous:
            success = self.merge_additive_peft(from_method)
            if success:
                logger.info(f"Successfully merged '{from_method}' parameters")
            else:
                logger.error(f"Failed to merge '{from_method}' parameters")
        
        # Prepare for the new method
        if not requires_reparameterization(to_method):
            self.prepare_for_selective_peft(to_method)
        
        # Update tracking
        if success:
            self.peft_history.append({
                'from_method': from_method,
                'to_method': to_method,
                'merged': merge_previous and requires_reparameterization(from_method)
            })
            self.current_peft_method = to_method
        
        return success
    
    def get_merge_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all PEFT transitions and merges.
        
        Returns:
            Dictionary containing merge history and statistics
        """
        total_transitions = len(self.peft_history)
        merged_transitions = sum(1 for h in self.peft_history if h.get('merged', False))
        
        return {
            'total_transitions': total_transitions,
            'merged_transitions': merged_transitions,
            'current_method': self.current_peft_method,
            'has_base_state': self.original_state_dict is not None,
            'history': self.peft_history.copy()
        }
    
    def save_merged_model(self, path: str):
        """
        Save the current model state with all merged parameters.
        
        Args:
            path: Path to save the model
        """
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Saved merged model to: {path}")
        except Exception as e:
            logger.error(f"Error saving merged model: {e}")
    
    def restore_base_model(self):
        """Restore the model to its original state before any PEFT methods."""
        if self.original_state_dict is not None:
            self.model.load_state_dict(self.original_state_dict)
            logger.info("Restored model to base state")
        else:
            logger.warning("No base model state saved, cannot restore")


def create_peft_merger(model: nn.Module) -> PEFTMerger:
    """
    Create a PEFT merger for the given model.
    
    Args:
        model: The model to create a merger for
        
    Returns:
        PEFTMerger instance
    """
    return PEFTMerger(model)


def validate_peft_transition(from_method: str, to_method: str) -> Tuple[bool, List[str]]:
    """
    Validate if a PEFT method transition is valid.
    
    Args:
        from_method: Current PEFT method
        to_method: Target PEFT method
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check if methods exist
    if from_method not in PEFT_CATEGORIZATION:
        warnings.append(f"Unknown source PEFT method: {from_method}")
    
    if to_method not in PEFT_CATEGORIZATION:
        warnings.append(f"Unknown target PEFT method: {to_method}")
    
    if warnings:
        return False, warnings
    
    from_category = PEFT_CATEGORIZATION[from_method]
    to_category = PEFT_CATEGORIZATION[to_method]
    
    # Warn about potentially problematic transitions
    if (from_category == PEFTCategory.ADDITIVE and 
        to_category == PEFTCategory.ADDITIVE and 
        from_method != to_method):
        warnings.append(
            f"Transitioning between additive methods ({from_method} -> {to_method}) "
            "requires merging the previous method"
        )
    
    if (from_category == PEFTCategory.SELECTIVE and 
        to_category == PEFTCategory.ADDITIVE):
        warnings.append(
            f"Transitioning from selective to additive method ({from_method} -> {to_method}) "
            "will add new parameters to the model"
        )
    
    return True, warnings 