import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple

class SSFLayer(nn.Module):
    """
    Scale-Shift Factor layer for parameter-efficient fine-tuning.
    
    This layer applies learnable scale and shift factors to the output of a target module.
    Original Paper: https://arxiv.org/abs/2110.11256
    """
    
    def __init__(
        self, 
        hidden_size: int,
        init_scale: float = 1.0,
        init_shift: float = 0.0
    ):
        super().__init__()
        # Initialize scale factor (gamma) with init_scale
        self.scale = nn.Parameter(torch.ones(hidden_size) * init_scale)
        # Initialize shift factor (beta) with init_shift
        self.shift = nn.Parameter(torch.ones(hidden_size) * init_shift)
    
    def forward(self, x):
        # Apply scale and shift: x_new = gamma * x + beta
        # Handling both 2D and 3D inputs
        if x.dim() == 2:
            return x * self.scale + self.shift
        elif x.dim() == 3:
            # For sequence data: [batch_size, seq_len, hidden_size]
            return x * self.scale.unsqueeze(0).unsqueeze(0) + self.shift.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")


def apply_ssf_to_model(
    model: nn.Module,
    init_scale: float = 1.0,
    init_shift: float = 0.0,
    verbose: bool = False
) -> nn.Module:
    """
    Apply Scale-Shift Factor layers to all linear modules in the model.
    
    Args:
        model: The model to modify
        init_scale: Initial value for scale factors
        init_shift: Initial value for shift factors
        verbose: Whether to print debug information
        
    Returns:
        Modified model with SSF layers
    """
    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Counter for modified modules
    modified_count = 0
    
    # This will wrap linear modules with SSF
    def _find_and_replace_linear_modules(module, path=""):
        nonlocal modified_count
        
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            
            # First recursively process all children
            _find_and_replace_linear_modules(child, child_path)
            
            # Check if this is a Linear module
            if isinstance(child, nn.Linear):
                # Get the output dimension of the module
                hidden_size = child.out_features
                
                # Create SSF wrapper with original linear module
                class SSFWrapper(nn.Module):
                    def __init__(self, module, ssf_layer):
                        super().__init__()
                        self.module = module
                        self.ssf_layer = ssf_layer
                        
                    def forward(self, *args, **kwargs):
                        output = self.module(*args, **kwargs)
                        return self.ssf_layer(output)
                
                # Create the SSF layer
                ssf_layer = SSFLayer(hidden_size, init_scale, init_shift)
                
                # Replace the module with the wrapped version
                wrapped_module = SSFWrapper(child, ssf_layer)
                setattr(module, name, wrapped_module)
                
                # Count successful modifications
                modified_count += 1
                if verbose:
                    print(f"Applied SSF to Linear module at {child_path} with hidden size {hidden_size}")
    
    # Start the recursive process to find and replace linear modules
    _find_and_replace_linear_modules(model)
    
    if verbose:
        print(f"Modified {modified_count} Linear modules with SSF")
    
    # Return the modified model
    return model 