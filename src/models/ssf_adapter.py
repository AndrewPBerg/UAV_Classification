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
        # Handle different input dimensions
        try:
            if x.dim() == 2:
                # Linear layer output: [batch_size, hidden_size]
                return x * self.scale + self.shift
            elif x.dim() == 3:
                # For sequence data: [batch_size, seq_len, hidden_size]
                return x * self.scale.unsqueeze(0).unsqueeze(0) + self.shift.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 4:
                # For CNN data: [batch_size, channels, height, width]
                # Apply scale and shift to the channel dimension
                return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
            elif x.dim() == 5:
                # For 3D CNN data: [batch_size, channels, depth, height, width]
                return x * self.scale.view(1, -1, 1, 1, 1) + self.shift.view(1, -1, 1, 1, 1)
            else:
                raise ValueError(f"Unsupported input dimension: {x.dim()}, shape: {x.shape}, type: {type(x)}")
        except Exception as e:
            # Add more debug information
            print(f"Error in SSFLayer.forward: {str(e)}")
            print(f"Input shape: {x.shape}, Input type: {type(x)}")
            print(f"Scale shape: {self.scale.shape}, Shift shape: {self.shift.shape}")
            raise


def apply_ssf_to_model(
    model: nn.Module,
    init_scale: float = 1.0,
    init_shift: float = 0.0,
    target_modules: List[str] = ["linear"],
    expected_input_channels: int = 1,  # Default for audio is 1 channel
    verbose: bool = False
) -> nn.Module:
    """
    Apply Scale-Shift Factor layers to specified module types in the model.
    
    Args:
        model: The model to modify
        init_scale: Initial value for scale factors
        init_shift: Initial value for shift factors
        target_modules: List of module types to apply SSF to (e.g., ["linear", "conv2d"])
        expected_input_channels: Expected number of input channels for the first conv layer
        verbose: Whether to print debug information
        
    Returns:
        Modified model with SSF layers
    """
    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Counter for modified modules
    modified_count = 0
    
    # Create a mapping of module type names to their corresponding classes
    module_type_mapping = {
        "linear": nn.Linear,
        "conv1d": nn.Conv1d,
        "conv2d": nn.Conv2d,
        "conv3d": nn.Conv3d,
        "layernorm": nn.LayerNorm,
        "batchnorm1d": nn.BatchNorm1d,
        "batchnorm2d": nn.BatchNorm2d,
        "batchnorm3d": nn.BatchNorm3d,
    }
    
    # Special handling for ResNet models
    resnet_first_conv_handled = False
    
    # Convert target_modules to lowercase for case-insensitive matching
    target_modules = [m.lower() for m in target_modules]
    
    # Get the actual module classes to target
    target_classes = [module_type_mapping[m] for m in target_modules if m in module_type_mapping]
    
    if not target_classes:
        raise ValueError(f"No valid module types found in {target_modules}. Valid types are: {list(module_type_mapping.keys())}")
    
    # This will wrap target modules with SSF
    def _find_and_replace_linear_modules(module, path=""):
        nonlocal modified_count, resnet_first_conv_handled
        
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            
            # Special handling for ResNet's first conv layer
            if not resnet_first_conv_handled and isinstance(child, nn.Conv2d) and child_path.endswith("conv1") and child.in_channels != expected_input_channels:
                if verbose:
                    print(f"Skipping first conv layer at {child_path} with in_channels={child.in_channels} (expected {expected_input_channels})")
                resnet_first_conv_handled = True
                continue
            
            # First recursively process all children
            _find_and_replace_linear_modules(child, child_path)
            
            # Check if this is a target module type
            if any(isinstance(child, cls) for cls in target_classes):
                # Get the output dimension of the module
                if isinstance(child, nn.Linear):
                    hidden_size = child.out_features
                elif isinstance(child, nn.Conv1d):
                    hidden_size = child.out_channels
                elif isinstance(child, nn.Conv2d):
                    hidden_size = child.out_channels
                elif isinstance(child, nn.Conv3d):
                    hidden_size = child.out_channels
                elif isinstance(child, nn.LayerNorm):
                    if isinstance(child.normalized_shape, tuple):
                        hidden_size = child.normalized_shape[0]
                    else:
                        hidden_size = child.normalized_shape
                elif isinstance(child, nn.BatchNorm1d):
                    hidden_size = child.num_features
                elif isinstance(child, nn.BatchNorm2d):
                    hidden_size = child.num_features
                elif isinstance(child, nn.BatchNorm3d):
                    hidden_size = child.num_features
                else:
                    # Skip if we can't determine the hidden size
                    if verbose:
                        print(f"Skipping {child_path}: cannot determine hidden size for {type(child).__name__}")
                    continue
                
                # Create SSF wrapper with original linear module
                class SSFWrapper(nn.Module):
                    def __init__(self, module, ssf_layer):
                        super().__init__()
                        self.module = module
                        self.ssf_layer = ssf_layer
                        
                        # Expose attributes for compatibility with existing code
                        if hasattr(module, 'weight'):
                            self.weight = module.weight
                        if hasattr(module, 'bias'):
                            self.bias = module.bias
                        if hasattr(module, 'in_features'):
                            self.in_features = module.in_features
                        if hasattr(module, 'out_features'):
                            self.out_features = module.out_features
                        if hasattr(module, 'in_channels'):
                            self.in_channels = module.in_channels
                        if hasattr(module, 'out_channels'):
                            self.out_channels = module.out_channels
                        if hasattr(module, 'kernel_size'):
                            self.kernel_size = module.kernel_size
                        if hasattr(module, 'stride'):
                            self.stride = module.stride
                        if hasattr(module, 'padding'):
                            self.padding = module.padding
                        if hasattr(module, 'dilation'):
                            self.dilation = module.dilation
                        if hasattr(module, 'groups'):
                            self.groups = module.groups
                        if hasattr(module, 'normalized_shape'):
                            self.normalized_shape = module.normalized_shape
                        if hasattr(module, 'num_features'):
                            self.num_features = module.num_features
                        
                    def forward(self, *args, **kwargs):
                        output = self.module(*args, **kwargs)
                        return self.ssf_layer(output)
                
                # Create the SSF layer
                try:
                    ssf_layer = SSFLayer(hidden_size, init_scale, init_shift)
                    
                    # Replace the module with the wrapped version
                    wrapped_module = SSFWrapper(child, ssf_layer)
                    setattr(module, name, wrapped_module)
                    
                    # Count successful modifications
                    modified_count += 1
                    if verbose:
                        print(f"Applied SSF to {child_path} with hidden size {hidden_size}")
                except Exception as e:
                    if verbose:
                        print(f"Error applying SSF to {child_path}: {str(e)}")
                    continue
    
    # Start the recursive process to find and replace linear modules
    _find_and_replace_linear_modules(model)
    
    if verbose:
        print(f"Modified {modified_count} modules with SSF")
    
    # Return the modified model
    return model 