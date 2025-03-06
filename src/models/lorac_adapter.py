import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class LoRACLayer(nn.Module):
    """
    LoRA-C (Low-Rank Adaptation for Convolutional layers) implementation.
    
    This is a parameter-efficient fine-tuning method specifically designed for CNN models.
    Instead of applying low-rank decomposition to individual kernels, LoRA-C applies it at the
    convolutional layer level to reduce the number of trainable parameters.
    
    Based on the paper: "LoRA-C: Parameter-Efficient Fine-Tuning of Robust CNN for IoT Devices"
    https://arxiv.org/pdf/2410.16954
    """
    
    def __init__(
        self,
        conv_layer: nn.Conv2d,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        """
        Initialize LoRA-C adapter for a convolutional layer.
        
        Args:
            conv_layer: The original convolutional layer to adapt
            r: Rank of the low-rank decomposition (r << min(cin*k*k, cout*k*k))
            alpha: Scaling factor for the LoRA-C branch
            dropout: Dropout probability for the LoRA-C branch
        """
        super().__init__()
        
        # Get dimensions from the original convolutional layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        if isinstance(self.kernel_size, tuple):
            self.kernel_size_h, self.kernel_size_w = self.kernel_size
        else:
            self.kernel_size_h = self.kernel_size_w = self.kernel_size
            
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.bias = conv_layer.bias is not None
        
        # Original weights (frozen)
        self.weight = conv_layer.weight
        self.bias_param = conv_layer.bias
        
        # Scaling factor
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Initialize LoRA-C matrices
        # A ∈ ℝ^(r×cin×k×k) - Down projection
        self.lora_A = nn.Parameter(
            torch.zeros((r, self.in_channels, self.kernel_size_h, self.kernel_size_w))
        )
        # B ∈ ℝ^(cout×k×k×r) - Up projection
        self.lora_B = nn.Parameter(
            torch.zeros((self.out_channels, self.kernel_size_h, self.kernel_size_w, r))
        )
        
        # Initialize A with random Gaussian and B with zeros
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Merge flag for inference
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA-C adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after convolution with LoRA-C adaptation
        """
        if self.merged:
            # If weights are merged, use standard convolution
            return F.conv2d(
                x, 
                self.weight, 
                self.bias_param, 
                self.stride, 
                self.padding, 
                self.dilation, 
                self.groups
            )
        
        # Standard convolution with frozen weights
        out = F.conv2d(
            x, 
            self.weight, 
            self.bias_param, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
        
        # LoRA-C branch
        # Compute the layer-wise low-rank adaptation: ΔW = B·A
        # We implement this as a series of operations that maintain the convolutional structure
        
        # Reshape input for efficient computation
        batch_size, _, height, width = x.shape
        
        # Compute the LoRA-C contribution
        # First, apply the down projection A
        # We need to reshape A to be compatible with conv2d
        lora_A_reshaped = self.lora_A.reshape(self.r * self.in_channels, 1, self.kernel_size_h, self.kernel_size_w)
        x_down = F.conv2d(
            x,
            lora_A_reshaped,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.in_channels  # Use groups=in_channels for depthwise convolution
        )
        
        # Apply dropout
        x_down = self.dropout(x_down)
        
        # Reshape for the up projection
        x_down = x_down.reshape(batch_size, self.r, -1)
        
        # Apply the up projection B
        # We need to reshape B to be compatible with our computation
        lora_B_reshaped = self.lora_B.permute(0, 3, 1, 2).reshape(
            self.out_channels * self.r, 1, self.kernel_size_h, self.kernel_size_w
        )
        
        # Reshape x_down to match the expected input shape for the up projection
        x_down = x_down.repeat_interleave(self.out_channels, dim=1)
        x_down = x_down.reshape(batch_size * self.out_channels, self.r, height, width)
        
        # Apply the up projection
        lora_output = F.conv2d(
            x_down,
            lora_B_reshaped,
            None,
            (1, 1),
            self.padding,
            self.dilation,
            self.r  # Use groups=r for grouped convolution
        )
        
        # Reshape the output to the expected shape
        lora_output = lora_output.reshape(batch_size, self.out_channels, height, width)
        
        # Scale and add to the output
        return out + self.scaling * lora_output
    
    def merge_weights(self) -> None:
        """
        Merge LoRA-C weights with the original weights for inference.
        This eliminates the need for the LoRA-C branch during inference.
        """
        if self.merged:
            return
            
        # Compute ΔW = B·A efficiently using tensor operations
        # Reshape A and B for matrix multiplication
        A_flat = self.lora_A.reshape(self.r, -1)  # r x (cin*kh*kw)
        B_flat = self.lora_B.reshape(self.out_channels, -1, self.r)  # cout x (kh*kw) x r
        
        # Compute the matrix multiplication
        delta_w_flat = torch.bmm(B_flat, A_flat.unsqueeze(0).expand(self.out_channels, -1, -1))  # cout x (kh*kw) x (cin*kh*kw)
        
        # Reshape back to the weight tensor shape
        delta_w = delta_w_flat.reshape(
            self.out_channels, self.kernel_size_h, self.kernel_size_w, 
            self.in_channels, self.kernel_size_h, self.kernel_size_w
        )
        
        # Sum over the appropriate dimensions to get the final weight update
        delta_w = delta_w.sum(dim=(1, 2)).permute(0, 1, 2, 3).reshape_as(self.weight)
        
        # Scale and update the original weights
        self.weight.data += self.scaling * delta_w
        self.merged = True
    
    def unmerge_weights(self) -> None:
        """
        Unmerge LoRA-C weights from the original weights.
        This restores the original weights for further training.
        """
        if not self.merged:
            return
            
        # Compute ΔW = B·A efficiently using tensor operations
        # Reshape A and B for matrix multiplication
        A_flat = self.lora_A.reshape(self.r, -1)  # r x (cin*kh*kw)
        B_flat = self.lora_B.reshape(self.out_channels, -1, self.r)  # cout x (kh*kw) x r
        
        # Compute the matrix multiplication
        delta_w_flat = torch.bmm(B_flat, A_flat.unsqueeze(0).expand(self.out_channels, -1, -1))  # cout x (kh*kw) x (cin*kh*kw)
        
        # Reshape back to the weight tensor shape
        delta_w = delta_w_flat.reshape(
            self.out_channels, self.kernel_size_h, self.kernel_size_w, 
            self.in_channels, self.kernel_size_h, self.kernel_size_w
        )
        
        # Sum over the appropriate dimensions to get the final weight update
        delta_w = delta_w.sum(dim=(1, 2)).permute(0, 1, 2, 3).reshape_as(self.weight)
        
        # Scale and restore the original weights
        self.weight.data -= self.scaling * delta_w
        self.merged = False


def apply_lorac_to_model(
    model: nn.Module,
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA-C to a model by replacing convolutional layers with LoRA-C layers.
    
    Args:
        model: The model to apply LoRA-C to
        r: Rank of the low-rank decomposition
        alpha: Scaling factor for the LoRA-C branch
        dropout: Dropout probability for the LoRA-C branch
        target_modules: List of module names to apply LoRA-C to. If None, apply to all Conv2d layers.
        
    Returns:
        Model with LoRA-C applied
    """
    # Dictionary to store LoRA-C layers
    lorac_layers = {}
    
    # If no target modules specified, apply to all Conv2d layers
    if target_modules is None or len(target_modules) == 0:
        target_modules_list = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_modules_list.append(name)
    else:
        # Convert target_modules to lowercase for case-insensitive matching
        target_modules_lower = [module.lower() for module in target_modules]
        target_modules_list = []
        
        # Find all modules that match the target module types
        for name, module in model.named_modules():
            module_type = module.__class__.__name__.lower()
            if any(target_type.lower() in module_type for target_type in target_modules_lower) and isinstance(module, nn.Conv2d):
                target_modules_list.append(name)
    
    # Replace target modules with LoRA-C layers
    for name, module in model.named_modules():
        if name in target_modules_list:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # Create LoRA-C layer
                lorac_layer = LoRACLayer(
                    module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout
                )
                
                # Replace the original layer with LoRA-C layer
                setattr(parent, child_name, lorac_layer)
                
                # Store the LoRA-C layer
                lorac_layers[name] = lorac_layer
            else:
                # Create LoRA-C layer
                lorac_layer = LoRACLayer(
                    module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout
                )
                
                # Replace the original layer with LoRA-C layer
                setattr(model, child_name, lorac_layer)
                
                # Store the LoRA-C layer
                lorac_layers[name] = lorac_layer
    
    # Add a property to the model to access LoRA-C layers
    model.lorac_layers = lorac_layers
    
    # Add methods to merge and unmerge weights
    def merge_lorac_weights(self):
        for layer in self.lorac_layers.values():
            layer.merge_weights()
    
    def unmerge_lorac_weights(self):
        for layer in self.lorac_layers.values():
            layer.unmerge_weights()
    
    # Add methods to the model
    model.merge_lorac_weights = merge_lorac_weights.__get__(model)
    model.unmerge_lorac_weights = unmerge_lorac_weights.__get__(model)
    
    return model
