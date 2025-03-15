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
        
        # Get output dimensions for later reshaping
        batch_size, _, out_height, out_width = out.shape
        
        # LoRA-C branch implementation
        # First, apply the down projection A
        lora_A_reshaped = self.lora_A.reshape(self.r, self.in_channels, self.kernel_size_h, self.kernel_size_w)
        
        # Apply down projection using grouped convolution
        x_down = F.conv2d(
            x,
            lora_A_reshaped,
            None,
            self.stride,
            self.padding,
            self.dilation,
            1  # No grouping for down projection
        )
        
        # Apply dropout
        x_down = self.dropout(x_down)
        
        # Apply the up projection B
        # Reshape B for efficient computation
        lora_B_reshaped = self.lora_B.reshape(self.out_channels, self.kernel_size_h, self.kernel_size_w, self.r)
        lora_B_reshaped = lora_B_reshaped.permute(0, 3, 1, 2)  # [out_channels, r, kernel_h, kernel_w]
        
        # Initialize lora_output with zeros that require grad
        lora_output = torch.zeros_like(out, requires_grad=True)
        
        # Process each rank dimension separately and sum the results
        for i in range(self.r):
            # Extract the i-th rank feature maps
            x_down_i = x_down[:, i:i+1]  # [batch_size, 1, height, width]
            
            # Extract the i-th rank filters for all output channels
            B_i = lora_B_reshaped[:, i:i+1]  # [out_channels, 1, kernel_h, kernel_w]
            
            # Apply convolution for this rank
            lora_output_i = F.conv2d(
                x_down_i,
                B_i,
                None,
                1,  # stride of 1
                self.padding,
                self.dilation,
                1  # No grouping
            )
            
            # Add to the output
            lora_output = lora_output + lora_output_i
        
        # Scale and add to the output, ensuring we maintain gradient information
        return out + (self.scaling * lora_output)
    
    def merge_weights(self) -> None:
        """
        Merge LoRA-C weights with the original weights for inference.
        This eliminates the need for the LoRA-C branch during inference.
        """
        if self.merged:
            return
            
        # Compute ΔW = B·A efficiently
        delta_w = torch.zeros_like(self.weight)
        
        # Reshape A and B for computation
        A = self.lora_A.reshape(self.r, self.in_channels, self.kernel_size_h, self.kernel_size_w)
        B = self.lora_B.reshape(self.out_channels, self.kernel_size_h, self.kernel_size_w, self.r)
        
        # Compute the weight update for each output channel
        for i in range(self.out_channels):
            for j in range(self.r):
                # For each rank, compute the outer product of B and A
                B_ij = B[i, :, :, j].unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_h, kernel_w]
                A_j = A[j]  # [in_channels, kernel_h, kernel_w]
                
                # Compute outer product for this rank and add to delta_w
                for k in range(self.in_channels):
                    delta_w[i, k] += F.conv2d(
                        A_j[k].unsqueeze(0).unsqueeze(0),
                        B_ij,
                        padding=self.kernel_size_h-1
                    )[0, 0, :self.kernel_size_h, :self.kernel_size_w]
        
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
            
        # Compute ΔW = B·A efficiently
        delta_w = torch.zeros_like(self.weight)
        
        # Reshape A and B for computation
        A = self.lora_A.reshape(self.r, self.in_channels, self.kernel_size_h, self.kernel_size_w)
        B = self.lora_B.reshape(self.out_channels, self.kernel_size_h, self.kernel_size_w, self.r)
        
        # Compute the weight update for each output channel
        for i in range(self.out_channels):
            for j in range(self.r):
                # For each rank, compute the outer product of B and A
                B_ij = B[i, :, :, j].unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_h, kernel_w]
                A_j = A[j]  # [in_channels, kernel_h, kernel_w]
                
                # Compute outer product for this rank and add to delta_w
                for k in range(self.in_channels):
                    delta_w[i, k] += F.conv2d(
                        A_j[k].unsqueeze(0).unsqueeze(0),
                        B_ij,
                        padding=self.kernel_size_h-1
                    )[0, 0, :self.kernel_size_h, :self.kernel_size_w]
        
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
