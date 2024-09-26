import torch.nn as nn
import torch

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, data_shape: torch.Size, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self._infer_features(input_shape, data_shape)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.flattened_features, 
                      out_features=output_shape)
        )

    def _infer_features(self, input_shape:int, data_shape: torch.Size):
        # Pass through conv layers to infer the flattened size
        x = torch.randn(1, input_shape, data_shape[1], data_shape[2]) # Must change if data changes
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        self.flattened_features = x.numel() # numel counts the total number of elements in a tensor
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))