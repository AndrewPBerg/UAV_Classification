from typing import Dict, Tuple, Any
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
)
from transformers import ViTForImageClassification, ViTConfig

class ModelArchitectures:
    @staticmethod
    def get_architecture_mapping():
        return {
            # ResNet variants
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
            "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
            "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
            "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
            "resnet152": (resnet152, ResNet152_Weights.DEFAULT),
            # EfficientNet variants
            "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
            "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
            "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
            "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
            "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.DEFAULT),
            "efficientnet_b5": (efficientnet_b5, EfficientNet_B5_Weights.DEFAULT),
            "efficientnet_b6": (efficientnet_b6, EfficientNet_B6_Weights.DEFAULT),
            "efficientnet_b7": (efficientnet_b7, EfficientNet_B7_Weights.DEFAULT),
            # MobileNet variants
            "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
            "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
        }

    @staticmethod
    def get_vit_configs() -> Dict[str, Dict[str, Any]]:
        return {
            "vit_tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
            "vit_small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
            "vit_base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
            "vit_large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        }

    @staticmethod
    def is_vit_model(model_size: str) -> bool:
        return "vit" in model_size
