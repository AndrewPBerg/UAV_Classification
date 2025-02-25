from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml
from icecream import ic
import sys

class _AdapterChoice(BaseModel):
    """
    valid PEFT choices
    """

    valid_peft_types: List[str] = ["lora", "ia3", "adalora", "oft", "fourier", "layernorm"]

class NoneClassifierConfig(BaseModel):
    """None classifier configuration"""
    task_type: str = "AUDIO_CLASSIFICATION"

class NoneFullConfig(BaseModel):
    """Full configuration"""
    task_type: str = "AUDIO_CLASSIFICATION"

class LoraConfig(BaseModel):
    """LoRA configuration"""
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = ["query", "key", "value", "dense"]
    lora_dropout: float = 0
    bias: str = "lora_only"
    task_type: str = "AUDIO_CLASSIFICATION"
    use_rslora: bool = False
    use_dora: bool = False
    class Config:
        strict = True

class IA3Config(BaseModel):
    """IA3 configuration"""
    target_modules: List[str] = ["query", "key", "value", "dense"]
    feedforward_modules: List[str] = ["dense", "query", "key", "value"]
    task_type: str = "AUDIO_CLASSIFICATION"
    class Config:
        strict = True

class AdaLoraConfig(BaseModel):
    """AdaLoRA configuration"""
    init_r: int = 100
    target_r: int = 16
    target_modules: List[str] = ["query", "key", "value", "dense"]
    lora_alpha: int = 8
    task_type: str = "AUDIO_CLASSIFICATION"
    class Config:
        strict = True

class OFTConfig(BaseModel):
    """OFT configuration"""
    r: int = 768
    target_modules: List[str] = ["query", "key", "value", "dense"]
    module_dropout: float = 0.0
    init_weights: bool = True
    class Config:
        strict = True

class FourierConfig(BaseModel):
    """Fourier configuration"""
    scaling: int = 100
    n_frequency: int = 1000
    target_modules: List[str] = ["query", "key", "value", "dense"]
    task_type: str = "AUDIO_CLASSIFICATION"
    class Config:
        strict = True

class LayernormConfig(BaseModel):
    """LayerNorm configuration"""
    target_modules: List[str] = ["layernorm"]
    task_type: str = "AUDIO_CLASSIFICATION"
    class Config:
        strict = True


def get_peft_config(config: dict) -> BaseModel:


    match config["general"]["adaptor_type"]:
        case "lora":
            # Handle LoRA configuration
            return LoraConfig(**config["lora"])
 
        case "ia3":
            # Handle IA3 configuration
            return IA3Config(**config["ia3"])

        case "adalora":
            # Handle AdaLoRA configuration
            return AdaLoraConfig(**config["adalora"])

        case "oft":
            # Handle OFT configuration
            return OFTConfig(**config["oft"])

        case "fourier":
            # Handle Fourier configuration
            return FourierConfig(**config["fourier"])

        case "layernorm":
            # Handle Layernorm configuration
            return LayernormConfig(**config["layernorm"])

        case "none-classifier":
            return NoneClassifierConfig()
        
        case "none-full":
            return NoneFullConfig()

        case _:
            raise ValueError(f"Unsupported adapter type: {config['general']['adaptor_type']}")


def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    peft_config = get_peft_config(config)
    ic(peft_config)




if __name__ == '__main__':
    main()


