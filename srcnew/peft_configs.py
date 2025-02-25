from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml
from icecream import ic
import sys

class AdapterType(BaseModel):
    """
    valid models for each model type
    
    """

class TransformerConfig(BaseModel):

class CnnConfig(BaseModel):

class LoraConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = ["query", "key", "value", "dense"]
    lora_dropout: float = 0
    bias: str = "lora_only"
    task_type: str = "AUDIO_CLASSIFICATION"
    use_rslora: bool = False
    use_dora: bool = False

class IA3Config(BaseModel):
    target_modules: List[str] = ["query", "key", "value", "dense"]
    feedforward_modules: List[str] = ["dense", "query", "key", "value"]
    task_type: str = "AUDIO_CLASSIFICATION"

class AdaLoraConfig(BaseModel):
    init_r: int = 100
    target_r: int = 16
    target_modules: List[str] = ["query", "key", "value", "dense"]
    lora_alpha: int = 8
    task_type: str = "AUDIO_CLASSIFICATION"

class OFTConfig(BaseModel):
    r: int = 768
    target_modules: List[str] = ["query", "key", "value", "dense"]
    module_dropout: float = 0.0
    init_weights: bool = True

class FourierConfig(BaseModel):
    scaling: int = 100
    n_frequency: int = 1000
    target_modules: List[str] = ["query", "key", "value", "dense"]
    task_type: str = "AUDIO_CLASSIFICATION"

class LayernormConfig(BaseModel):
    target_modules: List[str] = ["layernorm"]
    task_type: str = "AUDIO_CLASSIFICATION"


def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create GeneralConfig instance from the dictionary
    try:
        # general_config = GeneralConfig(**config["general"])
        # ic("GeneralConfig instance created successfully:")
        # ic(general_config)
    except ValidationError as e:
        ic("Validation error occurred:")
        ic(e)





if __name__ == '__main__':
    main()


