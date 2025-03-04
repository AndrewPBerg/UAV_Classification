from typing import Optional, Literal, Dict, Any, List, Union, TypeVar
import yaml
from icecream import ic
import sys

from peft import (
    LoraConfig, 
    IA3Config, 
    AdaLoraConfig, 
    OFTConfig, 
    HRAConfig,
    LNTuningConfig,
    TaskType
)
from dataclasses import dataclass

# Define custom configs for options not available in PEFT
@dataclass
class NoneClassifierConfig:
    """None classifier configuration"""
    adapter_type: str = "none-classifier"
    task_type: str = "SEQ_CLS"

    def __iter__(self):
        yield "adapter_type", self.adapter_type
        yield "task_type", self.task_type

    def to_dict(self):
        return {
            "adapter_type": self.adapter_type,
            "task_type": self.task_type
        }

@dataclass
class NoneFullConfig:
    """Full configuration"""
    adapter_type: str = "none-full"
    task_type: str = "SEQ_CLS"

    def __iter__(self):
        yield "adapter_type", self.adapter_type
        yield "task_type", self.task_type

    def to_dict(self):
        return {
            "adapter_type": self.adapter_type,
            "task_type": self.task_type
        }

# Define valid PEFT types
VALID_PEFT_TYPES = ["lora", "ia3", "adalora", "oft", "layernorm", "hra"]

# Define PEFTConfig type alias
PEFTConfig = Union[LoraConfig, IA3Config, AdaLoraConfig, OFTConfig, HRAConfig, LNTuningConfig, Any, NoneClassifierConfig, NoneFullConfig]

def get_peft_config(config: dict) -> Optional[PEFTConfig]:
    """
    Create a PEFT configuration based on the provided config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PEFT configuration object
    """
    try:
        adapter_type = config["general"]["adapter_type"]
        
        match adapter_type:
            case "lora":
                # Convert task_type string to TaskType enum
                lora_config = config["lora"].copy()
                # if "task_type" in lora_config and isinstance(lora_config["task_type"], str):
                #     lora_config["task_type"] = "AUDIO_CLASSIFICATION"
                lora_config["task_type"] = "SEQ_CLS"
                return LoraConfig(**lora_config)
    
            case "ia3":
                # Convert task_type string to TaskType enum
                ia3_config = config["ia3"].copy()
                ia3_config["task_type"] = "SEQ_CLS"
                return IA3Config(**ia3_config)

            case "adalora":
                # Convert task_type string to TaskType enum
                adalora_config = config["adalora"].copy()
                adalora_config["task_type"] = "SEQ_CLS"
                return AdaLoraConfig(**adalora_config)

            case "oft":
                # Convert task_type string to TaskType enum
                oft_config = config["oft"].copy()
                oft_config["task_type"] = "SEQ_CLS"
                return OFTConfig(**oft_config)
                
            case "hra":
                # Convert task_type string to TaskType enum
                hra_config = config["hra"].copy()
                # if "task_type" in hra_config and isinstance(hra_config["task_type"], str):
                hra_config["task_type"] = "SEQ_CLS"
                return HRAConfig(**hra_config)

            case "layernorm":
                # For layernorm, we use LNTuningConfig
                layernorm_config = config["layernorm"].copy()
                layernorm_config["task_type"] = "SEQ_CLS"
                return LNTuningConfig(**layernorm_config)

            case "none-classifier":
                return NoneClassifierConfig()
            
            case "none-full":
                return NoneFullConfig()

            case _:
                raise ValueError(f"Unsupported adapter type: {adapter_type}")
                
    except KeyError as e:
        ic("The adapter type is not included in the config, defaulting to sweeps case:", e)
        
        # This is for the sweeps case
        try:
            adapter_type = config["adapter_type"]
            
            match adapter_type:
                case "lora":
                    # Convert task_type string to TaskType enum
                    sweep_config = config.copy()
                    # if "task_type" in sweep_config and isinstance(sweep_config["task_type"], str):
                    #     sweep_config["task_type"] = "AUDIO_CLASSIFICATION"
                    sweep_config["task_type"] = "SEQ_CLS"
                    return LoraConfig(**sweep_config)
        
                case "ia3":
                    # Convert task_type string to TaskType enum
                    sweep_config = config.copy()
                    sweep_config["task_type"] = "SEQ_CLS"
                    return IA3Config(**sweep_config)

                case "adalora":
                    # Convert task_type string to TaskType enum
                    sweep_config = config.copy()
                    # if "task_type" in sweep_config and isinstance(sweep_config["task_type"], str):
                    #     sweep_config["task_type"] = "AUDIO_CLASSIFICATION"
                    sweep_config["task_type"] = "SEQ_CLS"
                    return AdaLoraConfig(**sweep_config)

                case "oft":
                    # Convert task_type string to TaskType enum
                    sweep_config = config.copy()
                    sweep_config["task_type"] = "SEQ_CLS"
                    return OFTConfig(**sweep_config)
                    
                case "hra":
                    # Convert task_type string to TaskType enum
                    sweep_config = config.copy()
                    sweep_config["task_type"] = "SEQ_CLS"
                    return HRAConfig(**sweep_config)

                case "layernorm":
                    sweep_config = config.copy()
                    sweep_config["task_type"] = "SEQ_CLS"
                    return LNTuningConfig(**sweep_config)

                case "none-classifier":
                    return NoneClassifierConfig()
                
                case "none-full":
                    return NoneFullConfig()

                case _:
                    raise ValueError(f"Unsupported adapter type: {adapter_type}")
        except KeyError as e:
            ic("Could not determine adapter type:", e)
            return None

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    peft_config = get_peft_config(config)

    ic(peft_config.to_dict())

if __name__ == '__main__':
    main()
