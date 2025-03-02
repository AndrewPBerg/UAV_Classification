from typing import Optional, Literal, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml
from icecream import ic
import sys




class WandbConfig(BaseModel):
    class Config:
        strict = True

    project: str # ex: "9class-Kfold-Results"
    name: str  # ex:"CNN-0augs-LR=0.001"
    reinit: bool = False # not useful unless interactive code env
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    dir: str = "wandb" # local dir to store wandb files (should resemble cloud)

class SweepConfig(BaseModel):
    class Config:
        strict = True
    
    project: str # ex: "9class-Kfold-sweep"
    name: str # ex: Augmentations/sample Tuning
    method: str  # ex: random
    metric: Dict[str, str] = {"name": "test_acc", "goal": "maximize"}
    parameters: Dict[str, Any] = {}
    

def get_wandb_config(config: dict) -> Tuple[WandbConfig, SweepConfig]:


    try:
        wandb_config = WandbConfig(**config["wandb"])
        sweep_config = SweepConfig(**config["sweep"])
        
    except ValidationError as e:
        ic("Validation error occurred:")
        ic(e)
    
    except KeyError as e:
        ic("Key error, defaulting to sweeps case: ", e)
        wandb_config = WandbConfig(**config)
        sweep_config = SweepConfig(**config)
    
    
    return wandb_config, sweep_config

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    wandb_config, sweep_config = get_wandb_config(config)
    ic(wandb_config)
    ic(sweep_config)


if __name__ == '__main__':
    main()