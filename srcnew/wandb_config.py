from typing import Optional, Literal, Dict, Any, List
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
    
    name: str # ex: Augmentations/sample Tuning
    method: str  # ex: random
    metric: Dict[str, str] = {"name": "test_acc", "goal": "maximize"}
    parameters: Dict[str, Any] = {}
    

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ic(config["wandb"])
    ic(config["sweep"])

    try:
        wandb_config = WandbConfig(**config["wandb"])
        ic("WandbConfig instance created successfully:")
        ic(wandb_config)
    except ValidationError as e:
        ic("Validation error occurred:")
        ic(e)

    try:
        sweep_config = SweepConfig(**config["sweep"])
        ic("SweepConfig instance created successfully:")
        ic(sweep_config)
    except ValidationError as e:
        ic("Validation error occurred:")
        ic(e)

if __name__ == '__main__':
    main()