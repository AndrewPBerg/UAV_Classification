import os
import yaml
from configs.augmentation_config import AugmentationConfig
import torch
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from icecream import ic

# Import the PyTorch Lightning implementation
from ptl_trainer import PTLTrainer
from datamodule import AudioDataModule
from model_factory import ModelFactory
from configs.configs_demo import (
    GeneralConfig, 
    FeatureExtractionConfig,
    load_configs,
    wandb_config_dict
)
from configs.peft_config import get_peft_config
from configs.wandb_config import get_wandb_config
from helper.util import wandb_login
import sys

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path: str = 'configs/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def model_pipeline(sweep_config=None):
    """
    Pipeline function for wandb sweep agent.
    
    Args:
        sweep_config: Configuration provided by wandb sweep
    """
    # Initialize wandb run
    with wandb.init(config=sweep_config):
        # Get the sweep configuration
        config = wandb.config
        
        # Load general config
        yaml_config = load_config(config_path="configs/config.yaml")

        # Combine sweep config with general config
        mixed_params = get_mixed_params(general_config=yaml_config, sweep_config=config)
        
        # ic(mixed_params)
        
        # sys.exit(0)
        (
        general_config,
        feature_extraction_config,
        peft_config,
        wandb_config,
        sweep_config,
        augmentation_config
         ) = load_configs(mixed_params)
        ic(wandb_config)
        ic(sweep_config)
        # sys.exit(0)
        # Create configuration objects
        # general_config = GeneralConfig(
        #     **mixed_params
        # )
        
        # # Feature extraction config
        # feature_extraction_config = FeatureExtractionConfig(
        #     **mixed_params
        # )
        
        # #PEFT config
        # peft_config = get_peft_config(mixed_params)
        
        
        # wandb_config, sweep_config = get_wandb_config(mixed_params)
        
        
        # augmentation_config = AugmentationConfig(
        #     **mixed_params
        # )
        
        # Create data module
        data_module = AudioDataModule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            augmentation_config=augmentation_config,
            wandb_config=wandb_config,
            sweep_config=sweep_config
        )
        
        # Get model factory function
        model_factory = ModelFactory.get_model_factory(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            peft_config=peft_config
        )
        
        # Create PTL trainer
        trainer = PTLTrainer(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            peft_config=peft_config,
            wandb_config=wandb_config,
            sweep_config=None,  # We're not using a separate sweep config object here
            data_module=data_module,
            model_factory=model_factory
        )
        
        # Train model
        if general_config.use_kfold:
            results = trainer.k_fold_cross_validation()
            # Log results summary
            wandb.log(results["avg_metrics"])
        else:
            model, results = trainer.train()
            # Log final test metrics
            wandb.log(results)

def get_mixed_params(general_config: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine sweep configuration with general configuration.
    
    """
    ic(sweep_config)
    # 1. get mixed params from config file, sweeps overrides all values
    mixed_params = {}
    
    # general
    mixed_params.update(general_config['general'])
    # augmentations
    mixed_params.update(general_config['augmentations'])
    # feature extraction
    mixed_params.update(general_config['feature_extraction'])
    # wandb
    mixed_params.update(general_config['wandb'])
    # sweep
    mixed_params.update(general_config['sweep'])
    # ic(mixed_params.get('project'))
    # ic(mixed_params.get('name'))
    # sweep project and name
    mixed_params['project'] = general_config['sweep']['project']
    mixed_params['name'] = general_config['sweep']['name']
    # ic(mixed_params.get('project'))
    # ic(mixed_params.get('name'))
    
    # sys.exit(0)
    
    # peft (tricky cases for specific peft general_configs)
    # ic("params before sweep", mixed_params)
    
    try: 
        peft_name = str(sweep_config['adapter_type'])
    except KeyError as e:
        ic("the adapter type is not included in the sweep config, defaulting to general config's: ", e)
        peft_name = str(general_config['general']['adapter_type'])
    
    # ic(peft_name)
    
   
    mixed_params.update(general_config[peft_name])
    mixed_params.update(sweep_config)
    
    # ic("after sweep mixing", mixed_params)

    # sys.exit(0)
    
    return mixed_params

def main():
    """
    Main function to initialize and run the sweep.
    """
    # Load configuration
    config = load_config(config_path="configs/config.yaml")
    
    # Set random seeds for reproducibility
    SEED = config['general']['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Login to wandb
    wandb_login()
    
    # Get sweep configuration
    sweep_config = config['sweep']

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=config['sweep']['project']
    )
    
    # Run sweep agent
    wandb.agent(
        sweep_id,
        function=model_pipeline,
        count=config['general'].get('sweep_count', 10)
    )

if __name__ == "__main__":
    main()
    #demo_3_1()