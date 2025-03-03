import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from icecream import ic
import copy
import os
import os
from pytorch_lightning.loggers import WandbLogger

# Import the PyTorch Lightning implementation
from ptl_trainer import PTLTrainer
from datamodule import AudioDataModule
from model_factory import ModelFactory
from configs.configs_demo import (
    load_configs,
)

from helper.util import wandb_login

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
    try:
        # Load the general configuration
        config = load_config()
        
        # If this is a sweep run, merge the configurations
        if sweep_config:
            # Use a more robust merging of configurations
            mixed_params = get_mixed_params(config, sweep_config)
            # Make sure all required configurations exist with defaults if needed
            if 'aug_config' not in locals() or 'aug_config' not in globals():
                aug_config = {}  # Default empty dict to avoid UnboundLocalError
            
            # Create a copy to avoid modifying the original during DDP
            config_copy = copy.deepcopy(mixed_params)
        else:
            config_copy = copy.deepcopy(config)
        
        # Get the number of GPUs and set up DDP strategy
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}, using strategy: {'ddp' if num_gpus > 1 else 'auto'}")
        
        # Make sure strategy is properly passed to Trainer
        trainer_kwargs = config_copy.get("trainer", {})
        if num_gpus > 1:
            trainer_kwargs["strategy"] = "ddp"
            # Add find_unused_parameters to avoid DDP issues with unused model parameters
            trainer_kwargs["plugins"] = [{"class_path": "pytorch_lightning.plugins.environments.LightningEnvironment"}]
            trainer_kwargs["strategy_kwargs"] = {"find_unused_parameters": False}
        
        # Set up wandb configuration properly for DDP
        # Only initialize wandb on the main process for DDP
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_main_process = local_rank == 0
        
        # Configure a proper WandbLogger for PyTorch Lightning
        if "wandb" in config_copy:
            try:
                # Make sure we don't have conflicting runs
                if wandb.run is not None and is_main_process:
                    wandb.finish()
                
                # Extract wandb config
                wandb_config = config_copy.get("wandb", {}).copy()
                # Remove problematic nested configurations
                if "config" in wandb_config:
                    del wandb_config["config"]
                
                # Only log from rank 0 in DDP
                wandb_logger = WandbLogger(
                    log_model=True,
                    save_dir="./wandb",
                    **wandb_config,
                    settings=wandb.Settings(start_method="thread")
                )
                
                # Add logger to trainer args
                if "logger" not in trainer_kwargs:
                    trainer_kwargs["logger"] = wandb_logger
            except Exception as e:
                ic("Error setting up wandb logger:", e)
                # Continue without wandb if it fails
        
        # load into pydantic models w/ load_configs()
        (
        general_config,
        feature_extraction_config,
        peft_config,
        wandb_config,
        sweep_config,
        augmentation_config
         ) = load_configs(config_copy)
        
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
        
        # Create the model
        from Model import AudioClassifier
        model = AudioClassifier(config_copy)
        
        # Run the training
        try:
            results = trainer.train()
        except Exception as e:
            ic(f"Error during training: {str(e)}")
            raise
        
        # Only log from main process
        if is_main_process and wandb.run:
            wandb.finish()
            
        return results
    except Exception as e:
        ic(f"Error in model_pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def get_mixed_params(general_config: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge general config with sweep config with improved error handling for DDP.
    """
    try:
        # Create a deep copy to avoid modifying the original during DDP
        mixed_params = copy.deepcopy(general_config)
        
        # Safely merge sweep parameters
        for key, value in sweep_config.items():
            # For nested configurations, use defensive coding
            if key in mixed_params and isinstance(mixed_params[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                mixed_params[key].update(value)
            else:
                # Direct assignment for non-nested or new parameters
                mixed_params[key] = value
                
        # Ensure critical configurations have defaults
        if 'adapter_type' not in mixed_params:
            try:
                mixed_params['adapter_type'] = general_config.get('adapter_type', 'default')
                ic("the adapter type is not included in the sweep config, defaulting to general config's: ", 
                   e=KeyError('adapter_type'))
            except KeyError as e:
                ic("Key error occurred, defaulting to sweeps case: ", e=e)
                mixed_params['adapter_type'] = 'default'
                
        # Ensure augmentations configuration exists
        if 'augmentations' not in mixed_params or not isinstance(mixed_params.get('augmentations'), dict):
            mixed_params['augmentations'] = {}
            ic("Key error occurred, defaulting to sweeps case: ", 
               e=KeyError('augmentations is not a dict'))
                
        return mixed_params
        
    except Exception as e:
        ic(f"Error in get_mixed_params: {str(e)}")
        # Return a minimal valid configuration to avoid crashing
        return {
            'adapter_type': 'default',
            'augmentations': {},
            'learning_rate': sweep_config.get('learning_rate', 0.001),
            'seed': sweep_config.get('seed', 42)
        }

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
    