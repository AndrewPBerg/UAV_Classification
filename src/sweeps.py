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
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

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
        # Import needed modules
        import os
        import copy
        import torch
        import torch.distributed as dist
        import pytorch_lightning as pl
        from pytorch_lightning.strategies import DDPStrategy
        from pytorch_lightning.loggers import WandbLogger
        
        # Determine rank for DDP coordination
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        is_distributed = local_rank >= 0
        
        # Load the general configuration
        config = load_config()
        
        # If this is a sweep run, merge the configurations
        if sweep_config:
            # Use a more robust merging of configurations
            mixed_params = get_mixed_params(config, sweep_config)
            # Make sure all required configurations exist with defaults if needed
            aug_config = {}  # Default empty dict to avoid UnboundLocalError
            
            # Create a copy to avoid modifying the original during DDP
            config_copy = copy.deepcopy(mixed_params)
        else:
            config_copy = copy.deepcopy(config)
        
        # Get the number of GPUs and set up DDP strategy
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}, using strategy: {'ddp' if is_distributed else 'auto'}")
        
        # Create trainer kwargs based on distributed status
        trainer_kwargs = config_copy.get("trainer", {}).copy()
        if is_distributed:
            # For distributed training, use explicit DDP configuration
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = [local_rank]  # Each process uses its specific GPU
            
            # Use explicit DDPStrategy for better control
            ddp_strategy = DDPStrategy(
                find_unused_parameters=False,
                static_graph=True  # Use static graph for better performance
            )
            trainer_kwargs["strategy"] = ddp_strategy
        else:
            # For non-distributed training, use auto strategy
            trainer_kwargs["accelerator"] = "gpu" if num_gpus > 0 else "cpu"
            trainer_kwargs["devices"] = "auto"
            trainer_kwargs["strategy"] = "auto"
        
        # Configure wandb - important: only create logger on non-distributed runs
        # or if explicitly running as the main process in a distributed setting
        if not is_distributed and "wandb" in config_copy:
            try:
                # Finish any existing wandb runs to avoid conflicts
                if wandb.run is not None:
                    wandb.finish()
                
                # Extract and clean wandb config
                wandb_config = config_copy.get("wandb", {}).copy()
                if "config" in wandb_config:
                    del wandb_config["config"]
                
                # Create WandbLogger for Lightning
                wandb_logger = WandbLogger(
                    save_dir="./wandb",
                    log_model="all",
                    **wandb_config
                )
                
                # Set logger in trainer args
                trainer_kwargs["logger"] = wandb_logger
            except Exception as e:
                print(f"Error setting up wandb: {e}")
                # Continue without wandb logger
                pass
        elif is_distributed:
            # For distributed workers, disable logging completely
            trainer_kwargs["logger"] = False
            # Ensure wandb is disabled
            os.environ["WANDB_MODE"] = "disabled"
        
        # Add deterministic flag for reproducibility
        trainer_kwargs["deterministic"] = True
        
        # load into pydantic models w/ load_configs()
        try:
            (
                general_config,
                feature_extraction_config,
                peft_config,
                wandb_config,
                sweep_config,
                aug_config,
            ) = load_configs(config_copy)
            
            # Safely access data
            data_module = AudioDataModule(
                general_config=general_config,
                feature_extraction_config=feature_extraction_config,
                augmentation_config=aug_config,
                wandb_config=wandb_config,
                sweep_config=sweep_config
            )
            
            model_factory = ModelFactory.get_model_factory(
                general_config=general_config,
                feature_extraction_config=feature_extraction_config,
                peft_config=peft_config
            )
            
            # Create PTL trainer - don't pass wandb_config in distributed settings
            if is_distributed or wandb_config is None:
                # Create a minimal dummy wandb config to avoid errors
                from types import SimpleNamespace
                dummy_wandb_config = SimpleNamespace()
                dummy_wandb_config.project = "dummy-project"
                dummy_wandb_config.entity = None
                dummy_wandb_config.name = "dummy-run"
                dummy_wandb_config.mode = "disabled"
                
                # Use the dummy wandb config
                trainer = PTLTrainer(
                    general_config=general_config,
                    feature_extraction_config=feature_extraction_config,
                    peft_config=peft_config,
                    wandb_config=dummy_wandb_config,
                    sweep_config=None,
                    data_module=data_module,
                    model_factory=model_factory
                )
            else:
                # Normal case for non-distributed runs
                trainer = PTLTrainer(
                    general_config=general_config,
                    feature_extraction_config=feature_extraction_config,
                    peft_config=peft_config,
                    wandb_config=wandb_config,
                    sweep_config=None,
                    data_module=data_module,
                    model_factory=model_factory
                )
        except Exception as e:
            print(f"Error in configuration/initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Run the training with proper error handling
        try:
            results = trainer.train()
        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Clean up distributed environment on error if needed
            raise
        
        # Final cleanup for wandb
        if not is_distributed and wandb.run is not None:
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
    # Import necessary modules for distributed setup
    import os
    import torch.distributed as dist
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    
    # Detect if this is a distributed run
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_distributed = local_rank >= 0
    
    # If this is a distributed worker (not the main process), don't use wandb sweeps at all
    if is_distributed:
        print(f"Running as distributed worker with local_rank={local_rank}")
        # Set environment variable to disable wandb for worker processes
        os.environ["WANDB_MODE"] = "disabled"
        
        # Load configuration - use fixed config rather than wandb sweep
        config = load_config(config_path="configs/config.yaml")
        
        # Set random seeds for reproducibility
        SEED = config['general']['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        
        # Create a simple fixed config for the worker process
        # The actual hyperparam values will be broadcast from the main process
        dummy_config = {
            "learning_rate": config['general'].get('learning_rate', 0.001),
            "seed": SEED
        }
        
        # Call the model pipeline directly
        try:
            model_pipeline(dummy_config)
        except Exception as e:
            print(f"Worker process error: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Worker rank {local_rank} finished")
        return
    
    # Continue with normal execution for the main process (non-distributed or rank 0)
    print("Running as main process - will manage wandb sweeps")
    
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
    
    print("All runs completed.")

if __name__ == "__main__":
    # Display PyTorch and GPU information
    print_pytorch_info()
    main()
    