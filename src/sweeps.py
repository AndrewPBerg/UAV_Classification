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
        from pytorch_lightning.loggers import WandbLogger
        
        # Determine rank for DDP coordination
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        global_rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        is_distributed = world_size > 1
        is_main_process = global_rank == 0
        
        # Basic configuration tracking
        if is_main_process:
            print(f"DDP Configuration: world_size={world_size}, rank={global_rank}, local_rank={local_rank}")
        
        # Load the general configuration - should be identical on all ranks
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
        if is_main_process:
            print(f"Available GPUs: {num_gpus}, using strategy: {'ddp' if num_gpus > 1 else 'auto'}")
        
        # Create DDP strategy configuration consistently across all ranks
        trainer_kwargs = config_copy.get("trainer", {}).copy()
        if is_distributed:
            # Use explicit strategy for DDP to avoid discrepancies
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = [local_rank]  # Each process uses its specific GPU
            trainer_kwargs["strategy"] = "ddp"
            trainer_kwargs["num_nodes"] = 1
            trainer_kwargs["use_distributed_sampler"] = True
            
            # Set strategy_kwargs only if not already set to avoid overwriting
            if "strategy_kwargs" not in trainer_kwargs:
                trainer_kwargs["strategy_kwargs"] = {"find_unused_parameters": False}
        
        # Configure wandb properly - ONLY on the main process
        if is_main_process and "wandb" in config_copy:
            try:
                # Finish any existing wandb runs to avoid conflicts
                if wandb.run is not None:
                    wandb.finish()
                
                # Extract and clean wandb config
                wandb_config = config_copy.get("wandb", {}).copy()
                if "config" in wandb_config:
                    del wandb_config["config"]
                
                # Create WandbLogger for Lightning - simpler and more compatible with DDP
                wandb_logger = WandbLogger(
                    save_dir="./wandb",
                    log_model=True,
                    **wandb_config,
                    settings=wandb.Settings(start_method="thread")
                )
                
                # Set logger in trainer args
                trainer_kwargs["logger"] = wandb_logger
            except Exception as e:
                print(f"Error setting up wandb: {e}")
        else:
            # For non-main processes, use a dummy logger
            trainer_kwargs["logger"] = False
        
        # Synchronize all processes before model creation to ensure consistency
        if is_distributed and dist.is_initialized():
            dist.barrier()
        
        # load into pydantic models w/ load_configs() - critical part where errors occur
        try:
            # Note: This is where the models are created from config
            # Make sure this code path is identical on all ranks
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
            
            # Create PTL trainer with all necessary configurations
            # IMPORTANT: For non-main processes, we still need to send a minimal wandb_config
            # to avoid 'NoneType' has no attribute 'project' errors
            if not is_main_process and wandb_config is None:
                # Create a dummy wandb config with just enough attributes to avoid errors
                # but disable actual logging for non-main processes
                from types import SimpleNamespace
                dummy_wandb_config = SimpleNamespace()
                dummy_wandb_config.project = general_config.project if hasattr(general_config, 'project') else "dummy-project"
                dummy_wandb_config.entity = general_config.entity if hasattr(general_config, 'entity') else None
                dummy_wandb_config.mode = "disabled"  # Disable actual logging
                
                # Use the dummy config for non-main processes
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
                # Main process uses the real wandb_config
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
        
        # Synchronize again before training
        if is_distributed and dist.is_initialized():
            dist.barrier()
        
        # Run the training with proper error handling
        try:
            # Disable wandb syncing for non-main processes
            if not is_main_process and wandb.run is not None:
                os.environ["WANDB_MODE"] = "offline"
            
            results = trainer.train()
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Clean up distributed environment on error
            if is_distributed and dist.is_initialized():
                dist.destroy_process_group()
            raise
        
        # Final cleanup
        if is_main_process and wandb.run is not None:
            wandb.finish()
            
        # Wait for all processes to finish
        if is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            
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
    
    # Determine rank for DDP coordination
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    is_main_process = global_rank == 0
    
    # Only print from main process to avoid duplicate outputs
    if is_main_process:
        print(f"Starting with rank={global_rank}, local_rank={local_rank}, world_size={world_size}")
    
    # Load configuration
    config = load_config(config_path="configs/config.yaml")
    
    # Set random seeds for reproducibility
    SEED = config['general']['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    # In distributed mode, only login to wandb on the main process
    if is_main_process:
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
    else:
        # Non-main processes should just call model_pipeline once
        # They will synchronize with the main process during distributed training
        print(f"Rank {global_rank}: waiting for main process to coordinate training")
        # Use a dummy sweep config with the same keys that would be in the real one
        dummy_sweep = {
            "learning_rate": config['general'].get('learning_rate', 0.001),
            "seed": config['general'].get('seed', 42)
        }
        model_pipeline(dummy_sweep)
    
    # Clean up distributed environment
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # Display PyTorch and GPU information
    print_pytorch_info()
    main()
    