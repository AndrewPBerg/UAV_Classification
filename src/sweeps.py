import os
import copy
import yaml
import random
import traceback
import numpy as np
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Union, Optional

import torch
import torch.distributed
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

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
    Load configuration from the given YAML file path.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing the configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Ensure all required sections exist
        required_sections = ['general', 'model', 'train', 'augmentation', 'sweep']
        for section in required_sections:
            if section not in config:
                config[section] = {}
        
        # Ensure general section has required fields
        if 'general' in config:
            if 'data_path' not in config['general']:
                config['general']['data_path'] = 'data/processed'
            if 'num_classes' not in config['general']:
                config['general']['num_classes'] = 10  # Set appropriate default
            if 'model_type' not in config['general']:
                config['general']['model_type'] = 'ast'
        
        return config
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Creating default config.")
        # Create a default config with all required sections and fields
        return {
            'general': {
                'data_path': 'data/processed',
                'num_classes': 10,  # Set appropriate default
                'model_type': 'ast',
                'seed': 42
            },
            'model': {
                'adapter_type': 'default',
                'trainable_layers': 'all',
                'freeze_feature_extractor': False
            },
            'train': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001
            },
            'augmentation': {},
            'sweep': {
                'method': 'random',
                'metric': {'name': 'val_acc', 'goal': 'maximize'},
                'parameters': {
                    'learning_rate': {'min': 0.0001, 'max': 0.01},
                    'batch_size': {'values': [16, 32, 64]}
                },
                'project': 'default_project'
            }
        }
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        traceback.print_exc()
        # Return minimal valid config
        return {
            'general': {
                'data_path': 'data/processed',
                'num_classes': 10, 
                'model_type': 'ast',
                'seed': 42
            },
            'model': {},
            'train': {},
            'augmentation': {},
            'sweep': {'project': 'default_project'}
        }

def model_pipeline(sweep_config=None):
    """
    The main pipeline for model training with wandb sweep support.
    
    Args:
        sweep_config: Configuration from wandb sweep, if applicable.
    """
    # Check if running as distributed worker
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank >= 0
    global_rank = int(os.environ.get("RANK", 0))
    
    try:
        # Load general configuration
        config = load_config('configs/config.yaml')
        config_copy = copy.deepcopy(config)
        
        # If sweep_config is provided (from wandb), merge it into our config
        if sweep_config is not None:
            try:
                config_copy = get_mixed_params(config_copy, sweep_config)
            except Exception as e:
                if local_rank <= 0:  # Only print from main process
                    print(f"Error in get_mixed_params: {str(e)}")
                    traceback.print_exc()
        
        # Make sure general config exists
        if 'general' not in config_copy:
            # Try to recover by setting default values
            if local_rank <= 0:  # Only print from main process
                print("Key error occurred, defaulting to general config from original config")
            
            if 'general' in config:
                config_copy['general'] = config['general']
            else:
                # Create minimal general config with required fields
                config_copy['general'] = {
                    'data_path': 'data/processed',
                    'num_classes': 10,  # Set appropriate default
                    'model_type': 'ast',
                    'seed': 42,
                    # Add other required fields
                }
        
        # Print GPU info only from one process
        if local_rank <= 0:
            num_gpus = torch.cuda.device_count()
            strategy = 'ddp' if is_distributed else 'auto'
            print(f"Available GPUs: {num_gpus}, using strategy: {strategy}")
        
        # Initialize all config variables to avoid UnboundLocalError
        general_config = None
        model_config = None
        train_config = None
        aug_config = None
        wandb_config = None
        sweep_config_obj = None
            
        # Load configurations into the pydantic models (will validate)
        try:
            (
                general_config, 
                model_config, 
                train_config, 
                aug_config, 
                wandb_config,
                sweep_config_obj
            ) = load_configs(config_copy)
            
            # For distributed workers, create a complete dummy wandb_config
            # with all required attributes to avoid AttributeError
            if is_distributed and local_rank > 0:
                # Create a more complete dummy wandb_config with all required attributes
                wandb_config = SimpleNamespace(
                    project="dummy_project",
                    entity="dummy_entity",
                    name="dummy_run",
                    mode="disabled",
                    tags=[],  # Add empty tags list
                    notes="",
                    group="",
                    job_type="",
                    id="",
                    save_code=False,
                    log_model=False,
                    reinit=False,
                    dir="",
                    config={},
                    settings=SimpleNamespace(
                        start_method="thread",
                        _disable_stats=True,
                        _disable_meta=True
                    )
                )
                
        except Exception as e:
            if local_rank <= 0:  # Only print from main process
                print(f"Error in configuration/initialization: {str(e)}")
                
            # Try to create a valid config with required fields
            for section in ['general', 'model', 'train', 'augmentation']:
                if section not in config_copy:
                    config_copy[section] = {}
            
            # Add required fields to general section
            if 'data_path' not in config_copy['general']:
                config_copy['general']['data_path'] = 'data/processed'
            if 'num_classes' not in config_copy['general']:
                config_copy['general']['num_classes'] = 10  # Set appropriate default
            if 'model_type' not in config_copy['general']:
                config_copy['general']['model_type'] = 'ast'
                
            # Try loading configs again with fixed config
            try:
                (
                    general_config, 
                    model_config, 
                    train_config, 
                    aug_config, 
                    wandb_config,
                    sweep_config_obj
                ) = load_configs(config_copy)
            except Exception as inner_e:
                # If still failing, create minimal valid objects
                from types import SimpleNamespace
                
                # Create minimal valid config objects
                if general_config is None:
                    general_config = SimpleNamespace(
                        data_path='data/processed',
                        num_classes=10,
                        model_type='ast',
                        seed=42
                    )
                
                if model_config is None:
                    model_config = SimpleNamespace(
                        adapter_type='default',
                        trainable_layers='all',
                        freeze_feature_extractor=False
                    )
                
                if train_config is None:
                    train_config = SimpleNamespace(
                        batch_size=32,
                        epochs=100,
                        learning_rate=0.001
                    )
                
                if aug_config is None:
                    aug_config = SimpleNamespace(
                        enabled=False
                    )
                
                if wandb_config is None:
                    wandb_config = SimpleNamespace(
                        project="dummy_project",
                        entity="dummy_entity",
                        name="dummy_run",
                        mode="disabled",
                        tags=[],
                        notes="",
                        group="",
                        job_type="",
                        id="",
                        save_code=False,
                        log_model=False,
                        reinit=False,
                        dir="",
                        config={}
                    )
                
                if sweep_config_obj is None:
                    sweep_config_obj = SimpleNamespace(
                        method="random",
                        metric={"name": "val_acc", "goal": "maximize"},
                        parameters={}
                    )
            
            # For distributed workers, create a complete dummy wandb_config
            if is_distributed and local_rank > 0:
                # Create a more complete dummy wandb_config with all required attributes
                wandb_config = SimpleNamespace(
                    project="dummy_project",
                    entity="dummy_entity",
                    name="dummy_run",
                    mode="disabled",
                    tags=[],  # Add empty tags list
                    notes="",
                    group="",
                    job_type="",
                    id="",
                    save_code=False,
                    log_model=False,
                    reinit=False,
                    dir="",
                    config={},
                    settings=SimpleNamespace(
                        start_method="thread",
                        _disable_stats=True,
                        _disable_meta=True
                    )
                )
        
        # Create data module and model factory
        data_module = AudioDataModule(
            general_config=general_config,
            feature_extraction_config=model_config,
            augmentation_config=aug_config,
            wandb_config=wandb_config,
            sweep_config=sweep_config_obj
        )
        
        model_factory = ModelFactory.get_model_factory(
            general_config=general_config,
            feature_extraction_config=model_config,
            peft_config=train_config
        )
        
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
        
        # Create PTL trainer - don't pass wandb_config in distributed settings
        if is_distributed or wandb_config is None:
            # Create a minimal dummy wandb config to avoid errors
            dummy_wandb_config = SimpleNamespace()
            dummy_wandb_config.project = "dummy-project"
            dummy_wandb_config.entity = None
            dummy_wandb_config.name = "dummy-run"
            dummy_wandb_config.mode = "disabled"
            
            # Use the dummy wandb config
            trainer = PTLTrainer(
                general_config=general_config,
                model_config=model_config,
                train_config=train_config,
                aug_config=aug_config,
                wandb_config=dummy_wandb_config,
                sweep_config=None,
                data_module=data_module,
                model_factory=model_factory
            )
        else:
            # Normal case for non-distributed runs
            trainer = PTLTrainer(
                general_config=general_config,
                model_config=model_config,
                train_config=train_config,
                aug_config=aug_config,
                wandb_config=wandb_config,
                sweep_config=None,
                data_module=data_module,
                model_factory=model_factory
            )
        
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
        if local_rank <= 0:  # Only print from main process
            print(f"Error in model_pipeline: {str(e)}")
            traceback.print_exc()
        raise

def get_mixed_params(general_config: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mix the sweep configuration parameters with the general configuration.
    
    Args:
        general_config: The general configuration.
        sweep_config: The sweep configuration.
        
    Returns:
        A mixed configuration.
    """
    try:
        config_copy = copy.deepcopy(general_config)
        
        # Handle different parameter locations in config
        for param_name, param_value in sweep_config.items():
            # For logging purposes, store original sweep params in their own section
            if 'sweep_params' not in config_copy:
                config_copy['sweep_params'] = {}
            config_copy['sweep_params'][param_name] = param_value
            
            # Now determine where each parameter belongs in the config structure
            if param_name in ['batch_size', 'epochs', 'learning_rate', 'weight_decay']:
                if 'train' not in config_copy:
                    config_copy['train'] = {}
                config_copy['train'][param_name] = param_value
            
            elif param_name in ['adapter_type', 'trainable_layers', 'freeze_feature_extractor']:
                if 'model' not in config_copy:
                    config_copy['model'] = {}
                config_copy['model'][param_name] = param_value
            
            # Add more parameter mappings as needed
            else:
                # For unknown params, add to general section by default
                if 'general' not in config_copy:
                    config_copy['general'] = {
                        'data_path': 'data/processed',
                        'num_classes': 10,
                        'model_type': 'ast',
                        'seed': 42
                    }
                config_copy['general'][param_name] = param_value
        
        # Ensure all required sections exist
        required_sections = ['general', 'model', 'train', 'augmentation']
        for section in required_sections:
            if section not in config_copy:
                config_copy[section] = {}
        
        # Ensure general section has required fields
        if 'data_path' not in config_copy['general']:
            config_copy['general']['data_path'] = 'data/processed'
        if 'num_classes' not in config_copy['general']:
            config_copy['general']['num_classes'] = 10  # Set appropriate default
        if 'model_type' not in config_copy['general']:
            config_copy['general']['model_type'] = 'ast'
        
        return config_copy
    except Exception as e:
        # Log error without using ic to avoid the keyword error
        print(f"Error in get_mixed_params: {str(e)}")
        traceback.print_exc()
        
        # Return a safe fallback config
        return {
            'general': {
                'data_path': 'data/processed',
                'num_classes': 10,
                'model_type': 'ast',
                'seed': 42
            },
            'model': sweep_config.copy() if isinstance(sweep_config, dict) else {},
            'train': {},
            'augmentation': {}
        }

def main():
    # Check if running as distributed worker
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    try:
        if local_rank >= 0:
            # Running as distributed worker
            print(f"Running as distributed worker with local_rank={local_rank}")
            
            # Set environment variable to disable wandb for worker processes
            if local_rank > 0:  # Non-main processes should disable wandb
                os.environ["WANDB_MODE"] = "disabled"
            
            # Load the full configuration
            config = load_config('configs/config.yaml')
            
            # Set seed for reproducibility
            seed = config['general'].get('seed', 42)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # For distributed workers, ensure wandb section exists
            if 'wandb' not in config:
                config['wandb'] = {
                    'project': 'default_project',
                    'entity': 'default_entity',
                    'tags': [],
                    'notes': '',
                    'group': '',
                    'job_type': '',
                    'save_code': False,
                    'log_model': False
                }
            
            # Run the model pipeline with the complete config
            model_pipeline(config)
            print(f"Worker rank {local_rank} finished")
            
            # Cleanup distributed environment
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                
        else:
            # Main process that handles wandb agent
            print("Starting main process for wandb sweep")
            
            # Load configuration
            config = load_config('configs/config.yaml')
            
            # Set seed for reproducibility
            seed = config['general'].get('seed', 42)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
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
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    import os
    import random
    import numpy as np
    import torch
    import wandb
    import traceback
    
    main()
    