import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from icecream import ic

# Import the PyTorch Lightning implementation
from helper.ptl_trainer import PTLTrainer
from esc50.esc50_datamodule import ESC50DataModule, create_esc50_datamodule
from models.model_factory import ModelFactory
from configs.configs_aggregate import load_configs

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
    """
    Pipeline function for wandb sweep agent.
    
    Args:
        sweep_config: Configuration provided by wandb sweep
    """
    # Enable WandB service for better distributed training reliability
    wandb.require("service")
    
    # Ensure any existing wandb run is finished before starting
    try:
        if wandb.run is not None:
            print(f"Warning: Finishing existing wandb run: {wandb.run.name}")
            wandb.finish()
    except Exception as e:
        print(f"Warning: Error finishing existing wandb run: {e}")
        # Force finish any lingering runs
        try:
            wandb.finish()
        except:
            pass
    
    # Initialize wandb run with error handling
    try:
        with wandb.init(config=sweep_config):
            # Get the sweep configuration
            config = wandb.config
            
            # Load general config
            yaml_config = load_config(config_path="configs/config.yaml")

            # Combine sweep config with general config
            mixed_params = get_mixed_params(general_config=yaml_config, sweep_config=config)
            
            # Load into pydantic models w/ load_configs()
            (
                general_config,
                feature_extraction_config,
                dataset_config,
                peft_config,
                wandb_config,
                sweep_config_obj,
                augmentation_config,
                optimizer_config,
                peft_scheduling_config
            ) = load_configs(mixed_params)
            
            # Log the augmentation config to the W&B run with error handling
            try:
                if augmentation_config:
                    # Convert to dict and filter out non-serializable objects
                    aug_dict = augmentation_config.model_dump()
                    # Remove any complex nested objects that might cause issues
                    if 'aug_configs' in aug_dict:
                        del aug_dict['aug_configs']
                    wandb.config.update({"augmentations": aug_dict}, allow_val_change=True)
            except Exception as e:
                print(f"Warning: Could not log augmentation config to WandB: {e}")

            # Create ESC50 data module
            data_module = create_esc50_datamodule(
                general_config=general_config,
                feature_extraction_config=feature_extraction_config,
                esc50_config=dataset_config,  # dataset_config contains ESC50Config
                augmentation_config=augmentation_config,
                use_filename_based_splits=True
            )
            
            # Get model factory function
            model_factory = ModelFactory.get_model_factory(
                general_config=general_config,
                feature_extraction_config=feature_extraction_config,
                dataset_config=dataset_config,
                peft_config=peft_config
            )
            
            # Create PTL trainer
            trainer = PTLTrainer(
                general_config=general_config,
                feature_extraction_config=feature_extraction_config,
                dataset_config=dataset_config,
                peft_config=peft_config,
                wandb_config=wandb_config,
                sweep_config=sweep_config_obj,
                data_module=data_module,
                model_factory=model_factory,
                augmentation_config=augmentation_config,
                optimizer_config=optimizer_config,
                peft_scheduling_config=peft_scheduling_config
            )
            
            # Train model
            if general_config.use_kfold:
                results = trainer.k_fold_cross_validation()
                # Log results summary
                wandb.log(results["avg_metrics"])
                
                # Log inference metrics if available
                inference_metrics = {k: v for k, v in results["avg_metrics"].items() if k.startswith("average_inference_") or k.startswith("std_inference_")}
                if inference_metrics:
                    print("\nLogging inference metrics to sweep:")
                    for key, value in inference_metrics.items():
                        print(f"  {key}: {value:.4f}")
            else:
                results = trainer.train()
                # Log final test metrics
                wandb.log(results)
                
                # Log inference metrics if available
                inference_metrics = {k: v for k, v in results.items() if k.startswith("inference_")}
                if inference_metrics:
                    print("\nLogging inference metrics to sweep:")
                    for key, value in inference_metrics.items():
                        print(f"  {key}: {value:.4f}")
            
            # Ensure proper cleanup for distributed training
            wandb.finish()
                        
    except Exception as e:
        print(f"Error in model_pipeline: {e}")
        import traceback
        traceback.print_exc()
        # Ensure cleanup even on error
        try:
            wandb.finish()
        except:
            pass
        # Re-raise the exception so WandB can handle it properly
        raise e

def set_nested_value(config_dict: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a nested value in a dictionary using dot notation.
    
    Args:
        config_dict: The dictionary to update
        key_path: Dot-separated path to the key (e.g., 'peft_scheduling.schedule.0.peft_method')
        value: The value to set
    """
    keys = key_path.split('.')
    current = config_dict
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        # Handle numeric keys (for list indices)
        if key.isdigit():
            key = int(key)
            # Ensure the list is long enough
            while len(current) <= key:
                current.append({})
            current = current[key]
        else:
            # Create the key if it doesn't exist
            if key not in current:
                current[key] = {}
            current = current[key]
    
    # Set the final value
    final_key = keys[-1]
    if final_key.isdigit():
        final_key = int(final_key)
        # Ensure the list is long enough
        while len(current) <= final_key:
            current.append({})
    
    current[final_key] = value

def get_mixed_params(general_config: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine sweep configuration with general configuration.
    
    Args:
        general_config: General configuration dictionary
        sweep_config: Sweep configuration dictionary
        
    Returns:
        Mixed parameters dictionary with proper nested structure
    """
    # Start with a deep copy of the general config to maintain structure
    import copy
    mixed_params = copy.deepcopy(general_config)
    
    # Modify distributed training settings for sweep compatibility
    if mixed_params['general']['distributed_training']:
        # Use ddp_spawn strategy which is more compatible with sweeps
        mixed_params['general']['strategy'] = 'ddp_spawn'
        print("Modified distributed training strategy to ddp_spawn for sweep compatibility")
    
    # Get the adapter type from sweep config or fall back to general config
    try: 
        peft_name = str(sweep_config['adapter_type'])
    except KeyError as e:
        ic("the adapter type is not included in the sweep config, defaulting to general config's: ", e)
        peft_name = str(general_config['general']['adapter_type']) 
   
    # Handle optimizer parameters specially since they were flattened in the sweep config
    optimizer_params = {}
    
    # Update general section with sweep config values
    for key, value in sweep_config.items():
        print(f"Processing sweep parameter: {key} = {value}")
        
        # Handle nested parameters with dot notation (e.g., peft_scheduling.schedule.0.peft_method)
        if '.' in key:
            try:
                set_nested_value(mixed_params, key, value)
                print(f"Set nested parameter: {key} = {value}")
                continue
            except Exception as e:
                print(f"Warning: Could not set nested parameter {key}: {e}")
                # Fall through to the original logic
        
        # Handle flattened optimizer parameters
        if key in ['optimizer_type', 'learning_rate', 'weight_decay', 'gradient_clipping_enabled']:
            if key == 'optimizer_type':
                optimizer_params['optimizer_type'] = value
            elif key == 'learning_rate':
                # Map learning_rate to the appropriate optimizer section
                optimizer_type = sweep_config.get('optimizer_type', mixed_params['optimizer']['optimizer_type'])
                if optimizer_type == 'adamw':
                    if 'adamw' not in optimizer_params:
                        optimizer_params['adamw'] = mixed_params['optimizer']['adamw'].copy()
                    optimizer_params['adamw']['lr'] = value
                elif optimizer_type == 'adam':
                    if 'adam' not in optimizer_params:
                        optimizer_params['adam'] = mixed_params['optimizer']['adam'].copy()
                    optimizer_params['adam']['lr'] = value
                elif optimizer_type == 'adamspd':
                    if 'adamspd' not in optimizer_params:
                        optimizer_params['adamspd'] = mixed_params['optimizer']['adamspd'].copy()
                    optimizer_params['adamspd']['lr'] = value
            elif key == 'weight_decay':
                # Map weight_decay to the appropriate optimizer section
                optimizer_type = sweep_config.get('optimizer_type', mixed_params['optimizer']['optimizer_type'])
                if optimizer_type == 'adamw':
                    if 'adamw' not in optimizer_params:
                        optimizer_params['adamw'] = mixed_params['optimizer']['adamw'].copy()
                    optimizer_params['adamw']['weight_decay'] = value
                elif optimizer_type == 'adamspd':
                    if 'adamspd' not in optimizer_params:
                        optimizer_params['adamspd'] = mixed_params['optimizer']['adamspd'].copy()
                    optimizer_params['adamspd']['weight_decay'] = value
                # Note: Adam doesn't typically use weight_decay in the same way
            elif key == 'gradient_clipping_enabled':
                optimizer_params['gradient_clipping_enabled'] = value
        elif key in mixed_params['general']:
            mixed_params['general'][key] = value
        elif key in mixed_params.get('dataset', {}):
            mixed_params['dataset'][key] = value
        elif key in mixed_params.get('feature_extraction', {}):
            mixed_params['feature_extraction'][key] = value
        elif key in mixed_params.get('augmentations', {}):
            mixed_params['augmentations'][key] = value
        elif key in mixed_params.get(peft_name, {}):
            mixed_params[peft_name][key] = value
        else:
            # If key doesn't exist in any section, add it to general
            mixed_params['general'][key] = value
    
    # Update optimizer config with the processed parameters
    if optimizer_params:
        mixed_params['optimizer'].update(optimizer_params)
    
    # Ensure inference_size is properly set if not in sweep_config
    if 'inference_size' not in sweep_config and 'inference_size' in general_config['general']:
        mixed_params['general']['inference_size'] = general_config['general']['inference_size']
    
    # Debug: Print the final peft_scheduling configuration
    if 'peft_scheduling' in mixed_params:
        print(f"Final peft_scheduling config: {mixed_params['peft_scheduling']}")
    
    return mixed_params

def main():
    """
    Main function to initialize and run the sweep.
    """
    # Setup WandB for spawned processes (required for ddp_spawn)
    wandb.setup()
    
    # Load configuration
    config = load_config(config_path="configs/config.yaml")
    
    # Set random seeds for reproducibility
    SEED = config['general']['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Print information about inference evaluation
    inference_size = config['general'].get('inference_size', 0.0)
    if inference_size > 0:
        print(f"\nInference evaluation is enabled with {inference_size * 100:.1f}% of data reserved for inference.")
        print("Inference metrics will be logged to WandB for each sweep run.")
    else:
        print("\nInference evaluation is disabled (inference_size is 0).")
        print("To enable inference evaluation, set inference_size > 0 in config.yaml.")
    
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
    