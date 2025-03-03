import yaml
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
        
        # mixed_params_table = wandb.Table(data=[mixed_params])
        # broken code for adding the mixed params to a wandb table
        # columns = [f"col_{i}" for i in range(54)]
        # mixed_params_table = wandb.Table(data=[mixed_params], columns=columns)
        
        # # Update wandb config with the mixed parameters
        # wandb.log({"mixed_params_table": mixed_params_table})

        # load into pydantic models w/ load_configs()
        (
        general_config,
        feature_extraction_config,
        peft_config,
        wandb_config,
        sweep_config,
        augmentation_config
         ) = load_configs(mixed_params)
        
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

def get_mixed_params(general_config: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine sweep configuration with general configuration.
    
    Args:
        general_config: General configuration dictionary
        sweep_config: Sweep configuration dictionary
        
    Returns:
        Mixed parameters dictionary
    """

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

    # sweep project and name (this isn't needed, but for my peace of mind)
    mixed_params['project'] = general_config['sweep']['project']
    mixed_params['name'] = general_config['sweep']['name']
    
    try: 
        peft_name = str(sweep_config['adapter_type'])
    except KeyError as e:
        ic("the adapter type is not included in the sweep config, defaulting to general config's: ", e)
        peft_name = str(general_config['general']['adapter_type']) 
   
    # finally update the mixed_params with the correct peft config(sweep agnostic)
    mixed_params.update(general_config[peft_name])
    
    # Update with sweep config values
    mixed_params.update(sweep_config)
    
    # Ensure inference_size is properly set if not in sweep_config
    if 'inference_size' not in sweep_config and 'inference_size' in general_config['general']:
        mixed_params['inference_size'] = general_config['general']['inference_size']
    
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
    