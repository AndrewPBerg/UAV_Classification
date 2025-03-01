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
    WandbConfig,
    SweepConfig,
    wandb_config_dict
)
from configs.peft_config import get_peft_config
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
        general_config_dict = yaml_config['general']
        
        # Combine sweep config with general config
        mixed_params = get_mixed_params(sweep_config=yaml_config['sweep'], general_config=yaml_config['general'], 
                                        wandb_config=yaml_config['wandb'], feature_extraction_config=yaml_config['feature_extraction'],
                                        augmentation_config=yaml_config['augmentation'])
        
        ic(mixed_params)
        
        sys.exit(0)
        
        # Create configuration objects
        general_config = GeneralConfig(
            model_type=mixed_params['model_type'],
            num_classes=mixed_params['num_classes'],
            epochs=mixed_params['epochs'],
            batch_size=mixed_params['batch_size'],
            learning_rate=mixed_params['learning_rate'],
            seed=mixed_params['seed'],
            patience=mixed_params['patience'],
            accumulation_steps=mixed_params['accumulation_steps'],
            use_wandb=mixed_params['use_wandb'],
            save_model=mixed_params['save_model'],
            use_kfold=mixed_params['use_kfold'],
            k_folds=mixed_params['k_folds'],
            adapter_type=mixed_params['adapter_type']
        )
        
        # Feature extraction config
        feature_extraction_config = FeatureExtractionConfig(
            type=mixed_params['feature_type'],
            sampling_rate=mixed_params['sampling_rate'],
            n_mels=mixed_params['n_mels'],
            n_fft=mixed_params['n_fft'],
            hop_length=mixed_params['hop_length'],
            power=mixed_params['power'],
            n_mfcc=mixed_params['n_mfcc']
        )
        
        #PEFT config
        peft_config = get_peft_config(mixed_params)
        
        # WandB config
        wandb_config = WandbConfig(
            project=mixed_params['wandb_project'],
            name=f"sweep_{wandb.run.id}",
            tags=["sweep"],
            notes="Automated hyperparameter sweep",
        )
        
        # Sweep config
        sweep_config = SweepConfig(
            project=mixed_params['sweep_project'],
            name=mixed_params['sweep_name'],
            method=mixed_params['sweep_method'],
            metric=mixed_params['sweep_metric'],
            parameters=mixed_params['sweep_parameters']
        )
        
        augmentation_config = AugmentationConfig(
            augmentations=mixed_params['augmentations'],
            augmentations_per_sample=mixed_params['augmentations_per_sample']
        )
        
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

# def get_mixed_params(sweep_config: Dict[str, Any], general_config: GeneralConfig,
#                      wandb_config: WandbConfig, feature_extraction_config: FeatureExtractionConfig,
#                      peft_config: Optional[Any], augmentation_config: Optional[AugmentationConfig]) -> Dict[str, Any]:
def get_mixed_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine sweep configuration with general configuration.
    
    """
    
    # 1. get mixed params from config file, sweeps overrides all values
    mixed_params = {}
    
    # general
    
    # augmentations
    
    # feature extraction
    
    # wandb
    
    # peft (tricky cases for specific peft configs)
    
    """
    pseudo code for this case
    
    if yaml_config['sweep'][general_config['adapter_type']: 
    # essentially checking if this adapter is part of the sweep
    
    then compare the parameters from the sweep config with that adapters    
    """
    
    # # Add all general config parameters first
    # for key, value in general_config.items():
    #     mixed_params[key] = value
    
    # # Override with sweep config parameters
    # for key, value in sweep_config.items():
    #     mixed_params[key] = value

    # # Override with wandb config parameters
    # for key, value in wandb_config.items():
    #     mixed_params[key] = value

    # # Override with feature extraction config parameters
    # for key, value in feature_extraction_config.items():
    #     mixed_params[key] = value

    # # Override with PEFT config parameters
    # # if peft_config:
    # #     for key, value in peft_config.items():
    # #         mixed_params[key] = value

    # # Override with augmentation config parameters
    # for key, value in augmentation_config.dict().items():
    #     mixed_params[key] = value
    
    # return mixed_params

def demo_3_1():
    # Load general config
    yaml_config = load_config(config_path="configs/config.yaml")
    general_config_dict = yaml_config['general']
    
    # Combine sweep config with general config
    mixed_params = get_mixed_params(yaml_config)
    
    ic(mixed_params)
    
    sys.exit(0)
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
        project=config['wandb']['project']
    )
    
    # Run sweep agent
    wandb.agent(
        sweep_id,
        function=model_pipeline,
        count=config['general'].get('sweep_count', 10)
    )

if __name__ == "__main__":
    # main()
    demo_3_1()