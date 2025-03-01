import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
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
    load_configs
)
from configs.augmentation_config import AugmentationConfig
from helper.util import wandb_login

def create_trainer_factory(config_path: str = 'configs/config.yaml'):
    """
    Create a factory function that produces PTLTrainer instances based on configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Function that creates PTLTrainer instances
    """
    def trainer_factory(mixed_params: Dict[str, Any]):
        """
        Factory function to create PTLTrainer instances.
        
        Args:
            mixed_params: Configuration parameters
            
        Returns:
            PTLTrainer instance
        """
        # Load the base configuration
        with open(config_path, 'r') as file:
            base_config = yaml.safe_load(file)
        
        # Update base config with mixed params
        for section in base_config:
            if isinstance(base_config[section], dict):
                for key, value in mixed_params.items():
                    if key in base_config[section]:
                        base_config[section][key] = value
        
        # Create configuration objects
        general_config, feature_extraction_config, peft_config, wandb_config, sweep_config, augmentation_config = load_configs(base_config)
        
        # Override specific parameters from mixed_params
        if 'learning_rate' in mixed_params:
            general_config.learning_rate = mixed_params['learning_rate']
        if 'batch_size' in mixed_params:
            general_config.batch_size = mixed_params['batch_size']
        if 'epochs' in mixed_params:
            general_config.epochs = mixed_params['epochs']
        
        # Create data module
        data_module = AudioDataModule(
            general_config=general_config,
            feature_extraction_config=feature_extraction_config,
            augmentation_config=augmentation_config
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
            sweep_config=sweep_config,
            data_module=data_module,
            model_factory=model_factory
        )
        
        return trainer
    
    return trainer_factory

def main():
    """
    Main function to run a sweep using the AudioDataModule.
    """
    # Set up configuration path
    config_path = 'configs/config.yaml'
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create configuration objects
    general_config, feature_extraction_config, peft_config, wandb_config, sweep_config, augmentation_config = load_configs(config)
    
    # Check if sweeps are enabled
    if not general_config.use_sweep:
        raise ValueError("Sweeps are not enabled in the configuration. Set use_sweep=True to enable sweeps.")
    # Enable wandb logging
    general_config.use_wandb = True
    
    # Ensure wandb is logged in
    wandb_login()
    
    # Set random seeds for reproducibility
    torch.manual_seed(general_config.seed)
    torch.cuda.manual_seed(general_config.seed)
    np.random.seed(general_config.seed)
    
    # Create data module with sweep configuration
    data_module = AudioDataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        augmentation_config=augmentation_config,
        wandb_config=wandb_config,
        sweep_config=sweep_config
    )
    
    # Create trainer factory
    trainer_factory = create_trainer_factory(config_path)
    
    # Create model pipeline for sweep
    model_pipeline = data_module.create_sweep_pipeline(trainer_factory, config_path)
    
    # Run sweep
    data_module.run_sweep(model_pipeline, config_path)

if __name__ == "__main__":
    main() 