import os
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
    GeneralConfig, 
    FeatureExtractionConfig,
    WandbConfig,
    SweepConfig,
    wandb_config_dict
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

def get_mixed_params(sweep_config: Dict[str, Any], general_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine sweep configuration with general configuration.
    
    Args:
        sweep_config: Sweep configuration from wandb
        general_config: General configuration from YAML file
        
    Returns:
        Combined configuration dictionary
    """
    mixed_params = {}
    
    # Add all general config parameters first
    for key, value in general_config.items():
        mixed_params[key] = value
    
    # Override with sweep config parameters
    for key, value in sweep_config.items():
        mixed_params[key] = value
    
    return mixed_params

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
        mixed_params = get_mixed_params(dict(config), general_config_dict)
        
        # Create configuration objects
        general_config = GeneralConfig(
            model_type=mixed_params.get('model_type', 'AST'),
            num_classes=mixed_params.get('num_classes', 4),
            epochs=mixed_params.get('epochs', 30),
            batch_size=mixed_params.get('batch_size', 32),
            learning_rate=mixed_params.get('learning_rate', 1e-4),
            seed=mixed_params.get('seed', 42),
            patience=mixed_params.get('patience', 5),
            accumulation_steps=mixed_params.get('accumulation_steps', 1),
            use_wandb=True,
            save_model=True,
            use_kfold=mixed_params.get('use_kfold', False),
            k_folds=mixed_params.get('k_folds', 5),
            adapter_type=mixed_params.get('adapter_type', 'lora')
        )
        
        # Feature extraction config
        feature_extraction_config = FeatureExtractionConfig(
            type=mixed_params.get('feature_type', 'melspectrogram'),
            sampling_rate=mixed_params.get('sampling_rate', 16000),
            n_mels=mixed_params.get('n_mels', 128),
            n_fft=mixed_params.get('n_fft', 1024),
            hop_length=mixed_params.get('hop_length', 512),
            power=mixed_params.get('power', 2.0),
            n_mfcc=mixed_params.get('n_mfcc', 40)
        )
        
        # WandB config
        wandb_config = WandbConfig(
            project=yaml_config['wandb']['project'],
            name=f"sweep_{wandb.run.id}",
            tags=["sweep"],
            notes="Automated hyperparameter sweep"
        )
        
        # PEFT config (if needed)
        peft_config = None
        if general_config.model_type.lower() == 'ast' and general_config.adapter_type != 'none':
            from peft import LoraConfig, TaskType
            
            # Example LoRA config - adjust parameters based on sweep
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=mixed_params.get('lora_r', 8),
                lora_alpha=mixed_params.get('lora_alpha', 16),
                lora_dropout=mixed_params.get('lora_dropout', 0.1),
                target_modules=["query", "key", "value"],
                bias="none"
            )
        
        # Create data module
        data_module = AudioDataModule(
            data_path=mixed_params.get('data_path', 'data'),
            batch_size=general_config.batch_size,
            test_size=mixed_params.get('test_size', 0.2),
            val_size=mixed_params.get('val_size', 0.2),
            inference_size=mixed_params.get('inference_size', 0.1),
            num_workers=mixed_params.get('num_cuda_workers', 4),
            seed=general_config.seed,
            feature_extraction_config=feature_extraction_config,
            augmentations=mixed_params.get('augmentations', ['None']),
            augmentations_per_sample=mixed_params.get('augmentations_per_sample', 1)
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
    main()