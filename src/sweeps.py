import yaml
from helper.util import train_test_split_custom, wandb_login
from helper.engine import train, inference_loop
from helper.ast import custom_AST
from helper.util import get_mixed_params
from helper.cnn_engine import TorchCNN
from helper.cnn_feature_extractor import CNNFeatureExtractor

import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
import torch.nn as nn
import wandb
import random
from torch.cuda.amp import GradScaler, autocast
import sys
from helper.util import count_parameters
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, cast, TypeVar
from transformers import PreTrainedModel
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraConfig
from peft.tuners.ia3 import IA3Config
from peft.tuners.adalora import AdaLoraConfig
from peft.tuners.oft import OFTConfig
from helper.MoA import AST_MoA, AST_SoftMoA

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access yaml general configuration 
general_config = config['general']

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

def create_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool = True, pin_memory: bool = True) -> DataLoader:
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      shuffle=shuffle)

ModelType = Union[PreTrainedModel, nn.Module, PeftModel, AST_MoA, AST_SoftMoA, TorchCNN]
AdaptorConfigType = Union[Dict[str, Any], LoraConfig, IA3Config, AdaLoraConfig, OFTConfig]
M = TypeVar('M', bound=ModelType)

def get_model_and_optimizer(config: Dict[str, Any], device: torch.device) -> Tuple[ModelType, Optimizer, Any, Dict[str, Any]]:
    """
    Factory function to create model and optimizer based on configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        device (torch.device): Device to put model on
    
    Returns:
        tuple: (model, optimizer, feature_extractor, adaptor_config)
    """
    model_type = config.get('model_type', 'AST')  # Default to AST if not specified
    num_classes = config['num_classes']
    learning_rate = config['learning_rate']
    
    model: Optional[ModelType] = None
    optimizer: Optional[Optimizer] = None
    feature_extractor: Any = None
    adaptor_config: Dict[str, Any] = {}
    
    if model_type == "AST":
        ast_model, ast_feature_extractor, ast_adaptor_config = custom_AST(num_classes, config['adaptor_type'], sweep_config=config)
        model = ast_model
        feature_extractor = ast_feature_extractor
        if isinstance(ast_adaptor_config, dict):
            adaptor_config = ast_adaptor_config
        else:
            adaptor_config = {"config": ast_adaptor_config}
        if model is not None:
            optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif model_type == "CNN":
        cnn_model = TorchCNN(
            num_classes=num_classes,
            hidden_units=config['cnn_config']['hidden_units']
        )
        model = cnn_model
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Get feature extraction parameters from config
        fe_config = config['cnn_config']['feature_extraction']
        feature_extractor = CNNFeatureExtractor(
            sampling_rate=fe_config['sampling_rate'],
            n_mels=fe_config['n_mels'],
            n_fft=fe_config['n_fft'],
            hop_length=fe_config['hop_length'],
            power=fe_config['power']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if model is None or optimizer is None:
        raise ValueError("Failed to initialize model or optimizer")
        
    if isinstance(model, (PreTrainedModel, nn.Module)):
        model = cast(nn.Module, model).to(device)
    
    return model, optimizer, feature_extractor, adaptor_config

def make(config: Dict[str, Any]) -> Tuple[ModelType, DataLoader, DataLoader, DataLoader, DataLoader, nn.CrossEntropyLoss, Optimizer, _LRScheduler, int]:
    DATA_PATH = config['data_path']
    TEST_SIZE = config['test_size']
    SEED = config['seed']
    INFERENCE_SIZE = config['inference_size']
    VAL_SIZE = config['val_size']
    BATCH_SIZE = config['batch_size']
    AUGMENTATIONS = config['augmentations']
    AUGMENTATIONS_PER_SAMPLE = config['augmentations_per_sample']
    NUM_CUDA_WORKERS = config['num_cuda_workers']
    NUM_CLASSES = general_config['num_classes']
    
    try:    
        NUM_AUGMENTATIONS = config['num_augmentations'] # sweeps exclusive
    except KeyError: # NUM_AUGMENTATIONS isn't being utilized
        NUM_AUGMENTATIONS = None

    if NUM_AUGMENTATIONS is None:
        # this means config is setup for general augmentation use
        selected_augmentations = AUGMENTATIONS
    elif NUM_AUGMENTATIONS == 0:
        selected_augmentations = ["None"]
    else:
        selected_augmentations = random.sample(AUGMENTATIONS, NUM_AUGMENTATIONS)

    print(f"selection augs: {selected_augmentations}")
    wandb.log({"selected_augmentations": selected_augmentations})

    # Get model, optimizer and feature extractor using factory function
    model, optimizer, feature_extractor, adaptor_config = get_model_and_optimizer(config, device)
    
    # Add adaptor config to the general config if it exists
    if adaptor_config:
        config['adaptor_config'] = adaptor_config

    train_dataset, val_dataset, test_dataset, inference_dataset = train_test_split_custom(
        DATA_PATH, 
        feature_extractor, 
        test_size=TEST_SIZE, 
        seed=SEED,
        inference_size=INFERENCE_SIZE,
        augmentations_per_sample=AUGMENTATIONS_PER_SAMPLE,
        val_size=VAL_SIZE,
        augmentations=selected_augmentations,
        config=config
    )

    inference_loader = create_dataloader(inference_dataset, BATCH_SIZE, num_workers=NUM_CUDA_WORKERS)
    train_loader = create_dataloader(train_dataset, BATCH_SIZE, num_workers=NUM_CUDA_WORKERS)
    val_loader = create_dataloader(val_dataset, BATCH_SIZE, num_workers=NUM_CUDA_WORKERS)
    test_loader = create_dataloader(test_dataset, BATCH_SIZE, num_workers=NUM_CUDA_WORKERS)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    scheduler = cast(_LRScheduler, ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2))

    return model, train_loader, val_loader, test_loader, inference_loader, criterion, optimizer, scheduler, NUM_CLASSES

def model_pipeline(config= None):
    with wandb.init(config=config):
        config = wandb.config
        mixed_params = get_mixed_params(config, general_config)

        # Assign dictionary values to variables
        EPOCHS = int(mixed_params['epochs'])
        ACCUMULATION_STEPS = int(mixed_params['accumulation_steps'])
        PATIENCE = int(mixed_params['patience'])
        NUM_CLASSES = int(mixed_params['num_classes'])

        model, train_loader, val_loader, test_loader, inference_loader, criterion, optimizer, scheduler, num_classes = make(mixed_params)
        print(model)
        total_params, trainable_params = count_parameters(model)
        wandb.log({"total_params": total_params})
        wandb.log({"trainable_parms": trainable_params})

        # Move model to device before creating scaler
        if isinstance(model, (PreTrainedModel, nn.Module)):
            model = cast(nn.Module, model).to(device)
        
        # Convert model to float32 before training
        model = model.float()

        # Initialize gradient scaler for mixed precision
        scaler = GradScaler()

        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True

        results = train(
            model=cast(nn.Module, model),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            val_dataloader=val_loader,
            optimizer=cast(AdamW, optimizer),
            scheduler=scheduler,
            loss_fn=criterion,
            epochs=EPOCHS,
            device=device_str,
            num_classes=NUM_CLASSES,
            accumulation_steps=ACCUMULATION_STEPS,
            patience=PATIENCE,
            scaler=scaler
        )

        inference_loop(
            model=model,
            device=device_str,
            loss_fn=criterion,
            inference_loader=inference_loader,
            num_classes=NUM_CLASSES
        )

    return model, results

def main():
    SEED = general_config['seed']
    PROJECT_NAME = config['wandb']['project']
    SWEEP_COUNT = config['sweep']['count']

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.RandomState(SEED)

    wandb_login()

    sweep_id = wandb.sweep(config['sweep'], project=PROJECT_NAME)
    wandb.agent(sweep_id, model_pipeline, count=SWEEP_COUNT)

if __name__ == "__main__":
    main()
