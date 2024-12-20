# DESCRIPTION
from helper.util import train_test_split_custom, save_model, wandb_login, calculated_load_time, generate_model_image, k_fold_split_custom
from helper.engine import train, inference_loop
from helper.fold_engine import k_fold_cross_validation
from helper.ast import custom_AST
from helper.models import TorchCNN
from helper.cnn_feature_extractor import CNNFeatureExtractor
from transformers import ASTFeatureExtractor

import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim import Adam
import torch.nn as nn
from torchinfo import summary
import yaml
from timeit import default_timer as timer 
import wandb
from icecream import ic
from torch.cuda.amp import GradScaler, autocast
import sys
import numpy as np

def get_model_and_optimizer(config, device):
    """
    Factory function to create model and optimizer based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        device (str): Device to put model on
    
    Returns:
        tuple: (model, optimizer, training_function, feature_extractor)
    """
    model_type = config['model_type']
    num_classes = config['num_classes']
    learning_rate = config['learning_rate']
    
    if model_type == "AST":
        model, feature_extractor, adaptor_config = custom_AST(num_classes, config['adaptor_type'])
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        train_fn = train
    elif model_type == "CNN":
        model = TorchCNN(
            num_classes=num_classes,
            hidden_units=config['cnn_config']['hidden_units']
        )
        optimizer = Adam(model.parameters(), lr=learning_rate)
    
        train_fn = train
        
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
        
    model = model.to(device)
    return model, optimizer, train_fn, feature_extractor

def main():
    # start logging data load time
    start = timer()
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    general_config = config['general']
    run_config = config['wandb']

    # Extract configuration
    DATA_PATH = general_config['data_path']
    BATCH_SIZE = general_config['batch_size']
    SEED = general_config['seed']
    EPOCHS = general_config['epochs']
    NUM_CUDA_WORKERS = general_config['num_cuda_workers']
    PINNED_MEMORY = general_config['pinned_memory']
    SHUFFLED = general_config['shuffled']
    ACCUMULATION_STEPS = general_config['accumulation_steps']
    TRAIN_PATIENCE = general_config['patience']
    SAVE_MODEL = general_config['save_model']
    TEST_SIZE = general_config['test_size']
    INFERENCE_SIZE = general_config['inference_size']
    VAL_SIZE = general_config['val_size']
    AUGMENTATIONS_PER_SAMPLE = general_config['augmentations_per_sample']
    AUGMENTATIONS = general_config['augmentations']
    USE_WANDB = general_config['use_wandb']
    NUM_CLASSES = general_config['num_classes']
    TORCH_VIZ = general_config['torch_viz']
    USE_KFOLD = general_config['use_kfold']
    K_FOLDS = general_config['k_folds']

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.RandomState(SEED)

    # Get model, optimizer, training function and feature extractor
    model, optimizer, train_fn, feature_extractor = get_model_and_optimizer(general_config, device)

    summary(model,
            col_names=["num_params","trainable"],
            col_width=20,
            row_settings=["var_names"])
    print(model)
    
    if TORCH_VIZ:
        generate_model_image(model, device)

    # Initialize gradient scaler for mixed precision
    scaler = torch.amp.GradScaler()

    # Enable cudnn benchmarking for faster training
    torch.backends.cudnn.benchmark = True

    if USE_WANDB:
        wandb_login()
        
        wandb.init(
            project=run_config['project'],
            name=run_config['name'],
            reinit=run_config['reinit'],
            notes=run_config['notes'],
            tags=run_config['tags'],
            dir=run_config['dir'],
            config=general_config
        )
    
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    
    if USE_KFOLD:
        # K-fold cross validation
        fold_datasets, inference_dataset = k_fold_split_custom(
            DATA_PATH,
            feature_extractor=feature_extractor,
            k_folds=K_FOLDS,
            inference_size=INFERENCE_SIZE,
            seed=SEED,
            augmentations_per_sample=AUGMENTATIONS_PER_SAMPLE,
            augmentations=AUGMENTATIONS,
            config=general_config
        )
        
        # Define model and optimizer creation functions
        def model_fn():
            model, optimizer, _, _ = get_model_and_optimizer(general_config, device)
            return model
        
        def optimizer_fn(parameters):
            if general_config['model_type'] == "AST":
                return AdamW(parameters, lr=general_config['learning_rate'])
            else:
                return Adam(parameters, lr=general_config['learning_rate'])
        
        def scheduler_fn(optimizer):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=2
            )
        
        # Perform k-fold cross validation
        k_fold_results = k_fold_cross_validation(
            model_fn=model_fn,
            fold_datasets=fold_datasets,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            loss_fn=loss_fn,
            device=device,
            num_classes=NUM_CLASSES,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_CUDA_WORKERS,
            pin_memory=PINNED_MEMORY,
            shuffle=SHUFFLED,
            accumulation_steps=ACCUMULATION_STEPS,
            patience=TRAIN_PATIENCE,
            scaler=scaler
        )

    else:
        # Original train-test split code
        train_dataset, val_dataset, test_dataset, inference_dataset = train_test_split_custom(
            DATA_PATH,
            feature_extractor=feature_extractor,
            test_size=TEST_SIZE,
            seed=SEED,
            inference_size=INFERENCE_SIZE,
            augmentations_per_sample=AUGMENTATIONS_PER_SAMPLE,
            val_size=VAL_SIZE,
            augmentations=AUGMENTATIONS,
            config=general_config
        )

        # Create data loaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_CUDA_WORKERS,
            pin_memory=PINNED_MEMORY,
            shuffle=SHUFFLED
        )
        
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_CUDA_WORKERS,
            pin_memory=PINNED_MEMORY,
            shuffle=SHUFFLED
        )
        
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_CUDA_WORKERS,
            pin_memory=PINNED_MEMORY,
            shuffle=SHUFFLED
        )
        
        inference_dataloader = DataLoader(
            dataset=inference_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_CUDA_WORKERS,
            pin_memory=PINNED_MEMORY,
            shuffle=SHUFFLED
        )
        
        end = timer()
        total_load_time = calculated_load_time(start, end)
        print(f"Load time in on path: {DATA_PATH} --> {total_load_time}")
        
        if wandb.run is not None:
            wandb.log({"load_time": total_load_time})
        
        train_fn(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=EPOCHS,
            device=device,
            num_classes=NUM_CLASSES,
            accumulation_steps=ACCUMULATION_STEPS,
            patience=TRAIN_PATIENCE,
            scaler=scaler
        )
        
        # Run inference after training
        inference_loop(
            model=model,
            inference_loader=inference_dataloader,
            loss_fn=loss_fn,
            device=device,
            num_classes=NUM_CLASSES
        )

    if USE_WANDB:
        wandb.finish()

    if SAVE_MODEL:
        model_name = f"{general_config['model_type']}_classifier.pt"
        save_model(
            model=model,
            target_dir="saved_models",
            model_name=model_name
        )

if __name__ == "__main__":
    main()
