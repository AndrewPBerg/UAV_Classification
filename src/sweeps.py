import yaml
from helper.util import train_test_split_custom, wandb_login
from helper.engine import train, inference_loop
from helper.ast import custom_AST
from helper.util import get_mixed_params

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb
import random
from torch.cuda.amp import GradScaler, autocast
import sys
from helper.teleBot import send_message

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# # Access yaml general configuration 
general_config = config['general']

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataloader(dataset, batch_size, num_workers, shuffle=True, pin_memory=True):
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      shuffle=shuffle)

def make(config):

    # MODEL_NAME = config['model_name'] # Deprecated; only AST now
    DATA_PATH = config['data_path']
    TEST_SIZE = config['test_size']
    SEED = config['seed']
    INFERENCE_SIZE = config['inference_size']
    VAL_SIZE = config['val_size']
    BATCH_SIZE = config['batch_size']
    AUGMENTATIONS = config['augmentations']
    AUGMENTATIONS_PER_SAMPLE = config['augmentations_per_sample']
    LEARNING_RATE = config['learning_rate']

    NUM_CUDA_WORKERS = config['num_cuda_workers']
    NUM_CLASSES = general_config['num_classes']
    ADAPTOR_TYPE = config['adaptor_type']
    try:    
        NUM_AUGMENTATIONS = config['num_augmentations'] # sweeps exclusive
    except KeyError: # NUM_AUGMENTATIONS isn't being utilized
        NUM_AUGMENTATIONS = None
    # Get the selected augmentations directly from the config

    if NUM_AUGMENTATIONS is None:
        # this means config is setup for general augmentation use
        selected_augmentations = AUGMENTATIONS
    elif NUM_AUGMENTATIONS == 0:
        selected_augmentations = ["None"]
    else:
        selected_augmentations = random.sample(AUGMENTATIONS, NUM_AUGMENTATIONS)

    print(f"selection augs: {selected_augmentations}")

    # Log the selected augmentations
    wandb.log({"selected_augmentations": selected_augmentations})

    # Update model creation to use custom_AST
    model, feature_extractor, adaptor_config = custom_AST(NUM_CLASSES, ADAPTOR_TYPE, sweep_config=config)
    
    # Add adaptor config to the general config
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
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    return model, train_loader, val_loader, test_loader, inference_loader, criterion, optimizer, scheduler, NUM_CLASSES

def model_pipeline(config=None):
    with wandb.init(config=config):
        config = wandb.config
        mixed_params = get_mixed_params(config, general_config)

        # Assign dictionary values to variables
        EPOCHS = mixed_params['epochs']
        ACCUMULATION_STEPS = mixed_params['accumulation_steps']
        PATIENCE = mixed_params['patience']
        NUM_CLASSES = mixed_params['num_classes']

        model, train_loader, val_loader, test_loader, inference_loader, criterion, optimizer, scheduler, num_classes = make(mixed_params)
        print(model)

        # Move model to device before creating scaler
        model = model.to(device)
        
        # Convert model to float32 before training
        model = model.float()

        # Initialize gradient scaler for mixed precision
        scaler = GradScaler()

        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True

        results = train(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=criterion,
            epochs=EPOCHS,
            device=device,
            num_classes=NUM_CLASSES,
            accumulation_steps=ACCUMULATION_STEPS,
            patience=PATIENCE,
            scaler=scaler
        )

        inference_loop(
            model=model,
            device=device,
            loss_fn=criterion,
            inference_loader=inference_loader
        )

    return model, results

def main():

    SEED = general_config['seed']
    PROJECT_NAME = general_config['project_name']
    SWEEP_COUNT = general_config['sweep_count']
    sweep_config = config['sweep']
    sweep_name = sweep_config['name']

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    wandb_login()

    sweep_id = wandb.sweep(config['sweep'], project=PROJECT_NAME)
    wandb.agent(sweep_id, model_pipeline, count=SWEEP_COUNT)
    send_message(f"Your Run has finished cooking...\n"
                 f"Project: {PROJECT_NAME}\n"
                 f"Sweep Count: {SWEEP_COUNT}\n"
                 f"Sweep Name: {sweep_name}\n"
                 f"Link to Your Wandb Project's page: https://wandb.ai/andberg9-self/projects")

if __name__ == "__main__":
    main()
