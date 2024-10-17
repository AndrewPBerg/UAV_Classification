import yaml
from helper.util import train_test_split_custom, wandb_login
from helper.engine import train, inference_loop
from helper.model import auto_extractor, custom_AST

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb
import random

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

def get_mixed_params(sweep_config, general_config):
    # copy sweep_config to result dict
    result = sweep_config 

    for key, value in general_config.items():
        # just like LeetCode isDuplicate problem
        if key in result:
            pass
        else:
            # if not already occupied by sweep config value add the current general parameter
            result[key] = value
    
    # final dict should contain all of the config.yaml parameters
    # where sweep parameters have priority over duplicates in the general configuration
    return result

def make(config):

    MODEL_NAME = config['model_name']
    DATA_PATH = config['data_path']
    TEST_SIZE = config['test_size']
    SEED = config['seed']
    INFERENCE_SIZE = config['inference_size']
    VAL_SIZE = config['val_size']
    BATCH_SIZE = config['batch_size']
    AUGMENTATIONS = config['augmentations']

    NUM_AUGMENTATIONS = config['num_augmentations']
    AUGMENTATIONS_PER_SAMPLE = config['num_train_transforms']
    LEARNING_RATE = config['learning_rate']
    NUM_CUDA_WORKERS = config['num_cuda_workers']
    feature_extractor = auto_extractor(MODEL_NAME)

    # Get the selected augmentations directly from the config
    num_augmentations = NUM_AUGMENTATIONS
    selected_augmentations = random.sample(k=num_augmentations, population=AUGMENTATIONS)
    
    # Log the selected augmentations
    wandb.log({"selected_augmentations": selected_augmentations})

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

    num_classes = len(train_dataset.get_classes() + test_dataset.get_classes() + inference_dataset.get_classes())

    # Make the model
    model = custom_AST(MODEL_NAME, num_classes, device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # type: ignore #AdamW class isn't exported correctly :(
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    return model, train_loader, val_loader, test_loader, inference_loader, criterion, optimizer, scheduler, num_classes

def model_pipeline(config=None):
    with wandb.init(config=config):
        # Updated wandb configuration handling
        config = wandb.config

        mixed_params = get_mixed_params(config, general_config)

        # Assign dictionary values to variables
        EPOCHS = mixed_params['epochs']
        ACCUMULATION_STEPS = mixed_params['accumulation_steps']
        PATIENCE = mixed_params['patience']

        model, train_loader, val_loader, test_loader, inference_loader, criterion, optimizer, scheduler, num_classes = make(mixed_params)
        print(model)

        results = train(model,
                        train_dataloader=train_loader,
                        test_dataloader=test_loader,
                        val_dataloader=val_loader,
                        optimizer=optimizer,
                        scheduler=scheduler, # type: ignore
                        loss_fn=criterion,
                        epochs=EPOCHS,
                        device=device,
                        num_classes=num_classes,
                        accumulation_steps=ACCUMULATION_STEPS,
                        patience=PATIENCE)

        inference_loop(model=model,
                       device=device,
                       loss_fn=criterion,
                       inference_loader=inference_loader)

    return model, results

def main():

    SEED = general_config['seed']
    PROJECT_NAME = general_config['project_name']
    SWEEP_COUNT = general_config['sweep_count']

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    wandb_login()

    sweep_id = wandb.sweep(config['sweep'], project=PROJECT_NAME)
    wandb.agent(sweep_id, model_pipeline, count=SWEEP_COUNT)

if __name__ == "__main__":
    main()
