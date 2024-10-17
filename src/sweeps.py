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

# Access yaml general configuration 
general_config = config['general']
augmentations_config = config['augmentations']

# Assign dictionary values to variables
NUM_CUDA_WORKERS = general_config['num_cuda_workers']
MODEL_NAME = general_config['model_name']
DATA_PATH = general_config['data_path']
TEST_SIZE = general_config['test_size']
SEED = general_config['seed']
INFERENCE_SIZE = general_config['inference_size']
VAL_SIZE = general_config['val_size']
BATCH_SIZE = general_config['batch_size']
EPOCHS = general_config['epochs']
ACCUMULATION_STEPS = general_config['accumulation_steps']
PATIENCE = general_config['patience']
PROJECT_NAME = general_config['project_name']
SWEEP_COUNT = general_config['sweep_count']

AUGMENTATIONS = augmentations_config['augmentations']
# access transform parameters by indexing the variable with the parameter name
# ex: PITCH_SHIFT_MIN_SEMITONES = PITCH_SHIFT['min_semitones']
PITCH_SHIFT = augmentations_config['pitch_shift']
TIME_STRETCH = augmentations_config['time_stretch']


device = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataloader(dataset, batch_size, num_workers=NUM_CUDA_WORKERS, shuffle=True):
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=True,
                      shuffle=shuffle)
def get_augmentation_params(config, augmentations_config):
    aug_params = {}

    for aug in config.augmentations:
        aug_params[aug] = {}
        for param in config[aug]:
            aug_params[aug][param] = random.choice(augmentations_config[aug][param])
    return aug_params

def make(config):
    # Make the data
    feature_extractor = auto_extractor(MODEL_NAME)

    # aug_params = get_augmentation_params(config, general_config, augmentations_config)
    print(f"Config: {config}")
    print(f"Config items: {config.items()}")
    # Get the selected augmentations directly from the config
    num_augmentations = config.num_augmentations
    selected_augmentations = random.sample(k=num_augmentations, population=AUGMENTATIONS)
    
    # Log the selected augmentations
    wandb.log({"selected_augmentations": selected_augmentations})

    # Updated dataset loading to match new format
    train_dataset, val_dataset, test_dataset, inference_dataset = train_test_split_custom(
        DATA_PATH, 
        feature_extractor, 
        test_size=TEST_SIZE, 
        seed=SEED,
        inference_size=INFERENCE_SIZE,
        augmentations_per_sample=config.num_train_transforms,
        val_size=VAL_SIZE,
        augmentations=selected_augmentations,
        config=config
    )

    inference_loader = create_dataloader(inference_dataset, BATCH_SIZE)
    train_loader = create_dataloader(train_dataset, BATCH_SIZE)
    val_loader = create_dataloader(val_dataset, BATCH_SIZE)
    test_loader = create_dataloader(test_dataset, BATCH_SIZE)

    num_classes = len(train_dataset.get_classes() + test_dataset.get_classes() + inference_dataset.get_classes())

    # Make the model
    model = custom_AST(MODEL_NAME, num_classes, device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate) # type: ignore
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    return model, train_loader, val_loader, test_loader, inference_loader, criterion, optimizer, scheduler, num_classes

def model_pipeline(config=None):
    with wandb.init(config=config):
        # Updated wandb configuration handling
        config = wandb.config

        model, train_loader, val_loader, test_loader, inference_loader, criterion, optimizer, scheduler, num_classes = make(config)
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
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    wandb_login()

    sweep_id = wandb.sweep(config['sweep'], project=PROJECT_NAME)
    wandb.agent(sweep_id, model_pipeline, count=SWEEP_COUNT)

if __name__ == "__main__":
    main()
