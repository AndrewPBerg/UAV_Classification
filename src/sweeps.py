import yaml
from helper.util import train_test_split_custom, wandb_login
from helper.engine import train, inference_loop
from helper.model import auto_extractor, custom_AST

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access yaml general configuration 
general_config = config['general']

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataloader(dataset, batch_size, num_workers=general_config['num_cuda_workers'], shuffle=True):
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=True,
                      shuffle=shuffle)

def make(config):
    # Make the data
    feature_extractor = auto_extractor(general_config['model_name'])
    # Get the selected augmentations directly from the config
    selected_augmentations = config.augmentations
    
    # Log the selected augmentations
    wandb.log({"selected_augmentations": selected_augmentations})

    # Updated dataset loading to match new format
    train_dataset, val_dataset, test_dataset, inference_dataset = train_test_split_custom(
        general_config['data_path'], 
        feature_extractor, 
        test_size=general_config['test_size'], 
        seed=general_config['seed'], 
        inference_size=general_config['inference_size'],
        augmentations_per_sample=config['num_train_transforms'],
        augmentation_probability=config['augmentation_probability']
    )
    
    inference_loader = create_dataloader(inference_dataset, config.batch_size)
    train_loader = create_dataloader(train_dataset, config.batch_size)
    val_loader = create_dataloader(val_dataset, config.batch_size)
    test_loader = create_dataloader(test_dataset, config.batch_size)

    num_classes = len(train_dataset.get_classes() + test_dataset.get_classes() + inference_dataset.get_classes())

    # Make the model
    model = custom_AST(general_config['model_name'], num_classes, device)

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
                        optimizer=optimizer, # type: ignore
                        scheduler=scheduler, # type: ignore
                        loss_fn=criterion,
                        epochs=general_config['epochs'],
                        device=device,
                        num_classes=num_classes,
                        accumulation_steps=config.accumulation_steps, # type: ignore
                        patience=general_config['patience'])

        inference_loop(model=model,
                       device=device,
                       loss_fn=criterion,
                       inference_loader=inference_loader)

    return model, results

def main():
    torch.manual_seed(general_config['seed'])
    torch.cuda.manual_seed(general_config['seed'])

    wandb_login()

    sweep_id = wandb.sweep(config['sweep'], project=general_config['project_name'])
    wandb.agent(sweep_id, model_pipeline, count=general_config['sweep_count'])

if __name__ == "__main__":
    main()