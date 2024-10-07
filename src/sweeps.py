import yaml
from helper.util import train_test_split_custom, save_model, wandb_login
from helper.engine import sweep_train, inference_loop
from helper.model import auto_extractor, custom_AST

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb


# Load configuration from YAML file
with open('src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

# Access general configuration
general_config = config['general']
# sweep_config = config['sweep']

device = "cuda" if torch.cuda.is_available() else "cpu"

wandb_login("src/.env")


def create_dataloader(dataset, batch_size, num_workers=general_config['num_cuda_workers'], shuffle=True):
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=True,
                      shuffle=shuffle)

def make(config):
    # Make the data
    feature_extractor = auto_extractor(general_config['model_name'])

    # Updated dataset loading to match new format
    train_subset, test_subset, inference_subset = train_test_split_custom(
        general_config['data_path'], 
        feature_extractor, 
        test_size=general_config['test_size'], 
        seed=general_config['seed'], 
        inference_size=general_config['inference_size'], 
        training_transforms=general_config['training_transforms']
    )
    
    inference_loader = create_dataloader(inference_subset, config.batch_size)            
    train_loader = create_dataloader(train_subset, config.batch_size, )
    test_loader = create_dataloader(test_subset, config.batch_size)

    # Make the model
    model = custom_AST(general_config['model_name'], general_config['num_classes'], device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate) # type: ignore
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    return model, train_loader, test_loader, inference_loader, criterion, optimizer, scheduler

def model_pipeline(config=None):
    with wandb.init(config=config):
        # Updated wandb configuration handling
        config = wandb.config

        model, train_loader, test_loader, inference_loader, criterion, optimizer, scheduler = make(config)
        print(model)

        results = sweep_train(model,
                              train_dataloader=train_loader,
                              test_dataloader=test_loader,
                              optimizer=optimizer,
                              scheduler=scheduler, # type: ignore
                              loss_fn=criterion,
                              epochs=config.epochs, # type: ignore
                              device=device,
                              num_classes=general_config['num_classes'],
                              accumulation_steps=config.accumulation_steps, # type: ignore
                              patience=general_config['patience']) # Added patience parameter

        inference_loop(model=model,
                       device=config.device, # type: ignore
                       loss_fn=criterion,
                       inference_loader=inference_loader)
        
        if general_config['save_model']:
            save_model(model=model,
                    target_dir="saved_models",
                    model_name="AST_classifier_true.pt")

    return model, results

def main():
    torch.manual_seed(general_config['seed'])
    torch.cuda.manual_seed(general_config['seed'])

    sweep_id = wandb.sweep(config['sweep'], project=general_config['project_name'])
    wandb.agent(sweep_id, model_pipeline, count=general_config['sweep_count'])



if __name__ == "__main__":
    main()
