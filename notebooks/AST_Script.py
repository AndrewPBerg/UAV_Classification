# DESCRIPTION
from AST_helper.util import AudioDataset, train_test_split_custom, save_model
from AST_helper.engine import train, inference_loop
from AST_helper.model import auto_extractor, custom_AST
from AST_helper.util import save_model # noqa: F401

import torch
from torch.utils.data import DataLoader
import torch.optim
import torch.nn as nn
from torchinfo import summary
import yaml

from icecream import ic

import wandb

def main():

    with open('notebooks\config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    general_config = config['general']
    run_config = config['wandb']

    data_path = general_config['data_path']
    model_name = general_config["model_name"]
    BATCH_SIZE = general_config['batch_size']
    SEED = general_config['seed']
    EPOCHS = general_config['epochs']
    NUM_CUDA_WORKERS = general_config['num_cuda_workers']
    PINNED_MEMORY = True
    SHUFFLED = general_config['shuffled']
    ACCUMULATION_STEPS = general_config['accumulation_steps']
    learning_rate = general_config['learning_rate']
    TRAIN_PATIENCE = general_config['patience']
    SAVE_MODEL = general_config['save_model']
    test_size = general_config['test_size']
    inference_size = general_config['inference_size']

    wandb_params = {
            "project": run_config['project'],
            "name": run_config['name'],
            "reinit": run_config['reinit'],
            "notes" : run_config['notes'],
            "tags": run_config['tags'],
            "dir" : run_config['dir'],
            "config": general_config
        }

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    feature_extractor = auto_extractor(model_name)

    dataset = AudioDataset(data_path, feature_extractor)

    train_subset, test_subset, inference_subset = train_test_split_custom(dataset, # type: ignore
                                                                          test_size=test_size, 
                                                                          inference_size=inference_size)  
    num_classes = len(dataset.get_classes()) 

    model = custom_AST(model_name, num_classes, device)

    summary(model,
            col_names=["num_params","trainable"],
            col_width=20,
            row_settings=["var_names"])
    print(model)
    
    train_dataloader_custom = DataLoader(dataset=train_subset, 
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_CUDA_WORKERS,
                                        pin_memory=PINNED_MEMORY,
                                        shuffle=SHUFFLED)

    test_dataloader_custom = DataLoader(dataset=test_subset,
                                        batch_size=BATCH_SIZE, 
                                        num_workers=NUM_CUDA_WORKERS,
                                        pin_memory=PINNED_MEMORY,
                                        shuffle=SHUFFLED)

    if inference_subset: # may not be defined
        inference_dataloader_custom = DataLoader(dataset=inference_subset,
                                        batch_size=BATCH_SIZE, 
                                        num_workers=NUM_CUDA_WORKERS,
                                        pin_memory=PINNED_MEMORY,
                                        shuffle=SHUFFLED) 
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2) #TODO experiment w/ diff hyperparams
    wandb.init(
            project=wandb_params.get("project"),
            config=wandb_params.get("config"),
            name=wandb_params.get("name"),
            reinit=wandb_params.get("reinit", True),
            tags=wandb_params.get("tags", []),
            notes=wandb_params.get("notes", ""),
            dir=wandb_params.get("dir", None)
        )
        
    results = train(model,
                    train_dataloader=train_dataloader_custom,
                    test_dataloader=test_dataloader_custom,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    epochs=EPOCHS,
                    device=device,
                    num_classes= num_classes,
                    accumulation_steps=ACCUMULATION_STEPS,
                    patience=TRAIN_PATIENCE)

    inference_loop(model=model,
                device=device,
                loss_fn=loss_fn,
                inference_loader= inference_dataloader_custom)




    wandb.finish()

    if SAVE_MODEL:
        save_model(model=model,
                target_dir="saved_models",
                model_name="AST_classifier_true.pt")
    
if __name__ == "__main__":
    main()
