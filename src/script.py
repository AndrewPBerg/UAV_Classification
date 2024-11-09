# DESCRIPTION
from helper.util import train_test_split_custom, save_model, wandb_login, calculated_load_time
from helper.engine import train, inference_loop
from helper.ast import custom_AST

import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
import torch.nn as nn
from torchinfo import summary
import yaml
from timeit import default_timer as timer 
import wandb
from icecream import ic

def main():

    # start logging data load time
    start = timer()
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    general_config = config['general']
    run_config = config['wandb']

    DATA_PATH = general_config['data_path']
    BATCH_SIZE = general_config['batch_size']
    SEED = general_config['seed']
    EPOCHS = general_config['epochs']
    NUM_CUDA_WORKERS = general_config['num_cuda_workers']
    PINNED_MEMORY = general_config['pinned_memory']
    SHUFFLED = general_config['shuffled']
    ACCUMULATION_STEPS = general_config['accumulation_steps']
    LEARNING_RATE = general_config['learning_rate']
    TRAIN_PATIENCE = general_config['patience']
    SAVE_MODEL = general_config['save_model']
    TEST_SIZE = general_config['test_size']
    INFERENCE_SIZE = general_config['inference_size']
    VAL_SIZE = general_config['val_size']
    AUGMENTATIONS_PER_SAMPLE = general_config['augmentations_per_sample']
    AUGMENTATIONS= general_config['augmentations']
    MODEL_NAME = general_config["model_name"]
    USE_WANDB = general_config['use_wandb']
    NUM_CLASSES = general_config['num_classes']

    ADAPTOR_TYPE = general_config['adaptor_type']

    if USE_WANDB:
        wandb_login()
        
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

    model, feature_extractor = custom_AST(NUM_CLASSES, ADAPTOR_TYPE)
    
    model.to(device)

    # dataset = AudioDataset(data_path, feature_extractor)
    train_dataset, val_dataset, test_dataset, inference_dataset = train_test_split_custom(
        DATA_PATH, 
        feature_extractor, 
        test_size=TEST_SIZE, 
        seed=SEED,
        inference_size=INFERENCE_SIZE,
        augmentations_per_sample=AUGMENTATIONS_PER_SAMPLE,
        val_size=VAL_SIZE,
        augmentations=AUGMENTATIONS,
        config=general_config
    )

    summary(model,
            col_names=["num_params","trainable"],
            col_width=20,
            row_settings=["var_names"])
    print(model)
    
    train_dataloader_custom = DataLoader(dataset=train_dataset, #transformed_train_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_CUDA_WORKERS,
                                        pin_memory=PINNED_MEMORY,
                                        shuffle=SHUFFLED)

    test_dataloader_custom = DataLoader(dataset=test_dataset,
                                        batch_size=BATCH_SIZE, 
                                        num_workers=NUM_CUDA_WORKERS,
                                        pin_memory=PINNED_MEMORY,
                                        shuffle=SHUFFLED)
    val_dataloader_custom = DataLoader(dataset=val_dataset,
                                        batch_size=BATCH_SIZE, 
                                        num_workers=NUM_CUDA_WORKERS,
                                        pin_memory=PINNED_MEMORY,
                                        shuffle=SHUFFLED)

    inference_dataloader_custom = DataLoader(dataset=inference_dataset,
                                    batch_size=BATCH_SIZE, 
                                    num_workers=NUM_CUDA_WORKERS,
                                    pin_memory=PINNED_MEMORY,
                                    shuffle=SHUFFLED) 
    end = timer()
    total_load_time = calculated_load_time(start, end)
    print(f"Load time in on path: {DATA_PATH} --> {total_load_time}")
    if wandb.run is not None:
        wandb.log({"load_time": total_load_time})

    loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2) #TODO experiment w/ diff hyperparams

    if USE_WANDB:
        wandb.init(
            project=wandb_params.get("project"),
            config=wandb_params.get("config"),
            name=wandb_params.get("name"),
            reinit=wandb_params.get("reinit", True),
            tags=wandb_params.get("tags", []),
            notes=wandb_params.get("notes", ""),
            dir=wandb_params.get("dir", None)
        )
    
    # ic(feature_extractor)
    # ic(feature_extractor)
    # ic(train_dataloader_custom.dataset[0][0].shape)
    # ic(train_dataloader_custom.dataset[0])
        
    # train(
    #     model=model,
    #     train_dataloader=train_dataloader_custom,
    #     test_dataloader=test_dataloader_custom,
    #     val_dataloader=val_dataloader_custom,
    #     optimizer=optimizer,
    #     scheduler=scheduler,  # type: ignore
    #     loss_fn=loss_fn,
    #     epochs=EPOCHS,
    #     device=device,
    #     num_classes=NUM_CLASSES,
    #     accumulation_steps=ACCUMULATION_STEPS,
    #     patience=TRAIN_PATIENCE
    # )

    # inference_loop(model=model,
    #             device=device,
    #             loss_fn=loss_fn,
    #             inference_loader= inference_dataloader_custom)




    if USE_WANDB:
        wandb.finish()

    if SAVE_MODEL:
        save_model(model=model,
                target_dir="saved_models",
                model_name="AST_classifier_true.pt")
    
if __name__ == "__main__":
    main()
