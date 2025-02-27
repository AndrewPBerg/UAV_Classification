from configs.configs_demo import load_configs
import yaml
from srcnew.datamodule import AudioDataModule
from icecream import ic





def main():
    
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    (general_config, 
     feature_extraction_config, 
     cnn_config, 
     peft_config, 
     wandb_config,
     sweep_config, 
     augmentation_config) = load_configs(config)
    
    print("_"*40+"\n")

    
    # Create data module directly from configs
    data_module = AudioDataModule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        augmentation_config=augmentation_config
    )
    ic("Created the audio data module")
    
    # Setup data module (this will also save dataloaders if save_dataloader is True)
    data_module.setup()
    ic("Setup the data module")
    
    # Get dataloaders
    try:
        if general_config.use_kfold:
            for fold in range(general_config.k_folds):
                fold_train_loader, fold_val_loader = data_module.get_fold_dataloaders(fold)
                # Use fold_train_loader and fold_val_loader for training
                train_samples = getattr(fold_train_loader.dataset, "__len__", lambda: "unknown")()
                val_samples = getattr(fold_val_loader.dataset, "__len__", lambda: "unknown")()
                ic(f"Fold {fold} train loader: {train_samples} samples")
                ic(f"Fold {fold} val loader: {val_samples} samples")
        elif "static" in general_config.data_path:
            ic("Loading from static dataloaders")
            data_module.load_dataloaders(general_config.data_path)
        else:
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            test_loader = data_module.test_dataloader()
            inference_loader = data_module.predict_dataloader()

            # Fixed: Check if dataset has __len__ before calling len()
            train_samples = getattr(train_loader.dataset, "__len__", lambda: "unknown")()
            val_samples = getattr(val_loader.dataset, "__len__", lambda: "unknown")()
            test_samples = getattr(test_loader.dataset, "__len__", lambda: "unknown")()
            inference_samples = getattr(inference_loader.dataset, "__len__", lambda: "unknown")()
            ic(f"Train loader: {train_samples} samples")
            ic(f"Val loader: {val_samples} samples")
            ic(f"Test loader: {test_samples} samples")
            ic(f"Inference loader: {inference_samples} samples")
            ic(f"number of augmentations: {train_loader.dataset.augmentations_per_sample}")
        
        # Get class information
        classes, class_to_idx, idx_to_class = data_module.get_class_info()
        print(f"Classes: {classes}")
        print(f"Number of classes: {len(classes)}")
        
    except Exception as e:
        ic(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

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
        # DATA_PATH : str
        # if DATA_PATH has static in the name then load the dataset from the path
        # if save flag is passed then save the dataset to the path 
        # Original train-test split code
        # if loading from static dataset:
        """
        train_dataloader = torch.load('path1')
        test_dataloader = torch.load('path2')
        val_dataloader = torch.load('path3')
        inference_dataloader = torch.load('path4')
        
        # Save the dataset to a .pth file
        torch.save(dataset, 'dataset.pth')
        
        # Later, you can load the dataset back
        loaded_dataset = torch.load('dataset.pth')
        
        """
        if "static" in DATA_PATH:
            ic(DATA_PATH)
            train_dataloader = torch.load(DATA_PATH+'/train_dataloader.pth', weights_only=False)
            test_dataloader = torch.load(DATA_PATH+'/test_dataloader.pth', weights_only=False)
            val_dataloader = torch.load(DATA_PATH+'/val_dataloader.pth', weights_only=False)
            inference_dataloader = torch.load(DATA_PATH+'/inference_dataloader.pth', weights_only=False)

            ic(train_dataloader.dataset.classes)
            ic(train_dataloader.dataset.augmentations_per_sample)
            # ic(test_dataloader)
            # ic(val_dataloader)
            # ic(inference_dataloader)

            # sys.exit()
        else:
            if AUGMENTATIONS_PER_SAMPLE > 0:
                ic("Augmenting the dataset, this might take a while...")
                
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
        if SAVE_DATALOADER:
            ic("saving the dataloaders, this might take a while...")
            fixed_pathing = '/app/src/datasets/static/'
            # TODO naming convention with Augmentations list is broken
            distinct_name = f"{NUM_CLASSES}"+f"-augs-{AUGMENTATIONS_PER_SAMPLE}"

            # add augmenatations to end of the path string
            all_together_now = f"{fixed_pathing}"+f"{distinct_name}"
            for s in AUGMENTATIONS:
                all_together_now += f"-{s.replace(' ', '-')}" #remove-white-space-from-the-string
            ic(all_together_now)

            os.makedirs(all_together_now, exist_ok=True)


            torch.save(train_dataloader, all_together_now+'/train_dataloader.pth')
            torch.save(val_dataloader, all_together_now+'/val_dataloader.pth')
            torch.save(test_dataloader, all_together_now+'/test_dataloader.pth')
            torch.save(inference_dataloader, all_together_now+'/inference_dataloader.pth')
            ic("Saved the dataloaders to the above path!")
            # TODO find a way to skip training and jump to end of main
            # sys.exit()
            return
        
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
