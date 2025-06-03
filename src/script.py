from configs import load_configs
import yaml
from helper.datamodule_factory import create_datamodule
from icecream import ic
import torch
import numpy as np
import sys
from models.model_factory import ModelFactory
from helper.ptl_trainer import PTLTrainer
from pathlib import Path
from torchinfo import summary




def main():

    print("_"*40+"\n")
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    (general_config, 
     feature_extraction_config,
     dataset_config,  
     peft_config, 
     wandb_config,
     sweep_config, 
     augmentation_config,
     optimizer_config) = load_configs(config)
    
    print("_"*40+"\n")

    # Set random seeds for reproducibility
    torch.manual_seed(general_config.seed)
    torch.cuda.manual_seed(general_config.seed)
    np.random.seed(general_config.seed)
    
    # Create data module using the factory
    data_module = create_datamodule(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        dataset_config=dataset_config,
        augmentation_config=augmentation_config
    )
    ic("Created the audio data module")
    
    # Setup data module (this will also save dataloaders if save_dataloader is True)
    data_module.setup()
    ic("Setup the data module")
    
    # Get dataloaders and print information
    try:
        if general_config.use_kfold:
            for fold in range(general_config.k_folds):

                fold_train_loader, fold_val_loader = data_module.get_fold_dataloaders(fold)

                # Use fold_train_loader and fold_val_loader for training
                train_samples = getattr(fold_train_loader.dataset, "__len__", lambda: "unknown")()
                val_samples = getattr(fold_val_loader.dataset, "__len__", lambda: "unknown")()

                ic(f"Fold {fold} train loader: {train_samples} samples")
                ic(f"Fold {fold} val loader: {val_samples} samples")

        elif "static" in dataset_config.data_path:

            ic("Loading from static dataloaders")
            data_module.load_dataloaders(dataset_config.data_path)

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
            ic(f"number of augmentations: {train_loader.dataset.augmentations_per_sample}") # type: ignore
        
        # Get class information
        classes, class_to_idx, idx_to_class = data_module.get_class_info()
        print(f"Classes: {classes}")
        print(f"Number of classes: {len(classes)}")
        
    except Exception as e:
        ic(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic(f"Using device: {device}")

    # Create model factory function
    model_factory = ModelFactory.get_model_factory(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        dataset_config=dataset_config,
        peft_config=peft_config
    )

    ic("model_factory created")

    # Unpack the tuple correctly
    model, feature_extractor = model_factory(device)
    
    # Now we can call parameters() on the model
    #print(model.parameters())
    #sys.exit(1)
    
    summary(model,
            col_names=["num_params","trainable"],
            col_width=20,
            row_settings=["var_names"])

    # print(model)
    # sys.exit(1)
    # Create PyTorch Lightning trainer
    trainer = PTLTrainer(
        general_config=general_config,
        feature_extraction_config=feature_extraction_config,
        dataset_config=dataset_config,
        peft_config=peft_config,
        wandb_config=wandb_config,
        sweep_config=sweep_config,
        data_module=data_module,
        model_factory=model_factory,
        augmentation_config=augmentation_config,
        optimizer_config=optimizer_config
    )
    
    ic("trainer created")
    
    # Train model
    try:
        if general_config.use_kfold:
            ic("Starting k-fold cross-validation training")
            results = trainer.k_fold_cross_validation()
            
            # Print k-fold results
            print("\nK-fold Cross-Validation Results:")
            print("-" * 40)
            
            for i, fold_result in enumerate(results["fold_results"]):
                print(f"Fold {i+1}:")
                print(f"  Val Loss: {fold_result['val_loss']:.4f}")
                print(f"  Val Accuracy: {fold_result['val_acc']:.4f}")
                print(f"  Val F1: {fold_result['val_f1']:.4f}")
                print(f"  Val Precision: {fold_result['val_precision']:.4f}")
                print(f"  Val Recall: {fold_result['val_recall']:.4f}")
            
            print("\nAverage Metrics:")
            print(f"  Val Loss: {results['avg_metrics']['average_val_loss']:.4f} ± {results['avg_metrics']['std_val_loss']:.4f}")
            print(f"  Val Accuracy: {results['avg_metrics']['average_val_acc']:.4f} ± {results['avg_metrics']['std_val_acc']:.4f}")
            print(f"  Val F1: {results['avg_metrics']['average_val_f1']:.4f} ± {results['avg_metrics']['std_val_f1']:.4f}")
            
            # Print average inference metrics if available
            if any(key.startswith('average_inference_') for key in results['avg_metrics'].keys()):
                print("\nAverage Inference Metrics:")
                print(f"  Inference Accuracy: {results['avg_metrics'].get('average_inference_acc', 'N/A'):.4f} ± {results['avg_metrics'].get('std_inference_acc', 0.0):.4f}")
                print(f"  Inference F1: {results['avg_metrics'].get('average_inference_f1', 'N/A'):.4f} ± {results['avg_metrics'].get('std_inference_f1', 0.0):.4f}")
                print(f"  Inference Precision: {results['avg_metrics'].get('average_inference_precision', 'N/A'):.4f} ± {results['avg_metrics'].get('std_inference_precision', 0.0):.4f}")
                print(f"  Inference Recall: {results['avg_metrics'].get('average_inference_recall', 'N/A'):.4f} ± {results['avg_metrics'].get('std_inference_recall', 0.0):.4f}")
            
        else:
            ic("Starting regular training")
            test_results = trainer.train()
            
            # Print test results
            print("\nTest Results:")
            print("-" * 40)
            print(f"Test Loss: {test_results.get('test_loss', 'N/A')}")
            print(f"Test Accuracy: {test_results.get('test_acc', 'N/A')}")
            print(f"Test F1: {test_results.get('test_f1', 'N/A')}")
            print(f"Test Precision: {test_results.get('test_precision', 'N/A')}")
            print(f"Test Recall: {test_results.get('test_recall', 'N/A')}")
            
            # Print inference results if available
            if any(key.startswith('inference_') for key in test_results.keys()):
                print("\nInference Results:")
                print("-" * 40)
                print(f"Inference Accuracy: {test_results.get('inference_acc', 'N/A')}")
                print(f"Inference F1: {test_results.get('inference_f1', 'N/A')}")
                print(f"Inference Precision: {test_results.get('inference_precision', 'N/A')}")
                print(f"Inference Recall: {test_results.get('inference_recall', 'N/A')}")
            
    except Exception as e:
        ic(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()