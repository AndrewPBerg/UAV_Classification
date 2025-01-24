# DESCRIPTION
import types
from helper.util import train_test_split_custom, save_model, wandb_login, calculated_load_time, generate_model_image, k_fold_split_custom
from helper.engine import train, inference_loop
from helper.fold_engine import k_fold_cross_validation
from helper.ast import custom_AST
from helper.models import TorchCNN
from helper.cnn_feature_extractor import MelSpectrogramFeatureExtractor, MFCCFeatureExtractor
from transformers import ASTFeatureExtractor
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from torchvision.models import efficientnet, EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_L_Weights
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights
from torchvision.models import Inception_V3_Weights
import torchvision.models as models
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim import Adam
import torch.nn as nn
from torchinfo import summary
import yaml
from timeit import default_timer as timer 
import wandb
import os
from icecream import ic
from torch.cuda.amp import GradScaler, autocast
import sys
import numpy as np
import torch.hub
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import (
    get_peft_model,
    LoraConfig,
    IA3Config,
    AdaLoraConfig,
    XLoraConfig,
    OFTConfig,
    FourierFTConfig,
    LNTuningConfig
)


torch.hub.set_dir('UAV_Classification/model_cache/torch/hub')  # Set custom cache directory


def get_feature_config(config):
    """
    Get feature extraction configuration from config.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (input_shape, feature_extractor)
    """
    fe_config = config['cnn_config']['feature_extraction']
    feature_type = fe_config.get('type', 'melspectrogram')

    if feature_type == 'melspectrogram':
        input_shape = (fe_config['n_mels'], 157)
        feature_extractor = MelSpectrogramFeatureExtractor(
            sampling_rate=fe_config['sampling_rate'],
            n_mels=fe_config['n_mels'],
            n_fft=fe_config['n_fft'],
            hop_length=fe_config['hop_length'],
            power=fe_config['power']
        )
    elif feature_type == 'mfcc':
        input_shape = (fe_config.get('n_mfcc', 40), 157)
        feature_extractor = MFCCFeatureExtractor(
            sampling_rate=fe_config['sampling_rate'],
            n_mfcc=fe_config.get('n_mfcc', 40),
            n_mels=fe_config.get('n_mels', 128),
            n_fft=fe_config.get('n_fft', 1024),
            hop_length=fe_config.get('hop_length', 512)
        )
    else:
        raise ValueError(f"Unknown feature extraction type: {feature_type}")
        
    return input_shape, feature_extractor

def parse_model_size(model_string, target):
    """Parse ResNet size from model string"""
    model_string = model_string.lower()
    if target not in model_string:
        return None
    size = ''.join(filter(str.isdigit, model_string))
    return size

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
    
    if model_type.lower() == "ast":
        model, feature_extractor, adaptor_config = custom_AST(num_classes, config['adaptor_type'])
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        train_fn = train
    if model_type.lower().startswith("vit"):
        # from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights
        input_shape, feature_extractor = get_feature_config(config)
        size = parse_model_size(model_type, target="vit")
        if size == "116": # short for Base-16
            model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif size == "132": # short for base-32
            model = models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        elif size == "216": # short for Large-16
            model = models.vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
        elif size == "232": # short for Large-32
            model = models.vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
        elif size == "3": # short for Huge-14
            model = models.vit_h_14(weights=ViT_H_14_Weights.DEFAULT)

        else:
            raise ValueError(f"Please add the Vit net weight to: {model_type}")



        # Add resize layer to match ViT's expected input size
            # Modify input processing
        model.image_size = 224  # Set fixed size expected by ViT
        resize_layer = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # Configure model dimensions
        model.image_size = 224
        hidden_dim = model.hidden_dim  # Get model's hidden dimension
        resize_layer = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        def new_forward(self, x):
            # Handle input
            x = x.float()
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            # Process through layers
            x = resize_layer(x)
            x = self.conv_proj(x)  # Now outputs correct hidden_dim
            
            # Shape for transformer
            x = x.flatten(2).transpose(1, 2)
            
            # Add class token
            n = x.shape[0]
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            
            # Forward through transformer and classification
            x = self.encoder(x)
            x = x[:, 0]
            x = self.heads(x)
            return x
        
        # Update model components
        model.forward = types.MethodType(new_forward, model)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        model.conv_proj = nn.Conv2d(1, hidden_dim, kernel_size=16, stride=16)
        
        # Freeze all except head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.parameters():
            param.requires_grad = True

        # print(model)
        # sys.exit()




        optimizer = AdamW(model.parameters(), lr=learning_rate)
        train_fn = train
    elif model_type.lower() == "cnn":
        
        input_shape, feature_extractor = get_feature_config(config)
            
        model = TorchCNN(
            num_classes=num_classes,
            hidden_units=config['cnn_config']['hidden_units'],
            input_shape=input_shape
        )
        optimizer = Adam(model.parameters(), lr=learning_rate)
        train_fn = train

    elif model_type.lower() == "test":
        input_shape, feature_extractor = get_feature_config(config)
        ic(input_shape)
        ic(feature_extractor)

        # Initialize the base model and modify conv1 BEFORE applying LoRA
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
        # Modify first conv layer for 1-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type="AUDIO_CLASSIFICATION",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"],
        )

        peft_config = IA3Config( 
                target_modules=["conv","fc"],
                feedforward_modules=["conv","fc"],
                task_type = "AUDIO_CLASSIFICATION"
                )
        peft_config = IA3Config( 
                target_modules=["fc"],
                feedforward_modules=["fc"],
                task_type = "AUDIO_CLASSIFICATION"
                )
        
        # Convert to PEFT model
        model = get_peft_model(model, peft_config)
        
        # # Freeze all parameters except LoRA parameters
        # model.requires_grad_(False)
        # for name, param in model.named_parameters():
        #     if "lora" in name:
        #         param.requires_grad = True
        
        # Modify the final layer for your number of classes
        model.base_model.model.fc = nn.Linear(model.base_model.model.fc.in_features, num_classes)
        # for param in model.base_model.model.fc.parameters():
        #     param.requires_grad = True
        
        # Initialize optimizer with only trainable parameters
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        
        train_fn = train


    elif model_type.lower().startswith("resnet"):
        input_shape, feature_extractor = get_feature_config(config)
        
        # Parse ResNet size
        size = parse_model_size(model_type, target="resnet")
        
        # Select appropriate ResNet model
        if size == "18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif size == "34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif size == "50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif size == "101":
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif size == "152":
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(f"Please add the resnet weight to: {model_type}")
        
        
        # Modify first conv layer for 1-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final layer for your number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Optionally freeze pretrained layers
        for param in model.parameters():
            # param.requires_grad = False
            param.requires_grad = True
        # Unfreeze final layers
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
            
        # optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        train_fn = train

    elif model_type.lower().startswith("densenet"):
        input_shape, feature_extractor = get_feature_config(config)

        
        size = parse_model_size(model_type, target="densenet")

        if size == "121":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        elif size == "161":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)
        elif size == "169":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
        elif size == "201":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        else:
            raise ValueError(f"Please add the densenet weight to: {model_type}")
        

        # # Modify first conv layer for 1-channel input
        # model.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # # Modify the final layer for your number of classes
            


        # Modify first conv layer for 1-channel input
        first_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # If using pretrained weights, adapt them for 1-channel input
        if True:  # assuming we always want to use pretrained weights
            # Average the weights across the 3 channels
            with torch.no_grad():
                new_weights = model.features.conv0.weight.data.mean(dim=1, keepdim=True)
                first_conv.weight.data = new_weights
        
        # Replace the first conv layer
        model.features.conv0 = first_conv
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
        for param in model.classifier.parameters():
            param.requires_grad = True
        # optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        train_fn = train
        print(model)
        # sys.exit()
    elif model_type.lower().startswith("efficientnet"):
        input_shape, feature_extractor = get_feature_config(config)

        size = parse_model_size(model_type, target="efficientnet")

        if size == "1": # this cooresponds to s
            model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        elif size == "2": #  this cooresponds to m
            model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        elif size == "3": #  this cooresponds to l
            model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        else:
            raise ValueError(f"Please add the efficientnet weight to: {model_type}")

        
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

        # Assuming input_channels = 1 for audio spectrograms
        first_conv = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        # Adapt pretrained weights for new input channels
        with torch.no_grad():
            # Average the weights across original 3 channels
            new_weights = model.features[0][0].weight.data.mean(dim=1, keepdim=True)
            first_conv.weight.data = new_weights
        
        # Replace first conv layer
        model.features[0][0] = first_conv
        
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # print(model)
        # sys.exit()
        # optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        train_fn = train

    elif model_type.lower().startswith("mobilenet"):
        input_shape, feature_extractor = get_feature_config(config)

        size = parse_model_size(model_type, target="mobilenet")

        if size == "1": # this cooresponds to small
            model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        elif size == "2": #  this cooresponds to large
            model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        else:
            raise ValueError(f"Please add the mobilenet weight to: {model_type}")

        
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        # # Assuming input_channels = 1 for audio spectrograms
        first_conv = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        # # Adapt pretrained weights for new input channels
        with torch.no_grad():
            # Average the weights across original 3 channels
            new_weights = model.features[0][0].weight.data.mean(dim=1, keepdim=True)
            first_conv.weight.data = new_weights
        
        # # Replace first conv layer
        model.features[0][0] = first_conv
        
        for param in model.classifier.parameters():
            param.requires_grad = True
        

        # print(model)
        # sys.exit()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        # optimizer = Adam(model.parameters(), lr=learning_rate)
        train_fn = train
    elif model_type.lower().startswith("inception"):
        input_shape, feature_extractor = get_feature_config(config)

        # size = parse_model_size(model_type, target="mobilenet")

        # if size == "1": # this cooresponds to small
        # elif size == "2": #  this cooresponds to large
        #     model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        # else:
        #     raise ValueError(f"Please add the mobilenet weight to: {model_type}")

        model = models.inception.inception_v3(weights=Inception_V3_Weights.DEFAULT)

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # # Assuming input_channels = 1 for audio spectrograms
       # Create new first conv layer with 1 input channel
        # first_conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # # Adapt pretrained weights
        # with torch.no_grad():
        #     # Average RGB channels -> grayscale
        #     new_weights = model.Conv2d_1a_3x3.conv.weight.data.mean(dim=1, keepdim=True)
        #     first_conv.weight.data = new_weights

        # # Replace first conv layer
        # model.Conv2d_1a_3x3.conv = first_conv
        
        # Modify first conv layer to accept 1 channel
        # model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)

        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)

        with torch.no_grad():
            # Average the weights across the 3 input channels
            new_weights = model.Conv2d_1a_3x3.conv.weight.data.mean(dim=1, keepdim=True)
            model.Conv2d_1a_3x3.conv.weight.data = new_weights

        # Modify auxiliary classifier for 1 color channel input
        model.AuxLogits.conv0 = nn.Conv2d(768, 128, kernel_size=1, stride=1)
        model.AuxLogits.conv1 = nn.Conv2d(128, 768, kernel_size=3, stride=1, padding=1)
            
        def _transform_input_grayscale(self, x):
            # Normalize grayscale input
            x = x * 0.5 + 0.5  # Scale to [0, 1]
            x = x * 0.224 + 0.456  # Apply normalization
            return x
        
        # Replace the original _transform_input method
        model._transform_input = _transform_input_grayscale.__get__(model)
        
        for param in model.fc.parameters():
            param.requires_grad = True
        # print(input_shape)        
        # print(model)
        # sys.exit()

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        # optimizer = Adam(model.parameters(), lr=learning_rate)
        train_fn = train

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
    NUM_CLASSES = general_config['num_classes']
    SAVE_DATALOADER = general_config['save_dataloader']
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
            sys.exit()
        
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
