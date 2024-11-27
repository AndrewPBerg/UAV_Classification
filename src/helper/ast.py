from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Type, Any
from .MoA import AST_MoA, AST_SoftMoA
import os
import torch
from torch import nn
from transformers import (
    ASTFeatureExtractor, ASTForAudioClassification, AutoModel
)
from icecream import ic
import yaml
import logging
import wandb
import sys
from .util import get_mixed_params

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
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")
pretrained_AST_model="MIT/ast-finetuned-audioset-10-10-0.4593"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('transformers').setLevel(logging.ERROR)

# downloads and store cached version in mounted Docker Volume
# Makes runs fast and the containers very tiny

def create_model():
    try:
        return ASTForAudioClassification.from_pretrained(pretrained_AST_model, attn_implementation="sdpa", torch_dtype=torch.float16, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError: # if model is not cached, download it
        return download_model(pretrained_AST_model, CACHE_DIR)
    return None

def create_processor():
    try:
        return ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError:
        return ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
    return None

def download_model(model_name, cache_dir):
    logger.info(f"Manually downloading {model_name} to {cache_dir}")
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

def custom_AST(num_classes: int, adaptor_type: str, sweep_config: dict=None):
    
    # if sweep_config is None:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # else: 
    if sweep_config:
        # config = sweep_config
        params = get_mixed_params(sweep_config, config[adaptor_type])
    else:
        params = config[adaptor_type]
    
    if adaptor_type == "moa":

        model = AST_MoA(
            max_length=params['max_length'],
            num_classes=num_classes, 
            final_output=params['final_output'],
            reduction_rate=params['reduction_rate'],
            adapter_type=params['adapter_type'],
            location=params['location'],
            adapter_module=params['adapter_module'],
            num_adapters=params['num_adapters'],
            model_ckpt=pretrained_AST_model)
        processor = create_processor()

        return model, processor, params
    
    elif adaptor_type == "soft-moa":
        # Initialize model
        model = AST_SoftMoA(
            max_length=params['max_length'],
            num_classes=num_classes,
            final_output=params['final_output'],
            reduction_rate=params['reduction_rate'], 
            adapter_type=params['adapter_type'],
            location=params['location'],
            adapter_module=params['adapter_module'],
            num_adapters=params['num_adapters'],
            num_slots=params['num_slots'],
            normalize=params['normalize'],
            model_ckpt=pretrained_AST_model)
        
        processor = create_processor()

        return model, processor, params

    elif adaptor_type == "none-classifier":
        model = create_model()
        processor = create_processor()
        model = create_model()
        processor = create_processor()
        model.config.num_labels = num_classes
        adaptor_config = {}
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        in_features = model.classifier.dense.in_features
        model.classifier.dense = nn.Linear(in_features, num_classes)
        
        return model, processor, {}

    elif adaptor_type == "none-full":
        model = create_model()
        processor = create_processor()
        model.config.num_labels = num_classes
        
        for param in model.parameters():
            param.requires_grad = True
        
        in_features = model.classifier.dense.in_features
        model.classifier.dense = nn.Linear(in_features, num_classes)
        return model, processor, {}
    
    elif adaptor_type == "inference":
        model = create_model()
        processor = create_processor()
        model.config.num_labels = num_classes
        
        for param in model.parameters():
            param.requires_grad = False
        
        in_features = model.classifier.dense.in_features
        model.classifier.dense = nn.Linear(in_features, num_classes)
        return model, processor, {}

    else:
        # Handle other adaptor types
        model = create_model()
        processor = create_processor()
        
        in_features = model.classifier.dense.in_features
        model.classifier.dense = nn.Linear(in_features, num_classes)
        
        model, adaptor_config = get_adaptor_model(model, adaptor_type, params)
        
        return model, processor, adaptor_config

def get_adaptor_config(adaptor_type: str, params: dict):
    # if sweep_config is None:
    #     with open('config.yaml', 'r') as file:
    #         config = yaml.safe_load(file)
    #         config = config[adaptor_type]
    # else:
    #     config = sweep_config
    config = params
    match adaptor_type:
        case "lora":
            # config = config["lora"]
            return LoraConfig(
                        r=config["r"],
                        lora_alpha=config["lora_alpha"],
                        target_modules=config["target_modules"],
                        lora_dropout=config["lora_dropout"],
                        bias=config["bias"],
                        task_type=config["task_type"],
                        use_rslora= config["use_rslora"],
                        use_dora = config["use_dora"],
                    )
        
        case "ia3":
            # config = config["ia3"]
            return IA3Config( 
                target_modules=config["target_modules"],
                feedforward_modules=config["feedforward_modules"],
                task_type = config['task_type']
                )
        
        case "adalora":
            # config = config["adalora"]
            return AdaLoraConfig(
                init_r=config["init_r"],
                target_r=config["target_r"],
                target_modules=config["target_modules"],
                lora_alpha=config["lora_alpha"],
                task_type=config["task_type"]
            )
            
        case "oft":
            # config = config["oft"]
            return OFTConfig(
                r=config['r'],
                target_modules=config['target_modules'],
                module_dropout=config['module_dropout'],
                init_weights=config['init_weights'],
            )
            
        case "fourier":
            # config = config["fourier"]
            return FourierFTConfig(
                target_modules=config["target_modules"],
                task_type=config["task_type"],
                n_frequency=config["n_frequency"],
                scaling=config["scaling"],
            )
            
        case "layernorm":
            # config = config["layernorm"]
            return LNTuningConfig(
                target_modules=config["target_modules"],
                task_type=config["task_type"]
            )
        case _:
            raise ValueError(f"Unknown adaptor type: {adaptor_type}")
    
        
def get_adaptor_model(model, adaptor_type: str, params:dict):

    adaptor_config = get_adaptor_config(adaptor_type, params)
    print("-----------------------------------------")
    # ic(adaptor_config)
    # match adaptor_type:
        # case "lora":
    ic(get_peft_model(model, adaptor_config))
    return get_peft_model(model, adaptor_config), adaptor_config
            # return (model, adaptor_config
        # case "ia3":
        #     return get_peft_model(model, adaptor_config)
    
    print("no case matched!!! check config.yaml")
    print("no case matched!!! check config.yaml")
    print("no case matched!!! check config.yaml")