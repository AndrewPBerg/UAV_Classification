from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Type, Any
from .MoA import AST_MoA
import os
import torch
from torch import nn
from transformers import (
    ASTFeatureExtractor, ASTForAudioClassification, AutoModel
)
from icecream import ic
import yaml
import logging
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
def download_model(model_name, cache_dir):
    logger.info(f"Manually downloading {model_name} to {cache_dir}")
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

def custom_AST(num_classes: int, adaptor_type: str) -> Tuple[ASTForAudioClassification, ASTFeatureExtractor]:
    if adaptor_type == "moa":
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        params = config["moa"]
        # Initialize model
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
        try:
            processor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
        except OSError:
            processor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
            
        return model, processor
    
    try:
        model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
        processor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError: # if model is not cached, download it
        model = download_model(pretrained_AST_model, CACHE_DIR)
        processor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
    
    in_features = model.classifier.dense.in_features
    model.classifier.dense = nn.Linear(in_features, num_classes)
    model = get_adaptor_model(model, adaptor_type)

    # ic(model)
    # if model is not None:
        # model.config.num_labels = num_classes
        
        # # # Add label mappings
        # # model.config.id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
        # # model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        
        # # for param in model.parameters():
        # #     param.requires_grad = False
        # # for param in model.classifier.parameters():
        # #     param.requires_grad = True

        # # Freeze all parameters first
        # for param in model.parameters():
        #     param.requires_grad = False
        
        # Model-specific customizations
        # try:
            # model.classifier.dense = nn.Linear(in_features, num_classes)
            # adaptor_config = get_peft_config(adaptor_type)

                
                # original_forward = model.forward
                # def new_forward(self, input_values=None, **kwargs):
                #     if isinstance(input_values, torch.Tensor):
                #         return self.base_model(input_values=input_values)
                    
                #     if input_values is None and kwargs:
                #         for k, v in kwargs.items():
                #             if isinstance(v, torch.Tensor):
                #                 return self.base_model(input_values=v)
                    
                #     return self.base_model(input_values=input_values)
                
                # model.forward = new_forward.__get__(model)
                
        # except Exception as e:
        #     # logger.error(f"Error in custom_AST: {e}")
        #     raise e
    
    return model, processor

def get_adaptor_config(adaptor_type: str):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    match adaptor_type:
        case "lora":
            config = config["lora"]
            return LoraConfig(
                        r=config["r"],
                        lora_alpha=config["lora_alpha"],
                        target_modules=config["target_modules"],
                        lora_dropout=config["lora_dropout"],
                        bias=config["bias"],
                        task_type=config["task_type"],
                    )
        
        case "ia3":
            config = config["ia3"]
            return IA3Config( 
                target_modules=config["target_modules"],
                feedforward_modules=config["feedforward_modules"],
                task_type = config['task_type']
                )
        
        case "adalora":
            config = config["adalora"]
            return AdaLoraConfig(
                init_r=config["init_r"],
                target_r=config["target_r"],
                target_modules=config["target_modules"],
                lora_alpha=config["lora_alpha"],
                task_type=config["task_type"]
            )
            
        case "oft":
            config = config["oft"]
            return OFTConfig(
                r=config['r'],
                target_modules=config['target_modules'],
                module_dropout=config['module_dropout'],
                init_weights=config['init_weights'],
            )
            
        case "fourier":
            config = config["fourier"]
            return FourierFTConfig(
                target_modules=config["target_modules"],
                task_type=config["task_type"]
            )
            
        case "layernorm":
            config = config["layernorm"]
            return LNTuningConfig(
                target_modules=config["target_modules"],
                task_type=config["task_type"]
            )
        
        case "none":
            pass
        case _:
            raise ValueError(f"Unknown adaptor type: {adaptor_type}")
    
        
def get_adaptor_model(model,adaptor_type: str):

    adaptor_config = get_adaptor_config(adaptor_type)
    print("-----------------------------------------")
    # ic(adaptor_config)
    # match adaptor_type:
        # case "lora":
    ic(get_peft_model(model, adaptor_config))
    return get_peft_model(model, adaptor_config)
            # return (model, adaptor_config
        # case "ia3":
        #     return get_peft_model(model, adaptor_config)
    
    print("no case matched!!! check config.yaml")
    print("no case matched!!! check config.yaml")
    print("no case matched!!! check config.yaml")


"""
Suggestions:
- X-Lora
- OFT
- FourierFt
- LayerNorm Tuning
"""