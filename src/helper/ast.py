from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Type, Any
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
    PrefixTuningConfig
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
    try:
        model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
        processor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError: # if model is not cached, download it
        model = download_model(pretrained_AST_model, CACHE_DIR)
        processor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
    
    # print(f"does the model exist:: {model}")
    model = get_adaptor_model(model, adaptor_type)
    # ic(model)
    # if model is not None:
        # model.config.num_labels = num_classes
    in_features = model.classifier.dense.in_features
    model.classifier.dense = nn.Linear(in_features, num_classes)
        
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
Sonnet Suggestions:
def get_adaptor_config(adaptor_type: str):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    match adaptor_type.lower():
        case "lora":
            return LoraConfig(
                r=config["lora"]["r"],  # Rank of the update matrices
                lora_alpha=config["lora"]["lora_alpha"],  # Alpha parameter for LoRA scaling
                target_modules=config["lora"]["target_modules"],  # List of module names to apply LoRA to
                lora_dropout=config["lora"]["lora_dropout"],
                bias=config["lora"]["bias"],  # "none", "all" or "lora_only"
                task_type="AUDIO_CLASSIFICATION"  # Specify correct task type
            )
        
        case "ia3":
            return IA3Config(
                target_modules=config["ia3"]["target_modules"],
                feedforward_modules=config["ia3"]["feedforward_modules"],
                task_type="AUDIO_CLASSIFICATION"
            )

        case "adalora":
            return AdaLoraConfig(
                target_modules=config["adalora"]["target_modules"],
                init_r=config["adalora"]["init_r"],
                target_r=config["adalora"]["target_r"],
                beta1=config["adalora"]["beta1"],
                beta2=config["adalora"]["beta2"],
                task_type="AUDIO_CLASSIFICATION"
            )

        case "prefix":
            return PrefixTuningConfig(
                num_virtual_tokens=config["prefix"]["num_virtual_tokens"],
                task_type="AUDIO_CLASSIFICATION"
            )
            
        case _:
            raise ValueError(f"Unknown adaptor type: {adaptor_type}")

def get_adaptor_model(model, adaptor_type: str):
    # Get a PEFT model with the specified adapter configuration
    adaptor_config = get_adaptor_config(adaptor_type)
    
    # Get PEFT model
    model = get_peft_model(model, adaptor_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


"""

# If I had more data I think I would try to use the soft-MoA and MoE approaches