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
    
    if model is not None:
        model.config.num_labels = num_classes
        in_features = model.classifier.dense.in_features
        # model.classifier.dense = nn.Linear(in_features, num_classes)
        
        # # Add label mappings
        # model.config.id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
        # model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.classifier.parameters():
        #     param.requires_grad = True

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Model-specific customizations
        try:
            # model.classifier.dense = nn.Linear(in_features, num_classes)
            # adaptor_config = get_peft_config(adaptor_type)

                
                model = get_adaptor_model(model, adaptor_type)
                ic(model)
                # original_forward = model.forward
                def new_forward(self, input_values=None, **kwargs):
                    if isinstance(input_values, torch.Tensor):
                        return self.base_model(input_values=input_values)
                    
                    if input_values is None and kwargs:
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor):
                                return self.base_model(input_values=v)
                    
                    return self.base_model(input_values=input_values)
                
                model.forward = new_forward.__get__(model)
                
        except Exception as e:
            # logger.error(f"Error in custom_AST: {e}")
            raise e
    
    return model, processor

def get_adaptor_config(adaptor_type: str):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    match adaptor_type:
        case "lora":
            return LoraConfig(
                        r=config.adaptor_type["r"],
                        lora_alpha=config.adaptor_type["lora_alpha"],
                        target_modules=config.adaptor_type["target_modules"],
                        lora_dropout=config.adaptor_type["lora_dropout"],
                        bias=config.adaptor_type["bias"],
                        task_type=config.adaptor_type["task_type"],
                    )
        
        case "ia3":
            return IA3Config( 
                target_modules=config.adaptor_type["target_modules"],
                feedforward_modules=config.adaptor_type["feedforward_modules"]
                )
        case _:
            raise ValueError(f"Unknown adaptor type: {adaptor_type}")
        
def get_adaptor_model(model,adaptor_type: str):

    adaptor_config = get_adaptor_config(adaptor_type)
    print("-----------------------------------------")
    ic(adaptor_config)
    match adaptor_config:
        case "lora":
            return get_peft_model(model, adaptor_config)
        case "ia3":
            return get_peft_model(model, adaptor_config)
