import os
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, Union
from transformers import (
    ASTFeatureExtractor, 
    ASTForAudioClassification, 
    AutoModel
)
from peft import get_peft_model, LoraConfig, IA3Config, AdaLoraConfig, OFTConfig, FourierFTConfig, LNTuningConfig
from peft.utils.peft_types import TaskType
from icecream import ic
from pathlib import Path

# Constants
CACHE_DIR = os.path.join("/app/src/model_cache")
DEFAULT_AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTModel:
    """
    Audio Spectrogram Transformer (AST) model wrapper.
    This class provides a clean interface for working with AST models,
    including support for PEFT methods.
    """
    
    @staticmethod
    def create_model(
        num_classes: int,
        adapter_type: str = "none",
        peft_config: Optional[Dict[str, Any]] = None,
        model_name: str = DEFAULT_AST_MODEL,
        device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, Any, Dict[str, Any]]:
        """
        Create an AST model with or without adapter.
        
        Args:
            num_classes: Number of classes
            adapter_type: Type of adapter to use
            peft_config: PEFT configuration
            model_name: Name of the AST model to use
            device: Device to put model on (no longer used with PyTorch Lightning)
            
        Returns:
            Tuple of (model, processor, adapter_config)
        """
        print(f"Creating AST model with {num_classes} classes, adapter_type: {adapter_type}")
        
        # Load base model and processor
        model = ASTModel._load_base_model(model_name)
        processor = ASTModel._load_processor(model_name)
        
        if model is None or processor is None:
            raise ValueError(f"Failed to load AST model or processor for {model_name}")
        
        # Set number of classes
        if hasattr(model.config, "num_labels"):
            model.config.num_labels = num_classes
        elif hasattr(model, "classifier") and hasattr(model.classifier, "out_features"):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features=in_features, out_features=num_classes)
        
        # Default empty adapter config
        adapter_config = {"type": "none"}
        
        # Handle PEFT if requested
        if adapter_type != "none" and peft_config is not None:
            adapter_config = peft_config
            print(f"Using adapter_type: {adapter_type}")
            
            # Create PEFT config
            peft_config_obj = ASTModel._create_peft_config(adapter_type, peft_config)
            if peft_config_obj is not None:
                model = get_peft_model(model, peft_config_obj)
                adapter_config["type"] = adapter_type
            
            # Print trainable parameters
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
        
        # Don't move model to device - PyTorch Lightning will handle this
        # Remove: if device is not None and model is not None:
        #     model = model.to(device)
            
        return model, processor, adapter_config
    
    @staticmethod
    def _load_base_model(model_name: str) -> Optional[nn.Module]:
        """
        Load the base AST model.
        
        Args:
            model_name: Name of the pretrained model
            
        Returns:
            Loaded model or None if loading failed
        """
        try:
            # Try to load from cache first
            return ASTForAudioClassification.from_pretrained(
                model_name, 
                attn_implementation="sdpa", 
                cache_dir=CACHE_DIR, 
                local_files_only=True
            )
        except OSError:
            # If not in cache, download it
            print(f"Model not found in cache. Downloading {model_name}...")
            try:
                return ASTForAudioClassification.from_pretrained(
                    model_name, 
                    attn_implementation="sdpa", 
                    cache_dir=CACHE_DIR
                )
            except Exception as e:
                print(f"Failed to download model: {str(e)}")
                return None
    
    @staticmethod
    def _load_processor(model_name: str) -> Optional[Any]:
        """
        Load the AST feature extractor.
        
        Args:
            model_name: Name of the pretrained model
            
        Returns:
            Loaded processor or None if loading failed
        """
        try:
            # Try to load from cache first
            return ASTFeatureExtractor.from_pretrained(
                model_name, 
                cache_dir=CACHE_DIR, 
                local_files_only=True
            )
        except OSError:
            # If not in cache, download it
            print(f"Processor not found in cache. Downloading {model_name}...")
            try:
                return ASTFeatureExtractor.from_pretrained(
                    model_name, 
                    cache_dir=CACHE_DIR
                )
            except Exception as e:
                print(f"Failed to download processor: {str(e)}")
                return None
    
    @staticmethod
    def _create_peft_config(adapter_type: str, config: Dict[str, Any]) -> Any:
        """
        Create a PEFT configuration based on the adapter type.
        
        Args:
            adapter_type: Type of adapter ('lora', 'ia3', etc.)
            config: Configuration parameters
            
        Returns:
            PEFT configuration object
        """
        if adapter_type == "lora":
            return LoraConfig(
                r=config.get("r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                target_modules=config.get("target_modules", ["query", "key", "value"]),
                lora_dropout=config.get("lora_dropout", 0.1),
                bias=config.get("bias", "none"),
                task_type=TaskType.SEQ_CLS,
                use_rslora=config.get("use_rslora", False),
                use_dora=config.get("use_dora", False),
            )
        
        elif adapter_type == "ia3":
            return IA3Config(
                target_modules=config.get("target_modules", ["query", "key", "value"]),
                feedforward_modules=config.get("feedforward_modules", ["output.dense"]),
                task_type=TaskType.SEQ_CLS
            )
        
        elif adapter_type == "adalora":
            return AdaLoraConfig(
                init_r=config.get("init_r", 12),
                target_r=config.get("target_r", 8),
                target_modules=config.get("target_modules", ["query", "key", "value"]),
                lora_alpha=config.get("lora_alpha", 16),
                task_type=TaskType.SEQ_CLS
            )
        
        elif adapter_type == "oft":
            return OFTConfig(
                r=config.get("r", 8),
                target_modules=config.get("target_modules", ["query", "key", "value"]),
                module_dropout=config.get("module_dropout", 0.1),
                init_weights=config.get("init_weights", True),
            )
        
        elif adapter_type == "fourier":
            return FourierFTConfig(
                target_modules=config.get("target_modules", ["query", "key", "value"]),
                task_type=TaskType.SEQ_CLS,
                n_frequency=config.get("n_frequency", 8),
                scaling=config.get("scaling", 1.0),
            )
        
        elif adapter_type == "layernorm":
            return LNTuningConfig(
                target_modules=config.get("target_modules", ["layernorm"]),
                task_type=TaskType.SEQ_CLS
            )
        
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}") 