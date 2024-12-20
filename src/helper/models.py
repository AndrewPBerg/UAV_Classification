from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Type, Any
import os
import torch
from torch import nn
from transformers import (
    ASTFeatureExtractor, ASTForAudioClassification, AutoFeatureExtractor,
    AutoModel, Wav2Vec2BertForSequenceClassification, Wav2Vec2Processor,
    AutoModelForSpeechSeq2Seq, WhisperForAudioClassification, WhisperProcessor,
    HubertForSequenceClassification, Wav2Vec2FeatureExtractor
)
from peft import (
    get_peft_model, 
    LoraConfig,
    TaskType  # Change PeftType to TaskType
)
import traceback
import torch.nn.functional as F

@dataclass
class ModelConfig:
    name: str
    model_id: str
    model_class: Type
    processor_class: Type
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    lora_config: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.extra_model_kwargs is None:
            self.extra_model_kwargs = {}
        if self.extra_processor_kwargs is None:
            self.extra_processor_kwargs = {}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")

MODEL_CONFIGS = {
    "AST": ModelConfig(
        name="AST",
        model_id="MIT/ast-finetuned-audioset-10-10-0.4593",
        model_class=ASTForAudioClassification,
        processor_class=ASTFeatureExtractor,
        lora_config={
            "r": 4,
            "lora_alpha": 8,
            # "target_modules": ["mlp.dense", "classifier.dense"],
            "target_modules": ["mlp.dense"],
            "lora_dropout": 0.2,
            "bias": "none",
            "task_type": "SEQ_CLS"
        }
    ),
    "BERT": ModelConfig(
        name="BERT",
        model_id="facebook/w2v-bert-2.0",
        model_class=Wav2Vec2BertForSequenceClassification,
        processor_class=AutoFeatureExtractor
    ),
    "WHISPER": ModelConfig(
        name="WHISPER",
        model_id="openai/whisper-large-v3-turbo",
        model_class=WhisperForAudioClassification,
        processor_class=WhisperProcessor,
        extra_model_kwargs={
            "use_weighted_layer_sum": True,
            "classifier_proj_size": 256
        }
    ),
    "HUBERT": ModelConfig(
        name="HUBERT",
        model_id="superb/hubert-base-superb-ks",
        model_class=HubertForSequenceClassification,
        processor_class=Wav2Vec2FeatureExtractor,
        extra_model_kwargs={"ignore_mismatched_sizes": True}
    ),
    "MERT": ModelConfig(
        name="MERT",
        model_id="m-a-p/MERT-v1-330M",
        model_class=AutoModel,
        processor_class=Wav2Vec2FeatureExtractor,
        extra_model_kwargs={"trust_remote_code": True}
    )
}

class TorchCNN(nn.Module):
    def __init__(self, num_classes: int = 9, hidden_units: int = 256):
        """
        Initialize the CNN model with configurable hidden units for the fully connected layers.
        
        Args:
            num_classes (int): Number of output classes
            hidden_units (int): Number of hidden units in the fully connected layer
        """
        super(TorchCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._get_conv_output_size((128, 157))  # Initialize _to_linear
        
        # Dense layers with configurable hidden units
        self.fc1 = nn.Linear(self._to_linear, hidden_units)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def _get_conv_output_size(self, shape):
        """Helper function to calculate conv output size"""
        bs = 1
        x = torch.rand(bs, *shape)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        self._to_linear = x.shape[1]
        return self._to_linear

    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.flatten(1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ModelFactory:
    @staticmethod
    def create_model_and_processor(model_name: str, num_classes: int) -> Tuple:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        config = MODEL_CONFIGS[model_name]
        model, processor = ModelFactory._load_model_and_processor(config, num_classes)
        return model, processor

    @staticmethod
    def _load_model_and_processor(config: ModelConfig, num_classes: int) -> Tuple:
        model_kwargs: Dict[str, Any] = {
            "num_labels": num_classes,
            "cache_dir": CACHE_DIR,
            "ignore_mismatched_sizes": True
        }
        processor_kwargs: Dict[str, Any] = {"cache_dir": CACHE_DIR}
        
        model_kwargs.update(config.extra_model_kwargs)
        processor_kwargs.update(config.extra_processor_kwargs)

        try:
            model_kwargs["local_files_only"] = True
            processor_kwargs["local_files_only"] = True
            model = config.model_class.from_pretrained(config.model_id, **model_kwargs)
            processor = config.processor_class.from_pretrained(config.model_id, **processor_kwargs)
        except OSError:
            model_kwargs.pop("local_files_only")
            processor_kwargs.pop("local_files_only")
            model = config.model_class.from_pretrained(config.model_id, **model_kwargs)
            processor = config.processor_class.from_pretrained(config.model_id, **processor_kwargs)

        # Customize model after loading
        model = ModelFactory._customize_model(model, num_classes, config.name)
        return model, processor

    @staticmethod
    def _customize_model(model, num_classes: int, model_name: str):
        if model is None:
            return None

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Model-specific customizations
        if model_name == "AST":
            try:
                in_features = model.classifier.dense.in_features
                model.classifier.dense = nn.Linear(in_features, num_classes)
                
                # Get the LoRA config for AST
                config = MODEL_CONFIGS["AST"]
                if config.lora_config:
                    lora_config = LoraConfig(
                        r=config.lora_config["r"],
                        lora_alpha=config.lora_config["lora_alpha"],
                        target_modules=config.lora_config["target_modules"],
                        lora_dropout=config.lora_config["lora_dropout"],
                        bias=config.lora_config["bias"],
                        task_type=TaskType.SEQ_CLS,
                        inference_mode=False,
                    )
                    
                    model = get_peft_model(model, lora_config)
                    
                    original_forward = model.forward
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
                pass

        elif model_name == "BERT":
            model.classifier = nn.Linear(model.config.hidden_size, num_classes)
            model.projector = nn.Linear(model.config.hidden_size, model.config.hidden_size)
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.projector.parameters():
                param.requires_grad = True
        elif model_name in ["WHISPER", "HUBERT", "MERT"]:
            for param in model.classifier.parameters():
                param.requires_grad = True

        # Update label mappings
        model.config.num_labels = num_classes
        model.config.id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        return model

def get_model_and_processor(model_name: str, num_classes: int) -> Tuple:
    """Main function to get model and processor."""
    return ModelFactory.create_model_and_processor(model_name, num_classes)

def download_model(model_name: str, cache_dir: str) -> None:
    """Downloads and stores cached version in mounted Docker Volume."""
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

