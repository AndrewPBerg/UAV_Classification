from transformers import (
    ASTFeatureExtractor, ASTForAudioClassification, AutoFeatureExtractor, AutoModel, 
    Wav2Vec2BertForSequenceClassification, Wav2Vec2BertModel, Wav2Vec2Processor, AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq, AutoProcessor
)
from torch import nn
import os
import torch
from typing import Optional, Tuple, Union

# Hard coded Pretrained models
pretrained_AST_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
pretrained_BERT_model = "facebook/w2v-bert-2.0"
pretrained_WHISPER_model = "openai/whisper-large-v3-turbo"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")

# Modify the get_model_and_processor function to use the new AudioClassifier
def get_model_and_processor(model_name: str, num_classes: int):
    if model_name == "AST":
        model, processor = custom_AST(num_classes)
    elif model_name == "BERT":
        model, processor = custom_BERT(num_classes)
    elif model_name == "WHISPER":
        model, processor = custom_WHISPER(num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    return model, processor

# downloads and store cached version in mounted Docker Volume
# Makes runs fast and the containers very tiny
def download_model(model_name, cache_dir):
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

def custom_BERT(num_classes: int, device: str):
    try:
        model = Wav2Vec2BertForSequenceClassification.from_pretrained(
            pretrained_BERT_model, 
            cache_dir=CACHE_DIR, 
            local_files_only=True,
            num_labels=num_classes,  # Specify the number of labels
            # ignore_mismatched_sizes=True  # Ignore mismatched sizes
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_BERT_model, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError: # if model is not cached, download it
        model = download_model(pretrained_BERT_model, CACHE_DIR)
        feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_BERT_model, cache_dir=CACHE_DIR)

    if model is not None:
        # Initialize the new layers with appropriate sizes
        model.classifier = nn.Linear(model.config.hidden_size, num_classes)
        model.projector = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        
        # Update the config to reflect the new number of labels
        model.config.num_labels = num_classes

        # Create a new label mapping
        model.config.id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.projector.parameters():
            param.requires_grad = True

    return model, feature_extractor

def custom_AST(num_classes: int, device: str):
    try:
        model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
        feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError: # if model is not cached, download it
        model = download_model(pretrained_AST_model, CACHE_DIR)
        feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
    
    if model is not None:
        model.config.num_labels = num_classes
        in_features = model.classifier.dense.in_features
        model.classifier.dense = nn.Linear(in_features, num_classes)
        
        # Add label mappings
        model.config.id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    
    return model, feature_extractor

def custom_WHISPER(num_classes: int):
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_WHISPER_model, cache_dir=CACHE_DIR, local_files_only=True)
        feature_extractor = AutoProcessor.from_pretrained(pretrained_WHISPER_model, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError: # if model is not cached, download it
        model = download_model(pretrained_AST_model, CACHE_DIR)
        feature_extractor = AutoProcessor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
        
    if model is not None:
        pass
    
    return model, feature_extractor
        
        