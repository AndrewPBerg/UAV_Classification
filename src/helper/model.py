from transformers import (
    ASTFeatureExtractor, ASTForAudioClassification, AutoFeatureExtractor, AutoModel, 
    Wav2Vec2BertForSequenceClassification
)
from torch import nn
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('transformers').setLevel(logging.ERROR)

# Hard coded Pretrained models
pretrained_AST_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
pretrained_BERT_model = "facebook/w2v-bert-2.0"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")

def get_model_and_processor(model_name: str, num_classes: int, device: str):
    if model_name == "AST":
        model, processor = custom_AST(num_classes, device)
        logger.info(f"Loading AST model from cache: {CACHE_DIR}")
    elif model_name == "BERT":
        model, processor = custom_BERT(num_classes, device)
        logger.info(f"Loading BERT model from cache: {CACHE_DIR}")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    # if isinstance(model, Wav2Vec2BertForSequenceClassification):
    #     # Adjust the classifier to match the projector output
    #     model.classifier = nn.Linear(768, num_classes)  # Changed from 1024 to 768
    
    return model, processor

# downloads and store cached version in mounted Docker Volume
# Makes runs fast and the containers very tiny
def download_model(model_name, cache_dir):
    logger.info(f"Manually downloading {model_name} to {cache_dir}")
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

# try: to use potentially cached model and feature extractor
# except: downloads and caches the new model and feature extractor
# then change the models classifier to correct num of labels

def custom_BERT(num_classes: int, device: str):
    try:
        model = Wav2Vec2BertForSequenceClassification.from_pretrained(
            pretrained_BERT_model, 
            cache_dir=CACHE_DIR, 
            local_files_only=True,
            num_labels=num_classes,  # Specify the number of labels
            ignore_mismatched_sizes=True  # Ignore mismatched sizes
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_BERT_model, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError: # if model is not cached, download it
        logger.info(f"Model not found in cache. Downloading {pretrained_BERT_model}")
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

        model = model.to(device) # type: ignore
    return model, feature_extractor

def custom_AST(num_classes: int, device: str):
    try:
        model = ASTForAudioClassification.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
        processor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR, local_files_only=True)
    except OSError: # if model is not cached, download it
        model = download_model(pretrained_AST_model, CACHE_DIR)
        processor = ASTFeatureExtractor.from_pretrained(pretrained_AST_model, cache_dir=CACHE_DIR)
    
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
        
        model = model.to(device) # type: ignore
    
    return model, processor

