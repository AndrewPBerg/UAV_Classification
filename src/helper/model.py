from transformers import ASTFeatureExtractor, ASTForAudioClassification
from torch import nn
# from peft import get_peft_model, LoraConfig





def auto_extractor(model_name: str) -> ASTFeatureExtractor:
    return ASTFeatureExtractor.from_pretrained(model_name)

def custom_AST(model_name: str, num_classes: int, device:str):

    model = ASTForAudioClassification.from_pretrained(model_name) 
    model.config.num_labels = num_classes

    in_features = model.classifier.dense.in_features  # Retain the input size of the pre-trained layer
    model.classifier.dense = nn.Linear(in_features, num_classes)  # Update for the new number of classes


    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    model.to(device) # type: ignore

    return model