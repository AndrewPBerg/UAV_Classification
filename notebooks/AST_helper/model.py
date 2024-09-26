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
    
    # lora_config = LoraConfig(
    #     r=8,  # Rank of the LoRA matrices
    #     lora_alpha=16,  # Scaling factor
    #     target_modules=["classifier"],  # Only apply LoRA to the classifier (lm_head)
    #     lora_dropout=0.1,  # Dropout for LoRA layers
    # )

    # # Apply PEFT (LoRA) to the model
    # peft_model = get_peft_model(model, lora_config)

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True

    # for name, param in model.named_parameters():
    #     if 'layernorm' in name.lower():  
    #         param.requires_grad = True
    #     if 'dropout' in name.lower():
    #         param.requires_grad = True
    
    model.to(device)

    return model
        