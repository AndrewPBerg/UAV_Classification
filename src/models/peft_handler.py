from peft import (
    get_peft_model,
    LoraConfig,
    IA3Config,
    TaskType,
    PeftModel
)
from config.model_config import PeftArgs
from transformers import PreTrainedModel
from typing import Optional

class PeftHandler:
    """Handles Parameter-Efficient Fine-Tuning (PEFT) configurations and application"""
    
    @staticmethod
    def get_default_target_modules() -> list:
        """Default attention modules to target in ViT models"""
        return ["query", "key", "value", "output.dense"]
    
    @staticmethod
    def create_peft_config(peft_args: PeftArgs):
        """Creates the appropriate PEFT configuration based on adapter type"""
        if peft_args.adapter_type == "lora":
            return LoraConfig(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                inference_mode=False,
                r=peft_args.r,
                lora_alpha=peft_args.alpha,
                lora_dropout=peft_args.dropout,
                bias=peft_args.bias,
                target_modules=peft_args.target_modules or PeftHandler.get_default_target_modules(),
                modules_to_save=peft_args.modules_to_save,
                init_lora_weights=peft_args.init_lora_weights
            )
        elif peft_args.adapter_type == "ia3":
            return IA3Config(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                inference_mode=False,
                target_modules=peft_args.target_modules or PeftHandler.get_default_target_modules(),
                modules_to_save=peft_args.modules_to_save,
                init_ia3_weights=True
            )
        else:
            raise ValueError(f"Unsupported adapter type: {peft_args.adapter_type}")
    
    @staticmethod
    def apply_peft(model: PreTrainedModel, peft_args: Optional[PeftArgs] = None) -> PreTrainedModel:
        """Applies PEFT to the model if peft_args is provided"""
        if peft_args is None:
            return model
            
        peft_config = PeftHandler.create_peft_config(peft_args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
