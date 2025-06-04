from typing import Optional, Literal, List, Dict, Set, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class PEFTCategory(Enum):
    """Categories of PEFT methods based on their implementation approach"""
    SELECTIVE = "selective"  # Methods that selectively enable/disable existing parameters
    ADDITIVE = "additive"    # Methods that add new trainable parameters (require merging)


class ModelType(Enum):
    """Supported model types"""
    CNN = "cnn"
    TRANSFORMER = "transformer"


# Define PEFT method categorization and model compatibility
PEFT_CATEGORIZATION = {
    # Selective methods (no reparameterization needed)
    "none-full": PEFTCategory.SELECTIVE,
    "none-classifier": PEFTCategory.SELECTIVE,
    "bitfit": PEFTCategory.SELECTIVE,
    "batchnorm": PEFTCategory.SELECTIVE,
    "layernorm": PEFTCategory.SELECTIVE,
    
    # Additive methods (require reparameterization/merging)
    "lora": PEFTCategory.ADDITIVE,
    "adalora": PEFTCategory.ADDITIVE,
    "hra": PEFTCategory.ADDITIVE,
    "ia3": PEFTCategory.ADDITIVE,
    "oft": PEFTCategory.ADDITIVE,
    "ssf": PEFTCategory.ADDITIVE,
    "lorac": PEFTCategory.ADDITIVE,
}

# Model compatibility mapping based on the models' peft_type lists
MODEL_PEFT_COMPATIBILITY = {
    ModelType.TRANSFORMER: {
        'lora', 'adalora', 'hra', 'ia3', 'oft', 'layernorm', 
        'none-full', 'none-classifier', 'ssf', 'bitfit'
    },
    ModelType.CNN: {
        'lorac', 'none-full', 'none-classifier', 'ssf', 'batchnorm'
    }
}

# Models that belong to each type
TRANSFORMER_MODELS = {
    'ast', 'mert', 'vit-base', 'vit-large',
    'deit-tiny', 'deit-small', 'deit-base',
    'deit-tiny-distil', 'deit-small-distil', 'deit-base-distil'
}

CNN_MODELS = {
    'resnet18', 'resnet50', 'resnet152', 'mobilenet_v3_small', 
    'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1', 
    'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
    'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'custom_cnn'
}


def get_model_type(model_name: str) -> ModelType:
    """Determine model type from model name"""
    if model_name in TRANSFORMER_MODELS:
        return ModelType.TRANSFORMER
    elif model_name in CNN_MODELS:
        return ModelType.CNN
    else:
        raise ValueError(f"Unknown model type for model: {model_name}")


def is_peft_compatible(model_name: str, peft_method: str) -> bool:
    """Check if a PEFT method is compatible with a model"""
    try:
        model_type = get_model_type(model_name)
        return peft_method in MODEL_PEFT_COMPATIBILITY[model_type]
    except ValueError:
        return False


def requires_reparameterization(peft_method: str) -> bool:
    """Check if a PEFT method requires reparameterization/merging"""
    return PEFT_CATEGORIZATION.get(peft_method, PEFTCategory.ADDITIVE) == PEFTCategory.ADDITIVE


class PEFTScheduleStep(BaseModel):
    """Configuration for a single PEFT scheduling step"""
    start_epoch: int = Field(description="Epoch to start this PEFT method (0-indexed)")
    peft_method: str = Field(description="PEFT method to use")
    merge_previous: bool = Field(
        default=True, 
        description="Whether to merge previous additive PEFT method before switching"
    )
    
    @field_validator('start_epoch')
    @classmethod
    def validate_start_epoch(cls, v):
        if v < 0:
            raise ValueError('start_epoch must be non-negative')
        return v
    
    @field_validator('peft_method')
    @classmethod
    def validate_peft_method(cls, v):
        if v not in PEFT_CATEGORIZATION:
            raise ValueError(f'Unknown PEFT method: {v}. Supported methods: {list(PEFT_CATEGORIZATION.keys())}')
        return v


class PEFTSchedulingConfig(BaseModel):
    """Configuration for PEFT method scheduling during training"""
    enabled: bool = Field(default=False, description="Whether to enable PEFT scheduling")
    model_name: Optional[str] = Field(default=None, description="Model name for compatibility validation")
    schedule: List[PEFTScheduleStep] = Field(
        default_factory=list,
        description="List of PEFT method changes during training"
    )
    auto_merge: bool = Field(
        default=True,
        description="Automatically merge additive PEFT methods when switching"
    )
    
    @field_validator('schedule')
    @classmethod
    def validate_schedule(cls, v, info):
        if not v:
            return v
        
        # Get model_name from the validation context
        model_name = info.data.get('model_name') if info.data else None
        
        # Check that start_epochs are in ascending order
        start_epochs = [step.start_epoch for step in v]
        if start_epochs != sorted(start_epochs):
            raise ValueError('Schedule steps must be ordered by start_epoch')
        
        # Check for duplicate start_epochs
        if len(set(start_epochs)) != len(start_epochs):
            raise ValueError('Schedule steps cannot have duplicate start_epochs')
        
        # Validate model compatibility if model_name is provided
        if model_name:
            for step in v:
                if not is_peft_compatible(model_name, step.peft_method):
                    model_type = get_model_type(model_name)
                    compatible_methods = MODEL_PEFT_COMPATIBILITY[model_type]
                    raise ValueError(
                        f'PEFT method "{step.peft_method}" is not compatible with {model_type.value} '
                        f'model "{model_name}". Compatible methods: {sorted(compatible_methods)}'
                    )
        
        # Validate merging logic
        for i in range(1, len(v)):
            current_step = v[i]
            previous_step = v[i-1]
            
            # If previous method requires merging and current step doesn't merge, warn
            if (requires_reparameterization(previous_step.peft_method) and 
                not current_step.merge_previous):
                print(f"Warning: Previous PEFT method '{previous_step.peft_method}' requires "
                      f"reparameterization but merge_previous=False for step at epoch {current_step.start_epoch}")
        
        return v
    
    def get_peft_config_for_epoch(self, current_epoch: int) -> str:
        """
        Get the appropriate PEFT method for the given epoch.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            
        Returns:
            PEFT method name for this epoch
        """
        if not self.enabled or not self.schedule:
            return "none-classifier"  # Default
        
        # Find the most recent schedule step that has started
        active_method = "none-classifier"  # Default before any scheduled changes
        for step in self.schedule:
            if current_epoch >= step.start_epoch:
                active_method = step.peft_method
            else:
                break
        
        return active_method
    
    def get_transition_info(self, current_epoch: int) -> Optional[Dict]:
        """
        Get information about PEFT method transitions at the current epoch.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            
        Returns:
            Dictionary with transition info if a transition occurs, None otherwise
        """
        if not self.enabled or not self.schedule:
            return None
        
        # Check if current epoch marks a transition
        for i, step in enumerate(self.schedule):
            if step.start_epoch == current_epoch:
                previous_method = None
                if i > 0:
                    previous_method = self.schedule[i-1].peft_method
                elif current_epoch > 0:
                    previous_method = "none-classifier"  # Default initial method
                
                return {
                    'from_method': previous_method,
                    'to_method': step.peft_method,
                    'requires_merge': (
                        previous_method and 
                        requires_reparameterization(previous_method) and 
                        step.merge_previous and 
                        self.auto_merge
                    ),
                    'from_category': PEFT_CATEGORIZATION.get(previous_method) if previous_method else None,
                    'to_category': PEFT_CATEGORIZATION.get(step.peft_method),
                    'step_config': step
                }
        
        return None
    
    def validate_model_compatibility(self, model_name: str) -> List[str]:
        """
        Validate all scheduled PEFT methods against a model.
        
        Args:
            model_name: Name of the model to validate against
            
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        try:
            model_type = get_model_type(model_name)
            compatible_methods = MODEL_PEFT_COMPATIBILITY[model_type]
            
            for step in self.schedule:
                if step.peft_method not in compatible_methods:
                    errors.append(
                        f'PEFT method "{step.peft_method}" at epoch {step.start_epoch} '
                        f'is not compatible with {model_type.value} model "{model_name}"'
                    )
        except ValueError as e:
            errors.append(str(e))
        
        return errors


def create_simple_schedule(
    switch_epoch: int = 10, 
    model_name: Optional[str] = None,
    from_method: str = "none-classifier",
    to_method: str = "none-full"
) -> PEFTSchedulingConfig:
    """
    Create a simple schedule that switches between two PEFT methods.
    
    Args:
        switch_epoch: Epoch to switch methods
        model_name: Model name for compatibility validation
        from_method: Initial PEFT method
        to_method: PEFT method to switch to
        
    Returns:
        PEFTSchedulingConfig with simple two-stage schedule
    """
    return PEFTSchedulingConfig(
        enabled=True,
        model_name=model_name,
        schedule=[
            PEFTScheduleStep(start_epoch=0, peft_method=from_method),
            PEFTScheduleStep(start_epoch=switch_epoch, peft_method=to_method)
        ]
    )


def create_progressive_schedule(
    model_name: str,
    classifier_epochs: int = 5,
    selective_epochs: int = 10,
    full_epochs: int = 20
) -> PEFTSchedulingConfig:
    """
    Create a progressive training schedule: classifier -> selective -> full training.
    
    Args:
        model_name: Model name for compatibility validation
        classifier_epochs: Epochs to train only classifier
        selective_epochs: Epochs to switch to selective method (bitfit/batchnorm)
        full_epochs: Epochs to switch to full fine-tuning
        
    Returns:
        PEFTSchedulingConfig with progressive schedule
    """
    model_type = get_model_type(model_name)
    
    # Choose appropriate selective method based on model type
    if model_type == ModelType.TRANSFORMER:
        selective_method = "bitfit"
    else:  # CNN
        selective_method = "batchnorm"
    
    return PEFTSchedulingConfig(
        enabled=True,
        model_name=model_name,
        schedule=[
            PEFTScheduleStep(start_epoch=0, peft_method="none-classifier"),
            PEFTScheduleStep(start_epoch=classifier_epochs, peft_method=selective_method),
            PEFTScheduleStep(start_epoch=selective_epochs, peft_method="none-full")
        ]
    )


def create_adapter_schedule(
    model_name: str,
    switch_epoch: int = 10,
    adapter_method: Optional[str] = None
) -> PEFTSchedulingConfig:
    """
    Create a schedule that uses adapter methods before full fine-tuning.
    
    Args:
        model_name: Model name for compatibility validation
        switch_epoch: Epoch to switch from adapter to full fine-tuning
        adapter_method: Specific adapter method to use (auto-selected if None)
        
    Returns:
        PEFTSchedulingConfig with adapter schedule
    """
    model_type = get_model_type(model_name)
    
    # Auto-select adapter method if not specified
    if adapter_method is None:
        if model_type == ModelType.TRANSFORMER:
            adapter_method = "lora"
        else:  # CNN
            adapter_method = "lorac"
    
    # Validate compatibility
    if not is_peft_compatible(model_name, adapter_method):
        compatible_methods = MODEL_PEFT_COMPATIBILITY[model_type]
        additive_methods = [m for m in compatible_methods if requires_reparameterization(m)]
        if additive_methods:
            adapter_method = additive_methods[0]
        else:
            raise ValueError(f"No compatible additive PEFT methods for model {model_name}")
    
    return PEFTSchedulingConfig(
        enabled=True,
        model_name=model_name,
        schedule=[
            PEFTScheduleStep(start_epoch=0, peft_method="none-classifier"),
            PEFTScheduleStep(start_epoch=5, peft_method=adapter_method),
            PEFTScheduleStep(start_epoch=switch_epoch, peft_method="none-full", merge_previous=True)
        ]
    )


def get_peft_scheduling_config(config: dict) -> Optional[PEFTSchedulingConfig]:
    """
    Extract PEFT scheduling configuration from the main config dictionary.
    
    Args:
        config: Main configuration dictionary
        
    Returns:
        PEFTSchedulingConfig instance or None if not configured
    """
    if "peft_scheduling" not in config:
        return PEFTSchedulingConfig()  # Default disabled config
    
    peft_scheduling_config_dict = config["peft_scheduling"]
    
    # Add model_name from the main config if not specified
    if "model_name" not in peft_scheduling_config_dict and "model" in config:
        peft_scheduling_config_dict["model_name"] = config["model"]["model_name"]
    
    return PEFTSchedulingConfig(**peft_scheduling_config_dict)


def get_compatible_peft_methods(model_name: str) -> Dict[str, List[str]]:
    """
    Get compatible PEFT methods for a model, categorized by type.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with 'selective' and 'additive' lists of compatible methods
    """
    model_type = get_model_type(model_name)
    compatible_methods = MODEL_PEFT_COMPATIBILITY[model_type]
    
    selective = [m for m in compatible_methods if not requires_reparameterization(m)]
    additive = [m for m in compatible_methods if requires_reparameterization(m)]
    
    return {
        'selective': sorted(selective),
        'additive': sorted(additive),
        'all': sorted(compatible_methods)
    } 