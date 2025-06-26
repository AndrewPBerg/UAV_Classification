from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator


class LossConfig(BaseModel):
    """
    Loss function configuration
    """
    class Config:
        strict = True

    # Loss function type
    type: Literal["cross_entropy", "focal", "weighted_cross_entropy"] = "cross_entropy"
    
    # Label smoothing parameter
    label_smoothing: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Class weights for imbalanced datasets (optional)
    class_weights: Optional[List[float]] = None
    
    # Focal loss parameters (if using focal loss)
    focal_alpha: Optional[float] = Field(default=None, ge=0.0)
    focal_gamma: float = Field(default=2.0, ge=0.0)
    
    @field_validator('label_smoothing')
    @classmethod
    def validate_label_smoothing(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('label_smoothing must be between 0.0 and 1.0')
        return v
    
    @field_validator('class_weights')
    @classmethod
    def validate_class_weights(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError('class_weights cannot be empty if provided')
        return v


def get_loss_config(config: dict) -> LossConfig:
    """
    Create LossConfig from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LossConfig instance
    """
    loss_config_dict = config.get("loss", {})
    return LossConfig(**loss_config_dict) 