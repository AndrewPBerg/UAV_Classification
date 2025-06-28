from typing import Optional, Literal, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator


class AdamConfig(BaseModel):
    """Adam optimizer configuration"""
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


class AdamWConfig(BaseModel):
    """AdamW optimizer configuration"""
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


class AdamSPDConfig(BaseModel):
    """AdamSPD (Selective Projection Decay) optimizer configuration"""
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


class WarmupConfig(BaseModel):
    """Learning rate warmup configuration"""
    enabled: bool = False
    warmup_steps: int = 1000
    warmup_start_lr: float = 1e-6
    warmup_method: Literal["linear", "cosine"] = "linear"


class ReduceLROnPlateauConfig(BaseModel):
    """ReduceLROnPlateau scheduler configuration"""
    mode: Literal["min", "max"] = "min"
    factor: float = 0.85
    patience: int = 5
    threshold: float = 1e-4
    threshold_mode: Literal["rel", "abs"] = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8


class StepLRConfig(BaseModel):
    """StepLR scheduler configuration"""
    step_size: int = 30
    gamma: float = 0.1


class CosineAnnealingLRConfig(BaseModel):
    """CosineAnnealingLR scheduler configuration"""
    T_max: int = 50
    eta_min: float = 0.0


class SGDConfig(BaseModel):
    """Stochastic Gradient Descent optimizer configuration"""
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False

    # Validate that nesterov is only enabled when momentum > 0
    @field_validator('nesterov')
    @classmethod
    def validate_nesterov_requires_momentum(cls, v, values):
        if v and values.get('momentum', 0.0) == 0.0:
            raise ValueError('nesterov requires momentum > 0.0')
        return v


class OptimizerConfig(BaseModel):
    """
    Optimizer configuration with support for different optimizers and schedulers
    """
    class Config:
        strict = True

    # Optimizer selection
    optimizer_type: Literal["adam", "adamw", "adamspd", "sgd"] = "adamw"  # default to adamw
    
    # Optimizer configurations
    adam: AdamConfig = Field(default_factory=AdamConfig)
    adamw: AdamWConfig = Field(default_factory=AdamWConfig)
    adamspd: AdamSPDConfig = Field(default_factory=AdamSPDConfig)
    sgd: SGDConfig = Field(default_factory=SGDConfig)
    
    # Warmup configuration
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    
    # Scheduler selection and configurations
    scheduler_type: Optional[Literal["reduce_lr_on_plateau", "step_lr", "cosine_annealing_lr"]] = "reduce_lr_on_plateau"
    reduce_lr_on_plateau: ReduceLROnPlateauConfig = Field(default_factory=ReduceLROnPlateauConfig)
    step_lr: StepLRConfig = Field(default_factory=StepLRConfig)
    cosine_annealing_lr: CosineAnnealingLRConfig = Field(default_factory=CosineAnnealingLRConfig)
    
    # Gradient clipping
    gradient_clipping_enabled: bool = True
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Literal["value", "norm"] = "value"


def get_optimizer_config(config: dict) -> OptimizerConfig:
    """
    Get optimizer configuration from config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OptimizerConfig instance
    """
    # Check if optimizer config exists in the config
    if "optimizer" in config:
        return OptimizerConfig(**config["optimizer"])
    else:
        # Return default config if not specified
        return OptimizerConfig()
