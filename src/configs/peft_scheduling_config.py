from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator


class PEFTScheduleStep(BaseModel):
    """Configuration for a single PEFT scheduling step"""
    start_epoch: int = Field(description="Epoch to start this PEFT method (0-indexed)")
    peft_method: Literal["none-classifier", "none-full"] = Field(description="PEFT method to use")
    
    @field_validator('start_epoch')
    @classmethod
    def validate_start_epoch(cls, v):
        if v < 0:
            raise ValueError('start_epoch must be non-negative')
        return v


class PEFTSchedulingConfig(BaseModel):
    """Configuration for PEFT method scheduling during training"""
    enabled: bool = Field(default=False, description="Whether to enable PEFT scheduling")
    schedule: List[PEFTScheduleStep] = Field(
        default_factory=list,
        description="List of PEFT method changes during training"
    )
    
    @field_validator('schedule')
    @classmethod
    def validate_schedule(cls, v):
        if not v:
            return v
        
        # Check that start_epochs are in ascending order
        start_epochs = [step.start_epoch for step in v]
        if start_epochs != sorted(start_epochs):
            raise ValueError('Schedule steps must be ordered by start_epoch')
        
        # Check for duplicate start_epochs
        if len(set(start_epochs)) != len(start_epochs):
            raise ValueError('Schedule steps cannot have duplicate start_epochs')
        
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


def create_simple_schedule(switch_epoch: int = 10) -> PEFTSchedulingConfig:
    """
    Create a simple schedule that switches from none-classifier to none-full at specified epoch.
    
    Args:
        switch_epoch: Epoch to switch from none-classifier to none-full
        
    Returns:
        PEFTSchedulingConfig with simple two-stage schedule
    """
    return PEFTSchedulingConfig(
        enabled=True,
        schedule=[
            PEFTScheduleStep(start_epoch=0, peft_method="none-classifier"),
            PEFTScheduleStep(start_epoch=switch_epoch, peft_method="none-full")
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
    return PEFTSchedulingConfig(**peft_scheduling_config_dict) 