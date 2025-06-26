from pydantic import BaseModel, Field, PositiveInt, field_validator

class EnsembleConfig(BaseModel):
    """Configuration for ensemble inference.

    Attributes
    ----------
    enabled : bool
        Whether to activate ensemble mode. If ``False`` (default) the training/inference
        pipeline behaves exactly as before.
    ensemble_size : PositiveInt
        Number of individual models that constitute the ensemble (denoted **M** in the
        documentation). A value of ``1`` is treated as *no* ensemble learning and the
        pipeline will fall back to single-model behaviour. Values below ``1`` are not
        allowed.
    same_minibatch : bool
        Controls the data strategy during inference.

        * ``True``  – every model in the ensemble receives the **same** mini-batch.
        * ``False`` – each model processes **different** mini-batches (first *M* batches
          are grouped together).  
        Setting this flag to ``False`` is useful for Monte-Carlo style ensembling or
        bagging where each model sees a different slice of the evaluation set.
    """

    enabled: bool = False
    ensemble_size: PositiveInt = Field(1, alias="size")
    same_minibatch: bool = True

    # Constant name used across the codebase when referring to ensemble size
    ENSEMBLE_SIZE_CONST: str = "M"

    @field_validator("ensemble_size")
    @classmethod
    def _validate_size(cls, v: int):
        if v < 1:
            raise ValueError("Ensemble size (M) must be >= 1. Use 1 to disable ensembling.")
        return v

    def is_active(self) -> bool:
        """Return ``True`` if ensemble mode is effectively active."""
        return self.enabled and self.ensemble_size > 1

    # Convenience property to access *M*
    @property
    def M(self) -> int:  # noqa: N802 – keep upper-case for mathematical convention
        return self.ensemble_size 