from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml
from icecream import ic



class AugmentationConfig(BaseModel):
    """
    nested pydantic model for general run configs

    Required Keys (will not be defaulted):
        N/A
    """
    class Config:
        strict = True

    augmentations_per_sample: int = 0
    augmentations: List[str] 
    aug_configs: Dict[str, BaseModel] | None # map  aug  name to `n` # of pydantic configs


class SinDistortionConfig(BaseModel):
    class Config:
        strict = True
    min_distortion: float = 0.1
    max_distortion: float = 0.6
    p: float = 1.0

class TanhDistortionConfig(BaseModel):
    class Config:
        strict = True
    min_distortion: float = 0.2
    max_distortion: float = 0.6
    p: float = 1.0

class AddGaussianNoiseConfig(BaseModel):
    class Config:
        strict = True
    min_amplitude: float = 0.1
    max_amplitude: float = 0.15
    p: float = 1.0

class PolarityInversionConfig(BaseModel):
    class Config:
        strict = True
    p: float = 1.0

class PitchShiftConfig(BaseModel):
    class Config:
        strict = True
    min_semitones: int = -12
    max_semitones: int = 9
    p: float = 1.0

class TimeStretchConfig(BaseModel):
    class Config:
        strict = True
    min_rate: float = 0.9
    max_rate: float = 1.0
    p: float = 1.0


def create_augmentation_configs(config: dict) -> Dict[str, BaseModel]:
    """
    Dynamically create augmentation configs based on the YAML config file
    """
    aug_config = config.get('augmentations', {})
    augmentations = aug_config.get('augmentations', [])
    
    config_map = {}
    
    # Mapping of augmentation names to their config classes
    aug_class_map = {
        'sin_distortion': {
            'class': SinDistortionConfig,
            'params': {
                'min_distortion': aug_config.get('sin_distortion_min_rate'),
                'max_distortion': aug_config.get('sin_distortion_max_rate'),
                'p': aug_config.get('sin_distortion_p')
            }
        },
        'time_stretch': {
            'class': TimeStretchConfig,
            'params': {
                'min_rate': aug_config.get('time_stretch_min_rate'),
                'max_rate': aug_config.get('time_stretch_max_rate'),
                'p': aug_config.get('time_stretch_p')
            }
        },
        'pitch_shift': {
            'class': PitchShiftConfig,
            'params': {
                'min_semitones': aug_config.get('pitch_shift_min_semitones'),
                'max_semitones': aug_config.get('pitch_shift_max_semitones'),
                'p': aug_config.get('pitch_shift_p')
            }
        },
        'polarity_inversion': {
            'class': PolarityInversionConfig,
            'params': {
                'p': aug_config.get('polarity_inversion_p')
            }
        },
        'gaussian_noise': {
            'class': AddGaussianNoiseConfig,
            'params': {
                'min_amplitude': aug_config.get('gaussian_noise_min_amplitude'),
                'max_amplitude': aug_config.get('gaussian_noise_max_amplitude'),
                'p': aug_config.get('gaussian_noise_p')
            }
        },
        'tanh_distortion': {
            'class': TanhDistortionConfig,
            'params': {
                'min_distortion': aug_config.get('tanh_distortion_min_rate'),
                'max_distortion': aug_config.get('tanh_distortion_max_rate'),
                'p': aug_config.get('tanh_distortion_p')
            }
        }
    }

    for aug_name in augmentations:
        if aug_name in aug_class_map:
            config_class = aug_class_map[aug_name]['class']
            params = aug_class_map[aug_name]['params']
            
            # Filter out None values to use defaults instead
            params = {k: v for k, v in params.items() if v is not None}
            
            try:
                config_map[aug_name] = config_class(**params)
            except ValidationError as e:
                print(f"Error creating config for {aug_name}: {e}")
                # Fall back to default values
                config_map[aug_name] = config_class()
        else:
            print(f"Warning: Unknown augmentation type '{aug_name}'")

    
    return AugmentationConfig(**config["augmentations"], aug_configs=config_map)


def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    ic(create_augmentation_configs(config))

if __name__ == "__main__":
    main()