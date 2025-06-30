import os
import torch
import sys
from icecream import ic

torch.set_float32_matmul_precision('medium')

# Import the rest of the modules
from helper.ptl_trainer import PTLTrainer
from helper.UAV_datamodule import UAVDataModule
from models.model_factory import ModelFactory
from configs import (
    load_configs,
)

from helper.util import wandb_login

from script import main as script_main
from sweeps import main as sweep_main
import yaml
from typing import Any
from helper.teleBot import send_message
import traceback
import wandb

def _get_nested(d: dict, path_parts: list[str]):
    """Return the nested value at *path_parts* or None if any segment is missing."""
    current = d
    for p in path_parts:
        if not isinstance(current, dict) or p not in current:
            return None
        current = current[p]
    return current


def _set_nested(d: dict, path_parts: list[str], value: Any):
    """Set *value* at the nested location described by *path_parts* creating dictionaries as required."""
    current = d
    for p in path_parts[:-1]:
        if p not in current or not isinstance(current[p], dict):
            current[p] = {}
        current = current[p]
    current[path_parts[-1]] = value

def change_config_value(file_path: str, key: str, value: Any) -> None:
    """
    Updates a specific key in the YAML configuration file.
    Creates nested structure if it doesn't exist.
    
    Args:
        file_path (str): Path to the YAML configuration file
        key (str): Dot-separated path to the configuration key
        value (Any): New value to set for the key
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Navigate to the nested key, creating structure as needed
    current = config
    key_parts = key.split('.')
    
    # Create nested dictionaries as needed
    for part in key_parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[key_parts[-1]] = value
    
    # Write back to file
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def alter(changes: dict, file_path: str='config.yaml') -> None:
    """
    Applies validated changes to the config file.
    For sweep parameters, replaces the entire parameters section.
    For other changes, updates individual keys.
    
    Args:
        changes (dict): Nested dictionary containing changes to apply
        file_path (str): Path to the config file
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    def apply_changes(target_dict: dict, changes_dict: dict, path: str = ''):
        for key, value in changes_dict.items():
            current_path = f"{path}.{key}" if path else key
            
            # If the key itself contains dots, treat it as a direct path reference regardless of nesting.
            if '.' in key:
                _set_nested(target_dict, key.split('.'), value)
                continue

            if isinstance(value, dict):
                # Special handling for sweep parameters - replace entirely
                if current_path == 'sweep.parameters':
                    target_dict[key] = value
                else:
                    # For other nested dicts, create if doesn't exist and recurse
                    if key not in target_dict:
                        target_dict[key] = {}
                    apply_changes(target_dict[key], value, current_path)
            else:
                # Direct value assignment
                target_dict[key] = value
    
    apply_changes(config, changes)
    
    # Write back to file
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def is_valid(run: dict) -> tuple[bool, str]:
    """
    Validates a single run configuration from orchestrate.yaml.
    """
    # Check if required keys exist
    required_keys = ['id', 'type', 'changes']
    for key in required_keys:
        if key not in run:
            return False, f"Missing required key: {key}"

    # Validate id
    if not isinstance(run['id'], int):
        return False, f"Invalid id type: {type(run['id'])}. Must be int"

    # Validate type
    valid_types = ['script', 'sweep']
    if run['type'] not in valid_types:
        return False, f"Invalid type: {run['type']}. Must be one of {valid_types}"

    # Validate changes structure
    changes = run['changes']
    if not isinstance(changes, dict):
        return False, f"Changes must be a dictionary, got {type(changes)}"

    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        def extract_paths(node: dict, prefix: str = '') -> list[str]:
            """Return a list of dotted paths for every key in *node* except those under sweep.parameters."""
            paths: list[str] = []
            for k, v in node.items():
                new_prefix = f"{prefix}.{k}" if prefix else k

                # Skip deep validation for sweep.parameters values –
                # they intentionally allow arbitrary dotted keys for wandb sweeps.
                if new_prefix.startswith('sweep.parameters'):
                    paths.append('sweep.parameters')
                    continue

                if isinstance(v, dict):
                    paths.extend(extract_paths(v, new_prefix))
                else:
                    paths.append(new_prefix)
            return paths

        for dotted_path in extract_paths(changes):
            # skip the special case handled above
            if dotted_path == 'sweep.parameters':
                continue
            if _get_nested(config, dotted_path.split('.')) is None:
                return False, f"Invalid key in changes: {dotted_path}"

        return True, ""

    except FileNotFoundError:
        return False, "Config file not found: config.yaml"
    except yaml.YAMLError:
        return False, "Error parsing config.yaml"

def main():
    # Load the orchestrate.yaml file
    with open('configs/orchestrate.yaml', 'r') as f:
        oc = yaml.safe_load(f)
    
    # Validate all runs first
    validation_errors = []
    for run in oc.get('runs', []):
        valid, error = is_valid(run)
        if not valid:
            validation_errors.append(f"Run {run.get('id', 'unknown')}: {error}")
    
    # If there are validation errors, log them and exit
    if validation_errors:
        ic("Validation errors found:")
        for error in validation_errors:
            ic(error)
        ic('Exiting...')
        sys.exit(1)
    
    ic('Orchestrate validation passed.')

    run_count = 0
    # Process valid runs
    try:
        for run in oc.get('runs'):
            
            id, changes, type = run.get('id'), run.get('changes'), run.get('type')
            
            # Ensure any existing wandb run is finished before starting a new one
            try:
                if wandb.run is not None:
                    ic(f"Finishing previous wandb run: {wandb.run.name}")
                    wandb.finish()
            except Exception as e:
                ic(f"Warning: Error finishing previous wandb run: {e}")
                # Force finish any lingering runs
                try:
                    wandb.finish()
                except:
                    pass
            
            ic(f'{id}: applying changes to config.yaml...')
            alter(changes, 'configs/config.yaml')

            if type == 'script':
                ic(f'{id}: running script...')
                script_main()

            else:
                ic(f'{id}: running sweep...')
                sweep_main()
        
            run_count += 1
        ic('All runs completed.')
        
        # Make sure to finish the last wandb run
        try:
            if wandb.run is not None:
                ic(f"Finishing final wandb run: {wandb.run.name}")
                wandb.finish()
        except Exception as e:
            ic(f"Warning: Error finishing final wandb run: {e}")
                
        if oc.get('SEND_MESSAGE'):
            send_message(f'Your Symphony has stopped playing\n {run_count} run(s) completed.')
            
    except Exception as e:
        # Make sure to finish the wandb run even if there's an error
        try:
            if wandb.run is not None:
                ic(f"Finishing wandb run due to error: {wandb.run.name}")
                wandb.finish()
        except Exception as finish_error:
            ic(f"Warning: Error finishing wandb run during cleanup: {finish_error}")
            
        ic('Error occurred during orchestration:', e)
        traceback_str = ''.join(traceback.format_exc())
        ic(traceback_str)
        if oc.get('SEND_MESSAGE'):
                send_message(f'Your Symphony has failed @ run number: {run_count}.\n\n Traceback: {e}')

if __name__ == "__main__":
    # Print GPU information
    print("-" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 50)
    
    main()
