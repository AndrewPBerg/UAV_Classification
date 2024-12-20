# from script import main as script_main
# from sweeps import main as sweep_main
import yaml
from typing import Any
from icecream import ic
import sys
from helper.teleBot import send_message

def change_config_value(file_path: str, key: str, value: Any) -> None:
    """
    Updates a specific key in the YAML configuration file.
    Assumes the key exists and the file path is valid due to prior validation.
    
    Args:
        file_path (str): Path to the YAML configuration file
        key (str): Dot-separated path to the configuration key
        value (Any): New value to set for the key
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Navigate to the nested key
    current = config
    key_parts = key.split('.')
    
    # Set nested value
    for part in key_parts[:-1]:
        current = current[part]
    current[key_parts[-1]] = value
    
    # Write back to file
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def alter(changes: dict, file_path: str='config.yaml') -> None:
    """
    Applies validated changes to the config file.
    Assumes changes dictionary structure matches config.yaml structure.
    
    Args:
        changes (dict): Nested dictionary containing changes to apply
        file_path (str): Path to the config file
    """
    def process_nested_dict(d: dict, prefix: str = ''):
        for key, value in d.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                process_nested_dict(value, current_path)
            else:
                change_config_value(file_path, current_path, value)
    
    process_nested_dict(changes)

def is_valid(run: dict) -> tuple[bool, str]:
    """
    Validates a single run configuration from orchestrate.yaml.
    
    Args:
        run (dict): A single run configuration dictionary
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    # Check if required keys exist
    required_keys = ['id', 'type', 'changes']
    for key in required_keys:
        if key not in run:
            return False, f"Missing required key: {key}"
    
    # Validate id
    if not isinstance(run['id'], (int)):
        return False, f"Invalid id type: {type(run['id'])}. Must be int"
    
    # Validate type
    valid_types = ['script', 'sweep']
    if run['type'] not in valid_types:
        return False, f"Invalid type: {run['type']}. Must be one of {valid_types}"
    
    # Validate changes structure
    changes = run['changes']
    if not isinstance(changes, dict):
        return False, f"Changes must be a dictionary, got {type(changes)}"
    
    # Validate changes against config structure
    try:
        with open('src/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        def validate_nested_dict(changes: dict, config: dict, path: str = '') -> tuple[bool, str]:
            for key, value in changes.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if key exists in config
                if key not in config:
                    return False, f"Invalid key in changes: {current_path}"
                
                # Recursively validate nested dictionaries
                if isinstance(value, dict):
                    if not isinstance(config[key], dict):
                        return False, f"Mismatch in structure at {current_path}"
                    valid, error = validate_nested_dict(value, config[key], current_path)
                    if not valid:
                        return False, error
                
            return True, ""
        
        return validate_nested_dict(changes, config)
        
    except FileNotFoundError:
        return False, "Config file not found: src/config.yaml"
    except yaml.YAMLError:
        return False, "Error parsing config.yaml"

def main():
    # Load the orchestrate.yaml file
    with open('src/orchestrate.yaml', 'r') as f:
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
        sys.exit()
    
    ic('Orchestrate validation passed.')

    run_count = 0
    # Process valid runs
    for run in oc.get('runs'):
        
        id, changes, type = run.get('id'), run.get('changes'), run.get('type')
        
        ic(f'{id}: applying changes to config.yaml...')
        alter(changes, 'src/config.yaml')

        if type == 'script':
            ic(f'{id}: running script...')
            pass  # script_main()

        else:
            ic(f'{id}: running sweep...')
            pass  # sweep_main()
    
        run_count += 1
    if oc.get('SEND_MESSAGE'):
        send_message(f'f"Your Symphony has stopped playing\n" {run_count} runs completed.')
        
    ic('All runs completed.')

if __name__ == "__main__":
    main()
