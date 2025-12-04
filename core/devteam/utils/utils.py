from typing import Any
from copy import deepcopy

def merge_dicts(base: dict[str, Any], override: dict[str, Any]):
    # create a deep copy of the base dictionary
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_dicts(result[key], value)
        else:
            # Merge non-dictionary values
            result[key] = value
    return result