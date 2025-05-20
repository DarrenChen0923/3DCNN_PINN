
"""
Tool function module
==========
Tool functions that provide various auxiliary functions

Including tool path compensation, result saving and loading, configuration management, etc.
"""

import os
import json
import numpy as np
import torch


def save_results(results, file_path):
    """
    Save evaluation results to JSON file

    Args:
    results: result dictionary
    file_path: save path

    Returns:
    None
    """
    # Convert NumPy data types to Python native types
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            # If it is an array, convert it to a list and make sure the elements are Python native types
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.number):
            # If it is a NumPy scalar type (such as float32, int64, etc.), convert it to a Python native type
            serializable_results[key] = value.item()
        else:
            # Checks if a list or nested structure contains NumPy types
            try:
                # Try using json.dumps to test if it is serializable
                json.dumps(value)
                serializable_results[key] = value
            except (TypeError, OverflowError):
                # If not serializable, try converting
                if isinstance(value, (list, tuple)):
                    # Recursively process each element in a list
                    serializable_results[key] = _make_serializable(value)
                else:
                    # For other types, try converting to string
                    serializable_results[key] = str(value)
    
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save as JSON
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f'Results saved to: {file_path}')

def _make_serializable(obj):
    """Recursively convert complex objects containing NumPy types to Python native types"""
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        # Handling other objects that may have a tolist method
        return obj.tolist()
    else:
        # For other types, try using them directly or converting them to strings
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


def save_model_info(model, file_path, additional_info=None):
    """
    Save model information

    Args:
    model: model
    file_path: save path
    additional_info: additional information dictionary

    Returns:
    None
    """
    # Get the number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Basic Information
    model_info = {
        'model_type': model.__class__.__name__,
        'num_parameters': num_params,
    }
    
    # Add additional information
    if additional_info:
        model_info.update(additional_info)
    
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save as JSON
    with open(file_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f'Model information has been saved to: {file_path}')


def inverse_transform_predictions(predictions, targets, y_scaler):
    """
    Convert normalized predictions and targets back to their original scale

    Args:
    predictions: normalized predictions
    targets: normalized targets
    y_scaler: normalizer

    Returns:
    orig_predictions: predictions on original scale
    orig_targets: targets on original scale
    """
    # Make sure the input is of the correct shape
    pred_reshaped = predictions.reshape(-1, 1)
    targets_reshaped = targets.reshape(-1, 1)
    
    # Inverse Transform
    orig_predictions = y_scaler.inverse_transform(pred_reshaped)
    orig_targets = y_scaler.inverse_transform(targets_reshaped)
    
    return orig_predictions, orig_targets


def set_seed(seed):
    """
    Set the random seed to ensure reproducible results

    Args:
    seed: random seed

    Returns:
    None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Setting up deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False