"""
Data processing utils module
==============
Handles loading, preprocessing and dataset construction of point series data

Contains data loading, normalization, partitioning, and dataset class definitions.
"""

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data_by_grid_size(grid_size):
    """
    Load all training data of the specified grid size
    
    Args:
        grid_size: Grid size（eg. 5mm, 10mm, 15mm, 20mm）
    
    Returns:
        point_series: All point series arrays of this grid size, shape is[n_samples, 9]
        errors: The corresponding error value array, shape is[n_samples]
    """
    # Build folder path
    folder_path = f"data/{grid_size}mm_file"
    
    # Check if a folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"File '{folder_path}' not exist")
    
    all_point_series = []
    all_errors = []
    
    # Traverse all outfile subfolders in a folder
    for outfile_dir in os.listdir(folder_path):
        outfile_path = os.path.join(folder_path, outfile_dir)
        
        # Make sure this is a directory
        if os.path.isdir(outfile_path):
            # Traverse all training files in the outfile directory
            for file_name in os.listdir(outfile_path):
                if file_name.startswith("trainingfile_") and file_name.endswith(".txt"):
                    file_path = os.path.join(outfile_path, file_name)
                    
                    # Loading data from a single file
                    point_series, errors = load_single_file(file_path)
                    
                    # Add to total dataset
                    all_point_series.append(point_series)
                    all_errors.append(errors)
    
    # Combine all data
    if all_point_series:
        combined_point_series = np.vstack(all_point_series)
        combined_errors = np.concatenate(all_errors)
        return combined_point_series, combined_errors
    else:
        raise ValueError(f"No valid training files found in '{folder_path}'")


def load_single_file(file_path):
    """
    Loading data from a single training file
    
    Args:
        file_path: Training file path
    
    Returns:
        point_series: Point series array，shape is [n_samples, 9]
        errors: Springback Error array，shape is [n_samples]
    """
    point_series = []
    errors = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ignore blank lines
                parts = line.split('|')
                if len(parts) == 2:
                    # Parsing point series
                    points_str = parts[0].strip()
                    points = [float(p.strip()) for p in points_str.split(',')]
                    
                    # Make sure the point series length is correct
                    if len(points) == 9:
                        # Parsing springback error 
                        error_str = parts[1].strip()
                        error = float(error_str)
                        
                        point_series.append(points)
                        errors.append(error)
    
    return np.array(point_series), np.array(errors)


def preprocess_data(point_series, errors, test_size=0.2, val_size=0.25, random_state=42):
    """
    Preprocessing point series data: Standardizing and partitioning the dataset
    
    Args:
        point_series: Point series data [n_samples, 9]
        errors: springback error [n_samples]
        test_size: Test set ratio
        val_size: Validation set ratio (based on training set)
        random_state: random_state
        
    Returns:
        Dictionary of dataset partitions and normalizers
    """
    # Normalize point series
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(point_series)
    
    # Normalize Springback error
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(errors.reshape(-1, 1)).flatten()
    
    # Divide the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )
    
    # Split into training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }


class PointSeriesDataset(Dataset):
    """
    PointSeriesDataset
    
    Convert 9-point series data into 3D voxel representation for use in 3D CNN models
    """
    def __init__(self, point_series, errors, transform=None):
        """
        Initialization point series dataset
        
        Args:
            point_series: point series data [n_samples, 9]
            errors: Corresponding springback error [n_samples]
            transform: transform
        """
        # Convert to 3D voxel format [n_samples, 3, 3, 1] - reshape the 9-point sequence into a 3x3 grid
        self.point_series = torch.FloatTensor(point_series.reshape(-1, 3, 3, 1)) 
        self.errors = torch.FloatTensor(errors).reshape(-1, 1)
        self.transform = transform
        
    def __len__(self):
        return len(self.errors)
    
    def __getitem__(self, idx):
        sample = {
            'point_series': self.point_series[idx],
            'error': self.errors[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def create_data_loaders(data_dict, batch_size=16):
    """
    create_data_loaders
    
    Args:
        data_dict: A dictionary containing partitioned datasets
        batch_size: batch size
        
    Returns:
        Data loader dictionary
    """
    train_dataset = PointSeriesDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = PointSeriesDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = PointSeriesDataset(data_dict['X_test'], data_dict['y_test'])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }