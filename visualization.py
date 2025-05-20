"""
Visualization module
========
Provides visualization functions for prediction results and error fields

Includes functions such as drawing prediction comparison charts, error heat maps, and training history curves.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the drawing style
plt.style.use('seaborn-v0_8-whitegrid')


def visualize_predictions(predictions, targets, save_path=None):
    """
    Visualize the comparison between predicted values ​​and true values

    Args:
    predictions: predicted values ​​[n_samples, 1]
    targets: true values ​​[n_samples, 1]
    save_path: save path

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    
    # Create an index as the X axis
    idx = np.arange(len(predictions))
    
    # Plot the true and predicted values
    plt.scatter(idx, targets, color='blue', label='Real error', alpha=0.7)
    plt.scatter(idx, predictions, color='red', label='Predicted error', alpha=0.7)
    
    # Draw the ideal prediction line (y=x)
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('Sample Index')
    plt.ylabel('Springback error')
    plt.title('Compare Springback')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add indicator text
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    ss_tot = np.sum((targets - np.mean(targets))**2)
    ss_res = np.sum((targets - predictions)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    plt.figtext(0.15, 0.85, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'The forecast comparison chart has been saved to: {save_path}')
    
    plt.show()


def visualize_error_heatmap(point_series, predictions, targets, save_path=None):
    """
    Generate error heat map visualization

    Args:
    point_series: point series data [n_samples, 9]
    predictions: predicted value [n_samples, 1]
    targets: true value [n_samples, 1]
    save_path: save path

    Returns:
    None
    """
    # Calculating prediction error
    errors = np.abs(predictions - targets)
    
    # Calculate the geometric features of the point sequence of each sample
    # Here we use the mean and variance of the points as features
    n_samples = point_series.shape[0]
    means = np.mean(point_series.reshape(n_samples, -1), axis=1)
    stds = np.std(point_series.reshape(n_samples, -1), axis=1)
    
    # Create a 2D scatter plot with colors representing the error size
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(means, stds, c=errors.flatten(), 
                          cmap='viridis', alpha=0.8, s=50)
    plt.colorbar(scatter, label='Prediction Error')
    
    plt.xlabel('Point series mean')
    plt.ylabel('Standard deviation of point sequence')
    plt.title('Relationship between point sequence characteristics and prediction error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'Error heatmap saved to: {save_path}')
    
    plt.show()


def visualize_training_history(history, save_path=None):
    """
    Visualize training history

    Args:
    history: training history dictionary
    save_path: save path

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and val loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'The training history curve has been saved to: {save_path}')
    
    plt.show()


def visualize_grid_representation(point_series, save_path=None):
    """
    Visualize a 3x3 grid representation of a point series

    Args:
    point_series: a single point series [9]
    save_path: save path

    Returns:
    None
    """
    # Reshape the point sequence into a 3x3 grid
    grid_data = point_series.reshape(3, 3)
    
    plt.figure(figsize=(8, 6))
    
    # Left: Scatter plot of point series
    plt.subplot(1, 2, 1)
    plt.scatter(range(9), point_series)
    plt.plot(range(9), point_series, 'b-', alpha=0.5)
    plt.title("Point sequence scatter plot")
    plt.xlabel("Sequence position")
    plt.ylabel("Height difference")
    plt.grid(True)
    
    # Right: 3×3 grid heat map representation
    plt.subplot(1, 2, 2)
    im = plt.imshow(grid_data, cmap='viridis')
    plt.colorbar(im, label='Height difference')
    plt.title("3×3 Grid Representation")
    
    # Adding grid value annotations
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{grid_data[i, j]:.2f}", 
                     ha="center", va="center", color="white", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'Grid representation saved to: {save_path}')
    
    plt.show()
