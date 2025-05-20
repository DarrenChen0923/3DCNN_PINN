
"""
Model training module
==========
Defines the functions of model training, evaluation and saving

Includes functions such as training loop, early stopping mechanism, model evaluation, etc.
"""

import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, device='cpu', scheduler=None, 
                patience=20, save_dir='./models'):
    """
    Model training
    
    Args:
        model: model
        train_loader: training data loader
        val_loader: validation data loader
        criterion: loss function
        optimizer: optimizer
        num_epochs: number of training cycles
        device: training device
        scheduler: learning rate scheduler
        patience: early stopping patience value
        save_dir: model save directory
        
    Returns:
        model: trained model
        history: training history
    """
    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Move the model to the device
    model.to(device)
    
    # Log training and validation losses
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    # Early stop setting
    best_val_loss = float('inf')
    counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training Stage
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            point_series = batch['point_series'].to(device)
            error = batch['error'].to(device)
            
            # Zero gradient
            optimizer.zero_grad()
            
            # Forward
            outputs, _, _ = model(point_series)
            loss = criterion(outputs, error, point_series)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate the average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Verification Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                point_series = batch['point_series'].to(device)
                error = batch['error'].to(device)
                
                # Forward Propagation
                outputs, _, _ = model(point_series)
                loss = criterion(outputs, error, point_series)
                
                val_loss += loss.item()
            
            # Calculate the average validation loss
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
        
        # Print the loss of the current epoch
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stop check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    
    return model, history


def evaluate_model(model, test_loader, device='cpu'):
    """evaluate_model"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            point_series = batch['point_series'].to(device)
            error = batch['error'].to(device)
            
            # forward
            outputs, _, _ = model(point_series)
            
            # Gather forecasts and goals
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(error.cpu().numpy())
    
    # Convert to NumPy array
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculating evaluation metrics
    mae = float(np.mean(np.abs(all_preds - all_targets)))  # 转换为Python float
    mse = float(np.mean((all_preds - all_targets)**2))     # 转换为Python float
    rmse = float(np.sqrt(mse))                             # 转换为Python float
    
    # Calculating R^2
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    ss_res = np.sum((all_targets - all_preds)**2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0  # 转换为Python float
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }


def init_training(model, lr=0.001, patience=10):
    """
    Initialize training components

    Args:
    model: model
    lr: learning rate
    patience: learning rate scheduler patience value

    Returns:
    optimizer and learning rate scheduler
    """
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=True
    )
    
    return optimizer, scheduler