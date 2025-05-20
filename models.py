
"""
Model definition module
==========
Define the 3D CNN and PINN fusion model architecture

Includes the definition of 3D CNN feature extraction module, PINN physical constraint module and fusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3D_Module(nn.Module):
    """
    3D CNN feature extraction module

    Extract geometric features from 3D voxel representation
    """
    def __init__(self):
        super(CNN3D_Module, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # First convolutional layer: input (1, 3, 3, 1), output (16, 3, 3, 1)
            nn.Conv3d(1, 16, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            
            # Second convolutional layer: input (16, 3, 3, 1), output (32, 3, 3, 1)
            nn.Conv3d(16, 32, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            
            # The third convolutional layer: input (32, 3, 3, 1), output (64, 3, 3, 1)
            nn.Conv3d(32, 64, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
        )
        
        # Calculate the flat size after feature extraction
        self.flatten_size = 64 * 3 * 3 * 1
        
        # Fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        forward
        
        Args:
            x: input tensor, shape: [batch, 3, 3, 1]
            
        Returns:
            Output tensor, shape is [batch, 1]
        """
        # Reshape to CNN3D input [batch, channels, depth, height, width]
        x = x.unsqueeze(1)  # shape change as [batch, 1, 3, 3, 1]
        
        # CNN layer
        x = self.cnn_layers(x)
        
        # Flattening
        x = x.view(-1, self.flatten_size)
        
        # Fully connected layer
        x = self.fc_layers(x)
        
        return x


class PINN_Module(nn.Module):
    """
    Physics-inspired neural network module

    Extracts differential features of point sequences and applies physical constraints
    """
    def __init__(self):
        super(PINN_Module, self).__init__()
        
        # Physics-inspired neural network
        self.physics_net = nn.Sequential(
            nn.Linear(8, 32),  #Input is 8 adjacent point spreads
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x, diffs=None):
        """
        forward
        
        Args:
            x: input tensor, shape is [batch, 3, 3, 1]
            diffs: pre-calculated difference value, if None, it will be automatically calculated
            
        Returns:
            Output tensor, shape is [batch, 1]
        """
        if diffs is None:
            # Reshape x to [batch, 9]
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            
            # Calculate the adjacent point difference
            diffs = x_flat[:, 1:] - x_flat[:, :-1]  # [batch, 8]
        
        # Physics-inspired neural network
        return self.physics_net(diffs)


class CNN3D_PINN_Model(nn.Module):
    """
    3D CNN and PINN fusion model

    Fusion model combining geometric feature extraction and physical constraints
    """
    def __init__(self):
        super(CNN3D_PINN_Model, self).__init__()
        
        # CNN feature extraction module
        self.cnn_module = CNN3D_Module()
        
        # PINN physical constraint module
        self.pinn_module = PINN_Module()
        
        # Fusion Layer
        self.fusion_layer = nn.Linear(2, 1)
        
    def calculate_differences(self, x):
        """
        Calculate the difference of a series of points
        
        Args:
           x: input tensor, shape: [batch, 3, 3, 1]
            
        Returns:
            Difference tensor, shape [batch, 8]
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [batch, 9]
        
        # Calculate the difference
        diffs = x_flat[:, 1:] - x_flat[:, :-1]  # [batch, 8]
        return diffs
        
    def forward(self, x):
        """
        forward
        
        Args:
            x: input tensor, shape: [batch, 3, 3, 1]
            
        Returns:
            final_pred: final prediction, shape is [batch, 1]
            cnn_pred: CNN module prediction, shape is [batch, 1]
            pinn_pred: PINN module prediction, shape is [batch, 1]
        """
        # CNN feature extraction prediction
        cnn_pred = self.cnn_module(x)
        
        # Calculate the difference value
        diffs = self.calculate_differences(x)
        
        # PINN physical constraint prediction
        pinn_pred = self.pinn_module(x, diffs)
        
        # Fusion prediction
        combined = torch.cat([cnn_pred, pinn_pred], dim=1)
        final_pred = self.fusion_layer(combined)
        
        return final_pred, cnn_pred, pinn_pred


class PhysicsLoss(nn.Module):
    """
    PhysicsLoss
    
    Combining data fitting loss and physical constraint loss
    """
    def __init__(self, boundary_weight=0.1, smoothness_weight=0.1):
        """
        Initialize the physical constraint loss function
        
        Args:
            boundary_weight: boundary_weight
            smoothness_weight: smoothness_weight
        """
        super(PhysicsLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.boundary_weight = boundary_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, pred, target, point_series):
        """
        Computing Losses with Physical Constraints
        
        Args:
            pred: model prediction value, shape is [batch, 1]
            target: true target value, shape is [batch, 1]
            point_series: original point series, shape is [batch, 3, 3, 1]
            
        Returns:
            total_loss: total loss (data loss + physical constraint loss)
        """
        # Data fitting loss
        data_loss = self.mse_loss(pred, target)
        
        # Calculating physical constraint losses
        batch_size = pred.size(0)
        
        # Smoothness constraint - springback deformation should be smooth
        if batch_size > 1:
            # The predicted values ​​of adjacent samples within a batch should change smoothly
            pred_diff = pred[1:] - pred[:-1]
            smoothness_loss = torch.mean(pred_diff**2)
            
            # Boundary constraints - springback is close to zero at the boundaries
            # Assume that the edge points of point_series represent the boundaries
            batch_point_series = point_series.view(batch_size, -1)  # [batch, 9]
            border_indices = [0, 2, 6, 8]  # Assume that the four corners are boundary points
            border_points = batch_point_series[:, border_indices]
            
            # The springback near the boundary should be small
            # Here we use the average absolute value of the boundary points as a constraint
            border_constraint = torch.mean(torch.abs(border_points))
            boundary_loss = border_constraint
            
            # Total loss = data loss + smoothness constraint + boundary constraint
            total_loss = data_loss + \
                         self.smoothness_weight * smoothness_loss + \
                         self.boundary_weight * boundary_loss
        else:
            total_loss = data_loss
            
        return total_loss