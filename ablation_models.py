import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CNN3D_Module, PINN_Module

class CNN3D_Only_Model(nn.Module):
    """仅使用CNN3D的模型"""
    def __init__(self):
        super(CNN3D_Only_Model, self).__init__()
        self.cnn_module = CNN3D_Module()
    
    def forward(self, x):
        cnn_pred = self.cnn_module(x)
        return cnn_pred, cnn_pred, torch.zeros_like(cnn_pred)  # 保持输出格式一致

class PINN_Only_Model(nn.Module):
    """仅使用PINN的模型"""
    def __init__(self):
        super(PINN_Only_Model, self).__init__()
        self.pinn_module = PINN_Module()
    
    def calculate_differences(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        diffs = x_flat[:, 1:] - x_flat[:, :-1]
        return diffs
    
    def forward(self, x):
        diffs = self.calculate_differences(x)
        pinn_pred = self.pinn_module(x, diffs)
        return pinn_pred, torch.zeros_like(pinn_pred), pinn_pred

class Baseline_MLP_Model(nn.Module):
    """传统MLP基线模型"""
    def __init__(self, input_size=9):
        super(Baseline_MLP_Model, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # 将3D输入展平为1D
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        mlp_pred = self.mlp(x_flat)
        return mlp_pred, mlp_pred, torch.zeros_like(mlp_pred)