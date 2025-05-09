#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型定义模块
==========
定义3D CNN和PINN融合模型架构

包含3D CNN特征提取模块、PINN物理约束模块和融合模型的定义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3D_Module(nn.Module):
    """
    3D CNN特征提取模块
    
    从3D体素表示中提取几何特征
    """
    def __init__(self):
        super(CNN3D_Module, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # 第一层卷积层: 输入 (1, 3, 3, 1), 输出 (16, 3, 3, 1)
            nn.Conv3d(1, 16, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            
            # 第二层卷积层: 输入 (16, 3, 3, 1), 输出 (32, 3, 3, 1)
            nn.Conv3d(16, 32, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            
            # 第三层卷积层: 输入 (32, 3, 3, 1), 输出 (64, 3, 3, 1)
            nn.Conv3d(32, 64, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
        )
        
        # 计算特征提取后的平坦大小
        self.flatten_size = 64 * 3 * 3 * 1
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch, 3, 3, 1]
            
        Returns:
            输出张量，形状为 [batch, 1]
        """
        # 重塑为CNN3D输入 [batch, channels, depth, height, width]
        x = x.unsqueeze(1)  # 形状变为 [batch, 1, 3, 3, 1]
        
        # 通过CNN层
        x = self.cnn_layers(x)
        
        # 平坦化
        x = x.view(-1, self.flatten_size)
        
        # 全连接层
        x = self.fc_layers(x)
        
        return x


class PINN_Module(nn.Module):
    """
    物理启发神经网络模块
    
    提取点序列的差分特征，并应用物理约束
    """
    def __init__(self):
        super(PINN_Module, self).__init__()
        
        # 物理约束网络
        self.physics_net = nn.Sequential(
            nn.Linear(8, 32),  # 输入为8个相邻点差值
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x, diffs=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch, 3, 3, 1]
            diffs: 预计算的差分值，如果为None则自动计算
            
        Returns:
            输出张量，形状为 [batch, 1]
        """
        if diffs is None:
            # 将x重塑为 [batch, 9]
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            
            # 计算相邻点差值
            diffs = x_flat[:, 1:] - x_flat[:, :-1]  # [batch, 8]
        
        # 物理约束网络
        return self.physics_net(diffs)


class CNN3D_PINN_Model(nn.Module):
    """
    3D CNN和PINN融合模型
    
    结合几何特征提取和物理约束的融合模型
    """
    def __init__(self):
        super(CNN3D_PINN_Model, self).__init__()
        
        # CNN特征提取模块
        self.cnn_module = CNN3D_Module()
        
        # PINN物理约束模块
        self.pinn_module = PINN_Module()
        
        # 融合层
        self.fusion_layer = nn.Linear(2, 1)
        
    def calculate_differences(self, x):
        """
        计算点序列的差分值
        
        Args:
            x: 输入张量，形状为 [batch, 3, 3, 1]
            
        Returns:
            差分张量，形状为 [batch, 8]
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [batch, 9]
        
        # 计算差分
        diffs = x_flat[:, 1:] - x_flat[:, :-1]  # [batch, 8]
        return diffs
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch, 3, 3, 1]
            
        Returns:
            final_pred: 最终预测，形状为 [batch, 1]
            cnn_pred: CNN模块预测，形状为 [batch, 1]
            pinn_pred: PINN模块预测，形状为 [batch, 1]
        """
        # CNN特征提取预测
        cnn_pred = self.cnn_module(x)
        
        # 计算差分值
        diffs = self.calculate_differences(x)
        
        # PINN物理约束预测
        pinn_pred = self.pinn_module(x, diffs)
        
        # 融合预测
        combined = torch.cat([cnn_pred, pinn_pred], dim=1)
        final_pred = self.fusion_layer(combined)
        
        return final_pred, cnn_pred, pinn_pred


class PhysicsLoss(nn.Module):
    """
    物理约束损失函数
    
    结合数据拟合损失和物理约束损失
    """
    def __init__(self, boundary_weight=0.1, smoothness_weight=0.1):
        """
        初始化物理约束损失函数
        
        Args:
            boundary_weight: 边界约束权重
            smoothness_weight: 光滑性约束权重
        """
        super(PhysicsLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.boundary_weight = boundary_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, pred, target, point_series):
        """
        计算带物理约束的损失
        
        Args:
            pred: 模型预测值，形状为 [batch, 1]
            target: 真实目标值，形状为 [batch, 1]
            point_series: 原始点序列，形状为 [batch, 3, 3, 1]
            
        Returns:
            total_loss: 总损失（数据损失 + 物理约束损失）
        """
        # 数据拟合损失
        data_loss = self.mse_loss(pred, target)
        
        # 计算物理约束损失
        batch_size = pred.size(0)
        
        # 光滑性约束 - 回弹变形应该是平滑的
        if batch_size > 1:
            # 批次内相邻样本的预测值应平滑变化
            pred_diff = pred[1:] - pred[:-1]
            smoothness_loss = torch.mean(pred_diff**2)
            
            # 边界约束 - 回弹在边界处接近于零
            # 假设point_series的边缘点代表边界
            batch_point_series = point_series.view(batch_size, -1)  # [batch, 9]
            border_indices = [0, 2, 6, 8]  # 假设四角是边界点
            border_points = batch_point_series[:, border_indices]
            
            # 边界附近回弹应该较小
            # 这里我们使用边界点的平均绝对值作为约束
            border_constraint = torch.mean(torch.abs(border_points))
            boundary_loss = border_constraint
            
            # 总损失 = 数据损失 + 光滑性约束 + 边界约束
            total_loss = data_loss + \
                         self.smoothness_weight * smoothness_loss + \
                         self.boundary_weight * boundary_loss
        else:
            total_loss = data_loss
            
        return total_loss