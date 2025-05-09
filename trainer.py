#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练模块
==========
定义模型训练、评估和保存的功能

包含训练循环、早停机制、模型评估等功能。
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
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练周期数
        device: 训练设备
        scheduler: 学习率调度器
        patience: 早停耐心值
        save_dir: 模型保存目录
        
    Returns:
        model: 训练好的模型
        history: 训练历史记录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将模型移至设备
    model.to(device)
    
    # 记录训练和验证损失
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    # 早停设置
    best_val_loss = float('inf')
    counter = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            point_series = batch['point_series'].to(device)
            error = batch['error'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs, _, _ = model(point_series)
            loss = criterion(outputs, error, point_series)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                point_series = batch['point_series'].to(device)
                error = batch['error'].to(device)
                
                # 前向传播
                outputs, _, _ = model(point_series)
                loss = criterion(outputs, error, point_series)
                
                val_loss += loss.item()
            
            # 计算平均验证损失
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
        
        # 打印当前epoch的损失
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    
    return model, history


def evaluate_model(model, test_loader, device='cpu'):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            point_series = batch['point_series'].to(device)
            error = batch['error'].to(device)
            
            # 前向传播
            outputs, _, _ = model(point_series)
            
            # 收集预测和目标
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(error.cpu().numpy())
    
    # 转换为NumPy数组
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 计算评估指标
    mae = float(np.mean(np.abs(all_preds - all_targets)))  # 转换为Python float
    mse = float(np.mean((all_preds - all_targets)**2))     # 转换为Python float
    rmse = float(np.sqrt(mse))                             # 转换为Python float
    
    # 计算R^2
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
    初始化训练组件
    
    Args:
        model: 模型
        lr: 学习率
        patience: 学习率调度器耐心值
        
    Returns:
        优化器和学习率调度器
    """
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=True
    )
    
    return optimizer, scheduler