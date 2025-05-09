#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模块
========
提供预测结果和误差场的可视化功能

包含绘制预测对比图、误差热力图、训练历史曲线等功能。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')


def visualize_predictions(predictions, targets, save_path=None):
    """
    可视化预测值与真实值对比
    
    Args:
        predictions: 预测值 [n_samples, 1]
        targets: 真实值 [n_samples, 1]
        save_path: 保存路径
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    
    # 创建索引作为X轴
    idx = np.arange(len(predictions))
    
    # 绘制真实值和预测值
    plt.scatter(idx, targets, color='blue', label='真实回弹值', alpha=0.7)
    plt.scatter(idx, predictions, color='red', label='预测回弹值', alpha=0.7)
    
    # 绘制理想预测线（y=x）
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('样本索引')
    plt.ylabel('回弹误差值')
    plt.title('回弹预测对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加指标文本
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
        print(f'预测对比图已保存至: {save_path}')
    
    plt.show()


def visualize_error_heatmap(point_series, predictions, targets, save_path=None):
    """
    生成误差热力图可视化
    
    Args:
        point_series: 点序列数据 [n_samples, 9]
        predictions: 预测值 [n_samples, 1]
        targets: 真实值 [n_samples, 1]
        save_path: 保存路径
        
    Returns:
        None
    """
    # 计算预测误差
    errors = np.abs(predictions - targets)
    
    # 计算每个样本的点序列几何特征
    # 这里我们用点的平均值和方差作为特征
    n_samples = point_series.shape[0]
    means = np.mean(point_series.reshape(n_samples, -1), axis=1)
    stds = np.std(point_series.reshape(n_samples, -1), axis=1)
    
    # 创建2D散点图，颜色代表误差大小
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(means, stds, c=errors.flatten(), 
                          cmap='viridis', alpha=0.8, s=50)
    plt.colorbar(scatter, label='预测误差')
    
    plt.xlabel('点序列均值')
    plt.ylabel('点序列标准差')
    plt.title('点序列特征与预测误差关系')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'误差热力图已保存至: {save_path}')
    
    plt.show()


def visualize_training_history(history, save_path=None):
    """
    可视化训练历史
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'训练历史曲线已保存至: {save_path}')
    
    plt.show()


def visualize_grid_representation(point_series, save_path=None):
    """
    可视化点序列的3x3网格表示
    
    Args:
        point_series: 单个点序列 [9]
        save_path: 保存路径
        
    Returns:
        None
    """
    # 将点序列重塑为3x3网格
    grid_data = point_series.reshape(3, 3)
    
    plt.figure(figsize=(8, 6))
    
    # 左侧：点序列散点图
    plt.subplot(1, 2, 1)
    plt.scatter(range(9), point_series)
    plt.plot(range(9), point_series, 'b-', alpha=0.5)
    plt.title("点序列散点图")
    plt.xlabel("序列位置")
    plt.ylabel("高度差值")
    plt.grid(True)
    
    # 右侧：3×3网格热力图表示
    plt.subplot(1, 2, 2)
    im = plt.imshow(grid_data, cmap='viridis')
    plt.colorbar(im, label='高度差值')
    plt.title("3×3网格表示")
    
    # 添加网格值标注
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{grid_data[i, j]:.2f}", 
                     ha="center", va="center", color="white", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'网格表示图已保存至: {save_path}')
    
    plt.show()


def visualize_compensation(original_profile, predicted_errors, save_path=None):
    """
    可视化工具路径补偿效果
    
    Args:
        original_profile: 原始剖面曲线 [n_points, 2]
        predicted_errors: 预测的回弹误差 [n_points]
        save_path: 保存路径
        
    Returns:
        None
    """
    # 计算补偿后的剖面
    compensated_profile = original_profile.copy()
    compensated_profile[:, 1] -= predicted_errors  # 负补偿
    
    # 计算假设的回弹后剖面
    springback_profile = original_profile.copy()
    springback_profile[:, 1] += predicted_errors
    
    plt.figure(figsize=(10, 6))
    
    # 绘制原始剖面
    plt.plot(original_profile[:, 0], original_profile[:, 1], 'k-', 
             label='目标剖面', linewidth=2)
    
    # 绘制回弹剖面
    plt.plot(springback_profile[:, 0], springback_profile[:, 1], 'r--', 
             label='回弹剖面(预测)', linewidth=2)
    
    # 绘制补偿剖面
    plt.plot(compensated_profile[:, 0], compensated_profile[:, 1], 'g-', 
             label='补偿剖面', linewidth=2)
    
    plt.xlabel('X坐标')
    plt.ylabel('Z坐标')
    plt.title('工具路径补偿效果可视化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'补偿效果图已保存至: {save_path}')
    
    plt.show()