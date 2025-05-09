#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理工具模块
==============
处理点序列数据的加载、预处理和数据集构建

包含数据加载、标准化、划分和数据集类的定义。
"""

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data_by_grid_size(grid_size):
    """
    加载指定网格大小的所有训练数据
    
    Args:
        grid_size: 网格大小（如 5mm, 10mm, 15mm, 20mm）
    
    Returns:
        point_series: 所有该网格大小的点序列数组，形状为[n_samples, 9]
        errors: 对应的误差值数组，形状为[n_samples]
    """
    # 构建文件夹路径
    folder_path = f"data/{grid_size}mm_file"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise ValueError(f"文件夹 '{folder_path}' 不存在")
    
    all_point_series = []
    all_errors = []
    
    # 遍历文件夹中的所有outfile子文件夹
    for outfile_dir in os.listdir(folder_path):
        outfile_path = os.path.join(folder_path, outfile_dir)
        
        # 确保这是一个目录
        if os.path.isdir(outfile_path):
            # 遍历该outfile目录中的所有训练文件
            for file_name in os.listdir(outfile_path):
                if file_name.startswith("trainingfile_") and file_name.endswith(".txt"):
                    file_path = os.path.join(outfile_path, file_name)
                    
                    # 加载单个文件中的数据
                    point_series, errors = load_single_file(file_path)
                    
                    # 添加到总数据集
                    all_point_series.append(point_series)
                    all_errors.append(errors)
    
    # 合并所有数据
    if all_point_series:
        combined_point_series = np.vstack(all_point_series)
        combined_errors = np.concatenate(all_errors)
        return combined_point_series, combined_errors
    else:
        raise ValueError(f"在 '{folder_path}' 中未找到有效的训练文件")


def load_single_file(file_path):
    """
    加载单个训练文件中的数据
    
    Args:
        file_path: 训练文件路径
    
    Returns:
        point_series: 点序列数组，形状为[n_samples, 9]
        errors: 误差值数组，形状为[n_samples]
    """
    point_series = []
    errors = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # 忽略空行
                parts = line.split('|')
                if len(parts) == 2:
                    # 解析点序列
                    points_str = parts[0].strip()
                    points = [float(p.strip()) for p in points_str.split(',')]
                    
                    # 确保点序列长度正确
                    if len(points) == 9:
                        # 解析误差值
                        error_str = parts[1].strip()
                        error = float(error_str)
                        
                        point_series.append(points)
                        errors.append(error)
    
    return np.array(point_series), np.array(errors)

# def load_point_series_data(file_path):
#     """
#     加载点序列数据文件
    
#     Args:
#         file_path: 数据文件路径，每行格式为"点1,点2,...,点9 | 误差值"
        
#     Returns:
#         point_series: 点序列数组，形状为[n_samples, 9]
#         errors: 误差值数组，形状为[n_samples]
#     """
#     point_series = []
#     errors = []
    
#     with open(file_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split('|')
#             if len(parts) == 2:
#                 # 解析点序列
#                 points = [float(p.strip()) for p in parts[0].split(',')]
#                 # 解析误差值
#                 error = float(parts[1].strip())
                
#                 point_series.append(points)
#                 errors.append(error)
    
#     return np.array(point_series), np.array(errors)


def preprocess_data(point_series, errors, test_size=0.2, val_size=0.25, random_state=42):
    """
    预处理点序列数据：标准化和划分数据集
    
    Args:
        point_series: 点序列数据 [n_samples, 9]
        errors: 误差值 [n_samples]
        test_size: 测试集比例
        val_size: 验证集比例（基于训练集）
        random_state: 随机种子
        
    Returns:
        数据集划分和标准化器的字典
    """
    # 标准化点序列
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(point_series)
    
    # 标准化误差
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(errors.reshape(-1, 1)).flatten()
    
    # 划分测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )
    
    # 划分训练集和验证集
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
    点序列数据集类
    
    将9点序列数据转换为3D体素表示，用于3D CNN模型
    """
    def __init__(self, point_series, errors, transform=None):
        """
        初始化点序列数据集
        
        Args:
            point_series: 点序列数据 [n_samples, 9]
            errors: 对应的回弹误差 [n_samples]
            transform: 数据转换函数
        """
        # 转换为3D体素格式 [n_samples, 3, 3, 1] - 将9点序列重塑为3x3网格
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
    创建数据加载器
    
    Args:
        data_dict: 包含划分数据集的字典
        batch_size: 批处理大小
        
    Returns:
        数据加载器字典
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