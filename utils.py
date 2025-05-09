#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块
==========
提供各种辅助功能的工具函数

包含工具路径补偿、结果保存与加载、配置管理等功能
"""

import os
import json
import numpy as np
import torch


def compensate_tool_path(original_path, predicted_errors):
    """
    基于预测的回弹误差生成补偿工具路径
    
    Args:
        original_path: 原始工具路径，形状为 [n_points, 3]
        predicted_errors: 预测的回弹误差，形状为 [n_points]
        
    Returns:
        compensated_path: 补偿后的工具路径，形状为 [n_points, 3]
    """
    # 复制原始路径
    compensated_path = original_path.copy()
    
    # 应用负补偿（向相反方向调整以抵消回弹）
    compensated_path[:, 2] -= predicted_errors
    
    return compensated_path


def save_results(results, file_path):
    """
    保存评估结果到JSON文件
    
    Args:
        results: 结果字典
        file_path: 保存路径
        
    Returns:
        None
    """
    # 转换NumPy数据类型为Python原生类型
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            # 如果是数组，转换为列表并确保元素是Python原生类型
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.number):
            # 如果是NumPy标量类型(如float32, int64等)，转换为Python原生类型
            serializable_results[key] = value.item()
        else:
            # 检查是否为包含NumPy类型的列表或嵌套结构
            try:
                # 尝试使用json.dumps测试是否可序列化
                json.dumps(value)
                serializable_results[key] = value
            except (TypeError, OverflowError):
                # 如果不可序列化，尝试转换
                if isinstance(value, (list, tuple)):
                    # 递归处理列表中的每个元素
                    serializable_results[key] = _make_serializable(value)
                else:
                    # 对于其他类型，尝试转换为字符串
                    serializable_results[key] = str(value)
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存为JSON
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f'结果已保存至: {file_path}')

def _make_serializable(obj):
    """递归转换包含NumPy类型的复杂对象为Python原生类型"""
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        # 处理其他可能有tolist方法的对象
        return obj.tolist()
    else:
        # 对其他类型，尝试直接使用或转为字符串
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def load_results(file_path):
    """
    从JSON文件加载评估结果
    
    Args:
        file_path: 文件路径
        
    Returns:
        结果字典
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # 转换列表为NumPy数组
    for key in ['predictions', 'targets']:
        if key in results:
            results[key] = np.array(results[key])
    
    return results


def save_model_info(model, file_path, additional_info=None):
    """
    保存模型信息
    
    Args:
        model: 模型
        file_path: 保存路径
        additional_info: 额外信息字典
        
    Returns:
        None
    """
    # 获取模型参数数量
    num_params = sum(p.numel() for p in model.parameters())
    
    # 基本信息
    model_info = {
        'model_type': model.__class__.__name__,
        'num_parameters': num_params,
    }
    
    # 添加额外信息
    if additional_info:
        model_info.update(additional_info)
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存为JSON
    with open(file_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f'模型信息已保存至: {file_path}')


def inverse_transform_predictions(predictions, targets, y_scaler):
    """
    将标准化的预测值和目标值转换回原始尺度
    
    Args:
        predictions: 标准化的预测值
        targets: 标准化的目标值
        y_scaler: 标准化器
        
    Returns:
        orig_predictions: 原始尺度的预测值
        orig_targets: 原始尺度的目标值
    """
    # 确保输入是正确的形状
    pred_reshaped = predictions.reshape(-1, 1)
    targets_reshaped = targets.reshape(-1, 1)
    
    # 逆变换
    orig_predictions = y_scaler.inverse_transform(pred_reshaped)
    orig_targets = y_scaler.inverse_transform(targets_reshaped)
    
    return orig_predictions, orig_targets


def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
        
    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性运算
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False