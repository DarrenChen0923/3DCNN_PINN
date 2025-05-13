#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单点渐进成型回弹误差预测系统 - 主程序
=================================
集成3D CNN与PINN的SPIF回弹误差预测系统

使用方法:
python main.py --grid_size 20 --batch_size 16 --epochs 1000 --patience 1000

作者: Du Chen
日期: 2025-04-28
"""

import os
import argparse
import torch
import time
import datetime

from data_utils import load_data_by_grid_size, preprocess_data, create_data_loaders
from models import CNN3D_PINN_Model, PhysicsLoss
from trainer import train_model, evaluate_model, init_training
from visualization import (visualize_predictions, visualize_error_heatmap, 
                           visualize_training_history, visualize_grid_representation)
from utils import save_results, save_model_info, inverse_transform_predictions, set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单点渐进成型回弹误差预测系统')
    
    # 数据相关参数
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.25, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--grid_size', type=int, default=10, help='网格大小（mm）')

    
    # 模型相关参数
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练周期数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--boundary_weight', type=float, default=0.05, help='边界约束权重')
    parser.add_argument('--smoothness_weight', type=float, default=0.05, help='光滑性约束权重')
    
    # 系统相关参数
    parser.add_argument('--device', type=str, default='auto', help='训练设备，可选: cuda, cpu, auto')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--visualize', action='store_true',default=True, help='是否可视化结果')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"grid_{args.grid_size}mm_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 修改为使用网格大小加载数据
    print(f"加载网格大小为 {args.grid_size}mm 的数据")
    point_series, errors = load_data_by_grid_size(args.grid_size)
    print(f"加载完成: {len(point_series)} 个样本")
    
    # 显示数据示例
    if args.visualize:
        print("样本点序列可视化...")
        sample_idx = 0
        sample_point_series = point_series[sample_idx]
        visualize_grid_representation(
            sample_point_series, 
            save_path=os.path.join(output_dir, 'sample_grid.png')
        )
    
    # 预处理数据
    print("预处理数据...")
    data_dict = preprocess_data(
        point_series, errors, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        random_state=args.seed
    )
    
    # 创建数据加载器
    print(f"创建数据加载器, 批大小: {args.batch_size}")
    loaders = create_data_loaders(data_dict, batch_size=args.batch_size)
    
    # 初始化模型
    print("初始化模型...")
    model = CNN3D_PINN_Model()
    
    # 保存模型信息
    save_model_info(
        model, 
        os.path.join(output_dir, 'model_info.json'),
        additional_info={
            'data_size': len(point_series),
            'train_size': len(data_dict['X_train']),
            'val_size': len(data_dict['X_val']),
            'test_size': len(data_dict['X_test']),
            'batch_size': args.batch_size,
            'device': str(device)
        }
    )
    
    # 初始化训练组件
    print(f"初始化优化器和学习率调度器, 学习率: {args.lr}")
    optimizer, scheduler = init_training(model, lr=args.lr, patience=10)
    
    # 定义损失函数
    print(f"定义损失函数, 边界权重: {args.boundary_weight}, 光滑性权重: {args.smoothness_weight}")
    criterion = PhysicsLoss(
        boundary_weight=args.boundary_weight, 
        smoothness_weight=args.smoothness_weight
    )
    
    # 训练模型
    print(f"开始训练, 最大周期数: {args.epochs}, 早停耐心值: {args.patience}")
    start_time = time.time()
    
    model, history = train_model(
        model=model,
        train_loader=loaders['train_loader'],
        val_loader=loaders['val_loader'],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        patience=args.patience,
        save_dir=output_dir
    )
    
    training_time = time.time() - start_time
    print(f"训练完成, 耗时: {training_time:.2f} 秒")
    
    # 可视化训练历史
    if args.visualize:
        print("可视化训练历史...")
        visualize_training_history(
            history, 
            save_path=os.path.join(output_dir, 'training_history.png')
        )
    
    # 评估模型
    print("评估模型...")
    eval_results = evaluate_model(
        model=model,
        test_loader=loaders['test_loader'],
        device=device
    )
    
    # 反标准化预测结果
    orig_predictions, orig_targets = inverse_transform_predictions(
        eval_results['predictions'], 
        eval_results['targets'], 
        data_dict['y_scaler']
    )
    
    # 保存评估结果
    eval_results.update({
        'orig_predictions': orig_predictions,
        'orig_targets': orig_targets,
        'training_time': training_time
    })
    
    save_results(
        eval_results, 
        os.path.join(output_dir, 'evaluation_results.json')
    )
    
    # 打印评估指标
    print("\n=== 评估指标 ===")
    print(f"MAE: {eval_results['mae']:.4f}")
    print(f"RMSE: {eval_results['rmse']:.4f}")
    print(f"R²: {eval_results['r2']:.4f}")
    
    # 可视化预测结果
    if args.visualize:
        print("可视化预测结果...")
        visualize_predictions(
            orig_predictions, 
            orig_targets, 
            save_path=os.path.join(output_dir, 'predictions.png')
        )
        
        print("生成误差热力图...")
        visualize_error_heatmap(
            data_dict['X_test'], 
            eval_results['predictions'], 
            eval_results['targets'], 
            save_path=os.path.join(output_dir, 'error_heatmap.png')
        )
    
    print(f"\n所有结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()