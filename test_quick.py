"""
简化版扩展实验运行脚本
直接在代码中指定模型路径，避免路径搜索问题

使用方法：
1. 修改下面的 MODEL_PATHS 字典，填入您的模型路径
2. 运行: python run_simple_extended_exp.py
"""

import os
import sys
import torch
import numpy as np

# 导入您的模块
from data_utils import load_data_by_grid_size, preprocess_data, create_data_loaders
from models import CNN3D_PINN_Model
from ablation_models import CNN3D_Only_Model, PINN_Only_Model
from extended_experiments import (
    ExtendedExperimentRunner,
    ExtendedExperimentConfig
)


# ============================================================================
# 配置区域 - 请在这里修改您的设置
# ============================================================================

# 1. 设置要分析的网格大小
GRID_SIZES = [10, 20]  # 可以修改为 [5, 10, 15, 20]

# 2. 设置模型路径（请根据您的实际情况修改）
MODEL_PATHS = {
    10: {  # 10mm网格的模型
        'CNN_Only': r'C:\Users\DuChe\Documents\3cdnnpinn\3DCNN_PINN\ablation_experiments\ablation_experiments_10mm\architecture_ablation\CNN3D_Only\best_model.pth',
        'PINN_Only': r'C:\Users\DuChe\Documents\3cdnnpinn\3DCNN_PINN\ablation_experiments\ablation_experiments_10mm\architecture_ablation\PINN_Only\best_model.pth',
        '3DCNN_PINN': r'C:\Users\DuChe\Documents\3cdnnpinn\3DCNN_PINN\ablation_experiments\ablation_experiments_10mm\architecture_ablation\CNN3D_PINN_Fusion\best_model.pth'
    },
    20: {  # 20mm网格的模型
         'CNN_Only': r'C:\Users\DuChe\Documents\3cdnnpinn\3DCNN_PINN\ablation_experiments\ablation_experiments_20mm\architecture_ablation\CNN3D_Only\best_model.pth',
        'PINN_Only': r'C:\Users\DuChe\Documents\3cdnnpinn\3DCNN_PINN\ablation_experiments\ablation_experiments_20mm\architecture_ablation\PINN_Only\best_model.pth',
        '3DCNN_PINN': r'C:\Users\DuChe\Documents\3cdnnpinn\3DCNN_PINN\ablation_experiments\ablation_experiments_20mm\architecture_ablation\CNN3D_PINN_Fusion\best_model.pth'
    },
    # 如果您有其他网格大小，按相同格式添加
    # 5: {
    #     'CNN_Only': r'路径',
    #     'PINN_Only': r'路径',
    #     '3DCNN_PINN': r'路径'
    # },
}

# 3. 输出目录
OUTPUT_DIR = r'.\pinn_validation_results'

# 4. 设备设置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 5. 实验配置
CONFIG = ExtendedExperimentConfig(
    boundary_percentage=0.1,   # 边界区域占10%
    center_percentage=0.6,     # 中心区域占60%
    grid_sizes=[5.0, 10.0, 15.0, 20.0]
)

# 6. 批次大小
BATCH_SIZE = 16

# ============================================================================
# 以下代码无需修改
# ============================================================================


def load_trained_model(model_path, model_type='CNN3D_PINN'):
    """加载训练好的模型"""
    print(f"  加载模型: {os.path.basename(model_path)}")
    
    if model_type == 'CNN3D_PINN':
        model = CNN3D_PINN_Model()
    elif model_type == 'CNN_Only':
        model = CNN3D_Only_Model()
    elif model_type == 'PINN_Only':
        model = PINN_Only_Model()
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            print(f"    ✓ 成功")
            return model
        except Exception as e:
            print(f"    ✗ 失败: {e}")
            return None
    else:
        print(f"    ✗ 文件不存在")
        return None


def get_model_predictions(model, data_loader, device='cpu'):
    """获取模型预测"""
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    print("  获取预测中...", end='', flush=True)
    
    with torch.no_grad():
        for batch in data_loader:
            point_series = batch['point_series'].to(device)
            error = batch['error'].to(device)
            
            outputs = model(point_series)
            if isinstance(outputs, tuple):
                predictions = outputs[0]
            else:
                predictions = outputs
            
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(error.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    ground_truth = np.concatenate(all_ground_truth, axis=0).flatten()
    
    print(f" 完成 ({len(predictions)} 个点)")
    
    return predictions, ground_truth


def reshape_to_grid(predictions, ground_truth):
    """重塑为2D网格"""
    n_points = len(predictions)
    grid_size = int(np.sqrt(n_points))
    
    # 如果不是完全平方数，调整
    if grid_size * grid_size != n_points:
        # 取最接近的平方数
        grid_size = int(np.sqrt(n_points))
        n_use = grid_size * grid_size
        predictions = predictions[:n_use]
        ground_truth = ground_truth[:n_use]
    
    pred_grid = predictions.reshape(grid_size, grid_size)
    gt_grid = ground_truth.reshape(grid_size, grid_size)
    
    return pred_grid, gt_grid


def main():
    """主函数"""
    print("="*70)
    print("PINN有效性验证扩展实验 - 简化版")
    print("="*70)
    
    print(f"\n设备: {DEVICE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"要分析的网格大小: {GRID_SIZES}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = {}
    
    for grid_size in GRID_SIZES:
        print(f"\n{'#'*70}")
        print(f"# 网格大小: {grid_size}mm")
        print(f"{'#'*70}")
        
        # 检查是否有该网格大小的模型配置
        if grid_size not in MODEL_PATHS:
            print(f"警告: 未配置 {grid_size}mm 的模型路径，跳过")
            continue
        
        try:
            # 1. 加载数据
            print(f"\n[1/4] 加载数据...")
            point_series, errors = load_data_by_grid_size(grid_size)
            print(f"  数据点数量: {len(point_series)}")
            
            # 2. 预处理
            print(f"\n[2/4] 数据预处理...")
            data_dict = preprocess_data(point_series, errors, random_state=42)
            data_loaders = create_data_loaders(data_dict, batch_size=BATCH_SIZE)
            print(f"  训练集: {len(data_dict['X_train'])}")
            print(f"  测试集: {len(data_dict['X_test'])}")
            
            # 3. 加载模型并获取预测
            print(f"\n[3/4] 加载模型并获取预测...")
            model_predictions_dict = {}
            
            for model_name, model_path in MODEL_PATHS[grid_size].items():
                print(f"\n模型: {model_name}")
                
                # 确定模型类型
                if 'PINN' in model_name and 'CNN' in model_name:
                    model_type = 'CNN3D_PINN'
                elif 'CNN' in model_name:
                    model_type = 'CNN_Only'
                elif 'PINN' in model_name:
                    model_type = 'PINN_Only'
                else:
                    model_type = 'CNN3D_PINN'
                
                # 加载模型
                model = load_trained_model(model_path, model_type)
                
                if model is not None:
                    # 获取预测
                    predictions, ground_truth = get_model_predictions(
                        model, data_loaders['test_loader'], DEVICE
                    )
                    
                    mae = np.mean(np.abs(predictions - ground_truth))
                    print(f"  MAE: {mae:.6f}")
                    
                    model_predictions_dict[model_name] = predictions
            
            if len(model_predictions_dict) == 0:
                print("没有成功加载任何模型！")
                continue
            
            # 4. 重塑为2D网格并运行实验
            print(f"\n[4/4] 运行扩展实验...")
            
            # 重塑为2D
            model_predictions_2d = {}
            for model_name, predictions_1d in model_predictions_dict.items():
                pred_grid, gt_grid = reshape_to_grid(predictions_1d, ground_truth)
                model_predictions_2d[model_name] = pred_grid
            
            ground_truth_2d = gt_grid
            print(f"  网格形状: {ground_truth_2d.shape}")
            
            # 运行实验
            grid_output_dir = os.path.join(OUTPUT_DIR, f'grid_{grid_size}mm')
            os.makedirs(grid_output_dir, exist_ok=True)
            
            runner = ExtendedExperimentRunner(CONFIG)
            results = runner.run_all_extended_experiments(
                model_predictions=model_predictions_2d,
                ground_truth=ground_truth_2d,
                output_dir=grid_output_dir
            )
            
            all_results[f'{grid_size}mm'] = results
            print(f"\n✓ {grid_size}mm 实验完成")
            
        except Exception as e:
            print(f"\n✗ 处理 {grid_size}mm 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成综合报告
    if all_results:
        print(f"\n{'='*70}")
        print("生成综合报告...")
        print(f"{'='*70}")
        generate_summary(all_results, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print("所有实验完成！")
    print(f"结果保存在: {OUTPUT_DIR}")
    print(f"{'='*70}")


def generate_summary(all_results, output_dir):
    """生成综合摘要"""
    summary_data = {}
    
    for grid_key, results in all_results.items():
        if 'boundary_error_stats' in results:
            improvements = results['boundary_error_stats'].get('improvement_analysis', {})
            
            for key, data in improvements.items():
                if '3DCNN_PINN' in key and 'boundary' in key:
                    summary_data[grid_key] = {
                        'MAE_improvement_%': data.get('MAE_improvement_%', 0),
                        'RMSE_improvement_%': data.get('RMSE_improvement_%', 0)
                    }
                    break
    
    # 保存摘要
    summary_path = os.path.join(output_dir, 'SUMMARY.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PINN有效性验证 - 结果摘要\n")
        f.write("="*50 + "\n\n")
        
        for grid_key in sorted(summary_data.keys()):
            data = summary_data[grid_key]
            f.write(f"{grid_key}:\n")
            f.write(f"  MAE改进: {data['MAE_improvement_%']:.2f}%\n")
            f.write(f"  RMSE改进: {data['RMSE_improvement_%']:.2f}%\n\n")
    
    print(f"\n摘要已保存: {summary_path}")
    print("\n结果预览:")
    with open(summary_path, 'r', encoding='utf-8') as f:
        print(f.read())


if __name__ == "__main__":
    main()