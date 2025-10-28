import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from trainer import train_model, evaluate_model
from models import CNN3D_PINN_Model, PhysicsLoss
class GeneralizationExperiment:
    def __init__(self, device, save_dir):
        self.device = device
        self.save_dir = save_dir
        self.grid_sizes = [5, 10, 15, 20]  # 可用的网格尺寸
    
    def run_cross_grid_generalization(self, epochs=100, lr=0.001):
        """运行跨网格尺寸泛化实验"""
        print("=== 开始跨网格尺寸泛化实验 ===")
        
        from data_utils import load_data_by_grid_size, preprocess_data, create_data_loaders
        
        generalization_results = {}
        
        # 对每个网格尺寸进行训练和测试
        for train_grid in self.grid_sizes:
            for test_grid in self.grid_sizes:
                if train_grid == test_grid:
                    continue  # 跳过相同尺寸的训练和测试
                
                experiment_name = f"Train_{train_grid}mm_Test_{test_grid}mm"
                print(f"\n--- {experiment_name} ---")
                
                try:
                    # 加载训练数据
                    train_point_series, train_errors = load_data_by_grid_size(train_grid)
                    train_data_dict = preprocess_data(train_point_series, train_errors, random_state=42)
                    train_loaders = create_data_loaders(train_data_dict, batch_size=16)
                    
                    # 加载测试数据
                    test_point_series, test_errors = load_data_by_grid_size(test_grid)
                    test_data_dict = preprocess_data(test_point_series, test_errors, random_state=42)
                    test_loaders = create_data_loaders(test_data_dict, batch_size=16)
                    
                    # 训练模型
                    model = CNN3D_PINN_Model()
                    criterion = PhysicsLoss(boundary_weight=0.001, smoothness_weight=0.001)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                    model_dir = os.path.join(self.save_dir, f"generalization/{experiment_name}")
                    os.makedirs(model_dir, exist_ok=True)
                    
                    trained_model, history = train_model(
                        model=model,
                        train_loader=train_loaders['train_loader'],
                        val_loader=train_loaders['val_loader'],
                        criterion=criterion,
                        optimizer=optimizer,
                        num_epochs=epochs,
                        device=self.device,
                        scheduler=None,
                        patience=20,
                        save_dir=model_dir
                    )
                    
                    # 在测试数据上评估
                    eval_results = evaluate_model(
                        model=trained_model,
                        test_loader=test_loaders['test_loader'],
                        device=self.device
                    )
                    
                    generalization_results[experiment_name] = {
                        'train_grid': train_grid,
                        'test_grid': test_grid,
                        'eval_metrics': eval_results
                    }
                    
                    print(f"结果 - MAE: {eval_results['mae']:.6f}, RMSE: {eval_results['rmse']:.6f}")
                    
                except Exception as e:
                    print(f"实验 {experiment_name} 失败: {str(e)}")
        
        # 可视化泛化结果
        self._visualize_generalization_results(generalization_results)
        return generalization_results
    
    def _visualize_generalization_results(self, results):
        """可视化泛化实验结果"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建泛化性能矩阵
        mae_matrix = np.full((len(self.grid_sizes), len(self.grid_sizes)), np.nan)
        
        for experiment_name, result in results.items():
            train_idx = self.grid_sizes.index(result['train_grid'])
            test_idx = self.grid_sizes.index(result['test_grid'])
            mae_matrix[train_idx, test_idx] = result['eval_metrics']['mae']
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(mae_matrix, cmap='RdYlBu_r', aspect='auto')
        plt.colorbar(label='MAE')
        plt.xlabel('Test Grid Size (mm)')
        plt.ylabel('Train Grid Size (mm)')
        plt.title('Cross-Grid Generalization Performance')
        
        # 设置刻度标签
        plt.xticks(range(len(self.grid_sizes)), [f'{size}mm' for size in self.grid_sizes])
        plt.yticks(range(len(self.grid_sizes)), [f'{size}mm' for size in self.grid_sizes])
        
        # 在每个格子中显示数值
        for i in range(len(self.grid_sizes)):
            for j in range(len(self.grid_sizes)):
                if not np.isnan(mae_matrix[i, j]):
                    plt.text(j, i, f'{mae_matrix[i, j]:.4f}', 
                            ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'generalization_matrix.png'), dpi=300)
        plt.show()