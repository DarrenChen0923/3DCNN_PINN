import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from trainer import train_model, evaluate_model
from ablation_models import CNN3D_Only_Model, PINN_Only_Model, Baseline_MLP_Model
from models import CNN3D_PINN_Model, PhysicsLoss
from itertools import product


class LossWrapper:
    """损失函数包装器，统一接口"""
    def __init__(self, loss_fn, use_physics=False):
        self.loss_fn = loss_fn
        self.use_physics = use_physics
    
    def __call__(self, pred, target, point_series=None):
        if self.use_physics:
            return self.loss_fn(pred, target, point_series)
        else:
            return self.loss_fn(pred, target)

class PhysicsLossVariants:
    """物理损失函数的不同变体"""
    
    @staticmethod
    def no_physics_loss():
        """无物理约束损失"""
        return nn.MSELoss()
    
    @staticmethod
    def boundary_only_loss(boundary_weight=0.002):
        """仅边界约束损失"""
        class BoundaryOnlyLoss(nn.Module):
            def __init__(self, boundary_weight):
                super().__init__()
                self.mse_loss = nn.MSELoss()
                self.boundary_weight = boundary_weight
            
            def forward(self, pred, target, point_series):
                data_loss = self.mse_loss(pred, target)
                
                batch_size = pred.size(0)
                if batch_size > 1:
                    batch_point_series = point_series.view(batch_size, -1)
                    border_indices = [0, 2, 6, 8]
                    border_points = batch_point_series[:, border_indices]
                    boundary_loss = torch.mean(torch.abs(border_points))
                    total_loss = data_loss + self.boundary_weight * boundary_loss
                else:
                    total_loss = data_loss
                
                return total_loss
        
        return BoundaryOnlyLoss(boundary_weight)
    
    @staticmethod
    def smoothness_only_loss(smoothness_weight=0.001):
        """仅平滑性约束损失"""
        class SmoothnessOnlyLoss(nn.Module):
            def __init__(self, smoothness_weight):
                super().__init__()
                self.mse_loss = nn.MSELoss()
                self.smoothness_weight = smoothness_weight
            
            def forward(self, pred, target, point_series):
                data_loss = self.mse_loss(pred, target)
                
                batch_size = pred.size(0)
                if batch_size > 1:
                    pred_diff = pred[1:] - pred[:-1]
                    smoothness_loss = torch.mean(pred_diff**2)
                    total_loss = data_loss + self.smoothness_weight * smoothness_loss
                else:
                    total_loss = data_loss
                
                return total_loss
        
        return SmoothnessOnlyLoss(smoothness_weight)

class AblationExperiment:
    def __init__(self, data_loaders, device, save_dir):
        self.data_loaders = data_loaders
        self.device = device
        self.save_dir = save_dir
        self.results = {}
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
    
    def run_architecture_ablation(self, epochs=1000, lr=0.001):
        """运行架构消融实验"""
        print("=== 开始架构消融实验 ===")
        
        # 定义要测试的模型
        models_to_test = {
            'CNN3D_Only': CNN3D_Only_Model(),
            'PINN_Only': PINN_Only_Model(), 
            'MLP_Baseline': Baseline_MLP_Model(),
            'CNN3D_PINN_Fusion': CNN3D_PINN_Model()
        }
        
        for model_name, model in models_to_test.items():
            print(f"\n--- 训练模型: {model_name} ---")
            
            # 创建模型专用目录
            model_dir = os.path.join(self.save_dir, f"architecture_ablation/{model_name}")
            os.makedirs(model_dir, exist_ok=True)
            
            # 设置优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler=None
            # 为不同模型使用不同的损失函数
            if model_name == 'CNN3D_PINN_Fusion':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10, verbose=True
                )
                criterion = LossWrapper(
                    PhysicsLoss(boundary_weight=0.002, smoothness_weight=0.001), 
                    use_physics=True
                )
            else:
                criterion = LossWrapper(nn.MSELoss(), use_physics=False)
            
            # 训练模型（使用原始的train_model函数）
            try:
                from trainer import train_model
                trained_model, history = train_model(
                    model=model,
                    train_loader=self.data_loaders['train_loader'],
                    val_loader=self.data_loaders['val_loader'],
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=epochs,
                    device=self.device,
                    scheduler=scheduler,
                    patience=50,
                    save_dir=model_dir
                )
                
                # 评估模型
                eval_results = evaluate_model(
                    model=trained_model,
                    test_loader=self.data_loaders['test_loader'],
                    device=self.device
                )
                
                # 保存结果
                self.results[model_name] = {
                    'eval_metrics': eval_results,
                    'training_history': history,
                    'model_params': sum(p.numel() for p in model.parameters()),
                    'model_type': model_name
                }
                
                print(f"{model_name} - MAE: {eval_results['mae']:.6f}, RMSE: {eval_results['rmse']:.6f}, R²: {eval_results['r2']:.6f}")
                
            except Exception as e:
                print(f"训练 {model_name} 时出错: {str(e)}")
                continue
        
        # 保存对比结果
        self._save_architecture_comparison()
        return self.results
    
    def run_physics_loss_ablation(self, epochs=1000, lr=0.001):
        """运行物理损失消融实验"""
        print("=== 开始物理约束损失消融实验 ===")
        
        # 定义不同的损失函数配置
        loss_configs = {
        'No_Physics': LossWrapper(PhysicsLossVariants.no_physics_loss(), use_physics=False),
        'Boundary_Only': LossWrapper(PhysicsLossVariants.boundary_only_loss(0.002), use_physics=True),
        'Smoothness_Only': LossWrapper(PhysicsLossVariants.smoothness_only_loss(0.001), use_physics=True),
        'Full_Physics': LossWrapper(PhysicsLoss(boundary_weight=0.002, smoothness_weight=0.001), use_physics=True)
        }
        
        physics_results = {}
        
        for loss_name, criterion in loss_configs.items():
            print(f"\n--- 测试损失函数: {loss_name} ---")
            
            # 创建新的模型实例
            model = CNN3D_PINN_Model()
            
            # 创建保存目录
            model_dir = os.path.join(self.save_dir, f"physics_ablation/{loss_name}")
            os.makedirs(model_dir, exist_ok=True)
            
            # 设置优化器
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10, verbose=True
                )
            try:
                # 训练模型
                trained_model, history = train_model(
                    model=model,
                    train_loader=self.data_loaders['train_loader'],
                    val_loader=self.data_loaders['val_loader'],
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=epochs,
                    device=self.device,
                    scheduler=scheduler,
                    patience=50,
                    save_dir=model_dir
                )
                
                # 评估模型
                eval_results = evaluate_model(
                    model=trained_model,
                    test_loader=self.data_loaders['test_loader'],
                    device=self.device
                )
                
                physics_results[loss_name] = {
                    'eval_metrics': eval_results,
                    'training_history': history
                }
                
                print(f"{loss_name} - MAE: {eval_results['mae']:.6f}, RMSE: {eval_results['rmse']:.6f}, R²: {eval_results['r2']:.6f}")
                
            except Exception as e:
                print(f"测试 {loss_name} 时出错: {str(e)}")
        
        # 保存物理约束实验结果
        self._save_physics_ablation_results(physics_results)
        return physics_results
    
    def run_physics_weight_optimization(self, epochs=1000, lr=0.001):
        """运行物理权重优化实验"""
        print("=== 开始物理权重优化实验 ===")
        
        # 定义权重搜索范围 - 使用对数空间进行更细致的搜索
        boundary_weights = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
        smoothness_weights = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]

        
        # 生成所有权重组合
        weight_combinations = list(product(boundary_weights, smoothness_weights))
        total_combinations = len(weight_combinations)
        
        print(f"总共需要测试 {total_combinations} 种权重组合")
        print(f"边界权重范围: {boundary_weights}")
        print(f"平滑权重范围: {smoothness_weights}")
        
        physics_weight_results = {}
        best_result = None
        best_config = None
        best_metric = float('inf')  # 使用MAE作为主要评估指标
        
        for idx, (boundary_weight, smoothness_weight) in enumerate(weight_combinations):
            config_name = f"B{boundary_weight}_S{smoothness_weight}"
            print(f"\n--- 进度: {idx+1}/{total_combinations} - 测试权重组合: {config_name} ---")
            print(f"边界权重: {boundary_weight}, 平滑权重: {smoothness_weight}")
            
            # 创建损失函数
            if boundary_weight == 0.0 and smoothness_weight == 0.0:
                criterion = LossWrapper(PhysicsLossVariants.no_physics_loss(), use_physics=False)
            else:
                criterion = LossWrapper(
                    PhysicsLoss(boundary_weight=boundary_weight, smoothness_weight=smoothness_weight), 
                    use_physics=True
                )
            
            # 创建新的模型实例
            model = CNN3D_PINN_Model()
            
            # 创建保存目录
            model_dir = os.path.join(self.save_dir, f"physics_weight_optimization/{config_name}")
            os.makedirs(model_dir, exist_ok=True)
            
            # 设置优化器
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=False  # 关闭verbose减少输出
            )
            
            try:
                # 训练模型
                trained_model, history = train_model(
                    model=model,
                    train_loader=self.data_loaders['train_loader'],
                    val_loader=self.data_loaders['val_loader'],
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=epochs,
                    device=self.device,
                    scheduler=scheduler,
                    patience=30,  # 减少patience加速实验
                    save_dir=model_dir,
                )
                
                # 评估模型
                eval_results = evaluate_model(
                    model=trained_model,
                    test_loader=self.data_loaders['test_loader'],
                    device=self.device
                )
                
                # 记录结果
                physics_weight_results[config_name] = {
                    'boundary_weight': boundary_weight,
                    'smoothness_weight': smoothness_weight,
                    'mae': eval_results['mae'],
                    'rmse': eval_results['rmse'],
                    'r2': eval_results['r2'],
                    'training_history': history
                }
                
                current_mae = eval_results['mae']
                print(f"{config_name} - MAE: {current_mae:.6f}, RMSE: {eval_results['rmse']:.6f}, R²: {eval_results['r2']:.6f}")
                
                # 更新最佳结果
                if current_mae < best_metric:
                    best_metric = current_mae
                    best_result = eval_results.copy()
                    best_config = config_name
                    print(f"🎉 发现新的最佳配置: {config_name} (MAE: {current_mae:.6f})")
                
            except Exception as e:
                print(f"测试 {config_name} 时出错: {str(e)}")
                # 记录失败的实验
                physics_weight_results[config_name] = {
                    'boundary_weight': boundary_weight,
                    'smoothness_weight': smoothness_weight,
                    'error': str(e)
                }
        
        # 保存详细结果
        self._save_physics_weight_optimization_results(physics_weight_results, best_config, best_result)
        
        # 生成分析报告
        self._analyze_physics_weight_results(physics_weight_results)
        
        print(f"\n=== 物理权重优化完成 ===")
        print(f"最佳配置: {best_config}")
        print(f"最佳MAE: {best_metric:.6f}")
        
        return physics_weight_results, best_config, best_result

    def _save_physics_weight_optimization_results(self, results, best_config, best_result):
        """保存物理权重优化结果"""
        # 保存详细结果
        results_file = os.path.join(self.save_dir, "physics_weight_optimization_results.json")
        
        # 准备保存的数据（移除不能序列化的部分）
        save_results = {}
        for config_name, result in results.items():
            if 'error' not in result:
                save_results[config_name] = {
                    'boundary_weight': result['boundary_weight'],
                    'smoothness_weight': result['smoothness_weight'],
                    'mae': result['mae'],
                    'rmse': result['rmse'],
                    'r2': result['r2']
                }
        
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=4)
        
        # 保存最佳配置
        best_config_file = os.path.join(self.save_dir, "best_physics_weights.json")
        best_info = {
            'best_config': best_config,
            'best_result': best_result,
            'boundary_weight': results[best_config]['boundary_weight'],
            'smoothness_weight': results[best_config]['smoothness_weight']
        }
        
        with open(best_config_file, 'w') as f:
            json.dump(best_info, f, indent=4)
        
        print(f"结果已保存到: {results_file}")
        print(f"最佳配置已保存到: {best_config_file}")

    def _analyze_physics_weight_results(self, results):
        """分析物理权重实验结果并生成可视化"""
        print("\n=== 生成分析报告 ===")
        
        # 准备数据用于分析
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("没有有效的实验结果用于分析")
            return
        
        # 创建DataFrame用于分析
        df_data = []
        for config_name, result in valid_results.items():
            df_data.append({
                'config': config_name,
                'boundary_weight': result['boundary_weight'],
                'smoothness_weight': result['smoothness_weight'],
                'mae': result['mae'],
                'rmse': result['rmse'],
                'r2': result['r2']
            })
        
        df = pd.DataFrame(df_data)
        
        # 生成热力图
        self._plot_weight_heatmaps(df)
        
        # 生成排名表
        self._generate_ranking_table(df)
        
        # 分析权重影响
        self._analyze_weight_effects(df)

    def _plot_weight_heatmaps(self, df):
        """生成权重影响的热力图"""
        # 创建透视表
        pivot_mae = df.pivot(index='smoothness_weight', columns='boundary_weight', values='mae')
        pivot_r2 = df.pivot(index='smoothness_weight', columns='boundary_weight', values='r2')
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAE热力图
        sns.heatmap(pivot_mae, annot=True, fmt='.4f', cmap='viridis_r', ax=ax1)
        ax1.set_title('MAE vs Physics Weights')
        ax1.set_xlabel('Boundary Weight')
        ax1.set_ylabel('Smoothness Weight')
        
        # R²热力图
        sns.heatmap(pivot_r2, annot=True, fmt='.4f', cmap='viridis', ax=ax2)
        ax2.set_title('R² vs Physics Weights')
        ax2.set_xlabel('Boundary Weight')
        ax2.set_ylabel('Smoothness Weight')
        
        plt.tight_layout()
        heatmap_file = os.path.join(self.save_dir, "physics_weights_heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"热力图已保存到: {heatmap_file}")

    def _generate_ranking_table(self, df):
        """生成性能排名表"""
        # 按MAE排序
        df_sorted = df.sort_values('mae').reset_index(drop=True)
        
        print("\n=== 性能排名 (按MAE排序) ===")
        print("排名\t配置\t\t边界权重\t平滑权重\tMAE\t\tRMSE\t\tR²")
        print("-" * 80)
        
        for idx, row in df_sorted.head(10).iterrows():  # 显示前10名
            print(f"{idx+1}\t{row['config']}\t{row['boundary_weight']:.3f}\t\t{row['smoothness_weight']:.3f}\t\t{row['mae']:.6f}\t{row['rmse']:.6f}\t{row['r2']:.6f}")
        
        # 保存完整排名
        ranking_file = os.path.join(self.save_dir, "physics_weights_ranking.csv")
        df_sorted.to_csv(ranking_file, index=False)
        print(f"\n完整排名已保存到: {ranking_file}")

    def _analyze_weight_effects(self, df):
        """分析权重对性能的影响"""
        print("\n=== 权重影响分析 ===")
        
        # 分析边界权重的影响
        boundary_effect = df.groupby('boundary_weight')['mae'].agg(['mean', 'std', 'min']).round(6)
        print("\n边界权重对MAE的影响:")
        print(boundary_effect)
        
        # 分析平滑权重的影响
        smoothness_effect = df.groupby('smoothness_weight')['mae'].agg(['mean', 'std', 'min']).round(6)
        print("\n平滑权重对MAE的影响:")
        print(smoothness_effect)
        
        # 找出最佳权重范围
        best_mae = df['mae'].min()
        tolerance = 0.001  # 容忍度
        good_configs = df[df['mae'] <= best_mae + tolerance]
        
        print(f"\n最佳性能配置 (MAE <= {best_mae + tolerance:.6f}):")
        for _, row in good_configs.iterrows():
            print(f"  {row['config']}: 边界={row['boundary_weight']}, 平滑={row['smoothness_weight']}, MAE={row['mae']:.6f}")
        
        # 权重建议
        print(f"\n=== 权重建议 ===")
        optimal_boundary = good_configs['boundary_weight'].mode().iloc[0] if len(good_configs) > 0 else df.loc[df['mae'].idxmin(), 'boundary_weight']
        optimal_smoothness = good_configs['smoothness_weight'].mode().iloc[0] if len(good_configs) > 0 else df.loc[df['mae'].idxmin(), 'smoothness_weight']
        
        print(f"推荐边界权重: {optimal_boundary}")
        print(f"推荐平滑权重: {optimal_smoothness}")

    # 如果需要更细致的搜索，可以使用这个版本
    def run_fine_grained_physics_weight_search(self, best_boundary, best_smoothness, epochs=1000, lr=0.001):
        """在最佳权重附近进行细致搜索"""
        print("=== 开始细致权重搜索 ===")
        
        # 在最佳权重附近生成更密集的搜索点
        boundary_range = np.linspace(max(0, best_boundary - 0.01), best_boundary + 0.01, 5)
        smoothness_range = np.linspace(max(0, best_smoothness - 0.01), best_smoothness + 0.01, 5)
        
        weight_combinations = list(product(boundary_range, smoothness_range))
        
        print(f"细致搜索范围:")
        print(f"边界权重: {boundary_range}")
        print(f"平滑权重: {smoothness_range}")
        
        fine_results = {}
        
        for boundary_weight, smoothness_weight in weight_combinations:
            config_name = f"Fine_B{boundary_weight:.4f}_S{smoothness_weight:.4f}"
            print(f"\n测试精细权重: {config_name}")
            
            # 类似的训练和评估过程...
            # (这里可以复用上面的训练代码)
        
        return fine_results
    
    def run_hyperparameter_sensitivity(self, epochs=100, lr=0.001):
        """运行超参数敏感性分析"""
        print("=== 开始超参数敏感性分析 ===")
        
        # 定义要测试的权重组合
        boundary_weights = [0.0, 0.001, 0.01, 0.05, 0.1]
        smoothness_weights = [0.0, 0.001, 0.01, 0.05, 0.1]
        
        sensitivity_results = {}
        
        for b_weight in boundary_weights:
            for s_weight in smoothness_weights:
                config_name = f"B{b_weight}_S{s_weight}"
                print(f"\n--- 测试配置: {config_name} ---")
                
                # 创建模型和损失函数
                model = CNN3D_PINN_Model()
                
                # 使用LossWrapper包装损失函数
                if b_weight == 0.0 and s_weight == 0.0:
                    # 如果两个权重都为0，使用纯MSE损失
                    criterion = LossWrapper(nn.MSELoss(), use_physics=False)
                else:
                    # 否则使用物理约束损失
                    criterion = LossWrapper(
                        PhysicsLoss(boundary_weight=b_weight, smoothness_weight=s_weight), 
                        use_physics=True
                    )
                
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10, verbose=True
                )
                # 创建保存目录
                model_dir = os.path.join(self.save_dir, f"sensitivity/{config_name}")
                os.makedirs(model_dir, exist_ok=True)
                
                try:
                    # 训练模型（使用较少的epochs以节省时间）
                    from trainer import train_model
                    trained_model, history = train_model(
                        model=model,
                        train_loader=self.data_loaders['train_loader'],
                        val_loader=self.data_loaders['val_loader'],
                        criterion=criterion,
                        optimizer=optimizer,
                        num_epochs=epochs,
                        device=self.device,
                        scheduler=scheduler,
                        patience=50,
                        save_dir=model_dir
                    )
                    
                    # 评估模型
                    eval_results = evaluate_model(
                        model=trained_model,
                        test_loader=self.data_loaders['test_loader'],
                        device=self.device
                    )
                    
                    sensitivity_results[config_name] = {
                        'boundary_weight': b_weight,
                        'smoothness_weight': s_weight,
                        'mae': eval_results['mae'],
                        'rmse': eval_results['rmse'],
                        'r2': eval_results['r2']
                    }
                    
                    print(f"{config_name} - MAE: {eval_results['mae']:.6f}")
                    
                except Exception as e:
                    print(f"测试 {config_name} 时出错: {str(e)}")
        
        # 可视化敏感性结果
        self._visualize_sensitivity_results(sensitivity_results)
        return sensitivity_results
    
    def _save_architecture_comparison(self):
        """保存架构对比结果"""
        comparison_file = os.path.join(self.save_dir, "architecture_comparison.json")
        
        # 提取关键指标进行对比
        comparison_data = {}
        for model_name, results in self.results.items():
            if 'eval_metrics' in results:
                comparison_data[model_name] = {
                    'MAE': results['eval_metrics']['mae'],
                    'RMSE': results['eval_metrics']['rmse'],
                    'R²': results['eval_metrics']['r2'],
                    'Parameters': results['model_params']
                }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=4)
        
        print(f"架构对比结果已保存到: {comparison_file}")
    
    def _save_physics_ablation_results(self, physics_results):
        """保存物理约束消融实验结果"""
        physics_file = os.path.join(self.save_dir, "physics_ablation_results.json")
        
        # 提取关键指标
        physics_data = {}
        for loss_name, results in physics_results.items():
            if 'eval_metrics' in results:
                physics_data[loss_name] = {
                    'MAE': results['eval_metrics']['mae'],
                    'RMSE': results['eval_metrics']['rmse'],
                    'R²': results['eval_metrics']['r2']
                }
        
        with open(physics_file, 'w') as f:
            json.dump(physics_data, f, indent=4)
        
        print(f"物理约束消融结果已保存到: {physics_file}")
    
    def visualize_architecture_results(self):
        """可视化架构消融实验结果"""
        if not self.results:
            print("没有结果可视化")
            return
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        model_names = []
        maes = []
        rmses = []
        r2s = []
        params = []
        
        for model_name, results in self.results.items():
            if 'eval_metrics' in results:
                model_names.append(model_name)
                maes.append(results['eval_metrics']['mae'])
                rmses.append(results['eval_metrics']['rmse'])
                r2s.append(results['eval_metrics']['r2'])
                params.append(results['model_params'])
        
        # MAE对比
        axes[0,0].bar(model_names, maes, color='skyblue')
        axes[0,0].set_title('Mean Absolute Error (MAE)')
        axes[0,0].set_ylabel('MAE')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # RMSE对比
        axes[0,1].bar(model_names, rmses, color='lightcoral')
        axes[0,1].set_title('Root Mean Square Error (RMSE)')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # R²对比
        axes[1,0].bar(model_names, r2s, color='lightgreen')
        axes[1,0].set_title('Coefficient of Determination (R²)')
        axes[1,0].set_ylabel('R²')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 参数数量对比
        axes[1,1].bar(model_names, params, color='gold')
        axes[1,1].set_title('Model Parameters')
        axes[1,1].set_ylabel('Number of Parameters')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'architecture_comparison.png'), dpi=300)
        plt.show()
        
        # 创建性能改进表
        self._create_improvement_table()
    
    def _create_improvement_table(self):
        """创建性能改进对比表"""
        if 'CNN3D_PINN_Fusion' not in self.results:
            return
        
        fusion_results = self.results['CNN3D_PINN_Fusion']['eval_metrics']
        
        print("\n=== 性能改进分析 ===")
        print(f"{'模型':<20} {'MAE改进':<12} {'RMSE改进':<12} {'R²改进':<12}")
        print("-" * 60)
        
        for model_name, results in self.results.items():
            if model_name != 'CNN3D_PINN_Fusion' and 'eval_metrics' in results:
                mae_improvement = (results['eval_metrics']['mae'] - fusion_results['mae']) / results['eval_metrics']['mae'] * 100
                rmse_improvement = (results['eval_metrics']['rmse'] - fusion_results['rmse']) / results['eval_metrics']['rmse'] * 100
                r2_improvement = (fusion_results['r2'] - results['eval_metrics']['r2']) / abs(results['eval_metrics']['r2']) * 100 if results['eval_metrics']['r2'] != 0 else 0
                
                print(f"{model_name:<20} {mae_improvement:>8.2f}%   {rmse_improvement:>8.2f}%   {r2_improvement:>8.2f}%")
    
    def _visualize_sensitivity_results(self, results):
        """可视化敏感性分析结果"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建热力图数据
        boundary_weights = sorted(list(set([r['boundary_weight'] for r in results.values()])))
        smoothness_weights = sorted(list(set([r['smoothness_weight'] for r in results.values()])))
        
        mae_matrix = np.zeros((len(smoothness_weights), len(boundary_weights)))
        
        for i, s_weight in enumerate(smoothness_weights):
            for j, b_weight in enumerate(boundary_weights):
                config_name = f"B{b_weight}_S{s_weight}"
                if config_name in results:
                    mae_matrix[i, j] = results[config_name]['mae']
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(mae_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='MAE')
        plt.xlabel('Boundary Weight')
        plt.ylabel('Smoothness Weight')
        plt.title('Hyperparameter Sensitivity Analysis (MAE)')
        
        # 设置刻度标签
        plt.xticks(range(len(boundary_weights)), boundary_weights)
        plt.yticks(range(len(smoothness_weights)), smoothness_weights)
        
        # 在每个格子中显示数值
        for i in range(len(smoothness_weights)):
            for j in range(len(boundary_weights)):
                plt.text(j, i, f'{mae_matrix[i, j]:.4f}', 
                        ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'sensitivity_heatmap.png'), dpi=300)
        plt.show()
        
        # 保存敏感性分析结果
        sensitivity_file = os.path.join(self.save_dir, "sensitivity_results.json")
        with open(sensitivity_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"敏感性分析结果已保存到: {sensitivity_file}")
    
    def visualize_physics_ablation_results(self, physics_results):
        """可视化物理约束消融实验结果"""
        if not physics_results:
            print("没有物理约束消融结果可视化")
            return
        
        # 提取数据
        loss_names = []
        maes = []
        rmses = []
        r2s = []
        
        for loss_name, results in physics_results.items():
            if 'eval_metrics' in results:
                loss_names.append(loss_name)
                maes.append(results['eval_metrics']['mae'])
                rmses.append(results['eval_metrics']['rmse'])
                r2s.append(results['eval_metrics']['r2'])
        
        # 创建对比图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MAE对比
        axes[0].bar(loss_names, maes, color='skyblue')
        axes[0].set_title('Mean Absolute Error (MAE)')
        axes[0].set_ylabel('MAE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE对比
        axes[1].bar(loss_names, rmses, color='lightcoral')
        axes[1].set_title('Root Mean Square Error (RMSE)')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R²对比
        axes[2].bar(loss_names, r2s, color='lightgreen')
        axes[2].set_title('Coefficient of Determination (R²)')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'physics_ablation_comparison.png'), dpi=300)
        plt.show()