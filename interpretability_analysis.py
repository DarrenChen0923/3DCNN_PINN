import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class InterpretabilityAnalysis:
    def __init__(self, model, data_loaders, device, save_dir):
        self.model = model
        self.data_loaders = data_loaders
        self.device = device
        self.save_dir = save_dir
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("=== 开始特征重要性分析 ===")
        
        # 收集测试数据
        X_test = []
        y_test = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.data_loaders['test_loader']:
                point_series = batch['point_series'].to(self.device)
                error = batch['error'].to(self.device)
                
                X_test.append(point_series.cpu().numpy())
                y_test.append(error.cpu().numpy())
        
        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)
        
        # 重塑为2D数组以便分析
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # 定义预测函数
        # 继续 interpretability_analysis.py
        def predict_function(X):
            X_tensor = torch.FloatTensor(X.reshape(-1, 3, 3, 1)).to(self.device)
            self.model.eval()
            with torch.no_grad():
                predictions, _, _ = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
            
            # 计算置换重要性
        print("计算置换重要性...")
        perm_importance = permutation_importance(
           predict_function, X_test_flat, y_test,
           n_repeats=10, random_state=42, n_jobs=1
        )
            
         # 可视化特征重要性
        self._visualize_feature_importance(perm_importance)
            
        return perm_importance
   
    def _visualize_feature_importance(self, perm_importance):
            """可视化特征重要性"""
            # 创建特征名称（对应3x3网格的位置）
            feature_names = []
            for i in range(3):
                for j in range(3):
                    feature_names.append(f'Position_{i}_{j}')
            
            # 排序特征重要性
            sorted_idx = perm_importance.importances_mean.argsort()[::-1]
            
            # 创建柱状图
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(sorted_idx)), 
                    perm_importance.importances_mean[sorted_idx],
                    xerr=perm_importance.importances_std[sorted_idx])
            
            plt.yticks(range(len(sorted_idx)), 
                        [feature_names[i] for i in sorted_idx])
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance Analysis')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'feature_importance.png'), dpi=300)
            plt.show()
            
            # 创建3x3热力图
            importance_matrix = perm_importance.importances_mean.reshape(3, 3)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
            plt.colorbar(label='Importance')
            plt.title('Feature Importance Heatmap (3x3 Grid)')
            
            # 添加数值标注
            for i in range(3):
                for j in range(3):
                    plt.text(j, i, f'{importance_matrix[i, j]:.4f}', 
                            ha='center', va='center', fontsize=12, fontweight='bold')
            
            plt.xticks(range(3), ['Col 0', 'Col 1', 'Col 2'])
            plt.yticks(range(3), ['Row 0', 'Row 1', 'Row 2'])
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'importance_heatmap.png'), dpi=300)
            plt.show()
        
    def analyze_attention_weights(self):
            """分析融合层的注意力权重"""
            print("=== 开始注意力权重分析 ===")
            
            # 收集CNN和PINN分支的预测以及最终融合结果
            cnn_predictions = []
            pinn_predictions = []
            final_predictions = []
            targets = []
            
            self.model.eval()
            with torch.no_grad():
                for batch in self.data_loaders['test_loader']:
                    point_series = batch['point_series'].to(self.device)
                    error = batch['error'].to(self.device)
                    
                    final_pred, cnn_pred, pinn_pred = self.model(point_series)
                    
                    cnn_predictions.append(cnn_pred.cpu().numpy())
                    pinn_predictions.append(pinn_pred.cpu().numpy())
                    final_predictions.append(final_pred.cpu().numpy())
                    targets.append(error.cpu().numpy())
            
            # 合并数据
            cnn_pred = np.concatenate(cnn_predictions).flatten()
            pinn_pred = np.concatenate(pinn_predictions).flatten()
            final_pred = np.concatenate(final_predictions).flatten()
            target = np.concatenate(targets).flatten()
            
            # 分析融合权重
            self._analyze_fusion_weights(cnn_pred, pinn_pred, final_pred, target)
            
            return {
                'cnn_predictions': cnn_pred,
                'pinn_predictions': pinn_pred,
                'final_predictions': final_pred,
                'targets': target
            }
        
    def _analyze_fusion_weights(self, cnn_pred, pinn_pred, final_pred, target):
            """分析融合权重"""
            # 通过线性回归估计融合权重
            from sklearn.linear_model import LinearRegression
            
            X = np.column_stack([cnn_pred, pinn_pred])
            reg = LinearRegression(fit_intercept=False).fit(X, final_pred)
            
            cnn_weight = reg.coef_[0]
            pinn_weight = reg.coef_[1]
            
            print(f"估计的融合权重:")
            print(f"CNN分支权重: {cnn_weight:.4f}")
            print(f"PINN分支权重: {pinn_weight:.4f}")
            print(f"权重和: {cnn_weight + pinn_weight:.4f}")
            
            # 计算各分支的性能
            cnn_mae = np.mean(np.abs(cnn_pred - target))
            pinn_mae = np.mean(np.abs(pinn_pred - target))
            final_mae = np.mean(np.abs(final_pred - target))
            
            print(f"\n各分支性能对比:")
            print(f"CNN分支 MAE: {cnn_mae:.6f}")
            print(f"PINN分支 MAE: {pinn_mae:.6f}")
            print(f"融合后 MAE: {final_mae:.6f}")
            
            # 可视化预测对比
            self._visualize_branch_comparison(cnn_pred, pinn_pred, final_pred, target)
        
    def _visualize_branch_comparison(self, cnn_pred, pinn_pred, final_pred, target):
            """可视化各分支预测对比"""
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # CNN预测 vs 真实值
            axes[0,0].scatter(target, cnn_pred, alpha=0.6, s=10)
            axes[0,0].plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
            axes[0,0].set_xlabel('True Values')
            axes[0,0].set_ylabel('CNN Predictions')
            axes[0,0].set_title('CNN Branch Performance')
            axes[0,0].grid(True, alpha=0.3)
            
            # PINN预测 vs 真实值
            axes[0,1].scatter(target, pinn_pred, alpha=0.6, s=10, color='orange')
            axes[0,1].plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
            axes[0,1].set_xlabel('True Values')
            axes[0,1].set_ylabel('PINN Predictions')
            axes[0,1].set_title('PINN Branch Performance')
            axes[0,1].grid(True, alpha=0.3)
            
            # 融合预测 vs 真实值
            axes[1,0].scatter(target, final_pred, alpha=0.6, s=10, color='green')
            axes[1,0].plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
            axes[1,0].set_xlabel('True Values')
            axes[1,0].set_ylabel('Final Predictions')
            axes[1,0].set_title('Fusion Model Performance')
            axes[1,0].grid(True, alpha=0.3)
            
            # 残差分析
            residuals = final_pred - target
            axes[1,1].hist(residuals, bins=50, alpha=0.7, color='purple')
            axes[1,1].set_xlabel('Residuals')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Residual Distribution')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'branch_comparison.png'), dpi=300)
            plt.show()