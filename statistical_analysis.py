import os
import numpy as np
from scipy import stats
import pandas as pd

class StatisticalAnalysis:
    def __init__(self, save_dir):
        self.save_dir = save_dir
    
    def statistical_significance_test(self, results_dict):
        """进行统计显著性检验"""
        print("=== 开始统计显著性检验 ===")
        
        # 提取所有模型的预测结果
        model_predictions = {}
        targets = None
        
        for model_name, results in results_dict.items():
            if 'eval_metrics' in results and 'predictions' in results['eval_metrics']:
                model_predictions[model_name] = results['eval_metrics']['predictions'].flatten()
                if targets is None:
                    targets = results['eval_metrics']['targets'].flatten()
        
        # 计算每个模型的绝对误差
        model_errors = {}
        for model_name, predictions in model_predictions.items():
            model_errors[model_name] = np.abs(predictions - targets)
        
        # 进行配对t检验
        significance_results = {}
        model_names = list(model_errors.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # 避免重复比较
                    # 配对t检验
                    t_stat, p_value = stats.ttest_rel(
                        model_errors[model1], 
                        model_errors[model2]
                    )
                    
                    # Wilcoxon符号秩检验（非参数检验）
                    w_stat, w_p_value = stats.wilcoxon(
                        model_errors[model1], 
                        model_errors[model2]
                    )
                    
                    significance_results[f"{model1}_vs_{model2}"] = {
                        'mean_error_1': np.mean(model_errors[model1]),
                        'mean_error_2': np.mean(model_errors[model2]),
                        't_statistic': t_stat,
                        't_p_value': p_value,
                        'wilcoxon_statistic': w_stat,
                        'wilcoxon_p_value': w_p_value,
                        'significant_at_0.05': p_value < 0.05
                    }
        
        # 创建结果报告
        self._create_significance_report(significance_results)
        return significance_results
    
    def _create_significance_report(self, results):
        """创建显著性检验报告"""
        print("\n=== 统计显著性检验结果 ===")
        print(f"{'比较':<30} {'平均误差1':<12} {'平均误差2':<12} {'t统计量':<12} {'p值':<12} {'显著性':<10}")
        print("-" * 100)
        
        for comparison, result in results.items():
            significance = "是" if result['significant_at_0.05'] else "否"
            print(f"{comparison:<30} {result['mean_error_1']:<12.6f} {result['mean_error_2']:<12.6f} "
                  f"{result['t_statistic']:<12.6f} {result['t_p_value']:<12.6f} {significance:<10}")
        
        # 保存详细结果到CSV
        df = pd.DataFrame(results).T
        df.to_csv(os.path.join(self.save_dir, 'significance_test_results.csv'))
        print(f"\n详细结果已保存到: {os.path.join(self.save_dir, 'significance_test_results.csv')}")
    
    def bootstrap_confidence_intervals(self, results_dict, n_bootstrap=1000):
        """计算bootstrap置信区间"""
        print("=== 开始Bootstrap置信区间计算 ===")
        
        bootstrap_results = {}
        
        for model_name, results in results_dict.items():
            if 'eval_metrics' in results and 'predictions' in results['eval_metrics']:
                predictions = results['eval_metrics']['predictions'].flatten()
                targets = results['eval_metrics']['targets'].flatten()
                errors = np.abs(predictions - targets)
                
                # Bootstrap重采样
                bootstrap_maes = []
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(errors, size=len(errors), replace=True)
                    bootstrap_maes.append(np.mean(bootstrap_sample))
                
                # 计算置信区间
                bootstrap_maes = np.array(bootstrap_maes)
                ci_lower = np.percentile(bootstrap_maes, 2.5)
                ci_upper = np.percentile(bootstrap_maes, 97.5)
                
                bootstrap_results[model_name] = {
                    'mean_mae': np.mean(bootstrap_maes),
                    'std_mae': np.std(bootstrap_maes),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }
                
                print(f"{model_name}: MAE = {np.mean(bootstrap_maes):.6f} "
                      f"[{ci_lower:.6f}, {ci_upper:.6f}]")
        
        return bootstrap_results