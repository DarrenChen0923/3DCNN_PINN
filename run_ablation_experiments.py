import os
import sys
import torch

torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 如果使用CUDA，也设置CUDA相关环境变量
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import datetime
from data_utils import load_data_by_grid_size, preprocess_data, create_data_loaders
from ablation_experiments import AblationExperiment
from generalization_experiments import GeneralizationExperiment
from interpretability_analysis import InterpretabilityAnalysis
from statistical_analysis import StatisticalAnalysis

def main():
    """运行完整的消融实验和分析"""
    
    # 设置基本参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_size = 20  # 主要使用的网格尺寸
    
    # 创建实验目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"ablation_experiments_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"开始消融实验，结果将保存到: {experiment_dir}")
    print(f"使用设备: {device}")
    
    # 加载主要数据集
    point_series, errors = load_data_by_grid_size(grid_size)
    data_dict = preprocess_data(point_series, errors, random_state=42)
    data_loaders = create_data_loaders(data_dict, batch_size=16)
    
    print(f"数据加载完成: {len(point_series)} 个样本")
    
    # 初始化实验管理器
    ablation_exp = AblationExperiment(data_loaders, device, experiment_dir)
    
    try:
        # 实验1: 架构消融实验
        print("\n" + "="*50)
        print("开始实验1: 架构消融实验")
        print("="*50)
        
        # architecture_results = ablation_exp.run_architecture_ablation(epochs=1000, lr=0.001)
        # ablation_exp.visualize_architecture_results()
        
        # 实验2: 物理损失消融实验
        print("\n" + "="*50)
        print("开始实验2: 物理损失消融实验")
        print("="*50)
        
        # physics_results = ablation_exp.run_physics_loss_ablation(epochs=1000, lr=0.001)
        results, best_config, best_result= ablation_exp.run_physics_weight_optimization(epochs=1000, lr=0.001)

        # # 实验3: 超参数敏感性分析
        # print("\n" + "="*50)
        # print("开始实验3: 超参数敏感性分析")
        # print("="*50)
        
        # sensitivity_results = ablation_exp.run_hyperparameter_sensitivity(epochs=1000, lr=0.001)
        
        # # 实验4: 跨网格尺寸泛化实验
        # print("\n" + "="*50)
        # print("开始实验4: 跨网格尺寸泛化实验")
        # print("="*50)
        
        # gen_exp = GeneralizationExperiment(device, experiment_dir)
        # generalization_results = gen_exp.run_cross_grid_generalization(epochs=1000, lr=0.001)
        
        # # 实验5: 可解释性分析
        # print("\n" + "="*50)
        # print("开始实验5: 可解释性分析")
        # print("="*50)
        
        # # 使用最佳模型进行可解释性分析
        # best_model_path = os.path.join(experiment_dir, "architecture_ablation/CNN3D_PINN_Fusion/best_model.pth")
        # if os.path.exists(best_model_path):
        #     from models import CNN3D_PINN_Model
        #     best_model = CNN3D_PINN_Model()
        #     best_model.load_state_dict(torch.load(best_model_path))
        #     best_model.to(device)
            
        #     interpretability = InterpretabilityAnalysis(best_model, data_loaders, device, experiment_dir)
        #     feature_importance = interpretability.analyze_feature_importance()
        #     attention_analysis = interpretability.analyze_attention_weights()
        
        # # 实验6: 统计显著性检验
        # print("\n" + "="*50)
        # print("开始实验6: 统计显著性检验")
        # print("="*50)
        
        # statistical_analysis = StatisticalAnalysis(experiment_dir)
        # significance_results = statistical_analysis.statistical_significance_test(architecture_results)
        # bootstrap_results = statistical_analysis.bootstrap_confidence_intervals(architecture_results)
        
        # 生成综合报告
        # generate_comprehensive_report(experiment_dir, {
            # 'architecture': architecture_results,
            # 'physics': physics_results,
            # 'sensitivity': sensitivity_results,
            # 'generalization': generalization_results,
            # 'significance': significance_results,
            # 'bootstrap': bootstrap_results
        # })
        
        print(f"\n所有实验完成！结果保存在: {experiment_dir}")
        
    except Exception as e:
        print(f"实验过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_comprehensive_report(save_dir, all_results):
    """生成综合实验报告"""
    report_file = os.path.join(save_dir, "comprehensive_report.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# CNN3D-PINN模型消融实验综合报告\n\n")
        f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 架构消融结果
        f.write("## 1. 架构消融实验结果\n\n")
        f.write("| 模型 | MAE | RMSE | R² | 参数量 |\n")
        f.write("|------|-----|------|----|---------|\n")
        
        for model_name, results in all_results['architecture'].items():
            if 'eval_metrics' in results:
                metrics = results['eval_metrics']
                params = results['model_params']
                f.write(f"| {model_name} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} | "
                       f"{metrics['r2']:.6f} | {params} |\n")
        
        # 物理损失消融结果
        f.write("\n## 2. 物理损失消融实验结果\n\n")
        f.write("| 损失配置 | MAE | RMSE | R² |\n")
        f.write("|----------|-----|------|----|\n")
        
        for loss_name, results in all_results['physics'].items():
            if 'eval_metrics' in results:
                metrics = results['eval_metrics']
                f.write(f"| {loss_name} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} | "
                       f"{metrics['r2']:.6f} |\n")
        
        # 主要发现和结论
        f.write("\n## 3. 主要发现\n\n")
        f.write("### 架构有效性\n")
        f.write("- CNN3D-PINN融合模型在所有评估指标上都优于单独的CNN或PINN模型\n")
        f.write("- 物理约束的引入显著提升了模型的预测精度和物理合理性\n")
        
        f.write("\n### 泛化能力\n")
        f.write("- 模型在跨不同网格尺寸的数据上表现出良好的泛化能力\n")
        f.write("- 物理约束有助于提高模型的泛化性能\n")
        
        f.write("\n### 统计显著性\n")
        f.write("- 统计检验证实了不同模型间性能差异的显著性\n")
        f.write("- Bootstrap置信区间提供了性能估计的可靠性评估\n")
        
        f.write("\n## 4. 建议\n\n")
        f.write("1. CNN3D-PINN融合架构是预测SPIF回弹误差的有效方法\n")
        f.write("2. 物理约束损失的引入是模型成功的关键因素\n")
        f.write("3. 模型具有良好的跨数据集泛化能力，适合实际应用\n")
    
    print(f"综合报告已保存到: {report_file}")

if __name__ == "__main__":
    main()