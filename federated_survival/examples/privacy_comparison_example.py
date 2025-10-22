"""
差分隐私效果对比示例

本示例对比有无差分隐私保护下的联邦学习性能差异。
"""
import matplotlib.pyplot as plt
import numpy as np
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter

def run_experiment(config, data, experiment_name):
    """运行实验并返回结果"""
    print(f"\n=== {experiment_name} ===")
    runner = FSARunner(config)
    
    # 获取隐私信息
    privacy_info = runner.get_privacy_info()
    if privacy_info["privacy_protection"]:
        print("差分隐私保护已启用（仅在客户端本地训练时应用）")
        print(f"隐私预算 (ε): {privacy_info['epsilon']}")
        print(f"噪声规模: {privacy_info['noise_scale']:.6f}")
    else:
        print("差分隐私保护未启用")
    
    # 运行训练
    results = runner.run(data, type='raw')
    
    print(f"最终性能:")
    print(f"  训练 C-index: {results['train_Cindex'][-1]:.4f}")
    print(f"  测试 C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"  训练 IBS: {results['train_IBS'][-1]:.4f}")
    print(f"  测试 IBS: {results['test_IBS'][-1]:.4f}")
    
    return results

def plot_comparison(results_no_dp, results_with_dp):
    """绘制对比结果"""
    # 使用更现代的样式，避免已弃用的seaborn样式
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('default')
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # 关闭所有已存在的图形窗口
    plt.close('all')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = list(range(1, len(results_no_dp['train_Cindex']) + 1))
    
    # C-index Comparison
    ax1.plot(epochs, results_no_dp['train_Cindex'], 'b-', label='No DP (Training)', linewidth=2)
    ax1.plot(epochs, results_no_dp['test_Cindex'], 'b--', label='No DP (Testing)', linewidth=2)
    ax1.plot(epochs, results_with_dp['train_Cindex'], 'r-', label='With DP (Training)', linewidth=2)
    ax1.plot(epochs, results_with_dp['test_Cindex'], 'r--', label='With DP (Testing)', linewidth=2)
    ax1.set_title('C-index Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('C-index')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IBS Comparison
    ax2.plot(epochs, results_no_dp['train_IBS'], 'b-', label='No DP (Training)', linewidth=2)
    ax2.plot(epochs, results_no_dp['test_IBS'], 'b--', label='No DP (Testing)', linewidth=2)
    ax2.plot(epochs, results_with_dp['train_IBS'], 'r-', label='With DP (Training)', linewidth=2)
    ax2.plot(epochs, results_with_dp['test_IBS'], 'r--', label='With DP (Testing)', linewidth=2)
    ax2.set_title('IBS Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IBS')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Performance Difference
    cindex_diff = np.array(results_with_dp['test_Cindex']) - np.array(results_no_dp['test_Cindex'])
    ibs_diff = np.array(results_with_dp['test_IBS']) - np.array(results_no_dp['test_IBS'])
    
    ax3.plot(epochs, cindex_diff, 'g-', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title('C-index Difference (With DP - No DP)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('C-index Difference')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(epochs, ibs_diff, 'orange', linewidth=2, marker='s', markersize=4)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title('IBS Difference (With DP - No DP)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('IBS Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数：对比有无差分隐私的效果"""
    print("=== 差分隐私效果对比实验 ===\n")
    
    # 基础配置
    base_config = {
        'n_samples': 1000,
        'n_features': 20,
        'num_clients': 5,
        'global_epochs': 30,
        'verbose': True,
    }
    
    # 生成数据
    print("生成模拟数据...")
    temp_config = FSAConfig(**base_config)
    sim_config = SimulationConfig(
        n_samples=temp_config.n_samples,
        n_features=temp_config.n_features,
        censoring_rate=temp_config.censor_rate,
        random_state=temp_config.random_seed
    )
    generator = DataGenerator(sim_config)
    raw_data = generator.generate('weibull')
    
    # 分割数据为联邦学习格式
    splitter = DataSplitter(
        n_clients=temp_config.num_clients,
        split_type='iid',
        test_size=0.2,
        random_state=temp_config.random_seed
    )
    data = splitter.split(raw_data)
    print(f"数据生成完成，客户端数量: {len(data.clients_set)}")
    
    # 实验1：无差分隐私
    config_no_dp = FSAConfig(**base_config, use_differential_privacy=False)
    results_no_dp = run_experiment(config_no_dp, data, "无差分隐私")
    
    # 实验2：有差分隐私
    config_with_dp = FSAConfig(
        **base_config,
        use_differential_privacy=True,
        dp_epsilon=0.1,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=1.0,
        dp_clip_norm=1.0,
    )
    results_with_dp = run_experiment(config_with_dp, data, "有差分隐私")
    
    # 绘制对比结果
    print("\n绘制对比结果...")
    plot_comparison(results_no_dp, results_with_dp)
    
    # 输出最终对比
    print("\n=== 最终性能对比 ===")
    print(f"{'指标':<15} {'无差分隐私':<12} {'有差分隐私':<12} {'差异':<12}")
    print("-" * 55)
    
    train_cindex_diff = results_with_dp['train_Cindex'][-1] - results_no_dp['train_Cindex'][-1]
    test_cindex_diff = results_with_dp['test_Cindex'][-1] - results_no_dp['test_Cindex'][-1]
    train_ibs_diff = results_with_dp['train_IBS'][-1] - results_no_dp['train_IBS'][-1]
    test_ibs_diff = results_with_dp['test_IBS'][-1] - results_no_dp['test_IBS'][-1]
    
    print(f"{'训练 C-index':<15} {results_no_dp['train_Cindex'][-1]:<12.4f} {results_with_dp['train_Cindex'][-1]:<12.4f} {train_cindex_diff:+.4f}")
    print(f"{'测试 C-index':<15} {results_no_dp['test_Cindex'][-1]:<12.4f} {results_with_dp['test_Cindex'][-1]:<12.4f} {test_cindex_diff:+.4f}")
    print(f"{'训练 IBS':<15} {results_no_dp['train_IBS'][-1]:<12.4f} {results_with_dp['train_IBS'][-1]:<12.4f} {train_ibs_diff:+.4f}")
    print(f"{'测试 IBS':<15} {results_no_dp['test_IBS'][-1]:<12.4f} {results_with_dp['test_IBS'][-1]:<12.4f} {test_ibs_diff:+.4f}")
    
    print("\n实验完成！")

if __name__ == "__main__":
    main()
