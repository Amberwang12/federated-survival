"""
历史客户端更新扰动示例（不提供端到端差分隐私保证）

本示例仅展示三类遗留噪声/选择工具的调用方式。当前训练路径没有逐样本裁剪、
隐私放大或跨轮 accountant，不能把 epsilon/delta 参数报告为 record-level DP 保证：
1. Gaussian Mechanism (高斯机制) - 适用于深度学习梯度保护
2. Laplace Mechanism (拉普拉斯机制) - 适用于计数查询
3. Exponential Mechanism (指数机制) - 适用于模型选择
"""
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
import torch
import numpy as np


def generate_data(n_samples=1000, n_features=20, num_clients=5):
    """生成联邦学习数据
    
    Args:
        n_samples: 样本数
        n_features: 特征数
        num_clients: 客户端数量
        
    Returns:
        分割后的联邦学习数据
    """
    sim_config = SimulationConfig(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42
    )
    generator = DataGenerator(sim_config)
    raw_data = generator.generate('weibull', c_mean=0.4)
    
    splitter = DataSplitter(
        n_clients=num_clients,
        split_type='iid',
        test_size=0.2,
        random_state=42
    )
    return splitter.split(raw_data)


def demonstrate_gaussian_mechanism():
    """演示高斯机制（Gaussian Mechanism）
    
    演示高斯噪声工具；当前联邦训练路径不据此提供 (ε, δ)-DP 保证。
    适用于深度学习梯度保护，通过添加高斯噪声保护模型参数。
    """
    print("\n" + "="*60)
    print("1. 高斯机制 (Gaussian Mechanism)")
    print("="*60)
    print("特点: (ε, δ)-差分隐私, 正态分布噪声")
    print("适用: 深度学习梯度保护\n")
    
    # 创建配置
    config = FSAConfig(
        n_samples=1000,
        n_features=20,
        num_clients=5,
        global_epochs=15,
        verbose=False,
        use_differential_privacy=True,
        dp_mechanism='gaussian',      # 高斯机制
        dp_epsilon=1.0,
        dp_delta=1e-5,                # 高斯机制需要 delta
        dp_noise_multiplier=1.0,      # 高斯噪声乘数
        dp_clip_norm=1.0,
    )
    
    # 生成数据
    print("生成数据...")
    data = generate_data(config.n_samples, config.n_features, config.num_clients)
    
    # 运行训练
    print("开始训练...")
    runner = FSARunner(config)
    results = runner.run(data, type='raw')
    
    # 显示隐私信息
    privacy_info = runner.get_privacy_info()
    print("\n隐私保护信息:")
    print(f"  机制: {privacy_info['mechanism']}")
    print(f"  隐私预算 (ε): {privacy_info['epsilon']}")
    print(f"  失败概率 (δ): {privacy_info['delta']}")
    print(f"  噪声乘数: {privacy_info['noise_multiplier']}")
    print(f"  梯度裁剪范数: {privacy_info['clip_norm']}")
    
    # 显示结果
    print("\n训练结果:")
    print(f"  最终训练 C-index: {results['train_Cindex'][-1]:.4f}")
    print(f"  最终测试 C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"  最终训练 IBS: {results['train_IBS'][-1]:.4f}")
    print(f"  最终测试 IBS: {results['test_IBS'][-1]:.4f}")
    
    return results


def demonstrate_laplace_mechanism():
    """演示拉普拉斯机制（Laplace Mechanism）
    
    演示拉普拉斯噪声工具；当前联邦训练路径不据此提供纯 ε-DP 保证。
    适用于计数查询和求和查询，噪声服从拉普拉斯分布。
    """
    print("\n" + "="*60)
    print("2. 拉普拉斯机制 (Laplace Mechanism)")
    print("="*60)
    print("特点: ε-差分隐私, 拉普拉斯分布噪声")
    print("适用: 计数查询, 求和查询\n")
    
    # 创建配置
    config = FSAConfig(
        n_samples=1000,
        n_features=20,
        num_clients=5,
        global_epochs=15,
        verbose=False,
        use_differential_privacy=True,
        dp_mechanism='laplace',       # 拉普拉斯机制
        dp_epsilon=1.0,
        # 注意: 拉普拉斯机制不需要 delta 和 noise_multiplier
        dp_clip_norm=1.0,
    )
    
    # 生成数据
    print("生成数据...")
    data = generate_data(config.n_samples, config.n_features, config.num_clients)
    
    # 运行训练
    print("开始训练...")
    runner = FSARunner(config)
    results = runner.run(data, type='raw')
    
    # 显示隐私信息
    privacy_info = runner.get_privacy_info()
    print("\n隐私保护信息:")
    print(f"  机制: {privacy_info['mechanism']}")
    print(f"  隐私预算 (ε): {privacy_info['epsilon']}")
    print(f"  梯度裁剪范数: {privacy_info['clip_norm']}")
    print(f"  注意: 拉普拉斯机制提供纯 ε-DP，无需 delta 参数")
    
    # 显示结果
    print("\n训练结果:")
    print(f"  最终训练 C-index: {results['train_Cindex'][-1]:.4f}")
    print(f"  最终测试 C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"  最终训练 IBS: {results['train_IBS'][-1]:.4f}")
    print(f"  最终测试 IBS: {results['test_IBS'][-1]:.4f}")
    
    return results


def demonstrate_exponential_mechanism():
    """演示指数机制（Exponential Mechanism）
    
    演示离散候选选择；只有另行证明质量函数敏感度时才可讨论其 DP 保证。
    常用于模型选择、超参数选择等离散优化问题。
    """
    print("\n" + "="*60)
    print("3. 指数机制 (Exponential Mechanism)")
    print("="*60)
    print("特点: ε-差分隐私, 概率采样")
    print("适用: 模型选择, 超参数选择, 离散优化\n")
    
    # 创建配置用于初始化差分隐私工具
    config = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism='exponential',
        dp_epsilon=2.0,
        dp_sensitivity=1.0,
    )
    
    # 创建差分隐私工具
    dp_tool = DifferentialPrivacy(config)
    
    # 定义候选模型名称和质量得分
    model_names = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E']
    quality_scores = torch.tensor([0.75, 0.82, 0.68, 0.79, 0.85])  # 模型质量得分
    
    # 创建候选项张量（这里使用简单的索引张量）
    candidates = torch.arange(len(model_names))
    
    print("候选模型和质量得分:")
    for i, (model, score) in enumerate(zip(model_names, quality_scores)):
        print(f"  {model}: {score:.4f}")
    print()
    
    # 使用指数机制选择模型（多次采样以观察概率分布）
    print("使用指数机制选择模型 (100次采样)...")
    n_trials = 100
    selection_counts = {model: 0 for model in model_names}
    
    for _ in range(n_trials):
        selected_idx = dp_tool.exponential_mechanism(
            candidates=candidates,
            quality_scores=quality_scores
        )
        selection_counts[model_names[selected_idx]] += 1
    
    # 显示选择统计
    print("\n选择统计 (基于隐私预算 ε=2.0):")
    for model in model_names:
        percentage = (selection_counts[model] / n_trials) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {model}: {selection_counts[model]:3d} 次 ({percentage:5.1f}%) {bar}")
    
    # 理论分析
    print("\n理论分析:")
    print("  指数机制选择概率与质量得分成指数关系")
    print(f"  最高得分模型 (Model_E: {quality_scores[4]:.4f}) 被选中概率最高")
    print(f"  实际选中次数: {selection_counts['Model_E']} 次")
    print("  说明: 当前示例未建立端到端隐私保证")
    
    return selection_counts


def main():
    """主函数：演示三种差分隐私机制"""
    print("\n" + "#"*60)
    print("# 差分隐私联邦学习生存分析 - 三种机制演示")
    print("#"*60)
    print("\n本示例演示三种差分隐私机制在联邦学习中的应用:")
    print("  1. Gaussian Mechanism (高斯机制)")
    print("  2. Laplace Mechanism (拉普拉斯机制)")
    print("  3. Exponential Mechanism (指数机制)")
    print("\n这些工具有不同实验用途，但当前训练流程没有端到端隐私会计。")
    
    # 演示三种机制
    results_gaussian = demonstrate_gaussian_mechanism()
    results_laplace = demonstrate_laplace_mechanism()
    selection_counts = demonstrate_exponential_mechanism()
    
    # 总结对比
    print("\n" + "="*60)
    print("总结对比")
    print("="*60)
    print("\n机制特性对比:")
    print("+" + "-"*18 + "+" + "-"*18 + "+" + "-"*20 + "+")
    print("| {:^16} | {:^16} | {:^18} |".format("工具", "当前软件保证", "实验用途"))
    print("+" + "-"*18 + "+" + "-"*18 + "+" + "-"*20 + "+")
    print("| {:^16} | {:^16} | {:^18} |".format("Gaussian", "(ε, δ)-DP", "深度学习梯度"))
    print("| {:^16} | {:^16} | {:^18} |".format("Laplace", "ε-DP", "计数/求和查询"))
    print("| {:^16} | {:^16} | {:^18} |".format("Exponential", "ε-DP", "模型选择"))
    print("+" + "-"*18 + "+" + "-"*18 + "+" + "-"*20 + "+")
    
    print("\n性能对比 (基于本次运行):")
    print(f"  高斯机制 - 测试 C-index: {results_gaussian['test_Cindex'][-1]:.4f}")
    print(f"  拉普拉斯机制 - 测试 C-index: {results_laplace['test_Cindex'][-1]:.4f}")
    print(f"  指数机制 - 模型选择: {selection_counts['Model_E']} 次")
    
    print("\n使用建议:")
    print("  • 深度学习模型训练 → 选择 Gaussian Mechanism")
    print("  • 简单统计查询 → 选择 Laplace Mechanism")
    print("  • 模型或参数选择 → 选择 Exponential Mechanism")
    
    print("\n" + "#"*60)
    print("示例运行完成！")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
