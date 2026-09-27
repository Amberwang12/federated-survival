import numpy as np
import matplotlib.pyplot as plt
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner

def test_plot(tmp_path):
    """测试绘图功能"""
    # 创建模拟数据
    n_epochs = 50
    np.random.seed(42)  # 设置随机种子以保证结果可重现
    
    # 生成训练集C-index：从0.6开始，逐渐提高到0.85，加入一些随机波动
    x = np.linspace(0, 5, n_epochs)
    train_cindex = 0.6 + 0.25 * (1 - np.exp(-x)) + np.random.normal(0, 0.02, n_epochs)
    train_cindex = np.clip(train_cindex, 0.5, 1.0)  # 确保在[0.5, 1.0]范围内
    
    # 生成测试集C-index：略低于训练集，有更大的波动
    test_cindex = train_cindex - 0.05 + np.random.normal(0, 0.03, n_epochs)
    test_cindex = np.clip(test_cindex, 0.5, 1.0)
    
    # 生成训练集IBS：从0.3开始，逐渐下降到0.15
    train_ibs = 0.3 * np.exp(-x/2) + 0.15 + np.random.normal(0, 0.01, n_epochs)
    train_ibs = np.clip(train_ibs, 0, 1.0)  # 确保在[0, 1.0]范围内
    
    # 生成测试集IBS：略高于训练集，有更大的波动
    test_ibs = train_ibs + 0.05 + np.random.normal(0, 0.02, n_epochs)
    test_ibs = np.clip(test_ibs, 0, 1.0)
    
    # 转换为Python列表并确保是float类型
    results = {
        'train_Cindex': [float(x) for x in train_cindex],
        'test_Cindex': [float(x) for x in test_cindex],
        'train_IBS': [float(x) for x in train_ibs],
        'test_IBS': [float(x) for x in test_ibs]
    }
    
    # 打印数据验证
    print("数据验证:")
    print(f"训练集C-index范围: [{min(results['train_Cindex']):.3f}, {max(results['train_Cindex']):.3f}]")
    print(f"测试集C-index范围: [{min(results['test_Cindex']):.3f}, {max(results['test_Cindex']):.3f}]")
    print(f"训练集IBS范围: [{min(results['train_IBS']):.3f}, {max(results['train_IBS']):.3f}]")
    print(f"测试集IBS范围: [{min(results['test_IBS']):.3f}, {max(results['test_IBS']):.3f}]")
    print(f"数据长度: {len(results['train_Cindex'])}")
    
    # 创建runner并绘图
    config = FSAConfig()
    runner = FSARunner(config)
    
    # 调用绘图函数
    output = tmp_path / 'training_metrics.png'
    figure = runner.plot_results(results, output_path=output, show=False)
    assert output.exists()
    assert len(figure.axes) == 2
    plt.close(figure)

