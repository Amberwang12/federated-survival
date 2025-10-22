# Differential Privacy Mechanisms Guide
# 差分隐私机制使用指南

## Overview | 概览

本项目现在支持三种主要的差分隐私机制:

1. **高斯机制 (Gaussian Mechanism)**
2. **拉普拉斯机制 (Laplace Mechanism)** 
3. **指数机制 (Exponential Mechanism)**

---

## 1. Gaussian Mechanism | 高斯机制

### 数学原理

对于查询函数 `f: D → ℝᵈ`，高斯机制定义为:

```
M(D) = f(D) + N(0, σ²I)
```

其中噪声标准差为:
```
σ = √(2·ln(1.25/δ)) × Δf / ε
```

### 隐私保证
- **(ε, δ)-差分隐私**
- 需要两个参数: `epsilon (ε)` 和 `delta (δ)`

### 适用场景
✅ **推荐使用**:
- 深度学习梯度扰动
- 机器学习模型训练
- 需要组合多次查询的场景

❌ **不推荐**:
- 需要纯ε-DP保证的场景
- 计数查询 (Laplace更优)

### 代码示例

```python
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.core.config import FSAConfig
import torch

# 配置
config = FSAConfig(
    n_features=10,
    num_clients=5,
    use_differential_privacy=True,
    dp_epsilon=1.0,
    dp_delta=1e-5,  # 高斯机制需要delta
    dp_sensitivity=1.0
)

dp = DifferentialPrivacy(config)

# 添加高斯噪声
gradient = torch.randn(100)
noisy_gradient = dp.add_gaussian_noise(gradient)
```

### 特点
- ✓ 噪声分布: 正态分布 N(0, σ²)
- ✓ 组合性质好 (适合多轮训练)
- ✓ 与矩量计数法兼容
- ✗ 需要额外的δ参数

---

## 2. Laplace Mechanism | 拉普拉斯机制

### 数学原理

对于查询函数 `f: D → ℝᵈ`，拉普拉斯机制定义为:

```
M(D) = f(D) + Lap(b)
```

其中尺度参数为:
```
b = Δf / ε
```

### 隐私保证
- **ε-差分隐私**
- 只需一个参数: `epsilon (ε)`
- 纯差分隐私 (不需要δ)

### 适用场景
✅ **推荐使用**:
- 计数查询 (counting queries)
- 求和查询 (sum queries)
- 需要纯ε-DP的场景
- 低维数值查询

❌ **不推荐**:
- 高维梯度扰动 (高斯更优)
- 深度学习场景

### 代码示例

```python
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.core.config import FSAConfig
import torch

# 配置
config = FSAConfig(
    n_features=10,
    num_clients=5,
    use_differential_privacy=True,
    dp_epsilon=1.0,
    dp_sensitivity=1.0
)

dp = DifferentialPrivacy(config)

# 计数查询示例
true_count = torch.tensor([1000.0])  # 真实计数
noisy_count = dp.add_laplace_noise(true_count, sensitivity=1.0, epsilon=1.0)
print(f"Noisy count: {int(noisy_count.item())}")
```

### 特点
- ✓ 噪声分布: 拉普拉斯分布 Lap(b)
- ✓ 纯ε-DP保证 (无需δ)
- ✓ 适合计数/求和查询
- ✗ 高维数据噪声较大

---

## 3. Exponential Mechanism | 指数机制

### 数学原理

对于输出空间 `R` 和质量函数 `q: D × R → ℝ`，指数机制以概率:

```
P(r) ∝ exp(ε · q(D,r) / (2·Δq))
```

选择输出 `r ∈ R`。

### 隐私保证
- **ε-差分隐私**
- 只需一个参数: `epsilon (ε)`

### 适用场景
✅ **推荐使用**:
- 模型选择 (model selection)
- 超参数调优
- 离散选择问题
- 非数值输出场景

❌ **不推荐**:
- 连续数值输出
- 需要精确数值的场景

### 代码示例

```python
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.core.config import FSAConfig
import torch

# 配置
config = FSAConfig(
    n_features=10,
    num_clients=5,
    use_differential_privacy=True,
    dp_epsilon=1.0,
    dp_sensitivity=1.0
)

dp = DifferentialPrivacy(config)

# 模型选择示例
candidates = torch.randn(5, 100)  # 5个候选模型配置
quality_scores = torch.tensor([0.80, 0.85, 0.90, 0.82, 0.88])  # 验证准确率

# 使用指数机制选择
selected_idx = dp.exponential_mechanism(candidates, quality_scores, epsilon=1.0)
print(f"Selected model: {selected_idx} (score: {quality_scores[selected_idx]:.4f})")

# 或直接获取选中的候选项
selected_model = dp.exponential_mechanism_tensor(candidates, quality_scores, epsilon=1.0)
```

### 特点
- ✓ 适合离散选择
- ✓ 不添加噪声，通过概率采样
- ✓ 保持输出格式
- ✗ 只适用于有限候选集

---

## Comparison Table | 对比表格

| 特性 | 高斯机制 | 拉普拉斯机制 | 指数机制 |
|------|---------|-------------|---------|
| **隐私保证** | (ε, δ)-DP | ε-DP | ε-DP |
| **噪声类型** | 正态分布 | 拉普拉斯分布 | 概率采样 |
| **输出类型** | 连续数值 | 连续数值 | 离散选择 |
| **最佳场景** | 深度学习 | 计数查询 | 模型选择 |
| **维度敏感性** | 适合高维 | 适合低维 | 与维度无关 |
| **组合性质** | 优秀 (RDP) | 一般 | 一般 |

---

## Privacy-Utility Tradeoff | 隐私-效用权衡

### Epsilon (ε) 参数的影响

| ε 值 | 隐私级别 | 噪声大小 | 数据效用 | 推荐场景 |
|------|---------|---------|---------|---------|
| 0.1  | 极高 | 极大 | 极低 | 极敏感数据 |
| 0.5  | 高 | 大 | 低 | 医疗数据 |
| 1.0  | 中等 | 中等 | 中等 | 一般推荐 ⭐ |
| 2.0  | 较低 | 较小 | 较高 | 商业数据 |
| 5.0+ | 低 | 小 | 高 | 公开数据 |

### Delta (δ) 参数 (仅高斯机制)

通常设置为 `δ = 1/n²`，其中 `n` 是数据集大小。

- 典型值: `1e-5` 到 `1e-7`
- δ 越小，隐私保护越强，但噪声也越大

---

## Usage in Federated Learning | 在联邦学习中的应用

### 场景1: 梯度扰动 (推荐高斯机制)

```python
# 在客户端训练中应用DP
class Client:
    def local_train(self, global_model, epoch):
        # ... 训练代码 ...
        
        # 应用高斯机制到梯度
        if self.dp_tool is not None:
            self.dp_tool.apply_dp_to_gradients(
                model=local_model.net, 
                optimizer=optimizer,
                mechanism='gaussian'  # 使用高斯机制
            )
        
        return local_model.net
```

### 场景2: 梯度扰动 (拉普拉斯机制)

```python
# 使用拉普拉斯机制
if self.dp_tool is not None:
    self.dp_tool.apply_dp_to_gradients(
        model=local_model.net, 
        optimizer=optimizer,
        mechanism='laplace'  # 使用拉普拉斯机制
    )
```

### 场景3: 模型选择 (指数机制)

```python
# 在服务器端选择最优客户端模型
quality_scores = torch.tensor([
    client_1_accuracy,
    client_2_accuracy,
    # ...
])

selected_client = dp_tool.exponential_mechanism(
    client_models, 
    quality_scores,
    epsilon=1.0
)
```

---

## Example Scripts | 示例脚本

### 运行对比示例

```bash
python federated_survival/examples/dp_mechanisms_comparison_example.py
```

该示例将:
1. 演示三种机制的使用
2. 可视化噪声分布
3. 对比隐私-效用权衡
4. 展示实际应用场景

### 输出文件
- `dp_mechanisms_noise_distributions.png` - 噪声分布可视化
- `dp_mechanisms_privacy_utility_tradeoff.png` - 隐私-效用权衡曲线

---

## Best Practices | 最佳实践

### 1. 选择合适的机制

```python
# 决策树
if output_type == "continuous":
    if scenario == "deep_learning":
        mechanism = "gaussian"  # 高斯机制
    elif scenario == "counting":
        mechanism = "laplace"   # 拉普拉斯机制
elif output_type == "discrete":
    mechanism = "exponential"   # 指数机制
```

### 2. 设置合理的隐私预算

```python
# 推荐配置
config = FSAConfig(
    dp_epsilon=1.0,           # 中等隐私级别
    dp_delta=1e-5,            # 高斯机制需要
    dp_sensitivity=1.0,       # 根据实际场景调整
    dp_clip_norm=1.0,         # 梯度裁剪
)
```

### 3. 监控隐私预算消耗

```python
# 计算总隐私预算
total_eps, per_round_eps = dp_tool.compute_privacy_budget(
    num_rounds=100,
    num_clients=10
)
print(f"Total privacy budget: ε={total_eps}")
```

---

## References | 参考文献

1. **高斯机制**:
   - Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy.

2. **拉普拉斯机制**:
   - Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating Noise to Sensitivity in Private Data Analysis.

3. **指数机制**:
   - McSherry, F., & Talwar, K. (2007). Mechanism Design via Differential Privacy.

4. **联邦学习中的差分隐私**:
   - Abadi, M., et al. (2016). Deep Learning with Differential Privacy.
   - McMahan, B., et al. (2017). Learning Differentially Private Recurrent Language Models.

---

## Support | 支持

如有问题或建议，请:
- 查看项目README
- 运行示例脚本
- 参考单元测试

**Last Updated**: 2025-10-19
