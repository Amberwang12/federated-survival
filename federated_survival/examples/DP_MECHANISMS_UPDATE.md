# 差分隐私机制更新说明
# Differential Privacy Mechanisms Update Guide

**更新日期**: 2025-10-19  
**版本**: v2.0

---

## 📢 重要更新

项目现已支持**三种**差分隐私机制,您可以通过配置参数自由选择!

### 新增内容

1. ✅ **拉普拉斯机制** (Laplace Mechanism) - 适用于计数查询
2. ✅ **指数机制** (Exponential Mechanism) - 适用于模型选择
3. ✅ **配置参数** `dp_mechanism` - 指定使用的机制类型

---

## 🔧 配置参数更新

### 新增参数

在 [`FSAConfig`](c:\Users\skyee\federated_survival\federated_survival\core\config.py) 中新增:

```python
dp_mechanism: str = 'gaussian'  # 差分隐私机制: 'gaussian', 'laplace', 'exponential'
```

### 完整配置示例

#### 1. 使用高斯机制 (默认)

```python
from federated_survival.core.config import FSAConfig

config = FSAConfig(
    n_features=10,
    num_clients=5,
    global_epochs=50,
    local_epochs=5,
    use_differential_privacy=True,
    dp_mechanism='gaussian',        # 高斯机制
    dp_epsilon=1.0,                 # 隐私预算
    dp_delta=1e-5,                  # 失败概率 (高斯机制需要)
    dp_sensitivity=1.0,             # 敏感度
    dp_noise_multiplier=1.0,        # 噪声乘数 (高斯机制)
    dp_clip_norm=1.0                # 梯度裁剪
)
```

**适用场景**: 深度学习梯度保护、联邦学习模型训练

---

#### 2. 使用拉普拉斯机制 (新增)

```python
config = FSAConfig(
    n_features=10,
    num_clients=5,
    global_epochs=50,
    local_epochs=5,
    use_differential_privacy=True,
    dp_mechanism='laplace',         # 拉普拉斯机制
    dp_epsilon=1.0,                 # 隐私预算
    dp_sensitivity=1.0,             # 敏感度
    dp_clip_norm=1.0                # 梯度裁剪
    # 注意: 拉普拉斯机制不需要 dp_delta 和 dp_noise_multiplier
)
```

**适用场景**: 计数查询、求和查询、需要纯ε-DP的场景

---

#### 3. 使用指数机制 (新增)

```python
config = FSAConfig(
    n_features=10,
    num_clients=5,
    global_epochs=50,
    local_epochs=5,
    use_differential_privacy=True,
    dp_mechanism='exponential',     # 指数机制
    dp_epsilon=1.0,                 # 隐私预算
    dp_sensitivity=1.0              # 质量函数敏感度
    # 注意: 指数机制不需要梯度裁剪参数
)
```

**适用场景**: 模型选择、超参数调优、离散选择

---

## 🔄 迁移指南

### 从旧版本迁移

如果您的代码之前使用差分隐私,**无需修改**即可继续运行!

**旧代码** (仍然有效):
```python
config = FSAConfig(
    use_differential_privacy=True,
    dp_epsilon=1.0,
    dp_delta=1e-5,
    # ...
)
```

**新代码** (推荐):
```python
config = FSAConfig(
    use_differential_privacy=True,
    dp_mechanism='gaussian',  # 明确指定机制
    dp_epsilon=1.0,
    dp_delta=1e-5,
    # ...
)
```

### 兼容性说明

- ✅ **向后兼容**: 不指定 `dp_mechanism` 时默认使用 `'gaussian'`
- ✅ **参数验证**: 系统会自动验证每种机制所需的参数
- ✅ **错误提示**: 缺少必要参数时会给出清晰的错误提示

---

## 📊 参数对照表

| 参数 | 高斯机制 | 拉普拉斯机制 | 指数机制 |
|------|---------|-------------|---------|
| `dp_mechanism` | `'gaussian'` | `'laplace'` | `'exponential'` |
| `dp_epsilon` | ✅ 必需 | ✅ 必需 | ✅ 必需 |
| `dp_delta` | ✅ 必需 | ❌ 不需要 | ❌ 不需要 |
| `dp_sensitivity` | ✅ 必需 | ✅ 必需 | ✅ 必需 |
| `dp_noise_multiplier` | ✅ 使用 | ❌ 不使用 | ❌ 不使用 |
| `dp_clip_norm` | ✅ 使用 | ✅ 使用 | ❌ 不使用 |

---

## 💡 使用建议

### 选择合适的机制

```python
# 决策流程
if task == "federated_learning":
    if output_type == "gradients":
        dp_mechanism = 'gaussian'      # 推荐: 高斯机制
    elif output_type == "counts":
        dp_mechanism = 'laplace'       # 推荐: 拉普拉斯机制
elif task == "model_selection":
    dp_mechanism = 'exponential'       # 推荐: 指数机制
```

### 参数调优建议

#### 高斯机制
```python
# 高隐私场景
dp_epsilon=0.5, dp_delta=1e-6, dp_clip_norm=0.5

# 中等隐私场景 (推荐)
dp_epsilon=1.0, dp_delta=1e-5, dp_clip_norm=1.0

# 低隐私场景
dp_epsilon=5.0, dp_delta=1e-4, dp_clip_norm=2.0
```

#### 拉普拉斯机制
```python
# 高隐私场景
dp_epsilon=0.5, dp_sensitivity=1.0

# 中等隐私场景 (推荐)
dp_epsilon=1.0, dp_sensitivity=1.0

# 低隐私场景
dp_epsilon=5.0, dp_sensitivity=1.0
```

---

## 🧪 测试新功能

### 运行对比示例

```bash
# 运行三种机制的对比示例
python federated_survival/examples/dp_mechanisms_comparison_example.py
```

### 查看详细文档

```bash
# 查看使用指南
cat federated_survival/examples/DP_MECHANISMS_GUIDE.md
```

### 运行单元测试

```bash
# 测试所有机制
pytest tests/test_dp_mechanisms.py -v

# 只测试高斯机制
pytest tests/test_dp_mechanisms.py::TestGaussianMechanism -v

# 只测试拉普拉斯机制
pytest tests/test_dp_mechanisms.py::TestLaplaceMechanism -v

# 只测试指数机制
pytest tests/test_dp_mechanisms.py::TestExponentialMechanism -v
```

---

## 📝 示例代码

### 完整训练示例

```python
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.data.generator import SurvivalDataGenerator

# 1. 生成模拟数据
generator = SurvivalDataGenerator(
    n_samples=1000,
    n_features=20,
    censor_rate=0.4
)
data = generator.generate()

# 2. 配置联邦学习 (使用拉普拉斯机制)
config = FSAConfig(
    n_features=20,
    num_clients=5,
    global_epochs=50,
    local_epochs=5,
    model_type='DeepSurv',
    use_differential_privacy=True,
    dp_mechanism='laplace',  # 使用拉普拉斯机制
    dp_epsilon=1.0,
    dp_sensitivity=1.0,
    dp_clip_norm=1.0,
    verbose=True
)

# 3. 运行训练
runner = FSARunner(config)
results = runner.run(data, type='raw')

# 4. 查看隐私信息
privacy_info = runner.get_privacy_info()
print(f"使用机制: {privacy_info['mechanism']}")
print(f"隐私预算: ε={privacy_info['epsilon']}")

# 5. 可视化结果
runner.plot_results(results)
```

---

## ⚠️ 注意事项

### 1. 参数验证

系统会自动验证参数的有效性:

```python
# ❌ 错误: 拉普拉斯机制不需要 delta
config = FSAConfig(
    dp_mechanism='laplace',
    dp_delta=1e-5  # 会被忽略，但建议不设置
)

# ✅ 正确
config = FSAConfig(
    dp_mechanism='laplace',
    dp_epsilon=1.0,
    dp_sensitivity=1.0
)
```

### 2. 机制选择

- **高斯机制**: 适合多轮训练,组合性质好
- **拉普拉斯机制**: 纯ε-DP,适合单次查询
- **指数机制**: 适合离散选择,不适合梯度

### 3. 性能影响

不同机制的计算开销:
- 高斯机制: 中等 (需要计算复杂的噪声规模)
- 拉普拉斯机制: 较低 (简单的噪声生成)
- 指数机制: 较高 (需要概率采样)

---

## 🔗 相关资源

### 文档
- [`DP_MECHANISMS_GUIDE.md`](./DP_MECHANISMS_GUIDE.md) - 详细使用指南
- [`README.md`](../../README.md) - 项目主文档

### 示例代码
- [`dp_mechanisms_comparison_example.py`](./dp_mechanisms_comparison_example.py) - 三种机制对比
- [`differential_privacy_example.py`](./differential_privacy_example.py) - 基础DP示例
- [`privacy_comparison_example.py`](./privacy_comparison_example.py) - 隐私效用对比

### 测试代码
- [`test_dp_mechanisms.py`](../../tests/test_dp_mechanisms.py) - 机制单元测试

---

## 📞 获取帮助

如有问题:
1. 查看 [`DP_MECHANISMS_GUIDE.md`](./DP_MECHANISMS_GUIDE.md)
2. 运行示例代码
3. 检查参数配置是否正确
4. 查看单元测试了解用法

---

**更新完成!** 🎉

现在您可以在联邦学习中灵活使用三种差分隐私机制,根据实际场景选择最合适的隐私保护方案!
