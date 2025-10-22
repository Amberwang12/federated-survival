# 示例脚本说明

本目录包含 Federated Survival 项目的完整示例脚本，涵盖框架的所有主要功能。

## 📁 示例文件列表

### 1. basic_usage_example.py ⭐

**功能**：演示联邦生存分析的完整工作流程

**运行方式**：
```bash
cd federated_survival/examples
python basic_usage_example.py
```

**包含内容**：
- ✅ 数据生成配置
- ✅ 模拟数据生成
- ✅ 数据分割为联邦学习格式
- ✅ 联邦学习配置
- ✅ 模型训练
- ✅ 结果评估和可视化

**适合人群**：初学者，第一次使用框架

---

### 2. data_generation_example.py

**功能**：演示所有可用的数据生成方法

**运行方式**：
```bash
cd federated_survival/examples
python differential_privacy_example.py
```

**特性**：
- ✅ 启用差分隐私保护
- ✅ 展示隐私预算和参数配置
- ✅ 演示原始数据和数据增强的训练
- ✅ 可视化训练结果

**关键参数**：
```python
use_differential_privacy=True  # 启用差分隐私
dp_epsilon=1.0                 # 隐私预算 (ε)
dp_delta=1e-5                  # 失败概率 (δ)
dp_sensitivity=1.0             # 敏感度
dp_noise_multiplier=1.0        # 噪声乘数
dp_clip_norm=1.0               # 梯度裁剪范数
```

**预期输出**：
- 配置信息
- 差分隐私保护信息
- 训练过程进度
- 最终性能指标（C-index 和 IBS）
- 训练曲线可视化

---

### 2. privacy_comparison_example.py

**功能**：对比有无差分隐私保护下的联邦学习性能差异

**运行方式**：
```bash
cd federated_survival/examples
python privacy_comparison_example.py
```

**特性**：
- ✅ 对比无差分隐私和有差分隐私的性能
- ✅ 使用相同数据集确保公平对比
- ✅ 详细的性能差异可视化
- ✅ 四象限对比图表

**可视化内容**：
1. **C-index 对比**：训练集和测试集的 C-index 随训练轮次的变化
2. **IBS 对比**：训练集和测试集的 IBS 随训练轮次的变化
3. **C-index 差异**：有 DP 和无 DP 的 C-index 差值
4. **IBS 差异**：有 DP 和无 DP 的 IBS 差值

**预期输出**：
- 两组实验的训练过程
- 性能对比表格
- 四象限对比图表

---

## 🚀 快速开始

### 前提条件

确保已安装 federated-survival 包：

```bash
pip install federated-survival
```

或者在开发模式下安装：

```bash
cd /path/to/federated_survival
pip install -e .
```

### 运行示例

#### 1. 基础示例
```bash
# 进入示例目录
cd federated_survival/examples

# 运行差分隐私示例
python differential_privacy_example.py
```

#### 2. 对比实验
```bash
# 运行性能对比示例
python privacy_comparison_example.py
```

## 📊 修改参数

您可以修改示例脚本中的参数来探索不同配置的效果：

### 数据集参数
```python
n_samples=1000      # 样本数量
n_features=20       # 特征数量
num_clients=5       # 客户端数量
```

### 训练参数
```python
global_epochs=20    # 全局训练轮数
local_epochs=2      # 本地训练轮数
learning_rate=0.01  # 学习率
```

### 差分隐私参数
```python
dp_epsilon=1.0      # 隐私预算（较小值=更强隐私）
dp_delta=1e-5       # 失败概率
dp_sensitivity=1.0  # 敏感度
dp_noise_multiplier=1.0  # 噪声乘数
dp_clip_norm=1.0    # 梯度裁剪范数
```

## 🔬 实验建议

### 1. 探索隐私预算的影响
修改 `dp_epsilon` 参数，观察不同隐私级别下的性能：

```python
# 强隐私保护（较大性能损失）
dp_epsilon=0.1

# 平衡隐私和性能
dp_epsilon=1.0

# 弱隐私保护（较小性能损失）
dp_epsilon=10.0
```

### 2. 测试不同数据规模
```python
# 小数据集
n_samples=100, num_clients=3

# 中等数据集
n_samples=1000, num_clients=5

# 大数据集
n_samples=10000, num_clients=10
```

### 3. 比较不同模型
修改 `model_type` 参数：
```python
model_type='PC-Hazard'      # 分段常数风险模型
model_type='DeepSurv'       # 深度生存模型
model_type='CoxPH'          # Cox 比例风险模型
```

## 📝 注意事项

1. **随机性**：由于差分隐私添加了随机噪声，每次运行结果会略有不同
2. **性能权衡**：启用差分隐私会降低一些模型性能，这是隐私保护的代价
3. **参数调优**：需要根据具体应用场景调整差分隐私参数
4. **计算资源**：较大的数据集和更多的训练轮次需要更多计算时间

## 🐛 故障排除

### 问题 1: ImportError
```
ImportError: No module named 'federated_survival'
```
**解决**：安装包 `pip install federated-survival` 或 `pip install -e .`

### 问题 2: 内存不足
**解决**：减少样本数量或客户端数量

### 问题 3: 训练过慢
**解决**：减少 `global_epochs` 或 `local_epochs`

### 问题 4: 图形窗口不显示
**解决**：确保支持图形界面，或使用 `plt.savefig()` 保存图片

## 📚 更多资源

- [完整文档](../../README.md)
- [差分隐私原理](../../README.md#differential-privacy)
- [API 参考](../../README.md#usage)

## 💡 贡献

欢迎提交新的示例脚本！请遵循以下准则：
1. 代码清晰易懂，包含详细注释
2. 提供运行说明和预期输出
3. 处理可能的异常情况
4. 添加到本 README 文档中
