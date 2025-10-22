# 差分隐私测试说明

本目录包含了差分隐私功能的完整测试套件。

## 测试文件说明

### 1. 基础功能测试
- **`test_differential_privacy.py`**: 测试差分隐私工具类的核心功能
  - 高斯噪声添加
  - 梯度裁剪
  - 隐私预算计算
  - Renyi差分隐私转换

### 2. 集成测试
- **`test_dp_federated_learning.py`**: 测试差分隐私在联邦学习中的集成
  - 服务器和客户端初始化
  - 模型更新和本地训练
  - 隐私信息获取

### 3. 性能测试
- **`test_dp_performance.py`**: 测试差分隐私对性能的影响
  - 计算开销测量
  - 不同参数对性能的影响
  - 内存使用测试
  - 收敛速度分析

### 4. 配置测试
- **`test_dp_configs.py`**: 测试差分隐私配置
  - 有效和无效配置验证
  - 参数范围测试
  - 配置一致性测试

## 运行测试

### 运行所有测试
```bash
python tests/run_dp_tests.py
```

### 运行特定类型的测试
```bash
# 运行基础功能测试
python tests/run_dp_tests.py --test basic

# 运行集成测试
python tests/run_dp_tests.py --test integration

# 运行性能测试
python tests/run_dp_tests.py --test performance

# 运行快速修复验证测试
python tests/run_dp_tests.py --test fix
```

### 运行单个测试文件
```bash
# 运行基础差分隐私测试
python -m unittest tests.test_differential_privacy -v

# 运行联邦学习集成测试
python -m unittest tests.test_dp_federated_learning -v

# 运行性能测试
python -m unittest tests.test_dp_performance -v

# 运行配置测试
python -m unittest tests.test_dp_configs -v
```

### 运行特定测试类
```bash
# 运行差分隐私工具测试
python -m unittest tests.test_differential_privacy.TestDifferentialPrivacy -v

# 运行差分隐私集成测试
python -m unittest tests.test_differential_privacy.TestDifferentialPrivacyIntegration -v
```

## 测试内容详解

### 基础功能测试 (`test_differential_privacy.py`)

**测试项目：**
- ✅ 差分隐私工具初始化
- ✅ 高斯噪声添加（标准敏感度和自定义敏感度）
- ✅ 梯度裁剪（大梯度和小梯度）
- ✅ 噪声规模计算
- ✅ 隐私预算计算
- ✅ 权重差分隐私应用（已弃用方法）
- ✅ 梯度差分隐私应用
- ✅ Renyi散度计算
- ✅ Renyi到ε转换
- ✅ 不同客户端数量下的噪声规模
- ✅ 隐私参数验证
- ✅ 差分隐私禁用时的行为

**测试方法：**
```python
def test_add_gaussian_noise(self):
    """测试高斯噪声添加"""
    tensor = torch.ones(10, 5)
    noisy_tensor = self.dp_tool.add_gaussian_noise(tensor)
    
    # 验证噪声被添加
    self.assertFalse(torch.equal(tensor, noisy_tensor))
    
    # 验证噪声统计特性
    noise = noisy_tensor - tensor
    noise_std = torch.std(noise)
    expected_std = self.dp_tool.sensitivity * self.dp_tool.noise_multiplier
    self.assertAlmostEqual(noise_std.item(), expected_std, delta=0.5)
```

### 集成测试 (`test_dp_federated_learning.py`)

**测试项目：**
- ✅ 服务器差分隐私初始化
- ✅ 客户端差分隐私初始化
- ✅ 无差分隐私的服务器/客户端初始化
- ✅ 差分隐私下的模型更新
- ✅ 差分隐私下的本地训练
- ✅ 隐私信息获取
- ✅ 无差分隐私时的隐私信息
- ✅ 差分隐私参数验证
- ✅ 差分隐私噪声一致性
- ✅ 不同ε值下的差分隐私行为
- ✅ 差分隐私梯度裁剪有效性

**测试方法：**
```python
def test_server_with_dp_initialization(self):
    """测试服务器差分隐私初始化"""
    server = Server(self.config)
    
    # 验证差分隐私工具被正确初始化
    self.assertIsNotNone(server.dp_tool)
    self.assertEqual(server.dp_tool.epsilon, self.config.dp_epsilon)
```

### 性能测试 (`test_dp_performance.py`)

**测试项目：**
- ✅ 差分隐私计算开销测量
- ✅ 不同噪声规模对性能的影响
- ✅ 不同敏感度对性能的影响
- ✅ 不同梯度裁剪范数对性能的影响
- ✅ 差分隐私内存使用测试
- ✅ 差分隐私对收敛速度的影响
- ✅ 不同参数组合的性能测试

**测试方法：**
```python
def test_dp_overhead_measurement(self):
    """测试差分隐私计算开销"""
    # 测量无差分隐私的训练时间
    start_time = time.time()
    runner_no_dp = FSARunner(config_no_dp)
    results_no_dp = runner_no_dp.run(self.data, type='raw')
    time_no_dp = time.time() - start_time
    
    # 测量有差分隐私的训练时间
    start_time = time.time()
    runner_with_dp = FSARunner(config_with_dp)
    results_with_dp = runner_with_dp.run(self.data, type='raw')
    time_with_dp = time.time() - start_time
    
    # 验证差分隐私增加了计算开销
    self.assertGreater(time_with_dp, time_no_dp)
```

### 配置测试 (`test_dp_configs.py`)

**测试项目：**
- ✅ 有效差分隐私配置测试
- ✅ 无效差分隐私配置测试
- ✅ 差分隐私参数范围测试
- ✅ 差分隐私配置一致性测试
- ✅ 差分隐私配置复制测试

**测试方法：**
```python
def test_valid_dp_configs(self):
    """测试有效的差分隐私配置"""
    valid_configs = [
        # 强隐私配置
        {'use_differential_privacy': True, 'dp_epsilon': 0.1, ...},
        # 平衡配置
        {'use_differential_privacy': True, 'dp_epsilon': 1.0, ...},
        # 弱隐私配置
        {'use_differential_privacy': True, 'dp_epsilon': 5.0, ...},
    ]
    
    for config_dict in valid_configs:
        config = FSAConfig(**config_dict)
        self.assertTrue(config.use_differential_privacy)
```

## 测试结果解读

### 成功指标
- ✅ 所有测试通过
- ✅ 噪声添加正确
- ✅ 梯度裁剪有效
- ✅ 隐私参数验证正确
- ✅ 性能影响在合理范围内

### 性能基准
- **计算开销**: 差分隐私增加的计算时间应该 < 50%
- **内存使用**: 差分隐私增加的内存应该 < 100MB
- **模型性能**: 在合理参数下，性能损失应该 < 10%

### 参数建议
- **ε = 1.0**: 平衡隐私和性能
- **δ = 1e-5**: 标准失败概率
- **敏感度 = 1.0**: 标准敏感度
- **噪声乘数 = 1.0**: 标准噪声规模
- **梯度裁剪范数 = 1.0**: 标准裁剪范数

## 故障排除

### 常见问题

1. **测试失败**: 检查依赖包是否正确安装
2. **内存不足**: 减少测试数据规模
3. **性能测试超时**: 减少训练轮数
4. **参数验证失败**: 检查配置参数是否在有效范围内

### 调试建议

1. **启用详细输出**: 使用 `-v` 参数
2. **单独运行测试**: 定位具体问题
3. **检查日志**: 查看详细的错误信息
4. **验证环境**: 确保所有依赖包版本正确

## 扩展测试

### 添加新测试
1. 在相应的测试文件中添加新的测试方法
2. 遵循命名约定：`test_功能描述`
3. 添加适当的断言和验证
4. 更新文档说明

### 自定义测试配置
```python
# 创建自定义测试配置
custom_config = FSAConfig(
    use_differential_privacy=True,
    dp_epsilon=0.5,
    dp_delta=1e-6,
    dp_sensitivity=0.5,
    dp_noise_multiplier=2.0,
    dp_clip_norm=0.5
)

# 运行自定义测试
dp_tool = DifferentialPrivacy(custom_config)
# 执行测试逻辑...
```

这个测试套件提供了全面的差分隐私功能验证，确保代码的正确性和可靠性。
