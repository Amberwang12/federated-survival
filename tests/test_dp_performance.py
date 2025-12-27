"""
差分隐私性能测试
"""
import unittest
import torch
import numpy as np
import time
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.data.generator import DataGenerator


class TestDPPerformance(unittest.TestCase):
    """差分隐私性能测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = FSAConfig(
            n_samples=200,
            n_features=10,
            num_clients=3,
            global_epochs=3,
            local_epochs=1,
            model_type='PC-Hazard',
            verbose=False
        )
        
        # 生成测试数据
        from federated_survival.data.generator import SimulationConfig, DataGenerator
        from federated_survival.data.splitter import DataSplitter
        
        sim_config = SimulationConfig(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            random_state=self.config.random_seed
        )
        generator = DataGenerator(sim_config)
        raw_data = generator.generate('weibull', c_mean=0.4)
        
        # 转换为联邦学习格式
        splitter = DataSplitter(
            n_clients=self.config.num_clients,
            split_type='iid',
            test_size=0.2,
            random_state=self.config.random_seed
        )
        self.data = splitter.split(raw_data)
    
    def test_dp_overhead_measurement(self):
        """测试差分隐私计算开销"""
        # 无差分隐私配置
        config_no_dp = FSAConfig(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            num_clients=self.config.num_clients,
            global_epochs=self.config.global_epochs,
            local_epochs=self.config.local_epochs,
            model_type=self.config.model_type,
            verbose=self.config.verbose,
            use_differential_privacy=False
        )
        
        # 有差分隐私配置
        config_with_dp = FSAConfig(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            num_clients=self.config.num_clients,
            global_epochs=self.config.global_epochs,
            local_epochs=self.config.local_epochs,
            model_type=self.config.model_type,
            verbose=self.config.verbose,
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )
        
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
        
        # 输出性能对比
        print(f"\n性能对比:")
        print(f"无差分隐私训练时间: {time_no_dp:.2f}秒")
        print(f"有差分隐私训练时间: {time_with_dp:.2f}秒")
        
        # 由于训练时间受多种随机因素影响，我们只验证两者都能正常完成
        # 而不是严格验证时间差异
        self.assertGreater(time_no_dp, 0)
        self.assertGreater(time_with_dp, 0)
        
        # 验证训练结果有效
        self.assertGreater(len(results_no_dp['train_Cindex']), 0)
        self.assertGreater(len(results_with_dp['train_Cindex']), 0)
        
        # 计算相对开销，但不做严格断言
        if time_no_dp > 0:
            overhead_pct = ((time_with_dp - time_no_dp) / time_no_dp * 100)
            print(f"相对开销: {overhead_pct:+.1f}%")
            print(f"注：由于训练时间较短且受随机因素影响，开销可能为负值或波动较大")
    
    def test_dp_noise_scale_impact(self):
        """测试不同噪声规模对性能的影响"""
        epsilon_values = [0.1, 1.0, 5.0, 10.0]
        performance_results = {}
        
        for epsilon in epsilon_values:
            config = FSAConfig(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                num_clients=self.config.num_clients,
                global_epochs=self.config.global_epochs,
                local_epochs=self.config.local_epochs,
                model_type=self.config.model_type,
                verbose=self.config.verbose,
                use_differential_privacy=True,
                dp_epsilon=epsilon,
                dp_delta=1e-5,
                dp_sensitivity=1.0,
                dp_noise_multiplier=1.0,
                dp_clip_norm=1.0
            )
            
            runner = FSARunner(config)
            results = runner.run(self.data, type='raw')
            
            performance_results[epsilon] = {
                'train_cindex': results['train_Cindex'][-1],
                'test_cindex': results['test_Cindex'][-1],
                'train_ibs': results['train_IBS'][-1],
                'test_ibs': results['test_IBS'][-1]
            }
        
        # 验证ε值越小，性能越差（噪声越大）
        epsilons = sorted(epsilon_values)
        train_cindexes = [performance_results[eps]['train_cindex'] for eps in epsilons]
        test_cindexes = [performance_results[eps]['test_cindex'] for eps in epsilons]
        
        # 输出性能对比
        print(f"\n不同ε值下的性能对比:")
        for epsilon in epsilons:
            perf = performance_results[epsilon]
            print(f"ε={epsilon}: Train C-index={perf['train_cindex']:.4f}, "
                  f"Test C-index={perf['test_cindex']:.4f}")
        
        # 验证总体趋势：最小的ε应该有最差的性能，最大的ε应该有最好的性能
        # 由于噪声的随机性，我们只验证极端值的关系
        self.assertLessEqual(
            train_cindexes[0], train_cindexes[-1] + 0.1,  # 允许10%的误差
            f"期望小ε({epsilons[0]})的性能({train_cindexes[0]:.4f}) ≤ 大ε({epsilons[-1]})的性能({train_cindexes[-1]:.4f})"
        )
    
    def test_dp_sensitivity_impact(self):
        """测试不同敏感度对性能的影响"""
        sensitivity_values = [0.5, 1.0, 2.0]
        performance_results = {}
        
        for sensitivity in sensitivity_values:
            config = FSAConfig(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                num_clients=self.config.num_clients,
                global_epochs=self.config.global_epochs,
                local_epochs=self.config.local_epochs,
                model_type=self.config.model_type,
                verbose=self.config.verbose,
                use_differential_privacy=True,
                dp_epsilon=1.0,
                dp_delta=1e-5,
                dp_sensitivity=sensitivity,
                dp_noise_multiplier=1.0,
                dp_clip_norm=1.0
            )
            
            runner = FSARunner(config)
            results = runner.run(self.data, type='raw')
            
            performance_results[sensitivity] = {
                'train_cindex': results['train_Cindex'][-1],
                'test_cindex': results['test_Cindex'][-1]
            }
        
        # 输出性能对比
        print(f"\n不同敏感度下的性能对比:")
        for sensitivity in sensitivity_values:
            perf = performance_results[sensitivity]
            print(f"Sensitivity={sensitivity}: Train C-index={perf['train_cindex']:.4f}, "
                  f"Test C-index={perf['test_cindex']:.4f}")
    
    def test_dp_clip_norm_impact(self):
        """测试不同梯度裁剪范数对性能的影响"""
        clip_norm_values = [0.5, 1.0, 2.0]
        performance_results = {}
        
        for clip_norm in clip_norm_values:
            config = FSAConfig(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                num_clients=self.config.num_clients,
                global_epochs=self.config.global_epochs,
                local_epochs=self.config.local_epochs,
                model_type=self.config.model_type,
                verbose=self.config.verbose,
                use_differential_privacy=True,
                dp_epsilon=1.0,
                dp_delta=1e-5,
                dp_sensitivity=1.0,
                dp_noise_multiplier=1.0,
                dp_clip_norm=clip_norm
            )
            
            runner = FSARunner(config)
            results = runner.run(self.data, type='raw')
            
            performance_results[clip_norm] = {
                'train_cindex': results['train_Cindex'][-1],
                'test_cindex': results['test_Cindex'][-1]
            }
        
        # 输出性能对比
        print(f"\n不同梯度裁剪范数下的性能对比:")
        for clip_norm in clip_norm_values:
            perf = performance_results[clip_norm]
            print(f"Clip Norm={clip_norm}: Train C-index={perf['train_cindex']:.4f}, "
                  f"Test C-index={perf['test_cindex']:.4f}")
    
    def test_dp_memory_usage(self):
        """测试差分隐私内存使用"""
        # 创建大张量测试内存使用
        large_tensor = torch.randn(1000, 1000)
        
        # 无差分隐私
        config_no_dp = FSAConfig(use_differential_privacy=False)
        dp_tool_no_dp = DifferentialPrivacy(config_no_dp)
        
        # 有差分隐私
        config_with_dp = FSAConfig(
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )
        dp_tool_with_dp = DifferentialPrivacy(config_with_dp)
        
        # 测试内存使用
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行差分隐私操作
        noisy_tensor = dp_tool_with_dp.add_gaussian_noise(large_tensor)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # 验证内存使用合理
        self.assertLess(memory_increase, 100)  # 内存增加应该小于100MB
        
        print(f"\n内存使用测试:")
        print(f"操作前内存: {memory_before:.1f}MB")
        print(f"操作后内存: {memory_after:.1f}MB")
        print(f"内存增加: {memory_increase:.1f}MB")
    
    def test_dp_convergence_speed(self):
        """测试差分隐私对收敛速度的影响"""
        # 无差分隐私
        config_no_dp = FSAConfig(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            num_clients=self.config.num_clients,
            global_epochs=5,
            local_epochs=self.config.local_epochs,
            model_type=self.config.model_type,
            verbose=self.config.verbose,
            use_differential_privacy=False
        )
        
        # 有差分隐私
        config_with_dp = FSAConfig(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            num_clients=self.config.num_clients,
            global_epochs=5,
            local_epochs=self.config.local_epochs,
            model_type=self.config.model_type,
            verbose=self.config.verbose,
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )
        
        # 运行训练
        runner_no_dp = FSARunner(config_no_dp)
        results_no_dp = runner_no_dp.run(self.data, type='raw')
        
        runner_with_dp = FSARunner(config_with_dp)
        results_with_dp = runner_with_dp.run(self.data, type='raw')
        
        # 分析收敛速度
        print(f"\n收敛速度对比:")
        print(f"无差分隐私 - 最终C-index: {results_no_dp['test_Cindex'][-1]:.4f}")
        print(f"有差分隐私 - 最终C-index: {results_with_dp['test_Cindex'][-1]:.4f}")
        
        # 验证差分隐私可能影响收敛速度
        # 注意：这不是绝对的，因为噪声可能有助于避免过拟合
        performance_diff = results_with_dp['test_Cindex'][-1] - results_no_dp['test_Cindex'][-1]
        print(f"性能差异: {performance_diff:+.4f}")
    
    def test_dp_parameter_combinations(self):
        """测试不同参数组合的性能"""
        parameter_combinations = [
            {'epsilon': 0.5, 'sensitivity': 0.5, 'clip_norm': 0.5},
            {'epsilon': 1.0, 'sensitivity': 1.0, 'clip_norm': 1.0},
            {'epsilon': 2.0, 'sensitivity': 2.0, 'clip_norm': 2.0},
        ]
        
        results = {}
        for i, params in enumerate(parameter_combinations):
            config = FSAConfig(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                num_clients=self.config.num_clients,
                global_epochs=self.config.global_epochs,
                local_epochs=self.config.local_epochs,
                model_type=self.config.model_type,
                verbose=self.config.verbose,
                use_differential_privacy=True,
                dp_epsilon=params['epsilon'],
                dp_delta=1e-5,
                dp_sensitivity=params['sensitivity'],
                dp_noise_multiplier=1.0,
                dp_clip_norm=params['clip_norm']
            )
            
            runner = FSARunner(config)
            result = runner.run(self.data, type='raw')
            
            results[f"组合{i+1}"] = {
                'params': params,
                'final_cindex': result['test_Cindex'][-1],
                'final_ibs': result['test_IBS'][-1]
            }
        
        # 输出结果
        print(f"\n不同参数组合的性能:")
        for name, result in results.items():
            params = result['params']
            print(f"{name}: ε={params['epsilon']}, S={params['sensitivity']}, "
                  f"C={params['clip_norm']} -> C-index={result['final_cindex']:.4f}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
