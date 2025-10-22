"""
差分隐私配置测试
"""
import unittest
from federated_survival.core.config import FSAConfig
from federated_survival.core.differential_privacy import DifferentialPrivacy


class TestDPConfigs(unittest.TestCase):
    """差分隐私配置测试类"""
    
    def test_valid_dp_configs(self):
        """测试有效的差分隐私配置"""
        valid_configs = [
            # 强隐私配置
            {
                'use_differential_privacy': True,
                'dp_epsilon': 0.1,
                'dp_delta': 1e-6,
                'dp_sensitivity': 0.5,
                'dp_noise_multiplier': 2.0,
                'dp_clip_norm': 0.5
            },
            # 平衡配置
            {
                'use_differential_privacy': True,
                'dp_epsilon': 1.0,
                'dp_delta': 1e-5,
                'dp_sensitivity': 1.0,
                'dp_noise_multiplier': 1.0,
                'dp_clip_norm': 1.0
            },
            # 弱隐私配置
            {
                'use_differential_privacy': True,
                'dp_epsilon': 5.0,
                'dp_delta': 1e-4,
                'dp_sensitivity': 2.0,
                'dp_noise_multiplier': 0.5,
                'dp_clip_norm': 2.0
            },
            # 禁用差分隐私
            {
                'use_differential_privacy': False
            }
        ]
        
        for i, config_dict in enumerate(valid_configs):
            with self.subTest(config=i):
                config = FSAConfig(**config_dict)
                
                if config_dict.get('use_differential_privacy', False):
                    # 验证差分隐私参数
                    self.assertTrue(config.use_differential_privacy)
                    self.assertEqual(config.dp_epsilon, config_dict['dp_epsilon'])
                    self.assertEqual(config.dp_delta, config_dict['dp_delta'])
                    self.assertEqual(config.dp_sensitivity, config_dict['dp_sensitivity'])
                    self.assertEqual(config.dp_noise_multiplier, config_dict['dp_noise_multiplier'])
                    self.assertEqual(config.dp_clip_norm, config_dict['dp_clip_norm'])
                    
                    # 验证差分隐私工具可以正确初始化
                    dp_tool = DifferentialPrivacy(config)
                    self.assertIsNotNone(dp_tool)
                else:
                    # 验证差分隐私被禁用
                    self.assertFalse(config.use_differential_privacy)
    
    def test_invalid_dp_configs(self):
        """测试无效的差分隐私配置"""
        invalid_configs = [
            # 负的ε值
            {
                'use_differential_privacy': True,
                'dp_epsilon': -1.0,
                'dp_delta': 1e-5,
                'dp_sensitivity': 1.0,
                'dp_noise_multiplier': 1.0,
                'dp_clip_norm': 1.0
            },
            # δ值大于等于1
            {
                'use_differential_privacy': True,
                'dp_epsilon': 1.0,
                'dp_delta': 1.0,
                'dp_sensitivity': 1.0,
                'dp_noise_multiplier': 1.0,
                'dp_clip_norm': 1.0
            },
            # 负的敏感度
            {
                'use_differential_privacy': True,
                'dp_epsilon': 1.0,
                'dp_delta': 1e-5,
                'dp_sensitivity': -1.0,
                'dp_noise_multiplier': 1.0,
                'dp_clip_norm': 1.0
            },
            # 负的噪声乘数
            {
                'use_differential_privacy': True,
                'dp_epsilon': 1.0,
                'dp_delta': 1e-5,
                'dp_sensitivity': 1.0,
                'dp_noise_multiplier': -1.0,
                'dp_clip_norm': 1.0
            },
            # 负的梯度裁剪范数
            {
                'use_differential_privacy': True,
                'dp_epsilon': 1.0,
                'dp_delta': 1e-5,
                'dp_sensitivity': 1.0,
                'dp_noise_multiplier': 1.0,
                'dp_clip_norm': -1.0
            }
        ]
        
        for i, config_dict in enumerate(invalid_configs):
            with self.subTest(config=i):
                with self.assertRaises(ValueError):
                    FSAConfig(**config_dict)
    
    def test_dp_parameter_ranges(self):
        """测试差分隐私参数范围"""
        # 测试边界值
        boundary_configs = [
            # 最小有效值
            {
                'use_differential_privacy': True,
                'dp_epsilon': 1e-10,  # 非常小的正数
                'dp_delta': 1e-10,    # 非常小的正数
                'dp_sensitivity': 1e-10,  # 非常小的正数
                'dp_noise_multiplier': 1e-10,  # 非常小的正数
                'dp_clip_norm': 1e-10   # 非常小的正数
            },
            # 较大值
            {
                'use_differential_privacy': True,
                'dp_epsilon': 100.0,
                'dp_delta': 0.999,  # 接近1但小于1
                'dp_sensitivity': 100.0,
                'dp_noise_multiplier': 100.0,
                'dp_clip_norm': 100.0
            }
        ]
        
        for i, config_dict in enumerate(boundary_configs):
            with self.subTest(config=i):
                # 这些配置应该有效
                config = FSAConfig(**config_dict)
                self.assertTrue(config.use_differential_privacy)
                
                # 验证差分隐私工具可以正确初始化
                dp_tool = DifferentialPrivacy(config)
                self.assertIsNotNone(dp_tool)
    
    def test_dp_config_consistency(self):
        """测试差分隐私配置一致性"""
        # 测试配置对象的一致性
        config = FSAConfig(
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )
        
        # 创建差分隐私工具
        dp_tool1 = DifferentialPrivacy(config)
        dp_tool2 = DifferentialPrivacy(config)
        
        # 验证两个工具的参数一致
        self.assertEqual(dp_tool1.epsilon, dp_tool2.epsilon)
        self.assertEqual(dp_tool1.delta, dp_tool2.delta)
        self.assertEqual(dp_tool1.sensitivity, dp_tool2.sensitivity)
        self.assertEqual(dp_tool1.noise_multiplier, dp_tool2.noise_multiplier)
        self.assertEqual(dp_tool1.clip_norm, dp_tool2.clip_norm)
    
    def test_dp_config_copy(self):
        """测试差分隐私配置复制"""
        original_config = FSAConfig(
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )
        
        # 复制配置
        copied_config = FSAConfig(**original_config.__dict__)
        
        # 验证复制的配置与原配置一致
        self.assertEqual(original_config.use_differential_privacy, copied_config.use_differential_privacy)
        self.assertEqual(original_config.dp_epsilon, copied_config.dp_epsilon)
        self.assertEqual(original_config.dp_delta, copied_config.dp_delta)
        self.assertEqual(original_config.dp_sensitivity, copied_config.dp_sensitivity)
        self.assertEqual(original_config.dp_noise_multiplier, copied_config.dp_noise_multiplier)
        self.assertEqual(original_config.dp_clip_norm, copied_config.dp_clip_norm)
        
        # 验证差分隐私工具行为一致
        dp_tool_original = DifferentialPrivacy(original_config)
        dp_tool_copied = DifferentialPrivacy(copied_config)
        
        self.assertEqual(dp_tool_original.epsilon, dp_tool_copied.epsilon)
        self.assertEqual(dp_tool_original.delta, dp_tool_copied.delta)
        self.assertEqual(dp_tool_original.sensitivity, dp_tool_copied.sensitivity)
        self.assertEqual(dp_tool_original.noise_multiplier, dp_tool_copied.noise_multiplier)
        self.assertEqual(dp_tool_original.clip_norm, dp_tool_copied.clip_norm)


if __name__ == '__main__':
    unittest.main(verbosity=2)
