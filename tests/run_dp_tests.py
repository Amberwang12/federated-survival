"""
差分隐私测试运行脚本
"""
import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_differential_privacy import TestDifferentialPrivacy, TestDifferentialPrivacyIntegration
from test_dp_federated_learning import TestDPFederatedLearning
from test_dp_performance import TestDPPerformance


def run_all_dp_tests():
    """运行所有差分隐私测试"""
    print("=" * 60)
    print("开始运行差分隐私测试套件")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加基础差分隐私测试
    print("\n1. 运行基础差分隐私功能测试...")
    test_suite.addTest(unittest.makeSuite(TestDifferentialPrivacy))
    test_suite.addTest(unittest.makeSuite(TestDifferentialPrivacyIntegration))
    
    # 添加联邦学习集成测试
    print("\n2. 运行联邦学习集成测试...")
    test_suite.addTest(unittest.makeSuite(TestDPFederatedLearning))
    
    # 添加性能测试
    print("\n3. 运行性能测试...")
    test_suite.addTest(unittest.makeSuite(TestDPPerformance))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # 返回测试是否全部通过
    return len(result.failures) == 0 and len(result.errors) == 0


def run_specific_test(test_name):
    """运行特定的测试类"""
    print(f"运行测试: {test_name}")
    print("=" * 60)
    
    # 根据测试名称选择测试类
    test_classes = {
        'basic': [TestDifferentialPrivacy, TestDifferentialPrivacyIntegration],
        'integration': [TestDPFederatedLearning],
        'performance': [TestDPPerformance],
        'all': [TestDifferentialPrivacy, TestDifferentialPrivacyIntegration, 
                TestDPFederatedLearning, TestDPPerformance]
    }
    
    if test_name not in test_classes:
        print(f"未知的测试名称: {test_name}")
        print(f"可用的测试: {list(test_classes.keys())}")
        return False
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    for test_class in test_classes[test_name]:
        test_suite.addTest(unittest.makeSuite(test_class))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行差分隐私测试')
    parser.add_argument('--test', '-t', 
                       choices=['basic', 'integration', 'performance', 'all', 'fix'],
                       default='all',
                       help='要运行的测试类型')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_dp_tests()
    elif args.test == 'fix':
        # 运行快速修复验证测试
        import test_dp_fix
        success = test_dp_fix.main()
    else:
        success = run_specific_test(args.test)
    
    if success:
        print("\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()
