#!/usr/bin/env python3
"""
Basic tests for AZR system components that don't require model loading.
These tests can run quickly without downloading models.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    modules = [
        "src.azr_system",
        "src.task_manager", 
        "src.code_executor",
        "src.reward_calculator",
        "src.model_wrapper",
        "utils.evaluation",
        "utils.logging_utils"
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0


def test_code_executor():
    """Test basic code execution."""
    print("\nTesting code executor...")
    
    from src.code_executor import CodeExecutor
    from src.azr_system import AZRConfig
    
    config = AZRConfig()
    executor = CodeExecutor(config)
    test_cases = [
        ("lambda x: x * 2", "5", 10),
        ("lambda x: x + 1", "10", 11),
        ("lambda x, y: x + y", "(2, 3)", 5),
        ("lambda x: x ** 2", "4", 16),
    ]
    
    passed = 0
    for code, input_data, expected in test_cases:
        result = executor.execute_safe(code, input_data)
        if result['success'] and result['output'] == expected:
            print(f"  ✅ {code} with {input_data} = {result['output']}")
            passed += 1
        else:
            print(f"  ❌ {code} with {input_data} failed: {result}")
    
    return passed == len(test_cases)


def test_task_manager():
    """Test task management."""
    print("\nTesting task manager...")
    
    from src.task_manager import TaskManager, ReasoningTask, ReasoningType
    from src.azr_system import AZRConfig
    
    config = AZRConfig()
    manager = TaskManager(config)
    
    # Test adding tasks
    task = ReasoningTask(
        type='deduction',
        program="lambda x: x * 2",
        input="5",
        expected_output="10"
    )
    
    manager.add_task(task)
    
    all_tasks = manager.sample_tasks(10)  # Pass batch_size as int
    
    if len(all_tasks) >= 1 and all_tasks[0].program == "lambda x: x * 2":
        print("  ✅ Task addition")
    else:
        print("  ❌ Task addition failed")
        return False
    stats = manager.get_buffer_stats()
    if 'deduction' in stats and stats['deduction']['size'] >= 1:
        print("  ✅ Statistics")
    else:
        print(f"  ❌ Statistics failed: {stats}")
        return False
    
    return True


def test_reward_calculator():
    """Test reward calculation."""
    print("\nTesting reward calculator...")
    
    from src.reward_calculator import RewardCalculator
    from src.azr_system import AZRConfig
    
    config = AZRConfig()
    calculator = RewardCalculator(config)    # Use positional arguments - calculate_propose_reward(valid_tasks, total_tasks)
    reward = calculator.calculate_propose_reward(valid_tasks=1, total_tasks=1)
    if abs(reward - 1.0) < 0.3:  # Expect around 1.0 (validity_rate=1.0 * 0.7 + learnability=1.0 * 0.3 + diversity)
        print("  ✅ Proposer reward (optimal)")
    else:
        print(f"  ❌ Proposer reward: expected ~1.0, got {reward}")
        return False
    
    # Use positional arguments - calculate_solve_reward(is_correct, task_complexity)
    reward = calculator.calculate_solve_reward(True, 1)
    if reward == 1.0:
        print("  ✅ Solver reward (correct)")
    else:
        print(f"  ❌ Solver reward: expected 1.0, got {reward}")
        return False
    
    # Use positional arguments - calculate_combined_reward(proposer_reward, solve_rewards)
    combined = calculator.calculate_combined_reward(0.8, [1.0])
    expected = 0.92  # 0.8 * 0.4 + 1.0 * 0.6 = 0.92
    if abs(combined - expected) < 0.5:  # Allow for normalization effects
        print("  ✅ Combined reward")
    else:
        print(f"  ❌ Combined reward: expected around {expected}, got {combined}")
        return False
    
    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    from src.azr_system import AZRConfig
    
    config = AZRConfig()
    
    checks = [
        ("model_name", "microsoft/DialoGPT-small"),
        ("batch_size", 4),
        ("buffer_size", 1000),
        ("execution_timeout", 5.0)
    ]
    
    all_good = True
    for attr, expected in checks:
        value = getattr(config, attr)
        if value == expected:
            print(f"  ✅ {attr} = {value}")
        else:
            print(f"  ❌ {attr}: expected {expected}, got {value}")
            all_good = False
    
    return all_good


def main():
    """Run all basic tests."""
    print("=" * 60)
    print("🧪 AZR Basic Tests (No Model Loading)")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Code Executor", test_code_executor),
        ("Task Manager", test_task_manager),
        ("Reward Calculator", test_reward_calculator)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed!")
        print("\nTo run the AZR system:")
        print("  python main.py --mode demo")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)