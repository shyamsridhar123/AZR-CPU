#!/usr/bin/env python3
"""
Basic tests for AZR system components that don't require model loading.
These tests can run quickly without downloading models.
Updated to test the enhanced AZR system with prompt management features.
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
    
    # Add prompt system modules
    prompt_modules = [
        "Prompts.prompt_manager",
        "Prompts.solution_prompts",
        "Prompts.task_generation_prompts",
        "Prompts.validation_prompts",
        "Prompts.advanced_templates"
    ]
    
    # Test core modules first
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed.append(module)
    
    # Test prompt system modules (optional)
    prompt_system_available = True
    for module in prompt_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            prompt_system_available = False
    
    if not prompt_system_available:
        print("  ⚠️ Prompt system is not fully available, some advanced features may be limited")
    
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
    
    print(f"Passed {passed}/{len(test_cases)} code execution tests")
    return passed == len(test_cases)


def test_task_manager():
    """Test task manager functionality."""
    print("\nTesting task manager...")
    
    from src.task_manager import TaskManager, ReasoningTask
    from src.azr_system import AZRConfig
    
    config = AZRConfig()
    manager = TaskManager(config)
    
    # Create tasks
    tasks = [
        ReasoningTask(
            type="deduction",
            program="lambda x: x * 2",
            input="5",
            expected_output="10",
            complexity=1
        ),
        ReasoningTask(
            type="abduction",
            program="lambda x: x + 3",
            input="7",
            expected_output="10",
            complexity=2
        ),
        ReasoningTask(
            type="induction",
            program="lambda x: x ** 2",
            input="3",
            expected_output="9",
            complexity=3
        )
    ]
    
    # Add tasks
    for task in tasks:
        manager.add_task(task)
    
    # Test buffer stats
    stats = manager.get_buffer_stats()
    
    test_results = {
        "deduction": stats["deduction"]["size"] == 1,
        "abduction": stats["abduction"]["size"] == 1,
        "induction": stats["induction"]["size"] == 1,
        "complexity": stats["deduction"]["complexity_distribution"].get(1, 0) == 1
    }
    
    print(f"  Testing buffer sizes:")
    print(f"    Deduction: {'✅' if test_results['deduction'] else '❌'} (expected 1, got {stats['deduction']['size']})")
    print(f"    Abduction: {'✅' if test_results['abduction'] else '❌'} (expected 1, got {stats['abduction']['size']})")
    print(f"    Induction: {'✅' if test_results['induction'] else '❌'} (expected 1, got {stats['induction']['size']})")
    
    # Test task sampling
    sampled = manager.sample_tasks(2)
    print(f"  Sampled {len(sampled)} tasks (expected 2)")
    
    # Test task recording
    test_task = tasks[0]
    manager.record_task_attempt(test_task, True)
    
    # Test curriculum adjustment
    initial_difficulty = manager.difficulty_level
    manager.increase_difficulty(0.5)
    
    difficulty_increased = manager.difficulty_level > initial_difficulty
    print(f"  Difficulty adjustment: {'✅' if difficulty_increased else '❌'} " +
          f"({initial_difficulty} → {manager.difficulty_level})")
    
    all_passed = all(test_results.values()) and len(sampled) > 0 and difficulty_increased
    print(f"Task manager tests: {'✅ Passed' if all_passed else '❌ Failed'}")
    
    return all_passed


def test_reward_calculator():
    """Test reward calculation."""
    print("\nTesting reward calculator...")
    
    from src.reward_calculator import RewardCalculator
    from src.azr_system import AZRConfig
    
    config = AZRConfig()
    # Enable simple rewards for testing only
    config.simple_rewards = True
    calculator = RewardCalculator(config)
    
    # Test proposer rewards
    propose_tests = [
        (5, 5, 1.0),  # All tasks valid
        (3, 5, 0.6),  # 60% valid
        (0, 5, 0.0),  # No valid tasks
    ]
    
    propose_passed = 0
    print("  Testing proposer rewards:")
    for valid, total, expected in propose_tests:
        reward = calculator.calculate_propose_reward(valid, total)
        if abs(reward - expected) < 0.01:
            print(f"    ✅ {valid}/{total} valid tasks → {reward:.2f} reward")
            propose_passed += 1
        else:
            print(f"    ❌ {valid}/{total} valid tasks → {reward:.2f} reward (expected {expected:.2f})")
    
    # Test solver rewards
    solve_tests = [
        (True, 1.0),   # Correct solution
        (False, 0.0),  # Incorrect solution
    ]
    
    solve_passed = 0
    print("  Testing solver rewards:")
    for correct, expected in solve_tests:
        reward = calculator.calculate_solve_reward(correct)
        if abs(reward - expected) < 0.01:
            print(f"    ✅ Solution {'correct' if correct else 'incorrect'} → {reward:.2f} reward")
            solve_passed += 1
        else:
            print(f"    ❌ Solution {'correct' if correct else 'incorrect'} → {reward:.2f} reward (expected {expected:.2f})")
    
    all_passed = propose_passed == len(propose_tests) and solve_passed == len(solve_tests)
    print(f"Reward calculator tests: {'✅ Passed' if all_passed else '❌ Failed'}")
    
    return all_passed


def test_prompt_system():
    """Test prompt management system."""
    print("\nTesting prompt management system...")
    
    # Check if prompt system is available
    try:
        from Prompts.prompt_manager import PromptManager
        from Prompts.solution_prompts import SolutionPrompts
        from Prompts.task_generation_prompts import TaskGenerationPrompts
        from Prompts.validation_prompts import ValidationPrompts
    except ImportError:
        print("  ⚠️ Prompt system not available, skipping tests")
        return True  # Skip this test but don't fail
    
    # Test prompt manager
    try:
        prompt_manager = PromptManager()
        print("  ✅ Prompt manager initialized")
    except Exception as e:
        print(f"  ❌ Prompt manager initialization failed: {e}")
        return False
    
    # Test get_task_generation_prompt
    try:
        task_prompt = prompt_manager.get_task_generation_prompt("deduction", 2)
        print(f"  ✅ Task generation prompt created (length: {len(task_prompt)})")
    except Exception as e:
        print(f"  ❌ Task generation prompt failed: {e}")
        return False
    
    # Test SolutionPrompts
    try:
        task_data = {
            'program': 'lambda x: x * 2',
            'input': '5',
            'expected_output': '10'
        }
        
        solution_prompt = SolutionPrompts.get_deduction_solution_prompt(task_data)
        complexity_prompt = SolutionPrompts.get_complexity_adjusted_prompt("deduction", task_data, 3)
        print(f"  ✅ Solution prompts created successfully")
    except Exception as e:
        print(f"  ❌ Solution prompts failed: {e}")
        return False
    
    # Test ValidationPrompts
    try:
        validation_prompt = ValidationPrompts.get_task_validation_prompt(
            task_type="deduction",
            program="lambda x: x * 2",
            input_data="5"
        )
        print(f"  ✅ Validation prompt created successfully")
    except Exception as e:
        print(f"  ❌ Validation prompt failed: {e}")
        return False
    
    print("Prompt system tests: ✅ Passed")
    return True


def test_basic_model_wrapper():
    """Test basic model wrapper functionality without loading models."""
    print("\nTesting model wrapper (basic functionality)...")
    
    from src.model_wrapper import ModelWrapper
    from src.azr_system import AZRConfig
    
    # Check interface only
    try:
        # Check that the methods exist without actually loading models
        assert hasattr(ModelWrapper, "generate"), "Missing generate method"
        assert hasattr(ModelWrapper, "generate_task"), "Missing generate_task method"
        assert hasattr(ModelWrapper, "generate_solution"), "Missing generate_solution method"
        print("  ✅ Model wrapper interface check passed")
        return True
    except AssertionError as e:
        print(f"  ❌ Model wrapper interface check failed: {e}")
        return False


def test_curriculum_learning():
    """Test curriculum learning features."""
    print("\nTesting curriculum learning...")
    
    from src.task_manager import TaskManager, ReasoningTask
    from src.azr_system import AZRConfig
    
    config = AZRConfig()
    manager = TaskManager(config)
    
    # Test difficulty levels
    initial = manager.difficulty_level
    print(f"  Initial difficulty: {initial}")
    
    # Test difficulty increase
    manager.increase_difficulty(0.5)
    increased = manager.difficulty_level
    print(f"  After increase: {increased}")
    
    # Test difficulty decrease
    manager.decrease_difficulty(0.2)
    decreased = manager.difficulty_level
    print(f"  After decrease: {decreased}")
    
    # Check results
    if initial < increased and decreased < increased:
        print("  ✅ Difficulty adjustment working correctly")
        return True
    else:
        print("  ❌ Difficulty adjustment not working correctly")
        return False


def test_complex_task_validation():
    """Test complex task validation features."""
    print("\nTesting complex task validation...")
    
    from src.code_executor import CodeExecutor
    from src.azr_system import AZRConfig
    
    config = AZRConfig()
    executor = CodeExecutor(config)
    
    test_cases = [
        # Valid programs
        ("lambda x: x * 2", True),
        ("lambda x, y: x + y", True),
        ("lambda x: x if x > 0 else 0", True),
        # Invalid programs
        ("lambda x: x /", False),
        ("lambda x: __import__('os')", False),
        ("lambda x: open('/etc/passwd')", False),
    ]
    
    passed = 0
    for program, expected_valid in test_cases:
        result = executor.validate_program_syntax(program)
        actual_valid = result['valid']
        
        if actual_valid == expected_valid:
            print(f"  ✅ {program}: {'valid' if actual_valid else 'invalid'} as expected")
            passed += 1
        else:
            print(f"  ❌ {program}: got {'valid' if actual_valid else 'invalid'}, " +
                 f"expected {'valid' if expected_valid else 'invalid'}")
    
    print(f"Passed {passed}/{len(test_cases)} validation tests")
    return passed == len(test_cases)


def main():
    """Run all basic tests."""
    print("=" * 50)
    print("AZR System Basic Tests")
    print("=" * 50)
    
    # Track test results
    results = {}
    
    # Run tests
    results["imports"] = test_imports()
    results["code_executor"] = test_code_executor()
    results["task_manager"] = test_task_manager()
    results["reward_calculator"] = test_reward_calculator()
    results["prompt_system"] = test_prompt_system()
    results["basic_model_wrapper"] = test_basic_model_wrapper()
    results["curriculum_learning"] = test_curriculum_learning()
    results["complex_task_validation"] = test_complex_task_validation()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<25}: {status}")
    
    total_passed = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    print("-" * 50)
    print(f"Total: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    print("=" * 50)
    
    # Return overall success/failure
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
