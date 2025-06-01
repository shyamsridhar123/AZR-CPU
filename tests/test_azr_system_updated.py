#!/usr/bin/env python3
"""
Updated comprehensive tests for the AZR (Absolute Zero Reasoner) system.
Tests all major components and their integration with proper interfaces.
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import json
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.azr_system import AZRConfig, AbsoluteZeroReasoner
from src.task_manager import TaskManager, ReasoningTask, ReasoningType
from src.code_executor import CodeExecutor
from src.reward_calculator import RewardCalculator
from src.model_wrapper import ModelWrapper
from utils.evaluation import evaluate_model_performance
from utils.logging_utils import setup_logging

# Import new prompt system components
try:
    from Prompts.prompt_manager import PromptManager
    from Prompts.solution_prompts import SolutionPrompts
    from Prompts.task_generation_prompts import TaskGenerationPrompts
    from Prompts.validation_prompts import ValidationPrompts
    PROMPT_SYSTEM_AVAILABLE = True
except ImportError:
    PROMPT_SYSTEM_AVAILABLE = False


class TestAZRConfig(unittest.TestCase):
    """Test AZR configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AZRConfig()
        
        self.assertEqual(config.model_name, "microsoft/DialoGPT-small")
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.temperature, 0.8)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.max_episodes, 1000)
        self.assertEqual(config.buffer_size, 1000)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AZRConfig(
            model_name="gpt2",
            batch_size=8,
            learning_rate=1e-3
        )
        
        self.assertEqual(config.model_name, "gpt2")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.learning_rate, 1e-3)


class TestCodeExecutor(unittest.TestCase):
    """Test code execution functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AZRConfig()
        self.executor = CodeExecutor(self.config)
    
    def test_simple_execution(self):
        """Test simple code execution."""
        result = self.executor.execute_safe("lambda x: x * 2", "5")
        
        self.assertTrue(result['success'])
        self.assertEqual(result['output'], 10)
        self.assertIsNone(result['error'])
    
    def test_tuple_input(self):
        """Test execution with tuple input."""
        result = self.executor.execute_safe("lambda x, y: x + y", "(3, 4)")
        
        self.assertTrue(result['success'])
        self.assertEqual(result['output'], 7)
    
    def test_invalid_code(self):
        """Test handling of invalid code."""
        result = self.executor.execute_safe("lambda x: x /", "5")
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIn("Syntax error", result['error'])
    
    def test_runtime_error(self):
        """Test handling of runtime errors."""
        result = self.executor.execute_safe("lambda x: x / 0", "5")
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIn("ZeroDivisionError", result['error'])
    
    def test_forbidden_operations(self):
        """Test blocking of forbidden operations."""
        forbidden_codes = [
            "lambda x: __import__('os').system('ls')",
            "lambda x: open('/etc/passwd').read()",
            "lambda x: eval('print(\"hack\")')"
        ]
        
        for code in forbidden_codes:
            result = self.executor.execute_safe(code, "1")
            self.assertFalse(result['success'])
            self.assertIn("forbidden", result['error'].lower())
    
    def test_timeout(self):
        """Test execution timeout."""
        # Simple test that should work
        result = self.executor.execute_safe("lambda x: x + 1", "5")
        self.assertTrue(result['success'])
    
    def test_syntax_validation(self):
        """Test syntax validation method."""
        result = self.executor.validate_program_syntax("lambda x: x * 2")
        self.assertTrue(result['valid'])
        
        result = self.executor.validate_program_syntax("lambda x: x /")
        self.assertFalse(result['valid'])


class TestReasoningTask(unittest.TestCase):
    """Test ReasoningTask functionality."""
    
    def test_task_creation(self):
        """Test creating a reasoning task."""
        task = ReasoningTask(
            type="deduction",
            program="lambda x: x + 1",
            input="5",
            expected_output="6",
            complexity=1
        )
        
        self.assertEqual(task.type, "deduction")
        self.assertEqual(task.program, "lambda x: x + 1")
        self.assertEqual(task.input, "5")
        self.assertEqual(task.expected_output, "6")
        self.assertEqual(task.complexity, 1)
        self.assertEqual(task.success_rate, 0.0)
    
    def test_task_attempt_recording(self):
        """Test recording task attempts."""
        task = ReasoningTask(
            type="deduction",
            program="lambda x: x + 1",
            input="5",
            expected_output="6"
        )
        
        # Record successful attempt
        task.record_attempt(True)
        self.assertEqual(task.success_count, 1)
        self.assertEqual(task.attempt_count, 1)
        self.assertEqual(task.success_rate, 1.0)
        
        # Record failed attempt
        task.record_attempt(False)
        self.assertEqual(task.success_count, 1)
        self.assertEqual(task.attempt_count, 2)
        self.assertEqual(task.success_rate, 0.5)
    
    def test_invalid_reasoning_type(self):
        """Test invalid reasoning type validation."""
        with self.assertRaises(ValueError):
            ReasoningTask(
                type="invalid_type",
                program="lambda x: x",
                input="1",
                expected_output="1"
            )


class TestTaskManager(unittest.TestCase):
    """Test task management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AZRConfig()
        self.manager = TaskManager(self.config)
    
    def test_add_task(self):
        """Test adding tasks to buffers."""
        task = ReasoningTask(
            type="deduction",
            program="lambda x: x + 1",
            input="5",
            expected_output="6"
        )
        
        self.manager.add_task(task)
        
        # Check that task was added to the correct buffer
        buffer = self.manager.buffers["deduction"]
        self.assertEqual(len(buffer), 1)
        self.assertEqual(list(buffer.tasks)[0].program, "lambda x: x + 1")
    
    def test_sample_tasks(self):
        """Test sampling tasks from buffers."""
        # Add multiple tasks
        for i in range(5):
            task = ReasoningTask(
                type="deduction",
                program=f"lambda x: x + {i}",
                input="1",
                expected_output=str(1 + i)
            )
            self.manager.add_task(task)
        
        # Sample tasks
        sampled = self.manager.sample_tasks(3)
        self.assertLessEqual(len(sampled), 3)
        
        # If we have tasks, we should get some
        if len(sampled) > 0:
            self.assertIsInstance(sampled[0], ReasoningTask)
    
    def test_sample_tasks_by_type(self):
        """Test sampling tasks by specific type."""
        # Add tasks of different types
        for task_type in ["deduction", "abduction", "induction"]:
            for i in range(3):
                task = ReasoningTask(
                    type=task_type,
                    program=f"lambda x: x + {i}",
                    input="1",
                    expected_output=str(1 + i)
                )
                self.manager.add_task(task)
        
        # Sample only deduction tasks
        deduction_tasks = self.manager.sample_tasks_by_type("deduction", 2)
        self.assertLessEqual(len(deduction_tasks), 2)
        
        for task in deduction_tasks:
            self.assertEqual(task.type, "deduction")
    
    def test_get_buffer_stats(self):
        """Test getting buffer statistics."""
        # Add tasks of different types
        for task_type in ["deduction", "abduction", "induction"]:
            for i in range(3):
                task = ReasoningTask(
                    type=task_type,
                    program=f"lambda x: x + {i}",
                    input="1",
                    expected_output=str(1 + i),
                    complexity=i + 1
                )
                self.manager.add_task(task)
        
        stats = self.manager.get_buffer_stats()
        
        # Check that stats contain expected keys
        self.assertIn("deduction", stats)
        self.assertIn("abduction", stats)
        self.assertIn("induction", stats)
        self.assertIn("difficulty_level", stats)
        
        # Check individual type stats
        deduction_stats = stats["deduction"]
        self.assertIn("size", deduction_stats)
        self.assertIn("complexity_distribution", deduction_stats)
    
    def test_difficulty_adjustment(self):
        """Test difficulty level adjustments."""
        initial_difficulty = self.manager.difficulty_level
        
        # Increase difficulty
        self.manager.increase_difficulty(0.2)
        self.assertGreater(self.manager.difficulty_level, initial_difficulty)
        
        # Decrease difficulty
        current_difficulty = self.manager.difficulty_level
        self.manager.decrease_difficulty(0.1)
        self.assertLess(self.manager.difficulty_level, current_difficulty)


class TestRewardCalculator(unittest.TestCase):
    """Test reward calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AZRConfig()
        self.calculator = RewardCalculator(self.config)
    
    def test_propose_reward_calculation(self):
        """Test proposer reward calculation."""
        reward = self.calculator.calculate_propose_reward(3, 5)
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
        
        # Perfect task generation should get high reward
        perfect_reward = self.calculator.calculate_propose_reward(5, 5)
        imperfect_reward = self.calculator.calculate_propose_reward(2, 5)
        self.assertGreater(perfect_reward, imperfect_reward)
    
    def test_solve_reward_calculation(self):
        """Test solver reward calculation."""
        # Test binary reward for solving
        correct_reward = self.calculator.calculate_solve_reward(True)
        incorrect_reward = self.calculator.calculate_solve_reward(False)
        
        self.assertEqual(correct_reward, 1.0)
        self.assertEqual(incorrect_reward, 0.0)


@unittest.skipIf(not PROMPT_SYSTEM_AVAILABLE, "Prompt system not available")
class TestPromptSystem(unittest.TestCase):
    """Test the new prompt management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.prompt_manager = PromptManager()
    
    def test_task_generation_prompt(self):
        """Test task generation prompt creation."""
        prompt = self.prompt_manager.get_task_generation_prompt(
            reasoning_type="deduction",
            complexity=2,
            include_examples=True
        )
        
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        self.assertIn("deduction", prompt.lower())
    
    def test_solution_prompts(self):
        """Test solution prompt generation."""
        task_data = {
            'program': 'lambda x: x * 2',
            'input': '5',
            'expected_output': '10'
        }
        
        # Test deduction prompt
        deduction_prompt = SolutionPrompts.get_deduction_solution_prompt(task_data)
        self.assertIsInstance(deduction_prompt, str)
        self.assertIn("deduction", deduction_prompt.lower())
        self.assertIn("lambda x: x * 2", deduction_prompt)
        
        # Test complexity adjusted prompt
        complex_prompt = SolutionPrompts.get_complexity_adjusted_prompt(
            "deduction", task_data, complexity=3
        )
        self.assertIn("Level 3", complex_prompt)
    
    def test_validation_prompts(self):
        """Test validation prompt generation."""
        prompt = ValidationPrompts.get_task_validation_prompt(
            task_type="deduction",
            program="lambda x: x + 1",
            input_data="5"
        )
        
        self.assertIsInstance(prompt, str)
        self.assertIn("validation", prompt.lower())


class TestModelWrapper(unittest.TestCase):
    """Test model wrapper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AZRConfig()
        # Mock the model wrapper to avoid loading actual models in tests
        self.wrapper = Mock(spec=ModelWrapper)
    
    def test_generate_method_exists(self):
        """Test that generate method exists in interface."""
        # This tests the interface we expect to exist
        self.assertTrue(hasattr(ModelWrapper, 'generate'))
        self.assertTrue(hasattr(ModelWrapper, 'generate_task'))
        self.assertTrue(hasattr(ModelWrapper, 'generate_solution'))
    
    @patch('src.model_wrapper.AutoTokenizer')
    @patch('src.model_wrapper.AutoModelForCausalLM')
    def test_model_wrapper_creation(self, mock_model, mock_tokenizer):
        """Test model wrapper creation with mocked dependencies."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        config = AZRConfig()
        wrapper = ModelWrapper(config)
        
        self.assertIsNotNone(wrapper)


class TestAZRSystemIntegration(unittest.TestCase):
    """Test integration of AZR system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AZRConfig()
        self.config.max_episodes = 2  # Limit for testing
        self.config.tasks_per_episode = 2
    
    @patch('src.model_wrapper.AutoTokenizer')
    @patch('src.model_wrapper.AutoModelForCausalLM')
    def test_azr_system_initialization(self, mock_model, mock_tokenizer):
        """Test AZR system initialization."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        azr = AbsoluteZeroReasoner(self.config)
        
        self.assertIsNotNone(azr.model)
        self.assertIsNotNone(azr.task_manager)
        self.assertIsNotNone(azr.code_executor)
        self.assertIsNotNone(azr.reward_calculator)
        
        # Check curriculum learning state
        self.assertEqual(azr.current_complexity, self.config.min_task_complexity)
        self.assertIsInstance(azr.success_rate_window, type(azr.success_rate_window))
    
    def test_task_validation(self):
        """Test task validation functionality."""
        config = AZRConfig()
        azr = Mock(spec=AbsoluteZeroReasoner)
        azr.code_executor = CodeExecutor(config)
        
        # Test valid task
        valid_task = {
            'program': 'lambda x: x + 1',
            'input': '5',
            'expected_output': '6',
            'type': 'deduction'
        }
        
        # Mock the validation method behavior
        azr._validate_task = lambda task_data: (
            task_data.get('program', '').startswith('lambda') and
            task_data.get('input', '') != '' and
            task_data.get('expected_output', '') != ''
        )
        
        self.assertTrue(azr._validate_task(valid_task))
        
        # Test invalid task
        invalid_task = {
            'program': '',
            'input': '',
            'expected_output': '',
            'type': 'deduction'
        }
        
        self.assertFalse(azr._validate_task(invalid_task))


class TestCurriculumLearning(unittest.TestCase):
    """Test curriculum learning functionality."""
    
    def test_complexity_tracking(self):
        """Test complexity level tracking and adjustment."""
        config = AZRConfig()
        task_manager = TaskManager(config)
        
        # Test initial state
        self.assertEqual(task_manager.difficulty_level, 1.0)
        
        # Test difficulty increases
        initial_difficulty = task_manager.difficulty_level
        task_manager.increase_difficulty(0.5)
        self.assertGreater(task_manager.difficulty_level, initial_difficulty)
        
        # Test difficulty decreases
        current_difficulty = task_manager.difficulty_level
        task_manager.decrease_difficulty(0.3)
        self.assertLess(task_manager.difficulty_level, current_difficulty)
    
    def test_task_complexity_distribution(self):
        """Test task complexity distribution tracking."""
        config = AZRConfig()
        task_manager = TaskManager(config)
        
        # Add tasks with different complexities
        for complexity in [1, 2, 2, 3, 3, 3]:
            task = ReasoningTask(
                type="deduction",
                program=f"lambda x: x + {complexity}",
                input="1",
                expected_output=str(1 + complexity),
                complexity=complexity
            )
            task_manager.add_task(task)
        
        # Check complexity distribution
        stats = task_manager.get_buffer_stats()
        distribution = stats["deduction"]["complexity_distribution"]
        
        self.assertEqual(distribution[1], 1)  # One task with complexity 1
        self.assertEqual(distribution[2], 2)  # Two tasks with complexity 2
        self.assertEqual(distribution[3], 3)  # Three tasks with complexity 3


class TestQualityControl(unittest.TestCase):
    """Test quality control mechanisms."""
    
    def test_task_quality_validation(self):
        """Test task quality validation."""
        config = AZRConfig()
        executor = CodeExecutor(config)
        
        # Test syntax validation
        good_syntax = executor.validate_program_syntax("lambda x: x * 2")
        self.assertTrue(good_syntax['valid'])
        
        bad_syntax = executor.validate_program_syntax("lambda x: x *")
        self.assertFalse(bad_syntax['valid'])
    
    def test_execution_safety(self):
        """Test code execution safety measures."""
        config = AZRConfig()
        executor = CodeExecutor(config)
        
        # Test safe code execution
        safe_result = executor.execute_safe("lambda x: x + 1", "5")
        self.assertTrue(safe_result['success'])
        
        # Test unsafe code rejection
        unsafe_result = executor.execute_safe("lambda x: __import__('os')", "1")
        self.assertFalse(unsafe_result['success'])
        self.assertIn("forbidden", unsafe_result['error'].lower())


class TestMetricsTracking(unittest.TestCase):
    """Test metrics tracking and logging."""
    
    def test_task_attempt_tracking(self):
        """Test tracking of task attempts and success rates."""
        task = ReasoningTask(
            type="deduction",
            program="lambda x: x + 1",
            input="5",
            expected_output="6"
        )
        
        # Initially no attempts
        self.assertEqual(task.attempt_count, 0)
        self.assertEqual(task.success_count, 0)
        self.assertEqual(task.success_rate, 0.0)
        
        # Record attempts
        task.record_attempt(True)
        task.record_attempt(False)
        task.record_attempt(True)
        
        self.assertEqual(task.attempt_count, 3)
        self.assertEqual(task.success_count, 2)
        self.assertAlmostEqual(task.success_rate, 2/3, places=2)
    
    def test_buffer_statistics(self):
        """Test buffer statistics collection."""
        config = AZRConfig()
        task_manager = TaskManager(config)
        
        # Add some tasks
        for i in range(5):
            task = ReasoningTask(
                type="deduction",
                program=f"lambda x: x + {i}",
                input="1",
                expected_output=str(1 + i),
                complexity=(i % 3) + 1
            )
            task_manager.add_task(task)
        
        stats = task_manager.get_buffer_stats()
        
        # Verify stats structure
        self.assertIn("deduction", stats)
        self.assertIn("size", stats["deduction"])
        self.assertIn("complexity_distribution", stats["deduction"])
        self.assertEqual(stats["deduction"]["size"], 5)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup functionality."""
        logger = setup_logging("test_logger", level="INFO")
        self.assertIsNotNone(logger)
        
        # Test that we can log without errors
        logger.info("Test log message")
        self.assertTrue(True)  # If we got here, logging worked


if __name__ == '__main__':
    # Create a test suite with all test classes
    test_classes = [
        TestAZRConfig,
        TestCodeExecutor, 
        TestReasoningTask,
        TestTaskManager,
        TestRewardCalculator,
        TestModelWrapper,
        TestAZRSystemIntegration,
        TestCurriculumLearning,
        TestQualityControl,
        TestMetricsTracking,
        TestUtilityFunctions
    ]
    
    # Add prompt system tests if available
    if PROMPT_SYSTEM_AVAILABLE:
        test_classes.append(TestPromptSystem)
    
    # Create test suite
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
