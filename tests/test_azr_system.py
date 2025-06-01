#!/usr/bin/env python3
"""
Comprehensive tests for the AZR (Absolute Zero Reasoner) system.
Tests all major components and their integration.
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.azr_system import AZRConfig, AbsoluteZeroReasoner
from src.task_manager import TaskManager, ReasoningTask, ReasoningType
from src.code_executor import CodeExecutor
from src.reward_calculator import RewardCalculator
from src.model_wrapper import ModelWrapper
from utils.evaluation import evaluate_model_performance
from utils.logging_utils import setup_logging


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
        result = self.executor.execute("lambda x: x * 2", "5")
        
        self.assertTrue(result['success'])
        self.assertEqual(result['output'], '10')
        self.assertIsNone(result['error'])
    
    def test_tuple_input(self):
        """Test execution with tuple input."""
        result = self.executor.execute("lambda x, y: x + y", "(3, 4)")
        
        self.assertTrue(result['success'])
        self.assertEqual(result['output'], '7')
    
    def test_invalid_code(self):
        """Test handling of invalid code."""
        result = self.executor.execute("lambda x: x /", "5")
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIn("SyntaxError", result['error'])
    
    def test_runtime_error(self):
        """Test handling of runtime errors."""
        result = self.executor.execute("lambda x: x / 0", "5")
        
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
            result = self.executor.execute(code, "1")
            self.assertFalse(result['success'])
            self.assertIn("forbidden", result['error'].lower())
    
    def test_timeout(self):
        """Test execution timeout."""
        # Code that would run forever
        result = self.executor.execute("lambda x: [i for i in iter(int, 1)]", "1")
        
        self.assertFalse(result['success'])
        self.assertIn("timeout", result['error'].lower())
    
    def test_memory_limit(self):
        """Test memory limit enforcement."""
        # Code that would use too much memory
        result = self.executor.execute(
            f"lambda x: [0] * {self.config.max_memory_mb * 1024 * 1024}", 
            "1"
        )
        
        self.assertFalse(result['success'])


class TestTaskManager(unittest.TestCase):
    """Test task management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AZRConfig()
        self.manager = TaskManager(self.config)
    
    def test_add_task(self):
        """Test adding tasks to buffers."""
        task = ReasoningTask(
            task_type=ReasoningType.DEDUCTION,
            program="lambda x: x + 1",
            input_data="5",
            expected_output="6"
        )
        
        self.manager.add_task(task)
        
        buffer = self.manager.get_buffer(ReasoningType.DEDUCTION)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer[0].program, "lambda x: x + 1")
    
    def test_sample_tasks(self):
        """Test sampling tasks from buffers."""
        # Add multiple tasks
        for i in range(10):
            task = ReasoningTask(
                task_type=ReasoningType.DEDUCTION,
                program=f"lambda x: x + {i}",
                input_data="1",
                expected_output=str(1 + i)
            )
            self.manager.add_task(task)
        
        # Sample tasks
        sampled = self.manager.sample_tasks(ReasoningType.DEDUCTION, n=5)
        
        self.assertEqual(len(sampled), 5)
        self.assertTrue(all(isinstance(t, ReasoningTask) for t in sampled))
    
    def test_buffer_size_limit(self):
        """Test buffer size limitation."""
        # Add more tasks than buffer size
        for i in range(self.config.buffer_size + 100):
            task = ReasoningTask(
                task_type=ReasoningType.DEDUCTION,
                program=f"lambda x: x + {i}",
                input_data="1",
                expected_output=str(1 + i)
            )
            self.manager.add_task(task)
        
        buffer = self.manager.get_buffer(ReasoningType.DEDUCTION)
        self.assertEqual(len(buffer), self.config.buffer_size)
    
    def test_get_statistics(self):
        """Test task statistics."""
        # Add tasks of different types
        for task_type in ReasoningType:
            for i in range(5):
                task = ReasoningTask(
                    task_type=task_type,
                    program=f"lambda x: x + {i}",
                    input_data="1",
                    expected_output=str(1 + i)
                )
                self.manager.add_task(task)
        
        stats = self.manager.get_statistics()
        
        self.assertEqual(stats['total_tasks'], 15)
        self.assertEqual(stats['deduction_tasks'], 5)
        self.assertEqual(stats['abduction_tasks'], 5)
        self.assertEqual(stats['induction_tasks'], 5)


class TestRewardCalculator(unittest.TestCase):
    """Test reward calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AZRConfig()
        self.calculator = RewardCalculator(self.config)
    
    def test_proposer_reward(self):
        """Test proposer reward calculation."""
        # Optimal learnability (50% success rate)
        reward = self.calculator.calculate_proposer_reward(0.5)
        self.assertAlmostEqual(reward, 1.0, places=2)
        
        # Too easy (100% success rate)
        reward = self.calculator.calculate_proposer_reward(1.0)
        self.assertAlmostEqual(reward, 0.5, places=2)
        
        # Too hard (0% success rate)
        reward = self.calculator.calculate_proposer_reward(0.0)
        self.assertAlmostEqual(reward, 0.5, places=2)
    
    def test_solver_reward(self):
        """Test solver reward calculation."""
        # Correct solution
        reward = self.calculator.calculate_solver_reward(True, 1.5)
        self.assertEqual(reward, 1.0)
        
        # Incorrect solution
        reward = self.calculator.calculate_solver_reward(False, 1.5)
        self.assertEqual(reward, 0.0)
    
    def test_combined_reward(self):
        """Test combined reward calculation."""
        proposer_reward = 0.8
        solver_reward = 1.0
        
        combined = self.calculator.calculate_combined_reward(
            proposer_reward, 
            solver_reward
        )
        
        # Should be weighted average
        expected = (proposer_reward + solver_reward) / 2
        self.assertAlmostEqual(combined, expected, places=2)
    
    def test_advantage_normalization(self):
        """Test advantage normalization."""
        rewards = [0.0, 0.5, 1.0, 0.3, 0.7]
        
        normalized = self.calculator.normalize_advantages(rewards)
        
        # Check mean is close to 0 and std is close to 1
        import numpy as np
        self.assertAlmostEqual(np.mean(normalized), 0.0, places=1)
        self.assertAlmostEqual(np.std(normalized), 1.0, places=1)


class TestModelWrapper(unittest.TestCase):
    """Test model wrapper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AZRConfig()
        # Mock the model loading to avoid downloading
        with patch('src.model_wrapper.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.model_wrapper.AutoModelForCausalLM.from_pretrained') as mock_model:
            
            # Create mock tokenizer
            self.mock_tokenizer = MagicMock()
            self.mock_tokenizer.pad_token = None
            self.mock_tokenizer.eos_token = "[EOS]"
            self.mock_tokenizer.pad_token_id = 0
            self.mock_tokenizer.eos_token_id = 1
            mock_tokenizer.return_value = self.mock_tokenizer
            
            # Create mock model
            self.mock_model = MagicMock()
            mock_model.return_value = self.mock_model
            
            self.wrapper = ModelWrapper(self.config)
    
    def test_initialization(self):
        """Test model wrapper initialization."""
        self.assertIsNotNone(self.wrapper.model)
        self.assertIsNotNone(self.wrapper.tokenizer)
        self.assertIsNotNone(self.wrapper.optimizer)
        self.assertEqual(self.wrapper.tokenizer.pad_token, "[EOS]")
    
    def test_parse_generated_task(self):
        """Test parsing of generated task text."""
        generated_text = """
        lambda x: x * 2
        input: 5
        output: 10
        """
        
        task = self.wrapper._parse_generated_task(generated_text, 'deduction')
        
        self.assertEqual(task['type'], 'deduction')
        self.assertEqual(task['program'], 'lambda x: x * 2')
        self.assertEqual(task['input'], '5')
        self.assertEqual(task['expected_output'], '10')
    
    def test_clean_generated_solution(self):
        """Test cleaning of generated solutions."""
        # Test with explanation text
        solution = "The answer is:\nlambda x: x + 1\nThis adds 1 to x."
        cleaned = self.wrapper._clean_generated_solution(solution)
        self.assertEqual(cleaned, "lambda x: x + 1")
        
        # Test with quotes
        solution = '"42"'
        cleaned = self.wrapper._clean_generated_solution(solution)
        self.assertEqual(cleaned, "42")
    
    def test_get_model_size(self):
        """Test model size calculation."""
        # Mock model parameters
        param1 = MagicMock()
        param1.numel.return_value = 1000
        param1.requires_grad = True
        
        param2 = MagicMock()
        param2.numel.return_value = 500
        param2.requires_grad = False
        
        self.mock_model.parameters.return_value = [param1, param2]
        
        size_info = self.wrapper.get_model_size()
        
        self.assertEqual(size_info['total_parameters'], 1500)
        self.assertEqual(size_info['trainable_parameters'], 1000)
        self.assertAlmostEqual(size_info['model_size_mb'], 1500 * 4 / (1024 * 1024), places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete AZR system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AZRConfig(
            checkpoint_dir=self.temp_dir,
            max_episodes=2,
            tasks_per_episode=2,
            batch_size=1
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.model_wrapper.AutoTokenizer.from_pretrained')
    @patch('src.model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_azr_system_initialization(self, mock_model, mock_tokenizer):
        """Test AZR system initialization."""
        # Mock the model and tokenizer
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        azr = AbsoluteZeroReasoner(self.config)
        
        self.assertIsNotNone(azr.model_wrapper)
        self.assertIsNotNone(azr.task_manager)
        self.assertIsNotNone(azr.code_executor)
        self.assertIsNotNone(azr.reward_calculator)
    
    def test_seed_tasks(self):
        """Test seed task initialization."""
        config = AZRConfig()
        
        # Check that seed tasks are properly defined
        self.assertEqual(len(config.seed_tasks), 3)
        
        for task in config.seed_tasks:
            self.assertIn('type', task)
            self.assertIn('program', task)
            self.assertIn('input', task)
            self.assertIn('expected_output', task)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = setup_logging("test_logger", log_file)
            
            self.assertIsNotNone(logger)
            
            # Test logging
            logger.info("Test message")
            
            # Check log file exists
            self.assertTrue(os.path.exists(log_file))
            
            # Check log content
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Test message", content)


def run_tests():
    """Run all tests with a nice output format."""
    print("=" * 70)
    print("🧪 Running AZR System Tests")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAZRConfig,
        TestCodeExecutor,
        TestTaskManager,
        TestRewardCalculator,
        TestModelWrapper,
        TestIntegration,
        TestUtils
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    print("=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[-2]}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)