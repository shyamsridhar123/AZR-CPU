"""
Absolute Zero Reasoner (AZR) - Core System Implementation
A self-contained system where a language model bootstraps reasoning capabilities through self-generated tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import json
import time
from tqdm import tqdm
from pathlib import Path

try:
    # Try relative imports first (when used as package)
    from .task_manager import TaskManager, ReasoningTask
    from .code_executor import CodeExecutor
    from .model_wrapper import ModelWrapper
    from .reward_calculator import RewardCalculator
except ImportError:
    # Fall back to absolute imports (when used directly)
    from task_manager import TaskManager, ReasoningTask
    from code_executor import CodeExecutor
    from model_wrapper import ModelWrapper
    from reward_calculator import RewardCalculator


@dataclass
class AZRConfig:
    """Configuration for the Absolute Zero Reasoner system."""
    # Model configuration
    model_name: str = "microsoft/DialoGPT-small"  # Lightweight for CPU
    max_length: int = 512
    temperature: float = 0.8
    use_local_models: bool = True  # Try to use locally cached models first
    save_models_locally: bool = True  # Save downloaded models to local cache
    
    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 4  # Small batch for CPU
    max_episodes: int = 1000
    buffer_size: int = 1000
    
    # Model saving configuration
    save_frequency: int = 50  # Save model every N episodes
    keep_best_models: int = 5  # Keep top N performing models
    auto_save_enabled: bool = True
    
    # Task generation
    max_task_complexity: int = 5
    min_task_complexity: int = 1
    tasks_per_episode: int = 10
    
    # Execution safety
    execution_timeout: float = 5.0
    max_memory_mb: int = 100
    
    # Curriculum learning
    success_rate_target: float = 0.5
    difficulty_adjustment_rate: float = 0.1
    
    # Logging
    log_level: str = "INFO"
    save_interval: int = 100


class AbsoluteZeroReasoner:
    """
    Main AZR system that implements self-bootstrapping reasoning through
    reinforcement learning with self-generated tasks.
    """
    
    def __init__(self, config: AZRConfig):
        self.config = config
        self.setup_logging()        # Initialize components
        self.model = ModelWrapper(config)
        self.task_manager = TaskManager(config)
        self.code_executor = CodeExecutor(config)
        self.reward_calculator = RewardCalculator(config)
        
        # Training state
        self.episode = 0
        self.total_tasks_generated = 0
        self.total_tasks_solved = 0
        
        # Metrics tracking
        self.metrics = {
            'propose_rewards': [],
            'solve_rewards': [],
            'success_rates': [],
            'task_complexity': [],
            'episode_times': []
        }
        
        self.logger.info("Absolute Zero Reasoner initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamp for log file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"azr_training_{timestamp}.log"
        
        # Configure logging to both file and console
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # Override existing configuration
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to file: {log_file}")
    
    def train(self):
        """
        Main training loop implementing the dual-phase TRR++ algorithm.
        """
        self.logger.info("Starting AZR training...")
        
        # Initialize with seed tasks
        self._initialize_seed_tasks()
        
        for episode in tqdm(range(self.config.max_episodes), desc="Training Episodes"):
            self.episode = episode
            episode_start_time = time.time()
            
            # Phase 1: PROPOSE - Generate new tasks
            propose_metrics = self._propose_phase()
            
            # Phase 2: SOLVE - Attempt to solve tasks
            solve_metrics = self._solve_phase()
            
            # Update model using combined rewards
            self._update_model(propose_metrics, solve_metrics)
            
            # Update curriculum difficulty
            self._update_curriculum(solve_metrics['success_rate'])
            
            # Log progress
            episode_time = time.time() - episode_start_time
            self._log_episode_metrics(propose_metrics, solve_metrics, episode_time)            # Save checkpoint, trained model, and tasks
            if episode % self.config.save_interval == 0:
                self._save_checkpoint(episode)
                self._save_trained_model(episode)
                self.task_manager.save_tasks_to_file(episode)
            
            # Save best model if this is the best episode
            if hasattr(self, 'best_episode') and episode == self.best_episode:
                self._save_trained_model(episode, force_save=True)
        
        # Save final model
        self._save_trained_model(self.config.max_episodes - 1, force_save=True)
        self.logger.info("Training completed!")
    
    def _initialize_seed_tasks(self):
        """Initialize the system with minimal seed tasks."""
        seed_tasks = [
            # Identity function
            {
                'type': 'deduction',
                'program': 'lambda x: x',
                'input': '5',
                'expected_output': '5',
                'complexity': 1
            },
            # Simple arithmetic
            {
                'type': 'deduction', 
                'program': 'lambda x, y: x + y',
                'input': '(3, 4)',
                'expected_output': '7',
                'complexity': 1
            },
            # Basic conditional
            {
                'type': 'abduction',
                'program': 'lambda x: x if x > 0 else 0',
                'input': '5',
                'expected_output': '5',
                'complexity': 2
            }
        ]
        
        for task_data in seed_tasks:
            task = ReasoningTask(**task_data)
            self.task_manager.add_task(task)
        
        self.logger.info(f"Initialized with {len(seed_tasks)} seed tasks")
    
    def _propose_phase(self) -> Dict[str, Any]:
        """
        PROPOSE phase: Generate new reasoning tasks using the current model.
        """
        proposed_tasks = []
        valid_tasks = 0
        
        for _ in range(self.config.tasks_per_episode):
            # Sample reasoning type
            reasoning_type = random.choice(['deduction', 'abduction', 'induction'])
            
            # Generate task using model
            task_prompt = self._create_task_prompt(reasoning_type)
            generated_task = self.model.generate_task(task_prompt)
            
            # Validate task through code execution
            if self._validate_task(generated_task):
                task = ReasoningTask(
                    type=reasoning_type,
                    program=generated_task['program'],
                    input=generated_task.get('input', ''),
                    expected_output=generated_task.get('output', ''),
                    complexity=self._estimate_complexity(generated_task['program'])
                )
                
                self.task_manager.add_task(task)
                proposed_tasks.append(task)
                valid_tasks += 1
                self.total_tasks_generated += 1
        
        # Calculate proposer reward
        propose_reward = self.reward_calculator.calculate_propose_reward(
            valid_tasks, self.config.tasks_per_episode
        )
        
        return {
            'valid_tasks': valid_tasks,
            'total_proposed': self.config.tasks_per_episode,
            'propose_reward': propose_reward,
            'tasks': proposed_tasks
        }
    
    def _solve_phase(self) -> Dict[str, Any]:
        """
        SOLVE phase: Attempt to solve tasks from the buffer.
        """
        tasks_to_solve = self.task_manager.sample_tasks(self.config.batch_size)
        solved_tasks = 0
        solve_rewards = []
        
        for task in tasks_to_solve:
            # Generate solution using model
            solution_prompt = self._create_solution_prompt(task)
            generated_solution = self.model.generate_solution(solution_prompt)
            
            # Verify solution through execution
            is_correct = self._verify_solution(task, generated_solution)
            
            if is_correct:
                solved_tasks += 1
                self.total_tasks_solved += 1
            
            # Calculate individual solve reward
            solve_reward = 1.0 if is_correct else 0.0
            solve_rewards.append(solve_reward)
        
        success_rate = solved_tasks / len(tasks_to_solve) if tasks_to_solve else 0.0
        
        return {
            'solved_tasks': solved_tasks,
            'total_tasks': len(tasks_to_solve),
            'success_rate': success_rate,
            'solve_rewards': solve_rewards,
            'average_solve_reward': np.mean(solve_rewards) if solve_rewards else 0.0
        }
    
    def _validate_task(self, task_data: Dict) -> bool:
        """Validate a generated task through code execution."""
        try:
            if 'program' not in task_data:
                return False
            
            program = task_data['program']
            input_data = task_data.get('input', '')
            
            # Basic syntax validation
            result = self.code_executor.execute_safe(program, input_data)
            return result['success']
            
        except Exception as e:
            self.logger.debug(f"Task validation failed: {e}")
            return False
    
    def _verify_solution(self, task: ReasoningTask, solution: str) -> bool:
        """Verify if a solution correctly solves the given task."""
        try:
            if task.type == 'deduction':
                # For deduction: run program with input and check output
                result = self.code_executor.execute_safe(task.program, task.input)
                if not result['success']:
                    return False
                expected = result['output']
                
                # Check if solution produces same output
                solution_result = self.code_executor.execute_safe(solution, task.input)
                return (solution_result['success'] and 
                       str(solution_result['output']) == str(expected))
            
            elif task.type == 'abduction':
                # For abduction: check if solution produces expected output
                result = self.code_executor.execute_safe(solution, task.input)
                return (result['success'] and 
                       str(result['output']) == str(task.expected_output))
            
            elif task.type == 'induction':
                # For induction: check against multiple I/O examples
                # Simplified: check single example for now
                result = self.code_executor.execute_safe(solution, task.input)
                return (result['success'] and 
                       str(result['output']) == str(task.expected_output))
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Solution verification failed: {e}")
            return False
    
    def _create_task_prompt(self, reasoning_type: str) -> str:
        """Create a prompt for task generation."""
        prompts = {
            'deduction': "Generate a simple Python function and input. Format: {'program': 'lambda x: ...', 'input': '...'}",
            'abduction': "Generate a Python function input and expected output. Format: {'input': '...', 'output': '...'}",
            'induction': "Generate input-output examples for pattern learning. Format: {'input': '...', 'output': '...'}"
        }
        return prompts.get(reasoning_type, prompts['deduction'])
    
    def _create_solution_prompt(self, task: ReasoningTask) -> str:
        """Create a prompt for solution generation."""
        if task.type == 'deduction':
            return f"Given program: {task.program} and input: {task.input}, what is the output?"
        elif task.type == 'abduction':
            return f"What program with input {task.input} produces output {task.expected_output}?"
        elif task.type == 'induction':
            return f"What program maps input {task.input} to output {task.expected_output}?"
        return ""
    
    def _estimate_complexity(self, program: str) -> int:
        """Estimate the complexity of a program."""
        # Simple heuristic based on program length and constructs
        complexity = 1
        complexity += len(program) // 20  # Length factor
        complexity += program.count('if')  # Conditionals
        complexity += program.count('for')  # Loops
        complexity += program.count('while')  # Loops
        complexity += program.count('lambda')  # Function definitions
        
        return min(complexity, self.config.max_task_complexity)
    
    def _update_model(self, propose_metrics: Dict, solve_metrics: Dict):
        """Update model using TRR++ algorithm with dual rewards."""
        # Combine propose and solve rewards
        total_reward = (propose_metrics['propose_reward'] +
                        solve_metrics['average_solve_reward'])

        # Update model (simplified for demonstration)
        self.model.update_weights(total_reward)
    
    def _update_curriculum(self, success_rate: float):
        """Update curriculum difficulty based on success rate."""
        target = self.config.success_rate_target
        adjustment = self.config.difficulty_adjustment_rate
        
        if success_rate > target + 0.1:
            # Too easy, increase difficulty
            self.task_manager.increase_difficulty(adjustment)
        elif success_rate < target - 0.1:
            # Too hard, decrease difficulty
            self.task_manager.decrease_difficulty(adjustment)
    
    def _log_episode_metrics(self, propose_metrics: Dict, solve_metrics: Dict, episode_time: float):
        """Log metrics for the current episode."""
        self.metrics['propose_rewards'].append(propose_metrics['propose_reward'])
        self.metrics['solve_rewards'].append(solve_metrics['average_solve_reward'])
        self.metrics['success_rates'].append(solve_metrics['success_rate'])
        self.metrics['episode_times'].append(episode_time)
        
        # Check if this is the best performance so far
        current_performance = (propose_metrics['propose_reward'] + solve_metrics['average_solve_reward']) / 2
        if not hasattr(self, 'best_performance') or current_performance > self.best_performance:
            self.best_performance = current_performance
            self.best_episode = self.episode
        
        if self.episode % 10 == 0:
            self.logger.info(
                f"Episode {self.episode}: "
                f"Propose Reward: {propose_metrics['propose_reward']:.3f}, "
                f"Solve Success: {solve_metrics['success_rate']:.3f}, "
                f"Valid Tasks: {propose_metrics['valid_tasks']}/{propose_metrics['total_proposed']}, "
                f"Time: {episode_time:.2f}s"
            )
    
    def _save_checkpoint(self, episode: int):
        """Save model and training state."""
        checkpoint = {
            'episode': episode,
            'model_state': self.model.get_state(),
            'task_buffers': self.task_manager.get_state(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        checkpoint_path = f"checkpoints/azr_checkpoint_{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_trained_model(self, episode: int, force_save: bool = False):
        """Save the trained AZR model with performance metrics."""
        should_save = (
            force_save or 
            (self.config.auto_save_enabled and episode % self.config.save_frequency == 0) or
            (hasattr(self, 'best_episode') and episode == self.best_episode)
        )
        
        if should_save:
            # Calculate current performance metrics
            recent_metrics = {}
            if len(self.metrics['propose_rewards']) > 0:
                recent_metrics['avg_propose_reward'] = np.mean(self.metrics['propose_rewards'][-50:])
                recent_metrics['avg_solve_reward'] = np.mean(self.metrics['solve_rewards'][-50:])
                recent_metrics['avg_success_rate'] = np.mean(self.metrics['success_rates'][-50:])
                recent_metrics['total_episodes'] = episode
                
            total_reward = recent_metrics.get('avg_propose_reward', 0) + recent_metrics.get('avg_solve_reward', 0)
            
            # Save the model
            model_path = self.model.save_azr_model(episode, total_reward, recent_metrics)
            self.logger.info(f"AZR model saved: {model_path}")
            
            return model_path
        return None
    
    def evaluate(self, test_tasks: List[ReasoningTask]) -> Dict[str, float]:
        """Evaluate the model on a set of test tasks."""
        correct = 0
        total = len(test_tasks)
        
        for task in test_tasks:
            solution_prompt = self._create_solution_prompt(task)
            solution = self.model.generate_solution(solution_prompt)
            
            if self._verify_solution(task, solution):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        self.logger.info(f"Evaluation results: {results}")
        return results
