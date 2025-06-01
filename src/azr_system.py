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

# Import new prompt management system
try:
    # Try relative imports first (when used as package)
    from ..Prompts.prompt_manager import PromptManager
    from ..Prompts.task_generation_prompts import TaskGenerationPrompts
    from ..Prompts.solution_prompts import SolutionPrompts
    from ..Prompts.validation_prompts import ValidationPrompts
    from ..Prompts.advanced_templates import AdvancedPromptTemplates
    PROMPT_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        # Fall back to adding the Prompts directory to path
        import sys
        from pathlib import Path
        prompts_path = Path(__file__).parent.parent / "Prompts"
        if str(prompts_path) not in sys.path:
            sys.path.insert(0, str(prompts_path))
        
        from prompt_manager import PromptManager
        from task_generation_prompts import TaskGenerationPrompts
        from solution_prompts import SolutionPrompts
        from validation_prompts import ValidationPrompts
        from advanced_templates import AdvancedPromptTemplates
        PROMPT_SYSTEM_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"Could not import prompt management system: {e}")
        # Set to None for fallback behavior
        PromptManager = None
        TaskGenerationPrompts = None
        SolutionPrompts = None
        ValidationPrompts = None
        AdvancedPromptTemplates = None
        PROMPT_SYSTEM_AVAILABLE = False


@dataclass
class AZRConfig:
    """Configuration for the Absolute Zero Reasoner system."""
    # Model configuration
    model_name: str = "Salesforce/codet5-small"  # Code-trained model for CPU
    max_length: int = 2000  # Increased from 512 for better generation
    temperature: float = 0.7  # Lower temperature for more focused code generation
    use_local_models: bool = True  # Try to use locally cached models first
    save_models_locally: bool = True  # Save downloaded models to local cache
      # AZR Model Configuration - Enhanced: Configurable model selection
    use_pretrained_azr: bool = False  # Whether to use a pre-trained AZR model
    pretrained_azr_path: Optional[str] = None  # Specific path to AZR model (if None, uses preference)
    azr_model_preference: str = "latest"  # "latest", "best", "specific", or model name
    fallback_to_base: bool = True  # Fallback to base model if AZR model not found
    
    # Model Selection Strategy
    model_selection_strategy: str = "auto"  # "auto", "manual", "performance_based"
    enable_runtime_model_switching: bool = True  # Allow switching models during runtime
    model_switching_threshold: float = 0.1  # Performance threshold for auto-switching
    
    # Model Performance Tracking
    track_model_performance: bool = True  # Track and compare model performance
    performance_window_size: int = 50  # Number of episodes to consider for performance
    save_model_comparison_data: bool = True  # Save performance comparison data
    
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
    
    # Testing configuration (DO NOT USE IN PRODUCTION)
    simple_rewards: bool = False  # Use simplified reward calculation for testing only


class AbsoluteZeroReasoner:
    """
    Main AZR system that implements self-bootstrapping reasoning through
    reinforcement learning with self-generated tasks.
    """
    def __init__(self, config: AZRConfig):
        self.config = config
        self.setup_logging()
        
        # Initialize components
        self.model = ModelWrapper(config)
        self.task_manager = TaskManager(config)
        self.code_executor = CodeExecutor(config)
        self.reward_calculator = RewardCalculator(config)
        
        # Initialize prompt management system
        if PromptManager is not None:
            self.prompt_manager = PromptManager()
            self.use_advanced_prompts = True
            self.logger.info("Using advanced prompt management system")
        else:
            self.prompt_manager = None
            self.use_advanced_prompts = False
            self.logger.warning("Using fallback basic prompts")
        
        # Training state
        self.episode = 0
        self.total_tasks_generated = 0
        self.total_tasks_solved = 0
        
        # Curriculum learning state
        self.current_complexity = config.min_task_complexity
        self.complexity_history = []
        self.success_rate_window = deque(maxlen=10)  # Track recent success rates
        
        # Task quality tracking
        self.task_quality_metrics = {
            'syntax_validity': deque(maxlen=50),
            'logic_correctness': deque(maxlen=50),
            'complexity_accuracy': deque(maxlen=50)
        }
        
        # Metrics tracking
        self.metrics = {
            'propose_rewards': [],
            'solve_rewards': [],
            'success_rates': [],
            'task_complexity': [],
            'episode_times': [],
            'task_quality_scores': []
        }
        
        self.logger.info("Absolute Zero Reasoner initialized with curriculum learning")
    
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
            self._log_episode_metrics(propose_metrics, solve_metrics, episode_time)
            
            # Save checkpoint, trained model, and tasks
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
            }        ]
        
        for task_data in seed_tasks:
            task = ReasoningTask(**task_data)
            self.task_manager.add_task(task)
        
        self.logger.info(f"Initialized with {len(seed_tasks)} seed tasks")
    
    def _propose_phase(self) -> Dict[str, Any]:
        """
        PROPOSE phase: Generate new reasoning tasks using the current model.
        Enhanced with advanced prompt system and quality control.
        """
        proposed_tasks = []
        valid_tasks = 0
        quality_scores = []
        
        for _ in range(self.config.tasks_per_episode):
            # Sample reasoning type
            reasoning_type = random.choice(['deduction', 'abduction', 'induction'])
              # Generate task using model with current complexity
            task_prompt = self._create_task_prompt(reasoning_type)
            generated_task = self.model.generate_task(
                task_prompt, 
                reasoning_type=reasoning_type,
                complexity=int(self.current_complexity)  # Ensure int type
            )
            
            # Enhanced validation with quality scoring
            if self._validate_task(generated_task):
                # Create task object
                task = ReasoningTask(
                    type=reasoning_type,
                    program=generated_task['program'],
                    input=generated_task.get('input', ''),
                    expected_output=generated_task.get('output', ''),
                    complexity=generated_task.get('complexity', self.current_complexity)
                )
                
                # Add task to manager and track
                self.task_manager.add_task(task)
                proposed_tasks.append(task)
                valid_tasks += 1
                self.total_tasks_generated += 1
                
                # Track quality if using advanced validation
                if hasattr(self, 'task_quality_metrics'):
                    if len(self.task_quality_metrics['logic_correctness']) > 0:
                        quality_scores.append(self.task_quality_metrics['logic_correctness'][-1])
        
        # Calculate enhanced proposer reward
        base_propose_reward = self.reward_calculator.calculate_propose_reward(
            valid_tasks, self.config.tasks_per_episode
        )
          # Bonus for high quality tasks
        quality_bonus = 0.0
        if quality_scores:
            avg_quality = float(np.mean(quality_scores))
            quality_bonus = max(0.0, (avg_quality - 0.6) * 0.5)  # Bonus for quality > 0.6
        
        total_propose_reward = base_propose_reward + quality_bonus
        return {
            'valid_tasks': valid_tasks,
            'total_proposed': self.config.tasks_per_episode,
            'propose_reward': total_propose_reward,
            'quality_bonus': quality_bonus,
            'avg_quality': np.mean(quality_scores) if quality_scores else 0.0,
            'tasks': proposed_tasks
        }
    
    def _solve_phase(self) -> Dict[str, Any]:
        """
        SOLVE phase: Attempt to solve tasks from the buffer.
        Enhanced with advanced solution prompts and error analysis.
        """
        tasks_to_solve = self.task_manager.sample_tasks(self.config.batch_size)
        solved_tasks = 0
        solve_rewards = []
        solution_attempts = []
        
        for task in tasks_to_solve:
            # Generate solution using enhanced prompts
            solution_prompt = self._create_solution_prompt(task)
            
            # Pass task data for advanced prompting
            task_data = {
                'type': task.type,
                'program': getattr(task, 'program', ''),
                'input': task.input,
                'expected_output': task.expected_output,
                'complexity': getattr(task, 'complexity', 1)
            }
            
            generated_solution = self.model.generate_solution(
                solution_prompt, 
                task_type=task.type,
                task_data=task_data
            )
            
            # Verify solution through execution
            is_correct = self._verify_solution(task, generated_solution)
            
            if is_correct:
                solved_tasks += 1
                self.total_tasks_solved += 1
            
            # Calculate individual solve reward with complexity bonus
            base_reward = 1.0 if is_correct else 0.0
            complexity_bonus = 0.0
            
            if is_correct and hasattr(task, 'complexity'):
                # Bonus for solving higher complexity tasks
                complexity_bonus = (task.complexity - 1) * 0.1
                
            total_reward = base_reward + complexity_bonus
            solve_rewards.append(total_reward)
            
            # Track solution attempts for analysis
            solution_attempts.append({
                'task_type': task.type,
                'complexity': getattr(task, 'complexity', 1),
                'correct': is_correct,
                'solution': generated_solution[:100]  # First 100 chars for logging
            })
        
        success_rate = solved_tasks / len(tasks_to_solve) if tasks_to_solve else 0.0
        
        # Calculate complexity-weighted success rate
        complexity_weighted_success = 0.0
        if solution_attempts:
            for attempt in solution_attempts:
                if attempt['correct']:
                    complexity_weighted_success += attempt['complexity']
            complexity_weighted_success /= sum(attempt['complexity'] for attempt in solution_attempts)
        return {
            'solved_tasks': solved_tasks,
            'total_tasks': len(tasks_to_solve),
            'success_rate': success_rate,
            'complexity_weighted_success': complexity_weighted_success,
            'solve_rewards': solve_rewards,
            'average_solve_reward': np.mean(solve_rewards) if solve_rewards else 0.0,
            'solution_attempts': solution_attempts[:5]  # Keep first 5 for logging
        }
    
    def _validate_task(self, task_data: Dict) -> bool:
        """Validate a generated task through code execution and quality checks."""
        try:
            if 'program' not in task_data:
                return False
            
            program = task_data['program']
            input_data = task_data.get('input', '')
            
            # Basic syntax validation through execution
            result = self.code_executor.execute_safe(program, input_data)
            if not result['success']:
                return False
            
            # Advanced quality validation if available
            if self.use_advanced_prompts and self.prompt_manager:
                quality_score = self._validate_task_quality(task_data)
                self.task_quality_metrics['syntax_validity'].append(1.0 if result['success'] else 0.0)
                self.task_quality_metrics['logic_correctness'].append(quality_score)
                
                # Return True only if quality meets threshold
                return quality_score > 0.6  # Adjustable threshold
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Task validation failed: {e}")
            return False
    
    def _validate_task_quality(self, task_data: Dict) -> float:
        """Validate task quality using advanced prompts."""
        if not self.use_advanced_prompts or not self.prompt_manager:
            return 1.0  # Default to accepting if no advanced validation
        
        try:
            validation_prompt = self.prompt_manager.get_validation_prompt(
                task_data=task_data,
                validation_type='quality'
            )
            
            # Use model to evaluate task quality
            quality_response = self.model.generate(validation_prompt)
            
            # Simple scoring based on response (could be more sophisticated)
            if 'excellent' in quality_response.lower() or 'good' in quality_response.lower():
                return 0.9
            elif 'acceptable' in quality_response.lower() or 'valid' in quality_response.lower():
                return 0.7
            elif 'poor' in quality_response.lower() or 'invalid' in quality_response.lower():
                return 0.3
            else:
                return 0.5  # Neutral if unclear
                
        except Exception as e:
            self.logger.debug(f"Quality validation failed: {e}")
            return 0.5
    
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
        """Create a prompt for task generation using advanced prompt system."""
        if self.use_advanced_prompts and self.prompt_manager:
            return self.prompt_manager.get_task_generation_prompt(
                reasoning_type=reasoning_type,
                complexity=int(self.current_complexity),
                include_examples=True
            )
        else:
            # Fallback to basic prompts
            prompts = {
                'deduction': "Generate a simple Python function and input. Format: {'program': 'lambda x: ...', 'input': '...'}",
                'abduction': "Generate a Python function input and expected output. Format: {'input': '...', 'output': '...'}",
                'induction': "Generate input-output examples for pattern learning. Format: {'input': '...', 'output': '...'}"            }
            return prompts.get(reasoning_type, prompts['deduction'])
    
    def _create_solution_prompt(self, task: ReasoningTask) -> str:
        """Create a prompt for solution generation using advanced prompt system."""
        if self.use_advanced_prompts and self.prompt_manager:
            # Prepare task data for the prompt manager with all possible fields
            task_data = {
                'program': getattr(task, 'program', ''),
                'input': task.input,
                'expected_output': task.expected_output,
                'type': task.type,
                'complexity': getattr(task, 'complexity', 1),
                'examples': getattr(task, 'examples', ''),
                'pattern': getattr(task, 'pattern', ''),
                'reasoning_type': task.type
            }
            
            # For induction tasks, if input looks like examples, format them properly
            if task.type == 'induction' and task.input:
                try:
                    # Try to parse input as examples for induction
                    import ast
                    if task.input.startswith('['):
                        examples_list = ast.literal_eval(task.input)
                        task_data['examples'] = examples_list
                except:
                    # If parsing fails, use input as is
                    task_data['examples'] = task.input
                    
            return self.prompt_manager.get_solution_prompt(
                reasoning_type=task.type,
                task_data=task_data
            )
        else:
            # Fallback to basic prompts
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
        """Update model using TRR++ algorithm with dual rewards."""        # Combine propose and solve rewards
        total_reward = (propose_metrics['propose_reward'] +
                       solve_metrics['average_solve_reward'])
                        
        # Update model (simplified for demonstration)
        self.model.update_weights(total_reward)
    
    def _update_curriculum(self, success_rate: float):
        """Update curriculum difficulty based on success rate with advanced curriculum learning."""
        self.success_rate_window.append(success_rate)
        target = self.config.success_rate_target
        adjustment = self.config.difficulty_adjustment_rate
        
        # Calculate moving average success rate
        avg_success_rate = np.mean(self.success_rate_window)
        
        # Update current complexity based on performance
        old_complexity = self.current_complexity
        
        if avg_success_rate > target + 0.1:
            # Too easy, increase difficulty
            self.current_complexity = min(
                self.current_complexity + adjustment,
                self.config.max_task_complexity
            )
        elif avg_success_rate < target - 0.1:
            # Too hard, decrease difficulty
            self.current_complexity = max(
                self.current_complexity - adjustment,
                self.config.min_task_complexity
            )
        
        # Log complexity changes
        if abs(self.current_complexity - old_complexity) > 0.05:
            self.logger.info(
                f"Complexity updated: {old_complexity:.2f} -> {self.current_complexity:.2f} "
                f"(Success rate: {avg_success_rate:.3f})"
            )
          # Update task manager difficulty if available (use alternative methods)
        if hasattr(self.task_manager, 'difficulty_level'):
            self.task_manager.difficulty_level = self.current_complexity
        
        # Record complexity history
        self.complexity_history.append(self.current_complexity)
    
    def _log_episode_metrics(self, propose_metrics: Dict, solve_metrics: Dict, episode_time: float):
        """Log metrics for the current episode with enhanced information."""
        self.metrics['propose_rewards'].append(propose_metrics['propose_reward'])
        self.metrics['solve_rewards'].append(solve_metrics['average_solve_reward'])
        self.metrics['success_rates'].append(solve_metrics['success_rate'])
        self.metrics['episode_times'].append(episode_time)
        
        # Track new quality metrics
        if 'quality_bonus' in propose_metrics:
            if 'task_quality_scores' not in self.metrics:
                self.metrics['task_quality_scores'] = []
            self.metrics['task_quality_scores'].append(propose_metrics.get('avg_quality', 0.0))
        
        if 'complexity_weighted_success' in solve_metrics:
            if 'complexity_weighted_success' not in self.metrics:
                self.metrics['complexity_weighted_success'] = []
            self.metrics['complexity_weighted_success'].append(solve_metrics['complexity_weighted_success'])
        
        # Track complexity progression
        self.metrics['task_complexity'].append(self.current_complexity)
        
        # Check if this is the best performance so far
        current_performance = (propose_metrics['propose_reward'] + solve_metrics['average_solve_reward']) / 2
        if not hasattr(self, 'best_performance') or current_performance > self.best_performance:
            self.best_performance = current_performance
            self.best_episode = self.episode
        
        if self.episode % 10 == 0:
            # Enhanced logging with new metrics
            quality_info = ""
            if 'avg_quality' in propose_metrics:
                quality_info = f", Avg Quality: {propose_metrics['avg_quality']:.3f}"
            
            complexity_info = ""
            if 'complexity_weighted_success' in solve_metrics:
                complexity_info = f", Complex Success: {solve_metrics['complexity_weighted_success']:.3f}"
            
            self.logger.info(
                f"Episode {self.episode}: "
                f"Propose Reward: {propose_metrics['propose_reward']:.3f}, "
                f"Solve Success: {solve_metrics['success_rate']:.3f}, "
                f"Valid Tasks: {propose_metrics['valid_tasks']}/{propose_metrics['total_proposed']}, "
                f"Complexity: {self.current_complexity:.2f}"
                f"{quality_info}{complexity_info}, "
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
