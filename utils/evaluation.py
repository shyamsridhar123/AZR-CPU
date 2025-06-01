"""
Evaluation utilities for the Absolute Zero Reasoner system.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path


class AZREvaluator:
    """Evaluation utilities for monitoring and analyzing AZR performance."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Standard benchmark tasks for evaluation
        self.benchmark_tasks = self._create_benchmark_tasks()
        
    def _create_benchmark_tasks(self) -> List[Dict[str, Any]]:
        """Create standard benchmark tasks for evaluation."""
        return [
            # Basic arithmetic
            {
                'type': 'deduction',
                'program': 'lambda x, y: x + y',
                'input': '(3, 5)',
                'expected_output': '8',
                'complexity': 1,
                'name': 'basic_addition'
            },
            {
                'type': 'deduction', 
                'program': 'lambda x, y: x * y',
                'input': '(4, 6)',
                'expected_output': '24',
                'complexity': 1,
                'name': 'basic_multiplication'
            },
            # Conditional logic
            {
                'type': 'deduction',
                'program': 'lambda x: x if x > 0 else 0',
                'input': '5',
                'expected_output': '5',
                'complexity': 2,
                'name': 'positive_filter'
            },
            {
                'type': 'deduction',
                'program': 'lambda x: x if x > 0 else 0',
                'input': '-3',
                'expected_output': '0',
                'complexity': 2,
                'name': 'negative_filter'
            },
            # List operations
            {
                'type': 'deduction',
                'program': 'lambda lst: sum(lst)',
                'input': '[1, 2, 3, 4]',
                'expected_output': '10',
                'complexity': 2,
                'name': 'list_sum'
            },
            {
                'type': 'deduction',
                'program': 'lambda lst: max(lst)',
                'input': '[1, 5, 3, 2]',
                'expected_output': '5',
                'complexity': 2,
                'name': 'list_max'
            },
            # Abduction tasks
            {
                'type': 'abduction',
                'input': '10',
                'expected_output': '100',
                'complexity': 2,
                'name': 'square_function'
            },
            {
                'type': 'abduction',
                'input': '(2, 3)',
                'expected_output': '5',
                'complexity': 2,
                'name': 'addition_reverse'
            },
            # Induction tasks
            {
                'type': 'induction',
                'input': '1',
                'expected_output': '2',
                'complexity': 3,
                'name': 'increment_pattern'
            },
            {
                'type': 'induction',
                'input': '3',
                'expected_output': '9',
                'complexity': 3,
                'name': 'square_pattern'
            }
        ]
    
    def evaluate_system(self, azr_system, save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the AZR system.
        
        Args:
            azr_system: The AZR system to evaluate
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = {
            'benchmark_performance': self._evaluate_benchmarks(azr_system),
            'task_generation_quality': self._evaluate_task_generation_quality(azr_system),
            'learning_progression': self._evaluate_learning_progression(azr_system),
            'computational_efficiency': self._evaluate_efficiency(azr_system),
            'robustness': self._evaluate_robustness(azr_system)
        }
          # Calculate overall score
        results['overall_score'] = self._calculate_overall_score(results)  # type: ignore
        
        if save_results:
            self._save_evaluation_results(results)
        
        return results
    def _evaluate_benchmarks(self, azr_system) -> Dict[str, Any]:
        """Evaluate performance on standard benchmark tasks."""
        try:
            from ..src.task_manager import ReasoningTask
        except ImportError:
            from src.task_manager import ReasoningTask
        
        # Convert benchmark tasks to ReasoningTask objects
        benchmark_reasoning_tasks = []
        for task_data in self.benchmark_tasks:
            task = ReasoningTask(
                type=task_data['type'],
                program=task_data.get('program', ''),
                input=task_data['input'],
                expected_output=task_data['expected_output'],
                complexity=task_data['complexity']
            )
            benchmark_reasoning_tasks.append((task, task_data['name']))
        
        # Evaluate each task
        results = {
            'total_tasks': len(benchmark_reasoning_tasks),
            'correct': 0,
            'task_results': [],
            'accuracy_by_complexity': {},
            'accuracy_by_type': {}
        }
        
        complexity_stats = {}
        type_stats = {}
        
        for task, name in benchmark_reasoning_tasks:
            # Generate solution
            solution_prompt = azr_system._create_solution_prompt(task)
            solution = azr_system.model.generate_solution(solution_prompt, task.type)
            
            # Verify solution
            is_correct = azr_system._verify_solution(task, solution)
            
            if is_correct:
                results['correct'] += 1
            
            # Track by complexity
            if task.complexity not in complexity_stats:
                complexity_stats[task.complexity] = {'correct': 0, 'total': 0}
            complexity_stats[task.complexity]['total'] += 1
            if is_correct:
                complexity_stats[task.complexity]['correct'] += 1
            
            # Track by type
            if task.type not in type_stats:
                type_stats[task.type] = {'correct': 0, 'total': 0}
            type_stats[task.type]['total'] += 1
            if is_correct:
                type_stats[task.type]['correct'] += 1
            
            results['task_results'].append({
                'name': name,
                'type': task.type,
                'complexity': task.complexity,
                'correct': is_correct,
                'solution': solution,
                'expected': task.expected_output
            })
        
        # Calculate accuracy by complexity and type
        for complexity, stats in complexity_stats.items():
            results['accuracy_by_complexity'][complexity] = stats['correct'] / stats['total']
        
        for task_type, stats in type_stats.items():
            results['accuracy_by_type'][task_type] = stats['correct'] / stats['total']
        
        results['overall_accuracy'] = results['correct'] / results['total_tasks']
        
        return results
    
    def _evaluate_task_generation_quality(self, azr_system) -> Dict[str, Any]:
        """Evaluate the quality of generated tasks."""
        # Generate sample tasks
        generated_tasks = []
        valid_tasks = 0
        
        for _ in range(20):
            reasoning_type = np.random.choice(['deduction', 'abduction', 'induction'])
            task_prompt = azr_system._create_task_prompt(reasoning_type)
            generated_task = azr_system.model.generate_task(task_prompt, reasoning_type)
            
            generated_tasks.append(generated_task)
            
            # Check validity
            if azr_system._validate_task(generated_task):
                valid_tasks += 1
        
        results = {
            'total_generated': len(generated_tasks),
            'valid_tasks': valid_tasks,
            'validity_rate': valid_tasks / len(generated_tasks),
            'diversity_score': self._calculate_task_diversity(generated_tasks),
            'complexity_distribution': self._analyze_complexity_distribution(generated_tasks)
        }
        
        return results
    
    def _evaluate_learning_progression(self, azr_system) -> Dict[str, Any]:
        """Evaluate learning progression over episodes."""
        metrics = azr_system.metrics
        
        if not metrics['success_rates']:
            return {'error': 'No training metrics available'}
        
        # Calculate learning trends
        success_rates = metrics['success_rates']
        propose_rewards = metrics['propose_rewards']
        solve_rewards = metrics['solve_rewards']
        
        results = {
            'initial_success_rate': success_rates[0] if success_rates else 0.0,
            'final_success_rate': success_rates[-1] if success_rates else 0.0,
            'max_success_rate': max(success_rates) if success_rates else 0.0,
            'learning_trend': self._calculate_trend(success_rates),
            'convergence_episode': self._find_convergence_point(success_rates),
            'stability_score': self._calculate_stability(success_rates[-20:] if len(success_rates) >= 20 else success_rates)
        }
        
        return results
    
    def _evaluate_efficiency(self, azr_system) -> Dict[str, Any]:
        """Evaluate computational efficiency metrics."""
        metrics = azr_system.metrics
        
        results = {
            'average_episode_time': np.mean(metrics['episode_times']) if metrics['episode_times'] else 0.0,
            'tasks_per_second': self._calculate_throughput(azr_system),
            'memory_usage': self._estimate_memory_usage(azr_system),
            'model_size': azr_system.model.get_model_size()
        }
        
        return results
    
    def _evaluate_robustness(self, azr_system) -> Dict[str, Any]:
        """Evaluate system robustness to various inputs."""
        # Test with edge cases
        edge_cases = [
            {'type': 'deduction', 'program': 'lambda x: x / 0', 'input': '1'},  # Division by zero
            {'type': 'deduction', 'program': 'lambda x: x + "string"', 'input': '5'},  # Type error
            {'type': 'abduction', 'input': 'invalid_input', 'output': '42'},  # Invalid input
        ]
        
        robust_count = 0
        for case in edge_cases:
            try:
                if azr_system._validate_task(case):
                    # Should not validate dangerous tasks
                    pass
                else:
                    robust_count += 1
            except Exception:
                robust_count += 1  # Properly handled exception
        
        results = {
            'edge_case_robustness': robust_count / len(edge_cases),
            'safety_score': self._evaluate_safety(azr_system)
        }
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate an overall performance score."""
        # Weight different aspects
        weights = {
            'benchmark_accuracy': 0.4,
            'task_quality': 0.2,
            'learning_progress': 0.2,
            'efficiency': 0.1,
            'robustness': 0.1
        }
        
        score = 0.0
        
        # Benchmark performance
        if 'benchmark_performance' in results:
            score += weights['benchmark_accuracy'] * results['benchmark_performance'].get('overall_accuracy', 0.0)
        
        # Task generation quality
        if 'task_generation_quality' in results:
            score += weights['task_quality'] * results['task_generation_quality'].get('validity_rate', 0.0)
        
        # Learning progression
        if 'learning_progression' in results:
            learning_score = min(1.0, results['learning_progression'].get('final_success_rate', 0.0))
            score += weights['learning_progress'] * learning_score
        
        # Efficiency (normalized)
        if 'computational_efficiency' in results:
            efficiency_score = min(1.0, 1.0 / max(1.0, results['computational_efficiency'].get('average_episode_time', 1.0)))
            score += weights['efficiency'] * efficiency_score
        
        # Robustness
        if 'robustness' in results:
            score += weights['robustness'] * results['robustness'].get('edge_case_robustness', 0.0)
        
        return score
    
    def _calculate_task_diversity(self, tasks: List[Dict]) -> float:
        """Calculate diversity score for generated tasks."""
        if not tasks:
            return 0.0
        
        # Count unique program patterns
        programs = [task.get('program', '') for task in tasks]
        unique_programs = len(set(programs))
        
        # Diversity is ratio of unique to total
        diversity = unique_programs / len(programs)
        
        return diversity
    
    def _analyze_complexity_distribution(self, tasks: List[Dict]) -> Dict[int, int]:
        """Analyze the distribution of task complexities."""
        distribution = {}
        
        for task in tasks:
            # Estimate complexity (simplified)
            program = task.get('program', '')
            complexity = self._estimate_task_complexity(program)
            distribution[complexity] = distribution.get(complexity, 0) + 1
        
        return distribution
    
    def _estimate_task_complexity(self, program: str) -> int:
        """Estimate task complexity based on program structure."""
        if not program:
            return 1
        
        complexity = 1
        complexity += program.count('if')
        complexity += program.count('for')
        complexity += program.count('while')
        complexity += program.count('lambda')
        complexity += len(program) // 20
        
        return min(complexity, 5)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate the trend (slope) of a series of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except:
            return 0.0
    
    def _find_convergence_point(self, values: List[float], 
                               window_size: int = 10, 
                               threshold: float = 0.01) -> Optional[int]:
        """Find the episode where the system converged."""
        if len(values) < window_size * 2:
            return None
        
        for i in range(window_size, len(values) - window_size):
            window1 = values[i-window_size:i]
            window2 = values[i:i+window_size]
            
            if abs(np.mean(window2) - np.mean(window1)) < threshold:
                return i
        
        return None
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score based on variance."""
        if not values:
            return 0.0
        
        # Lower variance = higher stability
        variance = np.var(values)
        stability = 1.0 / (1.0 + variance)
        
        return float(stability)
    
    def _calculate_throughput(self, azr_system) -> float:
        """Calculate tasks processed per second."""
        metrics = azr_system.metrics
        
        if not metrics['episode_times']:
            return 0.0
        
        total_time = sum(metrics['episode_times'])
        total_tasks = azr_system.total_tasks_generated + azr_system.total_tasks_solved
        
        if total_time == 0:
            return 0.0
        
        return total_tasks / total_time
    
    def _estimate_memory_usage(self, azr_system) -> Dict[str, float]:
        """Estimate memory usage of the system."""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def _evaluate_safety(self, azr_system) -> float:
        """Evaluate safety of the code execution system."""
        # Test with potentially dangerous code
        dangerous_codes = [
            "import os; os.system('rm -rf /')",
            "exec('while True: pass')",
            "open('/etc/passwd', 'r')",
            "__import__('subprocess')"
        ]
        
        safe_count = 0
        for code in dangerous_codes:
            try:
                result = azr_system.code_executor.execute_safe(code)
                if not result['success']:
                    safe_count += 1  # Properly rejected dangerous code
            except Exception:
                safe_count += 1  # Exception handling worked
        
        return safe_count / len(dangerous_codes)
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        output_path = Path("evaluation_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
    
    def plot_learning_curves(self, azr_system, save_path: Optional[str] = None):
        """Plot learning curves from training metrics."""
        metrics = azr_system.metrics
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('AZR Learning Curves')
        
        # Success rates
        if metrics['success_rates']:
            axes[0, 0].plot(metrics['success_rates'])
            axes[0, 0].set_title('Success Rate')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].grid(True)
        
        # Propose rewards
        if metrics['propose_rewards']:
            axes[0, 1].plot(metrics['propose_rewards'])
            axes[0, 1].set_title('Propose Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True)
        
        # Solve rewards
        if metrics['solve_rewards']:
            axes[1, 0].plot(metrics['solve_rewards'])
            axes[1, 0].set_title('Solve Rewards')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True)
        
        # Episode times
        if metrics['episode_times']:
            axes[1, 1].plot(metrics['episode_times'])
            axes[1, 1].set_title('Episode Times')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()


def evaluate_model(azr_system, test_tasks: Optional[List] = None) -> Dict[str, Any]:
    """
    Evaluate an AZR model with optional test tasks.
    
    Args:
        azr_system: The AZR system to evaluate
        test_tasks: Optional list of test tasks
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = AZREvaluator(azr_system.config)
    
    if test_tasks:
        # Evaluate on provided test tasks
        results = azr_system.evaluate(test_tasks)
    else:
        # Run comprehensive evaluation
        results = evaluator.evaluate_system(azr_system, save_results=False)
    
    return results
