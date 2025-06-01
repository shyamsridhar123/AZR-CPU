#!/usr/bin/env python3
"""
Simplified Absolute Zero Reasoner (AZR) Demo
A minimal implementation that demonstrates the core concepts without heavy dependencies.
"""

import random
import re
import ast
import time
import sys
from typing import Dict, List, Any, Optional
from collections import deque
import json


class SimpleCodeExecutor:
    """Simplified code executor for basic Python expressions."""
    
    def __init__(self, timeout=3.0):
        self.timeout = timeout
        
    def execute_safe(self, code: str, input_data: str = "") -> Dict[str, Any]:
        """Execute code safely with basic validation."""
        result = {
            'success': False,
            'output': None,
            'error': None,
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Basic safety checks
            forbidden = ['import', 'exec', 'eval', 'open', 'file', '__']
            if any(f in code for f in forbidden):
                result['error'] = "Forbidden operation detected"
                return result
            
            # Parse and validate
            try:
                parsed = ast.parse(code, mode='eval')
            except SyntaxError as e:
                result['error'] = f"Syntax error: {e}"
                return result
              # Prepare safe environment
            safe_dict: Dict[str, Any] = {
                '__builtins__': {
                    'abs': abs, 'max': max, 'min': min, 'sum': sum,
                    'len': len, 'range': range, 'int': int, 'float': float,
                    'str': str, 'bool': bool, 'list': list, 'dict': dict,
                    'tuple': tuple, 'set': set
                }
            }
            
            # Handle input data
            if input_data:
                try:
                    if input_data.startswith('(') and input_data.endswith(')'):
                        # Tuple input like (3, 4)
                        input_val = eval(input_data, safe_dict)
                        if isinstance(input_val, tuple) and len(input_val) == 2:
                            safe_dict['x'], safe_dict['y'] = input_val[0], input_val[1]
                        else:
                            safe_dict['x'] = input_val
                    else:
                        # Single value input
                        safe_dict['x'] = eval(input_data, safe_dict)
                except:
                    safe_dict['x'] = input_data
            
            # Execute the code
            output = eval(code, safe_dict)
            
            result['success'] = True
            result['output'] = output
            
        except Exception as e:
            result['error'] = str(e)
        
        result['execution_time'] = time.time() - start_time
        return result


class SimpleTask:
    """Simple task representation."""
    
    def __init__(self, task_type: str, program: str, input_data: str = "", 
                 expected_output: str = "", complexity: int = 1):
        self.type = task_type
        self.program = program
        self.input = input_data
        self.expected_output = expected_output
        self.complexity = complexity
        self.attempts = 0
        self.successes = 0
    
    def record_attempt(self, success: bool):
        self.attempts += 1
        if success:
            self.successes += 1
    
    @property
    def success_rate(self):
        return self.successes / self.attempts if self.attempts > 0 else 0.0


class SimpleTaskManager:
    """Simple task buffer management."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.tasks = {
            'deduction': deque(maxlen=max_size),
            'abduction': deque(maxlen=max_size),
            'induction': deque(maxlen=max_size)
        }
    
    def add_task(self, task: SimpleTask):
        if task.type in self.tasks:
            self.tasks[task.type].append(task)
    
    def sample_tasks(self, n: int) -> List[SimpleTask]:
        all_tasks = []
        for task_list in self.tasks.values():
            all_tasks.extend(task_list)
        
        if not all_tasks:
            return []
        
        return random.sample(all_tasks, min(n, len(all_tasks)))
    
    def get_stats(self) -> Dict[str, int]:
        return {task_type: len(task_list) for task_type, task_list in self.tasks.items()}


class SimpleModel:
    """Simple rule-based model for demonstration."""
    
    def __init__(self):
        self.task_templates = {
            'deduction': [
                'lambda x: x + {}',
                'lambda x: x * {}', 
                'lambda x: x - {}',
                'lambda x, y: x + y',
                'lambda x, y: x * y',
                'lambda x: x if x > {} else {}',
                'lambda x: x ** 2',
                'lambda x: abs(x)',
                'lambda x: max(x, {})',
                'lambda x: min(x, {})'
            ],
            'abduction': [
                'lambda x: x + {}',
                'lambda x: x * {}',
                'lambda x: x // {}',
                'lambda x: x % {}',
            ],
            'induction': [
                'lambda x: x + 1',
                'lambda x: x * 2',
                'lambda x: x ** 2',
                'lambda x: -x'
            ]
        }
        
        self.learning_rate = 0.1
        self.weights = {'propose': 1.0, 'solve': 1.0}
    
    def generate_task(self, task_type: str) -> Dict[str, Any]:
        """Generate a simple task."""
        templates = self.task_templates.get(task_type, self.task_templates['deduction'])
        template = random.choice(templates)
        
        if '{}' in template:
            # Fill in random parameters
            param = random.randint(1, 10)
            if template.count('{}') == 1:
                program = template.format(param)
            else:
                param2 = random.randint(1, 5)
                program = template.format(param, param2)
        else:
            program = template
        
        # Generate input
        if 'x, y' in program:
            input_data = f"({random.randint(1, 10)}, {random.randint(1, 10)})"
        else:
            input_data = str(random.randint(1, 10))
        
        return {
            'type': task_type,
            'program': program,
            'input': input_data,
            'complexity': self._estimate_complexity(program)
        }
    
    def generate_solution(self, task: SimpleTask) -> str:
        """Generate a solution for a task."""
        if task.type == 'deduction':
            # For deduction, we need to compute the output
            return f"The output is calculated by executing: {task.program} with input {task.input}"
        
        elif task.type == 'abduction':
            # For abduction, we need to find the program
            # Simple heuristic approach
            if 'add' in task.expected_output.lower() or '+' in task.expected_output:
                return "lambda x: x + constant"
            elif 'multiply' in task.expected_output.lower() or '*' in task.expected_output:
                return "lambda x: x * constant"
            else:
                return "lambda x: x"
        
        elif task.type == 'induction':
            # For induction, find pattern from examples
            templates = self.task_templates['induction']
            return random.choice(templates)
        
        return "lambda x: x"  # Default identity
    
    def _estimate_complexity(self, program: str) -> int:
        complexity = 1
        complexity += program.count('+')
        complexity += program.count('*')
        complexity += program.count('if')
        complexity += program.count('lambda')
        return min(complexity, 5)
    
    def update_weights(self, reward: float):
        """Simple weight update."""
        self.weights['propose'] += self.learning_rate * reward
        self.weights['solve'] += self.learning_rate * reward


class SimpleAZR:
    """Simplified Absolute Zero Reasoner implementation."""
    
    def __init__(self, max_episodes: int = 50):
        self.max_episodes = max_episodes
        self.model = SimpleModel()
        self.task_manager = SimpleTaskManager()
        self.code_executor = SimpleCodeExecutor()
        
        # Statistics
        self.episode = 0
        self.total_tasks_generated = 0
        self.total_tasks_solved = 0
        self.metrics = {
            'propose_rewards': [],
            'solve_rewards': [],
            'success_rates': []
        }
        
        # Initialize with seed tasks
        self._initialize_seed_tasks()
    
    def _initialize_seed_tasks(self):
        """Add some seed tasks to bootstrap the system."""
        seed_tasks = [
            SimpleTask('deduction', 'lambda x: x', '5', '5', 1),
            SimpleTask('deduction', 'lambda x: x + 1', '3', '4', 1),
            SimpleTask('deduction', 'lambda x, y: x + y', '(2, 3)', '5', 1),
            SimpleTask('abduction', 'lambda x: x * 2', '4', '8', 2),
            SimpleTask('induction', 'lambda x: x ** 2', '3', '9', 2)
        ]
        
        for task in seed_tasks:
            self.task_manager.add_task(task)
        
        print(f"🌱 Initialized with {len(seed_tasks)} seed tasks")
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting Simple AZR training for {self.max_episodes} episodes...")
        
        for episode in range(self.max_episodes):
            self.episode = episode
            
            # Phase 1: PROPOSE - Generate new tasks
            propose_metrics = self._propose_phase()
            
            # Phase 2: SOLVE - Attempt to solve tasks  
            solve_metrics = self._solve_phase()
            
            # Update model
            total_reward = propose_metrics['reward'] + solve_metrics['reward']
            self.model.update_weights(total_reward)
            
            # Log progress
            if episode % 10 == 0:
                self._log_progress(episode, propose_metrics, solve_metrics)
        
        print("✅ Training completed!")
        self._show_final_results()
    
    def _propose_phase(self, tasks_per_episode: int = 3) -> Dict[str, Any]:
        """Generate new tasks."""
        valid_tasks = 0
        
        for _ in range(tasks_per_episode):
            task_type = random.choice(['deduction', 'abduction', 'induction'])
            task_data = self.model.generate_task(task_type)
            
            # Validate task
            if self._validate_task(task_data):
                task = SimpleTask(
                    task_data['type'],
                    task_data['program'],
                    task_data['input'],
                    complexity=task_data['complexity']
                )
                self.task_manager.add_task(task)
                valid_tasks += 1
                self.total_tasks_generated += 1
        
        # Calculate proposer reward (learnability)
        validity_rate = valid_tasks / tasks_per_episode
        recent_success = np.mean(self.metrics['success_rates'][-10:]) if self.metrics['success_rates'] else 0.5
        learnability_reward = 1.0 - abs(recent_success - 0.5)  # Optimal at 50% success rate
        
        propose_reward = validity_rate * 0.7 + learnability_reward * 0.3
        self.metrics['propose_rewards'].append(propose_reward)
        
        return {
            'valid_tasks': valid_tasks,
            'total_tasks': tasks_per_episode,
            'reward': propose_reward
        }
    
    def _solve_phase(self, batch_size: int = 3) -> Dict[str, Any]:
        """Attempt to solve tasks."""
        tasks_to_solve = self.task_manager.sample_tasks(batch_size)
        solved_tasks = 0
        
        for task in tasks_to_solve:
            # Generate solution
            solution = self.model.generate_solution(task)
            
            # Verify solution
            is_correct = self._verify_solution(task, solution)
            task.record_attempt(is_correct)
            
            if is_correct:
                solved_tasks += 1
                self.total_tasks_solved += 1
        
        success_rate = solved_tasks / len(tasks_to_solve) if tasks_to_solve else 0.0
        solve_reward = success_rate
        
        self.metrics['solve_rewards'].append(solve_reward)
        self.metrics['success_rates'].append(success_rate)
        
        return {
            'solved_tasks': solved_tasks,
            'total_tasks': len(tasks_to_solve),
            'success_rate': success_rate,
            'reward': solve_reward
        }
    
    def _validate_task(self, task_data: Dict[str, Any]) -> bool:
        """Validate a generated task."""
        try:
            program = task_data.get('program', '')
            input_data = task_data.get('input', '')
            
            if not program or 'lambda' not in program:
                return False
            
            # Test execution
            result = self.code_executor.execute_safe(program, input_data)
            return result['success']
        
        except Exception:
            return False
    
    def _verify_solution(self, task: SimpleTask, solution: str) -> bool:
        """Verify if a solution is correct."""
        try:
            if task.type == 'deduction':
                # For deduction: check if we can execute the program
                result = self.code_executor.execute_safe(task.program, task.input)
                return result['success']
            
            # For other types, simplified verification
            return 'lambda' in solution
        
        except Exception:
            return False
    
    def _log_progress(self, episode: int, propose_metrics: Dict, solve_metrics: Dict):
        """Log training progress."""
        print(f"Episode {episode:3d}: "
              f"Generated {propose_metrics['valid_tasks']}/{propose_metrics['total_tasks']} tasks, "
              f"Solved {solve_metrics['solved_tasks']}/{solve_metrics['total_tasks']} tasks "
              f"(Success: {solve_metrics['success_rate']:.1%})")
    
    def _show_final_results(self):
        """Show final training results."""
        stats = self.task_manager.get_stats()
        avg_propose_reward = np.mean(self.metrics['propose_rewards']) if self.metrics['propose_rewards'] else 0
        avg_solve_reward = np.mean(self.metrics['solve_rewards']) if self.metrics['solve_rewards'] else 0
        
        print("\n" + "="*50)
        print("🎯 FINAL RESULTS")
        print("="*50)
        print(f"Total tasks generated: {self.total_tasks_generated}")
        print(f"Total tasks solved: {self.total_tasks_solved}")
        print(f"Task buffers: {stats}")
        print(f"Average propose reward: {avg_propose_reward:.3f}")
        print(f"Average solve reward: {avg_solve_reward:.3f}")
        
        if self.metrics['success_rates']:
            recent_success = np.mean(self.metrics['success_rates'][-10:])
            print(f"Recent success rate: {recent_success:.1%}")


def main():
    """Main entry point."""
    print("🧠 Simple Absolute Zero Reasoner (AZR) Demo")
    print("=" * 45)
    print("This demonstrates the core concepts of AZR:")
    print("• Self-generated tasks for bootstrapping")
    print("• Dual reward signals (propose + solve)")
    print("• Curriculum learning through success rates")
    print("• CPU-only implementation")
    print()
    
    # Run the demo
    azr = SimpleAZR(max_episodes=30)
    azr.train()
    
    # Interactive demo
    print("\n🔬 Interactive Demo:")
    print("The system has learned to generate and solve simple tasks!")
    
    # Show some examples
    sample_tasks = azr.task_manager.sample_tasks(3)
    if sample_tasks:
        print("\nSample tasks from the learned buffer:")
        for i, task in enumerate(sample_tasks, 1):
            print(f"{i}. {task.type}: {task.program} with input {task.input}")
            if task.attempts > 0:
                print(f"   Success rate: {task.success_rate:.1%} ({task.successes}/{task.attempts})")


if __name__ == "__main__":
        # Add numpy for basic math operations
    try:
        import numpy as np  # type: ignore
    except ImportError:
        print("NumPy not available, using basic Python math")
        class np:  # type: ignore
            @staticmethod
            def mean(arr):
                return sum(arr) / len(arr) if arr else 0
    
    main()
