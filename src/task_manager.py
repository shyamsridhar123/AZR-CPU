"""
Task Manager for the Absolute Zero Reasoner system.
Handles task buffers for deduction, abduction, and induction reasoning types.
"""

import random
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class ReasoningType(Enum):
    DEDUCTION = "deduction"
    ABDUCTION = "abduction" 
    INDUCTION = "induction"


@dataclass
class ReasoningTask:
    """Represents a reasoning task with its type and components."""
    type: str
    program: str
    input: str = ""
    expected_output: str = ""
    complexity: int = 1
    created_episode: int = 0
    success_count: int = 0
    attempt_count: int = 0
    
    def __post_init__(self):
        """Validate task after initialization."""
        if self.type not in [t.value for t in ReasoningType]:
            raise ValueError(f"Invalid reasoning type: {self.type}")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this task."""
        if self.attempt_count == 0:
            return 0.0
        return self.success_count / self.attempt_count
    
    def record_attempt(self, success: bool):
        """Record an attempt to solve this task."""
        self.attempt_count += 1
        if success:
            self.success_count += 1


class TaskBuffer:
    """Buffer for storing tasks of a specific reasoning type."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.tasks: deque = deque(maxlen=max_size)
        self.task_weights: List[float] = []
        
    def add_task(self, task: ReasoningTask):
        """Add a task to the buffer."""
        self.tasks.append(task)
        # Weight based on complexity and success rate
        weight = 1.0 / (1.0 + task.complexity * 0.1)
        self.task_weights.append(weight)
        
        # Maintain weight list size
        if len(self.task_weights) > self.max_size:
            self.task_weights.pop(0)
    
    def sample_tasks(self, n: int) -> List[ReasoningTask]:
        """Sample n tasks from the buffer using weighted sampling."""
        if not self.tasks:
            return []
        
        n = min(n, len(self.tasks))
        
        if len(self.task_weights) != len(self.tasks):
            # Recompute weights if sizes don't match
            self.task_weights = [1.0 / (1.0 + task.complexity * 0.1) 
                               for task in self.tasks]
        
        # Normalize weights
        total_weight = sum(self.task_weights)
        if total_weight == 0:
            weights = [1.0] * len(self.tasks)
        else:
            weights = [w / total_weight for w in self.task_weights]
        
        # Sample tasks
        indices = np.random.choice(
            len(self.tasks), 
            size=n, 
            replace=False if n <= len(self.tasks) else True,
            p=weights
        )
        
        return [self.tasks[i] for i in indices]
    
    def get_complexity_distribution(self) -> Dict[int, int]:
        """Get distribution of task complexities."""
        distribution = {}
        for task in self.tasks:
            complexity = task.complexity
            distribution[complexity] = distribution.get(complexity, 0) + 1
        return distribution
    
    def prune_easy_tasks(self, threshold: float = 0.9):
        """Remove tasks that are too easy (high success rate)."""
        tasks_to_keep = []
        weights_to_keep = []
        
        for i, task in enumerate(self.tasks):
            if task.success_rate < threshold or task.attempt_count < 5:
                tasks_to_keep.append(task)
                if i < len(self.task_weights):
                    weights_to_keep.append(self.task_weights[i])
        
        self.tasks = deque(tasks_to_keep, maxlen=self.max_size)
        self.task_weights = weights_to_keep
    
    def __len__(self):
        return len(self.tasks)


class TaskManager:
    """
    Manages task buffers for different reasoning types and handles curriculum learning.
    """
    
    def __init__(self, config):
        self.config = config
        self.current_episode = 0
        self.difficulty_level = 1.0
        
        # Initialize buffers for each reasoning type
        self.buffers = {
            ReasoningType.DEDUCTION.value: TaskBuffer(config.buffer_size),
            ReasoningType.ABDUCTION.value: TaskBuffer(config.buffer_size),
            ReasoningType.INDUCTION.value: TaskBuffer(config.buffer_size)
        }
        
        # Task generation templates for each type
        self.task_templates = {
            ReasoningType.DEDUCTION.value: self._get_deduction_templates(),
            ReasoningType.ABDUCTION.value: self._get_abduction_templates(),
            ReasoningType.INDUCTION.value: self._get_induction_templates()
        }
        
        # Statistics tracking
        self.stats = {
            'tasks_added': {t.value: 0 for t in ReasoningType},
            'tasks_solved': {t.value: 0 for t in ReasoningType},
            'average_complexity': {t.value: 0.0 for t in ReasoningType}
        }
    
    def add_task(self, task: ReasoningTask):
        """Add a task to the appropriate buffer."""
        task.created_episode = self.current_episode
        
        if task.type in self.buffers:
            self.buffers[task.type].add_task(task)
            self.stats['tasks_added'][task.type] += 1
            self._update_complexity_stats(task.type)
    
    def sample_tasks(self, batch_size: int) -> List[ReasoningTask]:
        """Sample tasks from all buffers for training."""
        tasks = []
        
        # Distribute batch across reasoning types
        tasks_per_type = max(1, batch_size // len(ReasoningType))
        
        for reasoning_type in ReasoningType:
            buffer = self.buffers[reasoning_type.value]
            type_tasks = buffer.sample_tasks(tasks_per_type)
            tasks.extend(type_tasks)
        
        # Fill remaining slots if needed
        remaining = batch_size - len(tasks)
        if remaining > 0:
            all_buffers = list(self.buffers.values())
            non_empty_buffers = [b for b in all_buffers if len(b) > 0]
            
            if non_empty_buffers:
                for _ in range(remaining):
                    buffer = random.choice(non_empty_buffers)
                    extra_tasks = buffer.sample_tasks(1)
                    tasks.extend(extra_tasks)
        
        random.shuffle(tasks)
        return tasks[:batch_size]
    
    def sample_tasks_by_type(self, reasoning_type: str, n: int) -> List[ReasoningTask]:
        """Sample tasks of a specific reasoning type."""
        if reasoning_type in self.buffers:
            return self.buffers[reasoning_type].sample_tasks(n)
        return []
    
    def record_task_attempt(self, task: ReasoningTask, success: bool):
        """Record an attempt to solve a task."""
        task.record_attempt(success)
        if success:
            self.stats['tasks_solved'][task.type] += 1
    
    def increase_difficulty(self, rate: float = 0.1):
        """Increase the difficulty level for task generation."""
        self.difficulty_level = min(self.config.max_task_complexity, 
                                  self.difficulty_level + rate)
    
    def decrease_difficulty(self, rate: float = 0.1):
        """Decrease the difficulty level for task generation."""
        self.difficulty_level = max(self.config.min_task_complexity,
                                  self.difficulty_level - rate)
    
    def prune_buffers(self):
        """Prune easy tasks from all buffers."""
        for buffer in self.buffers.values():
            buffer.prune_easy_tasks()
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the current buffer state."""
        stats = {}
        
        for reasoning_type, buffer in self.buffers.items():
            stats[reasoning_type] = {
                'size': len(buffer),
                'complexity_distribution': buffer.get_complexity_distribution(),
                'tasks_added': self.stats['tasks_added'][reasoning_type],
                'tasks_solved': self.stats['tasks_solved'][reasoning_type],
                'average_complexity': self.stats['average_complexity'][reasoning_type]
            }
        
        stats['difficulty_level'] = self.difficulty_level
        stats['current_episode'] = self.current_episode
        
        return stats
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state for checkpointing."""
        state = {
            'difficulty_level': self.difficulty_level,
            'current_episode': self.current_episode,
            'stats': self.stats,
            'buffers': {}
        }
        
        # Save buffer contents
        for reasoning_type, buffer in self.buffers.items():
            state['buffers'][reasoning_type] = {
                'tasks': list(buffer.tasks),
                'weights': buffer.task_weights
            }
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.difficulty_level = state.get('difficulty_level', 1.0)
        self.current_episode = state.get('current_episode', 0)
        self.stats = state.get('stats', self.stats)
        
        # Restore buffers
        if 'buffers' in state:
            for reasoning_type, buffer_data in state['buffers'].items():
                if reasoning_type in self.buffers:
                    buffer = self.buffers[reasoning_type]
                    buffer.tasks = deque(buffer_data['tasks'], 
                                       maxlen=buffer.max_size)
                    buffer.task_weights = buffer_data.get('weights', [])
    
    def _update_complexity_stats(self, reasoning_type: str):
        """Update average complexity statistics."""
        buffer = self.buffers[reasoning_type]
        if len(buffer) > 0:
            total_complexity = sum(task.complexity for task in buffer.tasks)
            self.stats['average_complexity'][reasoning_type] = total_complexity / len(buffer)
    
    def _get_deduction_templates(self) -> List[Dict[str, Any]]:
        """Get templates for deduction task generation."""
        return [
            {
                'name': 'arithmetic',
                'template': 'lambda x, y: x {op} y',
                'operations': ['+', '-', '*', '//', '%'],
                'complexity': 1
            },
            {
                'name': 'conditional',
                'template': 'lambda x: {expr} if {condition} else {alt_expr}',
                'complexity': 2
            },
            {
                'name': 'list_operation',
                'template': 'lambda lst: {operation}',
                'operations': ['sum(lst)', 'len(lst)', 'max(lst)', 'min(lst)'],
                'complexity': 2
            },
            {
                'name': 'string_operation',
                'template': 'lambda s: s.{method}()',
                'methods': ['upper', 'lower', 'strip', 'reverse'],
                'complexity': 2
            }
        ]
    
    def _get_abduction_templates(self) -> List[Dict[str, Any]]:
        """Get templates for abduction task generation."""
        return [
            {
                'name': 'reverse_arithmetic',
                'description': 'Given input and output, find the operation',
                'complexity': 2
            },
            {
                'name': 'pattern_matching',
                'description': 'Find the function that produces the given output',
                'complexity': 3
            },
            {
                'name': 'inverse_function',
                'description': 'Find the inverse of a given transformation',
                'complexity': 4
            }
        ]
    
    def _get_induction_templates(self) -> List[Dict[str, Any]]:
        """Get templates for induction task generation."""
        return [
            {
                'name': 'sequence_pattern',
                'description': 'Learn pattern from input-output sequences',
                'complexity': 3
            },
            {
                'name': 'function_synthesis',
                'description': 'Synthesize function from multiple examples',
                'complexity': 4
            },
            {
                'name': 'recursive_pattern',
                'description': 'Learn recursive patterns from examples',
                'complexity': 5
            }
        ]
    
    def update_episode(self, episode: int):
        """Update the current episode number."""
        self.current_episode = episode

    def save_tasks_to_file(self, episode: int):
        """Save all tasks to JSON files for analysis."""
        from pathlib import Path
        import json
        import datetime
        
        data_dir = Path(__file__).parent.parent / "data" / "generated"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for reasoning_type, buffer in self.buffers.items():
            if len(buffer) > 0:
                tasks_data = []
                for task in buffer.tasks:
                    task_dict = {
                        'type': task.type,
                        'program': task.program,
                        'input': task.input,
                        'expected_output': task.expected_output,
                        'complexity': task.complexity,
                        'created_episode': task.created_episode,
                        'success_count': task.success_count,
                        'attempt_count': task.attempt_count,
                        'success_rate': task.success_rate
                    }
                    tasks_data.append(task_dict)
                
                filename = f"{reasoning_type}_tasks_ep{episode}_{timestamp}.json"
                filepath = data_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump({
                        'episode': episode,
                        'reasoning_type': reasoning_type,
                        'task_count': len(tasks_data),
                        'timestamp': timestamp,
                        'tasks': tasks_data
                    }, f, indent=2)
                
                print(f"Saved {len(tasks_data)} {reasoning_type} tasks to {filepath}")
