"""
Reward Calculator for the Absolute Zero Reasoner system.
Implements the TRR++ reward calculation with dual reward signals.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging


class RewardCalculator:
    """
    Calculates rewards for the TRR++ algorithm with dual reward signals:
    - Learnability Reward (Proposer): r_propose = 1 - |success_rate - 0.5|
    - Accuracy Reward (Solver): r_solve = 1 if correct else 0
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Warn if simple rewards mode is enabled
        if hasattr(config, 'simple_rewards') and config.simple_rewards:
            self.logger.warning("=" * 60)
            self.logger.warning("WARNING: SIMPLE REWARDS MODE ENABLED")
            self.logger.warning("This mode is for testing only and should NEVER")
            self.logger.warning("be used for actual model training!")
            self.logger.warning("=" * 60)
        
        # Target success rate for optimal learnability
        self.target_success_rate = config.success_rate_target
        
        # Rolling windows for tracking metrics
        self.success_history = deque(maxlen=100)
        self.propose_history = deque(maxlen=100)
        self.solve_history = deque(maxlen=100)
        
        # Advantage normalization parameters
        self.advantage_mean = 0.0
        self.advantage_std = 1.0
        self.advantage_momentum = 0.9
        
        # Reward scaling factors
        self.propose_weight = 1.0
        self.solve_weight = 1.0
        self.complexity_bonus = 0.1
        
        # Statistics
        self.stats = {
            'total_propose_rewards': 0.0,
            'total_solve_rewards': 0.0,
            'reward_episodes': 0,
            'best_success_rate': 0.0,
            'reward_variance': 0.0
        }
    def calculate_propose_reward(self, valid_tasks: int, total_tasks: int) -> float:
        """
        Calculate proposer reward based on task generation success and learnability.
        
        Args:
            valid_tasks: Number of valid tasks generated
            total_tasks: Total number of tasks attempted
            
        Returns:
            Proposer reward value
        """
        if total_tasks == 0:
            return 0.0
          # For basic testing, use simple validity ratio
        if hasattr(self.config, 'simple_rewards') and self.config.simple_rewards:
            self.logger.warning("USING SIMPLE REWARDS MODE - FOR TESTING ONLY!")
            self.logger.warning("This mode should never be used in production training.")
            return valid_tasks / total_tasks
        
        # Basic validity reward
        validity_rate = valid_tasks / total_tasks
        
        # Estimate success rate from recent history
        if len(self.success_history) > 0:
            estimated_success_rate = np.mean(self.success_history)
        else:
            estimated_success_rate = 0.5  # Default to target
        
        # Learnability reward: optimal when success rate is near target
        learnability_reward = 1.0 - abs(estimated_success_rate - self.target_success_rate)
        learnability_reward = max(0.0, learnability_reward)
        
        # Combine validity and learnability
        propose_reward = (validity_rate * 0.7 + learnability_reward * 0.3) * self.propose_weight
        
        # Add diversity bonus (encourage varied task generation)
        diversity_bonus = self._calculate_diversity_bonus()
        propose_reward += diversity_bonus
        
        # Track history
        self.propose_history.append(propose_reward)
        self.stats['total_propose_rewards'] += propose_reward
        
        return propose_reward
    
    def calculate_solve_reward(self, is_correct: bool, task_complexity: int = 1) -> float:
        """
        Calculate solver reward based on correctness and task complexity.
        
        Args:
            is_correct: Whether the solution was correct
            task_complexity: Complexity level of the solved task
            
        Returns:
            Solver reward value
        """
        # Base accuracy reward
        base_reward = 1.0 if is_correct else 0.0
        
        # Complexity bonus for solving harder tasks
        complexity_factor = 1.0 + (task_complexity - 1) * self.complexity_bonus
        
        # Apply complexity scaling
        solve_reward = base_reward * complexity_factor * self.solve_weight
        
        # Track history
        self.solve_history.append(solve_reward)
        self.stats['total_solve_rewards'] += solve_reward
        
        return solve_reward
    
    def calculate_combined_reward(self, propose_reward: float, solve_rewards: List[float]) -> float:
        """
        Calculate combined reward for model update.
        
        Args:
            propose_reward: Reward from proposal phase
            solve_rewards: List of rewards from solve phase
            
        Returns:
            Combined reward value
        """
        if not solve_rewards:
            return propose_reward
        
        # Average solve reward
        avg_solve_reward = np.mean(solve_rewards)
        
        # Combine with appropriate weighting
        combined = propose_reward * 0.4 + avg_solve_reward * 0.6
          # Apply advantage normalization
        normalized_reward = self._normalize_advantage(float(combined))
        
        return normalized_reward
    
    def update_success_rate(self, success_rate: float):
        """Update the rolling success rate history."""
        self.success_history.append(success_rate)
        
        # Update best success rate
        if success_rate > self.stats['best_success_rate']:
            self.stats['best_success_rate'] = success_rate
    
    def _normalize_advantage(self, reward: float) -> float:
        """
        Normalize advantage using running mean and standard deviation.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Normalized advantage
        """
        # Update running statistics
        self.advantage_mean = (self.advantage_momentum * self.advantage_mean + 
                             (1 - self.advantage_momentum) * reward)
        
        squared_diff = (reward - self.advantage_mean) ** 2
        self.advantage_std = math.sqrt(
            self.advantage_momentum * (self.advantage_std ** 2) + 
            (1 - self.advantage_momentum) * squared_diff
        )
        
        # Avoid division by zero
        if self.advantage_std < 1e-8:
            self.advantage_std = 1.0
        
        # Normalize
        normalized = (reward - self.advantage_mean) / self.advantage_std
        
        return normalized
    
    def _calculate_diversity_bonus(self) -> float:
        """
        Calculate diversity bonus to encourage varied task generation.
        
        Returns:
            Diversity bonus value
        """
        if len(self.propose_history) < 10:
            return 0.0
        
        # Calculate variance in recent propose rewards
        recent_rewards = list(self.propose_history)[-10:]
        variance = np.var(recent_rewards)
          # Higher variance indicates more diverse task generation
        diversity_bonus = min(0.1, float(variance * 0.5))
        
        return diversity_bonus
    
    def calculate_curriculum_reward(self, task_difficulty: float, 
                                  success_rate: float) -> float:
        """
        Calculate reward bonus for curriculum learning progression.
        
        Args:
            task_difficulty: Current difficulty level
            success_rate: Current success rate
            
        Returns:
            Curriculum reward bonus
        """
        # Reward for maintaining optimal difficulty
        difficulty_target = self._get_optimal_difficulty(success_rate)
        difficulty_alignment = 1.0 - abs(task_difficulty - difficulty_target) / max(task_difficulty, 1.0)
        
        # Reward for improvement over time
        if len(self.success_history) >= 2:
            recent_improvement = (self.success_history[-1] - 
                                self.success_history[-2])
            improvement_bonus = max(0.0, recent_improvement * 2.0)
        else:
            improvement_bonus = 0.0
        
        curriculum_reward = (difficulty_alignment * 0.7 + 
                           improvement_bonus * 0.3) * 0.2
        
        return curriculum_reward
    
    def _get_optimal_difficulty(self, success_rate: float) -> float:
        """
        Determine optimal difficulty based on current success rate.
        
        Args:
            success_rate: Current success rate
            
        Returns:
            Optimal difficulty level
        """
        if success_rate > self.target_success_rate + 0.1:
            # Too easy, increase difficulty
            return min(self.config.max_task_complexity, 
                      self.config.max_task_complexity * 0.8)
        elif success_rate < self.target_success_rate - 0.1:
            # Too hard, decrease difficulty
            return max(self.config.min_task_complexity,
                      self.config.min_task_complexity * 1.2)
        else:
            # Just right, maintain current level
            return (self.config.min_task_complexity + 
                   self.config.max_task_complexity) / 2
    
    def calculate_exploration_reward(self, novelty_score: float) -> float:
        """
        Calculate reward for exploring novel task types.
        
        Args:
            novelty_score: Measure of task novelty (0-1)
            
        Returns:
            Exploration reward
        """
        # Encourage exploration of novel tasks
        exploration_reward = novelty_score * 0.1
        
        return exploration_reward
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics."""
        stats = self.stats.copy()
        
        # Add recent performance metrics
        if self.propose_history:
            stats['recent_propose_reward'] = np.mean(list(self.propose_history)[-10:])
            stats['propose_reward_std'] = np.std(list(self.propose_history))
        
        if self.solve_history:
            stats['recent_solve_reward'] = np.mean(list(self.solve_history)[-10:])
            stats['solve_reward_std'] = np.std(list(self.solve_history))
        
        if self.success_history:
            stats['recent_success_rate'] = np.mean(list(self.success_history)[-10:])
            stats['success_rate_trend'] = self._calculate_trend(self.success_history)
        
        # Advantage normalization stats
        stats['advantage_mean'] = self.advantage_mean
        stats['advantage_std'] = self.advantage_std
        
        return stats
    
    def _calculate_trend(self, history: deque) -> float:
        """
        Calculate trend (slope) of recent history.
        
        Args:
            history: Deque of historical values
            
        Returns:
            Trend value (positive = improving, negative = declining)
        """
        if len(history) < 5:
            return 0.0
        
        recent_values = list(history)[-10:]
        x = np.arange(len(recent_values))
        
        # Simple linear regression
        try:
            slope = np.polyfit(x, recent_values, 1)[0]
            return slope
        except:
            return 0.0
    
    def reset_statistics(self):
        """Reset all reward statistics."""
        self.success_history.clear()
        self.propose_history.clear()
        self.solve_history.clear()
        
        self.advantage_mean = 0.0
        self.advantage_std = 1.0
        
        self.stats = {
            'total_propose_rewards': 0.0,
            'total_solve_rewards': 0.0,
            'reward_episodes': 0,
            'best_success_rate': 0.0,
            'reward_variance': 0.0
        }
    
    def adjust_reward_weights(self, propose_weight: float, solve_weight: float):
        """
        Adjust the relative weights of propose and solve rewards.
        
        Args:
            propose_weight: Weight for proposer rewards
            solve_weight: Weight for solver rewards
        """
        self.propose_weight = max(0.0, propose_weight)
        self.solve_weight = max(0.0, solve_weight)
        
        self.logger.info(f"Reward weights updated: propose={self.propose_weight}, solve={self.solve_weight}")
    
    def get_adaptive_target(self) -> float:
        """
        Get adaptive target success rate based on learning progress.
        
        Returns:
            Adaptive target success rate
        """
        if len(self.success_history) < 20:
            return self.target_success_rate
        
        # Calculate recent performance
        recent_performance = np.mean(list(self.success_history)[-20:])
        
        # Adjust target based on consistent performance
        if recent_performance > self.target_success_rate + 0.05:
            # Consistently exceeding target, raise the bar
            adaptive_target = min(0.8, self.target_success_rate + 0.05)
        elif recent_performance < self.target_success_rate - 0.05:
            # Consistently below target, lower the bar temporarily
            adaptive_target = max(0.2, self.target_success_rate - 0.05)
        else:
            adaptive_target = self.target_success_rate
        
        return adaptive_target
