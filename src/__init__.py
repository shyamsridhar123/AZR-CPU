"""
Absolute Zero Reasoner (AZR) - Core Package

A self-contained system where a language model bootstraps reasoning capabilities
through self-generated tasks using reinforcement learning.
"""

from .azr_system import AbsoluteZeroReasoner, AZRConfig
from .task_manager import TaskManager, ReasoningTask, ReasoningType
from .code_executor import CodeExecutor
from .model_wrapper import ModelWrapper
from .reward_calculator import RewardCalculator

__version__ = "0.1.0"
__author__ = "AZR Development Team"

__all__ = [
    "AbsoluteZeroReasoner",
    "AZRConfig", 
    "TaskManager",
    "ReasoningTask",
    "ReasoningType",
    "CodeExecutor",
    "ModelWrapper",
    "RewardCalculator"
]
