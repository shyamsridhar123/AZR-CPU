"""
AZR Utilities Package

Common utilities for the Absolute Zero Reasoner system.
"""

from .evaluation import evaluate_model
from .logging_utils import setup_logging

__all__ = [
    "evaluate_model",
    "setup_logging"
]
