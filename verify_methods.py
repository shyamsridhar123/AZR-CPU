#!/usr/bin/env python3
"""
Verify the exact method names in each module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=== VERIFYING ACTUAL METHOD NAMES ===\n")

# Check CodeExecutor
print("1. CodeExecutor:")
from src.code_executor import CodeExecutor
from src.azr_system import AZRConfig

config = AZRConfig()
executor = CodeExecutor(config)
print("   Methods:", [m for m in dir(executor) if not m.startswith('_') and callable(getattr(executor, m))])

# Check TaskManager
print("\n2. TaskManager:")
from src.task_manager import TaskManager
manager = TaskManager(config)
print("   Methods:", [m for m in dir(manager) if not m.startswith('_') and callable(getattr(manager, m))])

# Check RewardCalculator
print("\n3. RewardCalculator:")
from src.reward_calculator import RewardCalculator
calculator = RewardCalculator(config)
print("   Methods:", [m for m in dir(calculator) if not m.startswith('_') and callable(getattr(calculator, m))])