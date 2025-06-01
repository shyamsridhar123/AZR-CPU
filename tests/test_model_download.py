#!/usr/bin/env python3
"""
Download and test model for AZR system.
This script ensures the model is properly downloaded and cached.
"""

import sys
import os
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Now import from src directory
from src.azr_system import AZRConfig
from src.model_wrapper import ModelWrapper
import logging

def main():
    """Download and test the model."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting model download and test...")
    
    # Create configuration
    config = AZRConfig()
    config.use_local_models = True
    config.save_models_locally = True
    
    try:
        # Initialize model wrapper (this will download the model)
        logger.info(f"Initializing model wrapper for: {config.model_name}")
        model_wrapper_instance = ModelWrapper(config)        # Test task generation
        logger.info("Testing task generation...")
        task_prompt = "Generate a simple Python function and input. Format: {'program': 'lambda x: ...', 'input': '...'}"
        task_result = model_wrapper_instance.generate_task(task_prompt)
        logger.info(f"Generated task: {task_result}")
        
        # Test solution generation
        logger.info("Testing solution generation...")
        solution_prompt = "Given program: lambda x: x * 2 and input: 5, what is the output?"
        solution_result = model_wrapper_instance.generate_solution(solution_prompt)
        logger.info(f"Generated solution: {solution_result}")
        logger.info("Model download and test completed successfully!")
        
        # Print model directory info
        models_dir = Path(__file__).parent.parent / "models"
        logger.info(f"Models directory: {models_dir}")
        if (models_dir / 'base_models').exists():
            logger.info(f"Base models: {list((models_dir / 'base_models').iterdir())}")
        else:
            logger.info("Base models directory not found or empty")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model download/test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
