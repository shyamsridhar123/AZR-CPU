#!/usr/bin/env python3
"""
Download and test model for AZR system.
This script ensures the model is properly downloaded and cached.
Updated to test the new generate() method and prompt system integration.
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Now import from src directory
from src.azr_system import AZRConfig
from src.model_wrapper import ModelWrapper
import logging

# Try to import prompt system
try:
    from Prompts.prompt_manager import PromptManager
    prompt_system_available = True
except ImportError:
    prompt_system_available = False

def main(use_small_model=False):
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
    if use_small_model:
        # Use a smaller model for faster testing
        logger.info("Using small model for testing")
        config.model_name = "distilgpt2"  # Smaller model
    
    config.use_local_models = True
    config.save_models_locally = True
    
    try:
        # Initialize model wrapper (this will download the model)
        logger.info(f"Initializing model wrapper for: {config.model_name}")
        model_wrapper_instance = ModelWrapper(config)
        
        # Test general text generation with new generate() method
        logger.info("Testing general text generation with generate()...")
        generation_prompt = "Complete this sentence: The Absolute Zero Reasoner is a"
        generation_result = model_wrapper_instance.generate(generation_prompt)
        logger.info(f"Generated completion: {generation_result[:100]}...")
        
        # Test task generation
        logger.info("Testing task generation...")
        task_prompt = "Generate a simple Python function and input. Format: {'program': 'lambda x: ...', 'input': '...'}"
        task_result = model_wrapper_instance.generate_task(task_prompt)
        logger.info(f"Generated task: {task_result}")
        
        # Test solution generation
        logger.info("Testing solution generation...")
        solution_prompt = "Given program: lambda x: x * 2 and input: 5, what is the output?"
        solution_result = model_wrapper_instance.generate_solution(solution_prompt)
        logger.info(f"Generated solution: {solution_result}")
        
        # Test solution generation with task data for complexity
        logger.info("Testing solution generation with task data...")
        task_data = {
            'type': 'deduction',
            'program': 'lambda x: x * 2',
            'input': '5',
            'complexity': 3
        }
        complex_solution = model_wrapper_instance.generate_solution(
            solution_prompt, 
            task_type='deduction',
            task_data=task_data
        )
        logger.info(f"Generated complex solution: {complex_solution[:100]}...")
        
        # Test prompt system integration if available
        if prompt_system_available:
            logger.info("Testing prompt system integration...")
            
            prompt_manager = PromptManager()
            
            # Test task generation with advanced prompt
            advanced_task_prompt = prompt_manager.get_task_generation_prompt(
                reasoning_type="deduction",
                complexity=2,
                include_examples=True
            )
            
            advanced_task = model_wrapper_instance.generate_task(
                advanced_task_prompt,
                reasoning_type="deduction",
                complexity=2
            )
            
            logger.info(f"Generated task with advanced prompt: {advanced_task}")
        else:
            logger.warning("Prompt system not available, skipping related tests")
        
        logger.info("Model download and test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in model testing: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model download and functionality")
    parser.add_argument("--small", action="store_true", help="Use a small model for faster testing")
    args = parser.parse_args()
    
    success = main(use_small_model=args.small)
    sys.exit(0 if success else 1)
