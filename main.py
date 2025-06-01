"""
Main entry point for the Absolute Zero Reasoner (AZR) system.
Implements self-bootstrapping reasoning through reinforcement learning.
"""

import argparse
import os
import sys
import logging
import datetime
import json
import traceback
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from src.azr_system import AbsoluteZeroReasoner, AZRConfig
from src.model_wrapper import ModelWrapper
from src.task_manager import ReasoningTask, ReasoningType
from utils.evaluation import evaluate_model
from utils.logging_utils import setup_logging


def save_evaluation_results(results: dict, azr_system):
    """Save evaluation results to file."""
    import json
    import datetime
    from pathlib import Path
    
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare comprehensive results
    eval_data = {
        'timestamp': timestamp,
        'evaluation_results': results,
        'training_metrics': {
            'total_episodes': len(azr_system.metrics['propose_rewards']),
            'final_propose_reward': azr_system.metrics['propose_rewards'][-1] if azr_system.metrics['propose_rewards'] else 0,
            'final_solve_reward': azr_system.metrics['solve_rewards'][-1] if azr_system.metrics['solve_rewards'] else 0,
            'final_success_rate': azr_system.metrics['success_rates'][-1] if azr_system.metrics['success_rates'] else 0,
            'average_episode_time': sum(azr_system.metrics['episode_times']) / len(azr_system.metrics['episode_times']) if azr_system.metrics['episode_times'] else 0,
            'total_tasks_generated': azr_system.total_tasks_generated,
            'total_tasks_solved': azr_system.total_tasks_solved
        },
        'task_manager_stats': azr_system.task_manager.get_buffer_stats(),
        'config': vars(azr_system.config)
    }
    
    filepath = outputs_dir / f"azr_evaluation_{timestamp}.json"
    with open(filepath, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"📄 Evaluation results saved to: {filepath}")


def create_demo_config() -> AZRConfig:
    """Create a demo configuration optimized for CPU execution."""
    return AZRConfig(
        model_name="distilgpt2",  # DistilGPT2 for code generation
        max_length=2000,
        temperature=0.7,  # Lower temperature for more focused code generation
        learning_rate=5e-5,
        batch_size=2,  # Small batch for CPU
        max_episodes=100,  # Reduced for demo
        buffer_size=500,
        max_task_complexity=3,
        min_task_complexity=1,
        tasks_per_episode=5,  # Reduced for faster demo
        execution_timeout=3.0,
        max_memory_mb=50,
        success_rate_target=0.5,
        difficulty_adjustment_rate=0.1,
        log_level="INFO",
        save_interval=20
    )


def create_coding_config() -> AZRConfig:
    """Create a configuration optimized for coding tasks."""
    return AZRConfig(
        model_name="microsoft/CodeGPT-small-py",  # Better code model if available
        max_length=2000,
        temperature=0.5,  # Even lower temperature for code
        learning_rate=3e-5,  # Lower learning rate for stability
        batch_size=1,  # Smaller batch for complex code tasks
        max_episodes=50,  # Shorter for testing
        buffer_size=200,
        max_task_complexity=2,  # Start with simpler tasks
        min_task_complexity=1,
        tasks_per_episode=3,  # Fewer tasks per episode
        execution_timeout=5.0,  # More time for code execution
        max_memory_mb=100,
        success_rate_target=0.3,  # Lower initial target
        difficulty_adjustment_rate=0.05,  # Slower progression
        log_level="INFO",
        save_interval=10
    )


def create_alternative_config() -> AZRConfig:
    """Create alternative configuration with different model."""
    return AZRConfig(
        model_name="Salesforce/codegen-350M-mono",  # CodeGen model for Python
        max_length=2000,
        temperature=0.6,
        learning_rate=2e-5,
        batch_size=1,
        max_episodes=30,
        buffer_size=150,
        max_task_complexity=2,
        min_task_complexity=1,
        tasks_per_episode=3,
        execution_timeout=4.0,
        max_memory_mb=80,
        success_rate_target=0.4,
        difficulty_adjustment_rate=0.08,
        log_level="INFO",
        save_interval=10
    )


def run_demo():
    """Run a demonstration of the AZR system."""
    print("🚀 Starting Absolute Zero Reasoner (AZR) Demo")
    print("=" * 50)
    
    # Setup
    config = create_demo_config()
    setup_logging(config.log_level)
    
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/generated", exist_ok=True)
    os.makedirs("models/azr_trained", exist_ok=True)
    
    # Initialize AZR system
    print("🔧 Initializing AZR system...")
    azr = AbsoluteZeroReasoner(config)
    
    # Run training
    print("🎯 Starting self-bootstrapping training...")
    try:
        azr.train()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    # Evaluation
    print("\n📊 Evaluating trained model...")
    test_tasks = create_test_tasks()
    results = azr.evaluate(test_tasks)
    
    # Save evaluation results
    save_evaluation_results(results, azr)
    
    print(f"\n✅ Final Results:")
    print(f"   Accuracy: {results['accuracy']:.2%}")
    print(f"   Correct: {results['correct']}/{results['total']}")
    print(f"   Total tasks generated: {azr.total_tasks_generated}")
    print(f"   Total tasks solved: {azr.total_tasks_solved}")
    
    return azr, results


def create_test_tasks() -> list:
    """Create a set of test tasks for evaluation."""
    test_tasks = [
        ReasoningTask(
            type=ReasoningType.DEDUCTION.value,
            program="lambda x: x * 2",
            input="5",
            expected_output="10",
            complexity=1
        ),
        ReasoningTask(
            type=ReasoningType.DEDUCTION.value,
            program="lambda x, y: x + y if x > 0 else y",
            input="(3, 4)",
            expected_output="7",
            complexity=2
        ),
        ReasoningTask(
            type=ReasoningType.ABDUCTION.value,
            program="lambda x: x if x > 0 else 0",
            input="5",
            expected_output="5",
            complexity=2
        ),
        ReasoningTask(
            type=ReasoningType.INDUCTION.value,
            program="lambda x: x ** 2",
            input="3",
            expected_output="9",
            complexity=2
        )
    ]
    return test_tasks


def run_demo_with_config(config: AZRConfig):
    """Run a demonstration with the specified configuration."""
    print("🚀 Starting Absolute Zero Reasoner (AZR) Demo")
    print("=" * 50)
    
    # Display model information
    if config.use_pretrained_azr:
        print(f"🎯 Using AZR trained model: {config.azr_model_preference}")
    else:
        print(f"📦 Using base model: {config.model_name}")
    
    # Setup
    setup_logging(config.log_level)
    
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/generated", exist_ok=True)
    os.makedirs("models/azr_trained", exist_ok=True)
    
    # Initialize AZR system
    print("🔧 Initializing AZR system...")
    azr = AbsoluteZeroReasoner(config)
    
    # Display loaded model information
    try:
        model_info = azr.model.get_model_size()
        print(f"📏 Model loaded: {model_info['total_parameters']:,} parameters "
              f"({model_info['model_size_mb']:.1f} MB)")
        
        if hasattr(azr.model, 'current_model_type'):
            print(f"🏷️  Model type: {azr.model.current_model_type}")
            
        if hasattr(azr.model, 'current_model_metadata'):
            metadata = azr.model.current_model_metadata
            if metadata:
                print(f"📈 Training info: {metadata.get('total_episodes', 'N/A')} episodes, "
                      f"Success rate: {metadata.get('avg_success_rate', 'N/A')}")
    except Exception as e:
        print(f"ℹ️  Could not retrieve model info: {e}")
    
    # Run training
    print("🎯 Starting self-bootstrapping training...")
    try:
        azr.train()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    # Evaluation
    print("\n📊 Evaluating trained model...")
    results = evaluate_model(azr)
    save_evaluation_results(results, azr)
    
    return azr, results


def run_enhanced_interactive_mode():
    """Run AZR in enhanced interactive mode with model selection."""
    print("🔬 Enhanced Interactive AZR Mode")
    print("=" * 40)
    
    # Let user choose model
    config = interactive_model_selection()
    azr = AbsoluteZeroReasoner(config)
    
    # Display current model info
    try:
        model_info = azr.model.get_model_size()
        print(f"\n✅ Loaded model: {model_info['total_parameters']:,} parameters")
        if hasattr(azr.model, 'current_model_type'):
            print(f"🏷️  Model type: {azr.model.current_model_type}")
    except:
        pass
    
    while True:
        print("\n" + "="*40)
        print("🎮 Interactive Options:")
        print("1. Generate a task")
        print("2. Solve a task")
        print("3. View task buffers")
        print("4. Switch model")
        print("5. Model information")
        print("6. Run mini-training (10 episodes)")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            generate_task_interactive(azr)
        elif choice == "2":
            solve_task_interactive(azr)
        elif choice == "3":
            view_task_buffers(azr)
        elif choice == "4":
            azr = switch_model_interactive(azr)
        elif choice == "5":
            display_model_info(azr)
        elif choice == "6":
            run_mini_training(azr)
        elif choice == "7":
            break
        else:
            print("❌ Invalid choice")


def interactive_model_selection() -> AZRConfig:
    """Interactive model selection interface."""
    print("\n🤖 Model Selection")
    print("-" * 30)
    print("1. Use base model")
    print("2. Use trained AZR model")
    print("3. List available models")
    
    while True:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            return select_base_model()
        elif choice == "2":
            return select_azr_model()
        elif choice == "3":
            list_available_models()
            continue
        else:
            print("❌ Invalid choice")


def select_base_model() -> AZRConfig:
    """Interactive base model selection."""
    print("\n📦 Base Model Selection")
    print("-" * 25)
    
    recommended_models = [
        ("distilgpt2", "DistilGPT2 - Fast and efficient"),
        ("microsoft/DialoGPT-small", "DialoGPT - Conversational"),
        ("microsoft/CodeGPT-small-py", "CodeGPT - Python focused"),
        ("gpt2", "GPT-2 - Classic language model"),
    ]
    
    print("Recommended models:")
    for i, (model, description) in enumerate(recommended_models, 1):
        print(f"  {i}. {model} - {description}")
    print(f"  {len(recommended_models) + 1}. Custom model (enter name)")
    
    while True:
        choice = input(f"\nSelect model (1-{len(recommended_models) + 1}): ").strip()
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(recommended_models):
                model_name = recommended_models[choice_num - 1][0]
                break
            elif choice_num == len(recommended_models) + 1:
                model_name = input("Enter model name: ").strip()
                if not model_name:
                    print("❌ Model name cannot be empty")
                    continue
                break
            else:
                print(f"❌ Invalid choice. Enter 1-{len(recommended_models) + 1}")
        except ValueError:
            print("❌ Please enter a number")
    
    episodes = get_episodes_input()
    
    return AZRConfig(
        model_name=model_name,
        use_pretrained_azr=False,
        max_episodes=episodes,
        batch_size=2,
        temperature=0.7,
        log_level="INFO"
    )


def select_azr_model() -> AZRConfig:
    """Interactive AZR model selection."""
    print("\n🎯 AZR Model Selection")
    print("-" * 25)
    
    # Try to list available AZR models
    try:
        temp_config = AZRConfig()
        temp_wrapper = ModelWrapper(temp_config)
        available_models = temp_wrapper.list_available_models()
        
        azr_models = available_models.get('trained_azr', [])
        
        if not azr_models:
            print("❌ No trained AZR models found.")
            print("Would you like to:")
            print("1. Use base model instead")
            print("2. Exit")
            
            choice = input("Select option (1-2): ").strip()
            if choice == "1":
                return select_base_model()
            else:
                exit(0)
        
        print("Available AZR models:")
        print("1. latest - Use the most recent model")
        print("2. best - Use the best performing model")
        
        for i, model in enumerate(azr_models, 3):
            print(f"  {i}. {model}")
        
        while True:
            choice = input(f"\nSelect model (1-{len(azr_models) + 2}): ").strip()
            
            try:
                choice_num = int(choice)
                if choice_num == 1:
                    preference = "latest"
                    break
                elif choice_num == 2:
                    preference = "best"
                    break
                elif 3 <= choice_num <= len(azr_models) + 2:
                    preference = azr_models[choice_num - 3]
                    break
                else:
                    print(f"❌ Invalid choice. Enter 1-{len(azr_models) + 2}")
            except ValueError:
                print("❌ Please enter a number")
        
    except Exception as e:
        print(f"❌ Error accessing models: {e}")
        preference = input("Enter AZR model preference (latest/best/model_name): ").strip()
        if not preference:
            preference = "latest"
    
    episodes = get_episodes_input()
    
    return AZRConfig(
        use_pretrained_azr=True,
        azr_model_preference=preference,
        fallback_to_base=True,
        max_episodes=episodes,
        batch_size=2,
        learning_rate=1e-5,
        temperature=0.6,
        log_level="INFO"
    )


def get_episodes_input() -> int:
    """Get number of episodes from user input."""
    while True:
        episodes_input = input("Number of episodes (default 100): ").strip()
        if not episodes_input:
            return 100
        try:
            episodes = int(episodes_input)
            if episodes > 0:
                return episodes
            else:
                print("❌ Episodes must be positive")
        except ValueError:
            print("❌ Please enter a valid number")


def generate_task_interactive(azr):
    """Interactive task generation."""
    print("\n🎯 Task Generation")
    reasoning_type = input("Reasoning type (deduction/abduction/induction): ").strip().lower()
    
    if reasoning_type in ['deduction', 'abduction', 'induction']:
        try:
            prompt = azr._create_task_prompt(reasoning_type)
            task = azr.model.generate_task(prompt, reasoning_type)
            print(f"\n✅ Generated task:")
            print(f"   Type: {task.get('type', 'Unknown')}")
            print(f"   Program: {task.get('program', 'N/A')}")
            print(f"   Input: {task.get('input', 'N/A')}")
            print(f"   Expected Output: {task.get('expected_output', 'N/A')}")
            print(f"   Complexity: {task.get('complexity', 'N/A')}")
        except Exception as e:
            print(f"❌ Error generating task: {e}")
    else:
        print("❌ Invalid reasoning type")


def solve_task_interactive(azr):
    """Interactive task solving."""
    print("\n🧠 Task Solving")
    program = input("Enter program (e.g., lambda x: x * 2): ").strip()
    input_data = input("Enter input (e.g., 5): ").strip()
    
    if program and input_data:
        try:
            from src.task_manager import ReasoningTask
            task = ReasoningTask(
                type="deduction",
                program=program,
                input=input_data,
                complexity=1
            )
            solution_prompt = azr._create_solution_prompt(task)
            solution = azr.model.generate_solution(solution_prompt)
            print(f"\n✅ Generated solution: {solution}")
            
            # Try to verify the solution
            try:
                verified = azr._verify_solution(task, solution)
                print(f"✅ Verification: {'Correct' if verified else 'Incorrect'}")
            except Exception as e:
                print(f"⚠️  Could not verify solution: {e}")
                
        except Exception as e:
            print(f"❌ Error solving task: {e}")
    else:
        print("❌ Program and input are required")


def view_task_buffers(azr):
    """View current task buffers."""
    print("\n📊 Task Buffers")
    print("-" * 20)
    
    try:
        stats = azr.task_manager.get_buffer_stats()
        for reasoning_type, count in stats.items():
            print(f"  {reasoning_type.capitalize()}: {count} tasks")
            
        total_tasks = sum(stats.values())
        print(f"\n  Total: {total_tasks} tasks")
        
    except Exception as e:
        print(f"❌ Error accessing task buffers: {e}")


def switch_model_interactive(azr):
    """Interactive model switching."""
    print("\n🔄 Model Switching")
    print("Current model will be replaced. Continue? (y/n)")
    
    if input().strip().lower() == 'y':
        new_config = interactive_model_selection()
        print("\n🔧 Loading new model...")
        try:
            new_azr = AbsoluteZeroReasoner(new_config)
            print("✅ Model switched successfully!")
            return new_azr
        except Exception as e:
            print(f"❌ Error switching model: {e}")
            print("Keeping current model")
            return azr
    else:
        print("Model switch cancelled")
        return azr


def display_model_info(azr):
    """Display current model information."""
    print("\n📋 Model Information")
    print("-" * 25)
    
    try:
        model_info = azr.model.get_model_size()
        print(f"Parameters: {model_info['total_parameters']:,}")
        print(f"Trainable: {model_info['trainable_parameters']:,}")
        print(f"Size: {model_info['model_size_mb']:.1f} MB")
        
        if hasattr(azr.model, 'current_model_type'):
            print(f"Type: {azr.model.current_model_type}")
        
        if hasattr(azr.model, 'current_model_metadata') and azr.model.current_model_metadata:
            metadata = azr.model.current_model_metadata
            print(f"Episodes trained: {metadata.get('total_episodes', 'N/A')}")
            print(f"Success rate: {metadata.get('avg_success_rate', 'N/A')}")
            print(f"Training completed: {metadata.get('timestamp', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error retrieving model info: {e}")


def run_mini_training(azr):
    """Run a short training session."""
    print("\n🏃‍♂️ Mini Training (10 episodes)")
    print("This will train the current model for 10 episodes...")
    
    if input("Continue? (y/n): ").strip().lower() == 'y':
        try:
            original_episodes = azr.config.max_episodes
            azr.config.max_episodes = 10
            
            print("🎯 Starting mini training...")
            azr.train()
            
            azr.config.max_episodes = original_episodes
            print("✅ Mini training completed!")
            
        except Exception as e:
            print(f"❌ Error during mini training: {e}")
    else:
        print("Mini training cancelled")


def run_evaluation_mode(config: AZRConfig):
    """Run evaluation mode to compare model performance."""
    print("📊 AZR Model Evaluation Mode")
    print("=" * 40)
    
    azr = AbsoluteZeroReasoner(config)
    
    print("🔍 Running comprehensive evaluation...")
    results = evaluate_model(azr)
    
    print("\n📈 Evaluation Results:")
    print("-" * 25)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    save_evaluation_results(results, azr)


def run_demo():
    """Original run demo function for backward compatibility."""
    config = create_demo_config()
    return run_demo_with_config(config)


def list_available_models():
    """List all available models (base and trained AZR models)."""
    print("🤖 Available Models:")
    print("=" * 50)
    
    from pathlib import Path
    
    # Define model directories
    models_dir = Path(__file__).parent / "models"
    base_models_dir = models_dir / "base_models"
    azr_models_dir = models_dir / "azr_trained"
    
    try:
        print("\n📦 Base Models:")
        base_models_found = False
        if base_models_dir.exists():
            for i, model_dir in enumerate(base_models_dir.iterdir(), 1):
                if model_dir.is_dir():
                    # Check if it has required model files
                    config_file = model_dir / "config.json"
                    if config_file.exists():
                        print(f"  {i}. {model_dir.name}")
                        base_models_found = True
        
        if not base_models_found:
            print("  No base models found in local cache")
            
        print("\n🎯 Trained AZR Models:")
        azr_models_found = False
        if azr_models_dir.exists():
            for i, model_dir in enumerate(azr_models_dir.iterdir(), 1):
                if model_dir.is_dir():
                    # Check if it has required model files
                    config_file = model_dir / "config.json"
                    tokenizer_file = model_dir / "tokenizer.json"
                    if config_file.exists() and tokenizer_file.exists():
                        print(f"  {i}. {model_dir.name}")
                        azr_models_found = True
        
        if not azr_models_found:
            print("  No trained AZR models found")
            
        print("\n💡 Recommended Base Models for Download:")
        recommended_models = [
            "distilgpt2",
            "microsoft/DialoGPT-small", 
            "gpt2"
        ]
        for i, model in enumerate(recommended_models, 1):
            print(f"  {i}. {model}")
            
    except Exception as e:
        print(f"Error listing models: {e}")
    
    print("\n" + "=" * 50)


def create_model_config(model_type: str = "base", model_selection: str = None, 
                       episodes: int = 100) -> AZRConfig:
    """Create configuration based on model type and selection."""
    
    if model_type == "azr":
        # AZR trained model configuration
        config = AZRConfig(
            use_pretrained_azr=True,
            azr_model_preference=model_selection or "latest",
            fallback_to_base=True,
            max_episodes=episodes,
            batch_size=2,
            learning_rate=1e-5,  # Lower learning rate for fine-tuning
            temperature=0.6,
            log_level="INFO"
        )
    else:
        # Base model configuration
        base_model = model_selection or "distilgpt2"
        config = AZRConfig(
            model_name=base_model,
            use_pretrained_azr=False,
            max_episodes=episodes,
            batch_size=2,
            learning_rate=1e-4,
            temperature=0.7,
            log_level="INFO"
        )
    
    return config


def main():
    """Main entry point with enhanced command line interface."""
    parser = argparse.ArgumentParser(
        description="Absolute Zero Reasoner (AZR) - Self-bootstrapping AI reasoning system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python main.py --list-models
  
  # Run with base model
  python main.py --mode demo --model-type base --model microsoft/CodeGPT-small-py
  
  # Run with latest trained AZR model
  python main.py --mode demo --model-type azr --model latest
  
  # Run with specific AZR model
  python main.py --mode train --model-type azr --model azr_model_ep100_20241201_143022
  
  # Interactive mode with model selection
  python main.py --mode interactive
        """
    )
    
    # Main execution modes
    parser.add_argument("--mode", choices=["demo", "train", "interactive", "evaluate"], 
                       default="demo", help="Execution mode")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of training episodes")
    
    # Model selection arguments
    parser.add_argument("--model-type", choices=["base", "azr"], default="base",
                       help="Type of model to use: 'base' for base models, 'azr' for trained AZR models")
    parser.add_argument("--model", type=str, default=None,
                       help="Model to use. For base: HuggingFace model name. For AZR: 'latest', 'best', or specific model name")
    parser.add_argument("--list-models", action="store_true",
                       help="List all available models and exit")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (auto-selected based on model type if not specified)")
    
    # System arguments
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--force-download", action="store_true",
                       help="Force download of base models even if cached")
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.list_models:
        list_available_models()
        return
    
    # Create configuration based on arguments
    config = create_model_config(
        model_type=args.model_type,
        model_selection=args.model,
        episodes=args.episodes
    )
    
    # Apply additional arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.verbose:
        config.log_level = "DEBUG"
    if args.force_download:
        config.use_local_models = False
    
    # Execute based on mode
    try:
        if args.mode == "demo":
            print(f"🚀 Running AZR Demo with {args.model_type} model")
            if args.model_type == "azr":
                print(f"   AZR Model: {args.model or 'latest'}")
            else:
                print(f"   Base Model: {args.model or config.model_name}")
            azr, results = run_demo_with_config(config)
            
        elif args.mode == "train":
            print(f"🎯 Training AZR with {args.model_type} model for {args.episodes} episodes")
            azr = AbsoluteZeroReasoner(config)
            azr.train()
            
        elif args.mode == "interactive":
            run_enhanced_interactive_mode()
            
        elif args.mode == "evaluate":
            print(f"📊 Evaluating model performance")
            run_evaluation_mode(config)
        
        print("\n🎉 AZR execution completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
