"""
Main entry point for the Absolute Zero Reasoner (AZR) system.
Implements self-bootstrapping reasoning through reinforcement learning.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from src.azr_system import AbsoluteZeroReasoner, AZRConfig
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
        model_name="microsoft/DialoGPT-small",  # Lightweight model for CPU
        max_length=256,
        temperature=0.8,
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


def run_interactive_mode():
    """Run AZR in interactive mode for manual testing."""
    print("🔬 Interactive AZR Mode")
    print("=" * 30)
    
    config = create_demo_config()
    azr = AbsoluteZeroReasoner(config)
    
    while True:
        print("\nOptions:")
        print("1. Generate a task")
        print("2. Solve a task")
        print("3. View task buffers")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            reasoning_type = input("Reasoning type (deduction/abduction/induction): ").strip()
            if reasoning_type in ['deduction', 'abduction', 'induction']:
                prompt = azr._create_task_prompt(reasoning_type)
                task = azr.model.generate_task(prompt, reasoning_type)
                print(f"Generated task: {task}")
            else:
                print("Invalid reasoning type")
        
        elif choice == "2":
            program = input("Enter program: ").strip()
            input_data = input("Enter input: ").strip()
            task = ReasoningTask(
                type="deduction",
                program=program,
                input=input_data,
                complexity=1
            )
            solution_prompt = azr._create_solution_prompt(task)
            solution = azr.model.generate_solution(solution_prompt)
            print(f"Generated solution: {solution}")
        
        elif choice == "3":
            for reasoning_type, buffer in azr.task_manager.buffers.items():
                print(f"{reasoning_type}: {len(buffer)} tasks")
        
        elif choice == "4":
            break
        
        else:
            print("Invalid choice")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description="Absolute Zero Reasoner (AZR)")
    parser.add_argument("--mode", choices=["demo", "train", "interactive"], 
                       default="demo", help="Execution mode")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of training episodes")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small",
                       help="Model name to use")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        azr, results = run_demo()
        
    elif args.mode == "train":
        config = create_demo_config()
        config.max_episodes = args.episodes
        config.model_name = args.model
        config.batch_size = args.batch_size
        if args.verbose:
            config.log_level = "DEBUG"
        
        azr = AbsoluteZeroReasoner(config)
        azr.train()
        
    elif args.mode == "interactive":
        run_interactive_mode()
    
    print("\n🎉 AZR execution completed!")


if __name__ == "__main__":
    main()
