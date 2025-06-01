#!/usr/bin/env python3
"""
AZR CPU System Requirements Analysis
Calculates detailed system requirements for each task type.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.azr_system import AZRConfig
from main import create_demo_config
from utils.model_manager import get_model_manager, ensure_model_cached
import psutil
import torch
import time
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_system_requirements():
    """Analyze CPU system requirements for AZR task types."""
    
    print("=" * 60)
    print("🧠 AZR CPU SYSTEM REQUIREMENTS ANALYSIS")
    print("=" * 60)
    print()
    
    # Get system information
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    print("🖥️  SYSTEM INFORMATION")
    print("-" * 30)
    print(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"  CPU Cores: {cpu_count}")
    if cpu_freq:
        print(f"  CPU Frequency: {cpu_freq.current:.0f} MHz")
    print()
    
    # Configuration analysis
    print("⚙️  CONFIGURATION ANALYSIS")
    print("-" * 30)
    
    default_config = AZRConfig()
    demo_config = create_demo_config()
    
    configs = {
        "Default": default_config,
        "Demo": demo_config
    }
    
    for name, config in configs.items():
        print(f"  {name} Configuration:")
        print(f"    Model: {config.model_name}")
        print(f"    Batch size: {config.batch_size}")
        print(f"    Buffer size: {config.buffer_size}")
        print(f"    Tasks per episode: {config.tasks_per_episode}")
        print(f"    Memory limit: {config.max_memory_mb} MB")
        print(f"    Execution timeout: {config.execution_timeout}s")
        print()
    
    # Model size analysis using model manager
    print("🤖 MODEL SIZE ANALYSIS")
    print("-" * 30)
    
    try:
        # Get model manager and ensure model is cached
        model_manager = get_model_manager()
        model_name = demo_config.model_name
        
        print(f"  Checking model cache for: {model_name}")
        
        # Ensure model is cached (downloads if necessary)
        model_info = ensure_model_cached(model_name)
        
        if model_info:
            print(f"  ✅ Model loaded from cache: models/{model_name.replace('/', '_')}")
            print(f"  Total Parameters: {model_info.get('total_parameters', 'N/A'):,}")
            print(f"  Trainable Parameters: {model_info.get('trainable_parameters', 'N/A'):,}")
            print(f"  Model Size: {model_info.get('model_size_mb', 0):.1f} MB")
            print(f"  Model Type: CPU-optimized (float32)")
        else:
            # If model info not available, use estimates
            print(f"  ⚠️  Using estimated values for {model_name}")
            model_info = {"model_size_mb": 250}  # Estimated for DialoGPT-small
        
        print()
        
        # Check all cached models
        cached_models = model_manager.get_cached_models()
        if cached_models:
            print("  📦 Cached Models:")
            for cached_name, cached_info in cached_models.items():
                print(f"    - {cached_name}: {cached_info.get('model_size_mb', 0):.1f} MB")
            cache_size = model_manager.get_cache_size()
            print(f"  Total Cache Size: {cache_size['total_size_mb']:.1f} MB ({cache_size['total_size_gb']:.2f} GB)")
        print()
        
        # Memory usage calculation
        base_memory = model_info.get('model_size_mb', 250)
        
        # Calculate per-task-type requirements
        print("📊 TASK TYPE REQUIREMENTS")
        print("-" * 30)
        
        task_types = {
            "Deduction": {
                "description": "Execute program with input → predict output",
                "complexity": "Low-Medium",
                "memory_multiplier": 1.0,
                "cpu_intensity": "Medium",
                "examples": ["lambda x: x * 2 + 1", "lambda x: x if x > 0 else 0"]
            },
            "Abduction": {
                "description": "Given input + output → discover program",
                "complexity": "Medium-High", 
                "memory_multiplier": 1.2,
                "cpu_intensity": "High",
                "examples": ["Find: lambda x: x**2 given input=3, output=9"]
            },
            "Induction": {
                "description": "Given examples → synthesize pattern",
                "complexity": "High",
                "memory_multiplier": 1.5,
                "cpu_intensity": "Very High",
                "examples": ["Find pattern: (1,3), (2,5), (3,7) → lambda x: 2*x+1"]
            }
        }
        
        buffer_overhead = demo_config.buffer_size * 0.001  # ~1KB per task
        
        for task_type, info in task_types.items():
            task_memory = base_memory * info["memory_multiplier"] + buffer_overhead
            
            print(f"  {task_type} Tasks:")
            print(f"    Description: {info['description']}")
            print(f"    Complexity: {info['complexity']}")
            print(f"    CPU Intensity: {info['cpu_intensity']}")
            print(f"    Memory Requirement: {task_memory:.1f} MB")
            print(f"    Buffer Contribution: {buffer_overhead:.1f} MB")
            if "examples" in info:
                print(f"    Examples: {'; '.join(info['examples'])}")
            print()
        
        # Overall system requirements
        print("💻 RECOMMENDED SYSTEM REQUIREMENTS")
        print("-" * 40)
        
        min_memory = max(task_types[t]["memory_multiplier"] for t in task_types) * base_memory + buffer_overhead
        recommended_memory = min_memory * 2  # 2x for safety margin
        
        print("  Minimum Requirements:")
        print(f"    RAM: {min_memory * 2:.0f} MB ({min_memory * 2 / 1024:.1f} GB)")
        print(f"    CPU: 2+ cores, 2.0+ GHz")
        print(f"    Storage: 2 GB free space")
        print(f"    Python: 3.10-3.11")
        print()
        
        print("  Recommended Requirements:")
        print(f"    RAM: {recommended_memory:.0f} MB ({recommended_memory / 1024:.1f} GB)")
        print(f"    CPU: 4+ cores, 2.5+ GHz")
        print(f"    Storage: 5 GB free space")
        print(f"    Python: 3.10+")
        print()
        
        # Performance estimates
        print("⚡ PERFORMANCE ESTIMATES")
        print("-" * 30)
        
        print("  Task Processing (per episode):")
        print(f"    Deduction: ~2-5 seconds")
        print(f"    Abduction: ~5-10 seconds") 
        print(f"    Induction: ~8-15 seconds")
        print()
        
        print("  Training Duration (100 episodes):")
        print(f"    Demo config: ~30-60 minutes")
        print(f"    Full config: ~2-4 hours")
        print()
        
        print("  Memory Usage During Training:")
        peak_memory = base_memory + (demo_config.buffer_size * 3 * 0.001) + 200  # overhead
        print(f"    Peak Memory: ~{peak_memory:.0f} MB")
        print(f"    Baseline: {base_memory:.0f} MB (model)")
        print(f"    Buffers: {demo_config.buffer_size * 3 * 0.001:.0f} MB (3 task types)")
        print(f"    Overhead: ~200 MB (Python + libraries)")
        print()
        
        # Scaling recommendations
        print("📈 SCALING RECOMMENDATIONS")
        print("-" * 30)
        
        print("  For Better Performance:")
        print("    • Increase batch_size to 4-8 (requires more RAM)")
        print("    • Increase buffer_size to 1000-2000")
        print("    • Use tasks_per_episode = 10-20")
        print()
        
        print("  For Lower Resource Usage:")
        print("    • Use batch_size = 1")
        print("    • Reduce buffer_size to 200-300")
        print("    • Set max_memory_mb = 25-50")
        print("    • Use execution_timeout = 2.0")
        print()
        
        return {
            "model_size_mb": base_memory,
            "min_memory_mb": min_memory,
            "recommended_memory_mb": recommended_memory,
            "task_requirements": task_types,
            "cached_models": cached_models
        }
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        print("\n💡 TIP: The model will be automatically downloaded on first use.")
        print("   You can also pre-download models using:")
        print("   python -c \"from utils.model_manager import get_model_manager; get_model_manager().download_and_cache_model('microsoft/DialoGPT-small')\"")
        return None

if __name__ == "__main__":
    results = analyze_system_requirements()
    
    if results:
        print("✅ Analysis complete! System requirements calculated.")
        if results.get('cached_models'):
            print(f"📦 {len(results['cached_models'])} model(s) available in cache.")
    else:
        print("⚠️  Analysis incomplete due to missing model.")
