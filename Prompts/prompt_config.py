"""
Prompt Configuration for AZR System
Contains configuration and templates for different prompt types.
"""

# Prompt template configurations
PROMPT_CONFIGS = {
    "task_generation": {
        "use_few_shot": True,
        "num_examples": 3,
        "include_validation": True,
        "curriculum_adjustment": True
    },
    "solution": {
        "step_by_step": True,
        "include_verification": True,
        "complexity_hints": True,
        "error_recovery": True
    },
    "validation": {
        "comprehensive_check": True,
        "scoring_system": True,
        "improvement_suggestions": True
    }
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "minimum_task_validity": 0.7,
    "minimum_solution_confidence": 0.6,
    "complexity_tolerance": 1,  # Allow complexity rating to differ by 1 level
    "batch_quality_threshold": 0.8
}

# Curriculum learning parameters
CURRICULUM_PARAMS = {
    "initial_complexity": 1,
    "max_complexity": 5,
    "success_rate_threshold_increase": 0.7,
    "success_rate_threshold_decrease": 0.3,
    "complexity_adjustment_episodes": 10
}

# Model-specific prompt adjustments
MODEL_ADJUSTMENTS = {
    "codet5": {
        "max_prompt_length": 400,
        "prefer_structured_format": True,
        "use_code_specific_examples": True
    },
    "codellama": {
        "max_prompt_length": 800,
        "prefer_detailed_explanations": True,
        "use_multi_step_reasoning": True
    },
    "default": {
        "max_prompt_length": 300,
        "prefer_simple_format": True,
        "use_basic_examples": True
    }
}
