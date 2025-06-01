"""
Advanced Prompt Templates for Different Scenarios
Contains specialized prompt templates for various training scenarios.
"""

from typing import Dict, Any


class AdvancedPromptTemplates:
    """Advanced prompt templates for specialized scenarios."""
    
    @staticmethod
    def get_bootstrap_prompt(reasoning_type: str) -> str:
        """Get prompt for initial bootstrapping with minimal examples."""
        templates = {
            'deduction': """BOOTSTRAP TASK GENERATION - DEDUCTION

You are starting to learn task generation. Begin with the simplest possible tasks.

GOAL: Generate a very simple Python function and input pair.

REQUIREMENTS:
- Use only basic arithmetic operations (+, -, *, //)
- Single input parameter
- Immediate calculation (no complex logic)
- Clear, predictable output

TEMPLATE:
{
    "program": "lambda x: x [operation] [number]",
    "input": "[simple_number]",
    "expected_output": "[calculated_result]",
    "complexity": 1
}

EXAMPLE:
{
    "program": "lambda x: x + 2",
    "input": "3",
    "expected_output": "5",
    "complexity": 1
}

Generate a similar bootstrap task:""",

            'abduction': """BOOTSTRAP TASK GENERATION - ABDUCTION

You are starting to learn abduction tasks. Begin with the simplest transformations.

GOAL: Provide input and output where the transformation is obvious.

REQUIREMENTS:
- Simple numeric transformation
- Single operation
- Clear mathematical relationship
- Easily expressible as lambda function

TEMPLATE:
{
    "input": "[simple_number]",
    "expected_output": "[transformed_result]",
    "complexity": 1
}

EXAMPLE:
{
    "input": "4",
    "expected_output": "8",
    "complexity": 1
}
(Solution would be: lambda x: x * 2)

Generate a similar bootstrap task:""",

            'induction': """BOOTSTRAP TASK GENERATION - INDUCTION

You are starting to learn induction tasks. Begin with very clear patterns.

GOAL: Provide simple input-output examples with obvious patterns.

REQUIREMENTS:
- 2-3 examples maximum
- Arithmetic patterns only
- Consistent, simple transformations
- Easy to recognize pattern

TEMPLATE:
{
    "examples": [["input1", "output1"], ["input2", "output2"]],
    "pattern": "[simple_description]",
    "complexity": 1
}

EXAMPLE:
{
    "examples": [["1", "3"], ["2", "6"], ["3", "9"]],
    "pattern": "multiply by 3",
    "complexity": 1
}

Generate a similar bootstrap task:"""
        }
        
        return templates.get(reasoning_type, templates['deduction'])
    
    @staticmethod
    def get_curriculum_transition_prompt(from_level: int, to_level: int, 
                                       reasoning_type: str) -> str:
        """Get prompt for transitioning between curriculum levels."""
        return f"""CURRICULUM TRANSITION - {reasoning_type.upper()}

You are transitioning from complexity level {from_level} to level {to_level}.

PREVIOUS LEVEL SKILLS ({from_level}):
{AdvancedPromptTemplates._get_level_skills(from_level)}

NEW LEVEL REQUIREMENTS ({to_level}):
{AdvancedPromptTemplates._get_level_skills(to_level)}

TRANSITION STRATEGY:
1. Build on previously mastered concepts
2. Introduce ONE new element at a time
3. Maintain clarity while adding complexity
4. Ensure the task is still solvable

GENERATE: A task that bridges level {from_level} and {to_level} concepts.

Task:"""
    
    @staticmethod
    def get_error_recovery_prompt(error_type: str, context: Dict[str, Any]) -> str:
        """Get prompt for recovering from specific error types."""
        error_templates = {
            'syntax_error': f"""SYNTAX ERROR RECOVERY

Previous attempt had syntax errors: {context.get('error_message', '')}

COMMON SYNTAX ISSUES TO AVOID:
- Missing colons after lambda
- Incorrect indentation
- Mismatched parentheses
- Invalid operator usage

CORRECTIVE APPROACH:
1. Use simple, valid Python syntax
2. Test mentally before generating
3. Stick to basic operations
4. Ensure proper lambda format

Generate a corrected task:""",

            'logic_error': f"""LOGIC ERROR RECOVERY

Previous task had logical inconsistencies: {context.get('error_message', '')}

COMMON LOGIC ISSUES TO AVOID:
- Expected output doesn't match program execution
- Inconsistent mathematical relationships
- Impossible transformations

CORRECTIVE APPROACH:
1. Verify the calculation manually
2. Ensure input/output relationship is correct
3. Use simple, verifiable operations
4. Double-check the mathematics

Generate a corrected task:""",

            'complexity_mismatch': f"""COMPLEXITY MISMATCH RECOVERY

Task complexity was incorrectly assessed: {context.get('error_message', '')}

COMPLEXITY CALIBRATION:
- Level 1: Single operation (x + 1, x * 2)
- Level 2: Simple conditions (x if x > 0 else 0)
- Level 3: Basic functions (len(x), sum(x))

CORRECTIVE APPROACH:
1. Match actual complexity to stated level
2. Use appropriate operations for the level
3. Ensure difficulty is consistent

Generate a properly calibrated task:"""
        }
        
        return error_templates.get(error_type, error_templates['syntax_error'])
    
    @staticmethod
    def get_diversity_enhancement_prompt(existing_patterns: list, reasoning_type: str) -> str:
        """Get prompt for enhancing task diversity."""
        avoided_patterns = ', '.join(existing_patterns)
        
        return f"""DIVERSITY ENHANCEMENT - {reasoning_type.upper()}

RECENTLY USED PATTERNS TO AVOID:
{avoided_patterns}

DIVERSITY GOALS:
1. Use different operation types
2. Vary input/output formats
3. Explore different mathematical concepts
4. Avoid repetitive patterns

OPERATION CATEGORIES TO CONSIDER:
- Arithmetic: +, -, *, //, %, **
- Comparison: >, <, ==, !=
- String: .upper(), .lower(), len(), slicing
- List: sum(), len(), indexing, slicing
- Boolean: and, or, not
- Conditional: if-else expressions

Generate a task using an underexplored pattern:"""
    
    @staticmethod
    def get_quality_improvement_prompt(quality_metrics: Dict[str, float]) -> str:
        """Get prompt for improving task quality based on metrics."""
        return f"""QUALITY IMPROVEMENT FOCUS

CURRENT QUALITY METRICS:
- Syntax Validity: {quality_metrics.get('syntax_validity', 0):.2f}
- Logic Correctness: {quality_metrics.get('logic_correctness', 0):.2f}
- Clarity Score: {quality_metrics.get('clarity', 0):.2f}
- Complexity Accuracy: {quality_metrics.get('complexity_accuracy', 0):.2f}

IMPROVEMENT PRIORITIES:
{AdvancedPromptTemplates._get_improvement_priorities(quality_metrics)}

QUALITY CHECKLIST:
□ Program is syntactically valid Python
□ Input/output relationship is mathematically sound
□ Task is clear and unambiguous
□ Complexity rating is accurate
□ Task is educational and meaningful

Generate a high-quality task addressing the priority areas:"""
    
    @staticmethod
    def _get_level_skills(level: int) -> str:
        """Get description of skills expected at each complexity level."""
        skills = {
            1: "Basic arithmetic operations, single parameter functions",
            2: "Simple conditionals, basic function calls",
            3: "List/string operations, multiple parameters",
            4: "Complex logic, nested operations, advanced functions",
            5: "Recursive patterns, sophisticated algorithms"
        }
        return skills.get(level, "Unknown level")
    
    @staticmethod
    def _get_improvement_priorities(metrics: Dict[str, float]) -> str:
        """Get improvement priorities based on quality metrics."""
        priorities = []
        
        if metrics.get('syntax_validity', 1.0) < 0.8:
            priorities.append("- CRITICAL: Fix syntax errors in generated code")
        
        if metrics.get('logic_correctness', 1.0) < 0.7:
            priorities.append("- HIGH: Ensure mathematical correctness")
        
        if metrics.get('clarity', 1.0) < 0.6:
            priorities.append("- MEDIUM: Improve task clarity and definitions")
        
        if metrics.get('complexity_accuracy', 1.0) < 0.7:
            priorities.append("- LOW: Calibrate complexity ratings better")
        
        return '\n'.join(priorities) if priorities else "- Continue maintaining current quality standards"


class ContextualPrompts:
    """Prompts that adapt based on training context."""
    
    @staticmethod
    def get_episode_context_prompt(episode: int, total_episodes: int, 
                                 recent_performance: Dict[str, float]) -> str:
        """Get prompt with episode context for adaptive training."""
        progress_percentage = (episode / total_episodes) * 100
        
        context = f"""TRAINING CONTEXT - Episode {episode}/{total_episodes} ({progress_percentage:.1f}% complete)

RECENT PERFORMANCE:
- Success Rate: {recent_performance.get('success_rate', 0):.2f}
- Task Quality: {recent_performance.get('task_quality', 0):.2f}
- Learning Progress: {recent_performance.get('learning_progress', 0):.2f}

TRAINING PHASE: {ContextualPrompts._get_training_phase(progress_percentage)}

FOCUS FOR THIS EPISODE:
{ContextualPrompts._get_episode_focus(progress_percentage, recent_performance)}

"""
        return context
    
    @staticmethod
    def _get_training_phase(progress: float) -> str:
        """Determine training phase based on progress."""
        if progress < 20:
            return "BOOTSTRAP - Learning basic task generation"
        elif progress < 50:
            return "DEVELOPMENT - Building reasoning capabilities"
        elif progress < 80:
            return "REFINEMENT - Improving quality and complexity"
        else:
            return "MASTERY - Achieving consistent high performance"
    
    @staticmethod
    def _get_episode_focus(progress: float, performance: Dict[str, float]) -> str:
        """Get focus areas for the current episode."""
        if progress < 20:
            return "Generate simple, valid tasks. Focus on syntax correctness."
        elif progress < 50:
            return "Increase task diversity while maintaining quality."
        elif progress < 80:
            return "Introduce more complex reasoning patterns."
        else:
            return "Optimize for consistent high-quality task generation."
