"""
Validation Prompts for AZR System
Contains prompts for validating task quality and solution correctness.
"""

from typing import Dict, Any, List


class ValidationPrompts:
    """
    Contains prompts for validating tasks and solutions to ensure quality.
    """
    
    @staticmethod
    def get_task_quality_validation_prompt(task_data: Dict[str, Any]) -> str:
        """Get prompt for validating task quality and correctness."""
        return f"""Validate this programming task for quality and correctness.

TASK TO VALIDATE:
Type: {task_data.get('type', 'unknown')}
Program: {task_data.get('program', 'N/A')}
Input: {task_data.get('input', 'N/A')}
Expected Output: {task_data.get('expected_output', 'N/A')}
Complexity: {task_data.get('complexity', 'N/A')}

VALIDATION CRITERIA:
1. SYNTAX VALIDITY
   - Is the program syntactically correct Python?
   - Are all functions and operations valid?
   - Check for common syntax errors

2. LOGICAL CORRECTNESS
   - Does the program produce the expected output for the given input?
   - Is the mathematical/logical relationship sound?
   - Are there any edge cases that could cause issues?

3. TASK CLARITY
   - Is the task well-defined and unambiguous?
   - Are the input and output formats appropriate?
   - Is the task solvable with the given information?

4. COMPLEXITY ASSESSMENT
   - Does the complexity rating match the actual difficulty?
   - Consider syntax complexity, conceptual difficulty, reasoning steps
   - Rate from 1 (trivial) to 5 (very complex)

5. EDUCATIONAL VALUE
   - Is this a meaningful programming/reasoning task?
   - Does it test relevant skills?
   - Is it appropriate for learning?

ANSWER FORMAT (JSON):
{{
    "is_valid": true/false,
    "syntax_valid": true/false,
    "logic_correct": true/false,
    "clarity_score": <1-5>,
    "suggested_complexity": <1-5>,
    "issues": ["list of specific issues found"],
    "suggestions": ["list of improvement suggestions"],
    "confidence": <0.0-1.0>
}}

Validation Result:"""

    @staticmethod
    def get_solution_correctness_prompt(task_data: Dict[str, Any], solution: str) -> str:
        """Get prompt for validating solution correctness."""
        return f"""Validate this solution to a programming task.

ORIGINAL TASK:
Type: {task_data.get('type', 'unknown')}
Program: {task_data.get('program', 'N/A')}
Input: {task_data.get('input', 'N/A')}
Expected Output: {task_data.get('expected_output', 'N/A')}

PROPOSED SOLUTION:
{solution}

VALIDATION CRITERIA:
1. CORRECTNESS
   - Does the solution correctly address the task?
   - For deduction: Does it produce the right output?
   - For abduction: Does the function transform input to expected output?
   - For induction: Does the function capture the pattern correctly?

2. SYNTAX VALIDATION
   - Is the solution syntactically valid Python?
   - Are all operations and functions used correctly?
   - Check for common syntax errors

3. COMPLETENESS
   - Does the solution handle the given input appropriately?
   - Are there missing edge cases?
   - Is the solution complete?

4. EFFICIENCY
   - Is the solution reasonably efficient?
   - Are there obvious performance issues?
   - Is the approach optimal?

5. CODE QUALITY
   - Is the code clean and readable?
   - Are variable names appropriate?
   - Is the logic clear?

ANSWER FORMAT (JSON):
{{
    "is_correct": true/false,
    "syntax_valid": true/false,
    "handles_input": true/false,
    "efficiency_score": <1-5>,
    "issues": ["list of specific issues"],
    "corrections": ["suggested fixes"],
    "confidence": <0.0-1.0>
}}

Validation Result:"""

    @staticmethod
    def get_complexity_validation_prompt(task_data: Dict[str, Any]) -> str:
        """Get prompt for validating task complexity rating."""
        return f"""Assess the complexity of this programming task.

TASK:
Type: {task_data.get('type', 'unknown')}
Program: {task_data.get('program', 'N/A')}
Input: {task_data.get('input', 'N/A')}
Expected Output: {task_data.get('expected_output', 'N/A')}
Current Complexity Rating: {task_data.get('complexity', 'N/A')}

COMPLEXITY ASSESSMENT FACTORS:
1. SYNTAX COMPLEXITY
   - Simple expressions vs complex nested operations
   - Number of operations involved
   - Use of advanced Python features

2. CONCEPTUAL DIFFICULTY
   - Basic arithmetic vs advanced mathematical concepts
   - Domain knowledge required
   - Abstract thinking needed

3. REASONING STEPS
   - Number of logical steps required
   - Intermediate calculations needed
   - Multi-step problem solving

4. INPUT/OUTPUT COMPLEXITY
   - Simple values vs complex data structures
   - Multiple inputs or complex transformations
   - Edge cases and special handling

COMPLEXITY SCALE:
- Level 1: Basic operations (add, multiply, simple conditionals)
- Level 2: Moderate operations (functions with conditions, basic loops)
- Level 3: Intermediate (list/string manipulation, multiple operations)
- Level 4: Advanced (complex logic, nested structures, algorithms)
- Level 5: Expert (recursive algorithms, advanced patterns, multiple concepts)

ANSWER FORMAT (JSON):
{{
    "assessed_complexity": <1-5>,
    "current_rating_accurate": true/false,
    "complexity_factors": ["list of contributing factors"],
    "reasoning": "detailed explanation of complexity assessment",
    "comparison_to_level": "how this compares to other tasks at this level"
}}

Assessment Result:"""

    @staticmethod
    def get_difficulty_progression_prompt(tasks: List[Dict[str, Any]]) -> str:
        """Get prompt for validating difficulty progression in curriculum learning."""
        task_list = "\n".join([
            f"Task {i+1}: {task.get('program', 'N/A')} (Complexity: {task.get('complexity', 'N/A')})"
            for i, task in enumerate(tasks)
        ])
        
        return f"""Validate the difficulty progression of these tasks for curriculum learning.

TASK SEQUENCE:
{task_list}

PROGRESSION CRITERIA:
1. GRADUAL INCREASE
   - Is there a logical progression from simple to complex?
   - Are the jumps in difficulty appropriate?
   - No sudden spikes in complexity?

2. CONCEPT BUILDING
   - Do earlier tasks prepare for later concepts?
   - Are foundational skills built before advanced ones?
   - Is there conceptual coherence?

3. LEARNING CURVE
   - Is the progression suitable for learning?
   - Are there enough practice opportunities at each level?
   - Is the pace appropriate?

4. COVERAGE
   - Do the tasks cover important concepts?
   - Is there good variety within complexity levels?
   - Are gaps in knowledge addressed?

ANSWER FORMAT (JSON):
{{
    "progression_quality": <1-5>,
    "appropriate_sequence": true/false,
    "difficulty_gaps": ["list of gaps or jumps identified"],
    "missing_concepts": ["concepts that should be included"],
    "recommendations": ["suggestions for improvement"],
    "overall_assessment": "summary of progression quality"
}}

Progression Analysis:"""

    @staticmethod
    def get_batch_quality_prompt(tasks: List[Dict[str, Any]]) -> str:
        """Get prompt for validating a batch of generated tasks."""
        task_summaries = []
        for i, task in enumerate(tasks):
            summary = f"Task {i+1}: Type={task.get('type', 'N/A')}, Complexity={task.get('complexity', 'N/A')}"
            task_summaries.append(summary)
        
        return f"""Validate this batch of generated tasks for overall quality and diversity.

BATCH SUMMARY:
Total Tasks: {len(tasks)}
{chr(10).join(task_summaries)}

BATCH VALIDATION CRITERIA:
1. DIVERSITY
   - Are different reasoning types represented?
   - Is there variety in complexity levels?
   - Are different types of operations covered?

2. CONSISTENCY
   - Are complexity ratings consistent across tasks?
   - Is the quality level similar across tasks?
   - Are formatting and structure consistent?

3. BALANCE
   - Is there appropriate distribution across types?
   - Are complexity levels well-distributed?
   - No over-emphasis on particular patterns?

4. OVERALL QUALITY
   - What percentage of tasks are high quality?
   - Are there systematic quality issues?
   - How does this batch compare to ideal standards?

ANSWER FORMAT (JSON):
{{
    "batch_quality_score": <1-5>,
    "diversity_score": <1-5>,
    "consistency_score": <1-5>,
    "valid_task_percentage": <0-100>,
    "type_distribution": {{"deduction": X, "abduction": Y, "induction": Z}},
    "complexity_distribution": {{"1": A, "2": B, "3": C, "4": D, "5": E}},
    "quality_issues": ["list of systematic issues"],
    "recommendations": ["suggestions for batch improvement"]
}}

Batch Analysis:"""

    @staticmethod
    def get_learning_progress_validation_prompt(historical_data: Dict[str, Any]) -> str:
        """Get prompt for validating learning progress and task quality trends."""
        return f"""Validate learning progress and task generation quality over time.

HISTORICAL DATA:
Episodes Analyzed: {historical_data.get('episodes', 'N/A')}
Total Tasks Generated: {historical_data.get('total_tasks', 'N/A')}
Average Success Rate: {historical_data.get('avg_success_rate', 'N/A')}
Complexity Trend: {historical_data.get('complexity_trend', 'N/A')}

PROGRESS VALIDATION CRITERIA:
1. LEARNING TRAJECTORY
   - Is task quality improving over time?
   - Are success rates trending upward?
   - Is the model generating more complex tasks appropriately?

2. SKILL DEVELOPMENT
   - Are there signs of improved reasoning capability?
   - Is the model learning from successful examples?
   - Are failure patterns being addressed?

3. CURRICULUM EFFECTIVENESS
   - Is the difficulty progression working?
   - Are complexity increases well-timed?
   - Is the model ready for next level challenges?

4. QUALITY TRENDS
   - Are tasks becoming more sophisticated?
   - Is there improvement in task diversity?
   - Are systematic errors being reduced?

ANSWER FORMAT (JSON):
{{
    "learning_progress_score": <1-5>,
    "quality_improvement": true/false,
    "readiness_for_complexity_increase": true/false,
    "strengths": ["observed improvements"],
    "weaknesses": ["areas needing improvement"],
    "recommendations": ["specific suggestions for continued learning"],
    "next_steps": ["recommended actions for next phase"]
}}

Progress Assessment:"""

    @staticmethod
    def get_task_validation_prompt(task_type: str, program: str, input_data: str) -> str:
        """Get prompt for validating a task execution."""
        return f"""Validate the execution of this task.

TASK TYPE: {task_type}
PROGRAM: {program}
INPUT: {input_data}

VALIDATION STEPS:
1. Parse the program syntax
2. Verify the program is safe to execute
3. Execute the program with the given input
4. Record the output

VALIDATION CRITERIA:
- Syntax correctness
- Execution safety
- Deterministic behavior
- Resource usage within limits

RESULT FORMAT:
{{
    "valid": true/false,
    "output": <execution_result>,
    "error": <error_message_if_any>
}}

Validation Result:"""
