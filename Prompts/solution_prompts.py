"""
Solution Generation Prompts for AZR System
Contains structured prompts for generating solutions to reasoning tasks.
"""

from typing import Dict, Any, Optional, List


class SolutionPrompts:
    """
    Contains prompts specifically designed for solution generation
    with step-by-step reasoning guidance.
    """
    
    @staticmethod
    def get_deduction_solution_prompt(task_data: Dict[str, Any]) -> str:
        """Get structured prompt for solving deduction tasks."""
        return f"""Solve this deduction problem step by step.

PROBLEM TYPE: Deduction
TASK: Given a program and input, determine the output.

GIVEN:
Program: {task_data.get('program', '')}
Input: {task_data.get('input', '')}

APPROACH:
1. Parse the lambda function carefully
2. Substitute the input value(s) into the function
3. Execute the operations step by step
4. Calculate the final result

REASONING STEPS:
Step 1: Identify the function structure
Step 2: Apply input to the function
Step 3: Perform calculations
Step 4: State the final output

ANSWER FORMAT: Provide only the final result value.

Solution:"""

    @staticmethod
    def get_abduction_solution_prompt(task_data: Dict[str, Any]) -> str:
        """Get structured prompt for solving abduction tasks."""
        return f"""Solve this abduction problem step by step.

PROBLEM TYPE: Abduction  
TASK: Given input and expected output, find the program that produces this transformation.

GIVEN:
Input: {task_data.get('input', '')}
Expected Output: {task_data.get('expected_output', '')}

APPROACH:
1. Analyze the transformation from input to output
2. Identify the mathematical or logical relationship
3. Consider common operations (arithmetic, string methods, list operations)
4. Express the transformation as a lambda function
5. Verify the function produces the correct output

REASONING STEPS:
Step 1: What type of transformation is this? (numeric, string, boolean, etc.)
Step 2: What operation could produce this change?
Step 3: How can this be expressed as a Python function?
Step 4: Verify the function works

ANSWER FORMAT: lambda x: <expression>

Solution:"""

    @staticmethod
    def get_induction_solution_prompt(task_data: Dict[str, Any]) -> str:
        """Get structured prompt for solving induction tasks."""
        examples = task_data.get('examples', [])
        pattern = task_data.get('pattern', '')
        
        formatted_examples = '\n'.join([f"  {inp} → {out}" for inp, out in examples])
        
        return f"""Solve this induction problem step by step.

PROBLEM TYPE: Induction
TASK: Given input-output examples, find the general pattern and express it as a function.

GIVEN EXAMPLES:
{formatted_examples}

PATTERN HINT: {pattern}

APPROACH:
1. Examine each input-output pair carefully
2. Look for mathematical or logical relationships
3. Test your hypothesis against all examples
4. Generalize the pattern into a function
5. Verify the function works for all given examples

REASONING STEPS:
Step 1: What is the relationship between input and output in each example?
Step 2: Is there a consistent mathematical operation or transformation?
Step 3: How can this pattern be expressed as a Python function?
Step 4: Does this function work for all examples?

ANSWER FORMAT: lambda x: <expression>

Solution:"""

    @staticmethod
    def get_multi_step_solution_prompt(reasoning_type: str, task_data: Dict[str, Any]) -> str:
        """Get enhanced solution prompt with detailed step-by-step reasoning."""
        base_prompts = {
            'deduction': SolutionPrompts.get_deduction_solution_prompt(task_data),
            'abduction': SolutionPrompts.get_abduction_solution_prompt(task_data), 
            'induction': SolutionPrompts.get_induction_solution_prompt(task_data)
        }
        
        base_prompt = base_prompts.get(reasoning_type, base_prompts['deduction'])
        
        # Add reasoning verification section
        verification_section = """
VERIFICATION:
Before providing your final answer, verify:
□ Your solution addresses the specific problem type
□ The logic is mathematically sound
□ The syntax is valid Python
□ The solution works for the given examples
□ The answer is in the correct format

CONFIDENCE CHECK:
Rate your confidence in this solution (1-10): ___
If confidence < 8, reconsider your approach.
"""
        return base_prompt + verification_section

    @staticmethod
    def get_hint_enhanced_prompt(reasoning_type: str, task_data: Dict[str, Any], 
                               hint: Optional[str] = None) -> str:
        """Get solution prompt enhanced with contextual hints."""
        base_prompt = SolutionPrompts.get_multi_step_solution_prompt(reasoning_type, task_data)
        
        if hint:
            hint_section = f"""
HELPFUL HINT: {hint}

Use this hint to guide your reasoning, but still work through the problem step by step.
"""
            # Insert hint after the problem statement but before the approach
            parts = base_prompt.split('APPROACH:')
            if len(parts) == 2:
                return parts[0] + hint_section + 'APPROACH:' + parts[1]
        
        return base_prompt

    @staticmethod
    def get_complexity_adjusted_prompt(reasoning_type: str, task_data: Dict[str, Any], 
                                     complexity: int) -> str:
        """Get solution prompt adjusted for task complexity."""
        base_prompt = SolutionPrompts.get_multi_step_solution_prompt(reasoning_type, task_data)
        
        complexity_guidance = {
            1: "This is a simple task. Focus on basic operations and straightforward logic.",
            2: "This is a moderate task. Consider simple conditionals or basic function calls.",
            3: "This is an intermediate task. May involve list/string operations or multiple steps.",
            4: "This is a complex task. Consider advanced operations, nested logic, or multiple transformations.",
            5: "This is a very complex task. May require recursive thinking, advanced algorithms, or sophisticated logic."
        }
        
        guidance = complexity_guidance.get(complexity, complexity_guidance[1])
        
        complexity_section = f"""
COMPLEXITY GUIDANCE (Level {complexity}):
{guidance}

"""
        
        # Insert complexity guidance after the problem type
        lines = base_prompt.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('TASK:'):
                lines.insert(i + 2, complexity_section)
                break
        
        return '\n'.join(lines)

    @staticmethod
    def get_error_recovery_prompt(reasoning_type: str, task_data: Dict[str, Any], 
                                previous_attempt: str, error_message: str) -> str:
        """Get solution prompt for error recovery scenarios."""
        base_prompt = SolutionPrompts.get_multi_step_solution_prompt(reasoning_type, task_data)
        
        error_section = f"""
PREVIOUS ATTEMPT ANALYSIS:
Previous attempt: {previous_attempt}
Error encountered: {error_message}

WHAT WENT WRONG:
Analyze why the previous attempt failed:
- Was it a syntax error?
- Was the logic incorrect?
- Did it not handle the input type correctly?
- Was the approach fundamentally wrong?

CORRECTIVE APPROACH:
Based on the error, adjust your strategy:
1. Fix any syntax issues
2. Reconsider the logical approach
3. Ensure proper input/output types
4. Verify against the problem requirements

"""
        
        return error_section + base_prompt

    @staticmethod
    def get_collaborative_prompt(reasoning_type: str, task_data: Dict[str, Any], 
                               alternative_solutions: Optional[List[str]] = None) -> str:
        """Get solution prompt that considers alternative approaches."""
        base_prompt = SolutionPrompts.get_multi_step_solution_prompt(reasoning_type, task_data)
        
        if alternative_solutions:
            alternatives_section = f"""
ALTERNATIVE APPROACHES TO CONSIDER:
{chr(10).join([f"- {alt}" for alt in alternative_solutions])}

Choose the most appropriate approach for this specific problem.

"""
            
            # Insert after the approach section
            parts = base_prompt.split('REASONING STEPS:')
            if len(parts) == 2:
                return parts[0] + alternatives_section + 'REASONING STEPS:' + parts[1]
        
        return base_prompt
