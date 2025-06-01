"""
Centralized Prompt Manager for AZR System
Manages all prompt templates and provides methods for dynamic prompt generation.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import random


class PromptManager:
    """
    Manages all prompts for the AZR system, including task generation,
    solution generation, and validation prompts.
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize prompt manager with prompt templates."""
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        
        self.prompts_dir = prompts_dir
        self.prompts = self._load_all_prompts()
        
    def _load_all_prompts(self) -> Dict[str, Any]:
        """Load all prompt templates from configuration files."""
        prompts = {
            'task_generation': self._load_task_generation_prompts(),
            'solution': self._load_solution_prompts(),
            'validation': self._load_validation_prompts(),
            'few_shot_examples': self._load_few_shot_examples()
        }
        return prompts
    
    def _load_task_generation_prompts(self) -> Dict[str, str]:
        """Load task generation prompts with few-shot examples."""
        return {
            'deduction': self._get_deduction_task_prompt(),
            'abduction': self._get_abduction_task_prompt(),
            'induction': self._get_induction_task_prompt()
        }
    
    def _load_solution_prompts(self) -> Dict[str, str]:
        """Load solution generation prompts."""
        return {
            'deduction': self._get_deduction_solution_prompt(),
            'abduction': self._get_abduction_solution_prompt(),
            'induction': self._get_induction_solution_prompt()
        }
    
    def _load_validation_prompts(self) -> Dict[str, str]:
        """Load validation prompts for task quality assessment."""
        return {
            'task_validation': self._get_task_validation_prompt(),
            'solution_validation': self._get_solution_validation_prompt(),
            'complexity_assessment': self._get_complexity_assessment_prompt()
        }
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Load few-shot examples for each reasoning type."""
        return {
            'deduction': self._get_deduction_examples(),
            'abduction': self._get_abduction_examples(),
            'induction': self._get_induction_examples()
        }
    
    def get_task_generation_prompt(self, reasoning_type: str, complexity: int = 1, 
                                 include_examples: bool = True) -> str:
        """
        Get a task generation prompt for the specified reasoning type.
        
        Args:
            reasoning_type: Type of reasoning (deduction, abduction, induction)
            complexity: Desired complexity level (1-5)
            include_examples: Whether to include few-shot examples
            
        Returns:
            Formatted prompt string
        """
        base_prompt = self.prompts['task_generation'].get(reasoning_type, '')
        
        if include_examples:
            examples = self._format_examples(reasoning_type, complexity)
            return base_prompt.format(examples=examples, complexity=complexity)
        else:
            return base_prompt.format(examples='', complexity=complexity)
    def get_solution_prompt(self, reasoning_type: str, task_data: Dict[str, Any]) -> str:
        """
        Get a solution generation prompt for the specified task.
        
        Args:
            reasoning_type: Type of reasoning task
            task_data: Task information including program, input, expected output
            
        Returns:
            Formatted prompt string
        """
        base_prompt = self.prompts['solution'].get(reasoning_type, '')
        
        # Provide default values for missing fields to prevent KeyError
        safe_task_data = {
            'program': '',
            'input': '',
            'expected_output': '',
            'examples': '',
            'pattern': '',
            'type': reasoning_type,
            'complexity': 1,
            **task_data  # Override with actual task_data if present
        }
        
        try:
            return base_prompt.format(**safe_task_data)
        except KeyError as e:
            # Fallback to basic prompt if formatting fails
            if reasoning_type == 'deduction':
                return f"Given program: {safe_task_data.get('program', '')} and input: {safe_task_data.get('input', '')}, what is the output?"
            elif reasoning_type == 'abduction':
                return f"What program with input {safe_task_data.get('input', '')} produces output {safe_task_data.get('expected_output', '')}?"
            elif reasoning_type == 'induction':
                return f"Find the pattern from these examples: {safe_task_data.get('examples', '')} and express it as a function."
            else:
                return f"Solve this {reasoning_type} problem: {str(safe_task_data)}"
    
    def get_validation_prompt(self, validation_type: str, **kwargs) -> str:
        """
        Get a validation prompt for task or solution validation.
        
        Args:
            validation_type: Type of validation needed
            **kwargs: Additional parameters for prompt formatting
            
        Returns:
            Formatted prompt string
        """
        base_prompt = self.prompts['validation'].get(validation_type, '')
        return base_prompt.format(**kwargs)
    
    def _format_examples(self, reasoning_type: str, complexity: int) -> str:
        """Format few-shot examples for the given reasoning type and complexity."""
        examples = self.prompts['few_shot_examples'].get(reasoning_type, [])
        
        # Filter examples by complexity if needed
        suitable_examples = [ex for ex in examples if ex.get('complexity', 1) <= complexity + 1]
        
        # Select a few random examples
        selected = random.sample(suitable_examples, min(3, len(suitable_examples)))
        
        formatted_examples = []
        for example in selected:
            if reasoning_type == 'deduction':
                formatted_examples.append(
                    f"Program: {example['program']}\n"
                    f"Input: {example['input']}\n"
                    f"Expected Output: {example['expected_output']}\n"
                    f"Complexity: {example['complexity']}"
                )
            elif reasoning_type == 'abduction':
                formatted_examples.append(
                    f"Input: {example['input']}\n"
                    f"Expected Output: {example['expected_output']}\n"
                    f"Solution: {example['program']}\n"
                    f"Complexity: {example['complexity']}"
                )
            elif reasoning_type == 'induction':
                formatted_examples.append(
                    f"Examples: {example['examples']}\n"
                    f"Pattern: {example['pattern']}\n"
                    f"Solution: {example['program']}\n"
                    f"Complexity: {example['complexity']}"
                )
        
        return "\n\n".join(formatted_examples)
    
    # Prompt template definitions follow...
    def _get_deduction_task_prompt(self) -> str:
        """Get the deduction task generation prompt with structure and examples."""
        return """You are generating Python programming tasks for deduction reasoning.

TASK: Generate a valid Python function and input that demonstrates logical deduction.

FORMAT (JSON):
{{
    "program": "lambda x: <expression>",
    "input": "<input_value>",
    "expected_output": "<expected_result>",
    "complexity": {complexity}
}}

REQUIREMENTS:
1. Program must be a valid Python lambda function
2. Input must be compatible with the function
3. Expected output must be correct when function is applied to input
4. Complexity should match the requested level (1=simple, 5=complex)
5. Use appropriate Python operations and syntax

EXAMPLES:
{examples}

Generate a new deduction task following the above format:"""

    def _get_abduction_task_prompt(self) -> str:
        """Get the abduction task generation prompt."""
        return """You are generating Python programming tasks for abduction reasoning.

TASK: Generate an input and expected output where the solver must find the program.

FORMAT (JSON):
{{
    "input": "<input_value>",
    "expected_output": "<expected_result>",
    "complexity": {complexity}
}}

REQUIREMENTS:
1. Input should be a simple value or data structure
2. Expected output should have a clear mathematical/logical relationship to input
3. The relationship should be expressible as a Python function
4. Complexity should match the requested level
5. Avoid ambiguous transformations

EXAMPLES:
{examples}

Generate a new abduction task following the above format:"""

    def _get_induction_task_prompt(self) -> str:
        """Get the induction task generation prompt."""
        return """You are generating Python programming tasks for induction reasoning.

TASK: Generate input-output examples that demonstrate a pattern for the solver to learn.

FORMAT (JSON):
{{
    "examples": [["input1", "output1"], ["input2", "output2"], ["input3", "output3"]],
    "pattern": "<description_of_pattern>",
    "complexity": {complexity}
}}

REQUIREMENTS:
1. Provide 3-5 input-output examples
2. Examples should clearly demonstrate a consistent pattern
3. Pattern should be learnable and expressible as a Python function
4. Complexity should match the requested level
5. Include diverse inputs to make pattern clear

EXAMPLES:
{examples}

Generate a new induction task following the above format:"""

    def _get_deduction_solution_prompt(self) -> str:
        """Get the deduction solution generation prompt."""
        return """Solve this deduction problem step by step.

PROBLEM:
Program: {program}
Input: {input}

TASK: Execute the program with the given input and provide the output.

APPROACH:
1. Parse the lambda function
2. Apply the input to the function
3. Calculate the result step by step
4. Provide the final output

ANSWER FORMAT: Just provide the result value.

Solution:"""

    def _get_abduction_solution_prompt(self) -> str:
        """Get the abduction solution generation prompt."""
        return """Solve this abduction problem step by step.

PROBLEM:
Input: {input}
Expected Output: {expected_output}

TASK: Find a Python lambda function that transforms the input to the expected output.

APPROACH:
1. Analyze the relationship between input and output
2. Identify the mathematical or logical transformation
3. Express the transformation as a lambda function
4. Verify the function works correctly

ANSWER FORMAT: lambda x: <expression>

Solution:"""

    def _get_induction_solution_prompt(self) -> str:
        """Get the induction solution generation prompt."""
        return """Solve this induction problem step by step.

PROBLEM:
Examples: {examples}
Pattern Description: {pattern}

TASK: Find a Python lambda function that captures the pattern shown in the examples.

APPROACH:
1. Analyze each input-output pair
2. Identify the common transformation pattern
3. Express the pattern as a general lambda function
4. Verify the function works for all examples

ANSWER FORMAT: lambda x: <expression>

Solution:"""

    def _get_task_validation_prompt(self) -> str:
        """Get the task validation prompt."""
        return """Validate this programming task for quality and correctness.

TASK:
{task_data}

VALIDATION CRITERIA:
1. Syntax: Is the program syntactically valid?
2. Logic: Is the expected output correct for the given input and program?
3. Clarity: Is the task well-defined and unambiguous?
4. Difficulty: Does the complexity match the claimed level?
5. Solvability: Is this a reasonable programming task?

ANSWER FORMAT:
{{
    "is_valid": true/false,
    "issues": ["list of any issues found"],
    "suggested_complexity": <1-5>,
    "confidence": <0.0-1.0>
}}

Validation Result:"""

    def _get_solution_validation_prompt(self) -> str:
        """Get the solution validation prompt."""
        return """Validate this solution to a programming task.

TASK: {task_description}
SOLUTION: {solution}

VALIDATION CRITERIA:
1. Correctness: Does the solution correctly solve the task?
2. Syntax: Is the solution syntactically valid Python?
3. Efficiency: Is the solution reasonably efficient?
4. Completeness: Does the solution handle the required inputs?

ANSWER FORMAT:
{{
    "is_correct": true/false,
    "syntax_valid": true/false,
    "issues": ["list of any issues"],
    "confidence": <0.0-1.0>
}}

Validation Result:"""

    def _get_complexity_assessment_prompt(self) -> str:
        """Get the complexity assessment prompt."""
        return """Assess the complexity of this programming task.

TASK: {task_data}

COMPLEXITY FACTORS:
1. Syntax complexity (simple expressions vs complex logic)
2. Conceptual difficulty (basic operations vs advanced concepts)
3. Required reasoning steps
4. Domain knowledge required

COMPLEXITY LEVELS:
1: Basic arithmetic/string operations
2: Simple conditionals and basic functions
3: Multiple operations, list/string manipulation
4: Complex logic, nested structures
5: Advanced algorithms, multiple complex concepts

ANSWER FORMAT:
{{
    "complexity": <1-5>,
    "reasoning": "<explanation of complexity assessment>",
    "key_factors": ["list of complexity contributing factors"]
}}

Assessment:"""

    def _get_deduction_examples(self) -> List[Dict]:
        """Get few-shot examples for deduction tasks."""
        return [
            {
                "program": "lambda x: x * 2",
                "input": "5",
                "expected_output": "10",
                "complexity": 1
            },
            {
                "program": "lambda x, y: x + y",
                "input": "(3, 4)",
                "expected_output": "7", 
                "complexity": 1
            },
            {
                "program": "lambda x: x ** 2 + 1",
                "input": "3",
                "expected_output": "10",
                "complexity": 2
            },
            {
                "program": "lambda x: x if x > 0 else -x",
                "input": "-5",
                "expected_output": "5",
                "complexity": 2
            },
            {
                "program": "lambda lst: sum(lst) / len(lst)",
                "input": "[1, 2, 3, 4, 5]",
                "expected_output": "3.0",
                "complexity": 3
            },
            {
                "program": "lambda s: ''.join(reversed(s.upper()))",
                "input": "'hello'",
                "expected_output": "'OLLEH'",
                "complexity": 3
            }
        ]

    def _get_abduction_examples(self) -> List[Dict]:
        """Get few-shot examples for abduction tasks."""
        return [
            {
                "input": "5",
                "expected_output": "25",
                "program": "lambda x: x ** 2",
                "complexity": 1
            },
            {
                "input": "[1, 2, 3]",
                "expected_output": "[2, 4, 6]",
                "program": "lambda lst: [x * 2 for x in lst]",
                "complexity": 2
            },
            {
                "input": "'hello'",
                "expected_output": "'HELLO'",
                "program": "lambda s: s.upper()",
                "complexity": 1
            },
            {
                "input": "10",
                "expected_output": "True",
                "program": "lambda x: x % 2 == 0",
                "complexity": 2
            },
            {
                "input": "[1, 2, 3, 4, 5]",
                "expected_output": "15",
                "program": "lambda lst: sum(lst)",
                "complexity": 2
            }
        ]

    def _get_induction_examples(self) -> List[Dict]:
        """Get few-shot examples for induction tasks."""
        return [
            {
                "examples": [["1", "2"], ["2", "4"], ["3", "6"], ["4", "8"]],
                "pattern": "multiply by 2",
                "program": "lambda x: x * 2",
                "complexity": 1
            },
            {
                "examples": [["0", "1"], ["1", "4"], ["2", "9"], ["3", "16"]],
                "pattern": "square then add 1", 
                "program": "lambda x: x ** 2 + 1",
                "complexity": 2
            },
            {
                "examples": [["5", "True"], ["3", "False"], ["8", "True"], ["7", "False"]],
                "pattern": "check if even",
                "program": "lambda x: x % 2 == 0",
                "complexity": 2
            },
            {
                "examples": [["'abc'", "3"], ["'hello'", "5"], ["'test'", "4"]],
                "pattern": "string length",
                "program": "lambda s: len(s)",
                "complexity": 1
            },
            {
                "examples": [["[1,2,3]", "6"], ["[4,5]", "9"], ["[1,2,3,4]", "10"]],
                "pattern": "sum of list elements",
                "program": "lambda lst: sum(lst)",
                "complexity": 2
            }
        ]
