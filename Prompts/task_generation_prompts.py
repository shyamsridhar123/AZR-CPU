"""
Task Generation Prompts for AZR System
Contains structured prompts for generating high-quality reasoning tasks.
"""

from typing import Dict, List


class TaskGenerationPrompts:
    """
    Contains prompts specifically designed for task generation
    with improved structure and few-shot examples.
    """
    
    @staticmethod
    def get_deduction_prompt_with_curriculum(complexity: int) -> str:
        """Get deduction prompt adjusted for curriculum learning."""
        base_prompt = """Generate a Python programming task for deduction reasoning.

COMPLEXITY LEVEL: {complexity}

TASK FORMAT (JSON):
{{
    "program": "lambda x: <expression>",
    "input": "<input_value>", 
    "expected_output": "<expected_result>",
    "complexity": {complexity}
}}

COMPLEXITY GUIDELINES:
- Level 1: Basic arithmetic (addition, subtraction, multiplication)
- Level 2: Simple functions with conditionals
- Level 3: List/string operations, multiple conditions
- Level 4: Complex logic, nested operations
- Level 5: Advanced algorithms, recursive patterns

QUALITY REQUIREMENTS:
1. Program must be syntactically valid Python
2. Input must be compatible with the function
3. Expected output must be mathematically correct
4. Task should be clear and unambiguous
5. Avoid edge cases that could cause errors

EXAMPLES FOR LEVEL {complexity}:
{examples}

Generate a new deduction task:"""
        
        examples = TaskGenerationPrompts._get_deduction_examples_by_complexity(complexity)
        return base_prompt.format(complexity=complexity, examples=examples)
    
    @staticmethod
    def get_abduction_prompt_with_curriculum(complexity: int) -> str:
        """Get abduction prompt adjusted for curriculum learning."""
        base_prompt = """Generate a Python programming task for abduction reasoning.

COMPLEXITY LEVEL: {complexity}

TASK FORMAT (JSON):
{{
    "input": "<input_value>",
    "expected_output": "<expected_result>",
    "complexity": {complexity}
}}

COMPLEXITY GUIDELINES:
- Level 1: Simple transformations (double, square, negate)
- Level 2: Basic conditions and simple operations
- Level 3: String/list manipulations, multiple steps
- Level 4: Complex transformations, pattern matching
- Level 5: Advanced logic, multiple operations combined

QUALITY REQUIREMENTS:
1. Input should be simple and clear
2. Transformation should have a clear logical relationship
3. Should be expressible as a single Python function
4. Avoid ambiguous transformations
5. Output should be deterministic

EXAMPLES FOR LEVEL {complexity}:
{examples}

Generate a new abduction task:"""
        
        examples = TaskGenerationPrompts._get_abduction_examples_by_complexity(complexity)
        return base_prompt.format(complexity=complexity, examples=examples)
    
    @staticmethod
    def get_induction_prompt_with_curriculum(complexity: int) -> str:
        """Get induction prompt adjusted for curriculum learning."""
        base_prompt = """Generate a Python programming task for induction reasoning.

COMPLEXITY LEVEL: {complexity}

TASK FORMAT (JSON):
{{
    "examples": [["input1", "output1"], ["input2", "output2"], ["input3", "output3"]],
    "pattern": "<description_of_pattern>",
    "complexity": {complexity}
}}

COMPLEXITY GUIDELINES:
- Level 1: Simple arithmetic patterns (multiply, add constant)
- Level 2: Basic function patterns (square, check property)
- Level 3: String/list patterns, multiple operations
- Level 4: Complex patterns, conditional logic
- Level 5: Advanced patterns, recursive or iterative logic

QUALITY REQUIREMENTS:
1. Provide 3-4 clear input-output examples
2. Pattern should be consistent across all examples
3. Pattern should be learnable from the examples
4. Include diverse inputs to clarify the pattern
5. Pattern description should be accurate

EXAMPLES FOR LEVEL {complexity}:
{examples}

Generate a new induction task:"""
        
        examples = TaskGenerationPrompts._get_induction_examples_by_complexity(complexity)
        return base_prompt.format(complexity=complexity, examples=examples)
    
    @staticmethod
    def get_structured_task_prompt(reasoning_type: str, complexity: int, 
                                 include_validation: bool = True) -> str:
        """Get a structured task generation prompt with validation instructions."""
        validation_section = ""
        if include_validation:
            validation_section = """
VALIDATION CHECKLIST:
Before generating, ensure:
□ Syntax is valid Python
□ Logic is mathematically sound  
□ Task is clear and unambiguous
□ Complexity matches the requested level
□ Task is solvable with the given information"""
        
        prompts = {
            'deduction': TaskGenerationPrompts.get_deduction_prompt_with_curriculum(complexity),
            'abduction': TaskGenerationPrompts.get_abduction_prompt_with_curriculum(complexity),
            'induction': TaskGenerationPrompts.get_induction_prompt_with_curriculum(complexity)
        }
        
        base_prompt = prompts.get(reasoning_type, prompts['deduction'])
        return base_prompt + validation_section
    
    @staticmethod
    def _get_deduction_examples_by_complexity(complexity: int) -> str:
        """Get deduction examples filtered by complexity level."""
        examples_by_level = {
            1: [
                'Program: lambda x: x + 1\nInput: 5\nExpected Output: 6',
                'Program: lambda x: x * 2\nInput: 3\nExpected Output: 6'
            ],
            2: [
                'Program: lambda x: x if x > 0 else 0\nInput: -3\nExpected Output: 0',
                'Program: lambda x, y: x + y\nInput: (4, 5)\nExpected Output: 9'
            ],
            3: [
                'Program: lambda lst: len(lst)\nInput: [1, 2, 3, 4]\nExpected Output: 4',
                'Program: lambda s: s.upper()\nInput: "hello"\nExpected Output: "HELLO"'
            ],
            4: [
                'Program: lambda lst: sum(x for x in lst if x > 0)\nInput: [-1, 2, -3, 4]\nExpected Output: 6',
                'Program: lambda s: "".join(reversed(s))\nInput: "abc"\nExpected Output: "cba"'
            ],
            5: [
                'Program: lambda n: n if n <= 1 else n * factorial(n-1)\nInput: 4\nExpected Output: 24',
                'Program: lambda lst: sorted(lst, key=lambda x: x[1])\nInput: [(1,3), (2,1), (3,2)]\nExpected Output: [(2,1), (3,2), (1,3)]'
            ]
        }
        
        examples = examples_by_level.get(complexity, examples_by_level[1])
        return '\n\n'.join(examples)
    
    @staticmethod
    def _get_abduction_examples_by_complexity(complexity: int) -> str:
        """Get abduction examples filtered by complexity level."""
        examples_by_level = {
            1: [
                'Input: 5\nExpected Output: 10\nSolution: lambda x: x * 2',
                'Input: 3\nExpected Output: 9\nSolution: lambda x: x ** 2'
            ],
            2: [
                'Input: -5\nExpected Output: 5\nSolution: lambda x: abs(x)',
                'Input: "hello"\nExpected Output: "HELLO"\nSolution: lambda s: s.upper()'
            ],
            3: [
                'Input: [1, 2, 3]\nExpected Output: 6\nSolution: lambda lst: sum(lst)',
                'Input: "abc"\nExpected Output: "cba"\nSolution: lambda s: s[::-1]'
            ],
            4: [
                'Input: [1, 2, 3, 4, 5]\nExpected Output: [2, 4]\nSolution: lambda lst: [x for x in lst if x % 2 == 0]',
                'Input: "hello world"\nExpected Output: 2\nSolution: lambda s: len(s.split())'
            ],
            5: [
                'Input: [3, 1, 4, 1, 5]\nExpected Output: [1, 1, 3, 4, 5]\nSolution: lambda lst: sorted(lst)',
                'Input: {"a": 1, "b": 2}\nExpected Output: 3\nSolution: lambda d: sum(d.values())'
            ]
        }
        
        examples = examples_by_level.get(complexity, examples_by_level[1])
        return '\n\n'.join(examples)
    
    @staticmethod  
    def _get_induction_examples_by_complexity(complexity: int) -> str:
        """Get induction examples filtered by complexity level."""
        examples_by_level = {
            1: [
                'Examples: [(1, 2), (2, 4), (3, 6)]\nPattern: multiply by 2\nSolution: lambda x: x * 2',
                'Examples: [(1, 4), (2, 5), (3, 6)]\nPattern: add 3\nSolution: lambda x: x + 3'
            ],
            2: [
                'Examples: [(1, 1), (2, 4), (3, 9)]\nPattern: square\nSolution: lambda x: x ** 2',
                'Examples: [(2, True), (3, False), (4, True)]\nPattern: check even\nSolution: lambda x: x % 2 == 0'
            ],
            3: [
                'Examples: [("abc", 3), ("hello", 5), ("test", 4)]\nPattern: string length\nSolution: lambda s: len(s)',
                'Examples: [([1,2], 3), ([3,4,5], 12), ([1], 1)]\nPattern: sum of list\nSolution: lambda lst: sum(lst)'
            ],
            4: [
                'Examples: [([1,3,2], [1,2,3]), ([5,1,4], [1,4,5])]\nPattern: sort list\nSolution: lambda lst: sorted(lst)',
                'Examples: [("hello", "olleh"), ("world", "dlrow")]\nPattern: reverse string\nSolution: lambda s: s[::-1]'
            ],
            5: [
                'Examples: [(5, 120), (4, 24), (3, 6)]\nPattern: factorial\nSolution: lambda n: 1 if n <= 1 else n * factorial(n-1)',
                'Examples: [([1,2,3], [1,4,9]), ([2,3], [4,9])]\nPattern: square each\nSolution: lambda lst: [x**2 for x in lst]'
            ]
        }
        
        examples = examples_by_level.get(complexity, examples_by_level[1])
        return '\n\n'.join(examples)
