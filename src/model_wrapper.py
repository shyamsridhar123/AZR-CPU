"""
Model Wrapper for the Absolute Zero Reasoner system.
Handles the language model that acts as both proposer and solver.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
import numpy as np
import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

# Import new prompt management system
try:
    # Try relative imports first (when used as package)
    from ..Prompts.prompt_manager import PromptManager
    from ..Prompts.task_generation_prompts import TaskGenerationPrompts
    from ..Prompts.solution_prompts import SolutionPrompts
except ImportError:
    try:
        # Fall back to adding the Prompts directory to path
        import sys
        from pathlib import Path
        prompts_path = Path(__file__).parent.parent / "Prompts"
        if str(prompts_path) not in sys.path:
            sys.path.insert(0, str(prompts_path))
        
        from prompt_manager import PromptManager
        from task_generation_prompts import TaskGenerationPrompts
        from solution_prompts import SolutionPrompts
    except ImportError as e:
        logging.warning(f"Could not import prompt management system: {e}")
        # Fallback to None, will use basic prompts
        PromptManager = None
        TaskGenerationPrompts = None
        SolutionPrompts = None


@dataclass
class ModelOutput:
    """Container for model generation output."""
    text: str
    logits: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    loss: Optional[float] = None


class ModelWrapper:
    """
    Wrapper for the language model that handles both task generation and solution.
    Implements the dual role architecture (Proposer/Solver) with shared weights.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")  # CPU-only implementation
        self.logger = logging.getLogger(__name__)
          # Set up models directory with subdirectories
        self.models_dir = Path(__file__).parent.parent / "models"
        self.base_models_dir = self.models_dir / "base_models"
        self.azr_models_dir = self.models_dir / "azr_trained"
        self.cache_dir = self.models_dir / "cache"
        
        # Ensure all directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.base_models_dir.mkdir(exist_ok=True)
        self.azr_models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)        # Check if we should load a pre-trained AZR model
        if hasattr(config, 'use_pretrained_azr') and config.use_pretrained_azr:
            self._load_pretrained_azr_model(config)
        else:
            self._load_base_model(config)

    def _detect_model_type(self, model_name_or_path: str) -> str:
        """Detect whether a model is encoder-decoder or decoder-only."""
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
            # Check if it's an encoder-decoder model
            if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
                return 'seq2seq'
            else:
                return 'causal'
        except Exception as e:
            self.logger.warning(f"Could not detect model type for {model_name_or_path}: {e}")
            # Default to causal for unknown models
            return 'causal'

    def _load_pretrained_azr_model(self, config):
        """Load a pre-trained AZR model with enhanced selection logic."""
        azr_model_path = None
        
        # Determine which model to load based on configuration
        if hasattr(config, 'pretrained_azr_path') and config.pretrained_azr_path:
            # Use specific path if provided
            azr_model_path = Path(config.pretrained_azr_path)
        else:
            # Use preference-based selection
            preference = getattr(config, 'azr_model_preference', 'latest')
            
            if preference == 'latest':
                azr_model_path = self._find_latest_azr_model()
            elif preference == 'best':
                azr_model_path = self._find_best_azr_model()
            elif preference == 'specific' or preference not in ['latest', 'best']:
                # Treat as specific model name
                azr_model_path = self._find_specific_azr_model(preference)
        
        # Fallback to latest if no model found and preference was not latest
        if not azr_model_path and getattr(config, 'azr_model_preference', 'latest') != 'latest':
            self.logger.warning(f"Could not find model with preference '{config.azr_model_preference}', trying latest")
            azr_model_path = self._find_latest_azr_model()
        
        if not azr_model_path or not azr_model_path.exists():
            if getattr(config, 'fallback_to_base', True):
                self.logger.warning("No pre-trained AZR model found, falling back to base model")
                self._load_base_model(config)
                return
            else:
                raise FileNotFoundError("No pre-trained AZR model found and fallback disabled")
        
        try:
            self.logger.info(f"Loading pre-trained AZR model from: {azr_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(azr_model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(azr_model_path),
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
            
            # Store current model info
            self.current_model_path = azr_model_path
            self.current_model_type = 'azr_trained'
            
            # Load training metadata if available
            metadata_path = azr_model_path / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.current_model_metadata = metadata
                    self.logger.info(f"Loaded model trained for {metadata.get('total_episodes', 'unknown')} episodes")
                    self.logger.info(f"Model training completed: {metadata.get('timestamp', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to load pre-trained AZR model: {e}")
            if getattr(config, 'fallback_to_base', True):
                self.logger.info("Falling back to base model")
                self._load_base_model(config)
                return
            else:
                raise
        
        # Common initialization after model loading
        self._initialize_common_components(config)
    
    def _find_latest_azr_model(self):
        """Find the latest trained AZR model."""
        if not self.azr_models_dir.exists():
            return None
        
        # Look for model directories
        model_dirs = [d for d in self.azr_models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            return None
        
        # Sort by modification time to get the latest
        latest_model = max(model_dirs, key=lambda d: d.stat().st_mtime)
        
        # Verify it has the required files
        required_files = ['config.json', 'tokenizer.json']
        if all((latest_model / f).exists() for f in required_files):
            return latest_model
        
        return None
    
    def _find_best_azr_model(self):
        """Find the best performing trained AZR model based on saved performance metrics."""
        if not self.azr_models_dir.exists():
            return None
        
        best_model = None
        best_performance = float('-inf')
        
        for model_dir in self.azr_models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            metadata_path = model_dir / "training_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Calculate performance score (can be customized)
                    performance_score = self._calculate_model_performance_score(metadata)
                    
                    if performance_score > best_performance:
                        # Verify required files exist
                        required_files = ['config.json', 'tokenizer.json']
                        if all((model_dir / f).exists() for f in required_files):
                            best_performance = performance_score
                            best_model = model_dir
                            
                except Exception as e:
                    self.logger.debug(f"Could not read metadata for {model_dir}: {e}")
                    continue
        
        return best_model
    
    def _calculate_model_performance_score(self, metadata: Dict) -> float:
        """Calculate a performance score for model comparison."""
        score = 0.0
        
        # Factor in total episodes (experience)
        episodes = metadata.get('total_episodes', 0)
        score += episodes * 0.001  # Small bonus for experience
        
        # Factor in success metrics
        avg_propose_reward = metadata.get('avg_propose_reward', 0)
        avg_solve_reward = metadata.get('avg_solve_reward', 0)
        avg_success_rate = metadata.get('avg_success_rate', 0)
        
        # Weighted combination of metrics
        score += avg_propose_reward * 0.3 + avg_solve_reward * 0.4 + avg_success_rate * 0.3
        
        return score
    
    def _find_specific_azr_model(self, model_name: str):
        """Find a specific AZR model by name."""
        if not self.azr_models_dir.exists():
            return None
        
        # Look for exact match first
        specific_path = self.azr_models_dir / model_name
        if specific_path.exists() and specific_path.is_dir():
            required_files = ['config.json', 'tokenizer.json']
            if all((specific_path / f).exists() for f in required_files):
                return specific_path
          # Look for partial matches
        for model_dir in self.azr_models_dir.iterdir():
            if model_dir.is_dir() and model_name.lower() in model_dir.name.lower():
                required_files = ['config.json', 'tokenizer.json']
                if all((model_dir / f).exists() for f in required_files):
                    return model_dir
        
        return None
    
    def _load_base_model(self, config):
        """Load the base model with support for different model types."""
        # Load model and tokenizer
        self.logger.info(f"Loading base model: {config.model_name}")
        
        # Check if model exists locally first
        local_model_path = self.base_models_dir / config.model_name.replace("/", "_")
        
        # Detect model type
        if local_model_path.exists() and getattr(config, 'use_local_models', True):
            model_path_for_detection = str(local_model_path)
        else:
            model_path_for_detection = config.model_name
            
        model_type = self._detect_model_type(model_path_for_detection)
        self.model_type = model_type
        self.logger.info(f"Detected model type: {model_type}")
        
        try:
            if local_model_path.exists() and getattr(config, 'use_local_models', True):
                self.logger.info(f"Loading model from local cache: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
                
                if model_type == 'seq2seq':
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        str(local_model_path),
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        low_cpu_mem_usage=True
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(local_model_path),
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        low_cpu_mem_usage=True
                    )
            else:
                self.logger.info(f"Downloading model: {config.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                
                if model_type == 'seq2seq':
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        config.model_name,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        low_cpu_mem_usage=True
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        config.model_name,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        low_cpu_mem_usage=True
                    )
                
                # Save model locally for future use
                if getattr(config, 'save_models_locally', True):
                    try:
                        self.logger.info(f"Saving model to local cache: {local_model_path}")
                        local_model_path.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(str(local_model_path))
                        self.tokenizer.save_pretrained(str(local_model_path))
                    except Exception as e:
                        self.logger.warning(f"Failed to save model locally: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise        # Store current model info
        self.current_model_path = config.model_name
        self.current_model_type = 'base_model'
        self.current_model_metadata = None

        # Common initialization after model loading
        self._initialize_common_components(config)
        
    def _initialize_common_components(self, config):
        """Initialize common components after model loading."""
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Initialize prompt management system
        if PromptManager is not None:
            self.prompt_manager = PromptManager()
            self.use_advanced_prompts = True
            self.logger.info("Using advanced prompt management system")
        else:
            self.prompt_manager = None
            self.use_advanced_prompts = False
            self.logger.warning("Using fallback basic prompts")
            
        # Initialize current complexity for curriculum learning
        self.current_complexity = 1

        # Role-specific prompts with examples and structure
        self.prompts = {
            'proposer': {
                'deduction': """Generate a Python function and input for deduction reasoning.

Format:
Task Type: Deduction
Program: <python function>
Input: <input value>
Expected Output: <expected result>
Complexity: <1-5>

Examples:
Task Type: Deduction
Program: lambda x: x * 2 + 1
Input: 5
Expected Output: 11
Complexity: 2

Task Type: Deduction  
Program: lambda x, y: x + y if x > 0 else x - y
Input: (3, 4)
Expected Output: 7
Complexity: 3

Generate a new task:
Task Type: Deduction
""",
                'abduction': """Generate an input and expected output for abduction reasoning.

Format:
Task Type: Abduction
Input: <input value>
Expected Output: <expected result>
Complexity: <1-5>

Examples:
Task Type: Abduction
Input: 5
Expected Output: 25
Complexity: 2

Task Type: Abduction
Input: [1, 2, 3]
Expected Output: [2, 4, 6]
Complexity: 3

Generate a new task:
Task Type: Abduction
""",
                'induction': """Generate input-output examples for induction reasoning.

Format:
Task Type: Induction
Examples: [(input1, output1), (input2, output2), ...]
Pattern: <description>
Complexity: <1-5>

Examples:
Task Type: Induction
Examples: [(1, 2), (2, 4), (3, 6)]
Pattern: multiply by 2
Complexity: 2

Task Type: Induction
Examples: [(0, 1), (1, 1), (2, 4), (3, 9)]
Pattern: square then add 1
Complexity: 4

Generate a new task:
Task Type: Induction
"""
            },
            'solver': {
                'deduction': """Solve this deduction problem step by step.

Given: Program and Input
Find: Output

Think through the execution step by step:
1. Identify the function
2. Apply the input
3. Calculate the result

Answer format: The output is <result>

Problem:
""",
                'abduction': """Solve this abduction problem step by step.

Given: Input and Expected Output
Find: Program that produces the output

Think through the pattern:
1. Analyze the transformation from input to output
2. Identify the mathematical relationship
3. Express as a Python function

Answer format: lambda x: <expression>

Problem:
""",
                'induction': """Solve this induction problem step by step.

Given: Input-output examples
Find: General pattern/rule

Think through the pattern:
1. Examine each input-output pair
2. Identify the common transformation
3. Express as a general rule

Answer format: lambda x: <expression>

Problem:
"""
            }
        }          # Generation parameters optimized for code generation
        self.generation_config = {
            'max_new_tokens': 200,  # Generate up to 200 new tokens
            'temperature': max(0.3, config.temperature * 0.7),  # Lower temperature for code
            'do_sample': True,
            'top_p': 0.8,  # Slightly more focused than 0.9
            'top_k': 40,   # Slightly more focused than 50
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'no_repeat_ngram_size': 3,
            'repetition_penalty': 1.1  # Reduce repetition in code
        }
        
        # Training state
        self.training_step = 0
        self.loss_history = []
        self.logger.info(f"Model wrapper initialized with {config.model_name}")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models without initializing them."""
        available_models = {
            'base_models': [],
            'trained_azr': []
        }
        
        # List base models
        if self.base_models_dir.exists():
            for model_dir in self.base_models_dir.iterdir():
                if model_dir.is_dir():
                    # Check if it has required model files
                    config_file = model_dir / "config.json"
                    if config_file.exists():
                        available_models['base_models'].append(model_dir.name)
        
        # List trained AZR models
        if self.azr_models_dir.exists():
            for model_dir in self.azr_models_dir.iterdir():
                if model_dir.is_dir():
                    # Check if it has required model files
                    config_file = model_dir / "config.json"
                    tokenizer_file = model_dir / "tokenizer.json"
                    if config_file.exists() and tokenizer_file.exists():
                        available_models['trained_azr'].append(model_dir.name)
        
        return available_models
    
    def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        """
        Basic text generation method for general prompts.
        
        Args:
            prompt: Text prompt to generate from
            max_length: Maximum length of generated text
            
        Returns:
            Generated text string
        """
        self.model.eval()
        
        with torch.no_grad():            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1800  # Leave room for generation (1800 + 200 tokens)
            ).to(self.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()

    def generate_task(self, task_prompt: str, reasoning_type: str = 'deduction',                     complexity: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a new task using the model in proposer role.
        
        Args:
            task_prompt: Basic prompt for task generation
            reasoning_type: Type of reasoning task to generate
            complexity: Desired complexity level (1-5)
            
        Returns:
            Dictionary containing generated task components
        """
        self.model.eval()
        
        # Determine complexity level
        if complexity is None:
            complexity = self.current_complexity
            
        with torch.no_grad():
            # Use advanced prompt system if available
            if self.use_advanced_prompts and self.prompt_manager:
                full_prompt = self.prompt_manager.get_task_generation_prompt(
                    reasoning_type, complexity, include_examples=True
                )
            else:
                # Fallback to basic prompts
                full_prompt = self.prompts['proposer'].get(reasoning_type, '') + task_prompt              # Tokenize input
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=1800  # Leave room for generation (1800 + 200 tokens)
            ).to(self.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Parse the generated text into task components
            task = self._parse_generated_task(generated_text, reasoning_type)
              # Add complexity to task
            task['complexity'] = complexity
            
        return task
    
    def generate_solution(self, solution_prompt: str, task_type: str = 'deduction', 
                         task_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a solution using the model in solver role.
        
        Args:
            solution_prompt: Basic prompt containing the problem
            task_type: Type of reasoning task
            task_data: Complete task data for advanced prompting
            
        Returns:
            Generated solution string
        """
        self.model.eval()
        
        with torch.no_grad():
            # Use advanced prompt system if available
            if self.use_advanced_prompts and self.prompt_manager and task_data:
                full_prompt = self.prompt_manager.get_solution_prompt(task_type, task_data)
            else:
                # Fallback to basic prompts
                full_prompt = self.prompts['solver'].get(task_type, '') + solution_prompt              # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1800  # Leave room for generation (1800 + 200 tokens)
            ).to(self.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
            # Decode
            solution = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Clean up the solution
            solution = self._clean_generated_solution(solution)
            
        return solution
    
    def update_weights(self, reward: float, batch_data: Optional[List[Dict]] = None):
        """
        Update model weights using reward signal.
        
        Args:
            reward: Reward signal from the environment
            batch_data: Optional batch of experiences for supervised fine-tuning
        """
        self.model.train()
        
        # Simple reward-weighted loss
        # In a full implementation, this would use PPO or similar RL algorithm
        base_loss = -reward  # Negative reward as loss
        
        if batch_data:
            # If batch data is provided, compute language modeling loss
            total_loss = 0.0
            
            for data in batch_data:
                text = data.get('text', '')
                if not text:
                    continue
                    
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length,
                    padding=True
                ).to(self.device)
                
                # Forward pass
                with torch.set_grad_enabled(True):
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['input_ids']
                    )
                    
                    # Combine reward-based loss with language modeling loss
                    total_loss += outputs.loss + base_loss
            
            # Average loss
            if batch_data:
                total_loss = total_loss / len(batch_data)
            else:
                total_loss = torch.tensor(base_loss, requires_grad=True, device=self.device)
        else:
            total_loss = torch.tensor(base_loss, requires_grad=True, device=self.device)
        
        # Backward pass
        self.optimizer.zero_grad()

        # Only do backward pass if total_loss is a tensor with gradients
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
        
        # Track training progress
        self.training_step += 1
        
        # Get loss value safely
        if isinstance(total_loss, torch.Tensor):
            loss_value = total_loss.item()
        else:
            loss_value = float(total_loss)
            
        self.loss_history.append(loss_value)
        
        self.logger.debug(f"Step {self.training_step}: Loss = {loss_value:.4f}, Reward = {reward:.4f}")
    
    def _parse_generated_task(self, generated_text: str, reasoning_type: str) -> Dict[str, Any]:
        """Parse generated text into structured task components."""
        task = {
            'type': reasoning_type,
            'program': '',
            'input': '',
            'expected_output': '',
            'examples': [],
            'raw_text': generated_text
        }
        
        # Simple parsing logic - can be enhanced with more sophisticated NLP
        lines = generated_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for program definitions
            if 'lambda' in line or 'def ' in line:
                task['program'] = line
            
            # Look for input specifications
            elif 'input:' in line.lower() or 'x =' in line:
                task['input'] = line.split(':', 1)[-1].strip() if ':' in line else line.split('=', 1)[-1].strip()
            
            # Look for output specifications
            elif 'output:' in line.lower() or 'result:' in line.lower():
                task['expected_output'] = line.split(':', 1)[-1].strip()
            
            # Look for examples (for induction)
            elif '->' in line or '→' in line:
                task['examples'].append(line)
        
        # If no program found, try to extract from the full text
        if not task['program']:
            # Look for lambda expressions
            lambda_match = re.search(r'lambda\s+[^:]+:[^,\n]+', generated_text)
            if lambda_match:
                task['program'] = lambda_match.group(0)
        
        return task
    
    def _clean_generated_solution(self, solution: str) -> str:
        """Clean and format generated solution."""
        # Remove common artifacts
        solution = solution.strip()
        
        # Remove explanation text if present
        if '\n' in solution:
            # Take the first line that looks like code
            for line in solution.split('\n'):
                line = line.strip()
                if 'lambda' in line or '=' in line or line.replace('.', '').replace('-', '').isdigit():
                    return line
        
        # Remove quotes if wrapped
        if solution.startswith('"') and solution.endswith('"'):
            solution = solution[1:-1]
        elif solution.startswith("'") and solution.endswith("'"):
            solution = solution[1:-1]
        
        return solution
    
    def get_state(self) -> Dict[str, Any]:
        """Get current model state for checkpointing."""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'loss_history': self.loss_history,
            'generation_config': self.generation_config
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load model state from checkpoint."""
        if 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
        
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        self.training_step = state.get('training_step', 0)
        self.loss_history = state.get('loss_history', [])
        
        if 'generation_config' in state:
            self.generation_config.update(state['generation_config'])
    def save_model(self, path: str, metadata: Optional[Dict] = None):
        """
        Save model, tokenizer, and training metadata to disk.
        
        Args:
            path: Path to save the model
            metadata: Optional metadata about the training (episodes, rewards, etc.)
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        # Save training metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'training_step': self.training_step,
            'loss_history': self.loss_history[-100:],  # Last 100 losses
            'model_name': self.config.model_name,
            'config': vars(self.config)
        })
        
        with open(save_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Model saved to {save_path}")
    
    def save_azr_model(self, episode: int, total_reward: float, performance_metrics: Dict):
        """
        Save the trained AZR model with episode information.
        
        Args:
            episode: Current training episode
            total_reward: Total reward achieved
            performance_metrics: Performance metrics from evaluation
        """        
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"azr_model_ep{episode}_{timestamp}"
        save_path = self.azr_models_dir / model_name
        
        metadata = {
            'episode': episode,
            'total_reward': total_reward,
            'performance_metrics': performance_metrics,
            'timestamp': timestamp,
            'training_type': 'absolute_zero_reasoning',
            'base_model': self.config.model_name
        }
        
        self.save_model(str(save_path), metadata)
          # Create symlink to latest model
        latest_path = self.azr_models_dir / "latest"
        if latest_path.exists():
            latest_path.unlink()
        try:
            latest_path.symlink_to(model_name, target_is_directory=True)
        except OSError:
            # Fallback for systems that don't support symlinks
            pass
            
        return str(save_path)

    def load_model(self, path: str):
        """Load model and tokenizer from disk."""
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
            
        self.model = AutoModelForCausalLM.from_pretrained(str(load_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(load_path))
        self.model.to(self.device)
        
        # Load training metadata if available
        metadata_path = load_path / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.training_step = metadata.get('training_step', 0)
                self.loss_history = metadata.get('loss_history', [])
                self.logger.info(f"Loaded training metadata: step {self.training_step}")
        
        self.logger.info(f"Model loaded from {path}")
    
    def load_latest_azr_model(self):
        """Load the latest trained AZR model."""
        latest_path = self.models_dir / "trained_azr" / "latest"
        if latest_path.exists():
            self.load_model(str(latest_path))
            return True
        else:
            self.logger.warning("No trained AZR model found")
            return False
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models in the models directory."""
        models = {
            'base_models': [],
            'trained_azr': []
        }
        
        # List base models
        for item in self.models_dir.iterdir():
            if item.is_dir() and item.name != "trained_azr":
                models['base_models'].append(item.name)
        
        # List trained AZR models
        azr_dir = self.models_dir / "trained_azr"
        if azr_dir.exists():
            for item in azr_dir.iterdir():
                if item.is_dir() and item.name != "latest":
                    models['trained_azr'].append(item.name)
        
        return models
    
    def evaluate_perplexity(self, text_samples: List[str]) -> float:
        """Evaluate model perplexity on text samples."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in text_samples:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    labels=inputs['input_ids']
                )
                
                total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def update_complexity(self, new_complexity: int):
        """Update the current complexity level for curriculum learning."""
        self.current_complexity = min(max(new_complexity, 1), 5)
        self.logger.info(f"Updated complexity level to {self.current_complexity}")
    
    def get_current_complexity(self) -> int:
        """Get the current complexity level."""
        return self.current_complexity
    
    def validate_task_quality(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate task quality using the prompt system.
        
        Args:
            task_data: Task to validate
            
        Returns:
            Validation results
        """
        if self.use_advanced_prompts and self.prompt_manager:
            validation_prompt = self.prompt_manager.get_validation_prompt(
                'task_validation', task_data=task_data
            )
            
            # Generate validation response
            with torch.no_grad():
                inputs = self.tokenizer(
                    validation_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 200,
                    temperature=0.3,  # Lower temperature for validation
                    do_sample=True
                )
                
                validation_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Parse validation results (simplified)
                return self._parse_validation_results(validation_text)
        else:
            # Basic validation fallback
            return {'is_valid': True, 'confidence': 0.5}
    
    def _parse_validation_results(self, validation_text: str) -> Dict[str, Any]:
        """Parse validation results from generated text."""
        # Simple parsing - could be enhanced with JSON parsing
        results = {
            'is_valid': 'true' in validation_text.lower() or 'valid' in validation_text.lower(),
            'confidence': 0.7,  # Default confidence
            'issues': []
        }
        
        # Look for specific issues mentioned
        if 'syntax' in validation_text.lower() and 'error' in validation_text.lower():
            results['issues'].append('syntax_error')
        if 'logic' in validation_text.lower() and ('error' in validation_text.lower() or 'incorrect' in validation_text.lower()):
            results['issues'].append('logic_error')
            
        return results
