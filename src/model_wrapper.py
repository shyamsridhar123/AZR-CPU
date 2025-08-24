"""
Model Wrapper for the Absolute Zero Reasoner system.
Handles the language model that acts as both proposer and solver.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass


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
        self.cache_dir.mkdir(exist_ok=True)
        
        # CPU-only fallback model templates
        self.fallback_templates = {
            'deduction': [
                'lambda x: x + {}',
                'lambda x: x * {}', 
                'lambda x: x - {}',
                'lambda x, y: x + y',
                'lambda x, y: x * y',
                'lambda x: x if x > {} else {}',
                'lambda x: x ** 2',
                'lambda x: abs(x)',
                'lambda x: max(x, {})',
                'lambda x: min(x, {})'
            ],
            'abduction': [
                'lambda x: x + {}',
                'lambda x: x * {}',
                'lambda x: x // {}',
                'lambda x: x % {}',
            ],
            'induction': [
                'lambda x: x + 1',
                'lambda x: x * 2',
                'lambda x: x ** 2',
                'lambda x: -x'
            ]
        }
        self.use_fallback = False
        
        # Load model and tokenizer
        self.logger.info(f"Loading model: {config.model_name}")
        
        # Check if model exists locally first
        local_model_path = self.base_models_dir / config.model_name.replace("/", "_")
        
        try:
            if local_model_path.exists() and config.use_local_models:
                self.logger.info(f"Loading model from local cache: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(local_model_path),
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    low_cpu_mem_usage=True
                )
            else:
                self.logger.info(f"Downloading model: {config.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    low_cpu_mem_usage=True
                )
                
                # Save model locally for future use
                if config.save_models_locally:
                    self.logger.info(f"Saving model to local cache: {local_model_path}")
                    local_model_path.mkdir(exist_ok=True)
                    self.model.save_pretrained(str(local_model_path))
                    self.tokenizer.save_pretrained(str(local_model_path))
                    
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.info("Falling back to CPU-only simple model...")
            self._init_cpu_fallback_model()
            return
        
        # Add padding token if not present
        if not self.use_fallback and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if not self.use_fallback:
            self.model.to(self.device)
        
        # Role-specific prompts
        self.prompts = {
            'proposer': {
                'deduction': "Generate a Python function and input for deduction reasoning:\n",
                'abduction': "Generate an input and expected output for abduction reasoning:\n", 
                'induction': "Generate input-output examples for induction reasoning:\n"
            },
            'solver': {
                'deduction': "Given the program and input, what is the output?\n",
                'abduction': "What program produces this output given this input?\n",
                'induction': "What program maps these inputs to these outputs?\n"
            }
        }
        
        # Generation parameters
        if not self.use_fallback:
            self.generation_config = {
                'max_length': config.max_length,
                'temperature': config.temperature,
                'do_sample': True,
                'top_p': 0.9,
                'top_k': 50,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'no_repeat_ngram_size': 3
            }
        else:
            self.generation_config = {}  # Not needed for fallback
        
        # Training state
        self.training_step = 0
        self.loss_history = []
        
        self.logger.info(f"Model wrapper initialized with {config.model_name}")
        
        # Initialize optimizer for non-fallback models
        if not self.use_fallback:
            # Initialize optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=config.learning_rate,
                weight_decay=1e-5
            )
    
    def generate_task(self, task_prompt: str, reasoning_type: str = 'deduction') -> Dict[str, Any]:
        """
        Generate a new task using the model in proposer role.
        
        Args:
            task_prompt: Prompt for task generation
            reasoning_type: Type of reasoning task to generate
            
        Returns:
            Dictionary containing generated task components
        """
        if self.use_fallback:
            return self._generate_task_fallback(reasoning_type)
        
        self.model.eval()
        
        with torch.no_grad():
            # Add role-specific prompt
            full_prompt = self.prompts['proposer'].get(reasoning_type, '') + task_prompt
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
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
            
        return task
    
    def generate_solution(self, solution_prompt: str, task_type: str = 'deduction') -> str:
        """
        Generate a solution using the model in solver role.
        
        Args:
            solution_prompt: Prompt containing the problem
            task_type: Type of reasoning task
            
        Returns:
            Generated solution string
        """
        if self.use_fallback:
            return self._generate_solution_fallback(solution_prompt, task_type)
        
        self.model.eval()
        
        with torch.no_grad():
            # Add role-specific prompt
            full_prompt = self.prompts['solver'].get(task_type, '') + solution_prompt
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
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
        if self.use_fallback:
            self._update_weights_fallback(reward)
            return
        
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
        if self.use_fallback:
            return {
                'fallback_weights': self.fallback_weights,
                'training_step': self.training_step,
                'loss_history': self.loss_history,
                'use_fallback': True
            }
        
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'loss_history': self.loss_history,
            'generation_config': self.generation_config,
            'use_fallback': False
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load model state from checkpoint."""
        if state.get('use_fallback', False):
            self.use_fallback = True
            self.fallback_weights = state.get('fallback_weights', {'propose': 1.0, 'solve': 1.0})
            self.training_step = state.get('training_step', 0)
            self.loss_history = state.get('loss_history', [])
            return
        
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
        
        if self.use_fallback:
            # Save fallback model state
            fallback_state = {
                'use_fallback': True,
                'fallback_weights': self.fallback_weights,
                'training_step': self.training_step,
                'loss_history': self.loss_history,
                'config_model_name': self.config.model_name
            }
            
            with open(save_path / "fallback_model.json", "w") as f:
                json.dump(fallback_state, f, indent=2)
                
            self.logger.info(f"Fallback model state saved to {save_path}")
        else:
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
            'config': vars(self.config),
            'use_fallback': self.use_fallback
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
        
        # Check if this is a fallback model
        fallback_path = load_path / "fallback_model.json"
        if fallback_path.exists():
            with open(fallback_path, "r") as f:
                fallback_state = json.load(f)
            
            self.use_fallback = True
            self.fallback_weights = fallback_state.get('fallback_weights', {'propose': 1.0, 'solve': 1.0})
            self.training_step = fallback_state.get('training_step', 0)
            self.loss_history = fallback_state.get('loss_history', [])
            self.logger.info(f"Fallback model loaded from {path}")
        else:
            # Load transformers model
            self.model = AutoModelForCausalLM.from_pretrained(str(load_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(load_path))
            self.model.to(self.device)
            self.use_fallback = False
            
        # Load training metadata if available
        metadata_path = load_path / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if not self.use_fallback:  # Only update these for non-fallback
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
        if self.use_fallback:
            # Return minimal size for fallback model
            return {
                'total_parameters': 1000,  # Minimal rule-based model
                'trainable_parameters': 1000,
                'model_size_mb': 0.1  # Very small
            }
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def _init_cpu_fallback_model(self):
        """Initialize a CPU-only fallback model when transformers fail."""
        self.use_fallback = True
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.generation_config = {}
        
        # Simple rule-based weights for fallback model
        self.fallback_weights = {'propose': 1.0, 'solve': 1.0}
        self.fallback_learning_rate = 0.1
        
        # Ensure training attributes exist (they should already be initialized)
        if not hasattr(self, 'training_step'):
            self.training_step = 0
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        
        self.logger.info("CPU-only fallback model initialized")
    
    def _generate_task_fallback(self, reasoning_type: str) -> Dict[str, Any]:
        """Generate a task using simple rule-based approach."""
        import random
        
        templates = self.fallback_templates.get(reasoning_type, self.fallback_templates['deduction'])
        template = random.choice(templates)
        
        if '{}' in template:
            # Fill in random parameters
            param = random.randint(1, 10)
            if template.count('{}') == 1:
                program = template.format(param)
            else:
                param2 = random.randint(1, 5)
                program = template.format(param, param2)
        else:
            program = template
        
        # Generate input
        if 'x, y' in program:
            input_data = f"({random.randint(1, 10)}, {random.randint(1, 10)})"
        else:
            input_data = str(random.randint(1, 10))
        
        # For deduction tasks, compute the expected output
        expected_output = ""
        if reasoning_type == 'deduction':
            # Import the code executor to compute expected output
            try:
                from .code_executor import CodeExecutor
                executor = CodeExecutor(self.config)
                result = executor.execute_safe(program, input_data)
                if result['success']:
                    expected_output = str(result['output'])
            except Exception:
                pass
        
        return {
            'type': reasoning_type,
            'program': program,
            'input': input_data,
            'output': expected_output,  # Use 'output' key to match azr_system expectation
            'expected_output': expected_output,  # Also include for compatibility
            'complexity': self._estimate_complexity_fallback(program)
        }
    
    def _generate_solution_fallback(self, solution_prompt: str, task_type: str) -> str:
        """Generate a solution using simple rule-based approach."""
        import random
        import re
        
        if task_type == 'deduction':
            # For deduction, try to extract the program from the prompt and return it
            # The prompt format is: "Given program: <program> and input: <input>, what is the output?"
            program_match = re.search(r'Given program:\s*([^a]+)\s+and input:', solution_prompt)
            if program_match:
                program = program_match.group(1).strip()
                return program  # Return the program itself - that's the deduction solution
            
            # Fallback to identity function
            return "lambda x: x"
        
        elif task_type == 'abduction':
            # For abduction, try to guess the program based on input/output pattern
            if 'input' in solution_prompt and 'output' in solution_prompt:
                # Try to extract numeric values to guess the operation
                numbers = re.findall(r'\d+', solution_prompt)
                if len(numbers) >= 2:
                    try:
                        input_val = int(numbers[0])
                        output_val = int(numbers[1])
                        
                        # Try common operations
                        if output_val == input_val * 2:
                            return "lambda x: x * 2"
                        elif output_val == input_val + 1:
                            return "lambda x: x + 1"
                        elif output_val == input_val ** 2:
                            return "lambda x: x ** 2"
                        elif input_val > 0 and output_val == input_val * 3:
                            return "lambda x: x * 3"
                        elif output_val == input_val + 5:
                            return "lambda x: x + 5"
                        elif output_val == input_val - 1:
                            return "lambda x: x - 1"
                    except ValueError:
                        pass
            
            # Fallback templates
            templates = self.fallback_templates['abduction']
            return random.choice(templates).replace('{}', str(random.randint(1, 5)))
        
        elif task_type == 'induction':
            # For induction, return a common pattern
            templates = self.fallback_templates['induction']
            return random.choice(templates)
        
        return "lambda x: x"  # Default identity
    
    def _update_weights_fallback(self, reward: float):
        """Update fallback model weights."""
        self.fallback_weights['propose'] += self.fallback_learning_rate * reward
        self.fallback_weights['solve'] += self.fallback_learning_rate * reward
        self.training_step += 1
        self.loss_history.append(-reward)  # Convert reward to loss
    
    def _estimate_complexity_fallback(self, program: str) -> int:
        """Estimate complexity for fallback model."""
        complexity = 1
        complexity += program.count('+')
        complexity += program.count('*')
        complexity += program.count('if')
        complexity += program.count('lambda')
        return min(complexity, 5)
