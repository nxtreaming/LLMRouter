"""
Prompt template loader utility.

This module provides functions to load prompt templates from YAML files.
Templates are organized in subfolders:
- task_prompts/ - Task-specific prompts for evaluation benchmarks
- agentic_role/ - Agent and multi-agent reasoning prompts
- router_prompts/ - Router-specific prompt templates
- data_prompts/ - Data conversion and processing prompts
"""

import os
import yaml
from pathlib import Path

# Get the directory where this file is located
_PROMPTS_DIR = Path(__file__).parent


def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from a YAML file.
    
    Searches recursively in all subfolders of llmrouter/prompts/.
    You can specify either:
    - Just the filename: "task_mc" (searches all subfolders)
    - With subfolder path: "task_prompts/task_mc" (searches specific subfolder)
    
    Args:
        template_name: Name of the template file (without .yaml extension)
                      Can include subfolder path like "task_prompts/task_mc"
    
    Returns:
        The prompt template string
    
    Raises:
        FileNotFoundError: If the template file doesn't exist
    """
    # Try direct path first (if subfolder is specified)
    if "/" in template_name or "\\" in template_name:
        template_path = _PROMPTS_DIR / f"{template_name}.yaml"
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if 'template' not in data:
                raise ValueError(f"YAML file {template_name}.yaml must contain a 'template' key")
            return data['template']
    
    # Search recursively in all subfolders
    for root, dirs, files in os.walk(_PROMPTS_DIR):
        root_path = Path(root)
        # Skip __pycache__ and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file == f"{template_name}.yaml":
                template_path = root_path / file
                with open(template_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if 'template' not in data:
                    raise ValueError(f"YAML file {template_name}.yaml must contain a 'template' key")
                
                return data['template']
    
    # If not found, raise error
    raise FileNotFoundError(
        f"Prompt template not found: {template_name}.yaml\n"
        f"Searched in: {_PROMPTS_DIR} and all subdirectories"
    )


def load_prompt_template_with_metadata(template_name: str) -> dict:
    """
    Load a prompt template with its metadata from a YAML file.
    
    Searches recursively in all subfolders of llmrouter/prompts/.
    
    Args:
        template_name: Name of the template file (without .yaml extension)
                      Can include subfolder path like "task_prompts/task_mc"
    
    Returns:
        Dictionary with 'template' and any other metadata keys
    """
    # Try direct path first (if subfolder is specified)
    if "/" in template_name or "\\" in template_name:
        template_path = _PROMPTS_DIR / f"{template_name}.yaml"
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    
    # Search recursively in all subfolders
    for root, dirs, files in os.walk(_PROMPTS_DIR):
        root_path = Path(root)
        # Skip __pycache__ and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file == f"{template_name}.yaml":
                template_path = root_path / file
                with open(template_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
    
    # If not found, raise error
    raise FileNotFoundError(
        f"Prompt template not found: {template_name}.yaml\n"
        f"Searched in: {_PROMPTS_DIR} and all subdirectories"
    )
