"""
Prompt formatting utilities for LLMRouter scripts
"""

from llmrouter.prompts import load_prompt_template

# Registry for custom prompt formatters (for extensibility)
PROMPT_REGISTRY = {}


def register_prompt(task_name: str):
    """
    Decorator to register a custom prompt formatter.
    
    Args:
        task_name: Name of the task to register (e.g., 'my_custom_task')
    
    Returns:
        Decorator function
    """
    def decorator(func):
        PROMPT_REGISTRY[task_name] = func
        return func
    return decorator


def format_mc_prompt(question, choices):
    """Format prompt for multiple choice tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    formatted_choices = ""
    options = ["A", "B", "C", "D"]

    for i, choice in enumerate(choices):
        formatted_choices += f"{options[i]}. {choice}\n"

    system_prompt = load_prompt_template("task_mc")
    user_query = f"""## Question:
{question}

## Options:
{formatted_choices}"""

    return {"system": system_prompt, "user": user_query}


def format_gsm8k_prompt(query):
    """Format prompt for GSM8K math tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    system_prompt = load_prompt_template("task_gsm8k")
    user_query = f"Question: {query}"
    return {"system": system_prompt, "user": user_query}


def format_math_prompt(query):
    """Format prompt for MATH tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    system_prompt = load_prompt_template("task_math")
    user_query = f"Question: {query}"
    return {"system": system_prompt, "user": user_query}


def format_commonsense_qa_prompt(query, choices):
    """Format prompt for commonsense QA tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    label = choices["label"]
    text = choices["text"]
    choice_text = ""
    for i, j in zip(label, text):
        choice_text += "\n" + "(" + i + ")" + " " + j

    system_prompt = load_prompt_template("task_mc")  # Uses same MC format
    user_query = f"Question: {query}\n{choice_text}"
    return {"system": system_prompt, "user": user_query}


def format_mbpp_prompt(text, tests):
    """Format prompt for MBPP code generation tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    tests_str = "\n".join(tests)
    system_prompt = load_prompt_template("task_mbpp")
    user_query = f"Task: {text}\n\nYour code should pass these tests:\n\n{tests_str}"
    return {"system": system_prompt, "user": user_query}


def format_humaneval_prompt(prompt):
    """Format prompt for HumanEval code generation tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    system_prompt = load_prompt_template("task_humaneval")
    user_query = f"Complete the following function:\n\n{prompt}"
    return {"system": system_prompt, "user": user_query}


def generate_task_query(task_name, sample_data):
    """Generate query prompt based on task name and sample_data.

    Returns:
        dict: {"system": system_prompt, "user": user_query}
              For simple tasks without special formatting, system will be None.
    """
    # First check custom registry (user-defined formatters take precedence)
    if task_name in PROMPT_REGISTRY:
        formatter = PROMPT_REGISTRY[task_name]
        result = formatter(sample_data)
        # Ensure result is in correct format
        if isinstance(result, dict) and "system" in result and "user" in result:
            return result
        elif isinstance(result, str):
            # Legacy format: return as user query only
            return {"system": None, "user": result}
        else:
            # Try to convert to dict format
            return {"system": None, "user": str(result)}
    
    # Built-in task formatters
    if task_name in ["natural_qa", "trivia_qa"]:
        # No special system prompt for these tasks
        return {"system": None, "user": sample_data['query']}
    elif task_name in ["mmlu"]:
        return format_mc_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "gpqa":
        return format_mc_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "mbpp":
        return format_mbpp_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "human_eval":
        return format_humaneval_prompt(sample_data['query'])
    elif task_name == "gsm8k":
        return format_gsm8k_prompt(sample_data['query'])
    elif task_name == "commonsense_qa":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "math":
        return format_math_prompt(sample_data['query'])
    elif task_name == "openbook_qa":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "arc_challenge":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    else:
        raise ValueError(f"Unknown task name: {task_name}")
