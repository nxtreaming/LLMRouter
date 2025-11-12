"""
API calling utilities using LiteLLM Router for load balancing

This module provides functions for making API calls to LLM services with
automatic load balancing across multiple API keys using LiteLLM Router.
"""

import os
import json
import time
from typing import Dict, List, Union, Optional, Any
from litellm import Router

# Global router cache: (api_endpoint, model_name, api_name, api_keys_tuple) -> Router
_router_cache: Dict[tuple, Router] = {}


def _parse_api_keys(api_keys_env: Optional[str] = None) -> List[str]:
    """
    Parse API keys from environment variable.
    
    Supports both single string and JSON list format:
    - Single key: "your-api-key"
    - Multiple keys: '["key1", "key2", "key3"]'
    
    Args:
        api_keys_env: Environment variable value for API_KEYS.
                     If None, reads from os.environ['API_KEYS']
    
    Returns:
        List of API key strings
    
    Raises:
        ValueError: If API_KEYS is not set or invalid
    """
    if api_keys_env is None:
        api_keys_env = os.environ.get('API_KEYS', '')
    
    if not api_keys_env:
        raise ValueError("API_KEYS environment variable is not set")
    
    # Try to parse as JSON (list format)
    try:
        parsed = json.loads(api_keys_env)
        if isinstance(parsed, list):
            return [str(key) for key in parsed if key]
        elif isinstance(parsed, str):
            return [parsed] if parsed else []
    except (json.JSONDecodeError, TypeError):
        pass
    
    # If not JSON, treat as single string
    if isinstance(api_keys_env, str) and api_keys_env.strip():
        return [api_keys_env.strip()]
    
    raise ValueError(f"Invalid API_KEYS format: {api_keys_env}")


def _create_router(
    api_endpoint: str,
    model_name: str,
    api_name: str,
    api_keys: List[str],
    timeout: int = 30,
    max_retries: int = 3
) -> Router:
    """
    Create or retrieve a cached LiteLLM Router instance for a specific model.
    
    Routers are cached by (api_endpoint, model_name, api_name, api_keys) tuple
    to avoid recreating routers for the same configuration.
    
    Args:
        api_endpoint: API endpoint URL (e.g., "https://integrate.api.nvidia.com/v1")
        model_name: Name identifier for the model (used in router)
        api_name: Actual API model name/path (e.g., "nvidia/llama3-chatqa-1.5-70b")
        api_keys: List of API keys for load balancing
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
    
    Returns:
        Configured LiteLLM Router instance (cached if available)
    """
    # Create cache key (using tuple of api_keys for hashing)
    cache_key = (api_endpoint, model_name, api_name, tuple(sorted(api_keys)), timeout, max_retries)
    
    # Return cached router if available
    if cache_key in _router_cache:
        return _router_cache[cache_key]
    
    # Create model list with all API keys for load balancing
    model_list = []
    for api_key in api_keys:
        model_list.append({
            "model_name": model_name,
            "litellm_params": {
                "model": api_name,
                "api_key": api_key,
                "api_base": api_endpoint,
                "timeout": timeout,
                "max_retries": max_retries
            }
        })
    
    # Create router with round-robin strategy for even distribution
    router = Router(
        model_list=model_list,
        routing_strategy="round_robin"
    )
    
    # Cache the router
    _router_cache[cache_key] = router
    
    return router


def call_api(
    request: Union[Dict[str, Any], List[Dict[str, Any]]],
    api_keys_env: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.9,
    timeout: int = 30,
    max_retries: int = 3
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Call LLM API using LiteLLM Router for load balancing across API keys.
    
    This function distributes API calls evenly across multiple API keys using
    LiteLLM's round-robin routing strategy.
    
    Args:
        request: Single dict or list of dicts, each containing:
            - api_endpoint (str): API endpoint URL
            - query (str): The query/prompt to send
            - model_name (str): Model identifier name
            - api_name (str): Actual API model name/path
        api_keys_env: Optional override for API_KEYS env var (for testing)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        timeout: Request timeout in seconds
        max_retries: Maximum retries for failed requests
    
    Returns:
        Single dict or list of dicts (matching input format) with added fields:
            - response (str): API response text
            - token_num (int): Total tokens used
            - prompt_tokens (int): Input tokens
            - completion_tokens (int): Output tokens
            - response_time (float): Time taken in seconds
            - error (str, optional): Error message if request failed
    
    Example:
        Single request:
        >>> request = {
        ...     "api_endpoint": "https://integrate.api.nvidia.com/v1",
        ...     "query": "What is 2+2?",
        ...     "model_name": "llama-70b",
        ...     "api_name": "nvidia/llama3-chatqa-1.5-70b"
        ... }
        >>> result = call_api(request)
        >>> print(result["response"])
        
        Batch requests:
        >>> requests = [request1, request2, request3]
        >>> results = call_api(requests)
    """
    # Parse API keys from environment
    api_keys = _parse_api_keys(api_keys_env)
    
    # Handle single request vs batch
    is_single = isinstance(request, dict)
    requests = [request] if is_single else request
    
    # Validate request format
    required_keys = {'api_endpoint', 'query', 'model_name', 'api_name'}
    for req in requests:
        missing = required_keys - set(req.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
    
    results = []
    
    # Process each request
    for req in requests:
        result = req.copy()
        start_time = time.time()
        
        try:
            # Create router for this model (cached per model_name + api_endpoint combo)
            router = _create_router(
                api_endpoint=req['api_endpoint'],
                model_name=req['model_name'],
                api_name=req['api_name'],
                api_keys=api_keys,
                timeout=timeout,
                max_retries=max_retries
            )
            
            # Make API call
            response = router.completion(
                model=req['model_name'],
                messages=[{"role": "user", "content": req['query']}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Extract response
            response_text = response.choices[0].message.content
            usage = response.usage.__dict__ if hasattr(response, 'usage') and response.usage else None
            
            # Extract token counts
            if usage:
                token_num = usage.get("total_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                # Fallback estimation
                prompt_tokens = len(req['query'].split()) if req['query'] else 0
                completion_tokens = len(response_text.split()) if isinstance(response_text, str) else 0
                token_num = prompt_tokens + completion_tokens
            
            end_time = time.time()
            
            # Add results to response
            result['response'] = response_text
            result['token_num'] = token_num
            result['prompt_tokens'] = prompt_tokens
            result['completion_tokens'] = completion_tokens
            result['response_time'] = end_time - start_time
            
        except Exception as e:
            error_msg = str(e)
            end_time = time.time()
            
            # Add error information
            result['response'] = f"API Error: {error_msg[:200]}"
            result['token_num'] = 0
            result['prompt_tokens'] = 0
            result['completion_tokens'] = 0
            result['response_time'] = end_time - start_time
            result['error'] = error_msg
        
        results.append(result)
    
    # Return single result or list based on input
    return results[0] if is_single else results

