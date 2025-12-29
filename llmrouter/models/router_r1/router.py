import os
import re
import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from llmrouter.models.router_r1.prompt_pool import *
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import generate_task_query, calculate_task_performance


def _get_env_var(*names, default=None):
    """Get first available environment variable from a list of names."""
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


class RouterR1(MetaRouter):
    """
    Router-R1
    -----------
    Example router that performs R1-like routing.

    This class:
        - Inherits MetaRouter to reuse configuration and utilities
        - Implements the `route_single()` method using the pre-trained model from official HF repo

    Environment Variables (fallback if not in YAML):
        - API Key: OPENAI_API_KEY, NVIDIA_API_KEY, NVAPI_KEY, or ROUTER_API_KEY
        - API Base: OPENAI_API_BASE, NVIDIA_API_BASE, or ROUTER_API_BASE
    """

    def __init__(self, yaml_path: str):
        """
        Args:
            yaml_path (str):
                Path to YAML config for this router.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Initialize hyperparameters
        self.model_id = self.cfg["hparam"]["model_id"]

        # Get api_base from config or environment variables
        self.api_base = self.cfg["hparam"].get("api_base") or _get_env_var(
            "OPENAI_API_BASE", "NVIDIA_API_BASE", "ROUTER_API_BASE"
        )

        # Get api_key from config or environment variables
        self.api_key = self.cfg["hparam"].get("api_key") or _get_env_var(
            "OPENAI_API_KEY", "NVIDIA_API_KEY", "NVAPI_KEY", "ROUTER_API_KEY"
        )

        # Validate required API configuration
        if not self.api_base:
            raise ValueError(
                "RouterR1 requires 'api_base'. Either:\n"
                "  1. Set in YAML config: hparam.api_base\n"
                "  2. Set environment variable: OPENAI_API_BASE, NVIDIA_API_BASE, or ROUTER_API_BASE"
            )
        if not self.api_key:
            raise ValueError(
                "RouterR1 requires 'api_key'. Either:\n"
                "  1. Set in YAML config: hparam.api_key\n"
                "  2. Set environment variable: OPENAI_API_KEY, NVIDIA_API_KEY, NVAPI_KEY, or ROUTER_API_KEY"
            )

    @staticmethod
    def get_query(text: str) -> Optional[str]:
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    @staticmethod
    def route(query: str, api_base: str, api_key: str) -> str:
        try:
            from llmrouter.models.router_r1.route_service import access_routing_pool
        except ImportError as e:
            raise ImportError(
                "RouterR1 requires the optional dependency `openai` (used in `route_service.py`). "
                "Install it with: `pip install openai`."
            ) from e

        ret = access_routing_pool(
            queries=[query],
            api_base=api_base,
            api_key=api_key,
        )
        return ret["result"][0]

    def route_single(self, query: Dict[str, Any], return_details: bool = False):
        """
        Perform inference on Router-R1.

        Args:
            query: Dict with 'query' field
            return_details: If True, return dict with response and token counts

        Returns:
            str if return_details=False, dict if return_details=True
        """
        # Prepare the question
        question = str(query.get("query", "")).strip()
        if not question:
            raise ValueError("RouterR1.route_single requires non-empty 'query' field")
        if not question.endswith("?"):
            question += "?"

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "RouterR1 requires the optional dependency `vllm`. Install it with: `pip install vllm`."
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError("RouterR1 currently requires CUDA (vLLM GPU runtime).")

        # Model path and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        tensor_parallel_size = max(1, torch.cuda.device_count())
        llm = LLM(model=self.model_id, dtype="float16", tensor_parallel_size=tensor_parallel_size)

        curr_route_template = '\n{output_text}\n<information>{route_results}</information>\n'

        # Initial prompt
        if self.model_id.lower().find("qwen") != -1:
            prompt = PROMPT_TEMPLATE_QWEN.format_map({"question": question})
        else:
            prompt = PROMPT_TEMPLATE_LLAMA.format_map({"question": question})
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                tokenize=False)

        # Sampling configuration
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1024,
            stop=["</search>", "</answer>"]
        )

        # Token tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_route_tokens = 0

        cnt = 0
        print('\n\n################# [Start Reasoning + Routing] ##################\n\n')
        STOP = False
        all_output = ""

        while True:
            if cnt > 4:
                break

            # Generate with vLLM and track tokens
            outputs = llm.generate(prompt, sampling_params=sampling_params)
            output = outputs[0]
            output_text = output.outputs[0].text

            # Track vLLM tokens
            # vLLM returns prompt_token_ids and output token count
            if hasattr(output, 'prompt_token_ids'):
                prompt_tokens = len(output.prompt_token_ids)
                total_prompt_tokens += prompt_tokens

            # Count completion tokens
            completion_tokens = len(tokenizer.encode(output_text))
            total_completion_tokens += completion_tokens

            if output_text.find("<answer>") != -1:
                STOP = True
                output_text += "</answer>"
            if not STOP:
                output_text += "</search>"

            print(f"[Generation {cnt}] Output:\n{output_text}")

            tmp_query = self.get_query(output_text)
            if tmp_query:
                # Call access_routing_pool directly to get token information
                try:
                    from llmrouter.models.router_r1.route_service import access_routing_pool
                    route_result_dict = access_routing_pool(
                        queries=[tmp_query],
                        api_base=self.api_base,
                        api_key=self.api_key,
                    )
                    # Extract response text and tokens
                    route_results = route_result_dict["result"][0]

                    # Track routing API tokens
                    if 'completion_tokens_list' in route_result_dict:
                        route_tokens = sum(route_result_dict['completion_tokens_list'])
                        total_route_tokens += route_tokens
                except Exception as e:
                    print(f"Warning: Failed to track routing tokens: {e}")
                    # Fallback to old route method
                    route_results = self.route(tmp_query, api_base=self.api_base, api_key=self.api_key)
            else:
                route_results = ''

            if not STOP:
                prompt += curr_route_template.format(output_text=output_text, route_results=route_results)
                all_output += curr_route_template.format(output_text=output_text, route_results=route_results)
            else:
                all_output += output_text + "\n"
                break

            cnt += 1

        print('\n\n################# [Output] ##################\n\n')
        print(all_output)
        print('\n\n################# [Output] ##################\n\n')

        if return_details:
            return {
                "response": all_output.strip(),
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "route_tokens": total_route_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens + total_route_tokens,
            }
        else:
            return all_output.strip()

    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using RouterR1's agentic reasoning and routing.

        This method performs end-to-end processing:
        1. Routes each query using vLLM-based agentic reasoning (route_single)
        2. Applies task-specific prompt formatting if task_name is provided
        3. Calculates performance metrics if ground truth is available

        Args:
            batch (Any, optional):
                If provided, routes the provided batch. If None, uses self.query_data_test from loaded data.
            task_name (str, optional):
                Task name for prompt formatting (e.g., "mmlu", "gsm8k", "commonsense_qa").

        Returns:
            list of dict:
                A list of query dictionaries with response, tokens, and performance metrics.
        """
        # Determine which data to use
        if batch is not None:
            query_data = batch if isinstance(batch, list) else [batch]
        else:
            if hasattr(self, "query_data_test") and self.query_data_test is not None:
                query_data = copy.copy(self.query_data_test)
            else:
                print("Warning: No batch provided and no test data available for batch routing.")
                return []

        query_data_output = []
        for row in query_data:
            # Handle both dict and non-dict inputs
            if isinstance(row, dict):
                row_copy = copy.copy(row)
                original_query = row_copy.get("query", "")
                row_task_name = row_copy.get("task_name", task_name)
            else:
                row_copy = {"query": str(row)}
                original_query = str(row)
                row_task_name = task_name

            # Step 1: Route using RouterR1's agentic reasoning
            # Note: RouterR1 doesn't assign a specific model_name since it's an agentic system
            # The model_name field represents the RouterR1 model itself
            row_copy["model_name"] = self.model_id

            # Step 2: Format query if task_name is provided (for evaluation purposes)
            if row_task_name:
                try:
                    sample_data = {
                        "query": original_query,
                        "choices": row_copy.get("choices", None) if isinstance(row_copy, dict) else None
                    }
                    formatted_query = generate_task_query(row_task_name, sample_data)
                    row_copy["formatted_query"] = formatted_query
                except (ValueError, KeyError) as e:
                    print(f"Warning: Failed to format query with task '{row_task_name}': {e}. Using original query.")

            # Step 3: Get response using agentic routing with token tracking
            try:
                result = self.route_single({"query": original_query}, return_details=True)
                response = result["response"]
                prompt_tokens = result["prompt_tokens"]
                completion_tokens = result["completion_tokens"]
                route_tokens = result.get("route_tokens", 0)
                success = True
            except Exception as e:
                print(f"Error during RouterR1 routing: {e}")
                response = ""
                prompt_tokens = 0
                completion_tokens = 0
                route_tokens = 0
                success = False

            # RouterR1 token breakdown:
            # - prompt_tokens: vLLM input tokens across all iterations
            # - completion_tokens: vLLM output tokens across all iterations
            # - route_tokens: External routing API tokens (from access_routing_pool)
            row_copy["response"] = response
            row_copy["prompt_tokens"] = prompt_tokens + route_tokens  # Include routing in prompt cost
            row_copy["completion_tokens"] = completion_tokens
            row_copy["input_token"] = prompt_tokens + route_tokens
            row_copy["output_token"] = completion_tokens
            row_copy["success"] = success

            # Step 4: Calculate task performance if ground truth is available
            ground_truth = row_copy.get("ground_truth") or row_copy.get("gt") or row_copy.get("answer")
            metric = row_copy.get("metric")
            if ground_truth:
                task_performance = calculate_task_performance(
                    prediction=response,
                    ground_truth=ground_truth,
                    task_name=row_task_name,
                    metric=metric
                )
                if task_performance is not None:
                    row_copy["task_performance"] = task_performance

            query_data_output.append(row_copy)

        return query_data_output
