from typing import Any, Dict, List, Optional
import os
import torch
import torch.nn as nn
import copy
from llmrouter.models.meta_router import MetaRouter


class CausalLMRouter(MetaRouter):
    """
    CausalLMRouter: A routing module that uses a finetuned Causal Language Model
    to predict the best LLM for a given query.

    The model is finetuned to predict the optimal LLM name based on query content.
    During inference, vLLM is used for efficient batch generation.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize CausalLMRouter.

        Args:
            yaml_path: Path to YAML configuration file
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Get model config
        self.model_config = self.cfg.get("hparam", {})
        self.base_model_name = self.model_config.get("base_model", "meta-llama/Llama-2-7b-hf")

        # Get available LLM names
        self.model_names = self.routing_data_train["model_name"].unique().tolist()

        # Prepare training data
        self._prepare_training_data()

        # vLLM model (initialized during inference)
        self.vllm_model = None

    def _prepare_training_data(self):
        """Prepare training data: query -> best LLM name pairs."""
        # Get best LLM for each query based on performance
        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        self.query_list = routing_best["query"].tolist()
        self.best_llm_list = routing_best["model_name"].tolist()

    def _build_prompt(self, query: str) -> str:
        """
        Build prompt for the causal LM.

        Args:
            query: User query string

        Returns:
            Formatted prompt string
        """
        llm_options = ", ".join(self.model_names)

        prompt = f"""You are an intelligent router that selects the best Large Language Model (LLM) for a given query.

Available LLMs: {llm_options}

Based on the query content, complexity, and requirements, predict which LLM would provide the best response.

Query: {query}

Best LLM:"""

        return prompt

    def _build_training_prompt(self, query: str, best_llm: str) -> str:
        """
        Build training prompt with the answer.

        Args:
            query: User query string
            best_llm: Ground truth best LLM name

        Returns:
            Formatted prompt with answer for training
        """
        prompt = self._build_prompt(query)
        return f"{prompt} {best_llm}"

    def get_training_data(self) -> List[Dict[str, str]]:
        """
        Get formatted training data for finetuning.

        Returns:
            List of dicts with 'prompt' and 'completion' keys
        """
        training_data = []
        for query, best_llm in zip(self.query_list, self.best_llm_list):
            training_data.append({
                "prompt": self._build_prompt(query),
                "completion": f" {best_llm}",
                "full_text": self._build_training_prompt(query, best_llm)
            })
        return training_data

    def _load_vllm_model(self):
        """Load model using vLLM for efficient inference."""
        if self.vllm_model is not None:
            return

        from vllm import LLM, SamplingParams

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(
            project_root,
            self.cfg["model_path"]["load_model_path"]
        )

        # Load finetuned model with vLLM
        self.vllm_model = LLM(
            model=load_model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.model_config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.model_config.get("gpu_memory_utilization", 0.9)
        )

        # Sampling parameters for generation
        self.sampling_params = SamplingParams(
            max_tokens=self.model_config.get("max_new_tokens", 32),
            temperature=self.model_config.get("temperature", 0.1),
            top_p=self.model_config.get("top_p", 0.95),
            stop=["\n", "Query:", "Available"]
        )

    def _parse_llm_name(self, generated_text: str) -> str:
        """
        Parse LLM name from generated text.

        Args:
            generated_text: Raw generated text from model

        Returns:
            Parsed LLM name (or first available if not found)
        """
        generated_text = generated_text.strip()

        # Try to find exact match
        for llm_name in self.model_names:
            if llm_name.lower() in generated_text.lower():
                return llm_name

        # Try to find partial match
        for llm_name in self.model_names:
            if any(part.lower() in generated_text.lower() for part in llm_name.split("-")):
                return llm_name

        # Default to first LLM if no match found
        return self.model_names[0]

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using the finetuned causal LM.

        Args:
            query: Dict containing 'query' key with the query string

        Returns:
            Dict with added 'model_name' key
        """
        self._load_vllm_model()

        prompt = self._build_prompt(query["query"])
        outputs = self.vllm_model.generate([prompt], self.sampling_params)

        generated_text = outputs[0].outputs[0].text
        model_name = self._parse_llm_name(generated_text)

        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using vLLM for efficient batch inference.

        Args:
            batch: Optional batch data (uses test data if None)

        Returns:
            List of dicts with added 'model_name' key
        """
        self._load_vllm_model()

        query_data_output = copy.copy(self.query_data_test)

        # Build prompts for all queries
        prompts = [self._build_prompt(row["query"]) for row in query_data_output]

        # Batch generation with vLLM
        outputs = self.vllm_model.generate(prompts, self.sampling_params)

        # Parse results
        for i, row in enumerate(query_data_output):
            generated_text = outputs[i].outputs[0].text
            model_name = self._parse_llm_name(generated_text)
            row["model_name"] = model_name

        return query_data_output
