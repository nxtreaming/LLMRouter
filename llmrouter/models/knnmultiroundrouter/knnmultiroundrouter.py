from typing import Any, Dict, List, Optional, Union
import os
import torch.nn as nn
import copy
from sklearn.neighbors import KNeighborsClassifier
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding, call_api

# Optional imports for local LLM inference
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AutoTokenizer = None


class KNNMultiRoundRouter(MetaRouter):
    """
    KNNMultiRoundRouter
    -------------------
    A routing module that leverages a K-Nearest Neighbors (KNN) classifier
    to select the most similar language model based on query embeddings.
    
    This router is designed for multi-round scenarios where queries are decomposed
    into sub-queries, and each sub-query is routed independently. The router works
    seamlessly with the decomposition → route → execute → aggregate pipeline.

    The router inherits from MetaRouter for consistent interface design.
    If no trained KNN model is found at the specified path, it can fall back
    to random selection.

    YAML Configuration Example:
    ---------------------------
    llm_data:
      GPT4:
        size: "175B"
        embedding: [0.12, 0.33, 0.78, 0.44]
      Claude3:
        size: "52B"
        embedding: [0.10, 0.25, 0.70, 0.50]
    optional:
      knn_model_path: "configs/knn_model.pkl"
      n_neighbors: 5
      metric: "cosine"
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the KNNMultiRoundRouter and load configuration.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        The initialization performs the following steps:
            1. Loads configuration and metadata from MetaRouter.
            2. Builds a KNN classifier using the specified hyperparameters.
            3. Prepares the training embeddings and corresponding model labels.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Initialize KNN classifier with user-defined hyperparameters
        knn_params = self.cfg["hparam"]
        self.knn_model = KNeighborsClassifier(**knn_params)

        # Select the best-performing model for each query
        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        # Prepare embedding and label arrays for KNN training
        query_embedding_id = routing_best["embedding_id"].tolist()
        self.query_embedding_list = [self.query_embedding_data[i].numpy() for i in query_embedding_id]
        self.model_name_list = routing_best["model_name"].tolist()
        
        # Initialize prompts for decomposition and aggregation
        self.DECOMP_PROMPT = """Given the query '{query}', decompose it into as many as 4 meaningful sub-queries (minimum 1, maximum 4). \
Try to cover the full scope of the original query by breaking it down into multiple specific and distinct sub-tasks \
whenever possible. Aim for the maximum number of high-quality sub-queries without introducing redundancy. \
Each sub-query should be clear, self-contained, and semantically coherent. \

**Output formatting rules (must be followed strictly):**
    - Output only the decomposed sub-queries, one per line.
    - Do not include any other text, explanations, or headers in your output.
    - Each line should contain only one sub-query.
"""
        
        self.AGENT_PROMPT = """You are a helpful assistant. \
You are participating in a multi-agent reasoning process, where a base model delegates sub-questions to specialized models like you. \
\nYour task is to do your **absolute best** to either: \n
    + Answer the question directly, if possible, and provide a brief explanation; or \n
    + Offer helpful and relevant context, background knowledge, or insights related to the question, even if you cannot fully answer it. \

If you are completely unable to answer the question or provide any relevant or helpful information, you must: \n
    + Clearly state that you are unable to assist with this question, and \n
    + Explicitly instruct the base model to consult other LLMs for further assistance. \

**Important Constraints**: \n
    + Keep your response clear, concise, and informative (preferably under 512 tokens). Your response will help guide the base model's reasoning and next steps. \n
    + Stay strictly on-topic. Do not include irrelevant or generic content. \

\n\nHere is the sub-question for you to assist with: {query}\n"""
        
        self.DECOMP_COT_PROMPT = """You are given a question along with auxiliary information, which consists of several sub-questions derived from the original question and their respective answers. Use this information to answer the original question if relevant, but make your own reasoning step by step before arriving at the final answer. 

Important: Your final answer MUST be clearly marked and enclosed within <answer> and </answer> tags at the end of your response. No other part of the output should be inside these tags.

Auxiliary Information: {info}

Question: {query}
Let's think step by step.
"""
        
        # Configuration for local LLM (for decomposition and aggregation)
        # Use .get() with defaults to handle missing config gracefully
        self.base_model = self.cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
        self.local_llm = None
        self.local_tokenizer = None
        self.use_local_llm = self.cfg.get("use_local_llm", False) and VLLM_AVAILABLE
        
        # API configuration for execution
        # Note: API keys are handled via environment variable API_KEYS in call_api()
        self.api_endpoint = self.cfg.get("api_endpoint", "https://integrate.api.nvidia.com/v1")
        
        # Load KNN model path for routing
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.knn_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query (sub-query) using the pre-trained KNN model.

        The method embeds the input query text using Longformer, then predicts
        the most similar LLM model based on the trained KNN classifier.

        Args:
            query (dict):
                A single query dictionary. Must contain the key:
                    - "query": textual input (sub-query) to be embedded.
                Optional keys:
                    - "original_query": The original query before decomposition
                    - "sub_query_index": Index of this sub-query in the sequence

        Returns:
            dict:
                Updated query dictionary containing:
                    - "model_name": predicted model name.
        """
        # Load KNN model if not already loaded
        if not hasattr(self, 'knn_model_path'):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.knn_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        
        self.knn_model = load_model(self.knn_model_path)

        # Compute query embedding and predict model
        query_text = query.get("query", "")
        query_embedding = [get_longformer_embedding(query_text).numpy()]
        model_name = self.knn_model.predict(query_embedding)[0]

        # Return updated query with prediction
        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries (sub-queries) using the pre-trained KNN model.

        Each query in the test set is embedded using Longformer, and
        the trained KNN classifier predicts the most similar model
        for each query.

        Args:
            batch (Any, optional):
                Placeholder argument for compatibility with other router interfaces.
                If None, uses self.query_data_test from loaded data.

        Returns:
            list of dict:
                A list of query dictionaries, each updated with:
                    - "model_name": predicted model name.
        """
        # Load KNN model if not already loaded
        if not hasattr(self, 'knn_model_path'):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.knn_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        
        self.knn_model = load_model(self.knn_model_path)

        query_data_output = copy.copy(self.query_data_test)
        for row in query_data_output:
            query_text = row.get("query", "")
            query_embedding = [get_longformer_embedding(query_text).numpy()]
            model_name = self.knn_model.predict(query_embedding)[0]
            row["model_name"] = model_name

        return query_data_output

    def route_with_context(
        self, 
        query: Dict[str, Any], 
        context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Route a query with optional context from previous sub-queries.
        
        This method allows for context-aware routing where information from
        previous sub-query responses can influence routing decisions. Currently,
        this is a placeholder that routes based on query only, but can be extended
        to incorporate context embeddings.

        Args:
            query (dict):
                Current sub-query to route. Must contain "query" key.
            context (list of dict, optional):
                List of previous sub-query results, each containing:
                    - "sub_query": The sub-query text
                    - "response": The response from the routed model
                    - "model_name": The model that was used
                    - "query": Original query (optional)

        Returns:
            dict:
                Updated query dictionary with "model_name" prediction.
        """
        # For now, route based on query only (same as route_single)
        # This can be extended to incorporate context embeddings
        return self.route_single(query)
    
    def _initialize_local_llm(self):
        """Initialize local LLM for decomposition and aggregation if not already initialized."""
        if not self.use_local_llm or self.local_llm is not None:
            return
        
        try:
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
            self.local_llm = LLM(
                model=self.base_model,
                trust_remote_code=True,
                dtype="float16",
                tensor_parallel_size=1
            )
            print(f"✅ Local LLM initialized: {self.base_model}")
        except Exception as e:
            print(f"⚠️  Failed to initialize local LLM: {e}. Will use API calls instead.")
            self.use_local_llm = False
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a query into sub-queries.
        
        Args:
            query: Original query to decompose
            
        Returns:
            List of sub-queries
        """
        decomp_prompt = self.DECOMP_PROMPT.format(query=query)
        
        if self.use_local_llm:
            self._initialize_local_llm()
            if self.local_llm is not None:
                # Use local LLM
                prompt_text = self.local_tokenizer.apply_chat_template(
                    [{"role": "user", "content": decomp_prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=256)
                outputs = self.local_llm.generate([prompt_text], sampling_params)
                decomp_output = outputs[0].outputs[0].text.strip()
            else:
                decomp_output = ""
        else:
            # Fallback: use API call (would need API key configured)
            # For now, return single query if API not available
            return [query]
        
        # Parse sub-queries from output
        decomp_lines = decomp_output.strip().split("\n")
        sub_queries = []
        for line in decomp_lines:
            line = line.strip()
            if line and len(line) > 2:  # Skip empty or very short lines
                sub_queries.append(line)
        
        # If no sub-queries generated, use original query
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    def _execute_sub_query(self, sub_query: str, model_name: str) -> Dict[str, Any]:
        """
        Execute a sub-query using the routed model via API.
        
        Args:
            sub_query: Sub-query to execute
            model_name: Model name to use
            
        Returns:
            Dict with response, tokens, etc.
        """
        agent_prompt = self.AGENT_PROMPT.format(query=sub_query)
        
        # Use call_api from utils
        request = {
            "api_endpoint": self.api_endpoint,
            "query": agent_prompt,
            "model_name": model_name,
            "api_name": model_name  # Assuming model_name is the API name
        }
        
        try:
            result = call_api(request, max_tokens=512, temperature=0.000001)
            return {
                "response": result.get("response", ""),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "success": "error" not in result
            }
        except Exception as e:
            print(f"Error executing sub-query with {model_name}: {e}")
            return {
                "response": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "success": False
            }
    
    def _aggregate_responses(
        self, 
        original_query: str, 
        sub_queries: List[str], 
        sub_responses: List[Dict[str, Any]],
        task_name: Optional[str] = None
    ) -> str:
        """
        Aggregate sub-query responses into final answer.
        
        Args:
            original_query: Original query
            sub_queries: List of sub-queries
            sub_responses: List of response dicts from sub-queries
            task_name: Optional task name for prompt selection
            
        Returns:
            Final aggregated answer
        """
        # Format auxiliary information
        input_info = ""
        for sub_q, sub_resp in zip(sub_queries, sub_responses):
            input_info += f"Sub-query: {sub_q}\n\n"
            input_info += f"Response: {sub_resp.get('response', '')}\n\n"
        
        # Select prompt based on task type
        mc_tasks = {"commonsense_qa", "openbook_qa", "arc_challenge", "mmlu", "gpqa"}
        if task_name and task_name in mc_tasks:
            # Multiple choice prompt
            agg_prompt = f"""You are given a multiple-choice question and supporting sub-answers. Use the information only if helpful.

Question: {original_query}

Supporting information:
{input_info}

Rules:
- Select exactly one option: A, B, C, D, or E.
- Output only the letter in <answer> tags. No explanation.

Format:
<answer>A</answer>"""
        else:
            # Standard decomposition prompt
            agg_prompt = self.DECOMP_COT_PROMPT.format(query=original_query, info=input_info)
        
        if self.use_local_llm:
            self._initialize_local_llm()
            if self.local_llm is not None:
                # Use local LLM
                prompt_text = self.local_tokenizer.apply_chat_template(
                    [{"role": "user", "content": agg_prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)
                outputs = self.local_llm.generate([prompt_text], sampling_params)
                final_answer = outputs[0].outputs[0].text.strip()
            else:
                final_answer = ""
        else:
            # Fallback: use API call with base model
            request = {
                "api_endpoint": self.api_endpoint,
                "query": agg_prompt,
                "model_name": self.base_model,
                "api_name": self.base_model
            }
            try:
                result = call_api(request, max_tokens=1024, temperature=0.0)
                final_answer = result.get("response", "")
            except Exception as e:
                print(f"Error aggregating responses: {e}")
                final_answer = ""
        
        return final_answer
    
    def answer_query(
        self, 
        query: str, 
        task_name: Optional[str] = None,
        return_intermediate: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Process a query through the full pipeline: decompose → route → execute → aggregate.
        
        This is the main method that handles the entire multi-round process.
        
        Args:
            query: Original query to answer
            task_name: Optional task name (for prompt selection)
            return_intermediate: If True, return intermediate results along with final answer
            
        Returns:
            If return_intermediate=False: Final answer string
            If return_intermediate=True: Dict with:
                - "final_answer": Final aggregated answer
                - "sub_queries": List of sub-queries
                - "routes": List of routed model names
                - "sub_responses": List of sub-query responses
                - "metadata": Additional metadata
        """
        # Step 1: Decompose query into sub-queries
        sub_queries = self._decompose_query(query)
        
        # Step 2: Route each sub-query using KNN
        routes = []
        for sub_query in sub_queries:
            routing_result = self.route_single({"query": sub_query})
            routes.append(routing_result["model_name"])
        
        # Step 3: Execute each sub-query
        sub_responses = []
        for sub_query, model_name in zip(sub_queries, routes):
            response = self._execute_sub_query(sub_query, model_name)
            sub_responses.append(response)
        
        # Step 4: Aggregate responses into final answer
        final_answer = self._aggregate_responses(query, sub_queries, sub_responses, task_name)
        
        if return_intermediate:
            return {
                "final_answer": final_answer,
                "sub_queries": sub_queries,
                "routes": routes,
                "sub_responses": sub_responses,
                "metadata": {
                    "num_sub_queries": len(sub_queries),
                    "task_name": task_name
                }
            }
        else:
            return final_answer

