"""
GMTRouter - Graph-based Multi-Turn Router

This module provides an adapter to integrate GMTRouter (https://github.com/ulab-uiuc/GMTRouter)
into the LLMRouter framework.

GMTRouter uses graph neural networks to provide personalized LLM routing across
multi-turn conversations, considering user preferences and interaction history.
"""

from typing import Any, Dict, List, Optional
import os
import copy
import torch
import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import (
    load_model,
    get_longformer_embedding,
    call_api,
    generate_task_query,
    calculate_task_performance
)


class GMTRouter(MetaRouter):
    """
    GMTRouter - Graph-based Multi-Turn Personalized Router

    A personalized LLM router that uses graph neural networks to optimize
    model selection across multi-turn conversations, considering:
    - User interaction history
    - Conversation context
    - Task-specific performance
    - Individual user preferences

    Requirements:
        - PyTorch 2.6+
        - PyTorch Geometric 2.6.1+
        - CUDA 12.4+ (for GPU training)

    YAML Configuration Example:
    ---------------------------
    data_path:
      query_data_train: 'data/gmtrouter/query_train.jsonl'
      query_data_test: 'data/gmtrouter/query_test.jsonl'
      routing_data_train: 'data/gmtrouter/routing_train_data.jsonl'
      conversation_history: 'data/gmtrouter/conversation_history.jsonl'  # Multi-turn context
      llm_data: 'data/example_data/llm_candidates/default_llm.json'

    model_path:
      gmt_checkpoint: 'saved_models/gmtrouter/gmt_model.pt'  # Pre-trained GMTRouter model
      load_model_path: 'saved_models/gmtrouter/gmtrouter.pkl'

    hparam:
      hidden_dim: 128           # GNN hidden dimension
      num_layers: 3             # Number of GNN layers
      dropout: 0.1              # Dropout rate
      aggregation: 'mean'       # Graph aggregation method
      personalization: true     # Enable user-specific routing
      context_window: 5         # Number of previous turns to consider
    """

    def __init__(self, yaml_path: str):
        """
        Initialize GMTRouter with configuration.

        Args:
            yaml_path: Path to YAML configuration file

        Note:
            This implementation provides an adapter framework. To use the full
            GMTRouter functionality, you need to:
            1. Clone https://github.com/ulab-uiuc/GMTRouter
            2. Copy graph.py and model components to this directory
            3. Install PyTorch Geometric: pip install torch-geometric==2.6.1
        """
        # Initialize with dummy model for base class
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # GMTRouter-specific configurations
        self.gmt_config = self.cfg.get("hparam", {})
        self.hidden_dim = self.gmt_config.get("hidden_dim", 128)
        self.num_layers = self.gmt_config.get("num_layers", 3)
        self.dropout = self.gmt_config.get("dropout", 0.1)
        self.personalization = self.gmt_config.get("personalization", True)
        self.context_window = self.gmt_config.get("context_window", 5)

        # User conversation history (for multi-turn routing)
        self.conversation_history = {}  # user_id -> list of (query, response, model)

        # Model will be loaded during routing
        self.gmt_model = None
        self.model_loaded = False

        print(f"[GMTRouter] Initialized with:")
        print(f"  - Hidden dim: {self.hidden_dim}")
        print(f"  - GNN layers: {self.num_layers}")
        print(f"  - Personalization: {self.personalization}")
        print(f"  - Context window: {self.context_window}")

    def _load_gmt_model(self):
        """
        Load pre-trained GMTRouter model.

        This method should load the graph neural network model trained
        using the GMTRouter framework.
        """
        if self.model_loaded:
            return

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        checkpoint_path = os.path.join(
            project_root,
            self.cfg["model_path"].get("gmt_checkpoint", "saved_models/gmtrouter/gmt_model.pt")
        )

        if os.path.exists(checkpoint_path):
            try:
                # Load GMTRouter checkpoint
                # NOTE: This requires GMTRouter's model architecture to be available
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # TODO: Initialize GMTRouter's GNN model here
                # from .graph import GMTModel  # Import from GMTRouter
                # self.gmt_model = GMTModel(...)
                # self.gmt_model.load_state_dict(checkpoint['model_state_dict'])
                # self.gmt_model.eval()

                self.model_loaded = True
                print(f"[GMTRouter] Loaded model from {checkpoint_path}")
            except Exception as e:
                print(f"[GMTRouter] Warning: Could not load checkpoint: {e}")
                print("[GMTRouter] Falling back to heuristic routing")
        else:
            print(f"[GMTRouter] Warning: Checkpoint not found at {checkpoint_path}")
            print("[GMTRouter] Using fallback heuristic routing")

    def _get_user_context(self, user_id: str, current_query: str) -> Dict[str, Any]:
        """
        Retrieve user's conversation history for personalized routing.

        Args:
            user_id: Unique user identifier
            current_query: Current query text

        Returns:
            Dictionary containing user context and history
        """
        history = self.conversation_history.get(user_id, [])

        # Get recent context within window
        recent_history = history[-self.context_window:] if history else []

        return {
            "user_id": user_id,
            "current_query": current_query,
            "history": recent_history,
            "num_previous_turns": len(history)
        }

    def _update_conversation_history(
        self,
        user_id: str,
        query: str,
        response: str,
        model_name: str
    ):
        """
        Update user's conversation history.

        Args:
            user_id: Unique user identifier
            query: User query
            response: Model response
            model_name: Selected model name
        """
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        self.conversation_history[user_id].append({
            "query": query,
            "response": response,
            "model": model_name
        })

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using GMTRouter.

        Args:
            query: Dictionary containing:
                - query (str): Query text
                - user_id (str, optional): User identifier for personalization
                - conversation_id (str, optional): Conversation thread ID

        Returns:
            Dictionary with added 'model_name' field
        """
        self._load_gmt_model()

        query_text = query.get("query", "")
        user_id = query.get("user_id", "default_user")

        # Get user context
        user_context = self._get_user_context(user_id, query_text)

        # Get query embedding
        query_embedding = get_longformer_embedding(query_text)

        if self.gmt_model is not None:
            # Use GMTRouter's GNN model for routing
            # TODO: Implement graph-based routing logic
            # model_scores = self.gmt_model.predict(query_embedding, user_context)
            # model_name = max(model_scores, key=model_scores.get)

            # Placeholder: Use fallback for now
            model_name = self._fallback_routing(query_embedding)
        else:
            # Fallback to simple embedding-based routing
            model_name = self._fallback_routing(query_embedding)

        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        query_output["routing_method"] = "gmt" if self.gmt_model else "fallback"

        return query_output

    def _fallback_routing(self, query_embedding: torch.Tensor) -> str:
        """
        Fallback routing method when GMTRouter model is not available.
        Uses simple embedding similarity.

        Args:
            query_embedding: Query embedding tensor

        Returns:
            Selected model name
        """
        # Simple fallback: use the first available model
        # In practice, this could use embedding similarity to LLM descriptions
        if hasattr(self, 'llm_data') and self.llm_data:
            return list(self.llm_data.keys())[0]
        return "default_model"

    def route_batch(
        self,
        batch: Optional[Any] = None,
        task_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Route a batch of queries and execute them.

        For multi-turn conversations, queries from the same user/conversation
        should include 'user_id' and/or 'conversation_id' fields.

        Args:
            batch: List of query dictionaries or None (uses test data)
            task_name: Optional task name for prompt formatting

        Returns:
            List of dictionaries with routing results and API responses
        """
        self._load_gmt_model()

        # Determine which data to use
        if batch is not None:
            query_data = batch if isinstance(batch, list) else [batch]
        else:
            if hasattr(self, "query_data_test") and self.query_data_test is not None:
                query_data = copy.copy(self.query_data_test)
            else:
                print("Warning: No batch provided and no test data available.")
                return []

        # Get API endpoint from config
        api_endpoint = self.cfg.get("api_endpoint", "https://integrate.api.nvidia.com/v1")

        query_data_output = []

        for row in query_data:
            # Handle both dict and non-dict inputs
            if isinstance(row, dict):
                row_copy = copy.copy(row)
                original_query = row_copy.get("query", "")
                user_id = row_copy.get("user_id", "default_user")
                row_task_name = row_copy.get("task_name", task_name)
            else:
                row_copy = {"query": str(row), "user_id": "default_user"}
                original_query = str(row)
                user_id = "default_user"
                row_task_name = task_name

            # Step 1: Route the query
            routing_result = self.route_single(row_copy)
            model_name = routing_result["model_name"]
            row_copy["model_name"] = model_name
            row_copy["routing_method"] = routing_result.get("routing_method", "gmt")

            # Step 2: Format query if task_name is provided
            if row_task_name:
                try:
                    sample_data = {
                        "query": original_query,
                        "choices": row_copy.get("choices")
                    }
                    formatted = generate_task_query(row_task_name, sample_data)
                    query_text = formatted["user"]
                    system_prompt = formatted["system"]
                except (ValueError, KeyError) as e:
                    print(f"Warning: Failed to format query: {e}")
                    query_text = original_query
                    system_prompt = None
            else:
                query_text = original_query
                system_prompt = None

            # Step 3: Call API
            api_model_name = model_name
            if hasattr(self, 'llm_data') and self.llm_data and model_name in self.llm_data:
                api_model_name = self.llm_data[model_name].get("model", model_name)

            request = {
                "api_endpoint": api_endpoint,
                "query": query_text,
                "system_prompt": system_prompt,
                "model_name": model_name,
                "api_name": api_model_name
            }

            try:
                result = call_api(request, max_tokens=1024, temperature=0.7)
                response = result.get("response", "")
                prompt_tokens = result.get("prompt_tokens", 0)
                completion_tokens = result.get("completion_tokens", 0)
                success = "error" not in result

                # Update conversation history
                self._update_conversation_history(user_id, original_query, response, model_name)
            except Exception as e:
                print(f"Error calling API: {e}")
                response = ""
                prompt_tokens = 0
                completion_tokens = 0
                success = False

            row_copy["response"] = response
            row_copy["prompt_tokens"] = prompt_tokens
            row_copy["completion_tokens"] = completion_tokens
            row_copy["input_token"] = prompt_tokens
            row_copy["output_token"] = completion_tokens
            row_copy["success"] = success

            # Step 4: Calculate performance
            ground_truth = row_copy.get("ground_truth") or row_copy.get("gt") or row_copy.get("answer")
            if ground_truth:
                task_performance = calculate_task_performance(
                    prediction=response,
                    ground_truth=ground_truth,
                    task_name=row_task_name,
                    metric=row_copy.get("metric")
                )
                if task_performance is not None:
                    row_copy["task_performance"] = task_performance

            query_data_output.append(row_copy)

        return query_data_output
