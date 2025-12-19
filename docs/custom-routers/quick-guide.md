# Custom Router Quick Guide

Learn how to create your own custom routing strategies in just 10 minutes!

## Why Create Custom Routers?

LLMRouter's plugin system allows you to implement domain-specific routing logic without modifying the core codebase:

- 🎯 **Domain-Specific Logic**: Route based on your unique requirements
- 💰 **Custom Cost Models**: Implement your own cost optimization
- 🔧 **Easy Integration**: Drop-in plugin system
- 🚀 **No Core Changes**: Keep your code separate from the main library

---

## Quick Start: 3-Step Router Creation

### Step 1: Create Router Directory

```bash
# Create your router directory
mkdir -p custom_routers/myrouter

# Create the router file
touch custom_routers/myrouter/router.py
```

### Step 2: Implement Router Class

```python
# custom_routers/myrouter/router.py
from llmrouter.routers.meta_router import MetaRouter
from typing import Dict, Any, List
import random

class MyRouter(MetaRouter):
    """My custom routing strategy."""

    def __init__(self, yaml_path: str):
        # Initialize without a model (for simple routers)
        super().__init__(model=None, yaml_path=yaml_path)

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        """Route a single query."""
        query = query_input.get("query", "")

        # Your custom routing logic here
        if len(query) < 50:
            selected_llm = "gpt-3.5-turbo"  # Short queries → fast model
        else:
            selected_llm = "gpt-4"  # Long queries → capable model

        return {
            "query": query,
            "model_name": selected_llm,
            "predicted_llm": selected_llm,
        }

    def route_batch(self, query_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route multiple queries."""
        return [self.route_single(q) for q in query_inputs]
```

### Step 3: Create Configuration

```yaml
# configs/myrouter.yaml
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
  query_data: 'data/example_data/queries/test_queries.jsonl'

api_endpoint: 'https://integrate.api.nvidia.com/v1'

router_params:
  # Your custom parameters
  threshold: 50
```

### Step 4: Test Your Router

```bash
llmrouter infer \
  --router myrouter \
  --config configs/myrouter.yaml \
  --query "What is AI?" \
  --route-only
```

✅ **Done!** Your custom router is ready to use.

---

## Router Templates

### Template 1: Rule-Based Router

Perfect for simple, interpretable logic:

```python
from llmrouter.routers.meta_router import MetaRouter
from typing import Dict, Any, List

class RuleBasedRouter(MetaRouter):
    """Route based on simple rules."""

    def __init__(self, yaml_path: str):
        super().__init__(model=None, yaml_path=yaml_path)
        self.keywords = self.config.get('router_params', {}).get('keywords', {})

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        query = query_input.get("query", "").lower()

        # Check for keywords
        if any(kw in query for kw in self.keywords.get('code', [])):
            selected_llm = "gpt-4"  # Good at code
        elif any(kw in query for kw in self.keywords.get('math', [])):
            selected_llm = "claude-3-opus"  # Good at math
        else:
            selected_llm = "gpt-3.5-turbo"  # Default

        return {
            "query": query_input.get("query", ""),
            "model_name": selected_llm,
            "predicted_llm": selected_llm,
        }

    def route_batch(self, query_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.route_single(q) for q in query_inputs]
```

**Configuration:**
```yaml
router_params:
  keywords:
    code: ["python", "javascript", "function", "class", "debug"]
    math: ["calculate", "solve", "equation", "math", "compute"]
```

---

### Template 2: Trainable Router

For learning from data:

```python
from llmrouter.routers.meta_router import MetaRouter
from llmrouter.trainer import Trainer
import torch
import torch.nn as nn
from typing import Dict, Any, List

class DifficultyEstimator(nn.Module):
    """Neural network to estimate query difficulty."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class TrainableRouter(MetaRouter):
    """Route based on learned difficulty estimation."""

    def __init__(self, yaml_path: str):
        # Get embedding dimension
        from llmrouter.utils.embedding import get_embedding_dim
        embedding_dim = get_embedding_dim()

        # Create model
        model = DifficultyEstimator(embedding_dim)
        super().__init__(model=model, yaml_path=yaml_path)

        # Thresholds for routing
        self.easy_threshold = 0.3
        self.hard_threshold = 0.7

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        query = query_input.get("query", "")

        # Get embedding
        embedding = self.get_query_embedding(query)

        # Estimate difficulty
        with torch.no_grad():
            difficulty = self.model(embedding).item()

        # Route based on difficulty
        if difficulty < self.easy_threshold:
            selected_llm = "gpt-3.5-turbo"
        elif difficulty < self.hard_threshold:
            selected_llm = "gpt-4"
        else:
            selected_llm = "claude-3-opus"

        return {
            "query": query,
            "model_name": selected_llm,
            "predicted_llm": selected_llm,
            "difficulty": difficulty,
        }

    def route_batch(self, query_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.route_single(q) for q in query_inputs]

# Make it trainable
Trainer.register_router('trainablerouter', TrainableRouter)
```

---

### Template 3: Cost-Optimized Router

Minimize costs while maintaining quality:

```python
from llmrouter.routers.meta_router import MetaRouter
from typing import Dict, Any, List
import numpy as np

class CostOptimizedRouter(MetaRouter):
    """Route to minimize cost while meeting quality threshold."""

    def __init__(self, yaml_path: str):
        super().__init__(model=None, yaml_path=yaml_path)

        # Load model costs and capabilities
        self.model_costs = self._load_model_costs()
        self.quality_threshold = self.config.get('router_params', {}).get('quality_threshold', 0.7)

    def _load_model_costs(self) -> Dict[str, float]:
        """Load model costs from LLM data."""
        llm_data = self.load_llm_data()
        return {
            name: info.get('cost_per_1k_tokens', 0.0)
            for name, info in llm_data.items()
        }

    def _estimate_quality(self, query: str, model: str) -> float:
        """Estimate quality for query-model pair."""
        # Implement your quality estimation logic
        # This is a placeholder
        query_complexity = len(query.split()) / 100.0

        model_capability = {
            "gpt-3.5-turbo": 0.6,
            "gpt-4": 0.9,
            "claude-3-opus": 0.95,
        }.get(model, 0.5)

        return min(1.0, model_capability / max(0.1, query_complexity))

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        query = query_input.get("query", "")

        # Find cheapest model that meets quality threshold
        best_model = None
        best_cost = float('inf')

        for model, cost in sorted(self.model_costs.items(), key=lambda x: x[1]):
            quality = self._estimate_quality(query, model)

            if quality >= self.quality_threshold and cost < best_cost:
                best_model = model
                best_cost = cost

        # Fallback to first available model
        if best_model is None:
            best_model = list(self.model_costs.keys())[0]

        return {
            "query": query,
            "model_name": best_model,
            "predicted_llm": best_model,
            "estimated_cost": best_cost,
        }

    def route_batch(self, query_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.route_single(q) for q in query_inputs]
```

---

## Advanced Features

### Using Pre-trained Embeddings

```python
class EmbeddingRouter(MetaRouter):
    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        # Get query embedding (uses Longformer by default)
        embedding = self.get_query_embedding(query_input.get("query", ""))

        # Get LLM embeddings
        llm_embeddings = self.get_llm_embeddings()

        # Compute similarity
        similarities = self.compute_similarity(embedding, llm_embeddings)

        # Select best match
        best_llm = max(similarities, key=similarities.get)

        return {
            "query": query_input.get("query", ""),
            "model_name": best_llm,
            "predicted_llm": best_llm,
        }
```

### Adding Training Support

```python
from llmrouter.trainer import Trainer

class MyTrainableRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        model = MyNeuralNetwork()
        super().__init__(model=model, yaml_path=yaml_path)

    def train_step(self, batch, optimizer, criterion):
        """Define training step."""
        queries, labels = batch

        # Forward pass
        embeddings = self.get_batch_embeddings(queries)
        predictions = self.model(embeddings)

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

# Register for training
Trainer.register_router('mytrainablerouter', MyTrainableRouter)
```

---

## Plugin System

LLMRouter automatically discovers routers in these locations:

1. **Local directory**: `./custom_routers/`
2. **Home directory**: `~/.llmrouter/plugins/`
3. **Environment variable**: `$LLMROUTER_PLUGINS`

### Directory Structure

```
custom_routers/
└── myrouter/
    ├── router.py          # Required: Router implementation
    ├── config.yaml        # Optional: Default config
    ├── README.md          # Optional: Documentation
    └── requirements.txt   # Optional: Dependencies
```

### Required Interface

Your router class must:

✅ Inherit from `MetaRouter`
✅ Implement `route_single(query_input) -> dict`
✅ Implement `route_batch(query_inputs) -> list`

---

## Testing Your Router

### Unit Tests

```python
# tests/test_myrouter.py
import pytest
from custom_routers.myrouter.router import MyRouter

def test_myrouter_route_single():
    router = MyRouter('configs/myrouter.yaml')

    result = router.route_single({
        "query": "What is AI?"
    })

    assert "model_name" in result
    assert "predicted_llm" in result
    assert result["query"] == "What is AI?"

def test_myrouter_route_batch():
    router = MyRouter('configs/myrouter.yaml')

    queries = [
        {"query": "Short query"},
        {"query": "This is a much longer query that requires more processing"}
    ]

    results = router.route_batch(queries)

    assert len(results) == 2
    assert results[0]["model_name"] == "gpt-3.5-turbo"  # Short → fast
    assert results[1]["model_name"] == "gpt-4"  # Long → capable
```

### Integration Tests

```bash
# Test with CLI
llmrouter infer \
  --router myrouter \
  --config configs/myrouter.yaml \
  --query "Test query" \
  --route-only \
  --verbose
```

---

## Best Practices

### 1. Handle Edge Cases

```python
def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
    # Validate input
    query = query_input.get("query", "")
    if not query or not query.strip():
        # Handle empty query
        return self._default_route()

    # Your routing logic
    ...

    # Validate output
    if selected_llm not in self.llm_names:
        # Fallback to default
        selected_llm = self.llm_names[0]

    return {
        "query": query,
        "model_name": selected_llm,
        "predicted_llm": selected_llm,
    }
```

### 2. Add Logging

```python
import logging

class MyRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        super().__init__(model=None, yaml_path=yaml_path)
        self.logger = logging.getLogger(__name__)

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        query = query_input.get("query", "")
        self.logger.info(f"Routing query: {query[:50]}...")

        selected_llm = self._select_model(query)
        self.logger.info(f"Selected model: {selected_llm}")

        return {
            "query": query,
            "model_name": selected_llm,
            "predicted_llm": selected_llm,
        }
```

### 3. Optimize Performance

```python
from functools import lru_cache

class MyRouter(MetaRouter):
    @lru_cache(maxsize=1000)
    def _compute_expensive_feature(self, query: str) -> float:
        """Cache expensive computations."""
        # Expensive operation
        return result
```

### 4. Document Your Router

```python
class MyRouter(MetaRouter):
    """
    Custom router that routes based on query length.

    Routing Strategy:
        - Short queries (<50 chars) → gpt-3.5-turbo (fast, cheap)
        - Long queries (≥50 chars) → gpt-4 (capable, expensive)

    Configuration Parameters:
        threshold (int): Character count threshold (default: 50)
        short_model (str): Model for short queries (default: gpt-3.5-turbo)
        long_model (str): Model for long queries (default: gpt-4)

    Example:
        >>> router = MyRouter('config.yaml')
        >>> result = router.route_single({"query": "Hi"})
        >>> print(result["model_name"])
        gpt-3.5-turbo
    """
```

---

## Examples Repository

Check out complete examples in the repository:

- **RandomRouter**: `custom_routers/randomrouter/`
- **ThresholdRouter**: `custom_routers/thresholdrouter/`

---

## Next Steps

<div class="grid cards" markdown>

-   :material-file-document:{ .lg .middle } __Detailed Tutorial__

    ---

    Complete guide with advanced techniques

    [:octicons-arrow-right-24: Full Tutorial](tutorial.md)

-   :material-puzzle:{ .lg .middle } __Plugin System__

    ---

    Deep dive into the plugin architecture

    [:octicons-arrow-right-24: Plugin System Docs](plugin-system.md)

-   :material-code-braces:{ .lg .middle } __Examples__

    ---

    Browse example implementations

    [:octicons-arrow-right-24: View Examples](examples.md)

-   :material-school:{ .lg .middle } __Colab Tutorial__

    ---

    Interactive notebook for custom routers

    [:octicons-arrow-right-24: Open in Colab](https://colab.research.google.com/github/ulab-uiuc/LLMRouter/blob/main/tutorials/notebooks/07_Creating_Custom_Routers.ipynb)

</div>

---

## Getting Help

Questions about custom routers?

- :fontawesome-brands-github: [GitHub Issues](https://github.com/ulab-uiuc/LLMRouter/issues)
- :fontawesome-brands-slack: [Slack Community](https://join.slack.com/t/llmrouteropen-ri04588/shared_invite/zt-3jz3cc6d1-ncwKEHvvWe0OczHx7K5c0g)
- :material-file-document: [API Reference](../api-reference/routers.md)
