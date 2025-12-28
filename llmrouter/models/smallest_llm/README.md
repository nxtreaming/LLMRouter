# Smallest LLM Router

## Overview

The **Smallest LLM Router** is a simple heuristic router that always selects the smallest available LLM based on model size. It prioritizes cost-efficiency over performance, making it ideal for cost-sensitive applications.

## Paper Reference

This router is a baseline method described in:

- **[GraphRouter: A Graph-based Router for LLM Selections](https://arxiv.org/abs/2410.03834)**
  - (2024). arXiv:2410.03834.
  - Uses smallest/largest LLM as baseline comparison for routing methods.

## How It Works

### Routing Logic

1. Load all LLMs from `llm_data`
2. Filter models with size ending in 'B' (billions of parameters)
3. Parse sizes (e.g., "7B" → 7.0, "70B" → 70.0)
4. Select model with minimum size
5. Route ALL queries to this single model

### No Training Required

This is a zero-shot heuristic - no training needed.

## Configuration

Requires only `llm_data` with model sizes:

```json
{
  "Qwen2.5-3B": {"size": "3B", "model": "qwen/qwen2.5-3b-instruct"},
  "Qwen2.5-7B": {"size": "7B", "model": "qwen/qwen2.5-7b-instruct"},
  "Llama-70B": {"size": "70B", "model": "meta/llama-3.1-70b-instruct"}
}
```

Router will select "Qwen2.5-3B" (smallest).

## Usage

```python
from llmrouter.models import SmallestLLM

router = SmallestLLM(yaml_path="configs/model_config_test/smallest_llm.yaml")

# All queries routed to smallest model
queries = [
    {"query": "Simple question"},
    {"query": "Complex question requiring reasoning"}
]

results = router.route_batch(queries)
# Both use same smallest model
```

## Advantages

- ✅ **Maximum Cost Savings**: Always uses cheapest model
- ✅ **Simple**: No training, no hyperparameters
- ✅ **Fast**: Instant routing decision
- ✅ **Predictable**: Deterministic behavior

## Limitations

- ❌ **Ignores Query Difficulty**: Treats all queries equally
- ❌ **May Sacrifice Quality**: Small models may underperform
- ❌ **No Adaptation**: Cannot improve with data
- ❌ **Single Model**: No load balancing

## When to Use

**Good For:**
- Extreme cost constraints
- Queries are mostly simple
- Baseline for comparison
- Development/testing with cheap model

**Alternatives:**
- Need quality → Largest LLM Router
- Balance cost-quality → Hybrid LLM Router
- Query-specific → KNN/MLP/SVM Router

## Related Routers

- **Largest LLM Router**: Opposite strategy (max quality)
- **Hybrid LLM Router**: Balances small and large models
- **ELO Router**: Data-driven single model selection

---

For questions or issues, please refer to the main LLMRouter documentation or open an issue on GitHub.
