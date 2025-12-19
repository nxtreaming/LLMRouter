# Features Overview

LLMRouter provides a comprehensive suite of features for intelligent LLM routing and optimization.

## Core Features

<div class="grid cards" markdown>

-   :brain:{ .lg .middle } __15+ Routing Strategies__

    ---

    From simple KNN to advanced neural routers

    [:octicons-arrow-right-24: Explore Routers](routers.md)

-   :material-speedometer:{ .lg .middle } __Fast Inference__

    ---

    Route queries in <50ms on average

    [:octicons-arrow-right-24: Benchmarks](#performance)

-   :wrench:{ .lg .middle } __Plugin System__

    ---

    Add custom routers without modifying core code

    [:octicons-arrow-right-24: Custom Routers](../custom-routers/quick-guide.md)

-   :material-chat:{ .lg .middle } __Interactive Chat UI__

    ---

    Gradio-based interface for testing

    [:octicons-arrow-right-24: Chat Interface](chat.md)

-   :material-code-braces:{ .lg .middle } __CLI & Python API__

    ---

    Use via command line or programmatically

    [:octicons-arrow-right-24: API Reference](../api-reference/overview.md)

-   :material-database:{ .lg .middle } __Flexible Data Formats__

    ---

    Support for ChatBot Arena, MT-Bench, custom datasets

    [:octicons-arrow-right-24: Data Guide](../tutorials/02-data-preparation.md)

</div>

---

## Routing Strategies

### Classic ML Approaches

**Fast, interpretable, and effective**

- **KNN Router** - Similarity-based routing
- **SVM Router** - Decision boundary learning
- **MLP Router** - Neural network classification
- **Matrix Factorization** - Collaborative filtering

[:octicons-arrow-right-24: Learn more about ML routers](routers.md#ml-based-routers)

### Graph-Based Methods

**Relationship-aware routing**

- **Graph Router** - Graph neural networks
- **Elo Rating Router** - Chess-style competitive ranking

[:octicons-arrow-right-24: Learn more about graph routers](routers.md#graph-based-routers)

### Neural Routers

**State-of-the-art deep learning**

- **BERT Router** - Semantic understanding
- **Causal LM Router** - Language model-based
- **Dual Contrastive Router** - Contrastive learning
- **Hybrid LLM Router** - Ensemble methods

[:octicons-arrow-right-24: Learn more about neural routers](routers.md#neural-routers)

### Multi-Round Routing

**Complex query decomposition**

- **KNN Multi-Round** - Iterative routing
- **LLM Multi-Round** - LLM-guided decomposition

[:octicons-arrow-right-24: Learn more about multi-round routers](routers.md#multi-round-routers)

---

## Training Capabilities

### Flexible Training Pipeline

```python
llmrouter train \
  --router mlprouter \
  --config configs/train.yaml \
  --device cuda \
  --epochs 20
```

**Features:**
- ✅ GPU acceleration (CUDA support)
- ✅ Automatic checkpointing
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Training/validation split
- ✅ Multiple optimization strategies

[:octicons-arrow-right-24: Training Guide](training.md)

### Supported Datasets

- **ChatBot Arena** - Real user preferences
- **MT-Bench** - Multi-turn conversations
- **Custom Datasets** - Your own data (JSONL format)

[:octicons-arrow-right-24: Data Preparation Tutorial](../tutorials/02-data-preparation.md)

---

## Inference Modes

### Single Query Inference

Route individual queries with minimal latency:

```bash
llmrouter infer \
  --router knnrouter \
  --config config.yaml \
  --query "What is machine learning?"
```

### Batch Inference

Process multiple queries efficiently:

```bash
llmrouter infer \
  --router mlprouter \
  --config config.yaml \
  --batch-file queries.jsonl \
  --output results.jsonl
```

### Route-Only Mode

Test routing logic without API calls:

```bash
llmrouter infer \
  --router graphrouter \
  --config config.yaml \
  --query "test" \
  --route-only
```

[:octicons-arrow-right-24: Inference Guide](inference.md)

---

## Interactive Chat Interface

### Gradio-Based UI

Launch a web interface for interactive testing:

```bash
llmrouter chat \
  --router knnrouter \
  --config config.yaml
```

**Features:**
- 💬 Multi-turn conversations
- 🔄 Real-time routing decisions
- 📊 Model selection visualization
- 🎨 Customizable interface
- 🌐 Shareable via public link

[:octicons-arrow-right-24: Chat Interface Guide](chat.md)

### Query Modes

- **current_only** - Only use current query
- **full_context** - Use entire conversation history
- **retrieval** - Retrieve relevant context

---

## Plugin System

### Zero-Code Custom Router Integration

Create custom routers without modifying LLMRouter:

```python
# custom_routers/myrouter/router.py
from llmrouter.routers.meta_router import MetaRouter

class MyRouter(MetaRouter):
    def route_single(self, query_input):
        # Your custom logic
        return {"model_name": selected_model, ...}
```

**Automatic Discovery:**
- ✅ `./custom_routers/` directory
- ✅ `~/.llmrouter/plugins/` directory
- ✅ `$LLMROUTER_PLUGINS` environment variable

[:octicons-arrow-right-24: Custom Router Guide](../custom-routers/quick-guide.md)

---

## API Integration

### Supported LLM Providers

Via **LiteLLM** integration:

<div class="grid" markdown>

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Meta (Llama via NVIDIA NIM)
- Mistral AI
- Cohere
- Custom endpoints

</div>

### Unified API Interface

```python
from llmrouter import LLMRouter

router = LLMRouter('knnrouter', 'config.yaml')
result = router.route("What is AI?")
print(result['model_name'])  # Selected model
print(result['response'])     # LLM response
```

---

## Data Formats

### LLM Candidates (JSON)

Define your model pool:

```json
{
  "gpt-4": {
    "model_name": "gpt-4",
    "provider": "openai",
    "cost_per_1k_tokens": 0.03,
    "max_tokens": 8192,
    "capabilities": ["reasoning", "code"]
  }
}
```

### Query Data (JSONL)

```jsonl
{"query": "What is machine learning?"}
{"query": "Explain quantum computing"}
```

### Routing Data (JSONL)

Training data with ground truth:

```jsonl
{"query": "Simple question", "best_llm": "gpt-3.5-turbo"}
{"query": "Complex reasoning task", "best_llm": "gpt-4"}
```

[:octicons-arrow-right-24: Data Format Specification](../tutorials/02-data-preparation.md)

---

## Performance

### Routing Speed

| Router | Avg Latency | Throughput |
|--------|------------|------------|
| Random | 5ms | 200 qps |
| KNN | 50ms | 20 qps |
| MLP | 20ms | 50 qps |
| BERT | 100ms | 10 qps |
| Graph | 40ms | 25 qps |

*Tested on NVIDIA A100 GPU*

### Accuracy

| Router | ChatBot Arena | MT-Bench | Custom Domain |
|--------|--------------|----------|---------------|
| Random | 33% | 31% | - |
| KNN | 76% | 72% | 65-80% |
| MLP | 81% | 78% | 70-85% |
| BERT | 85% | 83% | 75-90% |
| Hybrid | 89% | 87% | 80-93% |

### Cost Savings

Compared to always using the largest model:

- **KNN Router**: 40-50% cost reduction
- **MLP Router**: 45-55% cost reduction
- **Optimized Router**: 50-60% cost reduction

While maintaining **90%+** of the quality.

---

## CLI Commands

### Core Commands

```bash
# List available routers
llmrouter list-routers

# Train a router
llmrouter train --router ROUTER --config CONFIG

# Run inference
llmrouter infer --router ROUTER --config CONFIG --query "text"

# Launch chat interface
llmrouter chat --router ROUTER --config CONFIG
```

[:octicons-arrow-right-24: Complete CLI Reference](../api-reference/cli.md)

---

## Python API

### Programmatic Usage

```python
from llmrouter.routers import KNNRouter

# Initialize router
router = KNNRouter('configs/knn.yaml')

# Single query
result = router.route_single({"query": "What is AI?"})
print(result['model_name'])

# Batch queries
queries = [{"query": "Q1"}, {"query": "Q2"}]
results = router.route_batch(queries)

# Training
from llmrouter.trainer import Trainer
trainer = Trainer(router, 'configs/train.yaml')
trainer.train()
```

[:octicons-arrow-right-24: Python API Reference](../api-reference/overview.md)

---

## Advanced Features

### Multi-Round Routing

Decompose complex queries into sub-queries:

```python
from llmrouter.routers import KNNMultiRoundRouter

router = KNNMultiRoundRouter('config.yaml')
result = router.route_multi_round({
    "query": "Compare pros and cons of different ML approaches"
})
```

### Cost-Aware Routing

Optimize for cost while maintaining quality:

```yaml
router_params:
  cost_weight: 0.7      # Prioritize cost
  quality_weight: 0.3   # Minimum quality threshold
```

### Confidence Scores

Get routing confidence:

```python
result = router.route_single({"query": "..."})
print(result.get('confidence', 0))  # 0.0 - 1.0
```

### A/B Testing

Compare multiple routers:

```python
from llmrouter.evaluation import ABTest

test = ABTest(
    router_a='knnrouter',
    router_b='mlprouter',
    test_queries='queries.jsonl'
)
results = test.run()
print(results.summary())
```

---

## Monitoring & Logging

### Built-in Metrics

- Query routing time
- LLM inference time
- Cost per query
- Model selection distribution
- Error rates

### Custom Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

router = KNNRouter('config.yaml')
# Logs routing decisions automatically
```

### Export Results

```bash
llmrouter infer \
  --router mlprouter \
  --config config.yaml \
  --batch-file queries.jsonl \
  --output results.jsonl \
  --metrics metrics.json
```

---

## Extensibility

### Custom Embeddings

Use your own embedding model:

```python
class MyRouter(MetaRouter):
    def get_query_embedding(self, query: str):
        # Your custom embedding logic
        return custom_embedding
```

### Custom Metrics

Add custom evaluation metrics:

```python
from llmrouter.evaluation import Metric

class MyMetric(Metric):
    def compute(self, predictions, ground_truth):
        # Your metric calculation
        return score
```

### Hooks & Callbacks

```python
def on_route_complete(result):
    # Log to your monitoring system
    logger.info(f"Routed to {result['model_name']}")

router.add_callback('on_complete', on_route_complete)
```

---

## Next Steps

<div class="grid cards" markdown>

-   :rocket:{ .lg .middle } __Quick Start__

    ---

    Get started in 5 minutes

    [:octicons-arrow-right-24: Quick Start Guide](../getting-started/quick-start.md)

-   :bar_chart:{ .lg .middle } __Explore Routers__

    ---

    Detailed router documentation

    [:octicons-arrow-right-24: Router Overview](routers.md)

-   :material-school:{ .lg .middle } __Tutorials__

    ---

    Interactive Colab notebooks

    [:octicons-arrow-right-24: Browse Tutorials](../tutorials/index.md)

-   :wrench:{ .lg .middle } __Custom Routers__

    ---

    Build your own routing logic

    [:octicons-arrow-right-24: Custom Router Guide](../custom-routers/quick-guide.md)

</div>
