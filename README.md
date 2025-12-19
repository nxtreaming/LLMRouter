<div style="text-align:center;">
    <img src="assets/llmrouter.png" style="width: 100%; height: auto;">
</div>



<h1 align="center">LLMRouter: An Open-Source Library for LLM Routing</h1>

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/downloads/release/python-3109/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-orange)](https://github.com/ulab-uiuc/LLMRouter/pulls)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/llmrouteropen-ri04588/shared_invite/zt-3jz3cc6d1-ncwKEHvvWe0OczHx7K5c0g)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Introduction

**LLMRouter** is an intelligent routing system designed to optimize LLM inference by dynamically selecting the most suitable model for each query. To achieve intelligent routing, it defines:

1. 🚀 *Smart Routing*: Automatically routes queries to the optimal LLM based on task complexity, cost, and performance requirements.
2. 📊 *Multiple Router Models*: Support for **over 15 routing models**, including KNN, SVM, MLP, Matrix Factorization, Elo Rating, Graph-based routers, BERT-based routers, Hybrid probabilistic routers, transformed-score routers, multi-round routers, and many additional advanced strategies.
3. 🛠️ *Unified CLI*: Complete command-line interface for training, inference, and interactive chat with Gradio-based UI.

## Supported Routers

### Single-Round Routers
| Router | Training | Inference | Description |
|--------|:--------:|:---------:|-------------|
| `knnrouter` | ✅ | ✅ | K-Nearest Neighbors based routing |
| `svmrouter` | ✅ | ✅ | Support Vector Machine based routing |
| `mlprouter` | ✅ | ✅ | Multi-Layer Perceptron based routing |
| `mfrouter` | ✅ | ✅ | Matrix Factorization based routing |
| `elorouter` | ❌ | ✅ | Elo Rating based routing |
| `routerdc` | ✅ | ✅ | Dual Contrastive learning based routing |
| `automix` | ❌ | ✅ | Automatic model mixing |
| `hybrid_llm` | ✅ | ✅ | Hybrid LLM routing strategy |
| `graphrouter` | ✅ | ✅ | Graph-based routing |
| `causallm_router` | ✅ | ✅ | Causal Language Model router |
| `smallest_llm` | ❌ | ✅ | Always routes to smallest model |
| `largest_llm` | ❌ | ✅ | Always routes to largest model|

### Multi-Round Routers
| Router | Training | Inference | Description |
|--------|:--------:|:---------:|-------------|
| `router_r1` | ❌ | ✅ | Pre-trained Router-R1 model for multi-turn conversations |

### Agentic Routers
| Router | Training | Inference | Description |
|--------|:--------:|:---------:|-------------|
| `knnmultiroundrouter` | ✅ | ✅ | KNN-based agentic router for complex tasks |
| `llmmultiroundrouter` | ❌ | ✅ | LLM-based agentic router for complex tasks |

## Get Started

### Installation

Clone the repository and install from source using a virtual environment (e.g., with anaconda3):

```bash
# Clone the repository
git clone https://github.com/ulab-uiuc/LLMRouter.git
cd LLMRouter

# Create and activate virtual environment
conda create -n llmrouter python=3.10
conda activate llmrouter

# Install the package
pip install -e .
```

> **Note**: PyPI package coming soon! Once published, you'll be able to install directly with `pip install llmrouter`.
### Training a Router

Train various router models with your configuration:
```bash
# Train KNN router
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml

# Train MLP router with GPU
llmrouter train --router mlprouter --config configs/model_config_train/mlprouter.yaml --device cuda

# Train MF router quietly
llmrouter train --router mfrouter --config configs/model_config_train/mfrouter.yaml --quiet
```

### Running Inference

Perform inference with trained routers:
```bash
# Single query inference
llmrouter infer --router knnrouter --config config.yaml --query "What is machine learning?"

# Batch inference from file
llmrouter infer --router knnrouter --config config.yaml --input queries.txt --output results.json

# Route only (without calling LLM API)
llmrouter infer --router knnrouter --config config.yaml --query "Hello" --route-only

# Custom generation parameters
llmrouter infer --router knnrouter --config config.yaml --query "Explain AI" --temp 0.7 --max-tokens 2048 --verbose
```

Input file formats supported: `.txt` (one query per line), `.json` (list of strings or objects with `"query"` field), `.jsonl` (one JSON object per line).

### Interactive Chat Interface

<div style="text-align:center;">
    <img src="assets/llmrouter_chat.gif" style="width: 100%; height: auto;">
</div>

<p align="center">
    <strong>📱 Quick Preview:</strong> Animated overview of the LLMRouter chat interface showing real-time routing and model selection.
</p>

<div style="text-align:center;">
    <video width="100%" controls style="max-width: 800px; height: auto;">
        <source src="assets/llmrouter_chat_demo.mov" type="video/quicktime">
        Your browser does not support the video tag.
    </video>
</div>

<p align="center">
    <strong>🎥 Full Demo:</strong> Complete walkthrough demonstrating the interactive chat interface, including query routing, model selection, and response generation.
</p>

Launch a Gradio-based chat interface:
```bash
# Basic chat interface
llmrouter chat --router knnrouter --config config.yaml

# Custom host and port
llmrouter chat --router knnrouter --config config.yaml --host 0.0.0.0 --port 7860

# With public sharing link
llmrouter chat --router knnrouter --config config.yaml --share

# Specify query mode
llmrouter chat --router knnrouter --config config.yaml --mode full_context --top_k 5
```

Query Modes:
- `current_only`: Routes based on current query only (default)
- `full_context`: Combines all chat history with current query
- `retrieval`: Retrieves top-k similar historical queries for context

### Direct Script Execution

You can also run the CLI scripts directly:
```bash
# Training
python -m llmrouter.cli.router_train --router knnrouter --config config.yaml

# Inference
python -m llmrouter.cli.router_inference --router knnrouter --config config.yaml --query "Hello"

# Chat
python -m llmrouter.cli.router_chat --router knnrouter --config config.yaml
```

## 🔧 Creating Custom Routers

LLMRouter supports a **plugin system** that allows you to add custom router implementations without modifying the core codebase. This makes it easy to experiment with new routing strategies or domain-specific routers.

### Quick Start

**1. Create your router directory:**
```bash
mkdir -p custom_routers/my_router
```

**2. Implement your router** (`custom_routers/my_router/router.py`):
```python
from llmrouter.models.meta_router import MetaRouter
import torch.nn as nn

class MyRouter(MetaRouter):
    """Your custom router implementation."""

    def __init__(self, yaml_path: str):
        # Initialize with a model (can be nn.Identity() for simple routers)
        model = nn.Identity()
        super().__init__(model=model, yaml_path=yaml_path)

        # Get available LLM names from config
        self.llm_names = list(self.llm_data.keys())

    def route_single(self, query_input: dict) -> dict:
        """Route a single query to the best LLM."""
        query = query_input['query']

        # Your custom routing logic here
        # Example: route based on query length
        selected_llm = (self.llm_names[0] if len(query) < 50
                       else self.llm_names[-1])

        return {
            "query": query,
            "model_name": selected_llm,
            "predicted_llm": selected_llm,
        }

    def route_batch(self, batch: list) -> list:
        """Route multiple queries."""
        return [self.route_single(q) for q in batch]
```

**3. Create configuration** (`custom_routers/my_router/config.yaml`):
```yaml
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'

hparam:
  # Your hyperparameters here

api_endpoint: 'https://integrate.api.nvidia.com/v1'
```

**4. Use your custom router** (same as built-in routers!):
```bash
# Inference
llmrouter infer --router my_router \
  --config custom_routers/my_router/config.yaml \
  --query "What is machine learning?"

# List all routers (including custom ones)
llmrouter list-routers
```

### Plugin Discovery

Custom routers are automatically discovered from:
- `./custom_routers/` (recommended - project directory)
- `~/.llmrouter/plugins/` (user home directory)
- `$LLMROUTER_PLUGINS` environment variable (colon-separated paths)

### Example Routers

LLMRouter includes example custom routers you can learn from:

**RandomRouter** - Simple baseline that randomly selects an LLM
```bash
llmrouter infer --router randomrouter \
  --config custom_routers/randomrouter/config.yaml \
  --query "Hello world"
```

**ThresholdRouter** - Advanced trainable router with difficulty estimation
```bash
# Train the router
llmrouter train --router thresholdrouter \
  --config custom_routers/thresholdrouter/config.yaml

# Use for inference
llmrouter infer --router thresholdrouter \
  --config custom_routers/thresholdrouter/config.yaml \
  --query "Explain quantum computing"
```

### Documentation

For detailed guides on creating custom routers:
- 📖 **Quick Start**: [custom_routers/README.md](custom_routers/README.md)
- 📖 **Detailed Tutorial**: [docs/CUSTOM_ROUTERS.md](docs/CUSTOM_ROUTERS.md)
- 📖 **Implementation Summary**: [CUSTOM_ROUTER_SUMMARY.md](CUSTOM_ROUTER_SUMMARY.md)

### Common Routing Patterns

**Rule-based routing:**
```python
def route_single(self, query_input):
    query = query_input['query'].lower()
    if 'code' in query:
        return {"model_name": "code-specialist"}
    elif len(query) < 50:
        return {"model_name": "small-fast-model"}
    else:
        return {"model_name": "large-capable-model"}
```

**Embedding-based routing:**
```python
from llmrouter.utils import get_longformer_embedding

def route_single(self, query_input):
    embedding = get_longformer_embedding(query_input['query'])
    # Use embedding similarity to select best model
    selected = self._find_best_model(embedding)
    return {"model_name": selected}
```

**Cost-optimized routing:**
```python
def route_single(self, query_input):
    difficulty = self._estimate_difficulty(query_input)
    # Select cheapest model that can handle the difficulty
    for model_name, info in sorted(self.llm_data.items(),
                                   key=lambda x: x[1]['cost']):
        if info['capability'] >= difficulty:
            return {"model_name": model_name}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ulab-uiuc/LLMRouter&type=date&legend=top-left)](https://www.star-history.com/#ulab-uiuc/LLMRouter&type=date&legend=top-left)
