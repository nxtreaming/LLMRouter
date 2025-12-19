# Quick Start

Get up and running with LLMRouter in just 5 minutes!

## Prerequisites

Before starting, make sure you have:

- ✅ Python 3.10 or higher installed
- ✅ LLMRouter installed ([Installation Guide](installation.md))
- ✅ (Optional) API key for LLM provider

---

## Your First Inference

Let's route a simple query using a pre-trained KNN router.

### Step 1: Activate Environment

```bash
conda activate llmrouter
```

### Step 2: Run Route-Only Inference

Test the router without making actual API calls:

```bash
llmrouter infer \
  --router knnrouter \
  --config configs/model_config_test/knnrouter.yaml \
  --query "What is machine learning?" \
  --route-only
```

**Expected Output:**
```json
{
  "query": "What is machine learning?",
  "model_name": "meta/llama-3.1-70b-instruct",
  "predicted_llm": "meta/llama-3.1-70b-instruct",
  "routing_time": 0.045
}
```

!!! success "Congratulations!"
    You just ran your first LLM routing! The router analyzed your query and selected the most suitable model.

### Step 3: Run Full Inference (With API Call)

Set your API key and run a complete inference:

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"

llmrouter infer \
  --router knnrouter \
  --config configs/model_config_test/knnrouter.yaml \
  --query "What is machine learning?" \
  --verbose
```

**Output:**
```json
{
  "query": "What is machine learning?",
  "model_name": "meta/llama-3.1-70b-instruct",
  "response": "Machine learning is a subset of artificial intelligence...",
  "routing_time": 0.045,
  "inference_time": 1.234
}
```

---

## Try Different Routers

LLMRouter includes 15+ pre-trained routing strategies. Let's explore a few:

### Random Router (Baseline)

```bash
llmrouter infer \
  --router randomrouter \
  --config configs/model_config_test/randomrouter.yaml \
  --query "Explain quantum computing" \
  --route-only
```

Randomly selects an LLM - useful for baseline comparisons.

### MLP Router (Neural)

```bash
llmrouter infer \
  --router mlprouter \
  --config configs/model_config_test/mlprouter.yaml \
  --query "Explain quantum computing" \
  --route-only
```

Uses a neural network to learn routing patterns.

### Graph Router (Relationship-Aware)

```bash
llmrouter infer \
  --router graphrouter \
  --config configs/model_config_test/graphrouter.yaml \
  --query "Explain quantum computing" \
  --route-only
```

Models relationships between queries and LLMs as a graph.

### List All Available Routers

```bash
llmrouter list-routers
```

---

## Understanding Configuration Files

Configuration files define the router's behavior and model pool. Let's examine a simple config:

```yaml
# configs/model_config_test/knnrouter.yaml

data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
  query_data: 'data/example_data/queries/test_queries.jsonl'
  routing_data: 'data/example_data/routing_results/knn_routes.jsonl'

api_endpoint: 'https://integrate.api.nvidia.com/v1'

router_params:
  k: 5  # Number of nearest neighbors
  distance_metric: 'cosine'
```

**Key Sections:**

- **data_path**: Paths to data files
  - `llm_data`: JSON file defining available models
  - `query_data`: JSONL file with queries
  - `routing_data`: Training data for routers

- **api_endpoint**: LLM API endpoint

- **router_params**: Router-specific hyperparameters

---

## Train Your First Router

Now let's train a simple MLP router from scratch:

### Step 1: Prepare Training Data

```bash
# The example data is already included
ls data/example_data/routing_results/
```

### Step 2: Train the Router

```bash
llmrouter train \
  --router mlprouter \
  --config configs/model_config_train/mlprouter.yaml \
  --device cuda  # Use 'cpu' if no GPU
```

**Training Progress:**
```
Epoch 1/10: Loss = 1.234, Accuracy = 0.456
Epoch 2/10: Loss = 0.987, Accuracy = 0.567
...
Epoch 10/10: Loss = 0.234, Accuracy = 0.892

Model saved to: llmrouter/models/trained/mlprouter_20241219.pth
```

### Step 3: Test Your Trained Router

```bash
llmrouter infer \
  --router mlprouter \
  --config configs/model_config_test/mlprouter.yaml \
  --query "What is deep learning?" \
  --route-only
```

---

## Launch Interactive Chat

Try the Gradio-based chat interface:

```bash
llmrouter chat \
  --router knnrouter \
  --config configs/model_config_test/knnrouter.yaml
```

This will launch a local web interface at `http://127.0.0.1:7860`:

![Chat Interface](../assets/chat-interface.png)

**Features:**

- 💬 Interactive conversation
- 🔄 See which model was selected for each query
- 📊 View routing decisions in real-time
- 🎨 Customizable UI

---

## Batch Inference

Process multiple queries efficiently:

### Create a Batch File

```bash
cat > my_queries.jsonl << 'EOF'
{"query": "What is machine learning?"}
{"query": "Explain quantum computing"}
{"query": "How does a neural network work?"}
{"query": "What is the difference between AI and ML?"}
EOF
```

### Run Batch Inference

```bash
llmrouter infer \
  --router knnrouter \
  --config configs/model_config_test/knnrouter.yaml \
  --batch-file my_queries.jsonl \
  --output results.jsonl \
  --route-only
```

### View Results

```bash
cat results.jsonl
```

---

## Common Patterns

### Pattern 1: Cost-Optimized Routing

Route simple queries to cheaper models:

```yaml
# config.yaml
router_params:
  cost_weight: 0.7  # Prioritize cost
  quality_weight: 0.3
```

### Pattern 2: Quality-Optimized Routing

Route all queries to the best available model:

```bash
llmrouter infer \
  --router largestllm \
  --config configs/baselines/largest_llm.yaml \
  --query "Complex reasoning task"
```

### Pattern 3: Multi-Round Routing

Break complex queries into sub-queries:

```bash
llmrouter infer \
  --router knnmultiround \
  --config configs/model_config_test/knn_multiround.yaml \
  --query "Compare the pros and cons of different machine learning approaches"
```

---

## Understanding Routing Results

A routing result contains:

```json
{
  "query": "What is machine learning?",           // Original query
  "model_name": "meta/llama-3.1-70b-instruct",   // Selected model
  "predicted_llm": "meta/llama-3.1-70b-instruct", // Same as model_name
  "response": "Machine learning is...",           // LLM response (if not route-only)
  "routing_time": 0.045,                          // Time to route (seconds)
  "inference_time": 1.234,                        // Time for LLM call (seconds)
  "confidence": 0.892                             // Router confidence (optional)
}
```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-cog:{ .lg .middle } __Configuration__

    ---

    Learn about all configuration options

    [:octicons-arrow-right-24: Configuration Guide](configuration.md)

-   :bar_chart:{ .lg .middle } __Explore Routers__

    ---

    Understand different routing strategies

    [:octicons-arrow-right-24: Router Overview](../features/routers.md)

-   :material-school:{ .lg .middle } __Interactive Tutorials__

    ---

    Step-by-step Colab notebooks

    [:octicons-arrow-right-24: Browse Tutorials](../tutorials/index.md)

-   :wrench:{ .lg .middle } __Custom Routers__

    ---

    Build your own routing logic

    [:octicons-arrow-right-24: Custom Router Guide](../custom-routers/quick-guide.md)

</div>

---

## Quick Reference

### Common Commands

```bash
# List routers
llmrouter list-routers

# Route only (no API call)
llmrouter infer --router ROUTER --config CONFIG.yaml --query "QUERY" --route-only

# Full inference
llmrouter infer --router ROUTER --config CONFIG.yaml --query "QUERY"

# Train router
llmrouter train --router ROUTER --config CONFIG.yaml --device cuda

# Launch chat
llmrouter chat --router ROUTER --config CONFIG.yaml

# Batch inference
llmrouter infer --router ROUTER --config CONFIG.yaml --batch-file queries.jsonl
```

### Useful Flags

- `--route-only`: Skip LLM API calls, only test routing
- `--verbose`: Show detailed output
- `--device cuda`: Use GPU for training
- `--device cpu`: Force CPU usage
- `--output FILE`: Save results to file

---

## Getting Help

<div class="grid cards" markdown>

-   :material-help-circle:{ .lg .middle } __Documentation__

    ---

    Browse the full documentation

    [:octicons-arrow-right-24: Read Docs](../index.md)

-   :fontawesome-brands-github:{ .lg .middle } __GitHub Issues__

    ---

    Report bugs or request features

    [:octicons-arrow-right-24: Open Issue](https://github.com/ulab-uiuc/LLMRouter/issues)

-   :fontawesome-brands-slack:{ .lg .middle } __Slack Community__

    ---

    Get help from the community

    [:octicons-arrow-right-24: Join Slack](https://join.slack.com/t/llmrouteropen-ri04588/shared_invite/zt-3jz3cc6d1-ncwKEHvvWe0OczHx7K5c0g)

</div>

---

!!! tip "Pro Tip"
    Start with `--route-only` to test routing logic without spending API credits, then enable full inference once you're satisfied with the routing decisions.
