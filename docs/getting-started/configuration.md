# Configuration Guide

Learn how to configure LLMRouter for your specific use case.

## Configuration File Structure

LLMRouter uses YAML configuration files to define router behavior, data paths, and API settings.

### Basic Configuration Template

```yaml
# Data paths
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
  query_data: 'data/example_data/queries/test_queries.jsonl'
  routing_data: 'data/example_data/routing_results/training_data.jsonl'
  output_file: 'results/predictions.jsonl'

# API configuration
api_endpoint: 'https://integrate.api.nvidia.com/v1'
api_key_env: 'NVIDIA_API_KEY'

# Router-specific parameters
router_params:
  # Parameters vary by router type
  k: 5
  distance_metric: 'cosine'

# Training configuration (optional)
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  device: 'cuda'
```

---

## Data Path Configuration

### `llm_data` (Required)

Path to JSON file defining available LLM models.

**Format:**
```json
{
  "meta/llama-3.1-70b-instruct": {
    "model_name": "meta/llama-3.1-70b-instruct",
    "provider": "nvidia",
    "cost_per_1k_tokens": 0.001,
    "max_tokens": 4096,
    "capabilities": ["general", "reasoning"],
    "embedding": [0.123, 0.456, ...]  // Optional pre-computed embedding
  },
  "gpt-4": {
    "model_name": "gpt-4",
    "provider": "openai",
    "cost_per_1k_tokens": 0.03,
    "max_tokens": 8192,
    "capabilities": ["general", "reasoning", "code"]
  }
}
```

### `query_data` (Required for Inference)

Path to JSONL file containing queries.

**Format:**
```jsonl
{"query": "What is machine learning?"}
{"query": "Explain quantum computing"}
{"query": "How do neural networks work?"}
```

### `routing_data` (Required for Training)

Path to JSONL file with query-model pairs for training.

**Format:**
```jsonl
{"query": "What is ML?", "best_llm": "meta/llama-3.1-70b-instruct"}
{"query": "Simple math problem", "best_llm": "gpt-3.5-turbo"}
```

### `output_file` (Optional)

Path where results will be saved.

```yaml
data_path:
  output_file: 'results/my_predictions.jsonl'
```

---

## API Configuration

### `api_endpoint`

The base URL for LLM API calls.

**Common Endpoints:**
```yaml
# NVIDIA NIM
api_endpoint: 'https://integrate.api.nvidia.com/v1'

# OpenAI
api_endpoint: 'https://api.openai.com/v1'

# Custom endpoint
api_endpoint: 'http://localhost:8000/v1'
```

### `api_key_env`

Environment variable name containing the API key.

```yaml
api_key_env: 'NVIDIA_API_KEY'  # Will read from $NVIDIA_API_KEY
```

You can also use multiple API keys for different providers:

```yaml
api_keys:
  openai: 'OPENAI_API_KEY'
  anthropic: 'ANTHROPIC_API_KEY'
  nvidia: 'NVIDIA_API_KEY'
```

---

## Router-Specific Parameters

Different routers require different configuration parameters.

### KNN Router

```yaml
router_params:
  k: 5                          # Number of nearest neighbors
  distance_metric: 'cosine'     # 'cosine', 'euclidean', 'manhattan'
  weights: 'uniform'            # 'uniform' or 'distance'
```

### SVM Router

```yaml
router_params:
  kernel: 'rbf'                 # 'linear', 'rbf', 'poly'
  C: 1.0                        # Regularization parameter
  gamma: 'scale'                # Kernel coefficient
```

### MLP Router

```yaml
router_params:
  hidden_sizes: [256, 128, 64]  # Hidden layer sizes
  activation: 'relu'            # 'relu', 'tanh', 'sigmoid'
  dropout: 0.2                  # Dropout rate
```

### Graph Router

```yaml
router_params:
  num_layers: 3                 # Number of GNN layers
  hidden_dim: 128               # Hidden dimension
  num_heads: 4                  # Attention heads
  dropout: 0.1
```

### Matrix Factorization Router

```yaml
router_params:
  embedding_dim: 64             # Embedding dimension
  num_factors: 32               # Number of latent factors
  regularization: 0.01          # L2 regularization
```

### Causal LM Router

```yaml
router_params:
  model_name: 'gpt2'            # Base model
  max_length: 512               # Max sequence length
  num_labels: 10                # Number of LLMs to choose from
```

### Multi-Round Routers

```yaml
router_params:
  max_rounds: 3                 # Maximum routing rounds
  base_router: 'knnrouter'      # Base router to use
  decomposition_strategy: 'llm' # 'llm' or 'rule-based'
```

---

## Training Configuration

### Basic Training Parameters

```yaml
training:
  epochs: 10                    # Number of training epochs
  batch_size: 32                # Batch size
  learning_rate: 0.001          # Learning rate
  device: 'cuda'                # 'cuda' or 'cpu'
  seed: 42                      # Random seed for reproducibility
```

### Advanced Training Options

```yaml
training:
  # Optimizer settings
  optimizer: 'adam'             # 'adam', 'sgd', 'adamw'
  weight_decay: 0.0001          # L2 regularization

  # Learning rate scheduling
  lr_scheduler: 'cosine'        # 'step', 'cosine', 'plateau'
  warmup_epochs: 2              # Warmup period

  # Early stopping
  early_stopping: true
  patience: 5                   # Epochs without improvement

  # Checkpointing
  save_best_only: true
  checkpoint_dir: 'checkpoints/'

  # Validation
  validation_split: 0.2         # Train/val split ratio
  eval_every: 1                 # Evaluate every N epochs
```

---

## Complete Configuration Examples

### Example 1: Simple KNN Router for Testing

```yaml
# configs/my_knn_router.yaml
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
  query_data: 'data/example_data/queries/test_queries.jsonl'

api_endpoint: 'https://integrate.api.nvidia.com/v1'

router_params:
  k: 5
  distance_metric: 'cosine'
```

**Usage:**
```bash
llmrouter infer --router knnrouter --config configs/my_knn_router.yaml --query "test"
```

### Example 2: MLP Router Training

```yaml
# configs/train_mlp.yaml
data_path:
  llm_data: 'data/chatbot_arena/llm_candidates.json'
  routing_data: 'data/chatbot_arena/training_routes.jsonl'
  output_file: 'results/mlp_predictions.jsonl'

api_endpoint: 'https://integrate.api.nvidia.com/v1'

router_params:
  hidden_sizes: [512, 256, 128]
  activation: 'relu'
  dropout: 0.3

training:
  epochs: 20
  batch_size: 64
  learning_rate: 0.001
  device: 'cuda'
  validation_split: 0.2
  early_stopping: true
  patience: 5
```

**Usage:**
```bash
llmrouter train --router mlprouter --config configs/train_mlp.yaml
```

### Example 3: Production Multi-Provider Setup

```yaml
# configs/production.yaml
data_path:
  llm_data: 'data/production/llm_pool.json'
  query_data: 'data/production/queries.jsonl'
  output_file: 'results/production_routes.jsonl'

# Multiple API endpoints
api_config:
  default_endpoint: 'https://integrate.api.nvidia.com/v1'
  openai_endpoint: 'https://api.openai.com/v1'
  anthropic_endpoint: 'https://api.anthropic.com/v1'

api_keys:
  nvidia: 'NVIDIA_API_KEY'
  openai: 'OPENAI_API_KEY'
  anthropic: 'ANTHROPIC_API_KEY'

router_params:
  k: 7
  distance_metric: 'cosine'
  cost_weight: 0.6        # Optimize for cost
  quality_weight: 0.4     # While maintaining quality

# Logging and monitoring
logging:
  level: 'INFO'
  file: 'logs/production.log'
  metrics: true
```

---

## Environment Variables

LLMRouter supports configuration via environment variables:

### API Keys

```bash
export NVIDIA_API_KEY="your-nvidia-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Custom Plugin Directory

```bash
export LLMROUTER_PLUGINS="/path/to/custom/routers"
```

### Cache Directory

```bash
export LLMROUTER_CACHE_DIR="~/.cache/llmrouter"
```

### Logging Level

```bash
export LLMROUTER_LOG_LEVEL="DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

---

## Configuration Validation

Validate your configuration before running:

```python
from llmrouter.utils.config import load_config, validate_config

# Load configuration
config = load_config('configs/my_config.yaml')

# Validate
is_valid, errors = validate_config(config)
if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

---

## Configuration Best Practices

### 1. Use Separate Configs for Train/Test

```
configs/
├── training/
│   ├── mlp_train.yaml
│   ├── knn_train.yaml
│   └── graph_train.yaml
└── inference/
    ├── mlp_test.yaml
    ├── knn_test.yaml
    └── graph_test.yaml
```

### 2. Keep Sensitive Data in Environment Variables

❌ **Bad:**
```yaml
api_key: "sk-1234567890abcdef"  # Never hardcode!
```

✅ **Good:**
```yaml
api_key_env: 'OPENAI_API_KEY'
```

### 3. Use Relative Paths

```yaml
data_path:
  llm_data: 'data/llm_candidates.json'  # Relative to project root
```

### 4. Document Custom Parameters

```yaml
router_params:
  k: 5                    # Number of neighbors - tested 3,5,7; 5 performs best
  distance_metric: 'cosine'  # Cosine works better than euclidean for text
```

### 5. Version Your Configurations

```yaml
# Version tracking
config_version: '1.2.0'
created_date: '2024-12-19'
description: 'Production KNN router with cost optimization'
```

---

## Advanced Topics

### Dynamic Configuration

Load configuration programmatically:

```python
from llmrouter.utils.config import Config

# Create config object
config = Config()
config.set('data_path.llm_data', 'path/to/llms.json')
config.set('router_params.k', 5)

# Use config
router = KNNRouter(config)
```

### Configuration Inheritance

Base configuration:
```yaml
# configs/base.yaml
api_endpoint: 'https://integrate.api.nvidia.com/v1'
training:
  device: 'cuda'
  batch_size: 32
```

Derived configuration:
```yaml
# configs/my_router.yaml
extends: 'configs/base.yaml'

router_params:
  k: 5
```

### Configuration Profiles

```yaml
# config.yaml
profiles:
  development:
    data_path:
      llm_data: 'data/dev/llms.json'
    training:
      epochs: 5

  production:
    data_path:
      llm_data: 'data/prod/llms.json'
    training:
      epochs: 50
```

**Usage:**
```bash
llmrouter train --config config.yaml --profile production
```

---

## Troubleshooting

### Issue: "Configuration file not found"

**Solution:** Use absolute paths or paths relative to project root:
```bash
llmrouter infer --config ./configs/my_config.yaml --query "test"
```

### Issue: "Invalid parameter"

**Solution:** Check router-specific parameters in documentation:
```bash
llmrouter list-routers --show-params knnrouter
```

### Issue: "API key not found"

**Solution:** Verify environment variable is set:
```bash
echo $NVIDIA_API_KEY
```

---

## Next Steps

<div class="grid cards" markdown>

-   :bar_chart:{ .lg .middle } __Explore Routers__

    ---

    Learn about different routing strategies

    [:octicons-arrow-right-24: Router Overview](../features/routers.md)

-   :material-school:{ .lg .middle } __Tutorials__

    ---

    Interactive guides for all features

    [:octicons-arrow-right-24: Browse Tutorials](../tutorials/index.md)

-   :wrench:{ .lg .middle } __Custom Routers__

    ---

    Build your own routing logic

    [:octicons-arrow-right-24: Custom Router Guide](../custom-routers/quick-guide.md)

</div>
