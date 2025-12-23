# GMTRouter - Graph-based Multi-Turn Personalized Router

## Overview

GMTRouter is a graph neural network-based router designed specifically for multi-turn conversations with personalization capabilities. It leverages user interaction history and preferences to make routing decisions that adapt to individual users over time.

## Paper References

- **Graph Neural Networks**: Kipf & Welling (2017) - "Semi-Supervised Classification with Graph Convolutional Networks"
- **Personalized Recommendation**: Hamilton et al. (2017) - "Inductive Representation Learning on Large Graphs"
- **Multi-Turn Dialogue**: Serban et al. (2016) - "Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models"

## How It Works

GMTRouter employs a graph-based approach to model the relationships between:
1. **Queries** and **LLM candidates** (query-model performance)
2. **Users** and **LLM preferences** (personalization)
3. **Conversation history** (multi-turn context)

### Key Components

#### 1. Graph Construction
- Creates a heterogeneous graph with query, model, and user nodes
- Edges represent:
  - Query-Model: Historical performance relationships
  - User-Model: User preference patterns
  - Query-Query: Semantic similarity in conversation history

#### 2. Graph Neural Network
- Multi-layer GNN processes the graph structure
- Aggregates information from neighbors using message passing
- Learns representations for queries, models, and users

#### 3. Personalization Layer
- Maintains user-specific embeddings
- Updates based on interaction history
- Considers context window of recent conversations

#### 4. Routing Decision
- Combines current query embedding with:
  - User preference embedding
  - Conversation history context
- Predicts best model for current query-user pair

### Architecture Flow

```
User Query + User ID + Conversation History
          ↓
   Query Embedding
          ↓
   Graph Construction (Query-Model-User)
          ↓
   Graph Neural Network
          ↓
   Personalization Layer
          ↓
   Model Selection (Personalized)
```

## Configuration Parameters

### Training Parameters

Located in `configs/model_config_train/gmtrouter.yaml`:

#### GMTRouter-Specific Configuration (`gmt_config`)

- **`personalization`** (bool, default: `true`)
  - Enable user preference learning
  - When `true`, maintains user-specific embeddings
  - When `false`, falls back to non-personalized routing

- **`context_window`** (int, default: `5`)
  - Number of previous conversation turns to consider
  - Larger values = more conversation context
  - Range: 1-20 (typically 3-10 works best)

- **`hidden_dim`** (int, default: `128`)
  - Hidden layer dimension for GNN
  - Controls model capacity
  - Range: 64-512 (larger = more expressive but slower)

- **`num_layers`** (int, default: `3`)
  - Number of GNN layers
  - More layers = larger receptive field
  - Range: 2-5 (too many can cause over-smoothing)

- **`dropout`** (float, default: `0.1`)
  - Dropout rate for regularization
  - Prevents overfitting
  - Range: 0.0-0.5

- **`user_embedding_dim`** (int, default: `64`)
  - Dimension of user preference embeddings
  - Controls personalization capacity
  - Range: 32-256

- **`update_user_embeddings`** (bool, default: `true`)
  - Whether to update user embeddings during training
  - `true` for training, `false` for testing

- **`num_neighbors`** (int, default: `10`)
  - Number of neighbors for graph construction
  - Controls graph density
  - Range: 5-50

- **`edge_threshold`** (float, default: `0.5`)
  - Similarity threshold for edge creation
  - Higher = sparser graph
  - Range: 0.0-1.0

#### General Training Parameters (`hparam`)

- **`learning_rate`** (float, default: `0.001`)
  - Learning rate for optimizer
  - Range: 0.0001-0.01
  - Lower for stable convergence, higher for faster training

- **`weight_decay`** (float, default: `0.0001`)
  - L2 regularization weight
  - Prevents overfitting
  - Range: 0.0-0.01

- **`train_epoch`** (int, default: `100`)
  - Maximum number of training epochs
  - Training may stop earlier with early stopping

- **`batch_size`** (int, default: `32`)
  - Number of samples per gradient update
  - Larger = more stable gradients but more memory
  - Range: 8-128

- **`patience`** (int, default: `10`)
  - Early stopping patience (epochs)
  - Stops if validation doesn't improve for this many epochs

- **`val_split_ratio`** (float, default: `0.2`)
  - Validation set size (20% of training data)
  - Range: 0.1-0.3

- **`max_history_length`** (int, default: `10`)
  - Maximum conversation history to store per user
  - Older turns are discarded
  - Range: 5-50

- **`history_decay`** (float, default: `0.9`)
  - Decay factor for older conversation turns
  - Exponential decay: older turns have less weight
  - Range: 0.7-1.0 (1.0 = no decay)

### Testing Parameters

Located in `configs/model_config_test/gmtrouter.yaml`:

Most parameters match training config for model compatibility. Key differences:
- `update_user_embeddings`: Set to `false` (don't modify during inference)
- Training-specific parameters (learning_rate, etc.) are present but unused

## Usage

### Training

```python
from llmrouter.models.gmtrouter import GMTRouter, GMTRouterTrainer

# Initialize router with training config
router = GMTRouter(yaml_path='configs/model_config_train/gmtrouter.yaml')

# Create trainer
trainer = GMTRouterTrainer(router=router)

# Train the model
results = trainer.train()
print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
```

### Inference (Single Query)

```python
from llmrouter.models.gmtrouter import GMTRouter

# Initialize router with test config
router = GMTRouter(yaml_path='configs/model_config_test/gmtrouter.yaml')

# Route a query with user context
query = {
    "query_text": "Explain quantum computing in simple terms",
    "user_id": "user_123",  # Important for personalization
    "task": "qa"
}

# Get routing decision
result = router.route_single(query)
print(f"Selected model: {result['model_name']}")
```

### Multi-Turn Conversation

```python
# Conversation with history tracking
user_id = "user_456"

queries = [
    "What is machine learning?",
    "How is it different from deep learning?",
    "Can you give me a practical example?"
]

for i, query_text in enumerate(queries):
    query = {
        "query_text": query_text,
        "user_id": user_id,
        "task": "qa"
    }

    result = router.route_single(query)
    print(f"Turn {i+1}: {result['model_name']}")

    # Conversation history is automatically tracked
    # Each turn influences future routing decisions
```

### Batch Routing

```python
# Route multiple queries at once
batch = [
    {"query_text": "Calculate 123 * 456", "user_id": "user_789", "task": "math"},
    {"query_text": "Write a poem about AI", "user_id": "user_789", "task": "creative"},
    {"query_text": "Debug this Python code", "user_id": "user_789", "task": "code"}
]

results = router.route_batch(batch, task_name="mixed")
for query, result in zip(batch, results):
    print(f"{query['task']}: {result['model_name']}")
```

## YAML Configuration Example

### Training Configuration

```yaml
# configs/model_config_train/gmtrouter.yaml
gmt_config:
  personalization: true
  context_window: 5
  hidden_dim: 128
  num_layers: 3
  dropout: 0.1
  user_embedding_dim: 64

hparam:
  learning_rate: 0.001
  train_epoch: 100
  batch_size: 32
  patience: 10
```

### Testing Configuration

```yaml
# configs/model_config_test/gmtrouter.yaml
gmt_config:
  personalization: true
  context_window: 5
  hidden_dim: 128
  num_layers: 3
  update_user_embeddings: false  # Don't update during inference

hparam:
  random_state: 42
```

## Advantages

1. **Personalization**: Adapts to individual user preferences over time
2. **Multi-Turn Awareness**: Considers conversation history for context-aware routing
3. **Graph-Based**: Captures complex relationships between queries, models, and users
4. **Scalable**: Efficient graph operations handle large-scale routing
5. **Adaptive**: User embeddings evolve with interaction patterns
6. **Contextual**: Uses conversation windows for coherent multi-turn decisions

## Limitations

1. **Cold Start**: New users without history get generic routing initially
2. **Memory**: Storing user histories and embeddings requires additional memory
3. **Complexity**: More complex than simple routers (MLPRouter, KNNRouter)
4. **Training Data**: Requires user-labeled interaction data for best results
5. **Privacy**: User tracking may raise privacy concerns in some applications

## When to Use GMTRouter

### Ideal Use Cases

- **Chatbot Applications**: Multi-turn conversations with persistent users
- **Personalized Assistants**: Systems that adapt to individual user preferences
- **Long-Term Interactions**: Applications with returning users over time
- **Context-Dependent Tasks**: Queries that build on previous conversation

### Not Recommended When

- **Single-Turn Only**: No conversation history to leverage
- **Anonymous Users**: Cannot build user profiles without identification
- **Cold Start Critical**: Need instant optimal performance for new users
- **Simple Tasks**: Overhead not justified for straightforward routing

## Comparison with Other Routers

| Feature | GMTRouter | GraphRouter | KNNMultiRoundRouter |
|---------|-----------|-------------|---------------------|
| Personalization | ✅ | ❌ | ❌ |
| Multi-Turn | ✅ | ❌ | ✅ |
| Graph-Based | ✅ | ✅ | ❌ |
| User Embeddings | ✅ | ❌ | ❌ |
| Training Required | ✅ | ✅ | ✅ |
| Cold Start Performance | ❌ | ✅ | ⚠️ |

## Performance Tips

1. **Context Window**: Start with 5, adjust based on conversation patterns
2. **User Embeddings**: Larger `user_embedding_dim` for diverse user bases
3. **GNN Layers**: 3 layers usually optimal; more can cause over-smoothing
4. **History Length**: Balance between context and memory usage
5. **Warm-Up Period**: Performance improves as user history accumulates
6. **Batch Processing**: Use `route_batch()` for efficiency when possible

## Technical Requirements

- **Python**: 3.11+
- **PyTorch**: 2.6+
- **PyTorch Geometric**: 2.6.1+
- **Additional**: Graph construction utilities, user tracking system

## Example Output

```python
>>> result = router.route_single({
...     "query_text": "Solve this math problem",
...     "user_id": "user_123",
...     "task": "math"
... })
>>> print(result)
{
    'model_name': 'gpt-4',
    'confidence': 0.87,
    'user_preference': 0.92,
    'conversation_context': 0.81,
    'routing_time': 0.023
}
```

## References

For more information on graph neural networks and personalization:
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [GNN Tutorial](https://distill.pub/2021/gnn-intro/)
- [Original GMTRouter Repository](https://github.com/ulab-uiuc/GMTRouter)
