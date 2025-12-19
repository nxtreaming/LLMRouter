# Installation

This guide will help you install LLMRouter and set up your environment.

## Requirements

### System Requirements

- **Python**: 3.10 or higher
- **OS**: Linux, macOS, or Windows (with WSL recommended)
- **GPU**: Optional but recommended for training (CUDA 11.8+ compatible)

### Dependencies

LLMRouter automatically installs the following key dependencies:

- **PyTorch** 2.0+ - Deep learning framework
- **Transformers** 4.40+ - Hugging Face models
- **LiteLLM** 1.50+ - Unified LLM API interface
- **scikit-learn** - Machine learning utilities
- **torch-geometric** - Graph neural networks (for GraphRouter)
- **Gradio** 4.0+ - Chat interface UI

---

## Installation Methods

### Method 1: Install from Source (Recommended)

This method gives you access to the latest features and allows easy customization.

```bash
# Clone the repository
git clone https://github.com/ulab-uiuc/LLMRouter.git
cd LLMRouter

# Create a virtual environment
conda create -n llmrouter python=3.10
conda activate llmrouter

# Install in editable mode
pip install -e .
```

!!! tip "Why `-e` flag?"
    The `-e` (editable) flag allows you to modify the source code and see changes immediately without reinstalling.

---

### Method 2: Install with pip (Coming Soon)

```bash
# This will be available once published to PyPI
pip install llmrouter
```

!!! warning "Not Yet Available"
    PyPI distribution is planned for a future release. Use Method 1 for now.

---

## Verify Installation

After installation, verify that LLMRouter is correctly installed:

### Check CLI Access

```bash
llmrouter --help
```

**Expected Output:**
```
Usage: llmrouter [OPTIONS] COMMAND [ARGS]...

  LLMRouter CLI - Intelligent routing for Large Language Models

Commands:
  train          Train a router model
  infer          Run inference with a router
  chat           Launch interactive chat interface
  list-routers   List all available routers
```

### Check Python Import

```python
import llmrouter
print(llmrouter.__version__)
```

### Run a Quick Test

```bash
# List available routers
llmrouter list-routers

# Run a simple inference (requires API key)
llmrouter infer \
  --router knnrouter \
  --config configs/model_config_test/knnrouter.yaml \
  --query "What is machine learning?" \
  --route-only
```

The `--route-only` flag skips actual LLM API calls and only tests the routing logic.

---

## API Key Setup

LLMRouter uses LiteLLM to interface with various LLM providers. You'll need API keys for the models you want to use.

### Supported Providers

- **OpenAI** - GPT-4, GPT-3.5
- **Anthropic** - Claude models
- **Google** - Gemini models
- **Meta** - Llama models (via various providers)
- **Mistral AI** - Mistral models
- **Cohere** - Command models
- **NVIDIA** - NIM endpoints

### Setting API Keys

#### Option 1: Environment Variables

```bash
# Add to your ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

#### Option 2: Configuration File

Create a `.env` file in the LLMRouter directory:

```bash
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key
```

#### Option 3: In Code

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
```

!!! danger "Security Warning"
    Never commit API keys to version control. Add `.env` to your `.gitignore` file.

---

## GPU Setup (Optional)

### Check CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Install CUDA-enabled PyTorch

If CUDA is not detected, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'llmrouter'`

**Solution:** Make sure you've activated the correct conda environment:
```bash
conda activate llmrouter
```

#### Issue: `command not found: llmrouter`

**Solution:** The CLI might not be in your PATH. Try:
```bash
python -m llmrouter.cli.main --help
```

Or reinstall in editable mode:
```bash
pip install -e .
```

#### Issue: CUDA out of memory

**Solution:** Reduce batch size or use CPU:
```bash
llmrouter train --router mlprouter --config config.yaml --device cpu
```

#### Issue: `ImportError: torch_geometric`

**Solution:** Install torch-geometric separately:
```bash
pip install torch-geometric
```

#### Issue: API rate limits

**Solution:** LLMRouter respects rate limits. For development, use `--route-only` to skip API calls:
```bash
llmrouter infer --router knnrouter --config config.yaml --query "test" --route-only
```

---

## Development Installation

For contributors who want to modify LLMRouter:

```bash
# Clone and install in development mode
git clone https://github.com/ulab-uiuc/LLMRouter.git
cd LLMRouter

# Create environment
conda create -n llmrouter-dev python=3.10
conda activate llmrouter-dev

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Updating LLMRouter

### From Source Installation

```bash
cd LLMRouter
git pull origin main
pip install -e .
```

### From pip (Future)

```bash
pip install --upgrade llmrouter
```

---

## Uninstallation

```bash
pip uninstall llmrouter
```

To also remove the conda environment:
```bash
conda deactivate
conda env remove -n llmrouter
```

---

## Next Steps

<div class="grid cards" markdown>

-   :rocket:{ .lg .middle } __Quick Start__

    ---

    Run your first inference in 5 minutes

    [:octicons-arrow-right-24: Quick Start Guide](quick-start.md)

-   :material-cog:{ .lg .middle } __Configuration__

    ---

    Learn about configuration files

    [:octicons-arrow-right-24: Configuration Guide](configuration.md)

-   :material-school:{ .lg .middle } __Tutorials__

    ---

    Interactive Colab notebooks

    [:octicons-arrow-right-24: Browse Tutorials](../tutorials/index.md)

</div>

---

## Getting Help

If you encounter any issues during installation:

- :fontawesome-brands-github: [GitHub Issues](https://github.com/ulab-uiuc/LLMRouter/issues)
- :fontawesome-brands-slack: [Slack Community](https://join.slack.com/t/llmrouteropen-ri04588/shared_invite/zt-3jz3cc6d1-ncwKEHvvWe0OczHx7K5c0g)
- :material-file-document: [Documentation](../index.md)
