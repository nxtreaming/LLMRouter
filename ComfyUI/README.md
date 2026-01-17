# 🎨 ComfyUI Interface for LLMRouter

This directory contains the [ComfyUI](https://github.com/Comfy-Org/ComfyUI) custom nodes for **LLMRouter**, allowing you to visually construct and execute the data generation and routing pipeline.

By using ComfyUI, you can:
- **Visually configure** datasets and LLM candidates.
- **Automate** the entire pipeline: query generation -> embedding generation -> routing data creation.
- **Monitor** progress through a graphical interface.

## 🛠️ Installation & Setup

Prerequisites: You must have [ComfyUI](https://github.com/Comfy-Org/ComfyUI) installed.

To install the LLMRouter custom nodes, you need to create two symbolic links (soft links).

### 1. Link the Custom Nodes
This allows ComfyUI to load the LLMRouter Python backend logic in the ComfyUI "Nodes" category.

```bash
ln -s /path/to/LLMRouter/ComfyUI /path/to/ComfyUI/custom_nodes/LLMRouter
```

### 2. Link the Workflow Template (Optional)
This allows you to see the pre-configured workflow in the ComfyUI "Workflows" category.

```bash
ln -s /path/to/LLMRouter/ComfyUI/workflows/llm_router_template.json /path/to/ComfyUI/user/default/workflows/llm_router_template.json
```

## 🚀 How to Use

### Finding the Nodes
To use the nodes:
1.  Open the ComfyUI web interface in your browser.
2.  Navigate to the **`Nodes`** category.
3.  You will see nodes like `Start: Select Dataset`, `Start: Select LLMs`, `Step 1: Generate Query Data`, etc.

### Loading the Template
To use the template:
1.  Open the ComfyUI web interface in your browser.
2.  Click on **"Workflows"**.
3.  Select **`llm_router_template.json`**.

This will load a complete workflow connected and ready to run.