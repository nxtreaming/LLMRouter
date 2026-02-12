# 🎨 ComfyUI Interface for LLMRouter

This directory contains the [ComfyUI](https://github.com/Comfy-Org/ComfyUI) custom nodes for **LLMRouter**, allowing you to visually construct and execute the data generation and routing pipeline.

**Why ComfyUI?** A shift from command-line to fully visual:

- **No More CLI Commands**: Forget complex terminal scripts.
- **No More Config Files**: Stop searching for `config.yaml`. All parameters are right on the nodes.
- **Everything is Visual**: Configure and execute via **Nodes and Connections**:
  - **Visually Configure**: Select datasets (e.g., MMLU, GSM8K) and LLMs via toggle switches directly on the canvas.
  - **End-to-End Pipeline**: Visually connect Query Generation $\to$ API Inference $\to$ Embeddings.
  - **Instant Feedback**: Train routers (KNN, SVM, MLP) and watch performance metrics update in real-time.

<div align="center">
  <img src="assets/comfyui.png" alt="LLMRouter Example in ComfyUI" width="800">
</div>

## 🛠️ Installation & Setup

Prerequisites: You must have [ComfyUI](https://github.com/Comfy-Org/ComfyUI) installed.

To install the LLMRouter custom nodes, you need to create two symbolic links (soft links).

### 1. Link the Custom Nodes
This allows ComfyUI to load the LLMRouter Python backend logic in the ComfyUI "Nodes" category.

```bash
ln -s /path/to/LLMRouter/ComfyUI /path/to/ComfyUI/custom_nodes/LLMRouter
```

### 2. Link the Workflow Example (Optional)
This allows you to see the pre-configured workflow in the ComfyUI "Workflows" category.

```bash
ln -s /path/to/LLMRouter/ComfyUI/workflows/llm_router_example.json /path/to/ComfyUI/user/default/workflows/llm_router_example.json
```

### 3. Running the Application

To start the ComfyUI server with the LLMRouter nodes:

```bash
python /path/to/ComfyUI/main.py
```

### 4. Remote Access & Port Forwarding

If you are running ComfyUI on a remote server (e.g., a compute cluster) and wish to access the interface locally, you can use SSH tunneling. Once the tunnel is established, access the interface at `http://127.0.0.1:8188`.


## 🎮 How to Use

### Find the Nodes
To use the nodes:
1.  Open the ComfyUI web interface.
2.  Use the **Node Library** sidebar or **Right-click** on the canvas.
3.  Navigate to the **`LLMRouter`** category.
4.  You will find nodes organized by function:
    - **Data**: `Select Datasets`, `Select LLMs`, `Generate Data`.
    - **Single-Round**: `KNN Router`, `SVM Router`, `MLP Router`, etc.
    - **Multi-Round / Agentic**: Specialized routers for complex tasks.

### Load the Example
To use the ready-to-run example:
1.  Click the **`Workflows`** tab (if using a modern UI) or use the **"Load"** button.
2.  Select **`llm_router_example`**.
3.  This loads a complete pipeline connecting **Dataset Selection** $\to$ **Data Generation** $\to$ **Router Training** $\to$ **Evaluation**.