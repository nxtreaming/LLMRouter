import os
import sys
import json
import yaml
import subprocess

import llmrouter
from llmrouter.data.data_generation import generate_query_data, save_query_data_jsonl
from llmrouter.data.generate_llm_embeddings import generate_llm_embeddings

SAMPLE_CONFIG_PATH = os.path.join(os.path.dirname(llmrouter.__file__), "data", "sample_config.yaml")

def get_default_paths():
    with open(SAMPLE_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    data_path = config["data_path"]
    project_root = os.path.dirname(os.path.dirname(llmrouter.__file__))
    llm_config = os.path.join(project_root, data_path["llm_data"])
    full_p = os.path.join(project_root, data_path["query_data_train"])
    output_dir = os.path.dirname(full_p)
    return llm_config, output_dir

DEFAULT_LLM_CONFIG, DEFAULT_OUTPUT_DIR = get_default_paths()

class DatasetSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "natural_qa": ("BOOLEAN", {"default": True}),
                "trivia_qa": ("BOOLEAN", {"default": True}),
                "mmlu": ("BOOLEAN", {"default": True}),
                "gpqa": ("BOOLEAN", {"default": True}),
                "mbpp": ("BOOLEAN", {"default": True}),
                "human_eval": ("BOOLEAN", {"default": True}),
                "gsm8k": ("BOOLEAN", {"default": True}),
                "commonsense_qa": ("BOOLEAN", {"default": True}),
                "math": ("BOOLEAN", {"default": True}),
                "openbook_qa": ("BOOLEAN", {"default": True}),
                "arc_challenge": ("BOOLEAN", {"default": True}),
                "geometry3k": ("BOOLEAN", {"default": True}),
                "mathvista": ("BOOLEAN", {"default": True}),
                "charades_ego_activity": ("BOOLEAN", {"default": False}),
                "charades_ego_object": ("BOOLEAN", {"default": False}),
                "charades_ego_verb": ("BOOLEAN", {"default": False}),
                "charades_ego_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("selected_datasets", "charades_ego_path")
    FUNCTION = "get_dataset_list"
    CATEGORY = "LLMRouter"

    def get_dataset_list(self, **kwargs):
        charades_path = kwargs.pop("charades_ego_path", "")
        selected = [k for k, v in kwargs.items() if v]
        return (",".join(selected), charades_path)


class LLMSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_config_path": ("STRING", {"default": DEFAULT_LLM_CONFIG}),
                "qwen2.5-7b-instruct": ("BOOLEAN", {"default": True}),
                "llama-3.1-8b-instruct": ("BOOLEAN", {"default": True}),
                "mistral-7b-instruct-v0.3": ("BOOLEAN", {"default": True}),
                "llama-3.3-nemotron-super-49b-v1": ("BOOLEAN", {"default": True}),
                "llama3-70b-instruct": ("BOOLEAN", {"default": True}),
                "mixtral-8x7b-instruct-v0.1": ("BOOLEAN", {"default": True}),
                "mixtral-8x22b-instruct-v0.1": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("llms",)
    FUNCTION = "select_llms"
    CATEGORY = "LLMRouter"

    def select_llms(self, llm_config_path, **kwargs):
        # Load the full config
        with open(llm_config_path, 'r') as f:
            data = json.load(f)
            
        selected_models = [k for k, v in kwargs.items() if v]
        filtered = {k: v for k, v in data.items() if k in selected_models}

        output_path = os.path.join(os.path.dirname(llm_config_path), "filtered_llm.json")
        with open(output_path, 'w') as f:
            json.dump(filtered, f, indent=2)
            
        return (output_path,)

class QueryDataGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_datasets": ("STRING", {"forceInput": True}),
                "charades_ego_path": ("STRING", {"forceInput": True}),
                "sample_size": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "output_dir": ("STRING", {"default": DEFAULT_OUTPUT_DIR}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("query_train_path", "query_test_path")
    FUNCTION = "generate"
    CATEGORY = "LLMRouter"

    def generate(self, selected_datasets, charades_ego_path, sample_size=10, output_dir=DEFAULT_OUTPUT_DIR):
        datasets = selected_datasets
        # charades_ego_path, sample_size, output_dir are already passed as arguments
        dataset_list = datasets.split(',')
        
        # Check if Charades datasets are selected but path is missing
        if any("charades_ego" in d for d in dataset_list) and not charades_ego_path:
            raise ValueError("❌ Charades-Ego datasets are selected but 'Charades Ego Path' is missing. Please provide the dataset path or uncheck the Charades tasks.")
        
        os.makedirs(output_dir, exist_ok=True)
        query_train = os.path.join(output_dir, "query_train.jsonl")
        query_test = os.path.join(output_dir, "query_test.jsonl")
        
        print(f"Generating query data for datasets: {dataset_list} with sample size {sample_size}")
        if charades_ego_path:
            print(f"Using Charades-Ego path: {charades_ego_path}")
        
        train_data, test_data = generate_query_data(
            sample_size=sample_size,
            datasets=dataset_list,
            charades_ego_path=charades_ego_path
        )
        
        save_query_data_jsonl(train_data, query_train)
        save_query_data_jsonl(test_data, query_test)
        
        return (query_train, query_test)

class LLMEmbeddingsGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llms": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("llm_embeddings_path",)
    FUNCTION = "generate"
    CATEGORY = "LLMRouter"

    def generate(self, llms):
        llm_data_path = llms
        
        with open(llm_data_path, 'r') as f:
            llm_data = json.load(f)
            
        output_path = llm_data_path.replace(".json", "_embeddings.json")
        generate_llm_embeddings(llm_data, output_path)
        
        return (output_path,)

class RouterPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "query_train": ("STRING", {"forceInput": True}),
                "query_test": ("STRING", {"forceInput": True}),
                "llms": ("STRING", {"forceInput": True}),
                "workers": ("INT", {"default": 10}),
                "output_dir": ("STRING", {"default": DEFAULT_OUTPUT_DIR}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("routing_train_path", "routing_test_path")
    FUNCTION = "run_pipeline"
    CATEGORY = "LLMRouter"
    OUTPUT_NODE = True

    def run_pipeline(self, query_train, query_test, llms, workers, output_dir):
        # Create a temp config
        os.makedirs(output_dir, exist_ok=True)
        config = {
            "data_path": {
                "query_data_train": query_train,
                "query_data_test": query_test,
                "llm_data": llms,
                "query_embedding_data": os.path.join(output_dir, "query_embeddings.pt"),
                "routing_data_train": os.path.join(output_dir, "routing_train.jsonl"),
                "routing_data_test": os.path.join(output_dir, "routing_test.jsonl"),
            }
        }
        
        config_path = os.path.join(output_dir, "run_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        # Call the script via subprocess
        script_path = os.path.join(os.path.dirname(llmrouter.__file__), "data", "api_calling_evaluation.py")
        cmd = [sys.executable, script_path, "--config", config_path, "--workers", str(workers)]
        
        # Capture output to print to ComfyUI console
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
        for line in process.stdout:
            print(line, end='')
        process.wait()

        return (config["data_path"]["routing_data_train"], config["data_path"]["routing_data_test"])

NODE_CLASS_MAPPINGS = {
    "DatasetSelector": DatasetSelector,
    "LLMSelector": LLMSelector,
    "QueryDataGenerator": QueryDataGenerator,
    "LLMEmbeddingsGenerator": LLMEmbeddingsGenerator,
    "RouterPipeline": RouterPipeline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DatasetSelector": "Start: Select Datasets",
    "LLMSelector": "Start: Select LLMs",
    "QueryDataGenerator": "Step 1: Generate Query Data",
    "LLMEmbeddingsGenerator": "Step 2: Generate LLM Embeddings",
    "RouterPipeline": "Step 3: Generate Routing Data"
}
