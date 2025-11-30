import argparse
import os
from llmrouter.models import RouterR1


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # /data/taofeng2/LLMRouter
    default_yaml = os.path.join(project_root,  "configs", "model_config_test", "router_r1.yaml")

    parser = argparse.ArgumentParser(
        description="Test the LargestLLM router with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="ulab-ai/Router-R1-Qwen2.5-3B-Instruct",
        choices=["ulab-ai/Router-R1-Qwen2.5-3B-Instruct", "ulab-ai/Router-R1-Qwen2.5-3B-Instruct-Alpha0.9", "ulab-ai/Router-R1-Llama-3.2-3B-Instruct", "ulab-ai/Router-R1-Llama-3.2-3B-Instruct-Alpha0.9"],
        help="Router-R1 HF Pre-trained Weight",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        required=True,
        help="NVIDIA NIM API BASE",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="NVIDIA NIM API KEY",
    )
    args = parser.parse_args()

    # Verify file existence
    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    # Initialize the router
    print(f"📄 Using YAML file: {args.yaml_path}")
    llm = RouterR1(args.yaml_path)
    print("✅ RouterR1 initialized successfully!")

    # Run inference
    result = llm.route_single({"query":"How are you"})
    print(result)


if __name__ == "__main__":
    main()
