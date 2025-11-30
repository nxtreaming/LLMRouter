import argparse
import os
from llmrouter.models import LLMMultiRoundRouter


def main():
    # Note: LLMMultiRoundRouter does not require training (uses prompt-based routing)
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_yaml = os.path.join(project_root, "configs", "model_config_test", "llmmultiroundrouter.yaml")

    parser = argparse.ArgumentParser(
        description="Test the LLMMultiRoundRouter with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    args = parser.parse_args()

    # Verify file existence
    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    # Initialize the router
    print(f"📄 Using YAML file: {args.yaml_path}")
    router = LLMMultiRoundRouter(args.yaml_path)
    print("✅ LLMMultiRoundRouter initialized successfully!")

    # Run inference - routing only
    result = router.route_batch()
    print("Batch routing result:", result)
    result_ = router.route_single({"query": "How are you"})
    print("Single routing result:", result_)
    
    # Run full pipeline - decomposition+routing, execution, aggregation
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)
    test_query = "What is the capital of France and what is its population?"
    print(f"Query: {test_query}")
    
    # Option 1: Get just the final answer
    final_answer = router.answer_query(test_query)
    print(f"\nFinal Answer:\n{final_answer}")
    
    # Option 2: Get intermediate results
    full_result = router.answer_query(test_query, return_intermediate=True)
    print(f"\nSub-queries: {full_result['sub_queries']}")
    print(f"Routes: {full_result['routes']}")
    print(f"Number of sub-queries: {full_result['metadata']['num_sub_queries']}")
    print(f"Routing method: {full_result['metadata']['routing_method']}")



if __name__ == "__main__":
    main()

