"""
Automix Router - Complete Usage Example
========================================

This script demonstrates how to use the Automix router for complete training and inference workflows.

Usage:
    python main_automix.py [--config CONFIG_PATH]

Arguments:
    --config: Path to YAML configuration file (default: configs/model_config_train/automix.yaml)
"""

import os
import sys
import argparse
import pandas as pd
import yaml
import tempfile

# Add project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Running in: ", PROJECT_ROOT)


from llmrouter.models.automix import (
    AutomixRouter,
    AutomixRouterTrainer,
    AutomixModel,
    POMDP,
    Threshold,
    SelfConsistency,
)
from llmrouter.utils.data_convert import (
    convert_data,
    convert_train_data,
    merge_train_test,
)


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration file

    Args:
        config_path: Path to configuration file. If None, use default path

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(
            PROJECT_ROOT, "configs", "model_config_train", "automix.yaml"
        )
    elif not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Configuration file loaded: {config_path}")
    return config


def get_routing_method(method_name: str, num_bins: int):
    """
    Create routing method instance based on method name

    Args:
        method_name: Method name ("Threshold", "SelfConsistency", "POMDP")
        num_bins: Number of bins

    Returns:
        Routing method instance
    """
    method_map = {
        "Threshold": Threshold,
        "SelfConsistency": SelfConsistency,
        "POMDP": POMDP,
    }

    if method_name not in method_map:
        raise ValueError(
            f"Unknown routing method: {method_name}. "
            f"Available methods: {list(method_map.keys())}"
        )

    return method_map[method_name](num_bins=num_bins)


def convert_default_data_to_memory(config: dict) -> pd.DataFrame:
    """
    Convert default_data to required format (in-memory, no file output)

    Args:
        config: Configuration dictionary containing data_path settings

    Returns:
        Merged DataFrame with train and test data
    """
    data_cfg = config["data_path"]

    # Get input file paths
    test_input = data_cfg["routing_data_test"]
    train_input = data_cfg["routing_data_train"]

    # Handle relative paths
    if not os.path.isabs(test_input):
        test_input = os.path.join(PROJECT_ROOT, test_input)
    if not os.path.isabs(train_input):
        train_input = os.path.join(PROJECT_ROOT, train_input)

    # Load and process data in memory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_output = os.path.join(temp_dir, "test_data.jsonl")
        train_output = os.path.join(temp_dir, "train_data.json")
        merged_output = os.path.join(temp_dir, "merged_data.jsonl")

        # Convert test data to temporary file
        if os.path.exists(test_input):
            print(f"Converting test data from: {test_input}")
            convert_data(
                input_file=test_input,
                output_file=test_output,
                use_llm=False,
            )
        else:
            raise FileNotFoundError(f"Test data file not found: {test_input}")

        # Convert train data to temporary file
        if os.path.exists(train_input):
            print(f"Converting train data from: {train_input}")
            convert_train_data(
                input_file=train_input,
                output_file=train_output,
            )
        else:
            raise FileNotFoundError(f"Train data file not found: {train_input}")

        # Merge data to temporary file
        print(f"Merging train and test data in memory")
        merge_train_test(
            test_file=test_output,
            train_file=train_output,
            output_file=merged_output,
        )

        # Load merged data into memory
        merged_df = pd.read_json(merged_output, lines=True, orient="records")

    return merged_df


def train_and_evaluate(config: dict):
    """
    Train and evaluate using configuration

    Args:
        config: Configuration dictionary loaded from YAML file
    """
    hparam = config["hparam"]
    sep_width = 70

    print("=" * sep_width)
    print("Automix Router Training and Evaluation")
    print("=" * sep_width)

    print(f"\nStep 1: Load and prepare data")
    print("-" * sep_width)

    # Convert data in memory (no file output)
    try:
        df = convert_default_data_to_memory(config)
        print(f"Data loaded successfully! Dataset size: {len(df)}")
        print(f"Training set size: {len(df[df['split'] == 'train'])}")
        print(f"Test set size: {len(df[df['split'] == 'test'])}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure data files exist in the configured paths")
        return None

    print(f"\nStep 2: Create Automix router")
    print("-" * sep_width)

    # Create routing method from configuration
    method = get_routing_method(hparam["routing_method"], hparam["num_bins"])
    print(f"Routing method: {hparam['routing_method']} (num_bins={hparam['num_bins']})")

    # Create model
    model = AutomixModel(
        method=method,
        slm_column=hparam["slm_column"],
        llm_column=hparam["llm_column"],
        verifier_column=hparam["verifier_column"],
        costs=[hparam["small_model_cost"], hparam["large_model_cost"]],
        verifier_cost=hparam["verifier_cost"],
        verbose=hparam["verbose"],
    )
    print(
        f"Model configuration: Small model cost={hparam['small_model_cost']}, "
        f"Large model cost={hparam['large_model_cost']}, "
        f"Verifier cost={hparam['verifier_cost']}"
    )

    # Create router
    router = AutomixRouter(model=model)
    print("Router created successfully")

    print(f"\nStep 3: Train router")
    print("-" * sep_width)

    # Create trainer
    cost_constraint = hparam.get("cost_constraint", None)
    trainer = AutomixRouterTrainer(
        router=router,
        device=hparam["device"],
        cost_constraint=cost_constraint
    )

    # Split data
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Train and evaluate
    results = trainer.train_and_evaluate(train_df, test_df)

    print(f"\nStep 4: View results")
    print("-" * sep_width)

    print("\nTraining set results:")
    print(f"  Best parameter: {results['train']['best_param']}")
    print(f"  IBC Lift: {results['train']['metrics']['ibc_lift']:.4f}")
    print(f"  Average performance: {results['train']['metrics']['avg_performance']:.4f}")
    print(f"  Average cost: {results['train']['metrics']['avg_cost']:.2f}")

    print("\nTest set results:")
    print(f"  IBC Lift: {results['test']['ibc_lift']:.4f}")
    print(f"  Average performance: {results['test']['avg_performance']:.4f}")
    print(f"  Average cost: {results['test']['avg_cost']:.2f}")

    # Calculate routing statistics
    test_decisions = results["test"]["route_to_llm"]
    num_routed = int(test_decisions.sum())
    total = len(test_decisions)
    print(f"  Routed to large model: {num_routed}/{total} ({num_routed/total*100:.1f}%)")

    print(f"\nStep 5: Inference with trained router")
    print("-" * sep_width)

    # Select a few test samples for inference
    num_samples = hparam.get("num_inference_samples", 2)
    sample_data = test_df.head(num_samples)

    for idx, row in sample_data.iterrows():
        decision = router.model.infer(row)
        model_used = "Large model (70B)" if decision else "Small model (13B)"
        print(f"\nQuestion: {row['question'][:60]}...")
        print(f"  Verifier score: {row[hparam['verifier_column']]:.3f}")
        print(f"  Small model F1: {row[hparam['slm_column']]:.3f}")
        print(f"  Large model F1: {row[hparam['llm_column']]:.3f}")
        print(f"  Routing decision: {model_used}")

    return results

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Automix Router Training and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: configs/model_config_train/automix.yaml)",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the configuration file exists or use --config to specify the configuration file path")
        return

    sep_width = 70

    print("\n" + "=" * sep_width)
    print("Automix Router Training and Evaluation")
    print("=" * sep_width)

    try:
        results = train_and_evaluate(config)
        if results is None:
            print("\nTraining failed. Please check the error messages above.")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("Hint: Please ensure data files exist and configuration is correct")

    print("\n" + "=" * sep_width)
    print("Training completed!")
    print("=" * sep_width)


if __name__ == "__main__":
    main()
