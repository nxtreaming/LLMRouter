import argparse
import os

from llmrouter.models.Automix.main_automix import load_config, train_and_evaluate


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    default_yaml = os.path.join(
        project_root, "configs", "model_config_test", "automix_config.yaml"
    )

    parser = argparse.ArgumentParser(
        description="Train the Automix router with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(project_root, yaml_path)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    print(f"📄 Using YAML file: {yaml_path}")
    config = load_config(yaml_path)
    print("✅ Configuration loaded successfully!")

    print("\n🚀 Starting Automix router training and evaluation...")
    results = train_and_evaluate(config)
    if results is None:
        raise RuntimeError("Automix training did not complete successfully.")
    print("\n✅ Automix router train_test completed successfully!")


if __name__ == "__main__":
    main()
