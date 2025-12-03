"""
Automix Router
--------------
Automix router that conforms to the LLMRouter MetaRouter interface.

Original source: automix/colabs/
Adapted for LLMRouter framework.
"""

import os
import yaml
import pandas as pd
from typing import Any, Dict

import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter
from .model import AutomixModel
from .methods import Threshold, POMDP, SelfConsistency
from llmrouter.utils.data_convert import convert_data, convert_train_data, merge_train_test
from .data_pipeline import prepare_automix_data


class AutomixRouter(MetaRouter):
    """
    AutomixRouter
    -------------
    Router that uses self-verification to decide when to route queries
    from a small language model to a larger, more capable model.

    Key features:
    - Cost-effective routing based on verification confidence
    - Multiple routing methods: Threshold, POMDP, SelfConsistency
    - Automatic data preparation and preprocessing
    """

    def __init__(self, yaml_path: str):
        """
        Initialize AutomixRouter.

        Args:
            yaml_path (str): Path to YAML config file
        """
        # Load configuration
        with open(yaml_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Resolve project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Prepare data
        print("[AutomixRouter] Preparing data...")
        self.data_path = self._prepare_data()

        # Load data
        print("[AutomixRouter] Loading data...")
        self.df = pd.read_json(self.data_path, lines=True, orient="records")
        self.train_df = self.df[self.df["split"] == "train"].copy()
        self.test_df = self.df[self.df["split"] == "test"].copy()
        print(f"[AutomixRouter] Loaded {len(self.train_df)} training samples, {len(self.test_df)} test samples")

        # Create routing method
        hparam = self.cfg["hparam"]
        method = self._build_method(hparam["routing_method"], hparam["num_bins"])
        print(f"[AutomixRouter] Using routing method: {hparam['routing_method']}")

        # Create AutomixModel
        model = AutomixModel(
            method=method,
            slm_column=hparam["columns"]["slm"],
            llm_column=hparam["columns"]["llm"],
            verifier_column=hparam["columns"]["verifier"],
            costs=[hparam["costs"]["small_model"], hparam["costs"]["large_model"]],
            verifier_cost=hparam["costs"]["verifier"],
            verbose=self.cfg.get("train_param", {}).get("verbose", False),
        )
        print("[AutomixRouter] AutomixModel created successfully!")

        # Initialize parent class
        super().__init__(model=model, yaml_path=yaml_path)

    def _prepare_data(self) -> str:
        """
        Prepare and preprocess data if needed.

        Returns prepared data path with all required columns:
        - llama13b_f1, llama70b_f1 (F1 scores)
        - p_ver_13b (verifier confidence)
        """
        data_cfg = self.cfg["data_path"]
        prepared_path = data_cfg.get("prepared_data", "data/automix/router_automix_llamapair_ver_outputs.jsonl")

        if not os.path.isabs(prepared_path):
            prepared_path = os.path.join(self.project_root, prepared_path)

        # Check if data already exists
        if os.path.exists(prepared_path):
            print(f"  Using existing prepared data: {prepared_path}")
            return prepared_path

        # Need to prepare data - requires API calls
        print("  Prepared data not found.")
        print("  AutomixRouter requires data with LLM predictions and verifier scores.")
        print("  This involves calling LLM APIs to generate:")
        print("    - llama13b_pred_ans, llama70b_pred_ans (model predictions)")
        print("    - llama13b_f1, llama70b_f1 (F1 scores)")
        print("    - p_ver_13b (verifier confidence)")

        conv_cfg = data_cfg.get("conversion", {})
        output_dir = data_cfg.get("output_dir", "data/automix")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.project_root, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        merged_file = conv_cfg.get("merged_file", "train_test_nq_split.jsonl")
        merged_path = os.path.join(output_dir, merged_file)

        # Convert default data if needed
        if not os.path.exists(merged_path):
            print(f"\n  Step 1: Converting default data to {merged_path}...")
            default_data_dir = conv_cfg.get("default_data_dir", "data/default_data")
            if not os.path.isabs(default_data_dir):
                default_data_dir = os.path.join(self.project_root, default_data_dir)

            # Set up paths
            input_train = os.path.join(default_data_dir, conv_cfg.get("train_file", "default_routing_train_data.jsonl"))
            input_test = os.path.join(default_data_dir, conv_cfg.get("test_file", "default_routing_test_data.jsonl"))
            output_train = os.path.join(output_dir, conv_cfg.get("train_output_file", "router_train_data_nq.json"))
            output_test = os.path.join(output_dir, conv_cfg.get("test_output_file", "router_test_data_nq.jsonl"))

            # Convert test data
            if os.path.exists(input_test):
                print(f"    Converting test data...")
                convert_data(input_file=input_test, output_file=output_test, use_llm=False)

            # Convert train data
            if os.path.exists(input_train):
                print(f"    Converting train data...")
                convert_train_data(input_file=input_train, output_file=output_train)

            # Merge train and test
            print(f"    Merging train and test data...")
            merge_train_test(
                train_file=output_train,
                test_file=output_test,
                output_file=merged_path
            )
            print(f"  Data conversion completed: {merged_path}")

        # Now run data preparation pipeline (requires API)
        print(f"\n  Step 2: Running data preparation pipeline (calling LLM APIs)...")
        print(f"  Input: {merged_path}")
        print(f"  Output: {prepared_path}")

        model_cfg = self.cfg.get("model_path", {})
        engine_small = model_cfg.get("engine_small", "meta/llama-3.1-8b-instruct")
        engine_large = model_cfg.get("engine_large", "meta/llama-3.1-70b-instruct")

        df = prepare_automix_data(
            input_data_path=merged_path,
            output_dir=output_dir,
            engine_small=engine_small,
            engine_large=engine_large
        )

        print(f"  Data preparation completed: {prepared_path}\n")
        return prepared_path

    @staticmethod
    def _build_method(name: str, num_bins: int):
        """Build routing method from name."""
        mapping = {
            "POMDP": POMDP,
            "Threshold": Threshold,
            "SelfConsistency": SelfConsistency,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported routing method: {name}. Available: {list(mapping.keys())}")
        return mapping[name](num_bins=num_bins)

    # ------------------------------------------------------------------
    # MetaRouter interface
    # ------------------------------------------------------------------
    def route_batch(self, batch: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route a batch of queries.

        Args:
            batch (dict, optional): Batch data. If None, uses test_df.

        Returns:
            dict: Routing results with decisions, performance, and cost
        """
        if batch is None:
            # Use test data
            data = self.test_df
        else:
            data = self._prepare_batch(batch)

        # Run inference
        batch_dict = {"data": data, "mode": "infer"}
        outputs = self.model(batch_dict)

        return {
            "decisions": outputs["decisions"],
            "performance": outputs["performance"],
            "cost": outputs["cost"],
            "total": len(data),
            "num_routed": int(outputs["decisions"].sum().item()),
        }

    def route_single(self, sample: Any) -> Dict[str, Any]:
        """
        Route a single query.

        Args:
            sample: Single sample (dict, Series, or DataFrame row)

        Returns:
            dict: Routing result with decision, performance, and cost
        """
        single_batch = {"data": self._to_dataframe(sample), "mode": "infer"}
        outputs = self.model(single_batch)
        decision = bool(outputs["decisions"].item())

        return {
            "decision": decision,
            "route_to_llm": decision,
            "performance": outputs["performance"],
            "cost": outputs["cost"],
        }

    def route(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for route_batch for backward compatibility."""
        return self.route_batch(batch)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_metrics(self, outputs, batch) -> dict:
        """Compute routing metrics."""
        decisions = outputs["decisions"]
        num_routed = int(decisions.sum().item())
        total_samples = len(decisions)

        metrics = {
            "avg_performance": outputs["performance"],
            "avg_cost": outputs["cost"],
            "routing_percentage": (num_routed / total_samples * 100.0) if total_samples > 0 else 0.0,
            "num_routed": num_routed,
            "total_samples": total_samples,
        }

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_batch(batch: Dict[str, Any]) -> pd.DataFrame:
        """Prepare batch data."""
        if not isinstance(batch, dict):
            raise TypeError("AutomixRouter expects batch to be a dict.")
        if "data" not in batch:
            raise KeyError("Batch must contain a 'data' key with a pandas DataFrame.")

        return AutomixRouter._to_dataframe(batch["data"])

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame:
        """Convert data to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data.reset_index(drop=True)
        if isinstance(data, pd.Series):
            return data.to_frame().T.reset_index(drop=True)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        raise TypeError(
            "AutomixRouter expects pandas DataFrame/Series or dict as input data."
        )
