"""
Automix Router
--------------
Automix router that conforms to the LLMRouter MetaRouter interface.

Original source: automix/colabs/
Adapted for LLMRouter framework.
"""

import os
import re
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
        self.train_df, self.test_df = self._prepare_data()
        print(f"[AutomixRouter] Loaded {len(self.train_df)} training samples, {len(self.test_df)} test samples")

        # Create routing method
        hparam = self.cfg["hparam"]
        method = self._build_method(hparam["routing_method"], hparam["num_bins"])
        print(f"[AutomixRouter] Using routing method: {hparam['routing_method']}")

        # Extract column names (support both flat and nested structure)
        if "columns" in hparam:
            slm_column = hparam["columns"]["slm"]
            llm_column = hparam["columns"]["llm"]
            verifier_column = hparam["columns"]["verifier"]
        else:
            slm_column = hparam.get("slm_column", "llama13b_f1")
            llm_column = hparam.get("llm_column", "llama70b_f1")
            verifier_column = hparam.get("verifier_column", "p_ver_13b")

        # Extract costs (support both flat and nested structure)
        if "costs" in hparam:
            small_model_cost = hparam["costs"]["small_model"]
            large_model_cost = hparam["costs"]["large_model"]
            verifier_cost = hparam["costs"]["verifier"]
        else:
            small_model_cost = hparam.get("small_model_cost", 1)
            large_model_cost = hparam.get("large_model_cost", 50)
            verifier_cost = hparam.get("verifier_cost", 1)

        # Extract verbose setting
        verbose = hparam.get("verbose", self.cfg.get("train_param", {}).get("verbose", False))

        # Create AutomixModel
        model = AutomixModel(
            method=method,
            slm_column=slm_column,
            llm_column=llm_column,
            verifier_column=verifier_column,
            costs=[small_model_cost, large_model_cost],
            verifier_cost=verifier_cost,
            verbose=verbose,
        )
        print("[AutomixRouter] AutomixModel created successfully!")

        # Initialize parent class
        super().__init__(model=model, yaml_path=yaml_path)

    def _prepare_data(self):
        """
        Prepare and preprocess data using intermediate variables.

        Returns:
            tuple: (train_df, test_df) - Training and test DataFrames with all required columns
        """
        data_cfg = self.cfg["data_path"]
        hparam = self.cfg["hparam"]

        # Get paths for routing data
        train_path = data_cfg.get("routing_data_train")
        test_path = data_cfg.get("routing_data_test")

        if not train_path or not test_path:
            raise ValueError("Config must specify 'routing_data_train' and 'routing_data_test' in data_path section")

        # Resolve to absolute paths
        if not os.path.isabs(train_path):
            train_path = os.path.join(self.project_root, train_path)
        if not os.path.isabs(test_path):
            test_path = os.path.join(self.project_root, test_path)

        # Check if files exist
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")

        print(f"  Loading training data from: {train_path}")
        print(f"  Loading test data from: {test_path}")

        # Load data into memory
        train_df = pd.read_json(train_path, lines=True, orient="records")
        test_df = pd.read_json(test_path, lines=True, orient="records")

        # Add split column if not present
        if "split" not in train_df.columns:
            train_df["split"] = "train"
        if "split" not in test_df.columns:
            test_df["split"] = "test"

        # Determine ground truth column
        if "gt" in train_df.columns and "gt" in test_df.columns:
            ground_truth_col = "gt"
        elif "ground_truth" in train_df.columns and "ground_truth" in test_df.columns:
            ground_truth_col = "ground_truth"
        else:
            # Fallback to the first available candidate
            candidates = ["gt", "ground_truth", "answer", "output"]
            ground_truth_col = next((c for c in candidates if c in train_df.columns), "gt")

        # Get required column names
        slm_column = hparam.get("slm_column", "llama13b_f1")
        llm_column = hparam.get("llm_column", "llama70b_f1")
        verifier_column = hparam.get("verifier_column", "p_ver_13b")
        required_cols = [slm_column, llm_column, verifier_column]

        # Check if data already has required columns
        train_has_cols = all(col in train_df.columns for col in required_cols)
        test_has_cols = all(col in test_df.columns for col in required_cols)

        if train_has_cols and test_has_cols:
            print(f"  Data validation passed. Required columns present: {required_cols}")
            return train_df, test_df

        # If not, need to generate predictions and verification scores
        print("  Required columns not found. Starting data preparation pipeline...")
        print("  This involves calling LLM APIs to generate:")
        print(f"    - llama13b_pred_ans, llama70b_pred_ans (model predictions)")
        print(f"    - {slm_column}, {llm_column} (F1 scores)")
        print(f"    - {verifier_column} (verifier confidence)")

        # Merge train and test into a single DataFrame for processing
        merged_df = pd.concat([test_df, train_df], ignore_index=True)

        # Get model engine names from config
        engine_small = hparam.get("engine_small", "meta/llama-3.1-8b-instruct")
        engine_large = hparam.get("engine_large", "meta/llama-3.1-70b-instruct")

        # Run data preparation pipeline using prepare_automix_data logic
        # This will call LLM APIs and add required columns
        print("\n  Running data preparation pipeline...")

        # Import data pipeline function
        from .data_pipeline import init_providers, run_solver_job, prepare_row
        from .data_pipeline import run_verification, compute_fraction_correct
        from .data_pipeline import clean_answer, calculate_f1_for_models, categorize_rows

        # Initialize API providers
        init_providers()

        # Step 1: Solve queries with small and large models
        print("\n  Step 1: Solving queries with small and large models...")
        # Get max_workers from config (default to 5 for faster processing)
        max_workers = self.cfg.get('hparam', {}).get('max_workers', 5)
        print(f"  Using {max_workers} parallel workers for API calls...")
        results_13b = run_solver_job(merged_df, prepare_row, engine_small, max_workers=max_workers)
        results_70b = run_solver_job(merged_df, prepare_row, engine_large, max_workers=max_workers)

        # Clean answers - handle None values properly
        def safe_clean_answer(ans):
            """Safely clean answer, handling None and empty values."""
            if ans is None:
                return None
            if isinstance(ans, (float, type(pd.NA))) and pd.isna(ans):
                return None
            ans_str = str(ans).strip()
            if not ans_str:
                return None
            # Remove quotes but keep the answer
            return ans_str.replace("'", "").replace('"', '')
        
        merged_df["llama13b_pred_ans"] = [safe_clean_answer(ans) for ans in results_13b]
        merged_df["llama70b_pred_ans"] = [safe_clean_answer(ans) for ans in results_70b]
        
        # Debug: check a few predictions
        # print(f"  Sample predictions (first 3):")
        # for i in range(min(3, len(merged_df))):
        #     print(f"    [{i}] 13b: {repr(merged_df.iloc[i]['llama13b_pred_ans'])}, 70b: {repr(merged_df.iloc[i]['llama70b_pred_ans'])}")

        # Check if data is multiple choice (ground_truth is a single letter like "A", "B", "C", "D")
        # or if metric column indicates multiple choice
        is_multi_choice = False
        if "metric" in merged_df.columns:
            is_multi_choice = merged_df["metric"].str.contains("mc|multiple_choice", case=False, na=False).any()
        elif ground_truth_col in merged_df.columns:
            # Check if ground_truth values are single letters (A-D)
            sample_gt = merged_df[ground_truth_col].dropna().head(10)
            if len(sample_gt) > 0:
                is_multi_choice = sample_gt.astype(str).str.strip().str.match(r'^[A-D]$', na=False).any()

        # Calculate F1 scores
        model_sizes = ["13b", "70b"]
        
        if is_multi_choice:
            # For multiple choice: extract option letter from prediction and compare with ground_truth
            print("  Detected multiple choice format. Using option-based F1 calculation...")
            
            def extract_option_from_pred(pred) -> str:
                """Extract option letter (A, B, C, D) from prediction string."""
                # Handle None, NaN, or empty values
                if pred is None:
                    return None
                if isinstance(pred, float) and pd.isna(pred):
                    return None
                try:
                    if pd.isna(pred):
                        return None
                except:
                    pass
                
                pred_str = str(pred).strip()
                if not pred_str:
                    return None
                
                # Method 1: Look for option letter at the very start (most common: "B", "B.", "B. labels")
                # This handles: "B", "B.", "B. labels", "B. something else"
                if pred_str and len(pred_str) > 0:
                    first_char = pred_str[0].upper()
                    if first_char in ["A", "B", "C", "D", "E", "F"]:
                        # Check if it's followed by punctuation or space (not part of a word)
                        if len(pred_str) == 1 or pred_str[1] in [".", " ", ")", ":", ",", "\n", "\t"]:
                            return first_char
                
                # Method 2: Look for pattern like "Answer: B", "The answer is C", etc.
                # Match patterns: "Answer: B", "answer is C", "option B", etc.
                patterns = [
                    r'(?:answer|option|choice|select)[\s:]+([A-F])',  # "Answer: B" or "answer is C"
                    r'\(([A-F])\)',  # "(B)"
                    r'[\.\s]([A-F])[\.\)\s]',  # ".B." or " B " or " B)"
                    r'\b([A-F])\b',  # Standalone "B" (word boundary)
                ]
                for pattern in patterns:
                    match = re.search(pattern, pred_str, re.IGNORECASE)
                    if match:
                        return match.group(1).upper()
                
                # Method 3: Try to extract number and convert to option letter
                # Handle cases like "2" -> "B" (if ground_truth is option letter format)
                # Extract first number from string
                num_match = re.search(r'\b([1-4])\b', pred_str)
                if num_match:
                    num = int(num_match.group(1))
                    # Convert 1->A, 2->B, 3->C, 4->D
                    option_letter = chr(ord('A') + num - 1)
                    return option_letter
                
                return None
            
            for size in model_sizes:
                pred_col = f"llama{size}_pred_ans"
                f1_col = f"llama{size}_f1"
                
                # Extract option from prediction
                merged_df[f"llama{size}_pred_option"] = merged_df[pred_col].apply(extract_option_from_pred)
                
                # Debug: print some examples before comparison
                if len(merged_df) > 0:
                    sample_size = min(5, len(merged_df))
                    print(f"  Sample predictions for llama{size} (before comparison):")
                    for idx in range(sample_size):
                        pred_val = merged_df.iloc[idx][pred_col]
                        extracted = merged_df.iloc[idx][f"llama{size}_pred_option"]
                        gt_val = merged_df.iloc[idx].get(ground_truth_col, "N/A")
                        match = "✓" if (extracted is not None and str(gt_val).strip().upper() == str(extracted).strip().upper()) else "✗"
                        print(f"    [{idx}] {match} pred={repr(pred_val)}, extracted={repr(extracted)}, gt={repr(gt_val)}")
                
                # Compare extracted option with ground_truth
                merged_df[f1_col] = merged_df.apply(
                    lambda r: 1.0 if (
                        r.get(f"llama{size}_pred_option") is not None 
                        and str(r.get(ground_truth_col, "")).strip().upper() == str(r.get(f"llama{size}_pred_option")).strip().upper()
                    ) else 0.0,
                    axis=1
                )
                
                # Debug: print F1 statistics
                total = len(merged_df)
                correct = merged_df[f1_col].sum()
                # extracted_count = merged_df[f"llama{size}_pred_option"].notna().sum()
                # print(f"  llama{size} F1 statistics:")
                # print(f"    Total samples: {total}")
                # print(f"    Successfully extracted options: {extracted_count}")
                # print(f"    Correct matches: {correct}")
                # print(f"    F1 score: {correct/total*100:.1f}%")
                
                # Warning if all F1 scores are 0
                if correct == 0 and total > 0:
                    print(f"    ⚠️  WARNING: All F1 scores are 0 for llama{size}!")
                    print(f"       This may indicate:")
                    print(f"       1. Model predictions don't match ground truth format")
                    print(f"       2. Option extraction logic needs improvement")

        else:
            # For text-based answers: use standard F1 calculation
            merged_df = calculate_f1_for_models(
                merged_df, model_sizes, ground_truth_col=ground_truth_col
            )

        print(f"  Mean F1 scores - llama13b: {merged_df['llama13b_f1'].mean():.4f}")
        print(f"  Mean F1 scores - llama70b: {merged_df['llama70b_f1'].mean():.4f}")

        # Step 2: Self-verification
        print("\n  Step 2: Self-verification and categorization...")
        # Use same max_workers for verification
        ver_results = run_verification(
            merged_df,
            ans_col="llama13b_pred_ans",
            engine_name=engine_small,
            temperature=1.0,
            n=2,
            stop="---",
            max_tokens=250,
            max_workers=max_workers,
        )

        merged_df["llama13b_ver"] = ver_results
        merged_df["p_ver_13b"] = merged_df["llama13b_ver"].apply(compute_fraction_correct)

        # Categorize rows
        merged_df = categorize_rows(merged_df)

        print("  Data preparation completed!")

        # Split back into train and test
        train_df = merged_df[merged_df["split"] == "train"].copy().reset_index(drop=True)
        test_df = merged_df[merged_df["split"] == "test"].copy().reset_index(drop=True)

        # Verify required columns are present
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Training data missing required column after processing: {col}")
            if col not in test_df.columns:
                raise ValueError(f"Test data missing required column after processing: {col}")

        return train_df, test_df

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
