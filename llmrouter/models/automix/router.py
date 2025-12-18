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
import json
import pandas as pd
from typing import Any, Dict, Tuple

import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter
from .model import AutomixModel
from .methods import Threshold, POMDP, SelfConsistency
from .data_pipeline import prepare_automix_data


def parse_size(size_str: str) -> float:
    """
    Parse a model size string (e.g., '7B', '13B', '512M') into
    a numeric value in billions.

    Supported suffixes:
        - K: thousands
        - M: millions
        - B: billions
        - T: trillions

    If parsing fails, this function returns 0.0.
    """
    size_str = str(size_str).strip().upper()
    try:
        if size_str.endswith("K"):
            return float(size_str[:-1]) / 1e6
        elif size_str.endswith("M"):
            return float(size_str[:-1]) / 1e3
        elif size_str.endswith("B"):
            return float(size_str[:-1])
        elif size_str.endswith("T"):
            return float(size_str[:-1]) * 1e3
        else:
            return float(size_str)
    except Exception:
        return 0.0


def detect_small_large_models(llm_data_path: str) -> Tuple[str, str]:
    """
    Automatically detect the smallest and largest models from llm_data file.

    Args:
        llm_data_path: Path to the LLM data JSON file

    Returns:
        Tuple of (smallest_model_engine, largest_model_engine)

    Raises:
        ValueError: If llm_data is empty or no valid models found
    """
    with open(llm_data_path, 'r') as f:
        llm_data = json.load(f)

    if not llm_data:
        raise ValueError("llm_data is empty, cannot detect models")

    # Parse sizes and find smallest and largest
    model_info = []
    for name, info in llm_data.items():
        if isinstance(info.get("size", ""), str) and info["size"].strip():
            size = parse_size(info["size"])
            if size > 0:
                engine = info.get("model", name)
                model_info.append((name, engine, size))

    if not model_info:
        raise ValueError("No models with valid size information found in llm_data")

    # Sort by size
    model_info.sort(key=lambda x: x[2])

    smallest_name, smallest_engine, smallest_size = model_info[0]
    largest_name, largest_engine, largest_size = model_info[-1]

    print(f"✅ Auto-detected small model: {smallest_name} ({smallest_size}B) -> {smallest_engine}")
    print(f"✅ Auto-detected large model: {largest_name} ({largest_size}B) -> {largest_engine}")

    return smallest_engine, largest_engine


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

        # Auto-detect small and large models
        data_cfg = self.cfg["data_path"]
        llm_data_path = data_cfg.get("llm_data")
        if not os.path.isabs(llm_data_path):
            llm_data_path = os.path.join(self.project_root, llm_data_path)

        self.engine_small, self.engine_large = detect_small_large_models(llm_data_path)

        # Use generic column names for small/large models
        self.slm_column = "slm_f1"
        self.llm_column = "llm_f1"
        self.verifier_column = "p_ver_slm"

        # Check if inference mode (skip data preprocessing)
        hparam = self.cfg["hparam"]
        inference_mode = hparam.get("inference_mode", False)

        if inference_mode:
            # Inference mode: skip data loading and preprocessing
            print("✅ Inference mode enabled - skipping data preprocessing")
            self.train_df = pd.DataFrame()
            self.test_df = pd.DataFrame()
        else:
            # Training mode: load and prepare data
            self.train_df, self.test_df = self._prepare_data()

        # Create routing method
        method = self._build_method(hparam["routing_method"], hparam["num_bins"])

        # Extract costs
        small_model_cost = hparam.get("small_model_cost", 1)
        large_model_cost = hparam.get("large_model_cost", 50)
        verifier_cost = hparam.get("verifier_cost", 1)

        # Extract verbose setting
        verbose = hparam.get("verbose", self.cfg.get("train_param", {}).get("verbose", False))

        # Create AutomixModel
        model = AutomixModel(
            method=method,
            slm_column=self.slm_column,
            llm_column=self.llm_column,
            verifier_column=self.verifier_column,
            costs=[small_model_cost, large_model_cost],
            verifier_cost=verifier_cost,
            verbose=verbose,
        )

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

        # Load data into memory
        train_df = pd.read_json(train_path, lines=True, orient="records")
        test_df = pd.read_json(test_path, lines=True, orient="records")
        
        # Apply min_samples limit for quick testing
        # min_samples = hparam.get('min_samples')
        # if min_samples:
        #     train_df = train_df.head(min_samples)
        #     test_df = test_df.head(min_samples)

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

        # Use generic column names
        required_cols = [self.slm_column, self.llm_column, self.verifier_column]

        # Check if data already has required columns
        train_has_cols = all(col in train_df.columns for col in required_cols)
        test_has_cols = all(col in test_df.columns for col in required_cols)

        if train_has_cols and test_has_cols:
            return train_df, test_df

        # If not, need to generate predictions and verification scores

        # Merge train and test into a single DataFrame for processing
        merged_df = pd.concat([test_df, train_df], ignore_index=True)

        # Use auto-detected model engines
        engine_small = self.engine_small
        engine_large = self.engine_large

        # Run data preparation pipeline using prepare_automix_data logic
        # This will call LLM APIs and add required columns

        # Import data pipeline function
        from .data_pipeline import init_providers, run_solver_job, prepare_row
        from .data_pipeline import run_verification, compute_fraction_correct
        from .data_pipeline import clean_answer, calculate_f1_for_models, categorize_rows

        # Initialize API providers
        init_providers()

        # Step 1: Solve queries with small and large models
        # Get max_workers from config (default to 5 for faster processing)
        max_workers = self.cfg.get('hparam', {}).get('max_workers', 5)
        results_slm = run_solver_job(merged_df, prepare_row, engine_small, max_workers=max_workers)
        results_llm = run_solver_job(merged_df, prepare_row, engine_large, max_workers=max_workers)

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

        merged_df["slm_pred_ans"] = [safe_clean_answer(ans) for ans in results_slm]
        merged_df["llm_pred_ans"] = [safe_clean_answer(ans) for ans in results_llm]
        
        # Debug: check a few predictions
        # print(f"  Sample predictions (first 3):")
        # for i in range(min(3, len(merged_df))):
        #     print(f"    [{i}] slm: {repr(merged_df.iloc[i]['slm_pred_ans'])}, llm: {repr(merged_df.iloc[i]['llm_pred_ans'])}")

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
        model_names = ["slm", "llm"]

        if is_multi_choice:
            # For multiple choice: extract option letter from prediction and compare with ground_truth

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

            for model_name in model_names:
                pred_col = f"{model_name}_pred_ans"
                f1_col = f"{model_name}_f1"

                # Extract option from prediction
                merged_df[f"{model_name}_pred_option"] = merged_df[pred_col].apply(extract_option_from_pred)

                # Compare extracted option with ground_truth
                merged_df[f1_col] = merged_df.apply(
                    lambda r: 1.0 if (
                        r.get(f"{model_name}_pred_option") is not None
                        and str(r.get(ground_truth_col, "")).strip().upper() == str(r.get(f"{model_name}_pred_option")).strip().upper()
                    ) else 0.0,
                    axis=1
                )

                # Debug: print F1 statistics
                total = len(merged_df)
                correct = merged_df[f1_col].sum()
                # extracted_count = merged_df[f"{model_name}_pred_option"].notna().sum()
                # print(f"  {model_name} F1 statistics:")
                # print(f"    Total samples: {total}")
                # print(f"    Successfully extracted options: {extracted_count}")
                # print(f"    Correct matches: {correct}")
                # print(f"    F1 score: {correct/total*100:.1f}%")

                # Warning if all F1 scores are 0
                if correct == 0 and total > 0:
                    pass  # Silent warning

        else:
            # For text-based answers: use standard F1 calculation
            merged_df = calculate_f1_for_models(
                merged_df, model_names, ground_truth_col=ground_truth_col
            )

        # Step 2: Self-verification
        # Use same max_workers for verification
        ver_results = run_verification(
            merged_df,
            ans_col="slm_pred_ans",
            engine_name=engine_small,
            temperature=1.0,
            n=2,
            stop="---",
            max_tokens=250,
            max_workers=max_workers,
        )

        merged_df["slm_ver"] = ver_results
        merged_df[self.verifier_column] = merged_df["slm_ver"].apply(compute_fraction_correct)

        # Categorize rows
        merged_df = categorize_rows(merged_df, slm_column=self.slm_column, llm_column=self.llm_column)

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
        Route a batch of queries with full Automix inference pipeline.

        Process for each query:
        1. Call small model to get answer
        2. Perform self-verification
        3. Based on verification score, decide whether to call large model
        4. Return final answers and routing decisions

        Args:
            batch (dict, optional): Batch data. Should contain 'queries' list.
                                   If None, uses test_df.

        Returns:
            dict: Routing results with model_names, responses, and routing decisions
        """
        from .data_pipeline import prepare_row, run_solver_job, run_verification, compute_fraction_correct

        # Handle input
        if batch is None:
            # Use test data if available
            if hasattr(self, 'test_df') and not self.test_df.empty:
                query_df = self.test_df.copy()
            else:
                raise ValueError("No batch provided and no test data available")
        elif isinstance(batch, dict):
            if 'queries' in batch:
                # List of query strings
                query_df = pd.DataFrame([{'query': q} for q in batch['queries']])
            elif 'data' in batch:
                # DataFrame already provided
                query_df = batch['data'] if isinstance(batch['data'], pd.DataFrame) else pd.DataFrame(batch['data'])
            else:
                raise ValueError("Batch must contain 'queries' list or 'data' DataFrame")
        else:
            raise ValueError("Batch must be a dict or None")

        # Step 1: Call small model for all queries
        results_slm = run_solver_job(
            query_df, prepare_row, self.engine_small,
            max_workers=self.cfg.get('hparam', {}).get('max_workers', 5)
        )
        query_df['slm_pred_ans'] = results_slm

        # Step 2: Self-verification on small model answers
        ver_results = run_verification(
            query_df,
            ans_col='slm_pred_ans',
            engine_name=self.engine_small,
            temperature=1.0,
            n=2,
            stop="---",
            max_tokens=250,
            max_workers=self.cfg.get('hparam', {}).get('max_workers', 5),
        )

        query_df['slm_ver'] = ver_results
        query_df[self.verifier_column] = query_df['slm_ver'].apply(compute_fraction_correct)

        # Step 3: Make routing decisions
        decision_df = query_df[[self.verifier_column]].copy()
        decision_df[self.slm_column] = 0.0  # Placeholder
        decision_df[self.llm_column] = 0.0  # Placeholder

        batch_dict = {"data": decision_df, "mode": "infer"}
        outputs = self.model(batch_dict)
        decisions = outputs["decisions"].cpu().numpy()

        # Step 4: Call large model for queries that need it
        queries_needing_llm = query_df[decisions]
        if len(queries_needing_llm) > 0:
            results_llm = run_solver_job(
                queries_needing_llm, prepare_row, self.engine_large,
                max_workers=self.cfg.get('hparam', {}).get('max_workers', 5)
            )
            query_df.loc[decisions, 'llm_pred_ans'] = results_llm

        # Step 5: Select final answers and model names
        query_df['final_answer'] = query_df.apply(
            lambda row: row.get('llm_pred_ans', row['slm_pred_ans'])
            if decisions[row.name] else row['slm_pred_ans'],
            axis=1
        )
        query_df['model_name'] = [
            self.engine_large if dec else self.engine_small
            for dec in decisions
        ]

        return {
            "queries": query_df['query'].tolist(),
            "model_names": query_df['model_name'].tolist(),
            "responses": query_df['final_answer'].tolist(),
            "decisions": decisions.tolist(),
            "verification_scores": query_df[self.verifier_column].tolist(),
            "slm_answers": query_df['slm_pred_ans'].tolist(),
            "total": len(query_df),
            "num_routed_to_llm": int(decisions.sum()),
        }

    def route_single(self, sample: Any) -> Dict[str, Any]:
        """
        Route a single query with full Automix inference pipeline.

        Process:
        1. Call small model to get answer
        2. Perform self-verification
        3. Based on verification score, decide whether to call large model
        4. Return final answer and routing decision

        Args:
            sample: Single sample (dict with 'query' field)

        Returns:
            dict: Routing result with model_name, response, and routing decision
        """
        from .data_pipeline import prepare_row, run_solver_job, run_verification, compute_fraction_correct

        # Extract query
        if isinstance(sample, dict) and 'query' in sample:
            query = sample['query']
        else:
            raise ValueError("sample must be a dict with 'query' field")

        # Create a minimal dataframe for processing
        query_df = pd.DataFrame([{'query': query}])

        # Step 1: Call small model
        results_slm = run_solver_job(query_df, prepare_row, self.engine_small, max_workers=1)
        slm_answer = results_slm[0] if results_slm else None

        # Add answer to dataframe for verification
        query_df['slm_pred_ans'] = slm_answer

        # Step 2: Self-verification on small model answer
        ver_results = run_verification(
            query_df,
            ans_col='slm_pred_ans',
            engine_name=self.engine_small,
            temperature=1.0,
            n=2,
            stop="---",
            max_tokens=250,
            max_workers=1,
        )

        verification_score = compute_fraction_correct(ver_results[0])

        # Step 3: Make routing decision based on verification score
        decision_df = pd.DataFrame([{self.verifier_column: verification_score}])
        decision_df[self.slm_column] = 0.0  # Placeholder
        decision_df[self.llm_column] = 0.0  # Placeholder

        batch_dict = {"data": decision_df, "mode": "infer"}
        outputs = self.model(batch_dict)
        route_to_llm = bool(outputs["decisions"].item())

        # Step 4: If routing to large model, call it
        if route_to_llm:
            results_llm = run_solver_job(query_df, prepare_row, self.engine_large, max_workers=1)
            final_answer = results_llm[0] if results_llm else slm_answer
            model_name = self.engine_large
        else:
            final_answer = slm_answer
            model_name = self.engine_small

        return {
            "model_name": model_name,
            "response": final_answer,
            "route_to_llm": route_to_llm,
            "verification_score": verification_score,
            "slm_answer": slm_answer,
            "query": query,
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
