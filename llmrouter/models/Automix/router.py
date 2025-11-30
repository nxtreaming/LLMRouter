"""
Automix Router
--------------
Automix router that conforms to the latest LLMRouter MetaRouter interface.
"""

from typing import Any, Dict

import pandas as pd
import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter


class AutomixRouter(MetaRouter):
    """
    Router wrapper around AutomixModel.

    The router consumes pandas DataFrames containing Automix features and
    produces routing decisions, costs, and performance statistics.
    """

    def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None):
        super().__init__(model=model, yaml_path=yaml_path, resources=resources)

    # ------------------------------------------------------------------
    # MetaRouter interface
    # ------------------------------------------------------------------
    def route_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        validated = self._prepare_batch(batch)
        return self.model(validated)

    def route_single(self, sample: Any) -> Dict[str, Any]:
        single_batch = {"data": self._to_dataframe(sample), "mode": "infer"}
        outputs = self.model(single_batch)
        decision = bool(outputs["decisions"].item())
        return {
            "decision": decision,
            "performance": outputs["performance"],
            "cost": outputs["cost"],
        }

    # Keep backward compatibility with older MetaRouter.forward()
    def route(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.route_batch(batch)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_metrics(self, outputs, batch) -> dict:
        decisions = outputs["decisions"]
        num_routed = int(decisions.sum().item())
        total_samples = len(decisions)

        metrics = {
            "avg_performance": outputs["performance"],
            "avg_cost": outputs["cost"],
            "routing_percentage": (num_routed / total_samples * 100.0)
            if total_samples > 0
            else 0.0,
            "num_routed": num_routed,
            "total_samples": total_samples,
        }

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError("AutomixRouter expects batch to be a dict.")
        if "data" not in batch:
            raise KeyError("Batch must contain a 'data' key with a pandas DataFrame.")

        df = AutomixRouter._to_dataframe(batch["data"])
        mode = batch.get("mode", "infer")
        return {"data": df, "mode": mode}

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.reset_index(drop=True)
        if isinstance(data, pd.Series):
            return data.to_frame().T.reset_index(drop=True)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        raise TypeError(
            "AutomixRouter expects pandas DataFrame/Series or dict as input data."
        )
