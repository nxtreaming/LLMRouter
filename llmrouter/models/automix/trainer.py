"""
Automix Router Trainer
----------------------
Training implementation for AutomixRouter.

Original source: automix/colabs/Step3_MetaVerify.py
Adapted for LLMRouter framework.
"""

import torch
import pandas as pd
from typing import Any
from llmrouter.models.base_trainer import BaseTrainer


class AutomixRouterTrainer(BaseTrainer):
    """
    AutomixRouterTrainer
    -------------------
    Trainer implementation for AutomixRouter.

    Unlike typical neural network training with gradient descent,
    Automix training involves:
    1. Searching over candidate routing parameters
    2. Evaluating each on the training data
    3. Selecting the parameter with best IBC (Incremental Benefit over Cost) lift
    """

    def __init__(self, router, device: str = "cpu"):
        """
        Initialize AutomixRouterTrainer.

        Args:
            router: An AutomixRouter instance
            device (str): Device for computation (default: "cpu")
        """
        # Create a dummy optimizer for API compatibility
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([dummy_param], lr=1e-4)

        super().__init__(router=router, optimizer=optimizer, device=device)

        # Get config from router
        self.cfg = router.cfg
        self.train_df = router.train_df
        self.test_df = router.test_df

        # Get training parameters (support both train_param and hparam)
        hparam = self.cfg.get("hparam", {})

        # Try to get cost_constraint from train_param first, then hparam
        self.cost_constraint = hparam.get("cost_constraint", None)

        # Try to get verbose from train_param first, then hparam
        self.verbose = hparam.get("verbose", False)

        print("[AutomixRouterTrainer] Initialized successfully!")
        print(f"  Training samples: {len(self.train_df)}")
        print(f"  Test samples: {len(self.test_df)}")
        print(f"  Routing method: {hparam.get('routing_method', 'POMDP')}")

    def loss_func(self, outputs, batch) -> torch.Tensor:
        """
        Compute loss (not used for Automix).

        Automix uses discrete parameter search rather than gradient descent.

        Returns:
            torch.Tensor: Dummy loss tensor (always 0)
        """
        return torch.tensor(0.0, device=self.device)

    def train(self):
        """
        Train the AutomixRouter.

        For Automix, "training" means:
        1. Search over candidate parameters on training data
        2. Select best parameter based on IBC lift
        3. Evaluate on test data
        """
        print("\n" + "=" * 70)
        print("[AutomixRouterTrainer] Starting training (parameter search)")
        print("=" * 70)

        # Perform parameter search on training data
        print(f"\n[AutomixRouterTrainer] Training on {len(self.train_df)} samples...")
        best_param = self.router.model.train_routing(
            self.train_df,
            cost_constraint=self.cost_constraint
        )

        # Evaluate on training data
        train_metrics = self.router.model.evaluate(self.train_df, return_dict=True)

        print("\n[AutomixRouterTrainer] Training Results:")
        print(f"  Best parameter: {best_param}")
        print(f"  IBC Lift: {train_metrics['ibc_lift']:.4f}")
        print(f"  Avg Performance: {train_metrics['avg_performance']:.4f}")
        print(f"  Avg Cost: {train_metrics['avg_cost']:.2f}")

        # Evaluate on test data
        print(f"\n[AutomixRouterTrainer] Evaluating on {len(self.test_df)} samples...")
        test_metrics = self.router.model.evaluate(
            self.test_df,
            return_dict=True,
            return_decisions=True
        )

        decisions = test_metrics["route_to_llm"]
        num_routed = int(decisions.sum())
        total = len(self.test_df)

        print("\n[AutomixRouterTrainer] Test Results:")
        print(f"  IBC Lift: {test_metrics['ibc_lift']:.4f}")
        print(f"  Avg Performance: {test_metrics['avg_performance']:.4f}")
        print(f"  Avg Cost: {test_metrics['avg_cost']:.2f}")
        print(f"  Routed to LLM: {num_routed}/{total} ({num_routed/total*100:.1f}%)")

        print("\n" + "=" * 70)
        print("[AutomixRouterTrainer] Training complete!")
        print("=" * 70)

        return {
            "train": {
                "best_param": best_param,
                "metrics": train_metrics,
            },
            "test": test_metrics,
        }
