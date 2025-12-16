"""
DCRouter Router
---------------
Router implementation for the DCRouter routing strategy.

This module provides the DCRouter class that integrates with the
LLMRouter framework.

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework.
"""

import os
import yaml
import copy
import torch
from transformers import AutoTokenizer, DebertaV2Model
from llmrouter.models.meta_router import MetaRouter
from .dcmodel import RouterModule
from .dcdataset import DCDataset
from .dcdata_utils import preprocess_data


class DCRouter(MetaRouter):
    """
    DCRouter
    --------
    Router that uses dual-contrastive learning strategy for LLM routing decisions.

    DCRouter uses a pre-trained encoder (e.g., mDeBERTa) combined with learnable
    LLM embeddings to make routing decisions. The model is trained with three
    contrastive learning objectives:
    1. Sample-LLM contrastive loss
    2. Sample-Sample contrastive loss (task-level)
    3. Cluster contrastive loss
    """

    def __init__(self, yaml_path: str):
        """
        Initialize DCRouter.

        Args:
            yaml_path (str): Path to YAML config file
        """
        # Load configuration
        with open(yaml_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Resolve project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Prepare data
        self._prepare_data()

        # Initialize tokenizer and backbone
        backbone_model = self.cfg['model_path']['backbone_model']
        print(f"[DCRouter] Loading backbone model: {backbone_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone_model,
            truncation_side='left',
            padding=True
        )
        encoder_model = DebertaV2Model.from_pretrained(backbone_model)
        print("[DCRouter] Backbone model loaded successfully!")

        # Load datasets
        print(f"[DCRouter] Loading datasets...")
        hparam = self.cfg['hparam']
        self.train_dataset = DCDataset(
            data=self.train_data_processed,
            source_max_token_len=hparam.get('source_max_token_len', 512),
            target_max_token_len=hparam.get('target_max_token_len', 512),
            dataset_id=0
        )
        self.train_dataset.register_tokenizer(self.tokenizer)

        self.test_dataset = DCDataset(
            data=self.test_data_processed,
            source_max_token_len=hparam.get('source_max_token_len', 512),
            target_max_token_len=hparam.get('target_max_token_len', 512),
            dataset_id=1
        )
        self.test_dataset.register_tokenizer(self.tokenizer)

        num_llms = len(self.train_dataset.router_node)
        print(f"[DCRouter] Datasets loaded:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Number of LLMs: {num_llms}")
        print(f"  LLM names: {self.train_dataset.router_node}")

        # Create RouterModule
        model = RouterModule(
            backbone=encoder_model,
            hidden_state_dim=hparam['hidden_state_dim'],
            node_size=num_llms,
            similarity_function=hparam['similarity_function']
        )
        print("[DCRouter] RouterModule created successfully!")

        # Save cfg before calling super().__init__() since it will be reset
        saved_cfg = self.cfg
        
        # Initialize parent class (pass None to avoid duplicate data loading)
        # DCRouter handles its own data loading, so we pass yaml_path=None
        super().__init__(model=model, yaml_path=None)
        
        # Restore cfg since MetaRouter.__init__ resets it to {}
        self.cfg = saved_cfg
        
        # Load metric weights if provided
        weights_dict = self.cfg.get("metric", {}).get("weights", {})
        self.metric_weights = list(weights_dict.values())

    def _prepare_data(self):
        """Prepare and preprocess data."""
        data_path_config = self.cfg['data_path']
        train_data_raw = os.path.join(self.project_root, data_path_config['routing_data_train'])
        test_data_raw = os.path.join(self.project_root, data_path_config['routing_data_test'])

        print("\n[DCRouter] Starting data preprocessing...")

        hparam = self.cfg['hparam']
        n_clusters = hparam.get('n_clusters', 3)
        max_test_samples = hparam.get('max_test_samples', 500)

        # Preprocess training data
        # print("\n[DCRouter] Preprocessing training data...")
        self.train_data_processed = preprocess_data(
            input_path=train_data_raw,
            add_cluster_id=True,
            n_clusters=n_clusters,
            max_samples=None
        )

        # Preprocess test data
        # print("\n[DCRouter] Preprocessing test data...")
        self.test_data_processed = preprocess_data(
            input_path=test_data_raw,
            add_cluster_id=False,
            n_clusters=n_clusters,
            max_samples=max_test_samples
        )

        print("[DCRouter] Data preprocessing completed!\n")

    def route(self, batch):
        """
        Perform routing on a batch of data.

        Args:
            batch (dict): A batch containing tokenized inputs

        Returns:
            dict: A dictionary with routing outputs
        """
        # Extract temperature if provided, default to 1.0
        temperature = batch.get("temperature", 1.0)

        # Prepare inputs for the model
        input_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        # Forward pass through RouterModule
        scores, hidden_state = self.model(t=temperature, **input_kwargs)

        # Get predicted LLM indices (argmax)
        predictions = torch.argmax(scores, dim=1)

        return {
            "scores": scores,
            "hidden_state": hidden_state,
            "predictions": predictions,
        }

    def forward(self, batch):
        """
        PyTorch-compatible forward method.
        
        This delegates to route() for compatibility with nn.Module.
        
        Args:
            batch (dict): A batch containing tokenized inputs
            
        Returns:
            dict: A dictionary with routing outputs
        """
        return self.route(batch)

    def route_batch(self):
        """
        Route a batch of data from the test dataset.

        Returns:
            dict: Routing results
        """
        from torch.utils.data import DataLoader

        # Load model if exists
        hparam = self.cfg['hparam']
        device = hparam.get('device', 'cpu')

        # Try to load checkpoint
        save_dir = os.path.join(self.project_root, os.path.dirname(self.cfg['model_path']['save_model_path']))
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(save_dir, 'best_training_model.pth')

        if os.path.exists(checkpoint_path):
            print(f"[DCRouter] Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"[DCRouter] Warning: No checkpoint found. Using untrained model.")

        self.model = self.model.to(device)
        self.model.eval()

        # Run inference
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=hparam.get('inference_batch_size', 64),
            shuffle=False
        )

        query_data_output = []

        with torch.no_grad():
            sample_idx = 0
            for batch_data in test_dataloader:
                inputs, scores, _, _ = batch_data
                inputs = {k: v.to(device) for k, v in inputs.items()}
                scores = scores.to(device)

                batch = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "temperature": hparam.get('inference_temperature', 1.0),
                }

                outputs = self.route(batch)
                predictions = outputs["predictions"]

                # Process each sample in the batch
                for i in range(len(predictions)):
                    if sample_idx < len(self.test_dataset.data):
                        predicted_llm_idx = predictions[i].item()
                        predicted_llm = self.test_dataset.router_node[predicted_llm_idx]
                        query_text = self.test_dataset.data[sample_idx]['question']
                        query_data_output.append({
                            "query": query_text,
                            "model_name": predicted_llm
                        })
                        sample_idx += 1

        return query_data_output

    def route_single(self, data):
        """
        Route a single query.

        Args:
            data (dict): Query data with 'query' key

        Returns:
            dict: Routing result
        """
        hparam = self.cfg['hparam']
        device = hparam.get('device', 'cpu')

        # Load model if exists
        save_dir = os.path.join(self.project_root, os.path.dirname(self.cfg['model_path']['save_model_path']))
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(save_dir, 'best_training_model.pth')

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(device)
        self.model.eval()

        # Tokenize query
        query_text = data["query"]
        query_tokens = self.tokenizer(
            query_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        batch = {
            "input_ids": query_tokens["input_ids"],
            "attention_mask": query_tokens["attention_mask"],
            "temperature": hparam.get('inference_temperature', 1.0),
        }

        with torch.no_grad():
            outputs = self.route(batch)

        predicted_llm_idx = outputs["predictions"][0].item()
        predicted_llm = self.test_dataset.router_node[predicted_llm_idx]

        query_output = copy.copy(data)
        query_output["model_name"] = predicted_llm
        return query_output
