from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.preprocessing import MinMaxScaler

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import get_longformer_embedding
from graph_nn import FormData, GNNPredictor


class GraphRouter(MetaRouter):
    """
    GraphRouter: A routing module using Graph Neural Networks (GNN) to select
    the most suitable LLM based on query embeddings and LLM relationships.

    Graph Structure:
    - Query nodes: each query is a node with embedding features
    - LLM nodes: each LLM is a node with embedding features
    - Edges: connect each query to all LLMs, weighted by performance score
    """

    def __init__(self, yaml_path: str):
        """
        Initialize GraphRouter.

        Args:
            yaml_path: Path to YAML configuration file
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get hyperparameters from config
        self.gnn_params = self.cfg.get("hparam", {})
        self.hidden_dim = self.gnn_params.get("hidden_dim", 64)

        # Prepare training data
        self._prepare_training_data()

        # Prepare LLM embeddings
        self._prepare_llm_embeddings()

        # Initialize GNN config
        self.gnn_config = {
            'learning_rate': self.gnn_params.get('learning_rate', 0.001),
            'weight_decay': self.gnn_params.get('weight_decay', 1e-4),
            'train_epoch': self.gnn_params.get('train_epoch', 100),
            'batch_size': self.gnn_params.get('batch_size', 4),
            'train_mask_rate': self.gnn_params.get('train_mask_rate', 0.3),
            'llm_num': self.num_llms,
            'model_path': self._get_model_path('save_model_path'),
            'val_split_ratio': self.gnn_params.get('val_split_ratio', 0.2)
        }

        # Initialize FormData and GNNPredictor
        self.form_data = FormData(self.device)
        self.gnn_predictor = GNNPredictor(
            query_feature_dim=self.query_dim,
            llm_feature_dim=self.llm_dim,
            hidden_features_size=self.hidden_dim,
            in_edges_size=1,  # Only use performance
            config=self.gnn_config,
            device=self.device
        )

    def _get_model_path(self, key: str) -> str:
        """Get model path from config."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_paths = self.cfg.get("model_path", {})
        return os.path.join(project_root, model_paths.get(key, f"models/gnn_{key}.pt"))

    def _prepare_training_data(self):
        """
        Prepare training data: extract query embeddings and performance labels.
        """
        # Get unique model names
        self.model_names = self.routing_data_train["model_name"].unique().tolist()
        self.num_llms = len(self.model_names)
        self.model_to_idx = {name: idx for idx, name in enumerate(self.model_names)}

        # Get unique queries
        unique_queries = self.routing_data_train["query"].unique().tolist()
        self.num_queries_train = len(unique_queries)

        # Extract query embeddings (one per query)
        query_embedding_ids = []
        for query in unique_queries:
            query_data = self.routing_data_train[self.routing_data_train["query"] == query]
            embedding_id = query_data["embedding_id"].iloc[0]
            query_embedding_ids.append(embedding_id)

        self.query_embedding_list = np.array([
            self.query_embedding_data[i].numpy() for i in query_embedding_ids
        ])

        # Build performance matrix (num_queries x num_llms)
        self.performance_matrix = np.zeros((self.num_queries_train, self.num_llms))
        for i, query in enumerate(unique_queries):
            query_data = self.routing_data_train[self.routing_data_train["query"] == query]
            for _, row in query_data.iterrows():
                model_idx = self.model_to_idx[row["model_name"]]
                self.performance_matrix[i, model_idx] = row["performance"]

        # Normalize query embeddings
        scaler = MinMaxScaler()
        self.query_embedding_list = scaler.fit_transform(self.query_embedding_list)
        self.query_dim = self.query_embedding_list.shape[1]

        # Flatten and normalize performance
        self.performance_list = self.performance_matrix.flatten()
        self.performance_list = np.nan_to_num(self.performance_list, nan=0.0)

        # Create labels (one-hot encoding, best LLM = 1)
        best_llm_indices = np.argmax(self.performance_matrix, axis=1)
        self.label = np.eye(self.num_llms)[best_llm_indices].flatten().reshape(-1, 1)

    def _prepare_llm_embeddings(self):
        """Prepare LLM embeddings from config."""
        llm_data = self.cfg.get("llm_data", {})
        llm_embeddings = []

        for model_name in self.model_names:
            if model_name in llm_data and "embedding" in llm_data[model_name]:
                embedding = llm_data[model_name]["embedding"]
            else:
                # Random initialization if no embedding provided
                embedding = np.random.randn(self.query_dim).tolist()
            llm_embeddings.append(embedding)

        self.llm_embedding = np.array(llm_embeddings)

        # Normalize LLM embeddings
        scaler = MinMaxScaler()
        self.llm_embedding = scaler.fit_transform(self.llm_embedding)
        self.llm_dim = self.llm_embedding.shape[1]

    def _build_graph_data(self, query_embeddings, performance_list, is_train=True):
        """
        Build graph data for GNN.

        Args:
            query_embeddings: Query embedding matrix
            performance_list: Flattened performance list
            is_train: Whether this is training data

        Returns:
            train_data, val_data if is_train=True, else test_data
        """
        num_queries = len(query_embeddings)

        # Build edge indices
        edge_org_id = [q for q in range(num_queries) for _ in range(self.num_llms)]
        edge_des_id = list(range(self.num_llms)) * num_queries

        # Create labels
        performance_matrix = performance_list.reshape(num_queries, self.num_llms)
        best_llm_indices = np.argmax(performance_matrix, axis=1)
        label = np.eye(self.num_llms)[best_llm_indices].flatten().reshape(-1, 1)

        if is_train:
            # Split validation set from training set
            val_ratio = self.gnn_config.get('val_split_ratio', 0.2)
            num_val = int(num_queries * val_ratio)
            num_train = num_queries - num_val

            # Shuffle indices
            all_indices = np.arange(num_queries)
            np.random.shuffle(all_indices)
            train_indices = all_indices[:num_train]
            val_indices = all_indices[num_train:]

            # Create masks
            train_idx = []
            val_idx = []

            for q_idx in train_indices:
                start = q_idx * self.num_llms
                end = start + self.num_llms
                train_idx.extend(range(start, end))

            for q_idx in val_indices:
                start = q_idx * self.num_llms
                end = start + self.num_llms
                val_idx.extend(range(start, end))

            num_edges = num_queries * self.num_llms
            mask_train = torch.zeros(num_edges)
            mask_train[train_idx] = 1
            mask_val = torch.zeros(num_edges)
            mask_val[val_idx] = 1
            mask_test = torch.zeros(num_edges)

            # Create training data
            train_data = self.form_data.formulation(
                query_feature=query_embeddings,
                llm_feature=self.llm_embedding,
                org_node=edge_org_id,
                des_node=edge_des_id,
                edge_feature=performance_list,
                label=label,
                edge_mask=mask_train,
                train_mask=mask_train,
                valide_mask=mask_val,
                test_mask=mask_test
            )

            # Create validation data (same structure, different mask)
            val_data = self.form_data.formulation(
                query_feature=query_embeddings,
                llm_feature=self.llm_embedding,
                org_node=edge_org_id,
                des_node=edge_des_id,
                edge_feature=performance_list,
                label=label,
                edge_mask=mask_val,
                train_mask=mask_train,
                valide_mask=mask_val,
                test_mask=mask_test
            )

            return train_data, val_data
        else:
            # Test data
            num_edges = num_queries * self.num_llms
            mask_test = torch.ones(num_edges)
            mask_train = torch.zeros(num_edges)
            mask_val = torch.zeros(num_edges)

            test_data = self.form_data.formulation(
                query_feature=query_embeddings,
                llm_feature=self.llm_embedding,
                org_node=edge_org_id,
                des_node=edge_des_id,
                edge_feature=performance_list,
                label=label,
                edge_mask=mask_test,
                train_mask=mask_train,
                valide_mask=mask_val,
                test_mask=mask_test
            )

            return test_data

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using the trained GNN model.

        The new query is added to the graph and GNN predicts the best LLM.
        """
        # Load trained model
        load_model_path = self._get_model_path('load_model_path')
        if os.path.exists(load_model_path):
            state_dict = torch.load(load_model_path, map_location='cpu')
            self.gnn_predictor.model.load_state_dict(state_dict)

        # Get query embedding
        query_embedding = get_longformer_embedding(query["query"]).numpy().reshape(1, -1)

        # Normalize with training data
        combined = np.vstack([self.query_embedding_list, query_embedding])
        scaler = MinMaxScaler()
        scaler.fit(combined)
        query_embedding_normalized = scaler.transform(query_embedding)

        # Build prediction graph: new query + all training data
        all_query_embeddings = np.vstack([self.query_embedding_list, query_embedding_normalized])
        num_queries = len(all_query_embeddings)

        # Create fake performance for new query (zeros, to be predicted)
        new_performance = np.zeros(self.num_llms)
        all_performance = np.concatenate([self.performance_list, new_performance])

        # Build edges
        edge_org_id = [q for q in range(num_queries) for _ in range(self.num_llms)]
        edge_des_id = list(range(self.num_llms)) * num_queries

        # Create masks: only new query needs prediction
        num_edges = num_queries * self.num_llms
        new_query_idx = num_queries - 1

        # Training edges are visible
        mask_train = torch.zeros(num_edges)
        train_edge_end = (num_queries - 1) * self.num_llms
        mask_train[:train_edge_end] = 1

        # New query edges need prediction
        mask_predict = torch.zeros(num_edges)
        predict_edge_start = new_query_idx * self.num_llms
        predict_edge_end = predict_edge_start + self.num_llms
        mask_predict[predict_edge_start:predict_edge_end] = 1

        # Create labels
        performance_matrix = all_performance.reshape(num_queries, self.num_llms)
        best_llm_indices = np.argmax(performance_matrix, axis=1)
        label = np.eye(self.num_llms)[best_llm_indices].flatten().reshape(-1, 1)

        # Build data
        predict_data = self.form_data.formulation(
            query_feature=all_query_embeddings,
            llm_feature=self.llm_embedding,
            org_node=edge_org_id,
            des_node=edge_des_id,
            edge_feature=all_performance,
            label=label,
            edge_mask=mask_predict,
            train_mask=mask_train,
            valide_mask=torch.zeros(num_edges),
            test_mask=mask_predict
        )

        # Predict
        predicted_idx = self.gnn_predictor.predict(predict_data)

        # Get predicted model name (only last query's prediction)
        model_idx = predicted_idx[-1].item()
        model_name = self.model_names[model_idx]

        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using the trained GNN model.
        """
        # Load trained model
        load_model_path = self._get_model_path('load_model_path')
        if os.path.exists(load_model_path):
            state_dict = torch.load(load_model_path, map_location='cpu')
            self.gnn_predictor.model.load_state_dict(state_dict)

        query_data_output = copy.copy(self.query_data_test)

        # Prepare test embeddings
        test_embeddings = []
        for row in query_data_output:
            embedding = get_longformer_embedding(row["query"]).numpy()
            test_embeddings.append(embedding)
        test_embeddings = np.array(test_embeddings)

        # Normalize
        combined = np.vstack([self.query_embedding_list, test_embeddings])
        scaler = MinMaxScaler()
        scaler.fit(combined)
        test_embeddings_normalized = scaler.transform(test_embeddings)

        # Build prediction graph
        all_query_embeddings = np.vstack([self.query_embedding_list, test_embeddings_normalized])
        num_train_queries = self.num_queries_train
        num_test_queries = len(test_embeddings)
        num_queries = num_train_queries + num_test_queries

        # Create fake performance for test queries
        test_performance = np.zeros(num_test_queries * self.num_llms)
        all_performance = np.concatenate([self.performance_list, test_performance])

        # Build edges
        edge_org_id = [q for q in range(num_queries) for _ in range(self.num_llms)]
        edge_des_id = list(range(self.num_llms)) * num_queries

        # Create masks
        num_edges = num_queries * self.num_llms
        mask_train = torch.zeros(num_edges)
        train_edge_end = num_train_queries * self.num_llms
        mask_train[:train_edge_end] = 1

        mask_test = torch.zeros(num_edges)
        mask_test[train_edge_end:] = 1

        # Create labels
        performance_matrix = all_performance.reshape(num_queries, self.num_llms)
        best_llm_indices = np.argmax(performance_matrix, axis=1)
        label = np.eye(self.num_llms)[best_llm_indices].flatten().reshape(-1, 1)

        # Build data
        test_data = self.form_data.formulation(
            query_feature=all_query_embeddings,
            llm_feature=self.llm_embedding,
            org_node=edge_org_id,
            des_node=edge_des_id,
            edge_feature=all_performance,
            label=label,
            edge_mask=mask_test,
            train_mask=mask_train,
            valide_mask=torch.zeros(num_edges),
            test_mask=mask_test
        )

        # Predict
        predicted_idx = self.gnn_predictor.predict(test_data)

        # Get test query predictions
        test_predictions = predicted_idx[-num_test_queries:]

        # Update output
        for i, row in enumerate(query_data_output):
            model_idx = test_predictions[i].item()
            row["model_name"] = self.model_names[model_idx]

        return query_data_output

    def get_training_data(self):
        """
        Get graph data for training.

        Returns:
            train_data, val_data: PyG Data objects
        """
        return self._build_graph_data(
            self.query_embedding_list,
            self.performance_list,
            is_train=True
        )