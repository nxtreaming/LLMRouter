from typing import Any, Dict, List, Optional, Union
import os
import pickle
import random
import numpy as np
import torch.nn as nn
import copy
from sklearn.neighbors import KNeighborsClassifier
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding


class KNNRouter(MetaRouter):
    """
    KNNRouter
    ----------
    A routing module that leverages a K-Nearest Neighbors (KNN) classifier
    to select the most similar language model based on query embeddings.

    The router inherits from MetaRouter for consistent interface design.
    If no trained KNN model is found at the specified path, it can fall back
    to random selection.

    YAML Configuration Example:
    ---------------------------
    llm_data:
      GPT4:
        size: "175B"
        embedding: [0.12, 0.33, 0.78, 0.44]
      Claude3:
        size: "52B"
        embedding: [0.10, 0.25, 0.70, 0.50]
    optional:
      knn_model_path: "configs/knn_model.pkl"
      n_neighbors: 2
      metric: "cosine"
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the KNNRouter and load configuration.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        The initialization performs the following steps:
            1. Loads configuration and metadata from MetaRouter.
            2. Builds a KNN classifier using the specified hyperparameters.
            3. Prepares the training embeddings and corresponding model labels.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Initialize KNN classifier with user-defined hyperparameters
        knn_params = self.cfg["hparam"]
        self.knn_model = KNeighborsClassifier(**knn_params)

        # Select the best-performing model for each query
        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        # Prepare embedding and label arrays for KNN training
        query_embedding_id = routing_best["embedding_id"].tolist()
        self.query_embedding_list = [self.query_embedding_data[i].numpy() for i in query_embedding_id]
        self.model_name_list = routing_best["model_name"].tolist()

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using the pre-trained KNN model.

        The method embeds the input query text using Longformer, then predicts
        the most similar LLM model based on the trained KNN classifier.

        Args:
            query (dict):
                A single query dictionary. Must contain the key:
                    - "query": textual input to be embedded.

        Returns:
            dict:
                Updated query dictionary containing:
                    - "model_name": predicted model name.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.knn_model = load_model(load_model_path)

        # Compute query embedding and predict model
        query_embedding = [get_longformer_embedding(query["query"]).numpy()]
        model_name = self.knn_model.predict(query_embedding)[0]

        # Return updated query with prediction
        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using the pre-trained KNN model.

        Each query in the test set is embedded using Longformer, and
        the trained KNN classifier predicts the most similar model
        for each query.

        Args:
            batch (Any, optional):
                Placeholder argument for compatibility with other router interfaces.
                Not used in this implementation.

        Returns:
            list of dict:
                A list of query dictionaries, each updated with:
                    - "model_name": predicted model name.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.knn_model = load_model(load_model_path)

        query_data_output = copy.copy(self.query_data_test)
        for row in query_data_output:
            query_embedding = [get_longformer_embedding(row["query"]).numpy()]
            model_name = self.knn_model.predict(query_embedding)[0]
            row["model_name"] = model_name

        return query_data_output

