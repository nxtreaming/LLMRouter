import torch
import os
from llmrouter.models.base_trainer import BaseTrainer


class GMTRouterTrainer(BaseTrainer):
    """
    GMTRouterTrainer: A trainer class for GMTRouter using Graph Neural Networks
    for multi-turn personalized routing.

    Training workflow:
    1. Get training data with conversation history from router
    2. Split validation set from training set
    3. Train GNN model with personalization features
    4. Save best model with user preference embeddings
    """

    def __init__(self, router, optimizer=None, device=None):
        """
        Initialize GMTRouterTrainer.

        Args:
            router: GMTRouter instance
            optimizer: Optional optimizer (if None, use default Adam)
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.router = router
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Get model paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path_config = router.cfg.get("model_path", {})

        self.ini_model_path = os.path.join(
            project_root,
            model_path_config.get("ini_model_path", "models/gmt_model_init.pt")
        )
        self.save_model_path = os.path.join(
            project_root,
            model_path_config.get("save_model_path", "models/gmt_model.pt")
        )

        # GMTRouter-specific paths for user embeddings
        self.user_embeddings_path = os.path.join(
            project_root,
            model_path_config.get("user_embeddings_path", "models/gmt_user_embeddings.pt")
        )

        # Training hyperparameters from config
        training_config = router.cfg.get("training", {})
        self.epochs = training_config.get("epochs", 100)
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.batch_size = training_config.get("batch_size", 32)
        self.patience = training_config.get("patience", 10)
        self.val_split = training_config.get("val_split", 0.2)

        # Personalization settings
        self.enable_personalization = router.gmt_config.get("personalization", True)
        self.context_window = router.gmt_config.get("context_window", 5)

        # Set up optimizer if not provided
        if hasattr(router, 'gmt_model') and router.gmt_model is not None:
            if optimizer is None:
                self.optimizer = torch.optim.Adam(
                    router.gmt_model.parameters(),
                    lr=self.learning_rate
                )
            else:
                self.optimizer = optimizer
        else:
            self.optimizer = None

    def train(self):
        """
        Train the GMTRouter GNN model.

        Steps:
        1. Load initial model if exists
        2. Build training and validation data with conversation history
        3. Train model with personalization features
        4. Save best model and user embeddings

        Returns:
            dict: Training results including best validation accuracy
        """
        # Check if GMTRouter model is available
        if not hasattr(self.router, 'gmt_model') or self.router.gmt_model is None:
            print("Warning: GMTRouter GNN model not available. Training skipped.")
            return {"status": "skipped", "reason": "model_not_available"}

        # Load initial model if exists
        if os.path.exists(self.ini_model_path) and self.ini_model_path.endswith(".pt"):
            print(f"Loading initial model from {self.ini_model_path}")
            state_dict = torch.load(self.ini_model_path, map_location=self.device)
            self.router.gmt_model.load_state_dict(state_dict)

        # Load user embeddings if exists
        if self.enable_personalization and os.path.exists(self.user_embeddings_path):
            print(f"Loading user embeddings from {self.user_embeddings_path}")
            user_embeddings = torch.load(self.user_embeddings_path, map_location=self.device)
            if hasattr(self.router, 'user_embeddings'):
                self.router.user_embeddings = user_embeddings

        # Get training and validation data
        print("Preparing training data...")
        train_data, val_data = self.router.get_training_data()

        # Ensure save directories exist
        for path in [self.save_model_path, self.user_embeddings_path]:
            save_dir = os.path.dirname(path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # Move model to device
        self.router.gmt_model.to(self.device)

        # Training loop
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        print(f"Starting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Personalization: {self.enable_personalization}")
        print(f"Context window: {self.context_window}")

        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(train_data)

            # Validate
            val_loss, val_acc = self._validate(val_data)

            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0

                # Save model state
                torch.save(
                    self.router.gmt_model.state_dict(),
                    self.save_model_path
                )
                print(f"Saved best model to {self.save_model_path}")

                # Save user embeddings if personalization is enabled
                if self.enable_personalization and hasattr(self.router, 'user_embeddings'):
                    torch.save(
                        self.router.user_embeddings,
                        self.user_embeddings_path
                    )
                    print(f"Saved user embeddings to {self.user_embeddings_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

        return {
            "status": "completed",
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "final_epoch": epoch + 1,
            "model_path": self.save_model_path,
            "user_embeddings_path": self.user_embeddings_path if self.enable_personalization else None
        }

    def _train_epoch(self, train_data):
        """
        Train for one epoch.

        Args:
            train_data: Training data with graph structure

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.router.gmt_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Process training data in batches
        # Note: Actual implementation depends on data format from router
        for batch in self._create_batches(train_data, self.batch_size):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.router.gmt_model(batch)
            loss = self._compute_loss(outputs, batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            correct += self._count_correct(outputs, batch)
            total += len(batch)

        avg_loss = total_loss / max(len(train_data), 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    def _validate(self, val_data):
        """
        Validate the model.

        Args:
            val_data: Validation data

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.router.gmt_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self._create_batches(val_data, self.batch_size):
                outputs = self.router.gmt_model(batch)
                loss = self._compute_loss(outputs, batch)

                total_loss += loss.item()
                correct += self._count_correct(outputs, batch)
                total += len(batch)

        avg_loss = total_loss / max(len(val_data), 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    def _create_batches(self, data, batch_size):
        """
        Create batches from data.

        Args:
            data: Input data
            batch_size: Size of each batch

        Returns:
            list: List of batches
        """
        # This is a placeholder - actual implementation depends on data format
        # For now, return data as single batch
        return [data]

    def _compute_loss(self, outputs, batch):
        """
        Compute loss for the batch.

        Args:
            outputs: Model outputs
            batch: Input batch with labels

        Returns:
            torch.Tensor: Loss value
        """
        # Placeholder - actual implementation depends on model output format
        # Typically cross-entropy loss for classification
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, batch.get('labels', torch.zeros(len(outputs))))

    def _count_correct(self, outputs, batch):
        """
        Count correct predictions.

        Args:
            outputs: Model outputs
            batch: Input batch with labels

        Returns:
            int: Number of correct predictions
        """
        # Placeholder - actual implementation depends on output format
        predictions = torch.argmax(outputs, dim=-1)
        labels = batch.get('labels', torch.zeros(len(outputs)))
        return (predictions == labels).sum().item()
