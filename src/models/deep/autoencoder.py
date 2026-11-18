"""
src/models/deep/autoencoder.py

Deep Autoencoder Anomaly Detector for SENTINEL.

Mathematical Foundation:
--------------------------
The Autoencoder models the manifold of nominal data via a low-dimensional bottleneck:
    Encoder: f_theta(x) -> z in R^k,  k << d
    Decoder: g_phi(z)   -> x_hat in R^d

Loss function (Reconstruction MSE):
    L(x, x_hat) = (1 / d) * sum_{j=1}^d (x_j - x_hat_j)^2

Anomalies do not conform to the learned nominal manifold and therefore suffer
high reconstruction error.

Score convention:
    returns -L(x, x_hat) so that lower = higher reconstruction error = more anomalous.

Feature Attribution:
    Native per-feature squared reconstruction error:
        Attr_j(x) = (x_j - x_hat_j)^2
    identifies exactly which physical attributes failed reconstruction without
    requiring perturbation sampling.
"""

import time
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from src.models.base_detector import BaseDetector

if _HAS_TORCH:
    class _AENet(nn.Module):
        def __init__(self, n_features: int, latent_dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, n_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            return self.decoder(z)


class AutoencoderDetector(BaseDetector):
    """
    PyTorch Deep Autoencoder Anomaly Detector with native feature attribution.
    """

    MODEL_NAME = "Deep Autoencoder"

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 32,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 0.005,
        patience: int = 5,
        contamination: float = 0.05,
        device: str = "cpu",
        random_state: int = 42,
    ):
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for AutoencoderDetector. Install via 'pip install torch'.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.contamination = contamination
        self.device = device
        self.random_state = random_state

        self.model_: nn.Module | None = None
        self.scaler_: StandardScaler | None = None
        self._threshold: float = 0.0

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr).astype(np.float32)

        # Train/val split for early stopping
        val_size = max(5, int(n_samples * 0.15))
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_ds = TensorDataset(torch.from_numpy(X_scaled[train_idx]))
        val_tensor = torch.from_numpy(X_scaled[val_idx]).to(self.device)

        train_loader = DataLoader(train_ds, batch_size=min(self.batch_size, len(train_idx)), shuffle=True)

        actual_latent = min(self.latent_dim, max(1, n_features // 2))
        actual_hidden = max(actual_latent * 2, min(self.hidden_dim, n_features * 4))

        self.model_ = _AENet(n_features, actual_latent, actual_hidden).to(self.device)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model_.train()
        for _epoch in range(self.epochs):
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model_(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

            # Early stopping validation
            self.model_.eval()
            with torch.no_grad():
                val_recon = self.model_(val_tensor)
                val_loss = float(criterion(val_recon, val_tensor).item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
            self.model_.train()

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()

        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        train_scores = self.score_samples(X_arr)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def _get_reconstruction_errors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-sample mean reconstruction error and per-feature squared error.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X_arr).astype(np.float32) if self.scaler_ else X_arr

        self.model_.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_scaled).to(self.device)
            recon = self.model_(tensor_x).cpu().numpy()

        per_feature_error = (X_scaled - recon) ** 2
        per_sample_mse = np.mean(per_feature_error, axis=1)
        return per_sample_mse, per_feature_error

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative reconstruction MSE.
        Lower = higher reconstruction error = more anomalous.
        """
        per_sample_mse, _ = self._get_reconstruction_errors(X)
        return -per_sample_mse

    def get_feature_attribution(self, X: np.ndarray) -> np.ndarray:
        """
        Return normalized per-feature squared reconstruction errors.
        """
        _, per_feature_error = self._get_reconstruction_errors(X)
        return per_feature_error

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "contamination": self.contamination,
            "device": self.device,
        }
