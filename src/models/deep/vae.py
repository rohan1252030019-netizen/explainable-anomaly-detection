"""
src/models/deep/vae.py

Variational Autoencoder (VAE) Anomaly Detector for SENTINEL.

Mathematical Foundation:
--------------------------
The VAE models a probabilistic generative prior p(z) = N(0, I):
    Encoder q_theta(z|x): outputs mu(x) and log(sigma^2(x))
    Reparameterization: z = mu + sigma * epsilon, epsilon ~ N(0, I)
    Decoder p_phi(x|z): outputs x_hat in R^d

Objective Function (Evidence Lower Bound - ELBO):
    L(x) = MSE(x, x_hat) + beta * KL( q(z|x) || p(z) )
where:
    KL = -0.5 * sum_{k=1}^K (1 + log(sigma_k^2) - mu_k^2 - sigma_k^2)

Anomalies lie in regions of low nominal density and produce both poor reconstruction
and high latent KL divergence.

Score convention:
    returns -(MSE + beta * KL) so that lower = more anomalous.
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
    class _VAENet(nn.Module):
        def __init__(self, n_features: int, latent_dim: int, hidden_dim: int):
            super().__init__()
            self.encoder_shared = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, n_features),
            )

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder_shared(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar


class VAEDetector(BaseDetector):
    """
    PyTorch Variational Autoencoder (VAE) Anomaly Detector.
    """

    MODEL_NAME = "Variational Autoencoder"

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 32,
        beta: float = 0.01,
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
            raise ImportError("PyTorch is required for VAEDetector.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
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

    def fit(self, X: np.ndarray) -> "VAEDetector":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr).astype(np.float32)

        val_size = max(5, int(n_samples * 0.15))
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_ds = TensorDataset(torch.from_numpy(X_scaled[train_idx]))
        val_tensor = torch.from_numpy(X_scaled[val_idx]).to(self.device)
        train_loader = DataLoader(train_ds, batch_size=min(self.batch_size, len(train_idx)), shuffle=True)

        actual_latent = min(self.latent_dim, max(1, n_features // 2))
        actual_hidden = max(actual_latent * 2, min(self.hidden_dim, n_features * 4))

        self.model_ = _VAENet(n_features, actual_latent, actual_hidden).to(self.device)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model_.train()
        for _epoch in range(self.epochs):
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model_(batch_x)
                mse_loss = nn.functional.mse_loss(recon, batch_x)
                kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                total_loss = mse_loss + self.beta * kl_loss
                total_loss.backward()
                optimizer.step()

            # Early stopping validation
            self.model_.eval()
            with torch.no_grad():
                val_recon, val_mu, val_logvar = self.model_(val_tensor)
                val_mse = nn.functional.mse_loss(val_recon, val_tensor)
                val_kl = -0.5 * torch.mean(torch.sum(1.0 + val_logvar - val_mu.pow(2) - val_logvar.exp(), dim=1))
                val_total = float((val_mse + self.beta * val_kl).item())

            if val_total < best_val_loss:
                best_val_loss = val_total
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

    def _get_vae_errors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X_arr).astype(np.float32) if self.scaler_ else X_arr

        model: Any = self.model_
        assert model is not None
        model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_scaled).to(self.device)
            mu, _ = model.encode(tensor_x)
            recon = model.decode(mu).cpu().numpy()

        per_feature_error = (X_scaled - recon) ** 2
        per_sample_mse = np.mean(per_feature_error, axis=1)
        return per_sample_mse, per_feature_error

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        per_sample_mse, _ = self._get_vae_errors(X)
        return -per_sample_mse

    def get_feature_attribution(self, X: np.ndarray) -> np.ndarray:
        _, per_feature_error = self._get_vae_errors(X)
        return per_feature_error

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "beta": self.beta,
            "epochs": self.epochs,
            "contamination": self.contamination,
            "device": self.device,
        }
