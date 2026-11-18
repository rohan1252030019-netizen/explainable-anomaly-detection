"""
src/models/deep/deep_svdd.py

Deep Support Vector Data Description (Deep SVDD) for SENTINEL.

Mathematical Reference:
Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A.,
Müller, K. R., & Kloft, M. (2018).
"Deep One-Class Classification."
International Conference on Machine Learning (ICML), PMLR 80:4393-4402.

Principles:
- Maps nominal inputs via a deep neural network phi(x; W) into a minimum-volume
  hypersphere centered at c in the latent representation space.
- Hypersphere collapse is prevented by:
  1. Fixing the center c as the empirical mean of an initial forward pass.
  2. Omitting bias terms in intermediate layers.
- Anomaly score:
    dist_sq(x) = ||phi(x; W) - c||^2
- Score convention:
    returns -dist_sq(x) so that lower = outside hypersphere = more anomalous.
"""

import time
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
    class _SVDDNet(nn.Module):
        def __init__(self, n_features: int, rep_dim: int, hidden_dim: int):
            super().__init__()
            # Bias terms omitted in SVDD networks to prevent constant representation collapse
            self.network = nn.Sequential(
                nn.Linear(n_features, hidden_dim, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, rep_dim, bias=False),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


class DeepSVDDDetector(BaseDetector):
    """
    PyTorch Deep Support Vector Data Description (Deep SVDD) Anomaly Detector.
    """

    MODEL_NAME = "Deep SVDD"

    def __init__(
        self,
        rep_dim: int = 16,
        hidden_dim: int = 32,
        nu: float = 0.05,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        contamination: float = 0.05,
        device: str = "cpu",
        random_state: int = 42,
    ):
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for DeepSVDDDetector.")

        self.rep_dim = rep_dim
        self.hidden_dim = hidden_dim
        self.nu = nu
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.contamination = contamination
        self.device = device
        self.random_state = random_state

        self.model_: nn.Module | None = None
        self.scaler_: MinMaxScaler | None = None
        self.c_: torch.Tensor | None = None
        self._threshold: float = 0.0

    def fit(self, X: np.ndarray) -> "DeepSVDDDetector":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # MinMax scaling into positive range prevents trivial zero-bias representation collapse
        self.scaler_ = MinMaxScaler(feature_range=(0.1, 1.0))
        X_scaled = self.scaler_.fit_transform(X_arr).astype(np.float32)

        actual_rep = min(self.rep_dim, max(2, n_features))
        actual_hidden = max(actual_rep * 2, min(self.hidden_dim, n_features * 4))

        self.model_ = _SVDDNet(n_features, actual_rep, actual_hidden).to(self.device)

        # Initialize hypersphere center c as trimmed median/mean of initial representations
        self.model_.eval()
        with torch.no_grad():
            full_tensor = torch.from_numpy(X_scaled).to(self.device)
            initial_rep = self.model_(full_tensor)
            # Use median across samples to be robust to outliers in unlabelled training set
            c = torch.median(initial_rep, dim=0).values
            # Avoid center components being too close to zero to prevent representation collapse
            eps = 0.1
            zero_mask = torch.abs(c) < eps
            c[zero_mask] = torch.where(c[zero_mask] >= 0, torch.tensor(eps), torch.tensor(-eps))
            self.c_ = c

        # Train network to pull representations toward center c
        train_ds = TensorDataset(full_tensor)
        train_loader = DataLoader(train_ds, batch_size=min(self.batch_size, n_samples), shuffle=True)

        optimizer = optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model_.train()
        for _epoch in range(self.epochs):
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                # Smooth L1 loss prevents extreme outliers from dominating gradients
                loss = nn.functional.smooth_l1_loss(outputs, self.c_.expand_as(outputs))
                loss.backward()
                optimizer.step()

        self.model_.eval()
        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        train_scores = self.score_samples(X_arr)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative squared Euclidean distance from center c.
        Lower = further outside hypersphere = more anomalous.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X_arr).astype(np.float32) if self.scaler_ else X_arr

        self.model_.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_scaled).to(self.device)
            rep = self.model_(tensor_x)
            dist_sq = torch.sum((rep - self.c_) ** 2, dim=1).cpu().numpy()

        return -dist_sq

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {
            "rep_dim": self.rep_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "contamination": self.contamination,
            "device": self.device,
        }
