"""Matrix RLS update for online Koopman models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RLSConfig:
    forgetting_factor: float = 0.995
    p0: float = 100.0
    value_clip: float = 1.0e4
    max_spectral_radius: float = 1.25
    reject_worse_factor: float = 1.2
    reject_worse_margin: float = 1.0e-3
    max_relative_theta_update: float | None = None
    max_post_update_control_change: float | None = None


class MatrixRLS:
    """Update all rows of theta in y = theta @ phi with a shared covariance."""

    def __init__(self, theta0: np.ndarray, config: RLSConfig | None = None) -> None:
        self.theta = np.asarray(theta0, dtype=float).copy()
        self.config = config or RLSConfig()
        n_features = self.theta.shape[1]
        self.P = float(self.config.p0) * np.eye(n_features)
        self.sample_count = 0

    def update(self, phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(self.theta.shape[0])
        lam = float(self.config.forgetting_factor)
        denom = lam + float(phi.T @ self.P @ phi)
        gain = (self.P @ phi) / max(denom, 1.0e-12)
        err = y - self.theta @ phi
        self.theta += np.outer(err, gain)
        self.theta = np.clip(self.theta, -self.config.value_clip, self.config.value_clip)
        self.P = (self.P - np.outer(gain, phi) @ self.P) / lam
        self.sample_count += 1
        return err

    def apply_spectral_guard(self, n_lifted: int) -> None:
        A = self.theta[:, :n_lifted]
        try:
            radius = float(np.max(np.abs(np.linalg.eigvals(A))))
        except np.linalg.LinAlgError:
            return
        limit = float(self.config.max_spectral_radius)
        if radius > limit > 0.0:
            self.theta[:, :n_lifted] *= limit / radius
