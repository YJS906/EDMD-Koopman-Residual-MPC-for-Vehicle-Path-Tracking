"""EDMD-Koopman identification for path-tracking states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .features import BASE_FEATURE_NAMES, BASE_STATE_NAMES, default_feature_scales


BASE_STATE_DIM = 4


@dataclass
class EDMDConfig:
    ridge: float = 1.0e-6
    feature_names: tuple[str, ...] = BASE_FEATURE_NAMES
    feature_scales: tuple[float, ...] | None = None
    feature_clip: float = 8.0
    output_dim: int = BASE_STATE_DIM
    input_names: tuple[str, ...] = ("delta_rad",)
    max_spectral_radius: float | None = 1.02
    fit_output_direct: bool = False
    prediction_mode: str = "direct_output"


@dataclass
class EDMDKoopmanModel:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    config: EDMDConfig
    feature_names: tuple[str, ...]
    feature_scales: np.ndarray
    feature_clip: float
    output_dim: int
    input_names: tuple[str, ...]

    @property
    def lifted_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def input_dim(self) -> int:
        return int(self.B.shape[1])

    def lift(self, x: np.ndarray) -> np.ndarray:
        return lift_state(x, self.feature_scales, self.feature_clip)

    def predict_one(self, x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
        u_vec = np.asarray(u, dtype=float).reshape(self.input_dim)
        z_next = self.A @ self.lift(x) + self.B @ u_vec
        return self.C @ z_next

    def to_dict(self) -> Dict:
        return {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "C": self.C.tolist(),
            "ridge": self.config.ridge,
            "max_spectral_radius": self.config.max_spectral_radius,
            "fit_output_direct": self.config.fit_output_direct,
            "prediction_mode": self.config.prediction_mode,
            "input_dim": self.input_dim,
            "input_names": list(self.input_names),
            "feature_names": list(self.feature_names),
            "feature_scales": self.feature_scales.tolist(),
            "feature_clip": float(self.feature_clip),
            "output_names": list(BASE_STATE_NAMES[: self.output_dim]),
        }


def _scales_for(feature_names: Sequence[str], feature_scales: Sequence[float] | None) -> np.ndarray:
    if feature_scales is None:
        scales = default_feature_scales(feature_names)
    else:
        scales = np.asarray(feature_scales, dtype=float).reshape(len(feature_names))
    return np.maximum(np.abs(scales), 1.0e-9)


def lift_state(
    x: np.ndarray,
    feature_scales: Sequence[float] | None = None,
    feature_clip: float = 8.0,
) -> np.ndarray:
    """Polynomial observables: 1, scaled x, scaled x^2, pairwise cross terms."""
    x = np.asarray(x, dtype=float).reshape(-1)
    scales = np.ones_like(x) if feature_scales is None else np.asarray(feature_scales, dtype=float).reshape(x.size)
    x_scaled = x / np.maximum(np.abs(scales), 1.0e-9)
    if feature_clip is not None and feature_clip > 0.0:
        x_scaled = np.clip(x_scaled, -float(feature_clip), float(feature_clip))
    obs = [1.0]
    obs.extend(float(v) for v in x_scaled)
    obs.extend(float(v * v) for v in x_scaled)
    if x_scaled.size <= 6:
        pairs = [
            (i, j)
            for i in range(x_scaled.size)
            for j in range(i + 1, x_scaled.size)
        ]
    else:
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        for j in range(4, x_scaled.size):
            pairs.extend([(0, j), (1, j), (3, j)])
    for i, j in pairs:
        obs.append(float(x_scaled[i] * x_scaled[j]))
    return np.asarray(obs, dtype=float)


def lift_batch(X: np.ndarray, feature_scales: np.ndarray, feature_clip: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([lift_state(row, feature_scales, feature_clip) for row in X])


def output_matrix_for_lift(output_dim: int, feature_scales: np.ndarray) -> np.ndarray:
    lifted_dim = lift_state(np.zeros(feature_scales.size), feature_scales).size
    C = np.zeros((output_dim, lifted_dim), dtype=float)
    for idx in range(output_dim):
        C[idx, 1 + idx] = float(feature_scales[idx])
    return C


def fit_edmd_koopman(
    X: np.ndarray,
    U: np.ndarray,
    X_next: np.ndarray,
    config: EDMDConfig | None = None,
) -> EDMDKoopmanModel:
    """Fit z[k+1] = A z[k] + B u[k] by ridge least squares."""
    cfg = config or EDMDConfig()
    feature_names = tuple(cfg.feature_names)
    output_dim = int(cfg.output_dim)
    feature_scales = _scales_for(feature_names, cfg.feature_scales)
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    X_next = np.asarray(X_next, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(f"X must have shape (n, {len(feature_names)})")
    if X_next.shape != X.shape:
        raise ValueError("X_next must match X shape")
    if output_dim > X.shape[1]:
        raise ValueError("output_dim cannot exceed feature dimension")
    if U.shape[0] != X.shape[0]:
        raise ValueError("U length must match X")

    Z = lift_batch(X, feature_scales, cfg.feature_clip)
    Z_next = lift_batch(X_next, feature_scales, cfg.feature_clip)
    G = np.vstack([Z, U.T])
    ridge_eye = cfg.ridge * np.eye(G.shape[0])
    inv_term = np.linalg.solve(G @ G.T + ridge_eye, np.eye(G.shape[0]))
    if cfg.fit_output_direct:
        y_next = X_next[:, :output_dim].T
        theta_y = y_next @ G.T @ inv_term
        A = np.zeros((Z.shape[0], Z.shape[0]), dtype=float)
        B = np.zeros((Z.shape[0], U.shape[1]), dtype=float)
        A[0, 0] = 1.0
        for idx in range(output_dim):
            row = 1 + idx
            scale = float(feature_scales[idx])
            A[row, :] = theta_y[idx, :Z.shape[0]] / scale
            B[row, :] = theta_y[idx, Z.shape[0]:] / scale
    else:
        theta = Z_next @ G.T @ inv_term
        A = theta[:, :Z.shape[0]]
        B = theta[:, Z.shape[0]:]
    if cfg.max_spectral_radius is not None:
        try:
            radius = float(np.max(np.abs(np.linalg.eigvals(A))))
        except np.linalg.LinAlgError:
            radius = 0.0
        limit = float(cfg.max_spectral_radius)
        if radius > limit > 0.0:
            A *= limit / radius
    C = output_matrix_for_lift(output_dim, feature_scales)
    return EDMDKoopmanModel(
        A=A,
        B=B,
        C=C,
        config=cfg,
        feature_names=feature_names,
        feature_scales=feature_scales,
        feature_clip=float(cfg.feature_clip),
        output_dim=output_dim,
        input_names=tuple(cfg.input_names),
    )


def prediction_errors(
    model: EDMDKoopmanModel,
    X: np.ndarray,
    U: np.ndarray,
    X_next: np.ndarray,
    multi_step: int = 10,
) -> Dict[str, float]:
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    X_next = np.asarray(X_next, dtype=float)
    y_next = X_next[:, : model.output_dim]
    one = np.asarray([model.predict_one(x, u) for x, u in zip(X, U)])
    one_err = one - y_next
    out = {
        "one_step_rmse": float(np.sqrt(np.mean(one_err * one_err))),
        "one_step_max": float(np.max(np.linalg.norm(one_err, axis=1))),
    }
    if X.shape[0] > multi_step:
        errs = []
        for start in range(0, X.shape[0] - multi_step, max(1, multi_step)):
            z = model.lift(X[start])
            for k in range(multi_step):
                z = model.A @ z + model.B @ U[start + k]
            x_pred = model.C @ z
            errs.append(x_pred - X[start + multi_step, : model.output_dim])
        if errs:
            err = np.asarray(errs)
            out["multi_step_rmse"] = float(np.sqrt(np.mean(err * err)))
            out["multi_step_horizon"] = float(multi_step)
    return out
