"""Feature construction for EDMD-Koopman path-tracking predictors."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


BASE_STATE_NAMES = ("e_y", "e_psi", "v_y", "yaw_rate")
BASE_FEATURE_NAMES = BASE_STATE_NAMES
AUGMENTED_FEATURE_NAMES = (
    "e_y",
    "e_psi",
    "v_y",
    "yaw_rate",
    "vx",
    "delta_prev",
    "curvature",
    "curvature_rate",
    "curvature_preview_1",
    "curvature_preview_2",
    "curvature_preview_3",
    "steering_rate",
    "yaw_rate_prev_1",
    "yaw_rate_prev_2",
)
TIRE_AUGMENTED_FEATURE_NAMES = (
    "e_y",
    "e_psi",
    "v_y",
    "yaw_rate",
    "vx",
    "delta_prev",
    "steering_rate",
    "curvature",
    "curvature_rate",
    "curvature_preview_1",
    "curvature_preview_2",
    "curvature_preview_3",
    "yaw_rate_prev_1",
    "yaw_rate_prev_2",
    "alpha_front_mean",
    "alpha_rear_mean",
    "alpha_front_diff",
    "Fy_front_sum",
    "Fy_rear_sum",
    "Fz_front_sum",
    "Fz_rear_sum",
    "steering_angle_front_mean",
)

FEATURE_SCALE_MAP = {
    "e_y": 2.0,
    "e_psi": 0.6,
    "v_y": 4.0,
    "yaw_rate": 2.0,
    "vx": 15.0,
    "delta_prev": 0.5,
    "curvature": 0.2,
    "curvature_rate": 1.0,
    "curvature_preview_1": 0.2,
    "curvature_preview_2": 0.2,
    "curvature_preview_3": 0.2,
    "steering_rate": 2.0,
    "yaw_rate_prev_1": 2.0,
    "yaw_rate_prev_2": 2.0,
    "alpha_front_mean": 0.25,
    "alpha_rear_mean": 0.25,
    "alpha_front_diff": 0.15,
    "alpha_rear_diff": 0.15,
    "Fy_front_sum": 12000.0,
    "Fy_rear_sum": 12000.0,
    "Fz_front_sum": 12000.0,
    "Fz_rear_sum": 12000.0,
    "steering_angle_front_mean": 0.5,
}


def default_feature_scales(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([FEATURE_SCALE_MAP.get(name, 1.0) for name in feature_names], dtype=float)


def feature_map_from_context(base_state: np.ndarray, context: Mapping | None = None) -> dict[str, float]:
    context = context or {}
    x = np.asarray(base_state, dtype=float).reshape(4)
    values = {
        "e_y": float(x[0]),
        "e_psi": float(x[1]),
        "v_y": float(x[2]),
        "yaw_rate": float(x[3]),
        "vx": float(context.get("vx", 0.0)),
        "delta_prev": float(context.get("delta_prev", 0.0)),
        "curvature": float(context.get("curvature", 0.0)),
        "curvature_rate": float(context.get("curvature_rate", 0.0)),
        "curvature_preview_1": float(context.get("curvature_preview_1", context.get("curvature", 0.0))),
        "curvature_preview_2": float(context.get("curvature_preview_2", context.get("curvature", 0.0))),
        "curvature_preview_3": float(context.get("curvature_preview_3", context.get("curvature", 0.0))),
        "steering_rate": float(context.get("steering_rate", 0.0)),
        "yaw_rate_prev_1": float(context.get("yaw_rate_prev_1", x[3])),
        "yaw_rate_prev_2": float(context.get("yaw_rate_prev_2", context.get("yaw_rate_prev_1", x[3]))),
        "alpha_front_mean": float(context.get("alpha_front_mean", 0.0)),
        "alpha_rear_mean": float(context.get("alpha_rear_mean", 0.0)),
        "alpha_front_diff": float(context.get("alpha_front_diff", 0.0)),
        "alpha_rear_diff": float(context.get("alpha_rear_diff", 0.0)),
        "Fy_front_sum": float(context.get("Fy_front_sum", 0.0)),
        "Fy_rear_sum": float(context.get("Fy_rear_sum", 0.0)),
        "Fz_front_sum": float(context.get("Fz_front_sum", 0.0)),
        "Fz_rear_sum": float(context.get("Fz_rear_sum", 0.0)),
        "steering_angle_front_mean": float(context.get("steering_angle_front_mean", context.get("delta_prev", 0.0))),
    }
    return values


def build_feature_vector(
    base_state: np.ndarray,
    context: Mapping | None = None,
    feature_names: Sequence[str] = BASE_FEATURE_NAMES,
) -> np.ndarray:
    values = feature_map_from_context(base_state, context)
    return np.asarray([values[name] for name in feature_names], dtype=float)
