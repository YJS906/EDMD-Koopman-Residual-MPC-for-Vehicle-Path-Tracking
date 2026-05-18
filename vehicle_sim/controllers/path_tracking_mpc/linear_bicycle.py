"""Linear bicycle prediction model for path-tracking MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BicycleParams:
    mass: float
    izz: float
    lf: float
    lr: float
    cf: float
    cr: float


@dataclass
class BicycleModelScales:
    mass_scale: float = 1.0
    izz_scale: float = 1.0
    cf_scale: float = 1.0
    cr_scale: float = 1.0
    curvature_scale: float = 1.0

    def as_dict(self) -> dict:
        return {
            "mass_scale": float(self.mass_scale),
            "izz_scale": float(self.izz_scale),
            "cf_scale": float(self.cf_scale),
            "cr_scale": float(self.cr_scale),
            "curvature_scale": float(self.curvature_scale),
        }


def bicycle_params_from_vehicle(
    vehicle,
    scales: BicycleModelScales | None = None,
) -> BicycleParams:
    scales = scales or BicycleModelScales()
    c_alpha = float(vehicle.corners["FL"].lateral_tire.params.C_alpha)
    return BicycleParams(
        mass=float(vehicle.params.m_total) * float(scales.mass_scale),
        izz=float(vehicle.params.Izz) * float(scales.izz_scale),
        lf=abs(float(vehicle.corner_offsets["FL"]["x"])),
        lr=abs(float(vehicle.corner_offsets["RL"]["x"])),
        cf=2.0 * c_alpha * float(scales.cf_scale),
        cr=2.0 * c_alpha * float(scales.cr_scale),
    )


def discretize_linear_bicycle(
    params: BicycleParams,
    vx: float,
    dt: float,
    curvature: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return discrete x[k+1] = A x[k] + B delta[k] + d.

    State is [e_y, e_psi, v_y, yaw_rate].
    """
    vx = max(abs(float(vx)), 0.5)
    m, izz, lf, lr, cf, cr = (
        params.mass,
        params.izz,
        params.lf,
        params.lr,
        params.cf,
        params.cr,
    )

    Ac = np.zeros((4, 4), dtype=float)
    Bc = np.zeros((4, 1), dtype=float)
    dc = np.zeros(4, dtype=float)

    Ac[0, 1] = vx
    Ac[0, 2] = 1.0
    Ac[1, 3] = 1.0
    dc[1] = -vx * float(curvature)

    Ac[2, 2] = -(cf + cr) / (m * vx)
    Ac[2, 3] = (-lf * cf + lr * cr) / (m * vx) - vx
    Bc[2, 0] = cf / m

    Ac[3, 2] = (-lf * cf + lr * cr) / (izz * vx)
    Ac[3, 3] = -(lf * lf * cf + lr * lr * cr) / (izz * vx)
    Bc[3, 0] = lf * cf / izz

    try:
        from scipy.linalg import expm

        aug = np.zeros((6, 6), dtype=float)
        aug[:4, :4] = Ac
        aug[:4, 4:5] = Bc
        aug[:4, 5] = dc
        exp_aug = expm(aug * float(dt))
        A = exp_aug[:4, :4]
        B = exp_aug[:4, 4:5]
        d = exp_aug[:4, 5]
    except Exception:
        A = np.eye(4) + float(dt) * Ac
        B = float(dt) * Bc
        d = float(dt) * dc

    return A, B, d
