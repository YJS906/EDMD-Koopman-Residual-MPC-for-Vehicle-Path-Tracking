"""MPC controllers for path tracking with linear and Koopman predictors."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Optional

import numpy as np

from .edmd import EDMDKoopmanModel
from .features import build_feature_vector
from .linear_bicycle import BicycleModelScales, bicycle_params_from_vehicle, discretize_linear_bicycle
from .rls import MatrixRLS, RLSConfig

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - exercised only when cvxpy is absent
    cp = None


@dataclass
class MPCConfig:
    horizon: int = 10
    q: np.ndarray = field(default_factory=lambda: np.diag([20.0, 8.0, 0.5, 0.5]))
    r: float = 0.25
    rd: float = 2.0
    delta_max: float = 0.45
    delta_rate_max: float = 0.08
    y_max: np.ndarray = field(default_factory=lambda: np.asarray([3.5, 0.8, 8.0, 2.5], dtype=float))
    solver: str = "OSQP"
    verbose: bool = False


@dataclass
class ControlResult:
    delta: float
    info: Dict


class _CvxLinearMPC:
    def __init__(self, config: MPCConfig) -> None:
        self.config = config

    def solve(
        self,
        A: np.ndarray,
        B: np.ndarray,
        x0: np.ndarray,
        *,
        C: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        u_prev: float = 0.0,
    ) -> ControlResult:
        if cp is None:
            return ControlResult(float(np.clip(u_prev, -self.config.delta_max, self.config.delta_max)), {
                "status": "cvxpy_unavailable",
                "solve_time": 0.0,
            })

        t0 = perf_counter()
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float).reshape(A.shape[0], 1)
        x0 = np.asarray(x0, dtype=float).reshape(A.shape[0])
        C = np.eye(A.shape[0]) if C is None else np.asarray(C, dtype=float)
        d = np.zeros(A.shape[0]) if d is None else np.asarray(d, dtype=float).reshape(A.shape[0])
        q = np.asarray(self.config.q, dtype=float)
        if q.shape[0] != C.shape[0]:
            q = np.eye(C.shape[0])
        y_max = np.asarray(self.config.y_max, dtype=float).reshape(-1)
        if y_max.size != C.shape[0]:
            y_max = np.full(C.shape[0], np.inf)

        n, horizon = A.shape[0], int(self.config.horizon)
        x = cp.Variable((n, horizon + 1))
        u = cp.Variable(horizon)
        constraints = [x[:, 0] == x0]
        cost = 0.0
        last_u = float(u_prev)
        for k in range(horizon):
            y = C @ x[:, k]
            cost += cp.quad_form(y, q) + self.config.r * cp.square(u[k])
            cost += self.config.rd * cp.square(u[k] - last_u)
            constraints += [
                x[:, k + 1] == A @ x[:, k] + B[:, 0] * u[k] + d,
                cp.abs(u[k]) <= self.config.delta_max,
                cp.abs(u[k] - last_u) <= self.config.delta_rate_max,
                cp.abs(y) <= y_max,
            ]
            last_u = u[k]
        cost += cp.quad_form(C @ x[:, horizon], q)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=self.config.solver, warm_start=True, verbose=self.config.verbose)
        except Exception as exc:
            return ControlResult(float(np.clip(u_prev, -self.config.delta_max, self.config.delta_max)), {
                "status": f"solve_error:{type(exc).__name__}",
                "solve_time": perf_counter() - t0,
            })

        status = str(problem.status)
        if u.value is None or status not in {"optimal", "optimal_inaccurate"}:
            return ControlResult(float(np.clip(u_prev, -self.config.delta_max, self.config.delta_max)), {
                "status": status,
                "solve_time": perf_counter() - t0,
            })
        delta = float(np.clip(u.value[0], -self.config.delta_max, self.config.delta_max))
        return ControlResult(delta, {
            "status": status,
            "objective": float(problem.value) if problem.value is not None else None,
            "solve_time": perf_counter() - t0,
        })


class LinearBicycleMPCController:
    name = "linear_bicycle_mpc"

    def __init__(
        self,
        vehicle,
        config: Optional[MPCConfig] = None,
        model_scales: Optional[BicycleModelScales] = None,
    ) -> None:
        self.config = config or MPCConfig()
        self.model_scales = model_scales or BicycleModelScales()
        self.params = bicycle_params_from_vehicle(vehicle, self.model_scales)
        self.solver = _CvxLinearMPC(self.config)

    def compute_control(self, x: np.ndarray, u_prev: float, context: Dict) -> ControlResult:
        A, B, d = discretize_linear_bicycle(
            self.params,
            vx=float(context.get("vx", 0.0)),
            dt=float(context["dt"]),
            curvature=float(context.get("curvature", 0.0)) * float(self.model_scales.curvature_scale),
        )
        return self.solver.solve(A, B, x, d=d, u_prev=u_prev)


class KoopmanMPCController:
    name = "fixed_koopman_mpc"

    def __init__(
        self,
        model: EDMDKoopmanModel,
        config: Optional[MPCConfig] = None,
        use_output_linearization: bool = True,
    ) -> None:
        self.model = model
        self.config = config or MPCConfig()
        self.use_output_linearization = bool(use_output_linearization)
        self.solver = _CvxLinearMPC(self.config)

    def input_vector(self, u: float, context: Optional[Dict] = None) -> np.ndarray:
        context = context or {}
        values = []
        for name in self.model.input_names:
            if name == "delta_rad":
                values.append(float(u))
            elif name == "path_curvature_1_per_m":
                values.append(float(context.get("curvature", 0.0)))
            else:
                values.append(float(context.get(name, 0.0)))
        while len(values) < self.model.input_dim:
            values.append(0.0)
        return np.asarray(values[: self.model.input_dim], dtype=float)

    def feature_vector(self, x: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        context = context or {}
        if "koopman_features" in context:
            features = np.asarray(context["koopman_features"], dtype=float).reshape(-1)
            if features.size == self.model.feature_dim:
                return features
        return build_feature_vector(x, context, self.model.feature_names)

    def predict_state(self, x: np.ndarray, u: float, context: Optional[Dict] = None) -> np.ndarray:
        features = self.feature_vector(x, context)
        return self.model.C @ (
            self.model.A @ self.model.lift(features)
            + self.model.B @ self.input_vector(u, context)
        )

    def linearized_output_model(self, x: np.ndarray, context: Optional[Dict] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(4)
        context = dict(context or {})
        context.pop("koopman_features", None)
        eps = 1.0e-4
        f0 = self.predict_state(x, 0.0, context)
        F = np.zeros((4, 4), dtype=float)
        for idx in range(4):
            step = np.zeros(4, dtype=float)
            step[idx] = eps
            fp = self.predict_state(x + step, 0.0, context)
            fm = self.predict_state(x - step, 0.0, context)
            F[:, idx] = (fp - fm) / (2.0 * eps)
        gp = self.predict_state(x, eps, context)
        gm = self.predict_state(x, -eps, context)
        G = ((gp - gm) / (2.0 * eps)).reshape(4, 1)
        d = f0 - F @ x
        return F, G, d

    def compute_control(self, x: np.ndarray, u_prev: float, context: Dict) -> ControlResult:
        if self.use_output_linearization:
            A, B, d = self.linearized_output_model(x, context)
            return self.solver.solve(A, B, x, d=d, u_prev=u_prev)

        z0 = self.model.lift(self.feature_vector(x, context))
        B_delta = self.model.B[:, :1]
        d = None
        if self.model.input_dim > 1:
            exog = self.input_vector(0.0, context)[1:]
            d = self.model.B[:, 1:] @ exog
        return self.solver.solve(
            self.model.A,
            B_delta,
            z0,
            C=self.model.C,
            d=d,
            u_prev=u_prev,
        )


class ResidualKoopmanMPCController(KoopmanMPCController):
    name = "fixed_residual_koopman_mpc"

    def __init__(
        self,
        model: EDMDKoopmanModel,
        vehicle,
        config: Optional[MPCConfig] = None,
        model_scales: Optional[BicycleModelScales] = None,
    ) -> None:
        super().__init__(model, config=config, use_output_linearization=True)
        self.model_scales = model_scales or BicycleModelScales()
        self.nominal_params = bicycle_params_from_vehicle(vehicle, self.model_scales)

    def nominal_prediction(self, x: np.ndarray, u: float, context: Optional[Dict] = None) -> np.ndarray:
        context = context or {}
        A, B, d = discretize_linear_bicycle(
            self.nominal_params,
            vx=float(context.get("vx", 0.0)),
            dt=float(context.get("dt", 0.05)),
            curvature=float(context.get("curvature", 0.0)) * float(self.model_scales.curvature_scale),
        )
        return A @ np.asarray(x, dtype=float).reshape(4) + B[:, 0] * float(u) + d

    def predict_residual(self, x: np.ndarray, u: float, context: Optional[Dict] = None) -> np.ndarray:
        features = self.feature_vector(x, context)
        return self.model.C @ (
            self.model.A @ self.model.lift(features)
            + self.model.B @ self.input_vector(u, context)
        )

    def predict_state(self, x: np.ndarray, u: float, context: Optional[Dict] = None) -> np.ndarray:
        return self.nominal_prediction(x, u, context) + self.predict_residual(x, u, context)


class OnlineKoopmanMPCController(KoopmanMPCController):
    name = "online_koopman_mpc"

    def __init__(
        self,
        model: EDMDKoopmanModel,
        config: Optional[MPCConfig] = None,
        rls_config: Optional[RLSConfig] = None,
    ) -> None:
        super().__init__(
            EDMDKoopmanModel(
                A=model.A.copy(),
                B=model.B.copy(),
                C=model.C.copy(),
                config=model.config,
                feature_names=model.feature_names,
                feature_scales=model.feature_scales.copy(),
                feature_clip=model.feature_clip,
                output_dim=model.output_dim,
                input_names=model.input_names,
            ),
            config=config,
            use_output_linearization=True,
        )
        theta0 = np.hstack([self.model.A, self.model.B])
        self.rls = MatrixRLS(theta0, rls_config or RLSConfig())
        self.last_prediction_error = 0.0

    def observe_transition(
        self,
        x: np.ndarray,
        u: float,
        x_next: np.ndarray,
        context: Optional[Dict] = None,
    ) -> Dict:
        z = self.model.lift(self.feature_vector(x, context))
        if context is not None and "next_koopman_features" in context:
            next_features = np.asarray(context["next_koopman_features"], dtype=float).reshape(self.model.feature_dim)
        else:
            next_context = context.get("next_context", {}) if context else {}
            next_features = build_feature_vector(x_next, next_context, self.model.feature_names)
        z_next = self.model.lift(next_features)
        u_vec = self.input_vector(u, context)
        phi = np.concatenate([z, u_vec])
        before = self.model.C @ (self.model.A @ z + self.model.B @ u_vec)
        theta_before = self.rls.theta.copy()
        p_before = self.rls.P.copy()
        self.rls.update(phi, z_next)
        if self.model.config.fit_output_direct:
            keep_rows = np.zeros(self.rls.theta.shape[0], dtype=bool)
            keep_rows[1:1 + self.model.output_dim] = True
            self.rls.theta[~keep_rows, :] = theta_before[~keep_rows, :]
        self.rls.apply_spectral_guard(self.model.lifted_dim)
        self.model.A = self.rls.theta[:, :self.model.lifted_dim].copy()
        self.model.B = self.rls.theta[:, self.model.lifted_dim:].copy()
        after = self.model.C @ (self.model.A @ z + self.model.B @ u_vec)
        before_err = float(np.linalg.norm(before - x_next))
        after_err = float(np.linalg.norm(after - x_next))
        accepted = True
        limit = (
            self.rls.config.reject_worse_factor * before_err
            + self.rls.config.reject_worse_margin
        )
        rel_update = float(
            np.linalg.norm(self.rls.theta - theta_before)
            / max(np.linalg.norm(theta_before), 1.0e-12)
        )
        update_limit = self.rls.config.max_relative_theta_update
        update_too_large = update_limit is not None and rel_update > float(update_limit)
        input_jump_too_large = False
        jump_limit = self.rls.config.max_post_update_control_change
        if jump_limit is not None and context is not None and "next_context" in context:
            theta_after = self.rls.theta.copy()
            model_A_after = self.model.A.copy()
            model_B_after = self.model.B.copy()
            after_delta = self.compute_control(x_next, float(u), context["next_context"]).delta
            self.model.A = theta_before[:, :self.model.lifted_dim].copy()
            self.model.B = theta_before[:, self.model.lifted_dim:].copy()
            before_delta = self.compute_control(x_next, float(u), context["next_context"]).delta
            self.rls.theta = theta_after
            self.model.A = model_A_after
            self.model.B = model_B_after
            input_jump_too_large = abs(after_delta - before_delta) > float(jump_limit)
        if after_err > limit or update_too_large or input_jump_too_large:
            accepted = False
            self.rls.theta = theta_before
            self.rls.P = p_before
            self.model.A = theta_before[:, :self.model.lifted_dim].copy()
            self.model.B = theta_before[:, self.model.lifted_dim:].copy()
            after_err = before_err
        self.last_prediction_error = after_err
        return {
            "prediction_error_before": before_err,
            "prediction_error_after": after_err,
            "rls_samples": self.rls.sample_count,
            "rls_update_accepted": accepted,
            "rls_relative_theta_update": rel_update,
            "rls_input_jump_rejected": input_jump_too_large,
        }


class OnlineResidualKoopmanMPCController(ResidualKoopmanMPCController):
    name = "online_residual_koopman_mpc"

    def __init__(
        self,
        model: EDMDKoopmanModel,
        vehicle,
        config: Optional[MPCConfig] = None,
        model_scales: Optional[BicycleModelScales] = None,
        rls_config: Optional[RLSConfig] = None,
    ) -> None:
        copied = EDMDKoopmanModel(
            A=model.A.copy(),
            B=model.B.copy(),
            C=model.C.copy(),
            config=model.config,
            feature_names=model.feature_names,
            feature_scales=model.feature_scales.copy(),
            feature_clip=model.feature_clip,
            output_dim=model.output_dim,
            input_names=model.input_names,
        )
        super().__init__(
            copied,
            vehicle,
            config=config,
            model_scales=model_scales,
        )
        theta0 = np.hstack([self.model.A, self.model.B])
        self.rls = MatrixRLS(theta0, rls_config or RLSConfig())
        self.last_prediction_error = 0.0

    def observe_transition(
        self,
        x: np.ndarray,
        u: float,
        x_next: np.ndarray,
        context: Optional[Dict] = None,
    ) -> Dict:
        context = context or {}
        z = self.model.lift(self.feature_vector(x, context))
        next_context = context.get("next_context", {})
        if "next_koopman_features" in context:
            target_features = np.asarray(context["next_koopman_features"], dtype=float).reshape(self.model.feature_dim).copy()
        else:
            target_features = build_feature_vector(x_next, next_context, self.model.feature_names)
        nominal_next = self.nominal_prediction(x, u, context)
        true_residual = np.asarray(x_next, dtype=float).reshape(4) - nominal_next
        true_residual = np.clip(true_residual, -5.0, 5.0)
        target_features[: self.model.output_dim] = true_residual[: self.model.output_dim]
        z_target = self.model.lift(target_features)
        u_vec = self.input_vector(u, context)
        phi = np.concatenate([z, u_vec])

        before_residual = self.model.C @ (self.model.A @ z + self.model.B @ u_vec)
        before_state = nominal_next + before_residual
        theta_before = self.rls.theta.copy()
        p_before = self.rls.P.copy()
        self.rls.update(phi, z_target)
        keep_rows = np.zeros(self.rls.theta.shape[0], dtype=bool)
        keep_rows[1:1 + self.model.output_dim] = True
        self.rls.theta[~keep_rows, :] = theta_before[~keep_rows, :]
        self.rls.apply_spectral_guard(self.model.lifted_dim)
        self.model.A = self.rls.theta[:, :self.model.lifted_dim].copy()
        self.model.B = self.rls.theta[:, self.model.lifted_dim:].copy()
        after_residual = self.model.C @ (self.model.A @ z + self.model.B @ u_vec)
        after_state = nominal_next + after_residual

        before_residual_err = float(np.linalg.norm(before_residual - true_residual))
        after_residual_err = float(np.linalg.norm(after_residual - true_residual))
        before_state_err = float(np.linalg.norm(before_state - x_next))
        after_state_err = float(np.linalg.norm(after_state - x_next))
        accepted = True
        limit = (
            self.rls.config.reject_worse_factor * before_residual_err
            + self.rls.config.reject_worse_margin
        )
        rel_update = float(
            np.linalg.norm(self.rls.theta - theta_before)
            / max(np.linalg.norm(theta_before), 1.0e-12)
        )
        update_limit = self.rls.config.max_relative_theta_update
        update_too_large = update_limit is not None and rel_update > float(update_limit)
        input_jump_too_large = False
        jump_limit = self.rls.config.max_post_update_control_change
        if jump_limit is not None and "next_context" in context:
            theta_after = self.rls.theta.copy()
            model_A_after = self.model.A.copy()
            model_B_after = self.model.B.copy()
            after_delta = self.compute_control(x_next, float(u), context["next_context"]).delta
            self.model.A = theta_before[:, :self.model.lifted_dim].copy()
            self.model.B = theta_before[:, self.model.lifted_dim:].copy()
            before_delta = self.compute_control(x_next, float(u), context["next_context"]).delta
            self.rls.theta = theta_after
            self.model.A = model_A_after
            self.model.B = model_B_after
            input_jump_too_large = abs(after_delta - before_delta) > float(jump_limit)
        if after_residual_err > limit or update_too_large or input_jump_too_large:
            accepted = False
            self.rls.theta = theta_before
            self.rls.P = p_before
            self.model.A = theta_before[:, :self.model.lifted_dim].copy()
            self.model.B = theta_before[:, self.model.lifted_dim:].copy()
            after_residual_err = before_residual_err
            after_state_err = before_state_err
        self.last_prediction_error = after_state_err
        return {
            "prediction_error_before": before_state_err,
            "prediction_error_after": after_state_err,
            "residual_error_before": before_residual_err,
            "residual_error_after": after_residual_err,
            "rls_samples": self.rls.sample_count,
            "rls_update_accepted": accepted,
            "rls_relative_theta_update": rel_update,
            "rls_input_jump_rejected": input_jump_too_large,
        }
