"""Closed-loop VehicleBody plant helpers for path-tracking MPC experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import csv
import json
import numpy as np

from vehicle_sim.controllers.path_tracking_mpc.features import (
    AUGMENTED_FEATURE_NAMES,
    BASE_FEATURE_NAMES,
    TIRE_AUGMENTED_FEATURE_NAMES,
    build_feature_vector,
)
from vehicle_sim.controllers.path_tracking_mpc.linear_bicycle import BicycleParams, discretize_linear_bicycle
from vehicle_sim.models.vehicle_body.vehicle_body import VehicleBody
from vehicle_sim.utils.direct_ackermann_steering import DirectAckermannSteeringWrapper
from vehicle_sim.utils.yaw_sim_utils import set_initial_speed


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class SinePath:
    amplitude: float = 0.7
    wavelength: float = 35.0
    name: str = "mild_sine"

    def y(self, x: float) -> float:
        return float(self.amplitude * np.sin(2.0 * np.pi * x / self.wavelength))

    def dy(self, x: float) -> float:
        return float(self.amplitude * (2.0 * np.pi / self.wavelength) * np.cos(2.0 * np.pi * x / self.wavelength))

    def ddy(self, x: float) -> float:
        k = 2.0 * np.pi / self.wavelength
        return float(-self.amplitude * k * k * np.sin(k * x))

    def heading(self, x: float) -> float:
        return float(np.arctan(self.dy(x)))

    def curvature(self, x: float) -> float:
        yp = self.dy(x)
        return float(self.ddy(x) / ((1.0 + yp * yp) ** 1.5))

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "amplitude_m": float(self.amplitude),
            "wavelength_m": float(self.wavelength),
        }


@dataclass
class AggressiveSinePath(SinePath):
    amplitude: float = 1.5
    wavelength: float = 22.0
    name: str = "aggressive_sine"
    ramp_length: float = 8.0

    def _ramp_terms(self, x: float) -> tuple[float, float, float]:
        r = max(float(self.ramp_length), 1.0e-6)
        e = float(np.exp(-((float(x) / r) ** 2)))
        w = 1.0 - e
        wp = 2.0 * float(x) * e / (r * r)
        wpp = (2.0 / (r * r) - 4.0 * float(x) * float(x) / (r ** 4)) * e
        return w, wp, wpp

    def y(self, x: float) -> float:
        if self.ramp_length <= 0.0:
            return SinePath.y(self, x)
        k = 2.0 * np.pi / self.wavelength
        w, _, _ = self._ramp_terms(x)
        return float(self.amplitude * w * np.sin(k * float(x)))

    def dy(self, x: float) -> float:
        if self.ramp_length <= 0.0:
            return SinePath.dy(self, x)
        k = 2.0 * np.pi / self.wavelength
        xf = float(x)
        w, wp, _ = self._ramp_terms(xf)
        return float(self.amplitude * (wp * np.sin(k * xf) + w * k * np.cos(k * xf)))

    def ddy(self, x: float) -> float:
        if self.ramp_length <= 0.0:
            return SinePath.ddy(self, x)
        k = 2.0 * np.pi / self.wavelength
        xf = float(x)
        w, wp, wpp = self._ramp_terms(xf)
        return float(self.amplitude * (
            wpp * np.sin(k * xf)
            + 2.0 * wp * k * np.cos(k * xf)
            - w * k * k * np.sin(k * xf)
        ))

    def as_dict(self) -> Dict[str, float | str]:
        data = super().as_dict()
        data["ramp_length_m"] = float(self.ramp_length)
        return data


@dataclass
class DoubleLaneChangePath:
    lane_shift: float = 2.8
    x1: float = 20.0
    x2: float = 54.0
    k: float = 0.22
    name: str = "double_lane_change"

    def y(self, x: float) -> float:
        x = float(x)
        a = self.k * (x - self.x1)
        b = self.k * (x - self.x2)
        return float(0.5 * self.lane_shift * (np.tanh(a) - np.tanh(b)))

    def dy(self, x: float) -> float:
        x = float(x)
        a = self.k * (x - self.x1)
        b = self.k * (x - self.x2)
        sech2_a = 1.0 / (np.cosh(a) ** 2)
        sech2_b = 1.0 / (np.cosh(b) ** 2)
        return float(0.5 * self.lane_shift * self.k * (sech2_a - sech2_b))

    def ddy(self, x: float) -> float:
        x = float(x)
        a = self.k * (x - self.x1)
        b = self.k * (x - self.x2)
        sech2_a = 1.0 / (np.cosh(a) ** 2)
        sech2_b = 1.0 / (np.cosh(b) ** 2)
        second_a = -2.0 * self.k * self.k * sech2_a * np.tanh(a)
        second_b = -2.0 * self.k * self.k * sech2_b * np.tanh(b)
        return float(0.5 * self.lane_shift * (second_a - second_b))

    def heading(self, x: float) -> float:
        return float(np.arctan(self.dy(x)))

    def curvature(self, x: float) -> float:
        yp = self.dy(x)
        return float(self.ddy(x) / ((1.0 + yp * yp) ** 1.5))

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "lane_shift_m": float(self.lane_shift),
            "x1_m": float(self.x1),
            "x2_m": float(self.x2),
            "k_1_per_m": float(self.k),
        }


@dataclass
class AdaptiveLaneChangePath:
    lane_shift: float = 3.0
    x1: float = 32.0
    x2: float = 64.0
    k: float = 0.21
    adaptation_amplitude: float = 0.18
    adaptation_wavelength: float = 34.0
    adaptation_decay_x: float = 34.0
    name: str = "adaptive_lane_change"

    def _lane_terms(self, x: float) -> tuple[float, float, float]:
        a = self.k * (x - self.x1)
        b = self.k * (x - self.x2)
        sech2_a = 1.0 / (np.cosh(a) ** 2)
        sech2_b = 1.0 / (np.cosh(b) ** 2)
        y = 0.5 * self.lane_shift * (np.tanh(a) - np.tanh(b))
        dy = 0.5 * self.lane_shift * self.k * (sech2_a - sech2_b)
        ddy = 0.5 * self.lane_shift * (
            -2.0 * self.k * self.k * sech2_a * np.tanh(a)
            + 2.0 * self.k * self.k * sech2_b * np.tanh(b)
        )
        return float(y), float(dy), float(ddy)

    def _adapt_terms(self, x: float) -> tuple[float, float, float]:
        k = 2.0 * np.pi / self.adaptation_wavelength
        xd = max(float(self.adaptation_decay_x), 1.0e-6)
        gate = 0.5 * (1.0 - np.tanh((float(x) - xd) / 5.0))
        gate_p = -0.5 * (1.0 / np.cosh((float(x) - xd) / 5.0) ** 2) / 5.0
        gate_pp = (1.0 / 25.0) * (1.0 / np.cosh((float(x) - xd) / 5.0) ** 2) * np.tanh((float(x) - xd) / 5.0)
        s = np.sin(k * float(x))
        c = np.cos(k * float(x))
        y = self.adaptation_amplitude * gate * s
        dy = self.adaptation_amplitude * (gate_p * s + gate * k * c)
        ddy = self.adaptation_amplitude * (gate_pp * s + 2.0 * gate_p * k * c - gate * k * k * s)
        return float(y), float(dy), float(ddy)

    def y(self, x: float) -> float:
        lane, _, _ = self._lane_terms(float(x))
        adapt, _, _ = self._adapt_terms(float(x))
        return float(lane + adapt)

    def dy(self, x: float) -> float:
        _, lane, _ = self._lane_terms(float(x))
        _, adapt, _ = self._adapt_terms(float(x))
        return float(lane + adapt)

    def ddy(self, x: float) -> float:
        _, _, lane = self._lane_terms(float(x))
        _, _, adapt = self._adapt_terms(float(x))
        return float(lane + adapt)

    def heading(self, x: float) -> float:
        return float(np.arctan(self.dy(x)))

    def curvature(self, x: float) -> float:
        yp = self.dy(x)
        return float(self.ddy(x) / ((1.0 + yp * yp) ** 1.5))

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "lane_shift_m": float(self.lane_shift),
            "x1_m": float(self.x1),
            "x2_m": float(self.x2),
            "k_1_per_m": float(self.k),
            "adaptation_amplitude_m": float(self.adaptation_amplitude),
            "adaptation_wavelength_m": float(self.adaptation_wavelength),
            "adaptation_decay_x_m": float(self.adaptation_decay_x),
        }


def _smooth_step(x: float, center: float, width: float) -> float:
    return float(0.5 * (1.0 + np.tanh((float(x) - float(center)) / max(float(width), 1.0e-6))))


def _smooth_window(x: float, start: float, end: float, width: float) -> float:
    return float(_smooth_step(x, start, width) * (1.0 - _smooth_step(x, end, width)))


@dataclass
class CompositeFrictionAdaptationPath:
    target_speed: float = 10.0
    lane_shift: float = 3.0
    lane_k: float = 0.20
    chicane_amplitude: float = 0.9
    chicane_wavelength: float = 38.0
    adaptation_amplitude: float = 0.22
    adaptation_wavelength: float = 36.0
    mu_initial: float = 0.85
    first_mu_final: float = 0.70
    second_patch_mu: float = 0.65
    exit_mu: float = 0.80
    name: str = "composite_friction_adaptation_course"

    def _x_at_time(self, time_s: float) -> float:
        return float(self.target_speed) * float(time_s)

    @property
    def lane_x1(self) -> float:
        return self._x_at_time(9.7)

    @property
    def lane_x2(self) -> float:
        return self._x_at_time(13.1)

    @property
    def chicane_start_x(self) -> float:
        return self._x_at_time(16.0)

    @property
    def chicane_end_x(self) -> float:
        return self._x_at_time(22.3)

    def _lane_y(self, x: float) -> float:
        a = self.lane_k * (float(x) - self.lane_x1)
        b = self.lane_k * (float(x) - self.lane_x2)
        return float(0.5 * self.lane_shift * (np.tanh(a) - np.tanh(b)))

    def _adaptation_y(self, x: float) -> float:
        k = 2.0 * np.pi / max(float(self.adaptation_wavelength), 1.0e-6)
        gate = _smooth_window(float(x), self._x_at_time(0.5), self._x_at_time(9.0), 8.0)
        return float(self.adaptation_amplitude * gate * np.sin(k * float(x)))

    def _chicane_y(self, x: float) -> float:
        k = 2.0 * np.pi / max(float(self.chicane_wavelength), 1.0e-6)
        x0 = self.chicane_start_x
        gate = _smooth_window(float(x), self.chicane_start_x, self.chicane_end_x, 8.0)
        return float(self.chicane_amplitude * gate * np.sin(k * (float(x) - x0)))

    def y(self, x: float) -> float:
        return float(self._adaptation_y(x) + self._lane_y(x) + self._chicane_y(x))

    def dy(self, x: float) -> float:
        h = 0.05
        return float((self.y(float(x) + h) - self.y(float(x) - h)) / (2.0 * h))

    def ddy(self, x: float) -> float:
        h = 0.10
        xf = float(x)
        return float((self.y(xf + h) - 2.0 * self.y(xf) + self.y(xf - h)) / (h * h))

    def heading(self, x: float) -> float:
        return float(np.arctan(self.dy(x)))

    def curvature(self, x: float) -> float:
        yp = self.dy(x)
        return float(self.ddy(x) / ((1.0 + yp * yp) ** 1.5))

    def segments(self) -> List[Dict[str, float | str | bool]]:
        speed = max(float(self.target_speed), 1.0e-9)
        mu0 = float(self.mu_initial)
        mu1 = float(self.first_mu_final)
        mu2 = float(self.second_patch_mu)
        mu3 = float(self.exit_mu)
        specs = [
            ("initial_training", "Initial EDMD Training", 0.0, 5.0, mu0, mu0, "#9ecae1", "limited offline EDMD data collection", True, False, False),
            ("friction_transition_1", "Friction Transition", 5.0, 7.0, mu0, mu1, "#fdd0a2", "road friction ramp creating plant-model mismatch", False, False, False),
            ("online_adaptation_1", "Online Adaptation", 7.0, 9.0, mu1, mu1, "#c7e9c0", "low-mu response observation before maneuver", False, True, True),
            ("double_lane_change", "Double Lane Change", 9.0, 14.0, mu1, mu1, "#fdae6b", "transient evasive maneuver", False, False, True),
            ("recovery_1", "Recovery", 14.0, 16.0, mu1, mu1, "#d9d9d9", "tracking recovery and stabilization", False, False, False),
            ("moderate_chicane", "Moderate Chicane", 16.0, 20.0, mu1, mu1, "#bcbddc", "moderate S-course with repeated yaw excitation", False, False, True),
            ("friction_patch_2", "Low-Friction Patch", 20.0, 24.0, mu1, mu2, "#fb6a4a", "second low-friction patch over the chicane tail", False, False, True),
            ("exit_recovery", "Exit Recovery", 24.0, 27.0, mu2, mu3, "#c7e9c0", "friction recovery and final stabilization", False, False, False),
        ]
        rows: List[Dict[str, float | str | bool]] = []
        for name, display, start_t, end_t, mu0, mu1, color, purpose, is_train, is_adapt, is_maneuver in specs:
            rows.append({
                "segment_name": name,
                "display_name": display,
                "start_time_s": float(start_t),
                "end_time_s": float(end_t),
                "start_x_m": float(start_t * speed),
                "end_x_m": float(end_t * speed),
                "mu_start": float(mu0),
                "mu_end": float(mu1),
                "description": purpose,
                "visualization_color": color,
                "primary_purpose": purpose,
                "is_training_zone": bool(is_train),
                "is_online_adaptation_zone": bool(is_adapt),
                "is_maneuver_zone": bool(is_maneuver),
            })
        return rows

    def friction_piecewise(self) -> List[Dict[str, float]]:
        return [
            {
                "start_time": float(row["start_time_s"]),
                "end_time": float(row["end_time_s"]),
                "mu_start": float(row["mu_start"]),
                "mu_end": float(row["mu_end"]),
            }
            for row in self.segments()
        ]

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "target_speed_mps": float(self.target_speed),
            "lane_shift_m": float(self.lane_shift),
            "lane_k_1_per_m": float(self.lane_k),
            "lane_x1_m": float(self.lane_x1),
            "lane_x2_m": float(self.lane_x2),
            "chicane_amplitude_m": float(self.chicane_amplitude),
            "chicane_wavelength_m": float(self.chicane_wavelength),
            "adaptation_amplitude_m": float(self.adaptation_amplitude),
            "adaptation_wavelength_m": float(self.adaptation_wavelength),
            "mu_initial": float(self.mu_initial),
            "first_mu_final": float(self.first_mu_final),
            "second_patch_mu": float(self.second_patch_mu),
            "exit_mu": float(self.exit_mu),
        }


@dataclass
class NonstationaryAdaptiveTechnicalPath:
    initial_speed_kmh: float = 36.0
    mid_speed_kmh: float = 45.0
    high_speed_kmh: float = 50.0
    exit_speed_kmh: float = 40.0
    mu_initial: float = 0.90
    mu_mid: float = 0.72
    mu_chicane: float = 0.68
    mu_low: float = 0.62
    mu_exit: float = 0.80
    lane_shift: float = 3.2
    lane_k: float = 0.14
    chicane_amplitude: float = 0.7
    chicane_wavelength: float = 65.0
    adaptation_amplitude: float = 0.20
    adaptation_wavelength: float = 55.0
    final_lane_shift: float = 2.8
    final_lane_k: float = 0.12
    name: str = "nonstationary_adaptive_technical_course"

    @staticmethod
    def _kmh_to_mps(speed_kmh: float) -> float:
        return float(speed_kmh) / 3.6

    @property
    def target_speed(self) -> float:
        return self._kmh_to_mps(self.initial_speed_kmh)

    @property
    def key_segment_names(self) -> List[str]:
        return [
            "double_lane_change",
            "speed_up_chicane",
            "low_friction_patch",
            "final_evasive_maneuver",
        ]

    def _segment_specs(self) -> List[tuple]:
        s0 = float(self.initial_speed_kmh)
        s1 = float(self.mid_speed_kmh)
        s2 = float(self.high_speed_kmh)
        s3 = float(self.exit_speed_kmh)
        mu0 = float(self.mu_initial)
        mu1 = float(self.mu_mid)
        mu2 = float(self.mu_chicane)
        mu3 = float(self.mu_low)
        mu4 = float(self.mu_exit)
        return [
            (
                "initial_edmd_training",
                "Initial EDMD Training",
                0.0,
                6.0,
                s0,
                s0,
                mu0,
                mu0,
                "#9ecae1",
                "limited high-mu low-speed EDMD data collection",
                True,
                False,
                False,
            ),
            (
                "speed_ramp",
                "Speed Ramp",
                6.0,
                10.0,
                s0,
                43.0,
                mu0,
                mu0,
                "#c6dbef",
                "introduce speed-varying dynamics before friction shift",
                False,
                False,
                False,
            ),
            (
                "friction_transition_1",
                "Friction Transition",
                10.0,
                14.0,
                43.0,
                43.0,
                mu0,
                mu1,
                "#fdd0a2",
                "road friction ramp creating plant-model mismatch",
                False,
                False,
                False,
            ),
            (
                "online_adaptation_1",
                "Online Adaptation 1",
                14.0,
                18.0,
                43.0,
                s1,
                mu1,
                mu1,
                "#c7e9c0",
                "observe changed plant response before the first maneuver",
                False,
                True,
                False,
            ),
            (
                "double_lane_change",
                "Double Lane Change",
                18.0,
                26.0,
                s1,
                s1,
                mu1,
                mu1,
                "#fdae6b",
                "first key evasive maneuver under changed tire-road response",
                False,
                False,
                True,
            ),
            (
                "speed_up_chicane",
                "Speed-Up Chicane",
                26.0,
                34.0,
                s1,
                s2,
                mu1,
                mu2,
                "#bcbddc",
                "wide chicane with increasing speed and mild friction reduction",
                False,
                False,
                True,
            ),
            (
                "low_friction_patch",
                "Low-Friction Patch",
                34.0,
                40.0,
                48.0,
                s2,
                mu2,
                mu3,
                "#fb6a4a",
                "low-friction nonlinear tire response with curvature change",
                False,
                False,
                True,
            ),
            (
                "online_adaptation_2",
                "Online Adaptation 2",
                40.0,
                44.0,
                s1,
                s1,
                mu3,
                mu3,
                "#c7e9c0",
                "second low-mu adaptation opportunity before final maneuver",
                False,
                True,
                False,
            ),
            (
                "final_evasive_maneuver",
                "Final Evasive Maneuver",
                44.0,
                52.0,
                45.0,
                43.0,
                mu3,
                0.75,
                "#fd8d3c",
                "final long lane-change maneuver after online adaptation",
                False,
                False,
                True,
            ),
            (
                "exit_recovery",
                "Exit Recovery",
                52.0,
                55.0,
                43.0,
                s3,
                0.75,
                mu4,
                "#d9d9d9",
                "stabilization and friction recovery",
                False,
                False,
                False,
            ),
        ]

    def _speed_kmh_at_time(self, time_s: float) -> float:
        t = float(time_s)
        specs = self._segment_specs()
        for spec in specs:
            _, _, t0, t1, v0, v1, *_ = spec
            if t0 <= t < t1:
                blend = np.clip((t - t0) / max(t1 - t0, 1.0e-9), 0.0, 1.0)
                return float((1.0 - blend) * v0 + blend * v1)
        if t < specs[0][2]:
            return float(specs[0][4])
        return float(specs[-1][5])

    def speed_at_time(self, time_s: float) -> float:
        return self._kmh_to_mps(self._speed_kmh_at_time(time_s))

    def x_at_time(self, time_s: float) -> float:
        t_target = max(float(time_s), 0.0)
        specs = self._segment_specs()
        x = 0.0
        for spec in specs:
            _, _, t0, t1, v0_kmh, v1_kmh, *_ = spec
            if t_target <= t0:
                break
            dt_seg = min(t_target, t1) - t0
            if dt_seg > 0.0:
                full_dt = max(t1 - t0, 1.0e-9)
                v0 = self._kmh_to_mps(v0_kmh)
                v1 = self._kmh_to_mps(v1_kmh)
                slope = (v1 - v0) / full_dt
                x += v0 * dt_seg + 0.5 * slope * dt_seg * dt_seg
            if t_target <= t1:
                break
        if t_target > specs[-1][3]:
            x += self._kmh_to_mps(specs[-1][5]) * (t_target - specs[-1][3])
        return float(x)

    def time_at_x(self, x: float) -> float:
        target_x = max(float(x), 0.0)
        specs = self._segment_specs()
        x0 = 0.0
        for spec in specs:
            _, _, t0, t1, v0_kmh, v1_kmh, *_ = spec
            dt = max(t1 - t0, 1.0e-9)
            v0 = self._kmh_to_mps(v0_kmh)
            v1 = self._kmh_to_mps(v1_kmh)
            slope = (v1 - v0) / dt
            dx = v0 * dt + 0.5 * slope * dt * dt
            if target_x <= x0 + dx:
                local_x = target_x - x0
                if abs(slope) < 1.0e-9:
                    local_t = local_x / max(v0, 1.0e-9)
                else:
                    disc = max(v0 * v0 + 2.0 * slope * local_x, 0.0)
                    root = (-v0 + np.sqrt(disc)) / slope
                    local_t = np.clip(root, 0.0, dt)
                return float(t0 + local_t)
            x0 += dx
        return float(specs[-1][3] + (target_x - x0) / max(self._kmh_to_mps(specs[-1][5]), 1.0e-9))

    def _lane_y(self, x: float) -> float:
        x1 = self.x_at_time(18.8)
        x2 = self.x_at_time(24.8)
        a = float(self.lane_k) * (float(x) - x1)
        b = float(self.lane_k) * (float(x) - x2)
        return float(0.5 * self.lane_shift * (np.tanh(a) - np.tanh(b)))

    def _final_lane_y(self, x: float) -> float:
        x1 = self.x_at_time(44.8)
        x2 = self.x_at_time(50.8)
        a = float(self.final_lane_k) * (float(x) - x1)
        b = float(self.final_lane_k) * (float(x) - x2)
        return float(-0.5 * self.final_lane_shift * (np.tanh(a) - np.tanh(b)))

    def _weave_y(self, x: float) -> float:
        t = self.time_at_x(x)
        gate1 = _smooth_window(t, 0.3, 6.0, 0.6)
        gate2 = _smooth_window(t, 14.0, 18.0, 0.5)
        gate3 = _smooth_window(t, 40.0, 44.0, 0.5)
        k = 2.0 * np.pi / max(float(self.adaptation_wavelength), 1.0e-6)
        return float(self.adaptation_amplitude * (gate1 + gate2 + 0.85 * gate3) * np.sin(k * float(x)))

    def _transition_curve_y(self, x: float) -> float:
        t = self.time_at_x(x)
        gate = _smooth_window(t, 10.0, 14.0, 0.8)
        k = 2.0 * np.pi / max(95.0, 1.0e-6)
        return float(0.35 * gate * np.sin(k * (float(x) - self.x_at_time(10.0))))

    def _chicane_y(self, x: float) -> float:
        t = self.time_at_x(x)
        gate = _smooth_window(t, 26.0, 38.0, 0.9)
        k = 2.0 * np.pi / max(float(self.chicane_wavelength), 1.0e-6)
        return float(self.chicane_amplitude * gate * np.sin(k * (float(x) - self.x_at_time(26.0))))

    def _low_mu_curve_y(self, x: float) -> float:
        t = self.time_at_x(x)
        gate = _smooth_window(t, 34.0, 40.0, 0.8)
        center = self.x_at_time(37.0)
        return float(0.32 * gate * np.tanh((float(x) - center) / 45.0))

    def y(self, x: float) -> float:
        xf = float(x)
        return float(
            self._weave_y(xf)
            + self._transition_curve_y(xf)
            + self._lane_y(xf)
            + self._chicane_y(xf)
            + self._low_mu_curve_y(xf)
            + self._final_lane_y(xf)
        )

    def dy(self, x: float) -> float:
        h = 0.05
        return float((self.y(float(x) + h) - self.y(float(x) - h)) / (2.0 * h))

    def ddy(self, x: float) -> float:
        h = 0.10
        xf = float(x)
        return float((self.y(xf + h) - 2.0 * self.y(xf) + self.y(xf - h)) / (h * h))

    def heading(self, x: float) -> float:
        return float(np.arctan(self.dy(x)))

    def curvature(self, x: float) -> float:
        yp = self.dy(x)
        return float(self.ddy(x) / ((1.0 + yp * yp) ** 1.5))

    def segments(self) -> List[Dict[str, float | str | bool]]:
        rows: List[Dict[str, float | str | bool]] = []
        for spec in self._segment_specs():
            (
                name,
                display,
                start_t,
                end_t,
                speed0,
                speed1,
                mu0,
                mu1,
                color,
                purpose,
                is_train,
                is_adapt,
                is_maneuver,
            ) = spec
            rows.append({
                "segment_name": name,
                "display_name": display,
                "start_time_s": float(start_t),
                "end_time_s": float(end_t),
                "start_x_m": float(self.x_at_time(start_t)),
                "end_x_m": float(self.x_at_time(end_t)),
                "speed_start_kmh": float(speed0),
                "speed_end_kmh": float(speed1),
                "mu_start": float(mu0),
                "mu_end": float(mu1),
                "description": purpose,
                "visualization_color": color,
                "primary_purpose": purpose,
                "is_training_zone": bool(is_train),
                "is_online_adaptation_zone": bool(is_adapt),
                "is_maneuver_zone": bool(is_maneuver),
            })
        return rows

    def friction_piecewise(self) -> List[Dict[str, float]]:
        return [
            {
                "start_time": float(row["start_time_s"]),
                "end_time": float(row["end_time_s"]),
                "mu_start": float(row["mu_start"]),
                "mu_end": float(row["mu_end"]),
            }
            for row in self.segments()
        ]

    def speed_piecewise(self) -> List[Dict[str, float]]:
        return [
            {
                "start_time": float(row["start_time_s"]),
                "end_time": float(row["end_time_s"]),
                "speed_start": self._kmh_to_mps(float(row["speed_start_kmh"])),
                "speed_end": self._kmh_to_mps(float(row["speed_end_kmh"])),
            }
            for row in self.segments()
        ]

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "initial_speed_kmh": float(self.initial_speed_kmh),
            "mid_speed_kmh": float(self.mid_speed_kmh),
            "high_speed_kmh": float(self.high_speed_kmh),
            "exit_speed_kmh": float(self.exit_speed_kmh),
            "total_distance_m": float(self.x_at_time(55.0)),
            "lane_shift_m": float(self.lane_shift),
            "lane_k_1_per_m": float(self.lane_k),
            "chicane_amplitude_m": float(self.chicane_amplitude),
            "chicane_wavelength_m": float(self.chicane_wavelength),
            "adaptation_amplitude_m": float(self.adaptation_amplitude),
            "adaptation_wavelength_m": float(self.adaptation_wavelength),
            "mu_initial": float(self.mu_initial),
            "mu_mid": float(self.mu_mid),
            "mu_chicane": float(self.mu_chicane),
            "mu_low": float(self.mu_low),
            "mu_exit": float(self.mu_exit),
        }


@dataclass
class LateAdaptationGapBoostPath(NonstationaryAdaptiveTechnicalPath):
    """Nonstationary course with a second late-stage plant response shift.

    This variant keeps the award-ready course structure, but adds a distinct
    late friction transition before the low-friction patch and another friction
    shift before the final evasive maneuver. The goal is to give online RLS a
    second adaptation opportunity without changing the VehicleBody dynamics.
    """

    mu_final_recovery: float = 0.72
    final_friction_mode: str = "recovery"
    name: str = "late_adaptation_gap_boost_course"

    @property
    def key_segment_names(self) -> List[str]:
        return [
            "double_lane_change",
            "speed_up_chicane",
            "low_friction_patch",
            "final_evasive_maneuver",
        ]

    def _segment_specs(self) -> List[tuple]:
        s0 = float(self.initial_speed_kmh)
        s1 = float(self.mid_speed_kmh)
        s2 = float(self.high_speed_kmh)
        s3 = float(self.exit_speed_kmh)
        mu0 = float(self.mu_initial)
        mu1 = float(self.mu_mid)
        mu2 = float(self.mu_chicane)
        mu3 = float(self.mu_low)
        mu_final = float(self.mu_final_recovery)
        mode = str(self.final_friction_mode).lower()
        if mode == "second_drop":
            adapt2_mu0 = mu_final
            adapt2_mu1 = mu_final
            final_shift_mu0 = mu_final
            final_shift_mu1 = mu3
            final_mu0 = mu3
            final_mu1 = mu3
            final_desc = "second friction drop before the final maneuver"
        else:
            adapt2_mu0 = mu3
            adapt2_mu1 = mu3
            final_shift_mu0 = mu3
            final_shift_mu1 = mu_final
            final_mu0 = mu_final
            final_mu1 = mu_final
            final_desc = "friction recovery before the final maneuver"
        return [
            (
                "initial_edmd_training",
                "Initial EDMD Training",
                0.0,
                3.5,
                s0,
                s0,
                mu0,
                mu0,
                "#9ecae1",
                "limited high-mu low-speed EDMD data collection",
                True,
                False,
                False,
            ),
            (
                "speed_ramp",
                "Speed Ramp",
                3.5,
                8.0,
                s0,
                43.0,
                mu0,
                mu0,
                "#c6dbef",
                "speed-varying dynamics before the first friction shift",
                False,
                False,
                False,
            ),
            (
                "friction_transition_1",
                "Friction Transition 1",
                8.0,
                12.0,
                43.0,
                43.0,
                mu0,
                mu1,
                "#fdd0a2",
                "first road friction ramp creating plant-model mismatch",
                False,
                False,
                False,
            ),
            (
                "online_adaptation_1",
                "Online Adaptation 1",
                12.0,
                16.0,
                43.0,
                s1,
                mu1,
                mu1,
                "#c7e9c0",
                "first online RLS adaptation opportunity",
                False,
                True,
                False,
            ),
            (
                "double_lane_change",
                "Double Lane Change",
                16.0,
                24.0,
                s1,
                s1,
                mu1,
                mu1,
                "#fdae6b",
                "first key evasive maneuver under changed tire-road response",
                False,
                False,
                True,
            ),
            (
                "speed_up_chicane",
                "Speed-Up Chicane",
                24.0,
                33.0,
                s1,
                s2,
                mu1,
                mu2,
                "#bcbddc",
                "wide chicane with speed increase and friction reduction",
                False,
                False,
                True,
            ),
            (
                "late_friction_transition",
                "Late Friction Transition",
                33.0,
                36.0,
                s2,
                s2,
                mu2,
                mu3,
                "#fcbba1",
                "late-stage friction drop before the low-friction patch",
                False,
                False,
                False,
            ),
            (
                "low_friction_patch",
                "Low-Friction Patch",
                36.0,
                42.0,
                s2,
                s2,
                mu3,
                mu3,
                "#fb6a4a",
                "low-friction nonlinear tire response with moderate curvature",
                False,
                False,
                True,
            ),
            (
                "online_adaptation_2",
                "Online Adaptation 2",
                42.0,
                45.0,
                46.0,
                45.0,
                adapt2_mu0,
                adapt2_mu1,
                "#c7e9c0",
                "second online RLS adaptation opportunity before final shift",
                False,
                True,
                False,
            ),
            (
                "final_friction_recovery_or_shift",
                "Final Friction Shift",
                45.0,
                47.0,
                45.0,
                s3,
                final_shift_mu0,
                final_shift_mu1,
                "#fdd0a2",
                final_desc,
                False,
                False,
                False,
            ),
            (
                "final_evasive_maneuver",
                "Final Evasive Maneuver",
                47.0,
                55.0,
                s3,
                s3,
                final_mu0,
                final_mu1,
                "#fd8d3c",
                "final long evasive maneuver after a late friction shift",
                False,
                False,
                True,
            ),
        ]

    def _lane_y(self, x: float) -> float:
        x1 = self.x_at_time(16.8)
        x2 = self.x_at_time(23.2)
        a = float(self.lane_k) * (float(x) - x1)
        b = float(self.lane_k) * (float(x) - x2)
        return float(0.5 * self.lane_shift * (np.tanh(a) - np.tanh(b)))

    def _final_lane_y(self, x: float) -> float:
        x1 = self.x_at_time(47.6)
        x2 = self.x_at_time(54.0)
        a = float(self.final_lane_k) * (float(x) - x1)
        b = float(self.final_lane_k) * (float(x) - x2)
        return float(-0.5 * self.final_lane_shift * (np.tanh(a) - np.tanh(b)))

    def _weave_y(self, x: float) -> float:
        t = self.time_at_x(x)
        gate1 = _smooth_window(t, 0.3, 3.5, 0.45)
        gate2 = _smooth_window(t, 12.0, 16.0, 0.5)
        gate3 = _smooth_window(t, 42.0, 45.0, 0.45)
        k = 2.0 * np.pi / max(float(self.adaptation_wavelength), 1.0e-6)
        return float(self.adaptation_amplitude * (gate1 + gate2 + 0.9 * gate3) * np.sin(k * float(x)))

    def _transition_curve_y(self, x: float) -> float:
        t = self.time_at_x(x)
        gate1 = _smooth_window(t, 8.0, 12.0, 0.7)
        gate2 = _smooth_window(t, 33.0, 36.0, 0.55)
        k1 = 2.0 * np.pi / max(95.0, 1.0e-6)
        k2 = 2.0 * np.pi / max(80.0, 1.0e-6)
        return float(
            0.34 * gate1 * np.sin(k1 * (float(x) - self.x_at_time(8.0)))
            + 0.22 * gate2 * np.sin(k2 * (float(x) - self.x_at_time(33.0)))
        )

    def _chicane_y(self, x: float) -> float:
        t = self.time_at_x(x)
        gate = _smooth_window(t, 24.0, 42.0, 0.9)
        k = 2.0 * np.pi / max(float(self.chicane_wavelength), 1.0e-6)
        return float(self.chicane_amplitude * gate * np.sin(k * (float(x) - self.x_at_time(24.0))))

    def _low_mu_curve_y(self, x: float) -> float:
        t = self.time_at_x(x)
        gate = _smooth_window(t, 36.0, 42.0, 0.7)
        center = self.x_at_time(39.0)
        return float(0.36 * gate * np.tanh((float(x) - center) / 40.0))

    def as_dict(self) -> Dict[str, float | str]:
        data = super().as_dict()
        data.update({
            "type": type(self).__name__,
            "name": self.name,
            "total_distance_m": float(self.x_at_time(55.0)),
            "mu_final_recovery": float(self.mu_final_recovery),
            "final_friction_mode": str(self.final_friction_mode),
            "final_lane_shift_m": float(self.final_lane_shift),
            "final_lane_k_1_per_m": float(self.final_lane_k),
        })
        return data


def path_parameters(path) -> Dict[str, float | str]:
    if hasattr(path, "as_dict"):
        return dict(path.as_dict())
    if hasattr(path, "__dataclass_fields__"):
        data = asdict(path)
    else:
        data = dict(getattr(path, "__dict__", {}))
    data["type"] = type(path).__name__
    return data


def segment_metadata(path) -> List[Dict[str, float | str | bool]]:
    if hasattr(path, "segments"):
        return list(path.segments())
    return []


def segment_at_time(path, time_s: float) -> Dict[str, float | str | bool]:
    segments = segment_metadata(path)
    if not segments:
        return {
            "segment_name": "",
            "is_training_zone": False,
            "is_online_adaptation_zone": False,
            "is_maneuver_zone": False,
        }
    t = float(time_s)
    for row in segments:
        if float(row["start_time_s"]) <= t < float(row["end_time_s"]):
            return row
    if t < float(segments[0]["start_time_s"]):
        return segments[0]
    return segments[-1]


def segment_at_x(path, x: float) -> Dict[str, float | str | bool]:
    if hasattr(path, "time_at_x"):
        return segment_at_time(path, float(path.time_at_x(float(x))))
    speed = float(getattr(path, "target_speed", 0.0))
    if speed <= 0.0:
        return {
            "segment_name": "",
            "is_training_zone": False,
            "is_online_adaptation_zone": False,
            "is_maneuver_zone": False,
        }
    return segment_at_time(path, float(x) / speed)


def mu_for_segment(row: Mapping, time_s: float) -> float | None:
    if "mu_start" not in row or "mu_end" not in row:
        return None
    t0 = float(row.get("start_time_s", row.get("start_time", 0.0)))
    t1 = max(float(row.get("end_time_s", row.get("end_time", t0))), t0 + 1.0e-9)
    blend = np.clip((float(time_s) - t0) / (t1 - t0), 0.0, 1.0)
    return float((1.0 - blend) * float(row["mu_start"]) + blend * float(row["mu_end"]))


def speed_kmh_for_segment(row: Mapping, time_s: float) -> float | None:
    if "speed_start_kmh" not in row or "speed_end_kmh" not in row:
        return None
    t0 = float(row.get("start_time_s", row.get("start_time", 0.0)))
    t1 = max(float(row.get("end_time_s", row.get("end_time", t0))), t0 + 1.0e-9)
    blend = np.clip((float(time_s) - t0) / (t1 - t0), 0.0, 1.0)
    return float((1.0 - blend) * float(row["speed_start_kmh"]) + blend * float(row["speed_end_kmh"]))


def sample_reference_path(path, x_values: Iterable[float]) -> List[Dict[str, float | str]]:
    rows = []
    params = path_parameters(path)
    name = str(params.get("name", type(path).__name__))
    for x in x_values:
        xf = float(x)
        seg = segment_at_x(path, xf)
        if hasattr(path, "time_at_x"):
            t_ref = float(path.time_at_x(xf))
        else:
            t_ref = xf / max(float(getattr(path, "target_speed", 0.0)), 1.0e-9)
        mu_ref = mu_for_segment(seg, t_ref)
        speed_ref = (
            speed_kmh_for_segment(seg, t_ref)
            if "speed_start_kmh" in seg else (
                3.6 * float(getattr(path, "target_speed", 0.0))
                if float(getattr(path, "target_speed", 0.0)) > 0.0 else None
            )
        )
        rows.append({
            "scenario": name,
            "x": xf,
            "y_ref": float(path.y(xf)),
            "heading_ref": float(path.heading(xf)),
            "curvature_ref": float(path.curvature(xf)),
            "segment_name": str(seg.get("segment_name", "")),
            "mu_ref": "" if mu_ref is None else float(mu_ref),
            "speed_ref_kmh": "" if speed_ref is None else float(speed_ref),
        })
    return rows


@dataclass
class PlantConfig:
    control_dt: float = 0.05
    plant_dt: float = 0.01
    target_speed: float = 8.0
    speed_profile: str = "constant"
    speed_piecewise: Optional[Sequence[Mapping[str, float]]] = None
    drive_kp: float = 30.0
    drive_torque_limit: float = 160.0
    align_initial_heading: bool = True
    friction_profile: str = "constant"
    friction_mu_initial: Optional[float] = None
    friction_mu_final: Optional[float] = None
    friction_change_start_time: float = 0.0
    friction_change_end_time: float = 0.0
    friction_piecewise: Optional[Sequence[Mapping[str, float]]] = None

    @property
    def substeps(self) -> int:
        return max(1, int(round(self.control_dt / self.plant_dt)))


def friction_mu_at_time(cfg: PlantConfig, time: float) -> Optional[float]:
    profile = str(cfg.friction_profile).lower()
    if profile == "piecewise":
        schedule = list(cfg.friction_piecewise or [])
        if not schedule:
            return cfg.friction_mu_final if cfg.friction_mu_final is not None else cfg.friction_mu_initial
        t = float(time)
        for item in schedule:
            t0 = float(item["start_time"])
            t1 = max(float(item["end_time"]), t0 + 1.0e-9)
            if t0 <= t < t1:
                blend = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
                return float((1.0 - blend) * float(item["mu_start"]) + blend * float(item["mu_end"]))
        if t < float(schedule[0]["start_time"]):
            return float(schedule[0]["mu_start"])
        return float(schedule[-1]["mu_end"])
    if cfg.friction_mu_initial is None and cfg.friction_mu_final is None:
        return None
    mu_initial = (
        float(cfg.friction_mu_initial)
        if cfg.friction_mu_initial is not None
        else float(cfg.friction_mu_final)
    )
    mu_final = (
        float(cfg.friction_mu_final)
        if cfg.friction_mu_final is not None
        else mu_initial
    )
    if profile == "constant":
        return mu_final
    if profile == "step":
        return mu_initial if float(time) < float(cfg.friction_change_start_time) else mu_final
    if profile == "ramp":
        t0 = float(cfg.friction_change_start_time)
        t1 = max(float(cfg.friction_change_end_time), t0 + 1.0e-9)
        blend = np.clip((float(time) - t0) / (t1 - t0), 0.0, 1.0)
        return float((1.0 - blend) * mu_initial + blend * mu_final)
    raise ValueError(f"Unsupported friction_profile: {cfg.friction_profile}")


def speed_target_at_time(cfg: PlantConfig, time: float) -> float:
    profile = str(cfg.speed_profile).lower()
    if profile == "piecewise":
        schedule = list(cfg.speed_piecewise or [])
        if not schedule:
            return float(cfg.target_speed)
        t = float(time)
        for item in schedule:
            t0 = float(item["start_time"])
            t1 = max(float(item["end_time"]), t0 + 1.0e-9)
            if t0 <= t < t1:
                blend = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
                return float((1.0 - blend) * float(item["speed_start"]) + blend * float(item["speed_end"]))
        if t < float(schedule[0]["start_time"]):
            return float(schedule[0]["speed_start"])
        return float(schedule[-1]["speed_end"])
    if profile == "constant":
        return float(cfg.target_speed)
    raise ValueError(f"Unsupported speed_profile: {cfg.speed_profile}")


def apply_friction_schedule(vehicle: VehicleBody, cfg: PlantConfig, time: float) -> Optional[float]:
    mu = friction_mu_at_time(cfg, time)
    if mu is None:
        return None
    for corner in vehicle.corners.values():
        corner.lateral_tire.params.mu = float(mu)
        corner.longitudinal_tire.params.mu = float(mu)
    return float(mu)


def tire_state_snapshot(vehicle: VehicleBody) -> Dict[str, float]:
    """Return per-wheel tire states and compact axle aggregates for logging/features."""
    values: Dict[str, float] = {}
    for label in ("FL", "FR", "RR", "RL"):
        corner = vehicle.corners[label]
        values[f"alpha_{label}"] = float(getattr(corner.state, "slip_angle", 0.0))
        values[f"slip_ratio_{label}"] = float(getattr(corner.state, "slip_ratio", 0.0))
        values[f"Fx_{label}"] = float(corner.state.F_x_tire)
        values[f"Fy_{label}"] = float(corner.state.F_y_tire)
        values[f"Fz_{label}"] = float(corner.state.F_z)
        values[f"steering_angle_{label}"] = float(corner.state.steering_angle)

    alpha_fl, alpha_fr = values["alpha_FL"], values["alpha_FR"]
    alpha_rl, alpha_rr = values["alpha_RL"], values["alpha_RR"]
    values.update({
        "alpha_front_mean": 0.5 * (alpha_fl + alpha_fr),
        "alpha_rear_mean": 0.5 * (alpha_rl + alpha_rr),
        "alpha_front_diff": alpha_fl - alpha_fr,
        "alpha_rear_diff": alpha_rl - alpha_rr,
        "Fy_front_sum": values["Fy_FL"] + values["Fy_FR"],
        "Fy_rear_sum": values["Fy_RL"] + values["Fy_RR"],
        "Fz_front_sum": values["Fz_FL"] + values["Fz_FR"],
        "Fz_rear_sum": values["Fz_RL"] + values["Fz_RR"],
        "steering_angle_front_mean": 0.5 * (
            values["steering_angle_FL"] + values["steering_angle_FR"]
        ),
        "mu_current": float(vehicle.corners["FL"].lateral_tire.params.mu),
    })
    return values


def build_controller_context(
    *,
    vehicle: VehicleBody,
    path,
    cfg: PlantConfig,
    base_state: np.ndarray,
    time: float,
    delta_prev: float,
    delta_prev_prev: float,
    yaw_rate_prev_1: float,
    yaw_rate_prev_2: float,
    curvature_prev: float,
    feature_names: Sequence[str] = AUGMENTED_FEATURE_NAMES,
) -> Dict:
    x_pos = float(vehicle.state.x)
    vx = float(vehicle.state.velocity_x)
    curvature = float(path.curvature(x_pos))
    preview_distance = max(abs(vx), 0.1) * cfg.control_dt
    context = {
        "dt": cfg.control_dt,
        "vx": vx,
        "speed_target": speed_target_at_time(cfg, time),
        "speed_target_kmh": 3.6 * speed_target_at_time(cfg, time),
        "mu_current": float(vehicle.corners["FL"].lateral_tire.params.mu),
        "curvature": curvature,
        "curvature_rate": (curvature - float(curvature_prev)) / max(cfg.control_dt, 1.0e-9),
        "curvature_preview_1": float(path.curvature(x_pos + preview_distance)),
        "curvature_preview_2": float(path.curvature(x_pos + 2.0 * preview_distance)),
        "curvature_preview_3": float(path.curvature(x_pos + 3.0 * preview_distance)),
        "delta_prev": float(delta_prev),
        "steering_rate": (float(delta_prev) - float(delta_prev_prev)) / max(cfg.control_dt, 1.0e-9),
        "yaw_rate_prev_1": float(yaw_rate_prev_1),
        "yaw_rate_prev_2": float(yaw_rate_prev_2),
        "vehicle": vehicle,
        "time": time,
    }
    context.update(tire_state_snapshot(vehicle))
    context["koopman_features"] = build_feature_vector(base_state, context, feature_names)
    return context


def create_vehicle_plant(config_path: Optional[str] = None, target_speed: float = 8.0):
    vehicle = VehicleBody(config_path=config_path)
    set_initial_speed(vehicle, target_speed)
    steering = DirectAckermannSteeringWrapper(vehicle)
    return vehicle, steering


def initialize_vehicle_on_path(vehicle: VehicleBody, path, cfg: PlantConfig) -> None:
    if not cfg.align_initial_heading:
        return
    x0 = float(vehicle.state.x)
    vehicle.state.y = float(path.y(x0))
    vehicle.state.yaw = float(path.heading(x0))


def tracking_state(vehicle: VehicleBody, path) -> np.ndarray:
    x_pos = float(vehicle.state.x)
    y_ref = path.y(x_pos)
    psi_ref = path.heading(x_pos)
    dx = 0.0
    dy = float(vehicle.state.y) - y_ref
    e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
    e_psi = wrap_angle(float(vehicle.state.yaw) - psi_ref)
    return np.asarray([
        e_y,
        e_psi,
        float(vehicle.state.velocity_y),
        float(vehicle.state.yaw_rate),
    ], dtype=float)


def build_corner_inputs(vehicle: VehicleBody, drive_torque: float) -> Dict[str, Dict[str, float]]:
    return {
        label: {
            "T_steer": 0.0,
            "T_brk": 0.0,
            "T_Drv": float(drive_torque),
            "T_susp": 0.0,
            "z_road": 0.0,
            "z_road_dot": 0.0,
        }
        for label in vehicle.wheel_labels
    }


def speed_hold_drive_torque(vehicle: VehicleBody, cfg: PlantConfig, time: float = 0.0) -> float:
    err = float(speed_target_at_time(cfg, time)) - float(vehicle.state.velocity_x)
    return float(np.clip(cfg.drive_kp * err, -cfg.drive_torque_limit, cfg.drive_torque_limit))


def step_vehicle(
    vehicle: VehicleBody,
    steering: DirectAckermannSteeringWrapper,
    delta: float,
    cfg: PlantConfig,
    time: float = 0.0,
) -> Dict[str, float]:
    wheel_angles = {}
    for substep in range(cfg.substeps):
        sub_time = float(time) + substep * cfg.plant_dt
        apply_friction_schedule(vehicle, cfg, sub_time)
        drive = speed_hold_drive_torque(vehicle, cfg, sub_time)
        inputs = build_corner_inputs(vehicle, drive)
        wheel_angles = steering.update(cfg.plant_dt, delta, inputs)
    return wheel_angles


def generate_excitation_data(
    *,
    duration: float = 5.0,
    cfg: Optional[PlantConfig] = None,
    path: Optional[object] = None,
    config_path: Optional[str] = None,
    amplitude: float = 0.08,
    excitation_phase: float = 0.0,
    include_curvature_input: bool = False,
    use_augmented_features: bool = False,
    feature_names: Sequence[str] = AUGMENTED_FEATURE_NAMES,
) -> Dict[str, np.ndarray]:
    cfg = cfg or PlantConfig()
    path = path or SinePath()
    vehicle, steering = create_vehicle_plant(config_path, cfg.target_speed)
    initialize_vehicle_on_path(vehicle, path, cfg)
    n_steps = int(round(duration / cfg.control_dt))
    X, U, Xn = [], [], []
    delta_prev = 0.0
    delta_prev_prev = 0.0
    yaw_rate_prev_1 = 0.0
    yaw_rate_prev_2 = 0.0
    curvature_prev = float(path.curvature(float(vehicle.state.x)))
    for k in range(n_steps):
        t = k * cfg.control_dt
        apply_friction_schedule(vehicle, cfg, t)
        delta = amplitude * (
            0.6 * np.sin(2.0 * np.pi * 0.35 * t + excitation_phase)
            + 0.3 * np.sin(2.0 * np.pi * 0.73 * t + 0.4 + 0.5 * excitation_phase)
            + 0.1 * np.sign(np.sin(2.0 * np.pi * 0.17 * t + 0.25 * excitation_phase))
        )
        x0 = tracking_state(vehicle, path)
        context = build_controller_context(
            vehicle=vehicle,
            path=path,
            cfg=cfg,
            base_state=x0,
            time=t,
            delta_prev=delta_prev,
            delta_prev_prev=delta_prev_prev,
            yaw_rate_prev_1=yaw_rate_prev_1,
            yaw_rate_prev_2=yaw_rate_prev_2,
            curvature_prev=curvature_prev,
            feature_names=feature_names,
        )
        step_vehicle(vehicle, steering, float(delta), cfg, time=t)
        x1 = tracking_state(vehicle, path)
        next_context = build_controller_context(
            vehicle=vehicle,
            path=path,
            cfg=cfg,
            base_state=x1,
            time=t + cfg.control_dt,
            delta_prev=float(delta),
            delta_prev_prev=delta_prev,
            yaw_rate_prev_1=float(x0[3]),
            yaw_rate_prev_2=yaw_rate_prev_1,
            curvature_prev=float(context["curvature"]),
            feature_names=feature_names,
        )
        if use_augmented_features:
            X.append(context["koopman_features"])
            U.append(float(delta))
            Xn.append(next_context["koopman_features"])
        else:
            X.append(x0)
            if include_curvature_input:
                U.append([float(delta), float(context["curvature"])])
            else:
                U.append(float(delta))
            Xn.append(x1)
        delta_prev_prev = delta_prev
        delta_prev = float(delta)
        yaw_rate_prev_2 = yaw_rate_prev_1
        yaw_rate_prev_1 = float(x0[3])
        curvature_prev = float(context["curvature"])
    return {
        "X": np.asarray(X, dtype=float),
        "U": np.asarray(U, dtype=float),
        "X_next": np.asarray(Xn, dtype=float),
    }


def generate_multi_path_excitation_data(
    *,
    paths: Sequence[object],
    duration: float = 6.0,
    cfg: Optional[PlantConfig] = None,
    config_path: Optional[str] = None,
    amplitude: float = 0.08,
    include_curvature_input: bool = False,
    use_augmented_features: bool = False,
    feature_names: Sequence[str] = AUGMENTED_FEATURE_NAMES,
) -> Dict[str, np.ndarray]:
    cfg = cfg or PlantConfig()
    paths = list(paths) if paths else [SinePath()]
    duration_per_path = max(cfg.control_dt, float(duration) / len(paths))
    datasets = []
    for idx, path in enumerate(paths):
        datasets.append(generate_excitation_data(
            duration=duration_per_path,
            cfg=cfg,
            path=path,
            config_path=config_path,
            amplitude=amplitude,
            excitation_phase=0.73 * idx,
            include_curvature_input=include_curvature_input,
            use_augmented_features=use_augmented_features,
            feature_names=feature_names,
        ))
    return {
        "X": np.vstack([data["X"] for data in datasets]),
        "U": np.concatenate([data["U"] for data in datasets], axis=0),
        "X_next": np.vstack([data["X_next"] for data in datasets]),
    }


def _nominal_next_state(
    params: BicycleParams,
    x: np.ndarray,
    delta: float,
    context: Mapping,
    curvature_scale: float = 1.0,
) -> np.ndarray:
    A, B, d = discretize_linear_bicycle(
        params,
        vx=float(context.get("vx", 0.0)),
        dt=float(context.get("dt", 0.05)),
        curvature=float(context.get("curvature", 0.0)) * float(curvature_scale),
    )
    return A @ np.asarray(x, dtype=float).reshape(4) + B[:, 0] * float(delta) + d


def generate_closed_loop_training_data(
    *,
    duration: float = 5.0,
    cfg: Optional[PlantConfig] = None,
    path: Optional[object] = None,
    config_path: Optional[str] = None,
    controller=None,
    excitation_amplitude: float = 0.025,
    excitation_phase: float = 0.0,
    feature_names: Sequence[str] = TIRE_AUGMENTED_FEATURE_NAMES,
    residual_params: Optional[BicycleParams] = None,
    residual_curvature_scale: float = 1.0,
) -> Dict[str, np.ndarray | List[Dict]]:
    """Collect limited closed-loop data for direct or residual EDMD fitting."""
    cfg = cfg or PlantConfig()
    path = path or AdaptiveLaneChangePath()
    vehicle, steering = create_vehicle_plant(config_path, cfg.target_speed)
    initialize_vehicle_on_path(vehicle, path, cfg)
    n_steps = int(round(duration / cfg.control_dt))
    X, U, Xn, residuals, rows = [], [], [], [], []
    delta_prev = 0.0
    delta_prev_prev = 0.0
    yaw_rate_prev_1 = 0.0
    yaw_rate_prev_2 = 0.0
    curvature_prev = float(path.curvature(float(vehicle.state.x)))
    for k in range(n_steps):
        t = k * cfg.control_dt
        apply_friction_schedule(vehicle, cfg, t)
        x0 = tracking_state(vehicle, path)
        context = build_controller_context(
            vehicle=vehicle,
            path=path,
            cfg=cfg,
            base_state=x0,
            time=t,
            delta_prev=delta_prev,
            delta_prev_prev=delta_prev_prev,
            yaw_rate_prev_1=yaw_rate_prev_1,
            yaw_rate_prev_2=yaw_rate_prev_2,
            curvature_prev=curvature_prev,
            feature_names=feature_names,
        )
        if controller is not None:
            delta = float(controller.compute_control(x0, delta_prev, context).delta)
        else:
            delta = 0.0
        delta += float(excitation_amplitude) * (
            0.65 * np.sin(2.0 * np.pi * 0.45 * t + excitation_phase)
            + 0.35 * np.sin(2.0 * np.pi * 0.91 * t + 0.4 + excitation_phase)
        )
        delta = float(np.clip(delta, -0.45, 0.45))
        features = np.asarray(context["koopman_features"], dtype=float)
        step_vehicle(vehicle, steering, delta, cfg, time=t)
        x1 = tracking_state(vehicle, path)
        next_context = build_controller_context(
            vehicle=vehicle,
            path=path,
            cfg=cfg,
            base_state=x1,
            time=t + cfg.control_dt,
            delta_prev=delta,
            delta_prev_prev=delta_prev,
            yaw_rate_prev_1=float(x0[3]),
            yaw_rate_prev_2=yaw_rate_prev_1,
            curvature_prev=float(context["curvature"]),
            feature_names=feature_names,
        )
        target_features = np.asarray(next_context["koopman_features"], dtype=float).copy()
        residual = np.zeros(4, dtype=float)
        if residual_params is not None:
            nominal = _nominal_next_state(
                residual_params,
                x0,
                delta,
                context,
                curvature_scale=residual_curvature_scale,
            )
            residual = x1 - nominal
            target_features[:4] = residual
        X.append(features)
        U.append(delta)
        Xn.append(target_features)
        residuals.append(residual)
        row = {
            "time": t,
            "x": float(vehicle.state.x),
            "y": float(vehicle.state.y),
            "vx": float(vehicle.state.velocity_x),
            "speed_kmh": float(3.6 * vehicle.state.velocity_x),
            "speed_target_kmh": float(context["speed_target_kmh"]),
            "e_y": float(x0[0]),
            "e_y_mm": float(1000.0 * x0[0]),
            "abs_e_y_mm": float(1000.0 * abs(x0[0])),
            "e_psi": float(x0[1]),
            "v_y": float(x0[2]),
            "yaw_rate": float(x0[3]),
            "delta": delta,
            "delta_deg": float(np.degrees(delta)),
            "curvature": float(context["curvature"]),
            "residual_e_y": float(residual[0]),
            "residual_e_psi": float(residual[1]),
            "residual_v_y": float(residual[2]),
            "residual_yaw_rate": float(residual[3]),
        }
        seg = segment_at_time(path, t)
        row.update({
            "segment_name": str(seg.get("segment_name", "")),
            "is_training_zone": bool(seg.get("is_training_zone", False)),
            "is_online_adaptation_zone": bool(seg.get("is_online_adaptation_zone", False)),
            "is_maneuver_zone": bool(seg.get("is_maneuver_zone", False)),
        })
        for key, value in tire_state_snapshot(vehicle).items():
            row[key] = float(value)
        rows.append(row)
        delta_prev_prev = delta_prev
        delta_prev = delta
        yaw_rate_prev_2 = yaw_rate_prev_1
        yaw_rate_prev_1 = float(x0[3])
        curvature_prev = float(context["curvature"])
    return {
        "X": np.asarray(X, dtype=float),
        "U": np.asarray(U, dtype=float),
        "X_next": np.asarray(Xn, dtype=float),
        "residuals": np.asarray(residuals, dtype=float),
        "rows": rows,
    }


def generate_multi_path_closed_loop_training_data(
    *,
    paths: Sequence[object],
    duration: float = 6.0,
    cfg: Optional[PlantConfig] = None,
    config_path: Optional[str] = None,
    controller_factory=None,
    excitation_amplitude: float = 0.025,
    feature_names: Sequence[str] = TIRE_AUGMENTED_FEATURE_NAMES,
    residual_params: Optional[BicycleParams] = None,
    residual_curvature_scale: float = 1.0,
) -> Dict[str, np.ndarray | List[Dict]]:
    cfg = cfg or PlantConfig()
    paths = list(paths) if paths else [AdaptiveLaneChangePath()]
    duration_per_path = max(cfg.control_dt, float(duration) / len(paths))
    datasets = []
    for idx, path in enumerate(paths):
        controller = controller_factory() if controller_factory is not None else None
        datasets.append(generate_closed_loop_training_data(
            duration=duration_per_path,
            cfg=cfg,
            path=path,
            config_path=config_path,
            controller=controller,
            excitation_amplitude=excitation_amplitude,
            excitation_phase=0.51 * idx,
            feature_names=feature_names,
            residual_params=residual_params,
            residual_curvature_scale=residual_curvature_scale,
        ))
    return {
        "X": np.vstack([data["X"] for data in datasets]),
        "U": np.concatenate([data["U"] for data in datasets], axis=0),
        "X_next": np.vstack([data["X_next"] for data in datasets]),
        "residuals": np.vstack([data["residuals"] for data in datasets]),
        "rows": [row for data in datasets for row in data["rows"]],
    }


def run_closed_loop(
    controller,
    *,
    duration: float = 5.0,
    cfg: Optional[PlantConfig] = None,
    path: Optional[object] = None,
    config_path: Optional[str] = None,
    feature_names: Sequence[str] = AUGMENTED_FEATURE_NAMES,
) -> Dict:
    cfg = cfg or PlantConfig()
    path = path or SinePath()
    vehicle, steering = create_vehicle_plant(config_path, cfg.target_speed)
    initialize_vehicle_on_path(vehicle, path, cfg)
    n_steps = int(round(duration / cfg.control_dt))
    u_prev = 0.0
    u_prev_prev = 0.0
    yaw_rate_prev_1 = 0.0
    yaw_rate_prev_2 = 0.0
    curvature_prev = float(path.curvature(float(vehicle.state.x)))
    rows: List[Dict] = []
    for k in range(n_steps):
        t = k * cfg.control_dt
        apply_friction_schedule(vehicle, cfg, t)
        x_vec = tracking_state(vehicle, path)
        context = build_controller_context(
            vehicle=vehicle,
            path=path,
            cfg=cfg,
            base_state=x_vec,
            time=t,
            delta_prev=u_prev,
            delta_prev_prev=u_prev_prev,
            yaw_rate_prev_1=yaw_rate_prev_1,
            yaw_rate_prev_2=yaw_rate_prev_2,
            curvature_prev=curvature_prev,
            feature_names=feature_names,
        )
        t0 = perf_counter()
        result = controller.compute_control(x_vec, u_prev, context)
        solve_wall = perf_counter() - t0
        delta = float(result.delta)
        wheel_angles = step_vehicle(vehicle, steering, delta, cfg, time=t)
        x_next = tracking_state(vehicle, path)
        next_context = build_controller_context(
            vehicle=vehicle,
            path=path,
            cfg=cfg,
            base_state=x_next,
            time=t + cfg.control_dt,
            delta_prev=delta,
            delta_prev_prev=u_prev,
            yaw_rate_prev_1=float(x_vec[3]),
            yaw_rate_prev_2=yaw_rate_prev_1,
            curvature_prev=float(context["curvature"]),
            feature_names=feature_names,
        )
        context["next_context"] = next_context
        context["next_koopman_features"] = next_context["koopman_features"]
        update_info = {}
        if hasattr(controller, "observe_transition"):
            update_info = controller.observe_transition(x_vec, delta, x_next, context)
        row = {
            "time": t,
            "x": float(vehicle.state.x),
            "y": float(vehicle.state.y),
            "y_ref": float(path.y(float(vehicle.state.x))),
            "heading_ref": float(path.heading(float(vehicle.state.x))),
            "curvature": float(context["curvature"]),
            "curvature_ref": float(path.curvature(float(vehicle.state.x))),
            "curvature_rate": float(context["curvature_rate"]),
            "curvature_preview_1": float(context["curvature_preview_1"]),
            "curvature_preview_2": float(context["curvature_preview_2"]),
            "curvature_preview_3": float(context["curvature_preview_3"]),
            "yaw": float(vehicle.state.yaw),
            "vx": float(vehicle.state.velocity_x),
            "speed_kmh": float(3.6 * vehicle.state.velocity_x),
            "speed_target_kmh": float(context["speed_target_kmh"]),
            "mu_current": float(context["mu_current"]),
            "e_y": float(x_vec[0]),
            "e_y_mm": float(1000.0 * x_vec[0]),
            "abs_e_y_mm": float(1000.0 * abs(x_vec[0])),
            "e_psi": float(x_vec[1]),
            "v_y": float(x_vec[2]),
            "yaw_rate": float(x_vec[3]),
            "delta": delta,
            "delta_deg": float(np.degrees(delta)),
            "delta_prev": float(context["delta_prev"]),
            "steering_rate": float(context["steering_rate"]),
            "yaw_rate_prev_1": float(context["yaw_rate_prev_1"]),
            "yaw_rate_prev_2": float(context["yaw_rate_prev_2"]),
            "fl_angle": float(wheel_angles.get("FL", 0.0)),
            "fr_angle": float(wheel_angles.get("FR", 0.0)),
            "solve_time": float(result.info.get("solve_time", solve_wall)),
            "solver_status": str(result.info.get("status", "")),
            "prediction_error_before": float(update_info.get("prediction_error_before", np.nan)),
            "prediction_error_after": float(update_info.get("prediction_error_after", np.nan)),
            "residual_error_before": float(update_info.get("residual_error_before", np.nan)),
            "residual_error_after": float(update_info.get("residual_error_after", np.nan)),
            "residual_error_reduction": float(
                update_info.get("residual_error_before", np.nan)
                - update_info.get("residual_error_after", np.nan)
            ),
            "rls_update_accepted": update_info.get("rls_update_accepted", ""),
            "rls_relative_theta_update": float(update_info.get("rls_relative_theta_update", np.nan)),
            "rls_input_jump_rejected": update_info.get("rls_input_jump_rejected", ""),
        }
        seg = segment_at_time(path, t)
        row.update({
            "segment_name": str(seg.get("segment_name", "")),
            "is_training_zone": bool(seg.get("is_training_zone", False)),
            "is_online_adaptation_zone": bool(seg.get("is_online_adaptation_zone", False)),
            "is_maneuver_zone": bool(seg.get("is_maneuver_zone", False)),
        })
        for key, value in tire_state_snapshot(vehicle).items():
            row[key] = float(value)
        rows.append(row)
        u_prev_prev = u_prev
        u_prev = delta
        yaw_rate_prev_2 = yaw_rate_prev_1
        yaw_rate_prev_1 = float(x_vec[3])
        curvature_prev = float(context["curvature"])
    return {"controller": getattr(controller, "name", type(controller).__name__), "rows": rows}


def _segment_metrics(rows: List[Mapping], prefix: str) -> Dict[str, float | None]:
    if not rows:
        return {
            f"{prefix}_lateral_error_rmse": None,
            f"{prefix}_heading_error_rmse": None,
            f"{prefix}_max_abs_lateral_error": None,
        }
    e_y = np.asarray([r["e_y"] for r in rows], dtype=float)
    e_psi = np.asarray([r["e_psi"] for r in rows], dtype=float)
    return {
        f"{prefix}_lateral_error_rmse": float(np.sqrt(np.mean(e_y * e_y))),
        f"{prefix}_heading_error_rmse": float(np.sqrt(np.mean(e_psi * e_psi))),
        f"{prefix}_max_abs_lateral_error": float(np.max(np.abs(e_y))),
    }


def summarize_run(
    run: Mapping,
    *,
    adaptation_time: Optional[float] = None,
    post_friction_change_time: Optional[float] = None,
    maneuver_start_time: Optional[float] = None,
) -> Dict[str, float | str | None]:
    rows = list(run["rows"])
    e_y = np.asarray([r["e_y"] for r in rows], dtype=float)
    e_psi = np.asarray([r["e_psi"] for r in rows], dtype=float)
    delta = np.asarray([r["delta"] for r in rows], dtype=float)
    solve = np.asarray([r["solve_time"] for r in rows], dtype=float)
    ddelta = np.diff(delta, prepend=delta[0])
    pred_before = np.asarray([r["prediction_error_before"] for r in rows], dtype=float)
    pred_after = np.asarray([r["prediction_error_after"] for r in rows], dtype=float)
    residual_before = np.asarray([r.get("residual_error_before", np.nan) for r in rows], dtype=float)
    residual_after = np.asarray([r.get("residual_error_after", np.nan) for r in rows], dtype=float)
    finite_before = pred_before[np.isfinite(pred_before)]
    finite_pred = pred_after[np.isfinite(pred_after)]
    finite_res_before = residual_before[np.isfinite(residual_before)]
    finite_res_after = residual_after[np.isfinite(residual_after)]
    accepted = [r["rls_update_accepted"] for r in rows]
    accepted_count = sum(v is True or str(v).lower() == "true" for v in accepted)
    rejected_count = sum(v is False or str(v).lower() == "false" for v in accepted)
    rls_count = accepted_count + rejected_count
    rel_updates = np.asarray([r.get("rls_relative_theta_update", np.nan) for r in rows], dtype=float)
    finite_rel_updates = rel_updates[np.isfinite(rel_updates)]
    summary = {
        "controller": str(run["controller"]),
        "samples": len(rows),
        "lateral_error_rmse": float(np.sqrt(np.mean(e_y * e_y))),
        "heading_error_rmse": float(np.sqrt(np.mean(e_psi * e_psi))),
        "max_abs_lateral_error": float(np.max(np.abs(e_y))),
        "steering_effort_rms": float(np.sqrt(np.mean(delta * delta))),
        "steering_smoothness_rms": float(np.sqrt(np.mean(ddelta * ddelta))),
        "mean_solve_time_s": float(np.mean(solve)),
        "max_solve_time_s": float(np.max(solve)),
        "prediction_error_before_mean": float(np.mean(finite_before)) if finite_before.size else None,
        "prediction_error_after_mean": float(np.mean(finite_pred)) if finite_pred.size else None,
        "prediction_error_improvement_percent": (
            float(100.0 * (np.mean(finite_before) - np.mean(finite_pred)) / max(np.mean(finite_before), 1.0e-12))
            if finite_before.size and finite_pred.size else None
        ),
        "residual_prediction_error_before_mean": float(np.mean(finite_res_before)) if finite_res_before.size else None,
        "residual_prediction_error_after_mean": float(np.mean(finite_res_after)) if finite_res_after.size else None,
        "residual_prediction_error_improvement_percent": (
            float(100.0 * (np.mean(finite_res_before) - np.mean(finite_res_after)) / max(np.mean(finite_res_before), 1.0e-12))
            if finite_res_before.size and finite_res_after.size else None
        ),
        "rls_acceptance_rate": float(accepted_count / rls_count) if rls_count else None,
        "rls_rejection_rate": float(rejected_count / rls_count) if rls_count else None,
        "rls_relative_theta_update_mean": float(np.mean(finite_rel_updates)) if finite_rel_updates.size else None,
    }
    if adaptation_time is not None:
        post_rows = [r for r in rows if float(r["time"]) >= float(adaptation_time)]
        summary.update(_segment_metrics(post_rows, "post_adaptation"))
    if post_friction_change_time is not None:
        post_friction_rows = [r for r in rows if float(r["time"]) >= float(post_friction_change_time)]
        summary.update(_segment_metrics(post_friction_rows, "post_friction_change"))
    if maneuver_start_time is not None:
        maneuver_rows = [r for r in rows if float(r["time"]) >= float(maneuver_start_time)]
        metrics = _segment_metrics(maneuver_rows, "maneuver_segment")
        summary.update(metrics)
    return summary


def _detailed_metrics(rows: List[Mapping]) -> Dict[str, float | None]:
    if not rows:
        return {
            "samples": 0,
            "lateral_error_rmse": None,
            "heading_error_rmse": None,
            "max_abs_lateral_error": None,
            "steering_effort_rms": None,
            "steering_smoothness_rms": None,
            "mean_solve_time_s": None,
            "residual_prediction_error_before_mean": None,
            "residual_prediction_error_after_mean": None,
            "residual_prediction_error_improvement_percent": None,
            "residual_error_reduction_mean": None,
            "residual_error_reduction_positive_ratio": None,
            "rls_acceptance_rate": None,
            "rls_relative_theta_update_mean": None,
        }
    e_y = np.asarray([r["e_y"] for r in rows], dtype=float)
    e_psi = np.asarray([r["e_psi"] for r in rows], dtype=float)
    delta = np.asarray([r["delta"] for r in rows], dtype=float)
    solve = np.asarray([r.get("solve_time", np.nan) for r in rows], dtype=float)
    ddelta = np.diff(delta, prepend=delta[0])
    residual_before = np.asarray([r.get("residual_error_before", np.nan) for r in rows], dtype=float)
    residual_after = np.asarray([r.get("residual_error_after", np.nan) for r in rows], dtype=float)
    finite_res_before = residual_before[np.isfinite(residual_before)]
    finite_res_after = residual_after[np.isfinite(residual_after)]
    residual_reduction = residual_before - residual_after
    finite_res_reduction = residual_reduction[np.isfinite(residual_reduction)]
    accepted = [r.get("rls_update_accepted", "") for r in rows]
    accepted_count = sum(v is True or str(v).lower() == "true" for v in accepted)
    rejected_count = sum(v is False or str(v).lower() == "false" for v in accepted)
    rls_count = accepted_count + rejected_count
    rel_updates = np.asarray([r.get("rls_relative_theta_update", np.nan) for r in rows], dtype=float)
    finite_rel_updates = rel_updates[np.isfinite(rel_updates)]
    return {
        "samples": len(rows),
        "lateral_error_rmse": float(np.sqrt(np.mean(e_y * e_y))),
        "heading_error_rmse": float(np.sqrt(np.mean(e_psi * e_psi))),
        "max_abs_lateral_error": float(np.max(np.abs(e_y))),
        "steering_effort_rms": float(np.sqrt(np.mean(delta * delta))),
        "steering_smoothness_rms": float(np.sqrt(np.mean(ddelta * ddelta))),
        "mean_solve_time_s": float(np.nanmean(solve)),
        "residual_prediction_error_before_mean": float(np.mean(finite_res_before)) if finite_res_before.size else None,
        "residual_prediction_error_after_mean": float(np.mean(finite_res_after)) if finite_res_after.size else None,
        "residual_prediction_error_improvement_percent": (
            float(100.0 * (np.mean(finite_res_before) - np.mean(finite_res_after)) / max(np.mean(finite_res_before), 1.0e-12))
            if finite_res_before.size and finite_res_after.size else None
        ),
        "residual_error_reduction_mean": (
            float(np.mean(finite_res_reduction)) if finite_res_reduction.size else None
        ),
        "residual_error_reduction_positive_ratio": (
            float(np.mean(finite_res_reduction > 0.0)) if finite_res_reduction.size else None
        ),
        "rls_acceptance_rate": float(accepted_count / rls_count) if rls_count else None,
        "rls_relative_theta_update_mean": float(np.mean(finite_rel_updates)) if finite_rel_updates.size else None,
    }


def segment_metrics_for_run(run: Mapping, segments: Sequence[Mapping]) -> Dict[str, Dict[str, float | None]]:
    rows = list(run["rows"])
    output: Dict[str, Dict[str, float | None]] = {}
    for segment in segments:
        name = str(segment["segment_name"])
        start = float(segment["start_time_s"])
        end = float(segment["end_time_s"])
        segment_rows = [r for r in rows if start <= float(r["time"]) < end]
        output[name] = _detailed_metrics(segment_rows)
    return output


def combined_segment_metrics_for_run(
    run: Mapping,
    segments: Sequence[Mapping],
    segment_names: Sequence[str],
) -> Dict[str, float | None]:
    rows = list(run["rows"])
    selected = {str(name) for name in segment_names}
    windows = [
        (float(segment["start_time_s"]), float(segment["end_time_s"]))
        for segment in segments
        if str(segment["segment_name"]) in selected
    ]
    combined_rows = [
        r for r in rows
        if any(start <= float(r["time"]) < end for start, end in windows)
    ]
    return _detailed_metrics(combined_rows)


def write_run_csv(path: str | Path, rows: List[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
