"""Runtime wrapper for direct Ackermann steering-angle input.

This module does not modify the vehicle model source path. It patches the
steering update method on a given VehicleBody instance so front wheel angles are
set directly from a single bicycle-model steering angle, while rear steering is
held fixed.
"""

from __future__ import annotations

from types import MethodType
from typing import Dict, Mapping

import numpy as np


WheelAngleMap = Dict[str, float]


class DirectAckermannSteeringWrapper:
    """Apply direct Ackermann wheel angles to an existing VehicleBody instance.

    Args:
        vehicle: Existing VehicleBody instance.
        rear_angle: Fixed rear steering angle [rad].
        apply_limits: If True, each wheel angle is clipped by SteeringModel
            angle limits before being applied.

    Ackermann geometry is taken from ``vehicle.corner_offsets``, which is
    already loaded from the vehicle parameter YAML by VehicleBody.

    The wrapper bypasses steering torque actuator dynamics only for the wrapped
    vehicle instance. Drive, brake, suspension, tire, and body dynamics still run
    through the original model code.
    """

    def __init__(
        self,
        vehicle,
        *,
        rear_angle: float = 0.0,
        apply_limits: bool = True,
    ) -> None:
        self.vehicle = vehicle
        self.wheelbase = self._infer_wheelbase(vehicle)
        self.track = self._infer_front_track(vehicle)
        self.rear_angle = float(rear_angle)
        self.apply_limits = bool(apply_limits)

        if self.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if self.track < 0.0:
            raise ValueError("track must be non-negative")

        self._original_updates = {}
        self._angle_cmd: WheelAngleMap = {label: 0.0 for label in self.vehicle.wheel_labels}
        self.enable()

    @staticmethod
    def _infer_wheelbase(vehicle) -> float:
        offsets = vehicle.corner_offsets
        front_x = 0.5 * (float(offsets["FL"]["x"]) + float(offsets["FR"]["x"]))
        rear_x = 0.5 * (float(offsets["RL"]["x"]) + float(offsets["RR"]["x"]))
        return abs(front_x - rear_x)

    @staticmethod
    def _infer_front_track(vehicle) -> float:
        offsets = vehicle.corner_offsets
        return abs(float(offsets["FL"]["y"]) - float(offsets["FR"]["y"]))

    def enable(self) -> None:
        """Enable direct steering angle injection for this vehicle instance."""
        for label in self.vehicle.wheel_labels:
            steering = self.vehicle.corners[label].steering
            if label not in self._original_updates:
                self._original_updates[label] = steering.update

            def direct_update(steering_self, dt, T_str, T_align=0.0, *, wheel_label=label):
                prev_angle = steering_self.state.steering_angle
                next_angle = float(self._angle_cmd.get(wheel_label, 0.0))
                if self.apply_limits:
                    next_angle = steering_self.apply_angle_limits(next_angle)

                steering_self.state.steering_angle = next_angle
                steering_self.state.steering_rate = (
                    (next_angle - prev_angle) / dt if dt > 0.0 else 0.0
                )
                steering_self.state.steering_torque = 0.0
                steering_self.state.self_aligning_torque = T_align
                return steering_self.state.steering_angle

            steering.update = MethodType(direct_update, steering)

    def disable(self) -> None:
        """Restore original steering torque actuator updates."""
        for label, original_update in self._original_updates.items():
            self.vehicle.corners[label].steering.update = original_update

    def compute_wheel_angles(self, steering_angle: float) -> WheelAngleMap:
        """Convert one bicycle-model steering angle to per-wheel angles."""
        delta = float(steering_angle)
        if abs(delta) < 1e-12:
            front_left = 0.0
            front_right = 0.0
        else:
            if abs(np.cos(delta)) < 1e-12:
                raise ValueError("steering_angle is too close to +/- pi/2 for Ackermann geometry")
            curvature = np.tan(delta) / self.wheelbase
            half_track = 0.5 * self.track
            numerator = self.wheelbase * curvature
            left_denom = 1.0 - curvature * half_track
            right_denom = 1.0 + curvature * half_track
            front_left = float(np.arctan2(numerator, left_denom))
            front_right = float(np.arctan2(numerator, right_denom))

        return {
            "FL": front_left,
            "FR": front_right,
            "RL": self.rear_angle,
            "RR": self.rear_angle,
        }

    def set_steering_angle(self, steering_angle: float) -> WheelAngleMap:
        """Set the single steering command [rad] for the next vehicle update."""
        self._angle_cmd = self.compute_wheel_angles(steering_angle)
        return dict(self._angle_cmd)

    def get_wheel_angles(self) -> WheelAngleMap:
        """Return the latest commanded wheel angles."""
        return dict(self._angle_cmd)

    def update(
        self,
        dt: float,
        steering_angle: float,
        corner_inputs: Mapping[str, Mapping[str, float]],
        *,
        direction: int = 1,
    ) -> WheelAngleMap:
        """Set Ackermann angle command and step the wrapped vehicle."""
        wheel_angles = self.set_steering_angle(steering_angle)
        self.vehicle.update(dt, corner_inputs, direction=direction)
        return wheel_angles


def enable_direct_ackermann_steering(vehicle, **kwargs) -> DirectAckermannSteeringWrapper:
    """Convenience factory for DirectAckermannSteeringWrapper."""
    return DirectAckermannSteeringWrapper(vehicle, **kwargs)
