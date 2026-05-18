"""EDMD-Koopman path-tracking MPC benchmark on the VehicleBody plant.

The plant is always VehicleBody + ECorner 4-wheel dynamics. Linear bicycle and
Koopman models are used only inside MPC prediction.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import yaml

from vehicle_sim.controllers.path_tracking_mpc import (
    AUGMENTED_FEATURE_NAMES,
    BASE_FEATURE_NAMES,
    TIRE_AUGMENTED_FEATURE_NAMES,
    BicycleModelScales,
    EDMDConfig,
    KoopmanMPCController,
    LinearBicycleMPCController,
    MPCConfig,
    OnlineKoopmanMPCController,
    OnlineResidualKoopmanMPCController,
    ResidualKoopmanMPCController,
    fit_edmd_koopman,
)
from vehicle_sim.controllers.path_tracking_mpc.edmd import prediction_errors
from vehicle_sim.controllers.path_tracking_mpc.linear_bicycle import bicycle_params_from_vehicle
from vehicle_sim.controllers.path_tracking_mpc.rls import RLSConfig
from vehicle_sim.utils.path_tracking_sim import (
    AdaptiveLaneChangePath,
    AggressiveSinePath,
    CompositeFrictionAdaptationPath,
    DoubleLaneChangePath,
    LateAdaptationGapBoostPath,
    NonstationaryAdaptiveTechnicalPath,
    PlantConfig,
    SinePath,
    combined_segment_metrics_for_run,
    create_vehicle_plant,
    generate_multi_path_closed_loop_training_data,
    generate_multi_path_excitation_data,
    path_parameters,
    run_closed_loop,
    sample_reference_path,
    segment_metadata,
    segment_metrics_for_run,
    summarize_run,
    write_json,
    write_run_csv,
)


PLANT_DESCRIPTION = (
    "VehicleBody + ECorner 4-wheel dynamics with DirectAckermannSteeringWrapper; "
    "single bicycle steering command delta [rad] is directly mapped to FL/FR "
    "Ackermann wheel angles and rear steering is fixed."
)


@dataclass
class ScenarioSpec:
    name: str
    path: object
    default_eval_duration: float
    description: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke-sized comparison.")
    parser.add_argument(
        "--scenario",
        default="mild_sine",
        choices=[
            "mild_sine",
            "aggressive_sine",
            "double_lane_change",
            "adaptive_lane_change",
            "nonlinear_adaptive_lane_change",
            "low_mu_fiala_adaptive_lane_change",
            "friction_shift_adaptive_lane_change",
            "online_advantage_friction_shift_lane_change",
            "composite_friction_adaptation_course",
            "nonstationary_adaptive_technical_course",
            "late_adaptation_gap_boost_course",
            "all",
        ],
        help="Path scenario to evaluate.",
    )
    parser.add_argument("--out-dir", default="vehicle_sim/experiments/results/tire_informed_residual_koopman_benchmark")
    parser.add_argument("--train-duration", type=float, default=None)
    parser.add_argument("--eval-duration", type=float, default=None)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--plant-dt", type=float, default=0.01)
    parser.add_argument("--target-speed", type=float, default=9.5)
    parser.add_argument("--speed-profile", choices=["constant", "piecewise"], default="constant")
    parser.add_argument("--initial-speed-kmh", type=float, default=36.0)
    parser.add_argument("--mid-speed-kmh", type=float, default=45.0)
    parser.add_argument("--high-speed-kmh", type=float, default=50.0)
    parser.add_argument("--exit-speed-kmh", type=float, default=40.0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--path-amplitude", type=float, default=0.7)
    parser.add_argument("--path-wavelength", type=float, default=35.0)
    parser.add_argument("--aggressive-amplitude", type=float, default=1.5)
    parser.add_argument("--aggressive-wavelength", type=float, default=22.0)
    parser.add_argument("--lane-shift", type=float, default=2.8)
    parser.add_argument("--lane-x1", type=float, default=20.0)
    parser.add_argument("--lane-x2", type=float, default=54.0)
    parser.add_argument("--lane-k", type=float, default=0.22)
    parser.add_argument("--adaptation-time", type=float, default=3.0)
    parser.add_argument("--adaptation-weave-amplitude", type=float, default=0.22)
    parser.add_argument("--adaptation-weave-wavelength", type=float, default=32.0)
    parser.add_argument("--chicane-amplitude", type=float, default=0.9)
    parser.add_argument("--chicane-wavelength", type=float, default=38.0)
    parser.add_argument("--chicane-mu", type=float, default=0.68)
    parser.add_argument("--second-patch-mu", type=float, default=0.65)
    parser.add_argument("--mu-final-recovery", type=float, default=0.72)
    parser.add_argument("--final-friction-mode", choices=["recovery", "second_drop"], default="recovery")
    parser.add_argument("--maneuver-start-time", type=float, default=None)
    parser.add_argument("--maneuver-start-x", type=float, default=None)
    parser.add_argument("--use-augmented-koopman", action="store_true")
    parser.add_argument("--use-tire-augmented-koopman", action="store_true")
    parser.add_argument(
        "--koopman-mode",
        choices=["direct_output", "residual_output"],
        default="direct_output",
    )
    parser.add_argument(
        "--training-policy",
        choices=["excitation", "safe_linear_mpc_adaptation", "small_sine_plus_linear_mpc"],
        default="excitation",
    )
    parser.add_argument("--tire-model", choices=["linear", "linear_saturation", "fiala"], default="linear")
    parser.add_argument("--friction-mu", type=float, default=None)
    parser.add_argument("--friction-profile", choices=["constant", "step", "ramp", "piecewise"], default="constant")
    parser.add_argument("--friction-mu-initial", type=float, default=None)
    parser.add_argument("--friction-mu-final", type=float, default=None)
    parser.add_argument("--friction-change-start-time", type=float, default=1.5)
    parser.add_argument("--friction-change-end-time", type=float, default=3.0)
    parser.add_argument("--nonlinear-plant-benchmark", action="store_true")
    parser.add_argument("--excitation-amplitude", type=float, default=0.22)
    parser.add_argument("--edmd-ridge", type=float, default=1.0e-5)
    parser.add_argument("--rls-forgetting-factor", type=float, default=0.999)
    parser.add_argument("--rls-p0", type=float, default=0.01)
    parser.add_argument("--rls-max-relative-theta-update", type=float, default=0.01)
    parser.add_argument("--rls-max-post-update-control-change", type=float, default=0.12)
    parser.add_argument("--rls-reject-worse-factor", type=float, default=1.0)
    parser.add_argument("--rls-reject-worse-margin", type=float, default=1.0e-5)
    parser.add_argument("--run-small-sweep", action="store_true")
    parser.add_argument("--sweep-limit", type=int, default=8)
    parser.add_argument("--search-online-advantage", action="store_true")
    parser.add_argument("--search-max-candidates", type=int, default=40)
    parser.add_argument("--search-resume", action="store_true")
    parser.add_argument("--search-seed", type=int, default=7)
    parser.add_argument("--search-target-online-fixed-improvement", type=float, default=0.10)
    parser.add_argument(
        "--search-continue-after-success",
        action="store_true",
        help="Continue the candidate search even after the target ranking is found.",
    )
    parser.add_argument("--search-composite-course", action="store_true")
    parser.add_argument("--composite-search-max-candidates", type=int, default=24)
    parser.add_argument("--composite-search-resume", action="store_true")
    parser.add_argument("--composite-search-seed", type=int, default=11)
    parser.add_argument("--composite-target-online-fixed-improvement", type=float, default=0.05)
    parser.add_argument("--composite-target-online-linear-improvement", type=float, default=0.15)
    parser.add_argument("--composite-search-continue-after-success", action="store_true")
    parser.add_argument("--search-award-ready", action="store_true")
    parser.add_argument("--search-late-gap-boost", action="store_true")
    parser.add_argument(
        "--gap-boost",
        action="store_true",
        help="Use stronger but guarded award-ready search ranges to amplify Online-vs-Fixed differences.",
    )
    parser.add_argument("--target-online-best-baseline-improvement", type=float, default=0.05)
    parser.add_argument("--target-late-segment-improvement", type=float, default=0.05)
    parser.add_argument(
        "--continue-after-success",
        dest="search_continue_after_success",
        action="store_true",
        help="Continue award-ready search after finding an acceptable candidate.",
    )
    parser.add_argument(
        "--training-scope",
        default="scenario",
        choices=["scenario", "benchmark"],
        help=(
            "scenario trains each Koopman model on limited data from the evaluated path; "
            "benchmark trains on the full benchmark path set."
        ),
    )
    parser.add_argument(
        "--linear-cf-scale",
        type=float,
        default=0.60,
        help="Front cornering stiffness scale used only inside the linear MPC prediction model.",
    )
    parser.add_argument(
        "--linear-cr-scale",
        type=float,
        default=0.60,
        help="Rear cornering stiffness scale used only inside the linear MPC prediction model.",
    )
    parser.add_argument(
        "--linear-mass-scale",
        type=float,
        default=1.05,
        help="Mass scale used only inside the linear MPC prediction model.",
    )
    parser.add_argument(
        "--linear-izz-scale",
        type=float,
        default=1.15,
        help="Yaw inertia scale used only inside the linear MPC prediction model.",
    )
    parser.add_argument(
        "--linear-curvature-scale",
        type=float,
        default=1.0,
        help="Reference curvature feed-forward scale used only inside the linear MPC prediction model.",
    )
    return parser


def build_scenarios(args) -> Dict[str, ScenarioSpec]:
    maneuver_x1 = (
        float(args.maneuver_start_x)
        if args.maneuver_start_x is not None
        else float(args.target_speed) * float(args.adaptation_time) + 4.0
    )
    maneuver_x2 = maneuver_x1 + 32.0
    scenarios = {
        "mild_sine": ScenarioSpec(
            name="mild_sine",
            path=SinePath(
                amplitude=args.path_amplitude,
                wavelength=args.path_wavelength,
                name="mild_sine",
            ),
            default_eval_duration=5.0,
            description="Low-curvature sine path used as the nominal baseline.",
        ),
        "aggressive_sine": ScenarioSpec(
            name="aggressive_sine",
            path=AggressiveSinePath(
                amplitude=args.aggressive_amplitude,
                wavelength=args.aggressive_wavelength,
                name="aggressive_sine",
            ),
            default_eval_duration=7.0,
            description="High-curvature sine path that induces stronger lateral and yaw dynamics.",
        ),
        "double_lane_change": ScenarioSpec(
            name="double_lane_change",
            path=DoubleLaneChangePath(
                lane_shift=args.lane_shift,
                x1=args.lane_x1,
                x2=args.lane_x2,
                k=args.lane_k,
                name="double_lane_change",
            ),
            default_eval_duration=8.0,
            description="ISO-like evasive maneuver with lane change and return transient response.",
        ),
        "adaptive_lane_change": ScenarioSpec(
            name="adaptive_lane_change",
            path=AdaptiveLaneChangePath(
                lane_shift=max(float(args.lane_shift), 3.0),
                x1=maneuver_x1,
                x2=maneuver_x2,
                k=min(max(float(args.lane_k), 0.18), 0.25),
                name="adaptive_lane_change",
            ),
            default_eval_duration=9.0,
            description=(
                "Mild adaptation segment followed by a moderate double lane change, "
                "designed to test online model adaptation after limited initial data."
            ),
        ),
    }
    scenarios["nonlinear_adaptive_lane_change"] = ScenarioSpec(
        name="nonlinear_adaptive_lane_change",
        path=AdaptiveLaneChangePath(
            lane_shift=max(float(args.lane_shift), 3.0),
            x1=maneuver_x1,
            x2=maneuver_x2,
            k=min(max(float(args.lane_k), 0.18), 0.25),
            name="nonlinear_adaptive_lane_change",
        ),
        default_eval_duration=9.0,
        description=(
            "Adaptive lane change intended for nonlinear tire plant evaluation."
        ),
    )
    scenarios["low_mu_fiala_adaptive_lane_change"] = ScenarioSpec(
        name="low_mu_fiala_adaptive_lane_change",
        path=AdaptiveLaneChangePath(
            lane_shift=max(float(args.lane_shift), 3.0),
            x1=maneuver_x1,
            x2=maneuver_x2,
            k=min(max(float(args.lane_k), 0.20), 0.25),
            adaptation_amplitude=0.22,
            adaptation_wavelength=30.0,
            adaptation_decay_x=max(maneuver_x1 + 2.0, 32.0),
            name="low_mu_fiala_adaptive_lane_change",
        ),
        default_eval_duration=10.0,
        description=(
            "Low-friction Fiala tire adaptive lane change with a mild adaptation segment "
            "followed by a learnable evasive maneuver."
        ),
    )
    friction_x1 = (
        float(args.maneuver_start_x)
        if args.maneuver_start_x is not None
        else float(args.target_speed) * 4.2
    )
    scenarios["friction_shift_adaptive_lane_change"] = ScenarioSpec(
        name="friction_shift_adaptive_lane_change",
        path=AdaptiveLaneChangePath(
            lane_shift=max(float(args.lane_shift), 3.1),
            x1=friction_x1,
            x2=friction_x1 + 35.0,
            k=min(max(float(args.lane_k), 0.20), 0.25),
            adaptation_amplitude=float(args.adaptation_weave_amplitude),
            adaptation_wavelength=float(args.adaptation_weave_wavelength),
            adaptation_decay_x=friction_x1 - 2.0,
            name="friction_shift_adaptive_lane_change",
        ),
        default_eval_duration=11.0,
        description=(
            "Friction-ramp adaptive lane change: offline training is limited to the "
            "initial high-mu response, then the evaluation plant transitions to lower "
            "friction before the maneuver so Online RLS can adapt."
        ),
    )
    online_maneuver_time = (
        float(args.maneuver_start_time)
        if args.maneuver_start_time is not None
        else max(float(args.friction_change_end_time) + float(args.adaptation_time), 4.5)
    )
    online_x1 = (
        float(args.maneuver_start_x)
        if args.maneuver_start_x is not None
        else float(args.target_speed) * online_maneuver_time
    )
    scenarios["online_advantage_friction_shift_lane_change"] = ScenarioSpec(
        name="online_advantage_friction_shift_lane_change",
        path=AdaptiveLaneChangePath(
            lane_shift=max(float(args.lane_shift), 3.0),
            x1=online_x1,
            x2=online_x1 + max(32.0, 10.0 * float(args.lane_shift)),
            k=min(max(float(args.lane_k), 0.18), 0.28),
            adaptation_amplitude=float(args.adaptation_weave_amplitude),
            adaptation_wavelength=float(args.adaptation_weave_wavelength),
            adaptation_decay_x=online_x1 - 2.0,
            name="online_advantage_friction_shift_lane_change",
        ),
        default_eval_duration=11.5,
        description=(
            "Search scenario for online RLS benefit: high-mu limited offline training, "
            "evaluation friction ramp, adaptation weave, then a double lane-change maneuver."
        ),
    )
    scenarios["composite_friction_adaptation_course"] = ScenarioSpec(
        name="composite_friction_adaptation_course",
        path=CompositeFrictionAdaptationPath(
            target_speed=float(args.target_speed),
            lane_shift=max(float(args.lane_shift), 2.8),
            lane_k=min(max(float(args.lane_k), 0.18), 0.23),
            chicane_amplitude=float(args.chicane_amplitude),
            chicane_wavelength=float(args.chicane_wavelength),
            adaptation_amplitude=float(args.adaptation_weave_amplitude),
            adaptation_wavelength=float(args.adaptation_weave_wavelength),
            mu_initial=float(args.friction_mu_initial if args.friction_mu_initial is not None else 0.85),
            first_mu_final=float(args.friction_mu_final if args.friction_mu_final is not None else 0.70),
            second_patch_mu=float(args.second_patch_mu),
            name="composite_friction_adaptation_course",
        ),
        default_eval_duration=27.0,
        description=(
            "Long F1-like open technical course with initial EDMD training, friction "
            "transition, online adaptation, double lane change, chicane, low-friction "
            "patch, and exit recovery segments."
        ),
    )
    scenarios["nonstationary_adaptive_technical_course"] = ScenarioSpec(
        name="nonstationary_adaptive_technical_course",
        path=NonstationaryAdaptiveTechnicalPath(
            initial_speed_kmh=float(args.initial_speed_kmh),
            mid_speed_kmh=float(args.mid_speed_kmh),
            high_speed_kmh=float(args.high_speed_kmh),
            exit_speed_kmh=float(args.exit_speed_kmh),
            mu_initial=float(args.friction_mu_initial if args.friction_mu_initial is not None else 0.90),
            mu_mid=float(args.friction_mu_final if args.friction_mu_final is not None else 0.72),
            mu_chicane=float(args.chicane_mu),
            mu_low=float(args.second_patch_mu),
            mu_exit=0.80,
            lane_shift=max(float(args.lane_shift), 3.0),
            lane_k=min(max(float(args.lane_k), 0.10), 0.22),
            chicane_amplitude=float(args.chicane_amplitude),
            chicane_wavelength=float(args.chicane_wavelength),
            adaptation_amplitude=float(args.adaptation_weave_amplitude),
            adaptation_wavelength=float(args.adaptation_weave_wavelength),
            name="nonstationary_adaptive_technical_course",
        ),
        default_eval_duration=55.0,
        description=(
            "Long nonstationary technical course with speed ramps, friction ramps, "
            "limited high-mu EDMD training, online adaptation windows, lane changes, "
            "wide chicane, and low-friction patch key evaluation segments."
        ),
    )
    final_friction_mode = getattr(args, "final_friction_mode", "recovery")
    scenarios["late_adaptation_gap_boost_course"] = ScenarioSpec(
        name="late_adaptation_gap_boost_course",
        path=LateAdaptationGapBoostPath(
            initial_speed_kmh=float(args.initial_speed_kmh),
            mid_speed_kmh=float(args.mid_speed_kmh),
            high_speed_kmh=float(args.high_speed_kmh),
            exit_speed_kmh=float(args.exit_speed_kmh),
            mu_initial=float(args.friction_mu_initial if args.friction_mu_initial is not None else 0.90),
            mu_mid=float(args.friction_mu_final if args.friction_mu_final is not None else 0.72),
            mu_chicane=float(args.chicane_mu),
            mu_low=float(args.second_patch_mu),
            mu_exit=0.80,
            mu_final_recovery=float(getattr(args, "mu_final_recovery", 0.72)),
            final_friction_mode=str(final_friction_mode),
            lane_shift=max(float(args.lane_shift), 3.2),
            lane_k=min(max(float(args.lane_k), 0.10), 0.18),
            chicane_amplitude=float(args.chicane_amplitude),
            chicane_wavelength=float(args.chicane_wavelength),
            adaptation_amplitude=float(args.adaptation_weave_amplitude),
            adaptation_wavelength=float(args.adaptation_weave_wavelength),
            final_lane_shift=max(float(args.lane_shift), 3.2),
            final_lane_k=min(max(float(args.lane_k), 0.10), 0.16),
            name="late_adaptation_gap_boost_course",
        ),
        default_eval_duration=55.0,
        description=(
            "Late gap-boost technical course with an additional friction transition "
            "before the low-friction patch and another shift before the final "
            "evasive maneuver, designed to show continued Online RLS adaptation."
        ),
    )
    return scenarios


def selected_scenarios(args, scenarios: Mapping[str, ScenarioSpec]) -> List[ScenarioSpec]:
    if args.scenario == "all":
        return [
            scenarios["mild_sine"],
            scenarios["double_lane_change"],
            scenarios["adaptive_lane_change"],
        ]
    return [scenarios[args.scenario]]


def duration_defaults(args, spec: ScenarioSpec) -> tuple[float, float, int]:
    if args.smoke:
        train_duration = 1.8 if args.train_duration is None else args.train_duration
        eval_duration = 1.5 if args.eval_duration is None else args.eval_duration
        horizon = 5 if args.horizon is None else args.horizon
    else:
        if spec.name in {"adaptive_lane_change", "nonlinear_adaptive_lane_change", "low_mu_fiala_adaptive_lane_change"}:
            default_train = max(float(args.adaptation_time) + 5.0, 8.0)
        elif spec.name == "composite_friction_adaptation_course":
            default_train = 5.0
        elif spec.name == "nonstationary_adaptive_technical_course":
            default_train = 5.0
        elif spec.name == "late_adaptation_gap_boost_course":
            default_train = 3.5
        elif spec.name in {"friction_shift_adaptive_lane_change", "online_advantage_friction_shift_lane_change"}:
            default_train = 3.0
        else:
            default_train = 9.0
        train_duration = default_train if args.train_duration is None else args.train_duration
        eval_duration = spec.default_eval_duration if args.eval_duration is None else args.eval_duration
        horizon = 5 if args.horizon is None else args.horizon
    return train_duration, eval_duration, horizon


def controller_configs(horizon: int) -> MPCConfig:
    return MPCConfig(
        horizon=horizon,
        q=np.diag([24.0, 10.0, 0.7, 0.7]),
        r=0.18,
        rd=1.4,
        delta_max=0.50,
        delta_rate_max=0.10,
        y_max=np.asarray([4.5, 1.2, 10.0, 3.5], dtype=float),
    )


def selected_feature_names(args) -> tuple[str, ...]:
    if args.use_tire_augmented_koopman:
        return TIRE_AUGMENTED_FEATURE_NAMES
    if args.use_augmented_koopman:
        return AUGMENTED_FEATURE_NAMES
    return BASE_FEATURE_NAMES


def make_training_paths(args) -> List[object]:
    maneuver_x1 = (
        float(args.maneuver_start_x)
        if args.maneuver_start_x is not None
        else float(args.target_speed) * float(args.adaptation_time) + 4.0
    )
    return [
        SinePath(amplitude=args.path_amplitude, wavelength=args.path_wavelength, name="mild_sine_train"),
        AggressiveSinePath(
            amplitude=args.aggressive_amplitude,
            wavelength=args.aggressive_wavelength,
            name="aggressive_sine_train",
        ),
        DoubleLaneChangePath(
            lane_shift=args.lane_shift,
            x1=args.lane_x1,
            x2=args.lane_x2,
            k=args.lane_k,
            name="double_lane_change_train",
        ),
        AdaptiveLaneChangePath(
            lane_shift=max(float(args.lane_shift), 3.0),
            x1=maneuver_x1,
            x2=maneuver_x1 + 32.0,
            k=min(max(float(args.lane_k), 0.18), 0.25),
            name="adaptive_lane_change_train",
        ),
    ]


def has_matplotlib() -> bool:
    try:
        import matplotlib.pyplot  # noqa: F401

        return True
    except Exception:
        return False


def build_tire_config_override(args, out_root: Path) -> str | None:
    needs_override = (
        args.tire_model != "linear"
        or args.friction_mu is not None
        or args.friction_mu_initial is not None
        or args.friction_mu_final is not None
    )
    if not needs_override:
        return None

    base_config = Path(__file__).resolve().parents[1] / "models" / "params" / "vehicle_standard.yaml"
    with base_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tire_cfg = config.setdefault("tire", {})
    lateral_cfg = tire_cfg.setdefault("lateral", {})
    lateral_cfg["model_type"] = str(args.tire_model)
    config_mu = (
        args.friction_mu
        if args.friction_mu is not None
        else args.friction_mu_initial
        if args.friction_mu_initial is not None
        else args.friction_mu_final
    )
    if config_mu is not None:
        mu = float(config_mu)
        tire_cfg["mu"] = mu
        lateral_cfg["mu"] = mu

    config_dir = out_root / "_generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    mu_label = "default_mu" if config_mu is None else f"mu_{float(config_mu):.3f}".replace(".", "p")
    config_path = config_dir / f"vehicle_{args.tire_model}_{mu_label}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return str(config_path)


def apply_composite_defaults(args) -> None:
    args.scenario = "composite_friction_adaptation_course"
    args.tire_model = "fiala"
    args.use_tire_augmented_koopman = True
    args.koopman_mode = "residual_output"
    if args.training_policy == "excitation":
        args.training_policy = "small_sine_plus_linear_mpc"
    args.friction_profile = "piecewise"
    if args.friction_mu_initial is None:
        args.friction_mu_initial = 0.85
    if args.friction_mu_final is None:
        args.friction_mu_final = 0.70
    if args.friction_mu is None:
        args.friction_mu = args.friction_mu_initial
    if args.train_duration is None and not args.smoke:
        args.train_duration = 5.0
    if args.eval_duration is None and not args.smoke:
        args.eval_duration = 27.0
    if args.horizon is None:
        args.horizon = 6
    args.adaptation_time = 7.0
    if args.maneuver_start_time is None:
        args.maneuver_start_time = 9.0
    args.training_scope = "scenario"


def composite_friction_piecewise(path) -> List[Dict[str, float]]:
    if hasattr(path, "friction_piecewise"):
        return list(path.friction_piecewise())
    return []


def path_speed_piecewise(path) -> List[Dict[str, float]]:
    if hasattr(path, "speed_piecewise"):
        return list(path.speed_piecewise())
    return []


def apply_award_ready_defaults(args) -> None:
    args.scenario = "nonstationary_adaptive_technical_course"
    args.tire_model = "fiala"
    args.use_tire_augmented_koopman = True
    args.koopman_mode = "residual_output"
    if args.training_policy == "excitation":
        args.training_policy = "small_sine_plus_linear_mpc"
    args.friction_profile = "piecewise"
    args.speed_profile = "piecewise"
    if args.friction_mu_initial is None:
        args.friction_mu_initial = 0.90
    if args.friction_mu_final is None:
        args.friction_mu_final = 0.72
    if args.friction_mu is None:
        args.friction_mu = args.friction_mu_initial
    if args.train_duration is None and not args.smoke:
        args.train_duration = 5.0
    if args.eval_duration is None and not args.smoke:
        args.eval_duration = 55.0
    if args.horizon is None:
        args.horizon = 6
    args.target_speed = float(args.initial_speed_kmh) / 3.6
    args.adaptation_time = 14.0
    if args.maneuver_start_time is None:
        args.maneuver_start_time = 18.0
    args.training_scope = "scenario"


def apply_late_gap_boost_defaults(args) -> None:
    args.scenario = "late_adaptation_gap_boost_course"
    args.tire_model = "fiala"
    args.use_tire_augmented_koopman = True
    args.koopman_mode = "residual_output"
    if args.training_policy == "excitation":
        args.training_policy = "small_sine_plus_linear_mpc"
    args.friction_profile = "piecewise"
    args.speed_profile = "piecewise"
    if args.friction_mu_initial is None:
        args.friction_mu_initial = 0.90
    if args.friction_mu_final is None:
        args.friction_mu_final = 0.70
    if args.friction_mu is None:
        args.friction_mu = args.friction_mu_initial
    if args.train_duration is None and not args.smoke:
        args.train_duration = 3.5
    if args.eval_duration is None and not args.smoke:
        args.eval_duration = 55.0
    if args.horizon is None:
        args.horizon = 6
    args.target_speed = float(args.initial_speed_kmh) / 3.6
    args.adaptation_time = 12.0
    if args.maneuver_start_time is None:
        args.maneuver_start_time = 16.0
    args.training_scope = "scenario"


def run_scenario(
    *,
    args,
    spec: ScenarioSpec,
    all_training_paths: List[object],
    out_root: Path,
    config_path: str | None = None,
) -> Dict:
    train_duration, eval_duration, horizon = duration_defaults(args, spec)
    cfg = PlantConfig(
        control_dt=args.control_dt,
        plant_dt=args.plant_dt,
        target_speed=args.target_speed,
        speed_profile=args.speed_profile,
        speed_piecewise=(
            path_speed_piecewise(spec.path)
            if str(args.speed_profile) == "piecewise" else None
        ),
        friction_profile=args.friction_profile,
        friction_mu_initial=args.friction_mu_initial,
        friction_mu_final=args.friction_mu_final,
        friction_change_start_time=args.friction_change_start_time,
        friction_change_end_time=args.friction_change_end_time,
        friction_piecewise=(
            composite_friction_piecewise(spec.path)
            if str(args.friction_profile) == "piecewise" else None
        ),
    )
    training_mu = (
        args.friction_mu_initial
        if args.friction_mu_initial is not None
        else args.friction_mu
    )
    training_cfg = PlantConfig(
        control_dt=args.control_dt,
        plant_dt=args.plant_dt,
        target_speed=args.target_speed,
        speed_profile="constant",
        friction_profile="constant",
        friction_mu_initial=training_mu,
        friction_mu_final=training_mu,
        friction_change_start_time=args.friction_change_start_time,
        friction_change_end_time=args.friction_change_end_time,
    )
    mpc_cfg = controller_configs(horizon)
    scenario_dir = out_root / spec.name
    segments = segment_metadata(spec.path)
    key_segment_names = list(getattr(
        spec.path,
        "key_segment_names",
        ["double_lane_change", "moderate_chicane", "friction_patch_2"],
    ))
    if segments:
        write_run_csv(scenario_dir / "segment_metadata.csv", segments)
        write_json(scenario_dir / "segment_metadata.json", {"segments": segments})

    vehicle_for_params, _ = create_vehicle_plant(config_path=config_path, target_speed=cfg.target_speed)
    linear_scales = BicycleModelScales(
        mass_scale=args.linear_mass_scale,
        izz_scale=args.linear_izz_scale,
        cf_scale=args.linear_cf_scale,
        cr_scale=args.linear_cr_scale,
        curvature_scale=args.linear_curvature_scale,
    )
    residual_params = bicycle_params_from_vehicle(vehicle_for_params, linear_scales)
    feature_names = selected_feature_names(args)
    use_feature_context = args.use_augmented_koopman or args.use_tire_augmented_koopman
    input_names = ("delta_rad",) if use_feature_context else ("delta_rad", "path_curvature_1_per_m")

    if args.koopman_mode == "residual_output":
        def training_controller_factory():
            vehicle, _ = create_vehicle_plant(config_path=config_path, target_speed=cfg.target_speed)
            return LinearBicycleMPCController(vehicle, mpc_cfg, model_scales=linear_scales)

        training = generate_multi_path_closed_loop_training_data(
            paths=all_training_paths,
            duration=train_duration,
            cfg=training_cfg,
            config_path=config_path,
            controller_factory=(
                training_controller_factory
                if args.training_policy in {"safe_linear_mpc_adaptation", "small_sine_plus_linear_mpc"}
                else None
            ),
            excitation_amplitude=(
                min(float(args.excitation_amplitude), 0.04)
                if args.training_policy == "small_sine_plus_linear_mpc"
                else 0.0
                if args.training_policy == "safe_linear_mpc_adaptation"
                else float(args.excitation_amplitude)
            ),
            feature_names=feature_names,
            residual_params=residual_params,
            residual_curvature_scale=linear_scales.curvature_scale,
        )
    else:
        training = generate_multi_path_excitation_data(
            paths=all_training_paths,
            duration=train_duration,
            cfg=training_cfg,
            config_path=config_path,
            amplitude=args.excitation_amplitude,
            include_curvature_input=not use_feature_context,
            use_augmented_features=use_feature_context,
            feature_names=feature_names,
        )
    model = fit_edmd_koopman(
        training["X"],
        training["U"],
        training["X_next"],
        EDMDConfig(
            ridge=args.edmd_ridge,
            feature_names=tuple(feature_names),
            output_dim=4,
            input_names=input_names,
            fit_output_direct=bool(use_feature_context or args.koopman_mode == "residual_output"),
            prediction_mode=str(args.koopman_mode),
        ),
    )
    pred = prediction_errors(model, training["X"], training["U"], training["X_next"])
    write_run_csv(scenario_dir / "training_dataset.csv", list(training.get("rows", [])))
    residuals = np.asarray(training.get("residuals", np.zeros((0, 4))), dtype=float)
    training_summary = {
        "training_policy": args.training_policy,
        "koopman_mode": args.koopman_mode,
        "samples": int(training["X"].shape[0]),
        "feature_names": list(feature_names),
        "input_names": list(input_names),
        "residual_rmse": float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else None,
        "training_prediction_error": pred,
    }
    write_json(scenario_dir / "training_summary.json", training_summary)

    if args.koopman_mode == "residual_output":
        controllers = [
            LinearBicycleMPCController(vehicle_for_params, mpc_cfg, model_scales=linear_scales),
            ResidualKoopmanMPCController(model, vehicle_for_params, mpc_cfg, model_scales=linear_scales),
            OnlineResidualKoopmanMPCController(
                model,
                vehicle_for_params,
                mpc_cfg,
                model_scales=linear_scales,
                rls_config=RLSConfig(
                    forgetting_factor=args.rls_forgetting_factor,
                    p0=args.rls_p0,
                    max_spectral_radius=1.01,
                    reject_worse_factor=args.rls_reject_worse_factor,
                    reject_worse_margin=args.rls_reject_worse_margin,
                    max_relative_theta_update=args.rls_max_relative_theta_update,
                    max_post_update_control_change=args.rls_max_post_update_control_change,
                ),
            ),
        ]
    else:
        controllers = [
            LinearBicycleMPCController(vehicle_for_params, mpc_cfg, model_scales=linear_scales),
            KoopmanMPCController(model, mpc_cfg),
            OnlineKoopmanMPCController(
                model,
                mpc_cfg,
                RLSConfig(
                    forgetting_factor=0.9995,
                    p0=0.002,
                    max_spectral_radius=1.01,
                    reject_worse_factor=1.0,
                    reject_worse_margin=1.0e-6,
                    max_relative_theta_update=2.0e-2,
                    max_post_update_control_change=0.15,
                ),
            ),
        ]

    summaries = []
    runs = {}
    segment_metrics_by_controller = {}
    adaptation_time = float(args.adaptation_time)
    maneuver_start_time = (
        float(args.maneuver_start_time)
        if args.maneuver_start_time is not None
        else float(getattr(spec.path, "x1", cfg.target_speed * adaptation_time)) / max(cfg.target_speed, 1.0e-9)
    )
    if str(args.friction_profile) == "piecewise" and segments:
        friction_segments = [
            row for row in segments
            if "friction_transition" in str(row.get("segment_name", ""))
        ]
        post_friction_change_time = (
            float(friction_segments[0]["end_time_s"])
            if friction_segments else float(segments[0]["end_time_s"])
        )
    else:
        post_friction_change_time = (
            args.friction_change_end_time
            if args.friction_profile in {"step", "ramp"} else None
        )
    for controller in controllers:
        run = run_closed_loop(
            controller,
            duration=eval_duration,
            cfg=cfg,
            path=spec.path,
            config_path=config_path,
            feature_names=feature_names,
        )
        summary = summarize_run(
            run,
            adaptation_time=adaptation_time,
            post_friction_change_time=post_friction_change_time,
            maneuver_start_time=maneuver_start_time,
        )
        if segments:
            controller_segment_metrics = segment_metrics_for_run(run, segments)
            key_metrics = combined_segment_metrics_for_run(run, segments, key_segment_names)
            segment_metrics_by_controller[getattr(controller, "name", type(controller).__name__)] = controller_segment_metrics
            summary.update({
                "key_segments_lateral_error_rmse": key_metrics.get("lateral_error_rmse"),
                "key_segments_heading_error_rmse": key_metrics.get("heading_error_rmse"),
                "key_segments_max_abs_lateral_error": key_metrics.get("max_abs_lateral_error"),
                "key_segments_steering_effort_rms": key_metrics.get("steering_effort_rms"),
                "key_segments_steering_smoothness_rms": key_metrics.get("steering_smoothness_rms"),
                "key_segments_residual_prediction_error_improvement_percent": key_metrics.get("residual_prediction_error_improvement_percent"),
                "key_segments_residual_error_reduction_mean": key_metrics.get("residual_error_reduction_mean"),
                "key_segments_residual_error_reduction_positive_ratio": key_metrics.get("residual_error_reduction_positive_ratio"),
            })
        summary.update({
            "scenario": spec.name,
            "path_parameters": path_parameters(spec.path),
            "plant_description": PLANT_DESCRIPTION,
        })
        summaries.append(summary)
        runs[summary["controller"]] = run
        write_run_csv(scenario_dir / f"{summary['controller']}.csv", run["rows"])

    x_max = max(
        10.0,
        float(spec.path.x_at_time(eval_duration)) + 5.0
        if hasattr(spec.path, "x_at_time") else cfg.target_speed * eval_duration + 5.0,
    )
    ref_rows = sample_reference_path(spec.path, np.linspace(0.0, x_max, 401))
    write_run_csv(scenario_dir / "reference_path.csv", ref_rows)

    result = {
        "scenario": spec.name,
        "scenario_description": spec.description,
        "path_parameters": path_parameters(spec.path),
        "plant_description": PLANT_DESCRIPTION,
        "prediction_models": [
            "linear bicycle model MPC",
            "fixed residual EDMD-Koopman MPC" if args.koopman_mode == "residual_output" else "fixed EDMD-Koopman MPC",
            "online residual EDMD-Koopman MPC with matrix RLS" if args.koopman_mode == "residual_output" else "online EDMD-Koopman MPC with matrix RLS",
        ],
        "config": {
            "train_duration": train_duration,
            "eval_duration": eval_duration,
            "control_dt": cfg.control_dt,
            "plant_dt": cfg.plant_dt,
            "target_speed": cfg.target_speed,
            "speed_profile": cfg.speed_profile,
            "speed_schedule": cfg.speed_piecewise,
            "align_initial_heading": cfg.align_initial_heading,
            "horizon": horizon,
            "adaptation_time": adaptation_time,
            "maneuver_start_time": maneuver_start_time,
            "training_samples": int(training["X"].shape[0]),
            "training_paths": [path_parameters(path) for path in all_training_paths],
            "training_scope": args.training_scope,
            "excitation_amplitude_rad": float(args.excitation_amplitude),
            "training_friction_mu": training_mu,
            "lifted_dim": int(model.lifted_dim),
            "use_augmented_koopman": bool(args.use_augmented_koopman),
            "use_tire_augmented_koopman": bool(args.use_tire_augmented_koopman),
            "koopman_mode": str(args.koopman_mode),
            "koopman_feature_names": list(model.feature_names),
            "koopman_feature_scales": model.feature_scales.tolist(),
            "koopman_input_names": list(model.input_names),
            "linear_model_mismatch": linear_scales.as_dict(),
            "training_policy": args.training_policy,
            "tire_model": str(args.tire_model),
            "friction_mu": args.friction_mu,
            "friction_schedule": {
                "profile": args.friction_profile,
                "mu_initial": args.friction_mu_initial,
                "mu_final": args.friction_mu_final,
                "change_start_time": args.friction_change_start_time,
                "change_end_time": args.friction_change_end_time,
                "piecewise": cfg.friction_piecewise,
            },
            "rls_config": {
                "forgetting_factor": args.rls_forgetting_factor,
                "p0": args.rls_p0,
                "max_relative_theta_update": args.rls_max_relative_theta_update,
                "max_post_update_control_change": args.rls_max_post_update_control_change,
                "reject_worse_factor": args.rls_reject_worse_factor,
                "reject_worse_margin": args.rls_reject_worse_margin,
            },
            "vehicle_config_path": config_path,
            "matplotlib_available": has_matplotlib(),
        },
        "training_prediction_error": pred,
        "summaries": summaries,
        "segment_metadata": segments,
        "key_segments": key_segment_names,
        "segment_metrics": segment_metrics_by_controller,
    }
    write_json(scenario_dir / "summary.json", result)
    write_json(scenario_dir / "koopman_model.json", model.to_dict())
    write_visualization_summary(
        scenario_dir,
        summaries,
        segment_metrics_by_controller,
        key_segment_names,
    )

    return {
        "summary": result,
        "runs": runs,
        "path": spec.path,
        "scenario_dir": scenario_dir,
    }


def flatten_benchmark(results: Mapping[str, Mapping]) -> List[Dict]:
    rows = []
    for scenario_name, scenario_result in results.items():
        summary = scenario_result["summary"]
        for item in summary["summaries"]:
            row = {
                "scenario": scenario_name,
                "controller": item["controller"],
                "lateral_error_rmse": item["lateral_error_rmse"],
                "heading_error_rmse": item["heading_error_rmse"],
                "max_abs_lateral_error": item["max_abs_lateral_error"],
                "steering_effort_rms": item["steering_effort_rms"],
                "steering_smoothness_rms": item["steering_smoothness_rms"],
                "mean_solve_time_s": item["mean_solve_time_s"],
                "max_solve_time_s": item["max_solve_time_s"],
                "post_adaptation_lateral_error_rmse": item.get("post_adaptation_lateral_error_rmse"),
                "post_adaptation_heading_error_rmse": item.get("post_adaptation_heading_error_rmse"),
                "post_adaptation_max_abs_lateral_error": item.get("post_adaptation_max_abs_lateral_error"),
                "post_friction_change_lateral_error_rmse": item.get("post_friction_change_lateral_error_rmse"),
                "post_friction_change_heading_error_rmse": item.get("post_friction_change_heading_error_rmse"),
                "post_friction_change_max_abs_lateral_error": item.get("post_friction_change_max_abs_lateral_error"),
                "maneuver_segment_lateral_error_rmse": item.get("maneuver_segment_lateral_error_rmse"),
                "maneuver_segment_heading_error_rmse": item.get("maneuver_segment_heading_error_rmse"),
                "maneuver_segment_max_abs_lateral_error": item.get("maneuver_segment_max_abs_lateral_error"),
                "key_segments_lateral_error_rmse": item.get("key_segments_lateral_error_rmse"),
                "key_segments_heading_error_rmse": item.get("key_segments_heading_error_rmse"),
                "key_segments_max_abs_lateral_error": item.get("key_segments_max_abs_lateral_error"),
                "key_segments_steering_effort_rms": item.get("key_segments_steering_effort_rms"),
                "key_segments_steering_smoothness_rms": item.get("key_segments_steering_smoothness_rms"),
                "key_segments_residual_prediction_error_improvement_percent": item.get("key_segments_residual_prediction_error_improvement_percent"),
                "key_segments_residual_error_reduction_mean": item.get("key_segments_residual_error_reduction_mean"),
                "key_segments_residual_error_reduction_positive_ratio": item.get("key_segments_residual_error_reduction_positive_ratio"),
                "prediction_error_before_mean": item.get("prediction_error_before_mean"),
                "prediction_error_after_mean": item["prediction_error_after_mean"],
                "prediction_error_improvement_percent": item.get("prediction_error_improvement_percent"),
                "residual_prediction_error_before_mean": item.get("residual_prediction_error_before_mean"),
                "residual_prediction_error_after_mean": item.get("residual_prediction_error_after_mean"),
                "residual_prediction_error_improvement_percent": item.get("residual_prediction_error_improvement_percent"),
                "rls_acceptance_rate": item.get("rls_acceptance_rate"),
                "rls_rejection_rate": item.get("rls_rejection_rate"),
                "rls_relative_theta_update_mean": item.get("rls_relative_theta_update_mean"),
            }
            rows.append(row)
    return rows


def write_visualization_summary(
    scenario_dir: Path,
    summaries: List[Mapping],
    segment_metrics_by_controller: Mapping[str, Mapping[str, Mapping]],
    key_segment_names: List[str],
) -> None:
    controllers = [str(item["controller"]) for item in summaries]
    summary_by_controller = {str(item["controller"]): item for item in summaries}
    rows: List[Dict] = []

    def add_row(scope: str, segment_name: str, controller: str, metrics: Mapping) -> None:
        rmse = metrics.get("lateral_error_rmse")
        max_err = metrics.get("max_abs_lateral_error")
        rows.append({
            "scope": scope,
            "segment_name": segment_name,
            "controller": controller,
            "lateral_error_rmse_m": rmse,
            "lateral_error_rmse_mm": None if rmse is None else 1000.0 * float(rmse),
            "max_abs_lateral_error_m": max_err,
            "max_abs_lateral_error_mm": None if max_err is None else 1000.0 * float(max_err),
            "steering_effort_rms_rad": metrics.get("steering_effort_rms"),
            "steering_smoothness_rms_rad_per_step": metrics.get("steering_smoothness_rms"),
            "mean_solve_time_s": metrics.get("mean_solve_time_s"),
            "residual_prediction_error_improvement_percent": metrics.get("residual_prediction_error_improvement_percent"),
            "residual_error_reduction_mean": metrics.get("residual_error_reduction_mean"),
            "residual_error_reduction_positive_ratio": metrics.get("residual_error_reduction_positive_ratio"),
            "rls_acceptance_rate": metrics.get("rls_acceptance_rate"),
            "online_vs_fixed_improvement_percent": "",
            "online_vs_linear_improvement_percent": "",
            "online_vs_best_baseline_improvement_percent": "",
            "best_non_online_baseline": "",
        })

    for controller in controllers:
        add_row("overall", "overall", controller, summary_by_controller[controller])
        key_metric = {
            "lateral_error_rmse": summary_by_controller[controller].get("key_segments_lateral_error_rmse"),
            "max_abs_lateral_error": summary_by_controller[controller].get("key_segments_max_abs_lateral_error"),
            "steering_effort_rms": summary_by_controller[controller].get("key_segments_steering_effort_rms"),
            "steering_smoothness_rms": summary_by_controller[controller].get("key_segments_steering_smoothness_rms"),
            "residual_prediction_error_improvement_percent": summary_by_controller[controller].get("key_segments_residual_prediction_error_improvement_percent"),
            "rls_acceptance_rate": summary_by_controller[controller].get("rls_acceptance_rate"),
        }
        add_row("combined_key_segments", "key_segments", controller, key_metric)
        for segment_name, metrics in segment_metrics_by_controller.get(controller, {}).items():
            add_row("segment", str(segment_name), controller, metrics)

    online = summary_by_controller.get("online_residual_koopman_mpc")
    fixed = summary_by_controller.get("fixed_residual_koopman_mpc")
    linear = summary_by_controller.get("linear_bicycle_mpc")
    if online and fixed and linear:
        online_key = online.get("key_segments_lateral_error_rmse")
        fixed_key = fixed.get("key_segments_lateral_error_rmse")
        linear_key = linear.get("key_segments_lateral_error_rmse")
        fixed_improvement = (
            100.0 * (float(fixed_key) - float(online_key)) / max(float(fixed_key), 1.0e-12)
            if online_key is not None and fixed_key is not None else None
        )
        linear_improvement = (
            100.0 * (float(linear_key) - float(online_key)) / max(float(linear_key), 1.0e-12)
            if online_key is not None and linear_key is not None else None
        )
        best_baseline_name = "fixed_residual_koopman_mpc"
        best_baseline_value = fixed_key
        if fixed_key is not None and linear_key is not None and float(linear_key) < float(fixed_key):
            best_baseline_name = "linear_bicycle_mpc"
            best_baseline_value = linear_key
        best_improvement = (
            100.0 * (float(best_baseline_value) - float(online_key)) / max(float(best_baseline_value), 1.0e-12)
            if online_key is not None and best_baseline_value is not None else None
        )
        for row in rows:
            if row["scope"] == "combined_key_segments" and row["controller"] == "online_residual_koopman_mpc":
                row["online_vs_fixed_improvement_percent"] = fixed_improvement
                row["online_vs_linear_improvement_percent"] = linear_improvement
                row["online_vs_best_baseline_improvement_percent"] = best_improvement
                row["best_non_online_baseline"] = best_baseline_name

    write_run_csv(scenario_dir / "visualization_summary.csv", rows)
    write_json(scenario_dir / "visualization_summary.json", {
        "unit_note": "Metric columns ending in _mm are converted from meters for poster-scale visualization.",
        "key_segments": key_segment_names,
        "rows": rows,
    })


def generate_figures(out_root: Path, scenario_results: Mapping[str, Mapping], benchmark_rows: List[Dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    colors = {
        "linear_bicycle_mpc": "#1f77b4",
        "fixed_koopman_mpc": "#ff7f0e",
        "online_koopman_mpc": "#2ca02c",
        "fixed_residual_koopman_mpc": "#ff7f0e",
        "online_residual_koopman_mpc": "#2ca02c",
    }
    labels = {
        "linear_bicycle_mpc": "Linear MPC",
        "fixed_koopman_mpc": "Fixed Koopman MPC",
        "online_koopman_mpc": "Online Koopman MPC",
        "fixed_residual_koopman_mpc": "Fixed Residual Koopman MPC",
        "online_residual_koopman_mpc": "Online Residual Koopman MPC",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for scenario_name, result in scenario_results.items():
        path = result["path"]
        summary = result["summary"]
        eval_duration = float(summary["config"]["eval_duration"])
        target_speed = float(summary["config"]["target_speed"])
        xs = np.linspace(0.0, target_speed * eval_duration + 5.0, 500)
        ys = [path.y(x) for x in xs]
        ax.plot(xs, ys, label=scenario_name)
    ax.set_xlabel("Global X Position [m]")
    ax.set_ylabel("Reference Y Position [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "reference_paths_comparison.png", dpi=180)
    plt.close(fig)

    for scenario_name, result in scenario_results.items():
        path = result["path"]
        runs = result["runs"]
        scenario_dir = result["scenario_dir"]
        first_rows = next(iter(runs.values()))["rows"]
        xs = np.asarray([row["x"] for row in first_rows], dtype=float)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(xs, [path.y(x) for x in xs], "k--", label="Reference")
        for controller, run in runs.items():
            rows = run["rows"]
            ax.plot(
                [row["x"] for row in rows],
                [row["y"] for row in rows],
                color=colors.get(controller),
                label=labels.get(controller, controller),
            )
        ax.set_xlabel("Global X Position [m]")
        ax.set_ylabel("Global Y Position [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(scenario_dir / "trajectory_tracking.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for controller, run in runs.items():
            rows = run["rows"]
            ax.plot(
                [row["time"] for row in rows],
                [100.0 * row["e_y"] for row in rows],
                color=colors.get(controller),
                label=labels.get(controller, controller),
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Lateral Error e_y [cm]")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(scenario_dir / "lateral_error_time_series.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for controller, run in runs.items():
            rows = run["rows"]
            ax.plot(
                [row["time"] for row in rows],
                [np.degrees(row["delta"]) for row in rows],
                color=colors.get(controller),
                label=labels.get(controller, controller),
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Steering Command delta [deg]")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(scenario_dir / "steering_command.png", dpi=180)
        plt.close(fig)

        online = runs.get("online_koopman_mpc") or runs.get("online_residual_koopman_mpc")
        if online is not None:
            rows = online["rows"]
            before = np.asarray([row["prediction_error_before"] for row in rows], dtype=float)
            after = np.asarray([row["prediction_error_after"] for row in rows], dtype=float)
            if np.isfinite(before).any() or np.isfinite(after).any():
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                ax.plot([row["time"] for row in rows], before, label="Before RLS update")
                ax.plot([row["time"] for row in rows], after, label="After RLS update")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Prediction Error [-]")
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                fig.savefig(scenario_dir / "online_prediction_error.png", dpi=180)
                plt.close(fig)

    if any(row["controller"] == "fixed_residual_koopman_mpc" for row in benchmark_rows):
        controllers = ["linear_bicycle_mpc", "fixed_residual_koopman_mpc", "online_residual_koopman_mpc"]
    else:
        controllers = ["linear_bicycle_mpc", "fixed_koopman_mpc", "online_koopman_mpc"]
    scenario_names = list(scenario_results.keys())
    x = np.arange(len(scenario_names))
    width = 0.24
    for metric, ylabel, filename, scale in [
        ("lateral_error_rmse", "Lateral RMSE [cm]", "scenario_lateral_rmse.png", 100.0),
        ("max_abs_lateral_error", "Max |e_y| [cm]", "scenario_max_lateral_error.png", 100.0),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for idx, controller in enumerate(controllers):
            values = []
            for scenario_name in scenario_names:
                match = [
                    row for row in benchmark_rows
                    if row["scenario"] == scenario_name and row["controller"] == controller
                ]
                values.append(scale * float(match[0][metric]) if match else np.nan)
            ax.bar(x + (idx - 1) * width, values, width, label=labels[controller], color=colors[controller])
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=15, ha="right")
        ax.set_xlabel("Scenario")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_root / filename, dpi=180)
        plt.close(fig)

    return True


def _controller_metric(rows: List[Dict], controller: str, metric: str) -> float | None:
    for row in rows:
        if row.get("controller") == controller and row.get(metric) is not None:
            return float(row[metric])
    return None


def run_small_sweep(args, out_root: Path) -> List[Dict]:
    sweep_rows: List[Dict] = []
    combos = itertools.product(
        [0.65, 0.70, 0.75],
        [10.0, 10.5, 11.0],
        [3.0, 3.2],
        [0.01, 0.03, 0.05],
        [0.01, 0.02, 0.03],
    )
    for idx, (mu_final, speed, lane_shift, rls_p0, max_rel_update) in enumerate(combos):
        if idx >= int(args.sweep_limit):
            break
        case_args = copy.deepcopy(args)
        case_args.scenario = "friction_shift_adaptive_lane_change"
        case_args.tire_model = "fiala"
        case_args.use_tire_augmented_koopman = True
        case_args.koopman_mode = "residual_output"
        case_args.training_policy = "small_sine_plus_linear_mpc"
        case_args.friction_profile = "ramp"
        case_args.friction_mu_initial = 0.85
        case_args.friction_mu_final = float(mu_final)
        case_args.friction_mu = 0.85
        case_args.target_speed = float(speed)
        case_args.lane_shift = float(lane_shift)
        case_args.rls_p0 = float(rls_p0)
        case_args.rls_max_relative_theta_update = float(max_rel_update)
        case_args.train_duration = min(float(args.train_duration or 3.5), 3.5)
        case_args.eval_duration = min(float(args.eval_duration or 11.0), 11.0)
        case_args.horizon = args.horizon or 6
        case_root = out_root / "sweep_runs" / (
            f"case_{idx:02d}_mu{mu_final:.2f}_v{speed:.1f}_lane{lane_shift:.1f}_"
            f"p0{rls_p0:.2f}_rel{max_rel_update:.2f}"
        )
        config_path = build_tire_config_override(case_args, case_root)
        scenarios = build_scenarios(case_args)
        spec = scenarios["friction_shift_adaptive_lane_change"]
        result = run_scenario(
            args=case_args,
            spec=spec,
            all_training_paths=[spec.path],
            out_root=case_root,
            config_path=config_path,
        )
        rows = flatten_benchmark({spec.name: result})
        online = _controller_metric(rows, "online_residual_koopman_mpc", "maneuver_segment_lateral_error_rmse")
        fixed = _controller_metric(rows, "fixed_residual_koopman_mpc", "maneuver_segment_lateral_error_rmse")
        linear = _controller_metric(rows, "linear_bicycle_mpc", "maneuver_segment_lateral_error_rmse")
        max_online = _controller_metric(rows, "online_residual_koopman_mpc", "max_abs_lateral_error")
        steer_online = _controller_metric(rows, "online_residual_koopman_mpc", "steering_effort_rms")
        steer_linear = _controller_metric(rows, "linear_bicycle_mpc", "steering_effort_rms")
        values = [v for v in [online, fixed, linear] if v is not None]
        online_best = bool(values and online is not None and online <= min(values))
        sweep_row = {
            "case": idx,
            "mu_final": float(mu_final),
            "target_speed": float(speed),
            "lane_shift": float(lane_shift),
            "rls_p0": float(rls_p0),
            "rls_max_relative_theta_update": float(max_rel_update),
            "online_maneuver_rmse": online,
            "fixed_maneuver_rmse": fixed,
            "linear_maneuver_rmse": linear,
            "online_is_best_maneuver_rmse": online_best,
            "online_max_abs_lateral_error": max_online,
            "online_steering_effort_rms": steer_online,
            "linear_steering_effort_rms": steer_linear,
            "online_steering_not_excessive": (
                bool(steer_online is not None and steer_linear is not None and steer_online <= 1.25 * steer_linear)
            ),
            "case_dir": str(case_root),
        }
        sweep_rows.append(sweep_row)
    write_run_csv(out_root / "sweep_summary.csv", sweep_rows)
    write_json(out_root / "sweep_summary.json", {"cases": sweep_rows})
    return sweep_rows


def apply_online_advantage_defaults(args) -> None:
    args.scenario = "online_advantage_friction_shift_lane_change"
    args.tire_model = "fiala"
    args.use_tire_augmented_koopman = True
    args.koopman_mode = "residual_output"
    if args.training_policy == "excitation":
        args.training_policy = "small_sine_plus_linear_mpc"
    if args.friction_profile == "constant":
        args.friction_profile = "ramp"
    if args.friction_mu_initial is None:
        args.friction_mu_initial = 0.85
    if args.friction_mu_final is None:
        args.friction_mu_final = 0.65
    if args.friction_mu is None:
        args.friction_mu = args.friction_mu_initial
    if args.horizon is None:
        args.horizon = 6
    if args.eval_duration is None:
        args.eval_duration = 11.5
    args.training_scope = "scenario"


def _candidate_hash(params: Mapping) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _load_search_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_search_rows(path: Path, rows: List[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_for(rows: List[Dict], controller: str, metric: str) -> float | None:
    value = _controller_metric(rows, controller, metric)
    return None if value is None or not np.isfinite(value) else float(value)


def _search_candidate_sequence(args) -> Iterable[Dict]:
    hand_tuned = [
        {
            "stage": "baseline_reference",
            "mu_initial": 0.85,
            "mu_final": 0.70,
            "speed": 10.5,
            "lane_shift": 3.1,
            "lane_k": 0.22,
            "train_duration": 3.0,
            "friction_start": 1.5,
            "friction_end": 3.0,
            "adaptation_duration": 1.5,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 32.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.999,
            "rls_p0": 0.01,
            "rls_rel_update": 0.01,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.00,
        },
        {
            "stage": "baseline_shorter_training",
            "mu_initial": 0.85,
            "mu_final": 0.70,
            "speed": 10.5,
            "lane_shift": 3.1,
            "lane_k": 0.22,
            "train_duration": 2.0,
            "friction_start": 1.5,
            "friction_end": 3.0,
            "adaptation_duration": 1.5,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 32.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.999,
            "rls_p0": 0.01,
            "rls_rel_update": 0.01,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.00,
        },
        {
            "stage": "baseline_shorter_training_adaptive_rls",
            "mu_initial": 0.85,
            "mu_final": 0.70,
            "speed": 10.5,
            "lane_shift": 3.1,
            "lane_k": 0.22,
            "train_duration": 2.0,
            "friction_start": 1.5,
            "friction_end": 3.0,
            "adaptation_duration": 1.5,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 32.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.03,
        },
        {
            "stage": "baseline_more_adaptation_weave",
            "mu_initial": 0.85,
            "mu_final": 0.70,
            "speed": 10.5,
            "lane_shift": 3.1,
            "lane_k": 0.22,
            "train_duration": 2.5,
            "friction_start": 1.5,
            "friction_end": 3.0,
            "adaptation_duration": 1.8,
            "adaptation_amp": 0.30,
            "adaptation_wavelength": 30.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.999,
            "rls_p0": 0.01,
            "rls_rel_update": 0.01,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.00,
        },
        {
            "stage": "moderate_drop_conservative_rls",
            "mu_initial": 0.85,
            "mu_final": 0.65,
            "speed": 10.5,
            "lane_shift": 3.1,
            "lane_k": 0.22,
            "train_duration": 2.5,
            "friction_start": 1.5,
            "friction_end": 3.0,
            "adaptation_duration": 1.8,
            "adaptation_amp": 0.26,
            "adaptation_wavelength": 32.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.999,
            "rls_p0": 0.01,
            "rls_rel_update": 0.01,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.00,
        },
        {
            "stage": "short_fixed_data_conservative_rls",
            "mu_initial": 0.90,
            "mu_final": 0.70,
            "speed": 10.5,
            "lane_shift": 3.2,
            "lane_k": 0.22,
            "train_duration": 1.5,
            "friction_start": 1.5,
            "friction_end": 3.0,
            "adaptation_duration": 2.0,
            "adaptation_amp": 0.26,
            "adaptation_wavelength": 32.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.999,
            "rls_p0": 0.01,
            "rls_rel_update": 0.01,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.00,
        },
        {
            "stage": "stronger_friction_shift",
            "mu_initial": 0.90,
            "mu_final": 0.60,
            "speed": 11.0,
            "lane_shift": 3.2,
            "lane_k": 0.23,
            "train_duration": 2.5,
            "friction_start": 1.4,
            "friction_end": 2.8,
            "adaptation_duration": 1.7,
            "adaptation_amp": 0.30,
            "adaptation_wavelength": 30.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.03,
        },
        {
            "stage": "short_fixed_training",
            "mu_initial": 0.90,
            "mu_final": 0.62,
            "speed": 11.5,
            "lane_shift": 3.3,
            "lane_k": 0.24,
            "train_duration": 2.0,
            "friction_start": 1.3,
            "friction_end": 2.7,
            "adaptation_duration": 2.0,
            "adaptation_amp": 0.32,
            "adaptation_wavelength": 28.0,
            "cf_scale": 0.58,
            "cr_scale": 0.58,
            "mass_scale": 1.06,
            "izz_scale": 1.18,
            "rls_forgetting": 0.997,
            "rls_p0": 0.05,
            "rls_rel_update": 0.03,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.03,
        },
        {
            "stage": "low_mu_recovery",
            "mu_initial": 0.95,
            "mu_final": 0.55,
            "speed": 10.8,
            "lane_shift": 3.0,
            "lane_k": 0.20,
            "train_duration": 2.0,
            "friction_start": 1.2,
            "friction_end": 2.6,
            "adaptation_duration": 2.2,
            "adaptation_amp": 0.34,
            "adaptation_wavelength": 30.0,
            "cf_scale": 0.65,
            "cr_scale": 0.65,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.995,
            "rls_p0": 0.05,
            "rls_rel_update": 0.03,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.05,
        },
        {
            "stage": "faster_maneuver",
            "mu_initial": 0.90,
            "mu_final": 0.65,
            "speed": 12.0,
            "lane_shift": 3.4,
            "lane_k": 0.24,
            "train_duration": 2.5,
            "friction_start": 1.5,
            "friction_end": 3.0,
            "adaptation_duration": 1.5,
            "adaptation_amp": 0.30,
            "adaptation_wavelength": 28.0,
            "cf_scale": 0.55,
            "cr_scale": 0.60,
            "mass_scale": 1.08,
            "izz_scale": 1.20,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.03,
        },
    ]
    for params in hand_tuned:
        yield params

    rng = np.random.default_rng(int(args.search_seed))
    mu_initials = np.asarray([0.85, 0.90, 0.95])
    mu_finals = np.asarray([0.55, 0.60, 0.65, 0.70])
    speeds = np.asarray([10.0, 10.5, 11.0, 11.5, 12.0])
    lane_shifts = np.asarray([3.0, 3.2, 3.4, 3.6])
    lane_ks = np.asarray([0.20, 0.22, 0.24, 0.26])
    train_durations = np.asarray([2.0, 2.5, 3.0])
    adaptation_durations = np.asarray([1.2, 1.5, 1.8, 2.2])
    adaptation_amps = np.asarray([0.22, 0.28, 0.34, 0.40])
    rls_p0s = np.asarray([0.01, 0.03, 0.05, 0.10])
    rel_updates = np.asarray([0.01, 0.02, 0.03, 0.04])
    forgettings = np.asarray([0.995, 0.997, 0.999])
    cf_cr = np.asarray([0.55, 0.60, 0.65, 0.70])
    for _ in range(max(0, int(args.search_max_candidates) * 3)):
        mu_initial = float(rng.choice(mu_initials))
        mu_final = float(rng.choice(mu_finals[mu_finals < mu_initial]))
        friction_start = float(rng.choice([1.2, 1.4, 1.5, 1.7]))
        friction_end = friction_start + float(rng.choice([1.2, 1.5, 1.8]))
        cf = float(rng.choice(cf_cr))
        cr = float(rng.choice(cf_cr))
        yield {
            "stage": "seeded_search",
            "mu_initial": mu_initial,
            "mu_final": mu_final,
            "speed": float(rng.choice(speeds)),
            "lane_shift": float(rng.choice(lane_shifts)),
            "lane_k": float(rng.choice(lane_ks)),
            "train_duration": float(rng.choice(train_durations)),
            "friction_start": friction_start,
            "friction_end": friction_end,
            "adaptation_duration": float(rng.choice(adaptation_durations)),
            "adaptation_amp": float(rng.choice(adaptation_amps)),
            "adaptation_wavelength": float(rng.choice([26.0, 28.0, 30.0, 32.0, 36.0])),
            "cf_scale": cf,
            "cr_scale": cr,
            "mass_scale": float(rng.choice([1.00, 1.05, 1.10])),
            "izz_scale": float(rng.choice([1.05, 1.15, 1.25])),
            "rls_forgetting": float(rng.choice(forgettings)),
            "rls_p0": float(rng.choice(rls_p0s)),
            "rls_rel_update": float(rng.choice(rel_updates)),
            "rls_control_jump": float(rng.choice([0.08, 0.12, 0.15])),
            "rls_reject_worse": float(rng.choice([1.00, 1.03, 1.05])),
        }


def _apply_search_candidate(args, params: Mapping) -> None:
    apply_online_advantage_defaults(args)
    args.friction_mu_initial = float(params["mu_initial"])
    args.friction_mu_final = float(params["mu_final"])
    args.friction_mu = float(params["mu_initial"])
    args.friction_change_start_time = float(params["friction_start"])
    args.friction_change_end_time = float(params["friction_end"])
    args.target_speed = float(params["speed"])
    args.lane_shift = float(params["lane_shift"])
    args.lane_k = float(params["lane_k"])
    args.train_duration = float(params["train_duration"])
    args.adaptation_time = float(params["adaptation_duration"])
    args.maneuver_start_time = float(args.friction_change_end_time) + float(args.adaptation_time)
    args.maneuver_start_x = None
    args.eval_duration = max(float(args.eval_duration or 11.5), float(args.maneuver_start_time) + 6.0)
    args.adaptation_weave_amplitude = float(params["adaptation_amp"])
    args.adaptation_weave_wavelength = float(params["adaptation_wavelength"])
    args.linear_cf_scale = float(params["cf_scale"])
    args.linear_cr_scale = float(params["cr_scale"])
    args.linear_mass_scale = float(params["mass_scale"])
    args.linear_izz_scale = float(params["izz_scale"])
    args.linear_curvature_scale = 1.0
    args.rls_forgetting_factor = float(params["rls_forgetting"])
    args.rls_p0 = float(params["rls_p0"])
    args.rls_max_relative_theta_update = float(params["rls_rel_update"])
    args.rls_max_post_update_control_change = float(params["rls_control_jump"])
    args.rls_reject_worse_factor = float(params["rls_reject_worse"])
    args.rls_reject_worse_margin = 1.0e-5
    args.horizon = args.horizon or 6


def _search_result_row(
    *,
    idx: int,
    candidate_hash: str,
    params: Mapping,
    result: Mapping,
    rows: List[Dict],
    case_dir: Path,
    target_improvement: float,
) -> Dict:
    online = _metric_for(rows, "online_residual_koopman_mpc", "maneuver_segment_lateral_error_rmse")
    fixed = _metric_for(rows, "fixed_residual_koopman_mpc", "maneuver_segment_lateral_error_rmse")
    linear = _metric_for(rows, "linear_bicycle_mpc", "maneuver_segment_lateral_error_rmse")
    online_max = _metric_for(rows, "online_residual_koopman_mpc", "maneuver_segment_max_abs_lateral_error")
    fixed_max = _metric_for(rows, "fixed_residual_koopman_mpc", "maneuver_segment_max_abs_lateral_error")
    linear_max = _metric_for(rows, "linear_bicycle_mpc", "maneuver_segment_max_abs_lateral_error")
    online_steer = _metric_for(rows, "online_residual_koopman_mpc", "steering_effort_rms")
    fixed_steer = _metric_for(rows, "fixed_residual_koopman_mpc", "steering_effort_rms")
    linear_steer = _metric_for(rows, "linear_bicycle_mpc", "steering_effort_rms")
    pred_improve = _metric_for(rows, "online_residual_koopman_mpc", "residual_prediction_error_improvement_percent")
    acceptance = _metric_for(rows, "online_residual_koopman_mpc", "rls_acceptance_rate")
    rel_update = _metric_for(rows, "online_residual_koopman_mpc", "rls_relative_theta_update_mean")
    online_fixed_improvement = (
        (fixed - online) / fixed if fixed is not None and online is not None and fixed > 1.0e-12 else None
    )
    online_linear_improvement = (
        (linear - online) / linear if linear is not None and online is not None and linear > 1.0e-12 else None
    )
    steer_ref = max(v for v in [fixed_steer or 0.0, linear_steer or 0.0, 1.0e-9])
    max_ref = max(v for v in [fixed_max or 0.0, linear_max or 0.0, 0.05])
    online_best = bool(
        online is not None
        and fixed is not None
        and linear is not None
        and online < fixed
        and online < linear
    )
    max_ok = bool(online_max is not None and online_max <= max(0.75, 1.50 * max_ref))
    steer_ok = bool(online_steer is not None and online_steer <= 1.35 * steer_ref)
    prediction_improved = bool(pred_improve is not None and pred_improve > 0.0)
    target_met = bool(
        online_best
        and online_fixed_improvement is not None
        and online_fixed_improvement >= float(target_improvement)
        and max_ok
        and steer_ok
    )
    score = (
        (5.0 if target_met else 0.0)
        + (2.0 if online_best else 0.0)
        + float(online_fixed_improvement or -1.0)
        + 0.5 * float(online_linear_improvement or -1.0)
        + 0.02 * float(pred_improve or 0.0)
        - 0.05 * float(online_max or 0.0)
    )
    row = {
        "case": idx,
        "candidate_hash": candidate_hash,
        "case_dir": str(case_dir),
        "target_met": target_met,
        "online_best_maneuver": online_best,
        "score": score,
        "online_fixed_improvement_ratio": online_fixed_improvement,
        "online_linear_improvement_ratio": online_linear_improvement,
        "online_maneuver_rmse": online,
        "fixed_maneuver_rmse": fixed,
        "linear_maneuver_rmse": linear,
        "online_maneuver_max_abs": online_max,
        "fixed_maneuver_max_abs": fixed_max,
        "linear_maneuver_max_abs": linear_max,
        "online_steering_effort_rms": online_steer,
        "fixed_steering_effort_rms": fixed_steer,
        "linear_steering_effort_rms": linear_steer,
        "online_max_ok": max_ok,
        "online_steering_ok": steer_ok,
        "residual_prediction_improved": prediction_improved,
        "residual_prediction_error_improvement_percent": pred_improve,
        "rls_acceptance_rate": acceptance,
        "rls_relative_theta_update_mean": rel_update,
        "scenario_dir": str(result["scenario_dir"]),
    }
    row.update({f"param_{key}": value for key, value in params.items()})
    return row


def _copy_best_candidate(out_root: Path, row: Mapping) -> None:
    scenario_dir = Path(str(row["scenario_dir"]))
    best_dir = out_root / "best_candidate"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(scenario_dir, best_dir, dirs_exist_ok=True)
    write_json(best_dir / "best_candidate_metrics.json", row)
    write_json(out_root / "best_candidate_summary.json", row)


def run_online_advantage_search(args, out_root: Path) -> List[Dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "search_progress.csv"
    failed_path = out_root / "failed_candidates.csv"
    progress_rows: List[Dict] = _load_search_rows(progress_path) if args.search_resume else []
    completed_hashes = {str(row.get("candidate_hash", "")) for row in progress_rows}
    best_row = None
    for row in progress_rows:
        try:
            if best_row is None or float(row.get("score", -1.0e9)) > float(best_row.get("score", -1.0e9)):
                best_row = row
        except ValueError:
            continue

    started_count = len(completed_hashes)
    evaluated = 0
    for params in _search_candidate_sequence(args):
        if evaluated >= int(args.search_max_candidates):
            break
        candidate_hash = _candidate_hash(params)
        if candidate_hash in completed_hashes:
            continue
        idx = started_count + evaluated
        case_args = copy.deepcopy(args)
        _apply_search_candidate(case_args, params)
        case_dir = out_root / "candidate_runs" / f"case_{idx:03d}_{candidate_hash}"
        config_path = build_tire_config_override(case_args, case_dir)
        scenarios = build_scenarios(case_args)
        spec = scenarios["online_advantage_friction_shift_lane_change"]
        print(
            f"Search case {idx} [{candidate_hash}] "
            f"mu {params['mu_initial']:.2f}->{params['mu_final']:.2f}, "
            f"v={params['speed']:.1f}, lane={params['lane_shift']:.1f}, "
            f"train={params['train_duration']:.1f}s"
        )
        result = run_scenario(
            args=case_args,
            spec=spec,
            all_training_paths=[spec.path],
            out_root=case_dir,
            config_path=config_path,
        )
        rows = flatten_benchmark({spec.name: result})
        row = _search_result_row(
            idx=idx,
            candidate_hash=candidate_hash,
            params=params,
            result=result,
            rows=rows,
            case_dir=case_dir,
            target_improvement=float(args.search_target_online_fixed_improvement),
        )
        progress_rows.append(row)
        completed_hashes.add(candidate_hash)
        _write_search_rows(progress_path, progress_rows)
        failed_rows = [r for r in progress_rows if str(r.get("target_met", "")).lower() != "true"]
        _write_search_rows(failed_path, failed_rows)
        write_json(out_root / "search_progress.json", {"cases": progress_rows})
        if best_row is None or float(row["score"]) > float(best_row.get("score", -1.0e9)):
            best_row = row
            _copy_best_candidate(out_root, row)
        if row["target_met"]:
            _copy_best_candidate(out_root, row)
            print(
                "Target candidate found: "
                f"online/fixed improvement={100.0 * float(row['online_fixed_improvement_ratio']):.1f}%"
            )
            if not args.search_continue_after_success:
                break
        evaluated += 1

    if best_row is not None:
        _copy_best_candidate(out_root, best_row)
    write_json(out_root / "search_summary.json", {
        "target_online_fixed_improvement": float(args.search_target_online_fixed_improvement),
        "searched_cases_total": len(progress_rows),
        "new_cases_this_run": evaluated,
        "best_candidate": best_row,
        "success_cases": [
            row for row in progress_rows
            if str(row.get("target_met", "")).lower() == "true" or row.get("target_met") is True
        ],
    })
    return progress_rows


def _composite_candidate_sequence(args) -> Iterable[Dict]:
    hand_tuned = [
        {
            "stage": "patch20_lane32_success",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.68,
            "second_patch_mu": 0.60,
            "lane_shift": 3.2,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "excitation_amplitude": 0.015,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "patch20_best_guess",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.68,
            "second_patch_mu": 0.60,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "excitation_amplitude": 0.015,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "patch20_regularized_edmd",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.68,
            "second_patch_mu": 0.60,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "excitation_amplitude": 0.015,
            "edmd_ridge": 1.0e-3,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "patch20_more_regularized_edmd",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.68,
            "second_patch_mu": 0.60,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "excitation_amplitude": 0.015,
            "edmd_ridge": 1.0e-2,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "balanced_composite",
            "speed": 10.0,
            "train_duration": 5.0,
            "first_mu_final": 0.70,
            "second_patch_mu": 0.65,
            "lane_shift": 3.0,
            "lane_k": 0.20,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 36.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "milder_visual_course",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.72,
            "second_patch_mu": 0.68,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "milder_course_more_friction_shift",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.70,
            "second_patch_mu": 0.65,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.05,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "milder_course_low_initial_excitation",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.70,
            "second_patch_mu": 0.65,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "excitation_amplitude": 0.015,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "milder_course_low_excitation_stronger_rls",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.70,
            "second_patch_mu": 0.65,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "excitation_amplitude": 0.015,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.05,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "milder_course_deeper_patch",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.68,
            "second_patch_mu": 0.60,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "excitation_amplitude": 0.025,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "milder_course_deeper_patch_low_excitation",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.68,
            "second_patch_mu": 0.60,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "excitation_amplitude": 0.015,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "milder_course_conservative_rls",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.70,
            "second_patch_mu": 0.65,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.999,
            "rls_p0": 0.03,
            "rls_rel_update": 0.01,
        },
        {
            "stage": "milder_course_aggressive_rls",
            "speed": 9.5,
            "train_duration": 5.0,
            "first_mu_final": 0.70,
            "second_patch_mu": 0.65,
            "lane_shift": 2.8,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 38.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.995,
            "rls_p0": 0.10,
            "rls_rel_update": 0.03,
        },
        {
            "stage": "moderate_course_aggressive_rls",
            "speed": 10.0,
            "train_duration": 5.0,
            "first_mu_final": 0.70,
            "second_patch_mu": 0.65,
            "lane_shift": 3.0,
            "lane_k": 0.18,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.26,
            "adaptation_wavelength": 36.0,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "rls_forgetting": 0.995,
            "rls_p0": 0.10,
            "rls_rel_update": 0.03,
        },
        {
            "stage": "stronger_online_opportunity",
            "speed": 10.0,
            "train_duration": 5.0,
            "first_mu_final": 0.70,
            "second_patch_mu": 0.65,
            "lane_shift": 3.0,
            "lane_k": 0.20,
            "chicane_amplitude": 0.9,
            "chicane_wavelength": 38.0,
            "adaptation_amp": 0.26,
            "adaptation_wavelength": 34.0,
            "cf_scale": 0.60,
            "cr_scale": 0.65,
            "rls_forgetting": 0.997,
            "rls_p0": 0.05,
            "rls_rel_update": 0.02,
        },
        {
            "stage": "higher_speed_moderate",
            "speed": 10.5,
            "train_duration": 5.0,
            "first_mu_final": 0.72,
            "second_patch_mu": 0.68,
            "lane_shift": 3.0,
            "lane_k": 0.20,
            "chicane_amplitude": 0.9,
            "chicane_wavelength": 42.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 36.0,
            "cf_scale": 0.65,
            "cr_scale": 0.65,
            "rls_forgetting": 0.999,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
        },
    ]
    for params in hand_tuned:
        yield params

    rng = np.random.default_rng(int(args.composite_search_seed))
    for _ in range(max(0, int(args.composite_search_max_candidates) * 3)):
        yield {
            "stage": "seeded_composite",
            "speed": float(rng.choice([9.5, 10.0, 10.5])),
            "train_duration": float(rng.choice([5.0, 6.0])),
            "first_mu_final": float(rng.choice([0.70, 0.72, 0.75])),
            "second_patch_mu": float(rng.choice([0.65, 0.68, 0.70])),
            "lane_shift": float(rng.choice([2.8, 3.0, 3.2])),
            "lane_k": float(rng.choice([0.18, 0.20, 0.22])),
            "chicane_amplitude": float(rng.choice([0.7, 0.9, 1.1])),
            "chicane_wavelength": float(rng.choice([34.0, 38.0, 42.0])),
            "adaptation_amp": float(rng.choice([0.18, 0.22, 0.26])),
            "adaptation_wavelength": float(rng.choice([34.0, 38.0, 42.0])),
            "excitation_amplitude": float(rng.choice([0.015, 0.025, 0.04])),
            "edmd_ridge": float(rng.choice([1.0e-5, 1.0e-4, 1.0e-3])),
            "cf_scale": float(rng.choice([0.60, 0.65, 0.70])),
            "cr_scale": float(rng.choice([0.60, 0.65, 0.70])),
            "rls_forgetting": float(rng.choice([0.997, 0.999])),
            "rls_p0": float(rng.choice([0.01, 0.03, 0.05])),
            "rls_rel_update": float(rng.choice([0.01, 0.02])),
        }


def _apply_composite_candidate(args, params: Mapping) -> None:
    apply_composite_defaults(args)
    args.target_speed = float(params["speed"])
    args.train_duration = float(params["train_duration"])
    args.eval_duration = 27.0
    args.friction_mu_initial = 0.85
    args.friction_mu_final = float(params["first_mu_final"])
    args.friction_mu = 0.85
    args.second_patch_mu = float(params["second_patch_mu"])
    args.lane_shift = float(params["lane_shift"])
    args.lane_k = float(params["lane_k"])
    args.chicane_amplitude = float(params["chicane_amplitude"])
    args.chicane_wavelength = float(params["chicane_wavelength"])
    args.adaptation_weave_amplitude = float(params["adaptation_amp"])
    args.adaptation_weave_wavelength = float(params["adaptation_wavelength"])
    if "excitation_amplitude" in params:
        args.excitation_amplitude = float(params["excitation_amplitude"])
    args.linear_cf_scale = float(params["cf_scale"])
    args.linear_cr_scale = float(params["cr_scale"])
    args.linear_mass_scale = 1.05
    args.linear_izz_scale = 1.15
    args.linear_curvature_scale = 1.0
    args.rls_forgetting_factor = float(params["rls_forgetting"])
    args.rls_p0 = float(params["rls_p0"])
    args.rls_max_relative_theta_update = float(params["rls_rel_update"])
    args.rls_max_post_update_control_change = 0.12
    args.rls_reject_worse_factor = 1.03
    args.rls_reject_worse_margin = 1.0e-5
    if "edmd_ridge" in params:
        args.edmd_ridge = float(params["edmd_ridge"])


def _solver_failure_rate(run: Mapping) -> float:
    rows = list(run.get("rows", []))
    if not rows:
        return 1.0
    bad = 0
    for row in rows:
        status = str(row.get("solver_status", "")).lower()
        if status and "optimal" not in status:
            bad += 1
    return float(bad / len(rows))


def _composite_result_row(
    *,
    idx: int,
    candidate_hash: str,
    params: Mapping,
    result: Mapping,
    rows: List[Dict],
    case_dir: Path,
    fixed_target: float,
    linear_target: float,
) -> Dict:
    online = _metric_for(rows, "online_residual_koopman_mpc", "key_segments_lateral_error_rmse")
    fixed = _metric_for(rows, "fixed_residual_koopman_mpc", "key_segments_lateral_error_rmse")
    linear = _metric_for(rows, "linear_bicycle_mpc", "key_segments_lateral_error_rmse")
    online_overall = _metric_for(rows, "online_residual_koopman_mpc", "lateral_error_rmse")
    fixed_overall = _metric_for(rows, "fixed_residual_koopman_mpc", "lateral_error_rmse")
    linear_overall = _metric_for(rows, "linear_bicycle_mpc", "lateral_error_rmse")
    online_max = _metric_for(rows, "online_residual_koopman_mpc", "max_abs_lateral_error")
    fixed_max = _metric_for(rows, "fixed_residual_koopman_mpc", "max_abs_lateral_error")
    linear_max = _metric_for(rows, "linear_bicycle_mpc", "max_abs_lateral_error")
    online_steer = _metric_for(rows, "online_residual_koopman_mpc", "steering_effort_rms")
    linear_steer = _metric_for(rows, "linear_bicycle_mpc", "steering_effort_rms")
    pred_improve = _metric_for(rows, "online_residual_koopman_mpc", "residual_prediction_error_improvement_percent")
    acceptance = _metric_for(rows, "online_residual_koopman_mpc", "rls_acceptance_rate")
    online_fixed_improvement = (
        (fixed - online) / fixed if fixed is not None and online is not None and fixed > 1.0e-12 else None
    )
    online_linear_improvement = (
        (linear - online) / linear if linear is not None and online is not None and linear > 1.0e-12 else None
    )
    online_best = bool(
        online is not None
        and fixed is not None
        and linear is not None
        and online < fixed
        and online < linear
    )
    any_max = max(v for v in [online_max or 0.0, fixed_max or 0.0, linear_max or 0.0])
    online_max_ok = bool(online_max is not None and online_max <= 0.5)
    course_not_too_aggressive = bool(any_max <= 1.0)
    steering_ok = bool(
        online_steer is not None
        and linear_steer is not None
        and online_steer <= 1.5 * max(linear_steer, 1.0e-9)
    )
    prediction_improved = bool(pred_improve is not None and pred_improve > 0.0)
    solver_failure = max(_solver_failure_rate(run) for run in result.get("runs", {}).values())
    solver_ok = bool(solver_failure <= 0.10)
    target_met = bool(
        online_best
        and online_fixed_improvement is not None
        and online_linear_improvement is not None
        and online_fixed_improvement >= float(fixed_target)
        and online_linear_improvement >= float(linear_target)
        and online_max_ok
        and course_not_too_aggressive
        and steering_ok
        and prediction_improved
        and solver_ok
    )
    score = (
        (5.0 if target_met else 0.0)
        + (2.0 if online_best else 0.0)
        + float(online_fixed_improvement or -1.0)
        + float(online_linear_improvement or -1.0)
        + 0.01 * float(pred_improve or 0.0)
        - 0.2 * float(any_max)
        - 2.0 * float(solver_failure)
    )
    row = {
        "case": idx,
        "candidate_hash": candidate_hash,
        "case_dir": str(case_dir),
        "scenario_dir": str(result["scenario_dir"]),
        "target_met": target_met,
        "online_best_key_segments": online_best,
        "score": score,
        "online_fixed_improvement_ratio": online_fixed_improvement,
        "online_linear_improvement_ratio": online_linear_improvement,
        "online_key_segments_rmse": online,
        "fixed_key_segments_rmse": fixed,
        "linear_key_segments_rmse": linear,
        "online_overall_rmse": online_overall,
        "fixed_overall_rmse": fixed_overall,
        "linear_overall_rmse": linear_overall,
        "online_max_abs_lateral_error": online_max,
        "fixed_max_abs_lateral_error": fixed_max,
        "linear_max_abs_lateral_error": linear_max,
        "online_steering_effort_rms": online_steer,
        "linear_steering_effort_rms": linear_steer,
        "residual_prediction_improved": prediction_improved,
        "residual_prediction_error_improvement_percent": pred_improve,
        "rls_acceptance_rate": acceptance,
        "solver_failure_rate_max": solver_failure,
        "online_max_ok": online_max_ok,
        "course_not_too_aggressive": course_not_too_aggressive,
        "online_steering_ok": steering_ok,
    }
    row.update({f"param_{key}": value for key, value in params.items()})
    return row


def _copy_composite_best_candidate(out_root: Path, row: Mapping) -> None:
    scenario_dir = Path(str(row["scenario_dir"]))
    best_dir = out_root / "best_candidate"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(scenario_dir, best_dir, dirs_exist_ok=True)
    write_json(best_dir / "best_candidate_metrics.json", row)
    write_json(out_root / "composite_best_candidate_summary.json", row)


def run_composite_course_search(args, out_root: Path) -> List[Dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "composite_search_progress.csv"
    failed_path = out_root / "failed_candidates.csv"
    progress_rows = _load_search_rows(progress_path) if args.composite_search_resume else []
    completed_hashes = {str(row.get("candidate_hash", "")) for row in progress_rows}
    best_row = None
    for row in progress_rows:
        try:
            if best_row is None or float(row.get("score", -1.0e9)) > float(best_row.get("score", -1.0e9)):
                best_row = row
        except ValueError:
            continue
    started_count = len(completed_hashes)
    evaluated = 0
    for params in _composite_candidate_sequence(args):
        if evaluated >= int(args.composite_search_max_candidates):
            break
        candidate_hash = _candidate_hash(params)
        if candidate_hash in completed_hashes:
            continue
        idx = started_count + evaluated
        case_args = copy.deepcopy(args)
        _apply_composite_candidate(case_args, params)
        case_dir = out_root / "candidate_runs" / f"case_{idx:03d}_{candidate_hash}"
        config_path = build_tire_config_override(case_args, case_dir)
        scenarios = build_scenarios(case_args)
        spec = scenarios["composite_friction_adaptation_course"]
        print(
            f"Composite case {idx} [{candidate_hash}] "
            f"v={params['speed']:.1f}, train={params['train_duration']:.1f}s, "
            f"mu=0.85->{params['first_mu_final']:.2f}->{params['second_patch_mu']:.2f}, "
            f"lane={params['lane_shift']:.1f}, chicane={params['chicane_amplitude']:.1f}"
        )
        result = run_scenario(
            args=case_args,
            spec=spec,
            all_training_paths=[spec.path],
            out_root=case_dir,
            config_path=config_path,
        )
        rows = flatten_benchmark({spec.name: result})
        write_run_csv(result["scenario_dir"] / "benchmark_summary.csv", rows)
        write_json(result["scenario_dir"] / "benchmark_summary.json", {
            "plant_description": PLANT_DESCRIPTION,
            "scenario": spec.name,
            "key_segments": result["summary"].get("key_segments", []),
            "table": rows,
        })
        row = _composite_result_row(
            idx=idx,
            candidate_hash=candidate_hash,
            params=params,
            result=result,
            rows=rows,
            case_dir=case_dir,
            fixed_target=float(args.composite_target_online_fixed_improvement),
            linear_target=float(args.composite_target_online_linear_improvement),
        )
        progress_rows.append(row)
        completed_hashes.add(candidate_hash)
        _write_search_rows(progress_path, progress_rows)
        _write_search_rows(failed_path, [
            r for r in progress_rows
            if str(r.get("target_met", "")).lower() != "true"
        ])
        write_json(out_root / "composite_search_progress.json", {"cases": progress_rows})
        if best_row is None or float(row["score"]) > float(best_row.get("score", -1.0e9)):
            best_row = row
            _copy_composite_best_candidate(out_root, row)
        if row["target_met"]:
            _copy_composite_best_candidate(out_root, row)
            print(
                "Composite target candidate found: "
                f"online/fixed={100.0 * float(row['online_fixed_improvement_ratio']):.1f}%, "
                f"online/linear={100.0 * float(row['online_linear_improvement_ratio']):.1f}%"
            )
            if not args.composite_search_continue_after_success:
                break
        evaluated += 1
    if best_row is not None:
        _copy_composite_best_candidate(out_root, best_row)
    write_json(out_root / "composite_search_summary.json", {
        "target_online_fixed_improvement": float(args.composite_target_online_fixed_improvement),
        "target_online_linear_improvement": float(args.composite_target_online_linear_improvement),
        "searched_cases_total": len(progress_rows),
        "new_cases_this_run": evaluated,
        "best_candidate": best_row,
        "success_cases": [
            row for row in progress_rows
            if str(row.get("target_met", "")).lower() == "true" or row.get("target_met") is True
        ],
    })
    return progress_rows


def _award_candidate_sequence(args) -> Iterable[Dict]:
    if getattr(args, "gap_boost", False):
        yield from _gap_boost_candidate_sequence(args)
        return

    hand_tuned = [
        {
            "stage": "balanced_nonstationary",
            "initial_speed_kmh": 36.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 40.0,
            "mu_initial": 0.90,
            "mu_mid": 0.72,
            "mu_chicane": 0.68,
            "mu_low": 0.62,
            "lane_shift": 3.2,
            "lane_k": 0.14,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 65.0,
            "adaptation_amp": 0.20,
            "adaptation_wavelength": 55.0,
            "train_duration": 5.0,
            "excitation_amplitude": 0.02,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.997,
            "rls_p0": 0.03,
            "rls_rel_update": 0.02,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.03,
        },
        {
            "stage": "limited_training_stronger_shift",
            "initial_speed_kmh": 36.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 40.0,
            "mu_initial": 0.90,
            "mu_mid": 0.70,
            "mu_chicane": 0.66,
            "mu_low": 0.60,
            "lane_shift": 3.2,
            "lane_k": 0.14,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 70.0,
            "adaptation_amp": 0.22,
            "adaptation_wavelength": 55.0,
            "train_duration": 4.0,
            "excitation_amplitude": 0.02,
            "cf_scale": 0.60,
            "cr_scale": 0.65,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.997,
            "rls_p0": 0.05,
            "rls_rel_update": 0.02,
            "rls_control_jump": 0.12,
            "rls_reject_worse": 1.03,
        },
        {
            "stage": "high_speed_moderate_path",
            "initial_speed_kmh": 38.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 40.0,
            "mu_initial": 0.95,
            "mu_mid": 0.72,
            "mu_chicane": 0.68,
            "mu_low": 0.62,
            "lane_shift": 3.0,
            "lane_k": 0.12,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 75.0,
            "adaptation_amp": 0.20,
            "adaptation_wavelength": 60.0,
            "train_duration": 4.0,
            "excitation_amplitude": 0.025,
            "cf_scale": 0.55,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.20,
            "rls_forgetting": 0.995,
            "rls_p0": 0.05,
            "rls_rel_update": 0.03,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.05,
        },
        {
            "stage": "fixed_limited_online_window",
            "initial_speed_kmh": 34.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 40.0,
            "mu_initial": 0.90,
            "mu_mid": 0.68,
            "mu_chicane": 0.65,
            "mu_low": 0.60,
            "lane_shift": 3.4,
            "lane_k": 0.12,
            "chicane_amplitude": 0.9,
            "chicane_wavelength": 75.0,
            "adaptation_amp": 0.25,
            "adaptation_wavelength": 60.0,
            "train_duration": 3.0,
            "excitation_amplitude": 0.025,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.995,
            "rls_p0": 0.05,
            "rls_rel_update": 0.03,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.03,
        },
    ]
    for params in hand_tuned:
        yield params

    rng = np.random.default_rng(int(args.search_seed))
    for _ in range(max(0, int(args.search_max_candidates) * 3)):
        mu_initial = float(rng.choice([0.85, 0.90, 0.95]))
        mu_mid = float(rng.choice([0.68, 0.70, 0.72, 0.75]))
        mu_chicane = float(rng.choice([0.65, 0.68, 0.70]))
        mu_low = float(rng.choice([0.60, 0.62, 0.65, 0.68]))
        if not (mu_initial > mu_mid >= mu_chicane >= mu_low):
            mu_mid = min(mu_mid, mu_initial - 0.08)
            mu_chicane = min(mu_chicane, mu_mid)
            mu_low = min(mu_low, mu_chicane)
        yield {
            "stage": "seeded_award_ready",
            "initial_speed_kmh": float(rng.choice([34.0, 36.0, 38.0])),
            "mid_speed_kmh": float(rng.choice([42.0, 45.0])),
            "high_speed_kmh": float(rng.choice([47.0, 50.0])),
            "exit_speed_kmh": float(rng.choice([38.0, 40.0, 42.0])),
            "mu_initial": mu_initial,
            "mu_mid": mu_mid,
            "mu_chicane": mu_chicane,
            "mu_low": mu_low,
            "lane_shift": float(rng.choice([3.0, 3.2, 3.4])),
            "lane_k": float(rng.choice([0.10, 0.12, 0.14, 0.18])),
            "chicane_amplitude": float(rng.choice([0.5, 0.7, 0.9])),
            "chicane_wavelength": float(rng.choice([55.0, 65.0, 75.0, 80.0])),
            "adaptation_amp": float(rng.choice([0.15, 0.20, 0.25])),
            "adaptation_wavelength": float(rng.choice([55.0, 60.0, 70.0])),
            "train_duration": float(rng.choice([3.0, 4.0, 5.0, 6.0])),
            "excitation_amplitude": float(rng.choice([0.015, 0.02, 0.025, 0.035])),
            "edmd_ridge": float(rng.choice([1.0e-5, 1.0e-4, 1.0e-3])),
            "cf_scale": float(rng.choice([0.55, 0.60, 0.65, 0.70, 0.75])),
            "cr_scale": float(rng.choice([0.55, 0.60, 0.65, 0.70, 0.75])),
            "mass_scale": float(rng.choice([1.00, 1.05, 1.10])),
            "izz_scale": float(rng.choice([1.05, 1.15, 1.25])),
            "rls_forgetting": float(rng.choice([0.995, 0.997, 0.999])),
            "rls_p0": float(rng.choice([0.01, 0.03, 0.05, 0.10])),
            "rls_rel_update": float(rng.choice([0.01, 0.02, 0.03, 0.04])),
            "rls_control_jump": float(rng.choice([0.08, 0.12, 0.15])),
            "rls_reject_worse": float(rng.choice([1.00, 1.03, 1.05])),
        }


def _gap_boost_candidate_sequence(args) -> Iterable[Dict]:
    hand_tuned = [
        {
            "stage": "gap_short_training_mid_rls",
            "initial_speed_kmh": 36.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 40.0,
            "mu_initial": 0.90,
            "mu_mid": 0.68,
            "mu_chicane": 0.64,
            "mu_low": 0.58,
            "lane_shift": 3.4,
            "lane_k": 0.14,
            "chicane_amplitude": 0.9,
            "chicane_wavelength": 65.0,
            "adaptation_amp": 0.26,
            "adaptation_wavelength": 55.0,
            "train_duration": 3.0,
            "excitation_amplitude": 0.02,
            "edmd_ridge": 1.0e-4,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.995,
            "rls_p0": 0.05,
            "rls_rel_update": 0.03,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.05,
        },
        {
            "stage": "gap_very_limited_fixed",
            "initial_speed_kmh": 34.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 40.0,
            "mu_initial": 0.95,
            "mu_mid": 0.70,
            "mu_chicane": 0.66,
            "mu_low": 0.60,
            "lane_shift": 3.6,
            "lane_k": 0.14,
            "chicane_amplitude": 0.9,
            "chicane_wavelength": 70.0,
            "adaptation_amp": 0.30,
            "adaptation_wavelength": 60.0,
            "train_duration": 3.0,
            "excitation_amplitude": 0.02,
            "edmd_ridge": 1.0e-4,
            "cf_scale": 0.60,
            "cr_scale": 0.65,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.995,
            "rls_p0": 0.08,
            "rls_rel_update": 0.04,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.05,
        },
        {
            "stage": "gap_balanced_highlight",
            "initial_speed_kmh": 36.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 40.0,
            "mu_initial": 0.95,
            "mu_mid": 0.68,
            "mu_chicane": 0.64,
            "mu_low": 0.60,
            "lane_shift": 3.4,
            "lane_k": 0.16,
            "chicane_amplitude": 0.7,
            "chicane_wavelength": 70.0,
            "adaptation_amp": 0.26,
            "adaptation_wavelength": 55.0,
            "train_duration": 3.5,
            "excitation_amplitude": 0.025,
            "edmd_ridge": 1.0e-4,
            "cf_scale": 0.55,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.20,
            "rls_forgetting": 0.995,
            "rls_p0": 0.08,
            "rls_rel_update": 0.04,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.05,
        },
        {
            "stage": "gap_maneuver_strong_guarded",
            "initial_speed_kmh": 36.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 40.0,
            "mu_initial": 0.90,
            "mu_mid": 0.70,
            "mu_chicane": 0.66,
            "mu_low": 0.60,
            "lane_shift": 3.6,
            "lane_k": 0.16,
            "chicane_amplitude": 1.0,
            "chicane_wavelength": 70.0,
            "adaptation_amp": 0.30,
            "adaptation_wavelength": 60.0,
            "train_duration": 3.5,
            "excitation_amplitude": 0.02,
            "edmd_ridge": 1.0e-3,
            "cf_scale": 0.60,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.997,
            "rls_p0": 0.10,
            "rls_rel_update": 0.04,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.08,
        },
    ]
    for params in hand_tuned:
        yield params

    rng = np.random.default_rng(int(args.search_seed) + 173)
    for _ in range(max(0, int(args.search_max_candidates) * 4)):
        mu_initial = float(rng.choice([0.90, 0.95]))
        mu_mid = float(rng.choice([0.68, 0.70, 0.72]))
        mu_chicane = float(rng.choice([0.64, 0.66, 0.68]))
        mu_low = float(rng.choice([0.58, 0.60, 0.62]))
        if not (mu_initial > mu_mid >= mu_chicane >= mu_low):
            mu_mid = min(mu_mid, mu_initial - 0.12)
            mu_chicane = min(mu_chicane, mu_mid)
            mu_low = min(mu_low, mu_chicane)
        yield {
            "stage": "seeded_gap_boost",
            "initial_speed_kmh": float(rng.choice([34.0, 36.0, 38.0])),
            "mid_speed_kmh": float(rng.choice([42.0, 45.0])),
            "high_speed_kmh": float(rng.choice([47.0, 50.0])),
            "exit_speed_kmh": float(rng.choice([38.0, 40.0, 42.0])),
            "mu_initial": mu_initial,
            "mu_mid": mu_mid,
            "mu_chicane": mu_chicane,
            "mu_low": mu_low,
            "lane_shift": float(rng.choice([3.2, 3.4, 3.6])),
            "lane_k": float(rng.choice([0.14, 0.16, 0.18])),
            "chicane_amplitude": float(rng.choice([0.7, 0.9, 1.0])),
            "chicane_wavelength": float(rng.choice([60.0, 65.0, 70.0])),
            "adaptation_amp": float(rng.choice([0.22, 0.26, 0.30])),
            "adaptation_wavelength": float(rng.choice([50.0, 55.0, 60.0])),
            "train_duration": float(rng.choice([3.0, 3.5, 4.0])),
            "excitation_amplitude": float(rng.choice([0.015, 0.02, 0.025])),
            "edmd_ridge": float(rng.choice([1.0e-5, 1.0e-4, 1.0e-3])),
            "cf_scale": float(rng.choice([0.55, 0.60, 0.65, 0.70])),
            "cr_scale": float(rng.choice([0.55, 0.60, 0.65, 0.70])),
            "mass_scale": float(rng.choice([1.00, 1.05, 1.10])),
            "izz_scale": float(rng.choice([1.05, 1.15, 1.25])),
            "rls_forgetting": float(rng.choice([0.995, 0.997])),
            "rls_p0": float(rng.choice([0.05, 0.08, 0.10])),
            "rls_rel_update": float(rng.choice([0.03, 0.04, 0.05])),
            "rls_control_jump": float(rng.choice([0.12, 0.15])),
            "rls_reject_worse": float(rng.choice([1.03, 1.05, 1.08])),
        }


def _late_gap_boost_candidate_sequence(args) -> Iterable[Dict]:
    hand_tuned = [
        {
            "stage": "late_recovery_highlight_seed",
            "initial_speed_kmh": 34.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 44.0,
            "mu_initial": 0.95,
            "mu_mid": 0.70,
            "mu_chicane": 0.64,
            "mu_low": 0.60,
            "mu_final_recovery": 0.72,
            "final_friction_mode": "recovery",
            "lane_shift": 3.6,
            "lane_k": 0.14,
            "chicane_amplitude": 0.9,
            "chicane_wavelength": 70.0,
            "adaptation_amp": 0.30,
            "adaptation_wavelength": 60.0,
            "train_duration": 3.0,
            "excitation_amplitude": 0.02,
            "edmd_ridge": 1.0e-4,
            "cf_scale": 0.60,
            "cr_scale": 0.65,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.995,
            "rls_p0": 0.08,
            "rls_rel_update": 0.04,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.05,
        },
        {
            "stage": "late_second_drop_seed",
            "initial_speed_kmh": 36.0,
            "mid_speed_kmh": 45.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 44.0,
            "mu_initial": 0.95,
            "mu_mid": 0.72,
            "mu_chicane": 0.66,
            "mu_low": 0.60,
            "mu_final_recovery": 0.72,
            "final_friction_mode": "second_drop",
            "lane_shift": 3.6,
            "lane_k": 0.12,
            "chicane_amplitude": 0.9,
            "chicane_wavelength": 70.0,
            "adaptation_amp": 0.30,
            "adaptation_wavelength": 60.0,
            "train_duration": 3.5,
            "excitation_amplitude": 0.02,
            "edmd_ridge": 1.0e-4,
            "cf_scale": 0.60,
            "cr_scale": 0.65,
            "mass_scale": 1.05,
            "izz_scale": 1.20,
            "rls_forgetting": 0.995,
            "rls_p0": 0.10,
            "rls_rel_update": 0.04,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.05,
        },
        {
            "stage": "late_low_mu_stronger_final",
            "initial_speed_kmh": 34.0,
            "mid_speed_kmh": 43.0,
            "high_speed_kmh": 50.0,
            "exit_speed_kmh": 46.0,
            "mu_initial": 0.90,
            "mu_mid": 0.68,
            "mu_chicane": 0.64,
            "mu_low": 0.58,
            "mu_final_recovery": 0.70,
            "final_friction_mode": "recovery",
            "lane_shift": 3.8,
            "lane_k": 0.12,
            "chicane_amplitude": 1.0,
            "chicane_wavelength": 75.0,
            "adaptation_amp": 0.28,
            "adaptation_wavelength": 60.0,
            "train_duration": 3.0,
            "excitation_amplitude": 0.02,
            "edmd_ridge": 1.0e-4,
            "cf_scale": 0.55,
            "cr_scale": 0.60,
            "mass_scale": 1.05,
            "izz_scale": 1.20,
            "rls_forgetting": 0.995,
            "rls_p0": 0.10,
            "rls_rel_update": 0.05,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.08,
        },
        {
            "stage": "late_balanced_safe_shift",
            "initial_speed_kmh": 36.0,
            "mid_speed_kmh": 43.0,
            "high_speed_kmh": 48.0,
            "exit_speed_kmh": 42.0,
            "mu_initial": 0.90,
            "mu_mid": 0.72,
            "mu_chicane": 0.66,
            "mu_low": 0.60,
            "mu_final_recovery": 0.75,
            "final_friction_mode": "recovery",
            "lane_shift": 3.4,
            "lane_k": 0.14,
            "chicane_amplitude": 0.8,
            "chicane_wavelength": 70.0,
            "adaptation_amp": 0.28,
            "adaptation_wavelength": 60.0,
            "train_duration": 4.0,
            "excitation_amplitude": 0.02,
            "edmd_ridge": 1.0e-4,
            "cf_scale": 0.65,
            "cr_scale": 0.65,
            "mass_scale": 1.05,
            "izz_scale": 1.15,
            "rls_forgetting": 0.997,
            "rls_p0": 0.08,
            "rls_rel_update": 0.04,
            "rls_control_jump": 0.15,
            "rls_reject_worse": 1.05,
        },
    ]
    for params in hand_tuned:
        yield params

    rng = np.random.default_rng(int(args.search_seed) + 431)
    for _ in range(max(0, int(args.search_max_candidates) * 5)):
        mu_initial = float(rng.choice([0.90, 0.95]))
        mu_mid = float(rng.choice([0.68, 0.70, 0.72]))
        mu_chicane = float(rng.choice([0.64, 0.66, 0.68]))
        mu_low = float(rng.choice([0.58, 0.60, 0.62]))
        if not (mu_initial > mu_mid >= mu_chicane >= mu_low):
            mu_mid = min(mu_mid, mu_initial - 0.12)
            mu_chicane = min(mu_chicane, mu_mid)
            mu_low = min(mu_low, mu_chicane)
        yield {
            "stage": "seeded_late_gap_boost",
            "initial_speed_kmh": float(rng.choice([34.0, 36.0])),
            "mid_speed_kmh": float(rng.choice([43.0, 45.0])),
            "high_speed_kmh": float(rng.choice([48.0, 50.0])),
            "exit_speed_kmh": float(rng.choice([42.0, 44.0, 46.0])),
            "mu_initial": mu_initial,
            "mu_mid": mu_mid,
            "mu_chicane": mu_chicane,
            "mu_low": mu_low,
            "mu_final_recovery": float(rng.choice([0.70, 0.72, 0.75])),
            "final_friction_mode": str(rng.choice(["recovery", "second_drop"])),
            "lane_shift": float(rng.choice([3.4, 3.6, 3.8])),
            "lane_k": float(rng.choice([0.10, 0.12, 0.14, 0.16])),
            "chicane_amplitude": float(rng.choice([0.8, 0.9, 1.0])),
            "chicane_wavelength": float(rng.choice([65.0, 70.0, 75.0])),
            "adaptation_amp": float(rng.choice([0.24, 0.28, 0.32])),
            "adaptation_wavelength": float(rng.choice([55.0, 60.0, 65.0])),
            "train_duration": float(rng.choice([3.0, 3.5, 4.0])),
            "excitation_amplitude": float(rng.choice([0.015, 0.02, 0.025])),
            "edmd_ridge": float(rng.choice([1.0e-5, 1.0e-4, 1.0e-3])),
            "cf_scale": float(rng.choice([0.55, 0.60, 0.65])),
            "cr_scale": float(rng.choice([0.60, 0.65, 0.70])),
            "mass_scale": 1.05,
            "izz_scale": float(rng.choice([1.15, 1.20])),
            "rls_forgetting": float(rng.choice([0.995, 0.997])),
            "rls_p0": float(rng.choice([0.06, 0.08, 0.10, 0.12])),
            "rls_rel_update": float(rng.choice([0.03, 0.04, 0.05])),
            "rls_control_jump": float(rng.choice([0.12, 0.15])),
            "rls_reject_worse": float(rng.choice([1.03, 1.05, 1.08])),
        }


def _apply_award_candidate(args, params: Mapping) -> None:
    apply_award_ready_defaults(args)
    args.initial_speed_kmh = float(params["initial_speed_kmh"])
    args.mid_speed_kmh = float(params["mid_speed_kmh"])
    args.high_speed_kmh = float(params["high_speed_kmh"])
    args.exit_speed_kmh = float(params["exit_speed_kmh"])
    args.target_speed = float(args.initial_speed_kmh) / 3.6
    args.friction_mu_initial = float(params["mu_initial"])
    args.friction_mu_final = float(params["mu_mid"])
    args.friction_mu = float(params["mu_initial"])
    args.chicane_mu = float(params["mu_chicane"])
    args.second_patch_mu = float(params["mu_low"])
    args.lane_shift = float(params["lane_shift"])
    args.lane_k = float(params["lane_k"])
    args.chicane_amplitude = float(params["chicane_amplitude"])
    args.chicane_wavelength = float(params["chicane_wavelength"])
    args.adaptation_weave_amplitude = float(params["adaptation_amp"])
    args.adaptation_weave_wavelength = float(params["adaptation_wavelength"])
    args.train_duration = float(params["train_duration"])
    args.eval_duration = 55.0
    args.excitation_amplitude = float(params["excitation_amplitude"])
    if "edmd_ridge" in params:
        args.edmd_ridge = float(params["edmd_ridge"])
    args.linear_cf_scale = float(params["cf_scale"])
    args.linear_cr_scale = float(params["cr_scale"])
    args.linear_mass_scale = float(params["mass_scale"])
    args.linear_izz_scale = float(params["izz_scale"])
    args.linear_curvature_scale = 1.0
    args.rls_forgetting_factor = float(params["rls_forgetting"])
    args.rls_p0 = float(params["rls_p0"])
    args.rls_max_relative_theta_update = float(params["rls_rel_update"])
    args.rls_max_post_update_control_change = float(params["rls_control_jump"])
    args.rls_reject_worse_factor = float(params["rls_reject_worse"])
    args.rls_reject_worse_margin = 1.0e-5


def _apply_late_gap_candidate(args, params: Mapping) -> None:
    apply_late_gap_boost_defaults(args)
    args.initial_speed_kmh = float(params["initial_speed_kmh"])
    args.mid_speed_kmh = float(params["mid_speed_kmh"])
    args.high_speed_kmh = float(params["high_speed_kmh"])
    args.exit_speed_kmh = float(params["exit_speed_kmh"])
    args.target_speed = float(args.initial_speed_kmh) / 3.6
    args.friction_mu_initial = float(params["mu_initial"])
    args.friction_mu_final = float(params["mu_mid"])
    args.friction_mu = float(params["mu_initial"])
    args.chicane_mu = float(params["mu_chicane"])
    args.second_patch_mu = float(params["mu_low"])
    args.mu_final_recovery = float(params["mu_final_recovery"])
    args.final_friction_mode = str(params["final_friction_mode"])
    args.lane_shift = float(params["lane_shift"])
    args.lane_k = float(params["lane_k"])
    args.chicane_amplitude = float(params["chicane_amplitude"])
    args.chicane_wavelength = float(params["chicane_wavelength"])
    args.adaptation_weave_amplitude = float(params["adaptation_amp"])
    args.adaptation_weave_wavelength = float(params["adaptation_wavelength"])
    args.train_duration = float(params["train_duration"])
    args.eval_duration = 55.0
    args.excitation_amplitude = float(params["excitation_amplitude"])
    if "edmd_ridge" in params:
        args.edmd_ridge = float(params["edmd_ridge"])
    args.linear_cf_scale = float(params["cf_scale"])
    args.linear_cr_scale = float(params["cr_scale"])
    args.linear_mass_scale = float(params["mass_scale"])
    args.linear_izz_scale = float(params["izz_scale"])
    args.linear_curvature_scale = 1.0
    args.rls_forgetting_factor = float(params["rls_forgetting"])
    args.rls_p0 = float(params["rls_p0"])
    args.rls_max_relative_theta_update = float(params["rls_rel_update"])
    args.rls_max_post_update_control_change = float(params["rls_control_jump"])
    args.rls_reject_worse_factor = float(params["rls_reject_worse"])
    args.rls_reject_worse_margin = 1.0e-5


def _award_result_row(
    *,
    idx: int,
    candidate_hash: str,
    params: Mapping,
    result: Mapping,
    rows: List[Dict],
    case_dir: Path,
    target_improvement: float,
) -> Dict:
    online = _metric_for(rows, "online_residual_koopman_mpc", "key_segments_lateral_error_rmse")
    fixed = _metric_for(rows, "fixed_residual_koopman_mpc", "key_segments_lateral_error_rmse")
    linear = _metric_for(rows, "linear_bicycle_mpc", "key_segments_lateral_error_rmse")
    online_overall = _metric_for(rows, "online_residual_koopman_mpc", "lateral_error_rmse")
    fixed_overall = _metric_for(rows, "fixed_residual_koopman_mpc", "lateral_error_rmse")
    linear_overall = _metric_for(rows, "linear_bicycle_mpc", "lateral_error_rmse")
    online_max = _metric_for(rows, "online_residual_koopman_mpc", "max_abs_lateral_error")
    fixed_max = _metric_for(rows, "fixed_residual_koopman_mpc", "max_abs_lateral_error")
    linear_max = _metric_for(rows, "linear_bicycle_mpc", "max_abs_lateral_error")
    online_steer = _metric_for(rows, "online_residual_koopman_mpc", "steering_effort_rms")
    fixed_steer = _metric_for(rows, "fixed_residual_koopman_mpc", "steering_effort_rms")
    linear_steer = _metric_for(rows, "linear_bicycle_mpc", "steering_effort_rms")
    pred_improve = _metric_for(rows, "online_residual_koopman_mpc", "residual_prediction_error_improvement_percent")
    acceptance = _metric_for(rows, "online_residual_koopman_mpc", "rls_acceptance_rate")
    rel_update = _metric_for(rows, "online_residual_koopman_mpc", "rls_relative_theta_update_mean")
    solver_failure = max(_solver_failure_rate(run) for run in result.get("runs", {}).values())
    best_baseline = None
    best_baseline_name = ""
    if fixed is not None and linear is not None:
        if fixed <= linear:
            best_baseline = fixed
            best_baseline_name = "fixed_residual_koopman_mpc"
        else:
            best_baseline = linear
            best_baseline_name = "linear_bicycle_mpc"
    improvement = (
        (best_baseline - online) / best_baseline
        if best_baseline is not None and online is not None and best_baseline > 1.0e-12 else None
    )
    online_best = bool(
        online is not None
        and fixed is not None
        and linear is not None
        and online < min(fixed, linear)
    )
    any_max = max(v for v in [online_max or 0.0, fixed_max or 0.0, linear_max or 0.0])
    best_non_online_steer = min(
        v for v in [fixed_steer, linear_steer]
        if v is not None and np.isfinite(v)
    ) if any(v is not None and np.isfinite(v) for v in [fixed_steer, linear_steer]) else None
    online_max_ok = bool(online_max is not None and online_max <= 0.5)
    course_not_too_aggressive = bool(any_max <= 1.0)
    steering_ok = bool(
        online_steer is not None
        and best_non_online_steer is not None
        and online_steer <= 1.5 * max(best_non_online_steer, 1.0e-9)
    )
    prediction_improved = bool(pred_improve is not None and pred_improve > 0.0)
    solver_ok = bool(solver_failure <= 0.10)
    target_met = bool(
        online_best
        and improvement is not None
        and improvement >= float(target_improvement)
        and online_max_ok
        and course_not_too_aggressive
        and steering_ok
        and solver_ok
    )
    tier = (
        "poster_highlight" if improvement is not None and improvement >= 0.20 and target_met else
        "strong" if improvement is not None and improvement >= 0.10 and online_best else
        "acceptable" if improvement is not None and improvement >= 0.05 and online_best else
        "diagnostic"
    )
    score = (
        (6.0 if target_met else 0.0)
        + (2.0 if online_best else 0.0)
        + 2.0 * float(improvement or -1.0)
        + 0.01 * float(pred_improve or 0.0)
        - 0.2 * float(any_max)
        - 2.0 * float(solver_failure)
    )
    row = {
        "case": idx,
        "candidate_hash": candidate_hash,
        "case_dir": str(case_dir),
        "scenario_dir": str(result["scenario_dir"]),
        "target_met": target_met,
        "result_tier": tier,
        "online_best_key_segments": online_best,
        "best_non_online_baseline": best_baseline_name,
        "score": score,
        "online_best_baseline_improvement_ratio": improvement,
        "online_best_baseline_improvement_percent": None if improvement is None else 100.0 * float(improvement),
        "online_key_segments_rmse": online,
        "online_key_segments_rmse_mm": None if online is None else 1000.0 * float(online),
        "fixed_key_segments_rmse": fixed,
        "fixed_key_segments_rmse_mm": None if fixed is None else 1000.0 * float(fixed),
        "linear_key_segments_rmse": linear,
        "linear_key_segments_rmse_mm": None if linear is None else 1000.0 * float(linear),
        "online_overall_rmse": online_overall,
        "online_overall_rmse_mm": None if online_overall is None else 1000.0 * float(online_overall),
        "fixed_overall_rmse": fixed_overall,
        "fixed_overall_rmse_mm": None if fixed_overall is None else 1000.0 * float(fixed_overall),
        "linear_overall_rmse": linear_overall,
        "linear_overall_rmse_mm": None if linear_overall is None else 1000.0 * float(linear_overall),
        "online_max_abs_lateral_error": online_max,
        "online_max_abs_lateral_error_mm": None if online_max is None else 1000.0 * float(online_max),
        "fixed_max_abs_lateral_error": fixed_max,
        "fixed_max_abs_lateral_error_mm": None if fixed_max is None else 1000.0 * float(fixed_max),
        "linear_max_abs_lateral_error": linear_max,
        "linear_max_abs_lateral_error_mm": None if linear_max is None else 1000.0 * float(linear_max),
        "online_steering_effort_rms": online_steer,
        "fixed_steering_effort_rms": fixed_steer,
        "linear_steering_effort_rms": linear_steer,
        "residual_prediction_improved": prediction_improved,
        "residual_prediction_error_improvement_percent": pred_improve,
        "rls_acceptance_rate": acceptance,
        "rls_relative_theta_update_mean": rel_update,
        "solver_failure_rate_max": solver_failure,
        "online_max_ok": online_max_ok,
        "course_not_too_aggressive": course_not_too_aggressive,
        "online_steering_ok": steering_ok,
    }
    row.update({f"param_{key}": value for key, value in params.items()})
    return row


def _segment_metric_from_result(
    result: Mapping,
    controller: str,
    segment_name: str,
    metric_name: str,
) -> float | None:
    metrics = (
        result.get("summary", {})
        .get("segment_metrics", {})
        .get(controller, {})
        .get(segment_name, {})
    )
    value = metrics.get(metric_name)
    return None if value is None else float(value)


def _combined_segment_rmse_from_run(run: Mapping, segment_names: Iterable[str]) -> float | None:
    selected = {str(name) for name in segment_names}
    rows = [
        row for row in run.get("rows", [])
        if str(row.get("segment_name", "")) in selected
    ]
    if not rows:
        return None
    errors = np.asarray([row["e_y"] for row in rows], dtype=float)
    return float(np.sqrt(np.mean(errors * errors)))


def _late_gap_result_row(
    *,
    idx: int,
    candidate_hash: str,
    params: Mapping,
    result: Mapping,
    rows: List[Dict],
    case_dir: Path,
    target_improvement: float,
    target_late_improvement: float,
) -> Dict:
    row = _award_result_row(
        idx=idx,
        candidate_hash=candidate_hash,
        params=params,
        result=result,
        rows=rows,
        case_dir=case_dir,
        target_improvement=target_improvement,
    )
    late_segments = ["low_friction_patch", "final_evasive_maneuver"]
    controller_names = [
        "linear_bicycle_mpc",
        "fixed_residual_koopman_mpc",
        "online_residual_koopman_mpc",
    ]
    late_rmse = {
        name: _combined_segment_rmse_from_run(result.get("runs", {}).get(name, {}), late_segments)
        for name in controller_names
    }
    online_late = late_rmse["online_residual_koopman_mpc"]
    fixed_late = late_rmse["fixed_residual_koopman_mpc"]
    linear_late = late_rmse["linear_bicycle_mpc"]
    late_best_baseline = None
    late_best_name = ""
    if fixed_late is not None and linear_late is not None:
        if fixed_late <= linear_late:
            late_best_baseline = fixed_late
            late_best_name = "fixed_residual_koopman_mpc"
        else:
            late_best_baseline = linear_late
            late_best_name = "linear_bicycle_mpc"
    late_improvement = (
        (late_best_baseline - online_late) / late_best_baseline
        if late_best_baseline is not None and online_late is not None and late_best_baseline > 1.0e-12
        else None
    )
    low_online = _segment_metric_from_result(result, "online_residual_koopman_mpc", "low_friction_patch", "lateral_error_rmse")
    low_fixed = _segment_metric_from_result(result, "fixed_residual_koopman_mpc", "low_friction_patch", "lateral_error_rmse")
    low_linear = _segment_metric_from_result(result, "linear_bicycle_mpc", "low_friction_patch", "lateral_error_rmse")
    final_online = _segment_metric_from_result(result, "online_residual_koopman_mpc", "final_evasive_maneuver", "lateral_error_rmse")
    final_fixed = _segment_metric_from_result(result, "fixed_residual_koopman_mpc", "final_evasive_maneuver", "lateral_error_rmse")
    final_linear = _segment_metric_from_result(result, "linear_bicycle_mpc", "final_evasive_maneuver", "lateral_error_rmse")
    low_online_best = bool(
        low_online is not None
        and low_fixed is not None
        and low_linear is not None
        and low_online < min(low_fixed, low_linear)
    )
    final_online_best = bool(
        final_online is not None
        and final_fixed is not None
        and final_linear is not None
        and final_online < min(final_fixed, final_linear)
    )
    low_reduction = _segment_metric_from_result(
        result,
        "online_residual_koopman_mpc",
        "low_friction_patch",
        "residual_error_reduction_mean",
    )
    final_reduction = _segment_metric_from_result(
        result,
        "online_residual_koopman_mpc",
        "final_evasive_maneuver",
        "residual_error_reduction_mean",
    )
    low_positive_ratio = _segment_metric_from_result(
        result,
        "online_residual_koopman_mpc",
        "low_friction_patch",
        "residual_error_reduction_positive_ratio",
    )
    final_positive_ratio = _segment_metric_from_result(
        result,
        "online_residual_koopman_mpc",
        "final_evasive_maneuver",
        "residual_error_reduction_positive_ratio",
    )
    late_residual_visible = bool(
        (low_reduction is not None and low_reduction > 0.0)
        or (final_reduction is not None and final_reduction > 0.0)
    )
    late_target_met = bool(
        row.get("target_met") is True
        and low_online_best
        and final_online_best
        and late_improvement is not None
        and late_improvement >= float(target_late_improvement)
        and late_residual_visible
    )
    score = float(row.get("score", 0.0))
    score += 3.0 if late_target_met else 0.0
    score += 1.0 if low_online_best else 0.0
    score += 1.0 if final_online_best else 0.0
    score += 2.0 * float(late_improvement or -1.0)
    score += 0.2 * float((low_positive_ratio or 0.0) + (final_positive_ratio or 0.0))
    row.update({
        "target_met": late_target_met,
        "late_target_met": late_target_met,
        "score": score,
        "late_best_non_online_baseline": late_best_name,
        "online_late_segments_rmse": online_late,
        "online_late_segments_rmse_mm": None if online_late is None else 1000.0 * float(online_late),
        "fixed_late_segments_rmse": fixed_late,
        "fixed_late_segments_rmse_mm": None if fixed_late is None else 1000.0 * float(fixed_late),
        "linear_late_segments_rmse": linear_late,
        "linear_late_segments_rmse_mm": None if linear_late is None else 1000.0 * float(linear_late),
        "online_late_best_baseline_improvement_ratio": late_improvement,
        "online_late_best_baseline_improvement_percent": (
            None if late_improvement is None else 100.0 * float(late_improvement)
        ),
        "low_friction_patch_online_best": low_online_best,
        "final_evasive_maneuver_online_best": final_online_best,
        "low_friction_patch_online_rmse": low_online,
        "low_friction_patch_online_rmse_mm": None if low_online is None else 1000.0 * float(low_online),
        "low_friction_patch_fixed_rmse_mm": None if low_fixed is None else 1000.0 * float(low_fixed),
        "low_friction_patch_linear_rmse_mm": None if low_linear is None else 1000.0 * float(low_linear),
        "final_evasive_maneuver_online_rmse": final_online,
        "final_evasive_maneuver_online_rmse_mm": None if final_online is None else 1000.0 * float(final_online),
        "final_evasive_maneuver_fixed_rmse_mm": None if final_fixed is None else 1000.0 * float(final_fixed),
        "final_evasive_maneuver_linear_rmse_mm": None if final_linear is None else 1000.0 * float(final_linear),
        "low_friction_patch_residual_error_reduction_mean": low_reduction,
        "final_evasive_maneuver_residual_error_reduction_mean": final_reduction,
        "low_friction_patch_residual_error_reduction_positive_ratio": low_positive_ratio,
        "final_evasive_maneuver_residual_error_reduction_positive_ratio": final_positive_ratio,
        "late_residual_reduction_visible": late_residual_visible,
    })
    if late_target_met and float(row.get("online_best_baseline_improvement_percent") or 0.0) >= 20.0:
        row["result_tier"] = "poster_highlight_late_gap"
    elif late_target_met:
        row["result_tier"] = "strong_late_gap"
    return row


def _copy_award_best_candidate(out_root: Path, row: Mapping) -> None:
    scenario_dir = Path(str(row["scenario_dir"]))
    best_dir = out_root / "best_candidate"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(scenario_dir, best_dir, dirs_exist_ok=True)
    write_json(best_dir / "best_candidate_metrics.json", row)
    write_json(out_root / "best_candidate_summary.json", row)


def run_award_ready_search(args, out_root: Path) -> List[Dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "search_progress.csv"
    failed_path = out_root / "failed_candidates.csv"
    progress_rows = _load_search_rows(progress_path) if args.search_resume else []
    completed_hashes = {str(row.get("candidate_hash", "")) for row in progress_rows}
    best_row = None
    for row in progress_rows:
        try:
            if best_row is None or float(row.get("score", -1.0e9)) > float(best_row.get("score", -1.0e9)):
                best_row = row
        except ValueError:
            continue
    started_count = len(completed_hashes)
    evaluated = 0
    for params in _award_candidate_sequence(args):
        if evaluated >= int(args.search_max_candidates):
            break
        candidate_hash = _candidate_hash(params)
        if candidate_hash in completed_hashes:
            continue
        idx = started_count + evaluated
        case_args = copy.deepcopy(args)
        _apply_award_candidate(case_args, params)
        case_dir = out_root / "candidate_runs" / f"case_{idx:03d}_{candidate_hash}"
        config_path = build_tire_config_override(case_args, case_dir)
        scenarios = build_scenarios(case_args)
        spec = scenarios["nonstationary_adaptive_technical_course"]
        print(
            f"Award case {idx} [{candidate_hash}] "
            f"v={params['initial_speed_kmh']:.0f}->{params['high_speed_kmh']:.0f} km/h, "
            f"mu={params['mu_initial']:.2f}->{params['mu_mid']:.2f}->{params['mu_low']:.2f}, "
            f"train={params['train_duration']:.1f}s, lane={params['lane_shift']:.1f}"
        )
        result = run_scenario(
            args=case_args,
            spec=spec,
            all_training_paths=[spec.path],
            out_root=case_dir,
            config_path=config_path,
        )
        rows = flatten_benchmark({spec.name: result})
        write_run_csv(result["scenario_dir"] / "benchmark_summary.csv", rows)
        write_json(result["scenario_dir"] / "benchmark_summary.json", {
            "plant_description": PLANT_DESCRIPTION,
            "scenario": spec.name,
            "key_segments": result["summary"].get("key_segments", []),
            "table": rows,
        })
        row = _award_result_row(
            idx=idx,
            candidate_hash=candidate_hash,
            params=params,
            result=result,
            rows=rows,
            case_dir=case_dir,
            target_improvement=float(args.target_online_best_baseline_improvement),
        )
        progress_rows.append(row)
        completed_hashes.add(candidate_hash)
        _write_search_rows(progress_path, progress_rows)
        _write_search_rows(failed_path, [
            r for r in progress_rows
            if str(r.get("target_met", "")).lower() != "true"
        ])
        write_json(out_root / "search_progress.json", {"cases": progress_rows})
        if best_row is None or float(row["score"]) > float(best_row.get("score", -1.0e9)):
            best_row = row
            _copy_award_best_candidate(out_root, row)
        if row["target_met"]:
            _copy_award_best_candidate(out_root, row)
            print(
                "Award-ready target candidate found: "
                f"online/best-baseline={100.0 * float(row['online_best_baseline_improvement_ratio']):.1f}%, "
                f"tier={row['result_tier']}"
            )
            if not args.search_continue_after_success:
                break
        evaluated += 1
    if best_row is not None:
        _copy_award_best_candidate(out_root, best_row)
    write_json(out_root / "search_summary.json", {
        "gap_boost": bool(getattr(args, "gap_boost", False)),
        "target_online_best_baseline_improvement": float(args.target_online_best_baseline_improvement),
        "searched_cases_total": len(progress_rows),
        "new_cases_this_run": evaluated,
        "best_candidate": best_row,
        "success_cases": [
            row for row in progress_rows
            if str(row.get("target_met", "")).lower() == "true" or row.get("target_met") is True
        ],
    })
    return progress_rows


def run_late_gap_boost_search(args, out_root: Path) -> List[Dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "search_progress.csv"
    failed_path = out_root / "failed_candidates.csv"
    progress_rows = _load_search_rows(progress_path) if args.search_resume else []
    completed_hashes = {str(row.get("candidate_hash", "")) for row in progress_rows}
    best_row = None
    for row in progress_rows:
        try:
            if best_row is None or float(row.get("score", -1.0e9)) > float(best_row.get("score", -1.0e9)):
                best_row = row
        except ValueError:
            continue
    started_count = len(completed_hashes)
    evaluated = 0
    for params in _late_gap_boost_candidate_sequence(args):
        if evaluated >= int(args.search_max_candidates):
            break
        candidate_hash = _candidate_hash(params)
        if candidate_hash in completed_hashes:
            continue
        idx = started_count + evaluated
        case_args = copy.deepcopy(args)
        _apply_late_gap_candidate(case_args, params)
        case_dir = out_root / "candidate_runs" / f"case_{idx:03d}_{candidate_hash}"
        config_path = build_tire_config_override(case_args, case_dir)
        scenarios = build_scenarios(case_args)
        spec = scenarios["late_adaptation_gap_boost_course"]
        print(
            f"Late-gap case {idx} [{candidate_hash}] "
            f"v={params['initial_speed_kmh']:.0f}->{params['high_speed_kmh']:.0f} km/h, "
            f"mu={params['mu_initial']:.2f}->{params['mu_mid']:.2f}->{params['mu_low']:.2f}, "
            f"final={params['final_friction_mode']}, train={params['train_duration']:.1f}s"
        )
        result = run_scenario(
            args=case_args,
            spec=spec,
            all_training_paths=[spec.path],
            out_root=case_dir,
            config_path=config_path,
        )
        rows = flatten_benchmark({spec.name: result})
        write_run_csv(result["scenario_dir"] / "benchmark_summary.csv", rows)
        write_json(result["scenario_dir"] / "benchmark_summary.json", {
            "plant_description": PLANT_DESCRIPTION,
            "scenario": spec.name,
            "key_segments": result["summary"].get("key_segments", []),
            "table": rows,
        })
        row = _late_gap_result_row(
            idx=idx,
            candidate_hash=candidate_hash,
            params=params,
            result=result,
            rows=rows,
            case_dir=case_dir,
            target_improvement=float(args.target_online_best_baseline_improvement),
            target_late_improvement=float(args.target_late_segment_improvement),
        )
        progress_rows.append(row)
        completed_hashes.add(candidate_hash)
        _write_search_rows(progress_path, progress_rows)
        _write_search_rows(failed_path, [
            r for r in progress_rows
            if str(r.get("target_met", "")).lower() != "true"
        ])
        write_json(out_root / "search_progress.json", {"cases": progress_rows})
        if best_row is None or float(row["score"]) > float(best_row.get("score", -1.0e9)):
            best_row = row
            _copy_award_best_candidate(out_root, row)
        if row["target_met"]:
            _copy_award_best_candidate(out_root, row)
            print(
                "Late-gap target candidate found: "
                f"key online/best={float(row['online_best_baseline_improvement_percent']):.1f}%, "
                f"late online/best={float(row['online_late_best_baseline_improvement_percent']):.1f}%, "
                f"tier={row['result_tier']}"
            )
            if not args.search_continue_after_success:
                break
        evaluated += 1
    if best_row is not None:
        _copy_award_best_candidate(out_root, best_row)
    write_json(out_root / "search_summary.json", {
        "late_gap_boost": True,
        "target_online_best_baseline_improvement": float(args.target_online_best_baseline_improvement),
        "target_late_segment_improvement": float(args.target_late_segment_improvement),
        "searched_cases_total": len(progress_rows),
        "new_cases_this_run": evaluated,
        "best_candidate": best_row,
        "success_cases": [
            row for row in progress_rows
            if str(row.get("target_met", "")).lower() == "true" or row.get("target_met") is True
        ],
    })
    return progress_rows


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.search_late_gap_boost:
        apply_late_gap_boost_defaults(args)
        out_root = Path(args.out_dir)
        rows = run_late_gap_boost_search(args, out_root)
        print("Late-gap boost search complete.")
        print(f"Results: {out_root.resolve()}")
        if rows:
            best = max(rows, key=lambda row: float(row.get("score", -1.0e9)))
            print(
                "Best late-gap candidate: "
                f"online={float(best['online_key_segments_rmse']):.4f} m, "
                f"fixed={float(best['fixed_key_segments_rmse']):.4f} m, "
                f"linear={float(best['linear_key_segments_rmse']):.4f} m, "
                f"late_online={float(best.get('online_late_segments_rmse') or 0.0):.4f} m, "
                f"tier={best['result_tier']}, target_met={best['target_met']}"
            )
        return
    if args.search_award_ready:
        apply_award_ready_defaults(args)
        out_root = Path(args.out_dir)
        rows = run_award_ready_search(args, out_root)
        print("Award-ready search complete.")
        print(f"Results: {out_root.resolve()}")
        if rows:
            best = max(rows, key=lambda row: float(row.get("score", -1.0e9)))
            print(
                "Best award candidate: "
                f"online={float(best['online_key_segments_rmse']):.4f} m, "
                f"fixed={float(best['fixed_key_segments_rmse']):.4f} m, "
                f"linear={float(best['linear_key_segments_rmse']):.4f} m, "
                f"tier={best['result_tier']}, target_met={best['target_met']}"
            )
        return
    if args.search_composite_course:
        apply_composite_defaults(args)
        out_root = Path(args.out_dir)
        rows = run_composite_course_search(args, out_root)
        print("Composite course search complete.")
        print(f"Results: {out_root.resolve()}")
        if rows:
            best = max(rows, key=lambda row: float(row.get("score", -1.0e9)))
            print(
                "Best composite candidate: "
                f"online={float(best['online_key_segments_rmse']):.4f} m, "
                f"fixed={float(best['fixed_key_segments_rmse']):.4f} m, "
                f"linear={float(best['linear_key_segments_rmse']):.4f} m, "
                f"target_met={best['target_met']}"
            )
        return
    if args.search_online_advantage:
        apply_online_advantage_defaults(args)
        out_root = Path(args.out_dir)
        rows = run_online_advantage_search(args, out_root)
        print("Online advantage search complete.")
        print(f"Results: {out_root.resolve()}")
        if rows:
            best = max(rows, key=lambda row: float(row.get("score", -1.0e9)))
            print(
                "Best candidate: "
                f"online={float(best['online_maneuver_rmse']):.4f} m, "
                f"fixed={float(best['fixed_maneuver_rmse']):.4f} m, "
                f"linear={float(best['linear_maneuver_rmse']):.4f} m, "
                f"target_met={best['target_met']}"
            )
        return
    if args.nonlinear_plant_benchmark:
        args.tire_model = "fiala"
        args.use_tire_augmented_koopman = True
        args.koopman_mode = "residual_output"
        if args.training_policy == "excitation":
            args.training_policy = "small_sine_plus_linear_mpc"
        if args.scenario == "mild_sine":
            args.scenario = "low_mu_fiala_adaptive_lane_change"
        if args.friction_mu is None:
            args.friction_mu = 0.70
    if args.scenario == "low_mu_fiala_adaptive_lane_change":
        args.tire_model = "fiala"
        args.use_tire_augmented_koopman = True
        args.koopman_mode = "residual_output"
        if args.training_policy == "excitation":
            args.training_policy = "small_sine_plus_linear_mpc"
        if args.friction_mu is None:
            args.friction_mu = 0.70
    if args.scenario == "friction_shift_adaptive_lane_change":
        args.tire_model = "fiala"
        args.use_tire_augmented_koopman = True
        args.koopman_mode = "residual_output"
        if args.training_policy == "excitation":
            args.training_policy = "small_sine_plus_linear_mpc"
        if args.friction_profile == "constant":
            args.friction_profile = "ramp"
        if args.friction_mu_initial is None:
            args.friction_mu_initial = 0.85
        if args.friction_mu_final is None:
            args.friction_mu_final = 0.70
        if args.friction_mu is None:
            args.friction_mu = args.friction_mu_initial
    if args.scenario == "online_advantage_friction_shift_lane_change":
        apply_online_advantage_defaults(args)
    if args.scenario == "composite_friction_adaptation_course":
        apply_composite_defaults(args)
    if args.scenario == "nonstationary_adaptive_technical_course":
        apply_award_ready_defaults(args)
    if args.scenario == "late_adaptation_gap_boost_course":
        apply_late_gap_boost_defaults(args)
    out_root = Path(args.out_dir)
    config_path = build_tire_config_override(args, out_root)
    scenarios = build_scenarios(args)
    specs = selected_scenarios(args, scenarios)

    scenario_results = {}
    reference_rows = []
    for spec in specs:
        training_paths = make_training_paths(args) if args.training_scope == "benchmark" else [spec.path]
        result = run_scenario(
            args=args,
            spec=spec,
            all_training_paths=training_paths,
            out_root=out_root,
            config_path=config_path,
        )
        scenario_results[spec.name] = result
        eval_duration = result["summary"]["config"]["eval_duration"]
        x_max = max(
            80.0,
            float(spec.path.x_at_time(eval_duration)) + 5.0
            if hasattr(spec.path, "x_at_time") else args.target_speed * eval_duration + 5.0,
        )
        reference_rows.extend(sample_reference_path(spec.path, np.linspace(0.0, x_max, 501)))

    benchmark_rows = flatten_benchmark(scenario_results)
    write_run_csv(out_root / "benchmark_summary.csv", benchmark_rows)
    write_run_csv(out_root / "reference_paths.csv", reference_rows)
    figures_generated = generate_figures(out_root, scenario_results, benchmark_rows)

    benchmark_summary = {
        "plant_description": PLANT_DESCRIPTION,
        "scenario_order": list(scenario_results.keys()),
        "controller_order": [
            "linear_bicycle_mpc",
            "fixed_residual_koopman_mpc" if args.koopman_mode == "residual_output" else "fixed_koopman_mpc",
            "online_residual_koopman_mpc" if args.koopman_mode == "residual_output" else "online_koopman_mpc",
        ],
        "metric_units": {
            "lateral_error_rmse": "m",
            "heading_error_rmse": "rad",
            "max_abs_lateral_error": "m",
            "steering_effort_rms": "rad",
            "steering_smoothness_rms": "rad/step",
            "mean_solve_time_s": "s",
            "max_solve_time_s": "s",
            "post_adaptation_lateral_error_rmse": "m",
            "post_adaptation_heading_error_rmse": "rad",
            "post_adaptation_max_abs_lateral_error": "m",
            "post_friction_change_lateral_error_rmse": "m",
            "post_friction_change_heading_error_rmse": "rad",
            "post_friction_change_max_abs_lateral_error": "m",
            "maneuver_segment_lateral_error_rmse": "m",
            "maneuver_segment_heading_error_rmse": "rad",
            "maneuver_segment_max_abs_lateral_error": "m",
            "key_segments_lateral_error_rmse": "m",
            "key_segments_heading_error_rmse": "rad",
            "key_segments_max_abs_lateral_error": "m",
            "key_segments_steering_effort_rms": "rad",
            "key_segments_steering_smoothness_rms": "rad/step",
            "key_segments_residual_prediction_error_improvement_percent": "%",
            "prediction_error_before_mean": "-",
            "prediction_error_after_mean": "-",
            "prediction_error_improvement_percent": "%",
            "residual_prediction_error_before_mean": "-",
            "residual_prediction_error_after_mean": "-",
            "residual_prediction_error_improvement_percent": "%",
            "rls_acceptance_rate": "ratio",
            "rls_rejection_rate": "ratio",
            "rls_relative_theta_update_mean": "ratio",
        },
        "matplotlib_figures_generated": figures_generated,
        "tire_model": str(args.tire_model),
        "friction_mu": args.friction_mu,
        "vehicle_config_path": config_path,
        "table": benchmark_rows,
        "scenarios": {
            name: {
                "path_parameters": result["summary"]["path_parameters"],
                "config": result["summary"]["config"],
                "training_prediction_error": result["summary"]["training_prediction_error"],
                "segment_metadata": result["summary"].get("segment_metadata", []),
                "key_segments": result["summary"].get("key_segments", []),
                "segment_metrics": result["summary"].get("segment_metrics", {}),
            }
            for name, result in scenario_results.items()
        },
    }
    write_json(out_root / "benchmark_summary.json", benchmark_summary)
    sweep_rows = run_small_sweep(args, out_root) if args.run_small_sweep else []
    if sweep_rows:
        benchmark_summary["sweep_cases"] = sweep_rows
        write_json(out_root / "benchmark_summary.json", benchmark_summary)

    print("EDMD-Koopman MPC benchmark complete.")
    print(f"Results: {out_root.resolve()}")
    print(f"Figures generated: {figures_generated}")
    if sweep_rows:
        print(f"Sweep cases: {len(sweep_rows)}")
    for row in benchmark_rows:
        print(
            f"{row['scenario']} | {row['controller']}: "
            f"ey_rmse={row['lateral_error_rmse']:.4f} m, "
            f"max_ey={row['max_abs_lateral_error']:.4f} m, "
            f"mean_solve={row['mean_solve_time_s']:.4f} s"
        )


if __name__ == "__main__":
    main()
