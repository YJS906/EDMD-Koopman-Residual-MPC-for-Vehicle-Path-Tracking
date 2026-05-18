"""
E-Corner 모델 - 통합 코너 모듈
4개 액추에이터 입력을 받아 타이어 힘 출력
입력: T_steer, T_brk, T_Drv, T_susp
출력: F_s, F_x_tire, F_y_tire (Full Car Dynamics로 전달)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param

from .tire.longitudinal.longitudinal_tire import LongitudinalTireModel
from .tire.lateral.lateral_tire import LateralTireModel
from .suspension.suspension_model import SuspensionModel
from .drive.drive_model import DriveModel
from .drive.brake_model import BrakeModel
from .steering.steering_model import SteeringModel


@dataclass
class ECornerState:
    """E-Corner 상태 변수 (한 바퀴)"""
    # 출력
    F_s: float = 0.0           # 서스펜션 힘 [N]

    # 중간 상태
    F_x_tire: float = 0.0      # 종방향 타이어 힘 [N]
    F_y_tire: float = 0.0      # 횡방향 타이어 힘 [N]
    F_z: float = 0.0           # 타이어 수직력 [N]
    steering_angle: float = 0.0  # 조향 각도 [rad]
    omega_wheel: float = 0.0   # 휠 각속도 [rad/s]

    slip_angle: float = 0.0    # Tire slip angle [rad]
    slip_ratio: float = 0.0    # Tire longitudinal slip ratio [-]


@dataclass
class ECornerParameters:
    """E-Corner 파라미터"""
    corner_id: Optional[str] = None       # 바퀴 위치 ('FL', 'FR', 'RL', 'RR')
    corner_name: Optional[str] = None     # corner_id와 동일 의미 (호환용)
    config: Optional[Dict] = None         # 하위 모듈에 전달할 설정 딕셔너리
    config_path: Optional[str] = None     # YAML 설정 파일 경로

    def __post_init__(self) -> None:
        # corner_name이 전달된 경우 corner_id로 매핑
        if self.corner_id is None:
            self.corner_id = self.corner_name
        if self.corner_id is None:
            raise ValueError("ECornerParameters requires corner_id (one of 'FL', 'FR', 'RL', 'RR')")
        if self.corner_id not in ['FL', 'FR', 'RL', 'RR']:
            raise ValueError(f"Invalid corner_id: {self.corner_id}. Must be one of ['FL', 'FR', 'RL', 'RR']")


class ECorner:
    """
    E-Corner 통합 모듈 (한 바퀴)
    입력: T_steer, T_brk, T_Drv, T_susp, V_wheel_x, V_wheel_y, roll, pitch, heave, roll_dot, pitch_dot, heave_dot
    출력: F_s, F_x_tire, F_y_tire
    """

    def __init__(self, corner_id: Optional[str] = None,
                 params: Optional[ECornerParameters] = None,
                 config: Optional[Dict] = None,
                 config_path: Optional[str] = None):
        """
        E-Corner 모델 초기화

        Args:
            corner_id: 바퀴 위치 ('FL', 'FR', 'RL', 'RR')
            params: ECornerParameters 객체 (corner_id/config/config_path 포함)
            config: 설정 딕셔너리 (선택, 없으면 YAML 로드)
            config_path: YAML 설정 파일 경로. None이면 기본 vehicle_standard.yaml 사용
        """
        # params가 제공되면 corner_id/config/config_path를 우선 채운다
        if params is not None:
            if not isinstance(params, ECornerParameters):
                raise TypeError("params must be an ECornerParameters instance")
            corner_id = corner_id or params.corner_id
            if config is None:
                config = params.config
            if config_path is None:
                config_path = params.config_path

        if corner_id not in ['FL', 'FR', 'RL', 'RR']:
            raise ValueError(f"Invalid corner_id: {corner_id}. Must be one of ['FL', 'FR', 'RL', 'RR']")

        self.corner_id = corner_id
        self.config = config or {}
        self.config_path = config_path

        self.state = ECornerState()

        # 하위 컴포넌트 초기화 (steering은 좌/우에 따라 config 전달)
        steering_cfg = self._build_steering_config(corner_id, self.config, config_path)
        self.steering = SteeringModel(config=steering_cfg, config_path=config_path, corner_id=corner_id)

        self.brake = BrakeModel(config_path=config_path)
        self.drive = DriveModel(config_path=config_path, corner_id=corner_id)
        self.suspension = SuspensionModel(corner_id=corner_id, config_path=config_path)
        self.longitudinal_tire = LongitudinalTireModel(config_path=config_path)
        self.lateral_tire = LateralTireModel(config_path=config_path)

    def update(self, dt: float,
               T_steer: float, T_brk: float, T_Drv: float, T_susp: float,
               V_wheel_x: float, V_wheel_y: float,
               X_body: np.ndarray,
               z_road: float = 0.0,
               z_road_dot: float = 0.0,
               direction: int = 1) -> Tuple[float, float, float]:
        """
        E-Corner 업데이트 (한 바퀴)

        입력:
            - T_steer, T_brk, T_Drv, T_susp: 액추에이터 토크 입력
            - V_wheel_x, V_wheel_y: 휠 중심 속도
            - X_body: 차체 상태 벡터 [heave, roll, pitch, heave_dot, roll_rate, pitch_rate]
            - z_road: 노면 높이 [m]
            - z_road_dot: 노면 속도 [m/s]
            - direction: 전진/후진 (1: 전진, -1: 후진)

        출력: F_s, F_x_tire, F_y_tire
        """
        # 1. Steering Actuator: T_steer → steering_angle
        steering_angle = self.steering.update(dt, T_steer, self.steering.state.self_aligning_torque)

        # 2. Brake Actuator: T_brk → F_clamp
        F_clamp = self.brake.update(dt, T_brk)

        # 3. Wheel Dynamics: T_Drv, F_clamp, F_x_tire(이전 스텝), direction → ω_wheel
        omega_wheel = self.drive.update(
            dt,
            T_Drv,
            self.state.F_x_tire,
            F_clamp=F_clamp,
            direction=direction,
        )

        # 4. Suspension Dynamics: T_susp, X_body, z_road, z_road_dot → F_s, F_z
        F_s, F_z = self.suspension.update(dt, T_susp, X_body, z_road, z_road_dot)

        # 5. Tire Dynamics: steering_angle, ω_wheel, V_wheel_x, V_wheel_y, F_z → F_x_tire, F_y_tire
        #    입력 속도는 바디 프레임이므로 최신 steering_angle로 휠 프레임으로 회전
        c, s = np.cos(steering_angle), np.sin(steering_angle)
        V_wx_local = c * V_wheel_x + s * V_wheel_y
        V_wy_local = -s * V_wheel_x + c * V_wheel_y

        # 5-1. Slip angle 계산 (휠 프레임)
        alpha = self.lateral_tire.calculate_slip_angle(V_wx_local, V_wy_local)

        # 5-2. Slip ratio 계산 (R_eff는 longitudinal_tire 내부에서 YAML 로드)
        kappa = self.longitudinal_tire.calculate_slip_ratio(omega_wheel, V_wx_local)

        # 5-3. 타이어 힘 계산
        F_x_tire = self.longitudinal_tire.calculate_force(kappa, F_z)
        F_y_tire = self.lateral_tire.calculate_force(alpha, F_z)

        # 6. Self-aligning torque 계산 및 조향 상태 업데이트
        M_align = self.lateral_tire.calculate_aligning_torque(alpha, F_z)
        self.steering.state.self_aligning_torque = M_align
        self.lateral_tire.state.slip_angle = alpha
        self.lateral_tire.state.lateral_force = F_y_tire
        self.lateral_tire.state.aligning_torque = M_align

        # 상태 업데이트
        self.state.F_s = F_s
        self.state.F_x_tire = F_x_tire
        self.state.F_y_tire = F_y_tire
        self.state.F_z = F_z
        self.state.steering_angle = steering_angle
        self.state.omega_wheel = omega_wheel
        self.state.slip_angle = alpha
        self.state.slip_ratio = kappa

        return F_s, F_x_tire, F_y_tire

    def get_state(self) -> Dict:
        """현재 E-Corner 상태 조회"""
        return {
            "F_s": self.state.F_s,
            "F_x_tire": self.state.F_x_tire,
            "F_y_tire": self.state.F_y_tire,
            "F_z": self.state.F_z,
            "steering_angle": self.state.steering_angle,  # 휠 조향각
            "omega_wheel": self.state.omega_wheel,        # 휠 각속도
            "slip_angle": self.state.slip_angle,
            "slip_ratio": self.state.slip_ratio,
        }

    def reset(self) -> None:
        """E-Corner 상태 리셋"""
        self.state = ECornerState()
        self.steering.reset()
        self.brake.reset()
        self.drive.reset()
        self.suspension.reset()
        self.longitudinal_tire.reset()
        self.lateral_tire.reset()

    @staticmethod
    def _build_steering_config(corner_id: str,
                               user_config: Optional[Dict],
                               config_path: Optional[str]) -> Dict:
        """
        좌/우 바퀴에 맞춰 스티어링 파라미터를 완성해 SteeringModel로 전달한다.
        YAML 기본값 + 사용자 설정을 병합하고, side별 최대 각도를 꺼내어
        max_angle_pos/max_angle_neg 필드를 채운다.
        """
        # 1) 사용자 설정 우선, 없으면 YAML 로드
        steering_param = {}
        if user_config and 'steering' in user_config:
            steering_param = user_config['steering'] or {}
        else:
            steering_param = load_param('steering', config_path)

        # 2) 좌/우 선택
        side_key = 'left' if corner_id in ['FL', 'RL'] else 'right'
        side_cfg = steering_param.get(side_key, {}) if isinstance(steering_param, dict) else {}

        # 3) 최종 설정 구성 (필수 각도 필드 채움)
        steering_cfg = dict(steering_param) if isinstance(steering_param, dict) else {}
        steering_cfg['max_angle_pos'] = side_cfg.get(
            'max_angle_pos',
            steering_cfg.get('max_angle_pos', 0.0)
        )
        steering_cfg['max_angle_neg'] = side_cfg.get(
            'max_angle_neg',
            steering_cfg.get('max_angle_neg', 0.0)
        )

        return steering_cfg
