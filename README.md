<div align="center">
  <a href="assets/자동차 공학회_여진승_포스터_최종.pdf">
    <img src="assets/자동차 공학회_여진승_포스터_최종.png" alt="Final Poster PDF Preview" width="900">
  </a>
  <br>
  <sub>Click the poster image to open the PDF version.</sub>
</div>

<br>

# EDMD-Koopman Residual MPC for Vehicle Path Tracking

차량 횡방향 경로 추종 성능 향상을 위한
**EDMD-Koopman residual correction 기반 MPC 연구 프로토타입**입니다.

본 연구는 nonlinear vehicle plant와 nominal linear bicycle model 사이의 예측 오차를 residual로 정의하고, 이를 EDMD-Koopman predictor로 보정하는 MPC 구조를 구현하고 비교한 프로젝트입니다.

사용한 차량 plant는 연구실에서 제공받은 **E-Corner 기반 nonlinear vehicle simulation model**입니다.
따라서 본 저장소의 목적은 차량 plant 자체를 새로 개발하는 것이 아니라, 주어진 nonlinear plant를 기준으로 **linear MPC의 모델 불일치 문제를 어떻게 보정할 수 있는지**를 검증하는 데 있습니다.

또한 E-Corner 차량은 4륜 조향 구조를 가질 수 있지만, 본 연구에서는 일반적인 전륜 조향 차량 조건에 가깝게 실험하기 위해 rear steering은 고정하였습니다. 하나의 bicycle steering command `delta_cmd`를 전륜 FL/FR Ackermann steering angle로 변환하여 사용하였으며, 후륜 조향은 사용하지 않았습니다.

---

## Research Motivation

Linear bicycle model 기반 MPC는 계산이 빠르고 구조가 단순하다는 장점이 있습니다.
그러나 실제 차량 거동은 다음과 같은 요인으로 인해 선형 모델과 차이가 발생합니다.

* 타이어 비선형성
* 큰 조향 입력
* 고속 주행
* 횡가속도 증가
* 노면 마찰계수 변화
* Fiala tire saturation
* steering history와 yaw-rate history의 영향

이러한 조건에서는 nominal linear bicycle model만으로는 실제 nonlinear plant의 다음 상태를 충분히 정확하게 예측하기 어렵습니다.

본 연구에서는 이 문제를 다음과 같이 접근했습니다.

```text id="my7mnp"
linear bicycle model이 빠르게 기본 예측을 수행하고,
EDMD-Koopman residual predictor가 선형 모델이 틀릴 부분을 보정한다.
```

즉, MPC 내부 prediction model을 완전히 복잡한 nonlinear model로 바꾸는 대신,
기존 linear MPC 구조를 유지하면서 plant-model mismatch를 residual correction 방식으로 보완하는 것이 핵심입니다.

---

## Core Idea

본 연구의 residual은 다음과 같이 정의합니다.

```text id="23jl5p"
residual = 실제 nonlinear plant의 다음 상태 - linear bicycle model의 다음 상태 예측
```

상태 변수는 차량 횡방향 경로 추종에 필요한 4개 값으로 구성됩니다.

```text id="eu7ufo"
x = [e_y, e_psi, v_y, yaw_rate]^T
```

각 변수의 의미는 다음과 같습니다.

| State      | Description                        |
| ---------- | ---------------------------------- |
| `e_y`      | reference path 기준 lateral error    |
| `e_psi`    | reference heading 기준 heading error |
| `v_y`      | lateral velocity                   |
| `yaw_rate` | yaw rate                           |

Linear bicycle model은 다음 상태를 예측합니다.

```text id="lxe0bd"
x_nom_next = linear_bicycle_predictor(x_k, delta_k)
```

하지만 실제 nonlinear plant는 타이어 비선형성, 마찰 변화, 큰 조향 입력 등의 영향으로 선형 모델과 다른 다음 상태를 보입니다.

```text id="gecddd"
residual = VehicleBody_true_next - x_nom_next
```

EDMD-Koopman residual predictor는 이 residual을 예측합니다.

```text id="urass1"
r_koopman = KoopmanResidualPredictor(x_k, context_k, delta_k)
```

최종적으로 MPC는 다음과 같이 보정된 상태 예측을 사용합니다.

```text id="t4lopu"
x_pred = x_nom_next + r_koopman
```

---

## Controller Comparison

본 연구에서는 세 가지 제어기를 동일한 nonlinear plant에서 비교했습니다.

| Controller                       | Prediction Model                               | Online Update | Purpose                           |
| -------------------------------- | ---------------------------------------------- | ------------- | --------------------------------- |
| Linear Bicycle MPC               | Linear bicycle model                           | No            | 기본 baseline                       |
| Fixed Residual EDMD-Koopman MPC  | Linear model + offline Koopman residual        | No            | offline residual correction 효과 확인 |
| Online Residual EDMD-Koopman MPC | Linear model + Koopman residual updated by RLS | Yes           | 주행 조건 변화에 대한 online adaptation 확인 |

---

## 1. Linear Bicycle Model MPC

Linear MPC는 nominal linear bicycle model만을 prediction model로 사용합니다.

```text id="9e9h97"
x_pred = x_nom_next
```

장점은 구조가 단순하고 계산이 빠르다는 점입니다.
그러나 nonlinear tire effect나 friction shift가 포함된 조건에서는 plant와 prediction model 사이의 mismatch가 커질 수 있습니다.

본 연구에서는 Linear MPC를 baseline으로 사용했습니다.

---

## 2. Fixed Residual EDMD-Koopman MPC

Fixed Koopman MPC는 초기 학습 데이터로 EDMD-Koopman residual predictor를 학습한 뒤, 주행 중에는 모델을 업데이트하지 않습니다.

```text id="mffevz"
x_pred = x_nom_next + r_koopman
```

이 방식은 linear model이 설명하지 못하는 nonlinear residual을 보정할 수 있습니다.
다만 학습 데이터와 다른 주행 조건이 나타날 경우, residual prediction 성능이 제한될 수 있습니다.

예를 들어 다음과 같은 조건에서는 fixed model의 한계가 나타날 수 있습니다.

* 학습 데이터에 없던 마찰계수 변화
* 더 큰 조향 입력
* 더 높은 속도
* 다른 path curvature pattern
* nonlinear tire saturation 구간

---

## 3. Online Residual EDMD-Koopman MPC

Online Koopman MPC는 Fixed Koopman MPC와 같은 초기 EDMD 모델에서 시작합니다.
이후 주행 중 관측되는 plant transition data를 이용하여 residual predictor를 matrix RLS 방식으로 갱신합니다.

```text id="gj74l6"
W_0 : residual matrix initialized by offline EDMD
W_k : residual matrix updated by online RLS
```

이 구조의 목적은 제한된 초기 데이터만 사용하는 fixed model보다, 변화하는 주행 조건에 더 잘 적응하는 것입니다.

Online update는 다음 상황에서 의미가 있습니다.

* 노면 마찰계수가 변하는 경우
* 타이어 비선형성이 강하게 나타나는 경우
* plant-model mismatch가 시간에 따라 달라지는 경우
* 초기 학습 데이터가 모든 주행 조건을 충분히 포함하지 못하는 경우

단, online update가 항상 성능을 향상시키는 것은 아니므로, RLS update에는 guard를 적용했습니다.

```text id="shttfg"
- update가 residual error를 악화시키면 reject
- update 크기가 너무 크면 reject
- update 이후 제어 입력 변화가 과도하면 reject
- spectral radius guard 적용
```

---

## Feature and Observable Design

Koopman residual predictor는 단순히 4개 error state만 사용하지 않습니다.
Residual이 발생하는 원인을 더 잘 설명하기 위해 차량 상태, 경로 정보, 조향 이력, 타이어 관련 정보를 feature로 사용합니다.

본 연구에서 사용한 tire-augmented feature는 다음과 같습니다.

```text id="fxkm2v"
e_y
e_psi
v_y
yaw_rate
vx
delta_prev
steering_rate
curvature
curvature_rate
curvature_preview_1
curvature_preview_2
curvature_preview_3
yaw_rate_prev_1
yaw_rate_prev_2
alpha_front_mean
alpha_rear_mean
alpha_front_diff
Fy_front_sum
Fy_rear_sum
Fz_front_sum
Fz_rear_sum
steering_angle_front_mean
```

이 feature들은 다음과 같은 정보를 포함합니다.

| Feature Group              | Meaning                                     |
| -------------------------- | ------------------------------------------- |
| Path tracking state        | 현재 차량이 reference path에서 얼마나 벗어났는지           |
| Speed and steering history | 조향 입력과 속도 변화의 영향                            |
| Curvature preview          | 앞으로 나타날 path curvature 정보                   |
| Yaw-rate history           | 차량 회전 거동의 시간적 변화                            |
| Tire slip and force        | nonlinear tire effect를 설명하기 위한 물리 기반 정보     |
| Vertical load              | tire force 변화와 load transfer 영향을 반영하기 위한 정보 |

이 feature vector는 polynomial lifting을 통해 Koopman observable로 확장됩니다.

```text id="cm0e0w"
observable = [1, x_i, x_i^2, x_i * x_j]
```

즉, 기본 feature뿐 아니라 feature 간 상호작용도 residual prediction에 활용합니다.

---

## EDMD-Koopman Identification

EDMD는 Koopman predictor를 초기 학습하기 위해 사용됩니다.

학습 데이터는 다음 형태로 구성됩니다.

```text id="ob0yv6"
current observable: z_k
steering input:     delta_k
next target:        residual_{k+1}
```

EDMD는 다음 관계를 가장 잘 만족하는 선형 행렬을 찾습니다.

```text id="cz7foa"
z_{k+1} ≈ A z_k + B delta_k
```

그리고 output matrix `C`를 통해 lifted observable에서 residual prediction을 얻습니다.

```text id="6mp344"
r_koopman = C z_{k+1}
```

Fixed Koopman MPC는 이 EDMD 결과를 그대로 사용합니다.
Online Koopman MPC는 같은 EDMD 결과에서 시작한 뒤, 주행 중 RLS로 `A`, `B`를 갱신합니다.

---

## MPC Formulation

MPC의 최적화 문제 자체는 4차원 상태를 기준으로 구성됩니다.

```text id="h08oy4"
x = [e_y, e_psi, v_y, yaw_rate]
u = delta
```

Koopman observable은 MPC의 직접적인 optimization variable이 아닙니다.
105차원 observable은 residual prediction을 위한 내부 표현이며, 최종적으로는 보정된 4차원 next-state prediction이 MPC에 사용됩니다.

```text id="44b6fl"
x_next_pred = x_nom_next + r_koopman
```

코드에서는 이 보정 예측함수를 현재 상태 주변에서 국소 선형화하여 다음 형태로 만든 뒤 MPC QP에 사용합니다.

```text id="i8tnqs"
x_next ≈ F x + G delta + d
```

MPC는 조향각 크기와 조향각 변화율에 제한을 두고, lateral error와 heading error를 줄이는 방향으로 최적의 조향 입력을 계산합니다.

---

## Vehicle Model Assumption

본 연구에서 closed-loop plant는 연구실에서 제공받은 E-Corner 기반 VehicleBody plant입니다.

다만 본 연구의 목적은 E-Corner 차량의 독립 4륜 조향 성능 평가가 아니라, 일반적인 차량 경로 추종 문제에서 residual correction 기반 MPC의 효과를 검증하는 것입니다.

따라서 차량 입력 구조는 다음과 같이 단순화했습니다.

```text id="ndk7xv"
single bicycle steering command: delta_cmd [rad]
front steering: FL/FR Ackermann steering angle
rear steering: fixed
```

이를 위해 `DirectAckermannSteeringWrapper`를 사용했습니다.

이 wrapper는 하나의 bicycle steering command를 전륜 좌우 Ackermann 조향각으로 변환하고, 후륜 조향각은 고정합니다.
이때 차량의 drive, brake, suspension, tire, body dynamics는 기존 nonlinear plant를 그대로 사용합니다.

---

## Experiment Scenarios

실험 스크립트는 여러 path tracking scenario를 지원합니다.

대표적인 scenario는 다음과 같습니다.

| Scenario                                  | Description                                            |
| ----------------------------------------- | ------------------------------------------------------ |
| `mild_sine`                               | 낮은 곡률의 기본 sine path                                    |
| `aggressive_sine`                         | 큰 조향 입력이 필요한 aggressive sine path                      |
| `double_lane_change`                      | 차선 변경 형태의 path                                         |
| `adaptive_lane_change`                    | adaptation 성능 확인용 lane change                          |
| `friction_shift_adaptive_lane_change`     | 마찰계수 변화가 포함된 lane change                               |
| `composite_friction_adaptation_course`    | 복합 마찰 변화 및 경로 변화 시나리오                                  |
| `nonstationary_adaptive_technical_course` | 비정상 주행 조건을 포함한 technical course                        |
| `late_adaptation_gap_boost_course`        | online adaptation 차이를 확인하기 위한 late adaptation scenario |

주요 비교 지표는 다음과 같습니다.

* lateral error
* heading error
* yaw-rate behavior
* control input smoothness
* segment-level tracking metrics
* prediction error
* solver time
* RLS update acceptance
* online vs fixed performance gap

---

## Repository Structure

```text id="rm86o8"
vehicle_sim/
├── controllers/
│   └── path_tracking_mpc/
│       ├── linear_bicycle.py
│       ├── edmd.py
│       ├── features.py
│       ├── mpc.py
│       └── rls.py
│
├── experiments/
│   ├── edmd_koopman_mpc_mvp.py
│   └── results/
│
├── models/
│   └── e_corner/
│       └── tire/
│           └── lateral/
│               └── lateral_tire.py
│
└── utils/
    ├── direct_ackermann_steering.py
    └── path_tracking_sim.py

docs/
├── koopman_mpc_explanation.md
└── professor_qa.md

assets/
├── 자동차 공학회_여진승_포스터_최종.png
└── 자동차 공학회_여진승_포스터_최종.pdf
```

---

## Main Files

| File                                                          | Role                                                                   |
| ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `vehicle_sim/controllers/path_tracking_mpc/linear_bicycle.py` | nominal linear bicycle prediction model                                |
| `vehicle_sim/controllers/path_tracking_mpc/edmd.py`           | EDMD-Koopman model identification                                      |
| `vehicle_sim/controllers/path_tracking_mpc/features.py`       | feature vector and Koopman observable construction                     |
| `vehicle_sim/controllers/path_tracking_mpc/mpc.py`            | Linear, Fixed Koopman, Online Koopman MPC controller implementation    |
| `vehicle_sim/controllers/path_tracking_mpc/rls.py`            | matrix RLS update logic                                                |
| `vehicle_sim/utils/direct_ackermann_steering.py`              | single bicycle steering command to FL/FR Ackermann steering conversion |
| `vehicle_sim/utils/path_tracking_sim.py`                      | closed-loop simulation, path definition, metric logging                |
| `vehicle_sim/experiments/edmd_koopman_mpc_mvp.py`             | main benchmark experiment script                                       |
| `docs/koopman_mpc_explanation.md`                             | 연구 구조 설명                                                               |
| `docs/professor_qa.md`                                        | 발표 및 질의응답 대비 정리                                                        |

---

## Final Poster

본 연구의 최종 학회 포스터는 아래 경로에 포함되어 있습니다.

```text id="g0xx68"
assets/자동차 공학회_여진승_포스터_최종.pdf
assets/자동차 공학회_여진승_포스터_최종.png
```

README 상단의 포스터 이미지를 클릭하면 PDF 원본을 확인할 수 있습니다.

---

## Final Candidate Result

최종 후보 실험 결과는 다음 경로에 저장되어 있습니다.

```text id="cl7z8s"
vehicle_sim/experiments/results/award_ready_online_koopman_benchmark_gap_boost/best_candidate/
```

대규모 sweep 결과와 smoke-test output folder는 GitHub 저장소에서 제외했습니다.
필요한 경우 experiment script를 통해 다시 생성할 수 있습니다.

---

## Example Run

저장소 루트에서 다음 명령으로 benchmark를 실행할 수 있습니다.

```bash id="dhh9sd"
python -m vehicle_sim.experiments.edmd_koopman_mpc_mvp \
  --scenario nonstationary_adaptive_technical_course \
  --out-dir vehicle_sim/experiments/results/manual_run
```

Windows PowerShell에서는 다음과 같이 실행할 수 있습니다.

```powershell id="eaaem6"
python -m vehicle_sim.experiments.edmd_koopman_mpc_mvp `
  --scenario nonstationary_adaptive_technical_course `
  --out-dir vehicle_sim\experiments\results\manual_run
```

---

## Important Experiment Settings

대표 실험 설정은 다음과 같습니다.

| Parameter                   | Value                                         |
| --------------------------- | --------------------------------------------- |
| Controller period           | `control_dt = 0.05 s`                         |
| Control frequency           | `20 Hz`                                       |
| Plant integration period    | `plant_dt = 0.01 s`                           |
| Plant integration frequency | `100 Hz`                                      |
| Steering command            | single `delta_cmd [rad]`                      |
| Plant                       | lab-provided E-Corner based VehicleBody plant |
| Steering interface          | Direct Ackermann wrapper                      |
| Rear steering               | fixed                                         |
| Tire model                  | optional Fiala lateral tire model             |
| Solver                      | CVX / OSQP based prototype                    |

---

## Notes on Runtime

본 구현은 Python/CVX 기반 연구 프로토타입입니다.

최종 실험 로그 기준으로 평균 solve time은 20 Hz 제어 주기 근처에서 동작 가능한 수준이었지만, 일부 worst-case에서는 50 ms 제어 주기를 초과하는 구간이 있었습니다.

따라서 본 저장소는 다음 목적에 적합합니다.

* simulation validation
* controller structure comparison
* poster-level experimentation
* EDMD-Koopman residual correction 검증
* online RLS adaptation 가능성 확인

반면, hard real-time embedded deployment를 목적으로 한 최적화 구현은 아닙니다.

---

## Key Contributions

본 프로젝트의 핵심 기여는 다음과 같습니다.

1. **Residual correction 기반 MPC 구조 구현**

   nonlinear plant와 nominal linear bicycle model 사이의 mismatch를 residual로 정의하고, 이를 Koopman predictor로 보정하는 MPC 구조를 구성했습니다.

2. **EDMD-Koopman residual predictor 설계**

   차량 상태, 경로 곡률, 조향 이력, yaw-rate 이력, 타이어 slip/force/load 정보를 포함한 feature를 구성하고, polynomial lifting을 통해 residual prediction에 사용했습니다.

3. **Fixed vs Online Koopman MPC 비교**

   offline EDMD 모델을 그대로 사용하는 Fixed Koopman MPC와, 주행 중 matrix RLS로 업데이트하는 Online Koopman MPC를 비교했습니다.

4. **E-Corner plant의 일반 전륜 조향 조건화**

   연구실 제공 E-Corner plant를 사용하되, rear steering을 고정하고 single bicycle steering command를 FL/FR Ackermann steering angle로 변환하여 일반적인 path tracking 문제에 가깝게 구성했습니다.

5. **Nonstationary driving condition 평가**

   마찰 변화, 큰 조향 입력, 속도 변화, nonlinear tire saturation이 포함된 조건에서 residual correction 기반 MPC의 가능성을 검토했습니다.

---

## Limitations

본 연구에는 다음과 같은 한계가 있습니다.

* Python/CVX 기반 prototype이므로 hard real-time 보장은 어렵습니다.
* 실험은 simulation 환경에서 수행되었습니다.
* E-Corner plant 자체는 연구실 제공 모델을 사용했습니다.
* Online RLS update는 guard를 적용했지만, 모든 조건에서 성능 향상을 보장하지는 않습니다.
* 실제 차량 적용을 위해서는 C++ 기반 최적화, 실시간 QP solver, actuator dynamics, sensor noise, delay 등을 추가로 고려해야 합니다.

---

## Related Documents

추가 설명은 아래 문서에서 확인할 수 있습니다.

```text id="deu0k5"
docs/koopman_mpc_explanation.md
docs/professor_qa.md
```

`koopman_mpc_explanation.md`는 residual, feature, observable, EDMD, online update 구조를 이해하기 쉽게 정리한 문서입니다.
`professor_qa.md`는 발표 또는 질의응답 상황에서 받을 수 있는 핵심 질문과 답변을 정리한 문서입니다.

---

## Author

여진승
Chungbuk National University
Department of Intelligent Robotics Engineering
