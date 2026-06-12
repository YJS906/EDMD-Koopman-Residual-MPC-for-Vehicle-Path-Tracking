# EDMD-Koopman Residual MPC for Vehicle Path Tracking

본 저장소는 차량 횡방향 경로 추종을 위한
**EDMD-Koopman residual correction 기반 MPC 연구 프로토타입**을 정리한 저장소입니다.

본 연구에서는 연구실에서 제공받은 **E-Corner 기반 차량 시뮬레이션 plant**를 사용하였습니다.
해당 차량 모델 자체를 새로 개발한 것은 아니며, 본 연구의 초점은 차량 plant 개발이 아니라
**nominal linear bicycle model과 실제 nonlinear plant 사이의 예측 오차를 보정하는 MPC 구조 설계 및 검증**에 있습니다.

또한 E-Corner 차량은 4륜 조향이 가능한 구조를 가질 수 있지만,
본 연구에서는 일반적인 전륜 조향 차량 조건에 가깝게 구성하기 위해 **rear steering은 고정**하고,
하나의 bicycle steering command `delta_cmd`를 전륜 FL/FR Ackermann steering angle로 변환하여 사용하였습니다.

---

## 연구 개요

차량의 횡방향 거동은 타이어 비선형성, 노면 마찰계수 변화, 큰 조향 입력, 횡가속도 등의 영향으로 인해 본질적으로 비선형 특성을 가집니다.

Linear bicycle model 기반 MPC는 계산이 빠르고 구조가 단순하다는 장점이 있지만,
고속 주행, 큰 조향 입력, 마찰계수 변화가 포함된 조건에서는 실제 차량 plant와의 모델 불일치로 인해 경로 추종 성능이 저하될 수 있습니다.

본 연구에서는 이러한 한계를 보완하기 위해,
**linear bicycle model을 nominal prediction model로 사용하고, nonlinear plant와의 예측 오차를 EDMD-Koopman residual predictor로 보정하는 MPC 구조**를 구성하였습니다.

비교한 제어기는 다음과 같습니다.

1. Linear bicycle model MPC
2. Fixed residual EDMD-Koopman MPC
3. Online residual EDMD-Koopman MPC with matrix RLS updates

---

## 핵심 아이디어

본 연구에서 사용한 plant는 연구실에서 제공받은 E-Corner 기반 nonlinear vehicle simulation model입니다.
MPC 내부의 예측 모델은 plant 자체가 아니라, 제어 입력을 계산하기 위한 단순화된 prediction model입니다.

본 연구의 residual Koopman 구조는 다음과 같습니다.

```text
x_nom_next = linear_bicycle_predictor(x_k, delta_k)
residual   = VehicleBody_true_next - x_nom_next
x_pred     = x_nom_next + r_koopman
```

즉, linear bicycle model이 예측한 다음 상태 `x_nom_next`와
실제 nonlinear plant의 다음 상태 `VehicleBody_true_next` 사이의 차이를 residual로 정의합니다.

EDMD-Koopman residual predictor는 이 residual을 예측하고,
MPC는 nominal prediction에 residual prediction을 더한 보정된 상태 예측값을 사용합니다.

```text
x_pred = x_nom_next + r_koopman
```

---

## Controller 구성

### 1. Linear bicycle model MPC

Linear MPC는 linear bicycle model만을 prediction model로 사용합니다.

```text
x_pred = x_nom_next
```

구조가 단순하고 계산이 빠르지만,
타이어 비선형성이나 마찰계수 변화가 큰 조건에서는 nonlinear plant와의 모델 불일치가 커질 수 있습니다.

---

### 2. Fixed Residual EDMD-Koopman MPC

Fixed Koopman MPC는 초기 주행 데이터로 학습한 EDMD-Koopman residual predictor를 사용합니다.

```text
x_pred = x_nom_next + r_koopman
```

단, 주행 중 residual model을 업데이트하지 않습니다.
따라서 초기 학습 조건과 다른 주행 조건이 나타나면 residual prediction 성능이 제한될 수 있습니다.

---

### 3. Online Residual EDMD-Koopman MPC

Online Koopman MPC는 Fixed Koopman MPC와 동일한 초기 EDMD 모델에서 시작합니다.
이후 주행 중 관측되는 plant transition data를 이용하여 matrix RLS 방식으로 residual predictor를 온라인 갱신합니다.

```text
W_0 : offline EDMD로 초기화된 residual matrix
W_k : online RLS update로 갱신되는 residual matrix
```

이를 통해 제한된 초기 데이터만 사용하는 경우보다
마찰계수 변화, 급격한 조향, 속도 변화 등 비정상 주행 조건에서 예측 오차를 점진적으로 줄이는 것을 목표로 합니다.

---

## 차량 모델 사용 조건

본 연구에서는 연구실에서 제공받은 E-Corner 기반 차량 plant를 사용하였습니다.

다만 본 연구의 목적은 E-Corner 차량의 독립 4륜 조향 성능을 평가하는 것이 아니라,
일반적인 차량 경로 추종 문제에서 residual correction 기반 MPC의 효과를 확인하는 것입니다.

따라서 다음과 같이 차량 입력 구조를 단순화하였습니다.

```text
single bicycle steering command: delta_cmd [rad]
front steering: FL/FR Ackermann steering angle
rear steering: fixed
```

즉, 하나의 bicycle steering command를 전륜 Ackermann 조향각으로 변환하고,
후륜 조향은 사용하지 않도록 고정하였습니다.

---

## 주요 코드 구조

```text
vehicle_sim/controllers/path_tracking_mpc/
```

* `linear_bicycle.py`

  * Linear bicycle prediction model

* `edmd.py`

  * EDMD-Koopman model identification and prediction utilities

* `features.py`

  * Koopman observable / feature construction

* `mpc.py`

  * Linear MPC, fixed Koopman MPC, online Koopman MPC controller logic

* `rls.py`

  * Matrix RLS update logic

```text
vehicle_sim/utils/direct_ackermann_steering.py
```

* 하나의 bicycle steering command `delta_cmd`를 전륜 FL/FR Ackermann steering angle로 변환합니다.
* 본 연구에서는 rear steering을 고정하여 일반적인 전륜 조향 차량 조건에 가깝게 사용하였습니다.

```text
vehicle_sim/utils/path_tracking_sim.py
```

* Closed-loop vehicle simulation helper
* Path definition
* Segment metric calculation
* Tire-state logging
* Friction and speed schedule handling

```text
vehicle_sim/experiments/edmd_koopman_mpc_mvp.py
```

* EDMD-Koopman MPC benchmark를 수행하는 main experiment script입니다.

```text
vehicle_sim/models/e_corner/tire/lateral/lateral_tire.py
```

* Linear lateral tire model
* Optional Fiala lateral tire model

---

## 최종 포스터

아래 이미지는 본 연구의 최종 학회 포스터입니다.

![Final Poster](assets/poster_final.png)

PDF 원본은 다음 경로에 둘 수 있습니다.

```text
assets/poster_final.pdf
```

---

## Final Candidate Result

최종 포스터 후보 결과는 다음 경로에 저장되어 있습니다.

```text
vehicle_sim/experiments/results/award_ready_online_koopman_benchmark_gap_boost/best_candidate/
```

대규모 sweep 결과와 smoke-test output folder는 Git에서 제외하였으며,
필요한 경우 experiment CLI를 통해 다시 생성할 수 있습니다.

---

## Example Run

Codex workspace의 Python runtime을 사용하는 경우 다음 명령으로 실행할 수 있습니다.

```powershell
& 'C:\Users\HOME\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m vehicle_sim.experiments.edmd_koopman_mpc_mvp --scenario nonstationary_adaptive_technical_course --out-dir vehicle_sim\experiments\results\manual_run
```

주요 실험 설정은 다음과 같습니다.

* Controller period: `control_dt = 0.05 s`
* Control frequency: `20 Hz`
* Plant integration period: `plant_dt = 0.01 s`
* Plant integration frequency: `100 Hz`
* Steering command: single `delta_cmd [rad]`
* Closed-loop plant: lab-provided E-Corner based VehicleBody plant
* Tire model: optional Fiala lateral tire model
* Steering interface: Direct Ackermann wrapper
* Rear steering: fixed

---

## Notes

본 저장소는 연구 프로토타입입니다.

Python/CVX 기반 구현은 simulation validation 및 poster-level experimentation을 목적으로 작성되었으며,
hard real-time deployment를 목적으로 한 구현은 아닙니다.

추가 설명 문서는 다음 파일에서 확인할 수 있습니다.

```text
docs/koopman_mpc_explanation.md
docs/professor_qa.md
```
