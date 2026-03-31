# 🦾 UR10e Hand-Following Polishing System (2026-1 URP)

본 프로젝트는 **RealSense L515** 레이저 스캐너와 **MediaPipe**를 결합하여 사용자의 손 동작을 실시간 트래킹하고, 이를 통해 **UR10e 로봇**의 폴리싱 작업을 제어하는 ROS 2 기반 연구 프로젝트입니다.

> ⚠️ **중요:** 대용량 모델 파일 및 데이터셋은 제외되어 있습니다. 다른 환경에서 실행 전 [SETUP_EXTERNAL_ASSETS.md](./SETUP_EXTERNAL_ASSETS.md) 가이드를 확인하세요.

---

## 📌 주요 특징
- **Vision:** MediaPipe Index Tip 3D Tracking (RealSense L515 Depth 결합).
- **Control:** FSM(Finite State Machine) 기반 중앙 제어 (STANDBY, FOLLOWING, TEACHING, PLAYBACK).
- **Simulation:** MuJoCo Physics 기반 고정밀 시뮬레이션 환경.
- **Hardware:** Universal Robots UR10e + Intel RealSense L515.

---

## 🛠 환경 설정 (Requirements) - **중요**
L515 모델은 SDK 버전에 매우 민감하므로 아래 버전을 반드시 준수해야 합니다.

### 1. Hardware & OS
- **OS:** Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Middleware:** ROS 2 Humble Hawksbill
- **Camera:** Intel RealSense L515 (USB 3.0 연결 필수)

### 2. Specific SDK Versions
- **Intel RealSense SDK (librealsense):** `v2.53.1` 
  - *참고: 최신 버전(2.54+)은 L515 지원이 불안정할 수 있으므로 2.53.1 소스 빌드 권장.*
- **Python:** `3.10` 계열 (MediaPipe & pyrealsense2 안정성 확인)
- **MuJoCo:** `3.1.x` 이상

### 3. Dependencies
```bash
pip install pyrealsense2==2.53.1.* mediapipe mujoco open3d opencv-python
```

---

## 📂 프로젝트 구조 (Project Structure)
```text
.
├── central/                # 중앙 제어기 (FSM 로직)
│   └── nrs_central_controller_v17.py  # [Main] 최종 컨트롤러 코드
├── mp_hand_tracking/       # 비전 모듈 (MediaPipe 기반)
│   └── realsense_mediapipe_v7.py      # [Main] L515 트래킹 노드
├── simulation_mujoco/      # 시뮬레이션 환경 (UR10e 모델링)
│   ├── ros2_mujoco_pure.py            # MuJoCo-ROS 2 브릿지
│   └── ur10e_update.xml               # 로봇/작업대 시뮬레이션 에셋
├── references/             # [Optional] 이전 연구 자료 및 PCD 데이터
└── docs/                   # 인수인계 문서 (인수인계.pdf 등)
```

---

## 🚀 실행 방법 (Execution)

각 모듈은 ROS 2 통신망 상에서 독립적인 노드로 작동합니다.

1. **Simulation Bridge:**
   ```bash
   python3 simulation_mujoco/ros2_mujoco_pure.py
   ```
2. **Vision Node (Hand Tracking):**
   ```bash
   python3 mp_hand_tracking/realsense_mediapipe_v7.py
   ```
3. **Central Controller:**
   ```bash
   python3 central/nrs_central_controller_v17.py
   ```

---

## 🎮 운용 모드 및 제스처
- **STANDBY:** 로봇 대기 상태.
- **FOLLOWING:** 사용자의 검지 끝을 따라 로봇이 실시간 이동.
- **TEACHING:** 궤적 기록 모드 (특정 제스처 시 시작).
- **PLAYBACK:** 기록된 궤적 반복 수행 (폴리싱 작업).

---

## ⚠️ 주의 사항 (Disclaimer)
1. **대용량 데이터 제외:** `hamer/_DATA` 폴더(12GB) 및 각종 `.ckpt`, `.pth` 모델 파일은 깃허브 용량 문제로 제외되었습니다.
2. **L515 호환성:** 카메라 인식 문제 발생 시 `realsense-viewer`에서 L515가 정상적으로 노출되는지 먼저 확인하십시오. (Firmware 1.6.3.0+ 권장)
