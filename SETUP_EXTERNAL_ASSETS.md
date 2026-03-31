# 📦 External Assets Setup Guide

본 프로젝트는 깃허브 용량 제한 및 라이선스 정책으로 인해 일부 대용량 모델 파일과 데이터셋을 포함하고 있지 않습니다. 다른 환경에서 프로젝트를 클론(Clone)한 후, 아래 절차에 따라 필요한 에셋을 수동으로 추가해야 정상 작동합니다.

---

## 1. HAMER Model Weights (가장 중요)
실시간 핸드 트래킹을 위한 사전 학습된 가중치 파일들입니다.

- **대상 경로:** `2026-1_urp/hamer/_DATA/`
- **필요 파일:**
  - `hamer_ckpts/checkpoints/hamer.ckpt`
  - `vitpose_ckpts/vitpose_base/wholebody.pth`
  - `data/mano/MANO_LEFT.pkl`, `MANO_RIGHT.pkl`
- **다운로드 방법:**
  - [HAMER 공식 저장소](https://github.com/facebookresearch/hamer)의 `fetch_demo_data.sh` 스크립트를 실행하거나, 공식 링크에서 다운로드하여 위 경로 구조에 맞게 배치하세요.

---

## 2. MANO Model (Hand Mesh)
손의 3D 메쉬 표현을 위한 모델 파일입니다.

- **대상 경로:** `2026-1_urp/mano_v1_2/models/`
- **필요 파일:**
  - `MANO_LEFT.pkl`
  - `MANO_RIGHT.pkl`
  - `SMPLH_female.pkl`, `SMPLH_male.pkl`
- **다운로드 방법:**
  - [MANO 공식 웹사이트](https://mano.is.tue.mpg.de/)에서 계정 생성 후 다운로드하여 해당 폴더에 넣으세요.

---

## 3. Reference Data & Point Clouds
이전 연구에서 수집된 포인트 클라우드(.pcd) 및 시뮬레이션용 .stl 파일들입니다.

- **대상 경로:** `2026-1_urp/references/`
- **참고:** 
  - 실험 데이터 복구가 필요한 경우, 기존 작업 컴퓨터의 `references/zips/` 내 압축 파일들을 해제하여 각 폴더에 배치하세요.
  - 시뮬레이션 환경에 필요한 `.xml` 및 `.stl` 파일은 이미 레포지토리에 포함되어 있으므로, 기본 실행에는 문제가 없습니다.

---

## 4. 환경 변수 및 의존성
- **RealSense SDK:** `v2.53.1` 소스 빌드 버전이 필요합니다.
- **Python:** `pip install -r requirements.txt` (추후 생성 예정)

---

### 💡 팁
에셋 배치가 완료된 후의 전체 폴더 구조는 `README.md`의 `Project Structure` 섹션을 참고하여 확인하시기 바랍니다.
