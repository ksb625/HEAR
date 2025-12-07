# HEAR📢 :Hybrid audio restoration Enhancement with Adaptive Reinforcement

본 프로젝트는 강화학습 기반 multi-step 음성 노이즈 제거(RL-based Speech Denoising)를 목표로 하며,
noisy 음성을 입력으로 받아 주파수 도메인에서 여러 단계에 걸쳐 gain/mask를 점진적으로 조정하는 정책을 학습한다.
PPO 알고리즘을 기반으로, 다양한 고전적 DSP denoiser(Wiener, Spectral Subtraction, Spectral Gate, Wavelet)를 action으로 선택·조합하여 음질(SI-SDR)과 발음 명료도(STOI/ESTOI)의 균형을 함께 최적화하는 강화학습 프레임워크를 구현하였다.

**팀원:** 20231837 김상부 / 20231846 송지아

## 📊 프로젝트 개요

![Project Overview](hear_images.png)

## 📁 프로젝트 구조

```
HEAR/
├── Dockerfile                  # Docker 환경 설정
├── .dockerignore               # Docker 빌드 제외 파일
├── .gitignore                  # Git 제외 파일
├── README.md                   # 프로젝트 문서
├── hear_images.png             # 프로젝트 개요 이미지
├── inference.py                # 학습된 모델로 추론
├── run_experiments.py          # 실험 실행 스크립트
├── rl/                         # RL 관련 모듈
│   ├── __init__.py
│   ├── train_agent.py          # RL 에이전트 학습
│   ├── custom_algorithms.py    # PPO/SAC/TD3 구현
│   ├── env.py                  # Denoising 환경 정의
│   └── dataset.py              # 데이터셋 로딩
├── utils/                      # 유틸리티 모듈
│   ├── __init__.py
│   ├── extract_features.py     # 오디오 특징 추출
│   ├── mix_noisy_dataset.py    # 노이즈와 Clean Speech 혼합
│   └── denoise_metrics.py      # Denoising 성능 평가
├── train_data/                 # 학습 데이터
│   ├── train/
│   │   ├── clean/              # Clean 오디오 파일
│   │   ├── noisy/              # Noisy 오디오 파일
│   │   └── meta.csv            # 메타데이터
│   └── train_state_features.csv # 추출된 특징 데이터
├── data_sample/                # 오디오 샘플 예시
│   ├── clean_1.wav
│   ├── noisy_1.wav
│   ├── denoise_1.wav
│   └── ...
├── runs/                       # 학습 결과 저장 (학습시 생성)
│   └── experiments/            # 실험별 결과 저장 경로
└── weight/                     # 미리 학습된 모델 가중치
```

## 📈 주요 결과 (Summary)

다양한 SNR 조건에서의 성능 평가 결과:

- **SNR 0 dB:** SI-SDR ≈ 5.7 dB (noisy 대비: -0.02 dB)
- **SNR 5 dB:** SI-SDR ≈ 9.2 dB (noisy 대비: 5.01 dB)
- **SNR 10 dB:** SI-SDR ≈ 14.4 dB (noisy 대비: 10.03 dB)

## 🎵 오디오 예시

다음은 denoising 전후의 오디오 샘플입니다:

### Sample 1
- **Clean (원본)**
  <audio controls src="data_sample/clean_1.wav"></audio>
  [다운로드](data_sample/clean_1.wav)

- **Noisy (노이즈 포함)**
  <audio controls src="data_sample/noisy_1.wav"></audio>
  [다운로드](data_sample/noisy_1.wav)

- **Denoised (처리 후)**
  <audio controls src="data_sample/denoise_1.wav"></audio>
  [다운로드](data_sample/denoise_1.wav)

### Sample 2
- **Clean (원본)**
  <audio controls src="data_sample/clean_2.wav"></audio>
  [다운로드](data_sample/clean_2.wav)

- **Noisy (노이즈 포함)**
  <audio controls src="data_sample/noisy_2.wav"></audio>
  [다운로드](data_sample/noisy_2.wav)

- **Denoised (처리 후)**
  <audio controls src="data_sample/denoise_2.wav"></audio>
  [다운로드](data_sample/denoise_2.wav)

## 🚀 빠른 시작

### 1. 학습 데이터 준비

⚠️ **데이터 재배포 제한 사항**

- **Clean Speech**: AI Hub KsponSpeech (라이선스 제한)
- **Noise**: ESC-50 (공개 데이터셋)

AI Hub의 라이선스 정책에 따라 KsponSpeech 오디오를 포함한 혼합 데이터셋은 **공개적으로 재배포할 수 없습니다**.

사용자는 원본 데이터셋을 개별적으로 다운로드하고, 제공된 스크립트를 사용하여 로컬에서 혼합 데이터를 생성해야 합니다.

**데이터 준비 방법:**

1. [AI Hub](https://www.aihub.or.kr/)에서 KsponSpeech 데이터셋 다운로드
2. [ESC-50](https://github.com/karolpiczak/ESC-50) 노이즈 데이터셋 다운로드
3. 유틸리티 섹션의 "데이터셋 생성"을 참고하여 clean/noisy 데이터셋 생성

### 2. 저장소 클론

```bash
git clone https://github.com/ksb625/HEAR.git
cd HEAR
```

### 3. Docker 환경 설정

```bash
# Docker 이미지 빌드
cd /your_project_path/HEAR
docker build -t rl-audio-env .

# 컨테이너 실행 (GPU 사용)
docker run -it --gpus all \
  -v /your_project_path/HEAR:/workspace \
  rl-audio-env
```

## 📝 주요 스크립트 사용법

### 1. RL 에이전트 학습 (Training)

PPO, SAC, TD3 알고리즘으로 denoising 에이전트를 학습합니다.

```bash
python -m rl.train_agent \
    --algo ppo \
    --features train_data/train_state_features.csv \
    --total-steps 50000 \
    --max-steps 3 \
    --reward-scale 4.0 \
    --stoi-weight 10.0 \
    --estoi-weight 10.0 \
    --diversity-weight 0.1 \
    --device cuda \
    --wandb-project rl-denoise

```

**주요 옵션:**
- `--algo`: 알고리즘 선택 (`ppo`, `sac`, `td3`, 기본: `ppo`)
- `--features`: 특징 CSV 파일 경로 (기본: `train_data/train_state_features.csv`)
- `--total-steps`: 총 학습 스텝 수 (기본: 200000)
- `--max-steps`: 에피소드당 최대 denoising 스텝 (기본: 3)
- `--reward-scale`: 보상 스케일 (기본: 5.0)
- `--stoi-weight`, `--estoi-weight`: STOI/ESTOI 가중치 (기본: 0.0)
- `--diversity-weight`: 방법 다양성 가중치 (기본: 0.0)
- `--device`: 사용할 디바이스 (`cuda` 또는 `cpu`, 기본: 자동 감지)
- `--wandb-project`: Weights & Biases 프로젝트 이름 (기본: None)
- `--log-dir`: 로그 및 체크포인트 저장 디렉토리 (기본: `runs/rl_train`)

**PPO 관련 옵션:**
- `--clip-range`: PPO 클리핑 범위 (기본: 0.2)
- `--entropy-coef`: 엔트로피 계수 (기본: 0.0)
- `--rollout-steps`: 롤아웃 버퍼 크기 (기본: 2048)
- `--update-epochs`: 업데이트 에포크 수 (기본: 10)
- `--batch-size`: 배치 크기 (기본: 256)
- `--actor-lr`, `--critic-lr`: Actor/Critic 학습률 (기본: 3e-4)

**SAC/TD3 관련 옵션:**
- `--random-steps`: 랜덤 액션 스텝 수 (기본: 2000)
- `--warmup-steps`: 워밍업 스텝 수 (기본: 4000)
- `--tau`: 타겟 네트워크 업데이트 계수 (기본: 0.005)


### 2. RL 에이전트 추론 (inference)

학습된 모델로 noisy 오디오를 denoising합니다.

```bash
# 단일 모델 추론
python inference.py \
    --checkpoint runs/rl_train/model.pt \
    --input train_data/noisy/sample.wav \
    --output denoised_output.wav
```

**주요 옵션:**
- `--checkpoint`: 학습된 모델 체크포인트 경로 (필수, 여러 개 지정 가능)
- `--input`: 입력 noisy 오디오 파일 경로 (필수)
- `--output`: 출력 denoised 오디오 파일 경로 (필수)
- `--train-features`: 학습 시 사용한 특징 CSV (scaler용, 기본: `train_data/train_state_features.csv`)
- `--clean`: (선택) Clean reference 파일 (메트릭 계산용)
- `--max-steps`: Denoising 스텝 수 (기본: 체크포인트에 저장된 값 사용)
- `--device`: 사용할 디바이스 (`cuda` 또는 `cpu`, 기본: 자동 감지)
- `--target-sr`: 타겟 샘플레이트 (기본: 16000)
- `--n-mels`: Mel 스펙트로그램 밴드 수 (기본: 64)
- `--n-fft`: FFT 크기 (기본: 1024)
- `--hop-length`: Hop length (기본: 256)

**출력:**
- Denoised 오디오 파일 (.wav)
- 여러 체크포인트 사용 시: `denoised_output__{tag}.wav` 형식으로 저장
- Clean reference가 제공되면 SI-SDR, STOI, ESTOI 메트릭 출력

## 🔧 유틸리티 (Utils)

### 1. 데이터셋 생성 (Dataset Creation)

KsponSpeech clean 오디오와 ESC-50 노이즈를 혼합하여 학습용 데이터셋을 생성합니다.

```bash
python utils/mix_noisy_dataset.py \
    --clean-root KsponSpeech_01 \
    --noise-root noise_select \
    --output-root train_data \
    --split train \
    --snr-db 0,5,10 \
    --seed 1337 \
    --esc50-meta esc50-meta.xlsx
```

**주요 옵션:**
- `--clean-root`: KsponSpeech 데이터 루트 디렉토리 (기본: `KsponSpeech_01`)
- `--noise-root`: ESC-50 노이즈 디렉토리 (기본: `noise_select`)
- `--output-root`: 출력 디렉토리 (기본: `data_mixed`)
- `--split`: 데이터셋 분할 (기본: `train`)
- `--snr-db`: SNR 값들 (기본: `0,5,10`)
- `--clean-limit`: 처리할 clean 파일 수 제한 (0 = 전체)
- `--esc50-meta`: ESC-50 메타데이터 파일 (선택사항)
- `--seed`: 랜덤 시드 (기본: 1337)
- `--target-sr`: 타겟 샘플레이트 (기본: 16000)

**출력:**
- `train_data/train/clean/`: Clean 오디오 파일
- `train_data/train/noisy/`: Noisy 오디오 파일
- `train_data/train/meta.csv`: 메타데이터 (utt_id, clean_path, noisy_path, clean_source, noise_source, snr_db, duration_sec 등)

### 2. 특징 추출 (Feature Extraction)

오디오 파일에서 RL 상태 입력용 특징을 추출합니다.

```bash
python utils/extract_features.py \
    --meta-path train_data/train/meta.csv \
    --output-path train_data/train_state_features.csv \
    --target-sr 16000 \
    --n-mels 64 \
    --n-fft 1024 \
    --hop-length 256
```

**중요:** `meta.csv` 파일은 다음 구조를 가져야 합니다:
- `utt_id`: 발화 ID
- `clean_path`: Clean 오디오 파일 경로 (상대 경로)
- `noisy_path`: Noisy 오디오 파일 경로 (상대 경로)
- `clean_source`: Clean 오디오 소스 정보
- `noise_source`: 노이즈 소스 정보
- `snr_db`: SNR 값 (dB)
- `duration_sec`: 오디오 길이 (초)

**주요 옵션:**
- `--meta-path`: 메타데이터 CSV 파일 경로 (기본: `data_mixed/train/meta.csv`)
- `--output-path`: 출력 특징 CSV 파일 경로 (기본: `train_data/train_state_features.csv`)
- `--target-sr`: 타겟 샘플레이트 (기본: 16000)
- `--n-mels`: Mel 스펙트로그램 밴드 수 (기본: 64)
- `--n-fft`: FFT 크기 (기본: 1024)
- `--hop-length`: Hop length (기본: 256)
- `--limit`: 처리할 파일 수 제한 (기본: 전체)

### 3. 성능 평가 (Metrics Evaluation)

Denoising 전후의 메트릭을 비교합니다.

```bash
python utils/denoise_metrics.py \
    --checkpoint weight/model.pt \
    --meta train_data/train/meta.csv \
    --train-features train_data/train_state_features.csv \
    --sample-size 100 \
    --snr-db 0 --snr-db 5 --snr-db 10 \
    --csv results.csv \
    --plot-dir plots
```

**주요 옵션:**
- `--checkpoint`: 평가할 모델 체크포인트 (여러 개 가능)
- `--meta`: 메타데이터 CSV (기본: `train_data/meta.csv`)
- `--train-features`: 학습 시 사용한 특징 CSV (기본: `train_data/train_state_features.csv`)
- `--sample-size`: 평가할 샘플 수 (0 = 전체, 기본: 100)
- `--snr-db`: 평가할 SNR 값들 (여러 개 지정 가능)
- `--seed`: 샘플링 시드 (기본: 0)
- `--csv`: 결과 CSV 저장 경로
- `--plot-dir`: 플롯 저장 디렉토리
- `--output-dir`: Denoised 오디오 저장 디렉토리

**출력:**
- 메트릭 CSV 파일 (SI-SDR, STOI, ESTOI)
- 플롯 이미지 (지정 시)
- Denoised 오디오 파일 (지정 시)

## 💡 전체 워크플로우

```bash
# 1. 데이터셋 생성 (노이즈와 Clean Speech 섞기)
python utils/mix_noisy_dataset.py \
    --clean-root KsponSpeech_01 \
    --noise-root noise_select \
    --output-root train_data \
    --snr-db 0,5,10

# 2. 특징 추출
python utils/extract_features.py \
    --meta-path train_data/train/meta.csv \
    --output-path train_data/train_state_features.csv

# 3. 모델 학습
python -m rl.train_agent \
    --algo ppo \
    --features train_data/train_state_features.csv \
    --total-steps 50000 \
    --max-steps 3 \
    --reward-scale 4.0 \
    --stoi-weight 10.0 \
    --estoi-weight 10.0 \
    --diversity-weight 0.1 \
    --device cuda

# 4. 추론
python inference.py \
    --checkpoint runs/rl_train/model.pt \
    --input train_data/train/noisy/test_sample.wav \
    --output test_denoised.wav

# 5. 성능 평가
python utils/denoise_metrics.py \
    --checkpoint runs/rl_train/model.pt \
    --meta train_data/train/meta.csv \
    --train-features train_data/train_state_features.csv \
    --sample-size 100
```

