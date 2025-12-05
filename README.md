# HEAR
HEAR 📢: Hybrid audio restoration Enhancement with Adaptive Reinforcement
🎧 RL-based Multi-step Speech Denoising

Reinforcement Learning based Multi-step Speech Denoising System
본 프로젝트는 classical denoising 기법들을 강화학습으로 제어하여,
multi-step noise reduction을 통해 음질(SI-SDR)을 최대화하는 음성 향상 시스템을 구현한다.

📌 Key Features

✅ Multi-step Denoising RL Environment

✅ Classical Denoisers Controlled by RL

Spectral Subtraction

Wiener Filter

Spectral Gate

Wavelet Denoising

✅ Reward based on SI-SDR Improvement + Diversity Bonus

✅ Algorithms Supported

PPO

SAC

TD3

✅ Large-scale Korean Speech Dataset (KsponSpeech)

✅ Noise from ESC-50 (Environmental Sounds)

✅ Hyperparameter Ablation & Algorithm Comparison

✅ SNR-wise Performance Analysis (0 / 5 / 10 dB)

🧠 Method Overview

This system formulates speech denoising as a Markov Decision Process (MDP):

State:
Acoustic feature vector extracted from noisy audio
(RMS, ZCR, duration, log-mel mean & std)

Previous action context

Step progress

Action (4D Continuous):

[method_idx, strength, smoothing, bandmix]


method_idx → selects one denoiser

strength → denoising intensity

smoothing → post-filter smoothness

bandmix → frequency band emphasis

Reward:

ΔSI-SDR / reward_scale + diversity_bonus


(optional STOI / ESTOI available for ablation)

Multi-step Policy Execution:
The agent performs multiple denoising steps (max_steps) sequentially
within one episode.

📂 Dataset
✅ Clean Speech

KsponSpeech (Korean Conversational Speech Dataset)

16kHz mono resampled

Short utterances (3–5 sec)

✅ Noise

ESC-50 Environmental Sounds

Selected classes:

Vacuum cleaner

Helicopter

Train

Engine

Sea waves

Rain

Wind

Crackling fire

Washing machine

Thunderstorm

✅ Mixing

SNR levels: 0 / 5 / 10 dB

Clean–noise pairing with random offsets

Metadata stored in meta.csv

📊 Evaluation Metric

✅ Primary Metric:
SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

✅ Reference Metrics (Ablation):
STOI, ESTOI

✅ Final results reported as mean ± std across the test set.

🚀 Getting Started
1️⃣ Environment Setup
docker build -t rl-denoise .
docker run --gpus all -it rl-denoise


or using Conda:

conda create -n rl-denoise python=3.10
conda activate rl-denoise
pip install -r requirements.txt

2️⃣ Feature Extraction
python features/extract_features.py


This generates:

features/train_state_features.csv

3️⃣ Training (Example: PPO)
python -m rl.train_agent \
  --algo ppo \
  --features features/train_state_features.csv \
  --total-steps 50000 \
  --reward-scale 5.0 \
  --diversity-weight 0.1 \
  --max-steps 4 \
  --device cuda

4️⃣ Evaluation
python eval_denoise.py --run-dir runs/experiments/...

⚙️ Key Hyperparameters
Parameter	Description
reward_scale	Scales SI-SDR gain
diversity_weight	Penalizes repeated denoiser usage
max_steps	Number of denoising steps
gamma	Discount factor
gae_lambda	GAE parameter
clip_range	PPO clipping range
🧪 Experimental Highlights

✅ Multi-step denoising outperforms single-step

✅ SAC achieved higher SI-SDR than PPO in final performance

✅ PPO showed the most stable learning curve

✅ Intelligibility reward weight affects optimal policy depending on SNR

✅ High SNR benefits from balanced denoising policies

👥 Contributors
Name	Contribution
Member A	Dataset construction, feature extraction, RL environment, reward design
Member B	RL algorithms, hyperparameter tuning, evaluation, visualization
📜 License

This project is released for educational and research purposes.
