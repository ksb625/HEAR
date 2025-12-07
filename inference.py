"""Run inference with a trained PPO denoising policy (custom implementation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pystoi import stoi as stoi_metric

from utils.extract_features import compute_features, load_audio
from rl.custom_algorithms import PPOAgent
from rl.env import (
    METHODS,
    si_sdr,
    mmse_lsa,
    nmf_denoise,
    spectral_gate,
    spectral_sub,
    wavelet_denoise,
    wiener_filter,
)


EXCLUDE_COLS = {"utt_id", "scene", "macro_scene", "snr_db", "noisy_path", "clean_path"}


def resolve_device(requested: str | None) -> torch.device:
    """Return a valid torch.device, falling back to CPU if needed."""
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[inference] CUDA requested but not available, falling back to CPU.", file=sys.stderr)
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scaler(feature_csv: Path) -> Tuple[StandardScaler, List[str]]:
    df = pd.read_csv(feature_csv)
    feature_cols = [
        col
        for col in df.columns
        if col not in EXCLUDE_COLS and np.issubdtype(df[col].dtype, np.number)
    ]
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler, feature_cols


def make_feature_vector(
    audio_path: Path,
    target_sr: int,
    feature_cols: List[str],
    scaler: StandardScaler,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    audio = load_audio(audio_path, target_sr)
    feats = compute_features(
        audio,
        sr=target_sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    vector = np.array([feats[col] for col in feature_cols], dtype=np.float32)
    scaled = scaler.transform(vector.reshape(1, -1))[0]
    return scaled.astype(np.float32), feats, audio


def load_agent_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    obs_dim: int,
) -> Tuple[PPOAgent, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("algo") != "ppo":
        raise ValueError(f"Checkpoint algo={payload.get('algo')} is not PPO.")

    saved_args: Dict[str, Any] = payload.get("args", {})
    hidden_dims = tuple(saved_args.get("hidden_dims", [256, 256]))
    lr = float(saved_args.get("actor_lr", 3e-4))
    max_grad_norm = float(saved_args.get("max_grad_norm", 1.0))

    action_low = np.asarray(payload["action_low"], dtype=np.float32)
    action_high = np.asarray(payload["action_high"], dtype=np.float32)
    act_dim = action_low.shape[0]

    agent = PPOAgent(
        obs_dim,
        act_dim,
        action_low,
        action_high,
        device,
        hidden_dims=hidden_dims,
        lr=lr,
        max_grad_norm=max_grad_norm,
    )
    agent.policy.load_state_dict(payload["policy_state"])
    agent.value_net.load_state_dict(payload["value_state"])
    agent.policy.eval()
    agent.value_net.eval()
    return agent, saved_args


def apply_action(
    noisy: np.ndarray,
    sr: int,
    action: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> Tuple[np.ndarray, int, str, float, float, float]:
    raw_method = float(action[0])
    method_idx = int(np.clip(np.round(raw_method), 0, len(METHODS) - 1))
    strength = float(np.clip(action[1], 0.0, 1.0))
    smoothing = float(np.clip(action[2], 0.0, 1.0))
    bandmix = float(np.clip(action[3], 0.0, 1.0))
    method_name = METHODS[method_idx]

    if method_name == "spectral_sub":
        denoised = spectral_sub(noisy, sr, strength, smoothing, bandmix, n_fft=n_fft, hop_length=hop_length)
    elif method_name == "wiener":
        denoised = wiener_filter(noisy, sr, strength, smoothing, bandmix, frame_len=n_fft, hop_len=hop_length)
    elif method_name == "spectral_gate":
        denoised = spectral_gate(noisy, sr, strength, smoothing, bandmix, n_fft=n_fft, hop_length=hop_length)
    elif method_name == "mmse_lsa":
        denoised = mmse_lsa(noisy, sr, strength, smoothing, bandmix, n_fft=n_fft, hop_length=hop_length)
    elif method_name == "nmf":
        denoised = nmf_denoise(noisy, sr, strength, smoothing, bandmix, n_fft=n_fft, hop_length=hop_length)
    else:
        denoised = wavelet_denoise(noisy, strength, smoothing, bandmix)
    return denoised, method_idx, method_name, strength, smoothing, bandmix


def build_observation(
    base_features: np.ndarray,
    prev_action_onehot: np.ndarray,
    prev_strength: float,
    prev_smoothing: float,
    prev_bandmix: float,
    step: int,
    max_steps: int,
) -> np.ndarray:
    step_fraction = step / max_steps
    extras = np.concatenate(
        [
            prev_action_onehot,
            np.array([prev_strength, prev_smoothing, prev_bandmix, step_fraction], dtype=np.float32),
        ]
    )
    return np.concatenate([base_features, extras]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply one or more PPO checkpoints to a noisy file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Path to custom PPO checkpoint (model.pt). Repeat for multiple models.",
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        default=Path("train_data/train_state_features.csv"),
        help="Feature CSV used during training (for scaler).",
    )
    parser.add_argument("--input", type=Path, required=True, help="Noisy audio file to enhance.")
    parser.add_argument("--clean", type=Path, default=None, help="Optional clean reference for metrics.")
    parser.add_argument("--output", type=Path, required=True, help="Base path for denoised audio output.")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max denoising steps (default: value stored in checkpoint).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    scaler, feature_cols = build_scaler(args.train_features)

    base_features, feature_map, noisy_audio = make_feature_vector(
        args.input,
        args.target_sr,
        feature_cols,
        scaler,
        args.n_mels,
        args.n_fft,
        args.hop_length,
    )

    clean_audio = None
    if args.clean is not None:
        clean_audio = load_audio(args.clean, args.target_sr)
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

    obs_dim = base_features.shape[0] + len(METHODS) + 4

    results: List[Dict[str, Any]] = []
    multi = len(args.checkpoint) > 1

    for ckpt in args.checkpoint:
        agent, saved_args = load_agent_from_checkpoint(ckpt, device, obs_dim)
        max_steps = args.max_steps if args.max_steps is not None else int(saved_args.get("max_steps", 3))

        prev_onehot = np.zeros(len(METHODS), dtype=np.float32)
        prev_strength = 0.0
        prev_smoothing = 0.0
        prev_bandmix = 0.0
        current_audio = noisy_audio.copy()

        for step in range(max_steps):
            obs = build_observation(
                base_features, prev_onehot, prev_strength, prev_smoothing, prev_bandmix, step, max_steps
            )
            env_action, _, _, _ = agent.act(obs, deterministic=True)
            denoised_audio, method_idx, method_name, strength, smoothing, bandmix = apply_action(
                current_audio,
                args.target_sr,
                env_action,
                args.n_fft,
                args.hop_length,
            )
            current_audio = denoised_audio
            prev_onehot = np.zeros(len(METHODS), dtype=np.float32)
            prev_onehot[method_idx] = 1.0
            prev_strength = strength
            prev_smoothing = smoothing
            prev_bandmix = bandmix
            print(
                f"[inference:{ckpt}] step {step + 1}/{max_steps}: "
                f"method={method_name}, strength={strength:.3f}, "
                f"smoothing={smoothing:.3f}, bandmix={bandmix:.3f}"
            )

        tag = ckpt.parent.name
        if multi:
            output_path = args.output.with_name(f"{args.output.stem}__{tag}{args.output.suffix}")
        else:
            output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf  # lazy import

        sf.write(output_path, current_audio, args.target_sr)

        row: Dict[str, Any] = {"checkpoint": str(ckpt), "steps": max_steps, "output": str(output_path)}
        if clean_audio is not None:
            min_len = min(len(clean_audio), len(current_audio))
            ref = clean_audio[:min_len]
            hyp = current_audio[:min_len]
            row["si_sdr"] = float(si_sdr(ref, hyp))
            try:
                row["stoi"] = float(stoi_metric(ref, hyp, args.target_sr, extended=False))
                row["estoi"] = float(stoi_metric(ref, hyp, args.target_sr, extended=True))
            except Exception:
                row["stoi"] = 0.0
                row["estoi"] = 0.0
        results.append(row)

        print(
            f"[inference] saved denoised audio -> {output_path}\n"
            f"  steps={max_steps}, input_duration={feature_map['duration_sec']:.2f}s\n"
            f"  checkpoint={ckpt}"
        )

    if results:
        print("\n[inference] summary:")
        for row in results:
            metrics = []
            if "si_sdr" in row:
                metrics.append(f"SI-SDR={row['si_sdr']:.2f} dB")
            if "stoi" in row:
                metrics.append(f"STOI={row['stoi']:.3f}")
            if "estoi" in row:
                metrics.append(f"ESTOI={row['estoi']:.3f}")
            metric_str = ", ".join(metrics) if metrics else "metrics=n/a"
            print(f"  - {row['output']}: {metric_str}")


if __name__ == "__main__":
    main()

