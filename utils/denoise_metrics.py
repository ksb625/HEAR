"""Compare metrics before/after denoising for a sampled subset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.extract_features import load_audio
from inference import (
    apply_action,
    build_observation,
    build_scaler,
    load_agent_from_checkpoint,
    make_feature_vector,
    resolve_device,
)
from rl.env import METHODS, si_sdr
from pystoi import stoi as stoi_metric


def run_episode(
    agent,
    base_features: np.ndarray,
    noisy_audio: np.ndarray,
    target_sr: int,
    n_fft: int,
    hop_length: int,
    max_steps: int,
) -> np.ndarray:
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
        denoised_audio, method_idx, _, strength, smoothing, bandmix = apply_action(
            current_audio, target_sr, env_action, n_fft, hop_length
        )
        current_audio = denoised_audio
        prev_onehot = np.zeros(len(METHODS), dtype=np.float32)
        prev_onehot[method_idx] = 1.0
        prev_strength = strength
        prev_smoothing = smoothing
        prev_bandmix = bandmix
    return current_audio


def compute_metrics(clean: np.ndarray, ref: np.ndarray, sr: int) -> Dict[str, float]:
    min_len = min(len(clean), len(ref))
    clean = clean[:min_len]
    ref = ref[:min_len]
    sisdr_val = float(si_sdr(clean, ref))
    try:
        stoi_val = float(stoi_metric(clean, ref, sr, extended=False))
        estoi_val = float(stoi_metric(clean, ref, sr, extended=True))
    except Exception:
        stoi_val = 0.0
        estoi_val = 0.0
    return {"si_sdr": sisdr_val, "stoi": stoi_val, "estoi": estoi_val}


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(df: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["si_sdr", "stoi", "estoi"]
    models = df["model"].unique().tolist()

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [df[df["model"] == name][metric].dropna() for name in models]
        ax.boxplot(data, labels=models, showfliers=False)
        ax.set_title(f"{metric.upper()} by model")
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("Model")
        fig.tight_layout()
        out_path = plot_dir / f"{metric}_by_model.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    snr_values = sorted(df["snr_db"].unique())
    if len(snr_values) > 1:
        for metric in metrics:
            fig, axes = plt.subplots(len(snr_values), 1, figsize=(6, 4 * len(snr_values)), sharex=True)
            if len(snr_values) == 1:
                axes = [axes]
            for ax, snr in zip(axes, snr_values):
                subset = df[df["snr_db"] == snr]
                data = [subset[subset["model"] == name][metric].dropna() for name in models]
                ax.boxplot(data, labels=models, showfliers=False)
                ax.set_title(f"{metric.upper()} @ {snr:.0f} dB")
                ax.set_ylabel(metric.upper())
            axes[-1].set_xlabel("Model")
            fig.tight_layout()
            out_path = plot_dir / f"{metric}_by_model_snr.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare metrics before/after denoising.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Path to PPO checkpoint (model.pt). Repeat to compare multiple models.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("train_data/meta.csv"),
        help="Meta CSV with clean/noisy paths.",
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        default=Path("train_data/train_state_features.csv"),
        help="Feature CSV used during training (for scaler).",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples to evaluate (0 = all).")
    parser.add_argument(
        "--snr-db",
        type=float,
        action="append",
        default=None,
        help="Optional list of SNR values to include.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps (default: checkpoint value).")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: auto).")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV to log per-sample metrics.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directory to save comparison plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write denoised audio (per model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.meta.exists():
        raise SystemExit(f"Meta CSV not found: {args.meta}")

    device = resolve_device(args.device)
    scaler, feature_cols = build_scaler(args.train_features)
    obs_dim = len(feature_cols) + len(METHODS) + 4

    df = pd.read_csv(args.meta)
    if args.snr_db:
        df = df[df["snr_db"].isin(args.snr_db)]
    if args.sample_size > 0 and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.seed)

    agents: List[Tuple[str, Any, int]] = []
    for ckpt in args.checkpoint:
        agent, saved_args = load_agent_from_checkpoint(ckpt, device, obs_dim)
        max_steps = args.max_steps if args.max_steps is not None else int(saved_args.get("max_steps", 3))
        tag = ckpt.parent.name
        agents.append((tag, agent, max_steps))
        print(f"[denoise] loaded {ckpt} (steps={max_steps})")

    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(df.itertuples(index=False), start=1):
        clean_audio = load_audio(Path(entry.clean_path), args.target_sr)
        base_features, _, noisy_audio = make_feature_vector(
            Path(entry.noisy_path),
            args.target_sr,
            feature_cols,
            scaler,
            args.n_mels,
            args.n_fft,
            args.hop_length,
        )
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        baseline = compute_metrics(clean_audio, noisy_audio, args.target_sr)
        rows.append(
            {
                "idx": idx,
                "utt_id": entry.utt_id,
                "scene": entry.scene,
                "snr_db": float(entry.snr_db),
                "model": "noisy",
                **baseline,
            }
        )

        for tag, agent, max_steps in agents:
            enhanced = run_episode(
                agent, base_features, noisy_audio, args.target_sr, args.n_fft, args.hop_length, max_steps
            )
            enhanced = enhanced[:min_len]
            metrics = compute_metrics(clean_audio, enhanced, args.target_sr)
            rows.append(
                {
                    "idx": idx,
                    "utt_id": entry.utt_id,
                    "scene": entry.scene,
                    "snr_db": float(entry.snr_db),
                    "model": tag,
                    **metrics,
                }
            )

            if args.output_dir:
                out_dir = args.output_dir / tag
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / Path(entry.noisy_path).name
                import soundfile as sf

                sf.write(out_path, enhanced, args.target_sr)

        if idx % 10 == 0 or idx == len(df):
            print(f"[denoise] processed {idx}/{len(df)} samples")

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("No rows collected.")

    summary = out_df.groupby("model")[["si_sdr", "stoi", "estoi"]].agg(["mean", "std"])
    print("\n[denoise] summary by model (mean ± std):")
    for model in summary.index:
        si_mean, si_std = summary.loc[model, ("si_sdr", "mean")], summary.loc[model, ("si_sdr", "std")]
        stoi_mean, stoi_std = summary.loc[model, ("stoi", "mean")], summary.loc[model, ("stoi", "std")]
        estoi_mean, estoi_std = summary.loc[model, ("estoi", "mean")], summary.loc[model, ("estoi", "std")]
        print(
            f"  - {model}: SI-SDR={si_mean:.2f}±{si_std:.2f} dB, "
            f"STOI={stoi_mean:.3f}±{stoi_std:.3f}, ESTOI={estoi_mean:.3f}±{estoi_std:.3f}"
        )

    summary_snr = out_df.groupby(["snr_db", "model"])[["si_sdr", "stoi", "estoi"]].agg(["mean", "std"])
    print("\n[denoise] summary by model × SNR (mean ± std):")
    for (snr, model), values in summary_snr.iterrows():
        si_mean, si_std = values[("si_sdr", "mean")], values[("si_sdr", "std")]
        stoi_mean, stoi_std = values[("stoi", "mean")], values[("stoi", "std")]
        estoi_mean, estoi_std = values[("estoi", "mean")], values[("estoi", "std")]
        print(
            f"  - SNR {snr:.0f} dB / {model}: SI-SDR={si_mean:.2f}±{si_std:.2f} dB, "
            f"STOI={stoi_mean:.3f}±{stoi_std:.3f}, ESTOI={estoi_mean:.3f}±{estoi_std:.3f}"
        )

    if args.csv:
        write_csv(rows, args.csv)
        print(f"[denoise] wrote detailed metrics to {args.csv}")

    if args.plot_dir:
        plot_metrics(out_df, args.plot_dir)
        print(f"[denoise] saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()

