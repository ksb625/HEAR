"""Extract lightweight features from noisy audio for RL state inputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

EPS = 1e-10


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def compute_features(
    audio: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> Dict[str, float]:
    if len(audio) == 0:
        raise ValueError("Empty audio array")

    rms = float(np.sqrt(np.mean(audio ** 2) + EPS))
    peak = float(np.max(np.abs(audio)) + EPS)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel + EPS, ref=1.0)

    mean = np.mean(log_mel, axis=1)
    std = np.std(log_mel, axis=1)

    features = {
        "signal_rms": rms,
        "signal_peak": peak,
        "log_rms": math.log(rms + EPS),
        "zcr": zcr,
        "duration_sec": len(audio) / sr,
    }

    for idx, value in enumerate(mean):
        features[f"mel_mean_{idx:02d}"] = float(value)
    for idx, value in enumerate(std):
        features[f"mel_std_{idx:02d}"] = float(value)

    return features


def write_rows(output_path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No feature rows to write")

    header = rows[0].keys()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features for RL states.")
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("train_data/meta.csv"),
        help="Path to mixing meta CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("train_data/train_state_features.csv"),
        help="Where to save extracted features (default: %(default)s)",
    )
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of rows to process (0 = all)",
    )
    args = parser.parse_args()

    if not args.meta_path.exists():
        raise SystemExit(f"Meta file not found: {args.meta_path}")

    df = pd.read_csv(args.meta_path)
    if args.limit > 0:
        df = df.head(args.limit)

    rows: List[Dict[str, float]] = []

    for idx, row in df.iterrows():
        noisy_path = Path(row["noisy_path"])
        if not noisy_path.exists():
            print(f"[warn] noisy file missing: {noisy_path}")
            continue

        audio = load_audio(noisy_path, args.target_sr)
        feats = compute_features(
            audio,
            sr=args.target_sr,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )

        feature_row: Dict[str, float] = {
            "utt_id": row["utt_id"],
            "scene": row.get("scene", ""),
            "macro_scene": row.get("macro_scene", ""),
            "snr_db": row.get("snr_db", ""),
            "clean_path": row["clean_path"],
            "noisy_path": row["noisy_path"],
        }
        feature_row.update(feats)

        rows.append(feature_row)

        if (idx + 1) % 50 == 0:
            print(f"[feat] processed {idx + 1} / {len(df)} rows", flush=True)

    if not rows:
        raise SystemExit("No features were extracted. Check input paths.")

    write_rows(args.output_path, rows)
    print(f"[feat] wrote {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()

