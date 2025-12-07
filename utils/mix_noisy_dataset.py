"""Generate a noisy speech dataset by mixing clean KsponSpeech clips with ESC-50 noise."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

EPS = 1e-9


def _normalize_column(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("\n", " ")
        .replace("_", " ")
    )


def load_noise_label_map(meta_path: Optional[Path]) -> Dict[str, str]:
    """Map ESC-50 recording IDs (e.g., '1-100210-A') to ground-truth labels."""
    if meta_path is None or not meta_path.exists():
        return {}

    df = pd.read_excel(meta_path)
    col_map = { _normalize_column(col): col for col in df.columns }
    recording_col = col_map.get("recording")
    label_col = col_map.get("ground truth") or col_map.get("groundtruth")

    if recording_col is None or label_col is None:
        return {}

    mapping: Dict[str, str] = {}
    for rec, label in zip(df[recording_col], df[label_col]):
        if pd.isna(rec) or pd.isna(label):
            continue
        rec_id = str(rec).strip()
        if rec_id.endswith(".ogg"):
            rec_id = rec_id[:-4]
        mapping[rec_id] = str(label).strip()
    return mapping


def load_scene_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize keys for robust lookup
    return { key.strip().lower(): value for key, value in data.items() }


def list_audio_files(root: Path, exts: Sequence[str] = (".wav", ".flac")) -> List[Path]:
    return sorted(
        path
        for ext in exts
        for path in root.rglob(f"*{ext}")
        if path.is_file()
    )


def read_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def align_noise(noise: np.ndarray, target_len: int, rng: random.Random) -> np.ndarray:
    if len(noise) == 0:
        raise ValueError("Noise clip is empty")

    if len(noise) >= target_len:
        max_start = len(noise) - target_len
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return noise[start:start + target_len]

    aligned = np.zeros(target_len, dtype=np.float32)
    segments = rng.randint(1, 4)  # place noise 1~3 times
    for _ in range(segments):
        chunk_len = min(len(noise), rng.randint(max(1, len(noise) // 2), len(noise)))
        chunk_start = rng.randint(0, max(0, len(noise) - chunk_len))
        chunk = noise[chunk_start:chunk_start + chunk_len]
        start = rng.randint(0, max(0, target_len - chunk_len))
        aligned[start:start + chunk_len] += chunk

    if np.allclose(aligned, 0):
        reps = int(np.ceil(target_len / len(noise)))
        tiled = np.tile(noise, reps)
        return tiled[:target_len]

    return aligned


def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_power = np.mean(clean ** 2) + EPS
    noise_power = np.mean(noise ** 2) + EPS
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    noisy = clean + noise * scale
    noisy = np.clip(noisy, -1.0, 1.0)
    return noisy.astype(np.float32)


def build_scene_label(noise_path: Path, label_map: Dict[str, str]) -> str:
    if not label_map:
        return ""
    parts = noise_path.stem.split("-")
    if len(parts) < 3:
        return ""
    key = "-".join(parts[:3])
    return label_map.get(key, "")


def build_macro_label(scene_label: str, scene_map: Dict[str, str]) -> str:
    if not scene_label:
        return ""
    return scene_map.get(scene_label.lower(), scene_label)


def write_meta(meta_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "utt_id",
        "split",
        "clean_path",
        "noisy_path",
        "clean_source",
        "noise_source",
        "scene",
        "macro_scene",
        "snr_db",
        "duration_sec",
        "speaker_id",
    ]

    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix clean speech with ESC-50 noise.")
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("KsponSpeech_01"),
        help="KsponSpeech clean 데이터 루트 (default: %(default)s)",
    )
    parser.add_argument(
        "--noise-root",
        type=Path,
        default=Path("noise_select"),
        help="ESC-50 noise 데이터 루트 (default: %(default)s)",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data_mixed"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--snr-db",
        type=str,
        default="0,5,10",
        help="Comma-separated SNR values in dB (default: %(default)s)",
    )
    parser.add_argument(
        "--clean-limit",
        type=int,
        default=0,
        help="Optional limit on number of clean files (0 = no limit)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument(
        "--esc50-meta",
        type=Path,
        default=Path("esc50-meta.xlsx"),
        help="Path to esc50-meta.xlsx for scene labels (default: %(default)s)",
    )
    parser.add_argument(
        "--scene-map",
        type=Path,
        default=None,
        help="JSON file mapping ground-truth labels to macro scenes (optional)",
    )
    parser.add_argument(
        "--noise-fallback",
        type=Path,
        default=Path("noise"),
        help="noise-root이 없으면 대체로 사용할 경로 (default: %(default)s)",
    )
    args = parser.parse_args()

    clean_root = args.clean_root
    if not clean_root.exists():
        raise SystemExit(f"Clean root not found: {clean_root}")

    noise_root = args.noise_root
    if not noise_root.exists():
        if args.noise_fallback and args.noise_fallback.exists():
            noise_root = args.noise_fallback
            print(f"[mix] noise-root missing, fallback -> {noise_root}")
        else:
            raise SystemExit(f"Noise root not found: {noise_root}")

    rng = random.Random(args.seed)

    clean_files = list_audio_files(clean_root, exts=(".wav",))
    if not clean_files:
        raise SystemExit(f"No clean .wav files found under {clean_root}")

    noise_files = list_audio_files(noise_root, exts=(".wav",))
    if not noise_files:
        raise SystemExit(f"No noise .wav files found under {noise_root}")

    if args.clean_limit > 0:
        clean_files = clean_files[: args.clean_limit]

    snr_values = [float(v.strip()) for v in args.snr_db.split(",") if v.strip()]
    if not snr_values:
        raise SystemExit("At least one SNR value must be specified")

    label_map = load_noise_label_map(args.esc50_meta)
    scene_map = load_scene_map(args.scene_map)

    split_dir = args.output_root / args.split
    clean_out_dir = split_dir / "clean"
    noisy_out_dir = split_dir / "noisy"
    clean_out_dir.mkdir(parents=True, exist_ok=True)
    noisy_out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: List[Dict[str, str]] = []

    for idx, clean_path in enumerate(clean_files):
        clean_audio = read_audio_mono(clean_path, args.target_sr)
        if len(clean_audio) == 0:
            continue

        noise_path = rng.choice(noise_files)
        noise_audio = read_audio_mono(noise_path, args.target_sr)
        noise_audio = align_noise(noise_audio, len(clean_audio), rng)

        snr_db = rng.choice(snr_values)
        noisy_audio = mix_with_snr(clean_audio, noise_audio, snr_db)

        utt_id = f"{clean_path.stem}_mix{idx:05d}_snr{int(snr_db)}"
        clean_out_path = clean_out_dir / f"{utt_id}.wav"
        noisy_out_path = noisy_out_dir / f"{utt_id}.wav"

        sf.write(clean_out_path, np.clip(clean_audio, -1.0, 1.0), args.target_sr, subtype="PCM_16")
        sf.write(noisy_out_path, noisy_audio, args.target_sr, subtype="PCM_16")

        scene_label = build_scene_label(noise_path, label_map)
        macro_scene = build_macro_label(scene_label, scene_map)
        duration_sec = len(clean_audio) / args.target_sr
        speaker_id = clean_path.parent.name

        meta_rows.append(
            {
                "utt_id": utt_id,
                "split": args.split,
                "clean_path": str(clean_out_path),
                "noisy_path": str(noisy_out_path),
                "clean_source": str(clean_path),
                "noise_source": str(noise_path),
                "scene": scene_label,
                "macro_scene": macro_scene,
                "snr_db": f"{snr_db:.1f}",
                "duration_sec": f"{duration_sec:.3f}",
                "speaker_id": speaker_id,
            }
        )

        if (idx + 1) % 50 == 0:
            print(f"[mix] processed {idx + 1} / {len(clean_files)} clean files", flush=True)

    meta_path = split_dir / "meta.csv"
    write_meta(meta_path, meta_rows)
    print(
        f"Done. Wrote {len(meta_rows)} examples to {split_dir} "
        f"(meta: {meta_path})"
    )


if __name__ == "__main__":
    main()

