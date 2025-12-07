"""Utilities for loading feature tables for RL training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureDataset:
    features: np.ndarray
    snr_targets: np.ndarray
    meta: pd.DataFrame

    @property
    def num_samples(self) -> int:
        return len(self.features)


def load_feature_csv(
    csv_path: Path,
    feature_cols: Sequence[str] | None = None,
) -> Tuple[FeatureDataset, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    snr = df["snr_db"].astype(float).to_numpy()

    # Automatically pick numeric feature columns if not provided
    if feature_cols is None:
        exclude = {"utt_id", "scene", "macro_scene", "snr_db", "noisy_path", "clean_path"}
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude and np.issubdtype(df[col].dtype, np.number)
        ]
    feats = df[feature_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)

    dataset = FeatureDataset(
        features=feats,
        snr_targets=snr,
        meta=df[["utt_id", "scene", "macro_scene", "noisy_path", "clean_path"]],
    )
    return dataset, list(feature_cols)


def split_dataset(
    dataset: FeatureDataset,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[FeatureDataset, FeatureDataset]:
    idx_train, idx_val = train_test_split(
        np.arange(dataset.num_samples),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )

    def subset(idxs: np.ndarray) -> FeatureDataset:
        return FeatureDataset(
            features=dataset.features[idxs],
            snr_targets=dataset.snr_targets[idxs],
            meta=dataset.meta.iloc[idxs].reset_index(drop=True),
        )

    return subset(idx_train), subset(idx_val)

