from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DefenseConfig


@dataclass
class StatisticalBaseline:
    """Per-feature mean/std from normal traffic; used for explanations + hybrid rules."""

    means: dict[str, float]
    stds: dict[str, float]
    feature_names: list[str]

    def zscore_row(self, row: pd.Series) -> pd.Series:
        z = {}
        for c in self.feature_names:
            if c not in row.index:
                continue
            mu = self.means.get(c, 0.0)
            sigma = self.stds.get(c, 1.0) or 1.0
            z[c] = abs(float(row[c]) - mu) / sigma
        return pd.Series(z)

    def max_zscore(self, row: pd.Series) -> float:
        zs = self.zscore_row(row)
        return float(zs.max()) if len(zs) else 0.0

    def to_json(self, path: str | Path) -> None:
        payload = {"means": self.means, "stds": self.stds, "feature_names": self.feature_names}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> StatisticalBaseline:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            means={k: float(v) for k, v in payload["means"].items()},
            stds={k: float(v) for k, v in payload["stds"].items()},
            feature_names=list(payload["feature_names"]),
        )


def fit_baseline(X: pd.DataFrame, cfg: DefenseConfig | None = None) -> StatisticalBaseline:
    cfg = cfg or DefenseConfig()
    feature_names = list(X.columns)
    means = {c: float(np.mean(X[c].values)) for c in feature_names}
    stds = {c: float(np.std(X[c].values)) or 1e-6 for c in feature_names}
    return StatisticalBaseline(means=means, stds=stds, feature_names=feature_names)


def baseline_anomaly_mask(X: pd.DataFrame, baseline: StatisticalBaseline, cfg: DefenseConfig) -> pd.Series:
    """Row is flagged if any feature exceeds z-score threshold vs baseline."""
    flags = []
    for _, row in X.iterrows():
        flags.append(baseline.max_zscore(row) > cfg.zscore_threshold)
    return pd.Series(flags, index=X.index)
