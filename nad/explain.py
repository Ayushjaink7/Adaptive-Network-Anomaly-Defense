"""
SHAP explanations for the Isolation Forest (optional; needs `pip install shap`).

The detector is Pipeline([StandardScaler, IsolationForest]). TreeExplainer runs on the
IsolationForest step with the *scaled* features. The scaler works on each feature
separately, so every SHAP value still belongs to the same named feature.

SHAP explains the forest's average isolation path length:
    base value + sum(SHAP values) = E[path length]
A NEGATIVE SHAP value means a shorter path, i.e. the feature pushed the flow TOWARD "anomaly".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ShapExplainer:
    def __init__(self, pipeline_if, feature_names: list[str]):
        import shap  # imported here so the rest of the project works without shap

        self.feature_names = list(feature_names)
        self.scaler = pipeline_if.named_steps["scaler"]
        self.explainer = shap.TreeExplainer(pipeline_if.named_steps["clf"])
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])
        self._cache: dict[tuple, list[dict[str, Any]]] = {}

    @classmethod
    def try_create(cls, pipeline_if, feature_names: list[str]) -> "ShapExplainer | None":
        """Return an explainer, or None (with one warning) if shap or the IF model is unavailable."""
        if pipeline_if is None:
            return None
        try:
            return cls(pipeline_if, feature_names)
        except Exception as e:  # ImportError if shap is not installed
            print(f"[shap] explanations disabled: {e}")
            return None

    def top_k(self, x_row: pd.Series, k: int = 3) -> list[dict[str, Any]]:
        """Top-k features pushing this flow toward anomaly (most negative SHAP first)."""
        x = x_row.reindex(self.feature_names, fill_value=0).astype(float)
        key = tuple(np.round(x.values, 9))
        if key in self._cache:  # flood flows are often identical, so compute each unique flow once
            return self._cache[key]
        xs = self.scaler.transform(pd.DataFrame([x.values], columns=self.feature_names))
        sv = np.asarray(self.explainer.shap_values(xs))[0]
        order = np.argsort(sv)[:k]
        out = [
            {"feature": self.feature_names[i], "value": float(x.values[i]), "shap": round(float(sv[i]), 4)}
            for i in order
            if sv[i] < 0
        ]
        self._cache[key] = out
        return out


def format_top_k(items: list[dict[str, Any]]) -> str:
    """'packets_per_sec=1e+06 (-2.76); ...' for the CSV column."""
    return "; ".join(f"{d['feature']}={d['value']:.4g} ({d['shap']:+.2f})" for d in items)
