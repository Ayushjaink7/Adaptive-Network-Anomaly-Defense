from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .config import DefenseConfig


ModelKind = Literal["iforest", "ocsvm", "ensemble"]


@dataclass
class AnomalyDetectorBundle:
    kind: ModelKind
    feature_names: list[str]
    pipeline_if: Pipeline | None = None
    pipeline_svm: Pipeline | None = None

    def predict_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        Xa = X.reindex(columns=self.feature_names, fill_value=0)
        out = pd.DataFrame(index=X.index)
        if self.pipeline_if is not None:
            # decision_function: higher = more normal for IF in sklearn
            out["if_score"] = self.pipeline_if.decision_function(Xa)
            out["if_label"] = self.pipeline_if.predict(Xa)
        if self.pipeline_svm is not None:
            out["svm_score"] = self.pipeline_svm.decision_function(Xa)
            out["svm_label"] = self.pipeline_svm.predict(Xa)
        return out

    def combined_anomaly_label(self, scores: pd.DataFrame) -> pd.Series:
        votes = []
        if "if_label" in scores.columns:
            votes.append(scores["if_label"].eq(-1))
        if "svm_label" in scores.columns:
            votes.append(scores["svm_label"].eq(-1))
        if not votes:
            return pd.Series(False, index=scores.index)
        stacked = pd.concat(votes, axis=1)
        # Majority vote; tie -> anomaly if any says anomaly
        return stacked.sum(axis=1) >= max(1, len(votes) // 2)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"kind": self.kind, "feature_names": self.feature_names}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self.pipeline_if is not None:
            joblib.dump(self.pipeline_if, d / "iforest.joblib")
        if self.pipeline_svm is not None:
            joblib.dump(self.pipeline_svm, d / "ocsvm.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> AnomalyDetectorBundle:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        kind = meta["kind"]
        fn = list(meta["feature_names"])
        p_if = joblib.load(d / "iforest.joblib") if (d / "iforest.joblib").exists() else None
        p_svm = joblib.load(d / "ocsvm.joblib") if (d / "ocsvm.joblib").exists() else None
        return cls(kind=kind, feature_names=fn, pipeline_if=p_if, pipeline_svm=p_svm)


def build_iforest(cfg: DefenseConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                IsolationForest(
                    n_estimators=cfg.if_n_estimators,
                    contamination=cfg.if_contamination,
                    random_state=cfg.if_random_state,
                ),
            ),
        ]
    )


def build_ocsvm(cfg: DefenseConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", OneClassSVM(kernel="rbf", nu=cfg.ocsvm_nu, gamma=cfg.ocsvm_gamma)),
        ]
    )


def train_detector(
    X: pd.DataFrame,
    cfg: DefenseConfig | None = None,
    kind: ModelKind = "iforest",
) -> AnomalyDetectorBundle:
    cfg = cfg or DefenseConfig()
    feature_names = list(X.columns)
    Xa = X.fillna(0).astype(np.float64)

    bundle = AnomalyDetectorBundle(kind=kind, feature_names=feature_names)
    if kind in ("iforest", "ensemble"):
        p_if = build_iforest(cfg)
        p_if.fit(Xa)
        bundle.pipeline_if = p_if
    if kind in ("ocsvm", "ensemble"):
        p_svm = build_ocsvm(cfg)
        p_svm.fit(Xa)
        bundle.pipeline_svm = p_svm
    return bundle
