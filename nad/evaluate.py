from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ClassificationMetrics:
    accuracy: float
    fpr: float
    fnr: float
    precision: float
    recall: float
    tp: int
    fp: int
    tn: int
    fn: int


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> ClassificationMetrics:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    acc = (tp + tn) / max(len(yt), 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return ClassificationMetrics(
        accuracy=float(acc),
        fpr=float(fpr),
        fnr=float(fnr),
        precision=float(prec),
        recall=float(rec),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def detection_latency_sec(
    flows: pd.DataFrame,
    y_true_attack: pd.Series,
    y_pred_attack: pd.Series,
    time_col: str = "flow_start_ts",
) -> float | None:
    """
    Approximate detection delay: time from first true attack flow start to first predicted attack.
    Requires flow_start_ts column; returns None if unavailable.
    """
    if time_col not in flows.columns:
        return None
    true_idx = flows.loc[y_true_attack.astype(bool)].index
    pred_idx = flows.loc[y_pred_attack.astype(bool)].index
    if len(true_idx) == 0 or len(pred_idx) == 0:
        return None
    t0 = float(flows.loc[true_idx, time_col].min())
    t1 = float(flows.loc[pred_idx, time_col].min())
    return max(t1 - t0, 0.0)


def label_flows_by_attack_ip(df: pd.DataFrame, attacker_ips: set[str]) -> pd.Series:
    return df["src_ip"].isin(attacker_ips).astype(int)


def label_flows_by_time_window(df: pd.DataFrame, start: float, end: float, ts_col: str) -> pd.Series:
    return ((df[ts_col] >= start) & (df[ts_col] <= end)).astype(int)
