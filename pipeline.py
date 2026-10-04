```python
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .baseline import StatisticalBaseline, baseline_anomaly_mask, fit_baseline
from .config import DefenseConfig
from .detector import AnomalyDetectorBundle, ModelKind, train_detector
from .features import (
    augment_security_principle_flags,
    extract_flow_records_from_pcap,
    featurize_for_ml,
    label_bruteforce_by_ssh_frequency,
)
from .firewall import FirewallAutomation
from .risk import RiskScorer
from .secure_log import SecureLogger


@dataclass
class TrainedArtifacts:
    cfg: DefenseConfig
    baseline: StatisticalBaseline
    detector: AnomalyDetectorBundle
    feature_names: list[str]


def build_labeled_frame(pcap_path: str, cfg: DefenseConfig) -> pd.DataFrame:
    df = extract_flow_records_from_pcap(pcap_path, cfg)
    df = augment_security_principle_flags(df, cfg)
    df["authentication_bruteforce_hint"] = label_bruteforce_by_ssh_frequency(df, cfg)
    return df


def train_from_normal_pcaps(
    pcap_paths: list[str],
    out_dir: str | Path,
    cfg: DefenseConfig | None = None,
    model_kind: ModelKind = "iforest",
) -> TrainedArtifacts:
    cfg = cfg or DefenseConfig()

    frames = [build_labeled_frame(p, cfg) for p in pcap_paths]
    df = pd.concat(frames, ignore_index=True)

    X, names = featurize_for_ml(df)
    baseline = fit_baseline(X, cfg)
    detector = train_detector(X, cfg, kind=model_kind)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline.to_json(out / "baseline.json")
    detector.save(out / "detector")
    pd.Series(names).to_csv(out / "feature_names.csv", index=False, header=["name"])

    (out / "config.json").write_text(json.dumps({}, indent=2))

    return TrainedArtifacts(cfg=cfg, baseline=baseline, detector=detector, feature_names=names)


def load_artifacts(artifact_dir: str | Path, monitored_subnet: str | None = None) -> TrainedArtifacts:
    d = Path(artifact_dir)

    base_cfg = DefenseConfig()
    p_cfg = d / "config.json"

    if p_cfg.exists():
        raw = json.loads(p_cfg.read_text(encoding="utf-8"))
        base_cfg = DefenseConfig(
            monitored_subnet=str(raw.get("monitored_subnet", base_cfg.monitored_subnet)),
            flow_idle_timeout_sec=float(raw.get("flow_idle_timeout_sec", base_cfg.flow_idle_timeout_sec)),
            zscore_threshold=float(raw.get("zscore_threshold", base_cfg.zscore_threshold)),
            if_contamination=float(raw.get("if_contamination", base_cfg.if_contamination)),
            risk_low_max=float(raw.get("risk_low_max", base_cfg.risk_low_max)),
            risk_medium_max=float(raw.get("risk_medium_max", base_cfg.risk_medium_max)),
            ssh_bruteforce_flow_threshold=int(
                raw.get("ssh_bruteforce_flow_threshold", base_cfg.ssh_bruteforce_flow_threshold)
            ),
            syn_flood_syn_ratio=float(raw.get("syn_flood_syn_ratio", base_cfg.syn_flood_syn_ratio)),
            syn_flood_min_packets=int(raw.get("syn_flood_min_packets", base_cfg.syn_flood_min_packets)),
        )

    cfg = replace(base_cfg, monitored_subnet=monitored_subnet) if monitored_subnet else base_cfg

    baseline = StatisticalBaseline.from_json(d / "baseline.json")
    detector = AnomalyDetectorBundle.load(d / "detector")
    names = pd.read_csv(d / "feature_names.csv")["name"].astype(str).tolist()

    return TrainedArtifacts(cfg=cfg, baseline=baseline, detector=detector, feature_names=names)


def score_pcap(
    pcap_path: str,
    artifacts: TrainedArtifacts,
    log_path: Path | None = None,
    enforce: bool = False,
) -> pd.DataFrame:

    print("\n========== STARTING ANALYSIS ==========")
    print("Processing PCAP:", pcap_path)

    df = build_labeled_frame(pcap_path, artifacts.cfg)
    X, _ = featurize_for_ml(df)
    X = X.reindex(columns=artifacts.feature_names, fill_value=0)

    scores = artifacts.detector.predict_scores(X)
    ml_anom = artifacts.detector.combined_anomaly_label(scores)

    stat_mask = baseline_anomaly_mask(X, artifacts.baseline, artifacts.cfg)
    scorer = RiskScorer(artifacts.cfg)

    decisions: list[dict[str, Any]] = []

    fw = FirewallAutomation(artifacts.cfg) if enforce else None
    logger = SecureLogger(log_path) if log_path else None

    enforced_block: set[str] = set()
    enforced_limit: set[str] = set()

    for idx, row in df.iterrows():
        z = artifacts.baseline.max_zscore(X.loc[idx])
        rd = scorer.score_row(row, bool(ml_anom.loc[idx]), z)

        decisions.append(
            {
                "flow_index": int(idx),
                "src_ip": row["src_ip"],
                "dst_ip": row["dst_ip"],
                "dst_port": int(row["dst_port"]),
                "service_bucket": row["service_bucket"],
                "ml_anomaly": bool(ml_anom.loc[idx]),
                "baseline_z_max": z,
                "risk_score": rd.score,
                "action": rd.action,
                "reasons": ",".join(rd.reasons),
            }
        )

        if enforce and fw is not None:
            sip = str(row["src_ip"])

            if rd.action == "BLOCK" and sip not in enforced_block:
                r = fw.block_source(sip)
                print(f"[iptables] BLOCK {sip} ok={r.ok}")
                enforced_block.add(sip)

            elif rd.action == "RATE_LIMIT" and sip not in enforced_limit:
                rs = fw.rate_limit_source(sip)
                ok = all(x.ok for x in rs if x.command != "(ensure_rule)")
                print(f"[iptables] RATE_LIMIT {sip} ok={ok}")
                enforced_limit.add(sip)

    df_out = pd.DataFrame(decisions)

    print("\n========== DETECTION RESULTS ==========")
    print(df_out.head(20))

    print("\nTotal flows:", len(df_out))

    if "ml_anomaly" in df_out.columns:
        print("Anomalies detected:", df_out["ml_anomaly"].sum())

    if "action" in df_out.columns:
        print("\nAction summary:")
        print(df_out["action"].value_counts())

    print("\n========== ANALYSIS COMPLETE ==========\n")

    return df_out


# ================= MAIN ================= #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Network Anomaly Detection Pipeline")
    parser.add_argument("pcap", help="Path to PCAP file")
    parser.add_argument("--model", default="model.pkl", help="Path to trained model directory")
    parser.add_argument("--log", default=None, help="Optional log file")
    parser.add_argument("--enforce", action="store_true", help="Enable firewall enforcement")

    args = parser.parse_args()

    print("\n[DEBUG] Pipeline execution started")

    artifacts = load_artifacts(args.model)

    df = score_pcap(
        pcap_path=args.pcap,
        artifacts=artifacts,
        log_path=Path(args.log) if args.log else None,
        enforce=args.enforce,
    )

    print("\n========== FINAL OUTPUT ==========")
    print(df.head(20))


if __name__ == "__main__":
    main()
```
