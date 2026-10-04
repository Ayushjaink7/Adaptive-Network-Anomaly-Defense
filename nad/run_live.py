from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import load_artifacts, score_pcap


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detection pipeline with full visibility")
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--subnet", default=None)
    parser.add_argument("--log", default="defense_log.json")
    parser.add_argument("--enforce", action="store_true")

    args = parser.parse_args()

    print("\n=== RUN STARTED ===")

    artifacts = load_artifacts(args.artifacts, monitored_subnet=args.subnet)
    log_path = Path(args.log) if args.log else None

    df = score_pcap(
        args.pcap,
        artifacts,
        log_path=log_path,
        enforce=args.enforce
    )

    # =========================
    # SAVE CSV
    # =========================
    df.to_csv("output.csv", index=False)
    print("\n✅ Saved CSV → output.csv")

    # =========================
    # SHOW SCORES CLEARLY
    # =========================
    print("\n========== SAMPLE (WITH SCORES) ==========")
    print(df[["src_ip", "dst_ip", "dst_port", "risk_score", "action"]].head(20))

    # =========================
    # SUMMARY
    # =========================
    print("\n========== SUMMARY ==========")
    print("Total flows:", len(df))
    print("BLOCK:", (df["action"] == "BLOCK").sum())
    print("RATE_LIMIT:", (df["action"] == "RATE_LIMIT").sum())
    print("LOG:", (df["action"] == "LOG").sum())

    # =========================
    # SCORE DISTRIBUTION
    # =========================
    print("\n========== SCORE DISTRIBUTION ==========")
    print(df["risk_score"].describe())

    # =========================
    # SCORE BY ACTION
    # =========================
    print("\n========== SCORES BY ACTION ==========")
    print(df.groupby("action")["risk_score"].describe())

    # =========================
    # GROUPED ANALYSIS
    # =========================
    print("\n========== GROUPED ANALYSIS ==========")
    print(df.groupby(["src_ip", "dst_port", "action"]).size())

    # =========================
    # TOP RISK FLOWS
    # =========================
    print("\n========== TOP HIGH RISK FLOWS ==========")
    print(df.sort_values("risk_score", ascending=False)[
        ["src_ip", "dst_port", "risk_score", "action"]
    ].head(10))

    print("\n=== RUN COMPLETE ===")


if __name__ == "__main__":
    main()

