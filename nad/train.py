from __future__ import annotations

import argparse
from pathlib import Path

from .config import DefenseConfig
from .pipeline import train_from_normal_pcaps


def main() -> None:
    p = argparse.ArgumentParser(description="Train baseline + anomaly model on NORMAL pcaps only.")
    p.add_argument("--normal-pcap", action="append", required=True, help="Repeat for multiple baseline captures.")
    p.add_argument("--out", required=True, help="Artifact directory (created).")
    p.add_argument("--subnet", default="192.168.64.0/24", help="Monitored subnet on defense node.")
    p.add_argument(
        "--model",
        default="iforest",
        choices=["iforest", "ocsvm", "ensemble"],
        help="Isolation Forest, One-Class SVM, or both (vote).",
    )
    args = p.parse_args()

    cfg = DefenseConfig(monitored_subnet=args.subnet)
    art = train_from_normal_pcaps(args.normal_pcap, args.out, cfg=cfg, model_kind=args.model)  # type: ignore[arg-type]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print("Saved artifacts to", args.out)
    print("Features:", len(art.feature_names))


if __name__ == "__main__":
    main()
