from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass
class DefenseConfig:
    """Tunable parameters for capture, ML, risk, and enforcement."""

    # Network context (set to your Ubuntu defense NIC / monitored subnet)
    monitored_subnet: str = "192.168.64.0/24"

    # Flow aggregation: split flow if gap between packets exceeds this (seconds)
    flow_idle_timeout_sec: float = 120.0

    # Statistical baseline: z-score beyond this marks "statistical anomaly" feature
    zscore_threshold: float = 3.0

    # Isolation Forest (unsupervised; tune contamination to expected attack ratio in train set)
    if_contamination: float = 0.05
    if_n_estimators: int = 200
    if_random_state: int = 42

    # One-Class SVM (optional second model); nu ~ upper bound on training outliers
    ocsvm_nu: float = 0.05
    ocsvm_gamma: str = "scale"

    # Risk thresholds (0-100 scale from RiskScorer)
    risk_low_max: float = 35.0
    risk_medium_max: float = 70.0

    # Protocol sensitivity multipliers (applied in risk engine)
    protocol_weights: dict[str, float] = field(
        default_factory=lambda: {
            "SSH": 1.4,
            "HTTP": 1.2,
            "HTTPS": 1.1,
            "TCP_OTHER": 1.0,
            "UDP": 0.9,
            "ICMP": 1.0,
            "OTHER": 1.0,
        }
    )

    # Brute-force heuristic: many distinct flows to port 22 from same src in window
    ssh_bruteforce_flow_threshold: int = 15

    # DoS heuristic: SYN-heavy flow
    syn_flood_syn_ratio: float = 0.85
    syn_flood_min_packets: int = 30

    # Integrity: flag combinations considered suspicious (educational heuristics)
    suspicious_tcp_flag_sets: FrozenSet[str] = field(
        default_factory=lambda: frozenset(
            {
                "FIN,PSH,URG",
                "NULL",
                "FIN",
                "URG,PSH",
            }
        )
    )

    # iptables chain name (dedicated chain keeps rules organized)
    iptables_chain: str = "NAD_DEFENSE"
