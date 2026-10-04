from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .config import DefenseConfig

EnforcementAction = Literal["LOG", "RATE_LIMIT", "BLOCK"]


@dataclass
class RiskDecision:
    score: float
    action: EnforcementAction
    reasons: list[str]


class RiskScorer:
    """
    Maps ML/statistical/heuristic signals into 0-100 risk and discrete actions.
    """

    def __init__(self, cfg: DefenseConfig | None = None):
        self.cfg = cfg or DefenseConfig()

    def score_row(self, row: pd.Series, ml_anomaly: bool, baseline_z: float) -> RiskDecision:
        reasons: list[str] = []
        score = 0.0

        if ml_anomaly:
            score += 40.0
            reasons.append("ml_anomaly")

        if baseline_z > self.cfg.zscore_threshold:
            score += min(30.0, (baseline_z - self.cfg.zscore_threshold) * 5)
            reasons.append("statistical_deviation")

        svc = row.get("service_bucket", "OTHER")
        w = self.cfg.protocol_weights.get(str(svc), 1.0)
        score *= w
        reasons.append(f"protocol_weight={w}")

        if int(row.get("integrity_tcp_anomaly", 0)) == 1:
            score += 15.0
            reasons.append("tcp_flag_integrity")

        if int(row.get("availability_syn_flood_pattern", 0)) == 1:
            score += 25.0
            reasons.append("syn_flood_availability")

        if int(row.get("confidentiality_plainhttp_to_server", 0)) == 1:
            score += 5.0
            reasons.append("plaintext_http_confidentiality")

        if int(row.get("authentication_bruteforce_hint", 0)) == 1:
            score += 20.0
            reasons.append("ssh_bruteforce_authentication")

        # Demo-strength reliability: certain attack patterns should escalate to BLOCK
        # even if the unsupervised model is conservative.
        if int(row.get("availability_syn_flood_pattern", 0)) == 1:
            score = max(score, 85.0)
            reasons.append("force_block_floor_syn_flood")
        if int(row.get("authentication_bruteforce_hint", 0)) == 1 and int(row.get("dst_port", 0)) == 22:
            score = max(score, 75.0)
            reasons.append("force_rate_or_block_floor_ssh_bruteforce")

        score = float(max(0.0, min(100.0, score)))

        if score <= self.cfg.risk_low_max:
            action: EnforcementAction = "LOG"
        elif score <= self.cfg.risk_medium_max:
            action = "RATE_LIMIT"
        else:
            action = "BLOCK"

        return RiskDecision(score=score, action=action, reasons=reasons)
