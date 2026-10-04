from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class SecureLogger:
    """
    Append-only JSONL log with hash chain for non-repudiation / integrity verification.
    Each line: {payload..., prev_hash, line_hash}
    """

    path: Path
    _last_hash: str = ""

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._last_hash = self._tail_hash()

    def _tail_hash(self) -> str:
        last = ""
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    last = str(obj.get("line_hash", ""))
                except json.JSONDecodeError:
                    continue
        return last

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        record = dict(event)
        record["ts_utc"] = _utc_iso()
        record["prev_hash"] = self._last_hash
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record["line_hash"] = sha256_text(self._last_hash + "|" + canonical)
        line = json.dumps(record, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
        self._last_hash = record["line_hash"]
        return record

    @staticmethod
    def verify_file(log_path: str | Path) -> tuple[bool, str]:
        p = Path(log_path)
        if not p.exists():
            return False, "missing file"
        prev_chain = ""
        with p.open("r", encoding="utf-8") as f:
            for i, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    return False, f"line {i}: invalid json ({e})"
                lh = obj.get("line_hash")
                if not lh:
                    return False, f"line {i}: missing line_hash"
                if obj.get("prev_hash", "") != prev_chain:
                    return False, f"line {i}: prev_hash chain broken"
                body = {k: v for k, v in obj.items() if k != "line_hash"}
                canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
                expect = sha256_text(prev_chain + "|" + canonical)
                if expect != lh:
                    return False, f"line {i}: hash mismatch (tamper or corruption)"
                prev_chain = lh
        return True, "ok"
