from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass

from .config import DefenseConfig


@dataclass
class IptablesResult:
    ok: bool
    command: str
    stdout: str
    stderr: str
    returncode: int


class FirewallAutomation:
    """
    Wraps iptables with a dedicated chain. Requires passwordless sudo or running as root.
    """

    def __init__(
        self,
        cfg: DefenseConfig | None = None,
        sudo: bool = True,
        iptables_path: str = "/usr/sbin/iptables",
        wait_seconds: int = 2,
        debug: bool = True,
    ):
        self.cfg = cfg or DefenseConfig()
        self.sudo = sudo
        self.iptables_path = iptables_path
        self.wait_seconds = wait_seconds
        self.debug = debug
        self._prefix = ["sudo", "-n", iptables_path] if sudo else [iptables_path]

    def _run(self, args: list[str]) -> IptablesResult:
        # -w: wait for xtables lock (avoids sporadic failures under concurrent activity)
        cmd = self._prefix + ["-w", str(self.wait_seconds)] + args
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return IptablesResult(
                ok=proc.returncode == 0,
                command=cmd_str,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                returncode=proc.returncode,
            )
        except FileNotFoundError:
            return IptablesResult(
                ok=False,
                command=cmd_str,
                stdout="",
                stderr="iptables not found; run on Linux defense node with netfilter.",
                returncode=127,
            )

    def _sudo_ready(self) -> IptablesResult:
        if not self.sudo:
            if os.geteuid() != 0:
                return IptablesResult(
                    ok=False,
                    command="(euid-check)",
                    stdout="",
                    stderr="Not running as root and sudo=False. Run as root or enable sudo.",
                    returncode=1,
                )
            return IptablesResult(ok=True, command="(euid-check)", stdout="", stderr="", returncode=0)
        # sudo -n will fail fast if it would prompt
        try:
            proc = subprocess.run(["sudo", "-n", "true"], capture_output=True, text=True, check=False)
            return IptablesResult(
                ok=proc.returncode == 0,
                command="sudo -n true",
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                returncode=proc.returncode,
            )
        except FileNotFoundError:
            return IptablesResult(ok=False, command="sudo -n true", stdout="", stderr="sudo not found", returncode=127)

    def _chain_exists(self, chain: str) -> bool:
        r = self._run(["-S", chain])
        return r.ok

    def _ensure_jump(self, from_chain: str, to_chain: str) -> list[IptablesResult]:
        results: list[IptablesResult] = []
        chk = self._run(["-C", from_chain, "-j", to_chain])
        results.append(chk)
        if not chk.ok:
            results.append(self._run(["-I", from_chain, "1", "-j", to_chain]))
        return results

    def _ensure_rule(self, rule_args: list[str], insert: bool = True) -> list[IptablesResult]:
        """
        Idempotent rule add:
        - First try `-C ...` for exact match.
        - If missing, add with `-I` (higher priority) or `-A`.
        """
        results: list[IptablesResult] = []
        # Convert first token -A/-I into -C for checking
        if not rule_args or rule_args[0] not in {"-A", "-I"}:
            return [IptablesResult(False, "(ensure_rule)", "", f"bad rule_args: {rule_args}", 2)]
        check = ["-C"] + rule_args[1:]
        r0 = self._run(check)
        results.append(r0)
        if r0.ok:
            return results
        add = rule_args[:]
        if insert and add[0] == "-A":
            add[0] = "-I"
            # insert at top of target chain
            add.insert(2, "1")  # after chain name
        results.append(self._run(add))
        return results

    def ensure_chain(self, flush: bool = False) -> list[IptablesResult]:
        """Create NAD chain and jump from INPUT. Optionally flush rules (flush=False by default)."""
        chain = self.cfg.iptables_chain
        results: list[IptablesResult] = []

        results.append(self._sudo_ready())
        if not results[-1].ok:
            return results

        if not self._chain_exists(chain):
            results.append(self._run(["-N", chain]))
        if flush:
            results.append(self._run(["-F", chain]))
        results.extend(self._ensure_jump("INPUT", chain))
        return results

    def block_source(self, src_ip: str, comment: str = "nad-block") -> IptablesResult:
        chain = self.cfg.iptables_chain
        # Insert at top so blocks override any earlier ACCEPT/rate-limit rules.
        res = self._ensure_rule(
            ["-I", chain, "1", "-s", src_ip, "-m", "comment", "--comment", comment, "-j", "DROP"],
            insert=True,
        )
        # Return the final add (or check) result for convenience
        return res[-1]

    def rate_limit_source(
        self,
        src_ip: str,
        limit: str = "10/sec",
        burst: int = 20,
        comment: str = "nad-limit",
    ) -> list[IptablesResult]:
        """
        Token-bucket style cap: ACCEPT under limit, then DROP excess (two rules).
        """
        chain = self.cfg.iptables_chain
        # Important ordering:
        # 1) limited ACCEPT near top
        # 2) unconditional DROP immediately after
        # This makes excess packets drop while allowing up to the configured rate.
        out: list[IptablesResult] = []
        out.extend(
            self._ensure_rule(
                [
                    "-I",
                    chain,
                    "1",
                    "-s",
                    src_ip,
                    "-m",
                    "limit",
                    "--limit",
                    limit,
                    "--limit-burst",
                    str(burst),
                    "-m",
                    "comment",
                    "--comment",
                    comment + "-accept",
                    "-j",
                    "ACCEPT",
                ],
                insert=True,
            )
        )
        out.extend(
            self._ensure_rule(
                ["-I", chain, "2", "-s", src_ip, "-m", "comment", "--comment", comment + "-drop", "-j", "DROP"],
                insert=True,
            )
        )
        return out

    def list_rules(self) -> IptablesResult:
        return self._run(["-S", self.cfg.iptables_chain])

    def list_input(self) -> IptablesResult:
        return self._run(["-S", "INPUT"])
