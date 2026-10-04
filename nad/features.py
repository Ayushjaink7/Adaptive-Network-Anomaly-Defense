from __future__ import annotations

import ipaddress
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.packet import Packet
from scapy.utils import PcapReader

from .config import DefenseConfig


def _in_subnet(ip_str: str, network: str) -> bool:
    try:
        return ipaddress.ip_address(ip_str) in ipaddress.ip_network(network, strict=False)
    except ValueError:
        return False


def _tcp_flag_str(tcp: TCP) -> str:
    f = tcp.flags
    names = []
    for letter, mask in (
        ("FIN", 0x01),
        ("SYN", 0x02),
        ("RST", 0x04),
        ("PSH", 0x08),
        ("ACK", 0x10),
        ("URG", 0x20),
        ("ECE", 0x40),
        ("CWR", 0x80),
    ):
        if f & mask:
            names.append(letter)
    return ",".join(names) if names else "NULL"


def _service_bucket(dport: int, proto: str) -> str:
    if proto != "TCP":
        return proto if proto in {"UDP", "ICMP"} else "OTHER"
    if dport == 22:
        return "SSH"
    if dport == 80:
        return "HTTP"
    if dport == 443:
        return "HTTPS"
    return "TCP_OTHER"


@dataclass
class FlowState:
    key: tuple[str, str, int, int, str]
    packets: int = 0
    bytes_total: int = 0
    sizes: list[int] = field(default_factory=list)
    first_ts: float | None = None
    last_ts: float | None = None
    syn: int = 0
    fin: int = 0
    rst: int = 0
    psh: int = 0
    urg: int = 0
    ece: int = 0
    cwr: int = 0
    null_tcp: int = 0
    flag_strings: list[str] = field(default_factory=list)
    proto: str = "OTHER"
    dport: int = 0
    sport: int = 0


def _finalize_flow(st: FlowState, cfg: DefenseConfig) -> dict[str, Any]:
    src, dst, sport, dport, proto = st.key
    dur = max((st.last_ts or 0) - (st.first_ts or 0), 1e-6)
    mean_sz = float(np.mean(st.sizes)) if st.sizes else 0.0
    std_sz = float(np.std(st.sizes)) if len(st.sizes) > 1 else 0.0
    pps = st.packets / dur
    bps = (st.bytes_total * 8) / dur
    syn_ratio = st.syn / max(st.packets, 1)

    suspicious_flags = 0
    for fs in st.flag_strings:
        if fs == "NULL" or fs in cfg.suspicious_tcp_flag_sets:
            suspicious_flags += 1

    service = _service_bucket(dport, proto)

    return {
        "src_ip": src,
        "dst_ip": dst,
        "src_port": sport,
        "dst_port": dport,
        "protocol": proto,
        "service_bucket": service,
        "flow_start_ts": float(st.first_ts or 0.0),
        "packet_count": st.packets,
        "total_bytes": st.bytes_total,
        "mean_packet_size": mean_sz,
        "std_packet_size": std_sz,
        "flow_duration_sec": dur,
        "packets_per_sec": pps,
        "bytes_per_sec": bps,
        "syn_count": st.syn,
        "fin_count": st.fin,
        "rst_count": st.rst,
        "syn_ratio": syn_ratio,
        "suspicious_tcp_flag_events": suspicious_flags,
        "monitored_dst": _in_subnet(dst, cfg.monitored_subnet),
    }


def iter_packets(path: str) -> Iterator[Packet]:
    reader = PcapReader(path)
    try:
        for pkt in reader:
            yield pkt
    finally:
        reader.close()


def extract_flow_records_from_pcap(pcap_path: str, cfg: DefenseConfig | None = None) -> pd.DataFrame:
    """
    Build per-flow feature rows from a PCAP (offline). Flow key = 5-tuple; split on idle gap.
    """
    cfg = cfg or DefenseConfig()
    flows: dict[tuple[str, str, int, int, str], FlowState] = {}
    last_seen: dict[tuple[str, str, int, int, str], float] = {}

    finalized: list[dict[str, Any]] = []

    for pkt in iter_packets(pcap_path):
        if not pkt.haslayer(IP):
            continue
        ip = pkt[IP]
        ts = float(pkt.time)
        proto_name = "OTHER"
        sport = dport = 0
        tcp_layer = None
        if ip.haslayer(TCP):
            tcp_layer = ip[TCP]
            proto_name = "TCP"
            sport = int(tcp_layer.sport)
            dport = int(tcp_layer.dport)
        elif ip.haslayer(UDP):
            udp = ip[UDP]
            proto_name = "UDP"
            sport = int(udp.sport)
            dport = int(udp.dport)
        elif ip.haslayer(ICMP):
            proto_name = "ICMP"

        src, dst = ip.src, ip.dst
        key = (src, dst, sport, dport, proto_name)

        # Flush idle flows for this key and other keys (simple: only check this key)
        prev = last_seen.get(key)
        if prev is not None and (ts - prev) > cfg.flow_idle_timeout_sec and key in flows:
            finalized.append(_finalize_flow(flows.pop(key), cfg))

        st = flows.get(key)
        if st is None:
            st = FlowState(key=key, proto=proto_name, dport=dport, sport=sport)
            flows[key] = st

        st.packets += 1
        plen = int(ip.len) if ip.len is not None else len(ip)
        st.bytes_total += plen
        st.sizes.append(plen)
        st.first_ts = ts if st.first_ts is None else st.first_ts
        st.last_ts = ts

        if tcp_layer is not None:
            fs = _tcp_flag_str(tcp_layer)
            st.flag_strings.append(fs)
            f = int(tcp_layer.flags)
            if f == 0:
                st.null_tcp += 1
            if f & 0x02:
                st.syn += 1
            if f & 0x01:
                st.fin += 1
            if f & 0x04:
                st.rst += 1
            if f & 0x08:
                st.psh += 1
            if f & 0x20:
                st.urg += 1
            if f & 0x40:
                st.ece += 1
            if f & 0x80:
                st.cwr += 1

        last_seen[key] = ts

    for st in flows.values():
        finalized.append(_finalize_flow(st, cfg))

    return pd.DataFrame(finalized)


def featurize_for_ml(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Turn raw flow records into numeric matrix for sklearn.
    Returns (X_encoded, feature_names).
    """
    if df.empty:
        return pd.DataFrame(), []

    out = df.copy()
    proto_dum = pd.get_dummies(out["protocol"], prefix="proto")
    svc_dum = pd.get_dummies(out["service_bucket"], prefix="svc")
    num_cols = [
        "packet_count",
        "total_bytes",
        "mean_packet_size",
        "std_packet_size",
        "flow_duration_sec",
        "packets_per_sec",
        "bytes_per_sec",
        "syn_count",
        "fin_count",
        "rst_count",
        "syn_ratio",
        "suspicious_tcp_flag_events",
    ]
    X = pd.concat([out[num_cols].fillna(0), proto_dum, svc_dum], axis=1)
    return X, list(X.columns)


def augment_security_principle_flags(df: pd.DataFrame, cfg: DefenseConfig) -> pd.DataFrame:
    """
    Adds interpretable columns for report/presentation (CIA + auth + availability heuristics).
    """
    out = df.copy()
    out["confidentiality_plainhttp_to_server"] = (
        (out["service_bucket"] == "HTTP") & (out["monitored_dst"])
    ).astype(int)
    out["confidentiality_https"] = (out["service_bucket"] == "HTTPS").astype(int)
    out["integrity_tcp_anomaly"] = (out["suspicious_tcp_flag_events"] > 0).astype(int)
    out["availability_syn_flood_pattern"] = (
        (out["syn_ratio"] >= cfg.syn_flood_syn_ratio)
        & (out["packet_count"] >= cfg.syn_flood_min_packets)
    ).astype(int)
    return out


def label_bruteforce_by_ssh_frequency(df: pd.DataFrame, cfg: DefenseConfig) -> pd.Series:
    """
    Authentication / brute-force hint: many short flows to port 22 from one source in capture.
    """
    ssh = df[df["dst_port"] == 22]
    counts = ssh.groupby("src_ip").size()
    brute_ips = set(counts[counts >= cfg.ssh_bruteforce_flow_threshold].index)
    return df["src_ip"].isin(brute_ips).astype(int)
