#!/usr/bin/env bash
# Usage: sudo ./capture_tcpdump.sh eth0 /tmp/live.pcap
# Writes PCAP while printing a live preview path for tshark pipe (optional).
IFACE="${1:?interface}"
OUT="${2:?output.pcap}"
sudo tcpdump -i "$IFACE" -U -w "$OUT" 'not port 22 and not port 53' &
echo "tcpdump PID $! writing $OUT (filter excludes SSH/DNS for optional noise reduction — adjust as needed)"
