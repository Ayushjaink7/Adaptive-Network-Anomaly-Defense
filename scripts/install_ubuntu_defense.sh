#!/usr/bin/env bash
# Run on Ubuntu defense VM (ARM or x86). Installs capture, SSH server, iptables tooling, Python deps.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  openssh-server \
  tcpdump tshark wireshark \
  iptables \
  build-essential python3-dev libpcap-dev
sudo systemctl enable --now ssh || sudo systemctl enable --now sshd
echo "Create venv next:"
echo "  python3 -m venv ~/nad-venv && source ~/nad-venv/bin/activate"
echo "  pip install -U pip && pip install -r /path/to/NS\\ Project/requirements.txt"
