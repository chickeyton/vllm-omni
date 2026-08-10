#!/bin/bash
# Start MiniCPM-o 4.5 in native full-duplex mode.
#
# The duplex overlay (session_mode: duplex) registers the WS /v1/duplex
# endpoint, enables the engine duplex control plane, and bounds the
# Thinker/Talker/Code2Wav stages to two live duplex sessions on one GPU.
# See vllm_omni/deploy/minicpmo_4_5_duplex.yaml for the session limits
# (idle TTL 300s, disconnect grace 30s, 16MiB pending input per session).
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
PORT="${PORT:-8099}"

exec vllm-omni serve "$MODEL" \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
