#!/bin/bash
# Server side for barge_in_client.py: start MiniCPM-o 4.5 in native
# full-duplex mode.
#
# The duplex deploy profile registers the WS /v1/duplex endpoint (with
# /v1/realtime?duplex=1 projecting onto the same handler), enables the
# engine duplex control plane, and bounds the Thinker/Talker/Code2Wav
# stages to two live duplex sessions on one GPU. See
# vllm_omni/deploy/minicpmo_4_5_duplex.yaml for the session limits
# (idle TTL 300s, disconnect grace 30s, 16MiB pending input per session).
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
PORT="${PORT:-8099}"

exec vllm-omni serve "$MODEL" \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
# duplex endpoints: ws://<host>:$PORT/v1/duplex and /v1/realtime?duplex=1
