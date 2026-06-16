# SPDX-License-Identifier: Apache-2.0
"""Regression: KV-transfer ZMQ ports must be unique across inner DP groups.

Before the fix, ``kv_zmq_port`` keyed only on (replica, tp_rank, stage), so a
KV-sending stage with ``data_parallel_size > 1`` bound the same port for every
DP group's TP-rank-0 worker -> EADDRINUSE on the second DP engine. The advertised
endpoint (``get_kv_sender_info``) had no DP dimension either. Both now carry a
``dp_rank * KV_DP_PORT_STRIDE`` term.
"""
import pytest

from vllm_omni.distributed.omni_connectors.utils.initialization import (
    KV_DP_PORT_STRIDE,
    KV_RANK_PORT_STRIDE,
    KV_REPLICA_PORT_STRIDE,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import kv_zmq_port

BASE = 50051


def _port(stage, tp_rank, replica, dp_rank):
    return kv_zmq_port(BASE, stage, local_rank=tp_rank, replica_id=replica, dp_rank=dp_rank)


def test_dp_groups_get_distinct_ports():
    # AR stage: 2 replicas x DP2 x TP2, stage 0 -> 8 sender workers, all unique.
    ports = [
        _port(0, tp, rep, dp)
        for rep in (0, 1)
        for dp in (0, 1)
        for tp in (0, 1)
    ]
    assert len(set(ports)) == len(ports), f"port collision: {ports}"


def test_dp_offset_is_added_above_tp_range():
    # The two DP groups of the same replica must differ by exactly the DP stride
    # at TP rank 0, so the receiver's (advertised_base + tp_rank*RANK_STRIDE)
    # reconstruction stays correct within each DP group.
    dp0 = _port(0, 0, 0, 0)
    dp1 = _port(0, 0, 0, 1)
    assert dp1 - dp0 == KV_DP_PORT_STRIDE
    # TP rank offsets stay inside one DP group's block.
    assert _port(0, 1, 0, 0) - dp0 == KV_RANK_PORT_STRIDE
    assert KV_RANK_PORT_STRIDE < KV_DP_PORT_STRIDE <= KV_REPLICA_PORT_STRIDE


def test_backward_compatible_dp0():
    # dp_rank=0 reproduces the legacy port (replica 0 / dp 0 / rank 0).
    assert _port(0, 0, 0, 0) == kv_zmq_port(BASE, 0, local_rank=0, replica_id=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
