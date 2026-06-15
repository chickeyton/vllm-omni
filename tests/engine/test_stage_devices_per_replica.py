# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for ``get_stage_devices_per_replica``.

An LLM stage's per-replica device count must be ``tp x pp x pcp x dp``: every
inner vLLM-DP rank needs its own ``tp x pp`` GPUs, so a replica running
``data_parallel_size`` ranks consumes ``dp`` times the TP world. Returning
``tp`` alone (the prior behavior) under-provisioned every replica by a factor
of ``dp`` and made ``num_replicas > 1`` combined with ``data_parallel_size > 1``
unrunnable — the device split demanded ``tp`` GPUs/replica while the runtime
placed workers across ``tp x dp`` (PR1 §3.2 rule 9, RFC #3865).
"""

from types import SimpleNamespace

import pytest

from vllm_omni.engine.stage_init_utils import get_stage_devices_per_replica


def _llm_stage(**engine_args):
    return SimpleNamespace(stage_type="llm", engine_args=dict(engine_args))


@pytest.mark.parametrize(
    "engine_args, expected",
    [
        ({}, 1),  # bare LLM stage -> 1 GPU
        ({"tensor_parallel_size": 2}, 2),  # TP only
        ({"tensor_parallel_size": 2, "data_parallel_size": 2}, 4),  # TP x DP (the bug)
        ({"tensor_parallel_size": 1, "data_parallel_size": 2}, 2),  # DP spreads beyond TP
        ({"tensor_parallel_size": 2, "pipeline_parallel_size": 2}, 4),  # TP x PP
        (
            {
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "prefill_context_parallel_size": 2,
            },
            8,
        ),  # TP x DP x PCP
    ],
)
def test_llm_devices_per_replica_includes_dp(engine_args, expected):
    assert get_stage_devices_per_replica(_llm_stage(**engine_args)) == expected


def test_none_parallel_fields_collapse_to_one():
    # A YAML field present but left blank (``pipeline_parallel_size:``) arrives
    # as ``None``; it must collapse to 1, not crash or zero out the product.
    stage = _llm_stage(
        tensor_parallel_size=2,
        data_parallel_size=2,
        pipeline_parallel_size=None,
        prefill_context_parallel_size=None,
    )
    assert get_stage_devices_per_replica(stage) == 4


def test_engine_args_object_access():
    # ``engine_args`` may be an attribute-bearing object rather than a dict.
    stage = SimpleNamespace(
        stage_type="llm",
        engine_args=SimpleNamespace(tensor_parallel_size=2, data_parallel_size=2),
    )
    assert get_stage_devices_per_replica(stage) == 4
