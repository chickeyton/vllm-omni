"""Unit tests for _enforce_omni_parallel_config.

Covers Point 2's YAML validation rules: reject user-set values for fields
vllm-omni / vLLM own, force-set data_parallel_size = data_parallel_size_local.

For LLM stages, ``engine_args`` is FLAT (no nested ``parallel_config:`` key)
because that's how vLLM's ``EngineArgs`` dataclass is shaped.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine.stage_init_utils import _enforce_omni_parallel_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_llm_stage(devices: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        runtime=({"devices": devices} if devices else None),
    )


def _make_dit_stage() -> SimpleNamespace:
    return SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        runtime=None,
    )


# ---------------------------------------------------------------------------
# Reject user-set "must omit" fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_address",
        "data_parallel_rpc_port",
        "data_parallel_master_port",
        "data_parallel_rank",
        "data_parallel_rank_local",
    ],
)
def test_rejects_user_set_must_omit_field(field: str) -> None:
    """Fields vllm-omni / vLLM own must not appear in the YAML."""
    stage = _make_llm_stage()
    args_dict: dict[str, Any] = {field: 42, "data_parallel_size_local": 1}
    with pytest.raises(ValueError, match=f"engine_args.{field}"):
        _enforce_omni_parallel_config(stage, args_dict)


# ---------------------------------------------------------------------------
# data_parallel_size: equality-or-absent
# ---------------------------------------------------------------------------


def test_rejects_dp_size_mismatch_with_local() -> None:
    stage = _make_llm_stage()
    args_dict = {"data_parallel_size": 4, "data_parallel_size_local": 2}
    with pytest.raises(ValueError, match="must equal data_parallel_size_local"):
        _enforce_omni_parallel_config(stage, args_dict)


def test_accepts_dp_size_equal_to_local() -> None:
    stage = _make_llm_stage()
    args_dict = {"data_parallel_size": 2, "data_parallel_size_local": 2}
    _enforce_omni_parallel_config(stage, args_dict)  # should not raise
    assert args_dict["data_parallel_size"] == 2


def test_force_sets_dp_size_to_local_when_absent() -> None:
    stage = _make_llm_stage()
    args_dict: dict[str, Any] = {"data_parallel_size_local": 3}
    _enforce_omni_parallel_config(stage, args_dict)
    assert args_dict["data_parallel_size"] == 3


def test_force_sets_dp_size_to_one_when_neither_set() -> None:
    """Default dp_local = 1; data_parallel_size should also default to 1."""
    stage = _make_llm_stage()
    args_dict: dict[str, Any] = {}
    _enforce_omni_parallel_config(stage, args_dict)
    assert args_dict["data_parallel_size"] == 1


# ---------------------------------------------------------------------------
# Boolean-must-be-False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["data_parallel_external_lb", "data_parallel_hybrid_lb", "enable_elastic_ep"],
)
def test_rejects_lb_or_elastic_ep_true(field: str) -> None:
    stage = _make_llm_stage()
    args_dict: dict[str, Any] = {field: True, "data_parallel_size_local": 1}
    with pytest.raises(ValueError, match=f"engine_args.{field}"):
        _enforce_omni_parallel_config(stage, args_dict)


@pytest.mark.parametrize(
    "field",
    ["data_parallel_external_lb", "data_parallel_hybrid_lb", "enable_elastic_ep"],
)
def test_accepts_lb_or_elastic_ep_false(field: str) -> None:
    stage = _make_llm_stage()
    args_dict: dict[str, Any] = {field: False, "data_parallel_size_local": 1}
    _enforce_omni_parallel_config(stage, args_dict)  # should not raise


# ---------------------------------------------------------------------------
# Backend constraint
# ---------------------------------------------------------------------------


def test_rejects_ray_backend() -> None:
    stage = _make_llm_stage()
    args_dict = {"data_parallel_backend": "ray", "data_parallel_size_local": 1}
    with pytest.raises(ValueError, match="data_parallel_backend must be 'mp'"):
        _enforce_omni_parallel_config(stage, args_dict)


def test_accepts_mp_backend_explicitly() -> None:
    stage = _make_llm_stage()
    args_dict = {"data_parallel_backend": "mp", "data_parallel_size_local": 1}
    _enforce_omni_parallel_config(stage, args_dict)


# ---------------------------------------------------------------------------
# EPLB sanity
# ---------------------------------------------------------------------------


def test_rejects_eplb_when_dp_local_1_tp_1() -> None:
    stage = _make_llm_stage()
    args_dict = {
        "enable_eplb": True,
        "data_parallel_size_local": 1,
        "tensor_parallel_size": 1,
    }
    with pytest.raises(ValueError, match="enable_eplb requires"):
        _enforce_omni_parallel_config(stage, args_dict)


def test_accepts_eplb_when_world_sufficient() -> None:
    stage = _make_llm_stage()
    args_dict = {
        "enable_eplb": True,
        "enable_expert_parallel": True,
        "data_parallel_size_local": 2,
        "tensor_parallel_size": 2,
    }
    _enforce_omni_parallel_config(stage, args_dict)


# ---------------------------------------------------------------------------
# Device-count product check
# ---------------------------------------------------------------------------


def test_rejects_devices_not_divisible_by_per_replica() -> None:
    """devices=7 GPUs cannot host any whole replica when per_replica = 4."""
    stage = _make_llm_stage(devices="0,1,2,3,4,5,6")  # 7 devices
    args_dict = {
        "tensor_parallel_size": 2,
        "data_parallel_size_local": 2,
    }
    with pytest.raises(ValueError, match="not divisible by per-replica"):
        _enforce_omni_parallel_config(stage, args_dict)


def test_accepts_devices_divisible_by_per_replica() -> None:
    stage = _make_llm_stage(devices="0,1,2,3,4,5,6,7")  # 8 devices = 2 replicas × 4
    args_dict = {
        "tensor_parallel_size": 2,
        "data_parallel_size_local": 2,
    }
    _enforce_omni_parallel_config(stage, args_dict)


# ---------------------------------------------------------------------------
# Validator does not add unexpected keys to engine_args
# ---------------------------------------------------------------------------


def test_validator_only_writes_data_parallel_size() -> None:
    """The validator must not add fields like 'parallel_config' or
    'api_server_count' to engine_args — those aren't EngineArgs fields and
    would cause OmniEngineArgs(**dict) to error with 'unexpected keyword'.
    """
    stage = _make_llm_stage()
    args_dict: dict[str, Any] = {
        "tensor_parallel_size": 2,
        "data_parallel_size_local": 2,
    }
    snapshot_keys = set(args_dict.keys())
    _enforce_omni_parallel_config(stage, args_dict)
    new_keys = set(args_dict.keys()) - snapshot_keys
    # Only data_parallel_size is force-set (the others are vLLM defaults).
    assert new_keys == {"data_parallel_size"}, f"unexpected new keys: {new_keys}"


# ---------------------------------------------------------------------------
# DiT stages: early-return (no validation, no force-set)
# ---------------------------------------------------------------------------


def test_dit_stage_is_soft_pass() -> None:
    """DiT engine_args has nested ``parallel_config:`` with a different
    schema (DiffusionParallelConfig). The LLM validator must not touch it.
    """
    stage = _make_dit_stage()
    args_dict = {
        "parallel_config": {
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "ulysses_degree": 2,
            "cfg_parallel_size": 2,
            "enable_expert_parallel": True,
        },
    }
    snapshot = {"parallel_config": dict(args_dict["parallel_config"])}
    _enforce_omni_parallel_config(stage, args_dict)
    # No force-set, no mutation.
    assert args_dict == snapshot


def test_dit_stage_does_not_reject_legacy_fields() -> None:
    """Even with LLM-only fields present (rare but possible), DiT path
    is a no-op."""
    stage = _make_dit_stage()
    args_dict = {"data_parallel_size": 2, "tensor_parallel_size": 2}
    _enforce_omni_parallel_config(stage, args_dict)
