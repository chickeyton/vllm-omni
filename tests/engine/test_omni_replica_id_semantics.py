"""Tests for the omni_replica_id assignment fix (Point 2.4).

After the fix, every DP-rank subprocess spawned by a single
``OmniCoreEngineProcManager`` shares the same ``omni_replica_id`` —
one OmniMasterServer registration = one omni replica slot, regardless
of the inner DP fan-out.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dp_local", [1, 2, 4])
def test_all_dp_ranks_share_one_omni_replica_id(dp_local: int) -> None:
    """OmniCoreEngineProcManager assigns the SAME omni_replica_id to every
    DP-rank subprocess of one registration. Previously each rank got
    base + index, which produced ids beyond OmniMasterServer's
    allocation when dp_local > 1.
    """
    from vllm_omni.engine.omni_core_engine_proc_manager import OmniCoreEngineProcManager

    # Capture every (target, name, kwargs) the manager would have spawned.
    spawned: list[dict[str, Any]] = []

    def fake_process(*, target: Any, name: str, kwargs: dict[str, Any]) -> MagicMock:
        spawned.append({"target": target, "name": name, "kwargs": dict(kwargs)})
        proc = MagicMock()
        proc.start = MagicMock()
        proc.is_alive = MagicMock(return_value=True)
        proc.exitcode = None
        return proc

    vllm_config = MagicMock()
    vllm_config.parallel_config.data_parallel_size = dp_local
    vllm_config.parallel_config.use_ray = False

    omni_replica_base_id = 7  # arbitrary; the master could pick anything

    with patch(
        "vllm_omni.engine.omni_core_engine_proc_manager.get_mp_context"
    ) as mock_ctx, patch(
        "vllm_omni.engine.omni_core_engine_proc_manager.numa_utils.configure_subprocess"
    ), patch(
        "vllm_omni.engine.omni_core_engine_proc_manager.current_platform"
    ) as mock_platform, patch(
        "vllm_omni.engine.omni_core_engine_proc_manager.weakref.finalize"
    ):
        mock_ctx.return_value.Process = fake_process
        mock_platform.is_cuda_alike.return_value = True

        OmniCoreEngineProcManager(
            local_engine_count=dp_local,
            start_index=0,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=False,
            handshake_address="tcp://127.0.0.1:1234",
            executor_class=MagicMock(),
            log_stats=False,
            omni_stage_id=0,
            omni_coordinator_address="tcp://127.0.0.1:2222",
            omni_replica_base_id=omni_replica_base_id,
        )

    assert len(spawned) == dp_local
    # Critical invariant: every spawned subprocess gets the SAME omni_replica_id.
    omni_ids = [s["kwargs"]["omni_replica_id"] for s in spawned]
    assert omni_ids == [omni_replica_base_id] * dp_local, (
        f"Expected every DP rank to share omni_replica_id={omni_replica_base_id}; "
        f"got {omni_ids}"
    )
    # DP rank itself still varies per subprocess.
    dp_ranks = [s["kwargs"]["dp_rank"] for s in spawned]
    assert dp_ranks == list(range(dp_local))
    local_dp_ranks = [s["kwargs"]["local_dp_rank"] for s in spawned]
    assert local_dp_ranks == list(range(dp_local))
