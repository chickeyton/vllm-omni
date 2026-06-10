"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import contextlib
import os
import signal
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import (
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineProcManager,
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_engine_zmq_addresses,
    wait_for_engine_startup,
)
from vllm.v1.utils import shutdown

from vllm_omni.distributed.omni_coordinator import OmniCoordClientForStage
from vllm_omni.engine.stage_init_utils import set_death_signal

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process.

        Omni-specific kwargs:
          - ``omni_coordinator_address``: ROUTER address of the head-side
            :class:`OmniCoordinator`. When provided, this subprocess
            instantiates an :class:`OmniCoordClientForStage` after the
            HELLO/INIT/READY handshake completes and reports its status +
            queue length via heartbeats. The hook is wired so each
            heartbeat refreshes ``queue_length`` from the live scheduler.
          - ``omni_stage_id``: logical stage id this replica belongs to.
            Required when ``omni_coordinator_address`` is provided.
          - ``omni_replica_id``: cluster-unique replica id within the
            stage (assigned by :class:`OmniMasterServer`). Used for
            logging / metrics only.
        """
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()

        # Register vllm-omni reasoning parsers (e.g. step_audio) in this
        # subprocess so they are available when the engine core resolves
        # ``--reasoning-parser``.  The main process already registered them
        # at import time, but the forked subprocess starts with a fresh
        # ReasoningParserManager.
        try:
            import vllm_omni.reasoning  # noqa: F401
        except ImportError:
            logger.warning(
                "Failed to import vllm_omni.reasoning in subprocess; "
                "custom reasoning parsers (e.g. step_audio) will not be "
                "available."
            )

        engine_core: StageEngineCoreProc | None = None
        coord_client: OmniCoordClientForStage | None = None
        try:
            # NOTE: previous revisions hardcoded data_parallel_size=1 here
            # (TODO referencing issue #984). The hardcoding has been removed
            # so the DP fields propagate through from the caller exactly
            # like upstream vLLM.

            stage_label = f"stage{omni_stage_id}" if omni_stage_id is not None else "noid"
            set_death_signal(signal.SIGTERM)
            set_process_title(f"StageEngineCoreProc_{stage_label}_replica{omni_replica_id}_DP{dp_rank}")
            decorate_logs()
            # Workaround for flashinfer/jit-cache version mismatch in CI.
            # The parent process handles this gracefully via ring_globals.py,
            # but the subprocess hits an unprotected import in TopKTopPSampler.
            # Setting this env var allows the same graceful fallback to work.
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            os.environ["VLLM_OMNI_REPLICA_ID"] = str(max(int(omni_replica_id), 0))

            # Per-process DP wiring, mirroring
            # ``vllm.v1.engine.core.EngineCoreProc.run_engine_core``: every
            # engine-core proc must own a *distinct* DP rank. Without this each
            # proc kept ``data_parallel_rank == 0`` and collided binding the DP
            # rendezvous port (EADDRINUSE). ``StageEngineCoreProc`` adds no
            # engine-behavior overrides over ``EngineCoreProc``, so the MoE-DP
            # case can use ``DPEngineCoreProc`` directly to get wave/stat
            # coordination — the stage I/O lives in the executor/connectors.
            vllm_config = kwargs["vllm_config"]
            parallel_config = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                if data_parallel:
                    # Non-MoE DP ranks are fully independent — treat like DP=1;
                    # the outer DPLB client fans requests across the procs.
                    parallel_config.data_parallel_size = 1
                    parallel_config.data_parallel_size_local = 1
                    parallel_config.data_parallel_rank = 0
                engine_core = StageEngineCoreProc(
                    *args,
                    engine_index=dp_rank,
                    **kwargs,
                )

            # Only DP rank 0 of each omni replica heartbeats. Inner-DP
            # routing is DPLBAsyncMPClient's job; multiple coord clients
            # per replica would multiply ReplicaList entries by ``dp``
            # (it keys by input_addr). See docs/design/pr1-yaml-only.md §5.
            should_open_coord_client = (
                omni_coordinator_address is not None
                and int(local_dp_rank) == 0
            )
            if should_open_coord_client:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coordinator_address is set")
                addresses: EngineZmqAddresses = engine_core.addresses
                if not addresses.inputs or not addresses.outputs:
                    raise RuntimeError(
                        "EngineCore handshake did not populate input/output addresses; "
                        "cannot start OmniCoordClientForStage"
                    )
                coord_client = OmniCoordClientForStage(
                    coord_zmq_addr=omni_coordinator_address,
                    input_addr=addresses.inputs[0],
                    output_addr=addresses.outputs[0],
                    stage_id=int(omni_stage_id),
                )

                def _refresh_queue_length() -> None:
                    """Pre-heartbeat hook: refresh queue_length from scheduler."""
                    scheduler = getattr(engine_core, "scheduler", None)
                    if scheduler is None:
                        return
                    try:
                        coord_client._queue_length = int(  # type: ignore[union-attr]
                            scheduler.get_num_unfinished_requests()
                        )
                    except Exception:
                        # Live scheduler stats are best-effort — heartbeats
                        # must not fail because of a stats lookup error.
                        pass

                coord_client._on_heartbeat = _refresh_queue_length

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if coord_client is not None:
                with contextlib.suppress(RuntimeError):
                    coord_client.close()
            if engine_core is not None:
                engine_core.shutdown()


@dataclass
class StageCoreLaunch:
    """Handles for an in-process stage launch, returned by ``spawn_stage_core``.

    ``proc`` is set for the single-engine (``data_parallel_size == 1``) path.
    ``engine_manager`` / ``coordinator`` / ``engines_to_handshake`` are set for
    the inner-vLLM-DP path (``data_parallel_size > 1``), which spawns one
    engine-core process per local DP rank plus a DP coordinator.
    """

    addresses: EngineZmqAddresses
    handshake_address: str
    proc: BaseProcess | None = None
    engine_manager: CoreEngineProcManager | None = None
    coordinator: DPCoordinator | None = None
    engines_to_handshake: list[CoreEngine] | None = None


def spawn_stage_core(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool = False,
    *,
    omni_stage_id: int | None = None,
    omni_replica_id: int = 0,
) -> StageCoreLaunch:
    """Spawn the *StageEngineCoreProc* subprocess(es) without handshaking.

    Must be called while the correct device env vars are set (e.g. under
    the stage-launch lock).  Call ``complete_stage_handshake`` afterwards.

    For ``data_parallel_size == 1`` this spawns a single engine-core process
    (the lightweight path). For ``data_parallel_size > 1`` it spawns one
    engine-core process per *local* DP rank and (when the model needs it) a
    :class:`DPCoordinator`, mirroring ``vllm.v1.engine.utils.launch_core_engines``
    for the single-host internal-LB case but launching
    :meth:`StageEngineCoreProc.run_stage_core`.
    """
    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size

    # ---- Single-engine path (unchanged): no inner vLLM DP. ----
    if dp_size <= 1:
        addresses = get_engine_zmq_addresses(vllm_config)
        handshake_address = get_open_zmq_ipc_path()

        ctx = get_mp_context()
        proc = ctx.Process(
            target=StageEngineCoreProc.run_stage_core,
            name="StageEngineCoreProc",
            kwargs={
                "vllm_config": vllm_config,
                "local_client": True,
                "handshake_address": handshake_address,
                "executor_class": executor_class,
                "log_stats": log_stats,
                "dp_rank": 0,
                "local_dp_rank": 0,
            },
        )
        proc.start()
        return StageCoreLaunch(addresses=addresses, handshake_address=handshake_address, proc=proc)

    # ---- Inner vLLM-DP path: one engine-core proc per local DP rank. ----
    # Without this, a stage carrying ``data_parallel_size > 1`` in head mode
    # spawned only DP rank 0; ranks 1..N never joined the TP x DP NCCL world
    # and init blocked until the handshake timed out. See
    # docs/design/pr1-test-report.md (S2 root cause).
    from vllm_omni.engine.omni_core_engine_proc_manager import OmniCoreEngineProcManager

    local_engine_count = (
        parallel_config.data_parallel_size_local
        if parallel_config.data_parallel_size_local is not None and parallel_config.data_parallel_size_local > 0
        else max(1, dp_size)
    )
    # ``get_engine_zmq_addresses`` reads ``data_parallel_size_local`` to choose
    # IPC vs TCP transport; on a single host every DP rank is local, so make
    # the count explicit before allocating addresses.
    parallel_config.data_parallel_size_local = local_engine_count
    dp_rank = parallel_config.data_parallel_rank or 0

    addresses = get_engine_zmq_addresses(vllm_config)

    # Internal-LB / MoE DP needs a coordinator for queue-stats load balancing
    # and (for MoE) wave coordination. Rank 0 owns it.
    coordinator: DPCoordinator | None = None
    if vllm_config.needs_dp_coordinator and dp_rank == 0:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )
        addresses.coordinator_input, addresses.coordinator_output = coordinator.get_engine_socket_addresses()
        addresses.frontend_stats_publish_address = coordinator.get_stats_publish_address()

    handshake_address = get_open_zmq_ipc_path()

    engine_manager = OmniCoreEngineProcManager(
        local_engine_count=local_engine_count,
        start_index=dp_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=True,
        handshake_address=handshake_address,
        executor_class=executor_class,
        log_stats=log_stats,
        omni_stage_id=int(omni_stage_id) if omni_stage_id is not None else 0,
        omni_coordinator_address=None,
        omni_replica_base_id=omni_replica_id,
    )

    engines_to_handshake = [CoreEngine(index=dp_rank + i, local=True) for i in range(local_engine_count)]

    return StageCoreLaunch(
        addresses=addresses,
        handshake_address=handshake_address,
        engine_manager=engine_manager,
        coordinator=coordinator,
        engines_to_handshake=engines_to_handshake,
    )


def complete_stage_handshake(
    launch: StageCoreLaunch,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> None:
    """Perform the HELLO/INIT/READY handshake with the spawned engine(s).

    On failure the spawned process(es) and coordinator are torn down before
    re-raising.
    """
    if launch.engine_manager is not None:
        _complete_dp_handshake(launch, vllm_config)
        return

    assert launch.proc is not None
    try:
        _perform_handshake(launch.proc, launch.handshake_address, launch.addresses, vllm_config, handshake_timeout)
    except Exception:
        shutdown([launch.proc])
        raise


def _complete_dp_handshake(launch: StageCoreLaunch, vllm_config: VllmConfig) -> None:
    """Handshake every local DP-rank engine via vLLM's multi-engine waiter."""
    parallel_config = vllm_config.parallel_config
    coordinated_dp = parallel_config.data_parallel_size > 1 and vllm_config.model_config.is_moe
    assert launch.engine_manager is not None
    assert launch.engines_to_handshake is not None
    try:
        with zmq_socket_ctx(launch.handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
            wait_for_engine_startup(
                handshake_socket,
                launch.addresses,
                launch.engines_to_handshake,
                parallel_config,
                coordinated_dp,
                vllm_config.cache_config,
                launch.engine_manager,
                launch.coordinator.proc if launch.coordinator else None,
            )
    except Exception:
        with contextlib.suppress(Exception):
            launch.engine_manager.shutdown()
        if launch.coordinator is not None:
            with contextlib.suppress(Exception):
                launch.coordinator.close()
        raise


def _perform_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> None:
    """Run the HELLO / INIT / READY handshake with the subprocess."""
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)

        identity, msg = _recv(poller, handshake_socket, proc, "HELLO", handshake_timeout)
        if msg.get("status") != "HELLO":
            raise RuntimeError(f"Expected HELLO, got: {msg}")

        init_payload = EngineHandshakeMetadata(
            addresses=addresses,
            parallel_config={},
        )
        handshake_socket.send_multipart([identity, msgspec.msgpack.encode(init_payload)])

        identity, msg = _recv(poller, handshake_socket, proc, "READY", handshake_timeout)
        if msg.get("status") != "READY":
            raise RuntimeError(f"Expected READY, got: {msg}")
        num_gpu_blocks = msg.get("num_gpu_blocks")
        if num_gpu_blocks is not None:
            vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks


def _recv(
    poller: zmq.Poller,
    handshake_socket: zmq.Socket,
    proc: BaseProcess,
    expected: str,
    timeout_s: int = 600,
) -> tuple[bytes, dict]:
    """Wait for one handshake message; raise if the process dies first."""
    timeout_ms = timeout_s * 1000
    while True:
        events = dict(poller.poll(timeout=timeout_ms))
        if not events:
            raise TimeoutError(
                f"Timed out waiting for {expected} from StageEngineCoreProc after {timeout_s}s. "
                f"This typically indicates model loading or initialization is taking too long. "
                f"Consider increasing `stage_init_timeout` for large models."
            )
        if handshake_socket in events:
            identity, raw = handshake_socket.recv_multipart()
            return identity, msgspec.msgpack.decode(raw)
        if proc.exitcode is not None:
            raise RuntimeError(f"StageEngineCoreProc died during {expected} (exit code {proc.exitcode})")
