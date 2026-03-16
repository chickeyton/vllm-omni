"""
Stage Core Process for vLLM-Omni V1 architecture.

StageCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import signal
import weakref
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
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_engine_zmq_addresses,
)
from vllm.v1.utils import shutdown

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor

logger = init_logger(__name__)

# Timeout (seconds) for the handshake poll when waiting for engine startup.
_HANDSHAKE_POLL_TIMEOUT_S = 600


class StageCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  The busy loop, ZMQ I/O
    threads, and handshake protocol are all inherited from EngineCoreProc;
    only the process entry point is reimplemented so that we do **not**
    delegate to ``EngineCoreProc.run_engine_core()``.
    """

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageCoreProc busy loop in background process.

        This is a standalone implementation that does NOT delegate to
        ``EngineCoreProc.run_engine_core()``.
        """
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning.
        maybe_register_config_serialize_by_value()

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: StageCoreProc | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config = vllm_config.parallel_config

            process_title = f"StageCoreProc_DP{dp_rank}"
            set_process_title(process_title)
            decorate_logs()

            # Stage processes always run as a single DP rank.
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            parallel_config.data_parallel_index = dp_rank

            engine_core = StageCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )

            assert engine_core is not None
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageCoreProc exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("StageCoreProc failed to start.")
            else:
                logger.exception(
                    "StageCoreProc encountered a fatal error."
                )
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()


class StageCoreManager:
    """Manages the lifecycle of a single StageCoreProc subprocess.

    Mirrors ``CoreEngineProcManager`` from vLLM but targets
    ``StageCoreProc.run_stage_core`` instead of
    ``EngineCoreProc.run_engine_core``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        handshake_address: str,
    ) -> None:
        context = get_mp_context()
        self.processes: list[BaseProcess] = [
            context.Process(
                target=StageCoreProc.run_stage_core,
                name="StageCoreProc",
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
        ]
        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        self.processes[0].start()

    def close(self) -> None:
        """Shutdown the subprocess."""
        self._finalizer()

    def finished_procs(self) -> dict[str, int]:
        """Return dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


def launch_stage_core(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool = False,
) -> tuple[EngineZmqAddresses, StageCoreManager]:
    """Spawn a *StageCoreProc* subprocess and perform the startup handshake.

    Returns ``(addresses, manager)`` where *addresses* contains the ZMQ
    socket paths the client should bind to and *manager* owns the subprocess
    handle (attach to ``BackgroundResources.engine_manager`` for cleanup).
    """
    addresses = get_engine_zmq_addresses(vllm_config)
    handshake_address = get_open_zmq_ipc_path()

    # Spawn the subprocess.
    manager = StageCoreManager(
        vllm_config,
        executor_class,
        log_stats,
        handshake_address,
    )

    # Perform the handshake (HELLO → INIT → READY).
    try:
        with zmq_socket_ctx(
            handshake_address, zmq.ROUTER, bind=True
        ) as handshake_socket:
            poller = zmq.Poller()
            poller.register(handshake_socket, zmq.POLLIN)
            for sentinel in (
                proc.sentinel for proc in manager.processes
            ):
                poller.register(sentinel, zmq.POLLIN)

            # --- Wait for HELLO ---
            identity, msg_bytes = _recv_handshake(
                poller, handshake_socket, manager, "HELLO"
            )
            hello_msg = msgspec.msgpack.decode(msg_bytes)
            if hello_msg.get("status") != "HELLO":
                raise RuntimeError(
                    f"Expected HELLO from StageCoreProc, got: {hello_msg}"
                )

            # --- Send INIT with addresses ---
            init_payload = EngineHandshakeMetadata(
                addresses=addresses,
                parallel_config={},
            )
            handshake_socket.send_multipart(
                [identity, msgspec.msgpack.encode(init_payload)]
            )

            # --- Wait for READY ---
            identity, msg_bytes = _recv_handshake(
                poller, handshake_socket, manager, "READY"
            )
            ready_msg = msgspec.msgpack.decode(msg_bytes)
            if ready_msg.get("status") != "READY":
                raise RuntimeError(
                    f"Expected READY from StageCoreProc, got: {ready_msg}"
                )
            num_gpu_blocks = ready_msg.get("num_gpu_blocks")
            if num_gpu_blocks is not None:
                vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

    except Exception:
        manager.close()
        raise

    return addresses, manager


def _recv_handshake(
    poller: zmq.Poller,
    handshake_socket: zmq.Socket,
    manager: StageCoreManager,
    expected_status: str,
) -> tuple[bytes, bytes]:
    """Block until a message arrives on the handshake socket.

    Raises ``RuntimeError`` if the subprocess exits before the expected
    handshake message is received.
    """
    timeout_ms = _HANDSHAKE_POLL_TIMEOUT_S * 1000
    while True:
        events = dict(poller.poll(timeout=timeout_ms))
        if not events:
            raise TimeoutError(
                f"Timed out waiting for {expected_status} from "
                f"StageCoreProc after {_HANDSHAKE_POLL_TIMEOUT_S}s"
            )
        if handshake_socket in events:
            identity, msg_bytes = handshake_socket.recv_multipart()
            return identity, msg_bytes

        # A process sentinel fired – the subprocess died.
        finished = manager.finished_procs()
        if finished:
            raise RuntimeError(
                f"StageCoreProc died during startup handshake "
                f"({expected_status} phase). Exit codes: {finished}"
            )
