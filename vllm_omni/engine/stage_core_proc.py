"""
StageCoreProc - Stage-specific EngineCore subprocess for vLLM-Omni.

Extends vLLM's EngineCoreProc (which extends EngineCore) to run
stage-specific engine logic in a background subprocess.  Also provides
:meth:`launch_core_engines`, a stage-aware replacement for vLLM's
``launch_core_engines()`` that routes the subprocess target through
``StageCoreProc.run_engine_core``.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.tracing import maybe_init_worker_tracer
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.engine.core import EngineCoreProc

if TYPE_CHECKING:
    from vllm.v1.engine.utils import CoreEngineProcManager, EngineZmqAddresses

logger = init_logger(__name__)


class StageCoreProc(EngineCoreProc):
    """EngineCore subprocess specialized for omni pipeline stages.

    Extends vLLM's EngineCoreProc (a subclass of EngineCore) to support
    stage-specific engine behavior while reusing the full ZMQ IPC
    infrastructure (handshake, IO threads, busy loop).

    In addition to the subprocess entry-point (:meth:`run_engine_core`),
    this class provides :meth:`launch_core_engines` – a context-manager
    that starts the subprocess and performs the startup handshake, using
    ``cls.run_engine_core`` as the target so that stage-specific setup
    (DP=1, stage-scoped tracing, etc.) is applied automatically.
    """

    @staticmethod
    def run_engine_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageCoreProc in a background process.

        Mirrors EngineCoreProc.run_engine_core but instantiates
        StageCoreProc instead of EngineCoreProc.
        """
        shutdown_requested = False
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
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

            # Stages are single-engine; disable DP.
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0

            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.engine_core",
                process_kind="engine_core",
                process_name="StageCoreProc",
            )
            set_process_title("StageCoreProc")
            decorate_logs()

            engine_core = StageCoreProc(
                *args, engine_index=dp_rank, **kwargs
            )
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

    # ==================== Process Lifecycle ====================

    @classmethod
    @contextmanager
    def launch_core_engines(
        cls,
        vllm_config: VllmConfig,
        executor_class: type,
        log_stats: bool = False,
        addresses: EngineZmqAddresses | None = None,
    ) -> Iterator[tuple[CoreEngineProcManager | None, None, EngineZmqAddresses]]:
        """Context manager that launches stage engine core processes.

        Stage-specific replacement for vLLM's ``launch_core_engines()``
        that uses ``cls.run_engine_core`` as the subprocess target, ensuring
        stage-specific setup (DP=1 enforcement, stage-scoped tracing, etc.)
        runs in the background process.

        Stages always run as a single engine (DP=1) with no coordinator.

        Usage::

            with StageCoreProc.launch_core_engines(
                vllm_config, executor_class, addresses=addresses,
            ) as (engine_manager, coordinator, addresses):
                pass  # engine is started; wait_for_engine_startup runs on exit

        Or for split ``__enter__`` / ``__exit__`` (e.g. to release a lock
        between process start and startup wait)::

            cm = StageCoreProc.launch_core_engines(...)
            engine_manager, _, addresses = cm.__enter__()
            # ... release lock ...
            cm.__exit__(None, None, None)   # waits for startup

        Args:
            vllm_config: Engine configuration.
            executor_class: Executor class for the engine.
            log_stats: Whether to log stats in the engine process.
            addresses: Pre-created ZMQ addresses for input/output sockets.
                If *None*, addresses are created automatically via
                ``get_engine_zmq_addresses``.

        Yields:
            ``(engine_manager, None, addresses)`` – *coordinator* is always
            ``None`` for stage engines (no DP coordination).
        """
        import zmq
        from vllm.utils.network_utils import (
            get_open_zmq_ipc_path,
            zmq_socket_ctx,
        )
        from vllm.v1.engine.utils import (
            CoreEngine,
            CoreEngineProcManager,
            EngineZmqAddresses as _EngineZmqAddresses,
            wait_for_engine_startup,
        )

        parallel_config = vllm_config.parallel_config

        if addresses is None:
            from vllm.v1.engine.utils import get_engine_zmq_addresses
            addresses = get_engine_zmq_addresses(vllm_config)

        engines_to_handshake = [CoreEngine(index=0, local=True)]
        handshake_address = get_open_zmq_ipc_path()

        with zmq_socket_ctx(
            handshake_address, zmq.ROUTER, bind=True
        ) as handshake_socket:
            engine_manager = CoreEngineProcManager(
                cls.run_engine_core,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                local_client=True,
                local_engine_count=1,
                start_index=0,
                local_start_index=0,
            )

            yield engine_manager, None, addresses

            wait_for_engine_startup(
                handshake_socket,
                addresses,
                engines_to_handshake,
                parallel_config,
                False,
                vllm_config.cache_config,
                engine_manager,
                None,
            )
