# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
StageCoreClient: Client for a single pipeline stage.

Manages the lifecycle of a StageCoreProc background process and provides
async methods for ZMQ-based communication. Each pipeline stage has one
StageCoreClient instance in the orchestrator process.
"""
from __future__ import annotations

import weakref
from functools import partial
from typing import Any

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.utils import (
    CoreEngineProcManager,
    EngineHandshakeMetadata,
    EngineZmqAddresses,
)
from vllm.v1.executor import Executor

from vllm_omni.engine.stage_core_proc import StageCoreProc

logger = init_logger(__name__)

# Handshake polling period (milliseconds)
_HANDSHAKE_POLL_MS = 2000
# Handshake timeout (milliseconds) — 10 minutes
_HANDSHAKE_TIMEOUT_MS = 600_000


class StageCoreClient:
    """Client for a single pipeline stage.

    Manages the lifecycle of a StageCoreProc running in a background
    process, performing the ZMQ handshake and providing async methods
    that delegate to an internal AsyncMPClient.

    Each StageCoreClient:
    - Launches one StageCoreProc in a dedicated background process
    - Performs a 3-phase ZMQ handshake (HELLO → INIT → READY)
    - Creates an AsyncMPClient that handles ZMQ I/O

    Args:
        vllm_config: VllmConfig for this stage's model.
        executor_class: Executor implementation class.
        log_stats: Whether to log statistics.
        stage_id: Unique identifier for this stage.
        stage_type: Type of stage ("llm" or "diffusion").
        devices: Device specification (e.g., "0,1" for GPUs).
        connectors_config: Configuration for stage connectors.
        stage_init_timeout: Timeout for stage initialization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        *,
        stage_id: int = 0,
        stage_type: str = "llm",
        devices: str | None = None,
        connectors_config: dict[str, Any] | None = None,
        stage_init_timeout: int = 300,
    ) -> None:
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.devices = devices
        self.vllm_config = vllm_config

        # 1. Create ZMQ addresses for input/output communication.
        #    The client BINDS both; the engine CONNECTS to both.
        input_address = get_open_zmq_ipc_path()
        output_address = get_open_zmq_ipc_path()
        handshake_address = get_open_zmq_ipc_path()

        addresses = EngineZmqAddresses(
            inputs=[input_address],
            outputs=[output_address],
        )

        # 2. Create a partial target function that wraps StageCoreProc.run_stage_core
        #    with stage-specific kwargs.  CoreEngineProcManager will add the
        #    standard engine kwargs (vllm_config, handshake_address, etc.).
        target_fn = partial(
            StageCoreProc.run_stage_core,
            stage_id=stage_id,
            stage_type=stage_type,
            devices=devices,
            connectors_config=connectors_config,
            stage_init_timeout=stage_init_timeout,
        )

        # 3. Launch the StageCoreProc process via CoreEngineProcManager.
        self._engine_manager = CoreEngineProcManager(
            target_fn,
            local_engine_count=1,
            start_index=0,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=True,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        # 4. Perform the 3-phase handshake with the engine process.
        self._do_handshake(handshake_address, addresses)

        # 5. Create AsyncMPClient with client_addresses (skip internal launch).
        client_addresses = {
            "input_address": input_address,
            "output_address": output_address,
        }
        self._mp_client = AsyncMPClient(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        # Store engine_manager in client resources for proper lifecycle management
        self._mp_client.resources.engine_manager = self._engine_manager
        self._mp_client.start_engine_core_monitor()

        logger.info(
            "[StageCoreClient-%d] Initialized (type=%s, devices=%s)",
            stage_id,
            stage_type,
            devices,
        )

    def _do_handshake(
        self,
        handshake_address: str,
        addresses: EngineZmqAddresses,
    ) -> None:
        """Perform 3-phase handshake with the StageCoreProc process.

        Protocol:
        1. Engine sends HELLO
        2. Client sends INIT with ZMQ addresses
        3. Engine sends READY

        Raises:
            TimeoutError: If handshake times out.
            RuntimeError: If handshake fails or engine process dies.
        """
        ctx = zmq.Context()
        handshake_socket = make_zmq_socket(
            ctx, handshake_address, zmq.ROUTER, bind=True
        )

        try:
            poller = zmq.Poller()
            poller.register(handshake_socket, zmq.POLLIN)

            # Add engine process sentinels to detect early death
            for sentinel in self._engine_manager.sentinels():
                poller.register(sentinel, zmq.POLLIN)

            # Phase 1: Wait for HELLO
            hello_received = False
            elapsed_ms = 0
            while not hello_received:
                events = poller.poll(timeout=_HANDSHAKE_POLL_MS)
                elapsed_ms += _HANDSHAKE_POLL_MS

                if not events:
                    if elapsed_ms >= _HANDSHAKE_TIMEOUT_MS:
                        raise TimeoutError(
                            f"[StageCoreClient-{self.stage_id}] "
                            f"Handshake timed out waiting for HELLO"
                        )
                    logger.debug(
                        "[StageCoreClient-%d] Waiting for engine HELLO...",
                        self.stage_id,
                    )
                    continue

                # Check if any event is from engine process dying
                if len(events) > 1 or events[0][0] != handshake_socket:
                    finished = self._engine_manager.finished_procs()
                    raise RuntimeError(
                        f"[StageCoreClient-{self.stage_id}] "
                        f"Engine died during handshake: {finished}"
                    )

                eng_identity, msg_bytes = handshake_socket.recv_multipart()
                msg = msgspec.msgpack.decode(msg_bytes)

                if msg.get("status") != "HELLO":
                    raise RuntimeError(
                        f"[StageCoreClient-{self.stage_id}] "
                        f"Expected HELLO, got: {msg}"
                    )
                hello_received = True

            # Phase 2: Send INIT with addresses
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=addresses,
                    parallel_config={},
                )
            )
            handshake_socket.send_multipart(
                (eng_identity, init_message), copy=False
            )
            logger.debug(
                "[StageCoreClient-%d] Sent INIT to engine",
                self.stage_id,
            )

            # Phase 3: Wait for READY
            ready_received = False
            while not ready_received:
                events = poller.poll(timeout=_HANDSHAKE_POLL_MS)
                elapsed_ms += _HANDSHAKE_POLL_MS

                if not events:
                    if elapsed_ms >= _HANDSHAKE_TIMEOUT_MS:
                        raise TimeoutError(
                            f"[StageCoreClient-{self.stage_id}] "
                            f"Handshake timed out waiting for READY"
                        )
                    continue

                if len(events) > 1 or events[0][0] != handshake_socket:
                    finished = self._engine_manager.finished_procs()
                    raise RuntimeError(
                        f"[StageCoreClient-{self.stage_id}] "
                        f"Engine died during handshake: {finished}"
                    )

                _, msg_bytes = handshake_socket.recv_multipart()
                msg = msgspec.msgpack.decode(msg_bytes)

                if msg.get("status") != "READY":
                    raise RuntimeError(
                        f"[StageCoreClient-{self.stage_id}] "
                        f"Expected READY, got: {msg}"
                    )
                ready_received = True

                # Store num_gpu_blocks from handshake
                num_gpu_blocks = msg.get("num_gpu_blocks", 0)
                if num_gpu_blocks:
                    self.vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

            logger.info(
                "[StageCoreClient-%d] Handshake complete",
                self.stage_id,
            )
        finally:
            handshake_socket.close(linger=0)
            ctx.term()

    # ---- Async communication methods ----

    async def add_request(self, request: EngineCoreRequest) -> None:
        """Send a request to the stage engine."""
        await self._mp_client.add_request_async(request)

    async def get_output(self) -> EngineCoreOutputs:
        """Receive outputs from the stage engine."""
        return await self._mp_client.get_output_async()

    async def abort_requests(self, request_ids: list[str]) -> None:
        """Abort requests in the stage engine."""
        await self._mp_client.abort_requests_async(request_ids)

    async def profile(self, is_start: bool = True) -> None:
        """Start or stop profiling."""
        await self._mp_client.profile_async(is_start)

    async def reset_mm_cache(self) -> None:
        await self._mp_client.reset_mm_cache_async()

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        return await self._mp_client.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def reset_encoder_cache(self) -> None:
        await self._mp_client.reset_encoder_cache_async()

    async def sleep(self, level: int = 1) -> None:
        await self._mp_client.sleep_async(level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self._mp_client.wake_up_async(tags)

    async def is_sleeping(self) -> bool:
        return await self._mp_client.is_sleeping_async()

    def shutdown(self) -> None:
        """Shutdown the stage engine process."""
        self._mp_client.shutdown()

    @property
    def is_alive(self) -> bool:
        """Check if the engine process is still alive."""
        return not self._mp_client.resources.engine_dead
