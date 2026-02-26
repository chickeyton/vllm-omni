# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
StageCoreClient: Client for a single pipeline stage.

Manages the lifecycle of an LLM stage (StageCoreProc via ZMQ) or a diffusion
stage (AsyncOmniDiffusion in-process). Each pipeline stage has one
StageCoreClient instance in the orchestrator process.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields
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
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

# Handshake polling period (milliseconds)
_HANDSHAKE_POLL_MS = 2000
# Handshake timeout (milliseconds) — 10 minutes
_HANDSHAKE_TIMEOUT_MS = 600_000


# ---------------------------------------------------------------------------
# Lightweight wrappers so diffusion results duck-type EngineCoreOutputs
# ---------------------------------------------------------------------------

@dataclass
class _DiffusionOutput:
    """Duck-types ``EngineCoreOutput`` for a single diffusion result."""

    request_id: str
    finished: bool = True
    result: OmniRequestOutput | None = None


@dataclass
class _DiffusionOutputs:
    """Duck-types ``EngineCoreOutputs`` for a batch of diffusion results."""

    outputs: list[_DiffusionOutput] = field(default_factory=list)


# ---------------------------------------------------------------------------
# StageCoreClient
# ---------------------------------------------------------------------------

class StageCoreClient:
    """Client for a single pipeline stage (LLM or diffusion).

    For **LLM** stages the client:
    - Launches one StageCoreProc in a dedicated background process
    - Performs a 3-phase ZMQ handshake (HELLO → INIT → READY)
    - Creates an AsyncMPClient that handles ZMQ I/O

    For **diffusion** stages the client:
    - Creates an AsyncOmniDiffusion engine in-process
    - Exposes ``submit_diffusion_request`` for prompt-based submission
    - Wraps diffusion results so ``get_output`` works uniformly

    Args:
        vllm_config: VllmConfig for this stage's model (LLM only).
        executor_class: Executor implementation class (LLM only).
        log_stats: Whether to log statistics.
        stage_id: Unique identifier for this stage.
        stage_type: ``"llm"`` or ``"diffusion"``.
        devices: Device specification (e.g. ``"0,1"`` for GPUs).
        connectors_config: Configuration for stage connectors.
        stage_init_timeout: Timeout for stage initialization.
        model: Model name or path (diffusion only).
        engine_args_raw: Raw engine args dict (diffusion only).
        engine_input_source: Upstream stage IDs for data flow (diffusion only).
    """

    def __init__(
        self,
        vllm_config: VllmConfig | None = None,
        executor_class: type[Executor] | None = None,
        log_stats: bool = False,
        *,
        stage_id: int = 0,
        stage_type: str = "llm",
        devices: str | None = None,
        connectors_config: dict[str, Any] | None = None,
        stage_init_timeout: int = 300,
        # Diffusion-specific
        model: str | None = None,
        engine_args_raw: dict[str, Any] | None = None,
        engine_input_source: list[int] | None = None,
    ) -> None:
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.devices = devices
        self.vllm_config = vllm_config
        self._is_diffusion = stage_type == "diffusion"

        if self._is_diffusion:
            self._init_diffusion(
                model=model,
                engine_args_raw=engine_args_raw or {},
                stage_id=stage_id,
                devices=devices,
                engine_input_source=engine_input_source or [],
            )
        else:
            assert vllm_config is not None, "vllm_config required for LLM stages"
            assert executor_class is not None, "executor_class required for LLM stages"
            self._init_llm(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                stage_id=stage_id,
                stage_type=stage_type,
                devices=devices,
                connectors_config=connectors_config,
                stage_init_timeout=stage_init_timeout,
            )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_llm(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        stage_id: int,
        stage_type: str,
        devices: str | None,
        connectors_config: dict[str, Any] | None,
        stage_init_timeout: int,
    ) -> None:
        """Set up an LLM stage backed by StageCoreProc + ZMQ."""

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
            "[StageCoreClient-%d] LLM initialized (devices=%s)",
            stage_id,
            devices,
        )

    def _init_diffusion(
        self,
        model: str | None,
        engine_args_raw: dict[str, Any],
        stage_id: int,
        devices: str | None,
        engine_input_source: list[int],
    ) -> None:
        """Set up a diffusion stage backed by AsyncOmniDiffusion."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
        from vllm_omni.entrypoints.stage_utils import set_stage_devices

        # 1. Set device visibility before loading the model.
        if devices is not None:
            set_stage_devices(stage_id, devices)
            logger.info(
                "[StageCoreClient-%d] Diffusion device setup: devices=%s",
                stage_id,
                devices,
            )

        # 2. Build OmniDiffusionConfig from engine args.
        od_config = engine_args_raw.get("od_config", {})
        if not od_config:
            od_config = {"model": model or ""}
            od_field_names = {f.name for f in fields(OmniDiffusionConfig)}
            for key, value in engine_args_raw.items():
                if key in od_field_names:
                    od_config[key] = value

        # 3. Inject omni_kv_config so the diffusion worker can find stage info.
        if "omni_kv_config" not in od_config:
            od_config["omni_kv_config"] = {}
        od_config["omni_kv_config"]["stage_id"] = stage_id
        od_config["omni_kv_config"]["engine_input_source"] = engine_input_source

        # 4. Create AsyncOmniDiffusion engine.
        extra_args = {
            k: v
            for k, v in engine_args_raw.items()
            if k not in {"od_config", "model"}
        }
        self._diffusion_engine = AsyncOmniDiffusion(
            model=model or "",
            od_config=od_config,
            **extra_args,
        )

        # 5. Result queue for get_output() interop.
        self._diffusion_results: asyncio.Queue[_DiffusionOutput] = asyncio.Queue()

        logger.info(
            "[StageCoreClient-%d] Diffusion initialized (devices=%s)",
            stage_id,
            devices,
        )

    # ------------------------------------------------------------------
    # ZMQ handshake (LLM only)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Async communication methods
    # ------------------------------------------------------------------

    async def add_request(self, request: EngineCoreRequest) -> None:
        """Send an ``EngineCoreRequest`` to an LLM stage engine.

        Raises:
            TypeError: If called on a diffusion stage.
        """
        if self._is_diffusion:
            raise TypeError(
                "Diffusion stages do not accept EngineCoreRequest. "
                "Use submit_diffusion_request() instead."
            )
        await self._mp_client.add_request_async(request)

    async def submit_diffusion_request(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        lora_request: Any | None = None,
    ) -> None:
        """Submit a request to a diffusion stage and queue the result.

        The diffusion engine runs generation (potentially blocking in a
        thread-pool) and the result is placed on an internal queue so that
        ``get_output`` can retrieve it.

        Args:
            prompt: Input prompt (text or multimodal dict).
            sampling_params: ``OmniDiffusionSamplingParams``.
            request_id: Unique request identifier.
            lora_request: Optional LoRA request.
        """
        if not self._is_diffusion:
            raise TypeError(
                "submit_diffusion_request is only valid for diffusion stages."
            )
        result = await self._diffusion_engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
        wrapped = _DiffusionOutput(
            request_id=result.request_id or request_id,
            finished=True,
            result=result,
        )
        await self._diffusion_results.put(wrapped)

    async def get_output(self) -> EngineCoreOutputs | _DiffusionOutputs:
        """Receive outputs from the stage engine.

        For LLM stages, delegates to the internal ``AsyncMPClient``.
        For diffusion stages, returns the next result from the internal queue
        wrapped in ``_DiffusionOutputs``.
        """
        if self._is_diffusion:
            output = await self._diffusion_results.get()
            return _DiffusionOutputs(outputs=[output])
        return await self._mp_client.get_output_async()

    async def abort_requests(self, request_ids: list[str]) -> None:
        """Abort requests in the stage engine."""
        if self._is_diffusion:
            for rid in request_ids:
                await self._diffusion_engine.abort(rid)
            return
        await self._mp_client.abort_requests_async(request_ids)

    async def profile(self, is_start: bool = True) -> None:
        """Start or stop profiling."""
        if self._is_diffusion:
            engine = self._diffusion_engine.engine
            if is_start:
                engine.start_profile()
            else:
                engine.stop_profile()
            return
        await self._mp_client.profile_async(is_start)

    async def reset_mm_cache(self) -> None:
        if self._is_diffusion:
            return  # no-op for diffusion
        await self._mp_client.reset_mm_cache_async()

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        if self._is_diffusion:
            return True  # no-op for diffusion
        return await self._mp_client.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def reset_encoder_cache(self) -> None:
        if self._is_diffusion:
            return  # no-op for diffusion
        await self._mp_client.reset_encoder_cache_async()

    async def sleep(self, level: int = 1) -> None:
        if self._is_diffusion:
            return  # no-op for diffusion
        await self._mp_client.sleep_async(level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        if self._is_diffusion:
            return  # no-op for diffusion
        await self._mp_client.wake_up_async(tags)

    async def is_sleeping(self) -> bool:
        if self._is_diffusion:
            return False
        return await self._mp_client.is_sleeping_async()

    def shutdown(self) -> None:
        """Shutdown the stage engine."""
        if self._is_diffusion:
            self._diffusion_engine.shutdown()
            return
        self._mp_client.shutdown()

    @property
    def is_alive(self) -> bool:
        """Check if the engine is still alive."""
        if self._is_diffusion:
            return not self._diffusion_engine._closed
        return not self._mp_client.resources.engine_dead

    @property
    def is_diffusion(self) -> bool:
        """Whether this client manages a diffusion stage."""
        return self._is_diffusion
