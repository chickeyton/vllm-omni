"""
Stage Engine Core Client for vLLM-Omni V1 architecture.

Manages a StageCoreProc subprocess and communicates with it via ZMQ.
Inherits from vLLM's AsyncMPClient for the ZMQ client functionality.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import zmq
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineProcManager,
    EngineZmqAddresses,
    wait_for_engine_startup,
)
from vllm.v1.utils import get_engine_client_zmq_addr
from vllm.utils.network_utils import zmq_socket_ctx

from vllm_omni.engine.stage_core_proc import StageCoreProc
from vllm_omni.engine.stage_init import StageMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.executor import Executor

    from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


class StageEngineCoreClient(AsyncMPClient):
    """Stage async client that manages a StageCoreProc subprocess.

    Encapsulates the full lifecycle of a single pipeline stage:
    - Launches StageCoreProc as a child process with stage-specific setup
    - ZMQ-based communication with the subprocess (inherited from AsyncMPClient)
    - Stage metadata management for orchestrator use

    The subprocess runs StageCoreProc.run_stage_core which performs
    stage-specific device setup, plugin loading, and worker class
    resolution before delegating to EngineCoreProc.run_engine_core
    for the standard ZMQ server loop.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        metadata: StageMetadata,
        *,
        devices: str | None = None,
    ):
        """Create an async EngineCore client for a single stage.

        Launches a StageCoreProc subprocess, performs the engine startup
        handshake, and initializes ZMQ client sockets for communication.

        Args:
            vllm_config: Pre-built vLLM configuration for this stage.
            executor_class: Executor implementation class.
            metadata: Stage metadata extracted from stage config.
            devices: Device specification (e.g. "0,1") for the stage
                subprocess. Passed to StageCoreProc for CUDA_VISIBLE_DEVICES
                setup.
        """
        # -------- Stage metadata (public fields used at runtime) --------
        self.stage_id = metadata.stage_id
        self.stage_type = metadata.stage_type
        self.engine_output_type = metadata.engine_output_type
        self.is_comprehension = metadata.is_comprehension
        self.requires_multimodal_data = metadata.requires_multimodal_data
        self.engine_input_source = metadata.engine_input_source
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.custom_process_input_func = metadata.custom_process_input_func
        self.model_stage = metadata.model_stage

        self.engine_outputs: Any = None

        logger.info(
            "[StageEngineCoreClient] Stage-%s launching StageCoreProc subprocess",
            self.stage_id,
        )

        # -------- Launch StageCoreProc subprocess --------
        engine_manager, addresses = self._launch_stage_proc(
            vllm_config=vllm_config,
            executor_class=executor_class,
            metadata=metadata,
            devices=devices,
        )

        # -------- Connect to subprocess via ZMQ (parent class) --------
        client_addresses = {
            "input_address": addresses.inputs[0],
            "output_address": addresses.outputs[0],
        }
        if addresses.frontend_stats_publish_address is not None:
            client_addresses["stats_update_address"] = (
                addresses.frontend_stats_publish_address
            )

        try:
            super().__init__(
                vllm_config,
                executor_class,
                log_stats=False,
                client_addresses=client_addresses,
            )
            # Transfer process lifecycle management to BackgroundResources
            # so that shutdown() cleans up the subprocess automatically.
            self.resources.engine_manager = engine_manager
        except Exception:
            logger.exception(
                "[StageEngineCoreClient] Stage-%s ZMQ client init failed",
                self.stage_id,
            )
            engine_manager.close()
            raise

        logger.info(
            "[StageEngineCoreClient] Stage-%s StageCoreProc running",
            self.stage_id,
        )

    # ==================== Subprocess Management ====================

    @staticmethod
    def _launch_stage_proc(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        metadata: StageMetadata,
        devices: str | None,
    ) -> tuple[CoreEngineProcManager, EngineZmqAddresses]:
        """Launch StageCoreProc in a subprocess and wait for startup.

        Creates ZMQ addresses, spawns the subprocess with stage-specific
        parameters via functools.partial, and performs the engine startup
        handshake (HELLO -> init -> READY).

        Args:
            vllm_config: Pre-built vLLM configuration.
            executor_class: Executor implementation class.
            metadata: Stage metadata for StageCoreProc setup.
            devices: Device specification string for the subprocess.

        Returns:
            (engine_manager, addresses) tuple. The engine_manager owns the
            subprocess and should be assigned to BackgroundResources for
            lifecycle management.
        """
        # Generate ZMQ addresses for client <-> subprocess communication.
        # Always local (same machine) so use IPC paths.
        addresses = EngineZmqAddresses(
            inputs=[get_engine_client_zmq_addr(True, "127.0.0.1")],
            outputs=[get_engine_client_zmq_addr(True, "127.0.0.1")],
        )
        handshake_address = get_engine_client_zmq_addr(True, "127.0.0.1")

        # Wrap StageCoreProc.run_stage_core with stage-specific kwargs.
        # CoreEngineProcManager will supply the standard kwargs
        # (vllm_config, handshake_address, executor_class, etc.).
        target_fn = functools.partial(
            StageCoreProc.run_stage_core,
            stage_id=metadata.stage_id,
            stage_type=metadata.stage_type,
            devices=devices,
        )

        logger.info(
            "[StageEngineCoreClient] Stage-%s spawning subprocess "
            "(type=%s, devices=%s)",
            metadata.stage_id,
            metadata.stage_type,
            devices,
        )

        with zmq_socket_ctx(
            handshake_address, zmq.ROUTER, bind=True
        ) as handshake_socket:
            # Spawn StageCoreProc subprocess via CoreEngineProcManager.
            # Single engine, no data parallelism.
            engine_manager = CoreEngineProcManager(
                target_fn=target_fn,
                local_engine_count=1,
                start_index=0,
                local_start_index=0,
                vllm_config=vllm_config,
                local_client=True,
                handshake_address=handshake_address,
                executor_class=executor_class,
                log_stats=False,
            )

            # Wait for the subprocess to complete startup handshake.
            # The subprocess sends HELLO, we reply with addresses,
            # then it sends READY once initialized.
            wait_for_engine_startup(
                handshake_socket=handshake_socket,
                addresses=addresses,
                core_engines=[CoreEngine(index=0, local=True)],
                parallel_config=vllm_config.parallel_config,
                coordinated_dp=False,
                cache_config=vllm_config.cache_config,
                proc_manager=engine_manager,
                coord_process=None,
            )

        logger.info(
            "[StageEngineCoreClient] Stage-%s subprocess started",
            metadata.stage_id,
        )
        return engine_manager, addresses

    # ==================== Overrides ====================

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        """Add request to the stage engine core."""
        logger.info(
            "[StageEngineCoreClient] Stage-%s adding request: %s",
            self.stage_id,
            request.request_id,
        )
        await super().add_request_async(request)

    # ==================== Stage Methods ====================

    def set_engine_outputs(self, engine_outputs: EngineCoreOutput) -> None:
        """Set engine outputs (called by orchestrator)."""
        self.engine_outputs = engine_outputs

    def process_engine_inputs(
        self,
        stage_list: list[Any],
        prompt: OmniTokensPrompt | list[OmniTokensPrompt] | None = None,
    ) -> list[OmniTokensPrompt]:
        """Process inputs from upstream stages."""
        from vllm_omni.inputs.data import OmniTokensPrompt

        if self.custom_process_input_func is not None:
            return self.custom_process_input_func(
                stage_list,
                self.engine_input_source,
                prompt,
                self.requires_multimodal_data,
            )

        if not self.engine_input_source:
            raise ValueError(f"engine_input_source empty for stage {self.stage_id}")

        source_id = self.engine_input_source[0]
        source_outputs = stage_list[source_id].engine_outputs

        if not isinstance(prompt, list):
            prompt = [prompt]

        mm_data = {so.request_id: p.get("multi_modal_data") for so, p in zip(source_outputs, prompt)}

        return [
            OmniTokensPrompt(
                prompt_token_ids=so.outputs[0].token_ids,
                multi_modal_data=(mm_data[so.request_id] if self.requires_multimodal_data else None),
            )
            for so in source_outputs
        ]

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Forward control RPCs to the underlying AsyncMPClient stage engine.

        Each ``StageEngineCoreClient`` already represents one logical stage, so
        stage-scoped control operations should be executed here and then fanned
        in-core across the workers managed by this EngineCore client.
        """
        return await super().collective_rpc_async(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )
