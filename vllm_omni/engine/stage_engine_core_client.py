"""
Stage Engine Core Client for vLLM-Omni V1 architecture.

Inherits from vLLM's AsyncMPClient and uses :class:`StageCoreProc` as its
subprocess engine, ensuring stage-specific behaviour (DP=1, stage-scoped
tracing, etc.) is applied automatically.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import AsyncMPClient

from vllm_omni.engine.stage_core_proc import StageCoreProc
from vllm_omni.engine.stage_init import StageMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.engine.utils import CoreEngineProcManager, EngineZmqAddresses

    from vllm_omni.engine.stage_init import StartedLlmStage
    from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


class StageEngineCoreClient(AsyncMPClient):
    """Async EngineCore client for a single omni pipeline stage.

    Inherits from vLLM's ``AsyncMPClient`` for ZMQ communication, output
    queues, and all standard utility methods.

    Relationship with :class:`StageCoreProc`:

    * **Self-managed mode** – when *client_addresses* is ``None``, the
      constructor automatically launches a ``StageCoreProc`` subprocess
      and connects to it via ZMQ.
    * **Externally-managed mode** – when *client_addresses* is supplied
      (i.e. the process was already launched by the caller via
      :meth:`launch_engine`), the constructor simply connects to the
      running process.

    Use the :meth:`launch_engine` / :meth:`from_started_stage` class
    methods for a two-phase initialisation flow (useful when stages need
    to be started in parallel with device-lock coordination).
    """

    ENGINE_CORE_PROC_CLASS: type[StageCoreProc] = StageCoreProc
    """The subprocess class used to run the engine core process."""

    def __init__(
        self,
        vllm_config: Any,
        executor_class: type,
        metadata: StageMetadata,
        client_addresses: dict[str, str] | None = None,
        engine_manager: Any = None,
        coordinator: Any = None,
    ):
        """Create an async EngineCore client for a single stage.

        Args:
            vllm_config: Engine configuration.
            executor_class: Executor class for the engine.
            metadata: Stage metadata extracted from the stage config.
            client_addresses: ZMQ addresses of an already-running engine.
                If *None*, a :class:`StageCoreProc` process is launched
                automatically (self-managed mode).
            engine_manager: Pre-existing engine process manager (used in
                externally-managed mode to hand ownership to this client).
            coordinator: Pre-existing DP coordinator (always ``None`` for
                stage engines).
        """
        # -------- Stage metadata (public fields used at runtime) --------
        self._metadata = metadata
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

        # -------- Self-managed mode: launch StageCoreProc --------
        if client_addresses is None:
            logger.info(
                "[StageEngineCoreClient] Stage-%s launching %s",
                self.stage_id,
                self.ENGINE_CORE_PROC_CLASS.__name__,
            )
            from vllm.v1.engine.utils import get_engine_zmq_addresses

            addresses = get_engine_zmq_addresses(vllm_config)
            with self.ENGINE_CORE_PROC_CLASS.launch_core_engines(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
                addresses=addresses,
            ) as (em, coord, result_addresses):
                pass  # wait_for_engine_startup runs on context exit

            engine_manager = em
            coordinator = coord
            client_addresses = {
                "input_address": result_addresses.inputs[0],
                "output_address": result_addresses.outputs[0],
            }
            if result_addresses.frontend_stats_publish_address is not None:
                client_addresses["stats_update_address"] = (
                    result_addresses.frontend_stats_publish_address
                )

        # -------- Connect to the running engine --------
        logger.info(
            "[StageEngineCoreClient] Stage-%s initializing EngineCore",
            self.stage_id,
        )
        try:
            super().__init__(
                vllm_config,
                executor_class,
                log_stats=False,
                client_addresses=client_addresses,
            )
            if engine_manager is not None:
                self.resources.engine_manager = engine_manager
            if coordinator is not None:
                self.resources.coordinator = coordinator
        except Exception:
            logger.exception(
                "[StageEngineCoreClient] Stage-%s EngineCore init failed",
                self.stage_id,
            )
            try:
                self.shutdown()
            except Exception as shutdown_error:
                logger.warning(
                    "[StageEngineCoreClient] Stage-%s cleanup after init failure failed: %s",
                    self.stage_id,
                    shutdown_error,
                )
            raise
        logger.info(
            "[StageEngineCoreClient] Stage-%s EngineCore running",
            self.stage_id,
        )

    # ==================== Factory Methods ====================

    @classmethod
    @contextmanager
    def launch_engine(
        cls,
        vllm_config: VllmConfig,
        executor_class: type,
        log_stats: bool = False,
        addresses: EngineZmqAddresses | None = None,
    ) -> Iterator[tuple[CoreEngineProcManager | None, None, EngineZmqAddresses]]:
        """Context manager that launches the engine process.

        Delegates to :meth:`StageCoreProc.launch_core_engines` via the
        class-level :attr:`ENGINE_CORE_PROC_CLASS`, so that subclasses
        can override the proc class if needed.

        This is the recommended entry-point for the *launch* phase of the
        two-phase initialisation pattern used by
        :class:`~vllm_omni.engine.async_omni_engine.AsyncOmniEngine`.

        Yields:
            ``(engine_manager, None, addresses)``
        """
        with cls.ENGINE_CORE_PROC_CLASS.launch_core_engines(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            addresses=addresses,
        ) as result:
            yield result

    @classmethod
    def from_started_stage(cls, started: StartedLlmStage) -> StageEngineCoreClient:
        """Create a client from an already-launched stage.

        This is the *attach* phase of the two-phase initialisation
        pattern.  The engine process has already been started (via
        :meth:`launch_engine`) and the startup handshake is complete.

        Ownership of *started.engine_manager* and *started.coordinator*
        is transferred to the new client; the corresponding fields on
        *started* are set to ``None``.

        Args:
            started: Resources for a launched stage.

        Returns:
            A fully connected :class:`StageEngineCoreClient`.
        """
        client_addresses: dict[str, str] = {
            "input_address": started.addresses.inputs[0],
            "output_address": started.addresses.outputs[0],
        }
        if started.addresses.frontend_stats_publish_address is not None:
            client_addresses["stats_update_address"] = (
                started.addresses.frontend_stats_publish_address
            )

        client = cls(
            vllm_config=started.vllm_config,
            executor_class=started.executor_class,
            metadata=started.metadata,
            client_addresses=client_addresses,
            engine_manager=started.engine_manager,
            coordinator=started.coordinator,
        )
        # Ownership transferred to client.
        started.engine_manager = None
        started.coordinator = None
        return client

    # ==================== Overrides ====================

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        """Add request to the stage engine core."""
        logger.info(f"[StageEngineCoreClient] Stage-{self.stage_id} adding request: {request.request_id}")
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
