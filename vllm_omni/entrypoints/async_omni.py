# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncOmni: Asynchronous unified entry point for multi-stage pipelines.

Implements vLLM's EngineClient protocol, delegating multi-stage pipeline
orchestration to PipelineOrchestrator. This is the top-level async
interface used by the serving layer (OpenAI API server).
"""
from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Any

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config import OmniModelConfig
from vllm_omni.entrypoints.omni import omni_snapshot_download
from vllm_omni.entrypoints.pipeline_orchestrator import PipelineOrchestrator
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _weak_shutdown(orchestrator: PipelineOrchestrator) -> None:
    """Weak reference cleanup for AsyncOmni instances."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(orchestrator.shutdown())
        else:
            loop.run_until_complete(orchestrator.shutdown())
    except Exception as e:
        logger.debug("AsyncOmni cleanup failed: %s", e)


class AsyncOmni(EngineClient):
    """Asynchronous unified entry point for multi-stage pipelines.

    Implements vLLM's EngineClient protocol, delegating pipeline
    orchestration to PipelineOrchestrator. Each pipeline stage runs
    a StageCoreProc in a dedicated background process, communicating
    via ZMQ.

    Args:
        model: Model name or path to load.
        **kwargs: Pipeline configuration arguments.
            - stage_configs_path: Path to YAML stage config file.
            - log_stats: Whether to enable statistics logging.
            - stage_init_timeout: Per-stage init timeout (seconds).
            - shm_threshold_bytes: SHM threshold for IPC.
            - worker_backend: Backend type ("multi_process" or "ray").
            - init_timeout: Global init timeout (seconds).
            - Additional kwargs passed to stage engines.

    Example:
        >>> async_omni = AsyncOmni(model="Qwen/Qwen2.5-Omni-7B")
        >>> async for output in async_omni.generate(
        ...     prompt="Hello",
        ...     request_id="req-1",
        ...     sampling_params_list=[SamplingParams(), SamplingParams()]
        ... ):
        ...     print(output)
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        model = omni_snapshot_download(model)

        # Pause/resume control
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False

        # Load stage configurations
        stage_configs_path = kwargs.get("stage_configs_path")
        if stage_configs_path is None:
            config_path = resolve_model_config_path(model)
            stage_configs = load_stage_configs_from_model(model)
        else:
            config_path = stage_configs_path
            stage_configs = load_stage_configs_from_yaml(stage_configs_path)

        # Create the pipeline orchestrator
        self.orchestrator = PipelineOrchestrator(
            model=model,
            stage_configs=stage_configs,
            config_path=config_path,
            log_stats=kwargs.get("log_stats", False),
            stage_init_timeout=kwargs.get("stage_init_timeout", 20),
            init_timeout=kwargs.get("init_timeout", 300),
            shm_threshold_bytes=kwargs.get("shm_threshold_bytes", 65536),
            worker_backend=kwargs.get("worker_backend", "multi_process"),
        )

        # Set up processors from orchestrator
        self.input_processor = self.orchestrator.input_processor
        self.io_processor = getattr(self.orchestrator, "io_processor", None)
        self.model_config = getattr(self.orchestrator, "model_config", None)

        # Store vllm_config from comprehension stage
        comp_stage = self.orchestrator.get_comprehension_stage()
        if comp_stage and comp_stage.vllm_config:
            self.vllm_config = comp_stage.vllm_config
        else:
            # Fall back to first LLM stage
            for info in self.orchestrator.stage_infos:
                if info.vllm_config is not None:
                    self.vllm_config = info.vllm_config
                    break
            else:
                self.vllm_config = None

        # Register cleanup
        self._weak_finalizer = weakref.finalize(
            self, _weak_shutdown, self.orchestrator
        )

        logger.info(
            "[AsyncOmni] Initialized with %d stages",
            len(self.orchestrator.stage_infos),
        )

    # ---- EngineClient abstract properties ----

    @property
    def is_running(self) -> bool:
        return len(self.orchestrator.stage_clients) > 0

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    # ---- EngineClient abstract methods ----

    async def generate(
        self,
        prompt: OmniPromptType,
        sampling_params: SamplingParams | None = None,
        request_id: str = "",
        *,
        sampling_params_list: Sequence[OmniSamplingParams] | None = None,
        output_modalities: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs through the multi-stage pipeline.

        Supports both EngineClient-style single sampling_params and
        omni-style sampling_params_list (one per stage).

        Args:
            prompt: Input prompt (text, tokens, or multimodal).
            sampling_params: Single sampling params (EngineClient compat).
            request_id: Unique request identifier.
            sampling_params_list: Per-stage sampling params (omni-style).
            output_modalities: Which output modalities to produce.
            **kwargs: Additional arguments (lora_request, etc. — not yet supported).

        Yields:
            OmniRequestOutput objects as produced by each stage.
        """
        # Wait until generation is resumed if paused
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        # Resolve sampling params
        if sampling_params_list is None:
            if sampling_params is not None:
                sampling_params_list = [sampling_params] * len(
                    self.orchestrator.stage_infos
                )
            else:
                sampling_params_list = self.orchestrator.default_sampling_params_list

        if len(sampling_params_list) != len(self.orchestrator.stage_infos):
            raise ValueError(
                f"Expected {len(self.orchestrator.stage_infos)} sampling params, "
                f"got {len(sampling_params_list)}"
            )

        try:
            async for output in self.orchestrator.process_request(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=list(sampling_params_list),
                output_modalities=output_modalities,
            ):
                yield output
        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            logger.info("[AsyncOmni] Request %s aborted.", request_id)
            raise

    async def encode(self, *args: Any, **kwargs: Any) -> AsyncGenerator:
        """Encode is not supported for AsyncOmni."""
        raise NotImplementedError("encode() is not supported for AsyncOmni")

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort one or more requests across all stages."""
        if isinstance(request_id, str):
            await self.orchestrator.abort(request_id)
        else:
            for rid in request_id:
                await self.orchestrator.abort(rid)

    async def is_tracing_enabled(self) -> bool:
        comp = self.orchestrator.get_comprehension_stage()
        if comp:
            return comp.is_tracing_enabled
        return False

    async def do_log_stats(self) -> None:
        pass

    async def check_health(self) -> None:
        pass

    async def start_profile(self) -> None:
        await self.orchestrator.start_profile()

    async def stop_profile(self) -> None:
        await self.orchestrator.stop_profile()

    async def reset_mm_cache(self) -> None:
        for client in self.orchestrator.stage_clients:
            await client.reset_mm_cache()

    async def reset_encoder_cache(self) -> None:
        for client in self.orchestrator.stage_clients:
            await client.reset_encoder_cache()

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        for client in self.orchestrator.stage_clients:
            await client.reset_prefix_cache(
                reset_running_requests, reset_connector
            )
        return True

    async def sleep(self, level: int = 1) -> None:
        for client in self.orchestrator.stage_clients:
            await client.sleep(level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        for client in self.orchestrator.stage_clients:
            await client.wake_up(tags)

    async def is_sleeping(self) -> bool:
        for client in self.orchestrator.stage_clients:
            if await client.is_sleeping():
                return True
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    async def pause_generation(
        self,
        *,
        mode: str = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause generation to allow model weight updates."""
        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        if clear_cache:
            await self.reset_prefix_cache()
            await self.reset_mm_cache()

    async def resume_generation(self) -> None:
        """Resume generation after pause_generation."""
        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        async with self._pause_cond:
            return self._paused

    def shutdown(self) -> None:
        """Shutdown the pipeline."""
        if hasattr(self, "_weak_finalizer"):
            self._weak_finalizer()

    # ---- Backward compatibility with serving layer ----

    @property
    def stage_configs(self):
        """Stage configurations (backward compat)."""
        return self.orchestrator.stage_configs

    @property
    def stage_list(self):
        """Stage info list (backward compat with serving layer)."""
        return self.orchestrator.stage_list

    @property
    def default_sampling_params_list(self):
        """Default sampling params per stage (backward compat)."""
        return self.orchestrator.default_sampling_params_list

    @property
    def output_modalities(self):
        """Output modalities per stage."""
        return self.orchestrator.output_modalities

    @property
    def log_stats(self) -> bool:
        return self.orchestrator.log_stats

    @property
    def renderer(self):
        """Return the renderer from input_processor if available.

        Required by OpenAIServingModels.__init__ which accesses
        engine_client.renderer.
        """
        if self.input_processor is not None:
            return self.input_processor.renderer
        return None

    @property
    def connectors(self):
        """Connectors dict (backward compat)."""
        return self.orchestrator.connectors

    async def get_vllm_config(self) -> VllmConfig | None:
        return self.vllm_config

    async def get_model_config(self) -> OmniModelConfig | None:
        return self.model_config

    async def get_input_preprocessor(self) -> InputPreprocessor | None:
        return None

    async def get_tokenizer(self) -> TokenizerLike | None:
        comp = self.orchestrator.get_comprehension_stage()
        if comp and comp.tokenizer:
            return comp.tokenizer
        # Try to get tokenizer from vllm_config
        if self.vllm_config and not self.vllm_config.model_config.skip_tokenizer_init:
            from vllm.tokenizers import cached_tokenizer_from_config
            return cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
        return None
