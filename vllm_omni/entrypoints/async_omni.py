# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncOmni: thin EngineClient wrapper around PipelineOrchestrator.

Implements the EngineClient protocol expected by the OpenAI-compatible
API server, delegating all pipeline logic to PipelineOrchestrator.
"""

import asyncio
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Any

from vllm.config import VllmConfig
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config import OmniModelConfig
from vllm_omni.entrypoints.pipeline_orchestrator import PipelineOrchestrator
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class AsyncOmni:
    """Asynchronous unified entry point supporting multi-stage pipelines.

    Thin wrapper around PipelineOrchestrator that implements the
    EngineClient protocol for use with the OpenAI API server.

    Args:
        model: Model name or path to load.
        **kwargs: Configuration keyword arguments forwarded to
            PipelineOrchestrator (stage_configs_path, log_stats,
            stage_init_timeout, shm_threshold_bytes, worker_backend,
            batch_timeout, init_timeout, etc.)

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
        # Pause/resume control
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False

        # Create the orchestrator (does all init)
        self.orchestrator = PipelineOrchestrator(model, **kwargs)

        # Expose attributes that the API server / EngineClient protocol expects
        # These are populated by the orchestrator after stage init
        self.input_processor = self.orchestrator.input_processor
        self.io_processor = self.orchestrator.io_processor
        self.model_config = self.orchestrator.model_config

    # ----- Orchestrator-delegated properties -----

    @property
    def stage_list(self):
        return self.orchestrator.stage_list

    @property
    def stage_configs(self):
        return self.orchestrator.stage_configs

    @property
    def default_sampling_params_list(self):
        return self.orchestrator.default_sampling_params_list

    @property
    def output_modalities(self):
        return self.orchestrator.output_modalities

    @property
    def log_stats(self):
        return self.orchestrator.log_stats

    @property
    def renderer(self):
        """Return renderer from input_processor.

        Required by OpenAIServingModels.__init__ which accesses
        engine_client.renderer.
        """
        # Check local input_processor first (API server may set it directly)
        if self.input_processor is not None:
            return self.input_processor.renderer
        return self.orchestrator.renderer

    # ----- EngineClient status properties -----

    @property
    def is_running(self) -> bool:
        return len(self.orchestrator.stage_list) > 0

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    # ----- Generation -----

    async def generate(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: Sequence[OmniSamplingParams] | None = None,
        *,
        output_modalities: list[str] | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt asynchronously.

        Coordinates multi-stage pipeline. Yields OmniRequestOutput as
        outputs become available from each stage.

        Args:
            prompt: Input prompt (text, token IDs, or multimodal).
            request_id: Unique identifier for this request.
            sampling_params_list: Per-stage sampling parameters.
            output_modalities: Optional output modality filter.

        Yields:
            OmniRequestOutput objects from each stage.
        """
        # Wait until resumed if paused
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        async for output in self.orchestrator.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=output_modalities,
        ):
            yield output

    # ----- EngineClient protocol methods -----

    async def abort(self, request_id: str | Iterable[str]) -> None:
        await self.orchestrator.abort(request_id)

    async def get_vllm_config(self) -> VllmConfig | None:
        return await self.orchestrator.get_vllm_config()

    async def get_model_config(self) -> OmniModelConfig | None:
        return await self.orchestrator.get_model_config()

    async def get_input_preprocessor(self) -> InputPreprocessor | None:
        return await self.orchestrator.get_input_preprocessor()

    async def get_tokenizer(self) -> TokenizerLike | None:
        return await self.orchestrator.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        return await self.orchestrator.is_tracing_enabled()

    async def do_log_stats(self) -> None:
        pass

    async def check_health(self) -> None:
        pass

    async def reset_mm_cache(self) -> None:
        pass

    async def reset_prefix_cache(self, reset_running_requests: bool = False) -> bool:
        pass

    async def sleep(self, level: int = 1) -> None:
        pass

    async def wake_up(self, tags: list[str] | None = None) -> None:
        pass

    async def is_sleeping(self) -> bool:
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    async def encode(self, *args, **kwargs):
        raise NotImplementedError("encode() is not implemented for AsyncOmni")

    # ----- Profiling -----

    async def start_profile(self, stages: list[int] | None = None) -> None:
        self.orchestrator.start_profile(stages)

    async def stop_profile(self, stages: list[int] | None = None) -> None:
        self.orchestrator.stop_profile(stages)

    # ----- Pause/Resume -----

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause generation. New requests block until resumed."""
        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        if clear_cache:
            await self.reset_prefix_cache()
            await self.reset_mm_cache()

    async def resume_generation(self) -> None:
        """Resume generation after pause_generation()."""
        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        async with self._pause_cond:
            return self._paused

    # ----- Lifecycle -----

    def shutdown(self) -> None:
        """Shutdown all stages and clean up resources."""
        self.orchestrator.shutdown()

    def close(self) -> None:
        """Alias for shutdown()."""
        self.shutdown()
