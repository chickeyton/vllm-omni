# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PipelineOrchestrator: Manages multi-stage pipeline of StageCoreClients.

Replaces OmniBase + OmniStage management logic with a cleaner OOP design.
Handles stage lifecycle, request routing, inter-stage data transformation,
connectors for KV cache transfer, and metrics collection.
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from omegaconf import OmegaConf
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.executor import Executor

from vllm_omni.distributed.omni_connectors import (
    initialize_orchestrator_connectors,
)
from vllm_omni.distributed.omni_connectors.adapter import (
    compute_talker_prompt_ids_length,
    try_send_via_connector,
)
from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    get_stage_connector_config,
    resolve_omni_kv_config_for_stage,
)
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.engine.stage_core_client import StageCoreClient
from vllm_omni.entrypoints.stage_utils import _to_dict
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
    inject_omni_kv_config,
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniPromptType,
    OmniSamplingParams,
    OmniTokensPrompt,
)
from vllm_omni.metrics import OrchestratorAggregator
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


@dataclass
class StageInfo:
    """Metadata about a single pipeline stage.

    Replaces OmniStage for metadata purposes while providing
    duck-type compatibility with the serving layer.
    """

    stage_id: int
    stage_type: Literal["llm", "diffusion"] = "llm"
    model_stage: str = "thinker"
    engine_input_source: list[int] = field(default_factory=list)
    final_output: bool = False
    final_output_type: str | None = None
    default_sampling_params: SamplingParams | OmniDiffusionSamplingParams = field(
        default_factory=SamplingParams
    )
    requires_multimodal_data: bool = False
    custom_process_input_func: Callable | None = None
    is_comprehension: bool = False
    engine_output_type: str | None = None
    async_chunk: bool = False

    # Runtime state
    vllm_config: VllmConfig | None = None
    tokenizer: Any = None
    is_tracing_enabled: bool = False
    engine_outputs: Any = None

    def set_engine_outputs(self, outputs: Any) -> None:
        self.engine_outputs = outputs


class PipelineOrchestrator:
    """Orchestrates a multi-stage pipeline of StageCoreClients.

    Manages the lifecycle of multiple stage engine processes,
    coordinates request flow through the pipeline, and handles
    inter-stage data transformation and KV cache transfer.

    Args:
        model: Model name or path.
        stage_configs: List of stage configuration objects (OmegaConf).
        config_path: Path to the YAML config file (for connector init).
        log_stats: Whether to log statistics.
        stage_init_timeout: Per-stage init timeout in seconds.
        init_timeout: Global init timeout in seconds.
        shm_threshold_bytes: Threshold for shared memory IPC.
        worker_backend: Backend type ("multi_process" or "ray").
    """

    def __init__(
        self,
        model: str,
        stage_configs: list,
        config_path: str,
        log_stats: bool = False,
        stage_init_timeout: int = 300,
        init_timeout: int = 300,
        shm_threshold_bytes: int = 65536,
        worker_backend: str = "multi_process",
    ) -> None:
        self.model = model
        self.stage_configs = stage_configs
        self.config_path = config_path
        self.log_stats = log_stats
        self._stage_init_timeout = stage_init_timeout
        self._init_timeout = init_timeout
        self._shm_threshold_bytes = shm_threshold_bytes
        self._worker_backend = worker_backend

        # Stage clients and metadata
        self.stage_clients: list[StageCoreClient] = []
        self.stage_infos: list[StageInfo] = []

        # Connectors for KV cache transfer
        self.connectors: dict[tuple[str, str], OmniConnectorBase] = {}
        self.omni_transfer_config: Any = None

        # Pipeline state
        self.async_chunk: bool = False
        self.output_modalities: list[str | None] = []
        self.default_sampling_params_list: list[OmniSamplingParams] = []

        # Input/output processors (set during initialization from comprehension stage)
        self.input_processor: OmniInputProcessor | None = None
        self.output_processor: MultimodalOutputProcessor | None = None

        # Initialize the pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize all stage clients and connectors."""
        # 1. Initialize connectors from config
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path,
            worker_backend=self._worker_backend,
            shm_threshold_bytes=self._shm_threshold_bytes,
        )

        # 2. Parse stage configs into StageInfo objects
        self._parse_stage_configs()

        # 3. Detect async_chunk mode
        self.async_chunk = self._is_async_chunk_enabled()

        # 4. Create StageCoreClient for each LLM stage
        self._create_stage_clients()

        # 5. Build default sampling params and output modalities
        self.default_sampling_params_list = [
            info.default_sampling_params for info in self.stage_infos
        ]
        self.output_modalities = [
            info.final_output_type for info in self.stage_infos
        ]

        logger.info(
            "[PipelineOrchestrator] Initialized %d stages",
            len(self.stage_infos),
        )

    def _parse_stage_configs(self) -> None:
        """Parse OmegaConf stage configs into StageInfo objects."""
        for cfg in self.stage_configs:
            engine_args = getattr(cfg, "engine_args", {})
            runtime = getattr(cfg, "runtime", {})

            # Parse default_sampling_params
            stage_type: Literal["llm", "diffusion"] = getattr(cfg, "stage_type", "llm")
            default_sp_dict = _to_dict(getattr(cfg, "default_sampling_params", {}))
            try:
                sp_cls = SamplingParams if stage_type == "llm" else OmniDiffusionSamplingParams
                default_sp = sp_cls(**default_sp_dict)
            except TypeError:
                default_sp = SamplingParams()

            # Parse custom_process_input_func
            custom_func = None
            if hasattr(cfg, "custom_process_input_func"):
                func_path = cfg.custom_process_input_func
                module_path, func_name = func_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                custom_func = getattr(module, func_name)

            info = StageInfo(
                stage_id=getattr(cfg, "stage_id", 0),
                stage_type=stage_type,
                model_stage=getattr(engine_args, "model_stage", "thinker"),
                engine_input_source=list(getattr(cfg, "engine_input_source", [])),
                final_output=getattr(cfg, "final_output", False),
                final_output_type=getattr(cfg, "final_output_type", None),
                default_sampling_params=default_sp,
                requires_multimodal_data=getattr(runtime, "requires_multimodal_data", False),
                custom_process_input_func=custom_func,
                is_comprehension=getattr(cfg, "is_comprehension", False),
                engine_output_type=getattr(engine_args, "engine_output_type", None),
                async_chunk=getattr(engine_args, "async_chunk", False),
            )
            self.stage_infos.append(info)

    def _is_async_chunk_enabled(self) -> bool:
        """Check if async chunk mode is enabled for the pipeline."""
        if self.stage_infos:
            return self.stage_infos[0].async_chunk
        return False

    def _create_stage_clients(self) -> None:
        """Create StageCoreClient for each stage."""
        from vllm_omni.engine.arg_utils import OmniEngineArgs

        for idx, (info, cfg) in enumerate(zip(self.stage_infos, self.stage_configs)):
            if info.stage_type == "diffusion":
                # Diffusion stages are handled separately
                logger.info(
                    "[PipelineOrchestrator] Stage-%d is diffusion, "
                    "using DiffusionStageClient (TODO)",
                    idx,
                )
                # TODO: Create DiffusionStageClient wrapping AsyncOmniDiffusion
                continue

            runtime = getattr(cfg, "runtime", {})
            engine_args_raw = _to_dict(getattr(cfg, "engine_args", {}))
            devices = getattr(runtime, "devices", None)

            # Get connector config for this stage
            stage_connectors_config = get_stage_connector_config(
                self.omni_transfer_config, idx
            )

            # Inject omni_kv_config for in-engine usage
            try:
                omni_conn_cfg, omni_from, omni_to = resolve_omni_kv_config_for_stage(
                    self.omni_transfer_config, idx
                )
                if omni_conn_cfg:
                    # Inject into engine_args_raw
                    engine_args_raw["omni_kv_config"] = {
                        "connector_config": omni_conn_cfg,
                        "omni_from_stage": omni_from,
                        "omni_to_stage": omni_to,
                    }
            except Exception as e:
                logger.debug(
                    "[PipelineOrchestrator] Failed to inject omni connector "
                    "config into stage-%d: %s",
                    idx,
                    e,
                )

            # Inject async_chunk connector spec if needed
            if engine_args_raw.get("async_chunk", False):
                stage_connector_spec = {}
                for v in stage_connectors_config.values():
                    stage_connector_spec = dict(v.get("spec", {}))
                    break
                engine_args_raw["stage_connector_spec"] = stage_connector_spec
                engine_args_raw["stage_id"] = idx

            # Resolve worker class
            self._resolve_worker_cls(engine_args_raw)

            # Build VllmConfig from engine args
            try:
                omni_engine_args = OmniEngineArgs(
                    model=self.model, **engine_args_raw
                )
                vllm_config = omni_engine_args.create_engine_config()
            except Exception as e:
                logger.error(
                    "[PipelineOrchestrator] Failed to create VllmConfig "
                    "for stage-%d: %s",
                    idx,
                    e,
                )
                raise

            # Store vllm_config in stage info
            info.vllm_config = vllm_config

            # Determine executor class
            executor_class = Executor.get_class(vllm_config)

            # Create StageCoreClient
            client = StageCoreClient(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=self.log_stats,
                stage_id=idx,
                stage_type=info.stage_type,
                devices=str(devices) if devices is not None else None,
                connectors_config=stage_connectors_config,
                stage_init_timeout=self._stage_init_timeout,
            )
            self.stage_clients.append(client)

            # Initialize processors from comprehension stage
            if info.is_comprehension:
                self._init_processors(vllm_config, omni_engine_args)

        # If no comprehension stage found, try first LLM stage
        if self.input_processor is None and self.stage_clients:
            first_llm = next(
                (i for i in self.stage_infos if i.stage_type == "llm"), None
            )
            if first_llm and first_llm.vllm_config:
                self._init_processors(first_llm.vllm_config)

    def _init_processors(
        self,
        vllm_config: VllmConfig,
        engine_args: Any = None,
    ) -> None:
        """Initialize input/output processors from a stage's config."""
        from vllm.plugins.io_processors import get_io_processor
        from vllm.tokenizers import cached_tokenizer_from_config

        try:
            self.input_processor = OmniInputProcessor(vllm_config=vllm_config)

            tokenizer = None
            if not vllm_config.model_config.skip_tokenizer_init:
                tokenizer = cached_tokenizer_from_config(
                    model_config=vllm_config.model_config
                )

            engine_output_type = None
            if engine_args is not None:
                engine_output_type = getattr(engine_args, "engine_output_type", None)

            self.output_processor = MultimodalOutputProcessor(
                tokenizer=tokenizer,
                log_stats=self.log_stats,
                engine_core_output_type=engine_output_type,
            )

            # Set model_config and io_processor for serving layer
            self.model_config = vllm_config.model_config
            io_plugin = vllm_config.model_config.io_processor_plugin
            self.io_processor = get_io_processor(vllm_config, io_plugin)

            logger.info("[PipelineOrchestrator] Initialized processors")
        except Exception as e:
            logger.warning(
                "[PipelineOrchestrator] Failed to initialize processors: %s", e
            )

    @staticmethod
    def _resolve_worker_cls(engine_args: dict[str, Any]) -> None:
        """Resolve worker class for a stage based on worker_type."""
        worker_type = engine_args.get("worker_type")
        if not worker_type or engine_args.get("worker_cls"):
            return
        try:
            from vllm_omni.platforms import current_omni_platform

            worker_type = str(worker_type).lower()
            if worker_type == "ar":
                engine_args["worker_cls"] = (
                    current_omni_platform.get_omni_ar_worker_cls()
                )
            elif worker_type == "generation":
                engine_args["worker_cls"] = (
                    current_omni_platform.get_omni_generation_worker_cls()
                )
        except ImportError:
            pass

    # ---- Request Processing ----

    async def process_request(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: list[OmniSamplingParams],
        output_modalities: list[str] | None = None,
        metrics: OrchestratorAggregator | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process a request through the multi-stage pipeline.

        Sends the request to stage 0, collects outputs, transforms
        for the next stage, and repeats until the final stage.
        Yields OmniRequestOutput objects as they become available.

        Args:
            prompt: Input prompt (text, tokens, or multimodal).
            request_id: Unique request identifier.
            sampling_params_list: Sampling params per stage.
            output_modalities: Which output modalities to produce.
            metrics: Optional metrics aggregator.
        """
        num_stages = len(self.stage_infos)
        wall_start = time.time()

        final_stage_id = get_final_stage_id_for_e2e(
            output_modalities, self.output_modalities, self.stage_infos
        )

        if metrics is None:
            metrics = OrchestratorAggregator(
                num_stages=num_stages,
                log_stats=self.log_stats,
                wall_start_ts=wall_start,
                final_stage_id_for_e2e=final_stage_id,
            )

        # Convert prompt to EngineCoreRequest for stage 0
        sp0 = sampling_params_list[0]
        engine_request = self._create_engine_request(prompt, request_id, sp0)

        # Submit to stage 0
        await self.stage_clients[0].add_request(engine_request)
        metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()

        if self.async_chunk:
            async for output in self._process_async_chunk(
                request_id,
                prompt,
                sampling_params_list,
                metrics,
                final_stage_id,
            ):
                yield output
        else:
            async for output in self._process_sequential(
                request_id,
                prompt,
                sampling_params_list,
                metrics,
                final_stage_id,
            ):
                yield output

    async def _process_sequential(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params_list: list[OmniSamplingParams],
        metrics: OrchestratorAggregator,
        final_stage_id: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process request sequentially through stages."""
        for stage_id in range(final_stage_id + 1):
            if stage_id >= len(self.stage_clients):
                break

            client = self.stage_clients[stage_id]
            info = self.stage_infos[stage_id]

            # Collect outputs until stage finishes this request
            finished = False
            accumulated_outputs = []

            while not finished:
                engine_outputs: EngineCoreOutputs = await client.get_output()

                # Process each output in the batch
                for output in engine_outputs.outputs:
                    if output.request_id != request_id:
                        continue

                    accumulated_outputs.append(output)
                    finished = output.finished

                    # Yield output for final output stages
                    if info.final_output:
                        omni_output = self._make_omni_output(
                            info, stage_id, output, engine_outputs
                        )
                        if omni_output is not None:
                            yield omni_output

                    metrics.stage_last_ts[stage_id] = max(
                        metrics.stage_last_ts[stage_id] or 0.0, time.time()
                    )

            # Store outputs for inter-stage data flow
            info.set_engine_outputs(accumulated_outputs)

            # Forward to next stage if needed
            next_stage_id = stage_id + 1
            if next_stage_id <= final_stage_id and next_stage_id < len(self.stage_clients):
                next_info = self.stage_infos[next_stage_id]

                # Derive inputs for next stage
                with metrics.stage_postprocess_timer(stage_id, request_id):
                    next_inputs = self._transform_stage_output(
                        next_info, prompt
                    )

                sp_next = sampling_params_list[next_stage_id]

                # Try connector-based transfer first
                connector_key = (str(stage_id), str(next_stage_id))
                connector = self.connectors.get(connector_key)

                if connector:
                    sent = try_send_via_connector(
                        connector=connector,
                        stage_id=stage_id,
                        next_stage_id=next_stage_id,
                        req_id=request_id,
                        next_inputs=next_inputs,
                        sampling_params=sp_next,
                        original_prompt=prompt,
                        next_stage_queue_submit_fn=self._make_submit_fn(next_stage_id),
                        metrics=metrics,
                    )
                    if not sent:
                        raise RuntimeError(
                            f"Failed to send request {request_id} "
                            f"to stage-{next_stage_id} via connector"
                        )
                else:
                    # Direct submission without connector
                    for next_input in next_inputs:
                        next_request = self._create_engine_request(
                            next_input, request_id, sp_next
                        )
                        await self.stage_clients[next_stage_id].add_request(next_request)

                metrics.stage_first_ts[next_stage_id] = time.time()

    async def _process_async_chunk(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params_list: list[OmniSamplingParams],
        metrics: OrchestratorAggregator,
        final_stage_id: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process request with async chunk mode (parallel stages).

        In async chunk mode, stages run in parallel:
        - Stage 0 starts generating
        - On first output from stage 0, derive placeholder input for stage 1+
        - Both stages run concurrently
        - Collect outputs from all stages as they arrive
        """
        num_active_stages = final_stage_id + 1
        stage_queues: dict[int, asyncio.Queue] = {
            sid: asyncio.Queue() for sid in range(num_active_stages)
        }
        all_stages_finished: dict[int, bool] = {
            sid: False for sid in range(num_active_stages)
        }
        submit_flag = True

        # Start a background task to collect outputs from all stage clients
        collector_tasks = []
        for sid in range(min(num_active_stages, len(self.stage_clients))):
            task = asyncio.create_task(
                self._collect_stage_outputs(
                    sid, self.stage_clients[sid], stage_queues[sid]
                )
            )
            collector_tasks.append(task)

        try:
            while not all(all_stages_finished.values()):
                for stage_id in range(num_active_stages):
                    if all_stages_finished[stage_id]:
                        continue
                    if stage_id >= len(self.stage_clients):
                        all_stages_finished[stage_id] = True
                        continue

                    try:
                        output = stage_queues[stage_id].get_nowait()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.001)
                        continue

                    info = self.stage_infos[stage_id]

                    if isinstance(output, Exception):
                        raise output

                    finished = output.finished
                    all_stages_finished[stage_id] = finished

                    # On first output from stage 0, submit to downstream stages
                    if submit_flag and stage_id == 0:
                        submit_flag = False
                        # Derive placeholder input for downstream stages
                        prompt_token_ids = getattr(output, "prompt_token_ids", None)
                        if prompt_token_ids is None and hasattr(output, "new_token_ids"):
                            prompt_token_ids = output.new_token_ids

                        engine_input = copy.deepcopy(prompt)
                        if isinstance(engine_input, dict):
                            next_prompt_len = max(
                                1,
                                compute_talker_prompt_ids_length(
                                    prompt_token_ids or []
                                ),
                            )
                            engine_input["prompt_token_ids"] = [0] * next_prompt_len
                            engine_input["multi_modal_data"] = None
                            engine_input["mm_processor_kwargs"] = None

                        for i in range(1, num_active_stages):
                            if i < len(self.stage_clients):
                                sp_i = sampling_params_list[i]
                                req = self._create_engine_request(
                                    engine_input, request_id, sp_i
                                )
                                await self.stage_clients[i].add_request(req)
                                metrics.stage_first_ts[i] = time.time()

                    # Yield output for final output stages
                    if info.final_output:
                        omni_output = self._make_omni_output(
                            info, stage_id, output, None
                        )
                        if omni_output is not None:
                            yield omni_output

                    metrics.stage_last_ts[stage_id] = max(
                        metrics.stage_last_ts[stage_id] or 0.0, time.time()
                    )
        finally:
            for task in collector_tasks:
                task.cancel()

    async def _collect_stage_outputs(
        self,
        stage_id: int,
        client: StageCoreClient,
        output_queue: asyncio.Queue,
    ) -> None:
        """Background task to collect outputs from a stage client."""
        try:
            while True:
                engine_outputs = await client.get_output()
                for output in engine_outputs.outputs:
                    await output_queue.put(output)
        except Exception as e:
            await output_queue.put(e)

    # ---- Data Transformation ----

    def _create_engine_request(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params: OmniSamplingParams,
    ) -> EngineCoreRequest:
        """Convert an omni prompt to an EngineCoreRequest."""
        if self.input_processor is not None:
            return self.input_processor.process_inputs(
                request_id=request_id,
                prompt=prompt,
                params=sampling_params,
            )
        raise RuntimeError("Input processor not initialized")

    def _transform_stage_output(
        self,
        next_info: StageInfo,
        original_prompt: OmniPromptType,
    ) -> list[OmniTokensPrompt]:
        """Transform current stage outputs into inputs for the next stage.

        Uses engine_input_source to determine which upstream stage outputs
        to use, and applies any custom processing function.
        """
        if next_info.custom_process_input_func is not None:
            return next_info.custom_process_input_func(
                self.stage_infos,
                next_info.engine_input_source,
                original_prompt,
                next_info.requires_multimodal_data,
            )

        # Default transformation: use token_ids from source stage
        if not next_info.engine_input_source:
            raise ValueError(
                f"engine_input_source is empty for stage-{next_info.stage_id}"
            )

        source_stage_id = next_info.engine_input_source[0]
        source_outputs = self.stage_infos[source_stage_id].engine_outputs

        if not isinstance(original_prompt, list):
            prompts = [original_prompt]
        else:
            prompts = original_prompt

        engine_inputs = []
        for source_output in source_outputs:
            # Extract token_ids from EngineCoreOutput or RequestOutput
            if hasattr(source_output, "outputs") and source_output.outputs:
                # RequestOutput format
                token_ids = source_output.outputs[0].token_ids
            elif hasattr(source_output, "new_token_ids"):
                # EngineCoreOutput format
                token_ids = source_output.new_token_ids
            else:
                token_ids = []

            # Get multimodal data from original prompt if needed
            mm_data = None
            if next_info.requires_multimodal_data and prompts:
                p = prompts[0] if len(prompts) == 1 else prompts[0]
                if isinstance(p, dict):
                    mm_data = p.get("multi_modal_data")

            engine_input = OmniTokensPrompt(
                prompt_token_ids=list(token_ids),
                multi_modal_data=mm_data,
            )
            engine_inputs.append(engine_input)

        return engine_inputs

    def _make_submit_fn(self, stage_id: int) -> Callable:
        """Create a submit function for connector-based transfer.

        Returns a function compatible with try_send_via_connector's
        next_stage_queue_submit_fn parameter.
        """
        client = self.stage_clients[stage_id]

        def submit(task: dict[str, Any]) -> None:
            # Convert task dict to EngineCoreRequest and submit
            request_id = task.get("request_id", "")
            engine_inputs = task.get("engine_inputs")
            sampling_params = task.get("sampling_params", SamplingParams())

            request = self._create_engine_request(
                engine_inputs, request_id, sampling_params
            )
            # Use asyncio to submit from sync context
            loop = asyncio.get_event_loop()
            loop.create_task(client.add_request(request))

        return submit

    @staticmethod
    def _make_omni_output(
        info: StageInfo,
        stage_id: int,
        output: Any,
        engine_outputs: EngineCoreOutputs | None,
    ) -> OmniRequestOutput | None:
        """Create an OmniRequestOutput from stage output."""
        images = []
        if info.final_output_type == "image":
            if isinstance(output, OmniRequestOutput) and output.images:
                images = output.images
            elif hasattr(output, "images") and output.images:
                images = output.images

        if info.final_output_type == "image":
            return OmniRequestOutput(
                stage_id=stage_id,
                final_output_type=info.final_output_type,
                request_output=output,
                images=images,
            )
        return OmniRequestOutput(
            stage_id=stage_id,
            final_output_type=info.final_output_type,
            request_output=output,
        )

    # ---- Lifecycle Management ----

    async def abort(self, request_id: str) -> None:
        """Abort a request across all stages."""
        for client in self.stage_clients:
            try:
                await client.abort_requests([request_id])
            except Exception as e:
                logger.debug("Failed to abort request %s: %s", request_id, e)

    async def shutdown(self) -> None:
        """Shutdown all stage processes."""
        for client in self.stage_clients:
            try:
                client.shutdown()
            except Exception as e:
                logger.debug("Failed to shutdown stage client: %s", e)

    async def start_profile(self) -> None:
        """Start profiling for all stages."""
        for client in self.stage_clients:
            await client.profile(is_start=True)

    async def stop_profile(self) -> None:
        """Stop profiling for all stages."""
        for client in self.stage_clients:
            await client.profile(is_start=False)

    # ---- Accessors ----

    def get_comprehension_stage(self) -> StageInfo | None:
        """Return the first comprehension stage."""
        for info in self.stage_infos:
            if info.is_comprehension:
                return info
        return None

    @property
    def stage_list(self) -> list[StageInfo]:
        """Return stage metadata list (backward compat with serving layer)."""
        return self.stage_infos
