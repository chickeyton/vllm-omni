# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
PipelineOrchestrator: multi-stage pipeline coordinator.

Absorbs the responsibilities of OmniBase (stage init, lifecycle) and
parts of AsyncOmni (generate, output handling, result routing).
Uses StageCoreClient instead of OmniStage for stage management.
"""

from __future__ import annotations

import asyncio
import copy
import os
import time
import weakref
from collections.abc import AsyncGenerator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from omegaconf import OmegaConf
from vllm.config import VllmConfig
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.plugins.io_processors import get_io_processor
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

from vllm_omni.config import OmniModelConfig
from vllm_omni.distributed.omni_connectors import (
    get_stage_connector_config,
    initialize_orchestrator_connectors,
)
from vllm_omni.distributed.omni_connectors.adapter import (
    compute_talker_prompt_ids_length,
    try_send_via_connector,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    resolve_omni_kv_config_for_stage,
)
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.stage_core_client import StageCoreClient
from vllm_omni.entrypoints.stage_utils import OmniStageTaskType
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
    inject_omni_kv_config,
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.metrics import OrchestratorAggregator
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _weak_orchestrator_cleanup(
    stage_list: list[StageCoreClient],
    output_handler: asyncio.Task | None,
) -> None:
    """Weak reference cleanup for PipelineOrchestrator."""
    for stage in stage_list:
        try:
            stage.stop()
        except Exception as e:
            logger.warning("Failed to stop stage: %s", e)
    if output_handler is not None:
        output_handler.cancel()


class PipelineOrchestrator:
    """Multi-stage pipeline coordinator using StageCoreClient.

    Replaces the combination of OmniBase + parts of AsyncOmni with a
    single class that manages the full pipeline lifecycle.

    Args:
        model: Model name or path to load
        **kwargs: Configuration keyword arguments (same as OmniBase)
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        from vllm_omni.entrypoints.omni import omni_snapshot_download

        model = omni_snapshot_download(model)
        kwargs["model"] = model

        # Stage management
        self.stage_list: list[StageCoreClient] = []

        # Request state tracking
        self.request_states: dict[str, ClientRequestState] = {}
        self.output_handler: asyncio.Task | None = None

        # Processors (populated after stages are ready)
        self.input_processor: OmniInputProcessor | None = None
        self.io_processor: Any = None
        self.model_config: OmniModelConfig | None = None
        self._vllm_config: VllmConfig | None = None
        self._tokenizer: TokenizerLike | None = None

        # Initialize stages
        logger.info("Initializing orchestrator for model: %s", model)
        self._initialize_stages(model, kwargs)

        # Register weak reference cleanup
        self._weak_finalizer = weakref.finalize(
            self,
            _weak_orchestrator_cleanup,
            self.stage_list,
            self.output_handler,
        )

    # ----- Configuration helpers -----

    @staticmethod
    def _get_default_cache_config(cache_backend: str | None) -> dict[str, Any] | None:
        if cache_backend == "cache_dit":
            return {
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 0,
                "max_warmup_steps": 4,
                "residual_diff_threshold": 0.24,
                "max_continuous_cached_steps": 3,
                "enable_taylorseer": False,
                "taylorseer_order": 1,
                "scm_steps_mask_policy": None,
                "scm_steps_policy": "dynamic",
            }
        if cache_backend == "tea_cache":
            return {"rel_l1_thresh": 0.2}
        return None

    @staticmethod
    def _normalize_cache_config(cache_backend: str | None, cache_config: Any | None) -> Any | None:
        import json

        if isinstance(cache_config, str):
            try:
                cache_config = json.loads(cache_config)
            except json.JSONDecodeError:
                logger.warning("Invalid cache_config JSON, using defaults.")
                cache_config = None
        if cache_config is None and cache_backend not in (None, "", "none"):
            cache_config = PipelineOrchestrator._get_default_cache_config(cache_backend)
        return cache_config

    def _create_default_diffusion_stage_cfg(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Create default diffusion stage configuration."""
        from vllm_omni.diffusion.data import DiffusionParallelConfig

        cache_backend = kwargs.get("cache_backend", "none")
        cache_config = self._normalize_cache_config(cache_backend, kwargs.get("cache_config", None))

        devices = "0"
        if "parallel_config" in kwargs:
            parallel_config = kwargs["parallel_config"]
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"
        else:
            ulysses_degree = kwargs.get("ulysses_degree") or 1
            ring_degree = kwargs.get("ring_degree") or 1
            sequence_parallel_size = kwargs.get("sequence_parallel_size")
            tensor_parallel_size = kwargs.get("tensor_parallel_size") or 1
            cfg_parallel_size = kwargs.get("cfg_parallel_size") or 1
            if sequence_parallel_size is None:
                sequence_parallel_size = ulysses_degree * ring_degree
            num_devices = sequence_parallel_size * tensor_parallel_size * cfg_parallel_size
            for i in range(1, num_devices):
                devices += f",{i}"
            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                tensor_parallel_size=tensor_parallel_size,
                sequence_parallel_size=sequence_parallel_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                cfg_parallel_size=cfg_parallel_size,
            )

        default_stage_cfg = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "devices": devices,
                    "max_batch_size": 1,
                },
                "engine_args": {
                    "parallel_config": parallel_config,
                    "vae_use_slicing": kwargs.get("vae_use_slicing", False),
                    "vae_use_tiling": kwargs.get("vae_use_tiling", False),
                    "cache_backend": cache_backend,
                    "cache_config": cache_config,
                    "enable_cache_dit_summary": kwargs.get("enable_cache_dit_summary", False),
                    "enable_cpu_offload": kwargs.get("enable_cpu_offload", False),
                    "enable_layerwise_offload": kwargs.get("enable_layerwise_offload", False),
                    "enforce_eager": kwargs.get("enforce_eager", False),
                    "diffusion_load_format": kwargs.get("diffusion_load_format", "default"),
                    "custom_pipeline_args": kwargs.get("custom_pipeline_args", None),
                },
                "final_output": True,
                "final_output_type": "image",
            }
        ]
        default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
        return default_stage_cfg

    @staticmethod
    def _is_async_chunk_enable(stage_args: list) -> bool:
        engine_args = getattr(stage_args[0], "engine_args", None)
        return bool(getattr(engine_args, "async_chunk", False))

    # ----- Stage initialization -----

    def _initialize_stages(self, model: str, kwargs: dict[str, Any]) -> None:
        """Initialize stage list management."""
        stage_init_timeout = kwargs.get("stage_init_timeout", 20)
        shm_threshold_bytes = kwargs.get("shm_threshold_bytes", 65536)
        init_timeout = kwargs.get("init_timeout", 300)
        batch_timeout = kwargs.get("batch_timeout", 10)
        stage_configs_path = kwargs.get("stage_configs_path", None)
        log_stats = kwargs.get("log_stats", False)
        tokenizer = kwargs.get("tokenizer", None)

        base_engine_args = {"tokenizer": tokenizer} if tokenizer is not None else None

        # Load stage configurations
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model, base_engine_args=base_engine_args)
            if not self.stage_configs:
                default_cfg = self._create_default_diffusion_stage_cfg(kwargs)
                self.stage_configs = OmegaConf.create(default_cfg)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(
                stage_configs_path, base_engine_args=base_engine_args
            )

        # Inject diffusion LoRA knobs
        for cfg in self.stage_configs:
            try:
                if getattr(cfg, "stage_type", None) != "diffusion":
                    continue
                if not hasattr(cfg, "engine_args") or cfg.engine_args is None:
                    cfg.engine_args = OmegaConf.create({})
                if kwargs.get("lora_path") is not None:
                    if not hasattr(cfg.engine_args, "lora_path") or cfg.engine_args.lora_path is None:
                        cfg.engine_args.lora_path = kwargs["lora_path"]
                lora_scale = kwargs.get("lora_scale") or kwargs.get("static_lora_scale")
                if lora_scale is not None:
                    if not hasattr(cfg.engine_args, "lora_scale") or cfg.engine_args.lora_scale is None:
                        cfg.engine_args.lora_scale = lora_scale
            except Exception as e:
                logger.warning("Failed to inject LoRA config: %s", e)

        # Initialize connectors
        worker_backend = kwargs.get("worker_backend", "multi_process")
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        self.log_stats: bool = bool(log_stats)
        self.async_chunk = self._is_async_chunk_enable(self.stage_configs)

        # Build StageCoreClient instances
        def _build_stage(idx_cfg: tuple[int, Any]) -> tuple[int, StageCoreClient]:
            idx, cfg = idx_cfg
            return idx, StageCoreClient(cfg, stage_init_timeout=stage_init_timeout)

        with ThreadPoolExecutor(
            max_workers=min(len(self.stage_configs), max(1, os.cpu_count() or 1))
        ) as executor:
            futures = [executor.submit(_build_stage, (i, c)) for i, c in enumerate(self.stage_configs)]
            results: list[tuple[int, StageCoreClient]] = []
            for fut in as_completed(futures):
                results.append(fut.result())

        results.sort(key=lambda x: x[0])
        self.stage_list = [st for _, st in results]
        self.default_sampling_params_list = [st.default_sampling_params for st in self.stage_list]
        self.output_modalities = [st.final_output_type for st in self.stage_list]

        logger.debug("Loaded %d stages", len(self.stage_list))

        # Start stages
        self._start_stages(model, kwargs, shm_threshold_bytes, batch_timeout)
        # Wait for readiness
        self._wait_for_stages_ready(timeout=init_timeout)

    def _start_stages(
        self,
        model: str,
        kwargs: dict[str, Any],
        shm_threshold_bytes: int,
        batch_timeout: int,
    ) -> None:
        """Start all stage subprocesses."""
        for stage_id, stage in enumerate(self.stage_list):
            stage_connectors_config = get_stage_connector_config(
                self.omni_transfer_config, stage_id
            )

            # Inject connector config into engine_args
            try:
                omni_conn_cfg, omni_from, omni_to = resolve_omni_kv_config_for_stage(
                    self.omni_transfer_config, stage_id
                )
                if omni_conn_cfg:
                    inject_omni_kv_config(stage, omni_conn_cfg, omni_from, omni_to)
            except Exception as e:
                logger.debug("Failed to inject omni connector config into stage-%s: %s", stage_id, e)

            stage.start(
                model=model,
                shm_threshold_bytes=shm_threshold_bytes,
                batch_timeout=batch_timeout,
                connectors_config=stage_connectors_config,
                log_stats=self.log_stats,
            )
            logger.debug("Stage-%d process started", stage_id)

    def _wait_for_stages_ready(self, timeout: int = 120) -> None:
        """Wait for all stages to report readiness."""
        num_stages = len(self.stage_list)
        ready_set: set[int] = set()
        deadline = time.time() + max(0, int(timeout))

        logger.info("Waiting for %d stages to initialize (timeout: %ds)", num_stages, timeout)

        # Each stage blocks independently via wait_ready()
        # We do this sequentially since each stage may take different time
        for stage_id, stage in enumerate(self.stage_list):
            remaining = max(1, int(deadline - time.time()))
            stage.wait_ready(timeout=remaining)
            if stage.is_ready:
                ready_set.add(stage_id)
                logger.info("Stage-%d reported ready", stage_id)

        if len(ready_set) == num_stages:
            logger.info("All %d stages initialized successfully", num_stages)
        else:
            not_ready = sorted(set(range(num_stages)) - ready_set)
            logger.warning(
                "Initialization: %d/%d stages ready. Missing: %s",
                len(ready_set), num_stages, not_ready,
            )

        # Initialize processors from the first LLM stage that has vllm_config
        for stage in self.stage_list:
            if stage.vllm_config is not None and stage.tokenizer is not None:
                try:
                    vllm_config = stage.vllm_config
                    self._vllm_config = vllm_config
                    self._tokenizer = stage.tokenizer
                    self.input_processor = OmniInputProcessor(vllm_config=vllm_config)
                    self.model_config = vllm_config.model_config
                    io_plugin = self.model_config.io_processor_plugin
                    self.io_processor = get_io_processor(vllm_config, io_plugin)
                    logger.info(
                        "Initialized processors from stage-%d", stage.stage_id
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Failed to init processors from stage-%d: %s", stage.stage_id, e
                    )

        if self.input_processor is None:
            logger.warning("No LLM stage found, processors unavailable")
            self.io_processor = None
            self.model_config = None

    # ----- Properties -----

    @property
    def vllm_config(self) -> VllmConfig | None:
        return self._vllm_config

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self._tokenizer

    @property
    def renderer(self):
        """Return renderer from input_processor (required by OpenAIServingModels)."""
        if self.input_processor is not None:
            return self.input_processor.renderer
        return None

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
        they become available from each stage.
        """
        logger.debug("[Orchestrator] generate() called")
        try:
            self._run_output_handler()

            if sampling_params_list is None:
                sampling_params_list = self.default_sampling_params_list

            if len(sampling_params_list) != len(self.stage_list):
                raise ValueError(
                    f"Expected {len(self.stage_list)} sampling params, "
                    f"got {len(sampling_params_list)}"
                )

            num_stages = len(self.stage_list)
            _wall_start_ts = time.time()

            final_stage_id = get_final_stage_id_for_e2e(
                output_modalities, self.output_modalities, self.stage_list
            )

            metrics = OrchestratorAggregator(
                num_stages=num_stages,
                log_stats=self.log_stats,
                wall_start_ts=_wall_start_ts,
                final_stage_id_for_e2e=final_stage_id,
            )

            req_state = ClientRequestState(request_id)
            req_state.metrics = metrics
            self.request_states[request_id] = req_state

            # Submit to first stage
            sp0: SamplingParams = sampling_params_list[0]
            task = {
                "request_id": request_id,
                "engine_inputs": prompt,
                "sampling_params": sp0,
            }
            self.stage_list[0].submit(task)
            metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()

            _req_start_ts = {request_id: time.time()}
            logger.info(
                "[Orchestrator] Scheduling: stages=%d, final_stage=%d",
                num_stages, final_stage_id,
            )

            if self.async_chunk:
                stage_queues = {sid: asyncio.Queue() for sid in range(num_stages)}
                req_state.stage_queues = stage_queues
                async for output in self._process_async_results(
                    request_id, prompt, sampling_params_list,
                    req_state, metrics, final_stage_id,
                ):
                    yield output
            else:
                async for output in self._process_sequential_results(
                    request_id, req_state, metrics, final_stage_id,
                    sampling_params_list, prompt,
                ):
                    yield output

            # Finalize metrics
            logger.debug("[Orchestrator] Request %s finalized", request_id)
            try:
                metrics.on_finalize_request(
                    final_stage_id, request_id,
                    _req_start_ts.get(request_id, _wall_start_ts),
                )
                metrics.build_and_log_summary()
            except Exception as e:
                logger.exception("[Orchestrator] Failed to finalize: %s", e)
            finally:
                self.request_states.pop(request_id, None)

        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            logger.info("[Orchestrator] Request %s aborted", request_id)
            raise

    async def _process_async_results(
        self,
        request_id: str,
        prompt: Any,
        sampling_params_list: Sequence[OmniSamplingParams],
        req_state: ClientRequestState,
        metrics: OrchestratorAggregator,
        final_stage_id: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process results when async_chunk is enabled."""
        all_finished = {sid: False for sid in range(final_stage_id + 1)}
        submit_flag = True

        while not all(all_finished.values()):
            for stage_id, stage in enumerate(self.stage_list[:final_stage_id + 1]):
                if all_finished[stage_id]:
                    continue
                try:
                    result = req_state.stage_queues[stage_id].get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)
                    continue

                engine_outputs, finished, output_to_yield = self._process_single_result(
                    result, stage, stage_id, metrics,
                )
                if submit_flag and stage_id == 0:
                    submit_flag = False
                    prompt_token_ids = engine_outputs.prompt_token_ids
                    engine_input = copy.deepcopy(prompt)
                    next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
                    engine_input["prompt_token_ids"] = [0] * next_prompt_len
                    engine_input["multi_modal_data"] = engine_input["mm_processor_kwargs"] = None
                    for i in range(1, len(self.stage_list)):
                        task = {
                            "request_id": request_id,
                            "engine_inputs": engine_input,
                            "sampling_params": sampling_params_list[i],
                        }
                        self.stage_list[i].submit(task)
                        metrics.stage_first_ts[i] = time.time()

                all_finished[stage_id] = finished
                if output_to_yield:
                    yield output_to_yield

    async def _process_sequential_results(
        self,
        request_id: str,
        req_state: ClientRequestState,
        metrics: OrchestratorAggregator,
        final_stage_id: int,
        sampling_params_list: Sequence[OmniSamplingParams],
        prompt: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process results sequentially through stages."""
        for stage_id, stage in enumerate(self.stage_list[:final_stage_id + 1]):
            finished = False
            while not finished:
                result = await req_state.queue.get()
                assert stage_id == req_state.stage_id
                engine_outputs, finished, output_to_yield = self._process_single_result(
                    result, stage, stage_id, metrics,
                )
                if output_to_yield:
                    yield output_to_yield

            if not isinstance(engine_outputs, list):
                engine_outputs = [engine_outputs]
            stage.set_engine_outputs(engine_outputs)

            # Forward to next stage
            next_stage_id = stage_id + 1
            if next_stage_id <= final_stage_id:
                next_stage = self.stage_list[next_stage_id]
                with metrics.stage_postprocess_timer(stage_id, request_id):
                    next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                sp_next: SamplingParams = sampling_params_list[next_stage_id]

                connector_key = (str(stage_id), str(next_stage_id))
                connector = self.connectors.get(connector_key)
                sent = False
                if connector:
                    sent = try_send_via_connector(
                        connector=connector,
                        stage_id=stage_id,
                        next_stage_id=next_stage_id,
                        req_id=request_id,
                        next_inputs=next_inputs,
                        sampling_params=sp_next,
                        original_prompt=prompt,
                        next_stage_queue_submit_fn=self.stage_list[next_stage_id].submit,
                        metrics=metrics,
                    )
                if not sent:
                    raise RuntimeError(
                        f"Failed to send request {request_id} to stage-{next_stage_id} via connector."
                    )
                logger.debug("[Orchestrator] Forwarded %s to stage-%d", request_id, next_stage_id)
            else:
                logger.debug("[Orchestrator] Request %s fully completed", request_id)

    def _process_single_result(
        self,
        result: dict[str, Any],
        stage: StageCoreClient,
        stage_id: int,
        metrics: OrchestratorAggregator,
    ) -> tuple[Any, bool, OmniRequestOutput | None]:
        """Process a single result dict from a stage."""
        req_id = result.get("request_id")
        if "error" in result:
            logger.error("[Orchestrator] Stage %d error on %s: %s", stage_id, req_id, result["error"])
            raise RuntimeError(result)

        engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
        if isinstance(engine_outputs, list):
            engine_outputs = engine_outputs[0]

        finished = engine_outputs.finished
        output_to_yield = None

        if getattr(stage, "final_output", False):
            images = []
            if stage.final_output_type == "image":
                if hasattr(engine_outputs, "images") and engine_outputs.images:
                    images = engine_outputs.images

            if stage.final_output_type == "image":
                output_to_yield = OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage.final_output_type,
                    request_output=engine_outputs,
                    images=images,
                )
            else:
                output_to_yield = OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage.final_output_type,
                    request_output=engine_outputs,
                )

        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())
        metrics.process_stage_metrics(
            result=result,
            stage_type=stage.stage_type,
            stage_id=stage_id,
            req_id=req_id,
            engine_outputs=engine_outputs,
            finished=finished,
            final_output_type=stage.final_output_type,
            output_to_yield=output_to_yield,
        )

        return engine_outputs, finished, output_to_yield

    # ----- Output handler -----

    def _run_output_handler(self) -> None:
        """Start background task that polls all stages for results."""
        if self.output_handler is not None:
            return

        stage_list = self.stage_list
        request_states = self.request_states

        async def _handler():
            try:
                while True:
                    idle = True
                    for stage_id, stage in enumerate(stage_list):
                        result = stage.try_collect()
                        if result is None:
                            continue
                        idle = False
                        if result.get("type") == "stage_ready":
                            await asyncio.sleep(0.05)
                            continue
                        req_id = result.get("request_id")
                        req_state = request_states.get(req_id)
                        if req_state is None:
                            logger.debug(
                                "[Orchestrator] Dropping output for aborted req %s at stage-%d",
                                req_id, stage_id,
                            )
                            continue
                        if hasattr(req_state, "stage_queues") and stage_id in req_state.stage_queues:
                            await req_state.stage_queues[stage_id].put(result)
                        else:
                            await req_state.queue.put(result)
                            req_state.stage_id = stage_id
                    if idle:
                        await asyncio.sleep(0.001)
                    else:
                        await asyncio.sleep(0)
            except Exception as e:
                logger.exception("[Orchestrator] Output handler failed")
                for rs in request_states.values():
                    error_msg = {"request_id": rs.request_id, "error": str(e)}
                    if hasattr(rs, "stage_queues"):
                        for q in rs.stage_queues.values():
                            await q.put(error_msg)
                    else:
                        await rs.queue.put(error_msg)
                self.output_handler = None

        self.output_handler = asyncio.create_task(_handler())

    # ----- Abort / Profiling / Lifecycle -----

    async def abort(self, request_id: str) -> None:
        abort_task = {"type": OmniStageTaskType.ABORT, "request_id": request_id}
        for stage in self.stage_list:
            stage.submit(abort_task)

    def start_profile(self, stages: list[int] | None = None) -> None:
        if stages is None:
            stages = list(range(len(self.stage_list)))
        for sid in stages:
            if sid < len(self.stage_list):
                try:
                    self.stage_list[sid].submit({"type": OmniStageTaskType.PROFILER_START})
                except Exception as e:
                    logger.warning("Failed to start profile on stage-%d: %s", sid, e)

    def stop_profile(self, stages: list[int] | None = None) -> dict:
        if stages is None:
            stages = list(range(len(self.stage_list)))
        all_results = {"traces": [], "tables": []}
        for sid in stages:
            if sid < len(self.stage_list):
                data = self.stage_list[sid].stop_profile()
                if isinstance(data, dict):
                    traces = data.get("trace") or data.get("traces")
                    tables = data.get("table") or data.get("tables")
                    if traces:
                        if isinstance(traces, str):
                            all_results["traces"].append(traces)
                        elif isinstance(traces, list):
                            all_results["traces"].extend(traces)
                    if tables:
                        if isinstance(tables, str):
                            all_results["tables"].append(tables)
                        elif isinstance(tables, list):
                            all_results["tables"].extend(tables)
        return all_results

    def shutdown(self) -> None:
        """Shutdown all stages and clean up."""
        if hasattr(self, "_weak_finalizer"):
            self._weak_finalizer()

    # ----- EngineClient-compatible methods -----

    async def get_vllm_config(self) -> VllmConfig | None:
        return self._vllm_config

    async def get_model_config(self) -> OmniModelConfig | None:
        return self.model_config

    async def get_tokenizer(self) -> TokenizerLike | None:
        return self._tokenizer

    async def get_input_preprocessor(self) -> InputPreprocessor | None:
        return None

    async def is_tracing_enabled(self) -> bool:
        for stage in self.stage_list:
            if stage.is_comprehension:
                return stage.is_tracing_enabled
        return False
