# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
StageCoreProc: subprocess that wraps EngineCore (LLM) or AsyncOmniDiffusion
directly and communicates with the orchestrator via ZMQ.

Replaces the nested layers of OmniStage → AsyncOmniLLM → EngineCoreClient
with a single process that owns the EngineCore directly and exposes a
simple ZMQ PUSH/PULL interface.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import time
import traceback
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Literal, cast

import zmq
import zmq.asyncio

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers ported from omni_stage.py
# ---------------------------------------------------------------------------

def _resolve_worker_cls(engine_args: dict[str, Any]) -> None:
    """Resolve worker class based on worker_type in engine_args."""
    worker_type = engine_args.get("worker_type", None)
    if not worker_type:
        return
    if engine_args.get("worker_cls"):
        return
    from vllm_omni.platforms import current_omni_platform

    worker_type = str(worker_type).lower()
    if worker_type == "ar":
        engine_args["worker_cls"] = current_omni_platform.get_omni_ar_worker_cls()
    elif worker_type == "generation":
        engine_args["worker_cls"] = current_omni_platform.get_omni_generation_worker_cls()
    else:
        raise ValueError(f"Unknown worker_type: {worker_type}")


def _build_od_config(engine_args: dict[str, Any], model: str) -> dict[str, Any]:
    """Build OmniDiffusionConfig kwargs from engine args."""
    from dataclasses import fields as dc_fields

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    od_config = engine_args.get("od_config", {})
    if not od_config:
        od_config = {"model": model}
        od_field_names = {f.name for f in dc_fields(OmniDiffusionConfig)}
        for key, value in engine_args.items():
            if key in od_field_names:
                od_config[key] = value
    return od_config


@contextmanager
def _sequential_init_lock(engine_args: dict[str, Any], stage_init_timeout: int = 300):
    """Acquire device locks for sequential init if NVML is unavailable.

    Delegates to the existing implementation in omni_stage.py.
    """
    from vllm_omni.entrypoints.omni_stage import (
        _sequential_init_lock as _orig_lock,
    )

    with _orig_lock(engine_args, stage_init_timeout):
        yield


# ---------------------------------------------------------------------------
# Stats helpers ported from omni_stage.py
# ---------------------------------------------------------------------------

def make_request_stats(
    req_output: list[Any],
    stage_gen_time_ms: float,
    batch_id: int,
    batch_size: int,
    rx_decode_time_ms: float,
    rx_transfer_bytes: int,
    rx_in_flight_time_ms: float,
):
    from vllm_omni.metrics import StageRequestStats, count_tokens_from_outputs

    num_tokens_in = _count_prompt_tokens(req_output)
    num_tokens_out = count_tokens_from_outputs(req_output)
    return StageRequestStats(
        num_tokens_in=num_tokens_in,
        num_tokens_out=num_tokens_out,
        stage_gen_time_ms=stage_gen_time_ms,
        batch_id=batch_id,
        batch_size=batch_size,
        rx_decode_time_ms=rx_decode_time_ms,
        rx_transfer_bytes=rx_transfer_bytes,
        rx_in_flight_time_ms=rx_in_flight_time_ms,
        stage_stats=None,
    )


def make_stage_stats(agg_total_tokens: int, agg_total_gen_time_ms: float):
    from vllm_omni.metrics import StageStats

    return StageStats(total_token=agg_total_tokens, total_gen_time_ms=agg_total_gen_time_ms)


def _count_prompt_tokens(engine_outputs: list[Any]) -> int:
    total = 0
    for ro in engine_outputs:
        prompt_token_ids = getattr(ro, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            total += len(prompt_token_ids)
    return total


def output_strip(r_output: Any, final_output: bool, final_output_type: str | None):
    """Strip unnecessary multimodal outputs from stage results."""
    if final_output and final_output_type != "text":
        return r_output
    if getattr(r_output, "finished", False):
        return r_output
    mm_output = getattr(r_output, "multimodal_output", None)
    if mm_output is not None:
        r_output.multimodal_output = {}
    outputs = getattr(r_output, "outputs", None)
    if outputs is not None:
        for out in outputs:
            if getattr(out, "multimodal_output", None):
                out.multimodal_output = {}
    return r_output


# ---------------------------------------------------------------------------
# StageCoreProc entry points
# ---------------------------------------------------------------------------

def stage_core_proc_entry(
    model: str,
    stage_payload: dict[str, Any],
    zmq_send_addr: str,
    zmq_recv_addr: str,
    stage_init_timeout: int = 300,
    batch_timeout: int = 10,
    log_stats: bool = False,
    final_output: bool = False,
    final_output_type: str | None = None,
) -> None:
    """Entry point for the StageCoreProc subprocess.

    For LLM stages, runs a synchronous busy loop (like EngineCoreProc).
    For diffusion stages, launches an asyncio event loop.
    """
    stage_type: Literal["llm", "diffusion"] = stage_payload.get("stage_type", "llm")

    if stage_type == "diffusion":
        asyncio.run(
            _stage_core_proc_async(
                model=model,
                stage_payload=stage_payload,
                zmq_send_addr=zmq_send_addr,
                zmq_recv_addr=zmq_recv_addr,
                stage_init_timeout=stage_init_timeout,
                batch_timeout=batch_timeout,
                log_stats=log_stats,
                final_output=final_output,
                final_output_type=final_output_type,
            )
        )
    else:
        _stage_core_proc_sync(
            model=model,
            stage_payload=stage_payload,
            zmq_send_addr=zmq_send_addr,
            zmq_recv_addr=zmq_recv_addr,
            stage_init_timeout=stage_init_timeout,
            batch_timeout=batch_timeout,
            log_stats=log_stats,
            final_output=final_output,
            final_output_type=final_output_type,
        )


def _setup_process_env() -> None:
    """Common subprocess environment setup."""
    import multiprocessing as _mp

    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("[StageCoreProc] Set VLLM_WORKER_MULTIPROC_METHOD=spawn")
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def _setup_devices(stage_id: int, runtime_cfg: dict[str, Any]) -> str | None:
    """Configure per-stage device visibility. Returns device_type."""
    device_type = None
    try:
        from vllm_omni.platforms import current_omni_platform

        device_type = current_omni_platform.device_type
        from vllm_omni.entrypoints.stage_utils import set_stage_devices

        set_stage_devices(stage_id, runtime_cfg.get("devices"), device_type=device_type)
    except Exception as e:
        logger.warning("Device setup failed: %s", e)
    return device_type


def _init_connectors(stage_id: int, connectors_config: dict) -> dict:
    """Initialize OmniConnectors if configured."""
    connectors: dict = {}
    if connectors_config:
        from vllm_omni.distributed.omni_connectors import build_stage_connectors

        built = build_stage_connectors(stage_id=stage_id, connectors_config=connectors_config)
        if built is None:
            raise RuntimeError(f"[Stage-{stage_id}] Connector build returned None")
        connectors = built
    return connectors


def _zmq_connect_sockets(
    ctx: zmq.Context,
    send_addr: str,
    recv_addr: str,
) -> tuple[zmq.Socket, zmq.Socket]:
    """Create ZMQ sockets: PUSH to send results, PULL to receive requests."""
    sender = ctx.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 0)
    sender.setsockopt(zmq.LINGER, 0)
    sender.connect(send_addr)

    receiver = ctx.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVHWM, 0)
    receiver.connect(recv_addr)

    return sender, receiver


def _zmq_connect_sockets_async(
    ctx: zmq.asyncio.Context,
    send_addr: str,
    recv_addr: str,
) -> tuple[zmq.asyncio.Socket, zmq.asyncio.Socket]:
    """Create async ZMQ sockets: PUSH to send results, PULL to receive requests."""
    sender = ctx.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 0)
    sender.setsockopt(zmq.LINGER, 0)
    sender.connect(send_addr)

    receiver = ctx.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVHWM, 0)
    receiver.connect(recv_addr)

    return sender, receiver


def _zmq_send(socket: zmq.Socket, msg: dict[str, Any]) -> None:
    """Send a pickle-serialized message over ZMQ."""
    socket.send(pickle.dumps(msg))


def _zmq_recv_nowait(socket: zmq.Socket) -> dict[str, Any] | None:
    """Non-blocking receive of a pickle-serialized message."""
    try:
        data = socket.recv(zmq.NOBLOCK)
        return pickle.loads(data)
    except zmq.Again:
        return None


# ---------------------------------------------------------------------------
# Sync (LLM) path
# ---------------------------------------------------------------------------

def _stage_core_proc_sync(
    model: str,
    stage_payload: dict[str, Any],
    zmq_send_addr: str,
    zmq_recv_addr: str,
    stage_init_timeout: int = 300,
    batch_timeout: int = 10,
    log_stats: bool = False,
    final_output: bool = False,
    final_output_type: str | None = None,
) -> None:
    """Synchronous LLM stage worker using EngineCore-like busy loop."""
    _setup_process_env()

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    connectors_config = stage_payload.get("connectors_config", {})

    _resolve_worker_cls(engine_args)

    # Device setup
    _setup_devices(stage_id, runtime_cfg)

    # ZMQ sockets
    zmq_ctx = zmq.Context()
    sender, receiver = _zmq_connect_sockets(zmq_ctx, zmq_send_addr, zmq_recv_addr)

    # Aggregates for running average
    _agg_total_tokens = 0
    _agg_total_gen_time_ms = 0.0
    _batch_seq = 0

    # Init engine
    with _sequential_init_lock(engine_args, stage_init_timeout):
        logger.debug("[Stage-%s] Initializing LLM engine", stage_id)
        if engine_args.get("async_chunk", False):
            stage_connector_spec = {}
            for v in connectors_config.values():
                stage_connector_spec = dict(v.get("spec", {}))
                break
            engine_args["stage_connector_spec"] = stage_connector_spec
            engine_args["stage_id"] = stage_id

        from vllm_omni.entrypoints.omni_llm import OmniLLM

        stage_engine = OmniLLM(model=model, **engine_args)

    logger.debug("[Stage-%s] Engine initialized", stage_id)

    # Init connectors
    connectors = _init_connectors(stage_id, connectors_config)

    # Signal readiness
    _zmq_send(sender, {"type": "stage_ready", "stage_id": stage_id})

    max_batch_size = int(runtime_cfg.get("max_batch_size", 1) or 1)
    logger.info("[Stage-%s] Max batch size: %d", stage_id, max_batch_size)

    from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector
    from vllm_omni.entrypoints.stage_utils import (
        OmniStageTaskType,
        is_profiler_task,
        maybe_dump_to_shm,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType, OmniSamplingParams

    SHUTDOWN_TYPE = OmniStageTaskType.SHUTDOWN

    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))

    def _handle_profiler(task_type: OmniStageTaskType) -> dict:
        if task_type == OmniStageTaskType.PROFILER_START:
            try:
                stage_engine.start_profile()
                logger.info("[Stage-%s] vLLM profiler started", stage_id)
            except Exception as e:
                logger.warning("[Stage-%s] Failed to start profiler: %s", stage_id, e)
            return {}
        elif task_type == OmniStageTaskType.PROFILER_STOP:
            try:
                stage_engine.stop_profile()
                logger.info("[Stage-%s] vLLM profiler stopped", stage_id)
            except Exception as e:
                logger.warning("[Stage-%s] Failed to stop profiler: %s", stage_id, e)
            return {}
        return {}

    # Local buffer for tasks that couldn't be batched (different sampling params)
    _pending_tasks: list[dict[str, Any]] = []

    # Main processing loop
    while True:
        # Get next task: drain local buffer first, then block on ZMQ
        if _pending_tasks:
            task = _pending_tasks.pop(0)
        else:
            data = receiver.recv()
            task = pickle.loads(data)

        _recv_ts = time.time()
        task_type = task.get("type", OmniStageTaskType.GENERATE)

        if task_type == SHUTDOWN_TYPE:
            logger.info("[Stage-%s] Received shutdown signal", stage_id)
            break

        if is_profiler_task(task_type):
            profiler_data = _handle_profiler(task_type)
            if task_type == OmniStageTaskType.PROFILER_STOP:
                _zmq_send(sender, {"type": "profiler_result", "data": profiler_data})
            continue

        # Batch collection
        batch_tasks: list[dict[str, Any]] = [task]
        tasks_failed: list[dict[str, Any]] = []
        start_time = time.time()

        if max_batch_size > 1:
            while len(batch_tasks) < max_batch_size:
                extra = _zmq_recv_nowait(receiver)
                if extra is None:
                    elapsed = time.time() - start_time
                    if elapsed > batch_timeout:
                        break
                    time.sleep(0.05)
                    continue
                extra_type = extra.get("type") if isinstance(extra, dict) else None
                if extra_type == SHUTDOWN_TYPE:
                    # Buffer the shutdown so outer loop catches it next iteration
                    _pending_tasks.append(extra)
                    break
                if is_profiler_task(extra_type):
                    p_data = _handle_profiler(extra_type)
                    if extra_type == OmniStageTaskType.PROFILER_STOP:
                        _zmq_send(sender, {"type": "profiler_result", "data": p_data})
                    continue
                if task.get("sampling_params") != extra.get("sampling_params"):
                    tasks_failed.append(extra)
                else:
                    batch_tasks.append(extra)
                if time.time() - start_time > batch_timeout:
                    break

        # Buffer tasks that couldn't be batched for the next iteration
        _pending_tasks.extend(tasks_failed)

        batch_sampling_params: OmniSamplingParams = batch_tasks[0]["sampling_params"]
        batch_request_ids: list[Any] = []
        batch_engine_inputs: list[OmniPromptType] = []
        _rx_bytes: dict[Any, int] = {}
        _rx_decode_ms: dict[Any, float] = {}
        _in_flight_ms: dict[Any, float] = {}

        for t in batch_tasks:
            rid = t["request_id"]
            try:
                sent_ts = float(t.get("sent_ts")) if t.get("sent_ts") is not None else None
                _in_flight_ms[rid] = max(0.0, (_recv_ts - sent_ts) * 1000.0) if sent_ts else 0.0
            except Exception:
                _in_flight_ms[rid] = 0.0

            ein, _rx_metrics = try_recv_via_connector(task=t, connectors=connectors, stage_id=stage_id)
            ein = cast(OmniPromptType | Sequence[OmniPromptType] | None, ein)
            if ein is None or _rx_metrics is None:
                raise RuntimeError(
                    f"[Stage-{stage_id}] Missing connector payload for request {rid}."
                )
            _rx_decode_ms[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
            _rx_bytes[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))
            batch_request_ids.append(rid)
            if isinstance(ein, (str, dict)):
                batch_engine_inputs.append(ein)
            elif isinstance(ein, Sequence):
                batch_engine_inputs.extend(ein)
            else:
                batch_engine_inputs.append(ein)

        try:
            _batch_seq += 1
            gen_outputs = []
            _gen_t0 = time.time()

            from vllm.sampling_params import SamplingParams

            stage_engine_cast = cast(Any, stage_engine)
            batch_sampling_params_cast = cast(SamplingParams, batch_sampling_params)
            results = stage_engine_cast.generate(
                batch_engine_inputs,
                batch_sampling_params_cast,
                use_tqdm=False,
            )
            gen_outputs.extend(results)

            _gen_ms = (time.time() - _gen_t0) * 1000.0

            # Group outputs by request id
            req_to_outputs: dict[Any, list] = {rid: [] for rid in batch_request_ids}
            unmapped = []
            for ro in gen_outputs:
                rid = ro.request_id
                if rid in req_to_outputs:
                    req_to_outputs[rid].append(ro)
                else:
                    unmapped.append(ro)
            if unmapped:
                for idx, ro in enumerate(unmapped):
                    target = batch_request_ids[idx % len(batch_request_ids)]
                    ro.request_id = target
                    req_to_outputs[target].append(ro)

            _agg_total_gen_time_ms += _gen_ms

            for i, rid in enumerate(batch_request_ids):
                r_outputs = req_to_outputs.get(rid, [])
                _metrics = make_request_stats(
                    r_outputs, _gen_ms, _batch_seq, len(batch_request_ids),
                    _rx_decode_ms.get(rid, 0.0), _rx_bytes.get(rid, 0),
                    _in_flight_ms.get(rid, 0.0),
                )
                _agg_total_tokens += _metrics.num_tokens_out
                if i == len(batch_request_ids) - 1:
                    _metrics.stage_stats = make_stage_stats(_agg_total_tokens, _agg_total_gen_time_ms)

                try:
                    use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                    msg = {
                        "request_id": rid,
                        "stage_id": stage_id,
                        "metrics": _metrics,
                    }
                    if use_shm:
                        msg["engine_outputs_shm"] = payload
                    else:
                        msg["engine_outputs"] = payload
                    _zmq_send(sender, msg)
                except Exception:
                    _zmq_send(sender, {
                        "request_id": rid,
                        "stage_id": stage_id,
                        "engine_outputs": r_outputs,
                        "metrics": _metrics,
                    })

        except Exception as e:
            logger.exception("[Stage-%s] Failed on batch %s: %s", stage_id, batch_request_ids, e)
            _tb = traceback.format_exc()
            for rid in batch_request_ids:
                _zmq_send(sender, {
                    "request_id": rid,
                    "stage_id": stage_id,
                    "error": str(e),
                    "error_tb": _tb,
                })

    # Cleanup
    sender.close()
    receiver.close()
    zmq_ctx.term()
    logger.info("[Stage-%s] StageCoreProc exiting", stage_id)


# ---------------------------------------------------------------------------
# Async (Diffusion + Async LLM) path
# ---------------------------------------------------------------------------

async def _stage_core_proc_async(
    model: str,
    stage_payload: dict[str, Any],
    zmq_send_addr: str,
    zmq_recv_addr: str,
    stage_init_timeout: int = 300,
    batch_timeout: int = 10,
    log_stats: bool = False,
    final_output: bool = False,
    final_output_type: str | None = None,
) -> None:
    """Async stage worker for diffusion stages."""
    _setup_process_env()

    stage_id = stage_payload["stage_id"]
    engine_args = stage_payload.get("engine_args", {})
    runtime_cfg = stage_payload.get("runtime", {})
    connectors_config = stage_payload.get("connectors_config", {})
    stage_type = stage_payload.get("stage_type", "llm")

    if stage_type != "diffusion":
        _resolve_worker_cls(engine_args)

    # Device setup
    _setup_devices(stage_id, runtime_cfg)

    # Init connectors
    connectors = _init_connectors(stage_id, connectors_config)

    # ZMQ async sockets
    zmq_ctx = zmq.asyncio.Context()
    sender, receiver = _zmq_connect_sockets_async(zmq_ctx, zmq_send_addr, zmq_recv_addr)

    _agg_total_tokens = 0
    _agg_total_gen_time_ms = 0.0
    _batch_seq = 0

    shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))

    # Init engine
    with _sequential_init_lock(engine_args, stage_init_timeout):
        logger.debug("[Stage-%s] Initializing %s engine (async)", stage_id, stage_type)
        if engine_args.get("async_chunk", False):
            stage_connector_spec = {}
            for v in connectors_config.values():
                stage_connector_spec = dict(v.get("spec", {}))
                break
            engine_args["stage_connector_spec"] = stage_connector_spec
            engine_args["stage_id"] = stage_id

        if stage_type == "diffusion":
            from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

            od_config = _build_od_config(engine_args, model)
            if "omni_kv_config" not in od_config:
                od_config["omni_kv_config"] = {}
            od_config["omni_kv_config"]["stage_id"] = stage_id
            od_config["omni_kv_config"]["engine_input_source"] = stage_payload.get("engine_input_source", [])

            stage_engine = AsyncOmniDiffusion(
                model=model,
                od_config=od_config,
                **{k: v for k, v in engine_args.items() if k not in {"od_config", "model"}},
            )
            vllm_config = None
        else:
            from vllm.usage.usage_lib import UsageContext

            from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
            from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM

            omni_engine_args = AsyncOmniEngineArgs(model=model, **engine_args)
            usage_context = UsageContext.OPENAI_API_SERVER
            vllm_config = omni_engine_args.create_engine_config(usage_context=usage_context)
            stage_engine = AsyncOmniLLM.from_vllm_config(
                vllm_config=vllm_config,
                usage_context=usage_context,
                engine_args=omni_engine_args,
                disable_log_stats=bool(
                    engine_args.get("disable_log_stats", True)
                    or getattr(omni_engine_args, "disable_log_stats", True)
                ),
            )

    # Reset MM cache for LLM engines
    if stage_type != "diffusion":
        await stage_engine.reset_mm_cache()

    logger.debug("[Stage-%s] Engine initialized (async)", stage_id)

    # Send ready signal
    ready_payload: dict[str, Any] = {
        "type": "stage_ready",
        "stage_id": stage_id,
        "vllm_config": vllm_config,
        "tokenizer": getattr(stage_engine, "tokenizer", None),
    }
    if stage_type != "diffusion":
        ready_payload["is_tracing_enabled"] = await stage_engine.is_tracing_enabled()

    await sender.send(pickle.dumps(ready_payload))

    from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector
    from vllm_omni.entrypoints.stage_utils import (
        OmniStageTaskType,
        is_profiler_task,
        maybe_dump_to_shm,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

    SHUTDOWN_TYPE = OmniStageTaskType.SHUTDOWN

    generation_out_q: asyncio.Queue = asyncio.Queue()
    _rx_bytes: dict[Any, int] = {}
    _rx_decode_ms: dict[Any, float] = {}
    _in_flight_ms: dict[Any, float] = {}

    async def _handle_profiler_async(task_type: OmniStageTaskType) -> None:
        if task_type == OmniStageTaskType.PROFILER_START:
            if stage_type == "diffusion":
                try:
                    profile_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "./profiles")
                    os.makedirs(profile_dir, exist_ok=True)
                    trace_filename = f"stage_{stage_id}_diffusion_{int(time.time())}"
                    stage_engine.start_profile(trace_filename=trace_filename)
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to start profiler: %s", stage_id, e)
            else:
                try:
                    await stage_engine.start_profile()
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to start profiler: %s", stage_id, e)
        elif task_type == OmniStageTaskType.PROFILER_STOP:
            if stage_type == "diffusion":
                try:
                    stage_engine.stop_profile()
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to stop profiler: %s", stage_id, e)
            else:
                try:
                    await stage_engine.stop_profile()
                except Exception as e:
                    logger.warning("[Stage-%s] Failed to stop profiler: %s", stage_id, e)

    async def _generation_single(task: dict[str, Any]) -> None:
        _recv_ts = time.time()
        rid = task["request_id"]
        try:
            sent_ts = float(task.get("sent_ts")) if task.get("sent_ts") is not None else None
            _in_flight_ms[rid] = max(0.0, (_recv_ts - sent_ts) * 1000.0) if sent_ts else 0.0
        except Exception:
            _in_flight_ms[rid] = 0.0

        try:
            ein, _rx_m = try_recv_via_connector(task=task, connectors=connectors, stage_id=stage_id)
            ein = cast(OmniPromptType | Sequence[OmniPromptType] | None, ein)
            if ein is None or _rx_m is None:
                raise RuntimeError(f"[Stage-{stage_id}] Missing connector for {rid}")

            _rx_decode_ms[rid] = float(_rx_m.get("rx_decode_time_ms", 0.0))
            _rx_bytes[rid] = int(_rx_m.get("rx_transfer_bytes", 0))

            _gen_t0 = time.time()
            if isinstance(ein, Sequence) and not isinstance(ein, str):
                ein = ein[0]

            if stage_type == "diffusion":
                from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion as _AD

                diff_sp = cast(OmniDiffusionSamplingParams, task["sampling_params"])
                gen_output = await cast(_AD, stage_engine).generate(ein, diff_sp, rid)
                _gen_ms = (time.time() - _gen_t0) * 1000.0
                await generation_out_q.put((rid, gen_output, _gen_ms))
            else:
                from vllm import PromptType
                from vllm.sampling_params import SamplingParams
                from vllm.v1.engine.async_llm import AsyncLLM

                ein = cast(PromptType, ein)
                llm_sp: SamplingParams = task["sampling_params"]
                async for res in cast(AsyncLLM, stage_engine).generate(ein, llm_sp, rid):
                    _gen_ms = (time.time() - _gen_t0) * 1000.0
                    _gen_t0 = time.time()
                    await generation_out_q.put((rid, res, _gen_ms))
        except Exception as e:
            logger.exception("[Stage-%s] Failed on request %s: %s", stage_id, rid, e)
            await sender.send(pickle.dumps({
                "request_id": rid,
                "stage_id": stage_id,
                "error": str(e),
            }))

    _batch_gen_t0 = time.time()
    poller = zmq.asyncio.Poller()
    poller.register(receiver, zmq.POLLIN)

    while True:
        # Non-blocking poll for incoming tasks
        events = dict(await poller.poll(timeout=1))  # 1ms timeout
        if receiver in events:
            data = await receiver.recv()
            task = pickle.loads(data)
            task_type = task.get("type", OmniStageTaskType.GENERATE)

            if task_type == SHUTDOWN_TYPE:
                logger.debug("[Stage-%s] Received shutdown signal", stage_id)
                if hasattr(stage_engine, "shutdown"):
                    stage_engine.shutdown()
                break
            elif task_type == OmniStageTaskType.ABORT:
                rid = task["request_id"]
                if hasattr(stage_engine, "abort"):
                    asyncio.create_task(stage_engine.abort(rid))
            elif is_profiler_task(task_type):
                await _handle_profiler_async(task_type)
            else:
                asyncio.create_task(_generation_single(task))

        # Collect generation outputs
        batch_outputs = []
        batch_rids = []
        batch_gen_ms = []
        batch_metrics_list = []

        while True:
            try:
                rid, gen_output, _gen_ms = generation_out_q.get_nowait()
                _metrics = make_request_stats(
                    [gen_output], _gen_ms, _batch_seq, 1,
                    _rx_decode_ms.get(rid, 0.0), _rx_bytes.get(rid, 0),
                    _in_flight_ms.get(rid, 0.0),
                )
                batch_metrics_list.append(_metrics)
                batch_outputs.append(gen_output)
                batch_gen_ms.append(_gen_ms)
                batch_rids.append(rid)
                _agg_total_tokens += _metrics.num_tokens_out
            except asyncio.QueueEmpty:
                break

        if not batch_outputs:
            continue

        _batch_seq += 1
        _batch_gen_t1 = time.time()
        _agg_total_gen_time_ms += (_batch_gen_t1 - _batch_gen_t0) * 1000
        _batch_gen_t0 = _batch_gen_t1

        for idx, metrics in enumerate(batch_metrics_list):
            metrics.batch_size = len(batch_metrics_list)
            if idx == len(batch_metrics_list) - 1:
                metrics.stage_stats = make_stage_stats(_agg_total_tokens, _agg_total_gen_time_ms)

        for rid, out, _gen_ms, _metrics in zip(batch_rids, batch_outputs, batch_gen_ms, batch_metrics_list):
            try:
                r_outputs = [output_strip(out, final_output, final_output_type)]
                use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
                msg: dict[str, Any] = {
                    "request_id": rid,
                    "stage_id": stage_id,
                    "metrics": _metrics,
                }
                if use_shm:
                    msg["engine_outputs_shm"] = payload
                else:
                    msg["engine_outputs"] = payload
                await sender.send(pickle.dumps(msg))
            except Exception as e:
                logger.exception("[Stage-%s] Failed to send result for %s: %s", stage_id, rid, e)
                await sender.send(pickle.dumps({
                    "request_id": rid,
                    "stage_id": stage_id,
                    "engine_outputs": [out],
                    "metrics": _metrics,
                }))

    # Cleanup
    sender.close()
    receiver.close()
    zmq_ctx.term()
    logger.info("[Stage-%s] StageCoreProc (async) exiting", stage_id)
