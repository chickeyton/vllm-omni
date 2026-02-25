# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
StageCoreClient: client that spawns a StageCoreProc subprocess and
communicates with it via ZMQ.

Replaces the OmniStage class as the per-stage management unit in the
orchestrator. Each StageCoreClient owns a subprocess and a pair of ZMQ
sockets for bidirectional communication.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import pickle
import time
from typing import Any, Literal

import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

from vllm_omni.entrypoints.stage_utils import OmniStageTaskType, _to_dict
from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniPromptType,
    OmniSamplingParams,
    OmniTokensPrompt,
)

logger = init_logger(__name__)


def _get_zmq_ipc_path() -> str:
    """Generate a unique IPC path for ZMQ communication."""
    import uuid

    base = os.environ.get("VLLM_RPC_BASE_PATH", "/tmp/vllm_rpc")
    os.makedirs(base, exist_ok=True)
    return f"ipc://{base}/{uuid.uuid4()}"


class StageCoreClient:
    """Client-side handle for a single pipeline stage.

    Spawns a StageCoreProc subprocess, communicates via ZMQ PUSH/PULL.
    Carries stage metadata for the orchestrator to route data correctly.

    Args:
        stage_config: OmegaConf or dict stage configuration
        stage_init_timeout: Timeout for engine initialization in seconds
    """

    def __init__(self, stage_config: Any, stage_init_timeout: int = 300):
        logger.info("[StageCoreClient] stage_config: %s", stage_config)
        self.stage_config = stage_config

        # Stage metadata (ported from OmniStage.__init__)
        self.stage_id: int = stage_config.stage_id
        self.engine_args = stage_config.engine_args
        self.model_stage = stage_config.engine_args.model_stage
        self.stage_type: Literal["llm", "diffusion"] = getattr(stage_config, "stage_type", "llm")

        self.engine_input_source = getattr(stage_config, "engine_input_source", [])
        self.final_output: bool = getattr(stage_config, "final_output", False)
        self.final_output_type: str | None = getattr(stage_config, "final_output_type", None)
        self.is_comprehension: bool = getattr(stage_config, "is_comprehension", False)
        self.requires_multimodal_data: bool = getattr(
            getattr(stage_config, "runtime", None), "requires_multimodal_data", False
        )

        # Custom process input function
        if hasattr(stage_config, "custom_process_input_func"):
            module_path, func_name = stage_config.custom_process_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_input_func = getattr(module, func_name)
        else:
            self.custom_process_input_func = None

        # Default sampling params
        default_sp = getattr(stage_config, "default_sampling_params", {})
        default_sp = _to_dict(default_sp)
        try:
            self.default_sampling_params: OmniSamplingParams = (
                SamplingParams if self.stage_type == "llm" else OmniDiffusionSamplingParams
            )(**default_sp)
        except TypeError as error:
            raise TypeError(
                f"Invalid default_sampling_params for stage {self.stage_id}: {error}"
            ) from error

        # Runtime state
        self._proc: mp.Process | None = None
        self._stage_init_timeout = stage_init_timeout

        # ZMQ state
        self._zmq_ctx: zmq.Context | None = None
        self._sender: zmq.Socket | None = None  # PUSH to send requests to proc
        self._receiver: zmq.Socket | None = None  # PULL to receive results from proc
        self._send_addr: str = ""
        self._recv_addr: str = ""

        # Readiness data (populated after wait_ready())
        self._vllm_config: VllmConfig | None = None
        self._tokenizer: TokenizerLike | None = None
        self._is_tracing_enabled: bool = False
        self._ready: bool = False

        # Engine outputs (for process_engine_inputs cross-stage data wiring)
        self.engine_outputs: Any = None

    # ----- Properties -----

    @property
    def vllm_config(self) -> VllmConfig | None:
        return self._vllm_config

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self._tokenizer

    @property
    def is_tracing_enabled(self) -> bool:
        return self._is_tracing_enabled

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ----- Setters for compatibility -----

    def set_vllm_config(self, vllm_config: VllmConfig) -> None:
        self._vllm_config = vllm_config

    def set_tokenizer(self, tokenizer: TokenizerLike) -> None:
        self._tokenizer = tokenizer

    def set_is_tracing_enabled(self, val: bool) -> None:
        self._is_tracing_enabled = val

    def set_engine_outputs(self, engine_outputs: Any) -> None:
        self.engine_outputs = engine_outputs

    # ----- Lifecycle -----

    def start(
        self,
        model: str,
        *,
        shm_threshold_bytes: int = 65536,
        batch_timeout: int = 10,
        connectors_config: dict | None = None,
        log_stats: bool = False,
    ) -> None:
        """Spawn the StageCoreProc subprocess.

        Creates ZMQ sockets, then spawns the subprocess which connects
        to them.
        """
        # Create ZMQ context and bind sockets BEFORE spawning
        self._zmq_ctx = zmq.Context()

        # Orchestrator PUSH → Proc PULL (requests)
        self._send_addr = _get_zmq_ipc_path()
        self._sender = self._zmq_ctx.socket(zmq.PUSH)
        self._sender.setsockopt(zmq.SNDHWM, 0)
        self._sender.setsockopt(zmq.LINGER, 0)
        self._sender.bind(self._send_addr)

        # Proc PUSH → Orchestrator PULL (results)
        self._recv_addr = _get_zmq_ipc_path()
        self._receiver = self._zmq_ctx.socket(zmq.PULL)
        self._receiver.setsockopt(zmq.RCVHWM, 0)
        self._receiver.bind(self._recv_addr)

        # Build stage payload
        engine_args = _to_dict(self.engine_args)
        runtime_cfg = _to_dict(getattr(self.stage_config, "runtime", {}))
        stage_payload: dict[str, Any] = {
            "stage_id": self.stage_id,
            "engine_args": engine_args,
            "runtime": runtime_cfg,
            "shm_threshold_bytes": shm_threshold_bytes,
            "connectors_config": connectors_config or {},
            "stage_type": self.stage_type,
            "engine_input_source": self.engine_input_source,
        }

        # Set logging prefix
        old_env = os.environ.get("VLLM_LOGGING_PREFIX")
        new_env = f"[Stage-{self.stage_id}] {'' if old_env is None else old_env}"
        os.environ["VLLM_LOGGING_PREFIX"] = new_env

        try:
            from vllm_omni.entrypoints.stage_core_proc import stage_core_proc_entry

            ctx = mp.get_context("spawn")
            self._proc = ctx.Process(
                target=stage_core_proc_entry,
                args=(
                    model,
                    stage_payload,
                    # Note: addresses are swapped — proc connects to our bind addresses
                    # Proc sends to recv_addr (our PULL), Proc receives from send_addr (our PUSH)
                    self._recv_addr,  # proc's send target = our receiver
                    self._send_addr,  # proc's recv source = our sender
                    self._stage_init_timeout,
                    batch_timeout,
                    log_stats,
                    self.final_output,
                    self.final_output_type,
                ),
            )
            self._proc.start()
            logger.info("[StageCoreClient] Stage-%d process started (pid=%s)", self.stage_id, self._proc.pid)
        finally:
            if old_env is None:
                os.environ.pop("VLLM_LOGGING_PREFIX", None)
            else:
                os.environ["VLLM_LOGGING_PREFIX"] = old_env

    def wait_ready(self, timeout: int = 300) -> None:
        """Block until the subprocess sends a READY message.

        Extracts vllm_config, tokenizer, and is_tracing_enabled from
        the ready payload.
        """
        assert self._receiver is not None, "Must call start() before wait_ready()"

        deadline = time.time() + timeout
        poller = zmq.Poller()
        poller.register(self._receiver, zmq.POLLIN)

        while time.time() < deadline:
            events = dict(poller.poll(timeout=1000))  # 1s poll
            if self._receiver in events:
                data = self._receiver.recv()
                msg = pickle.loads(data)
                if msg.get("type") == "stage_ready":
                    vllm_config = msg.get("vllm_config")
                    if vllm_config is not None:
                        self._vllm_config = vllm_config
                    tokenizer = msg.get("tokenizer")
                    if tokenizer is not None:
                        self._tokenizer = tokenizer
                    is_tracing = msg.get("is_tracing_enabled")
                    if is_tracing is not None:
                        self._is_tracing_enabled = is_tracing
                    self._ready = True
                    logger.info("[StageCoreClient] Stage-%d ready", self.stage_id)
                    return

        logger.warning(
            "[StageCoreClient] Stage-%d ready timeout after %ds", self.stage_id, timeout
        )

    def stop(self) -> None:
        """Stop the subprocess gracefully."""
        from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK

        if self._sender is not None:
            try:
                self._sender.send(pickle.dumps(SHUTDOWN_TASK))
            except Exception as e:
                logger.warning("[StageCoreClient] Failed to send shutdown: %s", e)

        if self._proc is not None:
            try:
                self._proc.join(timeout=10)
            except Exception as e:
                logger.debug("[StageCoreClient] join() failed: %s", e)
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception as e:
                    logger.warning("[StageCoreClient] terminate() failed: %s", e)

        # Close ZMQ
        if self._sender is not None:
            self._sender.close()
            self._sender = None
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None

    # ----- Communication -----

    def submit(self, payload: dict[str, Any]) -> None:
        """Submit a task to the stage subprocess via ZMQ.

        Injects global_request_id into additional_information for
        cross-stage ID consistency (same as OmniStage.submit).
        """
        assert self._sender is not None, "Stage not started"

        # Inject global request_id
        if "request_id" in payload and "engine_inputs" in payload:
            req_id = payload["request_id"]
            ein = payload["engine_inputs"]

            def _inject_global_id(target_ein):
                if isinstance(target_ein, dict):
                    if "additional_information" not in target_ein:
                        target_ein["additional_information"] = {}
                    if target_ein["additional_information"] is None:
                        target_ein["additional_information"] = {}
                    if isinstance(target_ein["additional_information"], dict):
                        target_ein["additional_information"]["global_request_id"] = [str(req_id)]

            if isinstance(ein, list):
                for item in ein:
                    _inject_global_id(item)
            else:
                _inject_global_id(ein)

        self._sender.send(pickle.dumps(payload))

    def try_collect(self) -> dict[str, Any] | None:
        """Non-blocking poll for results from the stage subprocess."""
        assert self._receiver is not None, "Stage not started"
        try:
            data = self._receiver.recv(zmq.NOBLOCK)
            return pickle.loads(data)
        except zmq.Again:
            return None

    # ----- Profiling -----

    def stop_profile(self) -> dict:
        """Send profiler stop command and wait for result."""
        assert self._sender is not None and self._receiver is not None

        self.submit({"type": OmniStageTaskType.PROFILER_STOP})

        # Wait for profiler result with timeout
        poller = zmq.Poller()
        poller.register(self._receiver, zmq.POLLIN)
        deadline = time.time() + 600  # 10min timeout

        while time.time() < deadline:
            events = dict(poller.poll(timeout=1000))
            if self._receiver in events:
                data = self._receiver.recv()
                response = pickle.loads(data)
                if isinstance(response, dict) and response.get("type") == "profiler_result":
                    return response.get("data", {})
                # Got a non-profiler message; for now just log and continue
                logger.warning(
                    "[StageCoreClient] Stage-%d got unexpected msg while waiting for profiler: %s",
                    self.stage_id,
                    response.get("type", "unknown"),
                )

        logger.error("[StageCoreClient] Stage-%d profiler result timeout", self.stage_id)
        return {}

    # ----- Cross-stage input processing -----

    def process_engine_inputs(
        self, stage_list: list[Any], prompt: OmniPromptType | None = None
    ) -> list[OmniTokensPrompt]:
        """Derive inputs for this stage from upstream outputs.

        Ported from OmniStage.process_engine_inputs().
        """
        if self.custom_process_input_func is not None:
            return self.custom_process_input_func(
                stage_list, self.engine_input_source, prompt, self.requires_multimodal_data
            )

        engine_inputs = []
        if len(self.engine_input_source) == 0:
            raise ValueError("engine_input_source is empty")

        source_stage_id = self.engine_input_source[0]
        source_outputs = stage_list[source_stage_id].engine_outputs
        if not isinstance(prompt, list):
            prompt = [prompt]

        multi_modal_data = {
            source_output.request_id: p.get("multi_modal_data", None)
            for source_output, p in zip(source_outputs, prompt)
        }

        for source_output in source_outputs:
            engine_input = OmniTokensPrompt(
                prompt_token_ids=source_output.outputs[0].token_ids,
                multi_modal_data=(
                    multi_modal_data[source_output.request_id]
                    if self.requires_multimodal_data and multi_modal_data
                    else None
                ),
            )
            engine_inputs.append(engine_input)
        return engine_inputs
