# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
StageCoreProc: Stage-specific EngineCore process.

Extends vLLM's EngineCoreProc with stage-specific device setup,
plugin loading, and connector initialization. Each pipeline stage
runs one StageCoreProc in a dedicated background process.
"""
import multiprocessing as mp
import os
import signal
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor import Executor

from vllm_omni.entrypoints.stage_utils import set_stage_devices

logger = init_logger(__name__)


class StageCoreProc(EngineCoreProc):
    """Stage-specific EngineCore that runs in a background process.

    Each pipeline stage runs one StageCoreProc with:
    - Dedicated GPU(s) assigned via CUDA_VISIBLE_DEVICES
    - Omni-specific plugins loaded
    - Stage-specific model loaded (e.g., thinker, talker)
    - ZMQ-based communication with StageCoreClient (inherited from EngineCoreProc)
    """

    @staticmethod
    def run_stage_core(
        *,
        # Stage-specific parameters
        stage_id: int = 0,
        stage_type: str = "llm",
        devices: str | None = None,
        connectors_config: dict[str, Any] | None = None,
        stage_init_timeout: int = 300,
        # Standard EngineCoreProc parameters (passed by CoreEngineProcManager)
        vllm_config: VllmConfig,
        local_client: bool = True,
        handshake_address: str = "",
        executor_class: type[Executor] | None = None,
        log_stats: bool = False,
        client_handshake_address: str | None = None,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ) -> None:
        """Entry point for launching StageCoreProc in a background process.

        Performs stage-specific setup (device assignment, plugin loading)
        before delegating to the standard EngineCoreProc initialization.

        Args:
            stage_id: Unique identifier for this stage in the pipeline.
            stage_type: Type of stage ("llm" or "diffusion").
            devices: Device specification (e.g., "0,1" for GPUs 0 and 1).
            connectors_config: Configuration for stage connectors (KV transfer).
            stage_init_timeout: Timeout for stage initialization in seconds.
            vllm_config: vLLM configuration object.
            local_client: Whether the client is in the same process group.
            handshake_address: ZMQ address for handshake with client.
            executor_class: Executor implementation class.
            log_stats: Whether to log statistics.
            client_handshake_address: Optional secondary handshake address.
            dp_rank: Data parallel rank (always 0 for stages).
            local_dp_rank: Local data parallel rank (always 0 for stages).
        """
        # ---- Stage-specific setup (before engine core init) ----

        # 1. Set up stage-specific device visibility
        if devices is not None:
            set_stage_devices(stage_id, devices)
            logger.info(
                "[StageCoreProc-%d] Device setup complete: devices=%s",
                stage_id,
                devices,
            )

        # 2. Load omni-specific plugins
        try:
            from vllm_omni.plugins import load_omni_general_plugins

            load_omni_general_plugins()
        except Exception as e:
            logger.warning(
                "[StageCoreProc-%d] Failed to load omni plugins: %s",
                stage_id,
                e,
            )

        # 3. Ensure spawn method for child processes
        if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # 4. Resolve worker class for the stage
        try:
            _resolve_worker_cls_for_stage(stage_type, vllm_config)
        except Exception as e:
            logger.debug(
                "[StageCoreProc-%d] Worker class resolution: %s",
                stage_id,
                e,
            )

        logger.info(
            "[StageCoreProc-%d] Stage setup complete, launching EngineCore "
            "(type=%s, devices=%s)",
            stage_id,
            stage_type,
            devices,
        )

        # ---- Delegate to standard EngineCoreProc ----
        EngineCoreProc.run_engine_core(
            vllm_config=vllm_config,
            local_client=local_client,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=log_stats,
            client_handshake_address=client_handshake_address,
            dp_rank=dp_rank,
            local_dp_rank=local_dp_rank,
        )


def _resolve_worker_cls_for_stage(stage_type: str, vllm_config: VllmConfig) -> None:
    """Resolve and set the worker class based on stage type.

    For LLM stages, uses the omni AR worker or generation worker
    based on the model configuration.
    """
    if stage_type == "diffusion":
        return

    # Check if a worker class is already configured
    if vllm_config.parallel_config.worker_cls:
        return

    try:
        from vllm_omni.platforms import current_omni_platform

        model_stage = getattr(vllm_config.model_config, "model_stage", None)
        worker_type = getattr(vllm_config.model_config, "worker_type", None)

        if worker_type == "ar":
            vllm_config.parallel_config.worker_cls = (
                current_omni_platform.get_omni_ar_worker_cls()
            )
        elif worker_type == "generation":
            vllm_config.parallel_config.worker_cls = (
                current_omni_platform.get_omni_generation_worker_cls()
            )
        elif model_stage:
            logger.debug(
                "No worker_type specified for model_stage=%s, using default",
                model_stage,
            )
    except ImportError:
        pass
