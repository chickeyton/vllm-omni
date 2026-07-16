"""vLLM-Omni stage process/replica module.

Public surface for the refactored stage LLM **and** diffusion core clients /
procs / proc-managers, the shared stage core contract + data types, and the
head-side replica pool.

The package ``__init__`` is **lazy** (PEP 562): it imports nothing heavy at module
scope, so importing the lightweight ``stage_core_types`` submodule (e.g. from
``vllm_omni.patch``) does not transitively pull in the engine-client stack
(``AsyncMPClient``, the omni connector modules) via this init.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# name -> (submodule, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    # contract (stage_core_client)
    "StageCoreClientBase": ("stage_core_client", "StageCoreClientBase"),
    # data types (stage_core_types, lightweight)
    "StageCoreRequest": ("stage_core_types", "StageCoreRequest"),
    "StageCoreOutput": ("stage_core_types", "StageCoreOutput"),
    "StageCoreOutputs": ("stage_core_types", "StageCoreOutputs"),
    "StageLLMCoreRequest": ("stage_core_types", "StageLLMCoreRequest"),
    "StageLLMCoreOutput": ("stage_core_types", "StageLLMCoreOutput"),
    "StageLLMCoreOutputs": ("stage_core_types", "StageLLMCoreOutputs"),
    "StageDiffusionCoreRequest": ("stage_core_types", "StageDiffusionCoreRequest"),
    "StageDiffusionCoreOutput": ("stage_core_types", "StageDiffusionCoreOutput"),
    "StageDiffusionCoreOutputs": ("stage_core_types", "StageDiffusionCoreOutputs"),
    "PromptEmbedsPayload": ("stage_core_types", "PromptEmbedsPayload"),
    "AdditionalInformationEntry": ("stage_core_types", "AdditionalInformationEntry"),
    "AdditionalInformationPayload": ("stage_core_types", "AdditionalInformationPayload"),
    # LLM clients (stage_llm_core_client, heavy)
    "StageLLMCoreClientBase": ("stage_llm_core_client", "StageLLMCoreClientBase"),
    "StageLLMCoreClient": ("stage_llm_core_client", "StageLLMCoreClient"),
    "DPLBStageLLMCoreClient": ("stage_llm_core_client", "DPLBStageLLMCoreClient"),
    # LLM proc / proc-manager (heavy)
    "StageLLMCoreProc": ("stage_llm_core_proc", "StageLLMCoreProc"),
    "StageLLMCoreProcManager": ("stage_llm_core_proc_manager", "StageLLMCoreProcManager"),
    # diffusion client / proc / proc-manager (heavy)
    "StageDiffusionCoreClient": ("stage_diffusion_core_client", "StageDiffusionCoreClient"),
    "create_diffusion_client": ("stage_diffusion_core_client", "create_diffusion_client"),
    "StageDiffusionCoreProc": ("stage_diffusion_core_proc", "StageDiffusionCoreProc"),
    "StageDiffusionCoreProcManager": (
        "stage_diffusion_core_proc_manager",
        "StageDiffusionCoreProcManager",
    ),
    # head-side replica pool (heavy)
    "StageReplicaPool": ("stage_replica_pool", "StageReplicaPool"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        submodule, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, attr)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:
    from vllm_omni.engine.stage.stage_core_client import (
        StageCoreClientBase as StageCoreClientBase,
    )
    from vllm_omni.engine.stage.stage_core_types import (
        AdditionalInformationEntry as AdditionalInformationEntry,
        AdditionalInformationPayload as AdditionalInformationPayload,
        PromptEmbedsPayload as PromptEmbedsPayload,
        StageCoreOutput as StageCoreOutput,
        StageCoreOutputs as StageCoreOutputs,
        StageCoreRequest as StageCoreRequest,
        StageDiffusionCoreOutput as StageDiffusionCoreOutput,
        StageDiffusionCoreOutputs as StageDiffusionCoreOutputs,
        StageDiffusionCoreRequest as StageDiffusionCoreRequest,
        StageLLMCoreOutput as StageLLMCoreOutput,
        StageLLMCoreOutputs as StageLLMCoreOutputs,
        StageLLMCoreRequest as StageLLMCoreRequest,
    )
    from vllm_omni.engine.stage.stage_diffusion_core_client import (
        StageDiffusionCoreClient as StageDiffusionCoreClient,
        create_diffusion_client as create_diffusion_client,
    )
    from vllm_omni.engine.stage.stage_diffusion_core_proc import (
        StageDiffusionCoreProc as StageDiffusionCoreProc,
    )
    from vllm_omni.engine.stage.stage_diffusion_core_proc_manager import (
        StageDiffusionCoreProcManager as StageDiffusionCoreProcManager,
    )
    from vllm_omni.engine.stage.stage_llm_core_client import (
        DPLBStageLLMCoreClient as DPLBStageLLMCoreClient,
        StageLLMCoreClient as StageLLMCoreClient,
        StageLLMCoreClientBase as StageLLMCoreClientBase,
    )
    from vllm_omni.engine.stage.stage_llm_core_proc import StageLLMCoreProc as StageLLMCoreProc
    from vllm_omni.engine.stage.stage_llm_core_proc_manager import (
        StageLLMCoreProcManager as StageLLMCoreProcManager,
    )
    from vllm_omni.engine.stage.stage_replica_pool import StageReplicaPool as StageReplicaPool
