"""Stage core data types for vLLM-Omni.

Shared, backend-agnostic marker bases plus the backend-concrete request/output
types for the stage process/client model.

LLM concrete types
------------------
``StageLLMCore{Request,Output,Outputs}`` subclass both the shared stage marker and
the canonical omni engine-core type (``OmniEngineCore*`` from ``vllm_omni.engine``),
so they carry the omni fields (e.g. ``additional_information``) while also being
recognizable as stage-core types. ``vllm_omni.patch`` rebinds vLLM's ``EngineCore*``
module globals to **these** ``StageLLMCore*`` types, so the stage LLM client/proc
decode/encode exactly them (their read-only assertions therefore hold:
``EngineCoreOutputs is StageLLMCoreOutputs``).

Diffusion concrete types
------------------------
``StageDiffusionCore{Request,Output,Outputs}`` are defined here and subclass the
shared markers. Diffusion's runtime payloads (``OmniDiffusionRequest`` /
``OmniRequestOutput``) are plain dataclasses and cannot be msgspec bases, so these
structs formalize the diffusion *wire* payload directly; heavy nested payloads are
typed ``Any`` and (de)serialized by ``OmniMsgpackEncoder`` / ``OmniMsgpackDecoder``.

Shared markers
--------------
``StageCore{Request,Output,Outputs}`` are field-free marker bases so orchestration
code can treat values polymorphically. They must stay empty.
"""

from __future__ import annotations

from typing import Any

import msgspec

# Reuse the canonical omni engine-core types + serialized payloads (single source
# of truth; patch.py rebinds vLLM's EngineCore* globals to the OmniEngineCore*).
from vllm_omni.engine import (
    AdditionalInformationEntry as AdditionalInformationEntry,
    AdditionalInformationPayload as AdditionalInformationPayload,
    OmniEngineCoreOutput,
    OmniEngineCoreOutputs,
    OmniEngineCoreRequest,
    PromptEmbedsPayload as PromptEmbedsPayload,
)

# =============================================================================
# Shared marker bases (FIELD-FREE — never add fields)
# =============================================================================


class StageCoreRequest(msgspec.Struct):
    """Marker base for all stage core requests. Never add fields."""


class StageCoreOutput(msgspec.Struct):
    """Marker base for all stage core outputs. Never add fields."""


class StageCoreOutputs(msgspec.Struct):
    """Marker base for all stage core output batches. Never add fields."""


# =============================================================================
# Concrete LLM stage core types == canonical omni engine-core types (reused)
# =============================================================================
# Aliased rather than redefined so there is a single wire type: patch.py rebinds
# vLLM's EngineCore* globals to these, which is what the stage LLM client/proc
# assert against.

class StageLLMCoreRequest(StageCoreRequest, OmniEngineCoreRequest):
    """LLM stage request."""


class StageLLMCoreOutput(StageCoreOutput, OmniEngineCoreOutput):
    """LLM stage output."""


class StageLLMCoreOutputs(StageCoreOutputs, OmniEngineCoreOutputs):
    """LLM stage outputs."""


# =============================================================================
# Concrete diffusion stage core types (subclass the marker ONLY)
# =============================================================================
# Unlike the LLM types, diffusion's runtime payloads (``OmniDiffusionRequest`` /
# ``OmniRequestOutput``) are plain ``@dataclass`` objects and cannot serve as
# msgspec base classes. These structs formalize the diffusion *wire* payload
# directly: they subclass only the shared markers and re-declare the fields that
# cross the StageDiffusionCoreProc <-> StageDiffusionCoreClient ZMQ boundary.


class StageDiffusionCoreRequest(StageCoreRequest):
    """Wire payload for a diffusion stage add-request.

    ``sampling_params`` is the plain-dict form produced by
    ``StageDiffusionCoreClient.sampling_params_to_dict`` (non-serializable fields
    already stripped).
    """

    request_id: str
    prompt: Any
    sampling_params: dict[str, Any]
    kv_sender_info: dict[int, dict[str, Any]] | None = None


class StageDiffusionCoreOutput(StageCoreOutput):
    """One diffusion result produced by ``StageDiffusionCoreProc``.

    ``output`` carries the full ``OmniRequestOutput`` opaquely for a successful
    result. For a failed request the proc sends a control ``error`` frame; the
    client materializes it into this struct with ``error`` set (and ``output``
    left ``None``) so the consumer can build an error ``OmniRequestOutput``
    without the client itself depending on that type.
    """

    request_id: str
    finished: bool = True
    output: Any = None
    error: str | None = None
    status_code: int | None = None
    error_type: str | None = None


class StageDiffusionCoreOutputs(StageCoreOutputs):
    """Batch envelope of diffusion outputs (mirrors ``StageLLMCoreOutputs``).

    Streaming steps each send a single-element batch; the client drains
    ``outputs`` in order.
    """

    outputs: list[StageDiffusionCoreOutput] = []
