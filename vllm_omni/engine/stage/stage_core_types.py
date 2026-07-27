"""Stage core data types for vLLM-Omni.

Shared, backend-agnostic marker bases plus the backend-concrete request/output
types for the stage process/client model.

LLM concrete types
------------------
``StageLLMCore{Request,Output,Outputs}`` subclass both the shared stage marker and
vLLM's canonical engine-core type (``EngineCore*`` from ``vllm.v1.engine``), adding
the omni-specific fields (e.g. ``additional_information``, ``multimodal_output``)
while also being recognizable as stage-core types. ``vllm_omni.patch`` rebinds
vLLM's ``EngineCore*`` module globals to **these** ``StageLLMCore*`` types, so the
stage LLM client/proc decode/encode exactly them (their read-only assertions
therefore hold: ``EngineCoreOutputs is StageLLMCoreOutputs``).

These are the single source of truth for the omni LLM wire types; they supersede
the former ``OmniEngineCore*`` structs that used to live in ``vllm_omni.engine``.

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
import torch
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)

# Serialized payload structs, re-exported here so the stage wire types and their
# payloads are importable from one place.
from vllm_omni.engine import (
    AdditionalInformationEntry as AdditionalInformationEntry,
)
from vllm_omni.engine import (
    AdditionalInformationPayload as AdditionalInformationPayload,
)
from vllm_omni.engine import (
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
# Concrete LLM stage core types
# =============================================================================
# vLLM's EngineCore* extended with the omni fields. These are the single wire
# type: patch.py rebinds vLLM's EngineCore* globals to these, which is what the
# stage LLM client/proc assert against.


class StageLLMCoreRequest(StageCoreRequest, EngineCoreRequest):
    """LLM stage request.

    Note: prompt_embeds is inherited from EngineCoreRequest
    (torch.Tensor | None). PromptEmbedsPayload should be decoded to
    torch.Tensor before constructing this request.
    """

    # Optional additional information dictionary (serialized)
    additional_information: AdditionalInformationPayload | None = None
    # Runner-owned runtime payload. This is materialized directly into
    # GPUModelRunner.model_intermediate_buffer instead of using the deprecated
    # additional_information request transport.
    model_intermediate_buffer: dict[str, Any] | None = None

    @classmethod
    def from_vllm_request(
        cls,
        request: EngineCoreRequest,
        *,
        prompt_embeds: torch.Tensor | None = None,
        additional_information: AdditionalInformationPayload | None = None,
        model_intermediate_buffer: dict[str, Any] | None = None,
    ) -> StageLLMCoreRequest:
        """Clone an EngineCoreRequest into a StageLLMCoreRequest with optional payload overrides."""

        if prompt_embeds is None:
            prompt_embeds = request.prompt_embeds
        if additional_information is None:
            additional_information = getattr(request, "additional_information", None)
        if model_intermediate_buffer is None:
            model_intermediate_buffer = getattr(request, "model_intermediate_buffer", None)

        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            data_parallel_rank=request.data_parallel_rank,
            prompt_embeds=prompt_embeds,
            client_index=request.client_index,
            current_wave=request.current_wave,
            priority=request.priority,
            trace_headers=request.trace_headers,
            resumable=request.resumable,
            external_req_id=request.external_req_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
            additional_information=additional_information,
            model_intermediate_buffer=model_intermediate_buffer,
        )


class StageLLMCoreOutput(StageCoreOutput, EngineCoreOutput):
    """LLM stage output."""

    # Dedicated channel for multimodal outputs (image/audio/latent).
    # pooling_output is inherited from EngineCoreOutput as torch.Tensor | None
    # and retains its original vLLM semantics for pooling/embedding tasks.
    multimodal_output: dict[str, torch.Tensor] | None = None
    # Finished flag for streaming input segment
    is_segment_finished: bool | None = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None


class StageLLMCoreOutputs(StageCoreOutputs, EngineCoreOutputs):
    """LLM stage outputs."""

    outputs: list[StageLLMCoreOutput] = []


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
