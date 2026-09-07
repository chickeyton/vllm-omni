# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""PersonaPlex full-duplex model plugin: engine policy and session policy in one class."""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexInputMode,
    DuplexOutputDecision,
)
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    DuplexModelSessionState,
    DuplexRuntimeConfigError,
    EncodeAudio,
    reject_changed_runtime_value,
)
from vllm_omni.model_executor.models.personaplex.duplex.config import DEFAULT_PERSONA
from vllm_omni.model_executor.models.personaplex.duplex.data_plane import (
    PersonaPlexDataPlaneContext,
    PersonaPlexDataPlaneSession,
)
from vllm_omni.model_executor.models.personaplex.duplex.input import (
    PersonaPlexPcmAppendBuffer,
)

_PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "personaplex_prefill_slots",
        "personaplex_model_path",
    }
)
_FRAME_BYTES = 1920 * 4


def _validated_frame_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("PersonaPlex duplex append payload must be a mapping")
    if payload.get("format") != "pcm_f32le":
        raise ValueError("PersonaPlex duplex append format must be pcm_f32le")
    if payload.get("sample_rate_hz") != 24000:
        raise ValueError("PersonaPlex duplex append sample_rate_hz must be 24000")
    audio = payload.get("audio")
    if not isinstance(audio, str):
        raise ValueError("PersonaPlex duplex append audio must be base64 pcm_f32le")
    try:
        raw = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("PersonaPlex duplex append audio is not valid base64") from exc
    if len(raw) != _FRAME_BYTES:
        raise ValueError("PersonaPlex duplex append must contain exactly 1920 samples")
    return dict(payload)


@dataclass(slots=True)
class PersonaPlexServingSessionState(DuplexModelSessionState):
    """Model-owned state of one PersonaPlex duplex session (owned by the session runner)."""

    audio_buffer: PersonaPlexPcmAppendBuffer = field(default_factory=PersonaPlexPcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    context_locked: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None


class PersonaPlexDuplexPlugin(DuplexModelPlugin):
    """PersonaPlex is pure lockstep: every session is model-native duplex."""

    plugin_id = "personaplex"
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = _PRIVATE_RUNTIME_CONFIG_KEYS
    collect_outputs_on_append = False
    silence_continuation_samples = 16000

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = PersonaPlexDataPlaneSession(encode_audio)

    # ---- engine policy (the resumable Stage0 request) ----

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        if not defaults:
            return defaults
        configured = list(defaults)
        stage0 = defaults[0]
        if isinstance(stage0, SamplingParams):
            stage0 = stage0.clone()
            stage0.temperature = 0.0
            stage0.top_k = 1
            stage0.max_tokens = 1
            configured[0] = stage0
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        if mode is not DuplexInputMode.APPEND_AUDIO_CHUNK:
            raise ValueError(f"PersonaPlex does not support duplex input mode {mode.value!r}")
        normalized_payload = _validated_frame_payload(payload)
        prefill_slots = runtime_config.get("personaplex_prefill_slots", 0)
        try:
            prefill_slots = max(0, int(prefill_slots))
        except (TypeError, ValueError) as exc:
            raise ValueError("personaplex_prefill_slots must be a non-negative integer") from exc
        prompt_slots = 1 + (prefill_slots if seq <= 1 else 0)
        return DuplexAppendPlan(
            prompt={
                "prompt_token_ids": [0] * prompt_slots,
                "model_intermediate_buffer": {
                    "request_id": request_id,
                    "global_request_id": [fence.session_id],
                    "duplex": {
                        "data_plane": True,
                        "fence": fence,
                        "session_id": fence.session_id,
                        "incarnation": fence.incarnation,
                        "epoch": fence.epoch,
                        "turn_id": fence.turn_id,
                        "response_seq": fence.response_seq,
                        "seq": seq,
                        "turn_seq": turn_seq,
                        "mode": mode.value,
                        "payload": normalized_payload,
                        "final": final,
                        "session_config": dict(session_config),
                        "runtime_config": dict(runtime_config),
                        "scheduler_token_budget": prompt_slots,
                    },
                },
            }
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        del stage_id, final_stage_id, segment_finished
        del segment_token_ids, segment_output_metadata, output
        return None

    # ---- session policy ----

    def create_session_state(self) -> PersonaPlexServingSessionState:
        return PersonaPlexServingSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        supports_multi_session = max_sessions > 1
        return DuplexCapabilities(
            supports_model_native_turn_policy=True,
            supports_external_turn_signal=False,
            supports_client_commit=False,
            supports_barge_in=False,
            supports_playback_ack=True,
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_rollback_to_checkpoint=False,
            supports_turn_commit_only=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=True,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=supports_multi_session,
            supports_multi_session_same_replica=supports_multi_session,
            supports_session_lease=True,
            supports_session_resume=False,
            session_admission_mode="engine_managed",
            supports_audio_truncate=False,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            implementation_level="model_native_duplex",
            adapter_patterns=["scheduler_data_plane"],
            input_modes=["append_audio_chunk"],
            signal_sources=["model_native", "client_event"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=80,
            target_barge_in_latency_ms=None,
        )

    def validate_client_extra_body(self, extra_body: object) -> None:
        if not isinstance(extra_body, dict):
            return
        private = sorted(_PRIVATE_RUNTIME_CONFIG_KEYS.intersection(extra_body))
        if private:
            raise DuplexRuntimeConfigError("PersonaPlex runtime configuration is server-owned: " + ", ".join(private))

    async def prepare_runtime_config(self, config: DuplexSessionConfig, *, model_config: Any) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            raise DuplexRuntimeConfigError("PersonaPlex model path is unavailable")
        voice = self._voice_name(config.voice)
        instructions = config.instructions or DEFAULT_PERSONA
        from vllm_omni.model_executor.models.personaplex.duplex.stage0 import (
            personaplex_prefill_slots,
        )

        try:
            prefill_slots = await asyncio.to_thread(
                personaplex_prefill_slots,
                model_path,
                voice,
                str(instructions),
            )
        except Exception as exc:
            raise DuplexRuntimeConfigError(f"PersonaPlex voice/persona prefill could not be prepared: {exc}") from exc
        return {
            "personaplex_model_path": model_path,
            "personaplex_voice_prompt": voice,
            "personaplex_persona": str(instructions),
            "personaplex_prefill_slots": prefill_slots,
        }

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        new_persona = str(config.instructions or DEFAULT_PERSONA)
        reject_changed_runtime_value(
            new_persona,
            current.get("personaplex_persona"),
            message="PersonaPlex persona (instructions) cannot be changed after the session is created",
            code="persona_update_unsupported",
        )
        new_voice = self._voice_name(config.voice)
        reject_changed_runtime_value(
            new_voice,
            current.get("personaplex_voice_prompt"),
            message="PersonaPlex voice cannot be changed after the session is created",
            code="voice_update_unsupported",
        )
        runtime_config = deepcopy(dict(current))
        runtime_config["personaplex_voice_prompt"] = new_voice
        runtime_config["personaplex_persona"] = new_persona
        return runtime_config

    def data_plane_context(
        self,
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> PersonaPlexDataPlaneContext:
        return PersonaPlexDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )

    @staticmethod
    def _voice_name(value: object) -> str:
        voice = value if isinstance(value, str) and value else "NATF2.pt"
        path = PurePath(voice)
        if path.name != voice or path.suffix != ".pt" or any(part == ".." for part in path.parts):
            raise DuplexRuntimeConfigError("PersonaPlex voice must be a bundled .pt basename")
        return voice


__all__ = ["PersonaPlexDuplexPlugin", "PersonaPlexServingSessionState"]
