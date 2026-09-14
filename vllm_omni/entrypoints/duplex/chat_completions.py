# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``/v1/chat/completions`` on a duplex model, as an ordinary Realtime client.

A duplex server has no turn-based request path, so one chat request becomes one
short-lived duplex session. How the prompt goes in depends on what it is, and
that difference is the model's, not this layer's:

* **Speech is a turn.** Audio content is appended to the input buffer and
  committed, exactly as a websocket client does it.
* **Text is not.** A model-native model decides to speak from the audio it
  hears, so silence is its signal *not* to take a turn and a text prompt has no
  turn to start. It reaches the model instead as the session's seeded opening
  turn (``DuplexSessionConfig.initial_user_text``), and the session is given
  silence units to generate on. Only a model that declares
  ``DuplexCapabilities.supports_chat_completions`` can be reached this way; any
  other is told so before it can hang.

Either way the session answers by itself. There is no ``response.create``: it
cannot drive a seeded turn, because the priming units are consumed as "listen"
before it arrives.

Two properties a caller should know, both inherent to serving HTTP on a duplex
session rather than artefacts of this adapter:

* **Every request holds an admission slot** for its lifetime, so
  ``duplex_session.max_sessions`` bounds HTTP concurrency as well as realtime
  sessions.
* **Latency follows the model's own clock.** A model-native session generates
  per audio unit, so a short answer still costs what that many units cost.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import aclosing, suppress
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from openai.types.chat.chat_completion_audio import ChatCompletionAudio
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse, UsageInfo
from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.events import (
    AudioDelta,
    ErrorEvent,
    ResponseDone,
    SessionClosed,
    TextDelta,
    TranscriptDelta,
)
from vllm_omni.engine.duplex.messages import DuplexSessionError

if TYPE_CHECKING:
    from fastapi import Request

    from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle

logger = init_logger(__name__)

#: Sample rate of the silence units a seeded turn is given to generate on.
_PRIMING_SAMPLE_RATE_HZ = 16000

#: Chat content parts carrying audio input, by OpenAI content-part type.
_AUDIO_PART_TYPES = frozenset({"input_audio", "audio"})

#: Everything a Realtime conversation item can carry. Images and video have no
#: place in one (``realtime_item_to_history_message`` drops them), so a request
#: asking for them is refused rather than silently answered without them.
_SUPPORTED_PART_TYPES = _AUDIO_PART_TYPES | {"text", "input_text"}

#: Realtime ``response.status`` -> OpenAI ``finish_reason``.
_FINISH_REASON_BY_STATUS = {
    "completed": "stop",
    "incomplete": "length",
    "cancelled": "stop",
    "failed": "stop",
}

#: Duplex error codes that are the caller's fault rather than the server's.
_CLIENT_ERROR_CODES = frozenset(
    {
        "response_create_without_input",
        "unsupported_audio_format",
        "unsupported_turn_detection",
        "unsupported_native_response_options",
        "invalid_request",
        "invalid_value",
        "missing_required_parameter",
    }
)


class DuplexChatCompletionsAdapter:
    """Serve one chat request on one duplex session.

    Duck-typed against ``OmniOpenAIServingChat``: the route only calls
    ``create_chat_completion`` and ``create_error_response`` on whatever
    ``app.state.openai_serving_chat`` holds.
    """

    def __init__(self, *, duplex_omni: DuplexOmni, model_name: str) -> None:
        self._omni = duplex_omni
        self._model_name = model_name

    # ------------------------------------------------------------------ #
    # Entry point                                                        #
    # ------------------------------------------------------------------ #

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> ChatCompletionResponse | AsyncGenerator[str, None] | ErrorResponse:
        del raw_request  # disconnects are handled by closing the session in finally
        unsupported = self._unsupported_reason(request)
        if unsupported is not None:
            return self.create_error_response(unsupported)

        try:
            handle = await self._omni.open_session(self._session_config(request))
        except DuplexSessionError as exc:
            return self._error_from_code(str(exc), exc.code)
        except Exception as exc:
            logger.exception("failed to open a duplex session for a chat completion: %s", exc)
            return self.create_error_response(
                f"could not open a duplex session: {exc}",
                err_type="internal_server_error",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        # Capabilities are the model's answer, and they only arrive with the
        # open. A text prompt needs a model that can be seeded with it; without
        # that the turn would never complete and the caller would wait out the
        # session's idle timeout to learn so.
        if not self._has_audio(request) and not handle.capabilities.supports_chat_completions:
            await self._close(handle)
            return self.create_error_response(
                "this duplex model answers speech input only: send audio content, or use /v1/realtime?duplex=1",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        if request.stream:
            # The generator owns the session from here, including closing it.
            return self._stream(handle, request)
        try:
            return await self._collect(handle, request)
        finally:
            await self._close(handle)

    def create_error_response(
        self,
        message: str,
        *,
        err_type: str = "BadRequestError",
        status_code: int = HTTPStatus.BAD_REQUEST,
    ) -> ErrorResponse:
        return ErrorResponse(error=ErrorInfo(message=message, type=err_type, code=int(status_code)))

    # ------------------------------------------------------------------ #
    # Request -> session                                                 #
    # ------------------------------------------------------------------ #

    def _unsupported_reason(self, request: ChatCompletionRequest) -> str | None:
        """Options one duplex turn has no representation for."""
        if (request.n or 1) > 1:
            return "n > 1 is not supported by a duplex model: one request is one session turn"
        if request.logprobs:
            return "logprobs is not supported by a duplex model"
        if request.tools:
            return "tools are not supported on /v1/chat/completions for a duplex model"
        unsupported_parts = sorted(self._unsupported_content_types(request))
        if unsupported_parts:
            # A Realtime conversation item carries text and audio only
            # (``realtime_item_to_history_message``). Accepting an image and
            # dropping it would answer a question the caller did not ask.
            return (
                f"content of type {', '.join(unsupported_parts)} is not supported on a duplex model: "
                "a duplex turn takes text and audio"
            )
        return None

    @classmethod
    def _has_audio(cls, request: ChatCompletionRequest) -> bool:
        """Whether the prompt contains speech, which is a turn on its own."""
        return any(True for message in request.messages for _ in cls._audio_parts(message))

    @staticmethod
    def _unsupported_content_types(request: ChatCompletionRequest) -> set[str]:
        found: set[str] = set()
        for message in request.messages:
            content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                kind = part.get("type")
                if isinstance(kind, str) and kind not in _SUPPORTED_PART_TYPES:
                    found.add(kind)
        return found

    def _session_config(self, request: ChatCompletionRequest) -> DuplexSessionConfig:
        """One caller-driven turn.

        Built typed rather than as a Realtime session object: there is no wire
        payload to echo back, and ``turn_detection`` stays absent so the session
        never runs server VAD -- this turn is delimited by the request, not by
        voice activity. Generation options belong here because a model-native
        session refuses per-response overrides.
        """
        config = DuplexSessionConfig(model=request.model or self._model_name, overlap_policy="listen_only")
        config.extra_body = self._session_extra_body(request)
        instructions, prompt = self._split_messages(request)
        if instructions:
            config.instructions = instructions
        if prompt:
            # The session's opening turn. A model-native session is seeded once,
            # at open, so the whole prompt goes in here rather than arriving as
            # conversation items -- which reach the history but not the model's
            # own context. Text that accompanies audio is seeded too: it is the
            # only way it reaches the model at all.
            config.initial_user_text = prompt
        if not self._has_audio(request):
            # A seeded turn has to answer by itself. An explicit response.create
            # cannot drive it, because the priming units are consumed as
            # "listen" before it arrives. Speech needs none of this: its commit
            # asks for the response.
            config.extra_body.setdefault("auto_response", True)
        modalities = getattr(request, "modalities", None)
        config.modalities = [str(m) for m in modalities] if modalities else ["text"]
        if request.temperature is not None:
            config.temperature = float(request.temperature)
        max_tokens = request.max_completion_tokens or request.max_tokens
        if max_tokens is not None:
            config.max_tokens = int(max_tokens)
        self._apply_chat_template_kwargs(request, config)
        return config

    @classmethod
    def _split_messages(cls, request: ChatCompletionRequest) -> tuple[str, str]:
        """The messages as (instructions, opening turn).

        System messages become the session instructions, which is where a
        duplex session keeps them; everything else is flattened into the turn
        the model is asked to answer, speaker-labelled when there is more than
        one message so a multi-turn history survives the flattening.
        """
        system: list[str] = []
        turns: list[tuple[str, str]] = []
        for message in request.messages:
            role = message.get("role") if isinstance(message, Mapping) else getattr(message, "role", None)
            text = cls._message_text(message)
            if not text:
                continue
            if role == "system" or role == "developer":
                system.append(text)
            else:
                turns.append((str(role or "user"), text))
        if len(turns) == 1:
            prompt = turns[0][1]
        else:
            prompt = "\n".join(f"{role}: {text}" for role, text in turns)
        return "\n".join(system), prompt

    @staticmethod
    def _message_text(message: Any) -> str:
        """The text of one chat message, audio parts excluded."""
        content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        parts = [
            str(part["text"])
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") in {"text", "input_text"}
            and isinstance(part.get("text"), str)
        ]
        return " ".join(p for p in parts if p).strip()

    @staticmethod
    def _session_extra_body(request: ChatCompletionRequest) -> dict[str, object]:
        """The session's ``extra_body``, however the caller spelled it.

        A websocket client puts these inside the session object; an HTTP caller
        reaches the same place two ways, and both are honoured: a literal
        ``extra_body`` object (what raw JSON does) and bare unknown top-level
        keys (what the OpenAI SDK's ``extra_body=`` produces, since it merges
        them into the body). This is how a request asking for audio output
        supplies the ``ref_audio`` such a model requires.

        No more permissive than the surface beside it: the Realtime open path
        passes its own session ``extra_body`` through unfiltered too.
        """
        extra = dict(request.model_extra or {})
        nested = extra.pop("extra_body", None)
        # ``modalities`` is a session field of its own, not plugin input.
        extra.pop("modalities", None)
        if isinstance(nested, Mapping):
            extra.update(nested)
        return extra

    @staticmethod
    def _apply_chat_template_kwargs(request: ChatCompletionRequest, config: DuplexSessionConfig) -> None:
        """Carry across the one chat-template knob a session also has.

        ``chat_template_kwargs`` is a turn-based stage-0 input-processor
        feature, and a duplex session renders its own prompt, so most of it has
        no equivalent here. ``use_tts_template`` is the exception: it is the
        same switch under another name, and callers of this model already pass
        it. The rest is reported rather than silently dropped, because ignoring
        something like ``enable_thinking`` changes what the answer contains.
        """
        kwargs = getattr(request, "chat_template_kwargs", None)
        if not isinstance(kwargs, Mapping):
            return
        if isinstance(kwargs.get("use_tts_template"), bool):
            config.use_tts_template = bool(kwargs["use_tts_template"])
        ignored = sorted(key for key in kwargs if key != "use_tts_template")
        if ignored:
            logger.warning(
                "chat_template_kwargs %s are ignored on a duplex model: the session renders its own prompt",
                ", ".join(ignored),
            )

    async def _start_turn(self, handle: DuplexSessionHandle, request: ChatCompletionRequest) -> None:
        """Give the session its input, in the shape the model actually answers.

        Speech is a turn on its own: append it and commit. Text is not -- a
        model-native model decides to speak from the audio it hears, so the
        prompt was seeded into the session at open and the session only needs
        units to generate on. Either way the session answers by itself; there
        is no ``response.create``, which cannot drive a seeded turn.
        """
        appended_audio = False
        for message in request.messages:
            for audio, fmt, sample_rate in self._audio_parts(message):
                await handle.append_audio(audio, format=fmt, sample_rate_hz=sample_rate)
                appended_audio = True
        if appended_audio:
            # The commit both ends the input and asks for the response.
            await handle.commit(final=True, create_response=True)
            return
        # The seeded turn generates per unit, so it needs a clock; silence
        # keeps those units from adding content of their own.
        units = max(1, int(handle.capabilities.text_turn_priming_units))
        silence = self._silence_unit()
        for _ in range(units):
            await handle.append_audio(silence, format="pcm16", sample_rate_hz=_PRIMING_SAMPLE_RATE_HZ)

    @staticmethod
    def _silence_unit() -> bytes:
        return bytes(_PRIMING_SAMPLE_RATE_HZ * 2)  # 1 s of PCM16 silence

    @staticmethod
    def _audio_parts(message: Any) -> Iterator[tuple[bytes, str, int | None]]:
        """Decoded audio of one chat message, in content order."""
        content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
        if not isinstance(content, list):
            return
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") not in _AUDIO_PART_TYPES:
                continue
            audio = part.get("input_audio") or part.get("audio")
            if not isinstance(audio, Mapping) or not isinstance(audio.get("data"), str):
                continue
            try:
                pcm = base64.b64decode(audio["data"], validate=True)
            except (ValueError, TypeError):
                continue
            if not pcm:
                continue
            fmt = audio.get("format")
            sample_rate = audio.get("sample_rate_hz") or audio.get("sample_rate")
            yield (
                pcm,
                fmt if isinstance(fmt, str) else "pcm16",
                int(sample_rate) if isinstance(sample_rate, int) else None,
            )

    # ------------------------------------------------------------------ #
    # Session -> response                                                #
    # ------------------------------------------------------------------ #

    async def _collect(
        self, handle: DuplexSessionHandle, request: ChatCompletionRequest
    ) -> ChatCompletionResponse | ErrorResponse:
        await self._start_turn(handle, request)
        text: list[str] = []
        audio: list[bytes] = []
        transcript: list[str] = []
        finish_reason = "stop"

        async with aclosing(handle.events()) as events:
            async for event in events:
                if isinstance(event, ErrorEvent):
                    return self._error_from_code(event.message, event.code)
                if isinstance(event, TextDelta):
                    text.append(event.delta)
                elif isinstance(event, TranscriptDelta):
                    transcript.append(event.delta)
                elif isinstance(event, AudioDelta):
                    if event.audio:
                        audio.append(event.audio)
                elif isinstance(event, ResponseDone):
                    finish_reason = self._finish_reason(event)
                    break
                elif isinstance(event, SessionClosed):
                    return self.create_error_response(
                        f"the duplex session ended before the response completed: {event.reason}",
                        err_type="internal_server_error",
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

        # A duplex model answers by speaking, so its words arrive as the audio
        # transcript. That transcript is the assistant's text; a chat client
        # that asked for text would otherwise get an empty message.
        content = "".join(text) or "".join(transcript)
        message = ChatMessage(role="assistant", content=content)
        if audio:
            message.audio = ChatCompletionAudio(
                id=f"audio_{uuid.uuid4().hex[:16]}",
                data=base64.b64encode(b"".join(audio)).decode("ascii"),
                expires_at=int(time.time()) + 86400,
                transcript="".join(transcript),
            )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model or self._model_name,
            choices=[ChatCompletionResponseChoice(index=0, message=message, finish_reason=finish_reason)],
            # A duplex session reports no token counts, so there is nothing
            # honest to put here.
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def _stream(self, handle: DuplexSessionHandle, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """SSE: one ``chat.completion.chunk`` per text delta, then ``[DONE]``."""
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = request.model or self._model_name

        def chunk(delta: Mapping[str, object], finish_reason: str | None = None) -> str:
            body = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(body)}\n\n"

        try:
            await self._start_turn(handle, request)
            yield chunk({"role": "assistant", "content": ""})
            async with aclosing(handle.events()) as events:
                async for event in events:
                    if isinstance(event, ErrorEvent):
                        logger.warning("duplex chat completion stream failed: %s: %s", event.code, event.message)
                        yield chunk({}, finish_reason="stop")
                        break
                    # A duplex model answers by speaking, so the transcript is
                    # the assistant's text; a model that emits text directly
                    # sends TextDelta instead. Never both for the same words.
                    if isinstance(event, TextDelta | TranscriptDelta):
                        if event.delta:
                            yield chunk({"content": event.delta})
                    elif isinstance(event, ResponseDone):
                        yield chunk({}, finish_reason=self._finish_reason(event))
                        break
                    elif isinstance(event, SessionClosed):
                        logger.warning("duplex session %s ended mid-stream: %s", handle.session_id, event.reason)
                        yield chunk({}, finish_reason="stop")
                        break
        except Exception as exc:
            # The stream has already started, so the only way to report this is
            # to end it; the session still has to be released below.
            logger.exception("duplex chat completion stream failed: %s", exc)
            yield chunk({}, finish_reason="stop")
        finally:
            await self._close(handle)
            yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------ #
    # Mapping helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _finish_reason(event: ResponseDone) -> str:
        status = event.response.get("status")
        return _FINISH_REASON_BY_STATUS.get(status if isinstance(status, str) else "", "stop")

    def _error_from_code(self, message: str, code: str | None) -> ErrorResponse:
        """One duplex error code, as the HTTP status it deserves."""
        if code == "resource_exhausted":
            # One session per request means max_sessions caps HTTP concurrency
            # too: that is backpressure, not a malformed request.
            status = HTTPStatus.SERVICE_UNAVAILABLE
        elif code in _CLIENT_ERROR_CODES:
            status = HTTPStatus.BAD_REQUEST
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        err_type = "BadRequestError" if status == HTTPStatus.BAD_REQUEST else str(code or "internal_server_error")
        return self.create_error_response(message, err_type=err_type, status_code=status)

    async def _close(self, handle: DuplexSessionHandle) -> None:
        """Never leak an admission slot, whatever ended the request."""
        with suppress(Exception):
            await self._omni.close_session(handle.session_id, reason="chat_completion_done")
