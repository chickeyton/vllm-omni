# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``/v1/chat/completions`` on a duplex model, as an ordinary Realtime client.

A duplex server has no turn-based request path, so one chat request becomes one
short-lived duplex session: the messages go in through the same wire verbs any
Realtime client uses -- ``conversation.item.create`` for text and images,
``input_audio_buffer.append`` + ``commit`` for audio -- and the answer is read
off ``events()`` like any other response.

Nothing model-specific lives here, and nothing below this layer learns that
chat completions exist. A model that cannot answer the shape of turn a request
describes says so on the wire, and that becomes the HTTP error.

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
        modalities = getattr(request, "modalities", None)
        config.modalities = [str(m) for m in modalities] if modalities else ["text"]
        if request.temperature is not None:
            config.temperature = float(request.temperature)
        max_tokens = request.max_completion_tokens or request.max_tokens
        if max_tokens is not None:
            config.max_tokens = int(max_tokens)
        self._apply_chat_template_kwargs(request, config)
        return config

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
        """Feed the prompt in as ordinary Realtime input and ask for the answer."""
        appended_audio = False
        for message in request.messages:
            for audio, fmt, sample_rate in self._audio_parts(message):
                await handle.append_audio(audio, format=fmt, sample_rate_hz=sample_rate)
                appended_audio = True
            item = self._realtime_item(message)
            if item is not None:
                await handle.create_item(item)
        if appended_audio:
            # The commit both ends the input and asks for the response.
            await handle.commit(final=True, create_response=True)
        else:
            # Text or images only: the runner starts the turn from the items.
            await handle.create_response()

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

    @staticmethod
    def _realtime_item(message: Any) -> dict[str, object] | None:
        """The text of a chat message, as a Realtime conversation item.

        Audio has already gone to the input buffer, and any other modality was
        refused up front, so only text reaches this point.
        """
        if isinstance(message, Mapping):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if not isinstance(role, str):
            return None

        # Realtime spells assistant text "text" and everything the user or the
        # system says "input_text".
        text_type = "text" if role == "assistant" else "input_text"
        parts: list[dict[str, object]] = []
        if isinstance(content, str):
            if content:
                parts.append({"type": text_type, "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") in _AUDIO_PART_TYPES:
                    continue  # audio already went to the input buffer
                kind = part.get("type")
                if kind in {"text", "input_text"} and isinstance(part.get("text"), str) and part["text"]:
                    parts.append({"type": text_type, "text": part["text"]})
        if not parts:
            return None
        return {
            "id": f"item_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": role,
            "content": parts,
        }

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

        message = ChatMessage(role="assistant", content="".join(text))
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
                    if isinstance(event, TextDelta):
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
