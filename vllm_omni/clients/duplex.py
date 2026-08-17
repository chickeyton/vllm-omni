# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public asynchronous client for the vLLM-Omni full-duplex Realtime API.

Connects to ``/v1/realtime?duplex=1`` (the normative duplex contract) and
provides typed events, response demultiplexing, incremental playback acking,
barge-in, and transparent session resume on transport drops.

Example::

    from vllm_omni.clients.duplex import DuplexClient, SessionConfig

    cfg = SessionConfig.for_minicpmo45(ref_audio=audio_data_url(wav_bytes))
    async with DuplexClient("ws://localhost:8099", model=model, config=cfg) as client:
        await client.stream_pcm(pcm16)
        await client.commit()
        async for response in client.responses():
            if response.decision == "listen":
                continue
            async for chunk in response.audio():
                play(chunk)
                await client.ack_playback(response.played_ms)
            break

Media capture/playback, VAD, and resampling stay in the application; this
module moves bytes and events.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import random
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

__all__ = [
    "PCM16_BYTES_PER_SAMPLE",
    "PCM16_SAMPLE_RATE",
    "AudioFormat",
    "AudioDelta",
    "ConnectionResumed",
    "DuplexClient",
    "DuplexClientError",
    "DuplexConnectionError",
    "DuplexEvent",
    "DuplexProtocolError",
    "DuplexSessionClosedError",
    "ErrorEvent",
    "EventCollector",
    "ListenDecision",
    "ReconnectPolicy",
    "ResponseCreated",
    "ResponseDone",
    "ResponseHandle",
    "SessionConfig",
    "SessionCreated",
    "SpeakDecision",
    "WebSocketTransport",
    "TextDelta",
    "TranscriptDelta",
    "acknowledge_collected_playback",
    "audio_data_url",
    "build_realtime_url",
    "image_data_url",
    "read_pcm16_wav",
    "summarize_session_request_metrics",
    "wait_for_condition",
    "write_pcm16_wav",
]

# The default duplex input wire format (16 kHz mono PCM16), kept as plain
# constants for callers doing byte/duration arithmetic outside AudioFormat.
PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2


# ---------------------------------------------------------------------------
# Errors


class DuplexClientError(Exception):
    """Base class for all duplex client errors."""


class DuplexConnectionError(DuplexClientError):
    """The WebSocket transport could not be established or failed."""


class DuplexProtocolError(DuplexClientError):
    """The server rejected a request with an ``error`` event."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        event_id: str | None = None,
        raw: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.event_id = event_id
        self.raw = dict(raw or {})


class DuplexSessionClosedError(DuplexClientError):
    """The session ended and the client cannot continue."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"duplex session closed: {reason}")
        self.reason = reason


# ---------------------------------------------------------------------------
# Configuration

_BYTES_PER_SAMPLE = {"pcm16": 2, "pcm_f32le": 4}


@dataclass(frozen=True)
class AudioFormat:
    """One PCM wire format: encoding name plus sample rate."""

    encoding: str
    sample_rate_hz: int

    @property
    def bytes_per_sample(self) -> int:
        try:
            return _BYTES_PER_SAMPLE[self.encoding]
        except KeyError:
            raise ValueError(f"Unknown audio encoding: {self.encoding!r}") from None

    def duration_ms(self, byte_count: int) -> float:
        return byte_count * 1000.0 / (self.sample_rate_hz * self.bytes_per_sample)

    def byte_count(self, duration_ms: float) -> int:
        samples = round(self.sample_rate_hz * duration_ms / 1000.0)
        return int(samples) * self.bytes_per_sample


@dataclass
class SessionConfig:
    """Model-agnostic duplex session configuration.

    Model-specific knobs ride in ``extra_body`` (or the per-model classmethod
    presets, which are data only — the client itself stays model-neutral).
    """

    modalities: tuple[str, ...] = ("audio", "text")
    input_audio: AudioFormat = AudioFormat("pcm16", 16_000)
    output_audio: AudioFormat = AudioFormat("pcm16", 24_000)
    instructions: str | None = None
    ref_audio: str | None = None
    voice: str | None = None
    auto_response: bool = True
    temperature: float | None = None
    overlap_policy: str | None = None
    playback_commit_policy: str | None = None
    turn_detection: dict[str, object] | None = None
    extra_body: dict[str, object] = field(default_factory=dict)

    def to_session_payload(self, *, model: str, session_id: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "modalities": list(self.modalities),
            "input_audio_format": self.input_audio.encoding,
            "output_audio_format": self.output_audio.encoding,
            "turn_detection": self.turn_detection,
        }
        if session_id:
            payload["session_id"] = session_id
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.ref_audio is not None:
            payload["ref_audio"] = self.ref_audio
        if self.voice is not None:
            payload["voice"] = self.voice
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.overlap_policy is not None:
            payload["overlap_policy"] = self.overlap_policy
        if self.playback_commit_policy is not None:
            payload["playback_commit_policy"] = self.playback_commit_policy
        extra_body: dict[str, object] = {"auto_response": self.auto_response}
        extra_body.update(self.extra_body)
        payload["extra_body"] = extra_body
        return payload

    @classmethod
    def for_minicpmo45(cls, *, ref_audio: str | None = None, **overrides: object) -> SessionConfig:
        """Preset matching the MiniCPM-o 4.5 native duplex deployment."""
        extra_body = {"minicpmo45_native_duplex": True, "force_listen_count": 0}
        extra_body.update(overrides.pop("extra_body", {}))
        config = cls(
            ref_audio=ref_audio,
            overlap_policy="listen_only",
            playback_commit_policy="ack_only",
            extra_body=extra_body,
        )
        return replace(config, **overrides) if overrides else config

    @classmethod
    def for_personaplex(cls, *, voice: str = "NATF2.pt", persona: str = "", **overrides: object) -> SessionConfig:
        """Preset matching the PersonaPlex duplex deployment (24 kHz float32 in)."""
        config = cls(
            input_audio=AudioFormat("pcm_f32le", 24_000),
            output_audio=AudioFormat("pcm16", 24_000),
            voice=voice,
            instructions=persona,
        )
        return replace(config, **overrides) if overrides else config


@dataclass(frozen=True)
class ReconnectPolicy:
    """Jittered exponential backoff for ``session.resume`` reconnects."""

    max_attempts: int = 5
    backoff_s: tuple[float, float] = (0.25, 4.0)

    def delay(self, attempt: int) -> float:
        low, high = self.backoff_s
        ceiling = min(high, low * (2.0 ** max(0, attempt)))
        return random.uniform(low, max(low, ceiling))


# ---------------------------------------------------------------------------
# Typed events


class DuplexEvent:
    """Thin typed wrapper over one server event dict."""

    __slots__ = ("raw",)

    def __init__(self, raw: dict[str, object]) -> None:
        self.raw = raw

    @property
    def type(self) -> str:
        value = self.raw.get("type")
        return value if isinstance(value, str) else ""

    @property
    def event_id(self) -> str | None:
        value = self.raw.get("event_id")
        return value if isinstance(value, str) else None

    @property
    def session_id(self) -> str | None:
        value = self.raw.get("session_id")
        return value if isinstance(value, str) else None

    @property
    def response_id(self) -> str | None:
        value = self.raw.get("response_id")
        if isinstance(value, str):
            return value
        response = self.raw.get("response")
        if isinstance(response, dict):
            value = response.get("id")
            if isinstance(value, str):
                return value
        return None

    @property
    def item_id(self) -> str | None:
        value = self.raw.get("item_id")
        return value if isinstance(value, str) else None

    @property
    def server_event_seq(self) -> int | None:
        value = self.raw.get("server_event_seq")
        return value if isinstance(value, int) else None

    @property
    def sample_rate_hz(self) -> int | None:
        value = self.raw.get("sample_rate_hz")
        return value if isinstance(value, int) and value > 0 else None

    @property
    def audio(self) -> bytes | None:
        delta = self.raw.get("delta") or self.raw.get("audio")
        if not isinstance(delta, str) or not delta:
            return None
        try:
            return base64.b64decode(delta)
        except ValueError:
            return None

    @property
    def text(self) -> str | None:
        for key in ("text", "delta", "transcript"):
            value = self.raw.get(key)
            if isinstance(value, str):
                return value
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(type={self.type!r})"


class SessionCreated(DuplexEvent):
    @property
    def session(self) -> dict[str, object]:
        value = self.raw.get("session")
        return value if isinstance(value, dict) else {}

    @property
    def resume_token(self) -> str | None:
        value = self.raw.get("resume_token")
        return value if isinstance(value, str) else None

    @property
    def incarnation(self) -> int:
        value = self.raw.get("incarnation")
        return value if isinstance(value, int) else 0


class SessionResumed(SessionCreated):
    pass


class SessionClosed(DuplexEvent):
    pass


class SessionExpired(DuplexEvent):
    pass


class ResyncRequired(DuplexEvent):
    pass


class HeartbeatAck(DuplexEvent):
    pass


class ResponseCreated(DuplexEvent):
    pass


class ResponseDone(DuplexEvent):
    pass


class AudioDelta(DuplexEvent):
    pass


class TranscriptDelta(DuplexEvent):
    pass


class TextDelta(DuplexEvent):
    pass


class ListenDecision(DuplexEvent):
    pass


class SpeakDecision(DuplexEvent):
    pass


class ConnectionResumed(DuplexEvent):
    """Synthetic client event: the transport dropped and was resumed."""


class ErrorEvent(DuplexEvent):
    @property
    def _error(self) -> dict[str, object] | None:
        value = self.raw.get("error")
        return value if isinstance(value, dict) else None

    @property
    def code(self) -> str | None:
        error = self._error
        value = (error or self.raw).get("code")
        return value if isinstance(value, str) else None

    @property
    def message(self) -> str:
        error = self._error
        if error is not None:
            value = error.get("message")
            if isinstance(value, str):
                return value
        value = self.raw.get("error")
        return value if isinstance(value, str) else json.dumps(self.raw)

    @property
    def related_event_id(self) -> str | None:
        error = self._error
        value = error.get("event_id") if error is not None else None
        return value if isinstance(value, str) else None

    def to_exception(self) -> DuplexProtocolError:
        return DuplexProtocolError(
            self.message,
            code=self.code,
            event_id=self.related_event_id,
            raw=self.raw,
        )


_EVENT_TYPES: dict[str, type[DuplexEvent]] = {
    "session.created": SessionCreated,
    "session.resumed": SessionResumed,
    "session.closed": SessionClosed,
    "session.expired": SessionExpired,
    "session.resync_required": ResyncRequired,
    "session.heartbeat_ack": HeartbeatAck,
    "response.created": ResponseCreated,
    "response.done": ResponseDone,
    "response.audio.delta": AudioDelta,
    "response.output_audio.delta": AudioDelta,
    "response.audio_transcript.delta": TranscriptDelta,
    "response.output_audio_transcript.delta": TranscriptDelta,
    "response.text.delta": TextDelta,
    "response.output_text.delta": TextDelta,
    "response.listen": ListenDecision,
    "response.speak": SpeakDecision,
    "connection.resumed": ConnectionResumed,
    "error": ErrorEvent,
}

_AUDIO_DELTA_TYPES = frozenset(name for name, event_cls in _EVENT_TYPES.items() if event_cls is AudioDelta)

_FATAL_RESUME_CODES = frozenset(
    {
        "invalid_session_resume",
        "session_resume_expired",
        "unsupported_session_resume",
        "invalid_resume_token",
        "session_resume_conflict",
    }
)


def wrap_event(raw: dict[str, object]) -> DuplexEvent:
    event_type = raw.get("type")
    event_cls = _EVENT_TYPES.get(event_type) if isinstance(event_type, str) else None
    return (event_cls or DuplexEvent)(raw)


@dataclass(frozen=True)
class _ClosedMarker:
    reason: str
    expected: bool


# ---------------------------------------------------------------------------
# Response demultiplexing


class ResponseHandle:
    """One ``response.created`` → terminal lifecycle, demultiplexed.

    Listen decisions surface as already-finished handles with
    ``decision == "listen"`` and no audio, so a turn loop stays uniform.
    """

    def __init__(
        self,
        *,
        response_id: str | None,
        output_format: AudioFormat,
        max_buffered_events: int,
        decision: str | None = None,
        created_event: ResponseCreated | None = None,
    ) -> None:
        self.response_id = response_id
        self.decision = decision
        self.transcript = ""
        self.text = ""
        self.played_ms = 0.0
        self.created_event = created_event
        self.done_event: DuplexEvent | None = None
        self._output_format = output_format
        self._audio_queue: asyncio.Queue[tuple[bytes, int | None] | None] = asyncio.Queue(maxsize=max_buffered_events)
        self._done = asyncio.Event()

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    async def wait(self, timeout_s: float | None = None) -> None:
        if timeout_s is None:
            await self._done.wait()
        else:
            await asyncio.wait_for(self._done.wait(), timeout_s)

    async def audio(self) -> AsyncIterator[bytes]:
        """Yield decoded output-audio chunks until the response finishes.

        ``played_ms`` advances as chunks are yielded, so it reflects what the
        application has taken for playback — feed it to
        :meth:`DuplexClient.ack_playback`.
        """
        while True:
            item = await self._audio_queue.get()
            if item is None:
                return
            chunk, sample_rate_hz = item
            rate = sample_rate_hz or self._output_format.sample_rate_hz
            self.played_ms += len(chunk) * 1000.0 / (rate * self._output_format.bytes_per_sample)
            yield chunk

    async def _feed(self, event: DuplexEvent) -> None:
        if self._done.is_set():
            return
        if isinstance(event, AudioDelta):
            chunk = event.audio
            if chunk:
                await self._audio_queue.put((chunk, event.sample_rate_hz))
        elif isinstance(event, TranscriptDelta):
            self.transcript += event.text or ""
        elif isinstance(event, TextDelta):
            self.text += event.text or ""
        elif isinstance(event, SpeakDecision):
            self.decision = "speak"
        elif isinstance(event, ListenDecision):
            self.decision = "listen"

    def _finish(self, event: DuplexEvent | None) -> None:
        if self._done.is_set():
            return
        self.done_event = event
        self._done.set()
        # A full queue cannot block _finish: the sentinel is delivered by a
        # non-blocking put with a drop-oldest fallback.
        try:
            self._audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - racing consumer
                pass
            self._audio_queue.put_nowait(None)


# ---------------------------------------------------------------------------
# Client


class WebSocketTransport(Protocol):
    """Minimal WebSocket surface the client needs (satisfied by ``websockets``)."""

    async def send(self, data: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


ConnectFn = Callable[[str], Awaitable["WebSocketTransport"]]


class DuplexClient:
    """Async client for one duplex session over ``/v1/realtime?duplex=1``."""

    def __init__(
        self,
        url: str,
        *,
        model: str,
        config: SessionConfig | None = None,
        session_id: str | None = None,
        reconnect: ReconnectPolicy | None = ReconnectPolicy(),
        max_buffered_events: int = 1024,
        max_frame_bytes: int = 64 << 20,
        heartbeat_interval_s: float | None = 30.0,
        handshake_timeout_s: float = 30.0,
        auto_ack: bool = True,
        connect: ConnectFn | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.config = config or SessionConfig()
        self.session_id = session_id
        self.session_info: dict[str, object] = {}
        self.incarnation = 0
        self.resume_token: str | None = None
        self._reconnect = reconnect
        self._max_buffered_events = max_buffered_events
        self._max_frame_bytes = max_frame_bytes
        self._heartbeat_interval_s = heartbeat_interval_s
        self._handshake_timeout_s = handshake_timeout_s
        self._auto_ack = auto_ack
        self._connect_fn: ConnectFn = connect or self._default_connect
        self._ws: WebSocketTransport | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._subscribers: list[asyncio.Queue[DuplexEvent | _ClosedMarker]] = []
        self._response_queue: asyncio.Queue[ResponseHandle | _ClosedMarker] = asyncio.Queue(maxsize=max_buffered_events)
        self._responses: dict[str, ResponseHandle] = {}
        self._last_server_event_seq: int | None = None
        self._input_audio_end_ms = 0.0
        self._closing = False
        self._closed = asyncio.Event()
        self._closed_marker: _ClosedMarker | None = None

    # -- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> DuplexClient:
        self._ws = await self._connect_fn(self._target_url())
        handshake_queue = self._add_subscriber()
        try:
            self._reader_task = asyncio.create_task(self._read_loop(), name="duplex-client-reader")
            await self.send(
                {
                    "type": "session.update",
                    "session": self.config.to_session_payload(model=self.model, session_id=self.session_id),
                }
            )
            created = await self._wait_on_queue(
                handshake_queue,
                ("session.created",),
                timeout_s=self._handshake_timeout_s,
            )
        except BaseException:
            await self._teardown()
            raise
        finally:
            self._remove_subscriber(handshake_queue)
        assert isinstance(created, SessionCreated)
        self._adopt_session(created)
        if self._heartbeat_interval_s is not None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="duplex-client-heartbeat")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None and not self._closed.is_set():
                await self.close()
        finally:
            await self._teardown()

    async def close(self, *, timeout_s: float = 20.0) -> None:
        """Send ``session.close`` and wait for the server to confirm."""
        if self._closed.is_set():
            return
        self._closing = True
        try:
            await self.send({"type": "session.close"})
        except (DuplexConnectionError, DuplexSessionClosedError):
            self._finalize("closed", expected=True)
            return
        try:
            await asyncio.wait_for(self._closed.wait(), timeout_s)
        except asyncio.TimeoutError:
            self._finalize("close timed out", expected=True)

    async def update_session(self, config: SessionConfig) -> None:
        self.config = config
        await self.send(
            {
                "type": "session.update",
                "session": config.to_session_payload(model=self.model, session_id=self.session_id),
            }
        )

    # -- input ----------------------------------------------------------------

    async def append_audio(
        self,
        pcm: bytes,
        *,
        final: bool = False,
        is_speech: bool | None = None,
        video_frames: Sequence[str] | None = None,
    ) -> None:
        """Append one chunk of input audio in the session's input format.

        ``video_frames`` takes data-URL JPEG strings (one per ~1 s unit for
        omni models); see :func:`image_data_url`.
        """
        input_format = self.config.input_audio
        duration_ms = input_format.duration_ms(len(pcm))
        self._input_audio_end_ms += duration_ms
        event: dict[str, object] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
            "format": input_format.encoding,
            "sample_rate_hz": input_format.sample_rate_hz,
            "duration_ms": round(duration_ms),
            "audio_end_ms": round(self._input_audio_end_ms),
        }
        if final:
            event["final"] = True
        if is_speech is not None:
            event["is_speech"] = is_speech
        if video_frames:
            event["video_frames"] = list(video_frames)
        await self.send(event)

    async def stream_pcm(
        self,
        pcm: bytes,
        *,
        chunk_ms: int = 200,
        realtime: bool = True,
        is_speech: bool | None = None,
    ) -> None:
        """Convenience: slice ``pcm`` into ``chunk_ms`` chunks, pace, append."""
        input_format = self.config.input_audio
        chunk_bytes = max(input_format.byte_count(chunk_ms), input_format.bytes_per_sample)
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset : offset + chunk_bytes]
            await self.append_audio(chunk, is_speech=is_speech)
            if realtime:
                await asyncio.sleep(input_format.duration_ms(len(chunk)) / 1000.0)

    async def append_text(self, text: str) -> None:
        await self.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def commit(self, *, final: bool = True, create_response: bool | None = None) -> None:
        event: dict[str, object] = {"type": "input_audio_buffer.commit", "final": final}
        if create_response is not None:
            event["response_create"] = create_response
        await self.send(event)

    async def create_response(self, **overrides: object) -> None:
        await self.send({"type": "response.create", "response": overrides})

    # -- interruption ---------------------------------------------------------

    async def cancel_response(self, response_id: str | None = None) -> None:
        event: dict[str, object] = {"type": "response.cancel"}
        if response_id is not None:
            event["response_id"] = response_id
        await self.send(event)

    async def clear_input(self) -> None:
        await self.send({"type": "input_audio_buffer.clear"})

    async def clear_output(self) -> None:
        await self.send({"type": "output_audio_buffer.clear"})

    async def barge_in(self) -> None:
        """Interrupt the assistant: cancel the active response, drop pending input.

        Composed from normative Realtime events (``response.cancel`` +
        ``input_audio_buffer.clear``). Report playback truncation separately
        via :meth:`ack_playback` with the milliseconds actually played.
        """
        await self.cancel_response()
        await self.clear_input()

    # -- playback contract ------------------------------------------------------

    async def ack_playback(
        self,
        played_ms: float,
        *,
        response_id: str | None = None,
        item_id: str | None = None,
        committed_ms: float | None = None,
    ) -> None:
        """Report cumulative playback progress; call periodically while playing."""
        event: dict[str, object] = {
            "type": "playback.ack",
            "played_ms": int(played_ms),
            "committed_ms": int(committed_ms if committed_ms is not None else played_ms),
        }
        if response_id is not None:
            event["response_id"] = response_id
        if item_id is not None:
            event["item_id"] = item_id
        await self.send(event)

    # -- output ----------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[DuplexEvent]:
        return self.events()

    async def events(self) -> AsyncIterator[DuplexEvent]:
        """Iterate over every server event from now on (typed)."""
        queue = self._add_subscriber()
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _ClosedMarker):
                    if item.expected:
                        return
                    raise DuplexSessionClosedError(item.reason)
                yield item
        finally:
            self._remove_subscriber(queue)

    async def responses(self) -> AsyncIterator[ResponseHandle]:
        """Iterate over response lifecycles (single consumer)."""
        while True:
            item = await self._response_queue.get()
            if isinstance(item, _ClosedMarker):
                if item.expected:
                    return
                raise DuplexSessionClosedError(item.reason)
            yield item

    async def wait_for(self, *types: str, timeout_s: float) -> DuplexEvent:
        """Wait for the next event whose type is in ``types``.

        Raises :class:`DuplexProtocolError` if an ``error`` event arrives
        first, and :class:`DuplexSessionClosedError` if the session ends first.
        """
        queue = self._add_subscriber()
        try:
            return await self._wait_on_queue(queue, types, timeout_s=timeout_s)
        finally:
            self._remove_subscriber(queue)

    async def send(self, event: dict[str, object]) -> str:
        """Send one raw client event; returns the (possibly stamped) event_id."""
        if self._closed.is_set():
            marker = self._closed_marker
            raise DuplexSessionClosedError(marker.reason if marker else "closed")
        payload = dict(event)
        payload.setdefault("event_id", f"evt_{uuid4().hex}")
        try:
            await self._ws.send(json.dumps(payload))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise DuplexConnectionError(f"send failed: {exc}") from exc
        return payload["event_id"]

    # -- internals ---------------------------------------------------------------

    def _target_url(self) -> str:
        parts = urlsplit(self.url)
        path = parts.path if parts.path not in ("", "/") else "/v1/realtime"
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("duplex", "1")
        query.setdefault("model", self.model)
        # The client always opens with an explicit session.update (and resumes
        # with an explicit session.resume). Autostart would race that handshake:
        # with a model query param, the server creates a bare default session
        # before reading the first client event, silently dropping ref_audio
        # and the model-specific extra_body.
        query.setdefault("autostart", "0")
        return urlunsplit((parts.scheme or "ws", parts.netloc, path, urlencode(query), parts.fragment))

    async def _default_connect(self, url: str) -> WebSocketTransport:
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - websockets is a pinned dep
            raise DuplexConnectionError("The duplex client requires the 'websockets' package") from exc
        try:
            return await websockets.connect(url, max_size=self._max_frame_bytes)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise DuplexConnectionError(f"connect to {url} failed: {exc}") from exc

    def _adopt_session(self, event: SessionCreated) -> None:
        self.session_info = event.session
        session_id = event.session.get("session_id") or event.session.get("id") or event.session_id
        if isinstance(session_id, str) and session_id:
            self.session_id = session_id
        self.incarnation = event.incarnation
        if event.resume_token:
            self.resume_token = event.resume_token

    def _add_subscriber(self) -> asyncio.Queue[DuplexEvent | _ClosedMarker]:
        queue: asyncio.Queue[DuplexEvent | _ClosedMarker] = asyncio.Queue(maxsize=self._max_buffered_events)
        if self._closed.is_set() and self._closed_marker is not None:
            queue.put_nowait(self._closed_marker)
        else:
            self._subscribers.append(queue)
        return queue

    def _remove_subscriber(self, queue: asyncio.Queue[DuplexEvent | _ClosedMarker]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    async def _wait_on_queue(
        self,
        queue: asyncio.Queue[DuplexEvent | _ClosedMarker],
        types: tuple[str, ...],
        *,
        timeout_s: float,
    ) -> DuplexEvent:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"timed out waiting for {types}")
            item = await asyncio.wait_for(queue.get(), remaining)
            if isinstance(item, _ClosedMarker):
                raise DuplexSessionClosedError(item.reason)
            if isinstance(item, ErrorEvent) and item.type not in types:
                raise item.to_exception()
            if item.type in types:
                return item

    async def _read_loop(self) -> None:
        while True:
            try:
                while True:
                    raw = await self._ws.recv()
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        await self._dispatch(data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closing or self._closed.is_set():
                    self._finalize("closed", expected=True)
                    return
                if await self._attempt_resume():
                    continue
                self._finalize(f"connection lost: {exc}", expected=False)
                return

    async def _dispatch(self, data: dict[str, object]) -> None:
        seq = data.get("server_event_seq")
        if isinstance(seq, int):
            if self._last_server_event_seq is not None and seq <= self._last_server_event_seq:
                return  # duplicate delivered by resume replay
            self._last_server_event_seq = seq
        event = wrap_event(data)

        if isinstance(event, SessionResumed):
            self._adopt_session(event)
        elif isinstance(event, SessionCreated):
            self._adopt_session(event)
        elif isinstance(event, ResponseCreated):
            response_id = event.response_id
            if response_id and response_id not in self._responses:
                handle = ResponseHandle(
                    response_id=response_id,
                    output_format=self.config.output_audio,
                    max_buffered_events=self._max_buffered_events,
                    created_event=event,
                )
                self._responses[response_id] = handle
                await self._response_queue.put(handle)
        elif isinstance(event, (AudioDelta, TranscriptDelta, TextDelta, SpeakDecision)):
            handle = self._responses.get(event.response_id or "")
            if handle is not None:
                await handle._feed(event)
        elif isinstance(event, ListenDecision):
            handle = self._responses.pop(event.response_id or "", None)
            if handle is not None:
                # A listen decision against an active response is terminal.
                await handle._feed(event)
                handle._finish(event)
            else:
                listen_handle = ResponseHandle(
                    response_id=event.response_id,
                    output_format=self.config.output_audio,
                    max_buffered_events=self._max_buffered_events,
                    decision="listen",
                )
                listen_handle._finish(event)
                await self._response_queue.put(listen_handle)
        elif isinstance(event, ResponseDone):
            handle = self._responses.pop(event.response_id or "", None)
            if handle is not None:
                if handle.decision is None:
                    handle.decision = "speak"
                handle._finish(event)

        for queue in list(self._subscribers):
            await queue.put(event)

        if isinstance(event, SessionClosed):
            self._finalize("closed", expected=True)
        elif isinstance(event, SessionExpired):
            reason = data.get("reason")
            self._finalize(f"expired: {reason}" if reason else "expired", expected=False)

        if isinstance(seq, int) and self._auto_ack and not self._closed.is_set():
            try:
                await self._ws.send(json.dumps({"type": "session.event_ack", "server_event_seq": seq}))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # the reader will observe the transport failure itself

    async def _attempt_resume(self) -> bool:
        policy = self._reconnect
        if policy is None or not self.resume_token or not self.session_id:
            return False
        for attempt in range(policy.max_attempts):
            await asyncio.sleep(policy.delay(attempt))
            try:
                ws = await self._connect_fn(self._target_url())
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            try:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.resume",
                            "session_id": self.session_id,
                            "incarnation": self.incarnation,
                            "resume_token": self.resume_token,
                            "last_received_server_event_seq": self._last_server_event_seq or 0,
                        }
                    )
                )
                deadline = time.monotonic() + self._handshake_timeout_s
                pending: list[dict[str, object]] = []
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    raw = await asyncio.wait_for(ws.recv(), remaining)
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        continue
                    event_type = data.get("type")
                    if event_type == "session.resumed":
                        self._ws = ws
                        await self._dispatch({"type": "connection.resumed", "attempt": attempt})
                        await self._dispatch(data)
                        for entry in pending:
                            await self._dispatch(entry)
                        return True
                    if event_type == "session.resync_required":
                        await self._close_quietly(ws)
                        return False
                    if event_type == "error":
                        code = ErrorEvent(data).code
                        if code in _FATAL_RESUME_CODES:
                            await self._close_quietly(ws)
                            return False
                        await self._close_quietly(ws)
                        break  # transient error: next attempt
                    pending.append(data)
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._close_quietly(ws)
                continue
        return False

    @staticmethod
    async def _close_quietly(ws: WebSocketTransport) -> None:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass

    async def _heartbeat_loop(self) -> None:
        interval = self._heartbeat_interval_s
        assert interval is not None
        while not self._closed.is_set():
            await asyncio.sleep(interval)
            try:
                await self.send({"type": "session.heartbeat"})
            except DuplexClientError:
                return

    def _finalize(self, reason: str, *, expected: bool) -> None:
        if self._closed.is_set():
            return
        marker = _ClosedMarker(reason=reason, expected=expected)
        self._closed_marker = marker
        self._closed.set()
        for handle in self._responses.values():
            handle._finish(None)
        self._responses.clear()
        for queue in (*self._subscribers, self._response_queue):
            try:
                queue.put_nowait(marker)
            except asyncio.QueueFull:  # pragma: no cover - subscriber gave up
                pass

    async def _teardown(self) -> None:
        self._closing = True
        if not self._closed.is_set():
            self._finalize("closed", expected=True)
        for task in (self._heartbeat_task, self._reader_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Test/benchmark collector


def _rounded_ms(value: float) -> float:
    return round(float(value), 3)


def _interval_summary(values: list[float]) -> dict[str, float | int]:
    clean = sorted(_rounded_ms(value) for value in values if math.isfinite(value) and value >= 0)
    if not clean:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def nearest_rank(percentile: float) -> float:
        index = max(0, math.ceil(percentile * len(clean)) - 1)
        return clean[min(index, len(clean) - 1)]

    return {
        "count": len(clean),
        "mean": _rounded_ms(sum(clean) / len(clean)),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": clean[-1],
    }


def _event_stage_metrics(event: dict[str, object]) -> dict[str, object] | None:
    candidates: list[object] = [event.get("vllm_omni")]
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend((metadata, metadata.get("vllm_omni")))
    response = event.get("response")
    if isinstance(response, dict):
        response_metadata = response.get("metadata")
        if isinstance(response_metadata, dict):
            candidates.extend((response_metadata, response_metadata.get("vllm_omni")))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        stage_metrics = candidate.get("stage_metrics")
        if isinstance(stage_metrics, dict):
            return stage_metrics
    return None


class EventCollector:
    """Accumulate events for assertions and latency summaries (tests/benchmarks).

    Unbounded by design — attach one to a short-lived probe session, not to a
    production client that runs for hours.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.event_received_at_s: list[float] = []
        self.response_audio: dict[str, list[bytes]] = {}
        self.response_ids: list[str] = []
        self.output_sample_rate_hz = 24_000

    async def consume(self, client: DuplexClient) -> None:
        """Subscribe to ``client`` and collect until the session ends."""
        try:
            async for event in client.events():
                self.add(event.raw)
        except DuplexSessionClosedError:
            pass

    @staticmethod
    def response_id(event: dict[str, object]) -> str | None:
        """Response identity of one raw event (``response_id`` or ``response.id``)."""
        return wrap_event(event).response_id

    def add(self, event: dict[str, object] | DuplexEvent, *, received_at_s: float | None = None) -> None:
        raw = event.raw if isinstance(event, DuplexEvent) else event
        received_at = time.monotonic() if received_at_s is None else float(received_at_s)
        stored_event = dict(raw)
        stored_event.setdefault("_client_received_at_s", received_at)
        self.events.append(stored_event)
        self.event_received_at_s.append(received_at)
        wrapped = wrap_event(stored_event)
        response_id = wrapped.response_id
        if stored_event.get("type") == "response.created" and response_id and response_id not in self.response_ids:
            self.response_ids.append(response_id)
        if stored_event.get("type") in _AUDIO_DELTA_TYPES:
            chunk = wrapped.audio
            if chunk and response_id:
                self.response_audio.setdefault(response_id, []).append(chunk)
            sample_rate_hz = wrapped.sample_rate_hz
            if sample_rate_hz:
                self.output_sample_rate_hz = sample_rate_hz

    def count(self, event_type: str) -> int:
        return sum(event.get("type") == event_type for event in self.events)

    def audio_bytes(self, response_id: str | None = None) -> bytes:
        if response_id is not None:
            return b"".join(self.response_audio.get(response_id, ()))
        return b"".join(
            chunk for response_id in self.response_ids for chunk in self.response_audio.get(response_id, ())
        )

    def errors(self) -> list[dict[str, object]]:
        return [event for event in self.events if event.get("type") == "error"]

    def first_received_at(self, *event_types: str, after_s: float = 0.0) -> float | None:
        for event, received_at_s in zip(self.events, self.event_received_at_s, strict=True):
            if received_at_s >= after_s and event.get("type") in event_types:
                return received_at_s
        return None

    def last_received_at(self, event_type: str) -> float | None:
        for event, received_at_s in zip(
            reversed(self.events),
            reversed(self.event_received_at_s),
            strict=True,
        ):
            if event.get("type") == event_type:
                return received_at_s
        return None

    def timing_summary(
        self,
        *,
        after_s: float,
        input_committed_at_s: float | None = None,
        response_id: str | None = None,
        measurement_origin: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """Summarize engine token metrics and client-observed audio cadence."""
        stage0_metrics: dict[str, object] | None = None
        response_created_at_s: float | None = None
        first_text_received_at_s: float | None = None
        audio_received_at_s: list[float] = []
        cumulative_audio_ms: list[float] = []
        for event, received_at_s in zip(self.events, self.event_received_at_s, strict=True):
            if received_at_s < after_s:
                continue
            event_response_id = wrap_event(event).response_id
            if response_id is not None and event_response_id != response_id:
                continue
            if event.get("type") == "response.created" and response_created_at_s is None:
                response_created_at_s = received_at_s
            if (
                event.get("type")
                in {
                    "response.audio_transcript.delta",
                    "response.output_text.delta",
                    "response.text.delta",
                }
                and isinstance(event.get("delta"), str)
                and bool(event["delta"])
                and first_text_received_at_s is None
            ):
                first_text_received_at_s = received_at_s

            stage_metrics = _event_stage_metrics(event)
            stage0 = stage_metrics.get("0") if isinstance(stage_metrics, dict) else None
            if isinstance(stage0, dict):
                stage0_metrics = stage0

            if event.get("type") not in _AUDIO_DELTA_TYPES:
                continue
            delta = event.get("delta") or event.get("audio")
            if not isinstance(delta, str) or not delta:
                continue
            audio_received_at_s.append(received_at_s)
            metadata = event.get("metadata")
            duration_ms = metadata.get("audio_duration_ms") if isinstance(metadata, dict) else None
            if isinstance(duration_ms, int | float) and math.isfinite(float(duration_ms)):
                cumulative_audio_ms.append(max(0.0, float(duration_ms)))

        result: dict[str, object] = {}
        if stage0_metrics is not None:
            raw_itls = stage0_metrics.get("vllm_itls_ms")
            itls = (
                [float(value) for value in raw_itls if isinstance(value, int | float)]
                if isinstance(raw_itls, list)
                else []
            )
            result["stage0_tokens"] = {
                "source": "engine_stage_metrics",
                "output_token_count": int(stage0_metrics.get("num_tokens_out") or 0),
                "ttft_ms": float(stage0_metrics.get("vllm_ttft_ms") or 0.0),
                "tpot_ms": float(stage0_metrics.get("vllm_tpot_ms") or 0.0),
                "inter_token_interval_ms": _interval_summary(itls),
            }

        if audio_received_at_s:
            intervals_ms = [
                (current - previous) * 1000.0 for previous, current in zip(audio_received_at_s, audio_received_at_s[1:])
            ]
            chunk_durations_ms: list[float] = []
            previous_duration_ms = 0.0
            for duration_ms in cumulative_audio_ms:
                chunk_durations_ms.append(
                    duration_ms - previous_duration_ms if duration_ms >= previous_duration_ms else duration_ms
                )
                previous_duration_ms = duration_ms
            interval_summary = _interval_summary(intervals_ms)
            result["audio_output"] = {
                "source": "client_monotonic_receive",
                "chunk_count": len(audio_received_at_s),
                "response_created_to_first_audio_ms": (
                    _rounded_ms((audio_received_at_s[0] - response_created_at_s) * 1000.0)
                    if response_created_at_s is not None
                    else None
                ),
                "commit_to_first_audio_ms": (
                    _rounded_ms((audio_received_at_s[0] - input_committed_at_s) * 1000.0)
                    if input_committed_at_s is not None
                    else None
                ),
                "inter_chunk_interval_ms": interval_summary,
                "chunk_duration_ms": _interval_summary(chunk_durations_ms),
                "max_chunk_gap_ms": interval_summary["max"],
            }
            request_started_at_s = input_committed_at_s if input_committed_at_s is not None else response_created_at_s
            if request_started_at_s is not None:
                # Lazily import so this module stays importable without the
                # vllm_omni metrics stack (the client is otherwise standalone).
                from vllm_omni.metrics.definitions import compute_audio_rtf

                audio_duration_ms = (
                    max(cumulative_audio_ms)
                    if cumulative_audio_ms
                    else self._format_for_bytes().duration_ms(len(self.audio_bytes(response_id)))
                )
                audio_generation_ms = max(
                    0.0,
                    (audio_received_at_s[-1] - request_started_at_s) * 1000.0,
                )
                result["request_metrics"] = {
                    "source": "client_monotonic_receive",
                    "measurement_origin": measurement_origin
                    or {
                        "ttft": "input_audio_buffer.commit client send to first non-empty text delta",
                        "ttfp": "input_audio_buffer.commit client send to first audio packet",
                        "rtf": "commit-to-last-audio receive time divided by emitted audio duration",
                    },
                    "ttft_ms": (
                        _rounded_ms((first_text_received_at_s - request_started_at_s) * 1000.0)
                        if first_text_received_at_s is not None
                        else None
                    ),
                    "ttfp_ms": _rounded_ms((audio_received_at_s[0] - request_started_at_s) * 1000.0),
                    "rtf": round(
                        compute_audio_rtf(
                            audio_generation_ms / 1000.0,
                            audio_duration_ms / 1000.0,
                        ),
                        6,
                    )
                    if audio_duration_ms > 0
                    else None,
                    "audio_generation_ms": _rounded_ms(audio_generation_ms),
                    "audio_duration_ms": _rounded_ms(audio_duration_ms),
                }
        return result

    def _format_for_bytes(self) -> AudioFormat:
        return AudioFormat("pcm16", self.output_sample_rate_hz)


# ---------------------------------------------------------------------------
# Media helpers


def audio_data_url(data: bytes | Path, *, mime: str = "audio/wav") -> str:
    """Encode audio bytes (or a file path) as a data URL for ``ref_audio``."""
    payload = data.read_bytes() if isinstance(data, Path) else data
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def image_data_url(data: bytes | Path, *, mime: str = "image/jpeg") -> str:
    """Encode an image (or a file path) as a data URL for ``video_frames``."""
    payload = data.read_bytes() if isinstance(data, Path) else data
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def read_pcm16_wav(path: Path, *, sample_rate_hz: int = 16_000) -> bytes:
    """Read a mono, uncompressed PCM16 WAV file at the expected rate."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("input WAV must be mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError("input WAV must be 16-bit PCM")
        if wav_file.getframerate() != sample_rate_hz:
            raise ValueError(f"input WAV must be {sample_rate_hz} Hz")
        if wav_file.getcomptype() != "NONE":
            raise ValueError("input WAV must be uncompressed PCM")
        return wav_file.readframes(wav_file.getnframes())


def write_pcm16_wav(path: Path, pcm16: bytes, *, sample_rate_hz: int) -> None:
    """Write mono PCM16 bytes as a WAV file."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm16)


# ---------------------------------------------------------------------------
# Probe/benchmark helpers


def build_realtime_url(
    url: str,
    model: str,
    *,
    autostart: bool | None = None,
    session_id: str | None = None,
    extra_query: dict[str, str] | None = None,
) -> str:
    """Add explicit duplex query parameters to a Realtime URL.

    :class:`DuplexClient` builds its own URL; this helper is for drivers that
    speak the wire protocol directly. Model-specific query flags (e.g.
    MiniCPM-o's ``minicpmo45_native_duplex``) ride in ``extra_query``.
    """
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("duplex", "1")
    query.setdefault("model", model)
    for key, value in (extra_query or {}).items():
        query.setdefault(key, value)
    if autostart is not None:
        query.setdefault("autostart", "1" if autostart else "0")
    if session_id:
        query.setdefault("session_id", session_id)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


async def wait_for_condition(
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    label: str,
) -> None:
    """Poll a collector predicate without coupling to a scenario runner."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {label}")


def summarize_session_request_metrics(
    request_metrics: list[dict[str, object]],
    *,
    session_id: str | None,
) -> dict[str, object]:
    """Average client-observed metrics across turns that emitted audio."""

    def mean(metric: str, *, digits: int = 3) -> float | None:
        values = [
            float(request[metric])
            for request in request_metrics
            if isinstance(request.get(metric), int | float) and math.isfinite(float(request[metric]))
        ]
        return round(sum(values) / len(values), digits) if values else None

    return {
        "session_id": session_id,
        "audio_turn_count": len(request_metrics),
        "mean_ttft_ms": mean("ttft_ms"),
        "mean_ttfp_ms": mean("ttfp_ms"),
        "mean_rtf": mean("rtf", digits=6),
    }


async def acknowledge_collected_playback(client: DuplexClient, collector: EventCollector) -> None:
    """Ack full playback of every collected response's audio (probe shorthand)."""
    output_format = AudioFormat("pcm16", collector.output_sample_rate_hz)
    for response_id in collector.response_ids:
        pcm16 = collector.audio_bytes(response_id)
        if not pcm16:
            continue
        played_ms = output_format.duration_ms(len(pcm16))
        await client.ack_playback(
            played_ms,
            response_id=response_id,
            item_id=f"item_{response_id}",
        )
