# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``/v1/chat/completions`` served on a duplex session.

What these pin down: the adapter speaks only ordinary Realtime verbs, it never
asks the framework for anything chat-shaped, and it releases the admission slot
on every exit.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.events import (
    AudioDelta,
    DuplexEvent,
    ErrorEvent,
    ResponseDone,
    SessionClosed,
    SessionCreated,
    TextDelta,
    TranscriptDelta,
)
from vllm_omni.engine.duplex.messages import DuplexSessionError
from vllm_omni.entrypoints.duplex.chat_completions import DuplexChatCompletionsAdapter
from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SESSION_ID = "duplex-test"
_PCM = b"\x01\x02" * 8


class FakeHandle:
    """Records the wire verbs the adapter submits and replays a scripted response."""

    def __init__(
        self,
        session_id: str,
        script: list[DuplexEvent],
        capabilities: DuplexCapabilities | None = None,
    ) -> None:
        self.session_id = session_id
        self.capabilities = capabilities or DuplexCapabilities(
            supports_chat_completions=True, text_turn_priming_units=3
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self._script = list(script)

    async def append_audio(self, audio: bytes, *, format: str = "pcm16", sample_rate_hz: int | None = None) -> None:
        self.calls.append(("append_audio", {"audio": audio, "format": format, "sample_rate_hz": sample_rate_hz}))

    async def create_item(self, item: Any, *, previous_item_id: str | None = None) -> None:
        self.calls.append(("create_item", {"item": dict(item)}))

    async def commit(self, *, final: bool = True, create_response: bool | None = None, **_: Any) -> None:
        self.calls.append(("commit", {"final": final, "create_response": create_response}))

    async def create_response(self, options: Any = None) -> None:
        self.calls.append(("create_response", {"options": options}))

    async def events(self):
        yield SessionCreated(session_id=self.session_id, session={"id": self.session_id})
        for event in self._script:
            yield event

    def verbs(self) -> list[str]:
        return [name for name, _ in self.calls]


class FakeOmni:
    def __init__(
        self,
        script: list[DuplexEvent] | None = None,
        capabilities: DuplexCapabilities | None = None,
    ) -> None:
        self.model = "fake-duplex-model"
        self.opened: list[DuplexSessionConfig] = []
        self.closed: list[str] = []
        self.handle: FakeHandle | None = None
        self.open_error: Exception | None = None
        self.capabilities = capabilities
        self._script = script or [ResponseDone(response_id="r1", response={"status": "completed"})]

    async def open_session(self, config: Any) -> FakeHandle:
        if self.open_error is not None:
            raise self.open_error
        self.opened.append(config)
        self.handle = FakeHandle(_SESSION_ID, self._script, self.capabilities)
        return self.handle

    async def close_session(self, session_id: str, *, reason: str = "client_close", **_: Any) -> None:
        self.closed.append(session_id)


def _adapter(omni: FakeOmni) -> DuplexChatCompletionsAdapter:
    return DuplexChatCompletionsAdapter(duplex_omni=omni, model_name="fake-duplex-model")


def _request(**kwargs: Any) -> ChatCompletionRequest:
    kwargs.setdefault("model", "fake-duplex-model")
    kwargs.setdefault("messages", [{"role": "user", "content": "hello"}])
    return ChatCompletionRequest(**kwargs)


def _audio_message(data: bytes = _PCM, fmt: str = "pcm16") -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": fmt}},
        ],
    }


# --------------------------------------------------------------------------- #
# The prompt becomes ordinary Realtime input                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_text_prompt_seeds_the_session_and_is_primed_with_silence() -> None:
    """Text is not a turn for a model-native model, so it rides the session context.

    The model still generates per audio unit, so the seeded turn gets
    ``text_turn_priming_units`` units of silence to speak on -- silence so they
    add no content of their own. No ``response.create``: it cannot drive a
    seeded turn, because the units are consumed as "listen" before it lands.
    """
    omni = FakeOmni([TextDelta(delta="hi "), TextDelta(delta="there"), ResponseDone(response={"status": "completed"})])

    response = await _adapter(omni).create_chat_completion(
        _request(messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}])
    )

    config = omni.opened[0]
    assert config.instructions == "be brief"
    assert config.initial_user_text == "hello"
    assert config.extra_body["auto_response"] is True
    handle = omni.handle
    assert handle is not None
    assert handle.verbs() == ["append_audio"] * 3
    assert all(set(call["audio"]) == {0} for _, call in handle.calls), "priming units must be silent"
    assert isinstance(response, ChatCompletionResponse)
    assert response.choices[0].message.content == "hi there"
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_a_multi_turn_history_is_flattened_into_the_seeded_turn() -> None:
    """A seeded session takes its text once, so the history has to travel with it."""
    omni = FakeOmni()

    await _adapter(omni).create_chat_completion(
        _request(
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "and now?"},
            ]
        )
    )

    assert omni.opened[0].initial_user_text == "user: hello\nassistant: hi\nuser: and now?"


@pytest.mark.asyncio
async def test_a_model_that_cannot_be_seeded_is_refused_before_it_can_hang() -> None:
    """Without the capability the turn would never complete; say so instead."""
    omni = FakeOmni(capabilities=DuplexCapabilities(supports_chat_completions=False))

    response = await _adapter(omni).create_chat_completion(_request())

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert "speech input only" in response.error.message
    assert omni.handle is not None
    assert omni.handle.verbs() == [], "nothing may be submitted to a model that cannot answer it"
    assert omni.closed == [_SESSION_ID], "a refused request must not hold the admission slot"


@pytest.mark.asyncio
async def test_audio_content_goes_to_the_input_buffer_and_the_commit_starts_the_turn() -> None:
    omni = FakeOmni([TextDelta(delta="a bell"), ResponseDone(response={"status": "completed"})])

    await _adapter(omni).create_chat_completion(_request(messages=[_audio_message()]))

    handle = omni.handle
    assert handle is not None
    # Speech is a turn on its own: the commit both ends the input and asks for
    # the response, and nothing has to be seeded.
    assert handle.verbs() == ["append_audio", "commit"]
    assert handle.calls[0][1]["audio"] == _PCM  # decoded, not the base64 the caller sent
    assert handle.calls[1][1] == {"final": True, "create_response": True}
    # The commit asks for the response, so the session must not also auto-respond.
    assert "auto_response" not in omni.opened[0].extra_body
    # Text alongside the audio is still seeded: it is the only way it reaches
    # the model at all.
    assert omni.opened[0].initial_user_text == "what is this?"


@pytest.mark.asyncio
async def test_generation_options_are_session_scoped_not_response_scoped() -> None:
    """A model-native session refuses per-response overrides, so they must be set at open time."""
    omni = FakeOmni()

    await _adapter(omni).create_chat_completion(_request(temperature=0.3, max_tokens=64))

    config = omni.opened[0]
    assert isinstance(config, DuplexSessionConfig)
    assert (config.temperature, config.max_tokens) == (0.3, 64)
    assert config.modalities == ["text"]
    # Nothing in the session config asks for voice activity detection: this turn
    # is delimited by the request.
    assert "realtime_turn_detection" not in config.extra_body
    assert omni.handle is not None


# --------------------------------------------------------------------------- #
# The response                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_audio_output_becomes_the_message_audio_field() -> None:
    omni = FakeOmni(
        [
            AudioDelta(delta=base64.b64encode(b"\x00\x01").decode()),
            TranscriptDelta(delta="spoken"),
            AudioDelta(delta=base64.b64encode(b"\x02\x03").decode()),
            ResponseDone(response={"status": "completed"}),
        ]
    )

    response = await _adapter(omni).create_chat_completion(_request())

    assert isinstance(response, ChatCompletionResponse)
    audio = response.choices[0].message.audio
    assert audio is not None
    assert base64.b64decode(audio.data) == b"\x00\x01\x02\x03"
    assert audio.transcript == "spoken"


@pytest.mark.asyncio
async def test_an_incomplete_response_reports_length() -> None:
    omni = FakeOmni([TextDelta(delta="cut"), ResponseDone(response={"status": "incomplete"})])

    response = await _adapter(omni).create_chat_completion(_request())

    assert isinstance(response, ChatCompletionResponse)
    assert response.choices[0].finish_reason == "length"


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 2},
        {"logprobs": True},
        {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
    ],
    ids=["n", "logprobs", "tools"],
)
@pytest.mark.asyncio
async def test_options_a_duplex_turn_cannot_express_are_rejected_without_a_session(kwargs: dict[str, Any]) -> None:
    """Rejected before ``open_session``: a refused request must not cost an admission slot."""
    omni = FakeOmni()

    response = await _adapter(omni).create_chat_completion(_request(**kwargs))

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert omni.opened == []


@pytest.mark.asyncio
async def test_no_admission_slot_is_backpressure_not_a_bad_request() -> None:
    omni = FakeOmni()
    omni.open_error = DuplexSessionError("no slots", code="resource_exhausted")

    response = await _adapter(omni).create_chat_completion(_request())

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 503


@pytest.mark.asyncio
async def test_a_turn_the_model_refuses_is_the_callers_error() -> None:
    """The framework rejects the turn on the wire; the adapter forwards it as a 400."""
    omni = FakeOmni(
        [
            ErrorEvent(
                code="response_create_without_input",
                message="Duplex response.create requires committed audio or conversation items to answer.",
            )
        ]
    )

    response = await _adapter(omni).create_chat_completion(_request())

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert "conversation items" in response.error.message


@pytest.mark.asyncio
async def test_a_session_that_ends_before_the_response_is_a_server_error() -> None:
    omni = FakeOmni([TextDelta(delta="par"), SessionClosed(reason="engine_dead")])

    response = await _adapter(omni).create_chat_completion(_request())

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 500
    assert "engine_dead" in response.error.message


# --------------------------------------------------------------------------- #
# The admission slot                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "script",
    [
        [ResponseDone(response={"status": "completed"})],
        [ErrorEvent(code="internal_error", message="boom")],
        [SessionClosed(reason="engine_dead")],
    ],
    ids=["completed", "error", "session_closed"],
)
@pytest.mark.asyncio
async def test_the_session_is_released_however_the_turn_ends(script: list[DuplexEvent]) -> None:
    omni = FakeOmni(script)

    await _adapter(omni).create_chat_completion(_request())

    assert omni.closed == [_SESSION_ID]


@pytest.mark.asyncio
async def test_the_session_is_released_when_the_turn_raises() -> None:
    """A handle that fails mid-turn still costs a slot until it is closed."""
    omni = FakeOmni()
    adapter = _adapter(omni)

    async def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("submit failed")

    real_open = omni.open_session

    async def open_and_break(config: Any) -> FakeHandle:
        handle = await real_open(config)
        handle.append_audio = explode  # type: ignore[method-assign]
        return handle

    omni.open_session = open_and_break  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="submit failed"):
        await adapter.create_chat_completion(_request())

    assert omni.closed == [_SESSION_ID]


# --------------------------------------------------------------------------- #
# Streaming                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_streaming_emits_one_chunk_per_text_delta_and_releases_the_session() -> None:
    omni = FakeOmni([TextDelta(delta="one"), TextDelta(delta=" two"), ResponseDone(response={"status": "completed"})])

    generator = await _adapter(omni).create_chat_completion(_request(stream=True))
    chunks = [chunk async for chunk in generator]

    assert chunks[-1] == "data: [DONE]\n\n"
    assert all(chunk.startswith("data: ") for chunk in chunks)
    contents = [chunk for chunk in chunks if '"content": "one"' in chunk or '"content": " two"' in chunk]
    assert len(contents) == 2
    assert '"finish_reason": "stop"' in chunks[-2]
    assert omni.closed == [_SESSION_ID]


@pytest.mark.asyncio
async def test_a_stream_that_fails_after_it_started_says_so_rather_than_stopping() -> None:
    """Once bytes are on the wire the status is fixed, so the failure goes in the stream.

    Ending with ``finish_reason: "stop"`` would report a failed turn as a
    complete, empty answer -- which is what a real server did before this:
    HTTP 200 and an empty assistant message, after 305 seconds.
    """
    omni = FakeOmni([ErrorEvent(code="internal_error", message="boom")])

    generator = await _adapter(omni).create_chat_completion(_request(stream=True))
    chunks = [chunk async for chunk in generator]

    assert chunks[-1] == "data: [DONE]\n\n"
    assert '"error"' in chunks[-2] and "boom" in chunks[-2]
    assert not any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert omni.closed == [_SESSION_ID]


@pytest.mark.asyncio
async def test_a_session_that_dies_mid_stream_is_not_reported_as_a_complete_answer() -> None:
    omni = FakeOmni([TextDelta(delta="par"), SessionClosed(reason="engine_dead")])

    generator = await _adapter(omni).create_chat_completion(_request(stream=True))
    chunks = [chunk async for chunk in generator]

    assert '"error"' in chunks[-2] and "engine_dead" in chunks[-2]
    assert not any('"finish_reason": "stop"' in chunk for chunk in chunks)
    assert omni.closed == [_SESSION_ID]


# --------------------------------------------------------------------------- #
# The route                                                                   #
# --------------------------------------------------------------------------- #


def _route_app(omni: FakeOmni) -> FastAPI:
    """A duplex server's ``/v1/chat/completions``, served by the adapter."""
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.openai_serving_chat = _adapter(omni)
    app.state.serving_tokenization = None
    app.state.enable_server_load_tracking = False
    app.state.server_load_metrics = 0
    return app


def test_the_route_returns_the_adapters_completion() -> None:
    """The handler recognises the adapter's response as a ChatCompletionResponse, not a stream."""
    omni = FakeOmni([TextDelta(delta="pong"), ResponseDone(response={"status": "completed"})])

    with TestClient(_route_app(omni)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "fake-duplex-model", "messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "pong"
    assert body["object"] == "chat.completion"
    assert omni.closed == [_SESSION_ID]


def test_the_route_answers_no_admission_slot_with_503() -> None:
    """The duplex error code has to survive as an HTTP status, not collapse to 400."""
    omni = FakeOmni()
    omni.open_error = DuplexSessionError("no slots", code="resource_exhausted")

    with TestClient(_route_app(omni)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "fake-duplex-model", "messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 503
    assert "no slots" in response.json()["error"]["message"]


def test_the_route_streams_the_adapters_sse() -> None:
    omni = FakeOmni([TextDelta(delta="a"), TextDelta(delta="b"), ResponseDone(response={"status": "completed"})])

    with TestClient(_route_app(omni)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "fake-duplex-model", "messages": [{"role": "user", "content": "ping"}], "stream": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")
    assert omni.closed == [_SESSION_ID]


@pytest.mark.asyncio
async def test_the_tts_template_switch_crosses_over_and_the_rest_is_reported(caplog) -> None:
    """``use_tts_template`` is the same switch a session has; the rest of chat_template_kwargs is not."""
    omni = FakeOmni()

    await _adapter(omni).create_chat_completion(
        _request(chat_template_kwargs={"use_tts_template": False, "enable_thinking": False})
    )

    assert omni.opened[0].use_tts_template is False
    assert "enable_thinking" in caplog.text


@pytest.mark.asyncio
async def test_requested_output_modalities_reach_the_session() -> None:
    omni = FakeOmni()

    await _adapter(omni).create_chat_completion(_request(modalities=["text", "audio"]))

    assert omni.opened[0].modalities == ["text", "audio"]


@pytest.mark.asyncio
async def test_image_content_is_refused_rather_than_dropped() -> None:
    """A Realtime conversation item carries text and audio; answering without the image would mislead."""
    omni = FakeOmni()

    response = await _adapter(omni).create_chat_completion(
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this picture?"},
                        {"type": "image_url", "image_url": {"url": "https://example.invalid/cat.png"}},
                    ],
                }
            ]
        )
    )

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert "image_url" in response.error.message
    assert omni.opened == []


@pytest.mark.asyncio
async def test_extra_body_reaches_the_session_so_audio_output_is_askable() -> None:
    """A model that wants ``ref_audio`` for audio output has to be able to receive it."""
    omni = FakeOmni()

    await _adapter(omni).create_chat_completion(
        _request(modalities=["text", "audio"], extra_body={"ref_audio": "/tmp/voice.wav"})
    )

    assert omni.opened[0].extra_body["ref_audio"] == "/tmp/voice.wav"
