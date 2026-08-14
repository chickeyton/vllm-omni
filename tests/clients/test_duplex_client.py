# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the public duplex client (fake transport, no server)."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from vllm_omni.clients.duplex import (
    AudioFormat,
    ConnectionResumed,
    DuplexClient,
    DuplexProtocolError,
    DuplexSessionClosedError,
    EventCollector,
    ReconnectPolicy,
    SessionConfig,
    SessionResumed,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_CREATED = {
    "type": "session.created",
    "session": {"session_id": "sess-1"},
    "incarnation": 0,
    "resume_token": "tok-1",
    "server_event_seq": 1,
}
SESSION_CLOSED = {"type": "session.closed", "session_id": "sess-1", "server_event_seq": 99}


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, raw: str) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)

    async def close(self) -> None:
        self.closed = True

    def feed(self, event: dict[str, object] | BaseException) -> None:
        self.incoming.put_nowait(event)

    def sent_types(self) -> list[object]:
        return [event.get("type") for event in self.sent]


def make_client(*sockets: FakeSocket, **kwargs) -> tuple[DuplexClient, list[str]]:
    remaining = list(sockets)
    calls: list[str] = []

    async def connect(url: str):
        calls.append(url)
        if not remaining:
            raise ConnectionError("no more sockets")
        return remaining.pop(0)

    kwargs.setdefault("heartbeat_interval_s", None)
    kwargs.setdefault("reconnect", None)
    kwargs.setdefault("handshake_timeout_s", 5.0)
    client = DuplexClient("ws://test-host:8099", model="test-model", connect=connect, **kwargs)
    return client, calls


# ---------------------------------------------------------------------------
# SessionConfig


def test_session_config_payload_defaults():
    payload = SessionConfig().to_session_payload(model="m", session_id="s")
    assert payload["model"] == "m"
    assert payload["session_id"] == "s"
    assert payload["modalities"] == ["audio", "text"]
    assert payload["input_audio_format"] == "pcm16"
    assert payload["output_audio_format"] == "pcm16"
    assert payload["turn_detection"] is None
    assert payload["extra_body"] == {"auto_response": True}
    assert "voice" not in payload
    assert "ref_audio" not in payload


def test_session_config_minicpmo_preset():
    config = SessionConfig.for_minicpmo45(ref_audio="data:audio/wav;base64,AAA=", temperature=0.0)
    payload = config.to_session_payload(model="openbmb/MiniCPM-o-4_5")
    assert payload["ref_audio"] == "data:audio/wav;base64,AAA="
    assert payload["overlap_policy"] == "listen_only"
    assert payload["playback_commit_policy"] == "ack_only"
    assert payload["extra_body"]["minicpmo45_native_duplex"] is True
    assert payload["extra_body"]["auto_response"] is True
    assert payload["temperature"] == 0.0


def test_session_config_personaplex_preset():
    config = SessionConfig.for_personaplex(voice="NATF2.pt", persona="You are calm.")
    assert config.input_audio == AudioFormat("pcm_f32le", 24_000)
    payload = config.to_session_payload(model="nvidia/personaplex-7b-v1")
    assert payload["input_audio_format"] == "pcm_f32le"
    assert payload["voice"] == "NATF2.pt"
    assert payload["instructions"] == "You are calm."


def test_audio_format_math():
    fmt = AudioFormat("pcm16", 16_000)
    assert fmt.byte_count(100) == 3200
    assert fmt.duration_ms(3200) == 100.0
    f32 = AudioFormat("pcm_f32le", 24_000)
    assert f32.byte_count(80) == 1920 * 4
    with pytest.raises(ValueError):
        _ = AudioFormat("mp3", 16_000).bytes_per_sample


# ---------------------------------------------------------------------------
# Handshake and lifecycle


async def test_handshake_adopts_session_and_acks():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, calls = make_client(sock)
    async with client:
        assert client.session_id == "sess-1"
        assert client.resume_token == "tok-1"
        assert calls == ["ws://test-host:8099/v1/realtime?duplex=1&model=test-model&autostart=0"]
        assert sock.sent[0]["type"] == "session.update"
        assert sock.sent[0]["session"]["model"] == "test-model"
        await _drain(lambda: {"type": "session.event_ack", "server_event_seq": 1} in _acks(sock))
        close_task = asyncio.create_task(client.close())
        await _drain(lambda: "session.close" in sock.sent_types())
        sock.feed(SESSION_CLOSED)
        await close_task
    assert "session.close" in sock.sent_types()


async def test_handshake_error_raises_protocol_error():
    sock = FakeSocket()
    sock.feed({"type": "error", "error": {"code": "unsupported_audio_format", "message": "bad"}})
    client, _ = make_client(sock)
    with pytest.raises(DuplexProtocolError) as excinfo:
        async with client:
            pass
    assert excinfo.value.code == "unsupported_audio_format"


async def test_session_expired_raises_from_event_stream():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "session.expired", "reason": "lease_expired", "server_event_seq": 2})
        with pytest.raises(DuplexSessionClosedError) as excinfo:
            async for _ in client.events():
                pass
        assert "lease_expired" in excinfo.value.reason


# ---------------------------------------------------------------------------
# Input events


async def test_append_audio_tracks_cumulative_end_ms():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        pcm = b"\x01\x02" * 1600  # 100 ms of pcm16 @ 16 kHz
        await client.append_audio(pcm)
        await client.append_audio(pcm, is_speech=True, final=True)
        appends = [event for event in sock.sent if event.get("type") == "input_audio_buffer.append"]
        assert [a["audio_end_ms"] for a in appends] == [100, 200]
        assert appends[0]["format"] == "pcm16"
        assert appends[0]["sample_rate_hz"] == 16_000
        assert appends[0]["duration_ms"] == 100
        assert "is_speech" not in appends[0]
        assert appends[1]["is_speech"] is True
        assert appends[1]["final"] is True
        assert base64.b64decode(appends[0]["audio"]) == pcm
        sock.feed(SESSION_CLOSED)


async def test_stream_pcm_chunking():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        await client.stream_pcm(b"\x00" * 8000, chunk_ms=100, realtime=False)
        appends = [event for event in sock.sent if event.get("type") == "input_audio_buffer.append"]
        assert [a["duration_ms"] for a in appends] == [100, 100, 50]
        sock.feed(SESSION_CLOSED)


async def test_barge_in_composes_documented_events():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        await client.barge_in()
        types = sock.sent_types()
        assert types.index("response.cancel") < types.index("input_audio_buffer.clear")
        sock.feed(SESSION_CLOSED)


async def test_append_text_uses_conversation_item():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        await client.append_text("hello")
        item_events = [event for event in sock.sent if event.get("type") == "conversation.item.create"]
        assert item_events[0]["item"]["content"] == [{"type": "input_text", "text": "hello"}]
        sock.feed(SESSION_CLOSED)


# ---------------------------------------------------------------------------
# Response demultiplexing


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


async def test_response_handle_flow():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    chunk = b"\x00\x01" * 2400  # 100 ms of pcm16 @ 24 kHz
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        sock.feed({"type": "response.speak", "response_id": "resp-1", "server_event_seq": 3})
        sock.feed(
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "delta": _b64(chunk),
                "sample_rate_hz": 24_000,
                "server_event_seq": 4,
            }
        )
        sock.feed(
            {
                "type": "response.audio_transcript.delta",
                "response_id": "resp-1",
                "delta": "hi there",
                "server_event_seq": 5,
            }
        )
        sock.feed({"type": "response.done", "response_id": "resp-1", "server_event_seq": 6})

        async for response in client.responses():
            chunks = [piece async for piece in response.audio()]
            assert chunks == [chunk]
            assert response.decision == "speak"
            assert response.transcript == "hi there"
            assert response.played_ms == pytest.approx(100.0)
            assert response.finished
            break
        sock.feed(SESSION_CLOSED)


async def test_listen_decision_yields_finished_silent_handle():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "response.listen", "server_event_seq": 2})
        async for response in client.responses():
            assert response.decision == "listen"
            assert response.finished
            assert [piece async for piece in response.audio()] == []
            break
        sock.feed(SESSION_CLOSED)


async def test_listen_terminates_active_response():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        sock.feed({"type": "response.listen", "response_id": "resp-1", "server_event_seq": 3})
        async for response in client.responses():
            await response.wait(timeout_s=5.0)
            assert response.decision == "listen"
            break
        sock.feed(SESSION_CLOSED)


# ---------------------------------------------------------------------------
# Resume


async def test_resume_after_transport_drop():
    first = FakeSocket()
    first.feed(SESSION_CREATED)
    second = FakeSocket()
    second.feed(
        {
            "type": "session.resumed",
            "session": {"session_id": "sess-1"},
            "incarnation": 0,
            "resume_token": "tok-2",
            "server_event_seq": 5,
        }
    )
    client, calls = make_client(
        first,
        second,
        reconnect=ReconnectPolicy(max_attempts=2, backoff_s=(0.0, 0.0)),
    )
    async with client:
        stream = client.events()

        async def take_two():
            seen = []
            async for event in stream:
                seen.append(event)
                if len(seen) == 2:
                    return seen

        consumer = asyncio.create_task(take_two())
        await asyncio.sleep(0)  # let the consumer subscribe before the drop
        await asyncio.sleep(0)
        first.feed(RuntimeError("transport dropped"))
        seen = await asyncio.wait_for(consumer, timeout=5.0)
        assert isinstance(seen[0], ConnectionResumed)
        assert isinstance(seen[1], SessionResumed)
        assert client.resume_token == "tok-2"
        assert len(calls) == 2
        resume = second.sent[0]
        assert resume["type"] == "session.resume"
        assert resume["session_id"] == "sess-1"
        assert resume["incarnation"] == 0
        assert resume["resume_token"] == "tok-1"
        assert resume["last_received_server_event_seq"] == 1
        second.feed(SESSION_CLOSED)


async def test_no_reconnect_policy_surfaces_closed():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)  # reconnect=None
    async with client:
        stream = client.events()
        sock.feed(RuntimeError("transport dropped"))
        with pytest.raises(DuplexSessionClosedError):
            async for _ in stream:
                pass


async def test_fatal_resume_error_gives_up():
    first = FakeSocket()
    first.feed(SESSION_CREATED)
    second = FakeSocket()
    second.feed({"type": "error", "error": {"code": "invalid_resume_token", "message": "nope"}})
    client, calls = make_client(
        first,
        second,
        reconnect=ReconnectPolicy(max_attempts=3, backoff_s=(0.0, 0.0)),
    )
    async with client:
        stream = client.events()
        first.feed(RuntimeError("transport dropped"))
        with pytest.raises(DuplexSessionClosedError):
            async for _ in stream:
                pass
        assert len(calls) == 2  # no retry after a fatal resume error


# ---------------------------------------------------------------------------
# Collector


def test_event_collector_accumulates_audio():
    collector = EventCollector()
    collector.add({"type": "response.created", "response": {"id": "r1"}}, received_at_s=1.0)
    collector.add(
        {"type": "response.audio.delta", "response_id": "r1", "delta": _b64(b"ab"), "sample_rate_hz": 16_000},
        received_at_s=1.1,
    )
    collector.add(
        {"type": "response.output_audio.delta", "response_id": "r1", "audio": _b64(b"cd")},
        received_at_s=1.2,
    )
    assert collector.count("response.created") == 1
    assert collector.audio_bytes() == b"abcd"
    assert collector.output_sample_rate_hz == 16_000
    summary = collector.timing_summary(after_s=0.0)
    assert summary["audio_output"]["chunk_count"] == 2
    assert summary["audio_output"]["response_created_to_first_audio_ms"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# helpers


def _acks(sock: FakeSocket) -> list[dict[str, object]]:
    return [event for event in sock.sent if event.get("type") == "session.event_ack"]


async def _drain(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)
