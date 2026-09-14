# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The engine-side Silero VAD: backend selection and endpoint rules.

The endpoint rules are the part worth pinning hardest. They were reimplemented
when turn detection moved into the engine, and drifted from the serving-side
``ThresholdEndpointPolicy`` in two ways that only show up mid-utterance: a loud
frame used to cancel an in-progress silence timer, and ``audio_end_ms`` used to
exclude the trailing silence OpenAI says it should include. The first test here
runs both implementations over the same probability sequences so the two cannot
drift apart again silently.
"""

from __future__ import annotations

import base64
import hashlib

import numpy as np
import pytest

from vllm_omni.engine.duplex.vad import (
    SILERO_VAD_SHA256,
    ServerVADUnavailableError,
    SileroStreamingVAD,
    SileroVADBackend,
    SileroVADBackendProvider,
    SileroVADConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FRAME = 512
SAMPLE_RATE_HZ = 16_000


def _drive(config: SileroVADConfig, probabilities: list[float]) -> list[tuple[str, int | None]]:
    """Feed one probability per frame and collect the endpoint events."""
    scores = iter(probabilities)
    vad = SileroStreamingVAD(config, frame_scorer=lambda _frame: next(scores))
    events: list[tuple[str, int | None]] = []
    for _ in probabilities:
        result = vad.process(np.zeros(FRAME, dtype=np.float32))
        if result.speech_started:
            events.append(("start", result.speech_start_ms))
        if result.speech_stopped:
            events.append(("stop", result.speech_end_ms))
    return events


# --------------------------------------------------------------------------- #
# Parity with the serving-side policy this was ported from                    #
# --------------------------------------------------------------------------- #

_PARITY_CASES = {
    "one utterance": (
        [0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        {"threshold": 0.5, "prefix_padding_ms": 0, "silence_duration_ms": 100, "min_speech_duration_ms": 32},
    ),
    "prefix padding reaches back": (
        [0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0],
        {"threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 64, "min_speech_duration_ms": 32},
    ),
    "loud frames inside the silence timer": (
        [0.9, 0.9, 0.1, 0.45, 0.1, 0.1, 0.1],
        {"threshold": 0.5, "prefix_padding_ms": 0, "silence_duration_ms": 96, "min_speech_duration_ms": 32},
    ),
    "min speech duration rejects a blip": (
        [0.9, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0],
        {"threshold": 0.5, "prefix_padding_ms": 0, "silence_duration_ms": 64, "min_speech_duration_ms": 96},
    ),
    "negative threshold floor at a low threshold": (
        [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
        {"threshold": 0.16, "prefix_padding_ms": 0, "silence_duration_ms": 64, "min_speech_duration_ms": 32},
    ),
}


@pytest.mark.parametrize(("probabilities", "options"), _PARITY_CASES.values(), ids=list(_PARITY_CASES))
def test_endpoint_decisions_match_the_serving_side_policy(
    probabilities: list[float], options: dict[str, object]
) -> None:
    """The engine detector and ``ThresholdEndpointPolicy`` must agree frame for frame."""
    server_vad = pytest.importorskip(
        "vllm_omni.entrypoints.duplex.server_vad",
        reason="the serving-side policy is the oracle for this parity check",
    )
    policy = server_vad.ThresholdEndpointPolicy(
        server_vad.ServerVADConfig(type="server_vad", **options),
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    expected: list[tuple[str, int | None]] = []
    for index, probability in enumerate(probabilities):
        decision = policy.update(probability, frame_start_sample=index * FRAME, frame_samples=FRAME)
        if decision.speech_started:
            expected.append(("start", decision.audio_start_ms))
        if decision.speech_stopped:
            expected.append(("stop", decision.audio_end_ms))

    assert _drive(SileroVADConfig(**options), probabilities) == expected


# --------------------------------------------------------------------------- #
# The two rules that had drifted                                              #
# --------------------------------------------------------------------------- #


def test_a_loud_frame_does_not_restart_the_silence_timer() -> None:
    """Silero v6.2 hysteresis: once silence is running, a loud frame only delays it.

    The frame at 0.45 sits above the negative threshold (0.35) but below the
    activation threshold. It must not cancel the pending endpoint, or a speaker
    who trails off unevenly never gets their turn committed.
    """
    config = SileroVADConfig(threshold=0.5, prefix_padding_ms=0, silence_duration_ms=96, min_speech_duration_ms=32)
    assert _drive(config, [0.9, 0.9, 0.1, 0.45, 0.1, 0.1]) == [("start", 0), ("stop", 160)]


def test_audio_end_ms_includes_the_trailing_silence() -> None:
    """OpenAI defines audio_end_ms as the end of the audio sent to the model.

    That includes the silence spent deciding the turn was over, so it is the
    frame boundary the detector stopped on, not where speech last was.
    """
    config = SileroVADConfig(threshold=0.5, prefix_padding_ms=0, silence_duration_ms=100, min_speech_duration_ms=32)
    events = _drive(config, [0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stop_ms = next(ms for kind, ms in events if kind == "stop")
    # 8 frames consumed when the timer expires: 8 * 512 / 16 kHz = 256 ms.
    assert stop_ms == 256


def test_reset_keeps_the_session_clock_and_clamps_the_prefix() -> None:
    """A barge-in resets the detector but not the session's timeline.

    Timestamps have to stay comparable with the rest of the session, and prefix
    padding must not reach back into audio the reset discarded.
    """
    config = SileroVADConfig(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=64, min_speech_duration_ms=32)
    scores = iter([0.0] * 10 + [0.9, 0.9, 0.9])
    vad = SileroStreamingVAD(config, frame_scorer=lambda _frame: next(scores))
    for _ in range(10):
        vad.process(np.zeros(FRAME, dtype=np.float32))

    stream_start_ms = round(10 * FRAME * 1000 / SAMPLE_RATE_HZ)
    vad.reset()
    result = vad.process(np.zeros(FRAME * 3, dtype=np.float32))

    assert result.speech_started
    # Without the clock the start would land at 0; without the clamp, 300 ms of
    # prefix would reach back before the reset.
    assert result.speech_start_ms == stream_start_ms


# --------------------------------------------------------------------------- #
# Input handling                                                              #
# --------------------------------------------------------------------------- #


def _pcm16_b64(samples: np.ndarray) -> str:
    return base64.b64encode((samples * 32767).astype("<i2").tobytes()).decode("ascii")


def test_pcm16_input_is_accepted_alongside_float32() -> None:
    config = SileroVADConfig(threshold=0.5)
    tone = np.zeros(FRAME * 2, dtype=np.float32)
    seen: list[int] = []
    vad = SileroStreamingVAD(config, frame_scorer=lambda frame: seen.append(frame.size) or 0.0)
    vad.process_base64(_pcm16_b64(tone), fmt="pcm16", sample_rate_hz=SAMPLE_RATE_HZ)
    assert seen == [FRAME, FRAME]


def test_an_odd_pcm16_byte_count_is_rejected() -> None:
    vad = SileroStreamingVAD(SileroVADConfig(), frame_scorer=lambda _frame: 0.0)
    with pytest.raises(ValueError, match="incomplete pcm16"):
        vad.process_base64(base64.b64encode(b"\x00\x01\x02").decode(), fmt="pcm16", sample_rate_hz=SAMPLE_RATE_HZ)


def test_resampling_keeps_the_frame_grid_from_drifting() -> None:
    """The 24 kHz remainder carries across chunks, so frame boundaries cannot drift.

    This is a weaker claim than upstream's ``StreamingAudioResampler``: the
    interpolation is per chunk, so values near a chunk edge differ slightly.
    What must hold is the frame *count* and alignment, which is what the
    endpoint timestamps are derived from.
    """
    source = np.sin(np.linspace(0, 40 * np.pi, 24_000, dtype=np.float32))

    def frames_for(chunk_sizes: list[int]) -> int:
        frames = 0

        def count(_frame: np.ndarray) -> float:
            nonlocal frames
            frames += 1
            return 0.0

        vad = SileroStreamingVAD(SileroVADConfig(), frame_scorer=count)
        offset = 0
        for size in chunk_sizes:
            vad.process_base64(_pcm16_b64(source[offset : offset + size]), fmt="pcm16", sample_rate_hz=24_000)
            offset += size
        assert offset == source.size
        return frames

    whole = frames_for([source.size])
    split = frames_for([7, 13, 511, 512, 513, source.size - 1_556])
    assert whole == split == 16_000 // FRAME


def test_a_sample_rate_change_mid_stream_is_rejected() -> None:
    vad = SileroStreamingVAD(SileroVADConfig(), frame_scorer=lambda _frame: 0.0)
    chunk = _pcm16_b64(np.zeros(1_024, dtype=np.float32))
    vad.process_base64(chunk, fmt="pcm16", sample_rate_hz=24_000)
    with pytest.raises(ValueError, match="sample rate cannot change"):
        vad.process_base64(chunk, fmt="pcm16", sample_rate_hz=48_000)


# --------------------------------------------------------------------------- #
# Backend selection                                                           #
# --------------------------------------------------------------------------- #


def test_a_configured_model_path_that_does_not_exist_is_an_error() -> None:
    """An operator who names an artifact gets told it is missing, not a fallback."""
    provider = SileroVADBackendProvider(model_path="/nonexistent/silero_vad.onnx")
    with pytest.raises(ServerVADUnavailableError, match="does not exist"):
        provider.get()


def test_a_configured_model_with_the_wrong_checksum_is_refused(tmp_path) -> None:
    """A different Silero revision changes endpointing, so the pin is enforced."""
    impostor = tmp_path / "silero_vad.onnx"
    impostor.write_bytes(b"not the pinned model")
    provider = SileroVADBackendProvider(model_path=str(impostor))
    with pytest.raises(ServerVADUnavailableError, match="checksum mismatch"):
        provider.get()
    assert hashlib.sha256(impostor.read_bytes()).hexdigest() != SILERO_VAD_SHA256


def test_the_onnx_backend_is_shared_by_every_session() -> None:
    """One ONNX session for the whole engine: its state is passed in, not held."""
    provider = SileroVADBackendProvider()
    try:
        backend = provider.get()
    except ServerVADUnavailableError as exc:
        pytest.skip(f"no Silero backend available here: {exc}")
    if not isinstance(backend, SileroVADBackend):
        pytest.skip("no local ONNX artifact; the torch fallback is per-session by design")

    assert provider.get() is backend
    # State is the caller's, so two sessions on one backend cannot interfere.
    first, second = backend.new_state(), backend.new_state()
    assert first is not second
    probability, next_state = backend.infer(np.zeros(FRAME, dtype=np.float32), first)
    assert 0.0 <= probability <= 1.0
    assert next_state is not first
