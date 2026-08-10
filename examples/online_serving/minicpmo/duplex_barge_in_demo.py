"""Full-duplex use case on the native ``/v1/duplex`` endpoint: talk over the assistant.

Scenario
--------
1. The user asks a question (``--question-wav``); the model answers in speech.
2. Mid-answer, the user starts talking again (``--interrupt-wav``) — a
   follow-up or correction — while assistant audio is still streaming.
3. The model handles the overlap: depending on the session overlap policy and
   speech length, the in-flight response is barged in (cancelled) or the
   commit is deferred to the turn end, and a new response answers the
   interruption. Either way the user never waits for the first answer.

This demo speaks the native duplex event vocabulary directly (no OpenAI
Realtime projection) and therefore needs only the ``websockets`` package —
not the vllm_omni package. The wire contract it uses:

  client -> server
    session.create              {session: {model, modalities, ref_audio, extra_body}}
    input_audio_buffer.append   {audio: b64 pcm16, format, sample_rate_hz}
    input_audio_buffer.commit   {final, response_create}
    playback.ack                {committed_ms, played_ms}
    session.close               {}
  server -> client
    session.created / session.closed / error
    input.committed
    response.created / response.speak / response.listen / response.done
    response.output_audio.delta {audio: b64 pcm16, text, sample_rate_hz}
    audio.cancelled             (a barged-in response's audio is cut)

Run the server first (see serve_duplex.sh), then:

    python duplex_barge_in_demo.py \
        --url ws://127.0.0.1:8099/v1/duplex \
        --ref-audio /path/to/reference_voice.wav \
        --question-wav question_16k.wav \
        --interrupt-wav follow_up_16k.wav \
        --output-dir ./duplex_out

Input WAVs must be mono 16 kHz PCM16. ``--ref-audio`` is required for audio
output (MiniCPM-o native duplex needs a reference voice; without it the
session is rejected with ``ref_audio_required``). Outputs land in
--output-dir as one WAV per response plus a JSON event summary.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import wave
from pathlib import Path

try:
    import websockets
except ImportError as exc:  # pragma: no cover - example dependency
    raise SystemExit("Install websockets first: pip install websockets") from exc

PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2
OUTPUT_SAMPLE_RATE_DEFAULT = 24_000


def read_pcm16_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError(f"{path} must be mono PCM16")
        if wav_file.getframerate() != PCM16_SAMPLE_RATE:
            raise ValueError(f"{path} must be {PCM16_SAMPLE_RATE} Hz")
        return wav_file.readframes(wav_file.getnframes())


def write_pcm16_wav(path: Path, pcm16: bytes, *, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm16)


class NativeDuplexClient:
    """Minimal client for the native /v1/duplex websocket dialect."""

    def __init__(self, url: str, *, max_size: int = 64 * 1024 * 1024) -> None:
        self.url = url
        self.max_size = max_size
        self.events: list[dict[str, object]] = []
        self._ws = None
        self._reader: asyncio.Task | None = None

    async def __aenter__(self) -> NativeDuplexClient:
        self._ws = await websockets.connect(self.url, max_size=self.max_size)
        self._reader = asyncio.create_task(self._read_loop())
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            await self._ws.close()

    async def _read_loop(self) -> None:
        async for raw in self._ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                self.events.append(event)

    async def send(self, event: dict[str, object]) -> None:
        await self._ws.send(json.dumps(event))

    async def wait_for(self, predicate, *, timeout_s: float, label: str) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while not predicate():
            if loop.time() >= deadline:
                raise TimeoutError(f"timed out waiting for {label}")
            await asyncio.sleep(0.05)

    async def create_session(
        self,
        model: str,
        *,
        ref_audio_data_url: str,
        timeout_s: float,
    ) -> None:
        await self.send(
            {
                "type": "session.create",
                "session": {
                    "model": model,
                    "modalities": ["text", "audio"],
                    "ref_audio": ref_audio_data_url,
                    "extra_body": {"minicpmo45_native_duplex": True},
                },
            }
        )
        await self.wait_for(
            lambda: any(e.get("type") in {"session.created", "error"} for e in self.events),
            timeout_s=timeout_s,
            label="session.created",
        )
        errors = [e for e in self.events if e.get("type") == "error"]
        if errors:
            raise RuntimeError(f"session.create failed: {errors[0]}")

    async def stream_pcm16(self, pcm16: bytes, *, chunk_ms: int) -> None:
        chunk_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000
        for offset in range(0, len(pcm16), chunk_bytes):
            await self.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16[offset : offset + chunk_bytes]).decode("ascii"),
                    "format": "pcm16",
                    "sample_rate_hz": PCM16_SAMPLE_RATE,
                }
            )
            await asyncio.sleep(chunk_ms / 1000)  # realtime pacing

    async def commit(self) -> None:
        await self.send({"type": "input_audio_buffer.commit", "final": True, "response_create": False})

    async def acknowledge_playback(self, played_ms: int) -> None:
        await self.send({"type": "playback.ack", "committed_ms": played_ms, "played_ms": played_ms})

    async def close_session(self, *, timeout_s: float) -> None:
        await self.send({"type": "session.close"})
        try:
            await self.wait_for(
                lambda: any(e.get("type") == "session.closed" for e in self.events),
                timeout_s=timeout_s,
                label="session.closed",
            )
        except TimeoutError:
            pass


def _responses(events: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Fold the native event stream into per-response audio, text, and status."""
    out: dict[str, dict[str, object]] = {}

    def entry(response_id: object) -> dict[str, object] | None:
        if not isinstance(response_id, str) or not response_id:
            return None
        return out.setdefault(
            response_id,
            {"audio": b"", "text": "", "status": "in_progress", "sample_rate_hz": OUTPUT_SAMPLE_RATE_DEFAULT},
        )

    for event in events:
        etype = event.get("type")
        item = entry(event.get("response_id"))
        if etype == "response.output_audio.delta" and item is not None:
            audio = event.get("audio")
            if isinstance(audio, str) and audio:
                item["audio"] += base64.b64decode(audio)
            text = event.get("text")
            if isinstance(text, str):
                item["text"] += text
            sample_rate_hz = event.get("sample_rate_hz")
            if isinstance(sample_rate_hz, int) and sample_rate_hz > 0:
                item["sample_rate_hz"] = sample_rate_hz
        elif etype == "response.done" and item is not None:
            if item["status"] != "cancelled":
                item["status"] = "completed"
        elif etype == "audio.cancelled":
            cancelled = entry(event.get("response_id"))
            targets = [cancelled] if cancelled is not None else list(out.values())
            for target in targets:
                if target["status"] == "in_progress":
                    target["status"] = "cancelled"
    return out


async def run(args: argparse.Namespace) -> int:
    question = read_pcm16_wav(Path(args.question_wav))
    interrupt = read_pcm16_wav(Path(args.interrupt_wav))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_audio_data_url = "data:audio/wav;base64," + base64.b64encode(
        Path(args.ref_audio).read_bytes()
    ).decode("ascii")

    client = NativeDuplexClient(args.url)
    async with client:
        await client.create_session(
            args.model,
            ref_audio_data_url=ref_audio_data_url,
            timeout_s=args.timeout_s,
        )

        # --- Turn 1: the question. Stream at realtime pace, then commit the turn.
        print("[1] streaming question ...", file=sys.stderr)
        await client.stream_pcm16(question, chunk_ms=args.chunk_ms)
        await client.commit()
        await client.wait_for(
            lambda: any(e.get("type") == "response.created" for e in client.events),
            timeout_s=args.timeout_s,
            label="first response.created",
        )
        print("[1] assistant is answering", file=sys.stderr)

        # --- Let the answer play for a moment, as a listening user would.
        await asyncio.sleep(args.listen_s)

        # --- Turn 2: talk over the assistant while its audio still streams.
        # The serving overlap policy decides: enough overlapped speech barges
        # in (audio.cancelled + response.done) or a short commit is deferred
        # until the model yields the turn.
        print("[2] interrupting mid-answer ...", file=sys.stderr)
        events_before_interrupt = len(client.events)
        await client.stream_pcm16(interrupt, chunk_ms=args.chunk_ms)
        await client.commit()

        # --- Wait for the answer to the interruption to finish.
        await client.wait_for(
            lambda: any(
                e.get("type") == "response.done" for e in client.events[events_before_interrupt:]
            ),
            timeout_s=args.timeout_s,
            label="response.done after interruption",
        )
        responses = _responses(client.events)
        played_ms = sum(
            len(item["audio"]) * 1000 // (item["sample_rate_hz"] * PCM16_BYTES_PER_SAMPLE)
            for item in responses.values()
        )
        await client.acknowledge_playback(played_ms)
        await client.close_session(timeout_s=args.timeout_s)

    # --- Report: one WAV per response, statuses show what the overlap did.
    responses = _responses(client.events)
    summary: list[dict[str, object]] = []
    for index, (response_id, item) in enumerate(responses.items(), start=1):
        wav_path = output_dir / f"response_{index}_{item['status']}.wav"
        if item["audio"]:
            write_pcm16_wav(wav_path, item["audio"], sample_rate_hz=item["sample_rate_hz"])
        summary.append(
            {
                "response_id": response_id,
                "status": item["status"],
                "audio_s": round(len(item["audio"]) / (item["sample_rate_hz"] * PCM16_BYTES_PER_SAMPLE), 2),
                "text": item["text"],
                "wav": wav_path.name if item["audio"] else None,
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    cancelled = sum(1 for item in summary if item["status"] == "cancelled")
    print(
        f"\n{len(summary)} responses ({cancelled} interrupted); outputs in {output_dir}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/duplex")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--ref-audio", required=True, help="reference voice WAV (required for audio output)")
    parser.add_argument("--question-wav", required=True, help="mono 16 kHz PCM16 question")
    parser.add_argument("--interrupt-wav", required=True, help="mono 16 kHz PCM16 follow-up spoken mid-answer")
    parser.add_argument("--output-dir", default="./duplex_out")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--listen-s", type=float, default=2.0, help="how long to listen before interrupting")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
