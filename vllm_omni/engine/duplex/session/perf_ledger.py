# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session record of stage work and emitted audio, by turn and chunk (RFC #8325).

The ledger keeps raw data only: the stage processes' step spans, each chunk's
acceptance and handling times, and the audio each chunk produced. Clients derive
busy RTF and the stall ratio from a ``session.stats`` snapshot
(``vllm_omni.benchmarks.duplex_session_metrics``).

It lives on ``DuplexEngineSession`` and is mutated only on the orchestrator loop,
so it needs no locking. Times are ``time.monotonic_ns()``, which every process on
the host shares; snapshots report them relative to the session's open time.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from vllm_omni.core.sched.duplex_stage_work import PERF_CHUNK_KEY

SCHEMA_VERSION = 1
DEFAULT_MAX_TURNS = 256
DEFAULT_MAX_SPANS_PER_CHUNK = 4096

TURN_OPEN = "open"
TURN_DONE = "done"
TURN_CANCELLED = "cancelled"
TURN_LISTEN = "listen"

CHUNK_INPUT = "input"
CHUNK_SILENCE_CONTINUATION = "silence_continuation"


@dataclass
class AudioPiece:
    response_id: str | None
    emitted_ns: int
    duration_ms: float


@dataclass
class ChunkPerfRecord:
    seq: int
    turn_seq: int
    kind: str
    submitted_ns: int
    # stage_id -> [[start_ns, duration_ns], ...], absolute monotonic time
    stage_work: dict[int, list[list[int]]] = field(default_factory=dict)
    audio: list[AudioPiece] = field(default_factory=list)
    # Latest time the API process finished handling a stage output of this chunk.
    handled_ns: int | None = None
    complete: bool = False
    truncated: bool = False
    num_spans: int = 0

    def done_ns(self) -> int | None:
        """End of the in-engine interval: the last handling, span end or audio emission."""
        if not self.complete:
            return None
        ends = [self.submitted_ns]
        if self.handled_ns is not None:
            ends.append(self.handled_ns)
        ends.extend(start + duration for spans in self.stage_work.values() for start, duration in spans)
        ends.extend(piece.emitted_ns for piece in self.audio)
        return max(ends)


@dataclass
class TurnPerfRecord:
    epoch: int
    turn_id: int
    status: str = TURN_OPEN
    response_ids: list[str] = field(default_factory=list)
    chunks: dict[int, ChunkPerfRecord] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int]:
        return self.epoch, self.turn_id

    @property
    def finished(self) -> bool:
        return self.status != TURN_OPEN


class SessionPerfLedger:
    """Bounded per-session store of turns and their chunks."""

    def __init__(
        self,
        *,
        stages: Sequence[tuple[int, str]],
        opened_at_ns: int | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_spans_per_chunk: int = DEFAULT_MAX_SPANS_PER_CHUNK,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._clock = clock
        self.opened_at_ns = clock() if opened_at_ns is None else opened_at_ns
        self.stages = [(int(stage_id), str(name)) for stage_id, name in stages]
        self.max_turns = max(1, int(max_turns))
        self.max_spans_per_chunk = max(1, int(max_spans_per_chunk))
        self.dropped_turns = 0
        self.dropped_spans = 0
        # Insertion order is turn order; eviction takes the oldest finished turn.
        self._turns: dict[tuple[int, int], TurnPerfRecord] = {}
        self._chunk_turns: dict[tuple[int, int], tuple[int, int]] = {}

    def now_ns(self) -> int:
        return self._clock()

    # ------------------------------------------------------------------ #
    # Recording                                                          #
    # ------------------------------------------------------------------ #

    def open_chunk(
        self,
        *,
        epoch: int,
        seq: int,
        turn_id: int,
        turn_seq: int,
        kind: str,
        submitted_ns: int,
    ) -> None:
        turn = self._turn(epoch, turn_id, create=True)
        if turn is None:
            return
        turn.chunks[seq] = ChunkPerfRecord(seq=seq, turn_seq=turn_seq, kind=kind, submitted_ns=submitted_ns)
        self._chunk_turns[(epoch, seq)] = turn.key

    def record_stage_work(self, rows: Iterable[Sequence[int]]) -> None:
        """Add ``(stage_id, epoch, seq, start_ns, duration_ns)`` spans; unknown chunks are ignored."""
        for stage_id, epoch, seq, start_ns, duration_ns in rows:
            chunk = self._chunk(epoch, seq)
            if chunk is None:
                continue
            if chunk.num_spans >= self.max_spans_per_chunk:
                chunk.truncated = True
                self.dropped_spans += 1
                continue
            spans = chunk.stage_work.setdefault(int(stage_id), [])
            if spans and start_ns <= spans[-1][0] + spans[-1][1]:
                # The recorder coalesces within a flush; join across flushes too.
                last = spans[-1]
                last[1] = max(last[1], start_ns + duration_ns - last[0])
                continue
            spans.append([int(start_ns), int(duration_ns)])
            chunk.num_spans += 1

    def mark_handled(self, chunks: Iterable[tuple[int, int]], handled_ns: int | None = None) -> None:
        now = self._clock() if handled_ns is None else handled_ns
        for epoch, seq in chunks:
            chunk = self._chunk(epoch, seq)
            if chunk is not None and (chunk.handled_ns is None or now > chunk.handled_ns):
                chunk.handled_ns = now

    def record_audio(
        self,
        chunk_key: tuple[int, int] | None,
        *,
        response_id: str | None,
        duration_ms: float,
        emitted_ns: int | None = None,
        fallback_turn: tuple[int, int] | None = None,
    ) -> None:
        """Attribute one audio delta to the chunk whose step produced it.

        Output that carried no step span (e.g. out-of-band first audio) falls back
        to the latest chunk of ``fallback_turn``.
        """
        if duration_ms <= 0:
            return
        chunk = self._chunk(*chunk_key) if chunk_key is not None else None
        if chunk is None and fallback_turn is not None:
            turn = self._turns.get(fallback_turn)
            if turn is not None and turn.chunks:
                chunk = turn.chunks[max(turn.chunks)]
        if chunk is None:
            return
        chunk.audio.append(
            AudioPiece(
                response_id=response_id,
                emitted_ns=self._clock() if emitted_ns is None else emitted_ns,
                duration_ms=float(duration_ms),
            )
        )

    def bind_response(self, *, epoch: int, turn_id: int, response_id: str) -> None:
        turn = self._turn(epoch, turn_id, create=True)
        if turn is not None and response_id not in turn.response_ids:
            turn.response_ids.append(response_id)

    def close_turns(self, *, epoch: int, up_to_turn_id: int) -> None:
        """Close every open turn of ``epoch`` up to ``up_to_turn_id`` (the model turn completed)."""
        for turn in self._turns.values():
            if turn.epoch == epoch and turn.turn_id <= up_to_turn_id and not turn.finished:
                self._close(turn, TURN_DONE if _has_audio(turn) else TURN_LISTEN)

    def cancel_epoch(self, epoch: int) -> None:
        """Barge-in: every open turn of the old epoch ends cancelled."""
        for turn in self._turns.values():
            if turn.epoch == epoch and not turn.finished:
                self._close(turn, TURN_CANCELLED)

    def close_all(self) -> None:
        """Session close: nothing more will run, so open turns end where they are."""
        for turn in self._turns.values():
            if not turn.finished:
                self._close(turn, TURN_CANCELLED)

    # ------------------------------------------------------------------ #
    # Query                                                              #
    # ------------------------------------------------------------------ #

    def snapshot(self, *, turn_ids: Sequence[int] | None = None, epoch: int | None = None) -> dict[str, object]:
        """The ``session.stats`` document (RFC #8325 §3.3), times relative to the session open."""
        origin = self.opened_at_ns
        return {
            "schema_version": SCHEMA_VERSION,
            "clock": {"source": "CLOCK_MONOTONIC", "unit": "ns", "origin": "session_opened"},
            "generated_at_ns": self._clock() - origin,
            "stages": [{"stage_id": stage_id, "name": name} for stage_id, name in self.stages],
            "turns": [self._turn_document(turn, origin) for turn in self._select(turn_ids, epoch)],
            "dropped_turns": self.dropped_turns,
            "dropped_spans": self.dropped_spans,
        }

    def clear_finished(self, *, turn_ids: Sequence[int] | None = None, epoch: int | None = None) -> None:
        for turn in list(self._select(turn_ids, epoch)):
            if turn.finished:
                self._drop(turn)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _select(self, turn_ids: Sequence[int] | None, epoch: int | None) -> list[TurnPerfRecord]:
        if turn_ids is None and epoch is None:
            return list(self._turns.values())
        wanted = None if turn_ids is None else {int(turn_id) for turn_id in turn_ids}
        return [
            turn
            for turn in self._turns.values()
            if (epoch is None or turn.epoch == epoch) and (wanted is None or turn.turn_id in wanted)
        ]

    def _turn(self, epoch: int, turn_id: int, *, create: bool) -> TurnPerfRecord | None:
        key = (int(epoch), int(turn_id))
        turn = self._turns.get(key)
        if turn is None and create:
            if len(self._turns) >= self.max_turns:
                self._evict()
            turn = TurnPerfRecord(epoch=key[0], turn_id=key[1])
            self._turns[key] = turn
        return turn

    def _chunk(self, epoch: int, seq: int) -> ChunkPerfRecord | None:
        turn_key = self._chunk_turns.get((int(epoch), int(seq)))
        turn = self._turns.get(turn_key) if turn_key is not None else None
        return turn.chunks.get(int(seq)) if turn is not None else None

    def _close(self, turn: TurnPerfRecord, status: str) -> None:
        turn.status = status
        for chunk in turn.chunks.values():
            chunk.complete = True

    def _evict(self) -> None:
        victim = next((turn for turn in self._turns.values() if turn.finished), None)
        if victim is None:
            victim = next(iter(self._turns.values()))
        self._drop(victim)
        self.dropped_turns += 1

    def _drop(self, turn: TurnPerfRecord) -> None:
        self._turns.pop(turn.key, None)
        for seq in turn.chunks:
            if self._chunk_turns.get((turn.epoch, seq)) == turn.key:
                del self._chunk_turns[(turn.epoch, seq)]

    @staticmethod
    def _turn_document(turn: TurnPerfRecord, origin: int) -> dict[str, object]:
        return {
            "epoch": turn.epoch,
            "turn_id": turn.turn_id,
            "response_ids": list(turn.response_ids),
            "status": turn.status,
            "chunks": [_chunk_document(chunk, origin) for chunk in turn.chunks.values()],
        }


def _chunk_document(chunk: ChunkPerfRecord, origin: int) -> dict[str, object]:
    done_ns = chunk.done_ns()
    return {
        "seq": chunk.seq,
        "turn_seq": chunk.turn_seq,
        "kind": chunk.kind,
        "submitted_ns": chunk.submitted_ns - origin,
        "done_ns": None if done_ns is None else done_ns - origin,
        "complete": chunk.complete,
        "truncated": chunk.truncated,
        "stage_work": [
            {"stage_id": stage_id, "spans": [[start - origin, duration] for start, duration in spans]}
            for stage_id, spans in sorted(chunk.stage_work.items())
        ],
        "audio": [
            {
                "response_id": piece.response_id,
                "emitted_ns": piece.emitted_ns - origin,
                "duration_ms": piece.duration_ms,
            }
            for piece in chunk.audio
        ],
    }


def with_perf_chunk(prompt: dict[str, object], chunk: tuple[int, int]) -> dict[str, object]:
    """Return ``prompt`` with ``duplex.perf_chunk`` set, so stage schedulers time its steps.

    Prompts without a duplex envelope are returned unchanged; their chunks get no spans.
    """
    buffer = prompt.get("model_intermediate_buffer")
    duplex = buffer.get("duplex") if isinstance(buffer, Mapping) else None
    if not isinstance(duplex, Mapping):
        return prompt
    return {
        **prompt,
        "model_intermediate_buffer": {**buffer, "duplex": {**duplex, PERF_CHUNK_KEY: [int(chunk[0]), int(chunk[1])]}},
    }


def _has_audio(turn: TurnPerfRecord) -> bool:
    return any(chunk.audio for chunk in turn.chunks.values())
