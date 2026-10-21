# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The per-session ledger behind ``session.stats`` (RFC #8325)."""

import pytest

from vllm_omni.engine.duplex.session.perf_ledger import SessionPerfLedger, with_perf_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MS = 1_000_000


class _Clock:
    def __init__(self, now: int = 1_000 * MS) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _ledger(**kwargs) -> tuple[SessionPerfLedger, _Clock]:
    clock = _Clock()
    ledger = SessionPerfLedger(stages=[(0, "thinker"), (1, "talker")], clock=clock, **kwargs)
    return ledger, clock


def _open(ledger: SessionPerfLedger, *, seq: int, turn_id: int = 0, epoch: int = 0, at: int = 0) -> None:
    ledger.open_chunk(
        epoch=epoch,
        seq=seq,
        turn_id=turn_id,
        turn_seq=seq,
        submitted_ns=ledger.opened_at_ns + at,
    )


def test_snapshot_is_session_relative_and_keeps_turn_and_chunk_structure() -> None:
    ledger, clock = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1, at=5 * MS)
    ledger.record_stage_work([(0, 0, 1, origin + 6 * MS, 4 * MS), (1, 0, 1, origin + 8 * MS, 10 * MS)])
    ledger.bind_response(epoch=0, turn_id=0, response_id="resp-1")
    clock.now = origin + 20 * MS
    ledger.record_audio(response_id="resp-1", duration_ms=480.0)
    clock.now = origin + 21 * MS
    ledger.mark_handled([(0, 1)])

    stats = ledger.snapshot()
    assert stats["schema_version"] == 1
    assert stats["stages"] == [{"stage_id": 0, "name": "thinker"}, {"stage_id": 1, "name": "talker"}]
    (turn,) = stats["turns"]
    assert (turn["status"], turn["response_ids"]) == ("open", ["resp-1"])
    (chunk,) = turn["chunks"]
    assert chunk["submitted_ns"] == 5 * MS
    assert chunk["stage_work"] == [
        {"stage_id": 0, "spans": [[6 * MS, 4 * MS]]},
        {"stage_id": 1, "spans": [[8 * MS, 10 * MS]]},
    ]
    assert turn["audio"] == [{"response_id": "resp-1", "emitted_ns": 20 * MS, "duration_ms": 480.0}]
    # Not final until the turn closes.
    assert (chunk["complete"], chunk["done_ns"]) == (False, None)

    ledger.close_turns(epoch=0, up_to_turn_id=0)
    (turn,) = ledger.snapshot()["turns"]
    assert turn["status"] == "done"
    assert turn["chunks"][0]["complete"] is True
    # done_ns covers the API-side handling after the last span.
    assert turn["chunks"][0]["done_ns"] == 21 * MS


def test_turn_without_audio_closes_as_listen_and_barge_in_cancels_the_epoch() -> None:
    ledger, _ = _ledger()
    _open(ledger, seq=1, turn_id=0)
    _open(ledger, seq=2, turn_id=1)
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    ledger.cancel_epoch(0)
    statuses = [(turn["turn_id"], turn["status"]) for turn in ledger.snapshot()["turns"]]
    assert statuses == [(0, "listen"), (1, "cancelled")]


def test_late_spans_of_an_old_epoch_land_on_that_epochs_chunk() -> None:
    ledger, _ = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1, epoch=0)
    ledger.cancel_epoch(0)
    _open(ledger, seq=1, epoch=1)
    ledger.record_stage_work([(1, 0, 1, origin + MS, MS), (0, 1, 1, origin + 3 * MS, MS)])
    by_epoch = {turn["epoch"]: turn["chunks"][0]["stage_work"] for turn in ledger.snapshot()["turns"]}
    assert by_epoch == {0: [{"stage_id": 1, "spans": [[MS, MS]]}], 1: [{"stage_id": 0, "spans": [[3 * MS, MS]]}]}


def test_spans_of_unknown_chunks_and_audio_of_unbound_responses_are_ignored() -> None:
    ledger, _ = _ledger()
    _open(ledger, seq=1)
    ledger.record_stage_work([(0, 0, 99, 0, MS)])
    ledger.record_audio(response_id="unbound", duration_ms=100.0)
    (turn,) = ledger.snapshot()["turns"]
    assert turn["chunks"][0]["stage_work"] == []
    assert turn["audio"] == []


def test_touching_spans_coalesce_across_flushes() -> None:
    ledger, _ = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1)
    ledger.record_stage_work([(0, 0, 1, origin, 10)])
    ledger.record_stage_work([(0, 0, 1, origin + 10, 5)])
    ledger.record_stage_work([(0, 0, 1, origin + 30, 5)])
    spans = ledger.snapshot()["turns"][0]["chunks"][0]["stage_work"][0]["spans"]
    assert spans == [[0, 15], [30, 5]]


def test_span_cap_marks_the_chunk_truncated() -> None:
    ledger, _ = _ledger(max_spans_per_chunk=2)
    origin = ledger.opened_at_ns
    _open(ledger, seq=1)
    ledger.record_stage_work([(0, 0, 1, origin + start, 1) for start in (0, 10, 20)])
    stats = ledger.snapshot()
    assert stats["dropped_spans"] == 1
    assert stats["turns"][0]["chunks"][0]["truncated"] is True


def test_retention_evicts_the_oldest_finished_turn_first() -> None:
    ledger, _ = _ledger(max_turns=2)
    _open(ledger, seq=1, turn_id=0)
    _open(ledger, seq=2, turn_id=1)
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    _open(ledger, seq=3, turn_id=2)
    stats = ledger.snapshot()
    assert [turn["turn_id"] for turn in stats["turns"]] == [1, 2]
    assert stats["dropped_turns"] == 1
    # The evicted turn's chunk is no longer addressable.
    ledger.record_stage_work([(0, 0, 1, 0, MS)])


def test_with_perf_chunk_copies_the_envelope_and_skips_prompts_without_one() -> None:
    duplex = {"epoch": 0, "seq": 3}
    prompt = {"prompt_token_ids": [1], "model_intermediate_buffer": {"duplex": duplex}}
    stamped = with_perf_chunk(prompt, (0, 3))
    assert stamped["model_intermediate_buffer"]["duplex"]["perf_chunk"] == [0, 3]
    assert "perf_chunk" not in duplex
    plain = {"prompt_token_ids": [1]}
    assert with_perf_chunk(plain, (0, 3)) is plain
