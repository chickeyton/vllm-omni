# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duplex session stats: per-step spans recorded in a stage process (RFC #8325)."""

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.duplex_stage_work import (
    DuplexStageWorkRecorder,
    parse_perf_chunk,
    perf_chunk_from_buffer,
    request_perf_chunk,
    set_request_perf_chunk,
    stage_work_rows,
)
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def _request(request_id: str, chunk: list[int] | None) -> SimpleNamespace:
    duplex = {"epoch": 0, "seq": 9} if chunk is None else {"perf_chunk": chunk}
    return SimpleNamespace(request_id=request_id, model_intermediate_buffer={"duplex": duplex})


def _step(recorder, clock, requests, *, begin: int, end: int, emit: tuple[str, ...]):
    scheduler_output = SimpleNamespace(num_scheduled_tokens={request_id: 1 for request_id in requests})
    clock.now = begin
    recorder.step_begin(scheduler_output, requests)
    clock.now = end
    outputs = {
        0: OmniEngineCoreOutputs(outputs=[OmniEngineCoreOutput(request_id=rid, new_token_ids=[]) for rid in emit])
    }
    recorder.step_end(scheduler_output, outputs)
    return {output.request_id: output for output in outputs[0].outputs}


def test_perf_chunk_is_read_from_the_duplex_envelope_and_validated() -> None:
    assert parse_perf_chunk([3, 7]) == (3, 7)
    assert parse_perf_chunk((3, 7)) == (3, 7)
    for invalid in (None, [1], [1, 2, 3], [True, 1], ["a", 1], "37"):
        assert parse_perf_chunk(invalid) is None
    assert perf_chunk_from_buffer({"duplex": {"perf_chunk": [0, 4]}}) == (0, 4)
    # The stage-0 envelope always has epoch/seq; without perf_chunk the request is not measured.
    assert perf_chunk_from_buffer({"duplex": {"epoch": 0, "seq": 4}}) is None
    assert perf_chunk_from_buffer(None) is None


def test_an_update_or_payload_retags_the_request_and_none_keeps_the_previous_chunk() -> None:
    request = _request("r", [0, 1])
    assert request_perf_chunk(request) == (0, 1)
    set_request_perf_chunk(request, (0, 2))
    assert request_perf_chunk(request) == (0, 2)
    set_request_perf_chunk(request, None)
    assert request_perf_chunk(request) == (0, 2)


def test_each_step_span_is_attached_to_that_steps_output_with_its_chunk() -> None:
    clock = _Clock()
    recorder = DuplexStageWorkRecorder(clock=clock)
    measured = _request("measured", [0, 1])
    other = _request("other", None)
    requests = {"measured": measured, "other": other}

    outputs = _step(recorder, clock, requests, begin=100, end=130, emit=("measured", "other"))

    assert outputs["measured"].stage_work == [[0, 1, 100, 30]]
    assert outputs["other"].stage_work is None
    # Shipped spans are not sent twice.
    assert _step(recorder, clock, requests, begin=200, end=210, emit=("measured",))["measured"].stage_work == [
        [0, 1, 200, 10]
    ]


def test_steps_without_an_output_are_held_and_coalesced_until_the_next_output() -> None:
    clock = _Clock()
    recorder = DuplexStageWorkRecorder(clock=clock)
    request = _request("r", [0, 1])
    requests = {"r": request}

    # Chunked prefill: no output for two touching steps.
    _step(recorder, clock, requests, begin=0, end=10, emit=())
    _step(recorder, clock, requests, begin=10, end=25, emit=())
    set_request_perf_chunk(request, (0, 2))
    outputs = _step(recorder, clock, requests, begin=40, end=50, emit=("r",))

    # The producing step's chunk comes last.
    assert outputs["r"].stage_work == [[0, 1, 0, 25], [0, 2, 40, 10]]
    assert stage_work_rows(outputs["r"].stage_work) == [(0, 1, 0, 25), (0, 2, 40, 10)]


def test_async_scheduling_attributes_each_update_to_the_step_it_scheduled() -> None:
    clock = _Clock()
    recorder = DuplexStageWorkRecorder(clock=clock)
    request = _request("r", [0, 1])
    requests = {"r": request}
    first = SimpleNamespace(num_scheduled_tokens={"r": 1})
    second = SimpleNamespace(num_scheduled_tokens={"r": 1})

    clock.now = 0
    recorder.step_begin(first, requests)
    set_request_perf_chunk(request, (0, 2))
    clock.now = 5
    recorder.step_begin(second, requests)
    clock.now = 20
    out_first = {0: OmniEngineCoreOutputs(outputs=[OmniEngineCoreOutput(request_id="r", new_token_ids=[])])}
    recorder.step_end(first, out_first)
    clock.now = 30
    out_second = {0: OmniEngineCoreOutputs(outputs=[OmniEngineCoreOutput(request_id="r", new_token_ids=[])])}
    recorder.step_end(second, out_second)

    assert out_first[0].outputs[0].stage_work == [[0, 1, 0, 20]]
    assert out_second[0].outputs[0].stage_work == [[0, 2, 5, 25]]


def test_stage_work_rows_drop_malformed_rows() -> None:
    assert stage_work_rows([[0, 1, 2, 3], [0, 1], "x", [0, "a", 2, 3]]) == [(0, 1, 2, 3)]
    assert stage_work_rows(None) == []
