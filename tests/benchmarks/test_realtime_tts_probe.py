# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The Seed-TTS Realtime probe queues for a duplex session slot instead of failing."""

from __future__ import annotations

import pytest

from vllm_omni.benchmarks.patch import patch as bench_patch
from vllm_omni.clients import duplex as duplex_client

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _AdmissionGate:
    """Stands in for ``DuplexClient``: refuses admission ``refusals`` times, then admits."""

    instances: list[_AdmissionGate] = []
    refusals = 0
    refusal_code = "resource_exhausted"

    def __init__(self, url: str, **kwargs: object) -> None:
        del url, kwargs
        self.entered = False
        type(self).instances.append(self)

    async def __aenter__(self) -> _AdmissionGate:
        cls = type(self)
        if cls.refusals > 0:
            cls.refusals -= 1
            raise duplex_client.DuplexProtocolError("duplex_session_capacity_exhausted: limit=4", code=cls.refusal_code)
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    async def events(self):  # the collector's consume loop ends at once
        return
        yield  # pragma: no cover


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> type[_AdmissionGate]:
    _AdmissionGate.instances = []
    _AdmissionGate.refusals = 0
    _AdmissionGate.refusal_code = "resource_exhausted"
    monkeypatch.setattr(duplex_client, "DuplexClient", _AdmissionGate)
    return _AdmissionGate


async def _configure(probe: bench_patch._RealtimeTTSProbe) -> None:
    try:
        await probe.configure("test-model", auto_response=True)
    finally:
        await probe.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_configure_waits_for_a_free_session_slot(gate) -> None:
    """Two refusals, then a slot: the request is served, not counted as failed."""
    gate.refusals = 2
    probe = bench_patch._RealtimeTTSProbe("ws://test/v1/realtime?duplex=1")
    await _configure(probe)
    assert len(gate.instances) == 3
    assert gate.instances[-1].entered is True
    assert probe._client is gate.instances[-1]


@pytest.mark.asyncio
async def test_configure_raises_other_rejections_at_once(gate) -> None:
    gate.refusals = 1
    gate.refusal_code = "unsupported_turn_detection"
    probe = bench_patch._RealtimeTTSProbe("ws://test/v1/realtime?duplex=1")
    with pytest.raises(duplex_client.DuplexProtocolError):
        await _configure(probe)
    assert len(gate.instances) == 1


@pytest.mark.asyncio
async def test_configure_gives_up_when_the_slot_wait_budget_is_spent(gate, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_patch._RealtimeTTSProbe, "_SESSION_SLOT_WAIT_S", 0.0)
    gate.refusals = 5
    probe = bench_patch._RealtimeTTSProbe("ws://test/v1/realtime?duplex=1")
    with pytest.raises(duplex_client.DuplexProtocolError) as excinfo:
        await _configure(probe)
    assert excinfo.value.code == "resource_exhausted"
    assert len(gate.instances) == 1
