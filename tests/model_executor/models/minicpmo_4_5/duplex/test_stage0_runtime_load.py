# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 Stage-0 duplex runtime is built at load time, not in the first session."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(model_stage: str, session_mode: str) -> MiniCPMO45OmniForConditionalGeneration:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = model_stage
    model.thinker = None
    model.talker = None
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(session_mode=session_mode))
    return model


@pytest.mark.parametrize(
    ("model_stage", "session_mode", "expected_builds"),
    [
        ("llm", "duplex", 1),
        ("llm", "turn", 0),
        ("tts", "duplex", 0),
    ],
)
def test_load_weights_builds_duplex_runtime_only_for_duplex_thinker(
    monkeypatch: pytest.MonkeyPatch,
    model_stage: str,
    session_mode: str,
    expected_builds: int,
) -> None:
    model = _model(model_stage, session_mode)
    builds: list[bool] = []
    monkeypatch.setattr(model, "_duplex_data_plane_helper", lambda: builds.append(True))

    model.load_weights([])

    assert len(builds) == expected_builds
