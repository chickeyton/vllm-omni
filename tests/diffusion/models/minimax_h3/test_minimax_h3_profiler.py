# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-mode pipeline profiler contract for MiniMax H3 (#6965).

Issue #6965 claimed ``MiniMaxH3Pipeline.diffuse`` reported ~0.3 s while the
denoise loop ran for minutes, suspecting the loop executed outside the wrapped
method. The request-mode path is ``forward -> self.diffuse -> denoise loop``,
and ``diffuse`` is wrapped on the instance, so the recorded value must cover
the whole loop. These tests pin that contract without a model: ``diffuse`` is
replaced by a stub that runs a timed fake loop, and ``forward`` runs the real
code so the wrapper, the per-seed accumulation and the per-request reset are
exercised exactly as in serving.
"""

import time
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_STEP_SLEEP_S = 0.02
_NUM_STEPS = 5


def _build_profiled_pipeline(*, num_outputs: int, loop_wall_times: list[float]):
    """Build a MiniMaxH3Pipeline whose stages are stubs, with the profiler on.

    ``diffuse`` sleeps for ``_NUM_STEPS`` fake denoise steps and appends the
    wall time of that loop to ``loop_wall_times`` so the test can compare the
    profiler's number against what the loop really took.
    """
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)

    def diffuse(**kwargs):
        start = time.perf_counter()
        for _ in range(_NUM_STEPS):
            time.sleep(_STEP_SLEEP_S)
        loop_wall_times.append(time.perf_counter() - start)
        return torch.zeros(1), torch.zeros(1)

    def decode(video_latent, audio_latent, *, height, width):
        return torch.zeros((1, 3, 2, height, width), dtype=torch.float32), torch.zeros((1, 8))

    context = {"num_outputs": num_outputs, "seed": 7, "height": 4, "width": 4, "preencode_mp4": False}
    pipeline.od_config = SimpleNamespace()
    pipeline._prepare_request_inputs = lambda *args, **kwargs: context
    pipeline._denoise_kwargs = lambda ctx: {}
    pipeline.diffuse = diffuse
    pipeline.decode = decode
    pipeline._offload_model_cpu_stage_output = lambda value: value

    pipeline.setup_diffusion_pipeline_profiler(enable_diffusion_pipeline_profiler=True)
    return pipeline


def _request():
    return SimpleNamespace(prompts=["a prompt"], sampling_params=SimpleNamespace())


def test_request_mode_diffuse_duration_covers_the_whole_denoise_loop():
    loop_wall_times: list[float] = []
    pipeline = _build_profiled_pipeline(num_outputs=1, loop_wall_times=loop_wall_times)

    forward_start = time.perf_counter()
    output = pipeline.forward(_request())
    forward_wall_time = time.perf_counter() - forward_start

    assert len(loop_wall_times) == 1
    diffuse_duration = output.stage_durations["MiniMaxH3Pipeline.diffuse"]
    # The profiler wraps diffuse() itself, so it cannot report less than the
    # loop that ran inside it, and cannot exceed the forward it is part of.
    assert diffuse_duration >= loop_wall_times[0]
    assert diffuse_duration <= forward_wall_time
    assert diffuse_duration >= _NUM_STEPS * _STEP_SLEEP_S
    assert "MiniMaxH3Pipeline.decode" in output.stage_durations


def test_request_mode_diffuse_duration_accumulates_every_output_seed():
    loop_wall_times: list[float] = []
    pipeline = _build_profiled_pipeline(num_outputs=2, loop_wall_times=loop_wall_times)

    output = pipeline.forward(_request())

    assert len(loop_wall_times) == 2
    assert output.stage_durations["MiniMaxH3Pipeline.diffuse"] >= sum(loop_wall_times)


def test_request_mode_stage_durations_reset_between_requests():
    loop_wall_times: list[float] = []
    pipeline = _build_profiled_pipeline(num_outputs=1, loop_wall_times=loop_wall_times)

    pipeline.forward(_request())
    second = pipeline.forward(_request())

    assert len(loop_wall_times) == 2
    # forward() clears the previous request's records, so the second report
    # covers the second loop alone rather than the running total. The upper
    # bound leaves room for wrapper overhead on a slow CI host while still
    # rejecting an accumulated value (which would be at least twice the loop).
    second_diffuse = second.stage_durations["MiniMaxH3Pipeline.diffuse"]
    assert second_diffuse >= loop_wall_times[1]
    assert second_diffuse < loop_wall_times[1] + _NUM_STEPS * _STEP_SLEEP_S / 2
