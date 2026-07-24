from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.engine.stage.stage_core_types import StageDiffusionCoreOutput, StageDiffusionCoreRequest
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_engine():
    with patch("vllm_omni.diffusion.inline_stage_diffusion_client.DiffusionEngine") as mock:
        engine_instance = MagicMock()
        engine_instance.executor = SimpleNamespace(
            register_failure_callback=MagicMock(),
            check_health=MagicMock(),
        )
        mock.make_engine.return_value = engine_instance
        yield engine_instance


@pytest.fixture
def client(mock_engine):
    metadata = StageMetadata(
        stage_id=0,
        stage_type="diffusion",
        engine_output_type="image",
        is_comprehension=False,
        requires_multimodal_data=False,
        engine_input_source="prompt",
        final_output=True,
        final_output_type="image",
        default_sampling_params={},
        custom_process_input_func=None,
        model_stage=None,
        runtime_cfg=None,
    )
    with patch.object(InlineStageDiffusionClient, "_enrich_config"):
        od_config = MagicMock(spec=OmniDiffusionConfig)
        od_config.streaming_output = False
        c = InlineStageDiffusionClient(model="test_model", od_config=od_config, metadata=metadata, batch_size=1)
        yield c
        c.shutdown()


@pytest.mark.asyncio
async def test_inline_dispatch_request_success(client, mock_engine):
    mock_result = OmniRequestOutput.from_diffusion(request_id="req-1", images=[MagicMock()])

    async def _step_streaming(_request):
        yield [mock_result]

    mock_engine.step_streaming = _step_streaming

    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-1", prompt="A test prompt", sampling_params={})
    )

    # Wait for the task to be processed
    batch = None
    for _ in range(10):
        batch = client.get_outputs_nowait()
        if batch is not None:
            break
        await asyncio.sleep(0.01)

    assert batch is not None
    assert len(batch.outputs) == 1
    core_output = batch.outputs[0]
    assert core_output.request_id == "req-1"
    assert core_output.output is mock_result


@pytest.mark.asyncio
async def test_inline_dispatch_request_streaming_success(client, mock_engine):
    """Inline StageDiffusionClient correctly receives streaming chunk output from Diffusion Engine"""
    chunks = [
        OmniRequestOutput.from_diffusion(request_id="req-stream", images=[], finished=False),
        OmniRequestOutput.from_diffusion(request_id="req-stream", images=[], finished=True),
    ]

    async def _step_streaming(_request):
        for chunk in chunks:
            yield [chunk]

    client.od_config.streaming_output = True
    mock_engine.step_streaming = _step_streaming

    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-stream", prompt="A test prompt", sampling_params={})
    )

    collected: list[StageDiffusionCoreOutput] = []
    for _ in range(20):
        batch = client.get_outputs_nowait()
        if batch is not None:
            collected.extend(batch.outputs)
            if any(o.finished for o in batch.outputs):
                break
        await asyncio.sleep(0.01)

    assert [o.output for o in collected] == chunks


@pytest.mark.asyncio
async def test_inline_dispatch_request_error(client, mock_engine):
    async def _step_streaming(_request):
        raise RuntimeError("Engine failure")
        yield  # pragma: no cover

    mock_engine.step_streaming = _step_streaming

    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-err", prompt="A test prompt", sampling_params={})
    )

    batch = None
    for _ in range(10):
        batch = client.get_outputs_nowait()
        if batch is not None:
            break
        await asyncio.sleep(0.01)

    assert batch is not None
    core_output = batch.outputs[0]
    assert core_output.request_id == "req-err"
    assert core_output.error == "Engine failure"
    assert core_output.output is None


def test_inline_shutdown(client, mock_engine):
    assert not client._shutting_down

    # Shutting down should cleanly cancel anything queued and close engine
    client.shutdown()

    assert client._shutting_down
    mock_engine.close.assert_called_once()


def test_inline_registers_executor_failure_callback(client, mock_engine):
    mock_engine.executor.register_failure_callback.assert_called_once()


def test_inline_executor_failure_marks_engine_dead(client, mock_engine):
    callback = mock_engine.executor.register_failure_callback.call_args.args[0]

    callback()

    assert client._engine_dead is True
    with pytest.raises(EngineDeadError, match="inline diffusion engine is dead"):
        client.get_outputs_nowait()


def test_inline_returns_queued_output_before_engine_dead(client, mock_engine):
    callback = mock_engine.executor.register_failure_callback.call_args.args[0]
    core_output = StageDiffusionCoreOutput(
        request_id="req-queued",
        finished=True,
        output=OmniRequestOutput.from_diffusion(request_id="req-queued", images=[]),
    )
    client._output_queue.put_nowait(core_output)

    callback()

    batch = client.get_outputs_nowait()
    assert batch is not None
    assert batch.outputs == [core_output]
    with pytest.raises(EngineDeadError, match="inline diffusion engine is dead"):
        client.get_outputs_nowait()


def test_inline_check_health_marks_engine_dead(client, mock_engine):
    mock_engine.executor.check_health.side_effect = EngineDeadError()

    with pytest.raises(EngineDeadError):
        client.check_health()

    assert client._engine_dead is True


def test_inline_client_requires_replica_id(mock_engine):
    metadata = SimpleNamespace(
        stage_id=0,
        final_output=True,
        final_output_type="image",
        default_sampling_params={},
        requires_multimodal_data=False,
        custom_process_input_func=None,
        engine_input_source=[],
    )
    with patch.object(InlineStageDiffusionClient, "_enrich_config"):
        od_config = MagicMock(spec=OmniDiffusionConfig)
        with pytest.raises(AttributeError, match="replica_id"):
            InlineStageDiffusionClient(
                model="test_model",
                od_config=od_config,
                metadata=metadata,
                batch_size=1,
            )
