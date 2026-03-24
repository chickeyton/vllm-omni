"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- Qwen-Image-Edit: single image input
- Qwen-Image-Edit-2509: two image inputs
"""

import pytest
from pathlib import Path

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
    generate_synthetic_image,
)
from tests.utils import hardware_marks

EDIT_PROMPT = "Transform this modern, geometrist image into a Vincent van Gogh style impressionist painting."
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


# This test file targets two models, so I write a helper function.
# If a similar test only involves one model, one can just define a global list variable.
def _get_diffusion_feature_cases(model: str):
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cfg-parallel-size",
                    "2",
                    "--stage-configs-path",
                    path.join(base_dir, "vllm_omni", "model_executor", "stage_configs", "glm_image.yaml"),
                ],
            ),
            id="parallel_001",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("zai-org/GLM-Image"),
    indirect=True,
)
def test_glm_image(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test all diffusion features with GLM-Image in regular end-user scenarios."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    # CFG parallel is only activated when a negative prompt and true_cfg_scale > 1.0 are both present
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
