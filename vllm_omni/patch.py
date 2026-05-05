import sys
from functools import cached_property

from aenum import extend_enum
from vllm.config import ModelConfig as _OriginalModelConfig
from vllm.inputs import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest
from vllm.v1.request import RequestStatus
from vllm.v1.request import StreamingUpdate as _OriginalStreamingUpdate

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.rotary_embedding import OmniMRotaryEmbedding
from vllm_omni.request import OmniRequest, OmniStreamingUpdate

# =============================================================================
# Patch ModelConfig.is_mm_prefix_lm to support omni-specific models
# =============================================================================
# WHY: HunyuanImage-3.0 requires bidirectional attention for image tokens
# (cond_token_attn_type: "joint_full" in config.json). vLLM gates this on
# is_mm_prefix_lm, which checks an internal MM_PREFIX_LM_MODELS list that
# does not include "hunyuan_image_3_moe" (the upstream HF model_type).
#
# WHY NOT model-level: is_mm_prefix_lm is checked in vLLM core (scheduler,
# attention backend selection) before model code runs — no model-level hook.
#
# SCOPE: Only affects model_type in _OMNI_MM_PREFIX_LM_MODELS (currently
# just "hunyuan_image_3_moe"). All other models fall through to the
# original vLLM implementation unchanged.
#
# FRAGILITY: Relies on is_mm_prefix_lm being a cached_property on
# ModelConfig. The __dict__ access + __set_name__ dance works around a
# pydantic dataclass issue in vllm 0.19.0+. If vLLM changes
# is_mm_prefix_lm to a regular method or removes it, this will break.
#
# TODO: Upstream a configurable MM_PREFIX_LM_MODELS or a model_config flag
# so this patch can be removed.
_OMNI_MM_PREFIX_LM_MODELS = ("hunyuan_image_3_moe",)
# Access via __dict__ to avoid triggering cached_property.__get__ which fails
# with "Cannot use cached_property instance without calling __set_name__" in
# pydantic dataclasses (vllm 0.19.0+).
_cp = _OriginalModelConfig.__dict__["is_mm_prefix_lm"]
_original_is_mm_prefix_lm = _cp.func if hasattr(_cp, "func") else _cp.fget


def _patched_is_mm_prefix_lm(self):
    if _original_is_mm_prefix_lm(self):
        return True
    model_type = getattr(self.hf_config, "model_type", "")
    return model_type in _OMNI_MM_PREFIX_LM_MODELS


_patched_cp = cached_property(_patched_is_mm_prefix_lm)
_patched_cp.__set_name__(_OriginalModelConfig, "is_mm_prefix_lm")
_OriginalModelConfig.is_mm_prefix_lm = _patched_cp

# Sanity check: verify the patch is active. If vLLM changes the descriptor
# type or __set_name__ semantics, this will fail loudly at import time
# rather than silently falling back to unpatched behavior.
_installed = _OriginalModelConfig.__dict__.get("is_mm_prefix_lm")
assert _installed is _patched_cp, (
    "is_mm_prefix_lm patch failed to install — bidirectional attention "
    "for HunyuanImage3 will not work. Check vLLM ModelConfig changes."
)

# =============================================================================
# Patch GlmImageTextConfig to expose mrope_section in rope_parameters
# =============================================================================
# GLM-Image uses M-RoPE with mrope_section: [8, 12, 12], but transformers'
# implementation doesn't expose it in rope_parameters. vLLM's uses_mrope
# detection relies on "mrope_section" being present in rope_parameters.
# This patch ensures proper M-RoPE detection for GLM-Image.
try:
    from transformers.models.glm_image.configuration_glm_image import GlmImageTextConfig

    _original_glm_image_text_config_init = GlmImageTextConfig.__init__

    def _patched_glm_image_text_config_init(self, *args, **kwargs):
        _original_glm_image_text_config_init(self, *args, **kwargs)
        # Ensure rope_parameters exists and contains mrope_section
        if self.rope_parameters is None:
            self.rope_parameters = {}
        if isinstance(self.rope_parameters, dict) and "mrope_section" not in self.rope_parameters:
            # GLM-Image uses mrope_section: [8, 12, 12] for T/H/W dimensions
            self.rope_parameters["mrope_section"] = [8, 12, 12]

    GlmImageTextConfig.__init__ = _patched_glm_image_text_config_init
except ImportError:
    # GlmImageTextConfig not available, skip patching
    pass

# =============================================================================
# Patch Qwen3-VL / Qwen3-Omni-Thinker deepstack checks for CUDA-graph padding
# =============================================================================
# WHY: vllm's gpu_model_runner._preprocess calls embed_input_ids on the
# unpadded scheduled-token length, then runs forward on the CUDA-graph padded
# length. embed_input_ids → _set_deepstack_input_embeds stores
# `deepstack_input_embeds_num_tokens = unpadded`, but forward calls
# _get_deepstack_input_embeds(padded). The original strict check
# `num_tokens > self.deepstack_input_embeds_num_tokens` raises ValueError
# whenever scheduled tokens don't already sit on a CUDA-graph bucket boundary
# (e.g. judge model with 309 scheduled tokens padded to 320). The buffer
# itself (allocated to max_num_batched_tokens) has plenty of capacity; the
# guard is just over-strict. Tail rows are zero-initialised and the per-
# position add of zero is a no-op, so dropping the guard is safe.
#
# WHY NOT model-level: judge processes (e.g. Qwen3-VL-30B-A3B-Instruct-AWQ
# in GEdit-Bench) launch via `vllm_omni.entrypoints.cli.main` without the
# `--omni` flag and therefore never resolve through OmniModelRegistry, so
# a vllm-omni-side subclass + registry override would not be reachable.
#
# SCOPE: Replaces _get_deepstack_input_embeds and _clear_deepstack_input_embeds
# on:
#   - Qwen3VLForConditionalGeneration (Qwen3VLMoeForConditionalGeneration
#     inherits without overriding, so it picks up the patch automatically)
#   - Qwen3OmniMoeThinkerForConditionalGeneration (independent class, ships
#     its own duplicate copies of these methods upstream)
# vllm-omni's own qwen3_omni_moe_thinker.py override (used on the --omni
# path) already implements the fix; this patch covers the upstream-CLI path.
#
# FRAGILITY: If upstream rewrites these methods (e.g. dynamic buffer
# allocation) or renames the class, this patch becomes either redundant or
# inert. The replacement is semantically equivalent to upstream PR #40932,
# so re-applying on an already-fixed vllm is harmless.
#
# TODO: Remove once vllm pin is past commit 22631f80a (PR #40932, v0.20.2rc0).
# https://github.com/vllm-project/vllm/pull/40932
try:
    from vllm.sequence import IntermediateTensors as _IntermediateTensors

    def _patched_get_deepstack_input_embeds(self, num_tokens):
        if not getattr(self, "deepstack_input_embeds", None):
            return None
        if getattr(self, "deepstack_input_embeds_num_tokens", 0) == 0:
            return None
        return _IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][:num_tokens]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _patched_clear_deepstack_input_embeds(self, num_tokens):
        if not getattr(self, "deepstack_input_embeds", None):
            return
        if getattr(self, "deepstack_input_embeds_num_tokens", 0) == 0:
            return
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()
            self.deepstack_input_embeds_num_tokens = 0

    for _module_path, _class_name in (
        ("vllm.model_executor.models.qwen3_vl", "Qwen3VLForConditionalGeneration"),
        (
            "vllm.model_executor.models.qwen3_omni_moe_thinker",
            "Qwen3OmniMoeThinkerForConditionalGeneration",
        ),
    ):
        try:
            _mod = __import__(_module_path, fromlist=[_class_name])
            _cls = getattr(_mod, _class_name)
        except (ImportError, AttributeError):
            # Class not available in this vllm version — skip.
            continue
        _cls._get_deepstack_input_embeds = _patched_get_deepstack_input_embeds
        _cls._clear_deepstack_input_embeds = _patched_clear_deepstack_input_embeds
except ImportError:
    # vllm.sequence.IntermediateTensors missing — skip patching entirely.
    pass

# Extend RequestStatus enum with omni-specific statuses
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    # The value - 1 is intentionally chosen to ensure it is treated
    # as a non-finished state and remains compatible with existing comparisons.
    extend_enum(RequestStatus, "WAITING_FOR_CHUNK", -1)

# Snapshot sys.modules: `hasattr` below can trigger lazy submodule imports
# (e.g. transformers' `_LazyModule.__getattr__`), which mutate sys.modules
# during iteration and raise `dictionary changed size during iteration`.
for module_name, module in list(sys.modules.items()):
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = OmniMRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "StreamingUpdate") and module.StreamingUpdate == _OriginalStreamingUpdate:
        module.StreamingUpdate = OmniStreamingUpdate
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest
