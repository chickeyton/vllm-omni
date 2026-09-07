# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex full-duplex integration.

PersonaPlex (``nvidia/personaplex-7b-v1``) is a Moshi finetune: a pure-lockstep
speech-to-speech model. This package plugs it into the engine-side duplex
session runtime through one ``DuplexModelPlugin`` (the ``duplex_plugin`` dotted
string in the model's ``pipeline.py``):

- :class:`PersonaPlexConfig`  immutable session config (voice / persona / sampling)
- :class:`PersonaPlexDuplexPlugin`  the ``DuplexModelPlugin`` implementation
  (engine append/sampling policy and session policy in one class)
- :class:`PersonaPlexStage0DuplexRuntime`  Stage 0 session state and prefill
- :class:`PersonaPlexPcmAppendBuffer`  PCM input framing
"""

from .config import (
    PersonaPlexConfig,
)
from .input import (
    PersonaPlexPcmAppendBuffer,
)
from .plugin import (
    PersonaPlexDuplexPlugin,
    PersonaPlexServingSessionState,
)
from .policy import PrefillStep
from .stage0 import (
    PersonaPlexStage0DuplexRuntime,
)

__all__ = [
    "PersonaPlexConfig",
    "PersonaPlexDuplexPlugin",
    "PersonaPlexPcmAppendBuffer",
    "PersonaPlexServingSessionState",
    "PersonaPlexStage0DuplexRuntime",
    "PrefillStep",
]
