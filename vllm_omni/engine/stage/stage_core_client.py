"""Contract layer for the vLLM-Omni stage process/client model.

``StageCoreClientBase`` (``abc.ABC``) is the runtime base shared by every backend
(LLM, diffusion). It declares **only** the surface common to both backends;
LLM-specific methods (``get_outputs_async`` / ``get_outputs_nowait`` /
``process_core_inputs`` / ``get_kv_sender_info``) live on
``StageLLMCoreClientBase``, and diffusion-specific methods
(``get_diffusion_output_nowait``) live on ``StageDiffusionCoreClient``, so a
backend only implements what it actually supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.engine.stage.stage_core_types import StageCoreRequest
    from vllm_omni.engine.stage_init_utils import StageMetadata

from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.output_modality import FinalOutputModalityType
from vllm_omni.inputs.data import OmniSamplingParams


class StageCoreClientBase(ABC):
    """Runtime base shared by every stage client (LLM and diffusion).

    Declares only the backend-common contract. It intentionally does not declare
    any LLM-only method, nor anything whose name would shadow an
    ``AsyncMPClient`` / ``DPLBAsyncMPClient`` implementation in a concrete LLM
    client's MRO (see the OOP counter-design for the shadowing rule).
    """

    # ---- shared metadata (populated by __init__ from ``metadata``) ----
    stage_id: int
    # replica_id is late-bound: it stays unassigned (``None``) until
    # bind_replica_id() delivers the master's assignment. The base owns this
    # default so concrete clients never initialize it themselves.
    replica_id: int | None = None
    stage_type: str
    model_stage: str | None
    final_output: bool
    final_output_type: FinalOutputModalityType | None
    default_sampling_params: OmniSamplingParams
    prompt_expand_func: Callable | None
    requires_multimodal_data: bool
    custom_process_input_func: Callable | None
    engine_input_source: Sequence[int]
    is_comprehension: bool

    def __init__(self, *args: Any, metadata: StageMetadata | None = None, **kwargs: Any) -> None:
        """Populate the shared stage metadata, then continue cooperative init.

        ``metadata`` is consumed here to set the backend-common metadata fields;
        any remaining positional/keyword arguments are forwarded via
        ``super().__init__`` so a concrete LLM client can initialize its mixed-in
        transport base (``AsyncMPClient``) through the MRO. ``replica_id`` is
        deliberately not read from metadata — it is late-bound via
        :meth:`bind_replica_id`.
        """
        if metadata is not None:
            self.stage_id = metadata.stage_id
            self.stage_type = metadata.stage_type
            self.model_stage = metadata.model_stage
            self.final_output = metadata.final_output
            self.final_output_type = metadata.final_output_type
            self.default_sampling_params = metadata.default_sampling_params
            self.prompt_expand_func = metadata.prompt_expand_func
            self.requires_multimodal_data = getattr(metadata, "requires_multimodal_data", False)
            self.custom_process_input_func = getattr(metadata, "custom_process_input_func", None)
            self.engine_input_source = getattr(metadata, "engine_input_source", [])
            self.is_comprehension = getattr(metadata, "is_comprehension", False)
        super().__init__(*args, **kwargs)

    def check_health(self) -> None:
        """Raise ``EngineDeadError`` if the backing stage process is dead.

        Template method: each backend reports liveness (and may cache a detected
        death as a side effect) via :meth:`_engine_dead_reason`; the raise site is
        shared here.
        """
        reason = self._engine_dead_reason()
        if reason is not None:
            raise EngineDeadError(reason)

    @abstractmethod
    def _engine_dead_reason(self) -> str | None:
        """Return an error message if the backing engine/subprocess is dead.

        Returns ``None`` when healthy. Implementations may update internal
        dead-state (e.g. cache a detected death) as a side effect.
        """

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown."""

    @abstractmethod
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        "Abort requests."

    @abstractmethod
    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """RPC"""

    def bind_replica_id(self, replica_id: int) -> None:
        """Bind replica id"""
        self.replica_id = int(replica_id)

    @abstractmethod
    async def add_request_async(self, request: StageCoreRequest) -> None:
        """Add request. Backends accept their own concrete request type."""
