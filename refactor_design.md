# vLLM-Omni Refactoring Design

## 1. Overview

### Goal
Refactor the vLLM-Omni multi-stage pipeline architecture to be more OOP, eliminate unnecessary wrapper layers, and align with vLLM's native engine architecture.

### Before (Current Architecture)
```
AsyncOmni (extends OmniBase)
  └─ has OmniStage (#stages)
       └─ OmniLLM (extends LLM) / AsyncOmniLLM (extends AsyncLLM)
            └─ LLMEngine
                 └─ EngineCoreClient
                      └─ (ipc / multiprocessing.Queue)
                           └─ EngineCoreProc (extends EngineCore)
```
**6 layers** from user to engine core. Uses multiprocessing.Queue + shared memory for IPC.

### After (Target Architecture)
```
AsyncOmni (extends EngineClient)
  └─ has PipelineOrchestrator
       └─ has StageCoreClient (extends EngineCoreClient) (#stages)
            └─ (zmq)
                 └─ StageCoreProc (extends EngineCore)
```
**4 layers** from user to engine core. Uses ZMQ for IPC (matching vLLM's native mechanism).

### Key Changes
1. **AsyncOmni**: `OmniBase` → `EngineClient` — Becomes a proper EngineClient implementation
2. **PipelineOrchestrator**: New class replacing OmniBase + OmniStage management logic
3. **StageCoreClient**: New class replacing OmniLLM/AsyncOmniLLM + LLMEngine layers
4. **StageCoreProc**: New class replacing `_stage_worker`/`_stage_worker_async` functions
5. **Eliminated**: OmniBase, OmniStage, OmniLLM, AsyncOmniLLM wrapper layers
6. **IPC**: multiprocessing.Queue + shared memory → ZMQ sockets

---

## 2. Class Design

### 2.1 StageCoreProc (extends EngineCore)

**File**: `vllm_omni/engine/stage_core_proc.py`

**Purpose**: Runs in a separate process per stage. Extends vLLM's EngineCore with stage-specific device setup, model loading, and omni-specific processing. Replaces the `_stage_worker` and `_stage_worker_async` functions.

```python
class StageCoreProc(EngineCoreProc):
    """Stage-specific EngineCore that runs in a background process.

    Each pipeline stage runs one StageCoreProc with:
    - Dedicated GPU(s) assigned via CUDA_VISIBLE_DEVICES
    - Stage-specific model loaded (e.g., thinker, talker)
    - ZMQ-based communication with StageCoreClient
    - Omni-specific connectors for KV cache transfer
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        # Stage-specific parameters
        stage_id: int,
        stage_type: str,  # "llm" or "diffusion"
        devices: str | None,
        connectors_config: dict | None,
        stage_init_timeout: int = 300,
        **kwargs,
    ):
        # 1. Set up stage-specific devices BEFORE parent init
        set_stage_devices(stage_id, devices)

        # 2. Initialize omni connectors for this stage
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.connectors = build_stage_connectors(stage_id, connectors_config)

        # 3. Call parent init (creates executor, scheduler, etc.)
        super().__init__(
            vllm_config=vllm_config,
            local_client=local_client,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=log_stats,
            **kwargs,
        )

    @staticmethod
    def run_stage_core(
        vllm_config: VllmConfig,
        stage_id: int,
        stage_type: str,
        devices: str | None,
        connectors_config: dict | None,
        stage_init_timeout: int,
        **kwargs,
    ):
        """Static entry point for launching StageCoreProc in a background process.

        Similar to EngineCoreProc.run_engine_core() but with stage-specific setup.
        """
        # Set up signal handlers, initialize StageCoreProc, run busy loop
        ...
```

**Key differences from EngineCoreProc**:
- Calls `set_stage_devices()` before initialization
- Initializes omni connectors for KV cache transfer
- Passes stage-specific configuration to the model executor
- Has a dedicated `run_stage_core()` static entry point

**For diffusion stages**: StageCoreProc handles diffusion by using a different executor class or by delegating to a DiffusionEngine internally. The PipelineOrchestrator passes the appropriate `executor_class` and `vllm_config` based on `stage_type`.

---

### 2.2 StageCoreClient (extends EngineCoreClient)

**File**: `vllm_omni/engine/stage_core_client.py`

**Purpose**: Client-side component for one stage. Extends vLLM's EngineCoreClient to communicate with StageCoreProc via ZMQ. Replaces OmniLLM/AsyncOmniLLM and LLMEngine layers. Each instance manages one stage's engine process.

```python
class StageCoreClient(EngineCoreClient):
    """Client for a single pipeline stage.

    Manages ZMQ communication with a StageCoreProc running in a
    background process. Each stage in the pipeline has one
    StageCoreClient instance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        # Stage-specific parameters
        stage_id: int,
        stage_type: str,
        devices: str | None,
        connectors_config: dict | None,
        stage_init_timeout: int = 300,
    ):
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.devices = devices
        self.connectors_config = connectors_config
        self.stage_init_timeout = stage_init_timeout

        # Create the async MP client internally (reuses vLLM's ZMQ machinery)
        # This launches StageCoreProc in a background process
        self._mp_client = self._create_mp_client(
            vllm_config, executor_class, log_stats
        )

    @classmethod
    def make_stage_client(
        cls,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        stage_id: int,
        stage_type: str,
        devices: str | None = None,
        connectors_config: dict | None = None,
        stage_init_timeout: int = 300,
    ) -> "StageCoreClient":
        """Factory method to create a StageCoreClient.

        Launches StageCoreProc in a background process and establishes
        ZMQ communication.
        """
        ...

    # --- EngineCoreClient interface ---
    async def async_add_request(self, request: EngineCoreRequest) -> None: ...
    async def async_get_output(self) -> EngineCoreOutputs: ...
    async def async_abort_requests(self, request_ids: list[str]) -> None: ...
    async def async_shutdown(self) -> None: ...
    async def async_profile(self, is_start: bool) -> None: ...

    # --- Stage-specific methods ---
    async def wait_for_ready(self, timeout: int = 300) -> dict:
        """Wait for the stage process to report readiness."""
        ...

    def get_vllm_config(self) -> VllmConfig:
        """Return the VllmConfig for this stage."""
        ...
```

**Implementation approach**: StageCoreClient internally delegates ZMQ communication to vLLM's existing `AsyncMPClient` implementation. The key customization is launching `StageCoreProc` instead of `EngineCoreProc` in the background process, passing stage-specific parameters (devices, connectors, stage_id).

**ZMQ socket pattern** (same as vLLM):
```
StageCoreClient                    StageCoreProc
  input_socket (DEALER) ──────> input_socket (ROUTER)
  output_socket (PULL)  <────── output_socket (PUSH)
```

---

### 2.3 PipelineOrchestrator

**File**: `vllm_omni/entrypoints/pipeline_orchestrator.py`

**Purpose**: Manages multiple StageCoreClients, orchestrates the multi-stage pipeline flow, handles stage-to-stage data transformation, connectors for KV transfer, and metrics. Replaces OmniBase's stage management and OmniStage's orchestration logic.

```python
class StageInfo:
    """Metadata about a single pipeline stage (replaces OmniStage for metadata)."""

    stage_id: int
    stage_type: str  # "llm" or "diffusion"
    model_stage: str  # "thinker", "talker", "diffusion", etc.
    engine_input_source: list[int]
    final_output: bool
    final_output_type: str | None
    default_sampling_params: SamplingParams | OmniDiffusionSamplingParams
    requires_multimodal_data: bool
    custom_process_input_func: Callable | None
    is_comprehension: bool
    # Runtime state
    engine_outputs: Any  # Last outputs from this stage


class PipelineOrchestrator:
    """Orchestrates a multi-stage pipeline of StageCoreClients.

    Manages the lifecycle of multiple stage engine processes,
    coordinates request flow through the pipeline, and handles
    inter-stage data transformation and KV cache transfer.
    """

    def __init__(
        self,
        model: str,
        stage_configs: list,
        config_path: str,
        log_stats: bool = False,
        stage_init_timeout: int = 300,
        init_timeout: int = 300,
    ):
        self.model = model
        self.stage_configs = stage_configs
        self.config_path = config_path
        self.log_stats = log_stats

        # Stage clients and metadata
        self.stage_clients: list[StageCoreClient] = []
        self.stage_infos: list[StageInfo] = []

        # Connectors for KV cache transfer
        self.connectors: dict[tuple[str, str], OmniConnectorBase] = {}

        # Pipeline state
        self.async_chunk: bool = False
        self.output_modalities: list[str] = []
        self.default_sampling_params_list: list = []

        # Initialize everything
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize all stage clients and connectors."""
        # 1. Parse stage configs into StageInfo objects
        # 2. Initialize connectors
        # 3. Create StageCoreClient for each stage
        # 4. Wait for all stages to report readiness
        ...

    def _create_stage_client(self, stage_info: StageInfo) -> StageCoreClient:
        """Create a StageCoreClient for one stage."""
        # Build vllm_config from stage engine_args
        # Determine executor class
        # Create and return StageCoreClient
        ...

    async def process_request(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: list[OmniSamplingParams],
        output_modalities: list[str] | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process a request through the multi-stage pipeline.

        Sends the request to stage 0, collects output, transforms it
        for the next stage, and repeats until the final stage.
        Yields intermediate and final OmniRequestOutput objects.
        """
        ...

    async def process_request_async_chunk(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: list[OmniSamplingParams],
        metrics: OrchestratorAggregator,
        final_stage_id: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process request with async chunk mode (parallel stages)."""
        ...

    async def process_request_sequential(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: list[OmniSamplingParams],
        metrics: OrchestratorAggregator,
        final_stage_id: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process request sequentially through stages."""
        ...

    def transform_stage_output(
        self,
        stage_id: int,
        output: EngineCoreOutputs,
        original_prompt: OmniPromptType,
    ) -> EngineCoreRequest:
        """Transform a stage's output into input for the next stage.

        Uses stage_info.custom_process_input_func or default logic
        to derive next-stage inputs from current-stage outputs.
        """
        ...

    async def abort(self, request_id: str) -> None:
        """Abort a request across all stages."""
        for client in self.stage_clients:
            await client.async_abort_requests([request_id])

    async def shutdown(self) -> None:
        """Shutdown all stage processes."""
        for client in self.stage_clients:
            await client.async_shutdown()

    def get_comprehension_stage(self) -> StageInfo | None:
        """Return the first comprehension stage (for tokenizer/config access)."""
        for info in self.stage_infos:
            if info.is_comprehension:
                return info
        return None

    @property
    def stage_list(self) -> list[StageInfo]:
        """Return stage metadata list (backward compat with serving layer)."""
        return self.stage_infos
```

**Key responsibilities**:
- Replaces OmniBase._initialize_stages(), _start_stages(), _wait_for_stages_ready()
- Replaces OmniStage.process_engine_inputs() → transform_stage_output()
- Replaces the output_handler loop from AsyncOmni
- Manages connector lifecycle for KV cache transfer
- Provides stage_list/default_sampling_params_list for backward compatibility

---

### 2.4 AsyncOmni (extends EngineClient)

**File**: `vllm_omni/entrypoints/async_omni.py` (modified)

**Purpose**: Top-level async interface for multi-stage pipeline inference. Now extends vLLM's `EngineClient` protocol instead of `OmniBase`. Delegates pipeline management to `PipelineOrchestrator`.

```python
class AsyncOmni(EngineClient):
    """Asynchronous unified entry point for multi-stage pipelines.

    Implements vLLM's EngineClient protocol, delegating multi-stage
    pipeline orchestration to PipelineOrchestrator.

    Args:
        model: Model name or path to load.
        **kwargs: Pipeline configuration arguments.
    """

    def __init__(self, model: str, **kwargs) -> None:
        model = omni_snapshot_download(model)

        # Load stage configs
        stage_configs_path = kwargs.get("stage_configs_path")
        if stage_configs_path is None:
            config_path = resolve_model_config_path(model)
            stage_configs = load_stage_configs_from_model(model)
        else:
            config_path = stage_configs_path
            stage_configs = load_stage_configs_from_yaml(stage_configs_path)

        # Create the pipeline orchestrator
        self.orchestrator = PipelineOrchestrator(
            model=model,
            stage_configs=stage_configs,
            config_path=config_path,
            log_stats=kwargs.get("log_stats", False),
            stage_init_timeout=kwargs.get("stage_init_timeout", 20),
            init_timeout=kwargs.get("init_timeout", 300),
        )

        # Initialize processors from comprehension stage
        comp_stage = self.orchestrator.get_comprehension_stage()
        if comp_stage and comp_stage.vllm_config:
            self._vllm_config = comp_stage.vllm_config
            self.model_config = comp_stage.vllm_config.model_config
            self.input_processor = OmniInputProcessor(vllm_config=self._vllm_config)
            io_plugin = self.model_config.io_processor_plugin
            self.io_processor = get_io_processor(self._vllm_config, io_plugin)
        else:
            self._vllm_config = None
            self.model_config = None
            self.input_processor = None
            self.io_processor = None

        # Pause/resume control
        self._pause_cond = asyncio.Condition()
        self._paused = False

        # Request tracking
        self.request_states: dict[str, ClientRequestState] = {}

    # ---- EngineClient abstract properties ----

    @property
    def is_running(self) -> bool:
        return len(self.orchestrator.stage_clients) > 0

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    # ---- EngineClient abstract methods ----

    async def generate(
        self,
        prompt: OmniPromptType,
        sampling_params: SamplingParams | None = None,
        request_id: str = "",
        *,
        sampling_params_list: list[OmniSamplingParams] | None = None,
        output_modalities: list[str] | None = None,
        **kwargs,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs through the multi-stage pipeline.

        Supports both EngineClient-style single sampling_params and
        omni-style sampling_params_list (one per stage).
        """
        # If single sampling_params provided, wrap for backward compat
        if sampling_params_list is None:
            if sampling_params is not None:
                sampling_params_list = [sampling_params] * len(self.orchestrator.stage_infos)
            else:
                sampling_params_list = self.orchestrator.default_sampling_params_list

        async for output in self.orchestrator.process_request(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=output_modalities,
        ):
            yield output

    async def encode(self, *args, **kwargs):
        raise NotImplementedError("encode() not supported for AsyncOmni")

    async def abort(self, request_id: str | Iterable[str]) -> None:
        if isinstance(request_id, str):
            await self.orchestrator.abort(request_id)
        else:
            for rid in request_id:
                await self.orchestrator.abort(rid)

    async def is_tracing_enabled(self) -> bool: ...
    async def do_log_stats(self) -> None: ...
    async def check_health(self) -> None: ...
    async def start_profile(self) -> None: ...
    async def stop_profile(self) -> None: ...
    async def reset_mm_cache(self) -> None: ...
    async def reset_encoder_cache(self) -> None: ...
    async def reset_prefix_cache(self, ...) -> bool: ...
    async def sleep(self, level: int = 1) -> None: ...
    async def wake_up(self, tags=None) -> None: ...
    async def is_sleeping(self) -> bool: ...
    async def add_lora(self, lora_request) -> bool: ...
    async def pause_generation(self, **kwargs) -> None: ...
    async def resume_generation(self) -> None: ...
    async def is_paused(self) -> bool: ...

    def shutdown(self) -> None:
        """Shutdown the pipeline."""
        asyncio.run(self.orchestrator.shutdown())

    # ---- Backward compatibility ----

    @property
    def stage_configs(self):
        return self.orchestrator.stage_configs

    @property
    def stage_list(self):
        return self.orchestrator.stage_list

    @property
    def default_sampling_params_list(self):
        return self.orchestrator.default_sampling_params_list

    @property
    def output_modalities(self):
        return self.orchestrator.output_modalities

    @property
    def renderer(self):
        return self.input_processor.renderer if self.input_processor else None

    async def get_vllm_config(self) -> VllmConfig | None:
        return self._vllm_config

    async def get_model_config(self):
        return self.model_config

    async def get_tokenizer(self):
        comp = self.orchestrator.get_comprehension_stage()
        return comp.tokenizer if comp else None
```

---

### 2.5 Omni (sync, extends OmniBase — kept for backward compatibility)

**File**: `vllm_omni/entrypoints/omni.py` (minimal changes)

The sync `Omni` class is kept as-is for offline/batch inference. It continues to use the existing `OmniBase` → `OmniStage` → `OmniLLM` path for synchronous generation. This is acceptable because:
1. The serving layer (OpenAI API) uses only `AsyncOmni`
2. The sync path is mainly for offline batch inference
3. Gradual migration: sync path can be refactored separately later

---

## 3. Data Flow

### 3.1 Request Processing (Sequential Mode)

```
User calls AsyncOmni.generate(prompt, request_id, sampling_params_list)
  │
  ▼
AsyncOmni delegates to PipelineOrchestrator.process_request()
  │
  ▼
PipelineOrchestrator prepares EngineCoreRequest for Stage 0
  │  (using OmniInputProcessor to convert prompt)
  ▼
StageCoreClient[0].async_add_request(request)
  │  (ZMQ DEALER → ROUTER)
  ▼
StageCoreProc[0] receives request, runs EngineCore.step()
  │  (model forward pass, token generation)
  ▼
StageCoreProc[0] sends output via ZMQ (PUSH → PULL)
  │
  ▼
StageCoreClient[0].async_get_output() receives EngineCoreOutputs
  │
  ▼
PipelineOrchestrator.transform_stage_output()
  │  (converts Stage 0 output → Stage 1 input)
  │  (may use connector for KV cache transfer)
  ▼
StageCoreClient[1].async_add_request(transformed_request)
  │  ... (repeat for each stage)
  ▼
PipelineOrchestrator yields OmniRequestOutput for final stages
  │
  ▼
AsyncOmni yields to user
```

### 3.2 Request Processing (Async Chunk Mode)

```
PipelineOrchestrator sends request to Stage 0
  │
  ├─ Stage 0 starts generating tokens
  │  On first output: derive placeholder input for Stage 1+
  │    ├─ Send placeholder to Stage 1
  │    ├─ Stage 1 starts (receives KV via connector)
  │    └─ Both stages run in parallel
  │
  ├─ Collect outputs from all stages concurrently
  │  via per-stage async queues
  │
  └─ Yield outputs as they arrive from any stage
```

---

## 4. File Changes

### New Files
| File | Content |
|------|---------|
| `vllm_omni/engine/stage_core_proc.py` | StageCoreProc class |
| `vllm_omni/engine/stage_core_client.py` | StageCoreClient class |
| `vllm_omni/entrypoints/pipeline_orchestrator.py` | PipelineOrchestrator + StageInfo classes |

### Modified Files
| File | Changes |
|------|---------|
| `vllm_omni/entrypoints/async_omni.py` | AsyncOmni extends EngineClient, uses PipelineOrchestrator |
| `vllm_omni/entrypoints/__init__.py` | Keep existing exports (AsyncOmni, Omni, AsyncOmniDiffusion) |
| `vllm_omni/__init__.py` | No changes needed |

### Preserved Files (backward compatibility)
| File | Reason |
|------|--------|
| `vllm_omni/entrypoints/omni.py` | Sync Omni path preserved |
| `vllm_omni/entrypoints/omni_stage.py` | Used by sync Omni path |
| `vllm_omni/entrypoints/omni_llm.py` | Used by sync Omni path |
| `vllm_omni/entrypoints/async_omni_llm.py` | No longer used by AsyncOmni (can deprecate later) |
| `vllm_omni/entrypoints/omni_diffusion.py` | Used for diffusion stages |
| `vllm_omni/entrypoints/async_omni_diffusion.py` | Used for diffusion stages |

### Untouched Files
| File | Reason |
|------|--------|
| `vllm_omni/entrypoints/openai/*.py` | Serving layer uses EngineClient protocol; no changes needed |
| `vllm_omni/engine/arg_utils.py` | Engine args unchanged |
| `vllm_omni/engine/input_processor.py` | Input processor unchanged |
| `vllm_omni/engine/output_processor.py` | Output processor unchanged |
| `vllm_omni/engine/__init__.py` | Core data structures unchanged |
| `vllm_omni/inputs/data.py` | Data types unchanged |
| `vllm_omni/outputs.py` | Output types unchanged |

---

## 5. Serving Layer Compatibility

The serving layer (`api_server.py`, `serving_chat.py`, `serving_video.py`) accesses AsyncOmni through these interfaces:

| Interface | Current | After Refactor |
|-----------|---------|----------------|
| `engine_client.generate(prompt, request_id, sampling_params_list)` | Direct method | Same signature via EngineClient |
| `engine_client.stage_configs` | From OmniBase | Property delegating to orchestrator |
| `engine_client.stage_list` | List[OmniStage] | List[StageInfo] (compatible duck-typing) |
| `engine_client.default_sampling_params_list` | From OmniBase | Property delegating to orchestrator |
| `engine_client.input_processor` | Set in _wait_for_stages_ready | Set in __init__ |
| `engine_client.io_processor` | Set in _wait_for_stages_ready | Set in __init__ |
| `engine_client.model_config` | Set in _wait_for_stages_ready | Set in __init__ |
| `engine_client.renderer` | From input_processor | Property |
| `await engine_client.get_vllm_config()` | Async method | Same |
| `await engine_client.get_tokenizer()` | Async method | Same |
| `engine_client.shutdown()` | Direct method | Same |

**StageInfo backward compatibility**: The serving layer accesses `stage.stage_type`, `stage.final_output`, `stage.final_output_type`, `stage.is_comprehension`. The `StageInfo` dataclass provides all these attributes, making it duck-type compatible with `OmniStage`.

---

## 6. Diffusion Stage Handling

Diffusion stages (stage_type="diffusion") don't use vLLM's EngineCore because diffusion models have a fundamentally different execution model. The approach:

1. **PipelineOrchestrator** detects `stage_type="diffusion"` from stage config
2. For diffusion stages, it creates a **DiffusionStageClient** instead of StageCoreClient
3. DiffusionStageClient wraps AsyncOmniDiffusion (existing class, kept as-is)
4. Communication uses the existing threading approach (ThreadPoolExecutor for sync diffusion engine)

This keeps diffusion handling clean while refactoring LLM stages.

```python
# In PipelineOrchestrator._create_stage_client():
if stage_info.stage_type == "diffusion":
    return DiffusionStageClient(model, stage_info, engine_args)
else:
    return StageCoreClient.make_stage_client(vllm_config, executor_class, ...)
```

---

## 7. Implementation Order

1. **StageCoreProc** — Foundation: the per-stage engine process
2. **StageCoreClient** — Communication layer to StageCoreProc
3. **PipelineOrchestrator** — Pipeline management using StageCoreClients
4. **AsyncOmni** — Refactor to extend EngineClient, use PipelineOrchestrator
5. **Updates** — Update Omni sync class and all imports

Each step builds on the previous one and can be tested incrementally.

---

## 8. Key Design Decisions

### 8.1 Why extend EngineCoreClient/EngineCore directly?
- Eliminates 2 abstraction layers (LLM, LLMEngine)
- Reuses vLLM's proven ZMQ IPC mechanism
- Allows stages to leverage vLLM's scheduler, cache management, etc.
- Makes each stage a first-class vLLM engine

### 8.2 Why PipelineOrchestrator as a separate class?
- Separation of concerns: AsyncOmni handles EngineClient protocol, orchestrator handles pipeline logic
- Makes pipeline logic testable independently
- Keeps AsyncOmni clean and focused on the EngineClient interface

### 8.3 Why keep OmniBase/OmniStage/OmniLLM?
- Sync Omni path still uses them for backward compatibility
- Gradual migration reduces risk
- They can be deprecated and removed in a follow-up PR

### 8.4 Why ZMQ over multiprocessing.Queue?
- Matches vLLM's native IPC mechanism
- Better performance (zero-copy, non-blocking)
- More reliable than shared memory hacks
- Supports the same serialization format (msgpack)
