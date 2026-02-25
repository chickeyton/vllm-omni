# vLLM-Omni Architecture Refactor Design

## 1. Overview

This document describes the architectural refactor of vLLM-Omni from its current layered architecture to a flattened, more maintainable design that better aligns with vLLM's native abstractions.

### 1.1 Current Architecture (Before)

```
AsyncOmni ──extends──> OmniBase
    │
    ├── #stages (1..N)
    │       ▼
    │   OmniStage ──has-a──> OmniLLM / AsyncOmniLLM ──extends──> LLM / AsyncLLM
    │                              │
    │                              ├── LLMEngine (sync path)
    │                              └── EngineCoreClient (async path)
    │                                      │ ipc (mp.Queue)
    │                                      ▼
    │                              EngineCoreProc ──contains──> EngineCore
    │
    └── connectors: OmniConnectorBase (shared-memory / RDMA)
```

**Key characteristics:**
- `AsyncOmni` extends custom `OmniBase` (not a vLLM class)
- Each `OmniStage` manages its own worker process with its own event loop
- Inside each worker, `OmniLLM` (extends `LLM`) or `AsyncOmniLLM` (extends `AsyncLLM`) creates an entire vLLM engine stack: `LLMEngine` or `EngineCoreClient` → `EngineCoreProc` → `EngineCore`
- Communication between orchestrator and stage workers uses `multiprocessing.Queue`
- Communication between `EngineCoreClient` and `EngineCoreProc` uses `multiprocessing.Queue` (ipc)
- **Result: 3 layers of indirection** (AsyncOmni → OmniStage → OmniLLM/AsyncOmniLLM → LLMEngine/EngineCoreClient)

### 1.2 Target Architecture (After)

```
AsyncOmni ──extends──> EngineClient (vllm protocol)
    │
    ├── has-a (1)
    │       ▼
    │   PipelineOrchestrator
    │       │
    │       ├── #stages (1..N)
    │       │       ▼
    │       │   StageCoreClient ──extends──> EngineCoreClient
    │       │       │ zmq
    │       │       ▼
    │       │   StageCoreProc ──contains──> EngineCore
    │       │
    │       └── connectors: OmniConnectorBase
    │
    └── (EngineClient protocol methods delegated to PipelineOrchestrator)
```

**Key characteristics:**
- `AsyncOmni` implements vLLM's `EngineClient` protocol directly (no custom OmniBase)
- `PipelineOrchestrator` (new, vllm-omni) encapsulates all multi-stage pipeline logic
- `StageCoreClient` (new, vllm-omni) extends `EngineCoreClient` directly, **bypassing** the `LLM`/`AsyncLLM`/`LLMEngine` layers
- `StageCoreProc` (new, vllm-omni) wraps `EngineCore` in a dedicated process
- Communication uses **ZMQ** instead of `multiprocessing.Queue`
- **Result: 2 layers of indirection** (AsyncOmni → PipelineOrchestrator → StageCoreClient → StageCoreProc)

---

## 2. Motivation

### 2.1 Problems with the Current Architecture

1. **Redundant layering**: Each stage creates a full `AsyncOmniLLM` (extending `AsyncLLM`) which internally creates `EngineCoreClient` → `EngineCoreProc`. The `OmniLLM`/`AsyncOmniLLM` and `LLMEngine`/`AsyncLLM` layers add complexity without benefit in the multi-stage context.

2. **Not a native vLLM citizen**: `AsyncOmni` extends custom `OmniBase` instead of implementing vLLM's `EngineClient` protocol. This requires adapter/shim code everywhere the OpenAI serving layer expects an `EngineClient`.

3. **Dual worker management**: `OmniStage` spawns a worker process that runs its own event loop, and inside that process `AsyncOmniLLM` spawns *another* set of processes via `EngineCoreClient.make_async_mp_client()`. This creates nested process trees that are hard to reason about and debug.

4. **IPC overhead**: Using `multiprocessing.Queue` for orchestrator↔stage communication means pickling Python objects through OS pipes, which has limited throughput and high latency for large payloads.

5. **Tight coupling**: Stage worker functions (`_stage_worker`, `_stage_worker_async`) are 400+ line monolithic functions that mix device setup, engine initialization, batching, profiling, and IPC concerns.

### 2.2 Benefits of the Target Architecture

1. **Fewer layers**: Eliminates `OmniBase`, `OmniStage`, `OmniLLM`, `AsyncOmniLLM`, `LLMEngine` from the critical path. Each stage directly manages an `EngineCore` via `StageCoreClient`.

2. **Native EngineClient**: `AsyncOmni` implements vLLM's `EngineClient` protocol, making it a drop-in replacement for `AsyncLLM` in the OpenAI serving layer.

3. **ZMQ transport**: Replaces `multiprocessing.Queue` with ZMQ sockets for orchestrator↔stage communication, enabling higher throughput, network transparency, and easier distributed deployment.

4. **Clear separation of concerns**: `PipelineOrchestrator` owns pipeline logic; `StageCoreClient` owns per-stage communication; `StageCoreProc` owns per-stage execution.

5. **Simpler process model**: Each stage has exactly one process (`StageCoreProc`), not nested processes.

---

## 3. Detailed Component Design

### 3.1 AsyncOmni (refactored)

**File**: `vllm_omni/entrypoints/async_omni.py`

**Inherits**: Implements `vllm.engine.protocol.EngineClient` (Protocol class)

**Responsibilities**:
- Public API surface: `generate()`, `abort()`, `get_model_config()`, `get_tokenizer()`, etc.
- Lifecycle management: construction, shutdown, pause/resume
- Delegates all pipeline orchestration to `PipelineOrchestrator`

**Key changes from current**:
- Remove inheritance from `OmniBase`
- Implement all `EngineClient` protocol methods
- Move stage management, connector setup, queue management into `PipelineOrchestrator`
- Keep `input_processor`, `io_processor`, `model_config` as attributes (populated after pipeline init)
- Keep `renderer` property for OpenAI serving compatibility

**Interface**:
```python
class AsyncOmni:  # implements EngineClient protocol
    def __init__(self, model: str, **kwargs) -> None:
        self.orchestrator = PipelineOrchestrator(model, **kwargs)
        # Populate EngineClient-required attributes from orchestrator
        self.model_config = self.orchestrator.model_config
        self.input_processor = self.orchestrator.input_processor
        ...

    async def generate(self, prompt, request_id, sampling_params_list, ...) -> AsyncGenerator:
        async for output in self.orchestrator.generate(prompt, request_id, ...):
            yield output

    async def abort(self, request_id) -> None:
        await self.orchestrator.abort(request_id)

    # ... other EngineClient protocol methods
```

### 3.2 PipelineOrchestrator (new)

**File**: `vllm_omni/entrypoints/pipeline_orchestrator.py`

**Inherits**: None (standalone class)

**Responsibilities**:
- Load and parse stage configurations from YAML/model
- Create and manage `StageCoreClient` instances (one per stage)
- Initialize inter-stage connectors (`OmniConnectorBase`)
- Request routing: submit to stage-0, forward between stages via connectors
- Output collection: poll `StageCoreClient` instances for results
- Metrics aggregation via `OrchestratorAggregator`
- Sequential and async-chunk result processing modes
- Lifecycle: startup, shutdown, pause/resume propagation

**Absorbs logic from**:
- `OmniBase._initialize_stages()` → stage config loading, connector init
- `OmniBase._start_stages()` → starting stage processes
- `OmniBase._wait_for_stages_ready()` → readiness synchronization
- `AsyncOmni.generate()` → request submission and result routing
- `AsyncOmni._process_sequential_results()` → sequential pipeline flow
- `AsyncOmni._process_async_results()` → async chunk pipeline flow
- `AsyncOmni._run_output_handler()` → background output polling
- `OmniStage.process_engine_inputs()` → inter-stage input derivation

**Key design decisions**:
- Owns the list of `StageCoreClient` instances (replaces `stage_list: list[OmniStage]`)
- Owns the connectors dict (replaces `self.connectors` on `OmniBase`)
- Does NOT own input/output processing — that stays in `AsyncOmni` or `StageCoreClient`
- Exposes `model_config`, `vllm_config`, `tokenizer` obtained from the first LLM stage

**Interface**:
```python
class PipelineOrchestrator:
    def __init__(self, model: str, **kwargs) -> None:
        self.stages: list[StageCoreClient] = []
        self.connectors: dict[tuple[str, str], OmniConnectorBase] = {}
        self.model_config = None
        self.vllm_config = None
        self.tokenizer = None
        self.input_processor = None
        self._initialize(model, kwargs)

    async def generate(self, prompt, request_id, sampling_params_list, ...) -> AsyncGenerator:
        ...

    async def abort(self, request_id) -> None:
        ...

    def shutdown(self) -> None:
        for stage in self.stages:
            stage.shutdown()
```

### 3.3 StageCoreClient (new)

**File**: `vllm_omni/entrypoints/stage_core_client.py`

**Inherits**: `vllm.v1.engine.core_client.EngineCoreClient`

**Responsibilities**:
- Manage ZMQ connection to one `StageCoreProc`
- Submit requests (engine inputs + sampling params) via ZMQ
- Receive results (engine outputs + metrics) via ZMQ
- Support abort, profiling commands
- Expose stage metadata (stage_id, stage_type, final_output, vllm_config, tokenizer)

**Absorbs logic from**:
- `OmniStage.submit()` → request submission
- `OmniStage.try_collect()` → result polling
- `OmniStage.attach_queues()` / `init_stage_worker()` → process lifecycle
- `OmniStage.stop_stage_worker()` → shutdown
- `OmniStage` metadata (stage_id, stage_type, engine_input_source, final_output, etc.)
- `AsyncOmniLLM.__init__()` partially → omni-specific input/output processor setup
  (note: the heavyweight engine creation moves to `StageCoreProc`)

**Key design decisions**:
- Uses ZMQ DEALER/ROUTER sockets for async request-reply
- Inherits `EngineCoreClient` to reuse vLLM's request/response serialization
- Spawns `StageCoreProc` as a subprocess during initialization
- Receives `vllm_config` and `tokenizer` from the stage process after initialization
- Does NOT run its own event loop — communicates via ZMQ from the orchestrator's event loop

**Interface**:
```python
class StageCoreClient(EngineCoreClient):
    def __init__(self, stage_config, model, ...) -> None:
        self.stage_id = stage_config.stage_id
        self.stage_type = stage_config.stage_type
        ...
        # Spawn StageCoreProc and establish ZMQ connection
        self._proc = self._spawn_stage_proc(model, stage_config, ...)
        self._zmq_socket = ...

    async def submit_request(self, request_id, engine_inputs, sampling_params) -> None:
        """Send a generation request to the stage process."""
        ...

    async def try_collect(self) -> dict | None:
        """Non-blocking poll for results from the stage process."""
        ...

    async def abort_request(self, request_id) -> None:
        ...

    def shutdown(self) -> None:
        ...

    # Stage metadata
    @property
    def vllm_config(self) -> VllmConfig | None: ...
    @property
    def tokenizer(self) -> TokenizerLike | None: ...
```

### 3.4 StageCoreProc (new)

**File**: `vllm_omni/entrypoints/stage_core_proc.py`

**Inherits**: None (standalone process entry point, analogous to `vllm.v1.engine.core_proc.EngineCoreProc`)

**Responsibilities**:
- Run in a dedicated subprocess
- Initialize device environment (CUDA_VISIBLE_DEVICES, etc.)
- Create and manage a vLLM `EngineCore` instance
- Handle ZMQ communication with `StageCoreClient`
- Process generation requests, forward to `EngineCore`, return results
- Handle inter-stage connector communication (receive from upstream via connector)
- Support profiling, abort, shutdown commands

**Absorbs logic from**:
- `_stage_worker()` function → sync worker loop, device setup, engine init
- `_stage_worker_async()` function → async worker loop, device setup, engine init
- Engine creation logic from `OmniLLM.__init__()` and `AsyncOmniLLM.__init__()`
- Connector receive logic (`try_recv_via_connector`)
- Batching logic from `_stage_worker`
- Profiling handler logic

**Key design decisions**:
- Contains an `EngineCore` directly (not `OmniLLM` or `AsyncOmniLLM`)
- Uses the omni-specific `OmniInputProcessor` and `MultimodalOutputProcessor`
- ZMQ DEALER socket connects back to `StageCoreClient`
- Runs an asyncio event loop for concurrent request handling
- Initializes connectors for receiving data from upstream stages

**Interface**:
```python
class StageCoreProc:
    """Runs in a subprocess. Not instantiated directly by users."""

    @staticmethod
    def run(model: str, stage_payload: dict, zmq_address: str, ...) -> None:
        """Entry point for the subprocess."""
        proc = StageCoreProc(model, stage_payload, zmq_address, ...)
        proc.event_loop()

    def __init__(self, model, stage_payload, zmq_address, ...) -> None:
        # Device setup
        # EngineCore initialization
        # ZMQ socket setup
        # Connector initialization
        ...

    def event_loop(self) -> None:
        """Main loop: receive requests via ZMQ, process via EngineCore, send results."""
        ...
```

---

## 4. Communication Design

### 4.1 ZMQ Transport (Orchestrator ↔ Stage)

Replace `multiprocessing.Queue` with ZMQ for all orchestrator↔stage communication.

**Socket pattern**: `PUSH/PULL` or `DEALER/ROUTER`
- `StageCoreClient` uses a `PUSH` socket to send requests and a `PULL` socket to receive results
- `StageCoreProc` uses the corresponding `PULL` (receive) and `PUSH` (send) sockets
- Addresses: `ipc://` for local, `tcp://` for distributed

**Message format**: Use msgpack or pickle for serialization (matching vLLM's existing `EngineCoreProc` format).

**Advantages over mp.Queue**:
- No GIL contention from pickle serialization in the main process
- Network-transparent (same code works for local and distributed)
- Better flow control with ZMQ's HWM (high water mark)
- Non-blocking poll with `zmq.POLLIN`

### 4.2 Inter-Stage Data Transfer (Stage ↔ Stage)

The existing `OmniConnectorBase` framework (SharedMemory, Mooncake, Yuanrong) is preserved unchanged. Connectors continue to handle large payload transfer (embeddings, KV cache) between stages.

**Change**: The connector send logic currently in `AsyncOmni._process_sequential_results()` moves to `PipelineOrchestrator`. The connector receive logic currently in `_stage_worker_async.generation_single_request()` moves to `StageCoreProc`.

### 4.3 Readiness Protocol

Current: Stage worker sends `{"type": "stage_ready", ...}` via `mp.Queue`.
New: `StageCoreProc` sends a readiness message via ZMQ after `EngineCore` initialization. `StageCoreClient` blocks on this message during construction. The readiness message includes `vllm_config`, `tokenizer`, and `is_tracing_enabled`.

---

## 5. Diffusion Stage Handling

The current architecture supports both `llm` and `diffusion` stage types. In the target architecture:

- **LLM stages**: `StageCoreProc` creates `EngineCore` (vLLM) directly
- **Diffusion stages**: `StageCoreProc` creates `DiffusionEngine`/`AsyncOmniDiffusion` (the diffusion path does not use vLLM's `EngineCore`)

`StageCoreClient` abstracts over both — the `PipelineOrchestrator` does not need to know whether a stage is LLM or diffusion. The `StageCoreProc` handles the type-specific initialization internally.

---

## 6. Files to Create / Modify / Delete

### New Files
| File | Description |
|------|-------------|
| `vllm_omni/entrypoints/pipeline_orchestrator.py` | `PipelineOrchestrator` class |
| `vllm_omni/entrypoints/stage_core_client.py` | `StageCoreClient` class |
| `vllm_omni/entrypoints/stage_core_proc.py` | `StageCoreProc` class |

### Modified Files
| File | Changes |
|------|---------|
| `vllm_omni/entrypoints/async_omni.py` | Rewrite to implement `EngineClient`, delegate to `PipelineOrchestrator` |
| `vllm_omni/entrypoints/openai/api_server.py` | Minimal changes (AsyncOmni already returned as `EngineClient`) |

### Deprecated / To Remove (after migration complete)
| File | Reason |
|------|--------|
| `vllm_omni/entrypoints/omni.py` | `OmniBase` replaced by `PipelineOrchestrator` |
| `vllm_omni/entrypoints/omni_stage.py` | `OmniStage` + worker functions replaced by `StageCoreClient` + `StageCoreProc` |
| `vllm_omni/entrypoints/omni_llm.py` | `OmniLLM` eliminated (no more `LLM` wrapper) |
| `vllm_omni/entrypoints/async_omni_llm.py` | `AsyncOmniLLM` eliminated (no more `AsyncLLM` wrapper) |

### Preserved (no changes)
| File | Reason |
|------|--------|
| `vllm_omni/distributed/omni_connectors/*` | Connector framework unchanged |
| `vllm_omni/engine/input_processor.py` | `OmniInputProcessor` reused in `StageCoreProc` |
| `vllm_omni/engine/output_processor.py` | `MultimodalOutputProcessor` reused in `StageCoreProc` |
| `vllm_omni/entrypoints/async_omni_diffusion.py` | Diffusion engine reused inside `StageCoreProc` |
| `vllm_omni/entrypoints/client_request_state.py` | Request state tracking reused in `PipelineOrchestrator` |
| `vllm_omni/metrics.py` | Metrics framework reused |
| `vllm_omni/outputs.py` | Output types reused |

---

## 7. Backward Compatibility

### 7.1 Public API
The public API of `AsyncOmni` remains unchanged:
- `AsyncOmni(model, **kwargs)` constructor
- `async generate(prompt, request_id, sampling_params_list, output_modalities)`
- `async abort(request_id)`
- `shutdown()`
- All `EngineClient` protocol methods

### 7.2 Configuration
Stage YAML configuration format remains unchanged. All existing stage configs continue to work.

### 7.3 Sync Path (OmniBase / Omni)
The synchronous `Omni` class (extending `OmniBase`) and its sync worker path (`_stage_worker` with `OmniLLM`) will be deprecated. The refactored architecture is async-first. If sync support is needed, it can be provided via `asyncio.run()` wrappers.

### 7.4 OpenAI Serving Layer
The OpenAI API server already uses `AsyncOmni` as `EngineClient`. After refactoring, `AsyncOmni` will natively implement the `EngineClient` protocol, so no changes are needed in the serving layer.
