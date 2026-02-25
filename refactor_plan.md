# vLLM-Omni Architecture Refactor Implementation Plan

> **Prerequisite**: Read `refactor_design.md` for full architectural context.

## Phase Overview

| Phase | Description | Risk | Estimated Scope |
|-------|-------------|------|-----------------|
| 0 | Preparation & test baseline | Low | Infrastructure |
| 1 | Create `StageCoreProc` | Medium | New file, ~500 lines |
| 2 | Create `StageCoreClient` | Medium | New file, ~300 lines |
| 3 | Create `PipelineOrchestrator` | Medium | New file, ~600 lines |
| 4 | Refactor `AsyncOmni` | High | Major rewrite, ~300 lines |
| 5 | Integration testing & cleanup | Medium | Testing + deletions |
| 6 | Remove deprecated code | Low | Delete old files |

---

## Phase 0: Preparation & Test Baseline

**Goal**: Establish a reliable test baseline before any changes.

### Step 0.1: Capture current test results
```bash
# Run existing tests, save output as baseline
python -m pytest tests/ -v --tb=short > test_baseline.txt 2>&1
```

### Step 0.2: Identify all integration test entry points
- Find all tests that use `AsyncOmni`, `OmniBase`, `OmniStage`, `OmniLLM`, `AsyncOmniLLM`
- Document which tests exercise the multi-stage pipeline end-to-end
- These tests will be the acceptance criteria for the refactor

### Step 0.3: Create feature branch
```bash
git checkout -b refactor/flatten-architecture
```

### Step 0.4: Add ZMQ dependency
- Add `pyzmq` to project dependencies (it may already be present for diffusion workers)
- Verify `import zmq` works in the test environment

---

## Phase 1: Create `StageCoreProc`

**Goal**: Build the subprocess that wraps `EngineCore` and communicates via ZMQ.

**File**: `vllm_omni/entrypoints/stage_core_proc.py`

### Step 1.1: Define the class skeleton

```python
class StageCoreProc:
    """Stage engine core process. Runs EngineCore in a subprocess, communicates via ZMQ."""

    @staticmethod
    def run(model, stage_payload, zmq_addr, ...): ...
    def __init__(self, model, stage_payload, zmq_addr, ...): ...
    def _setup_devices(self, stage_id, runtime_cfg): ...
    def _init_engine_core(self, model, engine_args, stage_type): ...
    def _init_connectors(self, stage_id, connectors_config): ...
    def _setup_zmq(self, zmq_addr): ...
    def _send_ready(self): ...
    async def _event_loop(self): ...
    async def _handle_request(self, msg): ...
    async def _handle_abort(self, msg): ...
    async def _handle_profiler(self, msg): ...
    def _shutdown(self): ...
```

### Step 1.2: Port device setup logic
- Extract from `_stage_worker()` lines 683-691 and `_stage_worker_async()` lines 1066-1074
- `set_stage_devices()`, `VLLM_WORKER_MULTIPROC_METHOD`, `mp.set_start_method("spawn")`

### Step 1.3: Port engine initialization
- **LLM path**: Instead of creating `OmniLLM` or `AsyncOmniLLM`, create `EngineCore` directly
  - Use `AsyncOmniEngineArgs` to create `VllmConfig`
  - Use `VllmConfig` to create `EngineCore` (the same class that `EngineCoreProc` uses internally)
  - Also create `OmniInputProcessor` and `MultimodalOutputProcessor` for omni-specific I/O
- **Diffusion path**: Create `AsyncOmniDiffusion` (same as current `_stage_worker_async`)

### Step 1.4: Implement ZMQ communication
- Use `zmq.PULL` socket to receive requests from `StageCoreClient`
- Use `zmq.PUSH` socket to send results back to `StageCoreClient`
- Message format: `[msg_type, msgpack_payload]`
- Message types: `GENERATE`, `ABORT`, `PROFILER_START`, `PROFILER_STOP`, `SHUTDOWN`, `READY`, `RESULT`, `ERROR`

### Step 1.5: Implement readiness protocol
- After `EngineCore` initialization, send `READY` message via ZMQ
- Include `vllm_config`, `tokenizer`, `is_tracing_enabled` in the ready payload

### Step 1.6: Implement the main event loop
- Port the async event loop from `_stage_worker_async()` lines 1277-1375
- Replace `in_q.get_nowait()` with `zmq.POLLIN` on the PULL socket
- Replace `out_q.put()` with `zmq.send()` on the PUSH socket
- Port connector receive logic from `generation_single_request()`
- Port profiling handlers from `handle_profiler_task_async()`

### Step 1.7: Port batching and metrics
- Port `make_request_stats()`, `make_stage_stats()` calls
- Port SHM serialization (`maybe_dump_to_shm`) — evaluate if still needed with ZMQ
  (ZMQ handles large messages natively; may not need SHM for stage output)

### Step 1.8: Unit test `StageCoreProc` in isolation
- Write a test that spawns `StageCoreProc` as a subprocess
- Connects via ZMQ, sends a test request, verifies response
- Test shutdown and error handling

---

## Phase 2: Create `StageCoreClient`

**Goal**: Build the client that communicates with `StageCoreProc` via ZMQ.

**File**: `vllm_omni/entrypoints/stage_core_client.py`

### Step 2.1: Define the class skeleton

```python
class StageCoreClient:
    """Client for communicating with a StageCoreProc via ZMQ."""

    def __init__(self, stage_config, model, ...): ...
    def _spawn_proc(self, model, stage_config, ...): ...
    def _setup_zmq(self): ...
    def _wait_for_ready(self, timeout): ...
    async def submit_request(self, request_id, engine_inputs, sampling_params): ...
    async def try_collect(self) -> dict | None: ...
    async def abort_request(self, request_id): ...
    async def start_profile(self): ...
    async def stop_profile(self): ...
    def shutdown(self): ...
```

### Step 2.2: Implement process spawning
- Port from `OmniStage.init_stage_worker()` lines 394-503
- Use `multiprocessing.Process(target=StageCoreProc.run, args=...)`
- Pass ZMQ address (e.g., `ipc:///tmp/vllm-omni-stage-{stage_id}-{uuid}` or `tcp://127.0.0.1:{port}`)
- Support Ray backend: `start_ray_actor(StageCoreProc.run, ...)`

### Step 2.3: Implement ZMQ setup
- `zmq.PUSH` socket to send requests to `StageCoreProc`
- `zmq.PULL` socket to receive results from `StageCoreProc`
- Bind sockets before spawning the subprocess (so the subprocess connects to known addresses)

### Step 2.4: Implement readiness wait
- Block on `zmq.PULL` until `READY` message received
- Extract `vllm_config`, `tokenizer`, `is_tracing_enabled` from ready payload
- Timeout if ready not received within `stage_init_timeout`

### Step 2.5: Implement request submission
- Port from `OmniStage.submit()` lines 532-570
- Serialize request as msgpack and send via ZMQ PUSH socket
- Inject `global_request_id` into `additional_information` (same as current)

### Step 2.6: Implement result collection
- Port from `OmniStage.try_collect()` lines 572-583
- Non-blocking `zmq.POLLIN` on PULL socket
- Deserialize result and return

### Step 2.7: Implement abort, profiling, shutdown
- Send typed messages via ZMQ PUSH socket

### Step 2.8: Port stage metadata
- Port from `OmniStage.__init__()` lines 248-293
- `stage_id`, `stage_type`, `engine_input_source`, `final_output`, `final_output_type`
- `default_sampling_params`, `requires_multimodal_data`, `is_comprehension`
- `custom_process_input_func`

### Step 2.9: Port `process_engine_inputs()`
- Port from `OmniStage.process_engine_inputs()` lines 585-633
- This logic derives inputs for the current stage from upstream stage outputs
- Keep it on `StageCoreClient` (or as a standalone function used by `PipelineOrchestrator`)

### Step 2.10: Integration test StageCoreClient + StageCoreProc
- Test the full client↔proc lifecycle:
  - Spawn proc, wait for ready, submit request, collect result, shutdown
- Test with both LLM and diffusion stage types (mock or lightweight model)

---

## Phase 3: Create `PipelineOrchestrator`

**Goal**: Build the multi-stage pipeline coordinator.

**File**: `vllm_omni/entrypoints/pipeline_orchestrator.py`

### Step 3.1: Define the class skeleton

```python
class PipelineOrchestrator:
    """Multi-stage pipeline orchestrator."""

    def __init__(self, model, **kwargs): ...
    def _load_stage_configs(self, model, kwargs): ...
    def _init_connectors(self, config_path, kwargs): ...
    def _create_stages(self, model, kwargs): ...
    def _wait_for_stages_ready(self, timeout): ...
    async def generate(self, prompt, request_id, sampling_params_list, ...): ...
    async def _process_sequential_results(self, ...): ...
    async def _process_async_results(self, ...): ...
    def _process_single_result(self, result, stage, stage_id, metrics): ...
    def _run_output_handler(self): ...
    async def abort(self, request_id): ...
    def shutdown(self): ...
```

### Step 3.2: Port stage config loading
- Port from `OmniBase._initialize_stages()` lines 209-296
- `load_stage_configs_from_model()`, `load_stage_configs_from_yaml()`
- LoRA injection for diffusion stages
- Connector initialization via `initialize_orchestrator_connectors()`

### Step 3.3: Port stage creation
- Replace `OmniStage(cfg)` with `StageCoreClient(cfg, model, ...)`
- Parallel stage creation with `ThreadPoolExecutor` (same as current)
- No more `attach_queues()` — ZMQ setup is internal to `StageCoreClient`
- No more `init_stage_worker()` — process spawning is internal to `StageCoreClient`

### Step 3.4: Port readiness synchronization
- Port from `OmniBase._wait_for_stages_ready()` lines in omni.py
- Each `StageCoreClient` already waits for its own process during construction
- After all stages ready, populate `model_config`, `vllm_config`, `tokenizer`, `input_processor`
- Port from `AsyncOmni._wait_for_stages_ready()` lines 189-224

### Step 3.5: Port generate() and result processing
- Port `AsyncOmni.generate()` lines 235-361
- Port `AsyncOmni._process_sequential_results()` lines 409-474
- Port `AsyncOmni._process_async_results()` lines 363-407
- Port `AsyncOmni._process_single_result()` lines 476-545
- Replace `self.stage_list[i].submit(task)` with `self.stages[i].submit_request(...)`
- Replace `stage.try_collect()` with `stage.try_collect()` (same interface on `StageCoreClient`)

### Step 3.6: Port output handler
- Port `AsyncOmni._run_output_handler()` lines 547-599
- Background asyncio task that polls all stages for results

### Step 3.7: Port inter-stage forwarding
- Port connector send logic from `_process_sequential_results()` lines 435-474
- `try_send_via_connector()` call stays the same
- Submit function reference changes to `self.stages[next_stage_id].submit_request`

### Step 3.8: Port abort, shutdown, profiling
- Port abort from `AsyncOmni.abort()` lines 626-630
- Port shutdown cleanup
- Port start_profile/stop_profile from `OmniBase`

### Step 3.9: Integration test PipelineOrchestrator
- Test with a 2-stage pipeline configuration
- Verify request flows through both stages
- Verify metrics are aggregated correctly

---

## Phase 4: Refactor `AsyncOmni`

**Goal**: Rewrite `AsyncOmni` as a thin `EngineClient` wrapper around `PipelineOrchestrator`.

**File**: `vllm_omni/entrypoints/async_omni.py`

### Step 4.1: Remove `OmniBase` inheritance
- Remove `class AsyncOmni(OmniBase):`
- Change to `class AsyncOmni:` (implements `EngineClient` protocol implicitly)

### Step 4.2: Rewrite `__init__`
```python
def __init__(self, model: str, **kwargs) -> None:
    self._pause_cond = asyncio.Condition()
    self._paused = False
    self.request_states = {}
    self.output_handler = None

    # Create the pipeline orchestrator (replaces OmniBase.__init__)
    self.orchestrator = PipelineOrchestrator(model, **kwargs)

    # Populate EngineClient-required attributes
    self.model_config = self.orchestrator.model_config
    self.input_processor = self.orchestrator.input_processor
    self.io_processor = self.orchestrator.io_processor
    self.vllm_config = self.orchestrator.vllm_config

    # Cleanup via weakref
    self._weak_finalizer = weakref.finalize(self, self.orchestrator.shutdown)
```

### Step 4.3: Delegate generate() to orchestrator
- Keep the pause/resume check
- Delegate to `self.orchestrator.generate()`

### Step 4.4: Delegate all EngineClient methods
- `abort()` → `self.orchestrator.abort()`
- `get_model_config()` → return `self.model_config`
- `get_tokenizer()` → return `self.orchestrator.tokenizer`
- `get_vllm_config()` → return `self.orchestrator.vllm_config`
- `is_tracing_enabled()` → return from orchestrator
- `shutdown()` → `self.orchestrator.shutdown()`
- `start_profile()` / `stop_profile()` → delegate to orchestrator
- `pause_generation()` / `resume_generation()` → keep current implementation
- No-op methods: `do_log_stats()`, `check_health()`, `reset_mm_cache()`, etc.

### Step 4.5: Remove moved code
- Remove `_process_sequential_results`, `_process_async_results`, `_process_single_result`
- Remove `_run_output_handler`
- Remove all `OmniBase`-inherited code references

### Step 4.6: Verify OpenAI API server compatibility
- `api_server.py` creates `AsyncOmni` and uses it as `EngineClient`
- Verify all accessed attributes and methods still work:
  - `engine_client.generate()`, `engine_client.abort()`
  - `engine_client.model_config`, `engine_client.renderer`
  - `engine_client.get_model_config()`, `engine_client.get_tokenizer()`

---

## Phase 5: Integration Testing & Cleanup

### Step 5.1: Run full test suite
```bash
python -m pytest tests/ -v --tb=short
```
Compare against baseline from Phase 0.

### Step 5.2: End-to-end test with real models
- Test with single-stage LLM model (e.g., Qwen2.5-Omni-7B)
- Test with multi-stage LLM+Diffusion pipeline
- Test with OpenAI API server endpoint

### Step 5.3: Test connector paths
- Verify SharedMemoryConnector works with new architecture
- Verify MooncakeConnector works (if test environment supports it)

### Step 5.4: Test edge cases
- Request abort mid-generation
- Stage initialization timeout
- Pause/resume generation
- Profiling start/stop
- Multiple concurrent requests (async chunk mode)

### Step 5.5: Performance comparison
- Compare latency and throughput vs. old architecture
- ZMQ should reduce IPC overhead for large payloads
- Fewer process layers should reduce scheduling latency

---

## Phase 6: Remove Deprecated Code

**Only after all tests pass and the new architecture is validated.**

### Step 6.1: Remove old files
- Delete `vllm_omni/entrypoints/omni.py` (`OmniBase`)
- Delete `vllm_omni/entrypoints/omni_llm.py` (`OmniLLM`)
- Delete `vllm_omni/entrypoints/async_omni_llm.py` (`AsyncOmniLLM`)

### Step 6.2: Refactor `omni_stage.py`
- Remove `OmniStage` class and all worker functions (`_stage_worker`, `_stage_worker_async`, etc.)
- Keep any utility functions (like `output_strip`, `make_request_stats`) that are reused
- Or move shared utilities to a new `vllm_omni/entrypoints/stage_utils.py` (if not already there)

### Step 6.3: Clean up imports
- Remove all imports of deleted classes throughout the codebase
- Update `__init__.py` files

### Step 6.4: Update documentation
- Update any docs referencing the old class hierarchy
- Update architecture diagrams

---

## Risk Mitigation

### Risk 1: EngineCore direct usage
**Risk**: `EngineCore` may have assumptions about being managed by `EngineCoreProc` (e.g., specific IPC setup, process lifecycle hooks).
**Mitigation**: Study `EngineCoreProc` carefully. `StageCoreProc` may need to replicate some of its setup logic. Alternatively, `StageCoreProc` could delegate to `EngineCoreProc` internally and only replace the communication layer.

### Risk 2: ZMQ reliability
**Risk**: ZMQ socket errors, message loss, or ordering issues.
**Mitigation**: Use reliable patterns (PUSH/PULL for ordered delivery), add heartbeats, handle reconnection. ZMQ PUSH/PULL guarantees ordered delivery and no message loss for connected peers.

### Risk 3: Serialization compatibility
**Risk**: Objects sent via ZMQ may not serialize correctly (e.g., torch tensors, custom types).
**Mitigation**: Reuse vLLM's existing serialization from `EngineCoreProc` (msgpack + custom encoders). For large tensors, continue using shared memory connectors.

### Risk 4: Diffusion stage compatibility
**Risk**: Diffusion stages don't use `EngineCore` and have different lifecycle.
**Mitigation**: `StageCoreProc` handles both paths internally. The `stage_type` field determines which engine is created. Test diffusion path specifically.

### Risk 5: Ray backend
**Risk**: The Ray backend path (`start_ray_actor`) may need significant adaptation.
**Mitigation**: Phase 2 Step 2.2 addresses Ray backend. Test with Ray specifically. May need to use ZMQ over TCP for cross-node communication.

---

## Dependency Graph

```
Phase 0 (Preparation)
    │
    ▼
Phase 1 (StageCoreProc)
    │
    ▼
Phase 2 (StageCoreClient) ── depends on Phase 1
    │
    ▼
Phase 3 (PipelineOrchestrator) ── depends on Phase 2
    │
    ▼
Phase 4 (AsyncOmni refactor) ── depends on Phase 3
    │
    ▼
Phase 5 (Integration testing) ── depends on Phase 4
    │
    ▼
Phase 6 (Cleanup) ── depends on Phase 5
```

Each phase produces testable artifacts. The old and new code can coexist during development — the old `AsyncOmni(OmniBase)` continues to work until Phase 4 replaces it.
