# Duplex Refactor Design Plan

Branch: `duplex_refactor1` (follow-up to vllm-project/vllm-omni#6196).
Revision 3: alternatives A (engine-resident sessions), B (typed
command/event contract) and C (single model plugin) applied; `DuplexOmniEngine`
kept as a thin, clearly named engine sibling (see §7 D10-D13). Revision 4
(API design update, not yet implemented in code): session ids are always
allocated by the server and the `incarnation` counter is dropped (§3.9, §7 D16).
Revision 5 (implemented): the simplification pass of §7 D17: one session object
without a separate resources holder, no input-mode mechanism, typed objects
across the engine queue, and typed construction of stateless events.

This document records (1) what the current duplex code does and where its
problems are, (2) the target architecture, (3) the public API design of
`DuplexOmni`, `DuplexOrchestrator`, `DuplexClientBase` / `InlineDuplexClient`,
(4) the removals, and (5) a phased migration plan with test strategy.

---

## 1. Current state

### 1.1 Layers and the active path

```text
DuplexClient (websockets)
  -> /v1/realtime?duplex=1  (api_server.py)            [also /v1/duplex, native dialect]
  -> OmniDuplexSessionHandler                          (entrypoints/duplex/serving.py, 1904 lines)
       = DuplexSessionRunnerMixin                      (session_runner.py, 2097 lines, ONE 2000-line closure method)
       + NativeRuntimeBridgeMixin                      (runtime_bridge.py, 1424 lines)
       + ChatFallbackProjectorMixin                    (chat_fallback.py, 264 lines)
       owns: DuplexSessionRegistry/DuplexSession, ServingRuntimeAdapter (model plugin),
             DuplexSessionAttachmentRegistry (resume tokens + replay journal),
             lifecycle listener, per-session task handles, Realtime projection
  -> AsyncOmni.{open,append,collect,signal,close,touch,resume}_duplex_*_async   (7 proxies)
  -> DuplexRequestClient                               (entrypoints/duplex_request_client.py)
  -> AsyncOmniEngine.{open,append,signal,close,touch,resume}_duplex_*           (13 methods)
  -> DuplexControlClient -> CorrelatedRpcClient -> request queue
  -> Orchestrator._request_handler -> DuplexControlPlane (+ DuplexSessionRuntimeManager, lease reaper)
  -> _OrchestratorDuplexStagePort -> StagePool (resumable Stage0 request) -> Stage1 ...
  <- OutputMessage per request_id  -> AsyncOmni._final_output_handler -> request_states[rid].queue
  <- DuplexSessionLifecycleMessage -> AsyncOmni.duplex_lifecycle_events -> handler lifecycle listener
```

The same view as §2.1, drawn for the baseline
(`D:\repo\github\chickeyton\vllm-omni_duplex_refactor_baseline`, commit `7112347f`):

```text
              Python users                                WebSocket clients
        ┌──────────────────────┐                    ┌────────────────────────┐
        │  (no Python API:     │                    │      DuplexClient      │
        │   only 7 fenced      │                    │   clients/duplex.py    │
        │   AsyncOmni proxies) │                    └───────────┬────────────┘
        └──────────────────────┘                                │ /v1/realtime?duplex=1 (Realtime JSON)
                                                                │ /v1/duplex (native dialect, aliases)
                                                                ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ OmniDuplexSessionHandler          [entrypoints/duplex/serving.py, 1,904 lines] │
        │   = DuplexSessionRunnerMixin      session_runner.py: handle_session, ONE       │
        │                                   2,040-line method with 21 nonlocal closures  │
        │   + NativeRuntimeBridgeMixin      runtime_bridge.py: every engine call         │
        │   + ChatFallbackProjectorMixin    chat_fallback.py: generic (non-native) lane  │
        │   owns: DuplexSessionRegistry -> DuplexSession        SESSION STATE MACHINE #1 │
        │           (config, epoch/turn, input/response/playback/history ledgers)        │
        │         ServingRuntimeAdapter (serving-side model plugin, Protocol)            │
        │           -> ServingRuntimeSessionState               per-session model state  │
        │         RealtimeOutputProjector / RealtimeSessionState  wire projection state  │
        │         DuplexSessionAttachmentRegistry  resume tokens, replay, grace timer    │
        │         lifecycle listener, per-session task handles, server VAD              │
        │   ~40 `realtime_protocol is None` branches (native vs Realtime dialect)       │
        └──────────────────────────────────────┬───────────────────────────────────────┘
                                               │ open / append / collect / signal / close / touch / resume
                                               ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ AsyncOmni(EngineClient, OmniBase)                  [entrypoints/async_omni.py] │
        │   7 *_duplex_*_async proxies  ->  DuplexRequestClient                         │
        │   duplex_lifecycle_events queue; duplex arm in the output loop                │
        │   (turn-based request handling in the same class)                             │
        └──────────────────────────────────────┬───────────────────────────────────────┘
                                               │ fence-carrying RPC per operation
                                               ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ AsyncOmniEngine                             [engine/async_omni_engine.py]      │
        │   13 *_duplex_* methods  ->  DuplexControlClient  ->  CorrelatedRpcClient     │
        │   loads duplex_runtime_extension; duplex config plumbing                       │
        │   (turn-based request handling in the same class)                             │
        │  └─ Orchestrator                                    [engine/orchestrator.py]   │
        │       130 duplex mentions: duplex arm in _request_handler, 5 branches in       │
        │       _route_output, duplex fields on OrchestratorRequestState, reaper in run()│
        │       ├─ DuplexControlPlane           open/append/signal/close/touch/resume,   │
        │       │    + DuplexSessionRuntimeManager                                       │
        │       │      -> DuplexSessionRuntimeState          SESSION STATE MACHINE #2    │
        │       │         (fence, lease, stage bindings, append reservation,             │
        │       │          operation_id idempotency cache)                               │
        │       └─ _OrchestratorDuplexStagePort  -> StagePool (resumable Stage 0) ...   │
        │       plugin: DuplexRuntimeExtension  (engine-side model plugin, Protocol;     │
        │               pipeline.duplex_runtime_extension, separate from the serving     │
        │               plugin pipeline.duplex_serving_adapter)                          │
        └──────────────────────────────────────────────────────────────────────────────┘
          ▲ OutputMessage per request_id  -> AsyncOmni._final_output_handler -> request_states[rid].queue
          ▲ DuplexSessionLifecycleMessage -> AsyncOmni.duplex_lifecycle_events -> handler lifecycle listener
```

Class relationships in the baseline (one class per layer, duplex members mixed in):

```text
OmniBase                              AsyncOmniEngine (single class)         Orchestrator (single class)
 ├─ Omni        (sync, turn)           turn-based requests                    turn-based admission
 └─ AsyncOmni   (EngineClient)         + 13 duplex methods                    + DuplexControlPlane, duplex arms
     turn-based requests               + DuplexControlClient                  + _OrchestratorDuplexStagePort
     + 7 duplex proxies                                                       + lease reaper task
     + DuplexRequestClient

Model plugin = two dotted paths per pipeline, bound only by convention:
  duplex_runtime_extension  -> DuplexRuntimeExtension (Protocol, engine side)
  duplex_serving_adapter    -> ServingRuntimeAdapter  (Protocol, serving side)
```

Compared with §2.1: the session is described twice (state machines #1 and #2)
and kept in sync by the fence protocol across the RPC boundary; the serving
handler owns the domain; the generic classes carry the duplex members; and
every model needs two plugins.

Three model plugins use this path: MiniCPM-o 4.5, PersonaPlex, Nemotron
VoiceChat. Each pipeline sets `duplex_control_enabled=True`,
`duplex_runtime_extension=<engine plugin>`, `duplex_serving_adapter=<serving
plugin>`, and their deploy YAMLs set `session_mode: duplex`.

### 1.2 Problems (confirmed by code reading)

| # | Problem | Evidence |
| --- | --- | --- |
| P1 | Serving layer owns the duplex *domain*: session registry, admission, `DuplexSession` transitions, append ordering chain, silence continuation, response lifecycle, playback/history commit, engine open/append/signal/close bridging, model-adapter loading, lifecycle expiry handling. | `serving.py` (`_open_session`, `_apply_runtime_lifecycle`, `_overlap_decision`, `_handle_playback_ack`, `_apply_session_update`, `_cancel_active_response`...), `session_runner.py::handle_session` (2000-line closure), `runtime_bridge.py` (all engine calls). |
| P2 | No complete Python API. `AsyncOmni` only exposes 7 fenced low-level proxies; everything above them lives in serving, so a Python user cannot run a duplex session without a WebSocket server. | `async_omni.py:288-460`. |
| P3 | Generic classes carry duplex logic. `Orchestrator`: 130 duplex mentions, 5 duplex branches inside `_route_output`, duplex block bracketing `_cleanup_request_ids`, duplex fields on `OrchestratorRequestState`, duplex arm in `_request_handler`, reaper task in `run()`. `AsyncOmniEngine`: 13 duplex methods (259 lines), duplex config plumbing, extension loading. `AsyncOmni`: proxies, `DuplexRequestClient`, `duplex_lifecycle_events`, lifecycle branch in the output loop. | See §2.4, §2.5. |
| P4 | Redundant surfaces: `/v1/duplex` route, `extra_body.native_duplex` (+ alias, query params), `ChatFallbackProjectorMixin`, `PipelineConfig.duplex_control_enabled`. All three duplex models are native; PersonaPlex/Nemotron already ignore the flag. | `api_server.py:1747`, `protocol.py:35-52`, `chat_fallback.py`, `minicpmo_4_5/duplex/adapter.py:49`, `stage_config.py:317-320`. |
| P5 | Mixin host-method coupling: `DuplexSessionRunnerMixin` calls ~55 host methods defined in `serving.py`; `handle_session` is one 2040-line method with 21 `nonlocal`-capturing closures; nothing inside is independently callable or testable. | `serving.py:90-93`, `session_runner.py:56-2097`. |
| P6 | The `/v1/duplex` vs Realtime distinction (`realtime_protocol is None`) is threaded through ~40 conditionals and changes *domain* behavior; the Realtime protocol object is persisted per session and required on resume (`session_runner.py:690`). | `serving.py:305,321,362,371,902`, `session_runner.py:706,950`. |
| P7 | **Two session state machines.** `DuplexSession` (API side: config, epoch/turn, input/response/playback/history ledgers) and `DuplexSessionRuntimeState` (engine side: fence, lease, stage bindings, append reservation, idempotency) describe the same session; the `DuplexFence` protocol, `next_fence`, `expected_epoch`, `operation_id` idempotency, request-id preregistration and the `collect_outputs` hop exist only to keep the two in sync across the RPC boundary. A third per-session object (`ServingRuntimeSessionState`, the model adapter's `native` state) sits beside them. | `protocol.py:416`, `engine/duplex/session.py`, `runtime_adapter.py:78-106`, `duplex_request_client.py:106-175`. |
| P8 | Two dotted plugin paths per model (`duplex_runtime_extension`, `duplex_serving_adapter`) with no descriptor binding them; a mismatch is not detected. | `stage_config.py:312-316`, `docs/design/fullduplex.md` "Plugin descriptor". |

### 1.3 What must be preserved (validated invariants)

From `docs/design/fullduplex.md`; the refactor changes ownership, not
semantics:

- one ordered inbound mailbox per session; a later close/cancel never
  overtakes an earlier append or playback ACK;
- exactly one session transition owner (no second reducer / state machine);
- irreversible cancellation: a cancel advances the session epoch atomically
  with releasing/aborting the stage bindings it owned, and a late append with
  the old epoch is permanently rejected;
- prepare/submit/commit append ordering with PCM reservation rollback;
- terminal-event acceptance before domain effects; late stale filtering for
  streaming deltas;
- response-scoped `ResponseCreateOptions`;
- the normative Realtime event contract table (public wire vocabulary);
- engine-owned admission, lease TTL, disconnect grace, resume/takeover.

---

## 2. Target architecture

### 2.1 Overview

```text
              Python users                                WebSocket clients
        ┌──────────────────────┐                    ┌────────────────────────┐
        │  InlineDuplexClient  │                    │      DuplexClient      │
        │  (DuplexClientBase)  │                    │   (DuplexClientBase)   │
        └──────────┬───────────┘                    └───────────┬────────────┘
                   │   DuplexSessionHandle                      │ /v1/realtime?duplex=1 (JSON)
                   │                                            ▼
                   │                          ┌──────────────────────────────────────┐
                   │                          │ OmniDuplexSessionHandler (thin)      │
                   │                          │  websocket I/O, DuplexCommand.from_  │
                   │                          │  realtime(), event.to_realtime(),    │
                   │                          │  attachment / resume tokens / replay │
                   │                          └──────────────────┬───────────────────┘
                   ▼                                             ▼   DuplexSessionHandle
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ DuplexOmni(AsyncOmniBase)                  [entrypoints/duplex_omni.py, thin] │
        │   open_session / get_session / resume_session / close_session                │
        │   DuplexSessionHandle: submit(DuplexCommand) / events()                      │
        └──────────────────────────────────────┬───────────────────────────────────────┘
                                               │ engine.open/close/resume/touch_session (RPC), submit_command (one-way)
                                               ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ DuplexOmniEngine(OmniEngineBase)  [engine/duplex_omni_engine.py, thin]        │
        │   message construction + queue put/RPC; creates DuplexOrchestrator            │
        │   (OmniEngineBase: stage init, orchestrator thread, queues, RPC router)       │
        │  └─ DuplexOrchestrator(OrchestratorBase)        [engine/duplex_orchestrator.py]│
        │       stage forwarding/prewarm/cleanup with session-owned policy              │
        │       ├─ DuplexSessionManager      admission, lease reaper, open/close/resume, │
        │       │                            dispatch of commands to runners            │
        │       └─ DuplexSessionRunner ×N    ONE session state (DuplexEngineSession),   │
        │                                    ordered mailbox, append planning + stage   │
        │                                    submit, data-plane decisions, response /   │
        │                                    playback / history, VAD, typed events      │
        │       plugin: DuplexModelPlugin    (pipeline.duplex_plugin, one class/model)  │
        └──────────────────────────────────────────────────────────────────────────────┘
```

Class relationships:

```text
OmniBase                        OmniEngineBase                       OrchestratorBase
 ├─ Omni        (sync, turn)     ├─ AsyncOmniEngine  (turn requests)  ├─ Orchestrator        (turn-based admission)
 └─ AsyncOmniBase                └─ DuplexOmniEngine (session msgs)   └─ DuplexOrchestrator  (hosts DuplexSessionManager)
     ├─ AsyncOmni   (EngineClient)  _create_engine -> AsyncOmniEngine   _create_orchestrator -> Orchestrator
     └─ DuplexOmni                  _create_engine -> DuplexOmniEngine  _create_orchestrator -> DuplexOrchestrator
```

Each layer has one turn-based and one duplex sibling over a shared base
(`*Omni` = Python API facade, `*OmniEngine` = stage-process ownership and
transport, `*Orchestrator` = stage management). Each concrete class
constructs its own collaborator in an explicit factory method the base calls
at the right moment: `OmniBase.__init__` calls `self._create_engine(...)`
after it has built the metrics objects the engine needs;
`OmniEngineBase._bootstrap_orchestrator` calls `self._create_orchestrator(...)`
on the orchestrator thread after the stages are initialized. No class
attributes, no generic kwargs smuggled through for the duplex case:
`DuplexOmniEngine._create_orchestrator` passes the loaded plugin and the
session runtime config to `DuplexOrchestrator` directly.

Design rules:

1. **One session authority.** The whole duplex session (config, epoch/turn,
   ledgers, lease, stage bindings, model state, projection ids) is one object,
   `DuplexEngineSession`, owned by one `DuplexSessionRunner` on the
   orchestrator loop. Nothing above the engine keeps session state beyond a
   handle (session id, event queue).
2. **Typed contract.** Commands into a session are `DuplexCommand` dataclasses;
   outputs are `DuplexEvent` dataclasses. Both carry `from_realtime()` /
   `to_realtime()` so the OpenAI Realtime JSON is derived, never hand-built,
   and the same conversion serves the websocket handler and the inline client.
3. **Generic bases and turn-based siblings have zero duplex vocabulary.**
   `OmniBase`, `AsyncOmniBase`, `AsyncOmni`, `OmniEngineBase`, `AsyncOmniEngine`,
   `OrchestratorBase`, `Orchestrator` gain only template seams.
4. **Duplex models are served duplex-only** (§7 D9): `vllm-omni serve`
   constructs `DuplexOmni` when the pipeline declares `duplex_plugin`, exposes
   `/v1/realtime?duplex=1`, `/v1/models`, `/health`; other routes report
   unavailable. Turn-based Python use (`Omni`/`AsyncOmni`) of the same model
   stays available offline.
5. **Serving is transport only**: websocket I/O, wire-envelope validation,
   `DuplexCommand.from_realtime`, `event.to_realtime`, attachment / resume
   tokens / replay journal.
6. **No `typing.Protocol`** in the new duplex surfaces; plugin and client
   seams are ABCs.

> **Breaking change (D9): MiniCPM-o 4.5 loses `/v1/chat/completions`.**
> On the current branch a MiniCPM-o 4.5 server started with its default
> profile (`deploy/minicpmo_4_5.yaml`, `session_mode: duplex`) serves both
> `/v1/chat/completions` and `/v1/realtime?duplex=1` from one engine; the
> profile header says so and `tests/e2e/online_serving/test_minicpmo_4_5.py`
> plus the `examples/online_serving/minicpmo` chat scripts depend on it.
> After this refactor `vllm-omni serve openbmb/MiniCPM-o-4_5` is a
> duplex-only server: chat completions, speech, batch and every other
> turn-based HTTP route answer "not available". Turn-based use of MiniCPM-o
> remains possible only through the offline Python API (`Omni` /
> `AsyncOmni`). PersonaPlex and Nemotron VoiceChat (duplex overlay) never
> supported chat completions, so they lose nothing. The affected tests,
> examples and docs are listed in §2.7 and §6.1; the release notes must
> carry this entry.

### 2.2 Why engine-resident sessions (alternative A applied)

- Collapses P7: `DuplexSession` + `DuplexSessionRuntimeState` +
  `ServingRuntimeSessionState` + projector state -> `DuplexEngineSession`.
  The `DuplexFence` becomes an internal identity value (session_id,
  epoch, turn_id) used for stage request ids and stale-output
  filtering. The cross-boundary fence *protocol* disappears (`DuplexFence`
  in messages and results, `next_fence` handshake, `accepted_fence`,
  `operation_id` idempotency cache, `runtime_contract_invalid`) because there
  is no second copy to reconcile. Epoch re-validation after an
  `await` inside the runner stays (§3.8); it is a local check, not a protocol.
- Data-plane outputs no longer take the request-state preregistration and
  `collect_outputs` hop: `DuplexOrchestrator._intercept_stage_output` hands
  each Stage0/Stage1 output for a session-owned request straight to its
  runner, on the loop where it arrived.
- Cancellation becomes one atomic step on one loop: advance epoch, abort
  owned stage bindings, drop queued stale appends.
- The mailbox is the control plane's existing per-session serialized task
  chain (`_session_control_tails`), made explicit as one `asyncio.Queue` +
  worker task per session; ordering is preserved because `DuplexOmniEngine`
  puts commands on the single engine request queue in caller order and
  `_request_handler` is ordered.
- `DuplexOmni` shrinks to a pipe (§3.1); the inline client and the websocket
  handler are both trivially thin.
- Cost to control: audio decoding/resampling, Silero VAD and base64 encoding
  of output audio now run on the orchestrator thread. The runner offloads
  these to `loop.run_in_executor` (default thread pool) and awaits them inside
  the command handler, so per-session order holds and stage forwarding is not
  blocked. Budget: no synchronous CPU work > 1 ms on the orchestrator loop;
  verify with the existing `orchestrator_monitor` stall metrics on H20.

### 2.3 Why a single model plugin (alternative C applied)

`DuplexModelPlugin` (ABC, `engine/duplex/plugin.py`) merges today's
`DuplexRuntimeExtension` (`configure_sampling_params`, `plan_append`,
`decide_output`) and `ServingRuntimeAdapter` (`capabilities`,
`validate_client_extra_body`, `prepare_runtime_config`,
`runtime_config_for_update`, `create_session_state`, data-plane projection
context and `project`). Both halves now execute engine-side, so one class per
model is the natural unit: `MiniCPMO45DuplexPlugin`, `PersonaPlexDuplexPlugin`,
`NemotronVoiceChatDuplexPlugin`. `PipelineConfig.duplex_plugin: str | None`
replaces `duplex_runtime_extension`, `duplex_serving_adapter` and
`duplex_control_enabled`; "is a duplex model" == `duplex_plugin is not None`.
Startup validation (required callables, per-stage sampling-param parity) moves
to `DuplexSessionManager.__init__`.

### 2.4 `OrchestratorBase` / `Orchestrator` / `DuplexOrchestrator`

Files: `engine/orchestrator.py` keeps `OrchestratorBase` and `Orchestrator`;
new `engine/duplex_orchestrator.py`.

`OrchestratorBase` (generic; most of today's `Orchestrator` moves here
unchanged): constructor and metrics state, `run()`, `_request_handler`
skeleton, abort, collective RPC, the output loops, stage error and
dead-replica handling, `_dispatch_or_fail_request`, `_cleanup_request_ids`,
raw terminal finishing, `_route_output`, next-stage request building,
`_forward_to_next_stage*`, `_prewarm_async_chunk_stages`, PD decode params,
`_shutdown_stages`. Seams:

| Seam (in `OrchestratorBase`) | Default | `Orchestrator` | `DuplexOrchestrator` |
| --- | --- | --- | --- |
| `_dispatch_message(msg) -> bool` called by `_request_handler` before the shared abort/collective/shutdown arms | `False` | `add_request`, `streaming_update`, `add_companion_request`, `interaction` handlers | open/command/close/resume/touch messages -> `DuplexSessionManager` |
| `_background_tasks() -> list[Coroutine]` gathered in `run()` | `[]` | — | `[manager.reaper_loop()]` (orchestrator.py:719-732) |
| `async _shutdown_extensions()` in `run()` `finally`, after task cancel, before membership drain | no-op | — | `await manager.shutdown()` (:657) |
| `_on_stage_submitted(stage_id, request_id, replica_id, req_state)` after every successful `submit_initial/submit_update` | no-op | — | bind stage request to the owning session (:1940-1955, call sites :2646, :2699, :2824, :2964) |
| `_intercept_stage_output(stage_id, output, metrics, req_state) -> bool` in `_route_output`, after per-stage metrics are computed and before client emission / forwarding | `False` | — | hands every session-owned output (Stage0 text/decision, Stage1 audio) plus its per-segment metric snapshot to `runner.on_stage_output(...)`; returns `True` when the base must **not** forward to the next stage (listen decision, terminal segment, Stage1 output), `False` when Stage0 output should be forwarded to TTS as today. Client emission is already suppressed by `session_owned`. Replaces `_duplex_output_decision` + `_emit_duplex_direct_output` (:2034-2044) and the deleted `collect_outputs` metric merge |
| `_handle_forward_failure(req_state, exc) -> bool` in the `_forward_to_next_stage_unguarded` except-block | `False` (re-raise) | — | fail the owning session's response, keep the loop alive (:2743-2758) |
| `OrchestratorRequestState.session_owned: bool = False` | no synthetic terminal (:1886), no client output on Stage0 segment end (:1997), `finished` = segment finished (:2010), no terminal re-forward (:2078), no auto-cleanup on finish (:2091) | never set | set by the runner's `ensure_request` |
| `_cleanup_request_ids(ids, *, abort=False, release_owners=False)` | ignores `release_owners`; error paths pass `True` | — | override brackets `super()` with `manager.close_sessions_for_request_ids` / `defer_request_cleanups` / `finalize_closed_sessions` (:1765-1841) |

`DuplexOrchestrator(OrchestratorBase)`:

- constructor `DuplexOrchestrator(*, plugin: DuplexModelPlugin, duplex_session_config:
  DuplexSessionRuntimeConfig, **generic_kwargs)` builds
  `DuplexSessionManager(plugin=plugin, stage_port=self, output_sink=self.output_async_queue,
  result_sink=self.rpc_async_queue, runtime_config=duplex_session_config, executor=...)`;
- implements the stage port directly (`stage_count`, `sampling_defaults`,
  `ensure_request`, `submit`, `cleanup`, `abort_request`), replacing
  `_OrchestratorDuplexStagePort` (:281-413);
- `DuplexOrchestratorRequestState(OrchestratorRequestState)` with the
  owning `session_key` and per-stage identity (replaces `duplex_identity`,
  `duplex_stage_fences`, `duplex_config_generation`, :226-228).

### 2.5 `OmniEngineBase` / `AsyncOmniEngine` / `DuplexOmniEngine`

Today's `AsyncOmniEngine` (2149 lines) is ~1200 lines of shared machinery,
~620 lines of turn-based request building and ~260 lines of duplex control.
It becomes:

| Class | Content |
| --- | --- |
| `OmniEngineBase` (`engine/omni_engine_base.py`, generic base) | config resolution, `_initialize_stages`, `_bootstrap_orchestrator` calling the abstract `_create_orchestrator(**generic_kwargs)`, queues + `RpcResultRouter` + `CorrelatedRpcClient`, `try_get_output*`, `get_output_blocking_async`, `get_stage_metadata`, `abort*`, `collective_rpc*`, `is_alive`, `shutdown`, `get_diffusion_od_config` |
| `AsyncOmniEngine(OmniEngineBase)` | `_create_orchestrator(**kw) -> Orchestrator(**kw)`; `add_request*`, `add_streaming_update*`, `_build_add_request_message`, CFG companions, multimodal UUID / replica cache scoping, `submit_interaction*` |
| `DuplexOmniEngine(OmniEngineBase)` (`engine/duplex_omni_engine.py`, thin) | `__init__` imports and instantiates the plugin from `pipeline_config.duplex_plugin` (validation against stage sampling defaults happens in `DuplexSessionManager.__init__`, which has the stage pools) and reads `deploy_config.duplex_session`; `_create_orchestrator(**kw) -> DuplexOrchestrator(plugin=self.plugin, duplex_session_config=self.duplex_session_config, **kw)`; the session message surface: `open_session_async(session_id, config) -> DuplexControlResult` (the id is allocated by `DuplexOmni`, §3.9), `close_session_async(session_id, reason)`, `resume_session_async(session_id, expected_lease_generation)`, `touch_session_async(session_id, activity)` (correlated RPC through the base's `CorrelatedRpcClient`; the blocking `_open_session` ... `_submit_command` bodies are private, only the `*_async` surface is public) and `submit_command_async(session_id, command)` (one-way `request_queue.put_nowait(DuplexSessionCommandMessage)`). `open_session_async` takes the already normalized `DuplexSessionConfig` object and `DuplexControlResult` carries `capabilities: DuplexCapabilities`, `public_session` (wire dict) and plain `error_code` / `error_message` / `error_retryable` fields: the queue is in-process, so nothing is serialized to dicts and re-parsed (D17). `duplex_session_config` and `duplex_capabilities` read from `deploy_config` / the loaded plugin |

`DuplexOmniEngine` is deliberately small (message construction and queue
access only); it exists so the engine layer has the same clearly named
turn/duplex pair as the API and orchestrator layers, and so the RPC client
never has to be exposed above the engine. `AsyncOmniEngine` loses all duplex
lines (async_omni_engine.py:37-40, :61-67, :211-220, :255, :392-405,
:1579-1837). `DuplexControlClient` is not kept: its only remaining job
(build a message, run it through the correlated RPC client, map a failed
result to an exception) is exactly `DuplexOmniEngine`'s method bodies, so
`control_client.py` is deleted and `DuplexControlRequestError` is renamed
`DuplexSessionError` (`engine/duplex/messages.py`), the exception
`DuplexOmni` raises to callers.

### 2.6 `AsyncOmniBase` / `AsyncOmni` / `DuplexOmni`

`AsyncOmniBase(OmniBase)` (`entrypoints/async_omni_base.py`, extracted from
`AsyncOmni`): nothing engine-selection related (each concrete class
implements `OmniBase._create_engine`); the async output pump (`_final_output_handler`, loop
skeleton at async_omni.py:929-1049) with seam `_route_engine_message(msg) ->
bool` before ACK / error / output handling; `AsyncEventResolver`;
engine-dead fan-out; `is_running`, `errored`, `check_health`, `dead_error`,
`shutdown`; `vllm_config` / `model_config` / `renderer` accessors.

`AsyncOmni(EngineClient, AsyncOmniBase)` keeps everything turn-based:
`generate`, streaming inputs, admission gate, `encode`, `abort`, interaction,
pause/sleep/wake, profiling, cache resets, LoRA, weight update,
`collective_rpc`, tokenizer/preprocessor accessors. It loses every duplex
line (:173-174, :181-182, :288-460, :980-982, imports).

`DuplexOmni(AsyncOmniBase)` is described in §3.1. It does not use
`request_states`, `_process_single_result` or request metrics (turn-request
concepts); those stay on `OmniBase` for `Omni`/`AsyncOmni`.

### 2.7 API server in duplex mode

`build_async_omni_from_stage_config` resolves the pipeline config
(`StageConfigFactory.get_pipeline_config(...)`, the same call `OmniEngineBase`
makes) and constructs `DuplexOmni` when `duplex_plugin` is set, `AsyncOmni`
otherwise. `DuplexOmni.__init__` validates `deploy_config.session_mode ==
"duplex"` (scheduler contract). `omni_init_app_state` gains a duplex branch
next to the pure-diffusion branch: `openai_serving_models` (for
`/v1/models`), `openai_serving_duplex = OmniDuplexSessionHandler(duplex_omni=engine_client)`,
everything else `None` (routes already answer "not available"). MiniCPM-o
4.5 is a duplex model, so its online chat tests/examples
(`tests/e2e/online_serving/test_minicpmo_4_5*.py`, online cases of
`tests/dfx/reliability/.../test_invalid_minicpmo_4_5_omni.py`,
`examples/online_serving/minicpmo/run_curl_multimodal_generation.sh`,
`run_gradio_demo.sh`, README chat sections) are removed or converted to the
duplex client; offline MiniCPM-o use is unaffected.

### 2.8 Package layout (§7 D7: stay in `entrypoints/duplex/`)

Annotations: `[A]` engine-resident sessions, `[B]` typed contract, `[C]` single model plugin,
`(moved)` from `entrypoints/duplex/`, `(new)` file that did not exist before the refactor.

```text
vllm_omni/
├── entrypoints/
│   ├── omni_base.py                 OmniBase (+ abstract _create_engine)
│   ├── async_omni_base.py           AsyncOmniBase                                (new)
│   ├── async_omni.py                AsyncOmni (turn-based requests)
│   ├── duplex_omni.py               DuplexOmni, DuplexSessionHandle (thin pipe)  (new) [A]
│   ├── duplex/                      SERVING (thin)
│   │   ├── serving.py               OmniDuplexSessionHandler
│   │   ├── realtime_input.py        wire-envelope validation, resume request parsing
│   │   ├── session_attachment.py    DuplexSessionAttachmentRegistry (resume tokens, replay journal)
│   │   └── websocket.py             websocket send/close/receive helpers
│   │   DELETED: chat_fallback.py, realtime_session.py, realtime_state.py, realtime_output.py,
│   │            runtime_adapter.py, runtime_bridge.py, session_runner.py, protocol.py, capability.py
│   ├── duplex_request_client.py     DELETED
│   └── openai/
│       └── api_server.py            builds DuplexOmni for duplex models; /v1/realtime?duplex=1 only
│
├── engine/
│   ├── omni_engine_base.py          OmniEngineBase (generic base; _validate_deployment seam)  (new)
│   ├── async_omni_engine.py         AsyncOmniEngine (turn-based requests)
│   ├── duplex_omni_engine.py        DuplexOmniEngine (session message surface, creates DuplexOrchestrator)  (new)
│   ├── orchestrator.py              OrchestratorBase + Orchestrator
│   ├── duplex_orchestrator.py       DuplexOrchestrator (+ DuplexOrchestratorRequestState; implements DuplexStagePort)  (new)
│   └── duplex/
│       ├── commands.py              DuplexCommand dataclasses (+ payload())                    (new) [B]
│       ├── realtime_commands.py     DuplexCommand.from_realtime translation of client events   (new) [B]
│       ├── events.py                DuplexEvent dataclasses (+ to_realtime())                 (new) [B]
│       ├── realtime_events.py       RealtimeProjectionState: internal event -> typed events    (new) [B]
│       ├── messages.py              queue envelopes: OpenDuplexSession, CloseDuplexSession, ResumeDuplexSession,
│       │                            TouchDuplexSession, DuplexSessionCommand, DuplexControlResult, DuplexSessionEvent
│       ├── config.py                DuplexSessionConfig, DuplexCapabilities, ResponseCreateOptions  (new; from protocol.py)
│       ├── contracts.py             DuplexFence (session_id, epoch, turn_id, incarnation), DuplexOutputAction,
│       │                            stage request/submission records, DuplexStagePort ABC (trimmed)
│       ├── session.py               DuplexEngineSession: ledgers, lease, fence, stage request resources,
│       │                            append sequencing, model/projection state, signal_turn()  [A]
│       ├── session_runner.py        DuplexSessionRunner (per-session mailbox on the orchestrator loop)   (new) [A]
│       ├── session_manager.py       DuplexSessionManager (admission, backpressure, reaper, dispatch)    (new) [A]
│       ├── plugin.py                DuplexModelPlugin, DuplexModelSessionState, DuplexDataPlane,
│       │                            PcmAppendBuffer, PcmAppendReservation ABCs; load_duplex_plugin      (new) [C]
│       ├── lease.py                 DuplexLeaseState (disconnect grace, idle TTL)
│       ├── intermediate.py          stage-to-stage intermediate buffer helpers
│       ├── turn_detection.py        server-side VAD turn detector used by the runner            (new) [A]
│       ├── audio.py                                                                             (moved)
│       ├── vad.py                                                                               (moved)
│       └── commit_policy.py                                                                     (moved)
│       DELETED: control_client.py, control_plane.py, runtime.py
│
├── config/
│   └── stage_config.py              PipelineConfig.duplex_plugin; DuplexSessionRuntimeConfig stays here  [C]
│
├── model_executor/models/
│   ├── minicpmo_4_5/
│   │   ├── pipeline.py              duplex_plugin="...minicpmo_4_5.duplex.plugin.MiniCPMO45DuplexPlugin"
│   │   └── duplex/
│   │       ├── plugin.py            MiniCPMO45DuplexPlugin (engine policy + session policy in one class)  [C]
│   │       ├── capabilities.py, data_plane.py, input.py, policy.py, session.py, stage0.py, compat.py
│   │       DELETED: adapter.py, runtime.py, serving_adapter.py
│   ├── personaplex/
│   │   ├── pipeline.py              duplex_plugin="...personaplex.duplex.plugin.PersonaPlexDuplexPlugin"
│   │   └── duplex/
│   │       ├── plugin.py            PersonaPlexDuplexPlugin  [C]
│   │       ├── config.py, data_plane.py, input.py, policy.py, stage0.py
│   │       DELETED: runtime_extension.py, serving_adapter.py
│   └── nemotron_voicechat/
│       ├── pipeline.py              duplex_plugin="...nemotron_voicechat.duplex.plugin.NemotronVoiceChatDuplexPlugin"
│       └── duplex/
│           ├── plugin.py            NemotronVoiceChatDuplexPlugin  [C]
│           ├── data_plane.py, input.py
│           DELETED: runtime.py, serving_adapter.py
│
└── clients/
    ├── duplex.py                    DuplexClientBase (ABC), DuplexClient, WebSocketTransport (ABC),
    │                                client-side events, ResponseHandle, EventCollector
    ├── inline_duplex.py             InlineDuplexClient (in-process, over DuplexOmni)  (new)
    └── minicpmo_4_5.py              MiniCPM-o 4.5 session preset (native_duplex toggle removed)
```

---

## 3. API design

### 3.1 `DuplexOmni` (thin)

```python
class DuplexOmni(AsyncOmniBase):
    """Async Python API for full-duplex models. Sessions run inside the engine;
    this class opens/closes them and pipes commands and events.

    Example:
        >>> omni = DuplexOmni(model="openbmb/MiniCPM-o-4_5")
        >>> async with await omni.open_session(DuplexSessionConfig(ref_audio=...)) as session:
        ...     async def consume():
        ...         async for event in session.events():
        ...             if isinstance(event, AudioDelta): play(event.audio)
        ...     task = asyncio.create_task(consume())
        ...     await session.submit(AppendAudio(pcm, format="pcm16", sample_rate_hz=16000))
        ...     await session.submit(Commit())
    """

    def _create_engine(self, **engine_kwargs) -> DuplexOmniEngine:
        return DuplexOmniEngine(**engine_kwargs)          # which creates DuplexOrchestrator

    @property duplex_session_config -> DuplexSessionRuntimeConfig      # engine.duplex_session_config
    @property duplex_capabilities -> DuplexCapabilities                # engine.duplex_capabilities
    @property sessions -> Mapping[str, DuplexSessionHandle]

    async def open_session(self, config: DuplexSessionConfig | Mapping | None = None, *,
                           timeout: float | None = 10.0) -> DuplexSessionHandle
        # 1. session_id = f"duplex-{uuid4().hex}"  (ALWAYS allocated here, §3.9; callers cannot choose it;
        #    a "session_id" key inside a Mapping config is ignored)
        # 2. register a pending handle under session_id so no event can arrive before it exists
        # 3. engine.open_session_async(session_id, config: DuplexSessionConfig)
        #    -> DuplexControlResult(capabilities: DuplexCapabilities, public_session, error_*)
        #    on failure: drop the pending handle, raise DuplexSessionError(code) with today's codes
        #    (resource_exhausted, invalid_duplex_runtime_config, ...)
        # 4. the runner's first emitted event is SessionCreated (carries the allocated session id)
    def get_session(self, session_id: str) -> DuplexSessionHandle | None
    async def resume_session(self, session_id: str, *,
                             expected_lease_generation: int) -> DuplexSessionHandle
        # engine.resume_session_async (lease CAS); returns the EXISTING handle, whose events() may be
        # re-entered by the new consumer (buffered events are delivered in order)
    async def detach_session(self, session_id: str) -> None   # engine.touch_session_async(DETACH): starts the
                                                              # engine-owned disconnect grace; the API-side grace
                                                              # callback (`cancel_orphan_response_after_grace`) is
                                                              # dropped, expiry arrives as SessionExpired
    async def close_session(self, session_id: str, *, reason: str = "client_close") -> None   # engine.close_session_async
    async def close_all_sessions(self, *, reason: str = "shutdown") -> None
    def shutdown(self, timeout: float | None = None) -> None

    def _route_engine_message(self, msg) -> bool
        # DuplexSessionEventMessage -> handle.outbox; session.closed/expired -> drop handle
```

Estimated size: about 300 lines. It holds no session state beyond the handle
registry.

### 3.2 `DuplexSessionHandle`

```python
class DuplexSessionHandle:
    session_id: str                      # server-allocated, unique for the engine's lifetime (§3.9)
    capabilities: DuplexCapabilities
    @property closed -> bool

    async def submit(self, command: DuplexCommand) -> None
        # one-way: engine.submit_command_async(session_id, command)
        # caller order == mailbox order; rejections come back as ErrorEvent on events()
    # convenience wrappers, all == submit(...):
    async def append_audio(...), append_text(text), commit(*, final=True, create_response=None, is_speech=None),
          create_response(options=None), clear_input(), cancel_input(), cancel_response(response_id=None),
          barge_in(), clear_output_audio(), signal_turn(event, payload=None), update(session_patch),
          ack_playback(played_ms, *, response_id=None, item_id=None, committed_ms=None), heartbeat(),
          create_item(item, *, previous_item_id=None), delete_item(item_id), truncate_item(item_id, *, audio_end_ms)
    async def close(self, *, reason="client_close") -> None      # RPC via DuplexOmni.close_session; awaits SessionClosed

    def events(self) -> AsyncIterator[DuplexEvent]
        # single consumer (§7 D8); ends after SessionClosed / SessionExpired
    async def wait_closed(self) -> str
    async def __aenter__ / __aexit__
```

Backpressure (`max_pending_input_bytes_per_session`,
`max_pending_turns_per_session`) is enforced in `DuplexSessionManager.dispatch`
on the orchestrator loop, *before* the command is put on the session mailbox,
with the existing byte / pending-turn accounting; a rejected command is
dropped and answered with `ErrorEvent(code="input_backpressure")`. Bytes never
accumulate in an unbounded queue: the engine request queue is drained by the
ordered `_request_handler`, and the mailbox only ever holds admitted bytes.
`events()` may be re-entered after a previous iterator was closed (resume);
only one iterator may be active at a time.

### 3.3 Typed contract (alternative B applied)

`engine/duplex/commands.py`:

```python
@dataclass(frozen=True, slots=True)
class DuplexCommand:
    event_id: str | None = None                              # client correlation id, echoed on errors
    @classmethod
    def from_realtime(cls, payload: Mapping[str, object]) -> "DuplexCommand"   # dispatch on payload["type"]
    def to_realtime(self) -> dict[str, object]

AppendAudio(audio: bytes, format: str, sample_rate_hz: int, is_speech: bool | None, video_frames: tuple[str, ...],
            duration_ms: int | None, audio_end_ms: int | None)
AppendText(text)   Commit(final, create_response, is_speech)   CreateResponse(options: ResponseCreateOptions)
ClearInput()   CancelInput()   CancelResponse(response_id)   BargeIn()   ClearOutputAudio()
SignalTurn(event, payload)   UpdateSession(patch: Mapping)   AckPlayback(played_ms, response_id, item_id, committed_ms)
Heartbeat()   CreateItem(item, previous_item_id)   DeleteItem(item_id)   TruncateItem(item_id, audio_end_ms)
```

`engine/duplex/events.py`:

Stateless events (`error`, `session.expired`, `session.heartbeat_ack`,
`overlap.decision`, `input_audio_buffer.cleared`, `playback.acknowledged`,
`turn.event`) are constructed typed at the runner's emit site
(`_emit_error`, `session.signal_turn()`). Only events with domain effects
(`response.done`, `response.listen`, `audio.cancelled`, `input.cancelled`,
`session.closed`) or Realtime projection state (response / item / content
part bookkeeping, model output) still travel as internal dictionaries through
`DuplexSessionRunner.emit` and `realtime_events.project_internal_event` (D17).

```python
@dataclass(frozen=True, slots=True)
class DuplexEvent:
    session_id: str; epoch: int; event_id: str
    def to_realtime(self) -> list[dict[str, object]]           # 0..n wire events (e.g. ResponseDone -> output_audio.done,
                                                               # output_item.done, response.done, rate_limits.updated)
SessionCreated(public_session, resume_supported)   SessionUpdated   SessionResumed   SessionClosed(reason)   SessionExpired(reason)
ResponseCreated(response_id, item_id, metadata)   Speak(response_id, decision_metadata)   Listen(reason, model_turn_id)
AudioDelta(response_id, item_id, audio: bytes, transcript: str, sample_rate_hz)   TextDelta   AudioDone   TranscriptDone
ResponseDone(response_id, status, status_details, committed, playback, stage_metrics)
InputCommitted(item_id, deferred)   InputCancelled   InputCleared   SpeechStarted(audio_start_ms)   SpeechStopped(audio_end_ms)
AudioCancelled(response_id, reason, cancelled_epoch, committed_ms, playback)   PlaybackAcknowledged
ItemCreated / ItemDeleted / ItemTruncated   FunctionCallDone   RuntimeControl(redacted)   ErrorEvent(code, message, related_event_id)
```

The normative table in `docs/design/fullduplex.md` becomes the docstrings of
these classes plus a contract test that round-trips every `to_realtime()`
output against the catalogue in `docs/serving/realtime_duplex_api.md`. The
Realtime projection state (response/item ids, content-part bookkeeping from
`realtime_output.py` + `_RealtimeResponseState`) lives in the runner and is
consumed when constructing events; `to_realtime()` itself is pure.

Client side: `vllm_omni/clients/duplex.py` keeps its JSON-based `DuplexEvent`
wrapper classes (they are the *wire* view); `InlineDuplexClient` feeds
`event.to_realtime()` into the same `_dispatch`. Both clients build wire JSON
for commands; the inline client converts it with `DuplexCommand.from_realtime`,
exactly what the websocket handler does.

### 3.4 Engine-side session internals (not public API)

```python
class DuplexEngineSession:            # engine/duplex/session.py — the ONE session state (no resources sub-object)
    session_id, epoch, turn_id, state, turn_state, config: DuplexSessionConfig, capabilities
    runtime_config, config_generation                                       # from DuplexSessionRuntimeState
    lease: DuplexLeaseState, request_resources[(stage_id, request_id)], accepted_fence (monotonic high-water
    mark for stage requests), input_seq / input_turn_seq                    # from DuplexSessionRuntimeState
    touch_lease / detach_lease / resume_lease / begin_close / release_all_requests / prepare_append / commit_append
    signal_turn(event, payload) -> TurnEvent                                # was DuplexTurnController
    _input / _response / _playback / _conversation ledgers                  # from DuplexSession
    model_state: DuplexModelSessionState                                    # from ServingRuntimeSessionState
    projection: ResponseProjectionState                                     # from _RealtimeResponseState
    # transition methods: unchanged bodies from protocol.py (begin_response, end_response, mark_audio_sent,
    # acknowledge_playback, truncate_history_item, barge_in, commit_native_audio_input, ...) plus the fence /
    # binding / reservation methods from engine/duplex/session.py

class DuplexSessionRunner:            # engine/duplex/session_runner.py — one per session, on the orchestrator loop
    session: DuplexEngineSession; plugin: DuplexModelPlugin; stage_port; manager
    mailbox: asyncio.Queue[DuplexCommand | _Internal]; worker: asyncio.Task
    tasks: append tail / active response / pending silence           # from DuplexSessionTasks
    async run()                                                       # mailbox loop; one _on_<Command>() per command
    def on_stage_output(stage_id, output, metrics) -> bool           # enqueues _StageOutput on the mailbox; returns the
                                                                      # forward decision for the base (§2.4); handler bodies
                                                                      # replace collect_outputs + _drain_native_data_plane_stream
                                                                      # + _send_one_native_duplex_event
    async _append(payload, *, final, precreate_response, ...)         # from start_native_append + control_plane.handle_append
                                                                      # + append_via_data_plane: plan_append -> stage_port.submit
    async _schedule_silence_continuation(...)                         # unchanged body
    def emit(event: DuplexEvent)                                      # terminal acceptance / stale filter -> output sink
    async _offload(fn, *args)                                         # run_in_executor for audio decode, VAD, base64

class DuplexSessionManager:           # engine/duplex/session_manager.py — from DuplexControlPlane + DuplexSessionRuntimeManager
    sessions: dict[str, DuplexSessionRunner]; admission (max_sessions, closing sessions retain capacity)
    async open(msg) / close(msg) / resume(msg) / touch(msg)          # RPC results via result_sink
    def dispatch(msg: DuplexSessionCommandMessage)                    # session lookup (unknown_session), byte / pending-turn reservation
    # (no DuplexRuntimeCapabilities / input-mode conversion: every duplex model appends audio chunks, D17)
                                                                      # -> runner.mailbox.put_nowait (or ErrorEvent)
    executor: ThreadPoolExecutor                                      # dedicated pool for runner _offload work
    async reaper_loop() / reap_expired()                              # lease expiry -> SessionExpired event + cleanup
    close_sessions_for_request_ids / defer_request_cleanups / finalize_closed_sessions   # unchanged
    async shutdown()

class DuplexModelPlugin(ABC):         # engine/duplex/plugin.py
    plugin_id, private_runtime_config_keys, clean_response_done_prefix, interrupted_tts_prefix, collect_outputs_on_append
    configure_sampling_params(...) / plan_append(...) / decide_output(...)          # from DuplexRuntimeExtension
    capabilities(max_sessions) / validate_client_extra_body(...) / prepare_runtime_config(config, model_config)
    runtime_config_for_update(config, current) / create_session_state() / data_plane: DuplexDataPlane
    # is_enabled / session_states / remove_session_state REMOVED
```

### 3.5 Serving (thin)

`OmniDuplexSessionHandler.handle_realtime_session(websocket)`:

1. accept; build `RealtimeEnvelope` (autostart / default session payload from
   query params, `event_id` correlation, error-type map, `conversation.item.*`
   echo events);
2. handshake: first message `session.update` -> `omni.open_session(payload["session"])`
   (any `session_id` / `id` the client puts in the payload is ignored; the
   server-allocated id is announced in `session.created`);
   `session.resume` (`session_id`, `resume_token`, `last_received_server_event_seq`)
   -> `attachment.authenticate_resume` -> `omni.resume_session`;
3. reader loop: JSON -> `DuplexCommand.from_realtime` -> `handle.submit`;
   envelope-level errors (invalid JSON, oversize frame, unknown type) are
   answered locally;
4. writer loop: `async for ev in handle.events(): for payload in ev.to_realtime(): attachment.send_event(payload)`;
5. disconnect: detach with grace (`attachment.detach`) and `Heartbeat`/detach
   touch; takeover and replay unchanged (`DuplexSessionAttachmentRegistry`).

Target size: serving.py < 400 lines, realtime_input.py < 300 lines.

### 3.6 `DuplexClientBase`, `DuplexClient`, `InlineDuplexClient`

```python
class DuplexClientBase(ABC):                              # vllm_omni/clients/duplex.py
    def __init__(self, *, model: str, config: SessionConfig | None = None)
    session_id: str | None        # None until session.created; assigned by the server, never chosen by the client
    # shared public API (unchanged signatures): __aenter__/__aexit__/close, append_audio, stream_pcm, commit,
    # cancel_response, clear_input, ack_playback, __aiter__/events/responses/wait_for, send(event: dict) -> str
    # shared internals: _dispatch, _adopt_session, _announce, subscribers, _wait_on_queue, _finalize
    # abstract: async _open(); async _send_command(payload: dict); async _teardown()

class DuplexClient(DuplexClientBase):                      # websockets; reconnect / resume / heartbeat / event_ack
class InlineDuplexClient(DuplexClientBase):                # vllm_omni/clients/inline_duplex.py
    def __init__(self, omni: "DuplexOmni", *, model, config=None, handshake_timeout_s=30.0)
    # _open(): handle = await omni.open_session(config.to_session_payload(model=...)); pump handle.events()
    #          -> for payload in ev.to_realtime(): self._dispatch(payload)
    # _send_command(payload): await handle.submit(DuplexCommand.from_realtime(payload))   (session.close -> handle.close())
```

`vllm_omni.clients.inline_duplex` imports `DuplexOmni` and `DuplexCommand`
lazily / under `TYPE_CHECKING` (the caller passes the `DuplexOmni` instance),
so `vllm_omni.clients.duplex` stays import-pure. `EventCollector.consume` and
`acknowledge_collected_playback` are retyped to `DuplexClientBase`;
`WebSocketTransport` becomes an ABC. `barge_in_client.py` gains `--inline`.

### 3.7 Concurrency and ordering model of the runner

Everything below runs on the orchestrator asyncio loop; there is no lock.

- **Single writer per session.** `DuplexEngineSession` is mutated only by
  its runner. Inputs reach the runner through one mailbox in this order:
  client commands (from `_request_handler`, in request-queue order), stage
  outputs (`on_stage_output` enqueues an internal `_StageOutput` item; it
  does not mutate the session inline), timers (silence continuation, bounded
  auto-response) and lease events (from the manager) as internal items. The
  mailbox worker processes items one at a time.
- **Slow work is not awaited inline.** An `AppendAudio` handler reserves
  bytes, then schedules a tracked append task exactly like today's
  `start_native_append`: the task awaits `_offload(decode/resample/VAD)`,
  then `plugin.plan_append`, then `stage_port.submit`, serialized on the
  per-session append tail. The worker returns to the mailbox immediately, so
  a later `CancelResponse` is not blocked behind an in-flight append.
- **Re-validation after every `await`.** A tracked task captures
  `(epoch, turn_id)` when it starts and re-checks them before
  the stage submit and before committing ledgers (today's `expected_epoch` /
  `_native_silence_continuation_is_stale` checks, moved verbatim). A task
  that finds the epoch advanced rolls back its PCM reservation and exits.
- **Cancel is atomic at the session.** The `CancelResponse` / `BargeIn`
  handler advances the epoch, cancels tracked append tasks
  (`DuplexSessionTasks.cancel_append_tasks`), aborts the owned stage bindings
  through the stage port and emits `AudioCancelled`, all before yielding.
  An append task already past its last check but still inside
  `stage_port.submit` is harmless: the stage request it creates is bound to
  the old epoch's request id, which the abort covers, and the base's
  `session_owned` policy never emits its outputs to a client.
- **Ordered event stream, no late stale filter needed above the engine.**
  Because the epoch bump and every emission happen on one loop in program
  order, any `AudioDelta` ahead of an `AudioCancelled` in the stream was
  emitted before the cancel and is legitimately ordered; nothing can be
  emitted for the old epoch after the terminal event. `runner.emit()` keeps
  the terminal-acceptance rule (a terminal carrying a stale epoch is dropped)
  and stamps `epoch` on every event for clients that filter defensively.
- **Executors.** Audio decode / resample / VAD / base64 use a dedicated
  `ThreadPoolExecutor` owned by the manager (not the default pool, which the
  RPC waits on the API side already use). Plugin hooks that may block
  (`prepare_runtime_config` resolving `ref_audio` from disk or URL) are
  awaited in `open()` on the manager and offloaded the same way.
- **Per-response stage metrics.** `_route_output` already computes the
  per-segment `StageMetrics` (`vllm_ttft_ms`, `vllm_tpot_ms`, `vllm_itls_ms`,
  segment-local `num_generation_tokens`) before calling
  `_intercept_stage_output`; the hook passes that snapshot to
  `on_stage_output`, and the runner accumulates it into the active response
  with today's `accumulate_response_stage_metrics` logic, so
  `ResponseDone.stage_metrics` and `metadata.vllm_omni.stage_metrics` on the
  wire are unchanged. This replaces the metric merge in
  `DuplexRequestClient.collect_outputs`.

### 3.8 Hard spots and how the design resolves them

| Hard spot | Resolution |
| --- | --- |
| `handle_session` is one method with 21 closures capturing `session`, `actor`, `native`, `runtime_closed`, ... | `DuplexSessionRunner` fields replace captured locals; each closure and each `if event_type == ...` block becomes a method. Bodies move; ordering is kept. |
| `emit_event` domain/transport cycle (deferred overlap promotion re-enters `start_native_append`). | `runner.emit()` only accepts/filters/projects; deferred promotion is an internal mailbox command. |
| `send_json` threaded through ~40 methods. | Runner methods call `self.emit(Event(...))`. |
| `realtime_protocol` as a domain input (P6). | Gone with `/v1/duplex`; the Realtime branch is the only branch. |
| Three session states (P7) and the fence protocol between them. | One `DuplexEngineSession`; fences become internal identity; no idempotency cache, `next_fence`, preregistration or `collect_outputs`; epoch re-validation after awaits stays (§3.7). |
| Realtime protocol object required on resume. | Projection state lives in the session; serving is stateless across resume. |
| Stale-output filter in the websocket writer. | Not needed above the engine: the event stream is emitted in program order on one loop (§3.7); `runner.emit()` keeps terminal acceptance. |
| Backpressure in the read loop. | In `DuplexSessionManager.dispatch`, before the mailbox put (§3.2). |
| Second request-admission path (`ensure_request`/`submit` bypass `_dispatch_or_fail_request`). | Kept inside `DuplexOrchestrator`'s stage port; reconciling is a follow-up. |
| Orchestrator loop taking CPU work (new, from A). | `_offload()` to a dedicated executor for audio decode/resample, VAD, base64; stall budget checked with `orchestrator_monitor` on H20. |
| Two disconnect-grace implementations (API attachment callback + engine lease). | Engine lease only: `detach_session` touches `DETACH`, expiry arrives as `SessionExpired`; `DuplexSessionAttachmentRegistry` keeps replay/takeover but no grace timer. `test_duplex_session_attachment.py` grace cases move to the lease tests. |
| Open race: first session event could reach `DuplexOmni` before the handle exists. | Session id generated API-side; pending handle registered before the open RPC (§3.1). |
| Ordering of one-way commands across the request queue (new, from A). | Single ordered engine request queue + ordered `_request_handler` + per-session mailbox; `close`/`resume` RPCs are enqueued on the same queue so they cannot overtake earlier commands. |

---

### 3.9 Session identity: server-allocated ids, no incarnation (§7 D16)

Rules:

1. **The server allocates every session id.** `DuplexOmni.open_session`
   generates `duplex-<uuid4 hex>`; there is no `session_id` parameter on
   `open_session`, on `DuplexClientBase.__init__`, or on `InlineDuplexClient`.
   A `session_id` / `id` key in a `session.update` open payload is ignored
   (Realtime clients echo the session object back, so rejecting it would
   break them). The allocated id reaches the client in `session.created`
   and is the only handle for `session.resume`, `close`, and every command.
2. **A session id is never reused** within an engine's lifetime, so the id
   alone identifies a session. The `incarnation` counter, which only existed
   to tell apart successive sessions opened under the same client-chosen id,
   is dropped everywhere: `DuplexFence` becomes `(session_id, epoch, turn_id,
   response_seq)`; stage request ids become `duplex-s.<id>.e.<epoch>.r.<role>`;
   model-side per-session state is keyed by `session_id` alone; events,
   messages and the attachment registry lose the field; the `stale_incarnation`
   error code disappears.
3. **Stale detection.** A command, close, touch or resume for an id the
   manager no longer holds is answered with `ErrorEvent(code="unknown_session")`
   (or `DuplexSessionError("unknown_session")` on the RPC path). Within a live
   session, staleness is still `epoch`-based (§3.7): appends and continuations
   captured under an older epoch roll back and exit.
4. **Resume identity** is `(session_id, resume_token, expected_lease_generation)`:
   the token is bound to the attachment state, the lease generation is the
   engine-side compare-and-swap. Neither needs an incarnation to be unique.
5. **Correlation.** Clients that need their own identifier keep it in
   `DuplexSessionConfig.metadata` (echoed in `public_session`), not in the id.

Wire changes: `session.created` no longer carries `incarnation`;
`session.resume` no longer requires it. Message changes:
`DuplexSessionCommandMessage(session_id, command)`,
`CloseDuplexSessionMessage(session_id, reason)`,
`ResumeDuplexSessionMessage(session_id, expected_lease_generation)`,
`TouchDuplexSessionMessage(session_id, activity)`, `DuplexControlResult`
without `incarnation`.

Expected effect: about 150 references across the engine, serving, attachment
registry, three model plugins and the sampler go away; one identity dimension
fewer to explain. The size reduction is modest (roughly 1-2% of the duplex
engine + serving layers); the gain is conceptual.

---

## 4. Removals

| Item | Where | Replacement |
| --- | --- | --- |
| `/v1/duplex` route, native dialect on the wire, dialect input aliases | `api_server.py:1747-1755`, `websocket.py:_INPUT_EVENT_ALIASES`, `serving.py:_open_session`, docs "Native Protocol", route tests | `/v1/realtime?duplex=1` only |
| `extra_body.native_duplex` / `minicpmo45_native_duplex` / query params, in client, server and model code (experimental tree untouched, §7 D4) | `protocol.py:35-52,299`, `realtime_input.py:757`, `realtime_state.py:141-150`, `serving.py:1196-1227,1384-1394`, `clients/minicpmo_4_5.py:16`, `clients/duplex.py:build_realtime_url`, `benchmarks/omniinteract.py:841,867`, `benchmarks/patch/patch.py:1540,1729,1738`, model adapters' `is_enabled()` / `validate_client_extra_body`, `minicpmo_4_5/duplex/{adapter,capabilities}.py`, minicpmo web demo JS, e2e helpers | native lane unconditional; MiniCPM `implementation_level="model_native_duplex"` unconditional |
| `ChatFallbackProjectorMixin` and the generic lane | `chat_fallback.py`, `append_tokens` path, non-native capabilities, `_uses_native_input_append` branches, `response.message`, `chat_service` dependency | deleted |
| `PipelineConfig.duplex_control_enabled`, `duplex_runtime_extension`, `duplex_serving_adapter` (§7 D3, D12) | `stage_config.py:312-320`, three `pipeline.py`, `async_omni_engine.py`, `orchestrator.py`, `tests/config/test_config_factory.py` | `PipelineConfig.duplex_plugin` |
| `entrypoints/duplex/capability.py` | `api_server.py:1159` | `duplex_plugin is not None` check |
| **`/v1/chat/completions` (and all other turn-based HTTP routes) on MiniCPM-o 4.5 servers** (§7 D9, breaking change) | `api_server.py` chat/speech/batch wiring for duplex deployments, MiniCPM-o online chat tests/examples (§2.7), deploy YAML header comments | duplex client over `/v1/realtime?duplex=1`; offline `Omni`/`AsyncOmni` for turn-based use |
| Cross-boundary fence protocol: `DuplexFence` in messages/results, `next_fence`, `expected_epoch`, `operation_id` idempotency cache, `accepted_fence`, `runtime_contract_invalid`, request-state preregistration, `collect_duplex_data_plane_outputs`, `DuplexRequestClient`, `DuplexEnginePort`, `DuplexRequestOutputPort`, `duplex_lifecycle_events` queue | `engine/duplex/messages.py`, `control_client.py` (`DuplexControlClient`, deleted), `control_plane.py:344-512`, `duplex_request_client.py`, `runtime_bridge.py:34-226`, `async_omni.py` | typed `DuplexSessionCommandMessage` (session_id, command) and `DuplexSessionEventMessage`; identity checked once in `DuplexSessionManager.dispatch`; `DuplexOmniEngine` builds messages directly; `DuplexControlRequestError` -> `DuplexSessionError` |
| API-side `DuplexSession`, `DuplexSessionRegistry`, `DuplexTurnController`, `ServingRuntimeSessionState`/`ServingRuntimeAdapter` Protocols, `NativeRealtimeSessionProtocol`, `RealtimeOutputProjector`, `RealtimeSessionState` | `entrypoints/duplex/{protocol,runtime_adapter,realtime_session,realtime_output,realtime_state}.py` | `DuplexEngineSession`, `DuplexModelPlugin`, `DuplexEvent.to_realtime()` |
| `AsyncOmni` / `AsyncOmniEngine` / `Orchestrator` duplex members | §2.4-§2.6 | `DuplexOmni`, `DuplexOrchestrator`, `DuplexSessionManager` |
| Simplification pass (§7 D17): `DuplexSessionResources`, `DuplexStageBinding`, `stage_request_ids`, `DuplexRuntimeCapabilities`, `DuplexInputMode` and the `mode` parameter of `plan_append` / `prepare_append`, `DuplexCapabilities.input_modes` / `implementation_level` fields (wire values kept as constants), `DuplexFence.response_seq`, `DuplexTurnController`, `DuplexControlError`, the dict round trips of session config / capabilities across the engine queue, `held_events`, the internal-dict form of stateless events, the public blocking engine methods | `engine/duplex/{session,contracts,plugin,config,session_manager,session_runner,messages,realtime_events}.py`, `duplex_omni.py`, `duplex_omni_engine.py`, three model plugins | one `DuplexEngineSession`; typed queue objects; `_emit_error` / `session.signal_turn()`; `*_async` engine surface only |
| Client-chosen session ids and the `incarnation` counter (§3.9, §7 D16): `open_session(session_id=...)`, `DuplexClientBase(session_id=...)`, `session_id` in the `session.update` open payload, `incarnation` in `DuplexFence`, events, messages, `session.created` / `session.resume`, attachment registry, model-side session keys, `stale_incarnation` | `duplex_omni.py`, `session_manager.py`, `session_runner.py`, `session.py`, `contracts.py`, `messages.py`, `events.py`, `realtime_input.py`, `serving.py`, `session_attachment.py`, `clients/duplex.py`, model `duplex/` packages, `duplex_sampling.py` | server-allocated `duplex-<uuid4 hex>` ids; identity = `session_id`; `unknown_session` error |
| `typing.Protocol` classes (`DuplexRuntimeExtension`, `DuplexStagePort`, `DuplexControlPlanePort`, `CorrelatedRpcTransport`, `PcmAppendBuffer`, `PcmAppendReservation`, `RuntimeDataPlane`, `WebSocketTransport`) (§7 D5) | `engine/duplex/contracts.py`, `runtime_adapter.py`, `clients/duplex.py:601` | ABCs (`DuplexModelPlugin`, `DuplexDataPlane`, `DuplexModelSessionState`, `PcmAppendBuffer`, `WebSocketTransport`); the stage port is `DuplexOrchestrator` itself |

Not removed: `?duplex=1`, `session_mode: duplex` (scheduler contract),
the `session_mode: duplex` scheduler contract, the wire constants
`implementation_level="model_native_duplex"` / `input_modes=["append_audio_chunk"]`,
`DuplexSessionAttachmentRegistry`, `experimental/fullduplex/`.

---

## 5. Migration plan

Ordered to delete first, then move engine-side, then thin the API and serving
layers, keeping the tree green after every phase.

### Phase 0 — Base-class extraction (no behavior change)

- `OrchestratorBase` with the seams of §2.4; turn handlers in `Orchestrator`;
  duplex code still in place but routed through the seams.
- `OmniEngineBase` base with abstract `_create_orchestrator`;
  `AsyncOmniEngine` on top implementing it; duplex methods stay on
  `AsyncOmniEngine` for now.
- `AsyncOmniBase` with `_route_engine_message`; `OmniBase` gets abstract
  `_create_engine`, implemented by `Omni` and `AsyncOmni`.

### Phase 1 — Deletions

- `/v1/duplex`, dialect aliases, `native_duplex` (outside `experimental/`),
  `ChatFallbackProjectorMixin`, generic lane, `is_enabled()`,
  `duplex_control_enabled`; collapse the ~40 `realtime_protocol is None` /
  native-vs-generic conditionals.
- Tests/docs per §6.1 and the doc list in §4.

### Phase 2 — Single plugin (C) and `DuplexOrchestrator`

- `DuplexModelPlugin` ABC; `MiniCPMO45DuplexPlugin`, `PersonaPlexDuplexPlugin`,
  `NemotronVoiceChatDuplexPlugin` compose today's extension + adapter;
  `PipelineConfig.duplex_plugin`.
- `engine/duplex_orchestrator.py`: control-plane hosting, stage port,
  `DuplexOrchestratorRequestState`, reaper, `release_owners` override.
  `OrchestratorBase` / `Orchestrator` / `AsyncOmniEngine` lose duplex code.
- `DuplexOmniEngine` (creating `DuplexOrchestrator` in
  `_create_orchestrator`) taking over the duplex control methods from
  `AsyncOmniEngine`; `DuplexOmni` skeleton (creating `DuplexOmniEngine` in
  `_create_engine`), still exposing the old fenced proxies so
  the existing serving handler keeps working; `api_server` duplex-only branch
  (§2.7); MiniCPM-o online chat tests/examples removed or converted.
- Tests: `test_duplex_orchestrator.py`, import-boundary targets,
  `test_config_factory.py`.

### Phase 3 — Typed contract (B)

- `engine/duplex/commands.py`, `events.py` with `from_realtime` /
  `to_realtime`; round-trip contract test against
  `docs/serving/realtime_duplex_api.md`.
- Introduce at the serving boundary first: the handler parses wire JSON into
  `DuplexCommand` and serializes `DuplexEvent`, with temporary adapters to the
  still-dict-based runner. This isolates the wire-format risk before the move.

### Phase 4 — Engine-resident sessions (A, the big one)

- 4a `DuplexEngineSession`: merge `DuplexSession` ledgers + transition
  methods with `DuplexSessionRuntimeState` and the plugin session state;
  unit tests for the merged class from `test_duplex_protocol.py` +
  `engine/duplex/test_duplex_control_plane.py` session cases.
- 4b `DuplexSessionRunner`: move the `handle_session` closure bodies,
  `runtime_bridge.py` bodies and `serving.py` domain helpers into methods;
  `_append` calls `plan_append` + stage port directly;
  `on_stage_output` replaces the drain; `_offload` for CPU work; VAD in the
  runner.
- 4c `DuplexSessionManager` from `DuplexControlPlane` +
  `DuplexSessionRuntimeManager` + serving registry/lifecycle code; new
  message types; `DuplexOmniEngine` switches from the fenced control methods
  to `open/close/resume/touch_session_async` + `submit_command`;
  `DuplexOrchestrator._intercept_stage_output` -> runner.
- 4d `DuplexOmni` becomes the thin pipe of §3.1/§3.2; serving handler
  rewritten to §3.5; delete the API-side session files (§2.8).
- Tests: `tests/entrypoints/openai_api/test_duplex_handler.py` scenarios
  ported to `tests/engine/duplex/test_session_runner.py` driving the runner
  with a fake stage port and scripted stage outputs (no fake engine client
  needed any more); `tests/engine/duplex/test_session_manager.py` from the
  control-plane tests; thin `tests/entrypoints/test_duplex_omni.py` and
  `test_duplex_serving.py`.

### Phase 5 — Clients

- `DuplexClientBase`, `InlineDuplexClient`, `WebSocketTransport` ABC;
  parameterized client tests; `barge_in_client.py --inline`.

### Phase 6 — Docs

- `docs/design/fullduplex.md` (active path, state ownership: one owner,
  module table, plugin descriptor section resolved), module ownership docs,
  `realtime_duplex_api.md` Python API section, `stage_configs.md`
  (`duplex_plugin`).

### Validation

- CPU suites after each phase.
- GPU (owner) after Phase 2 (duplex-only server, old runner), after Phase 4
  (engine-resident runner, orchestrator stall metrics under two concurrent
  sessions), after Phase 5 (inline client). PersonaPlex and Nemotron drivers
  unchanged apart from the removed flag.

---

## 6. Impact inventory

### 6.1 Tests

| File | Lines | Change |
| --- | --- | --- |
| `tests/entrypoints/openai_api/test_duplex_handler.py` | 8134 | Phase 1: chat-fallback + `native_duplex` + `/v1/duplex` cases removed. Phase 4: scenarios ported to `tests/engine/duplex/test_session_runner.py` (fake stage port) and a thin serving test |
| `tests/entrypoints/openai/test_duplex_protocol.py` | 553 | flag/alias tests deleted; session-state cases move to `DuplexEngineSession` tests |
| `tests/engine/duplex/test_duplex_control_plane.py` | 1270 | becomes `test_session_manager.py` + runner append tests |
| `tests/engine/test_orchestrator*.py` | ~4300 | duplex cases -> `test_duplex_orchestrator.py`; generic cases target the base |
| `tests/entrypoints/test_async_omni_duplex.py`, `test_duplex_fence_propagation.py`, `tests/entrypoints/duplex/test_runtime_adapter_boundary.py`, `test_runtime_control_redaction.py` | ~990 | fence-propagation and proxy tests deleted with the fence protocol; redaction test moves to `RuntimeControl.to_realtime()` |
| `tests/clients/test_duplex_client.py`, `test_model_session_configs.py` | 965 | flag tests deleted; demux tests parameterized over `DuplexClientBase`; inline handshake tests added |
| `tests/entrypoints/openai_api/test_api_server_guards.py` | 887 | `/v1/duplex` rows removed; duplex-mode availability added |
| `tests/engine/test_duplex_import_boundary.py` | 178 | targets updated (`engine.duplex_orchestrator`, `entrypoints.duplex_omni`) |
| `tests/config/test_config_factory.py`, `tests/model_executor/models/*/test_pipeline.py`, `tests/engine/duplex/test_duplex_runtime.py` | — | `duplex_plugin` replaces the three fields |
| `tests/model_executor/models/{minicpmo_4_5,personaplex,nemotron_voicechat}/duplex/*` | — | adapters/extensions become one plugin class; `native_duplex` rejection test deleted |
| MiniCPM-o online chat: `tests/e2e/online_serving/test_minicpmo_4_5*.py`, online cases of `tests/dfx/reliability/.../test_invalid_minicpmo_4_5_omni.py` | — | removed or converted (D9) |
| e2e duplex drivers/helpers, `tests/benchmarks/*`, `tests/examples/test_minicpmo_realtime_web_static.py` | — | `native_duplex` removed from payloads/URLs/assertions |

Untouched: `tests/e2e/features/fullduplex/*` (experimental), scheduler/worker
suites, offline MiniCPM-o tests, `test_duplex_lease.py`,
`test_duplex_session_attachment.py`, `test_websocket_actor.py`.

### 6.2 Source files touched

| Area | Files |
| --- | --- |
| Engine | `engine/orchestrator.py` (split), `engine/async_omni_engine.py` (split), new `engine/omni_engine_base.py`, `engine/duplex_omni_engine.py`, `engine/duplex_orchestrator.py`; `engine/duplex/`: new `commands.py`, `events.py`, `plugin.py`, `session_runner.py`, `session_manager.py`; rewritten `session.py`, `messages.py`; trimmed `contracts.py`; deleted `control_client.py`, `control_plane.py`, `runtime.py` (folded into `plugin.py` loading); moved-in `audio.py`, `vad.py`, `commit_policy.py` |
| Config | `config/stage_config.py` (`duplex_plugin`), `deploy/minicpmo_4_5*.yaml` (header comment) |
| Python API | `entrypoints/omni_base.py`, new `entrypoints/async_omni_base.py`, `entrypoints/async_omni.py` (shrinks), new `entrypoints/duplex_omni.py`; deleted `entrypoints/duplex_request_client.py` |
| Serving | `entrypoints/duplex/{serving,realtime_input,websocket,session_attachment}.py`; deleted `chat_fallback.py`, `realtime_session.py`, `realtime_state.py`, `realtime_output.py`, `runtime_adapter.py`, `runtime_bridge.py`, `session_runner.py`, `protocol.py`, `capability.py`; `entrypoints/openai/api_server.py` |
| Model plugins | `model_executor/models/{minicpmo_4_5,personaplex,nemotron_voicechat}/pipeline.py` and `duplex/` (extension + serving adapter -> one plugin class) |
| Clients | `clients/duplex.py`, new `clients/inline_duplex.py`, `clients/minicpmo_4_5.py` |
| Benchmarks/examples | `benchmarks/omniinteract.py`, `benchmarks/patch/patch.py`, `examples/online_serving/barge_in_client.py`, `examples/online_serving/minicpmo/*`, minicpmo web demo JS, `examples/online_serving/personaplex/README.md` |
| Docs | `docs/design/fullduplex.md`, `docs/design/module/{entrypoints,engine_orchestration}.md`, `docs/serving/{full_duplex_api,realtime_duplex_api,README,standalone_servers}.md`, `docs/design/architecture_overview.md`, `docs/api/README.md`, `docs/configuration/stage_configs.md`, recipes/example READMEs |

---

## 7. Decisions (owner-confirmed)

| # | Decision |
| --- | --- |
| D1 | Sibling classes over shared bases at every layer; duplex serving does not offer the chat API. |
| D2 | Public event and config vocabulary is Realtime-compatible; derived via `to_realtime()` / `from_realtime()` in the engine-side contract. |
| D3 | `PipelineConfig.duplex_control_enabled` dropped (subsumed by D12). |
| D4 | `native_duplex` stripped from client, server and model code; `experimental/fullduplex/` untouched. |
| D5 | All duplex `Protocol`s become ABCs. |
| D6 | Server-side VAD lives in the session runner (now engine-side). |
| D7 | Serving files stay under `entrypoints/duplex/`; engine-side session code lives under `engine/duplex/`. |
| D8 | `DuplexSessionHandle.events()` is single-consumer. |
| D9 | `vllm-omni serve` always runs a duplex model in duplex mode. **Breaking:** MiniCPM-o 4.5 servers no longer serve `/v1/chat/completions`; its online chat examples/tests go away (§2.1 note, §2.7). |
| D10 | The engine layer is split like the others: `OmniEngineBase` base, `AsyncOmniEngine`, and a thin `DuplexOmniEngine` (session message surface, creates `DuplexOrchestrator`). Kept for naming clarity even though it is small. |
| D11 | Alternative A applied: sessions are engine-resident; one `DuplexEngineSession` owned by a `DuplexSessionRunner` on the orchestrator loop; `DuplexOmni` is a thin pipe. |
| D12 | Alternative C applied: one `DuplexModelPlugin` per model selected by `PipelineConfig.duplex_plugin`. |
| D13 | Alternative B applied: typed `DuplexCommand` / `DuplexEvent` contract. |
| D14 | Collaborators are constructed explicitly: `DuplexOmni._create_engine()` returns `DuplexOmniEngine`, `DuplexOmniEngine._create_orchestrator()` returns `DuplexOrchestrator` with duplex-specific constructor arguments. No `engine_cls` / `orchestrator_cls` attributes. |
| D15 | `DuplexControlClient` is deleted; `DuplexOmniEngine` builds messages directly; `DuplexControlRequestError` becomes `DuplexSessionError`. |
| D16 | Session ids are always allocated by the server (`DuplexOmni.open_session`); clients cannot pick one. Because ids are never reused, the `incarnation` counter is dropped from the fence, events, messages, wire protocol and model-side session keys (§3.9). Revision 4; not yet implemented in code. |
| D17 | Simplification pass (revision 5, implemented): (1) `DuplexEngineSession` absorbs the former `DuplexSessionResources` (lease, accepted fence, stage request resources, append sequencing); the duplicate config / capability copies and the second request tracker are gone. (2) The input-mode mechanism (`DuplexInputMode`, `input_modes`, `implementation_level`) is removed: every duplex model is model-native and appends audio chunks. (3) `DuplexSessionConfig` and `DuplexCapabilities` cross the in-process engine queue as objects; control errors are plain fields on the result. (4) Stateless events are constructed typed at the emit site; the projector keeps only the stateful and domain-terminal projections. Constraint recorded: queue messages now carry dataclasses, so a future out-of-process orchestrator would need an encoding step. |

---

## 8. Structure review

### 8.1 Is `DuplexOmni > DuplexOmniEngine > DuplexOrchestrator > DuplexControlPlane` overkill?

In revision 2 it was: `DuplexOmniEngine` held 13 fenced RPC wrappers and
`DuplexControlPlane` was a second dispatcher in front of a second session
state. With A applied the chain is:

```text
DuplexOmni            thin API facade: session handles, event routing                         (~300 lines)
DuplexOmniEngine      thin engine sibling: session message surface over the generic OmniEngineBase (~150 lines)
DuplexOrchestrator    stage management sibling; hosts the session manager                    (hooks + stage port)
  DuplexSessionManager   admission / lease / dispatch                                         (component, not a layer)
  DuplexSessionRunner    the session state machine                                            (component, not a layer)
```

Three layers with one job each (API facade / process and transport / stage
management), each the duplex sibling of a turn-based class with the same
suffix, plus two components inside the orchestrator that are composition
rather than layering. `DuplexControlPlane` does not survive as a name: its
algorithms split into the manager (open/close/resume/touch, admission,
reaper, cleanup) and the runner (append planning and stage submit).
`DuplexOmniEngine` is kept small on purpose (§7 D10): it is the named place
for "how session messages enter the engine", not a place for session logic.

Merging `DuplexOmni` into the engine is not recommended: the engine runs the
orchestrator on a background thread; the API facade must live on the caller's
event loop. Merging `DuplexOrchestrator` into `DuplexSessionManager` is also
not recommended: stage forwarding, prewarm and cleanup are 2500 lines of
generic machinery that both orchestrator siblings need; the duplex sibling
only overrides policy points.

### 8.2 Residual risks of the applied alternatives

- A moves ~4k lines of validated logic onto the orchestrator thread. The
  offload rule (§2.2) and the stall metrics gate must be enforced in review
  before the H20 re-validation.
- B touches every emission site; the round-trip contract test is the safety
  net, and Phase 3 lands it at the serving boundary before the move.
- C changes the plugin contract for three models at once; keep the old
  extension/adapter classes as thin subclasses of the plugin for one phase so
  the model-side unit tests keep running until they are re-pointed.

### 8.3 Self-review of revision 3 (what changed after re-reading)

| Finding | Fix in this document |
| --- | --- |
| Claimed `expected_epoch` checks disappear; only the cross-boundary fence protocol does. | §2.2 corrected; §3.7 keeps re-validation after every `await`. |
| No written concurrency model for a runner that receives commands, stage outputs, timers and lease events on one loop. | New §3.7: single writer through the mailbox, tracked tasks for slow work, atomic cancel, ordered emission, dedicated executor. |
| `_intercept_stage_output` did not say when the base still forwards Stage0 output to TTS. | §2.4 seam contract rewritten (`True` = do not forward). |
| First session event could reach `DuplexOmni` before its handle exists (RPC result and output pump are different paths). | Session id generated API-side, pending handle registered before the open RPC (§3.1). |
| Backpressure "in the runner" would let bytes sit in the mailbox. | Reservation in `DuplexSessionManager.dispatch` before the put (§3.2). |
| Per-response stage metrics lost their source with `collect_outputs`. | Metric snapshot passed through the seam and accumulated by the runner (§3.7). |
| Plugin validation placed in `DuplexOmniEngine.__init__`, where stage pools do not exist yet. | Load in the engine, validate in `DuplexSessionManager.__init__` (§2.5). |
| `DuplexSessionConfig` / `DuplexCapabilities` / `ResponseCreateOptions` had no home after `protocol.py` deletion. | `engine/duplex/config.py` (§2.8). |
| Disconnect grace implemented twice (API attachment callback and engine lease). | Engine lease only; `detach_session` added to `DuplexOmni` (§3.1, §3.8). |
| `_offload` on the default thread pool competes with API-side RPC waits. | Dedicated executor owned by the manager (§3.7). |

Known simplifications carried by `DuplexOmni` that are accepted, not
defects: it inherits `OmniBase` request bookkeeping (`request_states`,
metrics, PD state) that it never uses; `AsyncOmniBase` keeps the ACK resolver
for the shared output pump.
