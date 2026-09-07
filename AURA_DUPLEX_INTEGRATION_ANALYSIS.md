# Fitting AURA into the duplex framework: refactored vs baseline

Written 2026-09-05. Sources read for this analysis:

| Tree | What |
| --- | --- |
| AURA | `NumberWan/vllm-omni` branch `AURA_production_v026` (`f488ace`), model code under `vllm_omni/model_executor/models/aura_omni/`, stage processors `stage_input_processors/aura_*.py`, `entrypoints/openai/serving_video_stream.py`, `video_stream_base.py`, `aura_tool_executor.py`, deploy YAMLs, `docs/serving/aura_video_stream_api.md`, `docs/aura/*.md`, the two web demos |
| Baseline | `D:\repo\github\chickeyton\vllm-omni_duplex_refactor_baseline` (`7112347f`, before the refactor) |
| Refactored | `D:\repo\github\chickeyton\vllm-omni_duplex_refactor` (`6a151f9f`, after the refactor) |

## 1. What AURA is today

AURA is a **four-stage cascade**, not a single native-duplex model:

```text
Stage 0  Qwen3-ASR            microphone PCM -> transcript          (custom wrapper: forced EOS when stage is skipped)
Stage 1  AURA / Qwen3-VL(3.5) transcript + video frames -> text or <|silent|>
Stage 2  Qwen3-TTS Talker     text -> codec tokens                  (stock vLLM-Omni)
Stage 3  Code2Wav             codec tokens -> PCM                   (stock vLLM-Omni)
```

The AURA-specific code splits into two very different halves:

| Half | Where | Size (physical lines) | Nature |
| --- | --- | ---: | --- |
| Model wrappers | `models/aura_omni/{qwen3_asr,qwen3_vl,qwen3_5,pipeline}.py` | 318 | Thin: config-compat shims, stage bypass, pipeline topology. **Unaffected by either duplex framework.** |
| Stage processors | `stage_input_processors/aura_omni.py`, `aura_session_history.py`, `aura_tool_protocol.py`, `aura_cross_turn_penalty.py`, `stage_bypass.py` | ~3,500 | Prompt building, ASR->AURA->TTS handoff, sentence-level TTS chunking, **session history registry**, tool protocol, cross-turn penalty. |
| Serving | `entrypoints/openai/serving_video_stream.py` (`AuraStreamingVideoHandler`), `video_stream_base.py`, `aura_tool_executor.py` | ~4,500 | The streaming-video WebSocket: frame/audio buffering, auto-trigger, turn lock, `generate()` per turn, text/audio event projection, tool loop. |
| Demos | `native_gateway_web_demo/server.py`, `minicpm_style_web_demo/` | ~800 + | Protocol bridges that add what the server lacks: barge-in (drop pending TTS at the bridge), Native event names, one merged WAV per turn. |

Session semantics AURA implements itself, outside any duplex framework
(from `aura_video_stream_api.md` and the handler code):

- A **turn** starts when `auto_trigger_min_frames` video frames have arrived and no turn is locked; audio is snapshotted at turn start (push-to-talk must send the whole utterance first).
- `<|silent|>` from Stage 1 means "do not speak"; silent turns are still committed to history. The deploy YAML must stop on `151669`/`151645` or silent turns pad to `max_tokens`.
- **Early turn release** after `response.text.done` so the next turn can start while TTS still streams.
- **Session history** lives in a module-global dict in `aura_session_history.py`, registered from the API process and looked up *again* inside the Stage 1 worker through `aura_session_id` in `additional_information`; pending turns are recorded in the processor and committed by `aura2tts*`.
- Documented gaps: `video.query` ignored, **no interrupt, no cancel** of in-flight TTS ("Server TTS may still finish on GPU"), barge-in only at the demo bridge, "End session = new server session (no history)", one concurrent session in the demos, no resume/reconnect.

So the question is not "port a duplex model" but "give a cascaded, turn-triggered pipeline real session semantics: interruption, cancel, continuous audio, resume, multiple sessions", and do it once instead of once per demo.

## 2. What each framework asks a model to provide

### Baseline (`7112347f`)

Two plugins per model, bound only by two dotted strings in `PipelineConfig`
(`duplex_runtime_extension`, `duplex_serving_adapter`) with no descriptor
checking that they match:

| Plugin | Contract | Runs in | Must implement |
| --- | --- | --- | --- |
| `DuplexRuntimeExtension` | `typing.Protocol` (`engine/duplex/contracts.py`) | orchestrator thread | `configure_sampling_params`, `plan_append`, `decide_output` |
| `ServingRuntimeAdapter` | `typing.Protocol` (`entrypoints/duplex/runtime_adapter.py`) | API process | `create_session_state`, `session_state`, `remove_session_state`, `is_enabled`, `capabilities`, `validate_client_extra_body`, `prepare_runtime_config`, `runtime_config_for_update`, `data_plane_context` + a `RuntimeDataPlane` + a `ServingRuntimeSessionState` with 16 fields (`data_plane_task`, `pending_silence_task`, `continuation_owner_id`, `committed_audio_operation_id`...) |

The adapter's session state is the *third* copy of the session (next to the
API-side `DuplexSession` and the engine-side `DuplexSessionRuntimeState`), and
its fields are asyncio tasks and owner ids that the 2,040-line
`handle_session` closure in `session_runner.py` mutates directly. Per-model
glue on the baseline: MiniCPM-o 678 effective lines across three files,
PersonaPlex 335, Nemotron 452, each split between the two processes.

Behaviour a model wants to change (when a turn starts, what "silent" means,
whether TTS may be cancelled) is not in either plugin: it is in the
serving-side mixins (`DuplexSessionRunnerMixin`, `NativeRuntimeBridgeMixin`)
and reachable only by editing them or by threading more `native_duplex`-style
flags through ~40 dialect conditionals.

### Refactored (`6a151f9f`)

One `DuplexModelPlugin` ABC (`engine/duplex/plugin.py`), one dotted string
(`PipelineConfig.duplex_plugin`), validated at engine start
(`_validate_deployment`, `validate_duplex_plugin_sampling`):

```text
engine policy     configure_sampling_params, plan_append, decide_output
session policy    create_session_state, capabilities, validate_client_extra_body,
                  prepare_runtime_config, runtime_config_for_update, data_plane_context,
                  runtime_config_for_function_output (optional, for tools)
state             DuplexModelSessionState ABC (3 methods), DuplexDataPlane ABC (6 methods),
                  PcmAppendBuffer ABC
```

Everything runs on the orchestrator loop next to the one
`DuplexEngineSession`; the plugin never sees a websocket, a task handle, or an
owner id. Per-model glue is one file: MiniCPM-o 633 effective lines,
PersonaPlex 310, Nemotron 434.

## 3. Advantages of the refactored framework for AURA

### 3.1 One session object, in the same process as the stage processors

AURA's hardest structural problem today is that session history is a
process-global registry consulted from two processes: the API handler
registers it, Stage 1's `asr2aura` looks it up by id, `aura2tts*` commits
it. This works only because the demos run one session and one replica; it
is why "End session = new server session".

- **Refactored:** `DuplexEngineSession` is the one authority for config,
  epoch/turn, ledgers and model state, owned by one `DuplexSessionRunner` on
  the orchestrator loop. AURA's `SessionHistory` becomes the plugin's
  `DuplexModelSessionState`; the runner passes it into `plan_append`, which
  builds the Stage 0 prompt and puts the history snapshot into the request's
  `additional_information`. The stage processors stop reaching into a global
  dict; commit/discard become runner-side ledger updates keyed by
  `(epoch, turn_id)`, exactly what the existing `ConversationHistory` and
  `PlaybackLedger` in `engine/duplex/session.py` already do for MiniCPM-o.
- **Baseline:** the session is already described three times across two
  event loops in one process (the API loop and the orchestrator thread,
  joined by the RPC boundary); AURA's registry would be a fourth copy, and
  the fence protocol (`next_fence`, `expected_epoch`, `operation_id`
  idempotency) would have to be extended to carry history commits across
  that boundary. AURA's own history crossing today is a real process
  boundary: the API server process registers it, the Stage 1 worker process
  looks it up again by id (`aura_session_history.py` keeps one registry for
  each side).

### 3.2 Interrupt, cancel and barge-in come for free, and reach all four stages

AURA documents "no interrupt, no cancel"; the demo bridge drops audio
client-side while the GPU keeps decoding. The refactored runner already
implements the full sequence for a multi-stage pipeline: `CancelResponse` /
`BargeIn` advance the epoch, cancel tracked append tasks, abort the
session-owned stage bindings through `DuplexStagePort.abort_requests`, and
emit `output_audio.cleared` / `response.done(cancelled)`. Because
`DuplexOrchestrator._intercept_stage_output` routes *every* stage's output for
a session-owned request to the runner, cancelling Stage 1 also cleans up the
Stage 2/3 requests that were forwarded from it. On the baseline the same
machinery exists but is spread over `serving.py`, `runtime_bridge.py` and
`control_plane.py`, and the cancel path goes through an RPC with fence
re-validation on both sides. AURA would get it either way, but on the
refactored tree it is one handler in one file.

### 3.3 Turn detection and "silent" are plugin decisions, not serving edits

AURA's trigger rule (N frames buffered, turn not locked, audio snapshot) and
its `<|silent|>` convention are today hard-coded in
`AuraStreamingVideoHandler` and in the stage processors' finish handling.

- **Refactored:** `decide_output(stage_id, final_stage_id, segment_finished,
  segment_token_ids, ...)` is called per stage per segment. The plugin returns
  `DuplexOutputAction` LISTEN for a `<|silent|>` prefix at Stage 1 (and the
  runner emits the typed `Listen` event and keeps the session in its listening
  state), SPEAK otherwise. Server-side VAD (`turn_detection.py`,
  `vad.py`) and the commit policy already live in the runner, so
  push-to-talk, server VAD and frame-count triggers are all expressible as
  `Commit` / `SignalTurn` commands plus a plugin rule, without touching
  serving. MiniCPM-o's listen/speak decision helpers in its plugin are the
  template.
- **Baseline:** the equivalent decisions sit in `_overlap_decision`,
  `_apply_runtime_lifecycle` and the 2,040-line closure; adding a fourth
  model's variant means another branch there.

### 3.4 Video frames already have a typed input path

`AppendAudio` in the refactored contract carries `video_frames: tuple[str, ...]`
with validation in `realtime_commands.py` (added for MiniCPM-o's omni input).
AURA's frame batches map onto it directly; `plan_append` receives the frames
in `payload` and can pack them with `pack_aura_video_ndarray` /
`frames_to_video_tuple` into the Stage 0 prompt's deferred multimodal data,
which `asr2aura` already reads. Adding a frame-only command later (video
without audio, for AURA's silent-vision turns) is a new dataclass in
`commands.py` plus one `from_realtime` case, because the contract is typed
and the wire vocabulary is derived from it. On the baseline, video would go
through `protocol.py`, `realtime_input.py`, `serving.py`, `runtime_bridge.py`
and the fence-carrying append message, each with its own dict shape.

### 3.5 Multi-session, resume and backpressure are framework properties

AURA's demos run one session and rebuild on reconnect. The refactored
`DuplexSessionManager` provides admission (`max_sessions`, closing sessions
retain capacity), per-session byte and pending-turn backpressure enforced
before the mailbox, a lease reaper with disconnect grace and idle TTL, and
resume with replay through `DuplexSessionAttachmentRegistry`. None of this
is model-specific; AURA gets it by declaring `supports_session_resume` in
`capabilities()`. On the baseline the same features exist, but the
attachment registry, grace timer, lifecycle listener and expiry handling
are split between the API and engine halves and were the source of the
dead-timer and double-ownership problems the refactor removed.

### 3.6 Tools have a hook

AURA's bounded tool loop (`tool_mode: auto`, preamble speech while tools run,
final pass commits history once) is today a 400-line branch of
`_process_query_engine` in the serving handler. The refactored plugin has
`runtime_config_for_function_output` and the runner has the function-call
event family (`FunctionCallArgumentsDelta/Done`, `CreateItem` with a
`function_call_output` item), so the pass-1 / pass-2 structure maps onto
standard Realtime tool events: Stage 1 emits a `<tool_call>` -> plugin
projects it as a function call item -> the client (or a server-side executor
behind the same `CreateItem` path) returns the output -> plugin builds the
pass-2 runtime config. The baseline has no equivalent seam; tools would stay
in serving.

### 3.7 Less code to write, and it is checkable

| | Baseline | Refactored |
| --- | --- | --- |
| Plugin classes to write | 2 Protocols + 3 helper Protocols, in two processes | 1 ABC + 2 small ABCs, one process |
| Where AURA's turn/silent/tool rules go | serving mixins (edit framework code) | plugin methods (own file) |
| Wiring check | none; a mismatched pair fails at first request | `_validate_deployment` at engine start; static import check |
| Comparable existing plugin to copy | MiniCPM-o adapter + runtime + serving adapter (678 lines, 3 files) | MiniCPM-o plugin (633 lines, 1 file); Nemotron (434) is the closer cascade example |
| Typing | dict payloads and `native_duplex` toggles on the wire | `DuplexCommand` / `DuplexEvent` dataclasses, wire JSON derived |
| Serving layer that must understand AURA | `serving.py` 1,760 lines + mixins | none: `OmniDuplexSessionHandler` is transport only (399 lines) |

## 4. What the refactored framework does not give AURA (honest gaps)

1. **Cascade-specific stage handling still has to be written.** `plan_append`
   targets Stage 0 (ASR); the ASR->AURA->TTS handoffs stay in the existing
   `custom_process_input_func` / `async_chunk` processors. `decide_output`
   must recognise Stage 1 finishes (silent vs spoken) and Stage 3 audio, and
   the "keep Stage 3 alive on batched silent finishes" fix has to be
   re-expressed as a runner cleanup rule. Nemotron's plugin is the closest
   existing example of a plugin whose Stage 0 is not the speaking model.
2. **Stage 0 bypass** (`omni_skip_stages`, video-only turns) is an orchestrator
   feature on the AURA branch (`_bypass_stage0` mock-forward). It is not on
   either duplex tree; it has to be ported to `OrchestratorBase` and
   exposed to the plugin through `DuplexAppendPlan`.
3. **Session history semantics** (sliding-window pruning, `max_rounds`,
   cross-turn penalty via `logit_bias` / `bad_words`) are AURA policy and
   move into the plugin's session state and `configure_sampling_params`
   unchanged in logic, but they must be rewritten against the runner's
   ledger API rather than the global registry.
4. **The runner is large** (about 3.6k lines). Adding AURA does not require
   editing it if the plugin seams suffice, but debugging a cascade will mean
   reading it. The baseline is worse here (the same logic across five files
   in two processes), but neither tree is small.
5. **Nothing has run.** The refactored tree passes static checks only
   (no vllm/torch install here); tests, docs and examples are not yet
   updated. AURA integration should start after the existing three plugins
   are validated end to end on a GPU.
6. **Breaking change carried over from D9:** on the refactored tree a model
   with `duplex_plugin` is served duplex-only. AURA's `/v1/chat/completions`
   and `/v1/video/chat/stream` routes would go away on that server; the
   offline `Omni` / `AsyncOmni` path stays. If both surfaces must coexist,
   that is a second deploy profile, not a code change.

## 5. Closing the gaps in section 4

Each item names the change, where it goes, and its size. Items 5.1 to 5.3
are small seam additions to the framework; the existing three plugins keep
working because every addition has a no-op default.

### 5.1 Cascade stage handling: mostly already there

Verified in the refactored tree:

- `DuplexOrchestrator._on_stage_submitted` binds *every* forwarded stage
  request (stages 1-3, not only stage 0) to the session, and
  `_intercept_stage_output` routes every stage's output through
  `runner.on_stage_output`. Cancel and cleanup therefore already cover the
  whole cascade.
- `runner.on_stage_output` forwards an intermediate stage's output to the
  next stage when `decide_output` returns `None`, and consumes it (no
  forwarding) when it returns a decision. MiniCPM-o uses exactly this:
  `None` for spoken text so it reaches TTS, a decision for control-only
  segments.

So the AURA plugin's `decide_output` is:

| Stage | Segment | Return | Effect |
| --- | --- | ---: | --- |
| 0 ASR | any | `None` | transcript forwards to Stage 1 through `asr2aura` unchanged; optionally the plugin's data plane projects it as `InputTranscriptionCompleted` |
| 1 AURA | starts with `<|silent|>` | `DuplexOutputDecision(LISTEN)` | consumed by the runner: typed `Listen` event, session stays listening, **nothing is forwarded to Stage 2/3** |
| 1 AURA | text | `None` | forwards to the talker through `aura2tts_async_chunk` unchanged (sentence chunking stays where it is) |
| 3 Code2Wav | any | final stage, always consumed | audio projected by `AuraDataPlane.project` as `AudioDelta` |

The "keep Stage 3 alive on batched silent finishes" fix disappears by
construction: on the AURA branch a silent turn still produced an empty
Code2Wav request because the cascade forwarded unconditionally; here a
silent segment never leaves the runner, so there is no empty payload to
keep Stage 3 alive for. The ASR->AURA->TTS `custom_process_input_func`
chain is not touched; it runs in `OrchestratorBase`'s forward path, which
`DuplexOrchestrator` inherits.

Size: no framework change; about 60 lines in the plugin.

### 5.2 Stage 0 bypass: port once, drive it from the prompt

The AURA branch implements bypass generically already: `stage_bypass.py`
(106 lines, keyed on `omni_skip_stages` / `omni_bypass_stage_text` in the
prompt's `additional_information`) plus `_should_skip_stage_submission` and
`_bypass_stage0` in `orchestrator.py` (about 120 lines, mock-forwards a
finished empty Stage 0 output into Stage 1, prewarms later stages).

1. Port those two pieces onto `OrchestratorBase` in the refactored tree,
   where stage submission lives. They are turn-based features too
   (video-only chat turns), so they belong in the base, not the duplex
   sibling.
2. No new duplex seam is needed: `plan_append` already returns the Stage 0
   prompt, so the AURA plugin sets
   `additional_information["omni_skip_stages"] = [0]` for a frame-only
   append (no PCM since the last commit). The mock-forwarded Stage 1 request
   goes through the normal forward path, so `_on_stage_submitted` binds it
   to the session like any other.
3. `decide_output` for the mock Stage 0 output (finished, empty) returns
   `None`, same as a real transcript.

Size: about 230 lines ported, one branch in the plugin. The
`test_orchestrator_stage_bypass.py` tests come with it.

### 5.3 Session history and cross-turn penalty: two small seams

Today `plan_append` receives config, runtime config and the payload, but
not the model's per-session state, and sampling parameters are fixed per
config generation. AURA needs both per turn. Three additions, all with
no-op defaults:

1. **`plan_append(..., session_state: DuplexModelSessionState)`.** The
   runner passes `session.model_state` (it already owns it). MiniCPM-o,
   PersonaPlex and Nemotron ignore the argument. AURA's
   `AuraSessionState(DuplexModelSessionState)` holds a `SessionHistory`
   moved verbatim from `aura_session_history.py` into
   `aura_omni/duplex/history.py`, minus the three module-global dicts and
   their locks. `plan_append` renders a **snapshot** of the history into
   the Stage 0 prompt's `additional_information` (the rendered messages and
   the tool template); `asr2aura` builds the Stage 1 prompt from that
   snapshot instead of calling `get_session_history` on a registry. This is
   the one processor edit: replace a lookup with a read of what is already
   in `additional_information`.
2. **`DuplexModelSessionState.on_response_done(text, *, silent: bool)`**,
   a non-abstract method with an empty default, called once by the runner
   where it finalizes a response (the same place it emits `ResponseDone`).
   AURA commits the pending turn there, exactly what `commit_session_turn`
   does today, and discards it on cancel through the existing
   `clear_continuation` hook. Commit is now epoch-fenced by the runner, so
   a barge-in mid-turn cannot commit a half answer, which the current
   global registry cannot guarantee.
3. **`DuplexAppendPlan.sampling_overrides: Mapping[int, object] = {}`**
   (stage id to `SamplingParams`). `DuplexSessionManager` applies it when
   it builds the `DuplexStageRequestContext` for that append. AURA's
   `CrossTurnPenalty` computes `logit_bias` / `bad_words` from the history
   snapshot in `plan_append` and returns them for Stage 1. Sliding-window
   pruning (`max_rounds`, `num_rounds_keep`, `max_context_qas`) stays a
   pure method on `SessionHistory`, invoked in `on_response_done`.

Size: about 40 lines of framework seam, `SessionHistory` moved (about 700
lines, logic unchanged), the registry deleted.

### 5.4 Runner size: split by responsibility before adding a fourth model

`session_runner.py` is about 3.6k lines and scores 0 on the maintainability
index; the measurement in `DUPLEX_REFACTOR_LOC_BASELINE.md` shows that
splitting it into eight equal parts would lift each part to about 16, and
sixteen parts to about 33. The split that follows the code's own section
headers:

| Module | Responsibility (moved verbatim) |
| --- | --- |
| `runner_append.py` | PCM append tail, reservation, `plan_append` submit, silence continuation |
| `runner_response.py` | response create / cancel / barge-in, epoch advance, stage abort |
| `runner_output.py` | `on_stage_output`, `_decide_output`, data-plane pumping, model-output events |
| `runner_playback.py` | playback ack, history truncation, output-audio clear |
| `runner_items.py` | conversation items create / delete / truncate / retrieve |
| `runner_commands.py` | the `_on_<Command>` dispatch table |
| `session_runner.py` | the class shell: mailbox loop, lifecycle, emit |

Mechanically these are mixins or helper classes taking the runner; no
behaviour change, verifiable by the same static checks. Doing this first
means the AURA work reads one 500-line module per concern instead of one
file. It also makes 5.3's two call sites (pass `session_state`, call
`on_response_done`) easy to locate.

### 5.5 Validation order

Nothing in the refactored tree has run on a GPU. Sequence the risk:

1. Update the existing duplex tests to the new module layout (they still
   import deleted modules) and run the MiniCPM-o e2e suite. This validates
   the runner, manager and serving path with a native two-stage model.
2. Run Nemotron VoiceChat: its Stage 0 is not the speaking model, so it
   exercises the intermediate-stage forwarding that AURA relies on.
3. Land 5.2 to 5.4 with unit tests (`test_orchestrator_stage_bypass.py`
   ported, a `plan_append` snapshot test, an `on_response_done` fencing
   test).
4. Port `native_gateway_web_demo/verify_bridge.py`'s acceptance (ASR text,
   assistant text, one playable WAV, one turn done) onto `DuplexClient`
   over `/v1/realtime?duplex=1`, then add the two things AURA cannot do
   today: barge-in mid-TTS must yield `output_audio.cleared` and a
   cancelled `response.done`, and a second concurrent session must not
   share history.
5. Gate on the AURA branch's own numbers from the OmniInteract Smoke3
   run (spoken median TTFT 195 ms, AUDIO_TTFP 132 ms, Stage 1 TTFT 78 ms),
   warm-up `0001` excluded as their bench rule says. The duplex path adds
   one mailbox hop per command; it should not add measurable latency, and
   the bench will show it if it does.

### 5.6 Duplex-only serving (D9): two pipeline entries, no framework change

`_is_duplex_model` decides on `pipeline_config.duplex_plugin`, and the deploy
YAML selects the pipeline by name (`pipeline: aura_omni`). So:

```text
AURA_OMNI_PIPELINE          model_type="aura_omni"          no duplex_plugin   -> chat completions,
                                                                                    /v1/video/chat/stream (as today)
AURA_OMNI_DUPLEX_PIPELINE   model_type="aura_omni_duplex"   duplex_plugin=...  -> /v1/realtime?duplex=1 only
```

Same four stages, same processors, same weights; `aura_omni_duplex.yaml`
differs from `aura_omni.yaml` by the `pipeline:` line and
`session_mode: duplex`. Operators pick the surface per server. Revisiting
D9 itself (one server serving both) is possible later by letting the API
server construct `AsyncOmni` plus a `DuplexOmni` view over one engine, but
that is a framework decision with its own analysis; it is not needed to
ship AURA on the duplex path.

## 6. Recommended shape of the AURA plugin (refactored tree)

```text
vllm_omni/model_executor/models/aura_omni/
    pipeline.py            + duplex_plugin="...aura_omni.duplex.plugin.AuraDuplexPlugin"
    duplex/
        plugin.py          AuraDuplexPlugin(DuplexModelPlugin)
                             configure_sampling_params: Stage 1 stop ids (151669/151645 or v2 ids),
                                                        cross-turn penalty -> logit_bias/bad_words
                             plan_append:  PCM + video_frames -> Stage 0 prompt with history snapshot,
                                           tool template when tool_mode=auto
                             decide_output: Stage 1 <|silent|> -> LISTEN; text -> SPEAK; Stage 3 -> audio
                             create_session_state: AuraSessionState (SessionHistory + pending turn)
                             capabilities: turn_commit + server_vad, supports_session_resume
                             prepare_runtime_config: system prompt, ref audio / speaker, TTS args
                             runtime_config_for_function_output: pass-2 prompt
        input.py           AuraPcmAppendBuffer(PcmAppendBuffer): snapshot-at-commit semantics
        data_plane.py      AuraDataPlane(DuplexDataPlane): Stage 3 audio + Stage 1 text projection
        history.py         SessionHistory moved from stage_input_processors, no globals
```

Estimated size: 600-900 effective lines, in line with the three existing
plugins, versus editing five serving/engine files on the baseline.

## 7. Bottom line

For AURA the decisive advantages of the refactored framework are:

1. **One in-process session object** that can own AURA's history, instead of
   today's registry that is written in the API server process and read again
   in the Stage 1 worker process.
2. **Interrupt, cancel, barge-in, resume and multi-session with less to wire
   and debug.** The baseline duplex framework implements the same behaviours
   and also binds forwarded stage requests to the session, so AURA would
   inherit the same feature list on either tree once it is ported off the
   streaming-video handler (which has none of them). The difference is where
   they live: one handler in the runner reaching all four stages, versus the
   serving mixins, the RPC boundary and the engine control plane with fence
   re-validation on both sides.
3. **A single, validated plugin seam** (the one `DuplexModelPlugin` the engine
   loads and hands to the orchestrator) where AURA's turn, silent and tool
   rules live in AURA's own file, instead of the baseline's engine extension
   plus serving adapter pair.

On the baseline AURA would become the fourth model to spread its policy
across two Protocols, two event loops and the serving mixins, and would
still have to move its history out of the stage-worker registry itself.
