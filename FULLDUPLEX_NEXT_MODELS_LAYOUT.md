# Full-Duplex Next Models: Target Layout Sketch

Status: **design sketch only** — nothing here is implemented. This document
answers "where would the code live" if support for OmniFlatten, SyncLLM,
Freeze-Omni, and Moshi were added, following the layout conventions
established by the MiniCPM-o 4.5 graduation
(`FULLDUPLEX_MIGRATION_DESIGN.md` §2) and the 2026-08-05 genericity split
(§15 there): generic serving/engine code never imports model code; each
model contributes a self-contained `duplex/` folder under its
`model_executor/models/<model>/` package, wired in via dotted-string plugin
paths and a `DuplexCapabilities` preset.

Ground rules carried over:

- `vllm_omni/entrypoints/openai/duplex/` and `vllm_omni/engine/duplex/`
  stay model-free. New capability *fields* are allowed there (they are
  data); new model *behavior* is not.
- A model opts in through its `pipeline.py`
  (`duplex_runtime_extension`, `duplex_serving_adapter`,
  `duplex_control_enabled`) and a deployment enables the endpoint with
  `session_mode: duplex`.
- Every model folder mirrors the MiniCPM file roles: `capabilities.py`
  (preset), `adapter.py` (extra-body gate + client config validation),
  `serving_adapter.py` (`ServingRuntimeAdapter` impl), `runtime.py`
  (engine `DuplexRuntimeExtension`), plus model-specific internals.
- Tests mirror the package path under
  `tests/model_executor/models/<model>/duplex/`.

## 1. Freeze-Omni — the minimal native-duplex sibling

Chunk-boundary state head decides continue-listening / start-responding /
stop. Closest to MiniCPM; smallest folder. Exercises the
`supports_model_native_turn_policy` tier with the fewest moving parts.

```text
vllm_omni/
├── model_executor/models/freeze_omni/
│   ├── pipeline.py                  # stages + duplex_runtime_extension/-serving_adapter strings
│   ├── freeze_omni.py               # speech encoder + frozen LLM stage model
│   └── duplex/
│       ├── __init__.py
│       ├── capabilities.py          # freeze_omni_native_capabilities(): model_native_turn_policy=True,
│       │                            #   input_append=True, chunk_period_ms per encoder chunk
│       ├── adapter.py               # extra_body gate ("freeze_omni_native_duplex"), config validation
│       ├── serving_adapter.py       # ServingRuntimeAdapter: PCM chunk packing, output projection
│       ├── policy.py                # state-head decision mapping → listen/speak envelopes
│       ├── runtime.py               # DuplexRuntimeExtension: chunk → stage request, state-head readout
│       ├── session.py               # serving-side session state (chunk buffer, state history)
│       └── input.py                 # audio chunking/normalization for the speech encoder
├── deploy/freeze_omni_duplex.yaml   # session_mode: duplex
```

Generic changes required: **none.**

## 2. SyncLLM — time-synchronized chunks with latency compensation

Same segmented family, but with a strict wall-clock chunk cadence and
speculative future-chunk prediction to hide latency. Stresses the timing
assumptions (`chunk_period_ms`, silence padding) harder than MiniCPM.

```text
vllm_omni/
├── model_executor/models/sync_llm/
│   ├── pipeline.py
│   ├── sync_llm.py                  # token-synchronized dialogue LM stage model
│   └── duplex/
│       ├── __init__.py
│       ├── capabilities.py          # strict chunk_period_ms; model_native_turn_policy=True
│       ├── adapter.py
│       ├── serving_adapter.py
│       ├── cadence.py               # NEW ROLE: wall-clock chunk scheduler — silence-chunk
│       │                            #   synthesis when the user is quiet, dedup/rollback of
│       │                            #   speculatively predicted chunks when real audio arrives
│       ├── policy.py                # chunk-interleave turn policy
│       ├── runtime.py               # DuplexRuntimeExtension: fixed-cadence chunk submission
│       ├── session.py
│       └── input.py
├── deploy/sync_llm_duplex.yaml
```

Generic changes required: none expected; if the serving-side silence
ticker in `session_runner.py` proves too MiniCPM-shaped for strict
cadence, generalize the tick interval into a capability field (data-only
change).

## 3. OmniFlatten — flattened single-stream time-division

User and assistant audio+text interleaved into one autoregressive stream
as fixed-ratio chunk groups. Same time-division core as MiniCPM; the new
work is the interleave schema, not the serving flow.

```text
vllm_omni/
├── model_executor/models/omniflatten/
│   ├── pipeline.py
│   ├── omniflatten.py               # flattened-stream LM stage model
│   └── duplex/
│       ├── __init__.py
│       ├── capabilities.py          # model_native_turn_policy=True; input_modes append_audio_chunk
│       ├── adapter.py
│       ├── serving_adapter.py
│       ├── flatten.py               # NEW ROLE: chunk-interleave schema — packs user chunks and
│       │                            #   unpacks assistant chunks from the single flattened stream
│       │                            #   (text:audio token ratio, chunk sizes per training recipe)
│       ├── policy.py                # turn-taking read out of the flattened stream tokens
│       ├── runtime.py               # DuplexRuntimeExtension over resumable stage requests
│       ├── session.py
│       └── input.py
├── deploy/omniflatten_duplex.yaml
```

Generic changes required: **none** — the flattening is entirely inside the
model's data plane, invisible to serving.

## 4. Moshi — parallel-stream duplex (the tier the stack does not have yet)

Frame-synchronous: every ~80 ms frame the model consumes a user audio
frame and emits an assistant audio frame (plus inner-monologue text) —
no commits, no turns, no response objects, barge-in implicit. This is the
one model of the four that requires **new generic capacity**, not just a
model folder.

```text
vllm_omni/
├── entrypoints/openai/duplex/
│   └── frame_session.py             # NEW GENERIC MODULE: frame-synchronous session loop —
│                                    #   continuous bidirectional frame pump replacing the
│                                    #   commit/turn machinery when capabilities select it;
│                                    #   reuses websocket.py actor, session lease/lifecycle,
│                                    #   audio.py codecs, session_attachment.py
│   # protocol.py                    # (edit, data-only) new capability fields:
│   #                                #   supports_frame_sync_streams, frame_period_ms,
│   #                                #   implementation_level="model_frame_duplex"
│   # realtime_*.py                  # untouched — frame mode bypasses Realtime response
│   #                                #   projection; raw duplex events only (first iteration)
├── engine/duplex/
│   # runtime.py / control_plane.py  # (edit, small) a persistent always-RUNNING stage request
│   #                                #   mode: open pins it, frames flow via appends, close
│   #                                #   finishes it — fences/leases unchanged, epoch barge-in
│   #                                #   unused (interruption is implicit in the streams)
├── model_executor/models/moshi/
│   ├── pipeline.py                  # single stage (Mimi codec + temporal/depth transformer)
│   ├── moshi_lm.py                  # Helium temporal transformer + depth transformer stage model
│   ├── mimi.py                      # streaming Mimi codec (encode user frames / decode assistant frames)
│   └── duplex/
│       ├── __init__.py
│       ├── capabilities.py          # moshi_frame_capabilities(): frame_sync=True, barge_in
│       │                            #   implicit, no client commit, no turn signals
│       ├── adapter.py               # extra_body gate ("moshi_frame_duplex")
│       ├── serving_adapter.py       # frame packing: PCM16 ↔ Mimi frame boundary alignment
│       ├── frame_runtime.py         # DuplexRuntimeExtension: per-frame encode → LM step →
│       │                            #   decode loop over the persistent stage request
│       ├── monologue.py             # inner-monologue text stream extraction → text deltas
│       └── session.py               # frame cursors, stream offsets, codec streaming state
├── deploy/moshi_duplex.yaml
```

Generic additions summarized (the honest cost of Moshi):

| Piece | Kind | Why |
| --- | --- | --- |
| `frame_session.py` | new generic module | commit/turn runner cannot express a continuous frame pump |
| capability fields for frame mode | data-only edit to `protocol.py` | mode selection stays capability-driven, no model imports |
| persistent-request mode in `engine/duplex` | small engine edit | Moshi never parks; `WAITING_FOR_STREAMING_REQ` cycle is wrong shape |

## 5. Test layout

```text
tests/model_executor/models/
├── freeze_omni/duplex/    # test_policy.py, test_input.py, test_capabilities.py
├── sync_llm/duplex/       # + test_cadence.py (silence synthesis, speculative rollback)
├── omniflatten/duplex/    # + test_flatten.py (interleave/deinterleave round-trip)
└── moshi/duplex/          # + test_frame_runtime.py, test_monologue.py
tests/entrypoints/openai/duplex/
└── test_frame_session.py  # generic frame-pump loop (only if §4 lands)
```

Contract tests to extend, not duplicate: `test_runtime_adapter_boundary.py`
(no model imports in generic modules — add the three new model packages to
its scan), `test_duplex_capability.py` (demo-client isolation),
`test_duplex_import_boundary.py` (plugin-only loading for each new model
folder).

## 6. Capability-preset summary

| Model | implementation_level | Turn decision | Engine request shape | Generic changes |
| --- | --- | --- | --- | --- |
| MiniCPM-o 4.5 (exists) | `model_native_duplex` | model, per chunk | resumable, parks between segments | — |
| Freeze-Omni | `model_native_duplex` | model state head, per chunk | resumable | none |
| SyncLLM | `model_native_duplex` | model, fixed cadence | resumable, strict cadence | none expected |
| OmniFlatten | `model_native_duplex` | model, flattened stream | resumable | none |
| Moshi | `model_frame_duplex` (new) | none — always both | persistent, never parks | frame session + capability fields + engine mode |

The ordering is also the recommended implementation order: the first three
validate that the existing seams generalize (cheap, additive), and only
then does the Moshi tier justify touching generic code.
