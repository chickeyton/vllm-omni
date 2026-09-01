# Full-Duplex Realtime Wire Protocol

## Purpose

This document is the client-facing companion to
[Full-Duplex Runtime (MiniCPM-o 4.5)](fullduplex.md). That document owns the
runtime architecture and the normative Realtime event contract (ordering and
cardinality rules, accepted input aliases, the commit-to-response table). This
document enumerates the wire vocabulary of `/v1/realtime?duplex=1` from the
client's point of view:

- how the duplex dialect relates to the OpenAI Realtime protocol, expressed
  as three compatibility tiers;
- a catalogue of every client-to-server and server-to-client event, with its
  tier and a one-line description;
- a concrete JSON example for every event type.

The catalogue and the examples are derived from the translator and projector
code (`vllm_omni/entrypoints/duplex/realtime_input.py`,
`realtime_output.py`) and the session runner (`session_runner.py`,
`serving.py`). A compatibility change to any of these modules must update the
catalogue and the examples here together with the normative contract table
and its protocol tests, as required by [fullduplex.md](fullduplex.md).

## Transport summary

The endpoint is `ws(s)://<host>/v1/realtime?duplex=1`; the `duplex` query
parameter (`1`, `true`, or `on`) selects the duplex session handler, and the
route is mounted only for deployments whose deploy configuration declares
`session_mode: duplex`. Optional query parameters are `model`, `session_id`,
`autostart` (`0` means resume-only), `resume`, and
`minicpmo45_native_duplex`. Every message is one JSON object per WebSocket
text frame, discriminated by `type`. Inbound events are applied in arrival
order through one per-session mailbox; outbound events preserve that order
and carry a monotonically increasing `server_event_seq`, which is the replay
cursor for `session.resume`. Client events may carry an `event_id`, echoed in
`error.error.event_id` when the event is rejected.

Session identity is `(session_id, incarnation)`; `attachment_generation`
names the current socket; `epoch` is the interruption fence that advances on
every cancel, clear, or barge-in; `turn_id` counts committed model turns;
`response_id` (item id `item_<response_id>`) names one assistant utterance.

**Tier** in the tables below: **1** means identical to OpenAI Realtime in
name and semantics; **2** means an OpenAI event name carrying vLLM-Omni
extensions that a stock OpenAI client ignores; **3** means vLLM-Omni only,
with no OpenAI counterpart.

## Compatibility with the OpenAI Realtime protocol

The Realtime dialect is, by design, an OpenAI-Realtime-compatible surface:
`RealtimeInputTranslator` / `RealtimeOutputProjector` exist to translate
the native duplex vocabulary onto OpenAI's schema so stock Realtime clients
work unmodified. The overlap falls into the three tiers used by the event catalogue below.

### Tier 1: identical to OpenAI Realtime

| Direction | Messages |
| --- | --- |
| Client → server | `session.update`, `input_audio_buffer.append` / `.commit` / `.clear`, `conversation.item.create` / `.retrieve` / `.truncate` / `.delete`, `response.create`, `response.cancel`, `output_audio_buffer.clear` (OpenAI: WebRTC-only; here also a WebSocket event) |
| Server → client | `error` (same `{type, code, message, event_id, param}` envelope and the same classes `invalid_request_error` / `server_error` / `rate_limit_error`), `session.created`, `session.updated`, `input_audio_buffer.committed` / `.cleared` / `.speech_started` / `.speech_stopped`, `conversation.item.created` / `.added` / `.done` / `.deleted` / `.retrieved` / `.truncated`, `conversation.item.input_audio_transcription.completed`, `response.created`, `response.done`, `response.output_item.added` / `.done`, `response.content_part.added` / `.done`, `response.audio.delta` / `.done`, `response.audio_transcript.delta` / `.done`, `response.output_text.delta` / `.done`, `response.function_call_arguments.delta` / `.done`, `rate_limits.updated`, `output_audio_buffer.cleared` |
| Session fields | `model`, `modalities` / `output_modalities`, `instructions`, `voice`, `input_audio_format` / `output_audio_format` (beta spelling) **and** `audio.input.format` / `audio.output.format` objects (GA spelling), `turn_detection.server_vad {threshold, prefix_padding_ms, silence_duration_ms}`, `input_audio_transcription`, `input_audio_noise_reduction`, `tools`, `tool_choice`, `temperature`, `max_response_output_tokens`, `speed`, `tracing` |
| Object shapes | `realtime.session`, `realtime.item`, `realtime.response`; item content parts `input_text` / `input_audio` / `output_text` / `output_audio`; `function_call` and `function_call_output` items; `previous_item_id` chaining; `output_index` / `content_index` addressing |
| Sequencing | `response.created → output_item.added → content_part.added → deltas → *.done → content_part.done → output_item.done → response.done → rate_limits.updated`; `speech_started → speech_stopped → committed` |

The server deliberately emits and accepts **both** the beta and the GA
spellings at once (`response.audio.delta` alongside `output_modalities`,
`conversation.item.added` alongside `conversation.item.created`) so either
generation of OpenAI client parses the stream.

### Tier 2: OpenAI names carrying vLLM-Omni extensions

A stock client ignores the extra keys; the extensions are additive.

| Message | Extension |
| --- | --- |
| `session.created` / `session.updated` | top-level `incarnation`, `attachment_generation`, `resume_token`; inside `session`: `state`, `turn_state`, `epoch`, `turn_id`, `active_request_id`, `active_response_id`, `active_response_turn_id`, `overlap_policy`, `overlap_*_ms/rms`, `playback_commit_policy`, `playback`, `capabilities`, `ref_audio`, `extra_body`, `idle_timeout_s`, `response_format` |
| `response.created` / `response.done` / `response.listen` | `response_id` at top level; the raw duplex event under `response.metadata` (`duplex_event`); `status_details.reason` uses vLLM reasons (`barge_in`, `client_cancelled`, `new_response`, …) |
| `response.audio.delta` | `format`, `sample_rate_hz`, `metadata{session_id, epoch, model_speak, end_of_turn, audio_duration_ms, audio_text_marks, playback}` |
| `response.speak` (inserted before the first delta) | not an OpenAI event, but rides the OpenAI response envelope (`response_id`, `item_id`, `output_index`, `content_index`) |
| `input_audio_buffer.append` | `is_speech`, `video_frames` (+ `max_slice_nums`), `duration_ms`, `audio_end_ms`, per-event `format` / `sample_rate_hz`, `final` |
| `input_audio_buffer.commit` | `final`, `response_create`, `is_speech:false` (silence declaration) |
| `input_audio_buffer.committed` | the native `input.committed` event wrapped under `event` (`turn_id`, `epoch`, `history_len`, `message`, `response_create_deferred`) |
| `conversation.item.*` server events | echo of the originating event under `event` |
| `error` | the error **codes** are vLLM-Omni's (OpenAI standardises only the class); handshake-stage errors use the flat native shape `{type:"error", error:"…", code}` |

Semantic divergences hidden behind shared names:

- **Commit ≠ response.** In the native lane audio streams into the model
  before any commit; a commit may end in `response.listen` and no
  response, and the model may open a response with no commit at all.
  OpenAI: commit ⇒ item, `response.create` ⇒ exactly one response.
- `turn_detection.interrupt_response=false` is rejected;
  `create_response` is ignored; `semantic_vad` is unsupported; the VAD
  runs per session (Silero) and implies `overlap_policy=barge_in_on_speech`.
- `rate_limits.updated` is always an empty list (compatibility only).
- Cancellation never reuses a `response_id`; truncation is driven by
  `playback.ack` as well as `conversation.item.truncate`.

### Tier 3: vLLM-Omni only

| Area | Messages / fields |
| --- | --- |
| Session lifetime | `session.heartbeat` / `session.heartbeat_ack`, `session.event_ack`, `session.close` / `session.closed`, `session.resume` / `session.resumed` / `session.replaced` / `session.expired` / `session.resync_required`; `server_event_seq`, `resume_token`, `incarnation`, `attachment_generation`; query params `?duplex=1`, `autostart`, `resume`, `minicpmo45_native_duplex` |
| Turn-taking | `response.speak`, `response.listen`, `overlap.decision`, `overlap_policy`, `barge_in`, `turn.signal`, `input.cancel`, `input.text.append`, `epoch` / `turn_id` fencing, `force_listen` |
| Playback truth | `playback.ack` / `playback.acknowledged`, `playback_commit_policy` (`ack_only` \| `commit_all_on_done`), `playback{generated_ms, sent_ms, played_ms, committed_ms}`, `history_committed`, `audio_text_marks` |
| Model negotiation | `capabilities{implementation_level, supports_input_append, supports_barge_in, supports_session_resume, chunk_period_ms, input_modes, …}`, `ref_audio`, `extra_body` (`auto_response`, `minicpmo45_native_duplex`, `force_listen_count`, `duplex_initial_user_text`) |
| Diagnostics | `runtime.control`, `duplex.function_call.done`, the full error-code vocabulary (`REALTIME_ERROR_TYPES_BY_CODE` in `vllm_omni/entrypoints/duplex/realtime_state.py`) |

### Consequences for clients

- Event **names** are roughly 70 % OpenAI's; payloads are OpenAI's plus
  additive extensions; everything that makes the endpoint full-duplex —
  model-owned turn-taking, playback truth, interruption fencing, resumable
  sessions — lives in Tier 3.
- An **unmodified OpenAI Realtime client** can hold a conversation
  (commit-driven or `turn_detection: server_vad`), but it will never see
  `response.listen` / `response.speak`, cannot resume after a transport
  drop, will not receive replay after reconnect, and its server-side
  history is only as honest as `commit_all_on_done` allows because it never
  sends `playback.ack`.
- A **duplex-aware client** (`vllm_omni.clients.duplex.DuplexClient`) uses
  the Tier 3 surface on top of the same Tier 1 events, so both kinds of
  client can share one deployment and one event log.

## Event catalogue

Every event type the Realtime route accepts or emits, with a worked example in the next section: 21
client→server and 42 server→client. Aliases (`push_chunk`,
`input.audio.append`, `input_text.append`, `push_text`, `signal_turn`,
`audio.playback_ack`, `close_session`, `close`, `session_close`) share the
payload of their canonical event and are not listed separately; the alias
table lives in the normative contract section of [fullduplex.md](fullduplex.md).

### Client to server

| Event type | Tier | Description |
| --- | --- | --- |
| `session.update` | 2 | Open handshake on first send (creates the session); later sends reconfigure it. OpenAI name; `session` object carries vLLM-Omni keys (`ref_audio`, `overlap_policy`, `playback_commit_policy`, `extra_body`, ...). |
| `session.heartbeat` | 3 | Keepalive; refreshes the engine session lease. |
| `session.event_ack` | 3 | Acknowledges received server events by `server_event_seq`; trims the resume replay journal. |
| `session.close` | 3 | Graceful close; answered with `session.closed`. |
| `session.resume` | 3 | Re-attach a live session on a new socket (`resume_token`, `incarnation`, `last_received_server_event_seq`). |
| `input_audio_buffer.append` | 2 | Append one audio chunk. OpenAI name; extended with `is_speech`, `video_frames`, `duration_ms`, `audio_end_ms`, per-event `format`/`sample_rate_hz`. |
| `input_audio_buffer.commit` | 2 | Seal the buffered utterance into a user item. OpenAI name; extended with `final`, `response_create`, `is_speech:false`. |
| `input.commit` | 3 | Alias of `input_audio_buffer.commit` accepted by the runner. |
| `input_audio_buffer.clear` | 1 | Drop un-committed input audio. |
| `input.cancel` | 3 | Cancel pending input; advances `epoch`. |
| `input.text.append` | 3 | Append text to the open input item (chat-fallback lane only). |
| `conversation.item.create` | 1 | Add a completed user item (text, audio, or `function_call_output`) to history. |
| `conversation.item.retrieve` | 1 | Fetch a stored item by id. |
| `conversation.item.delete` | 1 | Delete a stored item by id. |
| `conversation.item.truncate` | 1 | Truncate an assistant item's audio/transcript at `audio_end_ms`. |
| `response.create` | 1 | Explicitly request a response with per-response overrides. |
| `response.cancel` | 1 | Cancel the active (or named) response; advances `epoch`. |
| `output_audio_buffer.clear` | 1 | Discard queued output audio; advances `epoch` (OpenAI: WebRTC-only, here also WebSocket). |
| `barge_in` | 3 | Explicit hard interrupt; requires `capabilities.supports_barge_in`. |
| `turn.signal` | 3 | External turn-taking signal (e.g. `event:"barge_in"`); same capability gate. |
| `playback.ack` | 3 | Report cumulative playback progress (`played_ms`, `committed_ms`, optional `truncate`). |

### Server to client

| Event type | Tier | Description |
| --- | --- | --- |
| `session.created` | 2 | Session opened. OpenAI name; adds `incarnation`, `attachment_generation`, `resume_token` and vLLM-Omni keys inside `session` (`epoch`, `turn_id`, `playback`, `capabilities`, ...). |
| `session.updated` | 2 | Echo of the effective session config after every `session.update` (same extended `session` object). |
| `session.heartbeat_ack` | 3 | Reply to `session.heartbeat`. |
| `session.closed` | 3 | Last event on the socket; `reason` ∈ `session_close`, `timeout`, `disconnect`, ... |
| `session.resumed` | 3 | Resume accepted; carries the new `attachment_generation` and rotated `resume_token`; journaled events are replayed after it. |
| `session.replaced` | 3 | Sent to the superseded socket when another socket resumes the session. |
| `session.expired` | 3 | Engine lease reaped (`disconnect_grace_expired`, `idle_ttl_expired`); socket closes after it. |
| `session.resync_required` | 3 | Replay impossible (`journal_gap` / `journal_overflow`); client must start a new session. |
| `input_audio_buffer.speech_started` | 1 | Speech onset detected (server VAD, client hint, or RMS heuristic). |
| `input_audio_buffer.speech_stopped` | 1 | Speech end detected. |
| `input_audio_buffer.committed` | 2 | Commit accepted. OpenAI name; wraps the native `input.committed` payload under `event` (`turn_id`, `epoch`, `response_create_deferred`, ...). |
| `input_audio_buffer.cleared` | 1 | Input buffer cleared (after `input_audio_buffer.clear` or `input.cancel`). |
| `conversation.item.added` | 2 | Item entered history (GA spelling). OpenAI name; carries `previous_item_id` and echoes the originating event. |
| `conversation.item.created` | 2 | Same as `conversation.item.added` in the beta spelling; both are emitted. |
| `conversation.item.done` | 2 | Item finalized (GA spelling). |
| `conversation.item.input_audio_transcription.completed` | 1 | Transcript of a committed user audio item (model-produced). |
| `conversation.item.deleted` | 2 | Reply to `conversation.item.delete` (echoes the request under `event`). |
| `conversation.item.retrieved` | 1 | Reply to `conversation.item.retrieve`. |
| `conversation.item.truncated` | 2 | Reply to `conversation.item.truncate` (echoes the request under `event`). |
| `response.created` | 2 | A visible assistant response starts. OpenAI name; top-level `response_id`, raw duplex event under `response.metadata`. |
| `response.output_item.added` | 1 | Assistant item (or `function_call` item) attached to the response. |
| `response.content_part.added` | 1 | Audio/text content part opened on the assistant item. |
| `response.speak` | 3 | Model chose to speak (native lane); at most once per response, before the first audio delta. |
| `response.audio.delta` | 2 | One ordered audio chunk. OpenAI name; adds `format`, `sample_rate_hz`, `metadata{epoch, model_speak, end_of_turn, audio_text_marks, playback}`. |
| `response.audio_transcript.delta` | 1 | Transcript text paired one-to-one with each audio delta. |
| `response.output_text.delta` | 1 | Text delta for text-modality responses. |
| `response.output_text.done` | 1 | Final text of a text-modality response. |
| `response.audio.done` | 1 | Audio stream closed for the response. |
| `response.audio_transcript.done` | 1 | Full transcript (concatenation of the deltas). |
| `response.content_part.done` | 1 | Content part finalized with its transcript/text. |
| `response.output_item.done` | 1 | Assistant item finalized. |
| `response.done` | 2 | Terminal event; `status` ∈ `completed` \| `cancelled` \| `failed`. OpenAI name; vLLM-Omni `status_details.reason` values and raw duplex event under `metadata`. |
| `rate_limits.updated` | 1 | Compatibility event after every `response.done`; always an empty list. |
| `response.listen` | 3 | Model chose to keep listening at a unit boundary or after a silence commit; no `response.created` precedes it. |
| `response.function_call_arguments.delta` | 1 | Tool-call arguments delta (Nemotron VoiceChat). |
| `response.function_call_arguments.done` | 1 | Tool-call arguments complete. |
| `duplex.function_call.done` | 3 | Raw fallback when a model tool call is malformed. |
| `output_audio_buffer.cleared` | 1 | Reply to `output_audio_buffer.clear`. |
| `playback.acknowledged` | 3 | Reply to `playback.ack`; returns the playback ledger and `history_committed`. |
| `overlap.decision` | 3 | How an append that overlapped an active response was handled (`drop` \| `listen` \| `barge_in`). |
| `runtime.control` | 3 | Diagnostic: an engine control signal was partially unsupported. |
| `error` | 2 | Error envelope. OpenAI shape and classes; vLLM-Omni error codes. |

## Message examples

Field values are illustrative; base64 payloads are shortened to `"…"`.
`C→S` = client sends, `S→C` = server sends. Every `S→C` event on the wire
additionally carries `"server_event_seq": <int>` (omitted below for brevity).

### Session lifecycle

`C→S session.update` (first one = open handshake)

```json
{
  "type": "session.update",
  "event_id": "evt_0001",
  "session": {
    "model": "openbmb/MiniCPM-o-4_5",
    "session_id": "sess_demo_01",
    "modalities": ["audio", "text"],
    "instructions": "You are a concise voice assistant.",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "ref_audio": "data:audio/wav;base64,…",
    "turn_detection": null,
    "overlap_policy": "listen_only",
    "playback_commit_policy": "ack_only",
    "temperature": 0.7,
    "idle_timeout_s": 300,
    "extra_body": {
      "auto_response": true,
      "minicpmo45_native_duplex": true,
      "force_listen_count": 0
    }
  }
}
```

`S→C session.created`

```json
{
  "type": "session.created",
  "incarnation": 0,
  "attachment_generation": 1,
  "resume_token": "rt_3f9c…",
  "session": {
    "object": "realtime.session",
    "type": "realtime",
    "id": "sess_demo_01",
    "model": "openbmb/MiniCPM-o-4_5",
    "state": "open",
    "turn_state": "idle",
    "epoch": 0,
    "turn_id": 0,
    "active_request_id": null,
    "active_response_id": null,
    "active_response_turn_id": null,
    "modalities": ["audio", "text"],
    "output_modalities": ["audio", "text"],
    "instructions": "You are a concise voice assistant.",
    "voice": null,
    "response_format": "pcm16",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "audio": {
      "input": {"format": {"type": "audio/pcm", "rate": 16000}, "sample_rate_hz": 16000},
      "output": {"format": {"type": "audio/pcm", "rate": 24000}}
    },
    "temperature": 0.7,
    "max_tokens": null,
    "speed": null,
    "idle_timeout_s": 300.0,
    "overlap_policy": "listen_only",
    "overlap_short_ack_ms": 700,
    "overlap_barge_in_ms": 1200,
    "overlap_silence_rms": 0.003,
    "playback_commit_policy": "ack_only",
    "turn_detection": null,
    "input_audio_transcription": null,
    "tracing": null,
    "playback": {"generated_ms": 0, "sent_ms": 0, "played_ms": 0, "committed_ms": 0},
    "capabilities": {
      "implementation_level": "model_native_duplex",
      "supports_input_append": true,
      "supports_barge_in": true,
      "supports_session_resume": true,
      "supports_session_lease": true,
      "supports_multi_session": true,
      "chunk_period_ms": 1000,
      "input_modes": ["append_audio_chunk"],
      "signal_sources": ["model_native", "client_event", "server_policy"],
      "target_barge_in_latency_ms": null
    }
  }
}
```

`S→C session.updated` (same `session` object; follows the first
`session.created` and every later `session.update`)

```json
{"type": "session.updated", "session": {"object": "realtime.session", "id": "sess_demo_01", "…": "…"}}
```

`C→S session.heartbeat` / `S→C session.heartbeat_ack`

```json
{"type": "session.heartbeat"}
```

```json
{"type": "session.heartbeat_ack", "session_id": "sess_demo_01"}
```

`C→S session.event_ack`

```json
{"type": "session.event_ack", "server_event_seq": 48}
```

`C→S session.close` / `S→C session.closed`

```json
{"type": "session.close"}
```

```json
{"type": "session.closed", "session_id": "sess_demo_01", "reason": "session_close"}
```

`C→S session.resume` (new socket, `?duplex=1&autostart=0`)

```json
{
  "type": "session.resume",
  "session_id": "sess_demo_01",
  "incarnation": 0,
  "resume_token": "rt_3f9c…",
  "last_received_server_event_seq": 40
}
```

`S→C session.resumed` (activation; the token is rotated, then journaled
events with `server_event_seq > 40` are replayed in order)

```json
{
  "type": "session.resumed",
  "session_id": "sess_demo_01",
  "incarnation": 0,
  "attachment_generation": 2,
  "resume_token": "rt_a71e…"
}
```

`S→C session.replaced` (sent to the superseded socket, which is then closed)

```json
{"type": "session.replaced", "session_id": "sess_demo_01", "attachment_generation": 1}
```

`S→C session.expired`

```json
{"type": "session.expired", "session_id": "sess_demo_01", "incarnation": 0, "reason": "disconnect_grace_expired"}
```

`S→C session.resync_required`

```json
{"type": "session.resync_required", "session_id": "sess_demo_01", "reason": "journal_gap"}
```

### Input

`C→S input_audio_buffer.append` (200 ms of 16 kHz PCM16 = 6400 bytes)

```json
{
  "type": "input_audio_buffer.append",
  "event_id": "evt_0042",
  "audio": "…",
  "input_audio_format": "pcm16",
  "sample_rate_hz": 16000,
  "duration_ms": 200,
  "audio_end_ms": 1200,
  "is_speech": true
}
```

with camera frames (omni video; rides the append that closes a 1 s model unit)

```json
{
  "type": "input_audio_buffer.append",
  "audio": "…",
  "input_audio_format": "pcm16",
  "sample_rate_hz": 16000,
  "duration_ms": 200,
  "audio_end_ms": 1200,
  "video_frames": ["data:image/jpeg;base64,…", "data:image/jpeg;base64,…"]
}
```

`S→C input_audio_buffer.speech_started` / `speech_stopped`

```json
{"type": "input_audio_buffer.speech_started", "audio_start_ms": 1040, "item_id": "item_9b2d…"}
```

```json
{"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 2860, "item_id": "item_9b2d…"}
```

`C→S input_audio_buffer.commit`

```json
{"type": "input_audio_buffer.commit", "event_id": "evt_0060", "final": true}
```

alias form accepted by the runner (`input.commit`)

```json
{"type": "input.commit", "final": true}
```

silence/noise declaration (no runtime append; answered with `response.listen`)

```json
{"type": "input_audio_buffer.commit", "final": true, "is_speech": false}
```

`S→C` commit fan-out (in this order)

```json
{
  "type": "conversation.item.added",
  "previous_item_id": null,
  "item": {
    "id": "item_9b2d…",
    "object": "realtime.item",
    "type": "message",
    "role": "user",
    "status": "completed",
    "content": [{"type": "input_audio", "transcript": "what is the weather today"}]
  }
}
```

```json
{"type": "conversation.item.created", "previous_item_id": null, "item": {"id": "item_9b2d…", "…": "…"}}
```

```json
{
  "type": "input_audio_buffer.committed",
  "previous_item_id": null,
  "item_id": "item_9b2d…",
  "event": {
    "type": "input.committed",
    "session_id": "sess_demo_01",
    "turn_id": 1,
    "epoch": 0,
    "history_len": 1,
    "message": {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}]},
    "realtime_item_id": "item_9b2d…"
  }
}
```

```json
{
  "type": "conversation.item.input_audio_transcription.completed",
  "item_id": "item_9b2d…",
  "content_index": 0,
  "transcript": "what is the weather today"
}
```

```json
{"type": "conversation.item.done", "previous_item_id": null, "item": {"id": "item_9b2d…", "status": "completed", "…": "…"}}
```

deferred commit during an active response (`event.response_create_deferred`)

```json
{
  "type": "input_audio_buffer.committed",
  "previous_item_id": "item_9b2d…",
  "item_id": "item_c044…",
  "event": {"type": "input.committed", "session_id": "sess_demo_01", "turn_id": 2, "epoch": 0, "history_len": 3, "response_create_deferred": true, "message": {"…": "…"}}
}
```

`C→S input_audio_buffer.clear` / `S→C input_audio_buffer.cleared`

```json
{"type": "input_audio_buffer.clear"}
```

```json
{"type": "input_audio_buffer.cleared"}
```

`C→S input.cancel` (advances `epoch`; also answered with `input_audio_buffer.cleared`)

```json
{"type": "input.cancel"}
```

`C→S input.text.append`

```json
{"type": "input.text.append", "text": "and tomorrow?"}
```

`C→S conversation.item.create` (text turn, or a tool result)

```json
{
  "type": "conversation.item.create",
  "previous_item_id": null,
  "item": {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "Summarise what I said."}]
  }
}
```

```json
{
  "type": "conversation.item.create",
  "item": {"type": "function_call_output", "call_id": "call_7d1f", "output": "{\"temperature_c\": 24}"}
}
```

`C→S conversation.item.delete` / `S→C conversation.item.deleted`

```json
{"type": "conversation.item.delete", "item_id": "item_9b2d…"}
```

```json
{"type": "conversation.item.deleted", "item_id": "item_9b2d…", "event": {"type": "conversation.item.deleted", "item_id": "item_9b2d…"}}
```

`C→S conversation.item.retrieve` / `S→C conversation.item.retrieved`

```json
{"type": "conversation.item.retrieve", "item_id": "item_resp_01"}
```

```json
{"type": "conversation.item.retrieved", "item": {"id": "item_resp_01", "object": "realtime.item", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_audio", "transcript": "It is sunny, 24 degrees."}]}}
```

`C→S conversation.item.truncate` / `S→C conversation.item.truncated`

```json
{"type": "conversation.item.truncate", "item_id": "item_resp_01", "content_index": 0, "audio_end_ms": 1850}
```

```json
{"type": "conversation.item.truncated", "item_id": "item_resp_01", "content_index": 0, "audio_end_ms": 1850, "event": {"…": "…"}}
```

### Response lifecycle

`C→S response.create`

```json
{
  "type": "response.create",
  "event_id": "evt_0070",
  "response": {
    "modalities": ["audio", "text"],
    "instructions": "Answer in one sentence.",
    "output_audio_format": "pcm16",
    "temperature": 0.6,
    "max_output_tokens": 256,
    "metadata": {"trace": "turn-3"}
  }
}
```

`S→C response.created`

```json
{
  "type": "response.created",
  "response_id": "resp_01",
  "response": {
    "id": "resp_01",
    "object": "realtime.response",
    "status": "in_progress",
    "status_details": null,
    "output": [],
    "modalities": ["audio", "text"],
    "metadata": {"duplex_event": {"type": "response.created", "session_id": "sess_demo_01", "response_id": "resp_01", "epoch": 0, "turn_id": 1}}
  }
}
```

`S→C conversation.item.added` / `conversation.item.created` (assistant item, `status: in_progress`)

```json
{
  "type": "conversation.item.added",
  "previous_item_id": "item_9b2d…",
  "item": {"id": "item_resp_01", "object": "realtime.item", "type": "message", "role": "assistant", "status": "in_progress", "content": []}
}
```

`S→C response.output_item.added`

```json
{"type": "response.output_item.added", "response_id": "resp_01", "output_index": 0, "item": {"id": "item_resp_01", "object": "realtime.item", "type": "message", "role": "assistant", "status": "in_progress", "content": []}}
```

`S→C response.content_part.added`

```json
{"type": "response.content_part.added", "response_id": "resp_01", "item_id": "item_resp_01", "output_index": 0, "content_index": 0, "part": {"type": "audio", "transcript": ""}}
```

`S→C response.speak`

```json
{
  "type": "response.speak",
  "response_id": "resp_01",
  "item_id": "item_resp_01",
  "output_index": 0,
  "content_index": 0,
  "metadata": {"session_id": "sess_demo_01", "epoch": 0, "model_speak": true}
}
```

`S→C response.audio.delta` (one ~1 s unit, 24 kHz PCM16)

```json
{
  "type": "response.audio.delta",
  "response_id": "resp_01",
  "item_id": "item_resp_01",
  "output_index": 0,
  "content_index": 0,
  "delta": "…",
  "format": "pcm16",
  "sample_rate_hz": 24000,
  "metadata": {
    "session_id": "sess_demo_01",
    "epoch": 0,
    "model_speak": true,
    "end_of_turn": false,
    "audio_duration_ms": 1000,
    "audio_text_marks": [{"text_chars": 11, "audio_end_ms": 1000}],
    "playback": {"generated_ms": 1000, "sent_ms": 1000, "played_ms": 0, "committed_ms": 0}
  }
}
```

`S→C response.audio_transcript.delta` (exactly one per audio delta)

```json
{"type": "response.audio_transcript.delta", "response_id": "resp_01", "item_id": "item_resp_01", "output_index": 0, "content_index": 0, "delta": "It is sunny"}
```

`S→C response.output_text.delta` / `response.output_text.done` (text modality)

```json
{"type": "response.output_text.delta", "response_id": "resp_02", "item_id": "item_resp_02", "output_index": 0, "content_index": 0, "delta": "Tomorrow looks"}
```

```json
{"type": "response.output_text.done", "response_id": "resp_02", "item_id": "item_resp_02", "output_index": 0, "content_index": 0, "text": "Tomorrow looks cloudy."}
```

`S→C response.audio.done` / `response.audio_transcript.done`

```json
{"type": "response.audio.done", "response_id": "resp_01", "item_id": "item_resp_01", "output_index": 0, "content_index": 0}
```

```json
{"type": "response.audio_transcript.done", "response_id": "resp_01", "item_id": "item_resp_01", "output_index": 0, "content_index": 0, "transcript": "It is sunny, 24 degrees."}
```

`S→C response.content_part.done`

```json
{"type": "response.content_part.done", "response_id": "resp_01", "item_id": "item_resp_01", "output_index": 0, "content_index": 0, "part": {"type": "audio", "transcript": "It is sunny, 24 degrees."}}
```

`S→C response.output_item.done` / `conversation.item.done`

```json
{"type": "response.output_item.done", "response_id": "resp_01", "output_index": 0, "item": {"id": "item_resp_01", "object": "realtime.item", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_audio", "transcript": "It is sunny, 24 degrees."}]}}
```

```json
{"type": "conversation.item.done", "previous_item_id": "item_9b2d…", "item": {"id": "item_resp_01", "status": "completed", "…": "…"}}
```

`S→C response.done` (completed)

```json
{
  "type": "response.done",
  "response_id": "resp_01",
  "response": {
    "id": "resp_01",
    "object": "realtime.response",
    "status": "completed",
    "status_details": null,
    "output": [{"id": "item_resp_01", "object": "realtime.item", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_audio", "transcript": "It is sunny, 24 degrees."}]}],
    "metadata": {"type": "response.done", "session_id": "sess_demo_01", "response_id": "resp_01", "epoch": 0, "committed": true, "playback": {"generated_ms": 2400, "sent_ms": 2400, "played_ms": 2400, "committed_ms": 2400}}
  }
}
```

`S→C response.done` (cancelled by barge-in)

```json
{
  "type": "response.done",
  "response_id": "resp_01",
  "response": {"id": "resp_01", "object": "realtime.response", "status": "cancelled", "status_details": {"type": "cancelled", "reason": "barge_in"}, "output": [{"id": "item_resp_01", "status": "cancelled", "…": "…"}], "metadata": {"…": "…"}}
}
```

`S→C rate_limits.updated`

```json
{"type": "rate_limits.updated", "rate_limits": []}
```

`S→C response.listen`

```json
{
  "type": "response.listen",
  "session_id": "sess_demo_01",
  "epoch": 0,
  "response": {"object": "realtime.response", "status": "listening", "metadata": {"type": "response.listen", "session_id": "sess_demo_01", "epoch": 0, "reason": "model_listen"}}
}
```

`C→S response.cancel`

```json
{"type": "response.cancel", "response_id": "resp_01"}
```

`C→S output_audio_buffer.clear` / `S→C output_audio_buffer.cleared`

```json
{"type": "output_audio_buffer.clear"}
```

```json
{"type": "output_audio_buffer.cleared", "response_id": "resp_01"}
```

`C→S barge_in` / `turn.signal`

```json
{"type": "barge_in"}
```

```json
{"type": "turn.signal", "event": "barge_in", "payload": {"source": "client_vad"}}
```

function-call response (Nemotron VoiceChat tools)

```json
{"type": "response.output_item.added", "response_id": "resp_fc_01", "output_index": 0, "item": {"id": "item_fc_01", "object": "realtime.item", "type": "function_call", "status": "completed", "name": "get_weather", "call_id": "call_7d1f", "arguments": "{\"city\":\"Hong Kong\"}"}}
```

```json
{"type": "response.function_call_arguments.delta", "response_id": "resp_fc_01", "item_id": "item_fc_01", "output_index": 0, "call_id": "call_7d1f", "delta": "{\"city\":\"Hong Kong\"}"}
```

```json
{"type": "response.function_call_arguments.done", "response_id": "resp_fc_01", "item_id": "item_fc_01", "output_index": 0, "call_id": "call_7d1f", "arguments": "{\"city\":\"Hong Kong\"}"}
```

raw fallback when a model tool call is malformed (`call_id`/`name` missing)

```json
{"type": "duplex.function_call.done", "event": {"type": "function_call.done", "session_id": "sess_demo_01", "epoch": 0, "call_id": null, "name": null, "arguments": "{"}}
```

### Playback

`C→S playback.ack` / `S→C playback.acknowledged`

```json
{"type": "playback.ack", "response_id": "resp_01", "item_id": "item_resp_01", "played_ms": 1000, "committed_ms": 1000}
```

```json
{
  "type": "playback.acknowledged",
  "event": {
    "type": "playback.acknowledged",
    "session_id": "sess_demo_01",
    "epoch": 0,
    "item_id": "item_resp_01",
    "played_ms": 1000,
    "committed_ms": 1000,
    "truncate": false,
    "playback": {"generated_ms": 2400, "sent_ms": 2400, "played_ms": 1000, "committed_ms": 1000},
    "history_committed": false
  }
}
```

truncating ack after an interruption (what was actually heard)

```json
{"type": "playback.ack", "response_id": "resp_01", "played_ms": 1850, "committed_ms": 1850, "truncate": true}
```

### Overlap and barge-in

`S→C overlap.decision`

```json
{
  "type": "overlap.decision",
  "session_id": "sess_demo_01",
  "epoch": 0,
  "policy": "listen_only",
  "action": "listen",
  "reason": "long_overlap_speech",
  "overlap_ms": 1400,
  "buffer_audio": true,
  "defer_runtime_append": true,
  "force_listen": true
}
```

```json
{"type": "overlap.decision", "session_id": "sess_demo_01", "epoch": 0, "policy": "barge_in_on_speech", "action": "barge_in", "reason": "server_vad_utterance_active"}
```

server-VAD session (`session.update` fragment) and the annotated append the
translator forwards internally (shown for reference; clients never see it)

```json
{"type": "session.update", "session": {"turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500}, "overlap_policy": "barge_in_on_speech"}}
```

```json
{"type": "input_audio_buffer.append", "audio": "…", "format": "pcm16", "sample_rate_hz": 16000, "is_speech": true, "force_listen": true, "vad": {"backend": "silero", "is_speech": true, "speech_active": true, "speech_started": true, "speech_stopped": false, "speech_probability": 0.93}}
```

`S→C runtime.control` (only when an engine signal was partially unsupported)

```json
{"type": "runtime.control", "session_id": "sess_demo_01", "epoch": 1, "result": {"…": "redacted engine control result"}}
```

### Errors

`S→C error` — Realtime shape

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "code": "input_audio_buffer_empty",
    "message": "input_audio_buffer.commit requires a non-empty input audio buffer.",
    "event_id": "evt_0060"
  }
}
```

```json
{"type": "error", "error": {"type": "invalid_request_error", "code": "unsupported_turn_detection", "message": "turn_detection.interrupt_response=false is unsupported; use turn_detection=null for model-owned listen/speak", "event_id": "evt_0001", "param": "turn_detection"}}
```

```json
{"type": "error", "error": {"type": "server_error", "code": "runtime_append_failed", "message": "stale epoch: expected 1, got 0"}}
```

```json
{"type": "error", "error": {"type": "rate_limit_error", "code": "duplex_session_busy", "message": "duplex session capacity reached (max_sessions=4)"}}
```

Handshake-stage errors (before the Realtime projector is bound) use the
flat native shape:

```json
{"type": "error", "error": "Unknown or expired duplex session: sess_demo_01", "code": "session_resume_expired"}
```

### Coverage

The examples above cover every event type the Realtime route emits or
accepts (verified by diffing the `type` literals in
`vllm_omni/entrypoints/duplex/` against this document). Not shown, because
they never reach a `/v1/realtime?duplex=1` client unprojected: the native
`/v1/duplex` dialect (`session.create` / `open_session` / `session.config`,
`input.committed`, `input.cancelled`, `audio.cancelled`,
`response.output_audio.delta`, `response.text.delta`, `response.message`,
`function_call.done`) — their Realtime projections are the examples above
(see the name map below) — and the runner-internal markers `__timeout__`,
`__disconnect__`, `__replaced_attachment__`. Input aliases (`push_chunk`,
`input.audio.append`, `input_text.append`, `push_text`, `signal_turn`,
`audio.playback_ack`, `close_session`, `close`, `session_close`) share the
payload of their canonical event (see the alias table in [fullduplex.md](fullduplex.md)).

### Native to Realtime name map

The native `/v1/duplex` dialect is what the session runner produces; the
Realtime projector renames and fans out these events before they reach a
`/v1/realtime?duplex=1` client.

| Native / internal event | Realtime projection |
| --- | --- |
| `session.create` (`open_session`, `session.config`) | first `session.update` |
| `input.committed` | `conversation.item.added` / `conversation.item.created`, `input_audio_buffer.committed`, `conversation.item.input_audio_transcription.completed`, `conversation.item.done` |
| `input.cancelled` | `input_audio_buffer.cleared` |
| `response.output_audio.delta` (with `audio_transcript`) | optional `response.speak`, then `response.audio.delta` and `response.audio_transcript.delta` |
| `response.output_audio.done` / `response.output_text.done` | `response.audio.done` / `response.output_text.done` |
| `response.text.delta` | `response.output_text.delta` |
| `response.message` (chat-fallback raw chunk) | passed through |
| `audio.cancelled` | optional `output_audio_buffer.cleared`, then the cancelled terminal events and `response.done` with `status: "cancelled"` |
| `function_call.done` | `response.created`, `response.output_item.added` (`function_call` item), `response.function_call_arguments.delta` / `.done`, `response.output_item.done`, `response.done` |
| `playback.acknowledged` | `playback.acknowledged` (wrapped verbatim under `event`) |
| `runtime.control`, `session.resumed`, `session.replaced`, `session.expired`, `session.resync_required`, `session.heartbeat_ack` | passed through unchanged |
