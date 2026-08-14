# Experimental Full-Duplex

This package contains two experimental integrations:

```text
core/        generic duplex scaffold (adapter, session, turn runtime)
joyvl/       JoyVL model-specific integration
personaplex/ PersonaPlex lockstep engine, model-owned runtime, and serving
```

To run JoyVL, see
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).

`personaplex/` is a Moshi-class, pure-lockstep speech-to-speech model on the
`core/` contracts. It keeps `core/` untouched: the lockstep lifecycle (ONE
eternal, frame-clocked response that drains on close, instead of the turn-style
start/cancel-per-trigger one) is model policy and lives in the model package as
`PersonaPlexDuplexRuntime`, mirroring the model-owned runtime shape of the
MiniCPM-o duplex work. Its runnable serving path is `personaplex/serving/`
(single-session lease or `--batch-size` elastic slots) over
`personaplex/session.py` (lockstep driver). See also
`recipes/NVIDIA/PersonaPlex.md`.

The MiniCPM-o 4.5 native full-duplex runtime graduated out of this package.
It now lives in the stable tree:

```text
vllm_omni/engine/duplex/                       engine control plane, sessions, leases
vllm_omni/entrypoints/duplex/           WebSocket serving and Realtime projection
vllm_omni/entrypoints/duplex_request_client.py request/output lifecycle
vllm_omni/model_executor/models/minicpmo_4_5/duplex/  MiniCPM adapter
vllm_omni/model_executor/duplex_sampling.py    AR-runner sampling hook helper
vllm_omni/outputs/duplex.py                    typed output decision envelope
```

For its architecture and validation scope, see
[`docs/design/fullduplex.md`](../../../docs/design/fullduplex.md).

## Adding a full-duplex model on the core contracts

The seam is `core.DuplexAdapter`. `core/` owns the session lifecycle,
epoch-based barge-in, playback cursor, and the event protocol; you implement
only model policy.

1. Create a sibling package `vllm_omni/experimental/fullduplex/<model>/`; keep
   model-specific code there and do not touch `core/`.
2. Implement one `DuplexAdapter` (`capabilities` / `on_input` / `respond`; the
   rest have defaults). Turn-based models run through `core.DuplexRuntime`
   unchanged; a model needing a different lifecycle carries its own runtime in
   its package (see `personaplex/adapter.py::PersonaPlexDuplexRuntime`).
3. Promote a helper from a model package up into `core/` only once a second
   model actually needs it.
