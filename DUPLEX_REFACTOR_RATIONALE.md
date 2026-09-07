# Duplex Refactor: The Three Most Important Reasons

Branch `duplex_refactor1`, written 2026-09-05. Numbers are from
`DUPLEX_REFACTOR_LOC_BASELINE.md` (effective lines, tests / examples / `experimental/` /
`clients/` excluded).

## 1. A duplex session now has exactly one owner

Before, one session was spread over two processes and two half-implementations: an API-side
`DuplexSession` / registry / session runner / runtime bridge, and an engine-side control plane.
Keeping the two halves consistent needed a cross-boundary protocol of its own: fences carried in
every message, `next_fence` / `expected_epoch` checks on both sides, an `operation_id` idempotency
cache, `DuplexControlClient`, `DuplexRequestClient` and a separate lifecycle-event queue. Every
bug in ordering, cancellation or resume had to be reasoned about across that seam.

After, the session is engine-resident: one `DuplexEngineSession` owned by one
`DuplexSessionRunner` on the orchestrator loop, admitted and fenced once in
`DuplexSessionManager.dispatch`. The API side is a pipe (`DuplexOmni`, `DuplexSessionHandle`)
and the serving layer is a websocket adapter. The whole cross-boundary protocol is deleted.

Evidence: the serving layer went from 18 files / 10,769 effective lines to 6 files / 1,283
lines, with no file left in radon's lowest maintainability rank (baseline had 6).
Engine + serving together shrank by 822 lines while absorbing all of the moved logic.

## 2. Explicit, typed contracts replace dicts, Protocols and per-model façades

Before, commands and events crossed layers as untyped dicts with ad-hoc aliases, eight
`typing.Protocol` classes described the seams structurally, and each model needed several
loosely coupled pieces (runtime extension, serving adapter, capability probe, `native_duplex`
toggles in client, server and model code) that had to agree with each other by convention.

After, there is one `DuplexCommand` / `DuplexEvent` dataclass vocabulary with
`from_realtime()` / `to_realtime()` at the edge, all seams are ABCs, and each model provides one
`DuplexModelPlugin` selected by `PipelineConfig.duplex_plugin`. The engine validates the
deployment (plugin present, `session_mode: duplex`) at startup instead of at first request,
and the `native_duplex` client toggle is gone: a duplex model is always duplex.

Evidence: three model façade layers folded into one class each; every `Protocol` in duplex
code removed; wiring is checkable statically (import checker, `ruff`, `compileall`) rather
than only at runtime.

## 3. The turn-based path no longer carries duplex code

Before, `AsyncOmni`, `AsyncOmniEngine` and `Orchestrator` each held duplex members, branches
and message routing that every non-duplex model paid for in complexity and that every duplex
change risked breaking.

After, each layer is split into a base plus two siblings: `AsyncOmniBase` -> `AsyncOmni` /
`DuplexOmni`, `OmniEngineBase` -> `AsyncOmniEngine` / `DuplexOmniEngine`, `OrchestratorBase`
-> `Orchestrator` / `DuplexOrchestrator`, wired by explicit `_create_engine()` /
`_create_orchestrator()` factories. Duplex behaviour enters only through named template seams.

Evidence: `async_omni_engine.py` 1,748 -> 516 lines, `async_omni.py` 1,252 -> 746,
`orchestrator.py` 2,375 -> 2,073; a search for "duplex" in the generic classes now finds only
comments. The two new base files (`omni_engine_base.py` 997, `async_omni_base.py` 362) hold
shared code, not duplex code.
