# Full-Duplex Update Sites: Existing Scripts That Must Be Edited

Companion to `FULLDUPLEX_MIGRATION_DESIGN.md` (what moves) and
`FULLDUPLEX_NEXT_MODELS_LAYOUT.md` (what a new model adds). This document
answers the third question: **which existing files are edited in place**
when the full-duplex graduation (both waves — MiniCPM-o first wave §2–§7,
PersonaPlex second wave §17) is replayed on upstream main.

Inventory source: `grep -rl "experimental.fullduplex"` over upstream main
(`d5d47e9b` → current) excluding `vllm_omni/experimental/` itself and the
test trees that §6/§17.2 relocate wholesale. Every file below exists today
and is **edited, not moved**. Unless stated otherwise the edit is an
import/path retarget: `vllm_omni.experimental.fullduplex.X` → the stable
home given in the migration design.

```text
vllm_omni/
├── engine/
│   ├── orchestrator.py                  # retarget control-plane/session/lease imports → engine/duplex/*;
│   │                                    #   make them eager top-level imports (§3 revised policy)
│   └── async_omni_engine.py             # retarget control-client/messages imports → engine/duplex/*; eager
├── entrypoints/
│   ├── async_omni.py                    # retarget request-client import → entrypoints/duplex_request_client;
│   │                                    #   duplex proxies unchanged (Python API surface stays)
│   └── openai/
│       └── api_server.py                # retarget handler import → entrypoints/openai/duplex/serving;
│                                        #   gate via duplex_capability.should_enable_duplex_endpoint(); eager
├── worker/
│   └── gpu_ar_model_runner.py           # retarget DuplexSamplingRow import → model_executor/duplex_sampling;
│                                        #   hook resolution logic unchanged (duck-typed)
├── model_executor/
│   ├── stage_input_processors/
│   │   └── minicpmo_4_5_omni.py         # retarget ref-audio decode import → models/minicpmo_4_5/duplex/input
│   └── models/
│       ├── minicpmo_4_5/
│       │   ├── minicpmo_4_5_omni.py     # retarget policy/sampling/handoff imports → models/minicpmo_4_5/duplex/*
│       │   ├── minicpmo_4_5_omni_llm.py #   + model_executor/duplex_sampling (eager at model level is fine —
│       │   ├── minicpmo_4_5_omni_tts.py #   these files load only when the model is served)
│       │   └── pipeline.py              # retarget the two dotted strings:
│       │                                #   duplex_runtime_extension / duplex_serving_adapter
│       │                                #   → vllm_omni.model_executor.models.minicpmo_4_5.duplex.*
│       └── personaplex/
│           ├── personaplex_talker.py    # retarget duplex imports → models/personaplex/duplex/* (wave 2)
│           ├── personaplex_code2wav.py  # retarget duplex imports → models/personaplex/duplex/* (wave 2)
│           └── pipeline.py              # retarget the two dotted strings
│                                        #   → vllm_omni.model_executor.models.personaplex.duplex.*
│
├── (config/pipeline_registry.py, engine/arg_utils.py — NO edit: they only
│    reference the stable model tree, which does not move)
│
.buildkite/cuda/
├── test-ready.yml                       # replace experimental/fullduplex/ path filters with the stable
└── test-merge.yml                       #   duplex paths (engine/duplex/, entrypoints/openai/duplex/,
                                         #   models/*/duplex/) + relocated test dirs
pyproject.toml                           # keep the joyvl prompts E501 ignore (joyvl stays experimental);
                                         #   no new entries needed — package discovery is vllm_omni*
examples/
├── online_serving/minicpmo/
│   ├── README.md                        # update module paths in prose
│   └── realtime_duplex_demo.py          # retarget client import →
│                                        #   models/minicpmo_4_5/duplex/client (§15 home)
├── online_serving/personaplex/
│   └── README.md                        # rewrite: standalone Moshi-web server is dropped (§17.2) —
│                                        #   document /v1/duplex + /v1/realtime?duplex=1 serving instead
└── offline_inference/personaplex/
    └── personaplex_offline.py           # ✔ RESOLVED (2026-08-10): dropped with the demo set — it imports
                                         #   PersonaPlexEngine/PersonaPlexSession, the single-process trio
                                         #   §17.2 drops. Offline PersonaPlex inference is unsupported until
                                         #   an offline path exists over the stage machinery.
tests/
├── engine/
│   ├── test_duplex_import_boundary.py   # contract: pin new eager/forbidden module lists (both waves)
│   ├── test_orchestrator.py             # import retargets
│   ├── test_orchestrator_stage_input_bridge.py
│   └── test_async_omni_engine_outputs.py
├── entrypoints/
│   ├── test_async_omni_duplex.py        # import retargets (Python-API duplex coverage)
│   ├── test_duplex_fence_propagation.py
│   ├── openai/test_duplex_protocol.py   # + capability preset import → models/minicpmo_4_5/duplex/capabilities
│   ├── openai/test_duplex_session_attachment.py
│   ├── openai_api/test_duplex_capability.py  # contract: demo-client path pin (§15 home)
│   └── openai_api/test_duplex_handler.py     # + generic accessors instead of _minicpmo_* (§15)
├── worker/test_native_duplex_hooks.py   # import retargets
├── model_executor/models/minicpmo_4_5/test_pipeline.py  # dotted-string expectations
└── e2e/online_serving/
    ├── test_minicpmo_4_5_duplex.py      # client import retarget
    ├── minicpmo_realtime_duplex_scenarios.py
    └── personaplex_realtime_duplex.py   # ✔ RESOLVED (2026-08-10): kept — one-line retarget of the demo
                                         #   client import to models/minicpmo_4_5/duplex/client.py (the
                                         #   client is protocol-generic; a neutral home for it is a
                                         #   possible future cleanup)
docs/design/
├── session-state-generalization.md      # path references → stable homes
├── session-state-generalization.engine-hosted-joyvl.bak.md  # historical — annotate, don't rewrite
└── dead-code-sweep.md                   # historical — annotate, don't rewrite
```

Notes:

- **Wave-1 edits are already proven.** Every `vllm_omni/` + minicpmo entry
  above was already executed once on the `fullduplex` branch (design doc
  §5, §11, §15) and validated by the relocated suites; on upstream they
  must be replayed as a 3-way merge against drift, not cherry-picked.
- **Both former ⚠ items are resolved** (see the ✔ annotations above): the
  offline example was dropped, the e2e driver kept with a retargeted client
  import. Everything else in this file was mechanical retargeting, executed
  on this branch on 2026-08-10 together with the upstream merge.
- **What is deliberately absent:** `config/pipeline_registry.py`,
  `engine/arg_utils.py`, and the model GPU files' registration — they
  point at the stable model tree, which never moves. The `/v1/duplex` and
  `/v1/realtime` route handlers in `api_server.py` change imports only;
  no route paths change.
