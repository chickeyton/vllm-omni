# Full-Duplex Migration — Test Plan & Results

Validation of the `experimental/fullduplex` graduation (commit
`1c7ecca8 "move code"`, branch `fullduplex`) on a remote GPU test server.
Companion documents: [`FULLDUPLEX_MIGRATION_DESIGN.md`](FULLDUPLEX_MIGRATION_DESIGN.md)
(design + execution log), [`docs/design/fullduplex.md`](docs/design/fullduplex.md)
(runtime architecture).

## 1. Test environment

| Item | Value |
| --- | --- |
| Server | shared test server, 4× NVIDIA L20X (143 GB each) |
| Workspace | `/workspace/ngngaifai/` |
| Repo | `/workspace/ngngaifai/repo/chickeyton/vllm-omni-work` (editable install) |
| Venv | `/workspace/ngngaifai/.venv-fdxtest` — created fresh for this test |
| Python / torch / vLLM | 3.12.13 / 2.11.0+cu129 / **0.25.0** (matches the DESIGN validation pairing; the pre-existing `.venv` had 0.23.0 and was not used) |
| vllm-omni | `0.26.0rc2.dev50+gcc324de3e`, editable from the migration commit |
| Extra test deps | `pytest 9.1.1`, `pytest-asyncio`, `pytest-mock`, `websockets` |
| Model weights | `HF_HOME=/workspace/models` (MiniCPM-o 4.5 present, run offline) |
| GPUs used | free GPUs only (`CUDA_VISIBLE_DEVICES` pinned per run; one GPU was in use by others) |

Code transfer: the migration commit was shipped by `git bundle` + `scp`
(nothing pushed until explicitly requested later), fetched into the server
repo, and checked out as a temporary branch. After validation the server was
re-pointed at the pushed `origin/fullduplex`.

## 2. Test plan

Three batches, cheapest first, each gating the next:

1. **Focused duplex batch** — every relocated test suite plus the
   contract/boundary tests that enforce the migration's import rules:
   - `tests/engine/duplex/` (6 files), `tests/entrypoints/openai/duplex/`
     (4 files), `tests/model_executor/models/minicpmo_4_5/duplex/` (2 files)
   - `tests/engine/test_duplex_import_boundary.py` (lazy-import boundary)
   - `tests/entrypoints/openai_api/test_duplex_capability.py`,
     `test_duplex_handler.py`
   - `tests/entrypoints/openai/test_duplex_protocol.py`,
     `test_duplex_session_attachment.py`
   - `tests/entrypoints/test_async_omni_duplex.py`,
     `test_duplex_fence_propagation.py`
2. **Affected stable batch** — suites covering the stable modules whose
   imports were rewritten:
   - `tests/engine/test_orchestrator.py`,
     `test_orchestrator_stage_input_bridge.py`,
     `test_async_omni_engine_outputs.py`
   - `tests/worker/test_native_duplex_hooks.py`
   - `tests/model_executor/models/minicpmo_4_5/test_pipeline.py` (also proves
     the rewritten dotted-string plugin paths resolve)
3. **E2E batch** — `tests/e2e/online_serving/test_minicpmo_4_5_duplex.py` at
   the `core_model` tier, exactly as CI runs it: real model load, two stages
   on two GPUs, live duplex session with audio in/out. This is the only tier
   that exercises the runtime plugin loading end to end.

## 3. Test scripts (as executed)

### 3.1 Environment setup

```bash
# on the test server
cd /workspace/ngngaifai
python3.12 -m venv .venv-fdxtest
source .venv-fdxtest/bin/activate
pip install -U pip
pip install vllm==0.25.0
pip install -e repo/chickeyton/vllm-omni-work
pip install pytest pytest-asyncio pytest-mock websockets

# import smoke test
python -c 'import vllm, vllm_omni; print("vllm", vllm.__version__); print("omni", vllm_omni.__file__)'
```

### 3.2 Batch 1 — focused duplex suites

```bash
source /workspace/ngngaifai/.venv-fdxtest/bin/activate
export HF_HOME=/workspace/models CUDA_VISIBLE_DEVICES=0,1,2
cd /workspace/ngngaifai/repo/chickeyton/vllm-omni-work
timeout 40m python -m pytest -q \
  tests/engine/duplex/ \
  tests/entrypoints/openai/duplex/ \
  tests/model_executor/models/minicpmo_4_5/duplex/ \
  tests/engine/test_duplex_import_boundary.py \
  tests/entrypoints/openai_api/test_duplex_capability.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/entrypoints/openai/test_duplex_protocol.py \
  tests/entrypoints/openai/test_duplex_session_attachment.py \
  tests/entrypoints/test_async_omni_duplex.py \
  tests/entrypoints/test_duplex_fence_propagation.py \
  2>&1 | tee /workspace/ngngaifai/fdx_test_run1.log
```

### 3.3 Batch 2 — affected stable suites

```bash
source /workspace/ngngaifai/.venv-fdxtest/bin/activate
export HF_HOME=/workspace/models CUDA_VISIBLE_DEVICES=0,1,2
cd /workspace/ngngaifai/repo/chickeyton/vllm-omni-work
timeout 40m python -m pytest -q \
  tests/engine/test_orchestrator.py \
  tests/engine/test_orchestrator_stage_input_bridge.py \
  tests/engine/test_async_omni_engine_outputs.py \
  tests/worker/test_native_duplex_hooks.py \
  tests/model_executor/models/minicpmo_4_5/test_pipeline.py \
  2>&1 | tee /workspace/ngngaifai/fdx_test_run2b.log
```

### 3.4 Batch 3 — MiniCPM-o 4.5 duplex E2E

```bash
source /workspace/ngngaifai/.venv-fdxtest/bin/activate
export HF_HOME=/workspace/models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1
cd /workspace/ngngaifai/repo/chickeyton/vllm-omni-work
timeout 25m python -m pytest -s -v \
  tests/e2e/online_serving/test_minicpmo_4_5_duplex.py \
  -m "core_model and cuda" --run-level core_model \
  2>&1 | tee /workspace/ngngaifai/fdx_test_e2e.log
```

Note: the `profile` file sets `HF_HOME` without `export`, so the variable is
exported explicitly in each script.

### 3.5 Server launch (implicit in the E2E batch)

No manual `vllm-omni serve` was run: the E2E test's `omni_server` pytest
fixture (`tests/helpers/minicpmo_4_5_duplex.py`) launches and tears down the
API server itself, parameterized with
`get_deploy_config_path("minicpmo_4_5_duplex.yaml")` — i.e. the duplex deploy
profile with engine-managed session capacity. The tests then connect over
WebSocket to `ws://<host>:<port>/v1/realtime?duplex=1`.

Manual equivalent, for reproducing the serving side outside pytest (e.g. to
drive `examples/online_serving/minicpmo/realtime_duplex_demo.py` against it):

```bash
source /workspace/ngngaifai/.venv-fdxtest/bin/activate
export HF_HOME=/workspace/models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1
cd /workspace/ngngaifai/repo/chickeyton/vllm-omni-work
vllm-omni serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
# duplex endpoint: ws://<host>:8099/v1/realtime?duplex=1  (or /v1/duplex)
```

## 4. Results

| Batch | Result | Wall time | Log |
| --- | --- | --- | --- |
| 1. Focused duplex | **399 passed** (392 + 7 on rerun after test fixes) | 4m44s + 1m59s | `fdx_test_run1.log` |
| 2. Affected stable | **150 passed** | 6.5s | `fdx_test_run2b.log` |
| 3. E2E duplex | **1 passed**, 2 deselected (other tiers) | 5m07s | `fdx_test_e2e.log` |

**Zero unresolved failures.** Expected warnings only (vllm/vllm-omni version
skew `RuntimeWarning` — normal for this branch; `audioop` deprecation;
pydantic v2 deprecations).

## 5. Failures found during validation (all fixed)

Both real failures were defects in test code rewritten during the migration,
not in the migrated runtime code. Fixes are amended into `1c7ecca8`.

1. `test_engine_duplex_uses_canonical_contract_module_names` asserted the old
   `vllm_omni/experimental/fullduplex/engine/` **directory** no longer exists.
   On the server, stale `__pycache__` from a previous branch checkout kept the
   directory alive. Fixed to assert no `*.py` source files remain
   (`glob("*.py")`), which is robust against pycache litter.
2. `test_api_server_does_not_import_duplex_at_module_load` used a bare
   `startswith("vllm_omni.entrypoints.openai.duplex")`, which falsely matched
   the legitimate stable module `...openai.duplex_capability`. Fixed to
   require an exact match or `prefix + "."` (module boundary).

Environment gaps hit and resolved (not code defects): `pytest-mock` missing
from the minimal dependency install; stale `__pycache__` cleaned with a scoped
`git clean -fdx vllm_omni/experimental/fullduplex/`.

## 6. What this run does and does not prove

Proven:

- All relocated modules import and pass their suites under vLLM 0.25.0.
- The lazy-import boundary holds at the new paths (ordinary deployments load
  zero duplex modules).
- The rewritten dotted-string plugin paths
  (`duplex_runtime_extension`, `duplex_serving_adapter`) resolve at runtime —
  the one failure mode static checks could not cover.
- The full duplex serving path (WebSocket → session → engine control plane →
  Stage0/Stage1 → Realtime projection) works end to end on the migrated tree
  at the `core_model` E2E tier.

Not covered by this run (unchanged claims from `docs/design/fullduplex.md`):

- the `advanced_model` E2E tier and multi-session/resume/takeover scenario
  drivers;
- audio-quality / ASR evaluation;
- joyvl paths (out of migration scope, left in `experimental/`).
