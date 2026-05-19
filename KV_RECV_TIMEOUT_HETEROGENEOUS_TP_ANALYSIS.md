# KV Cache Receive Timeout on Heterogeneous Head/Headless TP — Root Cause Analysis

## Scenario

Single machine, two CLI runtimes serving a 2-stage AR + DiT pipeline:

- **Head**: `vllm serve <model> --omni --stage-id 0 --omni-dp-size-local 1` with **TP=1** (AR stage)
- **Headless**: `vllm serve <model> --omni --headless --stage-id 1 --omni-dp-size-local 1` with **TP=2** (DiT stage)

**Symptom:** receiver (stage 1) logs `Timeout waiting for KV cache for request <id> after 30s`.

**Counter-scenario (no error):** both CLI runtimes use the **same** TP (e.g., TP=2 each).

---

## Two key-construction forks

`vllm_omni/distributed/omni_connectors/utils/kv_utils.py` builds the connector keys with two sibling helpers. Each contains a **simple-vs-rank-aware fork driven by the inferred `KVTPTopology`**:

```python
# kv_utils.py:193  build_rank_aware_send_keys
if topo.source_tp_size <= 1 and topo.target_tp_size <= 1:
    return [f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}"]   # SIMPLE key
target_ranks = get_kv_target_ranks(topo)
return [get_kv_connector_key(request_id, from_stage, 0, topo.local_rank, r)   # RANK-AWARE
        for r in target_ranks]
```

```python
# kv_utils.py:211  build_rank_aware_recv_keys  (same fork)
if topo.source_tp_size <= 1 and topo.target_tp_size <= 1:
    return [(f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}", None)] # SIMPLE
source_ranks = get_kv_source_ranks(topo)
return [(get_kv_connector_key(request_id, from_stage, 0, r, topo.local_rank), r)
        for r in source_ranks]                                                # RANK-AWARE
```

`get_kv_connector_key` (`kv_utils.py:179`) emits `{req_id}_{from_stage}_0_{from_rank}_{to_rank}` — a completely different string from the simple form. Sender and receiver **must take the same fork** and **infer the same rank routing** or the strings will not match.

---

## Where the topology is decided

In `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py:307-317`, each transfer-manager builds its own `KVTPTopology` from an `omni_kv_config["rank_mapping"]` dict:

```python
local_rank = get_local_tp_rank()
if config.from_tp <= 1 and config.to_tp <= 1:
    detected_tp = get_tp_world_size()
    from_tp = detected_tp
    to_tp   = detected_tp
else:
    from_tp = config.from_tp
    to_tp   = config.to_tp
self._tp_topo = KVTPTopology(source_tp_size=from_tp, target_tp_size=to_tp,
                              local_rank=local_rank)
```

`config.from_tp` / `config.to_tp` come from `omni_kv_config["rank_mapping"]`, populated by `_inject_inferred_kv_tp_topology` in `vllm_omni/engine/stage_init_utils.py:243-304`:

```python
current_tp = _tp_size_for_stage(stage_configs, stage_id)        # OWN stage — always correct
...
if str(omni_from_stage) == str(stage_id):           # sender side (head, stage 0)
    from_tp = current_tp
    to_tp   = _tp_size_for_stage(stage_configs, peer_stage_id)  # PEER stage — read locally
elif str(omni_to_stage) == str(stage_id):           # receiver side (headless, stage 1)
    from_tp = _tp_size_for_stage(stage_configs, peer_stage_id)  # PEER stage — read locally
    to_tp   = current_tp
```

Each process infers the peer's TP **purely from its own copy of `stage_configs`**. There is **no cross-process exchange** of the actual TP value.

---

## Why the two processes' views diverge

`stage_configs` is built by `load_and_resolve_stage_configs(model, …, kwargs=args_dict, …)` in `vllm_omni/entrypoints/cli/serve.py` (head: top-level `omni_run_server`; headless: `run_headless` around line 746).

The `args_dict` for **that process** — including its CLI `--tensor-parallel-size` — is passed as `base_engine_args` and merged into **every** stage's engine_args via `merge_configs(base, stage_engine_args)` (`vllm_omni/entrypoints/utils.py:446`). OmegaConf merge lets the second arg win, so the stage YAML overrides the CLI base — **but only for fields the YAML actually pins**. Anything not pinned by the YAML keeps the CLI value.

`_tp_size_for_stage` then reads (`stage_init_utils.py:217-239`):

```python
parallel_config = engine_args.get("parallel_config")
if parallel_config is not None:
    tp = parallel_config.get("tensor_parallel_size", 1)   # nested (diffusion-style)
else:
    tp = engine_args.get("tensor_parallel_size", 1)       # top-level (LLM-style)
```

For the failing scenario:

| Process | Stage 0 (AR) TP as seen | Stage 1 (DiT) TP as seen |
|---|---|---|
| Head, `--tensor-parallel-size 1`, owns stage 0 | **1** (own, correct) | **1** — CLI value leaks unless YAML pins stage 1 TP where `_tp_size_for_stage` looks |
| Headless, `--tensor-parallel-size 2`, owns stage 1 | **2** — CLI value leaks unless YAML pins stage 0 TP | **2** (own, correct) |

Head ends up with **`KVTPTopology(source=1, target=1)`** → takes the **simple-key** branch.
Headless ends up with **`KVTPTopology(source=2, target=2)`** (or `(1, 2)` if stage 0's TP is pinned by YAML) → takes the **rank-aware** branch.

---

## The mismatch and the timeout

Head publishes a single key:

```
omni_0_to_1_kv_cache_<req_id>
```

Headless rank 0 polls:

```
<req_id>_0_0_0_0
```

Headless rank 1 polls:

```
<req_id>_0_0_1_1         (or <req_id>_0_0_0_1 in the asymmetric (1,2) case)
```

Neither matches what the head put. The recv loop in `kv_transfer_manager.py:1030-1127` keeps calling `connector.get(get_key=…)` with exponential backoff (0.01s → 0.5s), nothing ever resolves, and the wall-clock check at line 1122 fires:

```python
if time.time() - start_time > timeout:
    logger.error(f"Timeout waiting for KV cache for request {request_id} after {timeout}s")
    return None, 0
```

with `timeout = self.config.recv_timeout` — default **30 s** (`kv_transfer_manager.py:72`, `:361`). That is the **"KV cache receive timeout"**.

---

## Why both-TP=N hides the bug

If both processes pass `--tensor-parallel-size N` (or both leave it default and rely on identical YAML), every stage in every process is merged with the **same** unspecified-by-YAML default `N`. Both processes infer the **same** `KVTPTopology(N, N)`, both take the rank-aware branch, both compute the same `{from_rank, to_rank}` mapping (since `source == target` ⇒ both `get_kv_target_ranks` and `get_kv_source_ranks` return `[local_rank]`), and the keys line up.

The disagreement is **hidden, not fixed**.

---

## Root cause in one sentence

`_inject_inferred_kv_tp_topology` infers each peer stage's TP from the **local process's** merged `stage_configs`, where the local CLI `--tensor-parallel-size` has been sprayed into every stage's engine_args. With heterogeneous TP across head and headless, that local view of the peer stage is wrong, the two processes build incompatible `KVTPTopology`, and `build_rank_aware_send_keys` / `build_rank_aware_recv_keys` emit non-overlapping connector keys — so the receiver polls keys the sender never published, and the wait loop hits `recv_timeout`.

---

## Safe deployment recipe (avoids the bug without code changes)

The mismatch only happens when the two processes end up with **different per-stage TP views**. Two conditions are sufficient to remove every source of that divergence:

1. **Do not pass `--tensor-parallel-size`** (or any other parallelism CLI override) on either runtime.
2. **Every CLI runtime reads the exact same YAML static file** (same path, same contents), with each stage's `tensor_parallel_size` pinned in that file at the location `_tp_size_for_stage` reads:
   - LLM / AR stages → top-level `tensor_parallel_size: N` (or `parallel_config.tensor_parallel_size`)
   - Diffusion / DiT stages → `parallel_config: { tensor_parallel_size: N }`

Under those conditions, each process's merged `stage_configs` carries identical per-stage TP values. `_inject_inferred_kv_tp_topology` returns the same `(from_tp, to_tp)` on both sides, both build the same `KVTPTopology`, both take the same fork in `build_rank_aware_send_keys` / `build_rank_aware_recv_keys`, and the connector keys line up. Heterogeneous per-stage TP (e.g. stage 0 TP=1, stage 1 TP=2) works — the constraint is **same YAML across processes**, not "same TP across stages".

## Fix surfaces, in order of robustness

1. **Exchange the authoritative TP via `OmniMasterServer`.** Each process registers its own `tensor_parallel_size` for the stage it owns; `_inject_inferred_kv_tp_topology` reads the peer's TP from the master rather than from the local YAML-merge result. This is the only fix that holds when the YAML is silent on per-stage TP.
2. **Stop spraying CLI parallelism flags onto peer stages.** In `load_stage_configs_from_model` / `load_stage_configs_from_yaml`, when `--stage-id` is set, apply `tensor_parallel_size` (and friends) from `base_engine_args` **only to the locally-owned stage**; for peer stages, read TP **only** from YAML.
3. **Make the YAML the single source of truth** by documenting that every stage must pin `tensor_parallel_size` (top-level for LLM stages, `parallel_config:` for diffusion stages) when running head/headless across heterogeneous TP — and warn loudly in `_inject_inferred_kv_tp_topology` when CLI base TP and YAML stage TP disagree on a peer stage. Workaround only, not a structural fix.

The simple/rank-aware fork at `source_tp_size <= 1 and target_tp_size <= 1` (`kv_utils.py:205, 241`) is the spot where the mismatch becomes terminal (different key formats), but removing that fork only narrows the failure to "wrong rank routing" — the underlying information-asymmetry fix (1 or 2 above) is what actually closes the bug.

---

## File / line reference summary

| Concern | File | Lines |
|---|---|---|
| Sender simple/rank-aware fork | `vllm_omni/distributed/omni_connectors/utils/kv_utils.py` | 193–208 |
| Receiver simple/rank-aware fork | `vllm_omni/distributed/omni_connectors/utils/kv_utils.py` | 211–244 |
| Connector key format | `vllm_omni/distributed/omni_connectors/utils/kv_utils.py` | 179–190 |
| `KVTPTopology` construction | `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py` | 307–317 |
| Peer-stage TP inference | `vllm_omni/engine/stage_init_utils.py` | 243–304 |
| Per-stage TP lookup (top-level vs `parallel_config:`) | `vllm_omni/engine/stage_init_utils.py` | 217–239 |
| CLI → every-stage engine_args merge | `vllm_omni/entrypoints/utils.py` | 441–453 |
| Headless `load_and_resolve_stage_configs` call | `vllm_omni/entrypoints/cli/serve.py` | ~746 |
| `recv_timeout` default (30 s) | `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py` | 72, 361 |
| Receive wait-loop & timeout log | `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py` | 1030–1127 (timeout at 1122–1124) |
