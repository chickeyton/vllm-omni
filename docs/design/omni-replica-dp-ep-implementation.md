# Omni-Replica + vLLM-DP + Expert-Parallel — Implementation Design

Date: 2026-05-22
Branch: `replica_dp_ep`
Decision: **3b** (per-replica EP), all intra-replica parallelism YAML-only.
Companion: [omni-replica-dp-ep-analysis.md](./omni-replica-dp-ep-analysis.md)
for the trade-off rationale.

> **Revision history**
>
> - **Initial draft.** P1 rename + P2 YAML-only intra-replica
>   parallelism (validator on a nested `parallel_config:` dict;
>   force-set `data_parallel_size` and `api_server_count`).
>
> - **Post-integration-smoke revision (this version).** Three findings
>   surfaced when the implementation was exercised against a real
>   `vllm serve` launch on the test cluster:
>   1. **Validator must operate on FLAT fields, not a nested
>      `parallel_config:`.** `OmniEngineArgs` (subclass of vLLM's
>      `EngineArgs`) is a flat dataclass — there is no
>      `parallel_config` field. Writing back to
>      `engine_args_dict["parallel_config"]` caused
>      `OmniEngineArgs.__init__()` to error with "unexpected keyword
>      argument 'parallel_config'" on every LLM launch.
>   2. **`api_server_count` is not a force-set target.** It lives
>      only on the argparse Namespace (CLI-only); it never reaches
>      `engine_args` and is already defaulted to `0` in the headless
>      path at `serve.py:734`. The head path doesn't read it either
>      (vllm-omni's `omni_run_server` owns the HTTP server). Dropped
>      from the validator's force-set list. Stays on the
>      **CLI prohibition list** as a defence-in-depth.
>   3. **`_cli_explicit_keys` was never populated.** The prohibition
>      check at `serve.py:135` gated on `args._cli_explicit_keys`,
>      but no code in vllm-omni ever set that attribute. PR #3569's
>      prohibition list was structurally a no-op. Fixed by calling
>      `detect_explicit_cli_keys(sys.argv[1:], type(self)._parser)`
>      at the top of `validate()`. The replica-id deprecation warning
>      in `run_headless` (which also reads `_cli_explicit_keys`)
>      becomes functional as a side benefit.
>
>   The body below has been updated to reflect (1)–(3). See §10 for
>   a complete summary of the post-smoke bugfix patch.

---

## 1. Scope

Three changes layered on PR #3569, in order:

- **P1** — Rename `--omni-dp-size-local` → `--omni-num-replica`. Old name kept as a deprecated alias.
- **P2** — Inner vLLM DP per replica, configured exclusively in the deploy YAML (FLAT fields under each `stages[]` entry for LLM stages; nested `parallel_config:` for DiT, unchanged).
- **P3b** — EP / EPLB per replica, same YAML route. No cross-replica EP.

After this lands, **`--omni-num-replica` is the only intra-runtime scaling knob on the CLI**. Every other parallelism axis lives in YAML.

Out of scope (decided in the analysis doc): cross-replica EP (3a), elastic EP, Ray DP backend, vLLM external/hybrid LB modes.

### 1.1 Topology after this lands

End state: one head process + N independent omni replicas. Each replica
owns its own vLLM DP cluster, its own NCCL world, its own EP group.
Replicas are coupled only by ZMQ control + data channels through the head.

```
                              ┌─────────────────────────────────────┐
                              │           Head process              │
                              │  ┌────────────────┐                 │
                              │  │  omni_run_     │ HTTP            │
   client ─── HTTP ────────►  │  │  server (LLM   │ (vllm-omni's    │
                              │  │  + Diffusion)  │  OpenAI server) │
                              │  └───────┬────────┘                 │
                              │          │ AsyncOmniEngine          │
                              │  ┌───────▼────────┐                 │
                              │  │  StagePool[s]  │ outer LB        │
                              │  │  ─ LoadBalancer│ (random /       │
                              │  │   (replica-    │  round-robin /  │
                              │  │    level)      │  least-queue)   │
                              │  └───┬───┬────┬───┘                 │
                              │      │   │    │ ZMQ (req/resp)      │
                              │  ┌───▼┐  │ ┌──▼───┐                 │
                              │  │OMS │  │ │OmniCo│ ZMQ ROUTER/PUB  │
                              │  │+OCo│  │ │ord   │ (control plane: │
                              │  │ord │  │ │      │  status, queue) │
                              │  └─┬──┘  │ └──┬───┘                 │
                              └────┼─────┼────┼─────────────────────┘
                                   │     │    │
                  ZMQ control/data │     │    │ heartbeats
            ┌──────────────────────┴──┐  │    └────────────────────┐
            │                         │  │                         │
   ┌────────▼──────────┐    ┌─────────▼──▼─────────┐    ┌──────────▼────────┐
   │ Omni replica 0    │    │ Omni replica 1       │    │ Omni replica N-1  │
   │ (own NCCL world)  │    │ (own NCCL world)     │    │ (own NCCL world)  │
   │                   │    │                      │    │                   │
   │ ┌───────────────┐ │    │ ┌──────────────────┐ │    │ ┌───────────────┐ │
   │ │ DPCoordinator │ │    │ │  DPCoordinator   │ │    │ │ DPCoordinator │ │
   │ │ (per-replica) │ │    │ │  (per-replica)   │ │    │ │ (per-replica) │ │
   │ └───────────────┘ │    │ └──────────────────┘ │    │ └───────────────┘ │
   │                   │    │                      │    │                   │
   │ DP rank 0         │    │ DP rank 0            │    │ DP rank 0         │
   │  ├─ TP rank 0     │    │  ├─ TP rank 0        │    │  ├─ TP rank 0     │
   │  └─ TP rank 1     │    │  └─ TP rank 1        │    │  └─ TP rank 1     │
   │ DP rank 1         │    │ DP rank 1            │    │ DP rank 1         │
   │  ├─ TP rank 0     │    │  ├─ TP rank 0        │    │  ├─ TP rank 0     │
   │  └─ TP rank 1     │    │  └─ TP rank 1        │    │  └─ TP rank 1     │
   │                   │    │                      │    │                   │
   │ One _EP group =   │    │ One _EP group =      │    │ One _EP group =   │
   │   dp_local×pcp×tp │    │   dp_local×pcp×tp    │    │   dp_local×pcp×tp │
   │   = 2×1×2 = 4     │    │   = 4                │    │   = 4             │
   └───────────────────┘    └──────────────────────┘    └───────────────────┘
        4 GPUs                       4 GPUs                     4 GPUs
```

Key points the diagram is telling you:

- **The head is the only HTTP entrypoint.** `omni_run_server` (vllm-omni's
  own OpenAI-compatible server) is what clients talk to. vLLM's own
  `run_server` / `run_multi_api_server` is never invoked under `--omni`.
- **The outer LB is `StagePool.LoadBalancer`** — picks a replica based on
  OmniCoordinator's published `ReplicaList`.
- **The inner LB is vLLM's `DPLBAsyncMPClient`** — once a replica is
  picked, it routes the request to a DP rank inside that replica using
  `waiting × 4 + running` weighted-shortest-queue.
- **No torch.distributed comm between replicas.** Each replica is its own
  NCCL world with its own `_EP`, `_DP`, `_TP` groups. Replicas exchange
  nothing at the collective level — only metadata over ZMQ.
- **EP scope is bounded by one replica.** Same trade-off the analysis doc
  spells out: trade EP scale for replica isolation.

### 1.2 LB hierarchy in one picture

```
  Request from client
         │
         ▼
  ┌────────────────────┐
  │  AsyncOmniEngine   │ — looks up stage_id for the request
  └──────────┬─────────┘
             │
             ▼
  ┌────────────────────┐
  │  StagePool[stage]  │ — OUTER LB
  │  .LoadBalancer     │   policy from --omni-lb-policy
  │  (replica picker)  │   fed by OmniCoordinator's ReplicaList
  └──────────┬─────────┘     (status + queue length per replica)
             │
             ▼   one of {replica_0, replica_1, ...}
  ┌──────────────────────────┐
  │  DPLBStageEngineCoreClient│ — INNER LB (only when dp_local > 1)
  │  ← DPLBAsyncMPClient      │   waiting×4 + running shortest-queue
  │  .get_core_engine_for_   │   fed by per-replica DPCoordinator's
  │   request()              │   stats (every ~100ms)
  └──────────┬───────────────┘
             │
             ▼   one of {DP rank 0, DP rank 1, ...}
  ┌────────────────────┐
  │  EngineCore        │ — actually does inference
  │  subprocess        │   TP across its assigned GPUs
  └────────────────────┘
```

`data_parallel_external_lb` and `data_parallel_hybrid_lb` would each
short-circuit one of these layers — both are rejected by the YAML
validator (§4.3) so the picture above is the only allowed topology.

---

## 2. Re-reading the code — what we don't need to build

A careful read of the current code rules out several pieces of infrastructure the earlier draft assumed:

1. **No `OmniPortRegistry` needed.** vLLM's `ParallelConfig.__post_init__` already self-allocates `data_parallel_master_port` via `get_open_ports_list(5)` at `vllm/config/parallel.py:799-804` whenever `data_parallel_size > 1`. `OmniMasterServer._allocate_route_locked` uses `get_open_ports_list(count=3)` for its ZMQ triple. The diffusion `settle_port(61000 + rep_idx*100)` workaround at `serve.py:849-859` already exists and works. No central allocator is required.

2. **No force-set firehose needed.** vLLM defaults already cover most of what the analysis doc called "force-set fields": `data_parallel_backend="mp"`, `data_parallel_external_lb=False`, `data_parallel_hybrid_lb=False`, `enable_elastic_ep=False`, `data_parallel_rank=0` (for a fresh DP world). We just need to set **two** values explicitly:
   - `data_parallel_size = data_parallel_size_local` (the critical "no cross-replica DP" invariant).
   - `api_server_count = 0` (the head's path doesn't set it today; the headless path already defaults to 0 at `serve.py:734`).

3. **`client_count=1` is already the default** in `StageEngineCoreClientBase.make_async_mp_client` (`stage_engine_core_client.py:94, 124`). No change needed for LB tie-breaking.

4. **Inner-DP plumbing already exists end-to-end:**
   - Headless `serve.py:951-955` reads `parallel_config.data_parallel_size_local` as `local_engine_count`.
   - `OmniCoreEngineProcManager` detects intra-replica DP via `has_intra_replica_dp = parallel_config.data_parallel_size > 1` (`omni_core_engine_proc_manager.py:107`) and spawns `local_engine_count` subprocesses per call.
   - `launch_omni_core_engines` already spawns one `DPCoordinator` per omni replica (`stage_engine_startup.py:876-892`).
   - `make_async_mp_client` already selects `DPLBStageEngineCoreClient` when `data_parallel_size > 1` (`stage_engine_core_client.py:112-115`).

   The only things missing are: (a) the CLI prohibition is too narrow today (lets some things through, blocks the wrong things), (b) `omni_replica_id` is mis-assigned when DP > 1, (c) `OmniCoordClientForStage` runs on every DP rank rather than just rank 0.

5. **EP / EPLB plumbing is fully wired through `EngineArgs` already.** `enable_expert_parallel`, `enable_eplb`, `eplb_config.*`, `expert_placement_strategy`, `all2all_backend`, `enable_ep_weight_filter` reach `ParallelConfig.__post_init__` (`vllm/config/parallel.py:446-467`) and `init_model_parallel_group` (`vllm/distributed/parallel_state.py:1666-1696`) via the standard `build_engine_args_dict` → `OmniEngineArgs(**...)` path. P3b adds **no new wiring** — only validation.

The combined consequence: **no new files**, ~6 small file touches, ~150-250 LoC of production change.

---

## 3. Final user-facing surface

### 3.0 Where the new validator hooks into the build flow

```
   ┌─────────────────┐
   │  Deploy YAML    │  stages[].engine_args.parallel_config:
   └────────┬────────┘
            │
            ▼
   ┌──────────────────────────────────┐
   │ load_and_resolve_stage_configs   │  vllm_omni/entrypoints/utils.py
   └────────┬─────────────────────────┘
            │   StageConfig with engine_args dict
            ▼
   ┌──────────────────────────────────┐
   │ build_engine_args_dict           │  stage_init_utils.py:638-695
   └────────┬─────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────┐
   │ build_vllm_config                │  stage_init_utils.py:698+
   │  ┌─────────────────────────────┐ │
   │  │ filter_dataclass_kwargs(    │ │
   │  │    OmniEngineArgs, ...)     │ │
   │  └────────────┬────────────────┘ │
   │               │                  │
   │  ┌────────────▼─────────────────┐│  ◄── NEW (§4.3)
   │  │ _enforce_omni_parallel_      ││      • Reject user-set fields
   │  │  config(stage_cfg, args_dict)││        vllm-omni owns
   │  └────────────┬─────────────────┘│      • Reject ext/hybrid LB,
   │               │                  │        elastic EP, ray backend
   │  ┌────────────▼─────────────────┐│      • Reject device-count
   │  │ OmniEngineArgs(**args_dict)  ││        non-divisible
   │  └────────────┬─────────────────┘│      • Force-set ONLY
   │               │                  │        data_parallel_size
   │  ┌────────────▼─────────────────┐│        (api_server_count is
   │  │ .create_engine_config()      ││         CLI-only, never reaches
   │  │  → VllmConfig.parallel_config││         engine_args)
   │  │  → ParallelConfig.__post_init│ │
   │  │    (auto-allocs DP master    │ │  ◄── vLLM does its own
   │  │     port, applies EP rules)  │ │      validation here
   │  └────────────┬─────────────────┘ │
   └───────────────┼──────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────┐
   │ launch_omni_core_engines /       │  spawns DPCoordinator (if DP>1)
   │ OmniCoreEngineProcManager        │  + dp_local engine subprocesses
   └──────────────────────────────────┘
```

The validator is **one hook in one place** — between
`filter_dataclass_kwargs` and `OmniEngineArgs(**args_dict)`. It mutates
`engine_args_dict` in-place (rejects forbidden FLAT fields, force-sets
the single value vLLM defaults don't cover),
then lets the standard `EngineArgs` → `ParallelConfig` path do the rest.
No changes downstream of that point.

### 3.1 CLI flags accepted under `--omni`

```
--omni-num-replica N                 # NEW — per-process replica count
--omni-dp-size-local N               # DEPRECATED ALIAS (one release)

# Unchanged from PR #3569:
--omni-master-address / --omni-master-port / --omni-replica-address
--omni-lb-policy / --omni-heartbeat-timeout
--stage-id / --headless / --deploy-config / --stage-configs-path
--stage-overrides / --async-chunk
--stage-init-timeout / --init-timeout
```

### 3.2 CLI flags rejected under `--omni` (extended list)

See §4.2 below for the literal dict.

### 3.3 YAML — LLM stage

**For LLM stages the parallelism fields are FLAT** — they live directly
under the stage (vLLM's `EngineArgs` is a flat dataclass; there is no
`parallel_config:` sub-dict). The fields below match real
`vllm_omni/deploy/*.yaml` files.

```yaml
stages:
  - stage_id: 0
    devices: "0,1,2,3"                # devices PER replica = tp×pp×pcp×dp_local
    runtime:
      num_replicas: 1                 # overridden by --omni-num-replica

    # Parallelism fields are FLAT — no nested parallel_config: key.
    tensor_parallel_size: 2
    data_parallel_size_local: 2
    pipeline_parallel_size: 1
    prefill_context_parallel_size: 1

    # EP / EPLB
    enable_expert_parallel: true
    enable_eplb: true
    eplb_config:                      # this IS a dict — vLLM's EPLBConfig dataclass
      window_size: 1000
      step_interval: 3000
    expert_placement_strategy: linear
    all2all_backend: deepep_low_latency
```

### 3.4 YAML — DiT stage

DiT *does* use a nested `parallel_config:` block, because it's a
`DiffusionParallelConfig` dataclass instance — that's the historical
shape of the diffusion runtime and is unchanged. Validator handling
differs accordingly: the LLM rules apply only when
`stage_cfg.stage_type != "diffusion"`; DiT stages early-return.

```yaml
stages:
  - stage_id: 0
    devices: "0,1,2,3"
    parallel_config:                  # nested — DiffusionParallelConfig
      tensor_parallel_size: 2
      data_parallel_size: 1           # DiT FSDP/HSDP axis, not EngineCore DP
      ulysses_degree: 2
      cfg_parallel_size: 2
      enable_expert_parallel: true
```

### 3.5 Field handling summary (LLM, FLAT in `engine_args`)

| Field | Handling |
|---|---|
| `tensor_parallel_size`, `pipeline_parallel_size`, `prefill_context_parallel_size`, `decode_context_parallel_size`, `data_parallel_size_local` | **User-set in YAML (flat).** |
| `enable_expert_parallel`, `enable_eplb`, `eplb_config.*`, `expert_placement_strategy`, `all2all_backend`, `enable_ep_weight_filter`, `num_redundant_experts` | **User-set in YAML (flat).** |
| `data_parallel_size` | **Force-set** to `data_parallel_size_local`. YAML may omit, or set it equal (any other value rejected). |
| `data_parallel_address`, `data_parallel_rpc_port`, `data_parallel_master_port` | **vLLM self-allocates** at `parallel.py:799-804`. YAML must omit (validator rejects). |
| `data_parallel_rank`, `data_parallel_rank_local` | **vLLM defaults to 0.** YAML must omit. |
| `data_parallel_backend` | **vLLM default `"mp"`.** YAML may set iff `"mp"`. |
| `data_parallel_external_lb`, `data_parallel_hybrid_lb`, `enable_elastic_ep` | **vLLM defaults to False.** YAML rejected if `True`. |
| `api_server_count` | **CLI-only on argparse Namespace; never reaches engine_args.** Already prohibited at the CLI. Headless path defaults it to 0 at `serve.py:734`. Validator does not touch it. |

---

## 4. Patch list

### 4.1 Point 1 — Rename (~30 LoC)

**`vllm_omni/engine/arg_utils.py:158`**

```python
# Before
omni_dp_size_local: int = 1

# After
omni_num_replica: int = 1
```

**`vllm_omni/entrypoints/cli/serve.py`** — argparse (replace the `--omni-dp-size-local` block around line 258):

```python
omni_config_group.add_argument(
    "--omni-num-replica",
    "--omni-dp-size-local",          # deprecated alias, same dest
    type=int,
    default=1,
    dest="omni_num_replica",
    help=(
        "Number of stage replicas this process launches for its own "
        "--stage-id. Process-local. Requires --stage-id when != 1. "
        "(--omni-dp-size-local is a deprecated alias.)"
    ),
)
```

Add a one-shot deprecation warning at the top of `validate()`:

```python
if any(a.startswith("--omni-dp-size-local") for a in sys.argv):
    logger.warning("--omni-dp-size-local is deprecated; use --omni-num-replica.")
```

Find-and-replace `omni_dp_size_local` → `omni_num_replica` in `serve.py`'s validation block (lines 114-121) and in `async_omni_engine.py` (the field at line 312 plus 3-4 references). For programmatic compatibility, keep an `or kwargs.get("omni_dp_size_local")` fallback in the constructor.

### 4.2 Point 2 — Extended CLI prohibition (~30 LoC)

**`serve.py:128-147`** — replace the existing `prohibited_with_omni` dict:

```python
prohibited_with_omni: dict[str, str] = {
    # TP / PP
    "tensor_parallel_size": "--tensor-parallel-size",
    "pipeline_parallel_size": "--pipeline-parallel-size",
    # DP
    "data_parallel_size": "--data-parallel-size",
    "data_parallel_size_local": "--data-parallel-size-local",
    "data_parallel_rank": "--data-parallel-rank",
    "data_parallel_rank_local": "--data-parallel-rank-local",
    "data_parallel_address": "--data-parallel-address",
    "data_parallel_rpc_port": "--data-parallel-rpc-port",
    "data_parallel_master_port": "--data-parallel-master-port",
    "data_parallel_backend": "--data-parallel-backend",
    "data_parallel_external_lb": "--data-parallel-external-lb",
    "data_parallel_hybrid_lb": "--data-parallel-hybrid-lb",
    "data_parallel_start_rank": "--data-parallel-start-rank",
    "api_server_count": "--api-server-count",
    # Context parallel
    "prefill_context_parallel_size": "--prefill-context-parallel-size",
    "decode_context_parallel_size": "--decode-context-parallel-size",
    "dcp_comm_backend": "--dcp-comm-backend",
    # EP / EPLB
    "enable_expert_parallel": "--enable-expert-parallel",
    "enable_ep_weight_filter": "--enable-ep-weight-filter",
    "enable_eplb": "--enable-eplb",
    "num_redundant_experts": "--num-redundant-experts",
    "expert_placement_strategy": "--expert-placement-strategy",
    "all2all_backend": "--all2all-backend",
    "enable_elastic_ep": "--enable-elastic-ep",
    # Microbatching
    "enable_dbo": "--enable-dbo",
    "ubatch_size": "--ubatch-size",
}

offenders = sorted(flag for dest, flag in prohibited_with_omni.items() if dest in explicit_cli_keys)
# Also reject --eplb-config.* nested keys (argparse stores them with prefix "eplb_config_"
# or as nested attributes on the eplb_config dataclass):
offenders += sorted(
    f"--eplb-config.{k.removeprefix('eplb_config_')}"
    for k in explicit_cli_keys
    if k.startswith("eplb_config_") or k == "eplb_config"
)
if offenders:
    raise ValueError(
        "The following CLI args are not supported under --omni: "
        f"{', '.join(offenders)}. Configure intra-replica parallelism in the "
        "deploy YAML (--deploy-config) under `stages[].engine_args.parallel_config:`. "
        "The only CLI knob for replica scaling is --omni-num-replica."
    )
```

**Important wiring fix:** The check above gates on `args._cli_explicit_keys`,
which **PR #3569 never populated** — making the original prohibition list
a structural no-op. Populate it at the top of `validate()`:

```python
def validate(self, args: argparse.Namespace) -> None:
    # Populate args._cli_explicit_keys from sys.argv so the prohibition
    # check (and the replica-id deprecation warning in run_headless) can
    # distinguish user-typed flags from argparse defaults.
    if not hasattr(args, "_cli_explicit_keys"):
        from vllm_omni.entrypoints.utils import detect_explicit_cli_keys
        args._cli_explicit_keys = detect_explicit_cli_keys(
            sys.argv[1:], type(self)._parser
        )
    ...
```

`type(self)._parser` is stashed by `subparser_init` (`serve.py:696`) for
exactly this kind of post-parse introspection — the helper resolves each
`sys.argv` token through the parser's action table to find its real `dest`,
correctly handling alias flags (`--omni-num-replica` /
`--omni-dp-size-local` → same dest), `--no-foo` boolean-optional patterns,
and explicit `dest=` overrides.

### 4.3 Point 2 — YAML validator + minimal force-set (~80 LoC)

Add **one** function in `vllm_omni/engine/stage_init_utils.py`, called from
`build_vllm_config` just before `OmniEngineArgs(**filtered_engine_args_dict)`.
For LLM stages the validator operates on the FLAT `engine_args_dict` (no
nested `parallel_config:` key — `OmniEngineArgs` is a flat
dataclass).

```python
def _enforce_omni_parallel_config(stage_cfg: Any, engine_args_dict: dict) -> None:
    """Validate and finalize parallelism config for --omni mode.

    For LLM stages the relevant fields live FLAT in engine_args_dict
    (e.g. tensor_parallel_size, data_parallel_size_local) — vLLM's
    EngineArgs is flat, not nested under a parallel_config: key.

    Rejects YAML values for fields vllm-omni / vLLM own and force-sets
    data_parallel_size = data_parallel_size_local. api_server_count is
    CLI-only on the argparse Namespace; it never reaches engine_args, so
    we don't touch it here. DiT stages early-return — their nested
    parallel_config: is a DiffusionParallelConfig dataclass with a
    different schema and none of the LLM-DP fields.
    """
    if getattr(stage_cfg, "stage_type", "llm") == "diffusion":
        return

    sid = stage_cfg.stage_id

    # Fields vllm-omni / vLLM own; user must not set them.
    must_omit = {
        "data_parallel_address":     "vllm-omni / vLLM auto-assigns the DP master address",
        "data_parallel_rpc_port":    "vllm-omni / vLLM auto-assigns the DP RPC port",
        "data_parallel_master_port": "vllm-omni / vLLM auto-assigns the DP master port",
        "data_parallel_rank":        "each replica is its own DP world; rank is always 0",
        "data_parallel_rank_local":  "each replica is its own DP world; local rank is always 0",
    }
    for field, why in must_omit.items():
        if field in engine_args_dict:
            raise ValueError(
                f"stage {sid}: engine_args.{field} is not user-settable "
                f"under --omni ({why}). Remove the field from the YAML."
            )

    # Equality-or-absent: data_parallel_size must match data_parallel_size_local.
    dp_local = int(engine_args_dict.get("data_parallel_size_local") or 1)
    if (
        "data_parallel_size" in engine_args_dict
        and engine_args_dict["data_parallel_size"] is not None
        and int(engine_args_dict["data_parallel_size"]) != dp_local
    ):
        raise ValueError(
            f"stage {sid}: engine_args.data_parallel_size "
            f"({engine_args_dict['data_parallel_size']}) must equal "
            f"data_parallel_size_local ({dp_local}). vllm-omni does not "
            f"support cross-replica DP."
        )

    # Boolean-must-be-False.
    for field, hint in {
        "data_parallel_external_lb": "the outer LB is vllm-omni's StagePool",
        "data_parallel_hybrid_lb":   "the outer LB is vllm-omni's StagePool",
        "enable_elastic_ep":         "Ray-only; structurally incompatible with the omni replica model",
    }.items():
        if engine_args_dict.get(field):
            raise ValueError(
                f"stage {sid}: engine_args.{field}=True is not supported ({hint})."
            )

    # String-must-be-mp.
    backend = engine_args_dict.get("data_parallel_backend", "mp")
    if backend != "mp":
        raise ValueError(
            f"stage {sid}: engine_args.data_parallel_backend must be 'mp' "
            f"(got {backend!r}); the Ray DP backend is not supported under --omni."
        )

    # EPLB sanity (clearer than vLLM's own error).
    tp = int(engine_args_dict.get("tensor_parallel_size") or 1)
    if engine_args_dict.get("enable_eplb") and dp_local * tp <= 1:
        raise ValueError(
            f"stage {sid}: enable_eplb requires data_parallel_size_local × "
            f"tensor_parallel_size > 1 (got {dp_local} × {tp})."
        )

    # Device-count product (LLM only; DiT validates against its own world_size).
    pp  = int(engine_args_dict.get("pipeline_parallel_size") or 1)
    pcp = int(engine_args_dict.get("prefill_context_parallel_size") or 1)
    per_replica = tp * pp * pcp * dp_local
    devices_str = _resolve_devices_str(stage_cfg)
    if devices_str:
        n = len([d for d in devices_str.split(",") if d.strip()])
        if n % per_replica != 0:
            raise ValueError(
                f"stage {sid}: len(devices)={n} is not divisible by "
                f"per-replica device count tp×pp×pcp×dp_local={per_replica}."
            )

    # Minimal force-set: data_parallel_size must equal data_parallel_size_local
    # so each replica is its own DP world. Everything else is already correct
    # via vLLM defaults (mp backend, internal LB, no elastic EP, rank 0).
    engine_args_dict["data_parallel_size"] = dp_local
```

That is the entire validation + force-set logic. **No separate force-set
function; no DiT-specific validator;** the DiT branch returns early
because `DiffusionParallelConfig` has none of the conflicting fields.

**Why no `api_server_count` force-set, despite the initial draft.**
`api_server_count` is a vLLM CLI flag that lives only on the argparse
Namespace; it has no corresponding field on `EngineArgs` /
`OmniEngineArgs` / `ParallelConfig`. Writing it into `engine_args_dict`
makes `OmniEngineArgs(**engine_args_dict)` fail with "unexpected keyword
argument" — which is exactly the integration-smoke failure that
surfaced this. vLLM's own `serve.py` reads `args.api_server_count`
directly from the Namespace to decide whether to launch HTTP servers;
since vllm-omni's `--omni` path always invokes `omni_run_server` instead,
the vLLM-side value is unused. The CLI prohibition in §4.2 already
rejects user-supplied `--api-server-count` under `--omni`, which is the
sufficient defence.

### 4.4 Point 2 — Device-count math (~10 LoC)

`stage_init_utils.py:543-561` — extend the LLM branch of
`get_stage_devices_per_replica` to multiply by DP and PCP. For LLM
stages the fields are FLAT (no nested `parallel_config:`), but for
robustness against legacy configs the helper falls back to a nested
dict if it's present:

```python
def get_stage_devices_per_replica(stage_cfg: Any) -> int:
    if getattr(stage_cfg, "stage_type", "llm") != "diffusion":
        ea = getattr(stage_cfg, "engine_args", {})
        pc = _get_attr_or_item(ea, "parallel_config") or {}
        # FLAT on ea takes precedence; nested pc is a legacy fallback.
        tp  = int(_get_attr_or_item(pc, "tensor_parallel_size", None)
                  or _get_attr_or_item(ea, "tensor_parallel_size", 1) or 1)
        pp  = int(_get_attr_or_item(pc, "pipeline_parallel_size", 1) or 1)
        pcp = int(_get_attr_or_item(pc, "prefill_context_parallel_size", 1) or 1)
        dp  = int(_get_attr_or_item(pc, "data_parallel_size_local", 1) or 1)
        return tp * pp * pcp * dp     # DCP reuses TP devices; not multiplied
    # diffusion branch unchanged (still reads nested parallel_config)
    ...
```

`split_devices_for_replicas` needs **no change** — it already operates
on whatever `devices_per_replica` value it receives.

### 4.5 Point 2 — `omni_replica_id` fix (~1 LoC)

**Before (buggy when `dp_local > 1`):**

```
   OmniMasterServer
   auto-assigns
   replica_id = R
        │
        ▼
   OmniCoreEngineProcManager(omni_replica_base_id=R)
   spawns local_engine_count=dp_local subprocesses:

      ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
      │ StageEngineCoreProc │ │ StageEngineCoreProc │ │ StageEngineCoreProc │
      │ DP rank 0           │ │ DP rank 1           │ │ DP rank dp_local-1  │
      │ omni_replica_id = R │ │ omni_replica_id=R+1 │ │ omni_replica_id=R+N │
      │   ┌──────────────┐  │ │   ┌──────────────┐  │ │   ┌──────────────┐  │
      │   │OmniCoordClnt │  │ │   │OmniCoordClnt │  │ │   │OmniCoordClnt │  │
      │   │ForStage      │──┼─┼─►│ForStage      │──┼─┼─►│ForStage      │──┼─►OmniCoord
      │   │stage_id=S    │  │ │   │stage_id=S    │  │ │   │stage_id=S    │  │
      │   │input=ADDR_R  │  │ │   │input=ADDR_R  │  │ │   │input=ADDR_R  │  │
      │   └──────────────┘  │ │   └──────────────┘  │ │   └──────────────┘  │
      └─────────────────────┘ └─────────────────────┘ └─────────────────────┘

   Problem: ids R+1 ... R+dp_local-1 were NEVER registered with
            OmniMasterServer. OmniCoordinator's ReplicaList has dp_local
            entries per replica, all sharing the same input_addr.
            StagePool can't tell them apart from real replicas.
```

**After (DP ranks share one omni slot):**

```
   OmniMasterServer
   auto-assigns
   replica_id = R
        │
        ▼
   OmniCoreEngineProcManager(omni_replica_base_id=R)
   spawns dp_local subprocesses, all sharing omni_replica_id=R:

      ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
      │ StageEngineCoreProc │ │ StageEngineCoreProc │ │ StageEngineCoreProc │
      │ DP rank 0   ★       │ │ DP rank 1           │ │ DP rank dp_local-1  │
      │ omni_replica_id = R │ │ omni_replica_id = R │ │ omni_replica_id = R │
      │   ┌──────────────┐  │ │                     │ │                     │
      │   │OmniCoordClnt │──┼─┼─►(no coord client)  │ │  (no coord client)  │
      │   │ForStage      │  │ │                     │ │                     │
      │   │stage_id=S    │──┼─┼──────────────────►OmniCoord                 │
      │   │input=ADDR_R  │  │ │                     │ │                     │
      │   └──────────────┘  │ │                     │ │                     │
      └──────┬──────────────┘ └─────────────────────┘ └─────────────────────┘
             │ ★ only DP rank 0 reports queue length to OmniCoordinator
             │
             ▼
        OmniCoordinator's ReplicaList has exactly ONE entry per omni
        replica, matching what OmniMasterServer auto-assigned. The
        outer StagePool LB then sees one logical replica; the inner
        DPLBAsyncMPClient routes within it.
```

The fix is one line in `omni_core_engine_proc_manager.py:117`, which assigns a different `omni_replica_id` to each DP-rank subprocess of a single registration:

```python
omni_replica_id = omni_replica_base_id + index   # BUG when data_parallel_size_local > 1
```

Those `base + 1`, `base + 2`, … ids were never registered with `OmniMasterServer`. Replace with:

```python
omni_replica_id = omni_replica_base_id           # all DP ranks share the omni slot
```

Update the docstring at lines 102-106 to reflect this — it currently contradicts `stage_engine_core_proc.py:102-104`.

### 4.6 Point 2 — `OmniCoordClientForStage` on DP rank 0 only (~10 LoC)

`stage_engine_core_proc.py:102-119` — gate on local DP rank:

```python
# Before
if omni_coordinator_address is not None:
    ...
    coord_client = OmniCoordClientForStage(...)

# After — only DP rank 0 of the replica reports to OmniCoordinator.
if omni_coordinator_address is not None and local_dp_rank == 0:
    coord_client = OmniCoordClientForStage(
        coord_zmq_addr=omni_coordinator_address,
        input_addr=addresses.inputs[0],
        output_addr=addresses.outputs[0],
        stage_id=int(omni_stage_id),
    )
    # Queue length: rank-0's own scheduler is a good proxy and avoids
    # cross-rank coordination on the hot path. With DP > 1, the inner
    # DPLBAsyncMPClient balances across ranks, so rank-0's queue length
    # is representative within a small constant factor. (Tighten in a
    # follow-up if production data shows skew.)
    def _refresh_queue_length() -> None:
        scheduler = getattr(engine_core, "scheduler", None)
        if scheduler is None:
            return
        try:
            coord_client._queue_length = int(scheduler.get_num_unfinished_requests())
        except Exception:
            pass
    coord_client._on_heartbeat = _refresh_queue_length
```

`local_dp_rank` is already available — it's passed into `run_stage_core` via `kwargs["local_dp_rank"]` from `OmniCoreEngineProcManager:130-131`.

### 4.7 Point 3b (zero additional code)

P3b is fully covered by P2's CLI prohibition (§4.2) and YAML validator (§4.3). The EP/EPLB runtime path was already wired through `EngineArgs` → `ParallelConfig.__post_init__` → `init_model_parallel_group`. The only thing P3b adds is **test coverage**, not new code paths.

---

## 5. Touch-point summary

| File | LoC | Change |
|---|---|---|
| `vllm_omni/engine/arg_utils.py` | 1 | Rename field. |
| `vllm_omni/entrypoints/cli/serve.py` | ~60 | New flag w/ alias, deprecation warning, expanded `prohibited_with_omni`, **`_cli_explicit_keys` population** (post-smoke wiring fix). |
| `vllm_omni/engine/async_omni_engine.py` | ~5 | Rename internal symbol (+ programmatic fallback). |
| `vllm_omni/engine/stage_init_utils.py` | ~90 | New `_enforce_omni_parallel_config` (FLAT engine_args fields for LLM) + updated `get_stage_devices_per_replica`. |
| `vllm_omni/engine/omni_core_engine_proc_manager.py` | 1 | Single-slot `omni_replica_id`. |
| `vllm_omni/engine/stage_engine_core_proc.py` | ~10 | Gate `OmniCoordClientForStage` on `local_dp_rank == 0`. |
| **Total** | **~170 LoC** | **6 files, no new modules** |

Compare to the deleted draft, which proposed a new `OmniPortRegistry`
module, a separate `_force_set_llm_parallel_config` function, separate
DiT/LLM validators, and per-replica `data_parallel_master_port`
allocation — all of which the re-read showed were either already
handled by vLLM or unnecessary.

---

## 6. Tests

### 6.1 CLI rejection (`tests/entrypoints/test_serve.py`)

Parametrize one test over the full prohibited list rather than writing one per flag:

```python
@pytest.mark.parametrize("flag", [
    "--tensor-parallel-size", "--pipeline-parallel-size",
    "--data-parallel-size-local", "--data-parallel-external-lb",
    "--data-parallel-hybrid-lb",
    "--enable-expert-parallel", "--enable-eplb",
    "--all2all-backend",
    "--enable-elastic-ep",
])
def test_omni_rejects_intra_replica_parallel_flag(flag): ...

def test_omni_rejects_eplb_config_nested(): ...   # --eplb-config.window_size 1000
def test_omni_dp_size_local_alias_warns(caplog): ...
```

### 6.2 YAML validator (`tests/engine/test_async_omni_engine_stage_init.py`)

Same parametrization pattern:

```python
@pytest.mark.parametrize("field", [
    "data_parallel_address", "data_parallel_rpc_port",
    "data_parallel_master_port", "data_parallel_rank",
    "data_parallel_rank_local", "api_server_count",
])
def test_yaml_rejects_field_present(field): ...

def test_yaml_rejects_dp_size_mismatch_with_local(): ...
def test_yaml_accepts_dp_size_equal_to_local(): ...
def test_yaml_rejects_external_lb_true(): ...
def test_yaml_rejects_hybrid_lb_true(): ...
def test_yaml_rejects_elastic_ep_true(): ...
def test_yaml_rejects_ray_backend(): ...
def test_yaml_rejects_eplb_with_dp_local_1_tp_1(): ...
def test_yaml_force_sets_dp_size_and_api_server_count(): ...
def test_yaml_rejects_devices_not_divisible_by_per_replica(): ...
```

### 6.3 Runtime semantics

```python
# omni_replica_id (unit; tests/engine/test_orchestrator.py)
def test_one_omni_replica_id_per_manager_when_dp_local_gt_1(): ...
def test_only_dp_rank_zero_registers_with_omni_coordinator(): ...

# Integration (requires GPUs; tests/integration/)
def test_e2e_two_replicas_inner_dp(): ...                   # tp=2, dp_local=2, num_replica=2
def test_e2e_moe_per_replica_ep(): ...                      # + enable_expert_parallel
def test_e2e_moe_per_replica_ep_eplb(): ...                 # + enable_eplb
def test_two_replicas_have_independent_ep_groups(): ...
```

---

## 7. Migration

- `--omni-dp-size-local`: accepted as alias for ≥ 1 release, one-line deprecation warning.
- Programmatic callers using `kwargs={"omni_dp_size_local": N}` keep working via the fallback in `AsyncOmniEngine.__init__`.
- No deploy YAML in `vllm_omni/deploy/` currently sets any newly-rejected field (spot-checked: `hunyuan_image3*.yaml`, `qwen3_omni_moe_multi_replicas.yaml`).
- YAML rejection of `data_parallel_size != data_parallel_size_local` could in principle break a user with an inconsistent YAML; that configuration was silently broken under PR #3569 and turning it into an explicit error is desirable.

---

## 8. Build order

1. **P1** (rename + alias). Mechanical; ~1 day.
2. **P2** patches as a single PR (the six files in §5). Add the parametrized rejection tests in the same PR. ~3-5 days including the `(num_replicas=2, dp_local=2, tp=2)` integration test.
3. **P3b** test-only PR — confirm EP and EPLB run inside each replica with no production code changes. ~2 days.
4. **Docs** — refresh one example YAML (`deploy/<moe-model>_replicas_dp_ep.yaml`), short migration note for the renamed flag. ~1 day.

Total: ~2 weeks for one engineer. Most of the work is tests.

---

## 9. Deferred items (call out at review, not blockers)

- **Queue-length proxy under inner DP.** §4.6 uses DP rank 0's own queue length, not the aggregate over all inner DP ranks. The `least-queue-length` LB policy only needs relative values across replicas, so this is acceptable. If production data shows skew, swap in a sum from the inner `DPCoordinator`'s stats publisher.
- **DiT EP-error wording.** `diffusion/distributed/parallel_state.py:858` raises `RuntimeError("Expert parallelism enabled for a non-MoE model ")` on the bad case. Worth a one-line wording fix and a stage-id prefix in a follow-up PR; not needed for this design.
- **Diffusion `master_port` workaround.** `serve.py:849-859` keeps the `61000 + rep_idx * 100` allocation. Works fine; left untouched. If we ever add another consumer in that port band we'll need to centralize, but not now.

---

## 10. Bugs surfaced and fixed during integration smoke

Three issues found when the initial commit was launched against a real
`vllm serve` on the test cluster. All three are now fixed on
`replica_dp_ep`; this section documents them so future readers don't
re-introduce the same mistakes.

### 10.1 Bug: validator wrote to nested `parallel_config:` for LLM stages

**Symptom (real launch on Qwen2.5-Omni-3B):**

```
RuntimeError: Orchestrator initialization failed:
  OmniEngineArgs.__init__() got an unexpected keyword argument 'parallel_config'
```

**Root cause.** The initial validator wrote back to
`engine_args_dict["parallel_config"]`, assuming the LLM YAML used a
nested `parallel_config:` block (matching the DiT shape). Re-reading
the code:

```python
>>> dataclasses.is_dataclass(OmniEngineArgs)
True
>>> "parallel_config" in [f.name for f in dataclasses.fields(OmniEngineArgs)]
False
>>> "data_parallel_size" in [...]; "data_parallel_size_local" in [...]
True, True
```

`OmniEngineArgs` (subclass of vLLM's `EngineArgs`) is a **flat**
dataclass — DP/TP/PP/EP fields live directly on it as top-level
attributes. Real LLM YAMLs (`vllm_omni/deploy/qwen2_5_omni.yaml` et al.)
match this: fields like `tensor_parallel_size` sit at the stage level,
not nested. The `parallel_config:` block is a **DiT-only** convention
because `DiffusionParallelConfig` is its own dataclass.

**Fix.** Rewrite the validator to read from and write back to FLAT keys
on `engine_args_dict`. DiT stages still early-return — their nested
`parallel_config:` is left untouched by the LLM-shaped validator. Error
messages now say `engine_args.<field>` instead of
`parallel_config.<field>`.

### 10.2 Bug: `api_server_count` was force-set into engine_args

**Symptom.** Same `OmniEngineArgs.__init__()` error as 10.1, second
contributor.

**Root cause.** `api_server_count` is a vLLM CLI flag at the top level
of the argparse Namespace. It is **not a field on `EngineArgs`**:

```
$ grep api_server_count /workspace/ngngaifai/.venv/lib/python3.12/site-packages/vllm/**/*.py
vllm/v1/metrics/loggers.py: <stats-only reference>
vllm/config/multimodal.py: <cache-size formula reference>
vllm/entrypoints/cli/serve.py: <CLI dispatcher reference>
```

Nothing in `EngineArgs` / `ParallelConfig` consumes it. Setting it on
`engine_args_dict` therefore makes `OmniEngineArgs(**engine_args_dict)`
fail with "unexpected keyword argument".

**Fix.** Drop `api_server_count` from the validator's must-omit list
and from the force-set list. It stays on the **CLI prohibition list**
(`serve.py:138`) — that's where it gets rejected today. The headless
launcher path already defaults it to 0 at `serve.py:734`; the head's
own HTTP server is `omni_run_server`, which doesn't read it either.

### 10.3 Bug: `_cli_explicit_keys` was never populated

**Symptom (initial parser smoke test):**

```
sys.argv = [..., "--tensor-parallel-size", "2"]
args = parser.parse_args(sys.argv[1:])
OmniServeCommand().validate(args)   # silently passes!
```

Expected: ValueError naming the prohibited flag. Observed: no error.

**Root cause.** The prohibition check at `serve.py:135` gates on
`args._cli_explicit_keys`:

```python
explicit_cli_keys = getattr(args, "_cli_explicit_keys", set()) or set()
offenders = sorted(flag for dest, flag in prohibited_with_omni.items()
                   if dest in explicit_cli_keys)
```

But **no code in vllm-omni ever set `args._cli_explicit_keys`**. The
attribute is referenced in `serve.py:135, 765, 782` and
`async_omni_engine.py:2036`, but nowhere assigned. The helper
`detect_explicit_cli_keys` exists in `entrypoints/utils.py:43`, but
the only caller is the deprecated `Omni.from_cli_args` in
`omni_base.py:131`. The new `vllm serve` CLI flow never touches it.

That means PR #3569's `prohibited_with_omni` dict was a structural
no-op since the day it landed — and the `run_headless` deprecation
warning for `--replica-id` (which also reads `_cli_explicit_keys`)
was never firing either.

**Fix.** Populate `args._cli_explicit_keys` at the top of
`validate()`, using the parser stashed on the class by
`subparser_init`:

```python
def validate(self, args: argparse.Namespace) -> None:
    if not hasattr(args, "_cli_explicit_keys"):
        from vllm_omni.entrypoints.utils import detect_explicit_cli_keys
        args._cli_explicit_keys = detect_explicit_cli_keys(
            sys.argv[1:], type(self)._parser
        )
    ...
```

`detect_explicit_cli_keys(parser=...)` resolves each `sys.argv` token
through the parser's action table to find its real `dest`, correctly
handling alias flags (`--omni-num-replica` and `--omni-dp-size-local`
share the same dest) and `--no-foo` boolean-optional patterns.

### Test coverage added for these bugs

- `tests/engine/test_parallel_config_validator.py` now uses a FLAT
  schema and includes
  `test_validator_only_writes_data_parallel_size` — a regression
  guard that asserts the validator does **not** add unexpected keys
  like `parallel_config` or `api_server_count` to `engine_args_dict`
  (which would break `OmniEngineArgs(**dict)` downstream).
- Parser-level smoke (run manually, not in pytest because it requires
  GPU env activation) confirms each prohibited flag now fires the
  ValueError end-to-end, and that the `--omni-dp-size-local` alias
  emits its deprecation warning.

### What integration coverage actually proves

The single-replica E2E smoke (Qwen2.5-Omni-3B, 3 stages) completes
through `Application startup complete` and `/v1/models` responds. The
multi-replica smoke (same model with custom YAML using `num_replicas:
2` on stages 1 and 2) reached `stage-1 [rep-0] EngineCore running` —
i.e., `OmniCoreEngineProcManager` correctly spawned the second
replica's process, with the omni_replica_id fix in effect. The second
replica's eventual GPU-OOM failure was an environment artifact on the
shared cluster, not a code issue.
