# [RFC]: Omni-Replica + vLLM-DP + Expert-Parallel

## Motivation

PR [#3569](https://github.com/vllm-project/vllm-omni/pull/3569) introduced
process-local **replica** fan-out for a single stage, governed today by
`--omni-dp-size-local`. The flag name leaks an implementation detail
("vllm-omni's data parallel") that overlaps confusingly with vLLM's own
`--data-parallel-size-local`, and the current code path leaves three
intra-replica scaling axes — **vLLM DP**, **Expert Parallel**, and **EPLB** —
unreachable from a user-facing entrypoint:

- `--data-parallel-size-local` and `--enable-expert-parallel` are both on
  the CLI prohibition list under `--omni`, so they can't be set on the
  command line.
- The YAML `engine_args` route partially works for these axes today, but
  the `omni_replica_id` slot accounting is wrong when DP > 1, the
  `OmniCoordClientForStage` is created on every DP rank instead of just
  rank 0, and the CLI prohibition list itself was a structural no-op
  (`_cli_explicit_keys` was never populated).

The infrastructure to host all three already exists end-to-end in vLLM
(`DPLBAsyncMPClient`, `DPCoordinator`, `ParallelConfig`'s self-allocating
DP master port, the `EngineArgs` → `ParallelConfig.__post_init__` path for
`enable_expert_parallel` / `enable_eplb` / `eplb_config.*`). vLLM-Omni
just needs to (a) expose it correctly under a clean name, (b) fix the
slot / client-gating bugs, and (c) decide a single configuration surface.

## Proposed Change

Three changes layered on PR #3569, in order.

### P1 — Rename the replica-count flag

```
--omni-dp-size-local N    →    --omni-num-replica N
```

The old name is kept as a **deprecated alias** for one release. The
implementation symbol `omni_dp_size_local` is renamed to
`omni_num_replica` end-to-end, with a programmatic `kwargs.get(...)`
fallback so embedders aren't broken.

### P2 — Inner vLLM DP per replica, **YAML-only**

After this RFC lands, `--omni-num-replica` is the **only** intra-runtime
scaling knob on the CLI. Every other parallelism axis — TP, PP, DP, PCP,
DCP, EP, EPLB, microbatching — lives in the deploy YAML.

For **LLM stages**, fields are FLAT under `stages[]`, because
`OmniEngineArgs` (a subclass of vLLM's `EngineArgs`) is a flat dataclass
with no `parallel_config` field:

```yaml
stages:
  - stage_id: 0
    devices: "0,1,2,3"                # devices PER replica = tp×pp×pcp×dp_local
    tensor_parallel_size: 2           # (existing) — already reachable pre-RFC
    pipeline_parallel_size: 1         # (existing) — already reachable pre-RFC
    data_parallel_size: 2             # NEW — inner vLLM DP per replica (P2)
    prefill_context_parallel_size: 1  # NEW — PCP per replica (P2)
    enable_expert_parallel: true      # NEW — EP per replica (P3b)
    enable_eplb: true                 # NEW — EPLB per replica (P3b)
    eplb_config:                      # NEW — EPLB tuning per replica (P3b)
      window_size: 1000
      step_interval: 3000
    expert_placement_strategy: linear # NEW — EP placement strategy (P3b)
    all2all_backend: deepep_low_latency  # NEW — EP all2all backend (P3b)
```

For **DiT stages**, the existing nested `parallel_config:` shape
(`DiffusionParallelConfig`) is unchanged. The DiT runtime is a separate
`StageDiffusionProc` with its own parallelism axes
(`ulysses_degree`, `ring_degree`, `cfg_parallel_size`, …); the LLM
validator early-returns on DiT stages and the LLM-only fields above
don't apply:

```yaml
stages:
  - stage_id: 1
    stage_type: diffusion
    devices: "4,5,6,7"                # devices PER replica = tp × sp × cfg × dp
    parallel_config:                  # nested — DiffusionParallelConfig
      tensor_parallel_size: 2
      data_parallel_size: 1           # DiT FSDP/HSDP axis, not EngineCore DP
      ulysses_degree: 2               # sequence parallel (Ulysses)
      ring_degree: 1                  # sequence parallel (Ring); sp = ulysses × ring
      cfg_parallel_size: 2            # classifier-free-guidance parallel
      enable_expert_parallel: true    # DiT MoE EP (already supported today)
```

DiT's per-replica EP group is `tp × sp × cfg × dp` per PP stage
(`vllm_omni/diffusion/distributed/parallel_state.py:193`), which is
typically larger than the LLM-side `dp_local × pcp × tp`.

A new validator `_enforce_omni_parallel_config(stage_cfg, engine_args_dict)`
runs in `build_vllm_config`, **before** `filter_dataclass_kwargs`, and
performs:

- Reject user-set fields that vLLM / vllm-omni own
  (`data_parallel_address`, `data_parallel_rpc_port`,
  `data_parallel_master_port`, `data_parallel_rank`,
  `data_parallel_rank_local`).
- Reject nested `parallel_config:` on LLM stages (DiT-only shape).
- Reject `data_parallel_external_lb=True`,
  `data_parallel_hybrid_lb=True`, `enable_elastic_ep=True`,
  `data_parallel_backend != "mp"`.
- Device-count product check: `len(devices) % (tp × pp × pcp × dp_local) == 0`.
- EPLB sanity: `enable_eplb` requires `dp_local × tp > 1`.
- DiT stages early-return (the LLM rules don't apply).

This RFC does **not** add a CLI prohibition list for parallelism flags.
The deploy YAML is the authoritative surface; the parallelism CLI flags
(`--tensor-parallel-size`, `--enable-expert-parallel`, …) are treated as
unsupported under `--omni`. See
[Supporting parallelism CLI arguments — possible, but hazardous today](#supporting-parallelism-cli-arguments--possible-but-hazardous-today)
for why a hard parse-time prohibition was dropped in favor of documenting
the underlying hazard (chiefly a cross-stage KV-transfer rank-mapping
mismatch).

Two correctness fixes that go with P2:

1. **`omni_replica_id` is shared across DP ranks.** Today
   `OmniCoreEngineProcManager` assigns `omni_replica_id = base + index`
   when spawning the `dp_local` subprocesses of one replica, leaking
   phantom slots into `OmniCoordinator`'s `ReplicaList`. Fix: all DP
   ranks of a replica share the same `omni_replica_id`.
2. **`OmniCoordClientForStage` is created on DP rank 0 only.** Today
   every DP rank inside a replica opens its own coord-client; only rank
   0 should publish status/queue-length to `OmniCoordinator`, which is
   what `StagePool.LoadBalancer` consumes. Inner DP-rank routing is
   already handled by vLLM's `DPLBAsyncMPClient`.

### P3b — EP / EPLB per replica (no cross-replica EP)

Each replica owns its own EP group of size `dp_local × pcp × tp`
(per PP stage), bounded by the replica. Replicas are coupled only via
ZMQ control + data through the head; they share no NCCL world and no
`_EP` group. This is the natural extension of the
"replicas are independent" invariant from PR #3569.

The alternative — **3a, EP spanning all replicas** — was considered and
rejected; see the [companion analysis doc](./omni-replica-dp-ep-analysis.md).
Briefly: it would require collapsing the outer replica loop into a
single `OmniCoreEngineProcManager`, dissolving the replica abstraction;
break `StagePool.LoadBalancer` (no per-replica queue length to balance
on); and lose the isolation property that makes per-replica rolling
restart possible. Elastic EP is Ray-only and structurally incompatible
with the omni replica model.

### Worked example — MoE with per-replica EP + EPLB

A typical P3b deployment: a Mixtral-style MoE on a single 16-GPU host
fanned out into two replicas via `--omni-num-replica 2`, each replica
running EP across its own `dp_local × tp` group with EPLB rebalancing
inside it.

```bash
# Single launcher; --omni-num-replica fans the stage into 2 replicas
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
vllm serve <moe-model> --omni \
  --omni-num-replica 2 \
  --deploy-config deploy.yaml
```

```yaml
# deploy.yaml
stages:
  - stage_id: 0
    devices: "0,1,2,3,4,5,6,7"        # template mode; per_replica = tp × dp_local = 4 × 2 = 8
    tensor_parallel_size: 4
    data_parallel_size: 2              # inner vLLM DP per replica
    enable_expert_parallel: true       # P3b
    enable_eplb: true                  # P3b
    eplb_config:
      window_size: 1000
      step_interval: 3000
    expert_placement_strategy: linear
    all2all_backend: deepep_low_latency
```

`split_devices_for_replicas` expands the 8-GPU template to two per-replica
slices; `set_stage_devices` maps each replica's logical indices onto the
physical pool:

```
  Replica 0 — CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
    DP rank 0 — TP=4 across GPUs 0,1,2,3   ┐
    DP rank 1 — TP=4 across GPUs 4,5,6,7   ┘ EP group size = dp_local × pcp × tp = 8
  Replica 1 — CUDA_VISIBLE_DEVICES = "8,9,10,11,12,13,14,15"
    DP rank 0 — TP=4 across GPUs 8,9,10,11   ┐
    DP rank 1 — TP=4 across GPUs 12,13,14,15 ┘ EP group size = 8
  Each replica: own NCCL world, own DPCoordinator, own _EP group; EPLB
  rebalances expert placement every 3000 steps over a 1000-step window,
  using deepep_low_latency for the in-replica all2all.
```

Two LB layers in action: the head's `StagePool.LoadBalancer` routes each
request to one of `{replica 0, replica 1}` by per-replica queue length;
inside the chosen replica, `DPLBAsyncMPClient` picks DP rank 0 or 1 by
`waiting × 4 + running` from the local `DPCoordinator`. The two replicas
share **no** NCCL world, **no** `_EP` group, **no** EPLB state — replica
1 can be rolling-restarted (or lost entirely) without disturbing replica
0's expert placement or in-flight requests.

## Topology after this lands

```
                              ┌─────────────────────────────────────┐
                              │           Head process              │
                              │  ┌────────────────┐                 │
   client ─── HTTP ────────►  │  │  omni_run_     │ HTTP            │
                              │  │  server        │ (vllm-omni's    │
                              │  └───────┬────────┘  OpenAI server) │
                              │          │                          │
                              │  ┌───────▼────────┐                 │
                              │  │  StagePool     │ outer LB across │
                              │  │  .LoadBalancer │ replicas        │
                              │  └───┬───┬────┬───┘                 │
                              └──────┼───┼────┼─────────────────────┘
                                     │   │    │ ZMQ
            ┌────────────────────────┘   │    └────────────────────┐
            │                            │                         │
   ┌────────▼──────────┐    ┌────────────▼─────────┐    ┌──────────▼────────┐
   │ Omni replica 0    │    │ Omni replica 1       │    │ Omni replica N-1  │
   │ (own NCCL world)  │    │ (own NCCL world)     │    │ (own NCCL world)  │
   │ DPCoordinator     │    │ DPCoordinator        │    │ DPCoordinator     │
   │  ↳ inner LB       │    │  ↳ inner LB          │    │  ↳ inner LB       │
   │ DP rank 0 (TP×PP) │    │ DP rank 0 (TP×PP)    │    │ DP rank 0 (TP×PP) │
   │ DP rank 1 (TP×PP) │    │ DP rank 1 (TP×PP)    │    │ DP rank 1 (TP×PP) │
   │ One _EP group =   │    │ One _EP group =      │    │ One _EP group =   │
   │   dp_local×pcp×tp │    │   dp_local×pcp×tp    │    │   dp_local×pcp×tp │
   └───────────────────┘    └──────────────────────┘    └───────────────────┘
```

Two LB layers, no more, no less:

| Layer | Scope | Picker | Signal |
|---|---|---|---|
| Outer | across replicas | `StagePool.LoadBalancer` | OmniCoordinator's `ReplicaList` (status + queue length per replica) |
| Inner | DP ranks within a replica | vLLM's `DPLBAsyncMPClient` | per-replica `DPCoordinator` stats (~100ms cadence), `waiting × 4 + running` |

`data_parallel_external_lb` and `data_parallel_hybrid_lb` would each
short-circuit one of these layers; both are rejected by the YAML
validator.

## Configuration surface — YAML is authoritative

All intra-replica parallelism (TP, PP, DP, PCP, DCP, EP, EPLB,
microbatching) is configured in the deploy YAML. The only intra-runtime
CLI knob is **`--omni-num-replica`**.

> **This RFC drops the CLI prohibition list.** An earlier draft hard-rejected
> every vLLM parallelism flag under `--omni` at parse time. That check is
> *not* part of this design — see the next section for why supporting these
> flags is desirable in principle but unsafe in the current implementation,
> and why the conservative choice is to leave them unenforced-but-undocumented
> rather than ship a check that gives a false sense of completeness.

### Worked example — no CLI overriding (this RFC)

A typical single-stage LLM under the YAML-authoritative model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve <llm> --omni \
  --omni-num-replica 2 \
  --deploy-config deploy.yaml
```

```yaml
stages:
  - stage_id: 0
    devices: "0,1"               # template mode; per_replica = tp × dp_local = 2 × 1 = 2
    tensor_parallel_size: 2
```

The CLI carries only `--omni-num-replica`; every parallelism axis is
declared in YAML. Two replicas are spawned, each TP=2 on its own pair
of GPUs (replica 0 → physical 0,1; replica 1 → physical 2,3). The
validator's `len(devices) % (tp × pp × pcp × dp_local) == 0` check
runs before any subprocess spawns, so a typo in `devices:` fails fast
with a clear message rather than at worker init. No
`--tensor-parallel-size`, `--data-parallel-size`, `--enable-eplb` or
other vLLM parallelism flag participates — they are treated as
unsupported under `--omni` for this RFC (see the next section).

## Supporting parallelism CLI arguments — possible, but hazardous today

It is *technically possible* to let the existing vLLM parallelism CLI
flags (`--tensor-parallel-size`, `--enable-expert-parallel`, etc.) drive
the per-stage engines instead of requiring everything in YAML — the CLI
namespace already flows into `load_and_resolve_stage_configs` as
`cli_overrides`. Doing so would be a real ergonomic win for single-stage,
single-model deployments.

**But it carries major correctness problems in the current code, and this
RFC deliberately does not enable it.** Two structural issues:

### 1. TP is per-stage; a CLI flag is global

`--tensor-parallel-size` is a single process-global value, but in a
vllm-omni pipeline TP is an **independent per-stage** quantity (the LLM
stage and a DiT stage routinely run different TP). A global flag cannot
express that, and worse, the merge precedence between the CLI namespace
and the per-stage YAML is **loader-dependent**:

- Legacy YAML loader (`load_stage_configs_from_yaml`,
  `prefer_stage_engine_args=True`): the per-stage YAML **wins**, so a
  user-supplied `--tensor-parallel-size` is **silently ignored** for any
  stage that sets TP — a silent disagreement between what the operator
  typed and what runs.
- Factory loader (`StageConfigFactory.create_from_model` with
  `cli_overrides`): the CLI value is applied, potentially **overriding
  every stage uniformly** — including stages whose YAML intended a
  different TP, changing their world size and breaking the `devices:`
  divisibility contract.

Same flag, two opposite behaviors depending on config format.

### 2. It can break cross-stage KV-cache transfer

This is the concrete failure mode, and it is grounded in the current
code. Stages that hand off KV cache (e.g. a prefill→decode or LLM→codec
disaggregation) compute their transfer rank mapping from the **TP of each
stage as read from the resolved deploy config**:

- `_inject_inferred_kv_tp_topology` (`vllm_omni/engine/stage_init_utils.py:243`)
  sets `rank_mapping["from_tp"]` / `["to_tp"]` for the KV connector.
- Both come from `_tp_size_for_stage(stage_configs, ...)`
  (`stage_init_utils.py:217`), which reads `tensor_parallel_size` out of
  the **stage config objects** — i.e. the deploy YAML's view of each
  stage.
- The KV transfer manager then uses `from_tp`/`to_tp` to map cache blocks
  between a sender sharded `from_tp` ways and a receiver sharded `to_tp`
  ways.

The danger: if a CLI `--tensor-parallel-size` changes a stage's **actual
runtime TP** without the topology-inference path (which reads the YAML
`stage_configs`) seeing the same value, the inferred `from_tp`/`to_tp`
no longer matches the real sharding. The KV transfer manager then maps
blocks to ranks that don't exist or don't correspond to the real layout
→ the transfer **hangs or silently corrupts the KV cache**. Because the
two read paths (engine build vs. topology inference) and the CLI-vs-YAML
precedence are not unified, this divergence is reachable rather than
hypothetical.

Example (LLM stage 0 → codec stage 1, the codec receives stage 0's KV):

```
deploy.yaml:  stage 0 tensor_parallel_size: 2     stage 1 tensor_parallel_size: 1
CLI:          vllm serve --omni --tensor-parallel-size 4 ...

  Intended by operator: bump stage 0 to TP=4.
  Topology inference reads YAML → from_tp=2, to_tp=1   (stale: still the YAML values)
  If the override reaches stage 0's engine → it actually runs TP=4.
  Result: connector expects a 2-way sender, the sender is 4-way →
          rank mapping wrong → KV blocks mis-routed → transfer hang / corruption.
```

### Why defer rather than prohibit

Two options were on the table — keep a hard CLI prohibition, or do the
work to support CLI flags safely (thread a single source of TP through
both the engine build and the KV-topology inference, and forbid a global
flag in multi-stage configs). This RFC does **neither**: it ships YAML as
the authoritative surface, documents the hazard above, and leaves CLI
parallelism support as scoped future work. The prohibition check is
dropped because, as written, it depended on `_cli_explicit_keys` and gave
the impression of a guarantee it did not actually provide; documenting the
real hazard is more honest than a partial gate. Operators should treat the
parallelism CLI flags as **unsupported under `--omni`** and configure
parallelism in the deploy YAML.

### Future support — handshake-propagated parallel config

If CLI parallelism overrides are added in a future iteration, the
correctness gap above (cross-stage KV rank-mapping read from a per-process
view of the YAML) can be closed without requiring every process to agree
on CLI flags up-front. The proposal is to move the source-of-truth for a
peer's parallel config **off the locally loaded YAML** and onto the
**OmniMasterServer / OmniCoordinator handshake** — the same handshake
each replica already performs to register with the orchestrator.

Two changes:

1. **At registration, each replica submits its own CLI-overridden,
   post-resolution parallel config** (the *actual* values its engine will
   shard by — TP, PP, DP-local, and, for DiT, the nested
   `parallel_config:` fields) to OmniMasterServer / OmniCoordinator. The
   master records this per `(stage_id, replica_id)` keyed entry alongside
   the existing handshake/coordinator endpoints already exchanged at
   `register_stage_with_omni_master`.
2. **When the orchestrator dispatches a task to a downstream replica, it
   encloses the registered parallel config of the immediate upstream
   replica** (the one whose KV cache the downstream is about to pull
   from). The connector inside the downstream replica reads
   `from_tp` / `from_rank` from that authoritative bundle rather than
   from its local `stage_configs` view — so the rank set it pulls from,
   the keys it constructs, and any head-slice/concat math it performs
   are all derived from the **sender's actual sharding** rather than
   from whatever the downstream's own YAML or CLI happens to say about
   the sender.

This works because the KV transports are already pull-based (every
shipped connector — `MooncakeTransferEngineConnector`,
`MooncakeStoreConnector`, `ShmConnector`, `YuanrongConnector`,
`YuanrongTransferEngineConnector` — uses producer `put` / consumer `get`
semantics, with the receiver initiating the actual transfer). The
receiver is the only side that needs to know both endpoints' TP to do
its rank arithmetic; the orchestrator handing it the upstream's
registered config gives it exactly that, sourced from the upstream's
own engine rather than reconstructed from a file. Upstream-side
key construction and any fan-out pre-slicing remain correct as long as
the upstream and the orchestrator agree on the upstream's own values,
which they trivially do (the upstream itself reported them).

**Concrete failure this fixes (multi-stage, multi-CLI-runtime).**
Consider a model with two stages whose deploy YAML declares:

```yaml
stages:
  - stage_id: 0
    tensor_parallel_size: 2
    ...
  - stage_id: 1
    tensor_parallel_size: 1
    ...
```

An operator wants to run two stage-0 replicas on two hosts with
different per-host parallelism, plus a stage-1 receiver:

```bash
# Host A — stage 0, replica 0
vllm serve <model> --omni --stage-id 0 --tensor-parallel-size 2 ...

# Host B — stage 0, replica 1 (different TP from host A)
vllm serve <model> --omni --stage-id 0 --tensor-parallel-size 4 --headless ...

# Some host — stage 1
vllm serve <model> --omni --stage-id 1 --headless ...
```

Under today's implementation this is a silent correctness bug. Stage 1
infers its `from_tp` by calling `_tp_size_for_stage(stage_configs, 0)`,
which reads the deploy YAML's `stage 0: tensor_parallel_size: 2`. It then
constructs pull keys and source-rank sets sized for a 2-way sender —
but the stage-0 replica on Host B is actually sharded 4 ways. Stage 1
has **no information** about either replica's actual TP, because:

1. Stage 1's own process never observed the `--tensor-parallel-size`
   flags the two stage-0 invocations were launched with.
2. Nothing in the runtime today carries the resolved per-replica
   parallel config from a stage-0 replica back to a stage-1 replica that
   pulls from it.

The result is exactly Problem 4: pulls under wrong keys for the
host-B-served requests, or wrong shard counts/shapes, depending on which
replica routed the request — a hang or KV corruption either way.

Under the proposed handshake-propagated parallel config:

- Host-A stage-0 registers `tensor_parallel_size=2` with the master at
  startup.
- Host-B stage-0 registers `tensor_parallel_size=4` with the master at
  startup.
- The master / orchestrator keeps both keyed by replica id.
- When the orchestrator routes a request to stage 1, it knows which
  stage-0 replica produced the KV (the same one StagePool's LB picked
  for the upstream step) and **attaches that replica's registered TP**
  to the work dispatched to stage 1.
- Stage 1's connector reads `from_tp` from the attached bundle —
  2 for Host-A-served requests, 4 for Host-B-served requests — and
  constructs the right keys / pulls the right source ranks for each.

The deploy YAML can still differ from the registered values on either
end without breaking the transfer, because the YAML is no longer the
authority for cross-process topology; the registering replica is.

### Worked example — with CLI overriding (under handshake-propagated parallel config)

The same single-stage LLM as the [no-CLI-overriding example](#worked-example--no-cli-overriding-this-rfc),
but with `--tensor-parallel-size` lifted onto the CLI so an operator can
re-shape parallelism from the launch command without editing YAML. Only
safe under the handshake-propagated design above; *not* enabled by this RFC.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve <llm> --omni \
  --omni-num-replica 2 \
  --tensor-parallel-size 2 \      # CLI override; post-resolution TP = 2
  --deploy-config deploy.yaml
```

```yaml
stages:
  - stage_id: 0
    devices: "0,1"                 # per_replica = tp × dp_local = 2 × 1 = 2 (under the override)
    tensor_parallel_size: 1        # baseline; CLI overrides to 2
```

What changes from the no-CLI-overriding case:

- Each replica computes its actual TP after CLI resolution (=2 here, CLI
  wins) and **registers `tensor_parallel_size=2`** with `OmniMasterServer`
  on the existing handshake.
- For any cross-stage receiver pulling KV produced by this stage, the
  orchestrator attaches *this replica's* registered TP to the work
  dispatched downstream. The receiver's connector reads `from_tp=2` off
  the attached bundle rather than from `_tp_size_for_stage` on its own
  local YAML view — so it constructs the right keys and pulls the right
  source ranks even when its own YAML still reads `tensor_parallel_size: 1`.
- The validator's device-count check uses the post-resolution per-replica
  product, so the same `devices:` field that was valid YAML-only stays
  valid under the override (here `len("0,1") = 2 = 2 × 1`).

The cross-host extension (different `--tensor-parallel-size` per replica
host) falls out the same way: each replica registers its own resolved
TP, and the orchestrator hands the correct upstream value to each
downstream work item — exactly the multi-host case worked through in
the previous subsection.

### Out-of-scope for this iteration

This RFC does **not** implement the registration/dispatch wiring above,
nor does it enable CLI parallelism overrides. The handshake-propagated
parallel-config proposal is sketched here so that operators and reviewers
have a concrete forward path for the hazard, and so the prohibition's
removal in this RFC is paired with a documented design for the eventual
safe-support path rather than left open-ended. Companion analysis lives
at `docs/design/cli-yaml-parallelism-conflicts.md`.

## Device mapping (single-stage mode, existing implementation)

Three independent things decide which physical GPU a worker lands on.
Keeping them straight is the single most common source of OOM /
"everything stacked on cuda:0" confusion, so this section spells out the
contract.

| Layer | Owned by | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | the launcher's environment (whoever runs `vllm serve`) | The **physical** GPU pool this process may touch, e.g. `CUDA_VISIBLE_DEVICES=1,3,4,5`. Everything below is relative to this. |
| YAML `devices:` | the deploy YAML, per stage | **Logical** indices into the launcher's `CUDA_VISIBLE_DEVICES`, e.g. `devices: "0,1"` means "the 1st and 2nd entries of CVD". |
| `--omni-num-replica N` | the CLI | How many replicas to fan the stage out into. Slices the YAML `devices:` into N per-replica subsets; each subset becomes that replica's own narrowed `CUDA_VISIBLE_DEVICES`. |

### Resolution order

```
  CUDA_VISIBLE_DEVICES = "1,3,4,5"        # launcher env: physical pool
            │
            ▼
  YAML devices: "0,1"                     # logical indices into the pool
  per-replica device count = tp×pp×pcp×dp_local   (validated divisible)
            │
            ▼  split_devices_for_replicas(devices, N, per_replica, stage_id)
            │
   --omni-num-replica 2 ──► replica 0 → logical "0,1"   replica 1 → logical "2,3"
            │
            ▼  set_stage_devices() maps logical → physical via the pool
   replica 0  CUDA_VISIBLE_DEVICES = "1,3"   (pool[0], pool[1])
   replica 1  CUDA_VISIBLE_DEVICES = "4,5"   (pool[2], pool[3])
```

`set_stage_devices` (`vllm_omni/entrypoints/stage_utils.py`) performs the
logical→physical remap: `physical = CUDA_VISIBLE_DEVICES[logical_index]`.
It is **idempotent** — if the YAML values already name entries that exist
in the visible pool (e.g. a parent harness already narrowed CVD before
spawning), it uses them as-is and skips the remap, so double-mapping
can't crash.

### How `devices:` is sliced — two accepted shapes

`split_devices_for_replicas` (`stage_init_utils.py`) accepts two YAML
shapes when `--omni-num-replica > 1`, distinguished by length
(`per_replica = tp × pp × pcp × dp_local`):

**1. Template mode** — `len(devices) == per_replica`. The YAML declares a
single replica's shape and is **replica-count-independent**. Replica `r`
gets logical offsets `[r*per_replica + a for a in template]`; the
template's entries must lie in `[0, per_replica)`.

```
devices: "0,1"   per_replica=2
  --omni-num-replica 2 → ["0,1", "2,3"]
  --omni-num-replica 4 → ["0,1", "2,3", "4,5", "6,7"]
```

The same `devices: "0,1"` works for any `--omni-num-replica` — the
launcher's `CUDA_VISIBLE_DEVICES` scales, the YAML does not. **This is
the recommended shape.**

**2. Pool / legacy mode** — `len(devices) == num_replicas * per_replica`.
The YAML enumerates the full per-stage pool; each replica takes
`per_replica` consecutive entries.

```
devices: "1,2,3,4"   per_replica=2   --omni-num-replica 2
  → ["1,2", "3,4"]
```

Any other length raises `ValueError` (the two modes are length-disjoint
for `num_replicas > 1`). When `--omni-num-replica == 1`, the YAML
`devices:` is used unchanged (still remapped through the pool).

### Worked example

```
Launcher:   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   (8 physical GPUs)
CLI:        vllm serve --omni --omni-num-replica 2 --deploy-config deploy.yaml
YAML stage: devices: "0,1,2,3"        # template mode, per_replica = tp×dp_local = 2×2 = 4
            tensor_parallel_size: 2
            data_parallel_size: 2

  split_devices_for_replicas("0,1,2,3", 2, 4, 0)
     → replica 0 logical "0,1,2,3"   replica 1 logical "4,5,6,7"

  set_stage_devices remap (pool = 0..7, identity here):
     replica 0  CUDA_VISIBLE_DEVICES = "0,1,2,3"
       └ DP rank 0 → GPUs 0,1 (TP=2)   DP rank 1 → GPUs 2,3 (TP=2)
     replica 1  CUDA_VISIBLE_DEVICES = "4,5,6,7"
       └ DP rank 0 → GPUs 4,5 (TP=2)   DP rank 1 → GPUs 6,7 (TP=2)
```

The validator's device-count check (`len(devices) % per_replica == 0`)
runs before any subprocess is spawned, so a mismatched `devices:` /
parallelism product fails fast with a clear message instead of an
opaque CUDA error at worker init.

### When the pool is too small (device shortfall)

There is **no up-front check** that `len(CUDA_VISIBLE_DEVICES)` is large
enough for `--omni-num-replica × per_replica`. The validator's
divisibility check operates on the YAML `devices:` string, not the
physical pool; `split_devices_for_replicas` likewise only inspects the
YAML length. A shortfall surfaces **late and inconsistently** in
`set_stage_devices → _map_device_list`, per replica:

```python
mapped = [pool[idx] for idx in logical_ids if idx < len(pool)]   # silently drops out-of-range
if not mapped:        raise ValueError("...none of which map to the visible devices...")
if len(mapped) < len(logical_ids):  logger.warning("...Falling back to mapped subset...")
```

So depending on the YAML shape you get one of:

- **Hard `ValueError`** — a replica whose logical indices are *all* out of
  range (e.g. template-mode replica 3's `"6,7"` against a 6-GPU pool).
  Acceptable, but it's the *N-th replica* that fails, not a clear
  "needed 8, have 6".
- **Silent truncation** — a replica *straddling* the boundary runs on a
  truncated GPU subset (warning only), then trips a vLLM world-size
  assertion at worker init, or two replicas **overlap on the same
  physical GPU** and you get an OOM that looks unrelated.

> **Recommendation:** add a pre-spawn guard to `_enforce_omni_parallel_config`
> (it already computes `per_replica`) that reads the visible-device count
> and fails fast:
> ```
> required = omni_num_replica * per_replica
> pool     = len(CUDA_VISIBLE_DEVICES split)  # or full GPU count if unset
> if pool < required:
>     raise ValueError(
>         f"stage {sid}: --omni-num-replica {N} × per-replica {per_replica} "
>         f"= {required} GPUs required, but only {pool} visible. "
>         f"Widen CUDA_VISIBLE_DEVICES or lower --omni-num-replica."
>     )
> ```

## Device mapping (multi-stage, single process)

The section above is the *headless* path — one `vllm serve --omni
--headless --stage-id N` per stage, replica-scaled by
`--omni-num-replica`. The other common deployment is the **head-owned
full pipeline**: a single `vllm serve --omni --deploy-config deploy.yaml`
(no `--headless`, no `--stage-id`) launches *every* stage in one process.
This is the natural way to run, say, an LLM stage feeding a DiT stage.

Here `--omni-num-replica` defaults to 1, so each stage has one replica.
The device contract changes in one important way:

> **The YAML `devices:` of each stage are logical indices into the same
> launcher `CUDA_VISIBLE_DEVICES` pool. vllm-omni does not auto-partition
> the pool across stages — you place each stage explicitly, and stages
> are free to *share* physical GPUs (co-location) or take disjoint
> slices.** When stages overlap, you own the memory budget: each stage's
> `gpu_memory_utilization` must be low enough that the co-located stages
> fit together on the shared cards.

`setup_stage_devices(stage_id, runtime_cfg)` (`async_omni_engine.py`)
brackets each stage's worker spawn: it sets the process
`CUDA_VISIBLE_DEVICES` to *that stage's* mapped physical devices, spawns
the stage's workers (which inherit the narrowed env), then **restores the
parent env** before moving to the next stage. So the two stages are
applied sequentially against the same pool, each remapped through
`set_stage_devices` exactly as in the single-stage case — independently,
which is *why* overlap is possible at all.

### Worked example — LLM + DiT partially overlapping (share one GPU)

Stages can also overlap *partially* — each keeps some exclusive GPUs and
shares one (or more) with the other stage. Here the LLM stage owns GPU 0,
the DiT stage owns GPU 2, and both share GPU 1:

```
Launcher:  CUDA_VISIBLE_DEVICES=0,1,2,3   (4 physical GPUs)
CLI:       vllm serve --omni --deploy-config deploy.yaml      # no --headless, no --stage-id

deploy.yaml:
  stages:
    - stage_id: 0                    # LLM
      devices: "0,1"                 # logical → pool[0,1]
      tensor_parallel_size: 2        # per-stage GPUs = tp = 2
      gpu_memory_utilization: 0.45   # uniform across THIS stage's GPUs (0 and 1)
    - stage_id: 1                    # DiT
      stage_type: diffusion
      devices: "1,2"                 # logical → pool[1,2]  ← GPU 1 shared, GPU 2 exclusive
      parallel_config:
        tensor_parallel_size: 2      # per-stage GPUs = tp = 2
      gpu_memory_utilization: 0.45   # uniform across THIS stage's GPUs (1 and 2)

  setup_stage_devices(0, ...) → CUDA_VISIBLE_DEVICES="0,1"  (spawn LLM workers)  → restore
     └ TP rank 0 → GPU 0   TP rank 1 → GPU 1
  setup_stage_devices(1, ...) → CUDA_VISIBLE_DEVICES="1,2"  (spawn DiT workers)  → restore
     └ TP rank 0 → GPU 1   TP rank 1 → GPU 2

Resulting physical layout:
     GPU 0 : LLM only        (0.45 used, 0.55 idle)
     GPU 1 : LLM + DiT       (0.45 + 0.45 = 0.90 < 1.0 → OK)   ← the shared card
     GPU 2 : DiT only        (0.45 used, 0.55 idle)
     GPU 3 : unused
```

The overlap is exactly one card (GPU 1). It works because each stage's
`devices:` is remapped **independently** through `set_stage_devices` —
vllm-omni never reserves a GPU exclusively for a stage, so the DiT
stage's logical `"1,2"` maps straight onto physical 1 and 2, with 1
landing back on the LLM's GPU.

The constraint that makes it safe lives only on the **shared** card:
`0.45 + 0.45 < 1.0` on GPU 1. But note the catch — `gpu_memory_utilization`
is a single per-stage scalar applied **uniformly** to all of that stage's
GPUs. To keep GPU 1 under budget you also hold GPU 0 and GPU 2 down to
0.45, wasting ~0.55 of each exclusive card. There is no per-GPU memory
knob, so **partial overlap usually trades capacity for layout
flexibility** — full overlap (every GPU shared) or fully disjoint stages
are the cleaner endpoints. The accidental version — partial overlap with
the **default** `gpu_memory_utilization=0.9` each — OOMs on GPU 1 and is
the footgun below.

### Footguns vllm-omni does not catch

Because vllm-omni does **not** partition the pool for you, two mistakes
are easy and currently only fail late:

1. **Unbudgeted overlap.** Two stages sharing one or more GPUs while
   *both* keep a high `gpu_memory_utilization` → the second stage OOMs (or
   the first never leaves room) at worker init. Overlap (full or partial)
   is legal and often intended (above); what's unchecked is that, **on
   every shared GPU**, the sum of the co-located stages' memory fractions
   stays under 1.0. The per-stage parallel-config validator checks each
   stage's `len(devices) % per_replica == 0` in isolation; it does not
   reason about cross-stage GPU sharing or memory.
2. **Pool overflow.** A stage whose logical `devices:` indices exceed
   `len(CUDA_VISIBLE_DEVICES)` is not checked up front (same gap as the
   single-stage shortfall above); out-of-range indices are silently
   truncated in `_map_device_list` or raise the late "none of which map"
   `ValueError`.

> Recommendation (same as the single-stage shortfall note): a pre-spawn
> guard that (a) rejects any stage whose max logical index exceeds the
> pool, and (b) warns when the stages co-located on any one physical GPU
> have summed `gpu_memory_utilization` exceeding 1.0, would turn both
> footguns into one clear up-front message. Worth folding into the P2
> validator work.

## Alternative design — replica count in YAML 

### Configuration surface

```yaml
stages:
  - stage_id: 0
    num_replica: 2                    # NEW field — per-stage, replaces --omni-num-replica
    devices: "0,1,2,3,4,5,6,7"        # MUST enumerate all replicas (see below)
    tensor_parallel_size: 2
    data_parallel_size: 2             # per_replica = tp×pp×pcp×dp = 4
```

### Device mapping consequence

This is the crux: once `num_replica` is fixed in the YAML, the
`devices:` field **must account for it** — it can no longer be a
single-replica template that the CLI multiplies. The contract becomes
**pool-mode only**:

```
len(devices)  ==  num_replica × per_replica          # strict equality
per_replica   =   tp × pp × pcp × dp_local
```

Replica `r` deterministically takes the `r`-th consecutive chunk:

```
devices: "0,1,2,3,4,5,6,7"   num_replica=2   per_replica=4
  replica 0 → logical "0,1,2,3"   replica 1 → logical "4,5,6,7"
```

Template mode (`len(devices) == per_replica`) is gone — there is no
external multiplier to expand it against, so a `devices:` that lists
fewer than `num_replica × per_replica` entries is now an error rather
than a shorthand.

### Trade-offs vs the CLI design

| Dimension | CLI flag (this RFC) | `num_replica` in YAML (this alternative) |
|---|---|---|
| Source of truth | Split: topology in YAML, scale on CLI | **Single**: everything in the deploy file |
| Per-stage replica count | One process-global value | **Per-stage** — `4` LLM replicas + `1` DiT replica in one file |
| Scaling to fill a bigger box | Change one CLI token; template `devices:` unchanged | **Edit YAML** — recompute the full `devices:` list |
| `devices:` field | Short, replica-count-independent template | Long, must be recomputed when `num_replica` changes |
| Reproducibility | Depends on launch command | **Fully captured** by the YAML |
| Autoscaling / sweeps | Natural (vary a flag) | Needs YAML templating per scale point |
| Device placement clarity | Implicit (template + flag) | **Explicit** (every replica's GPUs are written out) |

### Pros and cons at a glance

| | CLI flag — `--omni-num-replica` (this RFC) | `num_replica` in YAML (alternative) |
|---|---|---|
| **Pros** | • One-token scaling; no YAML edit to change replica count<br>• Same YAML is **portable** across boxes of different GPU counts (template `devices:` scales)<br>• Simplest validator + `devices:` contract<br>• Natural fit for autoscaling / parameter sweeps | • **Single source of truth** — deployment fully described by the file<br>• **Per-stage** replica counts (e.g. 4 LLM + 1 DiT) in one config<br>• **Reproducible** — re-running the YAML reproduces the exact topology<br>• **Explicit** device placement; every replica's GPUs are visible |
| **Cons** | • Split source of truth (topology in YAML, scale on CLI)<br>• One process-global replica count — no per-stage scaling<br>• Device placement is implicit (template × flag)<br>• Topology not captured by the YAML alone | • Must **edit YAML** (recompute the full `devices:` list) to scale<br>• `devices:` is long and **replica-count-dependent**; template mode gone<br>• Not portable across box sizes without re-templating<br>• Stricter validation; more room for hand-edit mistakes |


### Why this RFC defaults to the CLI flag

The common operational case is "same model, scale replicas to fill the
GPUs I was given" — a one-token change (`--omni-num-replica N`) against
an unchanged YAML, which also makes the same config portable across boxes
of different size. The YAML-only approach is strictly better when the
topology is **fixed and heterogeneous** (different replica counts per
stage, pinned device placement, config-as-artifact reproducibility) and
worse when replica count is the thing you tune most often.

The two are not mutually exclusive: a later iteration could accept
`stages[].num_replica` as an *optional* per-stage override and keep
`--omni-num-replica` as the global default (CLI flag wins only where the
YAML is silent). That hybrid is out of scope here — this RFC ships the
CLI-only knob to keep the validator and the `devices:` contract simple,
and records the YAML-only variant as the documented alternative.

## Implementation overview

| File | Added | Deleted | What |
|---|---:|---:|---|
| `vllm_omni/engine/stage_init_utils.py` | 156 | 2 | `_enforce_omni_parallel_config` + `_resolve_devices_str`; updated `get_stage_devices_per_replica`. |
| `vllm_omni/entrypoints/cli/serve.py` | ~40 | 34 | `--omni-num-replica` flag w/ deprecated alias. **No** parallelism CLI prohibition list (dropped — see [hazard section](#supporting-parallelism-cli-arguments--possible-but-hazardous-today)); `_cli_explicit_keys` wiring retained only for the deprecation-alias warning. |
| `vllm_omni/engine/async_omni_engine.py` | 16 | 10 | Field rename + programmatic-compat fallback. |
| `vllm_omni/engine/stage_engine_core_proc.py` | 16 | 5 | Gate `OmniCoordClientForStage` on `local_dp_rank == 0`. |
| `vllm_omni/engine/omni_core_engine_proc_manager.py` | 7 | 4 | `omni_replica_id = base` (all DP ranks share one slot). |
| `vllm_omni/engine/arg_utils.py` | 1 | 1 | Field rename. |
| `vllm_omni/engine/stage_engine_startup.py` | 1 | 1 | Docstring rename. |
| **Total (production)** | **~237** | **57** | ~180 net LoC (prohibition list dropped). |
| `tests/engine/test_parallel_config_validator.py` | 274 | 0 | 27 cases over the validator. |
| `tests/engine/test_omni_replica_id_semantics.py` | 82 | 0 | DP-rank `omni_replica_id` invariant. |
| `tests/entrypoints/test_serve.py` | 4 | 4 | Rename in existing tests. |
| **Total (tests)** | **360** | **4** | ~356 net LoC. |

No new files, no new daemons, no new ZMQ endpoints; `OmniPortRegistry` is
not needed because vLLM's `ParallelConfig.__post_init__` already
self-allocates `data_parallel_master_port` via `get_open_ports_list(5)`.

### What we don't build (re-verified against the code)

1. **No `OmniPortRegistry`.** vLLM auto-allocates DP master ports.
2. **No force-set firehose.** vLLM defaults already cover most of the
   "force-set fields" the initial draft assumed (`data_parallel_backend="mp"`,
   `data_parallel_external_lb=False`, `enable_elastic_ep=False`, etc.).
   Only `data_parallel_size` needs an explicit force-set.
3. **No EP / EPLB wiring.** The full `EngineArgs` → `ParallelConfig` →
   `init_model_parallel_group` path already supports EP / EPLB. P3b is
   pure validation.

## Alternatives considered

- **3a — EP spans all replicas.** Rejected. Would require collapsing
  the outer replica loop into one `OmniCoreEngineProcManager` (no more
  per-replica `omni_replica_id`, no per-replica NCCL world), which
  dissolves the abstraction PR #3569 just introduced and breaks
  `StagePool.LoadBalancer`. Trades a marginal EP-scale gain for losing
  replica-level rolling restart, fault isolation, and the outer LB.
- **Elastic EP.** Ray-only; the omni replica model is process-local
  multiprocessing. Structurally incompatible — rejected by the YAML
  validator (`enable_elastic_ep=True`).
- **Hard CLI prohibition of parallelism flags.** An earlier draft
  rejected every vLLM parallelism flag under `--omni` at parse time.
  Dropped: the check depended on `_cli_explicit_keys` and implied a
  guarantee it didn't deliver, and it foreclosed a genuinely useful
  future feature. This RFC instead documents *why* CLI parallelism
  support is unsafe today (cross-stage KV-transfer rank-mapping mismatch)
  and defers it — see
  [Supporting parallelism CLI arguments](#supporting-parallelism-cli-arguments--possible-but-hazardous-today).
- **Validator on a nested `parallel_config:` dict.** Initial draft;
  rejected after integration smoke: `OmniEngineArgs` is a flat
  dataclass, writing back to `engine_args_dict["parallel_config"]`
  fails with "unexpected keyword argument 'parallel_config'". Final
  validator operates on FLAT fields and explicitly rejects nested
  `parallel_config:` on LLM stages.
- **Replica count in YAML (`stages[].num_replica`), no CLI flag.** A
  viable symmetric alternative — single source of truth, per-stage
  replica counts — at the cost of a replica-count-dependent `devices:`
  field and YAML edits to scale. Documented in full under
  [Alternative design — replica count in YAML](#alternative-design--replica-count-in-yaml-no-cli-flag);
  deferred in favor of the CLI knob for scaling ergonomics.

## Backwards compatibility

- `--omni-dp-size-local` continues to work as a deprecated alias of
  `--omni-num-replica` for **one release**, with a one-shot warning.
- Programmatic embedders that pass `omni_dp_size_local=` as a kwarg to
  `AsyncOmniEngine.__init__` continue to work via a `kwargs.get(...)`
  fallback in the constructor.
- Existing deploy YAML files under `vllm_omni/deploy/*.yaml` keep working
  unchanged; the validator is a soft-pass for DiT and a no-op for LLM
  stages that don't set the rejected fields.
- No on-the-wire ZMQ protocol changes; `OmniCoordinator` message shapes
  are unchanged.

## Test plan

### 1. Unit — YAML validator (`tests/engine/test_parallel_config_validator.py`, 27 cases)

Target: `_enforce_omni_parallel_config(stage_cfg, engine_args_dict)` in
`vllm_omni/engine/stage_init_utils.py`.

| # | Case | Pass criterion |
|---:|---|---|
| 1 | `test_rejects_nested_parallel_config_on_llm_stage` | `ValueError` matching `"nested parallel_config: block is for DiT"`. |
| 2 | `test_accepts_none_parallel_config_on_llm_stage` | `parallel_config: None` on an LLM stage is a no-op. |
| 3–7 | `test_rejects_user_set_must_omit_field` (parametrized over `data_parallel_address`, `data_parallel_rpc_port`, `data_parallel_master_port`, `data_parallel_rank`, `data_parallel_rank_local`) | `ValueError` matching `engine_args.{field}`. |
| 8 | `test_rejects_dp_size_mismatch_with_local` | `data_parallel_size=4` + `data_parallel_size=2` raises `"must equal data_parallel_size"`. |
| 9 | `test_accepts_dp_size_equal_to_local` | `data_parallel_size=2` + `data_parallel_size=2` passes; value preserved. |
| 10 | `test_force_sets_dp_size_to_local_when_absent` | `data_parallel_size=3` only → `data_parallel_size == 3`. |
| 11 | `test_force_sets_dp_size_to_one_when_neither_set` | Empty dict → `data_parallel_size == 1`. |
| 12–14 | `test_rejects_lb_or_elastic_ep_true` (parametrized over `data_parallel_external_lb`, `data_parallel_hybrid_lb`, `enable_elastic_ep`) | `ValueError` on `True`. |
| 15–17 | `test_accepts_lb_or_elastic_ep_false` (same fields) | `False` passes silently. |
| 18 | `test_rejects_ray_backend` | `data_parallel_backend="ray"` raises `"data_parallel_backend must be 'mp'"`. |
| 19 | `test_accepts_mp_backend_explicitly` | `data_parallel_backend="mp"` passes. |
| 20 | `test_rejects_eplb_when_dp_local_1_tp_1` | `enable_eplb=True` with `dp_local=1, tp=1` raises `"enable_eplb requires"`. |
| 21 | `test_accepts_eplb_when_world_sufficient` | `enable_eplb=True` with `dp_local=2, tp=2` passes. |
| 22 | `test_rejects_devices_not_divisible_by_per_replica` | `devices="0,1,2,3,4,5,6"` (7) with `tp=2, dp_local=2` raises `"not divisible by per-replica"`. |
| 23 | `test_accepts_devices_divisible_by_per_replica` | 8 devices with per-replica=4 passes. |
| 24 | `test_validator_only_writes_data_parallel_size` | Regression guard: validator must not add any key other than `data_parallel_size` (prevents the integration-smoke `unexpected keyword 'parallel_config'` / `'api_server_count'` regressions). |
| 25 | `test_dit_stage_is_soft_pass` | DiT stage with a full `parallel_config:` dict — validator early-returns; dict unchanged. |
| 26 | `test_dit_stage_does_not_reject_legacy_fields` | DiT stage with LLM-only keys at the top level — no-op. |
| 27 | (covered above) | Wire-coverage for the validator's call site (runs **before** `filter_dataclass_kwargs`). |

### 2. Unit — `omni_replica_id` semantics (`tests/engine/test_omni_replica_id_semantics.py`, 3 cases)

Target: `OmniCoreEngineProcManager` slot assignment in
`vllm_omni/engine/omni_core_engine_proc_manager.py`.

Parametrized over `dp_local ∈ {1, 2, 4}`:

- Spawn `dp_local` `StageEngineCoreProc` subprocesses for one replica
  with `omni_replica_base_id = R`.
- Assert every spawned subprocess sees `omni_replica_id == R` (no `R+1`,
  `R+2`, …).
- Assert `OmniCoordinator.ReplicaList` reports exactly **one** entry
  for that replica regardless of `dp_local`.

Regression target: pre-fix code at `omni_core_engine_proc_manager.py:117`
assigned `omni_replica_id = omni_replica_base_id + index`, which leaked
`dp_local - 1` phantom slots into the `ReplicaList` that `StagePool`
could not distinguish from real replicas.

### 3. Unit — CLI rename regression (`tests/entrypoints/test_serve.py`, +4/-4)

Existing CLI / argparse tests are updated to use the new kwarg name
(`omni_num_replica`). The deprecated alias `--omni-dp-size-local` still
parses to the same `dest`. There is **no** parallelism CLI prohibition
list to test — it was dropped (see the [hazard
section](#supporting-parallelism-cli-arguments--possible-but-hazardous-today));
the YAML validator is the authoritative gate.

### 4. Integration smoke (manual, multi-stage cluster)

Run on the two-server test cluster
(`47.79.124.13:30913` head + `47.79.124.13:31050` worker) against a
real model:

| Scenario | Setup | Pass criterion |
|---|---|---|
| **S1. Single replica, dp_local=1, no EP** | `--omni-num-replica 1`, YAML `tensor_parallel_size: 2` | Server reaches `/v1/models`; one OAI completion round-trips end-to-end. |
| **S2. Single replica, dp_local=2** | `--omni-num-replica 1`, YAML `data_parallel_size: 2, tensor_parallel_size: 2` | One `DPCoordinator`, two `StageEngineCoreProc` subprocs, one `omni_replica_id`, one entry in `ReplicaList`. Inner LB (`DPLBAsyncMPClient`) routes by `waiting × 4 + running`. |
| **S3. Two replicas, dp_local=1** | `--omni-num-replica 2`, YAML `tensor_parallel_size: 2` | Two independent NCCL worlds; `StagePool.LoadBalancer` distributes across `omni_replica_id ∈ {0, 1}`; per-replica restart works. |
| **S4. Two replicas × dp_local=2 (the full matrix)** | `--omni-num-replica 2`, YAML `data_parallel_size: 2, tensor_parallel_size: 2` | Four `StageEngineCoreProc` subprocs total, two `omni_replica_id`s, `OmniCoordClientForStage` is created on DP rank 0 of each replica only. |
| **S5. Per-replica EP** | S4 + YAML `enable_expert_parallel: true, enable_eplb: true, all2all_backend: deepep_low_latency` | EP group size = `dp_local × pcp × tp = 2 × 1 × 2 = 4` per replica; EPLB rebalances on the per-replica window; no cross-replica all2all. |
| **S6. Deprecation alias** | Launch with `--omni-dp-size-local 2` | One-shot `WARNING` logged once; replica count parsed as 2. |
| **S7. Parallelism CLI flag is not silently honored** | Launch with `--omni --tensor-parallel-size 4` while YAML stage 0 sets `tensor_parallel_size: 2` | Document observed behavior: the YAML value is what runs (CLI flag does not silently win). Confirms the hazard write-up; no prohibition is asserted. |
| **S8. Nested `parallel_config:` on LLM stage** | YAML places `parallel_config: { tensor_parallel_size: 2 }` on an LLM stage | Validator raises with `"nested parallel_config: block is for DiT"` before `filter_dataclass_kwargs` silently strips it. |
| **S9. DiT stage soft-pass** | Diffusion stage YAML with nested `parallel_config:` (existing shape) | Validator early-returns; DiT runtime starts and serves. |
| **S10. Device-count mismatch** | YAML `devices: "0,1,2"` (3) with `tp=2, dp_local=2` (per-replica=4) | Validator raises with `"not divisible by per-replica"` before any subprocess is spawned. |

### 5. Regression — existing unit suite

`108/108` existing unit tests under `tests/` must continue to pass after
the rename + validator + slot-fix patch. No tolerance for flake or
skip on this set.

### 6. What is **not** tested (deliberately)

- **CLI prohibition argparse-level rejection.** Not applicable — this
  RFC ships no parallelism CLI prohibition list (it was dropped in favor
  of documenting the underlying hazard). The YAML validator is the
  authoritative gate and is fully covered.
- **Cross-replica EP (3a).** Out of scope; the design explicitly
  forbids it.
- **Ray DP backend, external LB, hybrid LB, elastic EP.** Validator
  rejects these statically; no runtime test needed.

### 7. Test infrastructure

- Validator + slot-semantics suites run on CPU (`pytest.mark.cpu`,
  `pytest.mark.core_model`) — no GPU needed for the 30 new cases.
- Integration smoke (S1–S10) runs on the test cluster against a real
  model; checked in manually as part of the PR review, not in CI.
- All tests use the standard `pytest` runner from `tests/`; no new
  fixtures, no new conftest changes.

## Companion documents

- [omni-replica-dp-ep-analysis.md](./omni-replica-dp-ep-analysis.md) —
  feasibility analysis: 3a (EP spans replicas) vs 3b (EP per replica),
  LB-model constraint, DiT-scope clarification.
- [omni-replica-dp-ep-implementation.md](./omni-replica-dp-ep-implementation.md) —
  full implementation design with diagrams, validator pseudo-code, and
  post-smoke / post-review revision notes.

## Feedback Period

2 weeks.

## CC List

@vllm-project/vllm-omni-maintainers

## Any Other Things

This RFC's branch is `replica_dp_ep` on
`chickeyton/vllm-omni-replica-dp-ep`. Total change against `main`:
~180 net LoC of production code (7 files, no new files; the parallelism
CLI prohibition list was dropped), ~356 net LoC of tests (3 files, 2 new),
plus design docs.

## Before submitting a new issue...

- [x] Read the [documentation](https://github.com/vllm-project/vllm-omni)
      and existing RFCs / issues.
- [x] No duplicate RFC for this feature exists.
- [x] Companion analysis + implementation design docs included.






## Cli arguments overriding

We must support users specifing the follow cli arguemtns to overrides the parallel configs:

```
# TP / PP
--tensor-parallel-size, --pipeline-parallel-size
# DP
--data-parallel-size,
# Context parallel
--prefill-context-parallel-size, --decode-context-parallel-size,
# EP / EPLB
--enable-expert-parallel, --enable-eplb, 
--expert-placement-strategy, --all2all-backend,
--eplb-config.*,
# DiT (vllm-omni own)
--ulysses-degree,
--ring-degree,
--cfg-parallel-size
```

However, since there 2 sources of truth, one from YAML and one from cli argument, that will cause 
some problem with current implementations, consider the following scenerio:

An operator wants to run two stage-0 replicas on two hosts with
different per-host parallelism, plus a stage-1 receiver:

```bash
# Host A — stage 0, replica 0
vllm serve <model> --omni --stage-id 0 --tensor-parallel-size 2 ...

# Host B — stage 0, replica 1 (different TP from host A)
vllm serve <model> --omni --stage-id 0 --tensor-parallel-size 4 --headless ...

# Some host — stage 1
vllm serve <model> --omni --stage-id 1 --headless ...
```

The stage 1 has no idea what rank to use when pulling KV or other data from the connector, since:

Stage 1 have no information about the --tensor-parallel-size of stage 0 cli runtimes
The --tensor-parallel-size of stage 0 replica is unknown to stage 1 replica




## Configuration surface — full prohibition list

Prohibited CLI flags under `--omni` (configure in YAML instead):



```
# TP / PP
--tensor-parallel-size, --pipeline-parallel-size
# DP
--data-parallel-size, --data-parallel-size-local,
--data-parallel-rank, --data-parallel-rank-local,
--data-parallel-address, --data-parallel-rpc-port,
--data-parallel-master-port, --data-parallel-backend,
--data-parallel-external-lb, --data-parallel-hybrid-lb,
--data-parallel-start-rank, --api-server-count
# Context parallel
--prefill-context-parallel-size, --decode-context-parallel-size,
--dcp-comm-backend
# EP / EPLB
--enable-expert-parallel, --enable-ep-weight-filter,
--enable-eplb, --num-redundant-experts,
--expert-placement-strategy, --all2all-backend,
--enable-elastic-ep, --eplb-config.*
# Microbatching
--enable-dbo, --ubatch-size
```

