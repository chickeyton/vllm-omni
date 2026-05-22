# Omni-Replica + vLLM-DP + Expert-Parallel — Feasibility Analysis

Date: 2026-05-21
Branch: `replica_dp_ep`
Baseline: PR [#3569](https://github.com/vllm-project/vllm-omni/pull/3569) merged on 2026-05-19 (commit `e277feac`)
vLLM source: `D:\repo\github\vllm\vllm`

> **Revision note (review pass):** Updated Point 2 after re-reading the
> code. The headless path at `serve.py:951-955` already consumes
> `parallel_config.data_parallel_size_local` as `local_engine_count`,
> `OmniCoreEngineProcManager` already detects intra-replica DP at
> `omni_core_engine_proc_manager.py:107`, `make_async_mp_client`
> already returns `DPLBStageEngineCoreClient` when DP>1, and
> `launch_omni_core_engines` already spawns a per-replica
> `DPCoordinator`. The CLI prohibition is the user-facing blocker;
> the YAML route via `engine_args.parallel_config` partially works
> today. Added an `omni_replica_id` accounting issue surfaced by the
> code re-read, corrected the `external_lb` framing (we run vLLM in
> *internal-LB* mode per replica), and tightened the 3a infeasibility
> argument (collapsing the outer loop into one
> `OmniCoreEngineProcManager` eliminates the replica abstraction
> entirely).
>
> **Revision note (LB-model pass):** Elevated the LB-model constraint
> to a top-level "Load-balancing model" section. The design commits to
> exactly two LB layers: vllm-omni's `StagePool.LoadBalancer` across
> replicas, and vLLM's internal `DPLBAsyncMPClient` within each
> replica's DP cluster. vLLM's `data_parallel_external_lb` and
> `data_parallel_hybrid_lb` modes are unsupported and must be rejected
> at **both the CLI and the YAML layer** — CLI-only rejection isn't
> enough since the YAML's `engine_args.parallel_config` can supply
> either today.
>
> **Revision note (DiT-scope pass):** Added an explicit "Scope: LLM
> stages only" section clarifying that Points 2 and 3 do not affect
> DiT stages. DiT runs on a separate `StageDiffusionProc` runtime
> with its own `DiffusionParallelConfig`; `--data-parallel-size-local`
> is silently ignored on the DiT path, and EP is already supported
> for DiT today via YAML `parallel_config.enable_expert_parallel`.
> Both CLI flags should emit a warning (not error) when the self-stage
> is a DiT stage. One narrow edge case: the no-YAML
> `_create_default_diffusion_stage_cfg` path *does* read
> `--enable-expert-parallel` into `DiffusionParallelConfig`
> (`async_omni_engine.py:1881`); document as an edge case, still emit
> the warning. Also: DiT's per-replica EP size is
> `tp × sp × cfg × dp` per PP stage
> (`vllm_omni/diffusion/distributed/parallel_state.py:193`), NOT just
> TP — typically much larger than the LLM-side `dp_local × pcp × tp`.
>
> **Revision note (YAML-only pass, final):** The original Point 2
> proposal to *un-prohibit* `--data-parallel-size-local` is withdrawn.
> Replaced the narrow "DP/EP flags" prohibition with a blanket rule:
> **every vLLM intra-replica parallelism CLI flag is prohibited under
> `--omni`** (TP, PP, DP, PCP, DCP, EP, EPLB, microbatching, all2all
> backend, expert placement, elastic-EP, `--api-server-count`). All
> intra-replica parallelism is configured in
> `engine_args.parallel_config` in the deploy YAML, for both LLM and
> DiT stages. The CLI keeps exactly one scaling knob:
> `--omni-num-replica`. Added a new top-level "Configuration
> surface" section, the complete CLI prohibition list, and the YAML
> validator's enforcement rules. The earlier DiT silent-inert warnings
> go away — there's no flag to be silently ignored.
>
> **Revision note (flag-rename pass):** Dropped the `-local` suffix
> *and* the plural-`s`, landing on `--omni-num-replica`. The
> intermediate name (`--omni-num-replicas-local` → `--omni-num-replicas`)
> never shipped — final name is `--omni-num-replica`. Each
> `vllm serve --omni` invocation owns exactly one process slice, so
> "local" was redundant and invited confusion with vLLM's
> `--data-parallel-size-local` semantics; "replica" (singular) matches
> the grammar of every other singular vllm-omni flag (`--stage-id`,
> `--replica-id`). Per-process semantics are unchanged. Keep
> `--omni-dp-size-local` as a deprecated alias for ≥ 1 release.

---

## 1. Background — what PR #3569 already does

PR #3569 introduced what we currently call *"vllm-omni data parallel"*: a
process-local *replica* fan-out of a single stage, governed by
`--omni-dp-size-local`.

Concretely:

- `OmniMasterServer` pre-allocates one `(handshake, input, output)` ZMQ
  triple per `(stage_id, replica_id)`
  (`vllm_omni/engine/stage_engine_startup.py:131-180`,
  `:204-237`).
- `OmniCoordinator` is a ZMQ ROUTER/PUB service that tracks each replica's
  UP/DOWN/ERROR status, queue length and heartbeats, and publishes the
  current `ReplicaList` to head-side hubs
  (`vllm_omni/distributed/omni_coordinator/omni_coordinator.py:19-106`).
- `StagePool` consults that replica list and dispatches requests via
  `LoadBalancer` (`vllm_omni/engine/stage_pool.py:44-150`).
- Replicas are launched either as **head-owned** (in `AsyncOmniEngine` via
  `launch_omni_core_engines` — `async_omni_engine.py:892-940`) or
  **headless / remote** via `vllm serve --omni ... --headless --stage-id ...
  --omni-dp-size-local N` (`vllm_omni/entrypoints/cli/serve.py:687-1099`).
- Each replica is split off the YAML's `runtime.devices:` with
  `split_devices_for_replicas`
  (`vllm_omni/engine/stage_init_utils.py:471-620`) and gets its own
  narrowed `CUDA_VISIBLE_DEVICES` before its subprocess is spawned.

Critically, **replicas are independent**: each replica has its own
`StageEngineCoreProc` subprocess(es), its own NCCL/torch.distributed
world, its own ZMQ triple. The OmniCoordinator only carries control-plane
gossip (status, queue length). There is no collective comm between
replicas, no shared wave clock, no inter-replica all-reduce.

To prevent silent disagreement with vLLM's own DP/EP machinery, the head
already *rejects* a subset of vLLM DP/EP CLI flags when `--omni` is set
(`vllm_omni/entrypoints/cli/serve.py:128-147`):

```
--data-parallel-size, --data-parallel-size-local,
--data-parallel-address, --data-parallel-rpc-port,
--data-parallel-start-rank, --data-parallel-backend,
--api-server-count, --enable-expert-parallel
```

**The current prohibition is partial and at the CLI layer only.** The
deploy YAML's `engine_args.parallel_config` block can already supply
`data_parallel_size`, `data_parallel_size_local`, and
`enable_expert_parallel` — and the headless path at
`serve.py:951-955` already reads `parallel_config.data_parallel_size_local`
out of the built `vllm_config` and uses it as `local_engine_count`.

The proposal below tightens this: **every intra-replica parallelism
knob must be specified in the deploy YAML, never on the CLI**, for both
LLM and DiT stages. The CLI keeps exactly one scaling knob —
`--omni-num-replica` — that overrides the YAML's
`runtime.num_replicas`. See "Configuration surface" below for the
complete enforcement list.

### Configuration surface (load-bearing constraint)

The design commits to a single, easy-to-state rule:

> **All intra-replica parallelism is configured in the deploy YAML.
> No CLI override exists. The CLI exposes exactly one scaling knob:
> `--omni-num-replica`, which overrides the YAML's
> `runtime.num_replicas` for this process.**

This rule applies symmetrically to LLM and DiT stages and covers every
parallelism axis vLLM (or vllm-omni's diffusion runtime) understands.
Removing the CLI surface for these knobs:

- eliminates the "silently inert for the other stage type" foot-gun
  that already bit `--enable-expert-parallel` and
  `--data-parallel-size-local` (see DiT-scope discussion further down);
- keeps a single source of truth, since `devices:` and `num_replicas`
  must be consistent with the parallelism shape, and `devices:` is
  YAML-only anyway;
- avoids the per-stage-routing problem (a CLI flag has only one value,
  so a multi-stage runtime cannot use it to express different shapes
  per stage);
- matches how TP/PP/EP for LLM stages and *every* DiT parallelism axis
  are already configured today.

#### Full prohibition list (CLI) — `--omni` mode

The following vLLM CLI flags must be rejected at parse time
(extension of the existing `prohibited_with_omni` dict in
`serve.py:128-147`). All are sourced from `vllm/config/parallel.py`.

**Tensor / pipeline:**
`--tensor-parallel-size` (`-tp`), `--pipeline-parallel-size` (`-pp`).

**Data parallel (every field):**
`--data-parallel-size`, `--data-parallel-size-local`,
`--data-parallel-rank`, `--data-parallel-rank-local`,
`--data-parallel-address`, `--data-parallel-rpc-port`,
`--data-parallel-master-port`, `--data-parallel-backend`,
`--data-parallel-external-lb`, `--data-parallel-hybrid-lb`,
`--api-server-count`.

**Context parallel:**
`--prefill-context-parallel-size`, `--decode-context-parallel-size`,
`--dcp-comm-backend`.

**Expert parallel + EPLB:**
`--enable-expert-parallel`, `--enable-ep-weight-filter`,
`--enable-eplb`, `--num-redundant-experts`,
`--expert-placement-strategy`, `--all2all-backend`,
`--enable-elastic-ep`, and any `--eplb-config.*` nested key
(`--eplb-window-size`, `--eplb-step-interval`,
`--eplb-log-balancedness`, `--eplb-policy-type`, etc.).

**Microbatching / DBO:**
`--enable-dbo`, `--ubatch-size`.

If any of these appears on the command line under `--omni`, error out
with a single message pointing the user at the YAML field they should
use instead.

#### YAML enforcement (LLM and DiT)

A new validator in `build_engine_args_dict` / `build_vllm_config`
(`stage_init_utils.py:638-715`) enforces these YAML constraints:

**LLM stages (`engine_args.parallel_config`, an LLM `ParallelConfig`):**

- `data_parallel_size` — if specified, must equal `data_parallel_size_local`. vllm-omni force-sets it internally; specifying a different value is an error (no cross-replica DP).
- `data_parallel_external_lb` and `data_parallel_hybrid_lb` — must be unset or `False`. See "Load-balancing model" below.
- `data_parallel_backend` — must be `"mp"` (or unset).
- `data_parallel_address`, `data_parallel_rpc_port`,
  `data_parallel_master_port`, `data_parallel_rank`,
  `data_parallel_rank_local` — must NOT appear; vllm-omni auto-assigns per replica from its central `OmniPortRegistry`.
- `api_server_count` — must NOT appear; **force-set to `0`** by vllm-omni (not "auto-assigned" — there is no vLLM-started API server). The vllm-omni HTTP server lives in the head process (`vllm_omni/entrypoints/openai/api_server.py:343` — `omni_run_server`), and vLLM's `run_server`/`run_multi_api_server`/`run_headless` (`vllm/entrypoints/cli/serve.py:115-122`) are never invoked under `--omni`. Setting it to `0` rather than just letting it default also keeps the multimodal cache-size formula (`vllm/config/multimodal.py:129`: `mm_processor_cache_gb * (api_server_count + data_parallel_size)`) honest, and prevents any latent dependence on vLLM's `DPLBAsyncMPClient.client_count` tie-breaking.
- `enable_elastic_ep` — must be unset or `False`. Elastic EP is
  incompatible with vllm-omni's per-replica torch.distributed world
  (`vllm/config/parallel.py:780-784` already forbids it with
  external/hybrid LB; vllm-omni's outer LB is structurally
  equivalent).

**DiT stages (`engine_args.parallel_config`, a `DiffusionParallelConfig`):**

No vLLM CLI flags exist for DiT-specific axes (`ulysses_degree`,
`ring_degree`, `cfg_parallel_size`, `vae_patch_parallel_size`,
`use_hsdp`, `hsdp_shard_size`, `hsdp_replicate_size`), so the
prohibition list above is vacuous for them — they were already
YAML-only. The rule "all intra-replica parallelism in YAML" just
inherits the status quo on the DiT side.

**Both stage types:**

- The product of all parallelism axes for a stage must equal
  `len(devices.split(","))`. Validate at YAML load.
- For LLM: `dp_local × tp × pp × pcp × num_replicas` ≤
  `len(devices)`.
- For DiT: `pp × dp × tp × ulysses × ring × cfg × num_replicas`
  ≤ `len(devices)`. (See `diffusion/data.py:170-178` for the existing
  `world_size` formula.)

#### What stays on the CLI

| CLI flag | Role |
|---|---|
| `--omni-num-replica` | Per-process omni replica count (overrides YAML `runtime.num_replicas`). Required because head and headless invocations on different hosts must each pick their own value. |
| `--omni-lb-policy`, `--omni-heartbeat-timeout`, `--omni-master-address`, `--omni-master-port`, `--omni-replica-address` | Coordinator/transport — orthogonal to parallelism. Unchanged from PR #3569. |
| `--stage-id`, `--stage-init-timeout`, `--init-timeout`, `--headless`, `--deploy-config`, `--stage-configs-path`, `--stage-overrides`, `--async-chunk` | Stage selection / launch mode. Unchanged from PR #3569. |
| `--replica-id` | Already deprecated and ignored as of PR #3569 (`serve.py:248-256`). |

### Scope: LLM stages only (DiT stages excluded from Points 2 & 3)

Points 2 and 3 apply to **LLM stages only**. DiT (diffusion) stages run
on a completely separate runtime — `spawn_diffusion_proc` →
`StageDiffusionProc` (`serve.py:807-906`) — with their own
torch.distributed world and their own `DiffusionParallelConfig`
(`vllm_omni/diffusion/data.py:85-178`) whose axes are
`tensor_parallel_size`, `data_parallel_size`, `sequence_parallel_size`
(= `ulysses_degree × ring_degree`), `cfg_parallel_size`,
`vae_patch_parallel_size`, `pipeline_parallel_size`, `use_hsdp` +
`hsdp_shard_size` / `hsdp_replicate_size`, and an independent
`enable_expert_parallel`. `world_size` is the product of all of these.

Concretely (post-YAML-only design):

| Knob | LLM stage | DiT stage |
|---|---|---|
| `--omni-num-replica` (Point 1) | applies (CLI) | applies (CLI) |
| Intra-replica DP | **YAML only** — `engine_args.parallel_config.data_parallel_size_local` | **YAML only** — `engine_args.parallel_config.data_parallel_size` (a different field; FSDP/HSDP replicate or batch axis, not an EngineCore DP cluster) |
| Intra-replica TP, PP | **YAML only** — `parallel_config.{tensor,pipeline}_parallel_size` | **YAML only** — same field names |
| Intra-replica EP | **YAML only** — `parallel_config.enable_expert_parallel`, `enable_eplb`, `eplb_config`, etc. | **YAML only** — `parallel_config.enable_expert_parallel` (already supported today; `deploy/hunyuan_image3.yaml:73-75`) |
| EP group size formula | `dp_local × pcp × tp` per PP stage (`vllm/distributed/parallel_state.py:1674-1683`) | `tp × sp × cfg × dp` per PP stage where `sp = ulysses × ring` (`vllm_omni/diffusion/distributed/parallel_state.py:193` + `:256-263`). **EP is NOT just TP** — it spans the full DiT world minus PP. |
| vLLM `DPCoordinator`, `DPLBAsyncMPClient`, EPLB | LLM only | not used |

The diffusion headless loop (`serve.py:807-906`) iterates only over
`omni_dp_size_local` and spawns one `StageDiffusionProc` per outer
replica, sizing devices off `DiffusionParallelConfig.world_size` (via
`get_stage_devices_per_replica` —
`stage_init_utils.py:540-559`). It never reads
`parallel_config.data_parallel_size_local`. DiT "intra-replica DP" is
expressed by `DiffusionParallelConfig.data_parallel_size` (an FSDP/HSDP
replicate axis or batch-dim split — *not* a wave-coordinated DP cluster
of EngineCores) and is configured only through YAML.

**DiT EP-size note.** A DiT EP group already spans much more than TP
— it's `tp × sp × cfg × dp` per PP stage (`vllm_omni/diffusion/distributed/parallel_state.py:193`),
where `sp = ulysses × ring` and `dp` is `DiffusionParallelConfig.data_parallel_size`.
So a single DiT replica with `(tp=2, ulysses=2, cfg=2, dp=2)` already
has `ep_size=16` without any cross-replica coordination — the
per-replica EP cap for DiT is comfortably higher than for LLM stages,
where `ep_size = dp_local × pcp × tp` (and most LLM deployments run
`pcp=1`, collapsing to `dp_local × tp`). Point 3b's per-replica
EP-bound constraint is therefore much less limiting for DiT than for
LLM. (`pcp` = vLLM's `prefill_context_parallel_size`, group `_PCP` —
prefill-time context/sequence parallelism. See `vllm/distributed/parallel_state.py:1285-1290`.)

A mixed-stage runtime (e.g., text-encoder LLM + DiT + VAE) will see
`--data-parallel-size-local` and `--enable-expert-parallel` apply to
the LLM stages and pass through the DiT stages unchanged. **Under the
YAML-only rule there is no DiT-only warning to emit** — every
intra-replica parallelism flag is rejected at the CLI for both stage
types uniformly. The earlier "warn if `--enable-expert-parallel` is set
against a DiT self-stage" wart is gone because the flag is no longer
accepted at all.

(The no-YAML default-diffusion edge case at
`async_omni_engine._create_default_diffusion_stage_cfg` becomes a non-issue
in the same stroke: if the CLI flag never reaches `normalized_kwargs`,
the `_create_default_diffusion_stage_cfg` path cannot pick it up either.)

### Load-balancing model

This is a corollary of the YAML-only rule above; the relevant
prohibitions are already in the "Configuration surface" section. Here
we just spell out what the resulting LB topology looks like.

The design commits to **exactly two LB layers**, with no third option:

1. **Outer LB — across omni replicas: vllm-omni's `StagePool.LoadBalancer`
   only** (`vllm_omni/engine/stage_pool.py:107-112`). Policies are
   `random` / `round-robin` / `least-queue-length`, fed by
   OmniCoordinator's `ReplicaList` (`omni_coordinator.py:19-106`).
2. **Inner LB — across DP ranks within one replica: vLLM's internal
   DP-LB only** (`DPLBStageEngineCoreClient` ← `DPLBAsyncMPClient`,
   selected at `stage_engine_core_client.py:112-115`). The algorithm is
   weighted shortest-queue (`waiting × 4 + running`), fed by the
   per-replica `DPCoordinator`'s queue-length stats — see
   `vllm/v1/engine/core_client.py:1350-1378`.

vLLM's two alternative LB modes — `data_parallel_external_lb=True`
and `data_parallel_hybrid_lb=True` — are unsupported and rejected
both at the CLI and the YAML layer (see "Configuration surface"
above). Every per-replica vLLM world runs in vLLM's default
(internal-LB) DP mode, period.

`AsyncOmniEngine._validate_single_stage_mode_replica_constraints`
overwrites the YAML's `runtime.num_replicas` with `--omni-dp-size-local`
for the self-stage (`async_omni_engine.py:472-509`).

The user-facing problem the new proposal addresses: **most useful EP
deployments require DP > 1.** Pedantically EP works without DP — the
EP group is sized `dp × pcp × tp` (`vllm/distributed/parallel_state.py:1655-1696`),
so with `dp=1, pcp=1` you get `ep_size = tp`, i.e. EP collapses onto
the existing TP group. EPLB is stricter and explicitly errors unless
`dp * tp > 1` (`vllm/config/parallel.py:454-458`). The deployments
people actually want EP for ("wide EP", `ep_size ≫ tp`) all require
DP > 1, so as long as we forbid vLLM-DP we effectively lose the
practically useful EP and lose all EPLB.

---

## 2. The three proposed points

### Point 1 — Rename "omni data parallel" → "omni replica"

> *Higher-level concept on top of vLLM data parallel; vLLM DP runs within
> each replica; backend mandatory `mp`; rename
> `--omni-dp-size-local` → `--omni-num-replica` (singular, no `-local`
> suffix), keep the old name as a deprecated alias.*

#### Feasibility
**High.** The change is mostly nomenclature. Internally the symbol is
already called `_omni_dp_size_local` and is consumed in only ~4 places
(`async_omni_engine.py:312-321`, `serve.py:115-130, 248-264, 758-771`),
plus user-facing docs and YAML examples. The runtime model already
matches "replica" semantics: independent ZMQ triples, independent device
slices, independent NCCL worlds. The PR's own commit message already
uses "replicas" — the word "data parallel" is the misleading part.

#### Limitations
- The flag is dropped the `-local` suffix on purpose: each
  `vllm serve --omni` invocation already owns exactly one process slice,
  so "local" is redundant. Internally the semantics are unchanged —
  head and headless invocations on different hosts still pick their own
  values; we just stop telling the user that fact through the flag
  name. argparse will derive `dest='omni_num_replica'` (singular)
  from the flag, so the internal symbol becomes `_omni_num_replica`
  on the field; PR #3569 named it
  `_omni_dp_size_local`, so a pure rename of the field is part of the
  same patch.
- The flag name `--omni-num-replica` shadows the YAML field
  `runtime.num_replicas` (note the singular vs. plural difference).
  Clarify in docs that the CLI flag is *the source of truth for this
  process's slice* and overwrites the YAML field at
  `_validate_single_stage_mode_replica_constraints`
  (`async_omni_engine.py:494-509`).
- Backwards compatibility: existing users have launchers wired to
  `--omni-dp-size-local`. Keep it as a deprecated alias for ≥ 1
  release. Argparse supports this via multiple option strings on the
  same `dest` — trivial.
- One subtle gotcha: there is also a deprecated `--replica-id`
  (`serve.py:248-256`, warned-and-ignored). The two together form a
  confusing trio (`replica-id`, `num-replicas`, plus
  `runtime.num_replicas` in YAML). Plan to clearly document the hierarchy
  in the deprecation note.

#### System complexity
**Negligible delta.** Pure rename plus alias. Touches CLI argparse,
`AsyncOmniEngine.__init__`, `OmniEngineArgs` (`engine/arg_utils.py:155-160`),
and docs/examples.

---

### Point 2 — Enable vLLM data parallel within each replica

> *Original proposal: enable `--data-parallel-size-local` and keep the
> rest prohibited.*
> *Revised design: keep **every** vLLM DP CLI flag prohibited
> (see "Configuration surface" — full list above). Intra-replica DP
> is configured exclusively through
> `engine_args.parallel_config.data_parallel_size_local` in the deploy
> YAML; vllm-omni force-sets `data_parallel_size = data_parallel_size_local`
> and auto-assigns the address/port/rank/backend fields per replica.*

#### Feasibility
**High at the plumbing level; medium-high once you account for the
known wiring inconsistencies.**

Re-reading the code, inner DP is *already partly wired through*:

- `OmniCoreEngineProcManager` (`omni_core_engine_proc_manager.py:107`)
  computes `has_intra_replica_dp = parallel_config.data_parallel_size > 1`
  and spawns `local_engine_count` engine subprocesses per call.
- `launch_omni_core_engines` / `connect_remote_engine_cores`
  (`stage_engine_startup.py:803-807`, `:863-867`) already derive
  `local_engine_count` from `parallel_config.data_parallel_size_local`.
- The head's `make_async_mp_client`
  (`vllm_omni/engine/stage_engine_core_client.py:96-115`) already
  selects `DPLBStageEngineCoreClient` (= vLLM's `DPLBAsyncMPClient`)
  when `data_parallel_size > 1 and not data_parallel_external_lb`.
- vLLM's `DPCoordinator` is conditionally spawned per omni replica
  (`stage_engine_startup.py:876-892`,
  `vllm_omni/entrypoints/cli/serve.py:984-993`) — currently a no-op
  when `data_parallel_size == 1`, but the plumbing is in place.

So Point 2 is less "add inner DP" and more "lift the CLI prohibition
and resolve the wiring inconsistencies that surface once
`data_parallel_size_local > 1` is actually used".

The wiring inconsistencies, all surfaced by re-reading the diff:

1. **omni_replica_id accounting:** `OmniCoreEngineProcManager`
   (`omni_core_engine_proc_manager.py:117`) sets
   `omni_replica_id = omni_replica_base_id + index` for each spawned
   DP-rank subprocess. The launch loop in `serve.py:1023-1067` and
   `async_omni_engine.py` calls `register_stage_with_omni_master` once
   per *outer* omni replica, getting back one auto-assigned
   `response.replica_id`. With `data_parallel_size_local > 1`, ranks
   1..N-1 of the inner DP group end up with `omni_replica_id` values
   that are NEVER registered with `OmniMasterServer`.
2. **OmniCoordinator reporting granularity:** the per-subprocess
   `OmniCoordClientForStage` (`stage_engine_core_proc.py:102-119`) sends
   `(stage_id, input_addr, output_addr, queue_length, status)`. The
   `omni_core_engine_proc_manager.py` docstring asserts "DP ranks share
   one engine, one DPCoordinator, one set of weights" — but the per-rank
   coordinator reporting means OmniCoordinator currently sees N×DP
   entries per stage, not N. The head's `StagePool` / `LoadBalancer`
   would route to a DP-rank-level granularity, conflicting with the
   `DPLBStageEngineCoreClient` that wraps DP and already does its own
   internal LB.
3. **`OmniMasterServer.stage_replica_counts` does not multiply by
   `data_parallel_size_local`** (`async_omni_engine.py:625-643` only
   counts `len(plan.replicas)`). If we keep "each DP rank is one omni
   replica slot", the pre-allocated slot count must be
   `num_replicas × data_parallel_size_local`. If we keep
   "one slot per omni replica, DP ranks share it", we have to change
   the omni_replica_id math.

Either resolution is fine but it must be chosen explicitly. The cleaner
one is **"DP ranks share one omni replica slot"**: OmniCoordinator's
queue length comes from the DP rank with the smallest queue (or the
shared coordinator's view), only the rank-0 DP process registers an
`OmniCoordClientForStage`, and `omni_replica_id` is single-valued per
manager rather than `base + index`. This aligns the runtime with the
docstring at `omni_core_engine_proc_manager.py:102-106`.

Beyond that, the user constraints translate to:

- Backend `mp` is the only one we support (per user). Ray DP would
  add a process-management layer the omni replica model does not need.
- The remaining vLLM DP flags (`--data-parallel-size`,
  `--data-parallel-address`, `--data-parallel-rpc-port`,
  `--data-parallel-start-rank`, `--data-parallel-backend`,
  `--api-server-count`) stay prohibited; vllm-omni auto-assigns them
  per-replica.

**Fields auto-assigned by vllm-omni** (none of these are user-facing
under `--omni`; the YAML validator rejects any user-supplied value):

| `ParallelConfig` field | Source of truth in vllm-omni |
|---|---|
| `data_parallel_size` | Force-set equal to YAML's `data_parallel_size_local` (no cross-replica DP) |
| `data_parallel_master_ip` | Replica's host (default `127.0.0.1` for same-host launches; `--omni-replica-address` for cross-host) |
| `data_parallel_rpc_port` / `data_parallel_master_port` | Auto-picked per-replica from the central `OmniPortRegistry` (must NOT collide with the master server's pre-allocated ZMQ ports nor with sibling replicas' RPC ports) |
| `data_parallel_rank` / `data_parallel_rank_local` | `0` (each replica is its own DP world) |
| `data_parallel_backend` | `"mp"` (forced) |
| `data_parallel_external_lb` / `data_parallel_hybrid_lb` | Forced `False` (see "Load-balancing model" below) |
| `api_server_count` | **Force-set to `0` for every replica** (head and headless). The HTTP server is vllm-omni's `omni_run_server`; vLLM never starts an API server under `--omni`. |

#### Limitations

1. **Port allocation is the main hazard.** The replica's
   `OmniMasterServer` already pre-allocates ZMQ ports in the ephemeral
   range; PR #3569 already had to push diffusion `master_port` to
   `>= 61000` to avoid clashing (`serve.py:849-859`). Adding a vLLM
   `DPCoordinator` per replica adds **more** sockets. We need a single
   port-allocation strategy that hands out non-overlapping ranges to:
   - OmniMasterServer's `(handshake, input, output)` ports per replica
   - OmniCoordinator ROUTER/PUB
   - vLLM `DPCoordinator` front_publish / back_publish / output_publish
   - vLLM DP init port (`get_next_dp_init_port`)
   - Diffusion torch-dist `master_port`

   A small `PortRegistry` helper (one per process) is the cleanest fix.

2. **Device split changes shape.** `split_devices_for_replicas` currently
   assumes each replica owns `tp_size` (or `world_size` for diffusion)
   devices. With inner DP, each replica owns `dp_local × tp_size` devices.
   `get_stage_devices_per_replica` (`stage_init_utils.py`) must be
   updated to multiply by the configured DP. We also need to ensure the
   YAML's `devices:` field exposes enough physical GPUs (validate at
   parse time, fail loudly).

3. **Wave coordination cost.** vLLM's `DPCoordinator` does
   `enable_wave_coordination=True` for MoE models
   (`async_omni_engine.py:_launch_core_engines`, mirrored in
   `serve.py:976-996`). Every wave transition involves an all-reduce
   across the inner-DP world. This is fine for a few ranks, but if a
   user picks (say) `omni-num-replica=4, data_parallel_size_local=8` (YAML)
   on a single 32-GPU box, you now have 4 inner-DP worlds each doing
   wave-sync independently, plus 4 inner DPCoordinator processes. CPU /
   process pressure goes up; we should document a recommended ceiling.

4. **No third LB mode is permitted** (cross-reference: "Load-balancing
   model" section above). vLLM's `data_parallel_external_lb` and
   `data_parallel_hybrid_lb` (`vllm/config/parallel.py:135-148`) must
   be rejected at *both* the CLI and the YAML layer. The only
   permitted topology is: vllm-omni's `StagePool.LoadBalancer` across
   replicas, vLLM's internal `DPLBAsyncMPClient` across DP ranks
   within a replica. The reason for the hard rejection is that
   `data_parallel_external_lb=True` swaps `DPLBAsyncMPClient` for
   `DPAsyncMPClient` (`vllm/v1/engine/core_client.py:124-130`), which
   pushes DP-rank selection up to the API server layer and
   double-routes against vllm-omni's outer LB. `data_parallel_hybrid_lb=True`
   presumes one API server per node plus an external HTTP LB across
   nodes — a topology vllm-omni's ZMQ-based StagePool routing does not
   model. Either is silently broken if allowed to slip through.

5. **API server count.** `--api-server-count` is forbidden by the
   existing prohibition list. With inner DP this is still correct: the
   head owns the single API server; headless replicas serve no HTTP.

6. **Test surface explodes.** Existing single-stage and multi-replica
   tests assume `dp_size == 1`. We need new integration tests with
   `(num_replicas=2, dp_size_local=2, tp=2)` shapes — at minimum
   one per `omni_coord_client_for_stage` / `test_async_omni_engine_stage_init`
   to exercise the full bring-up.

#### System complexity
**Moderate.** Most of the architectural plumbing is already in place;
the work is concentrated in:

- **CLI prohibition list** (`serve.py:128-147`): extend to cover the
  full set in "Configuration surface" above — TP/PP/DP/PCP/DCP/EP/EPLB/DBO
  and `--api-server-count`. The original Point 2 proposal to
  *un-prohibit* `--data-parallel-size-local` is **withdrawn**: keep it
  prohibited; configure DP via YAML only.
- **YAML validator** in `build_engine_args_dict` / `build_vllm_config`
  (`stage_init_utils.py:638-715`) that:
  - rejects `data_parallel_external_lb=True` / `data_parallel_hybrid_lb=True`;
  - rejects `data_parallel_size` if specified and != `data_parallel_size_local`;
  - rejects `data_parallel_backend != "mp"`;
  - rejects user-set `data_parallel_address` / `data_parallel_rpc_port` / `data_parallel_master_port` / `data_parallel_rank` / `data_parallel_rank_local` / `api_server_count`;
  - rejects `enable_elastic_ep=True`;
  - validates `len(devices.split(",")) == num_replicas × dp_local × tp × pp × pcp` (LLM) or the equivalent DiT formula.
- **`omni_replica_id` semantics**: pick one of the two resolutions
  described in the Feasibility section and apply it consistently
  across `OmniCoreEngineProcManager`, `StageEngineCoreProc`,
  `OmniMasterServer._next_free_replica_id`, and
  `OmniCoordClientForStage`.
- **`stage_replica_counts` accounting**: extend
  `async_omni_engine._start_omni_master_server`
  (`async_omni_engine.py:625-643`) to factor in DP-rank multiplicity
  if we keep "DP rank = omni replica slot"; otherwise leave as-is.
- **Per-replica `DPCoordinator`**: already implemented; becomes
  load-bearing once DP > 1. Validate against multiple replicas on
  one host (port collision via `data_parallel_rpc_port`).
- **Port hygiene**: introduce a central per-process `PortRegistry`
  that allocates non-overlapping ranges to OmniMasterServer's
  per-replica triples, OmniCoordinator's ROUTER/PUB,
  vLLM `DPCoordinator`'s front/back/output_publish, vLLM DP init
  port (`ParallelConfig.get_next_dp_init_port` —
  `vllm/config/parallel.py:509-524`), and diffusion torch-dist
  `master_port` (which currently has its own
  `61000 + rep_idx * 100` workaround at `serve.py:849-859`).
- **Device math + validation**: update
  `get_stage_devices_per_replica`
  (`stage_init_utils.py:540-559`) to multiply by
  `data_parallel_size_local`; update `split_devices_for_replicas`
  device-count assertions.
- **Docs + examples**: at least one new YAML
  (`deploy/<model>_replicas_with_inner_dp.yaml`).

Estimate: ~400-700 LoC of production code + ~500 LoC of tests,
mostly mechanical. Lower than my first estimate because the launch-path
plumbing is already done.

---

### Point 3 — Enable Expert Parallel and EPLB

> *Enable `--enable-expert-parallel` and all vLLM EPLB arguments.*

vLLM's EP group is sized `data_parallel_size × prefill_context_parallel_size ×
tensor_parallel_size` (`vllm/distributed/parallel_state.py:1655-1696`)
and **spans across DP ranks**. The all2all dispatch/combine uses the DP
group by default (`vllm/distributed/device_communicators/all2all.py:66,
101`). EPLB requires `dp * tp > 1` (`vllm/config/parallel.py:454-458`)
and runs its own collective on a separate EPLB group with the same rank
membership as EP.

This means EP fundamentally requires a torch.distributed world that
includes every rank that participates in the EP group. **Which world we
choose determines the entire system design.** That is the 3a vs 3b
question.

The proposal's two flavors:

- **3a** — EP group spans the whole CLI runtime (one `vllm serve --omni`
  invocation, or one `AsyncOmni`): `ep_size = num_replicas × dp_local × tp`.
- **3b** — EP group is per-replica: `ep_size = dp_local × tp`.

---

## 3. Comparison: 3a vs 3b

### 3a — EP group spans all replicas in the runtime

#### Feasibility
**Low.** Several hard blockers:

1. **Replicas are NOT in a shared torch.distributed world.** Each
   replica today is launched as an independent vLLM `EngineCore`
   subprocess (or DP group of subprocesses) that builds its own
   `WorldGroup` via `init_distributed_environment`
   (`vllm/distributed/parallel_state.py:1358`). Two replicas have two
   different masters (`data_parallel_master_ip/port`), two different
   worlds, and never see each other.
2. **The current launch loop already forks per-replica `DPCoordinator`
   processes.** `launch_omni_core_engines`
   (`stage_engine_startup.py:876-892`) spawns a `DPCoordinator` per
   omni replica, and the same is mirrored in
   `serve.py:984-993`. Each `OmniCoreEngineProcManager` invocation gets
   one of these and one independent DP world. To make EP span replicas,
   the outer loop in `serve.py:1023-1067` would have to be
   **collapsed**: one `OmniCoreEngineProcManager` with
   `local_engine_count = num_replicas × dp_local`, one
   `DPCoordinator`, one `parallel_config.data_parallel_size =
   num_replicas × dp_local`. That collapse *eliminates the omni
   replica abstraction itself* — `num_replicas` becomes a
   multiplier on the inner DP size, equivalent to setting
   `data_parallel_size_local = num_replicas × dp_local` directly.
   At that point we're just running vLLM-native DP with extra steps.
3. **To unify them as separate-but-coordinated worlds, we must build
   one giant world** spanning all replica processes. That means a
   *single* `data_parallel_master_ip/port` shared across every replica
   subprocess, a *single* `world_size = num_replicas × dp_local × tp`,
   and per-rank assignment of `data_parallel_rank` and
   `tensor_parallel_rank` that's consistent across the runtime — i.e.,
   the model vLLM already implements *without* the omni layer. The
   user explicitly says "rename omni-data-parallel to omni-replica
   because EP depends on DP", which signals they want to *layer above*
   vLLM DP, not replace it.
4. **Wave coordination collapses replicas back into lockstep.** For
   MoE models, vLLM enables wave coordination
   (`vllm_omni/engine/stage_engine_startup.py:881`,
   `serve.py:987`: `enable_wave_coordination=vllm_config.model_config.is_moe`).
   `DPEngineCoreProc._has_global_unfinished_reqs` does an all-reduce
   across all DP ranks every step
   (`vllm/v1/engine/coordinator.py:33-42`). If 3a's EP group includes
   all replicas, every replica must call the all-reduce every step —
   meaning a slow replica stalls every other replica, and a crashed
   replica deadlocks the runtime. This destroys the *independent
   failure domain* property that makes omni replicas useful in the
   first place.
5. **Cross-replica all2all bandwidth at scale.** With `num_replicas=N`,
   each MoE forward needs an all2all across `N × dp_local × tp` ranks.
   On a single host with NVLink this is fine. Across hosts (the use case
   the omni layer was *designed* for) it puts the entire MoE token
   exchange on the inter-host network every layer — exactly what
   per-replica isolation was meant to avoid. Note that `all2all`
   uses the DP group by default
   (`vllm/distributed/device_communicators/all2all.py:66, 101`:
   `dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()`).
6. **Headless / dynamic-attach incompatibility.** PR #3569 supports
   replicas registering *after* the head is up via the auto-assign
   path in `OmniMasterServer` (`stage_engine_startup.py:431-512`) and
   the `_build_remote_replica` factory (`async_omni_engine.py:673-803`).
   torch.distributed process groups in the standard NCCL backend are
   **not elastic**; you cannot add a rank to an existing world without
   tearing down and rebuilding the whole world. Elastic-EP exists
   (`vllm/distributed/parallel_state.py:1657-1664`, gated on
   `enable_elastic_ep`) but it requires a coord store, only supports a
   single API server, and is incompatible with external/hybrid LB
   (`vllm/config/parallel.py:780-784`) — i.e., incompatible with the
   omni replica model.

#### Limitations
- **Cross-host:** requires the EP master IP to be reachable from every
  headless host; one bad NIC or a transient network blip kills all
  replicas, not just one. This is fundamentally worse than 3b for
  multi-host serving — which is the dominant deployment target.
- **Static membership:** dynamic add/remove of replicas (already
  supported via `OmniCoordinator` in PR #3569) becomes impossible
  without elastic EP, and elastic EP forbids external LB.
- **Mixed-stage runtimes:** the analysis above assumes a single stage.
  In a multi-stage omni runtime (e.g. text-encoder + DiT + VAE),
  forming one EP world per CLI invocation is meaningless across
  stages — they don't share weights. So 3a would have to mean "one EP
  world per *stage* per *CLI runtime*", which is identical to 3b once
  the per-process replica count is treated as inner DP. The naming "per
  AsyncOmni" hides this — there's no architectural difference at the
  stage level.
- **vLLM-omni's existing "loose-coupling via ZMQ" model is the
  abstraction boundary.** Crossing it would require us to reimplement
  most of vLLM's DP-coordinator-and-launcher inside vllm-omni so the
  replicas' subprocesses agree on rank assignment, master IP, and wave
  state. This is double work, and worse, double-maintenance forever as
  vLLM evolves.

#### System complexity
**High.** Requires:

- A new "outer DP" rank-assignment service in vllm-omni that all
  headless launches consult.
- A way for headless launchers to receive `(global_rank, world_size,
  master_ip, master_port)` before each subprocess starts NCCL.
- Replacement or wrapping of vLLM's `DPCoordinator` to run *across*
  replicas instead of within one.
- Rewrite of `OmniCoordinator` from a status-publishing service to a
  control-plane service that participates in wave synchronization.
- Elastic-EP integration (or accept static replica membership).
- All of point 2's work, *plus* the above.

#### Code-change estimate
~3500-5000 LoC across vllm-omni and likely 200-500 LoC of upstream
vLLM changes (to make `DPCoordinator` pluggable / accept an external
rank assignment). Multiple new abstractions (PortRegistry, GlobalRankAssigner,
ElasticReplicaSet). Many weeks of work, plus a risky migration story
for users on PR #3569.

---

### 3b — EP group is per-replica

#### Feasibility
**High.** This is the natural fit:

1. **Each replica is already a self-contained vLLM world.** Once point 2
   gives each replica its own `data_parallel_size_local = dp_local`,
   the per-replica EP group is exactly what vLLM constructs by default
   (`vllm/distributed/parallel_state.py:1666-1696`). No new
   infrastructure.
2. **The fixed two-layer LB model lines up naturally with how vLLM
   constructs EP groups.** Inner LB is vLLM's `DPLBAsyncMPClient`
   (`stage_engine_core_client.py:112-115`), routing across DP ranks
   *within* one replica's EP world. Outer LB is vllm-omni's
   `StagePool.LoadBalancer` (`stage_pool.py:107-112`), routing across
   replicas that each have their own independent EP world. vLLM's
   `data_parallel_external_lb` / `data_parallel_hybrid_lb` modes are
   explicitly forbidden (see "Load-balancing model" section), so the
   per-replica EP world always uses the default
   `(data_parallel_external_lb=False, data_parallel_hybrid_lb=False)`
   configuration — exactly what vLLM's EP group construction at
   `vllm/distributed/parallel_state.py:1666-1696` expects.
3. **No cross-replica torch.distributed.** Replicas remain independent
   failure domains; OmniCoordinator's ZMQ pub/sub is preserved
   as-is for status + queue-length-based LB.
4. **EPLB works "for free" within each replica.** vLLM's EPLB only
   requires `dp * tp > 1` within the EP group
   (`vllm/config/parallel.py:454-458`); each replica trivially satisfies
   this once `dp_local >= 2` (or `tp >= 2`).

#### Limitations

1. **EP scope is bounded by the replica.** A model that needs an EP
   group of size 32 to amortize its expert footprint cannot be served
   by 4 replicas of size 8; the user must instead launch 1 replica of
   size 32 (which is `dp_local × tp = 32`). This is the fundamental
   trade-off: 3b trades EP scale for replica isolation. For most
   deployments this is the right trade.
2. **EP weight loading is per-replica.** If the model has
   `num_experts = 256` and we run 4 replicas with `ep_size = 8` each,
   every replica loads all 256 experts (sharded across its own 8 ranks).
   Total memory footprint is `4×` what a single `ep_size=32` deployment
   would use. Users wanting maximum expert-fanout should use fewer,
   larger replicas. Document this.
3. **EPLB rebalances per-replica only.** Expert popularity skew across
   replicas (e.g., one replica's traffic happens to hit experts 0-31
   more) cannot be balanced by EPLB; only the OmniCoordinator's
   request-level LB across replicas helps. For most workloads
   per-replica EPLB is sufficient.
4. **Inherits all point-2 limitations** (port hygiene, device math,
   YAML validator for the LB-mode fields). The decision on
   `external_lb` / `hybrid_lb` is already settled by the
   "Configuration surface" rule — both are forced to `False`, vLLM's
   `DPLBAsyncMPClient` always handles inner-DP routing.

#### System complexity
**Low-moderate** — essentially "point 2 + YAML-only EP/EPLB plumbing".
Specifically:

- **CLI**: keep `--enable-expert-parallel` prohibited (per
  "Configuration surface"); also add the EPLB / all2all / placement
  flags to the prohibition list. None of them get a CLI route. They
  are configured via YAML's `engine_args.parallel_config.enable_expert_parallel`,
  `.enable_eplb`, `.eplb_config.*`, `.expert_placement_strategy`,
  `.all2all_backend`, `.enable_ep_weight_filter`, etc.
- **YAML pass-through**: these fields already flow through
  `build_engine_args_dict` → `EngineArgs` → `ParallelConfig.__post_init__`
  (`vllm/config/parallel.py:446-467`) — no per-flag plumbing
  needed beyond the YAML validator.
- **Config validation**: when EP is enabled (via CLI or YAML), validate
  upstream that the per-replica vLLM world satisfies `dp_local * tp > 1`
  for EPLB (vLLM already errors out otherwise — we want a clearer
  vllm-omni-side error message).
- **All2All backend selection**: `--all2all-backend` (e.g.
  `deepep_low_latency`) is a per-engine concern; passes straight
  through to `parallel_config.all2all_backend`.
- **Worker subprocess kwargs**: spot-check that
  `OmniCoreEngineProcManager._common_kwargs`
  (`omni_core_engine_proc_manager.py:88-97`) does not strip EPLB args
  from `vllm_config` — it passes the whole `vllm_config` through, so
  this should be fine. Confirm with a Qwen-MoE smoke test.
- **Tests**: add EP-on integration tests; reuse Qwen-MoE / DeepSeek-MoE
  small variants. Cover `(num_replicas=2, dp_local=2, tp=2, ep=True)`.

#### Code-change estimate
~150-300 LoC on top of point 2's ~400-700 LoC. No new abstractions, no
upstream vLLM changes.

---

## 4. Summary table

| Dimension | 3a (EP spans replicas) | 3b (EP per replica) |
|---|---|---|
| Feasibility | Low — requires torch.dist world across replicas | High — fits vLLM's per-DP-cluster EP model unchanged |
| EP group size | `num_replicas × dp_local × tp` | `dp_local × tp` |
| Cross-replica all2all | Required every MoE layer | None |
| Replica isolation | Lost — replica failure ⇒ EP world collapse | Preserved |
| Dynamic replica add/remove | Requires elastic-EP (incompat. with external LB) | Works as-is |
| Cross-host serving | Per-layer collectives on inter-host net | Independent per replica |
| Expert weight memory | Best (sharded across all replicas) | `num_replicas × ` more (each replica loads full set) |
| Inter-replica skew handled by EPLB | Yes | No (request-LB only) |
| Wave-sync latency cost | Across all replicas | Within one replica |
| Code change (on top of pt 2) | ~3500-5000 LoC + upstream vLLM changes (or collapses the omni replica abstraction entirely) | ~150-300 LoC |
| Maintenance burden | New abstractions to maintain forever | Almost none |
| Compatibility with PR #3569 dynamic attach | Broken without elastic EP | Preserved |

---

## 5. Recommendation

**Adopt 3b.** It is the only design that:

1. **Preserves the architectural promise of PR #3569** — replicas as
   independent, dynamically-attachable, ZMQ-coordinated failure
   domains.
2. **Reuses vLLM's existing EP/EPLB machinery untouched** instead of
   forking the DPCoordinator concept.
3. **Aligns with how vLLM upstream expects "wide-EP at scale" to be
   deployed** — `data_parallel_external_lb` and "one-pod-per-rank" is
   precisely the abstraction our omni replica layer implements at a
   higher level.

The order of work is:

1. **Point 1** (rename) — land first, behind an alias. ~1 day.
2. **Point 2** (inner DP per replica, YAML-driven). The plumbing
   exists; what needs to happen is:
   1. Pick and apply the `omni_replica_id` resolution (DP ranks share
      one slot, recommended).
   2. **Extend** (don't remove from) the CLI prohibition list to cover
      every intra-replica parallelism flag — see "Configuration
      surface". `--data-parallel-size-local` stays prohibited; DP is
      YAML-only.
   3. Add the YAML validator (LB-mode rejection, `data_parallel_size`
      consistency, `data_parallel_backend == "mp"`, no
      address/port/rank user overrides, device-count product check).
      Symmetric for LLM and DiT stages.
   4. Centralize port allocation (`OmniPortRegistry`).
   5. Extend device-count math in `get_stage_devices_per_replica` /
      `split_devices_for_replicas`.
   6. Tests: `(num_replicas=2, dp_local=2, tp=2)` end-to-end, plus
      a battery of YAML-validator failure cases.

   ~1-2 weeks.
3. **Point 3b** (EP/EPLB per replica, YAML-driven). Mostly adding
   the EP/EPLB/all2all/placement flags to the CLI prohibition list
   (they're currently un-prohibited but consumed only via
   `EngineArgs`), wiring the YAML validator, and adding test coverage.
   ~3-5 days.

Re-evaluate 3a only if a user emerges who needs an EP group larger
than what a single replica's inner DP can provide *and* is willing to
forfeit replica-level isolation and dynamic membership. For that user,
the better answer is probably "launch one replica with a larger inner
DP" rather than "make replicas not really replicas".

---

## 6. Reference YAML (post-design)

Example LLM stage with inner DP + EP + EPLB, post-design.

Launched as (single host, 8 GPUs, 2 replicas × dp_local=2 × tp=2):

```
vllm serve <model> --omni \
    --deploy-config deploy/<model>_replicas_dp_ep.yaml \
    --omni-num-replica 2 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 50051
```

For a 2-host split (4 GPUs each), head on host A, headless on host B:

```
# Host A (head)
vllm serve <model> --omni \
    --deploy-config deploy/<model>_replicas_dp_ep.yaml \
    --omni-num-replica 1 \
    --omni-master-address <host-A-ip> --omni-master-port 50051

# Host B (headless, one extra replica)
vllm serve <model> --omni --headless --stage-id 0 \
    --deploy-config deploy/<model>_replicas_dp_ep.yaml \
    --omni-num-replica 1 \
    --omni-master-address <host-A-ip> --omni-master-port 50051 \
    --omni-replica-address <host-B-ip>
```

YAML (shared by all launches):

```yaml
stages:
  - stage_id: 0
    devices: "0,1,2,3"        # devices PER replica, sized = dp_local × tp
    runtime:
      num_replicas: 1         # CLI --omni-num-replica overrides this per
                              # process; the head+headless example above
                              # yields 2 replicas total across hosts.
    engine_args:
      parallel_config:
        tensor_parallel_size: 2
        data_parallel_size_local: 2
        # data_parallel_size               <- forbidden (or must equal _local)
        # data_parallel_external_lb        <- forbidden if True
        # data_parallel_hybrid_lb          <- forbidden if True
        # data_parallel_backend            <- forbidden if != "mp"
        # data_parallel_address/_rpc_port  <- forbidden (auto-assigned)
        # api_server_count                 <- forbidden (force-set to 0;
        #                                     vllm-omni owns the HTTP server,
        #                                     vLLM never starts one)
        enable_expert_parallel: true
        enable_eplb: true
        eplb_config:
          window_size: 1000
          step_interval: 3000
          num_redundant_experts: 0
        # enable_elastic_ep                <- forbidden (incompatible)
        all2all_backend: deepep_low_latency
        expert_placement_strategy: linear
```

Total physical GPUs consumed by this stage across the runtime =
`(total replicas across all hosts) × dp_local × tp` =
`(--omni-num-replica summed over invocations) × 2 × 2`.

Per-replica EP group size = `dp_local × pcp × tp = 2 × 1 × 2 = 4`
(per PP stage). Replicas do not share an EP group.

**DiT example** (unchanged from today's `deploy/hunyuan_image3_dit.yaml`
— the `DiffusionParallelConfig` block was always the only way to
configure DiT parallelism; the YAML validator just adds explicit error
messages around it):

```yaml
stages:
  - stage_id: 0
    devices: "0,1,2,3"        # devices per replica = world_size
    runtime:
      num_replicas: 1
    engine_args:
      parallel_config:          # this is a DiffusionParallelConfig
        tensor_parallel_size: 2
        data_parallel_size: 1   # DiT field — FSDP replicate / batch axis,
                                # NOT a wave-coordinated EngineCore DP cluster
        ulysses_degree: 2
        ring_degree: 1
        cfg_parallel_size: 1
        vae_patch_parallel_size: 1
        pipeline_parallel_size: 1
        enable_expert_parallel: true   # DiT EP — group size:
                                       #   tp × sp × cfg × dp = 2 × 2 × 1 × 1 = 4
        use_hsdp: false
        hsdp_shard_size: -1
        hsdp_replicate_size: 1
```

`world_size = tp × dp × ulysses × ring × cfg × pp = 4`, must equal
`len(devices.split(","))`. Launched with `--omni-num-replica K` for
`K` replicas; each replica is an independent `StageDiffusionProc` with
its own torch.distributed world of 4 GPUs.

## 7. Open questions for the team

- **YAML schema layout**: `parallel_config.enable_expert_parallel` is
  already used by diffusion stages today
  (`deploy/hunyuan_image3.yaml:73-75`, consumed in
  `async_omni_engine.py:1881-1894`). For LLM stages we'd add the same
  shape to `engine_args.parallel_config`. The deploy YAML already
  supports nested `parallel_config:`, so this is consistent — no new
  abstraction needed.
- **`omni_replica_id` semantics when `dp_local > 1`**: the doc at
  `omni_core_engine_proc_manager.py:102-106` says DP ranks share one
  omni replica; the code at `:117` and at
  `stage_engine_core_proc.py:102-104` treats each DP rank as a
  separate omni replica. Choose one and apply consistently. Strong
  recommendation: **DP ranks share one omni replica slot**, only the
  DP-rank-0 process runs `OmniCoordClientForStage`, and the reported
  queue length is the aggregate over the inner DP cluster (the
  DPCoordinator already aggregates per-rank queue stats).
- **Port allocation**: introduce a single `OmniPortRegistry` *before*
  point 2 lands. It cleans up the existing
  `master_port = settle_port(61000 + ...)` workaround at
  `serve.py:849-859` and prevents new collisions between
  OmniMasterServer ZMQ ports, OmniCoordinator ROUTER/PUB, vLLM
  DPCoordinator's three sockets, and `data_parallel_rpc_port`.
- **EPLB inside a per-replica EP group**: the EPLB rebalancing thread
  is local to a replica's worker subprocess. `OmniCoreEngineProcManager`
  passes the whole `vllm_config` through
  (`omni_core_engine_proc_manager.py:88-97`), so the EPLB config
  reaches the worker untouched. Add a Qwen-MoE / DeepSeek-MoE small
  smoke test to confirm at runtime.
- **OmniCoordinator queue-length signal under inner DP**: today each
  per-rank `OmniCoordClientForStage` reports its own queue length.
  Under the recommended "DP-ranks-share-a-slot" model, the reported
  value should be the sum over the inner DP cluster (or the value
  exposed by the inner `DPCoordinator`'s stats endpoint). The
  `LoadBalancer` policies (`random` / `round-robin` /
  `least-queue-length`) all already assume one queue-length value per
  replica; we just need to make sure the value is well-defined.
