# Supporting both CLI parallelism arguments and YAML — Problem 4

> **Status:** analysis / companion to `rfc-omni-replica-dp-ep.md`.
>
> **Scope:** this document focuses on **Problem 4 — cross-stage KV-cache
> transfer rank-mapping mismatch** under the assumption that CLI parallelism
> arguments can override the per-stage deploy-YAML parallel configs of both
> LLM and DiT stages. Other potential hazards (default leakage, opposite
> loader precedence, per-stage expressibility of a global flag, `devices:`
> decoupling, force-set DP, validator blindness, DiT validator gap,
> nested-vs-flat routing, DiT internal invariants firing late) are
> deliberately set aside here and addressed in earlier exploratory analysis.

---

## Background — what the rank mapping reads, and where it can diverge

Stages that exchange KV cache (e.g. prefill → decode, LLM → codec) compute
their transfer rank mapping at engine-build time:

- `_inject_inferred_kv_tp_topology` (`stage_init_utils.py:243-304`) writes
  `rank_mapping["from_tp"]` / `["to_tp"]` into the KV connector's
  `omni_kv_config`.
- Both values come from `_tp_size_for_stage(stage_configs, …)`
  (`stage_init_utils.py:217-240`), which reads `tensor_parallel_size` from
  the **resolved per-stage configs** (nested `parallel_config.…` for DiT,
  flat for LLM).
- The connector consumes the pair: `get_kv_target_ranks` /
  `get_kv_source_ranks` (`kv_utils.py:152-171`) compute *which* remote
  ranks to address; `_slice_transfer_data_for_target`
  (`kv_transfer_manager.py:516-537`) drives sender-side head-slicing for
  the fan-out case from the same `to_tp`; connector keys
  (`get_kv_connector_key`, `kv_utils.py:179-190`) embed both `from_rank`
  and `to_rank` so the keys the sender publishes must match the keys the
  receiver pulls.

Because the rank mapping is **inferred per-process** from each process's
view of the resolved stage_configs, the head and any headless invocation
must arrive at the same numbers for the transfer to compose. Under the
assumption above, a CLI override that changes a stage's actual TP in one
process but not in another breaks that agreement.

---

## Problem 4 — Cross-stage KV-cache rank-mapping mismatch *(can hang or corrupt)*

Each process re-parses its own CLI and re-runs `load_and_resolve_stage_configs`
on the same deploy YAML. If the parallelism flags differ between
invocations — head vs. headless, or two headless invocations of the same
stage on different hosts — the two views of the same stage's TP diverge.
Because `_tp_size_for_stage` reads from that local view, the inferred
`from_tp`/`to_tp` differ across the transfer pair, and the values are
baked into:

- the connector keys the sender publishes (encodes `to_rank` derived from
  `to_tp`),
- the **number and shape** of head-shards the sender pre-slices in
  fan-out (`source_tp < target_tp`), driven by
  `ratio = target_tp_size / source_tp_size` at `kv_transfer_manager.py:519`,
- the source-rank set the receiver enumerates and pulls from in fan-in.

The failure mode is concrete and not a quiet regression: the receiver
either pulls keys that don't exist (hang waiting for blocks that were
published under different `to_rank`s), or pulls shards whose shape doesn't
match what its rank-mapping expects (concat ValueError, or silent KV
corruption when the receiver overwrites its cache with the wrong head
range).

### DiT is not exempt under this assumption

Today DiT is accidentally shielded because the CLI flat
`tensor_parallel_size` cannot reach the nested
`parallel_config.tensor_parallel_size` the engine actually shards by.
Under the assumption that override is possible (e.g. via a flat→nested
re-nester, or a `--parallel-config.tensor-parallel-size` style flag), DiT
loses that protection and becomes a Problem-4 surface on both ends:

- **As source.** A CLI override of the DiT stage's nested TP that differs
  between head and the DiT-headless invocations produces divergent
  `from_tp` on the DiT-as-sender side → the next stage's receiver
  mis-routes.
- **As victim.** Even with DiT TP unchanged, a CLI override of the
  **upstream LLM** stage's TP that diverges across processes makes the
  head's `from_tp(LLM)` and the DiT-headless's `from_tp(LLM)` disagree —
  the DiT receiver pulls under the wrong rank map, and is the one that
  physically gets bad data.

### Fixability

**Detection feasible; transparent correction infeasible without
cross-process reconciliation.** Each process shards locally and computes
the rank mapping from its own view *before* the registration handshake
runs, so a divergent CLI value cannot be made to "just work" after the
fact. The feasible path is to make the master server / OmniCoordinator
authoritative for parallel configs: have each replica report its
**resolved post-CLI** parallelism at registration, and have the
orchestrator hand each downstream replica the immediate-upstream's
parallel config along with the work it dispatches, so the connector
inside each replica knows the right rank to pull from regardless of what
its own CLI / YAML said about the peer. That redesign is the subject of
the *Supporting parallelism CLI arguments* section in
`rfc-omni-replica-dp-ep.md`.

---

## Why this problem is the load-bearing one

Of all the CLI/YAML hazards explored, Problem 4 is the one that escalates
from "wrong but loud" (the rest, all of which surface as validation
errors or visible misbehavior on a single process) to **wrong and
silent across processes** — KV blocks routed between processes that
each *individually* validated their own configuration. That makes it
distinct from the others: every other hazard can be caught with
single-process validation; Problem 4 fundamentally requires cross-process
agreement on the parallel config, and that is what the proposed handshake
exchange supplies.
