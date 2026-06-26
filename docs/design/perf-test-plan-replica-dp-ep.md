# Performance Test Plan — Omni-Replica + vLLM-DP (PR `replica_dp_ep_2`)

**Model / workload:** HunyuanImage-3.0-Instruct, TI2I (`/v1/images/edits`).
**Date authored:** 2026-06-24. **Branch under test:** `replica_dp_ep_2` (merge `2910e695`).

## 1. Objective

Quantify the cost and benefit of this PR by comparing four configurations,
**all with 1 DiT stage replica** (AR replicas / inner-DP / EP vary as shown):

| # | Code | AR parallelism | EP | AR GPUs | Purpose |
|---|------|----------------|----|---------|---------|
| **C1** | `origin/main` (without PR) | 2 rep × TP2, **no inner DP** | off | 4 | Baseline |
| **C2** | this PR | 2 rep × TP2, **no inner DP** | off | 4 | Same topology as C1 → isolates **PR overhead** |
| **C3** | this PR | 2 rep × TP2 × **DP2** | **on** (EP=TP×DP=4) | 8 | Inner DP **+ EP** on 2 replicas → realistic **DP+EP MoE serving** on top of replicas |
| **C4** | this PR | **1 rep** × TP2 × **DP2** | **on** (EP=TP×DP=4) | 4 | Same 4-GPU AR budget as C2, scaled via **inner-DP+EP instead of a 2nd replica** → **DP+EP vs replica** axis |

> **EP is enabled on C3/C4 only** (the inner-DP configs) so they reflect how an MoE model is
> actually served — DP and EP move together as one feature stack. C1/C2 keep EP off as the
> no-DP/no-EP baseline. Consequence: the C2↔C3 and C2↔C4 deltas reflect **DP and EP combined**,
> not DP alone. If you need to attribute the two separately, run the EP-off variant of C3/C4
> (Appendix B) as a third arm.

Three comparisons:
- **C1 ↔ C2** — *regression gate.* Same topology (both no-DP/no-EP), different code. Any delta is pure PR overhead. **Pass = within ±3% (noise).**
- **C2 ↔ C3** — *feature-stack scaling test.* Baseline → DP2+EP, +4 GPUs. Measures throughput gained by the realistic MoE stack (inner-DP **and** expert-parallel) per added GPU. **Expect AR-stage throughput ≈ 2×;** e2e gain bounded by the DiT stage (see §9).
- **C2 ↔ C4** — *equal-budget axis test.* Both use **4 AR GPUs**; C2 spends them on 2 replicas (no DP/EP), C4 on 1 replica × DP2+EP. Isolates **which scaling axis** (omni-replica vs vLLM DP+EP) is more efficient at fixed GPU cost — no extra hardware, so any AR-throughput delta is the axis (DP+EP) itself.

## 2. Hypotheses

- H1: C2 ≈ C1 on throughput and latency (PR introduces no measurable regression for the no-DP path).
- H2: C3 roughly **doubles AR-stage throughput** vs C2 (two DP engines per replica); EP keeps the MoE expert weights sharded so per-GPU VRAM stays viable while DP adds the throughput.
- H3: C3 e2e throughput improves over C2 **only to the extent AR is the bottleneck**; with 1 fixed DiT replica the e2e ceiling is set by DiT.
- H4: C4 ≈ C2 on AR-stage throughput at equal 4-GPU budget — DP2+EP and a 2nd replica should scale AR similarly; any gap reveals per-axis overhead (DP coordination + EP all-to-all cost vs replica/KV-routing cost).
- H5: EP shards the MoE experts across the EP=4 group, so each GPU holds a **smaller expert
  shard** than the dense-replica baseline. At a **fixed `gpu_memory_utilization`** that freed
  weight memory is **reallocated to KV cache** (more KV blocks / higher sustainable concurrency),
  not returned — so the observable signal is **larger KV cache at ~equal VRAM peak**, plus expert
  all-to-all traffic. (VRAM *peak* stays ~gmu-bound and roughly equal across configs — do **not**
  expect it to drop.)

## 3. Fixed topology & hardware

**DiT is held identical across all four runs** (1 replica, TP2, **no DP**) so it cancels out of every comparison.

**Everything runs on test server 1 (8× ~140 GB GPU).** HunyuanImage-3.0 requires TP≥2, so C3's
`2 rep × TP2 × DP2` consumes all 8 GPUs — DiT therefore **colocates** with the AR stage. To keep
the result fair, DiT is **pinned to the same GPU pair (0,1) in all four configs**, where it
shares with exactly **one** AR TP2 engine in every config (so DiT's local contention is held
constant; C3 merely adds AR engines on GPUs 2–7 that the 4-GPU configs leave idle).

```
Server 1 — 8× ~140 GB GPU      (KV transfer is intra-host: Mooncake over loopback)
GPU:     0      1      2      3      4      5      6      7
DiT TP2 [█ 0–1 █]                                              (colocated, ALL configs)
C1/C2   [ rep0   ][ rep1   ]  idle   idle   idle   idle        AR 2 rep × TP2  (no DP, EP off)
C4      [r0·DP0  ][r0·DP1  ]  idle   idle   idle   idle        AR 1 rep × TP2 × DP2 + EP  (EP group=4; same 4-GPU budget as C1/C2)
C3      [r0·DP0  ][r0·DP1  ][r1·DP0 ][r1·DP1 ]                 AR 2 rep × TP2 × DP2 + EP  (EP group=4 within each replica)
```

- All processes (head + AR replicas + colocated DiT) on server 1; single deploy YAML
  (`hunyuan_ti2i_colocate.yaml`, colocate validated previously). No headless / second host.
- **EP scope (C3/C4):** expert-parallel runs **within each replica** over its `TP×DP = 4`
  ranks (experts sharded across that group); it does **not** span replicas. The DiT stage has no EP.
- DiT shares GPUs 0–1 with exactly one AR TP2 engine in **every** config (rep0 in C1/C2,
  rep0·DP0 in C3/C4) → DiT's local neighborhood is identical across configs.
- C4 and C1/C2 occupy the **same GPUs 0–3** with the AR stage idle on 4–7, so the
  C2↔C4 comparison adds no hardware — only the scaling axis changes.
- `gpu_memory_utilization` for the AR and DiT stages is chosen so they coexist on the shared
  pair, and the **same** values are used in all four configs (so AR engines are uniform even
  on the unshared GPUs 2–7). Note: at equal gmu the EP-on configs (C3/C4) get a larger KV cache
  from the smaller expert shard — recorded as a metric, see §6 / H5, not a confound to remove.

## 4. Controlled variables (held constant across C1/C2/C3/C4)

| Held constant | Value |
|---|---|
| Model + revision | `tencent/HunyuanImage-3.0-Instruct`, same snapshot |
| DiT replicas | 1 (AR replicas / inner-DP **vary** by config — see §1) |
| AR TP | 2 | DiT TP | 2 |
| **Expert parallel (EP)** | **VARIES by design** — off in C1/C2 (baseline), **on** in C3/C4 (EP group = TP×DP = 4). Bundled with inner-DP so C3/C4 reflect realistic MoE serving; see §1 note + §11 confounder. |
| dtype / quant | identical |
| `enable_prefix_caching` | **false** (deterministic, no cross-request cache skew) |
| Sampling params | `temperature=0, top_p=1, top_k=-1, max_tokens=8192, seed=fixed` |
| Diffusion steps / guidance / image size | fixed (e.g. 50 steps, guidance 7.5, 1024²) |
| Prompt set | identical fixed set (§5) |
| `gpu_memory_utilization` | same AR value + same DiT value in all 4 (sized to coexist on the shared pair 0–1) |
| DiT placement | colocated, pinned to GPUs **0–1**, sharing with one AR TP2 engine — identical in all 4 |
| **DiT inner-DP** | **none (DiT `data_parallel_size = 1`)** in all 4 — only the **AR** stage ever uses inner-DP |
| Client / load generator / concurrency schedule | identical |
| Hardware | **test server 1 only** — same 8 GPUs, **exclusive** (no co-tenants) |

Only the **AR stage** varies: code version (C1 vs C2/C3/C4), number of AR replicas, AR
inner-DP, and AR expert-parallel (on for C3/C4). The **DiT stage is byte-for-byte identical** in
all four configs (1 replica, TP2, no DP, no EP).

## 5. Workload

- **Prompt set:** a fixed pool of **64 TI2I requests** (reuse `ti2i_inputs/setA` + `setB`,
  repeated/extended to 64 with fixed image inputs and prompts). Identical pool for all runs.
- **Two diffusion profiles** to expose where DP helps:
  - **P-heavy:** 50 steps (DiT-bound — realistic; shows e2e ceiling).
  - **P-light:** 12 steps (shifts bottleneck toward AR — exposes the inner-DP benefit at e2e).
- **Concurrency sweep:** 1, 4, 8, 16 in-flight requests (find saturation; report the curve).
- **Load generator / reporter:** the built-in **omni benchmark harness**
  (`vllm_omni/benchmarks/`) with `return_stage_metrics=true`, so per-stage TTFT/e2el/throughput
  (§6.1) are produced identically for all four configs. Avoid a hand-rolled client.

## 6. Metrics

All latency/throughput metrics are captured **from the live serving path** (not unit tests),
via the built-in omni benchmark harness + per-stage metrics — see §6.1 for the exact sources.

**LLM (AR) stage — latency (per request, p50/p95/p99):**
- **AR-stage TTFT** — time to first AR output token (`ttft` on the text stage;
  `SERVING_TIME_TO_FIRST_OUTPUT_MS`). The headline LLM-stage latency for the DP claim.
- **AR-stage E2E latency** — arrival → AR completion (KV ready to hand off) — the text-stage
  `e2el`. This is where inner-DP should reduce queueing latency under load.
- **AR TPOT / ITL** — inter-token latency on the AR stage (optional, for AR compute detail).

**LLM (AR) stage — throughput:**
- **AR-stage throughput** — AR completions/sec at saturation. Inner-DP should ≈2× this.

**DiT stage:**
- **DiT-stage latency** — DiT `e2el` / first-output (per-stage image metrics); ~constant across configs.
- **DiT-stage throughput** — images/sec (denoise loops/sec); ~constant across configs.

**End-to-end (request arrival → final image):**
- **E2E throughput** — completed images/sec at saturation.
- **E2E latency** — p50/p95/p99 (`e2el`, overall).
- **Stage breakdown** — AR time, (intra-host) KV-transfer time, DiT time per request.

**Resource:**
- **VRAM peak** per GPU; **GPU util** per GPU (`nvidia-smi dmon` @1 Hz).
- **KV-cache blocks / max concurrent sequences** — from the engine's startup log. Under EP (C3/C4)
  this **rises at the same gmu** (smaller expert shard frees memory into KV) and is the true
  EP-sharding signal; VRAM peak itself stays ~gmu-bound and roughly constant (see H5). Record it so
  a throughput gain isn't misattributed to pure DP compute scaling.

### 6.1 How the LLM-stage metrics are captured (live, not UT)

1. **Per-stage metrics over the API** — send `return_stage_metrics=true` on the `/v1/images/edits`
   request. The response carries `stage_metrics` keyed by `stage_id`, including the text (AR)
   stage's first-output time and stage latency (`api_server.py` `return_stage_metrics` →
   `out.stage_metrics`). This is the primary, per-request source.
2. **Benchmark harness aggregation** — `vllm_omni/benchmarks/metrics/metrics.py` consumes those
   per-request `stage_metrics` and emits per-stage **TTFT / e2el / TPOT / ITL** percentiles
   (`_build_stage_metrics_from_outputs` → `print_stage_metrics` → `_print_text_stage_metrics`
   for the LLM stage, `_print_image_stage_metrics` for DiT). Use this harness as the load
   generator/reporter so all four configs are scored identically.
3. **Cross-check** — orchestrator `arrival_time`, `stage_submit_ts[stage_id]`, and
   `_emit_tx_edge(tx_ms)` give an independent per-stage timeline if the API metrics need auditing.

> Because TTFT/e2el are reported **per stage**, the LLM-stage numbers are isolated from the DiT
> stage automatically — directly addressing the §9 DiT-bottleneck caveat: even when *overall*
> e2e is DiT-bound, the **AR-stage TTFT and e2el** still expose the inner-DP benefit.

## 7. Methodology

- **Warmup:** discard first 8 completed requests per run (model warmup / autotune).
- **Measure:** next 64 requests at steady state.
- **Repeat:** 3 rounds per (config × profile × concurrency); report **mean ± stddev**.
- **Saturation:** report the concurrency at which throughput plateaus; headline numbers taken there.
- **Isolation:** one config up at a time; full teardown + GPU-free check + stale-lock cleanup
  between runs (`/tmp/vllm_omni_device_*_init.lock`).

## 8. Per-config setup

Single deploy YAML = `hunyuan_ti2i_colocate.yaml` (server 1), DiT pinned to GPUs `0,1` with
**DiT `data_parallel_size = 1`, no EP** in every config; the AR stage's **replica count,
`data_parallel_size`, and `enable_expert_parallel`** are the only differences.

| | Code checkout (server 1) | AR replicas | AR `data_parallel_size` | AR `enable_expert_parallel` | `--omni-dp-size-local` | AR GPUs | DiT GPUs (TP2, DP1, no EP) |
|---|---|---|---|---|---|---|---|
| **C1** | `origin/main` | 2 | 1 (omit) | false | 1 | `0,1,2,3` | `0,1` |
| **C2** | `replica_dp_ep_2` | 2 | 1 (omit) | false | 1 | `0,1,2,3` | `0,1` |
| **C3** | `replica_dp_ep_2` | 2 | 2 | **true** | 2 | `0,1,2,3,4,5,6,7` | `0,1` |
| **C4** | `replica_dp_ep_2` | 1 | 2 | **true** | 2 | `0,1,2,3` | `0,1` |

EP applies only to the **AR (MoE) stage**; the DiT stage keeps `enable_expert_parallel=false`.

Per-run procedure (all on server 1):
1. `git fetch && git checkout <ref>` on server 1 (editable install, no reinstall).
2. Confirm all 8 GPUs free + no co-tenants; clean stale device locks (`/tmp/vllm_omni_device_*_init.lock`).
3. Launch the colocated deploy → wait health 200; confirm `reg` = stages registered, AR ranks
   distinct, `EADDRINUSE=0`.
4. Start `nvidia-smi dmon` sampler.
5. Run warmup (8) then measured batch (64) at each concurrency level via the load generator.
6. Collect logs + dmon + client timings → `test_artifacts/perf_<config>_<profile>_<conc>/`.
7. Tear down; verify all 8 GPUs released before the next config.

## 9. The DiT-bottleneck caveat (read before interpreting C2↔C3)

There is **only 1 DiT replica (TP2, no DP) in all four configs** (per the requested topology). In an
AR→DiT pipeline the diffusion stage (50 steps) is usually the slowest stage, so **e2e
throughput is capped by DiT** regardless of AR DP. Therefore:

- Report **AR-stage throughput separately** — that is where C3's DP+EP benefit shows (~2×),
  even when e2e is flat.
- Use the **P-light (12-step)** profile to make AR a larger fraction of the pipeline so the
  inner-DP benefit becomes visible end-to-end.
- Do **not** conclude "DP gives no benefit" from a DiT-bound e2e number — state the bottleneck
  explicitly and cite the stage-level numbers.

## 10. Pass / interpretation criteria

- **Regression (C1↔C2):** PASS if **AR-stage TTFT (p95), AR-stage e2el (p95), e2e throughput, and
  e2e p95 latency** are all within **±3%** at every concurrency for both profiles. A consistent
  >3% slowdown in C2 on any of these is a PR regression to investigate.
- **Feature-stack scaling (C2↔C3):** report AR-stage throughput ratio (target ≈1.7–2.0×) **and the
  AR-stage TTFT/e2el reduction under load** (the clearest LLM-stage signal), plus the overall e2e
  ratio per profile, plus **KV-cache blocks / max concurrency** (the EP-sharding effect at fixed
  gmu; VRAM peak stays ~constant — see H5). Label results **"DP+EP"** — the delta bundles both.
  Document the bottleneck. No hard pass/fail — this characterizes the feature.
- **Equal-budget axis (C2↔C4):** both spend 4 AR GPUs. Report the AR-stage throughput / TTFT / e2el
  ratio C4/C2 (target ≈ 1.0 — parity between DP+EP and a 2nd dense replica) **and KV-cache blocks**
  (C4's EP shard buys extra KV headroom at the same gmu — part of why it may win, so report it). A
  material gap identifies the cheaper scaling axis and its per-axis overhead (incl. EP all-to-all).
  No hard pass/fail — characterizes the feature.
- **Correctness sanity:** spot-check 5 output images per config match their own input (note: C1
  on main carries the pre-fix multi-replica reference-image leak — outputs may be wrong, which
  does **not** affect throughput/latency numbers but should be recorded).

## 11. Confounders & mitigations

| Confounder | Mitigation |
|---|---|
| Co-tenant GPU contention (pqyin/xyz/lby) | Require **exclusive** 8-GPU box; verify 0 foreign procs before each run |
| DiT↔AR colocation contention | DiT pinned to GPUs 0–1 sharing with exactly one AR engine in **all** configs → local contention constant; C3/C4 add their extra AR engines only on the unshared GPUs 2–7 (C4: 2–3; C3: 2–7). Report AR-stage metrics (per-stage isolated) as primary. |
| Intra-host KV transfer | Loopback (Mooncake), constant across all 4 (cancels); still report KV-transfer time |
| Cold cache / JIT autotune | Warmup discard; identical `FLASHINFER` autotune setting |
| Thermal / clock drift across long runs | Randomize config order across rounds; 3 rounds; report stddev |
| DP↔EP confound (C3/C4) | **By design** EP is bundled with inner-DP, so C2↔C3 and C2↔C4 measure the combined feature, not DP alone. Mitigation: an **EP-off variant of C3/C4** (Appendix B) is the third arm if pure-DP attribution is needed. Report it as "DP+EP", never "DP", in conclusions. |
| EP all-to-all overhead | EP=4 adds expert all-to-all traffic; on a colocated single host this shares interconnect with KV/DiT. Capture GPU-util + the stage breakdown so EP comms cost is visible, not hidden in AR e2el. |
| EP frees weight VRAM → larger KV cache (C3/C4) | At **fixed gmu** the smaller EP expert shard frees memory that the engine fills with **extra KV cache**, raising sustainable concurrency independent of compute. This is part of the DP+EP benefit but must be **logged explicitly** (KV-cache blocks per engine) so a throughput gain isn't misattributed to pure DP compute scaling. VRAM peak stays ~gmu-bound (do not read it as "EP saved memory"). |
| Prefix-cache skew | `enable_prefix_caching=false` |
| C1 ref-image leak | Affects image content only, not timing; record but ignore for perf |

## 12. Results template

Per (config × profile), at saturation concurrency. **AR-stage TTFT and AR-stage e2el are the
headline LLM-stage metrics**; overall e2e is reported alongside.

| Config | Profile | AR TTFT p50/p95 (ms) | AR e2el p50/p95 (s) | AR throughput (req/s) | DiT throughput (img/s) | E2E throughput (img/s) | E2E p50/p95 (s) | KV xfer (ms) | VRAM peak (GB) / KV blocks |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 50-step | | | | | | | | |
| C2 | 50-step | | | | | | | | |
| C3 | 50-step | | | | | | | | |
| C4 | 50-step | | | | | | | | |
| C1 | 12-step | | | | | | | | |
| C2 | 12-step | | | | | | | | |
| C3 | 12-step | | | | | | | | |
| C4 | 12-step | | | | | | | | |

Derived: `C2/C1` (overhead) on AR TTFT + AR e2el + e2e throughput; `C3/C2` AR-throughput ratio,
`C3/C2` AR-TTFT/e2el reduction under load, `C3/C2` overall e2e ratio; `C4/C2` equal-budget
axis ratio (DP+EP vs 2nd dense replica) on AR throughput + AR TTFT/e2el + KV-cache blocks.

## 13. Pre-flight checklist

- [ ] **Test server 1:** all 8 GPUs free, no foreign procs (pqyin/xyz/lby), stale locks cleaned.
- [ ] Server 1 checked out to the correct ref per config (`origin/main` for C1, `replica_dp_ep_2` for C2/C3/C4).
- [ ] `hunyuan_ti2i_colocate.yaml` present, DiT pinned to GPUs 0–1 (DP1, no EP); AR `enable_expert_parallel` per config (false C1/C2, true C3/C4); AR gmu/DiT gmu sized to coexist.
- [ ] Model snapshot present on server 1.
- [ ] Load generator (omni benchmark harness) + 64-prompt pool staged.
- [ ] `nvidia-smi dmon` sampler ready on server 1.

---

### Appendix A — Multi-host variant (contention-free, if a 2nd server is available)

Move DiT to a separate host (server 2 GPUs 2,3, TP2, still **no DP**) so it never shares GPUs with AR; AR then has
all 8 server-1 GPUs (C1/C2/C4 on 0–3, C3 on 0–7). This removes the DiT↔AR colocation confound
entirely at the cost of cross-host KV (constant across configs). Scripts:
`run_head_multihost.sh` / `run_headless_multihost.sh` / `hunyuan_ti2i_multihost.yaml`
(validated: `test_artifacts/ti2i_multihost_09804b5c/`). Use if the single-host colocation
contention is judged to materially affect the C2↔C3 e2e read.

### Appendix B — Optional follow-up matrices
- **EP-off variant of C3/C4** (`enable_expert_parallel=false`, DP2 only) — the third arm that
  separates pure inner-DP from EP, resolving the DP↔EP confound in C2↔C3 / C2↔C4 (§11).
- **2 DiT replicas** — lifts the DiT ceiling so AR inner-DP benefit appears at e2e.
