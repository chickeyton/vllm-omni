# How jiuwenswarm uses vLLM-Omni for video generation

I checked jiuwenswarm at `HEAD ea68d6ed` (2026-10-26). All of its vLLM-Omni code is in one client file, `jiuwenswarm/agents/harness/common/tools/vllm_omni_gen.py`. It is called from `gen_toolkits.py`, which the chat agent uses, and from `server/runtime/designer/media_generation.py`, which the Design canvas uses. I didn't need the vLLM repo: jiuwenswarm only talks to vLLM-Omni's own routes.

## 1. Which vLLM-Omni HTTP APIs it uses

The default address is `http://127.0.0.1:8091/v1`, set in `model_vendor_registry.py`. The API key and model name are optional.

| Step | Call | Notes |
|---|---|---|
| Find the model | `GET /v1/models` | Called **on every video request**. It takes the first model id, sends it as `model`, and uses it to pick model-specific request settings. |
| Create a video | `POST /v1/videos` (multipart form) | Asynchronous: it returns an `id`. It never uses `/v1/videos/sync`. |
| Check progress | `GET /v1/videos/{id}` | Every 5 s, for up to 30 min (Design waits up to 2 h). |
| Download the video | `GET /v1/videos/{id}/content` | Saves an mp4. |
| Text-to-image | `POST /v1/images/generations` (JSON, `response_format=b64_json`) | |
| Image editing / image-to-image | `POST /v1/images/edits` (multipart, repeated `image` fields) | |

All of these routes exist in our server, in `vllm_omni/entrypoints/openai/api_server.py` (lines 2061, 2130, 2361, 2739, 2945 and 3040).

**What a video request contains:**

- **Always:** `prompt`, `model`, `fps=24`.
- **MiniMax-H3, hardcoded:**
  - `num_inference_steps=50` and `flow_shift=12.0`.
  - `width`/`height` rounded to multiples of 32, or `aspect_ratio` (+ `short_edge=768` when it uses the adaptive ratio).
  - `extra_params` = `{"task": "t2va"|"ref2va", "duration": 4–15 s, "audio_flow_shift": 3.0}`. It switches to `ref2va` whenever any reference image, video or audio is attached.
- **Any other model:** only standard fields — `size` + `seconds`, or `width`/`height`/`num_frames` when an fps is given. No `extra_params`.
- **Reference files:**
  - Uploaded files go in `input_reference` or `input_references`.
  - URLs go in `image_reference`, `video_reference` or `audio_reference` as JSON.
  - Limits: 9 images, 3 videos, 3 audios, 12 in total.

## 2. Which vLLM-Omni models it supports

From the frontend list `channels/web/frontend/src/features/settings/modules/agent/generationModels.ts`:

- **Video:** `MiniMaxAI/MiniMax-H3`. This is the only model with tuned request settings; there is a separate match for "H3-Max", which uses a 5–15 s duration range.
- **Image:** `Qwen/Qwen-Image-2512` and `black-forest-labs/FLUX.2-dev`.

Any other model the server loads will still work through the generic request path. In practice, though, MiniMax-H3 is the only video model tuned and tested end to end. `docs/en/Design.md` says the video models tested so far are H3, Seedance 2.5 and Wan 3.0, but the last two run through cloud providers, not vLLM-Omni.

## 3. Reference docs in jiuwenswarm

There is no dedicated tutorial for deploying vLLM-Omni. The closest material:

- **`docs/en/Design.md`** (Chinese version: `docs/zh/设计.md`):
  - §2 covers configuring the image and video generation models.
  - §5.5 covers importing a ComfyUI workflow. Only vLLM-Omni Generate Image / Generate Video nodes are recognised, and their sampling parameters (steps, guidance scale, seed and so on) are sent to vLLM-Omni unchanged.
  - The FAQ says a self-hosted vLLM-Omni node can wait up to 2 h before it is marked failed.
- **The header comment in `vllm_omni_gen.py`** points back to our own docs: `docs/serving/videos_api.md`, `image_generation_api.md`, `image_edit_api.md` and `recipes/MiniMaxAI/MiniMax-H3.md`. All of them exist in this repo, along with hardware-specific H3 recipes for 4090, 5090, NPU and others.
- **Tests:** `tests/unit_tests/agents/harness/test_vllm_omni_gen.py` and `test_video_gen_tools.py` show the exact request contents they check.

## What this means for optimisation

These are my reading of the code, not measured:

- The main workload is **MiniMax-H3, 50 steps, 24 fps, 4–15 s clips, usually `ref2va`**, since Design always generates video from reference images. A comment says one 50-step run takes about 9 minutes on 2×RTX 5090.
- Because the client always sends 50 steps, a server-side default will never take effect. Reducing steps would need a change in jiuwenswarm.
- The extra `GET /models` call and the 5 s polling add little next to minutes of generation time.

---

# Suggestions for optimising vLLM-Omni video generation (jiuwenswarm as the frontend)

Denoising takes up almost all of a MiniMax-H3 request. The recipe's MI300X runs (4 GPUs, 50 steps) spend 244 of 267 s denoising for text-to-video and 258 of 287 s for first-frame-to-video. So the biggest win is running fewer or cheaper denoise steps. Cache-aware replica routing only helps with the smaller costs: encoding inputs and switching models. I checked this against vllm-omni `0f7a408c3`; the speedups below come from the recipe's own measurements, not my runs.

## 1. Choosing a replica with the cache in mind

**What exists today:** `LoadBalancer.select(task, replicas)` (`vllm_omni/distributed/omni_coordinator/load_balancer.py`) already receives the request's inputs and sampling settings. All three policies (random, round-robin, least-queue-length) ignore them. Each replica reports only a `queue_length`. A request only sticks to a replica when it shares its own id or its CFG parent's id (`engine/stage_pool.py:333`).

Suggested changes, best first:

1. **Balance by estimated work, not request count.** Design submits many shots at once, and H3 jobs range from 4 to 15 s of video with varying reference counts and step counts. Counting queued requests treats a 4-step 4 s clip the same as a 50-step 15 s clip.
   - Each replica would report its outstanding cost in its heartbeat, roughly frames × tokens per frame × steps.
   - The balancer would pick the replica with the least outstanding cost.
   - This is the most reliable gain whenever there is more than one replica.
2. **Prefer replicas that already have the right state loaded.** Three things are expensive to switch on a replica:
   - **The active Turbo LoRA.** Loading a different adapter clears the AdaLN caches (`pipeline_minimax_h3.py:719-721`).
   - **The Cache-DiT profile.** It is installed or removed whenever `quality` changes between requests.
   - **The task's transformer.** FL2VA and Ref2VA use separate transformers, which matters most when weights are offloaded.

   Route by the key (task, LoRA, quality) to a replica that already has it loaded, unless that replica's queued work is more than a threshold above the least-loaded one.
3. **Send shots with the same reference images to the same replica.** Design reuses the same character and scene sheets across a project's shots, so hash the reference files and route by that hash. This **only pays off together with item 2c below**. Today H3 has no cache across requests:
   - Its `encode_prompt` takes a `PreparedEncoderInputs` object, which the generic prompt-embedding cache cannot key on, so it is skipped.
   - The reference-KV cache only lives for one request.

**Scope caveat:** this balancer only runs when vLLM-Omni is deployed with stage replicas under the coordinator. jiuwenswarm talks to a single address, so if people run several separate servers behind a proxy, the routing happens outside vLLM-Omni.

## 2. Other directions in vLLM-Omni

**a. Turbo LoRA, the biggest lever (50 → 4 or 8 steps).** It already exists, including an 8-step Ref2VA 768p adapter (recipe §LoRA). The blocker is that jiuwenswarm always sends `num_inference_steps=50, flow_shift=12`.

- **Server-side fix:** add a named serving profile, listed in `GET /v1/models` (e.g. `…-turbo`), that sets the LoRA, steps and shift together. jiuwenswarm already sends the first model id it finds, so it would need no change.
- **Quality:** must still be compared against 50 steps for Design-style Ref2VA shots. I have no data on that.

**b. Turn on `--cache-backend cache_dit` on the server. No code change.** jiuwenswarm never sends `quality`, so the server's startup profile applies to every request.

- **Measured:** 1.35× faster on 4×H200, with SSIM 0.971 against the exact path.
- **Limit:** it can't be combined with step execution.

**c. Cache reference encodings across requests.** Key on a hash of the reference file plus its preprocessing settings, and keep the vision-encoder and VAE outputs.

- **Expected size:** encoding is about 14 s out of about 287 s for first-frame-to-video, so roughly 5% per shot.
- **Ref2VA:** with up to 9 images plus video, the share may be larger. Measure Ref2VA encode time before building this.

**d. Don't pursue co-batching requests.** The recipe measured H3 step co-batching and found no throughput gain: each step is already a compute-bound GEMM over a long sequence. Add capacity with replicas or sequence parallelism instead, which is why item 1.1 matters.

**e. Changes on the jiuwenswarm side worth proposing:**

- When a user clicks "Regenerate", cancel the old job with `DELETE /v1/videos/{id}` so it doesn't keep using GPU time.
- Call `GET /v1/models` once and remember the result instead of calling it per request (small win).
- Show the `progress` field the server already returns (`protocol/videos.py:474`). That improves the experience, not speed.

## Suggested order

1. **2b** first (configuration only).
2. **2a** next (largest gain, needs a quality check).
3. **1.1** once you have more than one replica.
4. **1.3** and **2c** together, after measuring Ref2VA encode time.
