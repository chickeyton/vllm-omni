# Duplex Refactor: Line Counts and Maintainability, Baseline vs Refactored

Measured 2026-09-05; refactored side re-measured after the D17 simplification pass.

| Tree | Path | Revision |
| --- | --- | --- |
| Baseline | `D:\repo\github\chickeyton\vllm-omni_duplex_refactor_baseline` | `7112347f` (`duplex_refactor1`, clean checkout, before the refactor) |
| Refactored | `D:\repo\github\chickeyton\vllm-omni_duplex_refactor` | `duplex_refactor1` after the refactor (`6a151f9f`) and the simplification pass of plan revision 5 (D17) |

Scope and method, identical for both trees:

- Only `vllm_omni/**/*.py` is scanned. `vllm_omni/experimental/` and `vllm_omni/clients/` are
  skipped; `tests/`, `examples/`, `benchmarks/` and `docs/` are outside the package and therefore
  never counted.
- Effective lines = physical lines minus blank lines, pure comment lines and docstring lines
  (module/class/function docstrings and any bare string-literal statement). A multi-line statement
  counts every physical line it occupies. Counted with `ast` + `tokenize` (script in Appendix B; the
  baseline numbers reproduce the earlier baseline document exactly).
- "Duplex logic" = the engine-side duplex package, the duplex API/serving files, and the model
  duplex packages of MiniCPM-o 4.5 and PersonaPlex. Exact file groups are named in the tables; the
  per-file list is in Appendix A. Files that were moved keep counting in whichever group they live in.
  Duplex members that the baseline kept inside the generic `async_omni.py`, `async_omni_engine.py`
  and `orchestrator.py` are not part of the baseline duplex-logic total; their removal shows up in
  rows 2-4 of the requested numbers instead.
- Maintainability Index (MI) is radon 6.0.1's formula (Halstead volume, cyclomatic complexity,
  logical lines, comment percentage with docstrings counted as comments), computed per module.
  MI is a per-module metric, so a scope is summarised two ways: the mean over files weighted by
  each file's logical lines (large files dominate, which is what a reader of the code experiences)
  and the plain mean over files. radon ranks: A = MI >= 20, B = 10-19, C = < 10. The complexity,
  volume and logical-line totals that feed MI are listed alongside so the movement can be explained.
  Note that MI is dominated by per-file size and total branch count and clamps to 0 for any module
  with roughly 700+ branches; it does not reward structural changes (ownership, typing, layering).

## Requested numbers (effective lines)

| # | Scope | Baseline | Refactored | Delta |
| --- | --- | ---: | ---: | ---: |
| 1 | All `vllm_omni/**/*.py` excluding `vllm_omni/experimental/` and `vllm_omni/clients/` (1292 / 1289 files) | 356,499 | 354,549 | -1,950 (-0.5%) |
| 2 | `vllm_omni/entrypoints/async_omni.py` | 1,252 | 746 | -506 (-40.4%) |
| 3 | `vllm_omni/engine/async_omni_engine.py` | 1,748 | 516 | -1,232 (-70.5%) |
| 4 | `vllm_omni/engine/orchestrator.py` | 2,375 | 2,073 | -302 (-12.7%) |

New shared base files in the refactored tree (not present in the baseline; effective lines):

| File | Refactored |
| --- | ---: |
| `vllm_omni/entrypoints/async_omni_base.py` | 362 |
| `vllm_omni/engine/omni_engine_base.py` | 997 |
| `vllm_omni/entrypoints/omni_base.py` (existing, baseline 596) | 598 |

## Duplex logic: effective lines

| Group | Baseline files | Baseline effective | Refactored files | Refactored effective | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Engine side (`engine/duplex/**`, `engine/duplex_omni_engine.py`, `engine/duplex_orchestrator.py`) | 9 | 2,224 | 20 | 10,532 | +8,308 (+373.6%) |
| API + serving (`entrypoints/duplex/**`, `entrypoints/duplex_omni.py`, `entrypoints/duplex_request_client.py`) | 18 | 10,769 | 6 | 1,276 | -9,493 (-88.2%) |
| MiniCPM-o 4.5 (`model_executor/models/minicpmo_4_5/duplex/**`) | 11 | 2,785 | 9 | 2,736 | -49 (-1.8%) |
| PersonaPlex (`model_executor/models/personaplex/duplex/**`) | 8 | 1,188 | 7 | 1,156 | -32 (-2.7%) |
| **Duplex logic incl. MiniCPM-o and PersonaPlex** | **46** | **16,966** | **42** | **15,700** | **-1,266 (-7.5%)** |
| Nemotron VoiceChat (`model_executor/models/nemotron_voicechat/duplex/**`), for reference | 5 | 982 | 4 | 960 | -22 (-2.2%) |
| Duplex logic incl. all three model plugins | 51 | 17,948 | 46 | 16,660 | -1,288 (-7.2%) |

## Duplex logic: Maintainability Index

| Scope | | Baseline | Refactored | Delta |
| --- | --- | ---: | ---: | ---: |
| **Duplex logic incl. MiniCPM-o and PersonaPlex** | MI, LLOC-weighted mean | 12.6 | 15.1 | +2.5 |
| | MI, plain mean over files | 42.0 | 43.9 | +1.8 |
| | files ranked A / B / C | 35 / 1 / 10 | 32 / 3 / 7 | |
| | cyclomatic complexity, total | 4,198 | 4,113 | -85 (-2.0%) |
| | Halstead volume, total | 85,962 | 82,203 | -4.4% |
| | logical lines (radon LLOC), total | 11,773 | 11,936 | +163 (+1.4%) |
| | comment + docstring lines / SLOC | 3.5% | 5.8% | |
| Engine side | MI, LLOC-weighted mean | 20.3 | 12.1 | -8.1 |
| | MI, plain mean over files | 45.1 | 35.6 | -9.5 |
| | files ranked A / B / C | 7 / 0 / 2 | 14 / 1 / 5 | |
| | cyclomatic complexity, total | 442 | 2,763 | +2,321 (+525.1%) |
| | Halstead volume, total | 6,563 | 59,340 | +804.2% |
| | logical lines (radon LLOC), total | 1,497 | 8,001 | +6,504 (+434.5%) |
| | comment + docstring lines / SLOC | 1.8% | 5.0% | |
| API + serving | MI, LLOC-weighted mean | 7.5 | 34.3 | +26.9 |
| | MI, plain mean over files | 30.4 | 53.8 | +23.4 |
| | files ranked A / B / C | 12 / 0 / 6 | 6 / 0 / 0 | |
| | cyclomatic complexity, total | 2,686 | 301 | -2,385 (-88.8%) |
| | Halstead volume, total | 60,218 | 3,208 | -94.7% |
| | logical lines (radon LLOC), total | 7,260 | 974 | -6,286 (-86.6%) |
| | comment + docstring lines / SLOC | 2.6% | 10.1% | |
| MiniCPM-o 4.5 | MI, LLOC-weighted mean | 15.9 | 10.9 | -5.0 |
| | MI, plain mean over files | 48.6 | 46.6 | -2.0 |
| | files ranked A / B / C | 9 / 0 / 2 | 6 / 1 / 2 | |
| | cyclomatic complexity, total | 827 | 811 | -16 (-1.9%) |
| | Halstead volume, total | 15,408 | 15,862 | +2.9% |
| | logical lines (radon LLOC), total | 2,143 | 2,106 | -37 (-1.7%) |
| | comment + docstring lines / SLOC | 6.6% | 6.8% | |
| PersonaPlex | MI, LLOC-weighted mean | 33.6 | 31.3 | -2.3 |
| | MI, plain mean over files | 55.8 | 55.3 | -0.4 |
| | files ranked A / B / C | 7 / 1 / 0 | 6 / 1 / 0 | |
| | cyclomatic complexity, total | 243 | 238 | -5 (-2.1%) |
| | Halstead volume, total | 3,773 | 3,792 | +0.5% |
| | logical lines (radon LLOC), total | 873 | 855 | -18 (-2.1%) |
| | comment + docstring lines / SLOC | 6.6% | 6.6% | |
| Nemotron VoiceChat, for reference | MI, LLOC-weighted mean | 31.6 | 28.3 | -3.4 |
| | MI, plain mean over files | 50.5 | 48.2 | -2.2 |
| | files ranked A / B / C | 4 / 1 / 0 | 3 / 1 / 0 | |
| | cyclomatic complexity, total | 221 | 216 | -5 (-2.3%) |
| | Halstead volume, total | 2,931 | 2,962 | +1.0% |
| | logical lines (radon LLOC), total | 664 | 644 | -20 (-3.0%) |
| | comment + docstring lines / SLOC | 3.9% | 4.2% | |

## Duplex engine + serving layers only (model plugins excluded)

Scope: `engine/duplex/**`, `engine/duplex_omni_engine.py`, `engine/duplex_orchestrator.py`, `entrypoints/duplex/**`, `entrypoints/duplex_omni.py`, `entrypoints/duplex_request_client.py`. Same exclusions and method as above.

| Metric | Baseline | Refactored | Delta |
| --- | ---: | ---: | ---: |
| Engine side, effective lines | 2,224 | 10,532 | +8,308 (+373.6%) |
| Serving side, effective lines | 10,769 | 1,276 | -9,493 (-88.2%) |
| **Engine + serving, effective lines** | **12,993** | **11,808** | **-1,185 (-9.1%)** |
| Engine + serving, files | 27 | 26 | -1 (-3.7%) |
| MI, LLOC-weighted mean | 9.7 | 14.5 | +4.9 |
| MI, plain mean over files | 35.3 | 39.8 | +4.5 |
| Files ranked A / B / C | 19 / 0 / 8 | 20 / 1 / 5 | |
| Cyclomatic complexity, total | 3,128 | 3,064 | -64 (-2.0%) |
| Halstead volume, total | 66,781 | 62,549 | -6.3% |
| Logical lines (radon LLOC), total | 8,757 | 8,975 | +218 (+2.5%) |
| Comment + docstring lines / SLOC | 2.5% | 5.5% | |

## Appendix A: per-file numbers (duplex logic)

Effective lines and radon MI per file. A dash means the file does not exist in that tree.

| File | Baseline eff. | Baseline MI | Refactored eff. | Refactored MI |
| --- | ---: | ---: | ---: | ---: |
| `vllm_omni/engine/duplex/__init__.py` | 0 | 100.0 | 0 | 100.0 |
| `vllm_omni/engine/duplex/audio.py` | — | — | 159 | 32.6 |
| `vllm_omni/engine/duplex/commands.py` | — | — | 215 | 46.4 |
| `vllm_omni/engine/duplex/commit_policy.py` | — | — | 26 | 69.5 |
| `vllm_omni/engine/duplex/config.py` | — | — | 680 | 0.0 |
| `vllm_omni/engine/duplex/contracts.py` | 205 | 38.1 | 148 | 41.4 |
| `vllm_omni/engine/duplex/control_client.py` | 205 | 47.8 | — | — |
| `vllm_omni/engine/duplex/control_plane.py` | 982 | 0.0 | — | — |
| `vllm_omni/engine/duplex/events.py` | — | — | 528 | 31.4 |
| `vllm_omni/engine/duplex/intermediate.py` | 75 | 60.0 | 75 | 60.0 |
| `vllm_omni/engine/duplex/lease.py` | 101 | 41.7 | 101 | 41.7 |
| `vllm_omni/engine/duplex/messages.py` | 98 | 46.4 | 79 | 63.8 |
| `vllm_omni/engine/duplex/plugin.py` | — | — | 236 | 42.2 |
| `vllm_omni/engine/duplex/realtime_commands.py` | — | — | 750 | 0.0 |
| `vllm_omni/engine/duplex/realtime_events.py` | — | — | 1,319 | 0.0 |
| `vllm_omni/engine/duplex/runtime.py` | 94 | 65.7 | — | — |
| `vllm_omni/engine/duplex/session.py` | 464 | 6.7 | 1,241 | 0.0 |
| `vllm_omni/engine/duplex/session_manager.py` | — | — | 648 | 15.3 |
| `vllm_omni/engine/duplex/session_runner.py` | — | — | 3,409 | 0.0 |
| `vllm_omni/engine/duplex/turn_detection.py` | — | — | 210 | 42.8 |
| `vllm_omni/engine/duplex/vad.py` | — | — | 156 | 34.8 |
| `vllm_omni/engine/duplex_omni_engine.py` | — | — | 276 | 47.3 |
| `vllm_omni/engine/duplex_orchestrator.py` | — | — | 276 | 43.5 |
| `vllm_omni/entrypoints/duplex/__init__.py` | 2 | 100.0 | 2 | 100.0 |
| `vllm_omni/entrypoints/duplex/audio.py` | 159 | 32.6 | — | — |
| `vllm_omni/entrypoints/duplex/capability.py` | 25 | 79.7 | — | — |
| `vllm_omni/entrypoints/duplex/chat_fallback.py` | 241 | 33.8 | — | — |
| `vllm_omni/entrypoints/duplex/commit_policy.py` | 26 | 69.5 | — | — |
| `vllm_omni/entrypoints/duplex/protocol.py` | 1,321 | 0.0 | — | — |
| `vllm_omni/entrypoints/duplex/realtime_input.py` | 1,389 | 0.0 | 119 | 54.7 |
| `vllm_omni/entrypoints/duplex/realtime_output.py` | 927 | 0.0 | — | — |
| `vllm_omni/entrypoints/duplex/realtime_session.py` | 120 | 46.7 | — | — |
| `vllm_omni/entrypoints/duplex/realtime_state.py` | 190 | 36.5 | — | — |
| `vllm_omni/entrypoints/duplex/runtime_adapter.py` | 176 | 34.7 | — | — |
| `vllm_omni/entrypoints/duplex/runtime_bridge.py` | 1,302 | 0.0 | — | — |
| `vllm_omni/entrypoints/duplex/serving.py` | 1,760 | 0.0 | 399 | 31.0 |
| `vllm_omni/entrypoints/duplex/session_attachment.py` | 424 | 20.4 | 374 | 24.0 |
| `vllm_omni/entrypoints/duplex/session_runner.py` | 1,990 | 0.0 | — | — |
| `vllm_omni/entrypoints/duplex/vad.py` | 156 | 34.8 | — | — |
| `vllm_omni/entrypoints/duplex/websocket.py` | 224 | 30.3 | 31 | 79.6 |
| `vllm_omni/entrypoints/duplex_omni.py` | — | — | 351 | 33.7 |
| `vllm_omni/entrypoints/duplex_request_client.py` | 337 | 28.1 | — | — |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/__init__.py` | 11 | 100.0 | 11 | 100.0 |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/adapter.py` | 283 | 34.7 | — | — |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/capabilities.py` | 37 | 94.7 | 35 | 95.1 |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/compat.py` | 35 | 66.8 | 35 | 66.8 |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/data_plane.py` | 735 | 0.0 | 736 | 0.0 |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/input.py` | 374 | 26.4 | 375 | 26.3 |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/plugin.py` | — | — | 628 | 10.1 |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/policy.py` | 112 | 61.4 | 112 | 61.4 |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/runtime.py` | 313 | 24.1 | — | — |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/serving_adapter.py` | 82 | 66.4 | — | — |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/session.py` | 48 | 59.8 | 49 | 59.6 |
| `vllm_omni/model_executor/models/minicpmo_4_5/duplex/stage0.py` | 755 | 0.0 | 755 | 0.0 |
| `vllm_omni/model_executor/models/personaplex/duplex/__init__.py` | 24 | 100.0 | 22 | 100.0 |
| `vllm_omni/model_executor/models/personaplex/duplex/config.py` | 20 | 78.9 | 20 | 78.9 |
| `vllm_omni/model_executor/models/personaplex/duplex/data_plane.py` | 184 | 28.7 | 185 | 28.7 |
| `vllm_omni/model_executor/models/personaplex/duplex/input.py` | 181 | 35.5 | 182 | 35.4 |
| `vllm_omni/model_executor/models/personaplex/duplex/plugin.py` | — | — | 303 | 34.0 |
| `vllm_omni/model_executor/models/personaplex/duplex/policy.py` | 33 | 92.3 | 33 | 92.3 |
| `vllm_omni/model_executor/models/personaplex/duplex/runtime_extension.py` | 112 | 52.5 | — | — |
| `vllm_omni/model_executor/models/personaplex/duplex/serving_adapter.py` | 223 | 40.0 | — | — |
| `vllm_omni/model_executor/models/personaplex/duplex/stage0.py` | 411 | 18.2 | 411 | 18.2 |
| `vllm_omni/model_executor/models/nemotron_voicechat/duplex/__init__.py` | 0 | 100.0 | 0 | 100.0 |
| `vllm_omni/model_executor/models/nemotron_voicechat/duplex/data_plane.py` | 347 | 17.8 | 348 | 17.7 |
| `vllm_omni/model_executor/models/nemotron_voicechat/duplex/input.py` | 183 | 42.5 | 184 | 42.4 |
| `vllm_omni/model_executor/models/nemotron_voicechat/duplex/plugin.py` | — | — | 428 | 32.8 |
| `vllm_omni/model_executor/models/nemotron_voicechat/duplex/runtime.py` | 139 | 56.6 | — | — |
| `vllm_omni/model_executor/models/nemotron_voicechat/duplex/serving_adapter.py` | 313 | 35.4 | — | — |

## Appendix B: measurement script

Run as `python measure.py <repo_root> <out.json>` against each tree, then render the two JSON files side by side. Requires `radon` (`pip install radon`).

```python
"""Measure effective LOC and radon Maintainability Index for one tree.

usage: python measure.py <repo_root> <out.json>
"""
from __future__ import annotations

import ast
import io
import json
import sys
import tokenize
from pathlib import Path

from radon.complexity import ComplexityVisitor
from radon.metrics import h_visit_ast, mi_compute
from radon.raw import analyze

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PKG = ROOT / "vllm_omni"
EXCLUDE_DIRS = {"experimental", "clients"}


def docstring_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def effective_lines(src: str) -> tuple[int, int]:
    total = src.count("\n") + (0 if src.endswith("\n") or not src else 1)
    tree = ast.parse(src)
    doc = docstring_lines(tree)
    code_lines: set[int] = set()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
            tokenize.ENCODING, tokenize.ENDMARKER}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in skip:
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            if ln not in doc:
                code_lines.add(ln)
    return len(code_lines), total


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def is_duplex_logic(r: str) -> str | None:
    """Return the duplex-logic group for a file, or None."""
    if r.startswith("vllm_omni/engine/duplex/") or r in (
        "vllm_omni/engine/duplex_omni_engine.py",
        "vllm_omni/engine/duplex_orchestrator.py",
    ):
        return "engine"
    if r.startswith("vllm_omni/entrypoints/duplex/") or r in (
        "vllm_omni/entrypoints/duplex_request_client.py",
        "vllm_omni/entrypoints/duplex_omni.py",
    ):
        return "entrypoints"
    if r.startswith("vllm_omni/model_executor/models/minicpmo_4_5/duplex/"):
        return "minicpmo"
    if r.startswith("vllm_omni/model_executor/models/personaplex/duplex/"):
        return "personaplex"
    if r.startswith("vllm_omni/model_executor/models/nemotron_voicechat/duplex/"):
        return "nemotron"
    return None


EMPTY = {"files": 0, "effective": 0, "physical": 0, "volume": 0.0, "cc": 0, "lloc": 0, "sloc": 0,
         "comment_lines": 0, "mi_sum": 0.0, "mi_lloc_weighted": 0.0, "rank_A": 0, "rank_B": 0, "rank_C": 0}


def main() -> None:
    files: dict[str, dict] = {}
    for p in sorted(PKG.rglob("*.py")):
        parts = p.relative_to(PKG).parts
        if parts and parts[0] in EXCLUDE_DIRS:
            continue
        src = p.read_text(encoding="utf-8")
        eff, phys = effective_lines(src)
        r = rel(p)
        entry = {"effective": eff, "physical": phys, "group": is_duplex_logic(r)}
        if entry["group"] is not None:
            node = ast.parse(src)
            raw = analyze(src)
            volume = h_visit_ast(node).total.volume
            cc = ComplexityVisitor.from_ast(node).total_complexity
            comment_lines = raw.comments + raw.multi
            comments_pct = comment_lines / float(raw.sloc) * 100 if raw.sloc else 0.0
            entry.update(
                mi=mi_compute(volume, cc, raw.lloc, comments_pct),
                halstead_volume=volume,
                complexity=cc,
                lloc=raw.lloc,
                sloc=raw.sloc,
                comment_lines=comment_lines,
            )
        files[r] = entry

    requested = {
        "all_pkg": {
            "files": len(files),
            "effective": sum(v["effective"] for v in files.values()),
            "physical": sum(v["physical"] for v in files.values()),
        },
    }
    for t in (
        "vllm_omni/entrypoints/async_omni.py",
        "vllm_omni/engine/async_omni_engine.py",
        "vllm_omni/engine/orchestrator.py",
        "vllm_omni/entrypoints/async_omni_base.py",
        "vllm_omni/engine/omni_engine_base.py",
        "vllm_omni/entrypoints/omni_base.py",
    ):
        if t in files:
            requested[t] = {"effective": files[t]["effective"], "physical": files[t]["physical"]}

    groups: dict[str, dict] = {}
    for r, v in files.items():
        g = v["group"]
        if g is None:
            continue
        d = groups.setdefault(g, dict(EMPTY))
        d["files"] += 1
        d["effective"] += v["effective"]
        d["physical"] += v["physical"]
        d["volume"] += v["halstead_volume"]
        d["cc"] += v["complexity"]
        d["lloc"] += v["lloc"]
        d["sloc"] += v["sloc"]
        d["comment_lines"] += v["comment_lines"]
        d["mi_sum"] += v["mi"]
        d["mi_lloc_weighted"] += v["mi"] * v["lloc"]
        d["rank_A" if v["mi"] >= 20 else "rank_B" if v["mi"] >= 10 else "rank_C"] += 1

    def finish(d: dict) -> dict:
        return {
            **d,
            "mi_mean": d["mi_sum"] / (d["files"] or 1),
            "mi_lloc_weighted": d["mi_lloc_weighted"] / (d["lloc"] or 1),
            "comments_pct": 100.0 * d["comment_lines"] / (d["sloc"] or 1),
        }

    def combine(names: list[str]) -> dict:
        d = dict(EMPTY)
        for n in names:
            for k in d:
                d[k] += groups[n][k]
        return finish(d)

    out = {
        "requested": requested,
        "groups": {g: finish(d) for g, d in groups.items()},
        "totals": {
            "engine+entrypoints": combine(["engine", "entrypoints"]),
            "minicpmo+personaplex": combine(["minicpmo", "personaplex"]),
            "duplex_logic_mc_pp": combine(["engine", "entrypoints", "minicpmo", "personaplex"]),
            "duplex_logic_all_models": combine(["engine", "entrypoints", "minicpmo", "personaplex", "nemotron"]),
        },
        "files": {r: v for r, v in files.items() if v["group"] is not None},
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: files={requested['all_pkg']['files']} eff={requested['all_pkg']['effective']}")


if __name__ == "__main__":
    main()
```
