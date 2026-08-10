# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_isolated_import_succeeds(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_stable_engine_imports_load_duplex_kernel_eagerly() -> None:
    # The duplex kernel is imported eagerly by the stable engine modules.
    # Model-specific duplex adapters must still load only via the
    # dotted-string plugin paths, and the websockets demo client must never
    # be imported (it exits at import time when websockets is missing).
    _assert_isolated_import_succeeds("""
import sys

import vllm_omni.engine.async_omni_engine
import vllm_omni.engine.orchestrator
import vllm_omni.entrypoints.async_omni

expected_eager = (
    "vllm_omni.engine.duplex.contracts",
    "vllm_omni.engine.duplex.control_plane",
    "vllm_omni.entrypoints.duplex_request_client",
    "vllm_omni.outputs.duplex",
)
missing = sorted(name for name in expected_eager if name not in sys.modules)
if missing:
    raise SystemExit("duplex kernel modules not imported eagerly: " + ", ".join(missing))

forbidden_prefixes = (
    "vllm_omni.model_executor.models.minicpmo_4_5.duplex",
    "vllm_omni.model_executor.models.minicpmo_4_5.duplex.client",
    "vllm_omni.model_executor.models.personaplex.duplex",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
)
if loaded:
    raise SystemExit("stable imports loaded plugin-only duplex modules: " + ", ".join(loaded))
""")


def test_stable_engine_does_not_expose_duplex_contract_modules() -> None:
    _assert_isolated_import_succeeds("""
import importlib.util

from vllm_omni.engine import messages

legacy_modules = (
    "vllm_omni.engine.duplex_contracts",
    "vllm_omni.engine.duplex_lease",
    "vllm_omni.engine.resumable",
)
present = [name for name in legacy_modules if importlib.util.find_spec(name) is not None]
if present:
    raise SystemExit("stable duplex modules still exist: " + ", ".join(present))

duplex_exports = sorted(name for name in vars(messages) if name.startswith("Duplex"))
if duplex_exports:
    raise SystemExit("stable messages still expose duplex contracts: " + ", ".join(duplex_exports))
""")


def test_stable_outputs_do_not_declare_duplex_decision_field() -> None:
    _assert_isolated_import_succeeds("""
import dataclasses

from vllm_omni.outputs import OmniRequestOutput

fields = {field.name for field in dataclasses.fields(OmniRequestOutput)}
if "duplex_output_decision" in fields:
    raise SystemExit("stable output declares duplex_output_decision")
""")


def test_duplex_sampling_helper_lives_in_model_executor() -> None:
    _assert_isolated_import_succeeds("""
import importlib.util

if importlib.util.find_spec("vllm_omni.model_executor.duplex_sampling") is None:
    raise SystemExit("model_executor duplex_sampling helper module is missing")
""")
    assert not (REPO_ROOT / "vllm_omni" / "experimental" / "fullduplex" / "model_executor.py").exists()


def test_runtime_package_does_not_bundle_the_browser_demo() -> None:
    assert not (REPO_ROOT / "vllm_omni" / "experimental" / "fullduplex" / "web").exists()


def test_engine_duplex_uses_canonical_contract_module_names() -> None:
    engine_dir = REPO_ROOT / "vllm_omni" / "engine" / "duplex"
    core_dir = REPO_ROOT / "vllm_omni" / "experimental" / "fullduplex" / "core"

    for name in (
        "contracts.py",
        "lease.py",
        "messages.py",
        "session.py",
        "control_plane.py",
        "control_client.py",
        "runtime.py",
        "intermediate.py",
    ):
        assert (engine_dir / name).is_file()
    # the duplex_ prefix is dropped inside the duplex package
    assert not (engine_dir / "duplex_session.py").exists()
    assert not (engine_dir / "duplex_lease.py").exists()
    assert not (engine_dir / "duplex_types.py").exists()
    # the experimental package retains only the joyvl framework; stale
    # __pycache__ leftovers may exist, so assert on source files, not the dir
    old_engine_dir = REPO_ROOT / "vllm_omni" / "experimental" / "fullduplex" / "engine"
    assert not list(old_engine_dir.glob("*.py"))
    old_personaplex_dir = REPO_ROOT / "vllm_omni" / "experimental" / "fullduplex" / "personaplex"
    assert not list(old_personaplex_dir.glob("**/*.py"))
    assert not (core_dir / "identity.py").exists()
