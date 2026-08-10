from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).parents[3]
_CLIENT_MODULE = "vllm_omni.model_executor.models.minicpmo_4_5.duplex.client"
_CLIENT_PACKAGE = "vllm_omni.model_executor.models.minicpmo_4_5.duplex"


def _imports_demo_client(path: Path) -> bool:
    """True if the module imports the websockets demo client at module level."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Import):
            if any(alias.name == _CLIENT_MODULE for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == _CLIENT_MODULE:
                return True
            if node.module == _CLIENT_PACKAGE and any(alias.name == "client" for alias in node.names):
                return True
    return False


def test_api_server_does_not_import_duplex_demo_client() -> None:
    # client.py raises SystemExit at import time when the optional
    # websockets dependency is missing; the serving stack (imported eagerly
    # by the API server) must never reach it.
    api_server = _REPO_ROOT / "vllm_omni/entrypoints/openai/api_server.py"
    assert not _imports_demo_client(api_server)


def test_duplex_serving_stack_does_not_import_demo_client() -> None:
    duplex_pkg = _REPO_ROOT / "vllm_omni/entrypoints/openai/duplex"
    offenders = [
        path.name
        for path in sorted(duplex_pkg.glob("*.py"))
        if _imports_demo_client(path)
    ]
    assert not offenders, f"duplex serving modules import the demo client: {offenders}"
