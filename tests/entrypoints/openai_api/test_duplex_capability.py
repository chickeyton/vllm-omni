from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_api_server_does_not_import_duplex_at_module_load() -> None:
    api_server = Path(__file__).parents[3] / "vllm_omni/entrypoints/openai/api_server.py"
    module = ast.parse(api_server.read_text())

    top_level_imports = {
        node.module for node in module.body if isinstance(node, ast.ImportFrom) and isinstance(node.module, str)
    }

    duplex_prefixes = (
        "vllm_omni.entrypoints.openai.duplex",
        "vllm_omni.engine.duplex",
        "vllm_omni.entrypoints.duplex_request_client",
    )
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in top_level_imports
        for prefix in duplex_prefixes
    )
