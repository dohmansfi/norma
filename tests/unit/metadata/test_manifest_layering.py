"""Layering for the manifest loader and the inference module (ARCHITECTURE § 3; TST-P-02)."""

from __future__ import annotations

import ast
from pathlib import Path

import norma.metadata.infer as infer_module
import norma.metadata.manifest as manifest_module

FORBIDDEN = {"mutagen", "subprocess", "multiprocessing", "socket", "urllib", "http", "ssl", "requests",
             "norma.adapters", "norma.pipeline", "norma.cli", "norma.config", "norma.report", "norma.audio"}


def imports(module: object) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_manifest_and_infer_import_only_the_pure_core() -> None:
    for module in (manifest_module, infer_module):
        names = imports(module)
        assert not {n for n in names if n in FORBIDDEN or n.split(".")[0] in FORBIDDEN}, names
        assert {n for n in names if n.startswith("norma")} <= {
            "norma.errors", "norma.models", "norma.metadata", "norma.metadata.model"}, names


def test_infer_has_no_file_access_and_manifest_reads_one_file_only() -> None:
    assert "open(" not in Path(infer_module.__file__).read_text(encoding="utf-8")
    source = Path(manifest_module.__file__).read_text(encoding="utf-8")
    assert source.count("read_bytes(") == 1 and "write_" not in source and "open(" not in source
