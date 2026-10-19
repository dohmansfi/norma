"""IC-4 N-10: interface import edges, including lazy imports (architecture § 3)."""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import norma

SRC = Path(norma.__file__).resolve().parent
PURE_CONFIG_HELPERS = {
    "norma.metadata.model",  # Pure field/extra validation in _metadata_options.
    "norma.metadata.model.canonical_text",
    "norma.metadata.model.canonical_date",
    "norma.metadata.infer.compile_pattern",
}
EDGE_MODULES = {"norma.safety", "norma.discovery", "norma.paths"}


def import_targets(source: str, module: str) -> set[str]:
    """Collect absolute targets, also for relative and literal dynamic imports."""
    tree = ast.parse(source)
    targets = set()
    importlib_names = {"importlib"}
    dynamic_names = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
            importlib_names.update(alias.asname or alias.name for alias in node.names
                                   if alias.name == "importlib")
        elif isinstance(node, ast.ImportFrom):
            base = (importlib.util.resolve_name("." * node.level + (node.module or ""),
                                               module.rpartition(".")[0])
                    if node.level else node.module or "")
            targets.update(base + "." + alias.name for alias in node.names)
            if base == "importlib":
                dynamic_names.update(alias.asname or alias.name for alias in node.names
                                     if alias.name == "import_module")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dynamic = (isinstance(node.func, ast.Name) and node.func.id in dynamic_names
                   or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                   and isinstance(node.func.value, ast.Name) and node.func.value.id in importlib_names)
        if dynamic:
            name = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg == "name"), None)
            assert isinstance(name, ast.Constant) and isinstance(name.value, str), (
                "Non-literal dynamic import needs an explicit architecture review", ast.unparse(node))
            target = name.value
            if target.startswith("."):
                target = importlib.util.resolve_name(target, module.rpartition(".")[0])
            targets.add(target)
    return targets


def forbidden_targets(source: str, module: str) -> set[str]:
    forbidden = set()
    for target in import_targets(source, module):
        if target.startswith(("norma.metadata", "norma.adapters")):
            if module != "norma.config" or target not in PURE_CONFIG_HELPERS:
                forbidden.add(target)
        if any(target == edge or target.startswith(edge + ".") for edge in EDGE_MODULES):
            forbidden.add(target)
    return forbidden


@pytest.mark.parametrize("module", ["cli", "config"])
def test_interface_imports_only_permitted_metadata_helpers(module):
    assert not forbidden_targets((SRC / (module + ".py")).read_text(encoding="utf-8"),
                                 "norma." + module)


@pytest.mark.parametrize("statement", [
    "import norma.metadata.read",
    "from norma.metadata import read",
    "from .metadata.read import read_tags",
    "from norma import safety",
    "from . import adapters",
    "import norma.adapters.ffmpeg as ffmpeg",
    "import importlib; importlib.import_module('norma.metadata.read')",
    "import importlib as il; il.import_module('norma.safety')",
    "from importlib import import_module as load; load('norma.adapters.sips')",
    "__import__('norma.metadata.write')",
])
@pytest.mark.parametrize("module", ["norma.cli", "norma.config"])
def test_import_guard_catches_nested_static_relative_and_dynamic_edges(statement, module):
    assert forbidden_targets("def lazy():\n    " + statement, module)


@pytest.mark.parametrize("target", sorted(PURE_CONFIG_HELPERS))
def test_pure_validation_helpers_are_config_only(target):
    module, name = target.rsplit(".", 1)
    source = f"def validate():\n    from {module} import {name}"
    assert not forbidden_targets(source, "norma.config")
    assert forbidden_targets(source, "norma.cli") == {target}


def test_unapproved_helper_in_an_allowed_module_is_refused():
    assert forbidden_targets("from norma.metadata.model import other_helper", "norma.config")


def test_nonliteral_dynamic_import_requires_review():
    with pytest.raises(AssertionError, match="Non-literal dynamic import"):
        import_targets("import importlib; importlib.import_module(selected)", "norma.cli")


def test_fresh_cli_import_loads_no_execution_or_metadata_modules():
    result = subprocess.run(
        [sys.executable, "-c", "import sys,json; import norma.cli; "
         "print(json.dumps(sorted(name for name in sys.modules if "
         "name.startswith(('norma.metadata', 'norma.adapters')) or "
         "name in {'norma.safety','norma.discovery','norma.paths','norma.pipeline','norma.batch','mutagen'})))"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
