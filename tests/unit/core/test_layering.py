"""Layering and import edges (TST-P-02, TST-C-05 static half; ARCHITECTURE § 3).

Static AST checks over src/norma plus fresh-interpreter import checks.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import norma

SRC = Path(norma.__file__).resolve().parent
ALL_MODULES = sorted(SRC.rglob("*.py"))


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level:
            found.add("." * node.level + (node.module or ""))
    return found


def norma_imports(path: Path) -> set[str]:
    return {name for name in imports_of(path) if name == "norma" or name.startswith(("norma.", "."))}


def test_models_imports_no_norma_module() -> None:
    assert norma_imports(SRC / "models.py") == set()


def test_errors_imports_only_models() -> None:
    assert norma_imports(SRC / "errors.py") <= {"norma.models"}


@pytest.mark.parametrize("path", sorted((SRC / "adapters").glob("*.py")), ids=lambda p: p.name)
def test_adapters_import_only_models_and_errors(path: Path) -> None:
    assert norma_imports(path) <= {"norma.models", "norma.errors"}


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_subprocess_only_in_adapters(path: Path) -> None:
    uses = {"subprocess", "multiprocessing"} & {name.split(".")[0] for name in imports_of(path)}
    if "adapters" not in path.relative_to(SRC).parts:
        assert not uses, f"{path.name} imports {uses}"
    text = path.read_text(encoding="utf-8")
    assert "os.system" not in text and "os.popen" not in text and "shell=True" not in text


def forbidden_imports(path: Path, imports: set[str]) -> set[str]:
    roots = {name.split(".")[0] for name in imports}
    banned = {"socket", "urllib", "http", "ssl", "requests"}
    if path.relative_to(SRC) not in {Path("metadata/read.py"), Path("metadata/write.py")}:
        banned.add("mutagen")
    return roots & banned


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_network_or_disallowed_mutagen(path: Path) -> None:
    assert not forbidden_imports(path, imports_of(path))


@pytest.mark.parametrize("module, allowed", [
    ("metadata/read.py", True),
    ("metadata/write.py", True),
    ("metadata/model.py", False),
    ("metadata/third.py", False),
    ("metadata/nested/read.py", False),
    ("audio/read.py", False),
    ("read.py", False),
])
@pytest.mark.parametrize("name", ["mutagen", "mutagen.id3"])
def test_mutagen_allowed_only_in_metadata_read_and_write(module: str, allowed: bool, name: str) -> None:
    forbidden = forbidden_imports(SRC / module, {name})
    assert forbidden == (set() if allowed else {"mutagen"})


@pytest.mark.parametrize("module", ["metadata/read.py", "metadata/write.py", "metadata/third.py"])
@pytest.mark.parametrize("name", ["socket", "urllib.request", "http.client", "ssl", "requests"])
def test_network_imports_forbidden_even_in_mutagen_modules(module: str, name: str) -> None:
    assert forbidden_imports(SRC / module, {name}) == {name.split(".")[0]}


def test_popen_calls_never_use_a_shell_string() -> None:
    for path in sorted((SRC / "adapters").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) in {"Popen", "run", "call",
                                                                                   "check_output"}:
                shell = [kw for kw in node.keywords if kw.arg == "shell"]
                assert shell and isinstance(shell[0].value, ast.Constant) and shell[0].value.value is False
                assert not (node.args and isinstance(node.args[0], (ast.Constant, ast.JoinedStr)))


def _fresh(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)


def test_acceptance_import_line() -> None:
    result = _fresh("import norma, norma.models, norma.errors, norma.adapters.ffmpeg")
    assert result.returncode == 0, result.stderr


def test_package_initialisers_are_inert() -> None:
    # importlib.metadata (for __version__) loads the socket module via email.utils;
    # loading a module is not network use, so the check covers processes and tags.
    result = _fresh(
        "import sys\n"
        "before = set(sys.modules)\n"
        "import norma, norma.adapters, norma.models, norma.errors\n"
        "new = set(sys.modules) - before\n"
        "print(sorted(n for n in new if n.split('.')[0] in {'subprocess', 'multiprocessing', 'mutagen'}"
        " or n in {'norma.adapters.ffmpeg', 'norma.adapters.ffprobe'}))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_socket_guard_refuses_connections() -> None:
    import socket

    with pytest.raises(AssertionError, match="network access attempted"):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    with socket.socket() as sock, pytest.raises(AssertionError):
        sock.connect(("127.0.0.1", 9))


def test_version_comes_from_distribution_metadata() -> None:
    assert norma.__version__ == importlib.metadata.version("norma")
    assert norma.__version__ not in (SRC / "__init__.py").read_text(encoding="utf-8")


def test_runtime_dependencies_are_exactly_mutagen() -> None:
    requires = importlib.metadata.requires("norma") or []
    runtime = {req.split(";")[0].split(">")[0].split("=")[0].split("<")[0].strip() for req in requires
               if "extra ==" not in req}
    assert runtime == {"mutagen"}


@pytest.mark.skipif(importlib.util.find_spec("norma.cli") is not None, reason="norma.cli exists (S1 onwards)")
def test_main_delegates_to_cli_and_fails_clearly_before_s1() -> None:
    result = subprocess.run([sys.executable, "-m", "norma"], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode != 0
    assert "norma.cli" in result.stderr
    source = (SRC / "__main__.py").read_text(encoding="utf-8")
    assert "argparse" not in source
