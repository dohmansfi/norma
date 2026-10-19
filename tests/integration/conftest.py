"""Shared helpers for the S1 integration tests (TK-0706, WS-07).

Every test runs the real ``norma`` entry point (``python -m norma``) as a child
process in the hermetic sandbox from tests/conftest.py: temporary HOME, XDG
folders and working directory, with generated audio held only below the test
session's temporary folders. Nothing here imports an adapter or the pipeline;
the layer under test is the command line.

Stdout is not a terminal in a child process started with captured pipes, so
the commands print the machine-readable tab-separated form (CLI § 7.1).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "norma_integration_test_fixtures", Path(__file__).parents[1] / "fixtures" / "make_fixtures.py",
)
assert _spec is not None and _spec.loader is not None
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
fixture_path = fixtures.fixture_path


@pytest.fixture(scope="session")
def fixture_module():
    """The WS-07 recipe module, for its canonical constants (TITLE, COMMENT, MARK)."""
    return fixtures


FILE_STATUSES = ("OK", "WARN", "SKIP", "FAIL")
_COUNTER = re.compile(r"\d+/\d+")
_ESCAPES = {"\\": "\\", "t": "\t", "r": "\r", "n": "\n"}


def unescape(cell: str) -> str:
    """Invert ``report.escape_cell`` in one pass (backslash, tab, CR, LF)."""
    return re.sub(r"\\([\\trn])", lambda match: _ESCAPES[match[1]], cell)


@dataclass(frozen=True)
class Run:
    code: int
    stdout: str
    stderr: str

    @property
    def lines(self) -> list[list[str]]:
        return [[unescape(cell) for cell in line.split("\t")] for line in self.stdout.splitlines()]

    @property
    def files(self) -> list[list[str]]:
        """Per-file result rows: status, ``n/N``, path, I, TP, gain, action, three blanks, note."""
        return [line for line in self.lines if line[0] in FILE_STATUSES and len(line) > 1 and _COUNTER.fullmatch(line[1])]

    @property
    def summary(self) -> dict[str, str]:
        found = [line for line in self.lines if line[0] == "SUMMARY"]
        assert len(found) == 1, self.stdout
        return dict(cell.split("=", 1) for cell in found[0][1:])

    def records(self, kind: str) -> list[list[str]]:
        return [line for line in self.lines if line[0] == kind]

    def rows(self) -> dict[str, list[str]]:
        """Doctor rows by check name: ``name -> [status, detail]`` (the last row wins)."""
        return {line[1]: [line[0], line[2]] for line in self.lines if len(line) == 3}


@pytest.fixture
def norma(tmp_path: Path):
    """Run ``python -m norma`` in the sandbox; ``path`` replaces PATH when given."""

    def run(*arguments: object, path: str | None = None, cwd: Path | None = None) -> Run:
        environment = dict(os.environ)
        environment["NO_COLOR"] = "1"
        if path is not None:
            environment["PATH"] = path
        completed = subprocess.run(
            [sys.executable, "-m", "norma", *map(str, arguments)],
            capture_output=True, text=True, timeout=180, check=False,
            cwd=cwd or tmp_path, env=environment,
        )
        return Run(completed.returncode, completed.stdout, completed.stderr)

    return run


@pytest.fixture
def without_ffmpeg(tmp_path: Path) -> str:
    """A PATH holding no tool at all; the Python interpreter is started by absolute path."""
    empty = tmp_path / "empty-path"
    empty.mkdir()
    return str(empty)


@pytest.fixture
def demos(tmp_path: Path):
    """Copy generated inputs into a caller-owned folder; originals stay in the fixture cache."""

    def build(members: dict[str, Path], folder: str = "demos") -> Path:
        root = tmp_path / folder
        root.mkdir(parents=True, exist_ok=True)
        for name, source in members.items():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return root

    return build


def snapshot(root: Path) -> dict[str, tuple]:
    """Identity ledger for a tree: type, size, mtime_ns, SHA-256 or link target for every entry."""
    ledger: dict[str, tuple] = {}
    for current, folders, names in os.walk(root, followlinks=False):
        for name in sorted(folders + names):
            path = Path(current) / name
            relative = str(path.relative_to(root))
            status = path.lstat()
            if path.is_symlink():
                ledger[relative] = ("link", os.readlink(path), status.st_mtime_ns)
            elif path.is_dir():
                ledger[relative] = ("dir", status.st_mtime_ns)
            else:
                ledger[relative] = ("file", status.st_size, status.st_mtime_ns,
                                    hashlib.sha256(path.read_bytes()).hexdigest())
    return ledger


@pytest.fixture
def ledger(tmp_path: Path):
    """Callable that takes the identity ledger of the whole test sandbox."""
    return lambda: snapshot(tmp_path)
