"""Shared hermetic fixtures and environment evidence (TK-0704, stage S0).

The audit hook refuses detectable Python filesystem writes outside tmp_path
and Music reads/writes before they happen. It is inactive outside function
fixtures and is not an OS sandbox: native code, child processes and
filesystem races still need the suite identity ledger
(hermeticity rules). Tests must pass only temporary paths to real tools.

Two deliberate rules, both narrow (TK-0707, stage S2):

- CR-041: a file descriptor, or a name relative to a directory descriptor, is
  resolved to its real path with ``fcntl.F_GETPATH`` and then judged like any
  other path, so the descriptor-based ``shutil.rmtree`` works inside tmp_path.
  Containment and the Music refusal are not relaxed; a descriptor whose path
  cannot be resolved is still refused.
- CR-044: the ``allow_linked_names`` fixture lets names with more than one
  link be renamed, linked or removed, for the two safety routes that hold
  hard-link preimages. Writes outside tmp_path stay refused and the rule is off
  unless a test asks for it.
No fixture data or Music-media-root discovery is introduced at S0.
"""

from __future__ import annotations

import fcntl
import os
import platform
import socket
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pytest


_ENVIRONMENT = pytest.StashKey[Mapping[str, str]]()
_ORIGINAL_HOME = Path.home().resolve()
_ACTIVE_GUARD: _PathGuard | None = None


class HermeticityError(AssertionError):
    """A test attempted an observable operation outside its permitted paths."""


def _descriptor_path(fd: int) -> Path:
    """Real path of an open descriptor (macOS ``F_GETPATH``); refuse when unknown."""
    try:
        raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))  # type: ignore[attr-defined]
    except (AttributeError, OSError) as exc:
        raise HermeticityError(f"unassessed file descriptor refused: {fd}") from exc
    text = os.fsdecode(raw.split(b"\0", 1)[0])
    if not text:
        raise HermeticityError(f"unassessed file descriptor refused: {fd}")
    return Path(text)


class _PathGuard:
    allow_linked_names = False

    def __init__(self, root: Path, home: Path) -> None:
        self.root = root.resolve()
        self.music_roots = (_ORIGINAL_HOME / "Music", home / "Music")

    def _refuse_music(self, path: Path) -> None:
        parts = tuple(part.casefold() for part in path.parts)
        if any(
            parts[:len(root.parts)] == tuple(part.casefold() for part in root.parts)
            for root in self.music_roots
        ) or any(
            part.casefold().endswith((".musiclibrary", ".itl")) for part in path.parts
        ):
            raise HermeticityError(f"Music library access refused: {path}")

    def check(self, value: object, *, write: bool = False, dir_fd: object = None) -> Path:
        """Check lexical and resolved ancestry; do not open a protected library."""
        if isinstance(value, int):
            # CR-041: judge a descriptor by the real path it names (macOS).
            path = _descriptor_path(value)
        elif not isinstance(value, (str, bytes, os.PathLike)):
            raise TypeError(f"not a filesystem path: {value!r}")
        else:
            path = Path(os.fsdecode(value))
        if not path.is_absolute() and dir_fd not in (None, -1):
            assert isinstance(dir_fd, int)
            path = _descriptor_path(dir_fd) / path
        # Check before normalisation too, so Music/../x is never traversed.
        lexical = path if path.is_absolute() else Path.cwd() / path
        self._refuse_music(lexical)
        resolved = lexical.resolve()
        self._refuse_music(resolved)
        if write and not resolved.is_relative_to(self.root):
            raise HermeticityError(f"write outside test temporary directory refused: {resolved}")
        if (write and not self.allow_linked_names and resolved.exists() and resolved.is_file()
                and resolved.stat().st_nlink > 1):
            raise HermeticityError(f"hard-linked write target refused: {resolved}")
        return resolved

    def audit(self, event: str, args: tuple[object, ...]) -> None:
        if event == "open":
            path, mode, flags = args
            assert isinstance(flags, int)
            # subprocess.DEVNULL opens the character sink read/write; it
            # does not create a file. No mutation operation gets this bypass.
            if path == os.devnull and not flags & (os.O_CREAT | os.O_TRUNC | os.O_EXCL):
                if stat.S_ISCHR(os.stat(os.devnull).st_mode):
                    return
            write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            # Reads through already-open fds have no assessable pathname;
            # ordinary fd reads are not claimed to be protected by this hook.
            if not isinstance(path, int) or write:
                self.check(path, write=write)
        elif event in {"os.listdir", "os.scandir"}:
            if args[0] is not None and not isinstance(args[0], int):
                self.check(args[0])
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.utime"}:
            self.check(args[0], write=True, dir_fd=args[-1])
        elif event in {"os.truncate", "os.chflags", "os.setxattr", "os.removexattr"}:
            self.check(args[0], write=True)
        elif event == "os.rename":
            self.check(args[0], write=True, dir_fd=args[2])
            self.check(args[1], write=True, dir_fd=args[3])
        elif event == "os.link":
            # An external inode must not gain a writable alias inside tmp_path.
            self.check(args[0], write=True, dir_fd=args[2])
            self.check(args[1], write=True, dir_fd=args[3])
        elif event == "os.symlink":
            destination = self.check(args[1], write=True, dir_fd=args[2])
            assert isinstance(args[0], (str, bytes, os.PathLike))
            target = Path(os.fsdecode(args[0]))
            self.check(target if target.is_absolute() else destination.parent / target)
        elif event == "os.chdir":
            self.check(args[0], write=True)
        elif event == "subprocess.Popen":
            # Reject explicit Music paths passed to real tools too. This
            # cannot interpret shell strings, scripts or arbitrary options
            # as a complete child-process filesystem sandbox.
            executable, argv, cwd, env = args
            if cwd is not None:
                self.check(cwd, write=True)
            if isinstance(argv, (list, tuple)):
                for value in argv:
                    if isinstance(value, (str, bytes, os.PathLike)):
                        path = Path(os.fsdecode(value))
                        self._refuse_music(path if path.is_absolute() else Path.cwd() / path)
                        if path.is_absolute():
                            self.check(path)
        elif event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"}:
            raise HermeticityError("network access attempted")


def _audit(event: str, args: tuple[object, ...]) -> None:
    if _ACTIVE_GUARD is not None:
        _ACTIVE_GUARD.audit(event, args)


# Audit hooks cannot be removed. Register once when this conftest is loaded;
# the fixture activates/deactivates the guard, including on failed tests.
sys.addaudithook(_audit)


def _environment(config: pytest.Config) -> Mapping[str, str]:
    if _ENVIRONMENT not in config.stash:
        ffmpeg = Path("/usr/local/bin/ffmpeg")
        try:
            result = subprocess.run(
                [str(ffmpeg), "-version"], capture_output=True, text=True,
                timeout=5, check=False, shell=False,
            )
            lines = result.stdout.splitlines()
            version = lines[0] if result.returncode == 0 and lines else f"unavailable (exit {result.returncode})"
        except (OSError, subprocess.TimeoutExpired) as exc:
            version = f"unavailable ({type(exc).__name__})"
        config.stash[_ENVIRONMENT] = MappingProxyType({
            "platform": platform.system(),
            "macos": platform.mac_ver()[0] or "not macOS",
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "ffmpeg_path": str(ffmpeg),
            "ffmpeg_version": version,
        })
    return config.stash[_ENVIRONMENT]


def pytest_report_header(config: pytest.Config) -> list[str]:
    evidence = _environment(config)
    return [
        f"Norma environment: {evidence['platform']} {evidence['macos']}; "
        f"{evidence['architecture']}; Python {evidence['python']}",
        f"Norma FFmpeg ({evidence['ffmpeg_path']}): {evidence['ffmpeg_version']}",
        "Norma hermeticity: temporary HOME/XDG/cwd; Python path/network guards; "
        "native/child operations require the suite identity ledger",
    ]


@pytest.fixture(scope="session")
def norma_environment(pytestconfig: pytest.Config) -> Mapping[str, str]:
    """Actual environment evidence shared with the report header, even under -q."""
    return _environment(pytestconfig)


@pytest.fixture(autouse=True)
def norma_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, norma_environment: Mapping[str, str],
) -> Iterator[Path]:
    """Redirect each test without overriding workstream-local fixture names."""
    global _ACTIVE_GUARD
    home = tmp_path / "home"
    folders = {
        "HOME": home,
        "XDG_CONFIG_HOME": home / ".config",
        "XDG_CACHE_HOME": home / ".cache",
        "XDG_DATA_HOME": home / ".local" / "share",
        "XDG_STATE_HOME": home / ".local" / "state",
        "TMPDIR": home / "tmp",
    }
    for name, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    for name in ("TMP", "TEMP"):
        monkeypatch.setenv(name, str(folders["TMPDIR"]))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(tempfile, "tempdir", str(folders["TMPDIR"]))
    monkeypatch.chdir(tmp_path)

    def refuse_network(*args: object, **kwargs: object) -> None:
        raise HermeticityError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)

    original_open = os.open

    def guarded_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes], flags: int,
        mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> int:
        # The CPython open audit event omits dir_fd. Refuse unassessed
        # relative descriptors before that event can misread the path as cwd.
        if _ACTIVE_GUARD is not None and dir_fd is not None:
            write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            _ACTIVE_GUARD.check(path, write=write, dir_fd=dir_fd)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", guarded_open)
    previous = _ACTIVE_GUARD
    _ACTIVE_GUARD = _PathGuard(tmp_path, home)
    try:
        yield tmp_path
    finally:
        # Disable before monkeypatch restores the original working directory.
        _ACTIVE_GUARD = previous


@pytest.fixture
def allow_linked_names(norma_sandbox: Path) -> Iterator[None]:
    """CR-044: let names with several links be renamed, linked or removed in tmp_path.

    This is exactly the state of an output while its hard-link preimage is held
    (safety § 5.2) or of a candidate after the hard-link publication route. Those
    operations change names, not the inode's bytes. Only the link-count rule is
    lifted: containment in tmp_path, the Music refusal and the network guard stay.
    A local fixture of the same name (tests/unit/safety/conftest.py) still wins there.
    """
    assert _ACTIVE_GUARD is not None
    _ACTIVE_GUARD.allow_linked_names = True
    try:
        yield
    finally:
        _ACTIVE_GUARD.allow_linked_names = False


@pytest.fixture
def guard_music_path(norma_sandbox: Path) -> Callable[[object], Path]:
    """Explicit preflight for tool input paths; never inspect a real library."""
    assert _ACTIVE_GUARD is not None
    return _ACTIVE_GUARD.check
