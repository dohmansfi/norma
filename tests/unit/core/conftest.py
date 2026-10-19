"""Local fixtures for WS-04 core tests (tests/conftest.py belongs to WS-07).

Hermeticity (test plan): every test runs with HOME, XDG
folders, TMPDIR and the working directory redirected into its own tmp_path,
and with Python socket connections refused (TST-C-05). Real-tool tests are
marked ``real_ffmpeg`` and write only under tmp_path.
"""

from __future__ import annotations

import socket
import stat
import sys
import tempfile
from pathlib import Path

import pytest

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
REAL_FFPROBE = Path("/usr/local/bin/ffprobe")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_ffmpeg: runs the installed /usr/local/bin/ffmpeg and ffprobe, writing only to tmp_path"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if REAL_FFMPEG.is_file() and REAL_FFPROBE.is_file():
        return
    skip = pytest.mark.skip(reason="installed FFmpeg/ffprobe not found at /usr/local/bin")
    for item in items:
        if "real_ffmpeg" in item.keywords:
            item.add_marker(skip)


class NetworkAttempt(AssertionError):
    pass


@pytest.fixture(autouse=True)
def hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    for sub in ("", ".config", ".cache", ".local/share", "tmp"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local/share"))
    monkeypatch.setenv("TMPDIR", str(home / "tmp"))
    monkeypatch.setattr(tempfile, "tempdir", str(home / "tmp"))
    monkeypatch.chdir(tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise NetworkAttempt(f"network access attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return tmp_path


@pytest.fixture
def make_tool(tmp_path: Path):
    """Write a fake tool: an executable Python script standing in for ffmpeg/ffprobe."""

    def factory(name: str, body: str) -> Path:
        tools = tmp_path / "tools"
        tools.mkdir(exist_ok=True)
        path = tools / name
        path.write_text(f"#!{sys.executable}\nimport sys, os, time, json\n{body}\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return path

    return factory
