"""Local marker and skip rule for real-tool adapter tests (tests/conftest.py belongs to WS-07)."""

from __future__ import annotations

from pathlib import Path

import pytest

REAL_TOOLS = (Path("/usr/local/bin/ffmpeg"), Path("/usr/local/bin/ffprobe"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_ffmpeg: runs the installed /usr/local/bin/ffmpeg and ffprobe, writing only to tmp_path"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if all(tool.is_file() for tool in REAL_TOOLS):
        return
    skip = pytest.mark.skip(reason="installed FFmpeg/ffprobe not found at /usr/local/bin")
    for item in items:
        if "real_ffmpeg" in item.keywords:
            item.add_marker(skip)
