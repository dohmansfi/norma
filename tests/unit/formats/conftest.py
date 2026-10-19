"""Local fixtures for WS-02 format tests (tests/conftest.py belongs to WS-07).

The shared ``norma_sandbox`` fixture already confines every test to its own
temporary folder (hermeticity rule). This file only registers the
``real_ffmpeg`` marker for this folder too, so ``pytest tests/unit/formats``
works on its own under ``--strict-markers``, and skips real-tool tests when
the installed FFmpeg is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_ffmpeg: runs the installed /usr/local/bin/ffmpeg and ffprobe, writing only to tmp_path"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if REAL_FFMPEG.is_file():
        return
    skip = pytest.mark.skip(reason="installed FFmpeg not found at /usr/local/bin")
    for item in items:
        if "real_ffmpeg" in item.keywords and Path(str(item.fspath)).parent == Path(__file__).parent:
            item.add_marker(skip)
