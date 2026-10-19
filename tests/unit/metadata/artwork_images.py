"""Image and fake-sips builders for the artwork and sips tests (TK-0609, stage S4).

Not a test module: ``test_artwork*.py`` and ``test_sips*.py`` load it by path
(``--import-mode=importlib`` has no package imports). Images are built with the
standard library (PNG) or by the real ``/usr/bin/sips`` (TESTING § 4.7 recipes),
always inside the caller's ``tmp_path``. Fakes are shell scripts in ``tmp_path``.
"""

from __future__ import annotations

import os
import stat
import struct
import subprocess
import textwrap
import zlib
from pathlib import Path

SIPS = Path("/usr/bin/sips")
HAVE_SIPS = SIPS.is_file() and os.access(SIPS, os.X_OK)
CMYK_PROFILE = Path("/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc")
GRAY_PROFILE = Path("/System/Library/ColorSync/Profiles/Generic Gray Profile.icc")


def png_bytes(width: int, height: int, colour_type: int = 2, pixel: tuple[int, ...] = (40, 120, 200)) -> bytes:
    """A seedless 8-bit PNG: colour type 0 grey, 2 RGB or 6 RGBA, every pixel the same."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    row = b"\x00" + bytes(pixel) * width
    raw = row * height
    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def run_sips(*args: object) -> None:
    result = subprocess.run([str(SIPS), *map(str, args)], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr


def convert(source: Path, dest: Path, *args: object) -> Path:
    """The real sips writing ``dest`` from ``source`` with extra arguments."""
    run_sips(*args, source, "--out", dest)
    assert dest.is_file()
    return dest


def pad_to(path: Path, size: int) -> None:
    """Pad a valid image with trailing zero bytes to exactly ``size`` bytes (decoders ignore the tail)."""
    current = path.stat().st_size
    assert current <= size
    with open(path, "ab") as handle:
        handle.write(b"\0" * (size - current))


def fake_tool(tmp_path: Path, body: str, name: str = "sips") -> Path:
    """An executable ``/bin/sh`` script standing in for sips."""
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


INSPECT_OK = """
    case "$1" in
      --version) echo sips-fake-1; exit 0;;
    esac
"""


def fake_with(tmp_path: Path, tail: str, name: str = "sips") -> Path:
    """A fake that answers ``--version`` and then runs ``tail``."""
    return fake_tool(tmp_path, INSPECT_OK + textwrap.dedent(tail), name)
