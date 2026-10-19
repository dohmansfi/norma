"""Local helpers for WS-05 discovery tests (TK-0504, stage S1).

Everything is built by hand inside ``tmp_path``: tiny hand-written containers and a fake
ffprobe. The shared fixture catalogue is deliberately not used. Hermetic HOME/TMPDIR/cwd and
the Music refusal come from ``tests/conftest.py``.
"""

from __future__ import annotations

import json
import stat
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma.adapters.ffprobe import Prober, locate_ffprobe


# --- Hand-written containers -----------------------------------------------------------


def wav_bytes(frames: int = 100, *, channels: int = 2, bytes_per_sample: int = 2, tag: int = 1) -> bytes:
    align = channels * bytes_per_sample
    data = bytes(frames * align)
    fmt = struct.pack("<HHIIHH", tag, channels, 44100, 44100 * align, align, bytes_per_sample * 8)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def aiff_bytes(frames: int = 100, *, channels: int = 2, bits: int = 16, aifc: str | None = None) -> bytes:
    comm = struct.pack(">HIH", channels, frames, bits) + b"\x40\x0e\xac\x44" + bytes(6)  # 44100 Hz, 80-bit
    if aifc is not None:
        comm += aifc.encode("ascii") + b"\x00"  # compression id and an empty pascal name
        comm += b"\x00" * (len(comm) % 2)
    data = bytes(frames * channels * ((bits + 7) // 8))
    ssnd = struct.pack(">II", 0, 0) + data
    body = (b"AIFC" if aifc else b"AIFF") + b"COMM" + struct.pack(">I", len(comm)) + comm
    body += b"SSND" + struct.pack(">I", len(ssnd)) + ssnd + bytes(len(ssnd) % 2)
    return b"FORM" + struct.pack(">I", len(body)) + body


FLAC = b"fLaC" + bytes(64)
MP4 = struct.pack(">I", 24) + b"ftypM4A " + bytes(16)
CAF = b"caff\x00\x01\x00\x00" + bytes(32)
MP3 = b"\xff\xfb\x90\x00" + bytes(64)  # MPEG-1 layer III, 128 kbit/s, 44.1 kHz
ADTS = b"\xff\xf1\x50\x80" + bytes(64)


def id3(payload_after: bytes, *, tag_size: int = 20, footer: bool = False) -> bytes:
    sizes = bytes([(tag_size >> 21) & 0x7F, (tag_size >> 14) & 0x7F, (tag_size >> 7) & 0x7F, tag_size & 0x7F])
    header = b"ID3\x04\x00" + bytes([0x10 if footer else 0]) + sizes
    return header + bytes(tag_size) + (bytes(10) if footer else b"") + payload_after


# --- Tree building ---------------------------------------------------------------------


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    root.mkdir()
    return root


@pytest.fixture
def put() -> Callable[..., Path]:
    def write(path: Path, data: bytes = b"data") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    return write


# --- Fake ffprobe ----------------------------------------------------------------------

_SCRIPT = """#!{python}
import json, os, sys
if "-version" in sys.argv:
    print("ffprobe version 0.0-test Copyright the tests")
    sys.exit(0)
table = json.load(open({table!r}))
entry = table.get(os.path.basename(sys.argv[-1]))
if entry is None:
    sys.stderr.write("no canned answer\\n")
    sys.exit(1)
if entry.get("rc"):
    sys.stderr.write(entry.get("stderr", "boom") + "\\n")
    sys.exit(entry["rc"])
print(json.dumps(entry["doc"]))
"""


def audio_stream(**changes: object) -> dict:
    base = {"index": 0, "codec_name": "pcm_s16le", "codec_type": "audio", "sample_fmt": "s16",
            "sample_rate": "44100", "channels": 2, "channel_layout": "stereo", "bits_per_sample": 16,
            "duration": "1.0"}
    base.update(changes)
    return base


def probe_document(*streams: dict, format_name: str = "wav") -> dict:
    return {"streams": list(streams), "chapters": [], "format": {"format_name": format_name, "duration": "1.0"}}


class FakeProbe:
    """A scripted ffprobe: ``answer(name, ...)`` fixes what it says about a file basename."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(exist_ok=True)
        self.table_path = directory / "table.json"
        self.table: dict[str, dict] = {}
        self._save()
        script = directory / "ffprobe"
        script.write_text(_SCRIPT.format(python=sys.executable, table=str(self.table_path)), encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.prober: Prober = locate_ffprobe(script)

    def _save(self) -> None:
        self.table_path.write_text(json.dumps(self.table), encoding="utf-8")

    def answer(self, name: str, document: dict | None = None, *, rc: int = 0, stderr: str = "boom") -> None:
        self.table[name] = {"rc": rc, "stderr": stderr} if rc else {"doc": document}
        self._save()


@pytest.fixture
def fake_probe(tmp_path: Path) -> FakeProbe:
    return FakeProbe(tmp_path / "tools")


@pytest.fixture
def build() -> SimpleNamespace:
    """The hand-written containers above, reachable without importing this conftest."""
    return SimpleNamespace(
        wav=wav_bytes, aiff=aiff_bytes, id3=id3, flac=FLAC, mp4=MP4, caf=CAF, mp3=MP3, adts=ADTS,
        stream=audio_stream, document=probe_document,
    )


# --- Write-side helpers (TK-0506, stage S2) ----------------------------------------------


@pytest.fixture
def allow_linked_names(norma_sandbox: Path) -> object:
    """Let namespace operations act on multiply-linked names inside tmp_path.

    The suite guard (tests/conftest.py) refuses any rename, link or remove whose target name has
    more than one link, which is exactly the state of an output while its hard-link preimage is
    held (safety § 5.2) or of a candidate after the hard-link publication route. Those operations
    change names, not the inode's bytes. This fixture keeps every other rule, including the
    tmp_path containment of writes, and is used only by tests of those two routes.
    """
    import sys

    module = next(m for m in list(sys.modules.values())
                  if getattr(m, "_PathGuard", None) is not None and hasattr(m, "_ACTIVE_GUARD"))
    active = module._ACTIVE_GUARD
    assert active is not None

    class Relaxed(module._PathGuard):  # type: ignore[name-defined, misc]
        def check(self, value: object, *, write: bool = False, dir_fd: object = None) -> Path:
            resolved = super().check(value, write=False, dir_fd=dir_fd)
            if write and not resolved.is_relative_to(self.root):
                raise module.HermeticityError(f"write outside test temporary directory refused: {resolved}")
            return resolved

    relaxed = Relaxed.__new__(Relaxed)
    relaxed.__dict__.update(active.__dict__)
    module._ACTIVE_GUARD = relaxed
    try:
        yield relaxed
    finally:
        module._ACTIVE_GUARD = active
