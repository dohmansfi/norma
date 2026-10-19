"""Local helpers for WS-06 metadata reader tests (TK-0605, stage S1).

Tags are written by independent writers inside ``tmp_path``: mutagen in the
test, FFmpeg ``-metadata`` flags, or bytes built here by hand. The reader under
test never writes. Short generated sources are cached in the first requesting
test's sandbox and copied into later tests before tagging (D-055: nothing is
generated inside the repository). Hermetic HOME/TMPDIR/cwd and the Music
refusal come from ``tests/conftest.py``.
"""

from __future__ import annotations

import importlib.util
import shutil
import struct
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma.adapters import ffmpeg

_spec = importlib.util.spec_from_file_location(
    "norma_metadata_test_fixtures", Path(__file__).parents[2] / "fixtures" / "make_fixtures.py",
)
assert _spec is not None and _spec.loader is not None
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
fixture_path = fixtures.fixture_path

# kind -> (suffix, C-02 container, encoder options)
KINDS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "wav": (".wav", "wav", ("-c:a", "pcm_s16le")),
    "aiff": (".aif", "aiff", ("-c:a", "pcm_s16be")),
    "flac": (".flac", "flac", ("-c:a", "flac")),
    "mp3": (".mp3", "mp3", ("-c:a", "libmp3lame", "-q:a", "2")),
    "alac": (".m4a", "mp4", ("-c:a", "alac")),
    "aac": (".m4a", "mp4", ("-c:a", "aac_at", "-b:a", "128k")),
}

_CACHE: dict[str, Path] = {}


def _generate(kind: str, destination: Path, extra: tuple[str, ...] = ()) -> None:
    suffix, _, options = KINDS[kind]
    executable = ffmpeg.locate_ffmpeg()
    ffmpeg.check_result(ffmpeg.run_process(
        ffmpeg.ffmpeg_argv(executable, "-y", "-f", "lavfi", "-i", "sine=frequency=997:duration=0.2:sample_rate=48000",
                           *extra, "-map", "0:a:0", "-map_metadata", "-1", "-map_chapters", "-1", "-ac", "2",
                           "-fflags", "+bitexact", "-flags:a", "+bitexact", *options, str(destination)),
        executable=executable, deadline_s=60,
    ))


@pytest.fixture
def audio(tmp_path: Path) -> Callable[..., Path]:
    """``audio(kind, name=None)``: a fresh untagged 0.2 s source copied into ``tmp_path``."""
    counter = iter(range(1, 10_000))

    def make(kind: str, name: str | None = None) -> Path:
        suffix = KINDS[kind][0]
        cached = _CACHE.get(kind)
        if cached is None or not cached.exists():
            cache = tmp_path / "cache"
            cache.mkdir(exist_ok=True)
            cached = cache / f"{kind}{suffix}"
            _generate(kind, cached)
            _CACHE[kind] = cached
        target = tmp_path / (name or f"{kind}-{next(counter)}{suffix}")
        shutil.copyfile(cached, target)
        return target

    return make


@pytest.fixture
def encode(tmp_path: Path) -> Callable[..., Path]:
    """``encode(kind, name, *ffmpeg_args)``: a source written by FFmpeg with extra arguments."""

    def make(kind: str, name: str, *extra: str) -> Path:
        target = tmp_path / name
        _generate(kind, target, tuple(extra))
        return target

    return make


def container(kind: str) -> str:
    return KINDS[kind][1]


# --- Hand-built bytes (independent writers) ------------------------------------------------


def syncsafe(value: int) -> bytes:
    return bytes([(value >> 21) & 0x7F, (value >> 14) & 0x7F, (value >> 7) & 0x7F, value & 0x7F])


def id3_frame(frame_id: str, body: bytes, major: int = 4, flags: int = 0) -> bytes:
    raw_id = frame_id.encode("ascii")
    if major == 2:
        return raw_id + len(body).to_bytes(3, "big") + body
    size = syncsafe(len(body)) if major == 4 else struct.pack(">I", len(body))
    return raw_id + size + struct.pack(">H", flags) + body


def id3_tag(major: int, frames: list[bytes], *, flags: int = 0, padding: int = 16) -> bytes:
    body = b"".join(frames) + bytes(padding)
    return b"ID3" + bytes([major, 0, flags]) + syncsafe(len(body)) + body


def riff_chunk(chunk_id: bytes, payload: bytes, order: str = "little") -> bytes:
    size = struct.pack("<I" if order == "little" else ">I", len(payload))
    return chunk_id + size + payload + (b"\x00" if len(payload) & 1 else b"")


def append_chunks(path: Path, chunks: list[bytes]) -> None:
    """Append chunks to a RIFF or FORM file and fix the outer size."""
    data = path.read_bytes() + b"".join(chunks)
    order = "<I" if data[:4] in (b"RIFF", b"RF64") else ">I"
    path.write_bytes(data[:4] + struct.pack(order, len(data) - 8) + data[8:])


def info_list(items: list[tuple[bytes, bytes]]) -> bytes:
    return riff_chunk(b"LIST", b"INFO" + b"".join(riff_chunk(key, value) for key, value in items))


def flac_blocks(path: Path) -> tuple[bytes, list[tuple[int, bytes]], bytes]:
    """``(prefix up to fLaC, [(type, body)], audio frames)`` of a FLAC file."""
    data = path.read_bytes()
    start = data.index(b"fLaC")
    position = start + 4
    blocks = []
    while True:
        header = data[position]
        length = int.from_bytes(data[position + 1:position + 4], "big")
        blocks.append((header & 0x7F, data[position + 4:position + 4 + length]))
        position += 4 + length
        if header & 0x80:
            break
    return data[:start], blocks, data[position:]


def write_flac_blocks(path: Path, prefix: bytes, blocks: list[tuple[int, bytes]], frames: bytes) -> None:
    out = bytearray(prefix + b"fLaC")
    for index, (kind, body) in enumerate(blocks):
        last = 0x80 if index == len(blocks) - 1 else 0
        out += bytes([last | kind]) + len(body).to_bytes(3, "big") + body
    path.write_bytes(bytes(out) + frames)


def vorbis_block(comments: list[bytes], vendor: bytes = b"norma-test") -> bytes:
    out = struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", len(comments))
    for comment in comments:
        out += struct.pack("<I", len(comment)) + comment
    return out


@pytest.fixture
def build() -> SimpleNamespace:
    return SimpleNamespace(
        syncsafe=syncsafe, id3_frame=id3_frame, id3_tag=id3_tag, riff_chunk=riff_chunk,
        append_chunks=append_chunks, info_list=info_list, flac_blocks=flac_blocks,
        write_flac_blocks=write_flac_blocks, vorbis_block=vorbis_block, container=container,
        fixtures=fixtures,
    )
