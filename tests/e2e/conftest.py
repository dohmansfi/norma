"""Shared helpers for the S2 end-to-end suite (TK-0707, WS-07).

Every test runs the real ``norma`` entry point (``python -m norma``) as a child process in the
hermetic sandbox from tests/conftest.py. Audio is generated, never stored: the C-12 recipes of
tests/fixtures/make_fixtures.py plus the untagged derivatives built here. Nothing in this folder
imports an adapter, the pipeline or the CLI to *drive* Norma; the layer under test is the command
line. The FFmpeg ``ebur128`` filter and ``pyloudnorm``/``scipy`` serve as independent meters
(test independence rule).

The S2 build accepts untagged inputs only (D-066). FFmpeg writes an ``encoder`` tag unless asked
not to, which this build would (rightly) refuse, so the derivatives here are made with
``-map_metadata -1 -fflags +bitexact`` and are checked to carry no tag at all.

Stdout is not a terminal in a child process started with captured pipes, so the commands print
the machine-readable tab-separated form (CLI § 7.1).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from norma.adapters import ffmpeg

_spec = importlib.util.spec_from_file_location(
    "norma_e2e_test_fixtures", Path(__file__).parents[1] / "fixtures" / "make_fixtures.py",
)
assert _spec is not None and _spec.loader is not None
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
fixture_path = fixtures.fixture_path

FILE_STATUSES = ("OK", "WARN", "SKIP", "FAIL")
_COUNTER = re.compile(r"\d+/\d+")
_ESCAPES = {"\\": "\\", "t": "\t", "r": "\r", "n": "\n"}
AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".aac", ".caf"}


def unescape(cell: str) -> str:
    """Invert ``report.escape_cell`` in one pass (backslash, tab, CR, LF)."""
    return re.sub(r"\\([\\trn])", lambda match: _ESCAPES[match[1]], cell)


@pytest.fixture(scope="session")
def fixture_module():
    """The WS-07 recipe module, for its canonical constants and ``decode``."""
    return fixtures


# --- Running the real command -------------------------------------------------------------


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
        """Result rows: status, ``n/N``, path, I, TP, gain, mode, out I, out TP, out path, note."""
        return [line for line in self.lines
                if line[0] in FILE_STATUSES and len(line) > 1 and _COUNTER.fullmatch(line[1])]

    @property
    def summary(self) -> dict[str, str]:
        found = [line for line in self.lines if line[0] == "SUMMARY"]
        assert len(found) == 1, self.stdout
        return dict(cell.split("=", 1) for cell in found[0][1:])

    def row(self, name: str) -> list[str]:
        """The one result row whose input path ends in ``name``."""
        found = [row for row in self.files if row[2].endswith(name)]
        assert len(found) == 1, (name, self.stdout)
        return found[0]


def _environment(path: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    environment["NO_COLOR"] = "1"
    if path is not None:
        environment["PATH"] = path
    return environment


@pytest.fixture
def norma(tmp_path: Path) -> Callable[..., Run]:
    """Run ``python -m norma`` to completion in the sandbox and return its streams."""

    def run(*arguments: object, cwd: Path | None = None, umask: int | None = None) -> Run:
        previous = os.umask(umask) if umask is not None else None
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "norma", *map(str, arguments)],
                capture_output=True, text=True, timeout=240, check=False,
                cwd=cwd or tmp_path, env=_environment(None),
            )
        finally:
            if previous is not None:
                os.umask(previous)
        return Run(completed.returncode, completed.stdout, completed.stderr)

    return run


@pytest.fixture
def start(tmp_path: Path) -> Iterator[Callable[..., subprocess.Popen]]:
    """Start ``python -m norma`` without waiting, for signal tests."""
    started: list[subprocess.Popen] = []

    def spawn(*arguments: object) -> subprocess.Popen:  # noqa: D401
        process = subprocess.Popen(
            [sys.executable, "-m", "norma", *map(str, arguments)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=tmp_path, env=_environment(None),
        )
        started.append(process)
        return process

    yield spawn
    for process in started:
        if process.poll() is None:
            process.kill()
        process.communicate()


def wait_until(predicate: Callable[[], bool], timeout: float, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def read_report(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV report, keyed by the 53 C-09 column names."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Identity ledgers ---------------------------------------------------------------------


def snapshot(root: Path) -> dict[str, tuple]:
    """Identity ledger for a tree.

    Files carry size, mtime_ns and SHA-256; links their target; folders only their existence,
    because a folder's own mtime changes whenever a legitimate sibling appears.
    """
    ledger: dict[str, tuple] = {}
    for current, folders, names in os.walk(root, followlinks=False):
        for name in sorted(folders + names):
            path = Path(current) / name
            relative = str(path.relative_to(root))
            status = path.lstat()
            if path.is_symlink():
                ledger[relative] = ("link", os.readlink(path))
            elif path.is_dir():
                ledger[relative] = ("dir",)
            else:
                ledger[relative] = ("file", status.st_size, status.st_mtime_ns,
                                    hashlib.sha256(path.read_bytes()).hexdigest())
    return ledger


@pytest.fixture
def ledger(tmp_path: Path) -> Callable[[], dict[str, tuple]]:
    """Callable that takes the identity ledger of the whole test sandbox."""
    return lambda: snapshot(tmp_path)


def difference(before: dict[str, tuple], after: dict[str, tuple]) -> dict[str, set[str]]:
    return {
        "added": set(after) - set(before),
        "removed": set(before) - set(after),
        "changed": {name for name in set(before) & set(after) if before[name] != after[name]},
    }


# --- Generated untagged inputs ------------------------------------------------------------


def _bitexact_encode(source: Path, destination: Path, *options: str) -> None:
    executable = ffmpeg.locate_ffmpeg()
    result = subprocess.run(
        [str(executable.path), "-v", "error", "-y", "-i", str(source), "-map", "0:a:0",
         "-map_metadata", "-1", "-fflags", "+bitexact", *options, str(destination)],
        capture_output=True, text=True, check=False, timeout=240,
    )
    assert result.returncode == 0, result.stderr


def tone_file(path: Path, *, level: float = -20.0, seconds: float = 5.0, rate: int = 48000,
              subtype: str = "PCM_24", frequency: float = 1000.0) -> Path:
    """Dual-mono sine whose per-channel peak is ``level`` dBFS (so I is ``level`` LUFS)."""
    n = np.arange(round(seconds * rate), dtype=np.float64)
    x = 10 ** (level / 20) * np.sin(2 * np.pi * frequency * n / rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.column_stack((x, x)), rate, subtype=subtype)
    return path


class UntaggedCache:
    """Session cache of untagged derivatives; built in the first user's sandbox, then read-only."""

    BUILDERS = ("wav24", "float", "wav16", "flac24", "aiff24", "alac24")

    def __init__(self, base: Callable[..., Path]) -> None:
        self._base = base
        self._paths: dict[str, Path] = {}

    def __call__(self, kind: str) -> Path:
        if kind not in self._paths:
            self._paths[kind] = self._build(kind)
        return self._paths[kind]

    def _build(self, kind: str) -> Path:
        folder = Path.cwd() / "untagged"
        folder.mkdir(exist_ok=True)
        if kind == "wav24":
            return self._base("F-04")
        if kind == "float":
            return self._base("F-06")
        source = self._base("F-04")
        if kind == "wav16":
            destination = folder / "wav16.wav"
            _bitexact_encode(source, destination, "-af", "aresample=dither_method=none",
                             "-ar", "44100", "-c:a", "pcm_s16le")
        elif kind == "flac24":
            destination = folder / "flac24.flac"
            _bitexact_encode(source, destination, "-c:a", "flac", "-sample_fmt", "s32",
                             "-bits_per_raw_sample", "24")
        elif kind == "aiff24":
            destination = folder / "aiff24.aif"
            _bitexact_encode(source, destination, "-c:a", "pcm_s24be")
        elif kind == "alac24":
            destination = folder / "alac24.m4a"
            _bitexact_encode(source, destination, "-c:a", "alac", "-sample_fmt", "s32p",
                             "-bits_per_raw_sample", "24")
        else:
            raise KeyError(kind)
        return destination


@pytest.fixture(scope="session")
def untagged(fixture_path) -> UntaggedCache:
    """``untagged("flac24")``: a path to an untagged 30 s derivative of F-04 (read only)."""
    return UntaggedCache(fixture_path)


@pytest.fixture
def demos(tmp_path: Path) -> Callable[..., Path]:
    """Copy generated inputs into a caller-owned folder; the cached originals stay untouched."""

    def build(members: dict[str, Path], folder: str = "demos") -> Path:
        root = tmp_path / folder
        root.mkdir(parents=True, exist_ok=True)
        for name, source in members.items():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return root

    return build


# --- Independent meters and decoders ------------------------------------------------------


def decode(path: Path) -> tuple[np.ndarray, int]:
    """Decoded samples as float64 ``(frames, channels)`` and the sample rate."""
    if path.suffix.lower() == ".m4a":
        probe = subprocess.run(
            [str(ffmpeg.locate_ffmpeg().path.parent / "ffprobe"), "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.split(",")
        rate, channels = int(probe[0]), int(probe[1])
        return fixtures.decode(path).reshape(-1, channels), rate
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    return samples, rate


def pyloudnorm_meter(samples: np.ndarray, rate: int) -> tuple[float, float]:
    """K-weighted integrated loudness and a 4x polyphase true-peak estimate (dBTP)."""
    integrated = pyln.Meter(rate, filter_class="K-weighting", block_size=0.400).integrated_loudness(samples)
    peak = np.max(np.abs(resample_poly(samples, 4, 1, axis=0, window=("kaiser", 12.0), padtype="constant")))
    return float(integrated), float(20 * np.log10(peak))


def ebur128_meter(path: Path) -> tuple[float, float]:
    """Integrated loudness and true peak from FFmpeg's ``ebur128`` summary (an independent meter)."""
    result = subprocess.run(
        [str(ffmpeg.locate_ffmpeg().path), "-hide_banner", "-nostats", "-i", str(path),
         "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, check=True, timeout=240,
    )
    summary = result.stderr.rsplit("Summary:", 1)[1]
    found_i, found_peak = re.search(r"I:\s+(-?[\d.]+) LUFS", summary), re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    assert found_i is not None and found_peak is not None, summary
    integrated, peak = float(found_i[1]), float(found_peak[1])
    return integrated, peak


def stream_facts(path: Path) -> dict[str, str]:
    """``codec_name``, ``bits_per_raw_sample``, ``sample_rate``, ``channels`` from ffprobe."""
    probe = ffmpeg.locate_ffmpeg().path.parent / "ffprobe"
    text = subprocess.run(
        [str(probe), "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,bits_per_raw_sample,sample_rate,channels", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


@pytest.fixture(scope="session")
def tools() -> SimpleNamespace:
    """Helper functions, reachable without importing this conftest (the suite uses importlib mode)."""
    return SimpleNamespace(
        decode=decode, pyloudnorm=pyloudnorm_meter, ebur128=ebur128_meter, facts=stream_facts,
        report=read_report, tone=tone_file, snapshot=snapshot, difference=difference,
        audio_files=audio_files, wait_until=wait_until, sha256=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
