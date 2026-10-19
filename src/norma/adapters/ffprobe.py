"""ffprobe adapter: local probe of one file into a C-02 ``ProbeResult``.

The probe runs after WS-05's signature/structure check, as an argv list with
``-protocol_whitelist file`` and an absolute path, bounded at 60 s (safety § 6,
ADR-0018). Outcomes:

- no audio stream after a successful probe: ``SkipFile(not_audio)``;
- unreadable or malformed probe output: ``FileFailure(probe_error)``;
- deadline expiry: ``FileFailure(timeout)``;
- access denial or a vanished source: ``permission_denied`` / ``source_changed``;
- the executable vanished or changed after preflight: ``probe_error`` (C-13).
- ``force_kill`` escalates active process groups immediately to SIGKILL and
  calls end ``Cancelled`` after reaping, including during timeout teardown.

An unsupported codec is still returned as a ``ProbeResult`` with
``codec_class="unsupported"`` so the caller can report the skip. Imports only
the standard library and ``norma.models``/``norma.errors``.
"""

from __future__ import annotations

import errno
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from norma.errors import Cancelled, EnvironmentError_, FileFailure, SkipFile
from norma.models import Identity, ProbeResult

PROBE_DEADLINE_S = 60.0
_STDOUT_LIMIT = 64 * 1024 * 1024
_TAIL_LINES = 20
TERMINATE_GRACE_S = 2.0
_READ_CHUNK = 64 * 1024
_MAX_LINE_BYTES = 1024 * 1024
"""A line longer than this is emitted in pieces, so a newline-free stream stays bounded."""
_SETTLE_POLL_S = 0.01
_POLL_S = 0.05

LineCallback = Callable[[str], None]

# Canonical container names for ffprobe's demuxer format_name values.
_CONTAINERS = {
    "mov,mp4,m4a,3gp,3g2,mj2": "mp4",
    "aac": "adts",
}
_LINEAR_PCM_RE = re.compile(r"^pcm_[suf](8|16|24|32|64)(le|be)?(_planar)?$")
_LOSSLESS_CODECS = frozenset({"flac", "alac"})
_LOSSY_CODECS = frozenset({"mp3", "aac"})
_FLOAT_FORMATS = frozenset({"flt", "fltp", "dbl", "dblp"})


def codec_class(codec: str) -> str:
    """C-02 / formats F1 classification of an ffprobe codec_name."""
    if codec in _LOSSLESS_CODECS or _LINEAR_PCM_RE.match(codec):
        return "lossless"
    if codec in _LOSSY_CODECS:
        return "lossy"
    return "unsupported"


def canonical_container(format_name: str) -> str:
    """One canonical container name, never a comma-joined alias list."""
    if format_name in _CONTAINERS:
        return _CONTAINERS[format_name]
    return format_name.split(",")[0]


@dataclass(frozen=True)
class Prober:
    """A resolved ffprobe executable with its identity and version at preflight."""

    path: Path
    identity: Identity
    version: str


def locate_ffprobe(name: str | Path = "ffprobe", *, search_path: str | None = None) -> Prober:
    """Resolve ffprobe and read its version; failure is an environment error (exit 3)."""
    text = str(name)
    if os.sep in text:
        path = Path(text)
        if not (path.is_absolute() and path.is_file() and os.access(path, os.X_OK)):
            raise EnvironmentError_(f"ffprobe not found or not executable: {text}")
    else:
        found = shutil.which(text, path=search_path)
        if found is None:
            raise EnvironmentError_(f"required tool not found on PATH: {text}")
        path = Path(found).absolute()
    info = os.stat(path)
    try:
        returncode, stdout, tail = _run((str(path), "-hide_banner", "-version"), PROBE_DEADLINE_S)
    except (OSError, _Timeout) as exc:
        raise EnvironmentError_(f"ffprobe at {path} is unusable: {exc}") from None
    match = re.match(r"^ffprobe version (\S+)", stdout.decode("utf-8", "replace"))
    if returncode != 0 or match is None:
        raise EnvironmentError_(f"ffprobe at {path} gave no usable version ({' | '.join(tail)})")
    return Prober(path, (info.st_dev, info.st_ino), match.group(1))


def probe_argv(prober: Prober, source: Path) -> tuple[str, ...]:
    if not isinstance(source, Path) or not source.is_absolute():
        raise ValueError(f"probe inputs must be absolute local paths, got {source!r}")
    return (str(prober.path), "-hide_banner", "-v", "error", "-protocol_whitelist", "file",
            "-show_format", "-show_streams", "-show_chapters", "-of", "json", "-i", str(source))


class _Timeout(Exception):
    pass


@dataclass
class _Call:
    """One probe process group, visible until teardown has finished."""

    supervisor: _Supervisor
    forced: bool = False


_ACTIVE: dict[int, _Call] = {}


def force_kill() -> int:
    """SIGKILL active probe groups immediately, including graceful teardown.

    Mirrors FFmpeg's lock-free, one-shot escalation (CR-047). The application
    coordinator must call both adapters; no cross-adapter import is permitted.
    Affected calls reap their groups and raise Cancelled, even during timeout
    teardown. Calls started afterwards are unaffected.
    """
    active = _ACTIVE.copy()
    for call in active.values():
        call.forced = True
        call.supervisor._signal_group(signal.SIGKILL)
    return len(active)


# The bounded runner below mirrors ``norma.adapters.ffmpeg`` line for line:
# ADR-0018 rule 4 lets adapters import only models/errors from the package, so
# it cannot be shared by import. tests/unit/core/test_process_lifecycle.py
# checks that ``_Stream`` and ``_Supervisor`` stay source-identical.

class _Stream:
    """One captured pipe: bounded bytes, a bounded line tail and an optional line callback."""

    def __init__(self, stream, *, keep_bytes: int, tail_lines: int, on_line: LineCallback | None) -> None:
        self.stream = stream
        self.fd: int = stream.fileno()
        self._keep_bytes = keep_bytes
        self._on_line = on_line
        self._split = bool(tail_lines) or on_line is not None
        self._pending = b""
        self.data = bytearray()
        self.truncated = False
        self.tail: deque[str] = deque(maxlen=tail_lines)
        self.error: BaseException | None = None
        self.eof = False

    def feed(self, chunk: bytes, *, callbacks: bool) -> None:
        if self._keep_bytes:
            room = self._keep_bytes - len(self.data)
            if room > 0:
                self.data += chunk[:room]
            if len(chunk) > room:
                self.truncated = True
        if not self._split:
            return
        # CR, LF and CR LF all end a line (IC-2 F-2, D-072): a CR-terminated progress line
        # never prefixes the next one. A trailing CR stays pending until the next byte shows
        # whether it begins a CR LF pair, so a pair split across reads is still one line end.
        data = self._pending + chunk
        held = b"\r" if data.endswith(b"\r") else b""
        *lines, self._pending = re.split(rb"\r\n|\r|\n", data[:len(data) - len(held)])
        self._pending += held
        for raw in lines:
            self._emit(raw, callbacks)
        while len(self._pending) > _MAX_LINE_BYTES:
            piece, self._pending = self._pending[:_MAX_LINE_BYTES], self._pending[_MAX_LINE_BYTES:]
            self._emit(piece, callbacks)

    def finish(self, *, callbacks: bool) -> None:
        """EOF: emit a final unterminated line, as ``readline`` would."""
        self.eof = True
        if self._pending:
            pending, self._pending = self._pending, b""
            self._emit(pending, callbacks)

    def _emit(self, raw: bytes, callbacks: bool) -> None:
        while len(raw) > _MAX_LINE_BYTES:  # a long line completed by a later chunk
            self._emit(raw[:_MAX_LINE_BYTES], callbacks)
            raw = raw[_MAX_LINE_BYTES:]
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if self.tail.maxlen:
            self.tail.append(line)
        if callbacks and self._on_line is not None and self.error is None:
            try:
                self._on_line(line)
            except BaseException as exc:  # surfaced by the runner after the group is torn down
                self.error = exc


class _Supervisor:
    """The launched process group, its two pipes and their teardown.

    The group identity is the leader's pid (``start_new_session``), recorded at
    launch, so the group stays addressable after an early leader exit. Pipes
    are read in this thread through a selector: no reader thread can outlive
    the call, and the deadline is checked between reads.
    """

    def __init__(self, proc: subprocess.Popen[bytes], *streams: _Stream) -> None:
        self.proc = proc
        self.pgid = proc.pid
        self._streams = streams
        self._selector = selectors.DefaultSelector()
        self._closed = False
        for item in streams:
            self._selector.register(item.fd, selectors.EVENT_READ, item)

    @property
    def pipes_open(self) -> bool:
        return not self._closed and bool(self._selector.get_map())

    @property
    def finished(self) -> bool:
        """Leader reaped and both pipes at EOF: every buffer is complete."""
        return not self.pipes_open and self.proc.returncode is not None

    @property
    def callback_error(self) -> BaseException | None:
        return next((item.error for item in self._streams if item.error is not None), None)

    def step(self, timeout: float, *, callbacks: bool = True) -> None:
        """Wait at most ``timeout`` for pipe data, drain what is ready, then poll the leader."""
        timeout = max(0.0, timeout)
        if self.pipes_open:
            for key, _ in self._selector.select(timeout):
                item: _Stream = key.data
                try:
                    chunk = os.read(item.fd, _READ_CHUNK)
                except OSError:
                    chunk = b""
                if chunk:
                    item.feed(chunk, callbacks=callbacks)
                else:
                    self._selector.unregister(item.fd)
                    item.finish(callbacks=callbacks)
                    item.stream.close()
        elif timeout:
            time.sleep(timeout)
        self.proc.poll()

    def group_alive(self) -> bool:
        """True while any member of the launched group exists (signal 0 only)."""
        try:
            os.killpg(self.pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # a member exists that we may not signal
            return True
        return True

    def _signal_group(self, signum: int) -> None:
        try:
            os.killpg(self.pgid, signum)
        except (ProcessLookupError, PermissionError):
            pass

    def terminate(self) -> None:
        """SIGTERM the whole group, SIGKILL after the grace period, reap, then confirm it is empty.

        Pipes keep draining throughout, so a member blocked on a full pipe can
        act on SIGTERM; callbacks are not invoked during teardown.
        """
        self._signal_group(signal.SIGTERM)
        grace_end = time.monotonic() + TERMINATE_GRACE_S
        while time.monotonic() < grace_end:
            self.step(min(_SETTLE_POLL_S * 2, grace_end - time.monotonic()), callbacks=False)
            if self.proc.returncode is not None and not self.group_alive():
                break
        if self.proc.returncode is None or self.group_alive():
            self._signal_group(signal.SIGKILL)
        self.proc.wait()
        settle_end = time.monotonic() + TERMINATE_GRACE_S
        while self.group_alive() and time.monotonic() < settle_end:
            self.step(_SETTLE_POLL_S, callbacks=False)
        self.close()

    def close(self) -> None:
        """Release both pipes and the selector; buffers already read are kept. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for key in list(self._selector.get_map().values()):
            self._selector.unregister(key.fd)
            key.data.stream.close()
        self._selector.close()


def _run(argv: tuple[str, ...], deadline_s: float,
         cancel_event: threading.Event | None = None) -> tuple[int, bytes, tuple[str, ...]]:
    """Run argv with both pipes drained, a monotonic deadline and process-group teardown.

    Completes only when the leader has exited and both pipes reached EOF; the
    deadline and cancellation stay armed until then, and stragglers of a
    finished leader are terminated before returning (same rules as the FFmpeg
    runner). Raises ``_Timeout`` or ``Cancelled`` after the group is gone.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()
    started = time.monotonic()
    proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=False, close_fds=True, start_new_session=True,
    )
    out = _Stream(proc.stdout, keep_bytes=_STDOUT_LIMIT, tail_lines=0, on_line=None)
    err = _Stream(proc.stderr, keep_bytes=0, tail_lines=_TAIL_LINES, on_line=None)
    supervisor = _Supervisor(proc, out, err)
    call = _Call(supervisor)
    _ACTIVE[id(call)] = call
    deadline = started + deadline_s
    outcome: str | None = None
    try:
        while not supervisor.finished:
            if call.forced:
                outcome = "cancelled"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                outcome = "timeout"
                break
            if cancel_event is not None and cancel_event.is_set():
                outcome = "cancelled"
                break
            supervisor.step(min(_POLL_S, remaining))
        if call.forced:
            outcome = "cancelled"
        if outcome is not None or supervisor.group_alive():
            supervisor.terminate()  # timeout, cancellation or stragglers
        else:
            supervisor.close()
    except BaseException:
        supervisor.terminate()
        raise
    finally:
        _ACTIVE.pop(id(call), None)
    if call.forced:
        raise Cancelled()
    if outcome == "timeout":
        raise _Timeout(f"ffprobe exceeded its {deadline_s:g} s deadline")
    if outcome == "cancelled":
        raise Cancelled()
    return proc.returncode, bytes(out.data), tuple(err.tail)


def _int(value: Any) -> int | None:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return None
    return number


def _positive_float(value: Any) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 and number != float("inf") else None


def _layout(stream: Mapping[str, Any], channels: int) -> str | None:
    layout = str(stream.get("channel_layout") or "").strip()
    if not layout or layout == "unknown" or re.fullmatch(r"\d+ channels", layout):
        return None
    return layout


def parse_probe(document: Mapping[str, Any], *, version: str, stage: str = "probe") -> ProbeResult:
    """Map an ffprobe JSON document to C-02; raise SkipFile/FileFailure on the C-13 cases."""
    streams = document.get("streams")
    fmt = document.get("format")
    if not isinstance(streams, list) or not isinstance(fmt, Mapping):
        raise FileFailure("probe_error", stage, "ffprobe output lacks streams/format")
    audio = [s for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "audio"]
    if not audio:
        raise SkipFile("not_audio", stage, "probe found no audio stream")
    stream = audio[0]

    codec = str(stream.get("codec_name") or "")
    index = _int(stream.get("index"))
    rate = _int(stream.get("sample_rate"))
    channels = _int(stream.get("channels"))
    format_name = str(fmt.get("format_name") or "")
    if not codec or index is None or index < 0 or not format_name:
        raise FileFailure("probe_error", stage, "ffprobe stream lacks codec, index or container")
    if rate is None or rate <= 0 or channels is None or channels <= 0:
        raise FileFailure("probe_error", stage, f"unusable rate/channels: {rate!r}/{channels!r}")

    klass = codec_class(codec)
    sample_format = str(stream.get("sample_fmt") or "")
    bit_depth: int | None = None
    if klass == "lossless":
        for key in ("bits_per_raw_sample", "bits_per_sample"):
            bits = _int(stream.get(key))
            if bits is not None and bits > 0:
                bit_depth = bits
                break

    duration = _positive_float(stream.get("duration"))
    basis = "stream"
    if duration is None:
        duration = _positive_float(fmt.get("duration"))
        basis = "container" if duration is not None else "unknown"

    bitrate = _int(stream.get("bit_rate"))
    has_artwork = any(isinstance(s, Mapping) and s.get("codec_type") == "video"
                      and (s.get("disposition") or {}).get("attached_pic") == 1 for s in streams)
    chapters = document.get("chapters")
    tags: dict[str, str] = {}
    for key, value in (fmt.get("tags") or {}).items():
        tags[str(key)] = str(value)
    for key, value in (stream.get("tags") or {}).items():
        tags[f"stream:{key}"] = str(value)

    return ProbeResult(
        codec=codec,
        codec_class=klass,  # type: ignore[arg-type]
        container=canonical_container(format_name),
        audio_stream_index=index,
        ignored_streams=len(audio) - 1,
        sample_rate_hz=rate,
        bit_depth=bit_depth,
        sample_float=sample_format in _FLOAT_FORMATS,
        sample_format=sample_format,
        channels=channels,
        channel_layout=_layout(stream, channels),
        duration_s=duration,
        duration_basis=basis,  # type: ignore[arg-type]
        bitrate_bps=bitrate if bitrate is not None and bitrate > 0 else None,
        has_artwork=has_artwork,
        chapters=len(chapters) if isinstance(chapters, list) else 0,
        ffprobe_tags=tags,
        ffprobe_version=version,
    )


def probe(prober: Prober, source: Path, *, deadline_s: float = PROBE_DEADLINE_S,
          cancel_event: threading.Event | None = None) -> ProbeResult:
    """Probe one local file into C-02, bounded at 60 s; Cancelled propagates."""
    stage = "probe"
    try:
        os.stat(source)
        with open(source, "rb"):
            pass
    except PermissionError as exc:
        raise FileFailure("permission_denied", stage, str(exc)) from None
    except FileNotFoundError as exc:
        raise FileFailure("source_changed", stage, f"source disappeared: {exc}") from None
    except OSError as exc:
        if exc.errno == errno.EACCES:
            raise FileFailure("permission_denied", stage, str(exc)) from None
        raise FileFailure("probe_error", stage, str(exc)) from None

    try:
        current = os.stat(prober.path)
    except OSError:
        current = None
    if current is None or (current.st_dev, current.st_ino) != prober.identity:
        raise FileFailure("probe_error", stage, f"ffprobe disappeared or changed identity: {prober.path}")

    try:
        returncode, stdout, tail = _run(probe_argv(prober, source), deadline_s, cancel_event)
    except _Timeout as exc:
        raise FileFailure("timeout", stage, str(exc)) from None
    except OSError as exc:
        raise FileFailure("probe_error", stage, f"cannot start ffprobe: {exc}") from None
    if returncode != 0:
        raise FileFailure("probe_error", stage, f"ffprobe exited {returncode}: " + " | ".join(tail))
    try:
        document = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FileFailure("probe_error", stage, f"unreadable ffprobe JSON: {exc}") from None
    if not isinstance(document, Mapping):
        raise FileFailure("probe_error", stage, "ffprobe JSON is not an object")
    return parse_probe(document, version=prober.version, stage=stage)
