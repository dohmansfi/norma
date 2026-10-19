"""FFmpeg adapter: argv-only process runner, deadlines, version and capability evidence.

Rules (ARCHITECTURE §§ 3, 6, 9; ADR-0018):

- argv lists only, never a shell; FFmpeg always gets ``-nostdin -hide_banner``;
  file inputs get ``-protocol_whitelist file`` and an absolute local path.
- Every call has its own monotonic deadline: ``max(60 s, 4 x known duration)``,
  3,600 s for an unknown duration, 60 s for probes and inspection. On expiry the
  process group gets SIGTERM, then SIGKILL after 2 s, and is reaped.
- A call is finished only when the leader has exited **and** both pipes have
  reached EOF; the deadline and cancellation stay armed until then, so a
  descendant still holding a pipe cannot outlive the deadline. The launched
  group (pgid = leader pid, a new session) is torn down as a whole, including
  an early-exiting leader's descendants, and no call returns while a group
  member is still alive.
- Both pipes are drained continuously; stderr keeps a bounded 20-line tail.
- ``force_kill`` (CR-045, D-065) is the repeated-interrupt escalation: it sends
  SIGKILL at once to every process group a call of this module is running,
  including one already inside its two-second graceful teardown, and that call
  then ends ``Cancelled`` after reaping. It never waits, takes no lock and is
  safe to call from a signal handler.
- Capability evidence (C-04) comes from installed queries and explicitly
  build-matched primary evidence: tiny encodes, header checks and full decode.
  The lossy collectors remove their private temporary encode folders.
  Nothing here is a capability table; an unconfirmable setting is recorded as
  an explicit gap (empty choices, ``nominal_confirmation=not_applicable``).

This module imports only the standard library and ``norma.models``/``norma.errors``.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import selectors
import shutil
import signal
import subprocess
import threading
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from norma.errors import Cancelled, EnvironmentError_, ToolFailed, ToolMissing, ToolTimeout
from norma.models import CapabilityEvidence, Identity

# --- Deadlines (safety § 6, ADR-0018 ruling 1) ---------------------------------------

MIN_DEADLINE_S = 60.0
UNKNOWN_DURATION_DEADLINE_S = 3600.0
PROBE_DEADLINE_S = 60.0
TERMINATE_GRACE_S = 2.0
STDERR_TAIL_LINES = 20
DEFAULT_STDOUT_LIMIT = 64 * 1024 * 1024

FFMPEG_BASE_FLAGS: tuple[str, ...] = ("-nostdin", "-hide_banner")


def tool_deadline(duration_s: float | None) -> float:
    """Per-call deadline for a duration-bearing call: max(60, 4 x duration), else 3,600 s."""
    if duration_s is None:
        return UNKNOWN_DURATION_DEADLINE_S
    if not (duration_s >= 0 and duration_s != float("inf")):
        raise ValueError(f"duration must be a finite non-negative number, got {duration_s!r}")
    return max(MIN_DEADLINE_S, 4.0 * duration_s)


# --- Executables ------------------------------------------------------------------------


def executable_identity(path: Path) -> Identity:
    """``(st_dev, st_ino)`` of the executable file (symlinks followed)."""
    info = os.stat(path)
    return (info.st_dev, info.st_ino)


def resolve_executable(name: str | Path, *, search_path: str | None = None) -> Path:
    """Find a local executable: an absolute path as given, or a bare name on PATH.

    A missing or non-executable tool is an environment error (exit 3 in run preflight).
    """
    text = str(name)
    if os.sep in text:
        candidate = Path(text)
        if not candidate.is_absolute():
            raise EnvironmentError_(f"tool path must be absolute: {text}")
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise EnvironmentError_(f"tool not found or not executable: {text}")
    found = shutil.which(text, path=search_path)
    if found is None:
        raise EnvironmentError_(f"required tool not found on PATH: {text}")
    return Path(found).absolute()


@dataclass(frozen=True)
class Executable:
    """A resolved tool executable, its identity at resolution and its version token."""

    path: Path
    identity: Identity
    version: str

    def revalidate(self) -> None:
        """Raise ToolMissing if the executable disappeared or changed identity."""
        try:
            current = executable_identity(self.path)
        except OSError as exc:
            raise ToolMissing(f"executable disappeared: {self.path} ({exc.strerror})") from None
        if current != self.identity:
            raise ToolMissing(f"executable changed identity: {self.path}")


# --- Process runner ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessResult:
    """Outcome of one finished tool call (adapter value, not a contract record)."""

    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr_tail: tuple[str, ...]
    elapsed_s: float
    stdout_truncated: bool = False


LineCallback = Callable[[str], None]


_READ_CHUNK = 64 * 1024
_MAX_LINE_BYTES = 1024 * 1024
"""A line longer than this is emitted in pieces, so a newline-free stream stays bounded."""
_SETTLE_POLL_S = 0.01

@dataclass
class _Call:
    """One ``run_process`` call's process group, as ``force_kill`` sees it."""

    supervisor: _Supervisor
    forced: bool = False


_ACTIVE: dict[int, _Call] = {}
"""Process groups of calls in progress, for ``force_kill``. [judgement] Plain dict item
assignment, removal and ``copy()`` are each atomic under CPython's GIL, so no lock is
needed and a signal handler can never deadlock against a worker registering a call."""


def force_kill() -> int:
    """Kill every process group this module is running at once; return how many were signalled.

    The repeated-interrupt escalation (safety § 6, CR-045, D-065): SIGKILL with no grace
    period, also for a call already inside its graceful teardown, whose grace wait then ends
    as soon as the group is gone. Each affected ``run_process`` call still reaps its group
    and then raises ``Cancelled``. Calls started afterwards are unaffected, so this is a
    one-shot request, not a lasting mode. Quick, non-blocking, lock-free and safe in a
    signal handler; it never raises for a group that has already gone.
    """
    active = _ACTIVE.copy()
    for call in active.values():
        call.forced = True
        call.supervisor._signal_group(signal.SIGKILL)
    return len(active)


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


def run_process(
    argv: Sequence[str],
    *,
    deadline_s: float,
    capture_stdout: bool = True,
    stdout_limit: int = DEFAULT_STDOUT_LIMIT,
    on_stdout_line: LineCallback | None = None,
    on_stderr_line: LineCallback | None = None,
    cancel_event: threading.Event | None = None,
    executable: Executable | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    poll_interval_s: float = 0.05,
) -> ProcessResult:
    """Run one tool call under its own deadline; never through a shell.

    The call completes only when the leader has exited and both pipes have
    reached EOF; until then the deadline and ``cancel_event`` stay armed.
    Group members left behind by a finished leader are terminated before
    returning. Raises ToolMissing (executable gone/changed or not startable),
    ToolTimeout (deadline expired; group terminated and reaped) or Cancelled.
    A non-zero exit is returned, not raised; see ``check_result``.
    """
    args = tuple(argv)
    if not args or not all(isinstance(item, str) for item in args):
        raise TypeError("argv must be a non-empty sequence of str")
    if not os.path.isabs(args[0]):
        raise ValueError(f"argv[0] must be an absolute executable path, got {args[0]!r}")
    if not deadline_s > 0:
        raise ValueError("deadline_s must be positive")
    if executable is not None:
        executable.revalidate()
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()

    started = time.monotonic()
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            start_new_session=True,
            env=env,
            cwd=cwd,
        )
    except (FileNotFoundError, PermissionError, NotADirectoryError) as exc:
        raise ToolMissing(f"cannot start {args[0]}: {exc.strerror}", argv=args) from None

    out = _Stream(proc.stdout, keep_bytes=stdout_limit if capture_stdout else 0, tail_lines=0,
                  on_line=on_stdout_line)
    err = _Stream(proc.stderr, keep_bytes=0, tail_lines=STDERR_TAIL_LINES, on_line=on_stderr_line)
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
            supervisor.step(min(poll_interval_s, remaining))
            if supervisor.callback_error is not None:
                outcome = "callback"
                break
        if call.forced:  # killed by force_kill, perhaps after the leader already ended
            outcome = "cancelled"
        if outcome is not None or supervisor.group_alive():
            supervisor.terminate()  # timeout, cancellation, callback failure or stragglers
        else:
            supervisor.close()
    except BaseException:
        supervisor.terminate()
        raise
    finally:
        _ACTIVE.pop(id(call), None)

    elapsed = time.monotonic() - started
    tail = tuple(err.tail)
    if call.forced:  # a forced kill during any teardown is an interruption, not a timeout
        raise Cancelled()
    if outcome == "timeout":
        raise ToolTimeout(f"{Path(args[0]).name} exceeded its {deadline_s:g} s deadline",
                          deadline_s=deadline_s, argv=args, stderr_tail=tail)
    if outcome == "cancelled":
        raise Cancelled()
    callback_error = supervisor.callback_error
    if callback_error is not None:
        raise callback_error
    return ProcessResult(args, proc.returncode, bytes(out.data), tail, elapsed, out.truncated)


def check_result(result: ProcessResult) -> ProcessResult:
    """Raise ToolFailed for a non-zero exit, keeping the stderr tail."""
    if result.returncode != 0:
        raise ToolFailed(f"{Path(result.argv[0]).name} exited with status {result.returncode}",
                         returncode=result.returncode, argv=result.argv, stderr_tail=result.stderr_tail)
    return result


# --- FFmpeg argv helpers ------------------------------------------------------------------


def ffmpeg_argv(executable: Path | Executable, *args: str) -> tuple[str, ...]:
    """``[ffmpeg, -nostdin, -hide_banner, *args]``; every element is a separate argv value."""
    path = executable.path if isinstance(executable, Executable) else executable
    if not all(isinstance(item, str) for item in args):
        raise TypeError("FFmpeg arguments must be str")
    return (str(path), *FFMPEG_BASE_FLAGS, *args)


def input_args(path: Path) -> tuple[str, ...]:
    """Input arguments for a local file: protocol whitelist ``file`` and an absolute path."""
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"FFmpeg inputs must be absolute local paths, got {path!r}")
    return ("-protocol_whitelist", "file", "-i", str(path))


_VERSION_RE = re.compile(r"^(?:ffmpeg|ffprobe) version (\S+)")


def parse_version(text: str) -> str:
    """The version token from ``-version`` output's first line."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    match = _VERSION_RE.match(first)
    if match is None:
        raise ValueError(f"unrecognised version line: {first!r}")
    return match.group(1)


def read_version(path: Path, *, extra_flags: Sequence[str] = FFMPEG_BASE_FLAGS) -> str:
    """Run ``<tool> -version`` (bounded at 60 s) and return its version token."""
    result = check_result(run_process((str(path), *extra_flags, "-version"), deadline_s=PROBE_DEADLINE_S))
    try:
        return parse_version(result.stdout.decode("utf-8", "replace"))
    except ValueError as exc:
        raise EnvironmentError_(f"{path}: {exc}") from None


def locate_ffmpeg(name: str | Path = "ffmpeg", *, search_path: str | None = None) -> Executable:
    """Resolve FFmpeg, record its identity and version; failures are environment errors."""
    path = resolve_executable(name, search_path=search_path)
    identity = executable_identity(path)
    try:
        version = read_version(path)
    except (ToolFailed, ToolMissing, ToolTimeout) as exc:
        raise EnvironmentError_(f"FFmpeg at {path} is unusable: {exc}") from None
    return Executable(path, identity, version)


# --- Feature discovery ----------------------------------------------------------------------


def _listing_names(text: str) -> frozenset[str]:
    names: set[str] = set()
    in_body = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("------"):
            in_body = True
            continue
        if in_body and stripped:
            parts = stripped.split()
            if len(parts) >= 2:
                names.add(parts[1])
    return frozenset(names)


def parse_encoders(text: str) -> frozenset[str]:
    """Encoder names from ``ffmpeg -encoders``."""
    return _listing_names(text)


def parse_filters(text: str) -> frozenset[str]:
    """Filter names from ``ffmpeg -filters``."""
    return _listing_names(text)


_OPTION_RE = re.compile(r"^\s+-?([A-Za-z0-9_]+)\s+<")


def parse_filter_options(text: str) -> frozenset[str]:
    """Option names from ``ffmpeg -h filter=<name>``."""
    return frozenset(match.group(1) for line in text.splitlines() if (match := _OPTION_RE.match(line)))


def _query(executable: Executable, *args: str) -> str:
    result = check_result(run_process(ffmpeg_argv(executable, *args), deadline_s=PROBE_DEADLINE_S,
                                      executable=executable))
    return result.stdout.decode("utf-8", "replace")


def platform_basis() -> str:
    """OS/architecture basis recorded with capability evidence; no subprocess."""
    release = platform.mac_ver()[0]
    if release:
        return f"macOS {release} {platform.machine()}"
    return platform.platform()


@dataclass(frozen=True)
class FFmpegEnvironment:
    """Run-local FFmpeg facts captured once in preflight."""

    executable: Executable
    encoders: frozenset[str]
    filters: frozenset[str]
    platform_basis: str


def probe_environment(executable: Executable) -> FFmpegEnvironment:
    """List encoders and filters of a located FFmpeg (each call bounded at 60 s)."""
    try:
        encoders = parse_encoders(_query(executable, "-encoders"))
        filters = parse_filters(_query(executable, "-filters"))
    except (ToolFailed, ToolMissing, ToolTimeout) as exc:
        raise EnvironmentError_(f"cannot list FFmpeg features: {exc}") from None
    return FFmpegEnvironment(executable, encoders, filters, platform_basis())


def filter_options(executable: Executable, name: str) -> frozenset[str]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"invalid filter name {name!r}")
    return parse_filter_options(_query(executable, "-h", f"filter={name}"))


MEASURE_FILTERS: tuple[str, ...] = ("ebur128", "ametadata", "apad", "aformat", "astats")
"""Every filter ``audio/measure.py`` runs: the meter and the sample-zero scan (CR-038)."""
RENDER_FILTERS: tuple[str, ...] = ("volume", "aresample", "alimiter")
DYNAMIC_FILTERS: tuple[str, ...] = ("loudnorm",)
ALIMITER_OPTIONS: tuple[str, ...] = ("level", "latency")
NEVER_ENCODERS: frozenset[str] = frozenset({"alac_at"})


def check_requirements(env: FFmpegEnvironment, *, encoders: Iterable[str] = (), filters: Iterable[str] = (),
                       alimiter_options: Iterable[str] = ()) -> None:
    """Raise EnvironmentError_ (exit 3) naming every missing required facility."""
    wanted_encoders = tuple(encoders)
    if NEVER_ENCODERS.intersection(wanted_encoders):
        raise ValueError("alac_at is never a Norma encoder; use native alac")
    missing = [f"encoder {name}" for name in wanted_encoders if name not in env.encoders]
    missing += [f"filter {name}" for name in filters if name not in env.filters]
    wanted_options = tuple(alimiter_options)
    if wanted_options and "alimiter" in env.filters:
        present = filter_options(env.executable, "alimiter")
        missing += [f"alimiter option {name}" for name in wanted_options if name not in present]
    if missing:
        raise EnvironmentError_("FFmpeg lacks required facilities: " + ", ".join(missing))


# --- Written-header readers (capability confirmation) -------------------------------------


@dataclass(frozen=True)
class PcmHeader:
    channels: int
    sample_rate_hz: int
    bit_depth: int
    sample_float: bool


def read_wav_header(data: bytes) -> PcmHeader:
    """RIFF/RF64 WAVE ``fmt `` chunk, including WAVE_FORMAT_EXTENSIBLE valid bits."""
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RF64", b"BW64") or data[8:12] != b"WAVE":
        raise ValueError("not a WAVE file")
    pos = 12
    while pos + 8 <= len(data):
        chunk, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if chunk == b"fmt ":
            fmt = int.from_bytes(body[0:2], "little")
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            if fmt == 0xFFFE and len(body) >= 40:
                valid = int.from_bytes(body[18:20], "little")
                bits = valid or bits
                fmt = int.from_bytes(body[24:26], "little")
            if fmt not in (1, 3):
                raise ValueError(f"unexpected WAVE format code {fmt:#x}")
            return PcmHeader(channels, rate, bits, fmt == 3)
        pos += 8 + size + (size & 1)
    raise ValueError("WAVE file has no fmt chunk")


def _extended_to_float(raw: bytes) -> float:
    exponent = ((raw[0] & 0x7F) << 8) | raw[1]
    mantissa = int.from_bytes(raw[2:10], "big")
    if exponent == 0 and mantissa == 0:
        return 0.0
    value = mantissa * 2.0 ** (exponent - 16383 - 63)
    return -value if raw[0] & 0x80 else value


def read_aiff_header(data: bytes) -> PcmHeader:
    """AIFF/AIFF-C ``COMM`` chunk; AIFF-C ``fl32``/``fl64`` compression is float."""
    if len(data) < 12 or data[:4] != b"FORM" or data[8:12] not in (b"AIFF", b"AIFC"):
        raise ValueError("not an AIFF file")
    pos = 12
    while pos + 8 <= len(data):
        chunk, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "big")
        body = data[pos + 8:pos + 8 + size]
        if chunk == b"COMM":
            channels = int.from_bytes(body[0:2], "big")
            bits = int.from_bytes(body[6:8], "big")
            rate = _extended_to_float(body[8:18])
            compression = body[18:22] if data[8:12] == b"AIFC" else b"NONE"
            if compression in (b"fl32", b"FL32", b"fl64", b"FL64"):
                is_float = True
            elif compression in (b"NONE", b"twos", b"sowt"):
                is_float = False
            else:
                raise ValueError(f"unexpected AIFF-C compression {compression!r}")
            if rate != int(rate):
                raise ValueError(f"non-integer AIFF sample rate {rate}")
            return PcmHeader(channels, int(rate), bits, is_float)
        pos += 8 + size + (size & 1)
    raise ValueError("AIFF file has no COMM chunk")


def read_flac_header(data: bytes) -> PcmHeader:
    """FLAC STREAMINFO (the first metadata block)."""
    if data[:4] != b"fLaC" or len(data) < 8 + 34 or data[4] & 0x7F != 0:
        raise ValueError("not a FLAC file with STREAMINFO first")
    info = data[8:8 + 34]
    packed = int.from_bytes(info[10:14], "big")
    rate = packed >> 12
    channels = ((packed >> 9) & 0x7) + 1
    bits = ((packed >> 4) & 0x1F) + 1
    return PcmHeader(channels, rate, bits, False)


def read_alac_header(data: bytes) -> PcmHeader:
    """The ALACSpecificConfig ('alac' box inside the 'alac' sample entry) of an MP4 file."""
    for match in re.finditer(re.escape(b"\x00\x00\x00\x24alac"), data):
        body = data[match.start() + 8:match.start() + 36]
        if len(body) < 28 or body[0:4] != b"\x00\x00\x00\x00":
            continue
        bits = body[9]
        channels = body[13]
        rate = int.from_bytes(body[24:28], "big")
        return PcmHeader(channels, rate, bits, False)
    raise ValueError("no ALAC magic cookie found")


# ISO/IEC 11172-3 and 13818-3 Layer III header fields, used only to read written frames.
_MP3_BITRATES_KBPS = {
    "1": (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, -1),
    "2": (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, -1),
}
_MP3_RATES = {"1": (44100, 48000, 32000), "2": (22050, 24000, 16000), "2.5": (11025, 12000, 8000)}
_MP3_VERSIONS = {0b11: "1", 0b10: "2", 0b00: "2.5"}


@dataclass(frozen=True)
class Mp3Frame:
    offset: int
    version: str
    bitrate_bps: int
    sample_rate_hz: int
    channel_mode: int
    """0 stereo, 1 joint stereo, 2 dual channel, 3 mono."""
    length: int


@dataclass(frozen=True)
class XingTag:
    tag: bytes
    """b'Xing' (VBR) or b'Info' (CBR)."""
    quality: int | None
    vbr_method: int | None
    """LAME-tag VBR method nibble, when a LAME extension follows."""


def _id3v2_size(data: bytes) -> int:
    if len(data) >= 10 and data[:3] == b"ID3":
        size = 0
        for byte in data[6:10]:
            size = (size << 7) | (byte & 0x7F)
        footer = 10 if data[5] & 0x10 else 0
        return 10 + size + footer
    return 0


def read_mp3_frames(data: bytes) -> tuple[Mp3Frame, ...]:
    """Every Layer III frame header after an optional ID3v2 tag; gaps are errors."""
    pos = _id3v2_size(data)
    frames: list[Mp3Frame] = []
    while pos + 4 <= len(data):
        if data[pos:pos + 3] == b"TAG":  # ID3v1 trailer
            break
        header = int.from_bytes(data[pos:pos + 4], "big")
        if header >> 21 != 0x7FF:
            raise ValueError(f"lost MPEG frame sync at offset {pos}")
        version = _MP3_VERSIONS.get((header >> 19) & 0x3)
        layer = (header >> 17) & 0x3
        if version is None or layer != 0b01:
            raise ValueError(f"not an MPEG Layer III frame at offset {pos}")
        table = _MP3_BITRATES_KBPS["1" if version == "1" else "2"]
        kbps = table[(header >> 12) & 0xF]
        rate_index = (header >> 10) & 0x3
        if kbps <= 0 or rate_index == 3:
            raise ValueError(f"free-format or invalid frame header at offset {pos}")
        rate = _MP3_RATES[version][rate_index]
        padding = (header >> 9) & 0x1
        mode = (header >> 6) & 0x3
        length = (144000 if version == "1" else 72000) * kbps // rate + padding
        frames.append(Mp3Frame(pos, version, kbps * 1000, rate, mode, length))
        pos += length
    if not frames:
        raise ValueError("no MPEG audio frames")
    return tuple(frames)


def read_xing_tag(data: bytes, frame: Mp3Frame) -> XingTag | None:
    """The Xing/Info tag in the first frame, if present."""
    if frame.version == "1":
        side = 17 if frame.channel_mode == 3 else 32
    else:
        side = 9 if frame.channel_mode == 3 else 17
    pos = frame.offset + 4 + side
    tag = data[pos:pos + 4]
    if tag not in (b"Xing", b"Info"):
        return None
    flags = int.from_bytes(data[pos + 4:pos + 8], "big")
    cursor = pos + 8
    cursor += 4 if flags & 0x1 else 0
    cursor += 4 if flags & 0x2 else 0
    cursor += 100 if flags & 0x4 else 0
    quality = None
    if flags & 0x8:
        quality = int.from_bytes(data[cursor:cursor + 4], "big")
        cursor += 4
    vbr_method = None
    if len(data) >= cursor + 10:
        vbr_method = data[cursor + 9] & 0x0F
    return XingTag(tag, quality, vbr_method)


# --- Capability evidence (C-04) --------------------------------------------------------------

EVIDENCE_SAMPLE_DURATION_S = 0.25
MP3_CBR_CANDIDATES_BPS: tuple[int, ...] = tuple(range(64_000, 320_001, 8_000))
"""Requests probed for MP3 CBR: the CLI's 64-320 kbps outer bound in 8 kbps steps.

Candidates are questions, not answers: a value enters the evidence only when
every written frame carries it. Unprobed values are never evidenced."""
MP3_VBR_CANDIDATES: tuple[int, ...] = tuple(range(10))

AAC_CBR_CANDIDATES_BPS: tuple[int, ...] = (
    8_000, 16_000, 24_000, 32_000, 40_000, 48_000, 56_000,
    64_000, 72_000, 80_000, 96_000, 112_000, 128_000, 144_000,
    160_000, 192_000, 224_000, 256_000, 288_000, 320_000,
)
"""Questions from formats § 9 C3, not a capability table; each is checked locally.

The 8-56 kbps steps are low-rate questions (IC-3 F-2): stereo 8 kHz has only
48 kbps as its maximum, so a list starting at 64 kbps could never discover it.
A value enters the evidence only when every existing confirmation check passes."""

_LOSSY_EVIDENCE_VERSION = "N-125892-g406c5a37aa-https://www.martin-riedl.de"
_LAME_SOURCE = "https://github.com/FFmpeg/FFmpeg/blob/406c5a37aa/libavcodec/libmp3lame.c"


def _matched_lossy_build(env: FFmpegEnvironment) -> bool:
    """The primary-source V0 setting proof is restricted to the inspected basis."""
    return (env.executable.version == _LOSSY_EVIDENCE_VERSION
            and env.platform_basis == platform_basis()
            and platform.system() == "Darwin" and platform.machine() == "arm64"
            and platform.mac_ver()[0] == "15.6.1")


def _check_evidence_cancel(cancel_event: threading.Event | None) -> None:
    """Stop evidence admission, including after local queries and header parsing."""
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()


def _decoded_evidence_ok(env: FFmpegEnvironment, path: Path, rate: int, channels: int, *,
                         cancel_event: threading.Event | None = None) -> bool:
    """Decode every byte without rate/channel overrides; reject failure or empty PCM."""
    _check_evidence_cancel(cancel_event)
    result = run_process(ffmpeg_argv(env.executable, "-loglevel", "error", "-xerror",
                                    *input_args(path), "-map", "0:a:0", "-c:a", "pcm_s16le",
                                    "-f", "wav", "pipe:1"),
                         deadline_s=PROBE_DEADLINE_S, executable=env.executable, cancel_event=cancel_event)
    _check_evidence_cancel(cancel_event)
    if result.returncode or result.stdout_truncated:
        return False
    try:
        header = read_wav_header(result.stdout)
    except ValueError:
        _check_evidence_cancel(cancel_event)
        return False
    _check_evidence_cancel(cancel_event)
    data_at = result.stdout.find(b"data")
    return (header.sample_rate_hz == rate and header.channels == channels
            and data_at >= 0 and len(result.stdout) > data_at + 8)


def _aac_nominal_settings(rate: int, channels: int, candidates: Sequence[int], *,
                           cancel_event: threading.Event | None = None) -> tuple[int, ...]:
    """Installed AudioToolbox CBR set/get query; no averages, interpolation or table.

    AudioConverter.h/AudioCodec.h define ASBD, 'acbf' (CBR=0), 'brat'.
    Uses the OS framework already required by aac_at, through stdlib ctypes.
    Query failures produce an empty conservative gap; the converter is disposed
    on every path. Only conventional mono/stereo layouts are queried.
    """
    _check_evidence_cancel(cancel_event)
    if platform.system() != "Darwin" or channels not in (1, 2):
        return ()

    class ASBD(ctypes.Structure):
        _fields_ = [("rate", ctypes.c_double)] + [(name, ctypes.c_uint32) for name in
                    ("format", "flags", "bytes_packet", "frames_packet", "bytes_frame",
                     "channels", "bits", "reserved")]

    def fourcc(text: str) -> int:
        return int.from_bytes(text.encode("ascii"), "big")

    try:
        lib = ctypes.CDLL("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
        lib.AudioConverterNew.argtypes = [ctypes.POINTER(ASBD), ctypes.POINTER(ASBD),
                                         ctypes.POINTER(ctypes.c_void_p)]
        lib.AudioConverterNew.restype = ctypes.c_int32
        lib.AudioConverterSetProperty.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                                  ctypes.c_void_p]
        lib.AudioConverterSetProperty.restype = ctypes.c_int32
        lib.AudioConverterGetProperty.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                  ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        lib.AudioConverterGetProperty.restype = ctypes.c_int32
        lib.AudioConverterDispose.argtypes = [ctypes.c_void_p]
        lib.AudioConverterDispose.restype = ctypes.c_int32
    except (OSError, AttributeError):
        _check_evidence_cancel(cancel_event)
        return ()
    _check_evidence_cancel(cancel_event)
    source = ASBD(rate, fourcc("lpcm"), 4 | 8, channels * 2, 1, channels * 2, channels, 16, 0)
    # Mirror aac_at's AudioConverterNew descriptions (default AAC-LC).
    target = ASBD(rate, fourcc("aac "), 0, 0, 0, 0, channels, 0, 0)
    accepted: list[int] = []
    for bps in candidates:
        _check_evidence_cancel(cancel_event)
        converter = ctypes.c_void_p()
        status = lib.AudioConverterNew(ctypes.byref(source), ctypes.byref(target), ctypes.byref(converter))
        try:
            _check_evidence_cancel(cancel_event)
            if status or not converter.value:
                continue
            mode = ctypes.c_uint32(0)
            nominal = ctypes.c_uint32(bps)
            if lib.AudioConverterSetProperty(converter, fourcc("acbf"), 4, ctypes.byref(mode)):
                continue
            _check_evidence_cancel(cancel_event)
            size = ctypes.c_uint32(4)
            observed_mode = ctypes.c_uint32(0xFFFFFFFF)
            if (lib.AudioConverterGetProperty(converter, fourcc("acbf"), ctypes.byref(size),
                                              ctypes.byref(observed_mode))
                    or size.value != 4 or observed_mode.value != 0):
                continue
            _check_evidence_cancel(cancel_event)
            if lib.AudioConverterSetProperty(converter, fourcc("brat"), 4, ctypes.byref(nominal)):
                continue
            _check_evidence_cancel(cancel_event)
            size = ctypes.c_uint32(4)
            observed = ctypes.c_uint32()
            if (not lib.AudioConverterGetProperty(converter, fourcc("brat"), ctypes.byref(size),
                                                  ctypes.byref(observed))
                    and size.value == 4 and observed.value == bps):
                accepted.append(bps)
        finally:
            if converter.value:
                lib.AudioConverterDispose(converter)
    _check_evidence_cancel(cancel_event)
    return tuple(accepted)


def collect_aac_evidence(env: FFmpegEnvironment, sample_rate_hz: int, channels: int,
                         channel_layout: str | None, work_dir: Path | None = None, *,
                         cbr_candidates_bps: Sequence[int] = AAC_CBR_CANDIDATES_BPS,
                         cancel_event: threading.Event | None = None) -> CapabilityEvidence:
    """AAC-LC CBR installed query, tiny M4A encodes and full decoded read-back.

    A nominal request must survive AudioToolbox's CBR set/get unchanged and a
    separate FFmpeg encode with all warning diagnostics inspected. Inspect
    these diagnostics on subsequent renders (F12); never compare averages.
    Temporary encodes are removed even on cancellation. Unknown layouts and
    missing nominal proof retain an explicit gap, never borrow stereo evidence.
    """
    _check_evidence_cancel(cancel_event)
    _require_encoder(env, "aac_at")
    source = _source_args(sample_rate_hz, channels, channel_layout)
    candidates = tuple(cbr_candidates_bps)
    if any(type(bps) is not int or not 0 < bps <= 0xFFFFFFFF for bps in candidates):
        raise ValueError("nominal bitrate candidates must be positive uint32 integers")
    accepted: list[int] = []
    gaps: list[str] = []
    known_layout = (channels, channel_layout) in ((1, "mono"), (2, "stereo"))
    nominal = (_aac_nominal_settings(sample_rate_hz, channels, candidates, cancel_event=cancel_event)
               if known_layout else ())
    _check_evidence_cancel(cancel_event)
    if not nominal:
        gaps.append("no unchanged AudioToolbox CBR set/get setting for this rate/layout")
    else:
        gaps.extend(f"{bps}: no unchanged AudioToolbox CBR set/get setting"
                    for bps in candidates if bps not in nominal)
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="norma-aac-", dir=work_dir) as folder:
        for bps in nominal:
            _check_evidence_cancel(cancel_event)
            path = Path(folder) / f"{bps}.m4a"
            warnings: deque[str] = deque(maxlen=STDERR_TAIL_LINES)

            def inspect_warning(line: str) -> None:
                # This matched build's tiny AAC probe emits a flush-queue
                # bookkeeping message, not a setting adjustment. Successful
                # full decode is still required. Every other warning refuses.
                if (_matched_lossy_build(env)
                        and re.fullmatch(r"\[aac_at @ 0x[0-9a-f]+\] Trying to remove 1024 samples, "
                                         r"but the queue is empty", line)):
                    return
                warnings.append(line)

            result = run_process(ffmpeg_argv(env.executable, "-loglevel", "warning", "-y", *source,
                                            "-map", "0:a:0", "-c:a", "aac_at", "-aac_at_mode", "cbr",
                                            "-b:a", str(bps), "-f", "ipod", str(path)),
                                 deadline_s=PROBE_DEADLINE_S, capture_stdout=False,
                                 on_stderr_line=inspect_warning, executable=env.executable,
                                 cancel_event=cancel_event)
            _check_evidence_cancel(cancel_event)
            # Every non-bookkeeping warning disqualifies, including a nominal clamp.
            if result.returncode or warnings or not path.is_file():
                gaps.append(f"{bps}: encode failed or warned: {' | '.join(tuple(warnings)[-3:])}")
                continue
            try:
                headers = run_process(ffmpeg_argv(env.executable, "-loglevel", "error", "-xerror",
                                                  *input_args(path), "-map", "0:a:0", "-c:a", "copy",
                                                  "-f", "adts", "pipe:1"),
                                      deadline_s=PROBE_DEADLINE_S, executable=env.executable,
                                      cancel_event=cancel_event)
                _check_evidence_cancel(cancel_event)
                if headers.returncode or headers.stdout_truncated:
                    raise ValueError("AAC packet read-back failed")
                _check_adts(headers.stdout, sample_rate_hz, channels)
            except ValueError as exc:
                _check_evidence_cancel(cancel_event)
                gaps.append(f"{bps}: {exc}")
                continue
            _check_evidence_cancel(cancel_event)
            if _decoded_evidence_ok(env, path, sample_rate_hz, channels, cancel_event=cancel_event):
                accepted.append(bps)
            else:
                gaps.append(f"{bps}: decoded rate/channels or full decode failed")
    _check_evidence_cancel(cancel_event)
    return CapabilityEvidence(
        evidence_id=evidence_id(env, "aac_at", "cbr", sample_rate_hz, channels, channel_layout),
        executable_path=env.executable.path, executable_identity=env.executable.identity,
        ffmpeg_version=env.executable.version, platform_basis=env.platform_basis, encoder="aac_at",
        sample_rate_hz=sample_rate_hz, channels=channels, channel_layout=channel_layout,
        bitrate_mode="cbr", supported_bit_depths=(), supports_float=False,
        nominal_bitrates_bps=tuple(sorted(set(accepted))), vbr_qualities=(), basis="installed_query",
        source_locator="AudioToolbox CBR brat set/get; aac_at -aac_at_mode cbr; "
                       "all warnings inspected (matched-build 1024-sample flush-queue note excepted); "
                       "ADTS LC/rate/layout and full PCM decode checked; "
                       + ("; ".join(gaps) if gaps else "all queried settings confirmed"),
        nominal_confirmation="encoder_diagnostic" if accepted else "not_applicable")


def _check_adts(data: bytes, rate: int, channels: int) -> None:
    """Check every complete ADTS frame: AAC-LC object type, rate and channel config."""
    rates = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)
    pos = 0
    while pos < len(data):
        header = data[pos:pos + 7]
        if len(header) < 7 or header[0] != 0xFF or header[1] & 0xF6 != 0xF0:
            raise ValueError("invalid ADTS sync/header")
        index = (header[2] >> 2) & 15
        count = ((header[2] & 1) << 2) | (header[3] >> 6)
        length = ((header[3] & 3) << 11) | (header[4] << 3) | (header[5] >> 5)
        minimum = 7 if header[1] & 1 else 9
        if (header[2] >> 6 != 1 or index >= len(rates) or rates[index] != rate
                or count != channels or not minimum < length <= len(data) - pos):
            raise ValueError("ADTS profile/rate/channels or frame length mismatch")
        pos += length
    if pos == 0:
        raise ValueError("empty ADTS output")


def collect_mp3_v0_evidence(env: FFmpegEnvironment, sample_rate_hz: int, channels: int,
                            channel_layout: str | None, work_dir: Path | None = None, *,
                            cancel_event: threading.Event | None = None) -> CapabilityEvidence:
    """Only V0, using the existing collector, in a removed private temporary folder."""
    _check_evidence_cancel(cancel_event)
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="norma-mp3-", dir=work_dir) as folder:
        _, record = collect_mp3_evidence(env, sample_rate_hz, channels, channel_layout, Path(folder),
                                        cbr_candidates_bps=(), vbr_candidates=(0,), cancel_event=cancel_event)
        _check_evidence_cancel(cancel_event)
        if record.vbr_qualities and record.basis != "build_matched_evidence":
            path = Path(folder) / f"mp3-{sample_rate_hz}-{channels}-vbr0.mp3"
            if not _decoded_evidence_ok(env, path, sample_rate_hz, channels, cancel_event=cancel_event):
                record = replace(record, vbr_qualities=(), nominal_confirmation="not_applicable",
                                 source_locator=record.source_locator + "; gap: full decoded read-back failed")
        _check_evidence_cancel(cancel_event)
        return record

_LOSSLESS_CANDIDATES: dict[str, tuple[tuple[str, str, tuple[str, ...], int | None, bool], ...]] = {
    # encoder: (muxer, extension, extra encoder args, expected depth, expected float)
    "flac": (("flac", ".flac", ("-sample_fmt", "s16"), 16, False),
             ("flac", ".flac", ("-sample_fmt", "s32", "-bits_per_raw_sample", "24"), 24, False)),
    "alac": (("ipod", ".m4a", ("-sample_fmt", "s16p"), 16, False),
             ("ipod", ".m4a", ("-sample_fmt", "s32p", "-bits_per_raw_sample", "24"), 24, False)),
}
_PCM_RE = re.compile(r"^pcm_([sf])(16|24|32)(le|be)$")


def evidence_id(env: FFmpegEnvironment, encoder: str, mode: str, sample_rate_hz: int, channels: int,
                channel_layout: str | None) -> str:
    """Run-unique, deterministic evidence identifier bound to executable and basis."""
    dev, ino = env.executable.identity
    layout = channel_layout or f"{channels}c-unknown"
    return f"ev:{encoder}:{mode}:{sample_rate_hz}:{channels}:{layout}@{dev}:{ino}"


def _source_args(sample_rate_hz: int, channels: int, channel_layout: str | None) -> tuple[str, ...]:
    if not (isinstance(sample_rate_hz, int) and sample_rate_hz > 0 and isinstance(channels, int) and channels > 0):
        raise ValueError("sample rate and channels must be positive integers")
    layout = channel_layout or f"{channels}c"
    if not re.fullmatch(r"[A-Za-z0-9.()+_-]+", layout):
        raise ValueError(f"invalid channel layout {layout!r}")
    graph = (f"sine=frequency=997:sample_rate={sample_rate_hz}:duration={EVIDENCE_SAMPLE_DURATION_S},"
             f"aformat=channel_layouts={layout}")
    return ("-protocol_whitelist", "file", "-f", "lavfi", "-i", graph)


def _require_encoder(env: FFmpegEnvironment, encoder: str) -> None:
    if encoder in NEVER_ENCODERS:
        raise ValueError("alac_at is never a Norma encoder; use native alac")
    if encoder not in env.encoders:
        raise EnvironmentError_(f"FFmpeg lacks required encoder {encoder}")


def _encode_outputs(env: FFmpegEnvironment, source: tuple[str, ...],
                    outputs: Sequence[tuple[Path, tuple[str, ...]]], *,
                    cancel_event: threading.Event | None = None) -> tuple[dict[Path, bool], tuple[str, ...]]:
    """Encode every output in one call; fall back to one call each if the batch fails.

    Returns success per output path and the last stderr tail seen.
    """
    _check_evidence_cancel(cancel_event)
    deadline = tool_deadline(EVIDENCE_SAMPLE_DURATION_S)

    def argv_for(items: Sequence[tuple[Path, tuple[str, ...]]]) -> tuple[str, ...]:
        args: list[str] = ["-loglevel", "error", "-y", *source]
        for path, encode_args in items:
            args += ["-map", "0:a:0", *encode_args, str(path)]
        return ffmpeg_argv(env.executable, *args)

    batch = run_process(argv_for(outputs), deadline_s=deadline, capture_stdout=False, executable=env.executable,
                        cancel_event=cancel_event)
    _check_evidence_cancel(cancel_event)
    if batch.returncode == 0:
        return {path: path.is_file() for path, _ in outputs}, batch.stderr_tail
    results: dict[Path, bool] = {}
    tail = batch.stderr_tail
    for item in outputs:
        _check_evidence_cancel(cancel_event)
        single = run_process(argv_for((item,)), deadline_s=deadline, capture_stdout=False,
                             executable=env.executable, cancel_event=cancel_event)
        _check_evidence_cancel(cancel_event)
        results[item[0]] = single.returncode == 0 and item[0].is_file()
        if single.returncode != 0:
            tail = single.stderr_tail
    return results, tail


def _mp3_layout_ok(frame: Mp3Frame, channels: int) -> bool:
    """MP3 frames carry mono or two channels; anything else was remapped (never evidence)."""
    if channels == 1:
        return frame.channel_mode == 3
    if channels == 2:
        return frame.channel_mode in (0, 1)
    return False


MP3_STATS_PROBE_ARGS: tuple[str, ...] = (
    "-stats_enc_pre", "pipe:1", "-stats_enc_pre_fmt", "pre {n} {pts} {tb} {samp}",
    "-stats_enc_post", "pipe:1", "-stats_enc_post_fmt", "post {n} {pts} {tb}",
)
"""The encoder-statistics options ``audio.render.MP3_ENCODER_STATS`` passes (CR-063); kept equal by a test."""


def probe_mp3_encoder_stats(env: FFmpegEnvironment, *, cancel_event: threading.Event | None = None) -> bool:
    """CR-067: does this FFmpeg write the encoder statistics MP3 gapless data is read from?

    Probed by behaviour, not version string: one 0.25 s libmp3lame encode to the null muxer
    with the render's own options. True only when it exits 0 and stdout carries well-formed
    ``pre`` and ``post`` lines. A build without the options exits non-zero ("Unrecognized
    option"), so MP3 evidence is withheld up front instead of failing at render.
    """
    _check_evidence_cancel(cancel_event)
    _require_encoder(env, "libmp3lame")
    result = run_process(
        ffmpeg_argv(env.executable, "-loglevel", "error", "-y", *_source_args(44_100, 2, "stereo"),
                    "-map", "0:a:0", "-c:a", "libmp3lame", "-q:a", "0", *MP3_STATS_PROBE_ARGS,
                    "-f", "null", "-"),
        deadline_s=tool_deadline(EVIDENCE_SAMPLE_DURATION_S), executable=env.executable, cancel_event=cancel_event)
    _check_evidence_cancel(cancel_event)
    if result.returncode != 0 or result.stdout_truncated:
        return False
    usable = _usable_mp3_stats(result.stdout.decode("utf-8", "replace").splitlines(), "1/44100")
    _check_evidence_cancel(cancel_event)
    return usable


def _usable_mp3_stats(lines: Sequence[str], base: str) -> bool:
    """CR-070: are these ``pre``/``post`` lines a usable account, not merely lines of the right shape?

    Every ``pre``/``post`` line must carry integer numbers and the expected time base; frame and
    packet numbers run 0, 1, 2…; sample counts are non-negative; each frame's timestamp follows
    from the samples before it; packet timestamps strictly increase. Other lines are ignored.
    The adapter keeps its own check: it may not import ``audio.render`` (layering).
    """
    frames = packets = samples = 0
    first_pts = 0
    last_post: int | None = None
    for line in lines:
        fields = line.split()
        if fields[:1] not in (["pre"], ["post"]):
            continue
        if len(fields) != (5 if fields[0] == "pre" else 4):
            return False  # truncated or padded record: shape-check before indexing
        try:
            numbers = [int(fields[1]), int(fields[2])]
            if fields[:1] == ["pre"]:
                if fields[3] != base:
                    return False
                count = int(fields[4])
                if numbers[0] != frames or count < 0:
                    return False
                if frames == 0:
                    first_pts = numbers[1]
                elif numbers[1] != first_pts + samples:
                    return False
                frames += 1
                samples += count
            else:
                if fields[3] != base or numbers[0] != packets:
                    return False
                if last_post is not None and numbers[1] <= last_post:
                    return False
                last_post = numbers[1]
                packets += 1
        except ValueError:
            return False
    return frames > 0 and packets > 0 and samples > 0


def _mp3_stats_gap(env: FFmpegEnvironment, sample_rate_hz: int, channels: int, channel_layout: str | None,
                   ) -> tuple[CapabilityEvidence, CapabilityEvidence]:
    """Empty, typed MP3 gaps: no setting is evidenced, so resolution refuses MP3 as unsupported."""
    locator = ("installed_query: libmp3lame -stats_enc_pre/-stats_enc_post (CR-063, FFmpeg 6.1+) not supported "
               "by this FFmpeg; MP3 gapless data cannot be recovered, so no MP3 setting is evidenced")

    def gap(mode: Literal["cbr", "vbr"]) -> CapabilityEvidence:
        return CapabilityEvidence(
            evidence_id=evidence_id(env, "libmp3lame", mode, sample_rate_hz, channels, channel_layout),
            executable_path=env.executable.path,
            executable_identity=env.executable.identity,
            ffmpeg_version=env.executable.version,
            platform_basis=env.platform_basis,
            encoder="libmp3lame",
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            channel_layout=channel_layout,
            bitrate_mode=mode,
            supported_bit_depths=(),
            supports_float=False,
            nominal_bitrates_bps=(),
            vbr_qualities=(),
            basis="installed_query",
            source_locator=locator,
            nominal_confirmation="not_applicable",
        )

    return gap("cbr"), gap("vbr")


def collect_mp3_evidence(env: FFmpegEnvironment, sample_rate_hz: int, channels: int, channel_layout: str | None,
                         work_dir: Path, *, cbr_candidates_bps: Sequence[int] = MP3_CBR_CANDIDATES_BPS,
                         vbr_candidates: Sequence[int] = MP3_VBR_CANDIDATES,
                         encoder_stats: bool | None = None,
                         cancel_event: threading.Event | None = None,
                         ) -> tuple[CapabilityEvidence, CapabilityEvidence]:
    """Installed-query evidence for libmp3lame CBR and VBR at one rate and layout.

    CBR: a request is evidenced (``validated_setting``) only when the encode
    succeeds and every written frame header carries exactly that bitrate, the
    requested sample rate and a matching channel mode.

    VBR: confirm a written Xing quality indicator, or V0 alone by the exact
    inspected build/platform's primary source plus a real encode and full
    decode. A zero Xing field alone proves no quality. Every other unconfirmed
    request remains a conservative gap (C-04, formats F12/F15, U-14).
    """
    _check_evidence_cancel(cancel_event)
    _require_encoder(env, "libmp3lame")
    if encoder_stats is None:
        encoder_stats = probe_mp3_encoder_stats(env, cancel_event=cancel_event)
    _check_evidence_cancel(cancel_event)
    if not encoder_stats:
        return _mp3_stats_gap(env, sample_rate_hz, channels, channel_layout)
    work_dir.mkdir(parents=True, exist_ok=True)
    source = _source_args(sample_rate_hz, channels, channel_layout)
    stem = f"mp3-{sample_rate_hz}-{channels}"
    cbr_outputs = [(work_dir / f"{stem}-cbr{bps}.mp3",
                    ("-c:a", "libmp3lame", "-b:a", str(bps), "-write_xing", "0", "-id3v2_version", "0",
                     "-f", "mp3")) for bps in cbr_candidates_bps]
    vbr_outputs = [(work_dir / f"{stem}-vbr{q}.mp3",
                    ("-c:a", "libmp3lame", "-q:a", str(q), "-id3v2_version", "0", "-f", "mp3"))
                   for q in vbr_candidates]
    written, tail = _encode_outputs(env, source, cbr_outputs + vbr_outputs, cancel_event=cancel_event)
    _check_evidence_cancel(cancel_event)

    accepted_cbr: list[int] = []
    for bps, (path, _) in zip(cbr_candidates_bps, cbr_outputs, strict=True):
        _check_evidence_cancel(cancel_event)
        if not written.get(path):
            continue
        try:
            frames = read_mp3_frames(path.read_bytes())
        except ValueError:
            _check_evidence_cancel(cancel_event)
            continue
        _check_evidence_cancel(cancel_event)
        if all(f.bitrate_bps == bps and f.sample_rate_hz == sample_rate_hz and _mp3_layout_ok(f, channels)
               for f in frames):
            accepted_cbr.append(bps)

    vbr_observed: list[int] = []
    zero_quality_vbr: list[int] = []
    confirmed_vbr: list[int] = []
    quality_fields: list[str] = []
    for quality, (path, _) in zip(vbr_candidates, vbr_outputs, strict=True):
        _check_evidence_cancel(cancel_event)
        if not written.get(path):
            continue
        try:
            data = path.read_bytes()
            frames = read_mp3_frames(data)
        except ValueError:
            _check_evidence_cancel(cancel_event)
            continue
        _check_evidence_cancel(cancel_event)
        if not all(f.sample_rate_hz == sample_rate_hz and _mp3_layout_ok(f, channels) for f in frames):
            continue
        xing = read_xing_tag(data, frames[0])
        _check_evidence_cancel(cancel_event)
        if xing is None or xing.tag != b"Xing":
            continue
        vbr_observed.append(quality)
        if xing.quality == 0:
            zero_quality_vbr.append(quality)
        quality_fields.append(f"q{quality}:{xing.quality}")
        if xing.quality is not None and 0 < xing.quality <= 100 and (100 - xing.quality) // 10 == quality:
            confirmed_vbr.append(quality)

    def record(mode: Literal["cbr", "vbr"], bitrates: tuple[int, ...], qualities: tuple[int, ...],
               locator: str, confirmed: bool) -> CapabilityEvidence:
        return CapabilityEvidence(
            evidence_id=evidence_id(env, "libmp3lame", mode, sample_rate_hz, channels, channel_layout),
            executable_path=env.executable.path,
            executable_identity=env.executable.identity,
            ffmpeg_version=env.executable.version,
            platform_basis=env.platform_basis,
            encoder="libmp3lame",
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            channel_layout=channel_layout,
            bitrate_mode=mode,
            supported_bit_depths=(),
            supports_float=False,
            nominal_bitrates_bps=bitrates,
            vbr_qualities=qualities,
            basis="installed_query",
            source_locator=locator,
            nominal_confirmation="validated_setting" if confirmed else "not_applicable",
        )

    failure_note = f"; stderr: {' | '.join(tail[-3:])}" if tail and not accepted_cbr else ""
    first = cbr_candidates_bps[0] if cbr_candidates_bps else "-"
    last = cbr_candidates_bps[-1] if cbr_candidates_bps else "-"
    cbr = record("cbr", tuple(accepted_cbr), (),
                 f"installed_query: libmp3lame -b:a over {len(cbr_candidates_bps)} candidates {first}..{last} bps; "
                 f"accepted when every written frame header carries the request{failure_note}",
                 bool(accepted_cbr))
    vbr_confirmed = bool(confirmed_vbr) and confirmed_vbr == vbr_observed
    vbr = record("vbr", (), tuple(confirmed_vbr) if vbr_confirmed else (),
                 f"installed_query: libmp3lame -q:a over {list(vbr_candidates)}; VBR mode (Xing) observed for "
                 f"{vbr_observed}; Xing quality fields {quality_fields}; "
                 + ("quality confirmed from the Xing quality indicator" if vbr_confirmed else
                    "gap: written headers do not confirm the requested quality, so no VBR quality is evidenced"),
                 vbr_confirmed)
    # This muxer writes a zero Xing quality: it is NOT a V-level read-back.
    # Primary matching source instead proves the exact -q:a 0 -> LAME V0
    # mapping. Restrict that proof to the inspected build/platform, with a real
    # encode, Xing mode/header checks and a complete decoded read-back at this
    # very rate/layout. No proof for another build or any unobserved V quality.
    if (not vbr_confirmed and 0 in zero_quality_vbr and _matched_lossy_build(env)
            and (channels, channel_layout) in ((1, "mono"), (2, "stereo"))):
        path = vbr_outputs[list(vbr_candidates).index(0)][0]
        if _decoded_evidence_ok(env, path, sample_rate_hz, channels, cancel_event=cancel_event):
            vbr = replace(vbr, vbr_qualities=(0,), basis="build_matched_evidence",
                          nominal_confirmation="validated_setting",
                          source_locator=f"{_LAME_SOURCE}: mp3lame_encode_init maps -q:a 0 directly "
                                         "to lame_set_VBR_quality(0); matching build/platform; "
                                         "local V0 encode, Xing VBR mode, every frame rate/layout "
                                         "and full decoded PCM checked; Xing quality is not used")
    _check_evidence_cancel(cancel_event)
    return cbr, vbr


def _read_pcm_header(muxer: str, data: bytes) -> PcmHeader:
    readers = {"wav": read_wav_header, "aiff": read_aiff_header, "flac": read_flac_header, "ipod": read_alac_header}
    return readers[muxer](data)


def collect_lossless_evidence(env: FFmpegEnvironment, encoder: str, sample_rate_hz: int, channels: int,
                              channel_layout: str | None, work_dir: Path, *,
                              cancel_event: threading.Event | None = None) -> CapabilityEvidence:
    """Installed-query evidence for a lossless encoder at one rate and layout.

    Each candidate representation is encoded and accepted only when the
    written header shows the expected depth/float kind, rate and channels.
    """
    _check_evidence_cancel(cancel_event)
    _require_encoder(env, encoder)
    pcm = _PCM_RE.match(encoder)
    if pcm:
        kind, bits, endian = pcm.groups()
        muxer, extension = ("wav", ".wav") if endian == "le" else ("aiff", ".aiff")
        candidates = ((muxer, extension, (), None if kind == "f" else int(bits), kind == "f"),)
    elif encoder in _LOSSLESS_CANDIDATES:
        candidates = _LOSSLESS_CANDIDATES[encoder]
    else:
        raise ValueError(f"no lossless evidence procedure for encoder {encoder!r}")
    work_dir.mkdir(parents=True, exist_ok=True)
    source = _source_args(sample_rate_hz, channels, channel_layout)
    outputs = [(work_dir / f"{encoder}-{sample_rate_hz}-{channels}-{index}{extension}",
                ("-c:a", encoder, *extra, "-f", muxer))
               for index, (muxer, extension, extra, _, _) in enumerate(candidates)]
    written, tail = _encode_outputs(env, source, outputs, cancel_event=cancel_event)
    _check_evidence_cancel(cancel_event)

    depths: list[int] = []
    supports_float = False
    observed: list[str] = []
    for (muxer, _, _, depth, is_float), (path, _) in zip(candidates, outputs, strict=True):
        _check_evidence_cancel(cancel_event)
        if not written.get(path):
            observed.append(f"{path.name}: encode failed")
            continue
        try:
            header = _read_pcm_header(muxer, path.read_bytes())
        except ValueError as exc:
            _check_evidence_cancel(cancel_event)
            observed.append(f"{path.name}: {exc}")
            continue
        _check_evidence_cancel(cancel_event)
        observed.append(f"{path.name}: {header.bit_depth}-bit{' float' if header.sample_float else ''} "
                        f"{header.sample_rate_hz} Hz {header.channels} ch")
        if header.sample_rate_hz != sample_rate_hz or header.channels != channels:
            continue
        if is_float and header.sample_float and header.bit_depth == int(pcm.group(2) if pcm else 32):
            supports_float = True
        elif not is_float and not header.sample_float and header.bit_depth == depth:
            depths.append(header.bit_depth)
    _check_evidence_cancel(cancel_event)
    layout_note = ("; channel layout beyond the channel count is not verified from headers"
                   if channels > 2 else "")
    note = f"; stderr: {' | '.join(tail[-3:])}" if tail and not (depths or supports_float) else ""
    return CapabilityEvidence(
        evidence_id=evidence_id(env, encoder, "none", sample_rate_hz, channels, channel_layout),
        executable_path=env.executable.path,
        executable_identity=env.executable.identity,
        ffmpeg_version=env.executable.version,
        platform_basis=env.platform_basis,
        encoder=encoder,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        channel_layout=channel_layout,
        bitrate_mode="none",
        supported_bit_depths=tuple(sorted(set(depths))),
        supports_float=supports_float,
        nominal_bitrates_bps=(),
        vbr_qualities=(),
        basis="installed_query",
        source_locator="installed_query: written headers " + "; ".join(observed) + layout_note + note,
        nominal_confirmation="not_applicable",
    )


def revalidate_evidence(env: FFmpegEnvironment, evidence: CapabilityEvidence) -> bool:
    """True when the evidence still matches this executable and platform (C-04)."""
    try:
        env.executable.revalidate()
    except ToolMissing:
        return False
    return (evidence.executable_path == env.executable.path
            and evidence.executable_identity == env.executable.identity
            and evidence.ffmpeg_version == env.executable.version
            and evidence.platform_basis == env.platform_basis)
