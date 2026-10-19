"""sips adapter: local image inspection and conversion for artwork preparation (metadata § 8, MD24).

macOS ``/usr/bin/sips`` is run as an argv list (no shell), on local absolute paths
only, in its own process group, with both pipes drained and a monotonic deadline.
Inspection is bounded at 60 s (ADR-0018 image-inspection deadline); a conversion
gets the same bound. Outcomes, all ``ToolError`` subclasses so the caller maps
them by stage (``to_failure``):

- ``ToolTimeout``: the deadline expired; the process group was terminated (2 s
  grace, then killed) and reaped;
- ``ToolMissing``: the executable vanished after preflight;
- ``ToolFailed``: sips exited non-zero (stderr tail kept);
- ``NotAnImage``: sips exited 0 but did not recognise the file. sips reports an
  unreadable or non-image file as a warning with exit 0 and ``<nil>`` fields, so
  success is judged from the parsed properties, never from the exit status alone.

``locate_sips`` raises ``EnvironmentError_`` (exit 3) when sips is absent or
unusable at preflight. Nothing here imports at package start-up; the module
imports only the standard library and ``norma.models``/``norma.errors``.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from norma.errors import Cancelled, EnvironmentError_, ToolFailed, ToolError, ToolMissing, ToolTimeout
from norma.models import Identity

INSPECT_DEADLINE_S = 60.0
TERMINATE_GRACE_S = 2.0
_POLL_S = 0.05
_TAIL_LINES = 20
_NIL = "<nil>"
SRGB_PROFILE = Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc")
"""The profile that turns CMYK and other colour spaces into RGB (``sips -s space`` is not supported)."""


class NotAnImage(ToolError):
    """sips ran but could not read the file as an image."""


@dataclass(frozen=True)
class Sips:
    """A resolved sips executable with its identity and version at preflight."""

    path: Path
    identity: Identity
    version: str


@dataclass(frozen=True)
class ImageInfo:
    """Properties read with ``sips -g format -g pixelWidth -g pixelHeight -g space``."""

    format: str
    width_px: int
    height_px: int
    space: str


def locate_sips(name: str | Path = "sips", *, search_path: str | None = None) -> Sips:
    """Resolve sips and read its version; failure is an environment error (exit 3)."""
    text = str(name)
    if os.sep in text:
        path = Path(text)
        if not (path.is_absolute() and path.is_file() and os.access(path, os.X_OK)):
            raise EnvironmentError_(f"sips not found or not executable: {text}")
    else:
        found = shutil.which(text, path=search_path)
        if found is None:
            raise EnvironmentError_(f"required tool not found on PATH: {text}")
        path = Path(found).absolute()
    info = os.stat(path)
    try:
        returncode, stdout, _stderr = _run((str(path), "--version"), INSPECT_DEADLINE_S)
    except (OSError, ToolError) as exc:
        raise EnvironmentError_(f"sips at {path} is unusable: {exc}") from None
    version = stdout.decode("utf-8", "replace").strip()
    if returncode != 0 or not version:
        raise EnvironmentError_(f"sips at {path} gave no usable version")
    return Sips(path, (info.st_dev, info.st_ino), version)


def _absolute(path: Path, role: str) -> str:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"sips {role} must be an absolute local path, got {path!r}")
    return str(path)


def inspect_argv(tool: Sips, source: Path) -> tuple[str, ...]:
    return (str(tool.path), "-g", "format", "-g", "pixelWidth", "-g", "pixelHeight", "-g", "space",
            _absolute(source, "input"))


def convert_argv(tool: Sips, source: Path, dest: Path, *, image_format: str, quality: int | None = None,
                 max_side: int | None = None, profile: Path | None = None) -> tuple[str, ...]:
    """``sips`` arguments for one conversion.

    ``max_side`` is only passed when the image is larger: ``-Z`` also enlarges.
    ``profile`` converts the colour space (``-m``); sips cannot set ``space`` directly.
    """
    argv = [str(tool.path)]
    if profile is not None:
        argv += ["-m", _absolute(profile, "profile")]
    if max_side is not None:
        if max_side <= 0:
            raise ValueError("max_side must be positive")
        argv += ["-Z", str(max_side)]
    argv += ["-s", "format", image_format]
    if quality is not None:
        argv += ["-s", "formatOptions", str(quality)]
    argv += [_absolute(source, "input"), "--out", _absolute(dest, "output")]
    return tuple(argv)


def parse_properties(text: str) -> dict[str, str]:
    """The ``key: value`` property lines of ``sips -g`` output (the path lines carry no colon-space)."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith((" ", "\t")) and ": " in line:
            key, _, value = line.strip().partition(": ")
            found[key] = value.strip()
    return found


def parse_info(text: str) -> ImageInfo | None:
    """``ImageInfo`` from sips output, or ``None`` when any property is missing, ``<nil>`` or not valid."""
    props = parse_properties(text)
    try:
        fmt, space = props["format"], props["space"]
        width, height = int(props["pixelWidth"]), int(props["pixelHeight"])
    except (KeyError, ValueError):
        return None
    if _NIL in (fmt, space) or not fmt or not space or width <= 0 or height <= 0:
        return None
    return ImageInfo(fmt.lower(), width, height, space)


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    pgid = proc.pid
    for signum, grace in ((signal.SIGTERM, TERMINATE_GRACE_S), (signal.SIGKILL, None)):
        try:
            os.killpg(pgid, signum)
        except (ProcessLookupError, PermissionError):
            break
        if grace is not None:
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.01)
            if proc.poll() is not None:
                break
    proc.wait()
    try:
        proc.communicate(timeout=1)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


def _run(argv: tuple[str, ...], deadline_s: float,
         cancel_event: threading.Event | None = None) -> tuple[int, bytes, tuple[str, ...]]:
    """Run argv with both pipes drained, a monotonic deadline and process-group teardown."""
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()
    started = time.monotonic()
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, close_fds=True, start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise ToolMissing(f"sips could not be started: {exc}", argv=argv) from None
    except PermissionError as exc:
        raise ToolMissing(f"sips is not executable: {exc}", argv=argv) from None
    deadline = started + deadline_s
    out = err = b""
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(proc)
                raise ToolTimeout(f"sips exceeded its {deadline_s:g} s deadline", deadline_s=deadline_s, argv=argv)
            if cancel_event is not None and cancel_event.is_set():
                _terminate(proc)
                raise Cancelled()
            try:
                out, err = proc.communicate(timeout=min(_POLL_S, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        if proc.poll() is None:
            _terminate(proc)
        raise
    tail = tuple(deque((line for line in err.decode("utf-8", "replace").splitlines() if line), maxlen=_TAIL_LINES))
    return proc.returncode, out, tail


def inspect(tool: Sips, source: Path, *, deadline_s: float = INSPECT_DEADLINE_S,
            cancel_event: threading.Event | None = None) -> ImageInfo:
    """Format, pixel size and colour space of one local image."""
    argv = inspect_argv(tool, source)
    returncode, stdout, tail = _run(argv, deadline_s, cancel_event)
    if returncode != 0:
        raise ToolFailed(f"sips exited {returncode} inspecting {source.name}", returncode=returncode,
                         argv=argv, stderr_tail=tail)
    info = parse_info(stdout.decode("utf-8", "replace"))
    if info is None:
        raise NotAnImage(f"sips could not read {source.name} as an image", argv=argv, stderr_tail=tail)
    return info


def convert(tool: Sips, source: Path, dest: Path, *, image_format: str, quality: int | None = None,
            max_side: int | None = None, profile: Path | None = None,
            deadline_s: float = INSPECT_DEADLINE_S, cancel_event: threading.Event | None = None) -> None:
    """Write a converted and/or scaled copy of ``source`` to ``dest`` (which must not exist)."""
    if dest.exists():
        raise ValueError(f"sips output must not exist: {dest}")
    argv = convert_argv(tool, source, dest, image_format=image_format, quality=quality, max_side=max_side,
                        profile=profile)
    returncode, _stdout, tail = _run(argv, deadline_s, cancel_event)
    if returncode != 0:
        raise ToolFailed(f"sips exited {returncode} converting {source.name}", returncode=returncode,
                         argv=argv, stderr_tail=tail)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise NotAnImage(f"sips produced no output for {source.name}", argv=argv, stderr_tail=tail)
