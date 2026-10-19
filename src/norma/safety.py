"""Write-side file safety: the run write set, private candidates and verified publication.

Implements the file-safety rules: the filesystem half of the output-path modes,
report, template and log files, and candidate structure, producing and
consuming C-08 records. ``paths`` decides names; this module touches the
filesystem.

Guarantees, as far as the § 1 threat boundary allows:

* nothing here opens a source for writing, or renames, replaces or removes one;
* a final name is created only by an exclusive primitive (``renamex_np`` with
  ``RENAME_EXCL``, or a hard link followed by unlinking the private name), or
  replaced by ``os.replace`` only when ``--overwrite`` authorised it and a
  private preimage of the old output is held;
* every temporary (candidate, preimage, receipt, report candidate) is created
  exclusively in the destination folder, registered at once, and removed by
  identity, never by glob;
* failures roll back this run's publications in reverse order and report the
  real survivors;
* candidates stay 0600 while private; just before publication each takes the
  ordinary new-file mode (0666 less the umask captured at run start) and the
  audio candidate takes the source's captured modification time; the source
  itself is never touched (§ 5.1, CR-046);
* a per-volume space reservation, held under the registry's space lock, is
  checked before a candidate is allocated, so a shortfall refuses before any
  render with ``disk_full`` and leaves nothing behind (§ 5.1 Space).

Parents are pinned by canonical path and directory identity and rechecked at
each allocation and publication. [judgement] No directory descriptor is held
open across the run: relative ``dir_fd`` operations are refused by the test
hermeticity guard, and one descriptor per output folder would not scale. The
identity recheck narrows, but cannot close, the race window (§ 1).
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import secrets
import stat
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Literal

from norma.discovery import check_signature
from norma.errors import EXIT_FAILED, FileFailure, FileOutcomeError, NormaError, SkipFile, UsageError
from norma.models import (
    RUN_LEVEL_PURPOSES,
    BundleMember,
    Diagnostic,
    Execution,
    FrozenMap,
    Identity,
    InputItem,
    OutputPlan,
    ReadDependency,
    ReasonCode,
    RunOptions,
    RunWriteSet,
    Status,
    TargetFormat,
    WriteDestination,
)
from norma.paths import AUDIO_MEMBER_ID, clash_failure, comparison_key, find_clashes

RESERVE_STAGE = "writes_reserved"
ALLOCATE_STAGE = "allocated"
COMMIT_STAGE = "committing"
CLEANUP_STAGE = "discarded"
SOURCE_STAGE = "source"
STRUCTURE_STAGE = "content_verified"

PRIVATE_PREFIX = ".norma-"
"""Every private name starts with this; discovery ignores hidden entries (§ 2.2)."""

RENAME_EXCL = 0x00000004
"""``RENAME_EXCL`` from the macOS SDK's ``sys/stdio.h`` (§ 9.1)."""

_HASH_CHUNK = 1 << 20
_PREIMAGE_ATTEMPTS = 8
_LINK_REFUSED = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK})
_EXCL_UNSUPPORTED = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS})

EXTENSION_KINDS: FrozenMap = FrozenMap({
    ".wav": "wav", ".aif": "aiff", ".aiff": "aiff", ".aifc": "aiff", ".flac": "flac",
    ".m4a": "mp4", ".mp4": "mp4", ".mp3": "mpeg", ".aac": "adts", ".caf": "caf",
})
"""Signature kind (``discovery.check_signature``) each audio extension's candidate must show."""


class RunOutputError(NormaError):
    """A report or template file could not be published: a run error, exit 1, no file row (§ 6)."""

    exit_code = EXIT_FAILED


# --- Small helpers ----------------------------------------------------------------------


def _identity(info: os.stat_result) -> Identity:
    return (info.st_dev, info.st_ino)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _kind(info: os.stat_result | None) -> str:
    if info is None:
        return "absent"
    mode = info.st_mode
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    return "special"


def _io_reason(exc: OSError, default: ReasonCode = "commit_error") -> ReasonCode:
    """§ 8: specific ``disk_full``/``permission_denied``, otherwise the stage's generic code."""
    if exc.errno in (errno.ENOSPC, errno.EDQUOT):
        return "disk_full"
    if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        return "permission_denied"
    return default


def _failure(reason: ReasonCode, stage: str, detail: str) -> FileFailure:
    return FileFailure(reason, stage, detail)


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, _HASH_CHUNK, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _sha256_path(path: Path, expected: Identity | None = None) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if expected is not None and _identity(os.fstat(fd)) != expected:
            raise _failure("unsafe_output", COMMIT_STAGE, f"{path} changed identity while it was read")
        return _sha256_fd(fd)
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data``; ``os.write`` may write fewer than asked."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(errno.EIO, "the volume accepted no bytes")
        view = view[written:]


def _fsync_directory(path: Path) -> str | None:
    """Request directory synchronisation; return a note when the volume refuses it (§ 7)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as exc:
        return f"directory sync of {path} unavailable: {exc.strerror or exc}"
    try:
        os.fsync(fd)
    except OSError as exc:
        return f"directory sync of {path} unavailable: {exc.strerror or exc}"
    finally:
        os.close(fd)
    return None


# --- In-place refusal (§ 4.1) ---------------------------------------------------------------


def refuse_in_place(options: RunOptions) -> None:
    """``--in-place``/``--no-backup`` are usage errors before any discovery, prompt or write (D-028)."""
    flags = [name for name, value in (("--in-place", options.in_place), ("--no-backup", options.no_backup))
             if value]
    if flags:
        raise UsageError(f"{' and '.join(flags)} {'is' if len(flags) == 1 else 'are'} not available: Norma never "
                         "replaces a source; outputs are always new copies (use -o, -s or the default "
                         "Normalised folder)")


# --- Canonical parents and pins (§ 2.6 (b)) -----------------------------------------------


def canonical_directory(path: Path | str) -> Path:
    """The canonical form of an output folder, resolved through its nearest existing ancestor."""
    return Path(os.path.realpath(os.path.abspath(path)))


def canonical_destination(path: Path | str) -> Path:
    """A run-level file destination with a canonical parent and the final component kept as given."""
    absolute = Path(os.path.abspath(path))
    return canonical_directory(absolute.parent) / absolute.name


@dataclass(frozen=True)
class ParentPin:
    """The intended parent of a destination and its nearest existing ancestor's identity."""

    parent: Path
    anchor: Path
    identity: Identity

    @property
    def exists(self) -> bool:
        return self.anchor == self.parent


class UnsafeAncestry(Exception):
    """The destination's ancestry is not a plain canonical folder chain (§ 2.6 (b))."""


def pin_parent(final_path: Path) -> ParentPin:
    """Pin ``final_path``'s parent; raise ``UnsafeAncestry`` for links, files or changed folders."""
    parent = final_path.parent
    if os.path.realpath(parent) != str(parent):
        raise UnsafeAncestry(f"{parent} is reached through a symbolic link or is not canonical")
    anchor = parent
    while True:
        try:
            info = _lstat(anchor)
        except NotADirectoryError:
            info = None  # a file sits higher up; keep climbing to report it
        if info is not None:
            break
        if anchor.parent == anchor:
            raise UnsafeAncestry(f"no existing ancestor for {parent}")
        anchor = anchor.parent
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafeAncestry(f"{anchor} is a {_kind(info)}, not a folder")
    return ParentPin(parent, anchor, _identity(info))


def recheck_pin(pin: ParentPin) -> ParentPin:
    """Confirm the pinned ancestor is unchanged; return the current pin (it may now exist deeper)."""
    info = _lstat(pin.anchor)
    if info is None or not stat.S_ISDIR(info.st_mode) or _identity(info) != pin.identity:
        raise UnsafeAncestry(f"output folder {pin.anchor} changed since it was checked")
    current = pin_parent(pin.parent / "_")
    if current.anchor != pin.anchor and not (current.anchor.is_relative_to(pin.anchor)):
        raise UnsafeAncestry(f"output folder {pin.parent} changed since it was checked")
    return current


# --- Protected reads (§ 2.6) --------------------------------------------------------------


def protect(path: Path | str, purpose: str) -> ReadDependency:
    """A C-08 read dependency: canonical path and the identity of what the run reads."""
    info = os.stat(path)
    return ReadDependency(Path(os.path.realpath(path)), _identity(info), purpose)


def protect_inputs(items: Iterable[InputItem]) -> tuple[ReadDependency, ...]:
    """Every C-01 input, whatever its later outcome, as a protected read."""
    return tuple(ReadDependency(item.source_path, item.identity, "input") for item in items)


@dataclass(frozen=True)
class _Protected:
    keys: FrozenMap  # comparison key -> purpose
    identities: FrozenMap  # identity -> purpose

    @classmethod
    def of(cls, reads: Sequence[ReadDependency]) -> _Protected:
        return cls(FrozenMap({comparison_key(read.path): read.purpose for read in reads}),
                   FrozenMap({read.identity: read.purpose for read in reads}))


# --- The run write set (§ 2.6, § 3 table, § 5.3) ---------------------------------------------


@dataclass(frozen=True)
class RunOutput:
    """A run-level file destination supplied by WS-03: the log, a report or a template export."""

    purpose: Literal["log", "report", "template_export"]
    path: Path
    parent_must_exist: bool = True
    """False only for the automatic report, whose parent normalise may create (§ 6)."""

    def __post_init__(self) -> None:
        if self.purpose not in RUN_LEVEL_PURPOSES:
            raise ValueError(f"{self.purpose!r} is not a run-level purpose")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("run output paths must be absolute")

    @property
    def automatic(self) -> bool:
        """The ``normalise`` automatic report: it never refuses or replaces a taken name (§ 6, CR-049)."""
        return self.purpose == "report" and not self.parent_must_exist


@dataclass(frozen=True)
class Occupant:
    """What a destination name held when it was last checked."""

    kind: str  # absent, regular, symlink, directory, special
    identity: Identity | None
    links: int
    parent: ParentPin | None = None
    """The log's preflight parent pin, which ``open_log`` rechecks (IC-2 recheck R-2); else None."""


def _occupant(path: Path) -> Occupant:
    info = _lstat(path)
    if info is None:
        return Occupant("absent", None, 0)
    return Occupant(_kind(info), _identity(info), info.st_nlink)


@dataclass(frozen=True)
class Preflight:
    """The checked run write set, per-file refusals and the pins publication rechecks."""

    write_set: RunWriteSet
    plans: FrozenMap  # input_index -> OutputPlan with the collision state set
    refusals: FrozenMap  # input_index -> SkipFile | FileFailure, terminal before any worker starts
    pins: FrozenMap  # destination_id -> ParentPin
    occupants: FrozenMap  # destination_id -> Occupant at the check

    def eligible(self) -> tuple[int, ...]:
        return tuple(index for index in self.plans if index not in self.refusals)


def destination_id(input_index: int | None, member_or_purpose: str) -> str:
    return member_or_purpose if input_index is None else f"{input_index}:{member_or_purpose}"


def _check_member(member: BundleMember, protected: _Protected, *, overwrite: bool
                  ) -> tuple[ParentPin | None, Occupant | None, FileOutcomeError | None]:
    """Checks (a)–(e) for one bundle member; the first failure is returned, not raised."""
    try:
        pin = pin_parent(member.final_path)
    except UnsafeAncestry as exc:
        return None, None, _failure("unsafe_output", RESERVE_STAGE, str(exc))
    if member.comparison_key != comparison_key(member.final_path):
        return pin, None, _failure("unsafe_output", RESERVE_STAGE, f"{member.final_path}: stale comparison key")
    if member.comparison_key in protected.keys:
        return pin, None, _failure("unsafe_output", RESERVE_STAGE,
                                   f"{member.final_path} is the {protected.keys[member.comparison_key]} "
                                   "this run reads, once case and Unicode normalisation are ignored")
    try:
        occupant = _occupant(member.final_path)
    except PermissionError as exc:
        return pin, None, _failure("permission_denied", RESERVE_STAGE, str(exc))
    if occupant.kind in ("symlink", "directory", "special"):
        return pin, occupant, _failure("unsafe_output", RESERVE_STAGE,
                                       f"{member.final_path} exists as a {occupant.kind}; it is never replaced")
    if occupant.identity is not None and occupant.identity in protected.identities:
        return pin, occupant, _failure("unsafe_output", RESERVE_STAGE,
                                       f"{member.final_path} is another name for the "
                                       f"{protected.identities[occupant.identity]} this run reads")
    if occupant.kind == "regular" and not overwrite:
        return pin, occupant, SkipFile("output_exists", RESERVE_STAGE,
                                       f"{member.final_path} already exists; use --overwrite to replace it")
    return pin, occupant, None


def _rank(error: FileOutcomeError) -> int:
    return {"unsafe_output": 0, "permission_denied": 0, "name_clash": 1, "output_exists": 2}.get(error.reason, 0)


def preflight(plans: Mapping[int, OutputPlan], run_outputs: Sequence[RunOutput],
              protected: Sequence[ReadDependency], *, overwrite: bool, revision: int = 0) -> Preflight:
    """Build and check the immutable ``RunWriteSet`` before the first open for writing (§ 2.6).

    Raises ``UsageError`` (exit 2, no file rows) for every run-level refusal. Per-file results
    are returned as terminal refusals: ``unsafe_output`` before ``name_clash`` before
    ``output_exists`` (§ 3 decision order). Clash participants keep their reserved keys.
    A taken automatic-report name is no refusal: publication takes an ordinal (§ 6, CR-049).
    """
    guard = _Protected.of(protected)
    problems: list[str] = []
    destinations: list[WriteDestination] = []
    pins: dict[str, ParentPin] = {}
    occupants: dict[str, Occupant] = {}
    member_keys: dict[str, tuple[int, str]] = {}
    for index, plan in plans.items():
        for member in plan.members:
            member_keys.setdefault(member.comparison_key, (index, member.member_id))

    # Run-level destinations first: their refusal wins over any file-level result (§ 2.6).
    run_keys: dict[str, str] = {}
    for output in run_outputs:
        final = canonical_destination(output.path)
        key = comparison_key(final)
        label = output.purpose.replace("_", " ")
        dest_id = destination_id(None, output.purpose)
        mode: Literal["publish", "append"] = "append" if output.purpose == "log" else "publish"
        destinations.append(WriteDestination(dest_id, output.purpose, None, final, key, mode))
        try:
            pin = pin_parent(final)
        except UnsafeAncestry as exc:
            problems.append(f"{label} {final}: {exc}")
            continue
        pins[dest_id] = pin
        if output.parent_must_exist and not pin.exists:
            problems.append(f"{label} {final}: its folder {final.parent} does not exist")
        if output.automatic:
            # CR-049: a taken base name is no conflict. Publication never replaces anything
            # there and takes the next free ordinal, rechecking each name (§ 6). Only a clash
            # inside the write set itself, a malformed request, is still refused.
            if key in member_keys:
                index, member_id = member_keys[key]
                problems.append(f"{label} {final} is also planned as input {index}'s {member_id} output")
            if key in run_keys:
                problems.append(f"{label} {final} is also the {run_keys[key].replace('_', ' ')}")
            run_keys[key] = output.purpose
            occupants[dest_id] = _occupant(final)
            continue
        if key in guard.keys:
            problems.append(f"{label} {final} is the {guard.keys[key]} this run reads")
        if key in member_keys:
            index, member_id = member_keys[key]
            problems.append(f"{label} {final} is also planned as input {index}'s {member_id} output")
        if key in run_keys:
            problems.append(f"{label} {final} is also the {run_keys[key].replace('_', ' ')}")
        run_keys[key] = output.purpose
        occupant = _occupant(final)
        if output.purpose == "log":
            occupant = replace(occupant, parent=pin)  # rechecked when the log is opened (§ 6)
        occupants[dest_id] = occupant
        if occupant.kind in ("symlink", "directory", "special"):
            problems.append(f"{label} {final} exists as a {occupant.kind}")
        elif occupant.identity is not None and occupant.identity in guard.identities:
            problems.append(f"{label} {final} is another name for the "
                            f"{guard.identities[occupant.identity]} this run reads")
        elif occupant.kind == "regular":
            if output.purpose == "log":
                if occupant.links != 1:
                    problems.append(f"log {final} has {occupant.links} links; refusing to append")
            elif not overwrite:
                problems.append(f"{label} {final} already exists; use --overwrite to replace it")
    if problems:
        raise UsageError("; ".join(problems))

    refusals: dict[int, FileOutcomeError] = {}

    def refuse(index: int, error: FileOutcomeError) -> None:
        current = refusals.get(index)
        if current is None or _rank(error) < _rank(current):
            refusals[index] = error

    for index, plan in plans.items():
        for member in plan.members:
            dest_id = destination_id(index, member.member_id)
            destinations.append(WriteDestination(dest_id, member.purpose, index, member.final_path,
                                                 member.comparison_key, "publish"))
            pin, occupant, error = _check_member(member, guard, overwrite=plan.overwrite_authorised)
            if pin is not None:
                pins[dest_id] = pin
            if occupant is not None:
                occupants[dest_id] = occupant
            if error is not None:
                refuse(index, error)
    for index, clashes in find_clashes(plans).items():
        refuse(index, clash_failure(clashes))

    updated: dict[int, OutputPlan] = {}
    for index, plan in plans.items():
        error = refusals.get(index)
        exists = any(occupants.get(destination_id(index, m.member_id), Occupant("absent", None, 0)).kind
                     == "regular" for m in plan.members)
        if error is not None and error.reason == "name_clash":
            collision = "clash_between_inputs"
        elif exists:
            collision = "exists_overwrite" if plan.overwrite_authorised and error is None else "exists_skip"
        else:
            collision = "none"
        updated[index] = replace(plan, collision=collision)
    write_set = RunWriteSet(revision, tuple(destinations), tuple(protected), True)
    return Preflight(write_set, FrozenMap(updated), FrozenMap(refusals), FrozenMap(pins), FrozenMap(occupants))


class ReservationBook:
    """Run-long reservations for action-changing retries (§ 5.3).

    Every key reserved at preflight stays held for the whole run. A retry's newly introduced
    member is checked, under the book's lock, against every protected read, run destination
    and existing reservation; only the retrying file is refused.
    """

    def __init__(self, checked: Preflight, protected: Sequence[ReadDependency]) -> None:
        self._lock = threading.Lock()
        self._guard = _Protected.of(protected)
        self._holders: dict[str, tuple[int | None, str]] = {}
        self._pins: dict[str, ParentPin] = dict(checked.pins)
        for dest in checked.write_set.destinations:
            owner = dest.purpose if dest.input_index is None else dest.destination_id.split(":", 1)[1]
            self._holders.setdefault(dest.comparison_key, (dest.input_index, owner))

    def holder(self, key: str) -> tuple[int | None, str] | None:
        with self._lock:
            return self._holders.get(key)

    def pin(self, input_index: int, member_id: str) -> ParentPin | None:
        with self._lock:
            return self._pins.get(destination_id(input_index, member_id))

    def reserve_retry(self, input_index: int, plan: OutputPlan) -> OutputPlan:
        """Reserve a retry's new destinations or raise the retrying file's refusal."""
        with self._lock:
            fresh: list[BundleMember] = []
            for member in plan.members:
                holder = self._holders.get(member.comparison_key)
                if holder == (input_index, member.member_id):
                    continue
                if holder is not None:
                    owner = "a run output" if holder[0] is None else f"input {holder[0]} ({holder[1]})"
                    if holder[0] is None:
                        raise _failure("unsafe_output", RESERVE_STAGE,
                                       f"a retry needed {member.final_path}, which is this run's {holder[1]}")
                    raise _failure("name_clash", RESERVE_STAGE,
                                   f"a retry needed {member.final_path}, which this run has already reserved "
                                   f"for {owner}")
                if any(other.comparison_key == member.comparison_key for other in fresh):
                    raise _failure("name_clash", RESERVE_STAGE,
                                   f"a retry planned {member.final_path} for two members of this file")
                fresh.append(member)
            pins: dict[str, ParentPin] = {}
            for member in fresh:
                pin, _occupant_now, error = _check_member(member, self._guard, overwrite=plan.overwrite_authorised)
                if error is not None:
                    raise error
                assert pin is not None
                pins[destination_id(input_index, member.member_id)] = pin
            for member in fresh:
                self._holders[member.comparison_key] = (input_index, member.member_id)
            self._pins.update(pins)
            return plan


# --- Source snapshots (§ 5.1 "Source stability") -----------------------------------------


@dataclass(frozen=True)
class SourceSnapshot:
    """Identity, size, nanosecond times and SHA-256 of a source, captured before processing."""

    identity: Identity
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


def _stat_tuple(info: os.stat_result) -> tuple[Identity, int, int, int]:
    return _identity(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns


def snapshot_source(item: InputItem, *, stage: str = SOURCE_STAGE) -> SourceSnapshot:
    """Read-only capture with stats checked around the hash read; never written, never touched."""
    try:
        fd = os.open(item.source_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except FileNotFoundError:
        raise _failure("source_changed", stage, f"{item.source_path} disappeared") from None
    except PermissionError as exc:
        raise _failure("permission_denied", stage, str(exc)) from None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _failure("source_changed", stage, f"{item.source_path} was replaced by a link") from None
        raise _failure("probe_error", stage, str(exc)) from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or _identity(before) != item.identity:
            raise _failure("source_changed", stage, f"{item.source_path} is not the file discovered")
        try:
            digest = _sha256_fd(fd)
        except OSError as exc:
            raise _failure(_io_reason(exc, "probe_error"), stage, f"reading {item.source_path}: {exc}") from None
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _stat_tuple(before) != _stat_tuple(after):
        raise _failure("source_changed", stage, f"{item.source_path} changed while it was read")
    return SourceSnapshot(_identity(before), before.st_size, before.st_mtime_ns, before.st_ctime_ns, digest)


def recheck_source(item: InputItem, snapshot: SourceSnapshot, *, rehash: bool = False,
                   stage: str = SOURCE_STAGE) -> None:
    """Raise ``source_changed`` when identity, size, times or (with ``rehash``) content differ."""
    try:
        info = os.stat(item.source_path, follow_symlinks=False)
    except FileNotFoundError:
        raise _failure("source_changed", stage, f"{item.source_path} disappeared") from None
    except PermissionError as exc:
        raise _failure("permission_denied", stage, str(exc)) from None
    if (not stat.S_ISREG(info.st_mode)
            or _stat_tuple(info) != (snapshot.identity, snapshot.size, snapshot.mtime_ns, snapshot.ctime_ns)):
        raise _failure("source_changed", stage, f"{item.source_path} changed during processing")
    if rehash and snapshot_source(item, stage=stage) != snapshot:
        raise _failure("source_changed", stage, f"{item.source_path} content changed during processing")


# --- Final mode, timestamps and space (§ 5.1, CR-046) ----------------------------------------

NEW_FILE_MODE = 0o666
"""What an ordinary new file asks for before the umask; published outputs end with this less it."""

TIME_GRANULARITIES_NS = (1, 1_000, 10_000_000, 1_000_000_000, 2_000_000_000)
"""Readback granularities accepted for an applied mtime: exact, µs, 10 ms, 1 s, 2 s (truncation only)."""

SPACE_MARGIN_BYTES = 50_000_000
CONTAINER_ALLOWANCE_BYTES = 1_000_000
UNKNOWN_DURATION_BOUND_S = 3_600
"""§ 6's unknown-duration deadline, used as the duration bound of a space estimate."""

COMPRESSED_MIN_BYTES_PER_SAMPLE = 4
_UNCOMPRESSED_FORMATS = frozenset({"wav", "aiff"})
_UMASK_LOCK = threading.Lock()


def process_umask() -> int:
    """Read the process umask without leaving it changed.

    POSIX offers no read-only query, so the mask is briefly set to 077 and restored under a
    module lock. [judgement] A file another thread creates in that instant is more private,
    never less. ``RunRegistry`` reads it once, before any worker starts.
    """
    with _UMASK_LOCK:
        current = os.umask(0o077)
        os.umask(current)
    return current


def output_mode_for(umask: int) -> int:
    """The permission bits an ordinary new file gets under ``umask`` (0666 less the mask)."""
    return NEW_FILE_MODE & ~umask & 0o777


def _times_match(observed_ns: int, wanted_ns: int) -> bool:
    """The volume stored ``wanted_ns`` exactly or truncated to one of its known granularities."""
    return any(observed_ns == wanted_ns - wanted_ns % step for step in TIME_GRANULARITIES_NS)


def free_space_bytes(path: Path) -> int:
    """Bytes available to this (unprivileged) process on the volume holding ``path``."""
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def _existing_ancestor(path: Path) -> tuple[Path, os.stat_result]:
    """The nearest existing ancestor of ``path`` (itself included) and its stat."""
    current = path
    while True:
        try:
            return current, os.stat(current)
        except (FileNotFoundError, NotADirectoryError):
            if current.parent == current:
                raise
            current = current.parent


@dataclass(frozen=True)
class SpaceEstimate:
    """One file's § 5.1 space estimate; ``required_bytes`` is what must be free before a render."""

    audio_bytes: int
    sidecar_bytes: int = 0
    preimage_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("audio_bytes", "sidecar_bytes", "preimage_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def required_bytes(self) -> int:
        total = self.audio_bytes + self.sidecar_bytes + self.preimage_bytes
        return (6 * total + 4) // 5 + SPACE_MARGIN_BYTES  # ceil(1.2 × total) + margin, exactly


def estimate_space(source_size: int, duration_s: float | None, target: TargetFormat, *,
                   sidecar_bytes: int = 0, preimage_bytes: int = 0,
                   stage: str = ALLOCATE_STAGE) -> SpaceEstimate:
    """The § 5.1 estimate for rendering one source to ``target``.

    Audio is the larger of the source size and ``ceil(duration × rate) × channels × bytes per
    stored sample``, plus the container allowance; compressed targets count at least four bytes
    per sample, and an unknown duration uses the unknown-duration bound. A rate or channel count
    that cannot be established fails ``probe_error`` rather than inventing zero space.
    """
    if target.sample_rate_hz <= 0 or target.channels <= 0:
        raise _failure("probe_error", stage, "the output rate or channel count is unknown, so the space it "
                       "needs cannot be estimated")
    if duration_s is None or duration_s != duration_s or duration_s < 0:  # None, NaN or negative
        duration_s = float(UNKNOWN_DURATION_BOUND_S)
    if target.bit_depth is not None and target.bit_depth > 0:
        sample_bytes = (target.bit_depth + 7) // 8
    else:
        sample_bytes = COMPRESSED_MIN_BYTES_PER_SAMPLE  # float or unknown depth: the widest common sample
    if target.format not in _UNCOMPRESSED_FORMATS:
        sample_bytes = max(sample_bytes, COMPRESSED_MIN_BYTES_PER_SAMPLE)
    frames = math.ceil(duration_s * target.sample_rate_hz)
    pcm = frames * target.channels * sample_bytes
    return SpaceEstimate(max(int(source_size), pcm) + CONTAINER_ALLOWANCE_BYTES, sidecar_bytes, preimage_bytes)


@dataclass(frozen=True)
class SpaceReservation:
    """Bytes held for one input on one volume until its files are removed, published or retained."""

    input_index: int
    device: int
    folder: Path
    required_bytes: int


# --- Exclusive publication primitives (§ 5.2 "No-clobber") ---------------------------------

_RENAMEX: Callable[..., int] | None | bool = False  # False = not looked up yet


def _renamex() -> Callable[..., int] | None:
    global _RENAMEX
    if _RENAMEX is False:
        try:
            function = ctypes.CDLL(None, use_errno=True).renamex_np
            function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            function.restype = ctypes.c_int
            _RENAMEX = function
        except (OSError, AttributeError):
            _RENAMEX = None
    return _RENAMEX  # type: ignore[return-value]


class ExclusiveRenameUnsupported(OSError):
    """The volume or platform has no exclusive rename; the hard-link route is used instead."""


def exclusive_rename(source: Path, destination: Path) -> None:
    """``renamex_np(source, destination, RENAME_EXCL)``: ``FileExistsError`` when the name is taken."""
    function = _renamex()
    if function is None:
        raise ExclusiveRenameUnsupported(errno.ENOTSUP, "renamex_np is unavailable")
    # The same audit event os.rename raises, so audit hooks see this native namespace change.
    sys.audit("os.rename", source, destination, None, None)
    if function(os.fsencode(source), os.fsencode(destination), RENAME_EXCL) == 0:
        return
    code = ctypes.get_errno()
    if code in _EXCL_UNSUPPORTED:
        raise ExclusiveRenameUnsupported(code, os.strerror(code), str(source), None, str(destination))
    if code == errno.EEXIST:
        raise FileExistsError(code, os.strerror(code), str(source), None, str(destination))
    raise OSError(code, os.strerror(code), str(source), None, str(destination))


def publish_no_clobber(candidate: Path, final: Path) -> bool:
    """Make ``candidate`` visible as ``final`` only if ``final`` is absent; never check-then-replace.

    Returns True when the hard-link route was used: ``final`` is visible and the caller must
    still unlink the private ``candidate`` name (and treat ``final`` as published if that fails).
    ``FileExistsError`` means the name was taken and nothing changed.
    """
    try:
        exclusive_rename(candidate, final)
        return False
    except ExclusiveRenameUnsupported:
        pass
    os.link(candidate, final, follow_symlinks=False)  # FileExistsError when taken
    return True


# --- Candidate registry (§ 5.1, § 5.2 cleanup, § 7) ----------------------------------------

TempKind = Literal["candidate", "preimage", "receipt", "run_output"]


@dataclass(frozen=True)
class TempRecord:
    path: Path
    identity: Identity
    input_index: int | None
    member_id: str | None
    purpose: str
    kind: TempKind
    worker: str | None = None
    final_path: Path | None = None
    """Allocated bundle destination, binding a candidate to its member, not just its inode."""


FaultHook = Callable[[str, str | None], None]
"""``hook(phase, member_id)``: raises to inject a fault at a controlled point (TESTING § 5.4)."""


def _no_fault(phase: str, member_id: str | None) -> None:
    return None


class RunRegistry:
    """Everything private this run created, by pathname and identity, plus the commit lock.

    Thread-safe. ``commit_lock`` is the run's commit/cancellation lock (§ 5.2, § 6): a first
    SIGINT sets cancellation while holding it, so no commit starts afterwards.

    ``umask`` (default: the process umask, read once here, before any worker starts) fixes
    ``output_mode``, the mode every published output takes. ``free_space`` (default
    ``free_space_bytes``) reports a volume's available bytes for the § 5.1 space check.
    """

    def __init__(self, run_id: str, *, fault_hook: FaultHook | None = None, umask: int | None = None,
                 free_space: Callable[[Path], int] | None = None) -> None:
        if not run_id or "/" in run_id or any(ord(c) < 33 for c in run_id):
            raise ValueError("run_id must be a non-empty token without separators or spaces")
        self.run_id = run_id
        self.commit_lock = threading.RLock()
        self.fault_hook: FaultHook = fault_hook or _no_fault
        self.output_mode = output_mode_for(process_umask() if umask is None else umask)
        self._free_space = free_space or free_space_bytes
        self._space_lock = threading.Lock()
        self._space: dict[int, list[SpaceReservation]] = {}
        self._lock = threading.RLock()
        self._temps: dict[Path, TempRecord] = {}
        self._created_dirs: list[Path] = []
        self._folders: dict[Path, Identity] = {}

    # -- inspection
    def records(self, input_index: int | None = None) -> tuple[TempRecord, ...]:
        with self._lock:
            return tuple(r for r in self._temps.values() if input_index is None or r.input_index == input_index)

    @property
    def created_directories(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(self._created_dirs)

    def _register(self, record: TempRecord) -> None:
        with self._lock:
            self._temps[record.path] = record

    def forget(self, path: Path) -> None:
        with self._lock:
            self._temps.pop(path, None)

    # -- folders
    def ensure_folder(self, folder: Path, pin: ParentPin | None = None) -> Identity:
        """Create the missing folders of ``folder`` after the barrier, recording each one created."""
        with self._lock:
            try:
                current = recheck_pin(pin) if pin is not None else pin_parent(folder / "_")
            except UnsafeAncestry as exc:
                raise _failure("unsafe_output", ALLOCATE_STAGE, str(exc)) from None
            if current.parent != folder:
                raise _failure("unsafe_output", ALLOCATE_STAGE, f"{folder} is not the pinned folder")
            missing = [p for p in [folder, *folder.parents] if p.is_relative_to(current.anchor)
                       and p != current.anchor]
            for path in reversed(missing):
                try:
                    os.mkdir(path)
                    self._created_dirs.append(path)
                except FileExistsError:
                    info = _lstat(path)
                    if info is None or not stat.S_ISDIR(info.st_mode):
                        raise _failure("unsafe_output", ALLOCATE_STAGE,
                                       f"{path} appeared as a {_kind(info)} while folders were created") from None
                except OSError as exc:
                    raise _failure(_io_reason(exc), ALLOCATE_STAGE, f"cannot create {path}: {exc}") from None
            if os.path.realpath(folder) != str(folder):
                raise _failure("unsafe_output", ALLOCATE_STAGE, f"{folder} is now reached through a link")
            info = os.lstat(folder)
            identity = _identity(info)
            known = self._folders.setdefault(folder, identity)
            if known != identity:
                raise _failure("unsafe_output", ALLOCATE_STAGE, f"output folder {folder} was replaced")
            return identity

    def check_folder(self, folder: Path, stage: str = COMMIT_STAGE) -> None:
        """The folder still is the one this run allocated in, by canonical path and identity."""
        with self._lock:
            known = self._folders.get(folder)
        info = _lstat(folder)
        if (known is None or info is None or not stat.S_ISDIR(info.st_mode) or _identity(info) != known
                or os.path.realpath(folder) != str(folder)):
            raise _failure("unsafe_output", stage, f"output folder {folder} changed since allocation")

    # -- allocation
    def _private(self, folder: Path, input_index: int | None, tag: str, extension: str) -> tuple[int, Path]:
        prefix = f"{PRIVATE_PREFIX}{self.run_id}-{'run' if input_index is None else input_index}-{tag}-"
        fd, name = tempfile.mkstemp(prefix=prefix, suffix=extension, dir=folder)
        return fd, Path(name)

    def allocate(self, input_index: int, member: BundleMember, *, pin: ParentPin | None = None,
                 worker: str | None = None) -> BundleMember:
        """An exclusive 0600 candidate beside ``member``'s final name, registered at once (§ 5.1)."""
        folder = member.final_path.parent
        self.ensure_folder(folder, pin)
        try:
            fd, path = self._private(folder, input_index, "cand", member.final_path.suffix)
        except OSError as exc:
            raise _failure(_io_reason(exc), ALLOCATE_STAGE, f"cannot create a candidate in {folder}: {exc}") from None
        try:
            info = os.fstat(fd)
        finally:
            os.close(fd)
        self._register(TempRecord(path, _identity(info), input_index, member.member_id, member.purpose,
                                  "candidate", worker, member.final_path))
        return replace(member, temporary_path=path)

    def stage_member(self, input_index: int, member: BundleMember, data: bytes) -> BundleMember:
        """Fill one non-audio member allocated by ``allocate_plan``; never open a final name.

        Allocation exclusively creates the 0600 private file in its pinned parent. This step
        opens that registered inode with O_NOFOLLOW, verifies the descriptor before truncation,
        writes every byte and fsyncs it. The returned member is unchanged: pass its member_id
        to commit_bundle only after staging succeeds. Publication supplies the public mode;
        sidecars/exports keep their own modification times (§ 5.1).

        On a write, flush or interrupt failure, remove this member's private file by identity.
        Other members stay registered for the caller's settlement/rollback. Prepared JSON and
        picture lengths must be summed into estimate_space(sidecar_bytes=...) before allocation.
        """
        if member.purpose == "audio":
            raise ValueError("stage_member takes only non-audio bundle members")
        if not isinstance(data, bytes):
            raise TypeError("staged member data must be bytes")
        path = member.temporary_path
        with self._lock:
            record = self._temps.get(path) if path is not None else None
            if (record is None or record.kind != "candidate" or record.input_index != input_index
                    or record.member_id != member.member_id or record.purpose != member.purpose
                    or record.final_path != member.final_path):
                raise _failure("unsafe_output", "tags_written", "member is not this input's allocated candidate")
            try:
                self.check_folder(member.final_path.parent, "tags_written")
                self.check_candidate(member, "tags_written")
                assert path is not None
                # No O_TRUNC: a racer substituting another regular inode must not lose bytes.
                fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
                try:
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or _identity(info) != record.identity
                            or info.st_nlink != 1):
                        raise _failure("unsafe_output", "tags_written", f"candidate {path} was substituted or linked")
                    self.check_folder(member.final_path.parent, "tags_written")
                    self.check_candidate(member, "tags_written")
                    os.ftruncate(fd, 0)
                    _write_all(fd, data)
                    os.fsync(fd)
                    self.check_folder(member.final_path.parent, "tags_written")
                    self.check_candidate(member, "tags_written")
                finally:
                    os.close(fd)
            except BaseException as exc:
                self.cleanup_paths((path,) if path is not None else ())
                if isinstance(exc, OSError):
                    reason = "unsafe_output" if exc.errno == errno.ELOOP else _io_reason(exc)
                    raise _failure(reason, "tags_written", f"cannot stage {member.member_id}: {exc}") from None
                raise
        return member

    def allocate_plan(self, input_index: int, plan: OutputPlan, *,
                      pins: Mapping[str, ParentPin] | None = None, worker: str | None = None,
                      space: SpaceEstimate | None = None) -> OutputPlan:
        """Allocate every member's candidate; with ``space``, reserve it first (§ 5.1 Space).

        A shortfall raises ``disk_full`` before any folder or candidate is created.
        """
        if space is not None:
            self.reserve_space(input_index, plan.audio_member.final_path.parent, space)
        members = tuple(self.allocate(input_index, member,
                                      pin=(pins or {}).get(destination_id(input_index, member.member_id)),
                                      worker=worker)
                        for member in plan.members)
        return replace(plan, members=members)

    # -- space (§ 5.1 Space)
    def reserve_space(self, input_index: int, folder: Path, estimate: SpaceEstimate | int, *,
                      stage: str = ALLOCATE_STAGE) -> SpaceReservation:
        """Hold ``estimate`` on ``folder``'s volume for this input, or raise ``disk_full``.

        Under the space lock: available bytes minus every other input's reservation on the same
        volume must cover ``required_bytes``. This input's earlier reservations (a previous
        attempt) are replaced. The check is advisory, not an allocation guarantee: ENOSPC later
        is still ``disk_full``. Held until ``cleanup`` of this input (deletion, publication or
        retention) or ``close``.
        """
        required = estimate.required_bytes if isinstance(estimate, SpaceEstimate) else int(estimate)
        return self._reserve(input_index, folder, required, stage=stage, replace_own=True)

    def _reserve(self, input_index: int, folder: Path, required: int, *, stage: str,
                 replace_own: bool) -> SpaceReservation:
        with self._space_lock:
            try:
                anchor, info = _existing_ancestor(folder)
                available = self._free_space(anchor)
            except OSError as exc:
                raise _failure(_io_reason(exc), stage, f"cannot read the free space for {folder}: {exc}") from None
            held = sum(r.required_bytes for index, rows in self._space.items() if index != input_index
                       for r in rows if r.device == info.st_dev)
            if available - held < required:
                raise _failure("disk_full", stage,
                               f"{folder} needs {required:,} bytes free for this file; the volume has "
                               f"{available:,} available and this run holds {held:,} for other files")
            reservation = SpaceReservation(input_index, info.st_dev, folder, required)
            if replace_own:
                self._space[input_index] = [reservation]
            else:
                self._space.setdefault(input_index, []).append(reservation)
            return reservation

    def release_space(self, input_index: int | None) -> None:
        """Drop this input's (or, with ``None``, every) space reservation."""
        with self._space_lock:
            if input_index is None:
                self._space.clear()
            else:
                self._space.pop(input_index, None)

    def reserved_bytes(self, input_index: int | None = None) -> int:
        """Bytes currently held, for one input or the whole run."""
        with self._space_lock:
            return sum(r.required_bytes for index, rows in self._space.items()
                       if input_index is None or index == input_index for r in rows)

    def check_candidate(self, member: BundleMember, stage: str = COMMIT_STAGE) -> os.stat_result:
        """The registered candidate is still our private regular inode with one link."""
        path = member.temporary_path
        if path is None:
            raise _failure("commit_error", stage, f"{member.member_id} has no allocated candidate")
        with self._lock:
            record = self._temps.get(path)
        if record is None or record.kind != "candidate":
            raise _failure("unsafe_output", stage, f"{path} is not a candidate this run allocated")
        if (record.member_id != member.member_id or record.purpose != member.purpose
                or record.final_path != member.final_path):
            raise _failure("unsafe_output", stage, f"{path} belongs to a different bundle member")
        info = _lstat(path)
        if info is None or not stat.S_ISREG(info.st_mode) or _identity(info) != record.identity:
            raise _failure("unsafe_output", stage, f"candidate {path} was removed or substituted")
        if info.st_nlink != 1:
            raise _failure("unsafe_output", stage, f"candidate {path} gained another link")
        return info

    def check_structure(self, input_index: int, member: BundleMember) -> str | None:
        """Structural check of a closed candidate before publication (§ 5.2 gate, § 8).

        The audio member must be a non-empty file whose leading signature matches its extension
        and, for WAV/AIFF, whose chunk walk proves no shortfall. Returns the signature kind.
        A structurally refused candidate is a ``render_error``: the source passed its own check.
        """
        info = self.check_candidate(member, STRUCTURE_STAGE)
        if member.purpose != "audio":
            return None
        if info.st_size == 0:
            raise _failure("render_error", STRUCTURE_STAGE, f"candidate {member.temporary_path} is empty")
        assert member.temporary_path is not None
        probe_item = InputItem(member.temporary_path, member.temporary_path.parent,
                               PurePosixPath(member.temporary_path.name), _identity(info), input_index, True)
        try:
            result = check_signature(probe_item)
        except FileOutcomeError as exc:
            raise _failure("render_error", STRUCTURE_STAGE,
                           f"candidate {member.temporary_path} failed its structural check: {exc.reason}"
                           + (f" ({exc.detail})" if exc.detail else "")) from None
        expected = EXTENSION_KINDS.get(member.final_path.suffix.casefold())
        if expected is not None and result.kind != expected:
            raise _failure("render_error", STRUCTURE_STAGE,
                           f"candidate {member.temporary_path} is {result.kind}, not the {expected} "
                           f"its name {member.final_path.name} promises")
        return result.kind

    # -- final mode and timestamps (§ 5.1 Metadata/timestamps, CR-046)
    def finalise_candidate(self, member: BundleMember, *, mtime_ns: int | None = None,
                           stage: str = COMMIT_STAGE) -> None:
        """Give a closed, verified candidate its public mode and, for audio, the source's mtime.

        Called just before publication, while the candidate still has its private name. Only
        the candidate is changed, by pathname without following links and between identity
        rechecks; the source is never touched. Each value is read back: a volume that cannot
        store the mode, or the mtime even at a coarser granularity, fails the file with a
        specific I/O reason (``permission_denied``, ``disk_full``, else ``commit_error``).
        """
        info = self.check_candidate(member, stage)
        path = member.temporary_path
        assert path is not None
        try:
            os.chmod(path, self.output_mode, follow_symlinks=False)
            if mtime_ns is not None:
                os.utime(path, ns=(info.st_atime_ns, mtime_ns), follow_symlinks=False)
        except OSError as exc:
            raise _failure(_io_reason(exc), stage,
                           f"cannot set the final mode or modification time of {path}: {exc}") from None
        after = self.check_candidate(member, stage)
        if stat.S_IMODE(after.st_mode) != self.output_mode:
            raise _failure("commit_error", stage,
                           f"{path} kept mode {stat.S_IMODE(after.st_mode):o}, not {self.output_mode:o}")
        if mtime_ns is not None and not _times_match(after.st_mtime_ns, mtime_ns):
            raise _failure("commit_error", stage,
                           f"{path} stored modification time {after.st_mtime_ns} ns, not the source's {mtime_ns} ns")

    # -- removal
    def remove(self, path: Path) -> None:
        """Unlink one registered private entry, only while it is still the inode we created."""
        with self._lock:
            record = self._temps.get(path)
        if record is None:
            return
        info = _lstat(path)
        if info is not None:
            if _identity(info) != record.identity or not stat.S_ISREG(info.st_mode):
                self.forget(path)
                raise _failure("cleanup_error", CLEANUP_STAGE, f"{path} is no longer this run's file; left alone")
            os.unlink(path)
        self.forget(path)

    def cleanup(self, input_index: int | None, *, keep: Iterable[Path] = ()) -> tuple[str, ...]:
        """Remove this input's (or, with ``None``, every) registered private entry; report failures.

        The input's space reservation is released afterwards: its bytes are now deleted,
        published or deliberately retained (§ 5.1 Space).
        """
        kept = set(keep)
        problems: list[str] = []
        for record in self.records(input_index):
            if record.path in kept:
                self.forget(record.path)
                continue
            try:
                self.remove(record.path)
            except FileFailure as exc:
                problems.append(exc.detail or str(exc))
            except OSError as exc:
                problems.append(f"cannot remove {record.path}: {exc.strerror or exc}")
        self.release_space(input_index)
        return tuple(problems)

    def cleanup_paths(self, paths: Iterable[Path]) -> tuple[str, ...]:
        """Remove the given registered private entries; report failures."""
        problems: list[str] = []
        for path in paths:
            try:
                self.remove(path)
            except (FileFailure, OSError) as exc:
                problems.append(str(exc))
        return tuple(problems)

    def remove_created_folders(self) -> tuple[str, ...]:
        """Remove run-created folders that are empty, deepest first; never anything else (§ 5.2).

        A folder this run allocated in is removed only while it still has the identity the run
        recorded; a replacement is left alone and stays recorded, so a later allocation there
        is still refused. A folder the run did remove is forgotten with its identity (CR-051):
        its name is free again, and a later legitimate allocation in the same run (the automatic
        report after every file failed, cli § 7.3) re-creates it as a new run-created folder.
        """
        problems: list[str] = []
        with self._lock:
            created = list(self._created_dirs)
        for path in sorted(created, key=lambda p: len(p.parts), reverse=True):
            with self._lock:
                known = self._folders.get(path)
            if known is not None:
                info = _lstat(path)
                if info is not None and (not stat.S_ISDIR(info.st_mode) or _identity(info) != known):
                    continue  # no longer the folder this run created: never removed, never forgotten
            try:
                os.rmdir(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    problems.append(f"cannot remove folder {path}: {exc.strerror or exc}")
                continue
            else:
                with self._lock:
                    for folder in [f for f in self._folders if f == path or f.is_relative_to(path)]:
                        del self._folders[folder]
            with self._lock:
                self._created_dirs.remove(path)
        return tuple(problems)

    def close(self) -> tuple[str, ...]:
        """End-of-run cleanup: every registered private entry, then empty run-created folders."""
        return self.cleanup(None) + self.remove_created_folders()

    # -- receipts and preimages
    def write_receipt(self, input_index: int, folder: Path, content: Mapping[str, object]) -> Path:
        """Create, register, then fill a private receipt; on an I/O failure it is removed (§ 5.2).

        The receipt is registered by identity straight after its exclusive creation, before any
        fallible write or ``fsync`` (IC-2 recheck R-1), so item rollback and end-of-run cleanup
        always know it. A failure while filling it also removes it here, by identity, and then
        re-raises the original error; should that removal fail, it stays registered for cleanup.
        """
        fd, path = self._private(folder, input_index, "receipt", ".json")
        try:
            info = os.fstat(fd)
        except OSError:
            os.close(fd)
            raise
        self._register(TempRecord(path, _identity(info), input_index, None, "receipt", "receipt"))
        try:
            try:
                _write_all(fd, json.dumps(content, sort_keys=True, default=str).encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            try:
                self.remove(path)
            except (FileFailure, OSError):
                pass  # still registered: rollback or close() retries and reports it
            raise
        return path

    def update_receipt(self, path: Path, content: Mapping[str, object]) -> None:
        with self._lock:
            record = self._temps[path]
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if _identity(os.fstat(fd)) != record.identity:
                raise OSError(errno.ESTALE, "receipt was substituted", str(path))
            _write_all(fd, json.dumps(content, sort_keys=True, default=str).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def make_preimage(self, input_index: int, member_id: str, final: Path, old: Identity) -> tuple[Path, str]:
        """Retain the old output without moving it: a private hard link, else a checked copy (§ 5.2)."""
        folder = final.parent
        last: OSError | None = None
        for _attempt in range(_PREIMAGE_ATTEMPTS):
            name = folder / f"{PRIVATE_PREFIX}{self.run_id}-{input_index}-pre-{secrets.token_hex(6)}{final.suffix}"
            try:
                os.link(final, name, follow_symlinks=False)
            except FileExistsError:
                continue
            except OSError as exc:
                if exc.errno in _LINK_REFUSED:
                    last = exc
                    break
                raise
            info = os.lstat(name)
            self._register(TempRecord(name, _identity(info), input_index, member_id, "preimage", "preimage"))
            if _identity(info) != old:
                raise _failure("backup_error", COMMIT_STAGE, f"{final} changed while its preimage was taken")
            return name, _sha256_path(name, old)
        else:
            raise _failure("backup_error", COMMIT_STAGE, f"no free private name for a preimage of {final}")
        return self._copy_preimage(input_index, member_id, final, old, last)

    def _copy_preimage(self, input_index: int, member_id: str, final: Path, old: Identity,
                       cause: OSError | None) -> tuple[Path, str]:
        source_fd = os.open(final, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            old_info = os.fstat(source_fd)
            if _identity(old_info) != old:
                raise _failure("backup_error", COMMIT_STAGE, f"{final} changed while its preimage was copied")
            # The copy fallback is the only preimage that consumes space (§ 5.1 Space).
            self._reserve(input_index, final.parent, SpaceEstimate(0, preimage_bytes=old_info.st_size).required_bytes,
                          stage=COMMIT_STAGE, replace_own=False)
            fd, path = self._private(final.parent, input_index, "pre", final.suffix)
            info = os.fstat(fd)
            self._register(TempRecord(path, _identity(info), input_index, member_id, "preimage", "preimage"))
            try:
                digest = hashlib.sha256()
                offset = 0
                while True:
                    chunk = os.pread(source_fd, _HASH_CHUNK, offset)
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = os.write(fd, view)
                        view = view[written:]
                    digest.update(chunk)
                    offset += len(chunk)
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            os.close(source_fd)
        copied = _sha256_path(path)
        if copied != digest.hexdigest() or _sha256_path(final, old) != copied:
            raise _failure("backup_error", COMMIT_STAGE, f"the preimage copy of {final} does not match it")
        try:  # a restore must give back the old output as it was, not a private 0600 copy
            os.chmod(path, stat.S_IMODE(old_info.st_mode), follow_symlinks=False)
            os.utime(path, ns=(old_info.st_atime_ns, old_info.st_mtime_ns), follow_symlinks=False)
        except OSError as exc:
            raise _failure("backup_error", COMMIT_STAGE,
                           f"cannot carry the mode and times of {final} to its preimage copy: {exc}") from None
        return path, copied


# --- Commit and rollback (§ 5.2) ------------------------------------------------------------


@dataclass(frozen=True)
class CommitResult:
    """What publication, rollback and cleanup actually did; C-09 fields for WS-04 to carry."""

    status: Status
    reason: ReasonCode
    execution: Execution
    published_paths: tuple[Path, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    retained_paths: tuple[Path, ...] = ()
    """Private evidence kept after a failed rollback or cleanup (preimages, receipt, candidates)."""
    retained_failed_path: Path | None = None
    notes: tuple[str, ...] = ()


@dataclass
class _Step:
    member: BundleMember
    action: Literal["create", "replace"]
    old: Identity | None = None
    preimage: Path | None = None
    preimage_sha256: str | None = None
    published: Identity | None = None
    done: bool = False


def _classify_occupant(final: Path, guard: _Protected, *, overwrite: bool) -> tuple[str, Identity | None]:
    """Commit-time occupant: ``create``, ``replace`` or a raised refusal (§ 3 table)."""
    info = _lstat(final)
    kind = _kind(info)
    if kind == "absent":
        return "create", None
    assert info is not None
    if kind != "regular":
        raise _failure("unsafe_output", COMMIT_STAGE, f"{final} is now a {kind}; it is never replaced")
    if _identity(info) in guard.identities:
        raise _failure("unsafe_output", COMMIT_STAGE,
                       f"{final} is now another name for the {guard.identities[_identity(info)]} this run reads")
    if not overwrite:
        raise SkipFile("output_exists", COMMIT_STAGE, f"{final} appeared after planning; it was left untouched")
    return "replace", _identity(info)


def _receipt(registry: RunRegistry, item: InputItem, plan: OutputPlan, steps: Sequence[_Step],
             phase: str) -> dict[str, object]:
    identities: dict[Path | None, Identity] = {
        r.path: r.identity for r in registry.records(item.input_index) if r.kind == "candidate"
    }
    return {
        "run_id": registry.run_id, "input_index": item.input_index, "source": str(item.source_path),
        "source_identity": list(item.identity), "plan_revision": plan.plan_revision, "phase": phase,
        "members": [{
            "member_id": s.member.member_id, "final": str(s.member.final_path),
            "candidate": str(s.member.temporary_path), "action": s.action,
            "candidate_identity": list(s.published or identities[s.member.temporary_path]),
            "preimage": str(s.preimage) if s.preimage else None, "preimage_sha256": s.preimage_sha256,
            "published": s.done,
        } for s in steps],
    }


def _diag(code: ReasonCode, message: str, detail: str | None = None) -> Diagnostic:
    return Diagnostic(code, message, COMMIT_STAGE, detail)


def commit_bundle(registry: RunRegistry, item: InputItem, plan: OutputPlan, *,
                  snapshot: SourceSnapshot, protected: Sequence[ReadDependency],
                  metadata_revision: int, staged_member_ids: Sequence[str],
                  cancel_event: threading.Event | None = None,
                  interruption_requested: Callable[[], bool] | None = None) -> CommitResult:
    """Publish one verified bundle: sidecars in pathname order, audio last; roll back on failure.

    Preconditions: every member was allocated by ``registry`` and is closed and verified
    (audio by WS-01, tags by WS-06). Returns the truthful C-09 execution state; never raises
    a ``FileOutcomeError``.

    ``interruption_requested`` observes a pending interrupt only for failed settlement
    diagnostics. It never gates publication or changes a committed outcome; the
    caller supplies a quick, non-raising predicate (for example ``lambda: control.interrupted``).
    """
    guard = _Protected.of(protected)
    index = item.input_index
    with registry.commit_lock:
        if cancel_event is not None and cancel_event.is_set():
            return settle(registry, index, SkipFile("cancelled", COMMIT_STAGE, "interrupted before publication"))
        steps: list[_Step] = []
        receipt: Path | None = None
        try:
            _gate(registry, item, plan, snapshot, metadata_revision, staged_member_ids)
            ordered = sorted((m for m in plan.members if m.purpose != "audio"), key=lambda m: str(m.final_path))
            ordered.append(plan.audio_member)
            for member in ordered:
                action, old = _classify_occupant(member.final_path, guard, overwrite=plan.overwrite_authorised)
                steps.append(_Step(member, action, old))  # type: ignore[arg-type]
            registry.fault_hook("receipt", None)
            try:
                receipt = registry.write_receipt(index, plan.audio_member.final_path.parent,
                                                 _receipt(registry, item, plan, steps, "prepared"))
            except OSError as exc:
                raise _failure(_io_reason(exc), COMMIT_STAGE, f"cannot write the recovery receipt: {exc}") from None
            for step in steps:
                if step.action == "replace":
                    registry.fault_hook("preimage", step.member.member_id)
                    assert step.old is not None
                    try:
                        step.preimage, step.preimage_sha256 = registry.make_preimage(
                            index, step.member.member_id, step.member.final_path, step.old)
                    except OSError as exc:
                        raise _failure("backup_error", COMMIT_STAGE,
                                       f"cannot retain the existing {step.member.final_path}: {exc}") from None
            # § 5.1: public mode for every member, the source's mtime for the audio, just before
            # publication; nothing is visible yet, so a failure here publishes nothing.
            for step in steps:
                registry.fault_hook("finalise", step.member.member_id)
                registry.finalise_candidate(
                    step.member, mtime_ns=snapshot.mtime_ns if step.member.purpose == "audio" else None)
            registry.update_receipt(receipt, _receipt(registry, item, plan, steps, "publishing"))
            for step in steps:
                if cancel_event is not None and cancel_event.is_set():
                    raise SkipFile("cancelled", COMMIT_STAGE, "interrupted during publication")
                _publish_step(registry, step, guard)
                registry.update_receipt(receipt, _receipt(registry, item, plan, steps, "publishing"))
            if cancel_event is not None and cancel_event.is_set():
                raise SkipFile("cancelled", COMMIT_STAGE, "interrupted during publication")
        except KeyboardInterrupt:
            result = _rollback(registry, index, steps,
                             SkipFile("cancelled", COMMIT_STAGE, "interrupted during publication"))
        except FileOutcomeError as exc:
            result = _rollback(registry, index, steps, exc)
        except OSError as exc:
            result = _rollback(registry, index, steps,
                             _failure(_io_reason(exc), COMMIT_STAGE, f"publication failed: {exc}"))
        except Exception as exc:  # noqa: BLE001 - unexpected: roll back truthfully, report internal
            result = _rollback(registry, index, steps,
                             _failure("internal", COMMIT_STAGE, f"unexpected {type(exc).__name__}: {exc}"))
        else:
            result = _finish(registry, index, steps, receipt)
        # Cancellation can arrive after the triggering error, during rollback or
        # required cleanup. Keep that specific failure primary, but do not lose
        # the interruption diagnostic (§ 6).
        if (result.status == "failed"
                and not any(d.code == "cancelled" for d in result.diagnostics)
                and ((cancel_event is not None and cancel_event.is_set())
                     or (interruption_requested is not None and interruption_requested()))):
            result = replace(result, diagnostics=(*result.diagnostics,
                             _diag("cancelled", "interrupted during publication or settlement")))
        return result


def _gate(registry: RunRegistry, item: InputItem, plan: OutputPlan, snapshot: SourceSnapshot,
          metadata_revision: int, staged_member_ids: Sequence[str]) -> None:
    if plan.source_identity != item.identity:
        raise _failure("commit_error", COMMIT_STAGE, "the output plan belongs to a different source")
    if plan.plan_revision != metadata_revision:
        raise _failure("commit_error", COMMIT_STAGE,
                       f"output plan revision {plan.plan_revision} is not the final metadata revision "
                       f"{metadata_revision}")
    wanted = sorted(m.member_id for m in plan.members if m.member_id != AUDIO_MEMBER_ID)
    if sorted(staged_member_ids) != wanted:
        raise _failure("commit_error", COMMIT_STAGE,
                       f"staged members {sorted(staged_member_ids)} do not match the requested {wanted}")
    registry.fault_hook("gate", None)
    recheck_source(item, snapshot, rehash=True, stage=COMMIT_STAGE)
    for member in plan.members:
        registry.check_candidate(member)
        registry.check_folder(member.final_path.parent)
        assert member.temporary_path is not None
        with registry._lock:
            record = registry._temps[member.temporary_path]
        if record.input_index != item.input_index:
            raise _failure("unsafe_output", COMMIT_STAGE, "candidate belongs to a different input")


def _publish_step(registry: RunRegistry, step: _Step, guard: _Protected) -> None:
    member = step.member
    candidate = member.temporary_path
    assert candidate is not None
    registry.check_folder(member.final_path.parent)
    info = registry.check_candidate(member)
    registry.fault_hook("publish", member.member_id)
    linked = False
    if step.action == "create":
        try:
            linked = publish_no_clobber(candidate, member.final_path)
        except FileExistsError:
            _classify_occupant(member.final_path, guard, overwrite=False)  # raises the right refusal
            raise _failure("commit_error", COMMIT_STAGE, f"{member.final_path} was taken during publication") \
                from None
    else:
        current = _lstat(member.final_path)
        if current is None or not stat.S_ISREG(current.st_mode) or _identity(current) != step.old:
            raise _failure("unsafe_output", COMMIT_STAGE,
                           f"{member.final_path} changed after its preimage was taken; left untouched")
        os.replace(candidate, member.final_path)
    step.published = _identity(info)
    step.done = True
    if linked:
        registry.remove(candidate)  # a failure here leaves the final published and rolls it back
    else:
        registry.forget(candidate)
    registry.fault_hook("published", member.member_id)
    after = _lstat(member.final_path)
    if after is None or _identity(after) != step.published:
        raise _failure("commit_error", COMMIT_STAGE, f"{member.final_path} does not hold the published file")


def _finish(registry: RunRegistry, index: int, steps: Sequence[_Step], receipt: Path | None) -> CommitResult:
    published = tuple(s.member.final_path for s in steps if s.done)
    notes = tuple(note for note in {_fsync_directory(p.parent) for p in published} if note)
    problems: list[str] = []
    try:
        registry.fault_hook("cleanup", None)
    except OSError as exc:
        problems.append(f"cleanup failed: {exc}")
    if not problems:
        problems.extend(registry.cleanup(index))
    if problems:
        retained = tuple(r.path for r in registry.records(index))
        return CommitResult("failed", "cleanup_error", "publication_failed", published,
                            (_diag("cleanup_error", "; ".join(problems)),), retained, None, notes)
    return CommitResult("ok", "ok", "committed", published, notes=notes)


def _rollback(registry: RunRegistry, index: int, steps: Sequence[_Step],
              primary: FileOutcomeError) -> CommitResult:
    """Undo this run's publications, newest first, only where our identity still holds the name."""
    survivors: list[Path] = []
    diagnostics: list[Diagnostic] = [primary.diagnostic()]
    keep: list[Path] = []
    for step in reversed([s for s in steps if s.done]):
        final = step.member.final_path
        current = _lstat(final)
        ours = current is not None and _identity(current) == step.published
        try:
            registry.fault_hook("rollback", step.member.member_id)
            if not ours:
                continue  # a third party holds the name now; never remove or overwrite it
            if step.action == "create":
                os.unlink(final)
            else:
                assert step.preimage is not None
                if _sha256_path(step.preimage) != step.preimage_sha256:
                    raise OSError(errno.EIO, "preimage content changed", str(step.preimage))
                os.replace(step.preimage, final)
                registry.forget(step.preimage)
        except (OSError, FileOutcomeError) as exc:
            survivors.append(final)
            if step.action == "replace":
                diagnostics.append(_diag("backup_error", f"could not restore {final}", str(exc)))
                if step.preimage is not None:
                    keep.append(step.preimage)
            else:
                diagnostics.append(_diag("commit_error", f"could not remove {final}", str(exc)))
    if survivors:
        receipts = [r.path for r in registry.records(index) if r.kind == "receipt"]
        keep.extend(receipts)
        problems = registry.cleanup(index, keep=keep)
        if problems:
            diagnostics.append(_diag("cleanup_error", "; ".join(problems)))
        reason = primary.reason if primary.status == "failed" else "commit_error"
        # cleanup forgets deliberate keeps, but failed removals remain registered.
        # Both kinds are recovery evidence, including unpublished candidates.
        retained = tuple(dict.fromkeys([*keep, *(r.path for r in registry.records(index))]))
        return CommitResult("failed", reason, "publication_failed", tuple(reversed(survivors)),
                            tuple(diagnostics), retained)
    return settle(registry, index, primary)


def settle(registry: RunRegistry, input_index: int, primary: FileOutcomeError, *,
           keep_failed: Path | None = None) -> CommitResult:
    """Discard an unpublished file: remove its private entries and return the not-written result.

    ``keep_failed`` names a verification-failed candidate to retain (``--keep-failed``); it is
    ignored for cancellation (§ 5.2). A cleanup failure keeps the earlier failure primary with
    ``cleanup_error`` secondary, or becomes the primary failure for a skip.
    """
    retain = None if primary.reason == "cancelled" else keep_failed
    problems = registry.cleanup(input_index, keep=(retain,) if retain is not None else ())
    diagnostics = [primary.diagnostic()]
    if problems:
        diagnostics.append(Diagnostic("cleanup_error", "; ".join(problems), CLEANUP_STAGE))
        retained = tuple(r.path for r in registry.records(input_index))
        if primary.status != "failed":
            return CommitResult("failed", "cleanup_error", "not_written", (), tuple(diagnostics), retained, retain)
        return CommitResult("failed", primary.reason, "not_written", (), tuple(diagnostics), retained, retain)
    return CommitResult(primary.status, primary.reason, "not_written", (), tuple(diagnostics), (), retain)


# --- Run outputs: report, template and log (§ 6) ------------------------------------------------

_MAX_ORDINAL = 999
"""The last ordinal the automatic report tries before it gives up with a run-output error."""


def publish_run_output(registry: RunRegistry, destination: WriteDestination, data: bytes, *,
                       overwrite: bool, protected: Sequence[ReadDependency],
                       reserved_keys: Iterable[str] = (), ordinal_retry: bool = False,
                       create_parent: bool = False, pin: ParentPin | None = None) -> Path:
    """Stage ``data`` privately beside ``destination`` and publish it without clobbering (§ 6).

    ``ordinal_retry`` (the automatic report only) never refuses and never replaces (CR-049,
    D-069): when the name is taken at publication it appends ``-2``, ``-3`` … to the stem
    (cli § 7.3), rechecking every name, the base one included, against ``reserved_keys`` and
    the protected reads, and publishing each exclusively. ``overwrite`` is ignored for it.
    Otherwise (an explicit report or template) a taken name is replaced only with
    ``overwrite`` and a safe regular occupant. Raises ``RunOutputError`` on any failure;
    committed audio is never touched.

    ``pin`` is the destination's preflight ``ParentPin`` (``Preflight.pins``). Without
    ``create_parent`` the parent must still exist and, when ``pin`` is given, still be the
    folder preflight checked; nothing is ever created for it (CR-048). With ``create_parent``
    (the automatic report) missing folders below the pinned ancestor are created.
    """
    if destination.open_mode != "publish" or destination.purpose not in ("report", "template_export"):
        raise ValueError("only report and template destinations are published this way")
    guard = _Protected.of(protected)
    final = destination.final_path
    folder = final.parent
    try:
        if create_parent:
            registry.ensure_folder(folder, pin)
        else:
            current = recheck_pin(pin) if pin is not None else pin_parent(final)
            if not current.exists:
                raise RunOutputError(f"cannot write {final}: its folder {folder} no longer exists; "
                                     "it is not re-created")
            registry.ensure_folder(folder, current)
    except (FileOutcomeError, UnsafeAncestry) as exc:
        raise RunOutputError(f"cannot write {final}: {exc}") from None
    if not os.path.isdir(folder):
        raise RunOutputError(f"cannot write {final}: its folder does not exist")
    try:
        fd, candidate = registry._private(folder, None, "out", final.suffix)
    except OSError as exc:
        raise RunOutputError(f"cannot stage {final}: {exc.strerror or exc}") from None
    try:
        info = os.fstat(fd)
        registry._register(TempRecord(candidate, _identity(info), None, None, destination.purpose, "run_output"))
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
        os.chmod(candidate, registry.output_mode, follow_symlinks=False)  # public mode before publication
        after = os.lstat(candidate)
        if _identity(after) != _identity(info) or stat.S_IMODE(after.st_mode) != registry.output_mode:
            raise OSError(errno.EPERM, "the staged copy was substituted or kept a private mode", str(candidate))
    except OSError as exc:
        os.close(fd)
        registry.cleanup_paths((candidate,))
        raise RunOutputError(f"cannot stage {final}: {exc.strerror or exc}") from None
    os.close(fd)
    reserved = set(reserved_keys) - {destination.comparison_key}
    name = final
    for ordinal in (0, *range(2, _MAX_ORDINAL + 1)) if ordinal_retry else (0,):
        if ordinal:
            name = folder / f"{final.stem}-{ordinal}{final.suffix}"
        if ordinal_retry and (comparison_key(name) in reserved or comparison_key(name) in guard.keys):
            continue  # another destination's name, or a file this run reads: never ours
        try:
            if publish_no_clobber(candidate, name):
                problems = registry.cleanup_paths((candidate,))
                if problems:
                    raise RunOutputError(f"{name} was published but its private copy remains: "
                                         + "; ".join(problems))
            registry.forget(candidate)
            return name
        except FileExistsError:
            if ordinal_retry:
                continue  # the automatic report never replaces: the next ordinal is tried (CR-049)
            occupant = _occupant(name)
            safe = occupant.kind == "regular" and occupant.identity not in guard.identities
            if overwrite and safe:
                try:
                    os.replace(candidate, name)
                except OSError as exc:
                    registry.cleanup_paths((candidate,))
                    raise RunOutputError(f"cannot replace {name}: {exc.strerror or exc}") from None
                registry.forget(candidate)
                return name
            registry.cleanup_paths((candidate,))
            raise RunOutputError(f"{name} appeared while the run was working; it was left untouched") \
                from None
        except OSError as exc:
            registry.cleanup_paths((candidate,))
            raise RunOutputError(f"cannot publish {name}: {exc.strerror or exc}") from None
    registry.cleanup_paths((candidate,))
    raise RunOutputError(f"no free report name beside {final}")


def _log_parent_changed(destination: WriteDestination, pin: ParentPin) -> bool:
    """The log's folder is no longer the existing, canonical folder preflight pinned (§ 2.6 (b))."""
    try:
        current = recheck_pin(pin)
    except (UnsafeAncestry, OSError):
        return True
    return not (pin.exists and current.exists and current.parent == destination.final_path.parent)


def open_log(destination: WriteDestination, occupant: Occupant, protected: Sequence[ReadDependency], *,
             pin: ParentPin | None = None) -> int:
    """Open the checked ``normalise`` log for appending; ``UsageError`` before the first byte (§ 6).

    ``occupant`` is what preflight saw. The descriptor must be a regular single-link file whose
    identity equals the checked one or, for a new log, is no protected identity.

    ``pin`` (default: the one preflight recorded on ``occupant``) is the log folder's preflight
    pin. It is rechecked before the open, so a folder replaced since the barrier is refused
    before anything is created or appended there, and again after it, with the log's name
    still naming the opened file (IC-2 recheck R-2). [judgement] No folder descriptor is used
    (see the module notes), so this narrows, but cannot close, a live race (§ 1). Without any
    pin only the checks above apply.
    """
    if destination.purpose != "log" or destination.open_mode != "append":
        raise ValueError("open_log takes the log destination")
    guard = _Protected.of(protected)
    pin = pin if pin is not None else occupant.parent
    final = destination.final_path
    if pin is not None and _log_parent_changed(destination, pin):
        raise UsageError(f"log {final}: its folder {final.parent} changed after it was checked; "
                         "nothing was written to it")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(final, flags, 0o600)
    except OSError as exc:
        raise UsageError(f"log {final}: cannot open it safely ({exc.strerror or exc})") from None
    try:
        info = os.fstat(fd)
        identity = _identity(info)
        expected = occupant.identity
        bad = (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or identity in guard.identities
               or (expected is not None and identity != expected))
        if not bad and pin is not None:
            named = _lstat(final)
            bad = named is None or _identity(named) != identity or _log_parent_changed(destination, pin)
    except OSError:
        bad = True
    if bad:
        os.close(fd)
        raise UsageError(f"log {final} changed after it was checked; nothing was written to it")
    return fd


__all__ = [
    "CommitResult", "ExclusiveRenameUnsupported", "Occupant", "ParentPin", "Preflight", "ReservationBook",
    "RunOutput", "RunOutputError", "RunRegistry", "SourceSnapshot", "SpaceEstimate", "SpaceReservation",
    "TempRecord", "UnsafeAncestry", "canonical_destination", "canonical_directory", "commit_bundle",
    "destination_id", "estimate_space", "exclusive_rename", "free_space_bytes", "open_log", "output_mode_for",
    "pin_parent", "preflight", "process_umask", "protect", "protect_inputs", "publish_no_clobber",
    "publish_run_output", "recheck_pin", "recheck_source", "refuse_in_place", "settle", "snapshot_source",
]
