"""Input discovery, ordering, deduplication and the signature/structure step.

Implements input discovery (with the structural checks on candidates),
producing C-01 ``InputItem`` records and, per item, the C-02 probe through
``norma.adapters.ffprobe``. Everything here is read-only (§ 2.7): directories
are listed, files are opened ``O_RDONLY | O_NOFOLLOW`` and nothing is written.

Two entry points:

* ``discover`` expands arguments into a ``DiscoveryResult`` (§§ 2.1-2.4). Bad
  explicit paths raise ``UsageError`` (exit 2) before any write.
* ``classify`` runs the signature step (§ 2.5) and then the adapter probe for
  one item, raising ``SkipFile``/``FileFailure`` with the C-13 reason.
"""

from __future__ import annotations

import errno
import os
import re
import stat
import struct
import threading
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from norma.adapters.ffprobe import PROBE_DEADLINE_S, Prober, probe
from norma.errors import FileFailure, SkipFile, UsageError
from norma.models import FrozenMap, Identity, InputItem, ProbeResult

# --- Folder-expansion rules (§ 2.1, § 2.2; formats F1) --------------------------------

RECOGNISED_EXTENSIONS: frozenset[str] = frozenset(
    {"wav", "aif", "aiff", "flac", "mp3", "m4a", "aac", "mp4", "caf"}
)
GENERATED_FOLDER_NAMES: frozenset[str] = frozenset({".norma-originals", "normalised"})

EXCLUSION_KINDS: tuple[str, ...] = ("hidden", "generated", "suffix", "extension", "special")
"""Keys of ``DiscoveryResult.excluded``; every key is always present."""


@dataclass(frozen=True)
class DiscoveryResult:
    """Deduplicated C-01 items in discovery order plus the counts § 2.2/§ 2.3 require."""

    items: tuple[InputItem, ...]
    excluded: FrozenMap  # kind (EXCLUSION_KINDS) -> entries skipped during folder expansion
    duplicates: int  # accepted files dropped because their (st_dev, st_ino) was already taken
    ignored_links: int  # unresolved symlinks found during expansion (§ 2.3)
    directory_links: int  # directory symlinks found inside a folder and not followed
    unreadable: tuple[Path, ...]  # discovered directories/entries that could not be read
    hints: tuple[str, ...]  # human-readable explanations of the counts above


# --- Ordering (§ 2.4) ----------------------------------------------------------------

_RUNS = re.compile(r"([0-9]+)")


def natural_key(name: str) -> tuple[tuple[tuple[int, int, str], ...], bytes]:
    """Sort key: NFC/case-folded digit and text runs, numeric runs numerically, then UTF-8 bytes.

    ``2`` precedes ``10``; ``02`` and ``2`` tie on the first part and fall to the byte tie-breaker.
    """
    folded = unicodedata.normalize("NFC", name).casefold()
    runs: list[tuple[int, int, str]] = []
    for part in _RUNS.split(folded):
        if not part:
            continue
        if part.isascii() and part.isdigit():
            runs.append((0, int(part), ""))
        else:
            runs.append((1, 0, part))
    return tuple(runs), os.fsencode(name)


# --- Expansion (§ 2.1-2.3) -----------------------------------------------------------


class _Collector:
    def __init__(self) -> None:
        self.items: list[InputItem] = []
        self.seen: set[Identity] = set()
        self.excluded = dict.fromkeys(EXCLUSION_KINDS, 0)
        self.duplicates = 0
        self.ignored_links = 0
        self.directory_links = 0
        self.unreadable: list[Path] = []

    def accept(self, source: Path, root: Path, relative: PurePosixPath, identity: Identity,
               explicit: bool) -> None:
        if identity in self.seen:
            self.duplicates += 1
            return
        self.seen.add(identity)
        self.items.append(InputItem(source, root, relative, identity, len(self.items), explicit))


def _identity(info: os.stat_result) -> Identity:
    return (info.st_dev, info.st_ino)


def _within(path: Path, ancestor: Path | None) -> bool:
    return ancestor is not None and (path == ancestor or ancestor in path.parents)


def _extension(name: str) -> str:
    suffix = Path(name).suffix
    return suffix[1:].casefold() if suffix else ""


def _stem_has_suffix(name: str, suffix: str) -> bool:
    if not suffix:
        return False
    stem = unicodedata.normalize("NFC", Path(name).stem).casefold()
    return stem.endswith(unicodedata.normalize("NFC", suffix).casefold())


def _file_target(real: Path, root: Path) -> tuple[Path, PurePosixPath]:
    """Root and relative path for a discovered file with canonical location ``real`` (§ 2.3)."""
    try:
        return root, PurePosixPath(*real.relative_to(root).parts)
    except ValueError:
        return real.parent, PurePosixPath(real.name)  # escaping link: the target's parent is the root


def _walk(directory: Path, root: Path, *, recursive: bool, suffix: str, output: Path | None,
          ancestors: frozenset[Identity], out: _Collector, errors: list[str]) -> None:
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        out.unreadable.append(directory)
        if directory == root:
            errors.append(f"cannot read folder {directory}: {exc.strerror or exc}")
        return
    entries.sort(key=lambda entry: natural_key(entry.name))
    for entry in entries:
        name = entry.name
        if name.startswith("."):  # hidden, AppleDouble (._x) and .norma-originals
            out.excluded["hidden"] += 1
            continue
        path = Path(entry.path)
        is_link = entry.is_symlink()
        try:
            info = os.stat(entry.path)  # follows links
        except OSError:
            if is_link:
                out.ignored_links += 1  # dangling or looping link
            else:
                out.unreadable.append(path)
            continue

        if stat.S_ISDIR(info.st_mode):
            if is_link:
                out.directory_links += 1  # never recursed into when discovered (§ 2.3)
                continue
            if unicodedata.normalize("NFC", name).casefold() in GENERATED_FOLDER_NAMES:
                out.excluded["generated"] += 1
                continue
            if _within(Path(os.path.realpath(entry.path)), output):
                out.excluded["generated"] += 1
                continue
            if recursive and _identity(info) not in ancestors:
                _walk(path, root, recursive=True, suffix=suffix, output=output,
                      ancestors=ancestors | {_identity(info)}, out=out, errors=errors)
            continue

        if not stat.S_ISREG(info.st_mode):
            out.excluded["special"] += 1  # FIFOs, sockets, devices: never opened
            continue
        real = Path(os.path.realpath(entry.path))
        if _within(real, output):
            out.excluded["generated"] += 1
            continue
        if _stem_has_suffix(name, suffix):
            out.excluded["suffix"] += 1
            continue
        if _extension(name) not in RECOGNISED_EXTENSIONS:
            out.excluded["extension"] += 1
            continue
        file_root, relative = _file_target(real, root)
        out.accept(real, file_root, relative, _identity(info), explicit=False)


def discover(arguments: Sequence[str | os.PathLike[str]], *, recursive: bool = False, suffix: str = "",
             output_dir: str | os.PathLike[str] | None = None) -> DiscoveryResult:
    """Expand explicit files and folders into deduplicated, ordered C-01 items (§ 2.1-2.4).

    ``suffix`` is the active expanded ``-s`` suffix and ``output_dir`` the canonical configured
    output folder; both only exclude entries found during folder expansion. Missing, dangling,
    looping, FIFO/device and unreadable-folder arguments raise one ``UsageError`` naming all of them.
    """
    output = Path(os.path.realpath(output_dir)) if output_dir is not None else None
    out = _Collector()
    errors: list[str] = []
    for argument in arguments:
        text = os.fspath(argument)
        try:
            info = os.stat(text)
        except FileNotFoundError:
            broken = os.path.islink(text)
            errors.append(f"{text}: " + ("broken symbolic link" if broken else "no such file or folder"))
            continue
        except OSError as exc:
            errors.append(f"{text}: {exc.strerror or exc}")
            continue
        real = Path(os.path.realpath(text))
        if stat.S_ISREG(info.st_mode):
            out.accept(real, real.parent, PurePosixPath(real.name), _identity(info), explicit=True)
        elif stat.S_ISDIR(info.st_mode):
            _walk(real, real, recursive=recursive, suffix=suffix, output=output,
                  ancestors=frozenset({_identity(info)}), out=out, errors=errors)
        else:
            errors.append(f"{text}: not a regular file or folder")
    if errors:
        raise UsageError("; ".join(errors))

    hints: list[str] = []
    if out.ignored_links:
        hints.append(f"{out.ignored_links} symbolic link(s) with unresolved targets were ignored")
    if out.directory_links:
        hints.append(f"{out.directory_links} folder link(s) inside the input were not followed; "
                     "name the folder as an argument to include it")
    if out.duplicates:
        hints.append(f"{out.duplicates} duplicate file(s) (same device and inode) were dropped")
    if out.unreadable:
        hints.append(f"{len(out.unreadable)} unreadable entry(ies) were skipped")
    return DiscoveryResult(
        items=tuple(out.items),
        excluded=FrozenMap(out.excluded),
        duplicates=out.duplicates,
        ignored_links=out.ignored_links,
        directory_links=out.directory_links,
        unreadable=tuple(out.unreadable),
        hints=tuple(hints),
    )


# --- Signature and structure (§ 2.5, § 8) ---------------------------------------------

SIGNATURE_STAGE = "signature"
MAX_CHANNELS = 2  # ADR-0020: more channels end SKIP ``unsupported_channels`` after the probe
_HEAD_BYTES = 16
_ID3_HEADER_BYTES = 10
_ID3_FOOTER_BYTES = 10
_PCM_AIFC = frozenset({b"NONE", b"twos", b"sowt", b"in24", b"in32", b"42ni", b"23ni",
                       b"fl32", b"FL32", b"fl64", b"FL64"})
_PCM_WAVE_TAGS = frozenset({1, 3, 0xFFFE})


@dataclass(frozen=True)
class SignatureResult:
    """An identified container: ``kind`` is wav, aiff, flac, mp4, caf, mpeg or adts."""

    kind: str
    size: int


def _fail(reason: str, detail: str) -> FileFailure:
    return FileFailure(reason, SIGNATURE_STAGE, detail)  # type: ignore[arg-type]


def _open(item: InputItem) -> int:
    try:
        fd = os.open(item.source_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError as exc:
        raise _fail("source_changed", f"source disappeared: {exc}") from None
    except PermissionError as exc:
        raise _fail("permission_denied", str(exc)) from None
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            raise _fail("permission_denied", str(exc)) from None
        if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.ENXIO):
            raise _fail("source_changed", f"source was replaced: {exc}") from None
        raise _fail("probe_error", str(exc)) from None
    return fd


def _read(fd: int, length: int, offset: int) -> bytes:
    try:
        return os.pread(fd, length, offset)
    except PermissionError as exc:
        raise _fail("permission_denied", str(exc)) from None
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            raise _fail("permission_denied", str(exc)) from None
        raise _fail("probe_error", f"read error: {exc}") from None


def _mpeg_sync(head: bytes) -> str | None:
    """``mpeg`` or ``adts`` when ``head`` starts with a plausible frame header, else None."""
    if len(head) < 4 or head[0] != 0xFF or head[1] & 0xE0 != 0xE0:
        return None
    if head[1] & 0xF6 == 0xF0:  # 12-bit sync, layer bits 00: ADTS
        return "adts" if (head[2] >> 2) & 0xF < 13 else None
    version = (head[1] >> 3) & 0b11
    layer = (head[1] >> 1) & 0b11
    if version == 0b01 or layer == 0:
        return None
    if (head[2] >> 4) == 0xF or (head[2] >> 2) & 0b11 == 0b11:  # bad bitrate / sample-rate index
        return None
    return "mpeg"


def _identify(fd: int, size: int) -> str | None:
    head = _read(fd, _HEAD_BYTES, 0)
    if head[:4] in (b"RIFF", b"RF64", b"BW64") and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"fLaC":
        return "flac"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head[:4] == b"caff":
        return "caf"
    kind = _mpeg_sync(head)
    if kind is not None:
        return kind
    if head[:3] == b"ID3" and len(head) >= _ID3_HEADER_BYTES:
        sizes = head[6:10]
        if any(byte & 0x80 for byte in sizes):
            return None
        length = (sizes[0] << 21) | (sizes[1] << 14) | (sizes[2] << 7) | sizes[3]
        after = _ID3_HEADER_BYTES + length + (_ID3_FOOTER_BYTES if head[5] & 0x10 else 0)
        if after >= size:
            return None
        tail = _read(fd, 4, after)
        if tail == b"fLaC":
            return "flac"
        return _mpeg_sync(tail)
    return None


def _walk_wave(fd: int, size: int) -> None:
    head = _read(fd, 12, 0)
    rf64 = head[:4] in (b"RF64", b"BW64")
    riff_size = struct.unpack("<I", head[4:8])[0]
    form_end = 8 + riff_size
    ds64: tuple[int, int] | None = None  # RF64 riff size, data size
    if rf64:
        # The ds64 chunk must follow WAVE immediately and carry the real 64-bit sizes.
        chunk = _read(fd, 36, 12)
        if len(chunk) < 36 or chunk[:4] != b"ds64" or struct.unpack("<I", chunk[4:8])[0] < 24:
            raise _fail("probe_error", "RF64/BW64 file without a usable ds64 chunk")
        riff64, data64 = struct.unpack("<QQ", chunk[8:24])
        ds64 = (riff64, data64)
        form_end = 8 + riff64
    elif riff_size in (0, 0xFFFFFFFF):
        raise _fail("probe_error", "RIFF size is unspecified; streamed WAV sizes are unsupported")

    position = 12
    fmt: tuple[int, int, int] | None = None  # tag, channels, block align
    data_length: int | None = None
    last_odd_end = -1
    while position + 8 <= min(form_end, size):
        header = _read(fd, 8, position)
        if len(header) < 8:
            break
        ident, chunk_size = header[:4], struct.unpack("<I", header[4:8])[0]
        start = position + 8
        if ds64 is not None and ident == b"data" and chunk_size == 0xFFFFFFFF:
            chunk_size = ds64[1]
        elif ident == b"data" and chunk_size == 0xFFFFFFFF:
            raise _fail("probe_error", "data chunk size is unspecified; unsupported")
        if start + chunk_size > size:
            raise _fail("truncated", f"chunk {ident!r} declares {chunk_size} bytes but only "
                                      f"{max(size - start, 0)} are present")
        if ident == b"fmt ":
            body = _read(fd, min(chunk_size, 40), start)
            if chunk_size < 16 or len(body) < 16:
                raise _fail("probe_error", "fmt chunk shorter than 16 bytes")
            tag, channels, _rate, _byte_rate, align = struct.unpack("<HHIIH", body[:14])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]  # sub-format GUID starts with the real tag
            fmt = (tag, channels, align)
        elif ident == b"data" and data_length is None:
            data_length = chunk_size
        if chunk_size & 1:
            last_odd_end = start + chunk_size
        position = start + chunk_size + (chunk_size & 1)
    if form_end > size and not (form_end - size == 1 and last_odd_end == size):
        raise _fail("truncated", f"RIFF form declares {form_end} bytes but the file holds {size}")
    if fmt is None or data_length is None:
        raise _fail("probe_error", "WAVE file lacks a fmt or data chunk")
    tag, channels, align = fmt
    if channels == 0 or align == 0:
        raise _fail("probe_error", "fmt chunk declares zero channels or block alignment")
    if tag in (1, 3) and data_length % align:
        raise _fail("probe_error", f"data length {data_length} is not a multiple of block alignment {align}")


def _walk_aiff(fd: int, size: int) -> None:
    head = _read(fd, 12, 0)
    aifc = head[8:12] == b"AIFC"
    form_end = 8 + struct.unpack(">I", head[4:8])[0]
    position = 12
    comm: tuple[int, int, int, bytes] | None = None  # channels, frames, bits, compression
    ssnd: tuple[int, int, int] | None = None  # data start, declared payload, offset
    last_odd_end = -1
    while position + 8 <= min(form_end, size):
        header = _read(fd, 8, position)
        if len(header) < 8:
            break
        ident, chunk_size = header[:4], struct.unpack(">I", header[4:8])[0]
        start = position + 8
        if start + chunk_size > size:
            raise _fail("truncated", f"chunk {ident!r} declares {chunk_size} bytes but only "
                                      f"{max(size - start, 0)} are present")
        if ident == b"COMM" and comm is None:
            need = 22 if aifc else 18
            body = _read(fd, need, start)
            if chunk_size < need or len(body) < need:
                raise _fail("probe_error", f"COMM chunk shorter than {need} bytes")
            channels, frames, bits = struct.unpack(">HIH", body[:8])
            comm = (channels, frames, bits, body[18:22] if aifc else b"NONE")
        elif ident == b"SSND" and ssnd is None:
            body = _read(fd, 8, start)
            if chunk_size < 8 or len(body) < 8:
                raise _fail("probe_error", "SSND chunk shorter than 8 bytes")
            offset = struct.unpack(">I", body[:4])[0]
            if 8 + offset > chunk_size:
                raise _fail("probe_error", "SSND offset lies beyond its chunk")
            ssnd = (start + 8 + offset, chunk_size - 8 - offset, offset)
        if chunk_size & 1:
            last_odd_end = start + chunk_size
        position = start + chunk_size + (chunk_size & 1)
    if form_end > size and not (form_end - size == 1 and last_odd_end == size):
        raise _fail("truncated", f"FORM declares {form_end} bytes but the file holds {size}")
    if comm is None:
        raise _fail("probe_error", "AIFF file lacks a COMM chunk")
    channels, frames, bits, compression = comm
    if channels == 0 or bits == 0:
        raise _fail("probe_error", "COMM chunk declares zero channels or bit depth")
    if compression not in _PCM_AIFC:
        return  # compressed payload: extents were checked, frame arithmetic does not apply
    needed = frames * channels * ((bits + 7) // 8)
    if ssnd is None:
        if frames:
            raise _fail("probe_error", "AIFF file declares frames but has no SSND chunk")
        return
    data_start, declared, _offset = ssnd
    if size - data_start < needed:
        raise _fail("truncated", f"COMM declares {needed} payload bytes but only "
                                  f"{max(size - data_start, 0)} are present")
    if declared < needed:
        raise _fail("probe_error", f"SSND chunk holds {declared} payload bytes, COMM needs {needed}")


def check_signature(item: InputItem) -> SignatureResult:
    """The bounded, read-only § 2.5 check; raises ``SkipFile``/``FileFailure`` with the C-13 reason."""
    fd = _open(item)
    try:
        try:
            info = os.fstat(fd)
        except OSError as exc:
            raise _fail("probe_error", str(exc)) from None
        if not stat.S_ISREG(info.st_mode) or _identity(info) != item.identity:
            raise _fail("source_changed", "opened file is not the one discovered")
        kind = _identify(fd, info.st_size)
        if kind is None:
            raise SkipFile("not_audio", SIGNATURE_STAGE, "no recognised audio or container signature")
        try:
            if kind == "wav":
                _walk_wave(fd, info.st_size)
            elif kind == "aiff":
                _walk_aiff(fd, info.st_size)
        except struct.error as exc:  # defensive: every read length is checked above
            raise _fail("probe_error", f"malformed {kind} structure: {exc}") from None
        return SignatureResult(kind, info.st_size)
    finally:
        os.close(fd)


def classify(item: InputItem, prober: Prober, *, deadline_s: float = PROBE_DEADLINE_S,
             cancel_event: threading.Event | None = None,
             is_processed: Callable[[InputItem], bool] | None = None,
             force: bool = False) -> ProbeResult:
    """Signature step, then the adapter probe, with the § 2.5 results for one item.

    Returns the C-02 probe for an item that may proceed. ``is_processed`` is the caller's
    ``NORMA`` mark test (owned by WS-06); it runs only for an otherwise usable item and ends
    ``SKIP already_processed`` unless ``force``, which bypasses nothing else (§ 2.2). A probed audio
    stream with more than two channels ends ``SKIP unsupported_channels`` (ADR-0020), before the mark test.
    """
    check_signature(item)
    result = probe(prober, item.source_path, deadline_s=deadline_s, cancel_event=cancel_event)
    if result.codec_class == "unsupported":
        raise SkipFile("unsupported_codec", "probe", f"codec {result.codec!r} is not supported")
    if result.channels > MAX_CHANNELS:  # ADR-0020: v1 handles mono and stereo only
        raise SkipFile("unsupported_channels", "probe",
                       f"{result.channels} audio channels; v1 handles mono and stereo only")
    if is_processed is not None and not force and is_processed(item):
        raise SkipFile("already_processed", "probe", "the file carries a NORMA processing mark")
    return result
