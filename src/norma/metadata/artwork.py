"""Artwork inspection and preparation over sips (metadata § 8, MD24; TST-M-06, F-16).

This is the ``metadata_prepared`` edge for pictures. It turns a supplied image
file (``--artwork``, a manifest cell, ``--folder-artwork``) or an embedded source
picture into an immutable ``ArtworkPlan`` of ``Artwork`` bytes. It never ranks
against the source tags under ``--prefer`` (that is resolution, lane 3): it only
orders the explicit sources (E1 > E2 > E3 > E4, ``explicit_artwork``) and prepares
what the caller asks for.

Rules applied to a supplied file, once per distinct file identity:

- over 52,428,800 bytes (checked by size before any decoding), or wider or taller
  than 16,384 pixels: ``artwork_invalid`` ("too large to decode"); exactly the limit
  is accepted if sips can decode it;
- JPEG and PNG in RGB or grey keep their format and, unless they must be scaled, their
  exact bytes; every other format and every other colour space becomes an RGB JPEG at
  quality 90 (non-RGB spaces through the sRGB profile: ``sips -s space`` is unsupported);
- a longest side above ``--artwork-max-size`` is scaled down keeping the aspect ratio
  (a scaled PNG stays PNG; a scaled JPEG is re-encoded at quality 90); an image is
  never enlarged;
- a non-square image gets a note (``ArtworkResult.notes``), not a C-13 warning;
- sips works on a private copy of the bytes that were hashed, so a file swapped
  between hashing and decoding cannot reach sips; ``recheck`` re-reads the original
  (identity, size, mtime and SHA-256) before use and raises ``source_changed``.

Failure classes (all ``FileFailure`` except the explicit-path usage error):
``artwork_invalid`` for a missing, unreadable, empty, non-image, oversize or
unconvertible file; ``timeout`` when sips exceeds the 60 s inspection deadline.
A sips executable lost or changed after preflight reports the reason
``errors.tool_loss_reason(STAGE)`` (``artwork_invalid``, CR-079) with the cause in the
detail, as ``ToolMissing.to_failure`` does for every stage (CR-086). A missing sips at preflight is
``adapters.sips.locate_sips`` (exit 3).

Embedded source pictures keep their bytes. Only a front cover that MP4 ``covr``
cannot type (neither JPEG nor PNG) is converted, as above.

The module imports no subprocess or tag library; process work is ``adapters.sips``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import threading
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from norma.adapters import sips
from norma.errors import FileFailure, ToolError, ToolMissing, ToolTimeout, UsageError, tool_loss_reason
from norma.models import Artwork, ArtworkPlan, Identity, ManifestCell, Provenance, ReadDependency

STAGE = "metadata_prepared"
MAX_BYTES = 52_428_800
MAX_SIDE_PX = 16_384
JPEG_QUALITY = 90
FRONT_COVER = 3
FOLDER_NAMES = ("cover.jpg", "cover.png", "folder.jpg")
_KEEP_FORMATS = {"jpeg": "image/jpeg", "png": "image/png"}
_KEEP_SPACES = frozenset({"rgb", "gray", "grey"})
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_READ_CHUNK = 1024 * 1024


# --- Records ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceStamp:
    """What was read from a supplied file; ``recheck`` compares the file with it."""

    path: Path
    identity: Identity
    size: int
    mtime_ns: int
    sha256: str

    def read_dependency(self, purpose: str = "artwork") -> ReadDependency:
        """The C-08 protected read for this file."""
        return ReadDependency(self.path, self.identity, purpose)


@dataclass(frozen=True)
class PreparedImage:
    artwork: Artwork
    stamp: SourceStamp | None
    notes: tuple[str, ...]
    converted: bool
    scaled: bool


@dataclass(frozen=True)
class ArtworkResult:
    """An ``ArtworkPlan`` (``None`` = no opinion) with its notes and the files to recheck."""

    plan: ArtworkPlan | None
    notes: tuple[str, ...] = ()
    stamps: tuple[SourceStamp, ...] = ()


@dataclass(frozen=True)
class ExplicitArtwork:
    """The winning explicit artwork opinion (E1 to E4) for one file."""

    operation: str  # "set" or "clear"
    path: Path | None
    provenance: Provenance


# --- Explicit sources and ranks ----------------------------------------------------------


def check_cli_artwork(path: Path | str, *, cwd: Path | None = None) -> Path:
    """An explicit ``--artwork`` path must exist as a file: otherwise a usage error (exit 2, U-13)."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (cwd if cwd is not None else Path.cwd()) / candidate
    if not candidate.is_file():
        raise UsageError(f"--artwork file does not exist: {path}")
    return Path(os.path.abspath(candidate))


def find_folder_artwork(folder: Path) -> Path | None:
    """The first of ``cover.jpg``, ``cover.png``, ``folder.jpg`` in ``folder`` (case-insensitive), or ``None``."""
    try:
        entries = {entry.name.casefold(): entry for entry in os.scandir(folder) if entry.name}
    except (FileNotFoundError, NotADirectoryError):
        return None
    for name in FOLDER_NAMES:
        entry = entries.get(name)
        if entry is not None and (entry.is_file() or entry.is_symlink()):
            return Path(entry.path)
    return None


def explicit_artwork(*, remove: bool = False, cli_path: Path | None = None, row_cell: ManifestCell | None = None,
                     default_cell: ManifestCell | None = None, manifest_dir: Path | None = None,
                     folder_path: Path | None = None, folder_provenance: Provenance = "config",
                     ) -> ExplicitArtwork | None:
    """Order the explicit artwork opinions: E1 command line, E2 manifest row, E3 manifest defaults, E4 folder.

    ``--remove-artwork`` and a ``clear`` cell are clear opinions (R1). A manifest path is relative to the
    manifest's folder. ``None`` means no explicit opinion.
    """
    if remove:
        return ExplicitArtwork("clear", None, "cli")
    if cli_path is not None:
        return ExplicitArtwork("set", cli_path, "cli")
    for cell, provenance in ((row_cell, "manifest"), (default_cell, "manifest_default")):
        if cell is None:
            continue
        if cell.operation == "clear":
            return ExplicitArtwork("clear", None, provenance)  # type: ignore[arg-type]
        value = cell.value
        if not isinstance(value, Path):
            raise ValueError("a manifest artwork cell holds a path")
        if not value.is_absolute():
            if manifest_dir is None:
                raise ValueError("a relative manifest artwork path needs the manifest folder")
            value = manifest_dir / value
        return ExplicitArtwork("set", Path(os.path.abspath(value)), provenance)  # type: ignore[arg-type]
    if folder_path is not None:
        return ExplicitArtwork("set", folder_path, folder_provenance)
    return None


# --- Failures -----------------------------------------------------------------------------


def _invalid(path: Path | str, detail: str) -> FileFailure:
    return FileFailure("artwork_invalid", STAGE, f"{path}: {detail}")


def _tool_failure(path: Path | str, exc: ToolError) -> FileFailure:
    if isinstance(exc, ToolTimeout):
        return exc.to_failure("timeout", STAGE)
    if isinstance(exc, ToolMissing):
        return FileFailure(tool_loss_reason(STAGE), STAGE, f"{path}: sips is no longer available ({exc})")
    detail = str(exc)
    if exc.stderr_tail:  # a crashing sips prints a long stack: keep the first lines, bounded
        detail += " (" + "; ".join(line[:160] for line in exc.stderr_tail[:2]) + ")"
    return _invalid(path, detail)


def sniff_storable(data: bytes) -> str | None:
    """``image/jpeg`` or ``image/png`` from the signature, else ``None`` (MP4 ``covr`` stores only these)."""
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    return None


def _check_complete(data: bytes, fmt: str, shown: Path | str) -> None:
    """CR-100: sips may accept a header or repair truncation; kept bytes must be complete."""
    if fmt == "png":
        if not data.startswith(_PNG_MAGIC):
            raise _invalid(shown, "invalid PNG signature")
        position, seen_header, seen_data = 8, False, False
        while position + 12 <= len(data):
            size = int.from_bytes(data[position:position + 4], "big")
            kind = data[position + 4:position + 8]
            end = position + 12 + size
            if end > len(data):
                break
            if zlib.crc32(data[position + 4:end - 4]) != int.from_bytes(data[end - 4:end], "big"):
                raise _invalid(shown, "invalid PNG chunk CRC")
            if not seen_header:
                if kind != b"IHDR" or size != 13:
                    raise _invalid(shown, "invalid PNG header")
                seen_header = True
            seen_data |= kind == b"IDAT"
            if kind == b"IEND":
                if size != 0 or not seen_data:
                    raise _invalid(shown, "invalid PNG end chunk")
                return
            position = end
        raise _invalid(shown, "truncated PNG: missing complete IEND")
    if fmt == "jpeg":
        if not data.startswith(b"\xff\xd8"):
            raise _invalid(shown, "invalid JPEG signature")
        position, in_scan, seen_scan = 2, False, False
        while position < len(data):
            if in_scan:
                position = data.find(b"\xff", position)
                if position < 0:
                    break
            elif data[position] != 0xFF:
                raise _invalid(shown, "invalid JPEG marker")
            while position < len(data) and data[position] == 0xFF:
                position += 1
            if position >= len(data):
                break
            marker = data[position]
            position += 1
            if in_scan and (marker == 0 or 0xD0 <= marker <= 0xD7):
                continue  # stuffed entropy byte or restart marker, not the end
            in_scan = False
            if marker == 0xD9:
                if not seen_scan:
                    raise _invalid(shown, "JPEG has no image scan")
                return
            if marker in (0, 0xD8) or 0xD0 <= marker <= 0xD7:
                raise _invalid(shown, "invalid JPEG marker")
            if marker == 1:  # standalone TEM marker
                continue
            if position + 2 > len(data):
                break
            size = int.from_bytes(data[position:position + 2], "big")
            if size < 2:
                raise _invalid(shown, "invalid JPEG segment size")
            if position + size > len(data):
                break
            position += size
            if marker == 0xDA:
                in_scan, seen_scan = True, True
        raise _invalid(shown, "truncated JPEG: missing complete EOI")


# --- Read-only inspection references (D-101, MD27) ------------------------------------------


@dataclass(frozen=True)
class ArtworkReference:
    """Exact supplied bytes and protected-read evidence; not a prepared rendition."""

    data: bytes
    stamp: SourceStamp


def _reference_bytes(path: Path, expected: SourceStamp | None = None) -> ArtworkReference:
    stage = "content_verified"
    before = None
    try:
        before = path.stat()
        if expected is not None and ((before.st_dev, before.st_ino) != expected.identity
                or before.st_size != expected.size or before.st_mtime_ns != expected.mtime_ns):
            raise FileFailure("source_changed", stage, f"{path}: artwork reference changed")
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_BYTES:
            raise FileFailure("artwork_invalid", stage, f"{path}: empty, non-regular or too large to decode")
        with open(path, "rb") as handle:
            opened = os.fstat(handle.fileno())
            data = handle.read(MAX_BYTES + 1)
            after = os.fstat(handle.fileno())
        current = path.stat()
    except OSError as exc:
        changed = expected is not None
        if before is not None:
            try:
                now = path.stat()
                changed |= ((now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns)
                            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns))
            except OSError:
                changed = True
        reason = "source_changed" if changed else "artwork_invalid"
        raise FileFailure(reason, stage, f"{path}: cannot read artwork reference ({exc.strerror or exc})") from None
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if any((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) != identity for s in (opened, after, current)) or len(data) != before.st_size:
        raise FileFailure("source_changed", stage, f"{path}: artwork reference changed while reading")
    stamp = SourceStamp(path, (before.st_dev, before.st_ino), before.st_size, before.st_mtime_ns,
                        hashlib.sha256(data).hexdigest())
    if expected is not None and stamp.sha256 != expected.sha256:
        raise FileFailure("source_changed", stage, f"{path}: artwork reference changed content")
    return ArtworkReference(data, stamp)


def recheck_artwork_reference(reference: ArtworkReference) -> None:
    """Protected read recheck before use/publication; a change raises ``source_changed``."""
    _reference_bytes(reference.stamp.path, reference.stamp)


def read_artwork_reference(path: Path, tool: sips.Sips, *, deadline_s: float = sips.INSPECT_DEADLINE_S,
                           cancel_event: threading.Event | None = None) -> ArtworkReference:
    """Read and validate an MD24 image without preparation, conversion or temporary files.

    Caller resolves manifest-relative paths first and registers ``stamp.read_dependency()``.
    Call ``recheck_artwork_reference`` before using cached evidence. All per-file failures
    are at ``content_verified``; image/reference failures are ``artwork_invalid``.
    """
    reference = _reference_bytes(Path(os.path.abspath(path)))
    try:
        info = sips.inspect(tool, reference.stamp.path, deadline_s=deadline_s, cancel_event=cancel_event)
    except ToolError as exc:
        recheck_artwork_reference(reference)
        failure = _tool_failure(path, exc)
        raise FileFailure(failure.reason, "content_verified", failure.detail) from None
    recheck_artwork_reference(reference)
    if info.width_px > MAX_SIDE_PX or info.height_px > MAX_SIDE_PX:
        raise FileFailure("artwork_invalid", "content_verified", f"{path}: too large to decode ({info.width_px}x{info.height_px})")
    try:
        _check_complete(reference.data, info.format, path)
    except FileFailure as exc:
        raise FileFailure(exc.reason, "content_verified", exc.detail) from None
    return reference


# --- Preparation ----------------------------------------------------------------------------


class ArtworkPreparer:
    """Prepares supplied images through sips, caching one result per distinct file identity.

    ``workdir`` is an existing directory for private scratch copies, removed after each operation.
    Thread-safe: concurrent requests for one identity prepare once.
    """

    def __init__(self, tool: sips.Sips, workdir: Path, *, max_size: int = 3000,
                 deadline_s: float = sips.INSPECT_DEADLINE_S, cancel_event: threading.Event | None = None) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if not workdir.is_dir():
            raise ValueError(f"workdir must be an existing directory: {workdir}")
        self.tool = tool
        self.workdir = workdir
        self.max_size = max_size
        self.deadline_s = deadline_s
        self.cancel_event = cancel_event
        self._lock = threading.Lock()
        self._cache: dict[tuple[Identity, int, int, int], PreparedImage | FileFailure] = {}
        self._key_locks: dict[tuple[Identity, int, int, int], threading.Lock] = {}
        self.preparations = 0  # distinct preparations run (cache effectiveness, for tests)

    # -- supplied files

    def prepare(self, path: Path, *, picture_type: int = FRONT_COVER, description: str = "") -> PreparedImage:
        """Prepare one supplied image file; raises ``FileFailure`` (``artwork_invalid``, ``timeout``, ``source_changed``)."""
        path = Path(os.path.abspath(path))
        real = Path(os.path.realpath(path))
        try:
            info = os.stat(real)
        except FileNotFoundError:
            raise _invalid(path, "the file does not exist") from None
        except OSError as exc:
            raise _invalid(path, f"the file cannot be read ({exc.strerror or exc})") from None
        if not stat.S_ISREG(info.st_mode):
            raise _invalid(path, "not a regular file")
        if info.st_size > MAX_BYTES:
            raise _invalid(path, f"too large to decode ({info.st_size} bytes; the limit is {MAX_BYTES})")
        if info.st_size == 0:
            raise _invalid(path, "the file is empty")
        key = ((info.st_dev, info.st_ino), info.st_size, info.st_mtime_ns, self.max_size)
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            cached = self._cache.get(key)
            if cached is None:
                try:
                    cached = self._prepare_file(real, path)
                except FileFailure as failure:
                    if failure.reason == "artwork_invalid":  # content verdicts repeat; timeouts and changes do not
                        self._cache[key] = failure
                    raise
                self._cache[key] = cached
            elif isinstance(cached, FileFailure):
                raise FileFailure(cached.reason, cached.stage, cached.detail)
            else:
                self.recheck(cached.stamp)  # a cache hit is a use: the file must still be the one prepared
        return _with(cached, picture_type, description)

    def recheck(self, stamp: SourceStamp | None) -> None:
        """Raise ``source_changed`` unless the file still has the stamp's identity, size, mtime and SHA-256."""
        if stamp is None:
            return
        try:
            before = os.stat(stamp.path)
            digest, size = _hash_file(stamp.path)
            after = os.stat(stamp.path)
        except OSError as exc:
            raise FileFailure("source_changed", STAGE, f"{stamp.path} can no longer be read ({exc.strerror or exc})") from None
        for info in (before, after):
            if ((info.st_dev, info.st_ino) != stamp.identity or info.st_size != stamp.size
                    or info.st_mtime_ns != stamp.mtime_ns):
                raise FileFailure("source_changed", STAGE, f"{stamp.path} changed after it was prepared")
        if digest != stamp.sha256 or size != stamp.size:
            raise FileFailure("source_changed", STAGE, f"{stamp.path} changed content after it was prepared")

    def _prepare_file(self, real: Path, shown: Path) -> PreparedImage:
        try:
            with open(real, "rb") as handle:
                before = os.fstat(handle.fileno())
                data = handle.read(MAX_BYTES + 1)
                after = os.fstat(handle.fileno())
        except OSError as exc:
            raise _invalid(shown, f"the file cannot be read ({exc.strerror or exc})") from None
        if len(data) > MAX_BYTES:
            raise _invalid(shown, f"too large to decode (more than {MAX_BYTES} bytes)")
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) != after.st_size:
            raise FileFailure("source_changed", STAGE, f"{shown} changed while it was read")
        stamp = SourceStamp(real, (after.st_dev, after.st_ino), after.st_size, after.st_mtime_ns,
                            hashlib.sha256(data).hexdigest())
        image = self._prepare_bytes(data, shown)
        return PreparedImage(image.artwork, stamp, image.notes, image.converted, image.scaled)

    def _prepare_bytes(self, data: bytes, shown: Path | str) -> PreparedImage:
        """Inspect, convert and scale ``data`` (already within the byte limit) in a private scratch folder."""
        self.preparations += 1
        scratch = Path(tempfile.mkdtemp(prefix="artwork-", dir=self.workdir))
        try:
            source = scratch / "source"
            source.write_bytes(data)
            try:
                info = sips.inspect(self.tool, source, deadline_s=self.deadline_s, cancel_event=self.cancel_event)
            except ToolError as exc:
                raise _tool_failure(shown, exc) from None
            if info.width_px > MAX_SIDE_PX or info.height_px > MAX_SIDE_PX:
                raise _invalid(shown, f"too large to decode ({info.width_px}x{info.height_px} pixels; "
                                      f"the limit is {MAX_SIDE_PX} per side)")
            keep = info.format in _KEEP_FORMATS and info.space.lower() in _KEEP_SPACES
            longest = max(info.width_px, info.height_px)
            scale_to = self.max_size if longest > self.max_size else None
            notes = ((f"artwork is not square ({info.width_px}x{info.height_px})",)
                     if info.width_px != info.height_px else ())
            if keep and scale_to is None:
                _check_complete(data, info.format, shown)
                return PreparedImage(_artwork(data, info), None, notes, False, False)
            target_format = info.format if keep else "jpeg"
            dest = scratch / "prepared"
            try:
                sips.convert(
                    self.tool, source, dest, image_format=target_format,
                    quality=JPEG_QUALITY if target_format == "jpeg" else None, max_side=scale_to,
                    profile=None if info.space.lower() == "rgb" or keep else sips.SRGB_PROFILE,
                    deadline_s=self.deadline_s, cancel_event=self.cancel_event,
                )
                out = dest.read_bytes()
                out_info = sips.inspect(self.tool, dest, deadline_s=self.deadline_s, cancel_event=self.cancel_event)
            except ToolError as exc:
                raise _tool_failure(shown, exc) from None
            except OSError as exc:
                raise _invalid(shown, f"the prepared image cannot be read ({exc.strerror or exc})") from None
            self._check_output(shown, info, out_info, target_format, keep, scale_to)
            return PreparedImage(_artwork(out, out_info), None, notes, not keep, scale_to is not None)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _check_output(self, shown: Path | str, src: sips.ImageInfo, out: sips.ImageInfo, fmt: str, keep: bool,
                      scale_to: int | None) -> None:
        wanted_space = src.space.lower() if keep else "rgb"
        if out.format != fmt or out.space.lower() != wanted_space:
            raise _invalid(shown, f"sips produced {out.format}/{out.space} instead of {fmt}/{wanted_space}")
        if scale_to is None:
            expected = (src.width_px, src.height_px)
        else:
            factor = scale_to / max(src.width_px, src.height_px)
            expected = (src.width_px * factor, src.height_px * factor)
        if abs(out.width_px - expected[0]) > 1 or abs(out.height_px - expected[1]) > 1:
            raise _invalid(shown, f"sips produced {out.width_px}x{out.height_px}, expected about "
                                  f"{expected[0]:.0f}x{expected[1]:.0f}")

    # -- embedded pictures

    def convert_embedded(self, picture: Artwork) -> Artwork:
        """An embedded picture MP4 can store (JPEG or PNG signature) is returned untouched; any other is converted."""
        if sniff_storable(picture.data) is not None:
            return picture
        if len(picture.data) > MAX_BYTES:
            raise _invalid("embedded picture", f"too large to decode ({len(picture.data)} bytes)")
        if not picture.data:
            raise _invalid("embedded picture", "the picture is empty")
        image = self._prepare_bytes(picture.data, "embedded picture")
        return _with(image, picture.picture_type, picture.description).artwork


# --- Plans --------------------------------------------------------------------------------------


def clear_plan(provenance: Provenance) -> ArtworkPlan:
    """``--remove-artwork`` or ``[clear]``: an explicit plan with no pictures."""
    return ArtworkPlan(pictures=(), provenance=provenance, source_locator=None)


def embedded_plan(preparer: ArtworkPreparer, pictures: Sequence[Artwork], *, target_mp4: bool) -> ArtworkResult:
    """The source's embedded pictures as a plan (provenance ``source``); no pictures is no opinion.

    Bytes are kept. For an MP4 target only a front cover that is not JPEG or PNG is converted
    (``changed``); other pictures have no ``covr`` slot and are left for the loss accounting.
    """
    if not pictures:
        return ArtworkResult(None)
    kept = tuple(preparer.convert_embedded(item) if target_mp4 and item.picture_type == FRONT_COVER else item
                 for item in pictures)
    return ArtworkResult(ArtworkPlan(kept, "source", "embedded"))


def supplied_plan(preparer: ArtworkPreparer, path: Path, provenance: Provenance, *,
                  source_pictures: Sequence[Artwork] = (), description: str = "") -> ArtworkResult:
    """One front cover (type 3) from a supplied file; the source's other pictures stay (MD23).

    Source front covers are replaced. ``source_pictures`` of another type are carried as embedded
    pictures would be.
    """
    image = preparer.prepare(path, description=description)
    others = tuple(item for item in source_pictures if item.picture_type != FRONT_COVER)
    pictures = (image.artwork, *others)
    assert image.stamp is not None
    return ArtworkResult(ArtworkPlan(pictures, provenance, str(image.stamp.path)), image.notes, (image.stamp,))


def plan_explicit(preparer: ArtworkPreparer, explicit: ExplicitArtwork | None, *,
                  source_pictures: Sequence[Artwork] = ()) -> ArtworkResult:
    """The plan for the winning explicit opinion; ``None`` gives no opinion (embedded handling is the caller's)."""
    if explicit is None:
        return ArtworkResult(None)
    if explicit.operation == "clear":
        return ArtworkResult(clear_plan(explicit.provenance))
    assert explicit.path is not None
    return supplied_plan(preparer, explicit.path, explicit.provenance, source_pictures=source_pictures)


def recheck_result(preparer: ArtworkPreparer, result: ArtworkResult) -> None:
    """Recheck every supplied file behind a result before it is used (``source_changed``)."""
    for stamp in result.stamps:
        preparer.recheck(stamp)


# --- Helpers ----------------------------------------------------------------------------------


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
            total += len(chunk)
            if total > MAX_BYTES:
                break
    return digest.hexdigest(), total


def _artwork(data: bytes, info: sips.ImageInfo, picture_type: int = FRONT_COVER, description: str = "") -> Artwork:
    return Artwork(data, _KEEP_FORMATS.get(info.format, "image/jpeg"), info.width_px, info.height_px,
                   picture_type, description)


def _with(image: PreparedImage, picture_type: int, description: str) -> PreparedImage:
    art = image.artwork
    if art.picture_type == picture_type and art.description == description:
        return image
    return PreparedImage(Artwork(art.data, art.mime, art.width_px, art.height_px, picture_type, description),
                         image.stamp, image.notes, image.converted, image.scaled)
