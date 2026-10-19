"""Tag writing and read-back: C-07 ``TagWriteResult`` (WS-06, I/O edge).

Behaviour: MD22 (read-back), MD25/MD36/MD41
(sidecar members), MD29–MD31 (writing and ID3 conversion) and MD39 (structural
residue). ``mutagen`` is imported here and in ``read.py`` only.

Three jobs, one code path:

- **Preparation** (``prepare_store``, the ``metadata_prepared`` edge of MD29):
  ``mapping.native_recipes`` and same-family passthrough items are turned into
  mutagen frames, atoms and comments, rendered into an in-memory store with
  the same save code the writer uses, and read back with ``read.py``. The
  expected native items therefore carry exactly the MD32 payloads that
  read-back will produce. ID3 passthrough frames go through the target
  version's conversion one by one (MD31), so each requested occurrence is
  linked to the expected occurrences it became, or to none.
- **Writing** (``write_tags``): each output store is rebuilt completely from
  ``MetadataPlan.expected_native`` alone, by decoding its MD32 payloads; WAV
  RIFF INFO is written by the render (``audio/render.py`` R8), never here. Only
  the private candidate path the caller gives is opened for writing; a path
  that is, or is the same file as, a protected path is refused.
- **Read-back** (``verify_tags``): the candidate is read with ``read.py`` and
  compared with the plan (native items with duplicate order, canonical
  projection, pictures, the mark, structure). Matching tags stage the sidecar
  members by ``member_id``; the bytes are handed back, never written (WS-05
  publishes them).
"""

from __future__ import annotations

import base64
import errno
import hashlib
import io
import json
import os
import stat
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutagen.aiff import AIFF
from mutagen.flac import FLAC, Picture
from mutagen.id3 import APIC, ID3, Frames, Frames_2_2
from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm, MP4Tags
from mutagen.wave import WAVE

from norma import __version__
from norma.errors import FileFailure
from norma.metadata import mapping, model
from norma.metadata.artwork import ArtworkReference
from norma.metadata import read as reader
from norma.models import (
    Artwork, Diagnostic, FieldStatus, FrozenMap, InventoryRef, MetadataPlan, NonTagItem, RawNativeItem,
    ManifestCell, SourceTags, TagWriteResult,
)

RAW = reader.RAW
JSON = reader.JSON
STAGE_WRITE = "tags_written"
STAGE_VERIFY = "content_verified"
PAYLOAD_LIMIT = reader.PAYLOAD_LIMIT
SIDECAR_VERSION = 1

TARGET_CONTAINER = FrozenMap({"alac": "mp4", "aac": "mp4", "flac": "flac", "mp3": "mp3", "aiff": "aiff", "wav": "wav"})
"""C-04 target format → the MD3 container row its output is read with."""

V24_ONLY_FRAMES = frozenset({
    "ASPI", "EQU2", "RVA2", "SEEK", "SIGN", "TDEN", "TDOR", "TDRC", "TDRL", "TDTG", "TIPL", "TMCL", "TMOO",
    "TPRO", "TSOA", "TSOP", "TSOT", "TSST",
})
"""ID3v2.4-only frame IDs; none may appear in an ID3v2.3 tag (MD31, P4, P12)."""

_INFO_KEYS = FrozenMap({**{native: field for field, native in (
    ("title", "INAM"), ("artist", "IART"), ("album", "IPRD"), ("date", "ICRD"), ("track", "IPRT"),
    ("genre", "IGNR"), ("comment", "ICMT"))}, **dict(mapping.INFO_PASSTHROUGH)})
"""RIFF INFO ID → FFmpeg ``-metadata`` key for the render's INFO subset (MD30)."""

_SKELETON_MP4 = (struct.pack(">I4s", 20, b"ftyp") + b"M4A \x00\x00\x02\x00M4A "
                 + struct.pack(">I4s", 116, b"moov") + struct.pack(">I4s", 108, b"mvhd") + bytes(100))


def _skeleton_flac() -> bytes:
    info = bytearray(34)
    info[0:2] = info[2:4] = (4096).to_bytes(2, "big")
    info[10:18] = ((48000 << 44) | (1 << 41) | (15 << 36)).to_bytes(8, "big")
    return b"fLaC" + bytes([0x80]) + (34).to_bytes(3, "big") + bytes(info)


class _Collision(ValueError):
    """Two occurrences that one mutagen store cannot hold separately (a planning bug when strict)."""


# --- Payload decoding (MD32 → mutagen objects) ------------------------------------------------


def _id3_field(value: Any, version: str) -> Any:
    if isinstance(value, bool):
        raise ValueError("dn-json-1 ID3 payloads hold no booleans")
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, list):
        return [_id3_field(item, version) for item in value]
    if isinstance(value, dict) and set(value) == {"b64"}:
        return base64.b64decode(value["b64"], validate=True)
    if isinstance(value, dict) and set(value) == {"ts"}:
        return value["ts"]
    if isinstance(value, dict) and set(value) == {"frames"}:
        return [_id3_frame_from_document(document, version) for document in value["frames"]]
    raise ValueError(f"not an MD32 ID3 field value: {value!r}")


def _id3_frame_from_document(document: Mapping[str, Any], version: str) -> Any:
    table = Frames_2_2 if version == "2.2" else Frames
    frame_class = table.get(document.get("frame"))
    if frame_class is None:
        raise ValueError(f"unknown ID3 frame {document.get('frame')!r}")
    return frame_class(**{name: _id3_field(value, version) for name, value in document["fields"]})


def id3_frame(item: RawNativeItem) -> Any:
    """The mutagen frame a ``dn-json-1`` ``id3v2`` item describes (MD29: built from parsed values)."""
    if item.store != "id3v2" or item.payload_encoding != JSON:
        raise ValueError(f"{item.identity}: only dn-json-1 ID3 items can be written (MD32 fall-back rows are evidence)")
    return _id3_frame_from_document(json.loads(item.payload), item.native_version or "2.4")


def _mp4_value(value: Any) -> Any:
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, dict) and set(value) == {"pair"}:
        number, total = value["pair"]
        return (int(number), int(total))
    if isinstance(value, dict) and set(value) == {"freeform", "dataformat", "version"}:
        return MP4FreeForm(base64.b64decode(value["freeform"], validate=True), dataformat=value["dataformat"],
                           version=value["version"])
    if isinstance(value, dict) and set(value) == {"cover", "imageformat"}:
        return MP4Cover(base64.b64decode(value["cover"], validate=True), imageformat=value["imageformat"])
    if isinstance(value, dict) and set(value) == {"b64"}:
        return base64.b64decode(value["b64"], validate=True)
    raise ValueError(f"not an MD32 MP4 value: {value!r}")


def mp4_entry(item: RawNativeItem) -> tuple[str, Any]:
    """``(atom key, mutagen value)`` for a ``dn-json-1`` ``mp4`` item."""
    if item.store != "mp4" or item.payload_encoding != JSON:
        raise ValueError(f"{item.identity}: only dn-json-1 MP4 items can be written (MD32 fall-back rows are evidence)")
    document = json.loads(item.payload)
    values = [_mp4_value(value) for value in document["values"]]
    return document["atom"], values[0] if document["scalar"] else values


def vorbis_entry(item: RawNativeItem) -> tuple[str, str] | Picture:
    """A Vorbis comment ``(stored key, text)`` or a FLAC ``Picture``; invalid UTF-8 cannot be written."""
    if item.store != "vorbis" or item.payload_encoding != RAW:
        raise ValueError(f"{item.identity}: not a Vorbis item")
    if item.native_type == "PICTURE":
        return Picture(item.payload)
    return item.key, item.payload.decode("utf-8")


# --- Recipes (mapping.native_recipes → mutagen objects) -----------------------------------------


def _recipe_frame(recipe: Mapping[str, Any]) -> Any:
    attributes = dict(recipe["attributes"])
    native_type, value = recipe["native_type"], recipe["value"]
    if isinstance(value, Artwork):
        return APIC(encoding=attributes["encoding"], mime=value.mime, type=value.picture_type,
                    desc=value.description, data=value.data)
    text = list(value)
    if native_type == "TCON":
        # ID3 genre references are not literal text. Escape the first parenthesis;
        # bare numeric/RX/CR references have no exact literal representation.
        text = ["(" + entry if entry.startswith("(") else entry for entry in text]
        frame = Frames[native_type](text=text, **attributes)
        if frame.genres != list(value):
            raise FileFailure("metadata_loss_error", "metadata_prepared",
                              f"genre: ID3 TCON cannot represent this text exactly: {list(value)!r}")
        return frame
    return Frames[native_type](text=text, **attributes)


def _recipe_mp4(recipe: Mapping[str, Any]) -> tuple[str, Any]:
    key, value, attributes = recipe["key"], recipe["value"], dict(recipe["attributes"])
    if isinstance(value, Artwork):
        return key, [MP4Cover(value.data, imageformat=attributes["imageformat"])]
    if key.startswith("----:"):
        return key, [MP4FreeForm(text.encode("utf-8"), dataformat=attributes["dataformat"],
                                 version=attributes["version"]) for text in value]
    if isinstance(value, bool):
        return key, value
    return key, list(value)


def _picture(artwork: Artwork) -> Picture:
    picture = Picture()
    picture.type = artwork.picture_type
    picture.mime = artwork.mime
    picture.desc = artwork.description
    picture.width = artwork.width_px or 0
    picture.height = artwork.height_px or 0
    picture.data = artwork.data
    return picture


def _recipe_vorbis(recipe: Mapping[str, Any]) -> tuple[str, str] | Picture:
    value = recipe["value"]
    if isinstance(value, Artwork):
        return _picture(value)
    return recipe["key"], value


# --- ID3 conversion and assembly (MD29, MD31) ------------------------------------------------------


def _convert_id3(frame: Any, source_version: str, target_version: str) -> list[Any]:
    """One passthrough frame converted alone to the target version; ``[]`` if the conversion removes it."""
    tags = ID3()
    tags.add(frame)  # upgrades an ID3v2.2 frame class (mutagen's own conversion)
    if source_version != target_version:
        if target_version == "2.4":
            tags.update_to_v24()
        else:
            tags.update_to_v23()
    if target_version == "2.3":
        for key in [key for key, value in tags.items() if value.FrameID in V24_ONLY_FRAMES]:
            del tags[key]  # update_to_v23 leaves RVA2:<desc>, EQU2:<desc> (P12) and a keyed SIGN
    return list(tags.values())


def _id3_tags(frames: Sequence[Any], *, strict: bool = True) -> tuple[ID3, list[bool]]:
    """Assemble frames without mutagen's merge: pictures with equal keys get a salt, other equal keys collide.

    Returns the tags and, per frame, whether it was kept (``strict`` raises instead of dropping).
    """
    tags = ID3()
    kept = []
    for frame in frames:
        key = frame.HashKey
        if key in tags and isinstance(frame, APIC):
            counter = 1
            while frame.HashKey in tags:
                counter += 1
                frame.salt = f"#{counter}"
            key = frame.HashKey
        if key in tags:
            if strict:
                raise _Collision(key)
            kept.append(False)
            continue
        tags[key] = frame
        kept.append(True)
    return tags, kept


def _render_id3(tags: ID3, version: str) -> bytes:
    buffer = io.BytesIO()
    tags.save(buffer, v1=0, v2_version=int(version[-1]), padding=lambda info: 0)
    return buffer.getvalue()


def _mp4_tags(entries: Sequence[tuple[str, Any]], *, strict: bool = True) -> tuple[dict[str, Any], list[bool]]:
    tags: dict[str, Any] = {}
    kept = []
    for key, value in entries:
        if key == "covr" and key in tags:
            tags[key] = list(tags[key]) + list(value)
        elif key in tags:
            if strict:
                raise _Collision(key)
            kept.append(False)
            continue
        else:
            tags[key] = value
        kept.append(True)
    return tags, kept


def _render_mp4(tags: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO(_SKELETON_MP4)
    mp4 = MP4Tags()
    mp4.update(tags)
    mp4.save(buffer, padding=lambda info: 0)
    return buffer.getvalue()


def _apply_vorbis(flac: FLAC, entries: Sequence[tuple[str, str] | Picture]) -> None:
    if flac.tags is None:
        flac.add_tags()
    del flac.tags[:]
    flac.clear_pictures()
    for entry in entries:
        if isinstance(entry, Picture):
            flac.add_picture(entry)
        else:
            flac.tags.append(entry)


def _render_vorbis(entries: Sequence[tuple[str, str] | Picture]) -> bytes:
    buffer = io.BytesIO(_skeleton_flac())
    flac = FLAC(buffer)
    _apply_vorbis(flac, entries)
    buffer.seek(0)
    flac.save(buffer, padding=lambda info: 0)
    return buffer.getvalue()


# --- Preparation (MD29) ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedStore:
    """One output store's expected native items and their links (implementation-local, not a contract record).

    ``items`` are in written order with read-back identities. ``recipe_ids[i]`` names the expected
    occurrence(s) recipe *i* produced. ``passthrough_ids`` maps each requested passthrough identity
    to the expected occurrences it became after conversion; ``()`` means the conversion or the store
    removed it (MD31: a ``dropped_format_limit`` for the planner, never a silent omission).
    """

    store: str
    items: tuple[RawNativeItem, ...]
    recipe_ids: tuple[tuple[str, ...], ...]
    passthrough_ids: FrozenMap


def _signature(item: RawNativeItem) -> tuple[Any, ...]:
    return item.store, item.key, item.native_type, item.native_version, item.payload_encoding, item.payload


def _link(store: str, items: tuple[RawNativeItem, ...], singles: Sequence[tuple[RawNativeItem, ...]]) -> list[tuple[str, ...]]:
    """Link each rendered entry (its own single-entry read) to the whole store's read-back occurrences."""
    pool: dict[tuple[Any, ...], list[str]] = {}
    for item in items:
        pool.setdefault(_signature(item), []).append(item.identity)
    links = []
    for single in singles:
        ids = []
        for item in single:
            candidates = pool.get(_signature(item))
            if not candidates:
                raise RuntimeError(f"{store}: a prepared occurrence did not survive the store render")
            ids.append(candidates.pop(0))
        links.append(tuple(ids))
    if any(pool.values()):
        raise RuntimeError(f"{store}: the store render holds an occurrence no entry produced")
    return links


def prepare_store(store: str, recipes: Sequence[Mapping[str, Any]] = (), passthrough: Sequence[RawNativeItem] = (),
                  *, id3_version: str = "2.3") -> PreparedStore:
    """MD29: the expected native items of one target store, encoded exactly as read-back will read them.

    ``recipes`` come from ``mapping.native_recipes`` for this store and version. ``passthrough`` are
    same-family ``requested_native`` items (MD19) in source form; ID3 items are converted to
    ``id3_version`` one by one (MD31). An item that cannot be held separately or written (a duplicate
    key mutagen would merge, invalid Vorbis text, a frame the conversion removes) gets no link.
    """
    if id3_version not in ("2.3", "2.4"):
        raise ValueError(f"not a target ID3 version: {id3_version!r}")
    for item in passthrough:
        if item.store != store:
            raise ValueError(f"{item.identity}: passthrough stays in its own family (MD19)")
    if store == "riff-info":
        return _prepare_info(recipes, passthrough)
    if store == "id3v2":
        recipe_entries = [[_recipe_frame(recipe)] for recipe in recipes]
        passthrough_entries = []
        for item in passthrough:
            frame = id3_frame(item)
            passthrough_entries.append(_convert_id3(frame, item.native_version or "2.4", id3_version))
    elif store == "mp4":
        recipe_entries = [[_recipe_mp4(recipe)] for recipe in recipes]
        passthrough_entries = [[mp4_entry(item)] for item in passthrough]
    elif store == "vorbis":
        recipe_entries = [[_recipe_vorbis(recipe)] for recipe in recipes]
        passthrough_entries = []
        for item in passthrough:
            try:
                entry = vorbis_entry(item)
            except UnicodeDecodeError:
                passthrough_entries.append([])
                continue
            valid_key = isinstance(entry, Picture) or (
                entry[0] and all(0x20 <= ord(char) <= 0x7D and char != "=" for char in entry[0]))
            passthrough_entries.append([entry] if valid_key else [])
    else:
        raise ValueError(f"not a target store: {store!r}")
    groups = recipe_entries + passthrough_entries
    flat = [entry for group in groups for entry in group]
    kept = _assemble_kept(store, flat, len([e for g in recipe_entries for e in g]))
    # A group survives whole or not at all, so a converted frame never half-arrives.
    survivors: list[list[Any]] = []
    position = 0
    for group in groups:
        flags = kept[position:position + len(group)]
        position += len(group)
        survivors.append(group if all(flags) else [])
    entries = [entry for group in survivors for entry in group]
    items = _read_render(store, _render(store, entries, id3_version))
    singles = [_read_render(store, _render(store, [entry], id3_version)) for entry in entries]
    links = iter(_link(store, items, singles))
    grouped = [tuple(identity for _ in group for identity in next(links)) for group in survivors]
    recipe_ids = tuple(grouped[:len(recipe_entries)])
    passthrough_ids = FrozenMap({item.identity: ids for item, ids in zip(passthrough, grouped[len(recipe_entries):])})
    if any(not ids for ids in recipe_ids):
        raise ValueError(f"{store}: a canonical recipe could not be stored")
    return PreparedStore(store, items, recipe_ids, passthrough_ids)


def _assemble_kept(store: str, entries: Sequence[Any], recipe_count: int) -> list[bool]:
    """Which entries the store can hold; a recipe collision is a planning bug."""
    if store == "id3v2":
        _, kept = _id3_tags(entries, strict=False)
    elif store == "mp4":
        _, kept = _mp4_tags(entries, strict=False)
    else:
        kept = [True] * len(entries)
    if not all(kept[:recipe_count]):
        raise ValueError(f"{store}: two canonical recipes share one native key")
    return kept


def _render(store: str, entries: Sequence[Any], id3_version: str) -> bytes:
    if store == "id3v2":
        return _render_id3(_id3_tags(entries)[0], id3_version)
    if store == "mp4":
        return _render_mp4(_mp4_tags(entries)[0])
    return _render_vorbis(entries)


def _read_render(store: str, data: bytes) -> tuple[RawNativeItem, ...]:
    return reader.native_items_from_bytes(store, data)


def _prepare_info(recipes: Sequence[Mapping[str, Any]], passthrough: Sequence[RawNativeItem]) -> PreparedStore:
    """INFO values as FFmpeg writes them: UTF-8 with a terminal NUL, no CSET (MD29, MD30; formats M5)."""
    entries: list[tuple[str, str] | None] = []
    for recipe in recipes:
        entries.append((recipe["key"], recipe["value"]))
    if len({entry[0] for entry in entries if entry}) != len(entries):
        raise ValueError("riff-info: FFmpeg writes one value per INFO key")
    for item in passthrough:
        try:
            text = item.payload.rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        taken = {entry[0] for entry in entries if entry is not None}
        writable = item.key in mapping.INFO_PASSTHROUGH and text and "\x00" not in text and item.key not in taken
        entries.append((item.key, text) if writable else None)  # a repeated key keeps its first occurrence
    items, ids = [], []
    for entry in entries:
        if entry is None:
            ids.append(())
            continue
        key, text = entry
        if key not in _INFO_KEYS or not isinstance(text, str) or not text or "\x00" in text:
            raise ValueError(f"riff-info: {key!r} cannot be rendered as an INFO item")
        identity = model.native_identity("riff-info", key)
        items.append(RawNativeItem(identity, "riff-info", key, key, None, RAW, text.encode("utf-8") + b"\x00",
                                   "planned:LIST/INFO"))
        ids.append((identity,))
    return PreparedStore("riff-info", tuple(items), tuple(ids[:len(recipes)]),
                         FrozenMap({item.identity: link for item, link in zip(passthrough, ids[len(recipes):])}))


def riff_info_metadata(expected_native: Sequence[RawNativeItem]) -> tuple[tuple[str, str], ...]:
    """The render's ``-metadata`` pairs for the expected INFO items (R8 ``wav_info``; MD30)."""
    pairs = []
    for item in expected_native:
        if item.store != "riff-info":
            continue
        if item.key not in _INFO_KEYS or item.payload_encoding != RAW or not item.payload.endswith(b"\x00"):
            raise ValueError(f"{item.identity}: not a renderable INFO item")
        pairs.append((_INFO_KEYS[item.key], item.payload[:-1].decode("utf-8")))
    return tuple(pairs)


# --- Writing (MD29) ------------------------------------------------------------------------------


@dataclass(frozen=True)
class TagWrite:
    """``write_tags``'s outcome: the C-07 result and the staged member bytes, keyed by ``member_id``."""

    result: TagWriteResult
    staged: FrozenMap


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return info.st_dev, info.st_ino


def _check_candidate(path: Path, protected: Sequence[Path]) -> None:
    if not path.is_absolute():
        raise ValueError("the candidate path must be absolute")
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path}: tags are written only to a regular private candidate file")
    own = (info.st_dev, info.st_ino)
    for other in protected:
        if Path(other) == path or _identity(Path(other)) == own:
            raise ValueError(f"{path}: refusing to write tags to a protected file ({other})")


def _check_plan(plan: MetadataPlan, target: str, wav_tags: str, id3_version: str) -> tuple[str, ...]:
    if plan.conditional:
        raise ValueError("a conditional plan is never executable (MD35)")
    if plan.mark_state == "pending":
        raise ValueError("a plan with a pending mark cannot be written or pass read-back (MD22)")
    if target not in TARGET_CONTAINER:
        raise ValueError(f"unknown target format: {target!r}")
    stores = mapping.target_stores(target, wav_tags)
    if id3_version not in ("2.3", "2.4"):
        raise ValueError(f"not a target ID3 version: {id3_version!r}")
    marks = [item for item in plan.expected_native if model.is_mark_key(item.store, item.key)]
    if (plan.mark_state == "final") != bool(marks):
        raise ValueError("the expected inventory must hold the mark exactly when mark_state is final")
    for item in plan.expected_native:
        if item.store not in stores:
            raise ValueError(f"{item.identity}: store {item.store!r} is not written for {target} ({wav_tags})")
        if item.store == "id3v2":
            if item.native_version != id3_version:
                raise ValueError(f"{item.identity}: expected ID3v{item.native_version}, writing ID3v{id3_version}")
            if id3_version == "2.3" and item.native_type in V24_ONLY_FRAMES:
                raise ValueError(f"{item.identity}: an ID3v2.4-only frame cannot be stored as ID3v2.3")
    return stores


def _write_failure(path: Path, exc: Exception) -> FileFailure:
    cause = exc if isinstance(exc, OSError) else exc.__cause__ or exc.__context__  # mutagen wraps OSError
    if isinstance(cause, OSError) and cause.errno == errno.ENOSPC:
        return FileFailure("disk_full", STAGE_WRITE, f"{path}: {cause}")
    if isinstance(cause, OSError) and cause.errno in (errno.EACCES, errno.EPERM):
        return FileFailure("permission_denied", STAGE_WRITE, f"{path}: {cause}")
    return FileFailure("verification_tags", STAGE_WRITE, f"tags could not be written to {path.name}: {exc}")


def _write_stores(path: Path, plan: MetadataPlan, target: str, wav_tags: str, id3_version: str) -> None:
    items = plan.expected_native
    container = TARGET_CONTAINER[target]
    version = int(id3_version[-1])
    if container == "mp4":
        tags, _ = _mp4_tags([mp4_entry(item) for item in items if item.store == "mp4"])
        audio = MP4(path)
        if audio.tags is None:
            audio.add_tags()
        audio.tags.clear()
        audio.tags.update(tags)
        audio.save()
    elif container == "flac":
        audio = FLAC(path)
        _apply_vorbis(audio, [vorbis_entry(item) for item in items if item.store == "vorbis"])
        audio.save(deleteid3=True)
    elif container == "mp3":
        tags, _ = _id3_tags([id3_frame(item) for item in items if item.store == "id3v2"])
        tags.save(path, v1=0, v2_version=version)  # always rewritten at the target version (MD39)
    elif container == "aiff" or (container == "wav" and wav_tags == "info+id3"):
        tags, _ = _id3_tags([id3_frame(item) for item in items if item.store == "id3v2"])
        audio = (AIFF if container == "aiff" else WAVE)(path)
        if audio.tags is None:
            audio.add_tags()
        audio.tags.clear()
        for key, frame in tags.items():
            audio.tags[key] = frame
        audio.save(v2_version=version)
    # WAV ``info`` and ``none``: INFO is the render's (R8); no ID3 chunk is written.


def write_tags(candidate_path: Path, plan: MetadataPlan, *, target: str, wav_tags: str = "info+id3",
               id3_version: str = "2.3", protected: Sequence[Path] = (), audio_name: str | None = None,
               clears: Sequence[str] = (), tool_version: str | None = None) -> TagWrite:
    """Write the final plan revision's expected stores into the private candidate, then verify them.

    ``protected`` lists the source and every other path that must never be written; the candidate
    must be a regular file that is none of them. ``audio_name`` (the final audio file name) and
    ``clears`` (roll-up keys whose winning opinion is a clear, MD41) are needed only when the plan
    requests sidecar members. Raises ``ValueError`` for a plan the writer must never execute and
    ``FileFailure`` when the candidate cannot be written or read.
    """
    path = Path(candidate_path)
    _check_plan(plan, target, wav_tags, id3_version)
    _check_candidate(path, protected)
    if plan.sidecar_requests and not audio_name:
        raise ValueError("sidecar members need the final audio file name")
    try:
        _write_stores(path, plan, target, wav_tags, id3_version)
    except ValueError:
        raise  # a plan the writer must not execute (undecodable payload, colliding keys)
    except Exception as exc:  # noqa: BLE001 - any mutagen or OS failure on the candidate fails this file
        raise _write_failure(path, exc) from None
    read_back, mismatches = verify_tags(path, plan, target=target, wav_tags=wav_tags, id3_version=id3_version)
    staged: dict[str, bytes] = {}
    if not mismatches and plan.sidecar_requests:
        assert audio_name is not None
        staged = dict(stage_members(plan, audio_name=audio_name, clears=clears, tool_version=tool_version))
    diagnostics = () if not mismatches else (Diagnostic(
        "verification_tags", "tag read-back does not match the plan", STAGE_VERIFY, ",".join(mismatches)),)
    result = TagWriteResult(path, plan.plan_revision, read_back, not mismatches, tuple(mismatches),
                            tuple(staged), diagnostics)
    return TagWrite(result, FrozenMap(staged))


@dataclass(frozen=True)
class InspectionTags:
    """MD27 result; failure and warning codes remain distinct in ``diagnostics``."""

    mismatches: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def matched(self) -> bool:
        return not self.mismatches


def check_inspection_tags(tags: SourceTags, references: Mapping[str, ManifestCell] | None = None, *,
                          artwork_reference: ArtworkReference | None = None) -> InspectionTags:
    """Check mark presence and explicit cells only, without I/O (MD27/MD37).

    Callers layer parsed manifest defaults, row cells and explicitly supplied CLI
    fields, in that order. Config, inferred and source values are not references.
    Parsing/matching errors remain the manifest module's responsibility.

    Non-null artwork cells must hold an absolute path, with the matching validated
    ``read_artwork_reference`` result supplied separately. The caller registers its
    protected read and rechecks before use; this checker never opens even that file.
    """
    mark = reader.read_processing_mark(tags)
    mismatches: list[str] = []
    causes = []
    if mark.validity == "absent":
        mismatches.append("extra:NORMA")
        causes.append("processing mark missing")
    extras = {name.casefold(): tag for name, tag in tags.extras.items()}
    for name, cell in (references or {}).items():
        canonical = name in model.CANONICAL_FIELDS
        if not canonical and model.extra_name_problem(name) is not None:
            raise ValueError(f"invalid inspection reference: {name!r}")
        key = ("field:" if canonical else "extra:") + name
        have = tags.fields.get(name) if canonical else extras.get(name.casefold())
        want = cell.value
        if name == "artwork":
            if cell.operation == "clear":
                matches = not tags.artwork
            else:
                if (not isinstance(want, Path) or not want.is_absolute() or artwork_reference is None
                        or want != artwork_reference.stamp.path):
                    raise ValueError("artwork: supply the validated reference for this absolute path")
                fronts = [tag.value for tag in tags.artwork
                          if isinstance(tag.value, Artwork) and tag.value.picture_type == 3]
                matches = len(fronts) == 1 and fronts[0].data == artwork_reference.data
        elif cell.operation == "clear":
            matches = have is None
        else:
            if name in model.TEXT_FIELDS or not canonical:
                if not isinstance(want, str):
                    raise ValueError(f"{name}: expected text")
                want = model.canonical_text(want, name if canonical else model.EXTRA)
            elif name == "date":
                if not isinstance(want, str):
                    raise ValueError("date: expected a canonical date")
                want = model.canonical_date(want)
                # The winning read value points to its actual native occurrences;
                # rendering defaults are not evidence of the inspected version.
                versions = {item.native_version for item in tags.native_inventory
                            if have is not None and item.identity in have.locator.split(",")}
                if have is not None and have.system == "id3v2" and versions == {"2.3"} and len(want) == 7:
                    want = want[:4]
            matches = have is not None and model.values_equal(want, have.value)
        if not matches:
            mismatches.append(key)
            causes.append(key + " differs from explicit reference")
    diagnostics = list(mark.diagnostics)
    if mismatches:
        diagnostics.insert(0, Diagnostic("verification_tags", "; ".join(causes), STAGE_VERIFY,
                                         ",".join(mismatches)))
    return InspectionTags(tuple(mismatches), tuple(diagnostics))


# --- Read-back (MD22, MD39) ------------------------------------------------------------------------


def verify_tags(candidate_path: Path, plan: MetadataPlan, *, target: str, wav_tags: str = "info+id3",
                id3_version: str = "2.3") -> tuple[SourceTags, tuple[str, ...]]:
    """Read every store of the candidate back and compare it with the plan (MD22, MD39).

    Returns the read-back inventory and the mismatch keys: native identities (expected or found),
    ``field:``/``extra:`` keys, ``field:artwork``, ``item:`` non-tag items and ``store:<store>`` for
    structure. An empty tuple means the tags match.
    """
    path = Path(candidate_path)
    stores = _check_plan(plan, target, wav_tags, id3_version)
    read_back = reader.read_source_tags(path, TARGET_CONTAINER[target])
    mismatches: list[str] = []
    mismatches += _native_mismatches(plan.expected_native, read_back.native_inventory)
    mismatches += [item.identity for item in read_back.non_tag_items]
    mismatches += _canonical_mismatches(plan, read_back)
    if plan.mark_state == "final" and reader.read_processing_mark(read_back).validity != "valid":
        mismatches.append("extra:" + model.MARK_NAME)
    mismatches += [f"store:{store}" for store in read_back.stores if store not in stores]
    try:
        mismatches += _structure_mismatches(path, target, wav_tags, id3_version)
    except OSError as exc:
        raise reader._failure(exc) from None  # noqa: SLF001 - the reader's own access mapping
    return read_back, tuple(dict.fromkeys(mismatches))


def _groups(items: Sequence[RawNativeItem]) -> dict[tuple[str, str], list[RawNativeItem]]:
    groups: dict[tuple[str, str], list[RawNativeItem]] = {}
    for item in items:
        groups.setdefault((item.store, item.key), []).append(item)
    return groups


def _native_mismatches(expected: Sequence[RawNativeItem], found: Sequence[RawNativeItem]) -> list[str]:
    """MD22 (1): every item with type, version, payload and duplicate order; no allow-list."""
    want, have = _groups(expected), _groups(found)
    out = []
    for key in list(want) + [key for key in have if key not in want]:
        left, right = want.get(key, []), have.get(key, [])
        if [_signature(item) for item in left] != [_signature(item) for item in right]:
            out += [item.identity for item in left] + [item.identity for item in right]
    return out


def _canonical_mismatches(plan: MetadataPlan, read_back: SourceTags) -> list[str]:
    """MD22 (2) and (3): canonical projection against ``effective``; pictures by bytes and type."""
    out = []
    for name in list(plan.effective) + [name for name in read_back.fields if name not in plan.effective]:
        want, have = plan.effective.get(name), read_back.fields.get(name)
        if want is None or have is None or not model.values_equal(want.value, have.value):
            out.append(f"field:{name}")
    wanted = {name.casefold(): value.value for name, value in plan.effective_extras.items()}
    found = {name.casefold(): value.value for name, value in read_back.extras.items()}
    names = {name.casefold(): name for name in list(plan.effective_extras) + list(read_back.extras)}
    for folded, name in names.items():
        if folded not in wanted or folded not in found or not model.values_equal(wanted[folded], found[folded]):
            out.append(f"extra:{name}")
    pictures = Counter((picture.data, picture.picture_type) for picture in plan.effective_artwork)
    read_pictures = Counter((tag.value.data, tag.value.picture_type) for tag in read_back.artwork
                            if isinstance(tag.value, Artwork))
    if pictures != read_pictures:
        out.append("field:artwork")
    return out


def _structure_mismatches(path: Path, target: str, wav_tags: str, id3_version: str) -> list[str]:
    """MD39's structural checks; residue is never an item, and nothing else may hide in it."""
    major = int(id3_version[-1])
    container = TARGET_CONTAINER[target]
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if container == "mp3":
            return _mp3_structure(handle, major)
        if container in ("wav", "aiff"):
            return _iff_structure(handle, size, container, wav_tags, major)
        if container == "flac":
            return _flac_structure(handle, size)
        return _mp4_structure(handle, size)


def _mp3_structure(handle: Any, major: int) -> list[str]:
    head = reader._read_at(handle, 0, 10)  # noqa: SLF001 - shared stdlib walk helpers
    tag_size = reader._id3_tag_size(head)  # noqa: SLF001
    if tag_size is None or head[3] != major:
        return ["store:id3v2"]
    after = reader._read_at(handle, tag_size, 10)  # noqa: SLF001
    if reader._id3_tag_size(after) is not None or len(after) < 2 or after[0] != 0xFF or after[1] & 0xE0 != 0xE0:  # noqa: SLF001
        return ["store:id3v2"]
    return []


def _iff_structure(handle: Any, size: int, container: str, wav_tags: str, major: int) -> list[str]:
    order = "little" if container == "wav" else "big"
    sizes: dict[bytes, int] | None = {} if container == "wav" else None
    id3_chunks, info_lists = [], 0
    for chunk_id, offset, chunk_size in reader._chunks(handle, 12, size, order, sizes=sizes):  # noqa: SLF001
        if chunk_id in (b"ID3 ", b"id3 "):
            id3_chunks.append(reader._read_at(handle, offset, min(chunk_size, 10)))  # noqa: SLF001
        elif chunk_id == b"LIST" and reader._read_at(handle, offset, 4) == b"INFO":  # noqa: SLF001
            info_lists += 1
    wants_id3 = container == "aiff" or wav_tags == "info+id3"
    out = []
    if len(id3_chunks) != (1 if wants_id3 else 0) or any(
            reader._id3_tag_size(head) is None or head[3] != major for head in id3_chunks):  # noqa: SLF001
        out.append("store:id3v2")
    if info_lists > (0 if container == "aiff" or wav_tags == "none" else 1):
        out.append("store:riff-info")
    return out


def _flac_structure(handle: Any, size: int) -> list[str]:
    if reader._read_at(handle, 0, 4) != b"fLaC":  # noqa: SLF001
        return ["store:vorbis"]
    offset, comments = 4, 0
    while offset + 4 <= size:
        header = reader._read_at(handle, offset, 4)  # noqa: SLF001
        comments += (header[0] & 0x7F) == 4
        offset += 4 + int.from_bytes(header[1:4], "big")
        if header[0] & 0x80:
            break
    return [] if comments == 1 else ["store:vorbis"]


def _mp4_structure(handle: Any, size: int) -> list[str]:
    moov = next(((offset + header, offset + atom_size) for name, offset, header, atom_size
                 in reader._atoms(handle, 0, size) if name == b"moov"), None)  # noqa: SLF001
    if moov is None:
        return ["store:mp4"]
    udtas = [(offset + header, offset + atom_size) for name, offset, header, atom_size
             in reader._atoms(handle, *moov) if name == b"udta"]  # noqa: SLF001
    if len(udtas) != 1:
        return ["store:mp4"]
    children = list(reader._atoms(handle, *udtas[0]))  # noqa: SLF001
    if [name for name, *_ in children] != [b"meta"]:
        return ["store:mp4"]
    _, offset, header, atom_size = children[0]
    ilsts = [name for name, *_ in reader._atoms(handle, offset + header + 4, offset + atom_size) if name == b"ilst"]  # noqa: SLF001
    return [] if len(ilsts) == 1 else ["store:mp4"]


# --- Sidecar members (MD25, MD36, MD41) ------------------------------------------------------------


def _payload_record(payload: bytes | None, size: int, digest: str) -> dict[str, Any]:
    record: dict[str, Any] = {"payload_size": size, "payload_sha256": digest}
    if payload is None or size > PAYLOAD_LIMIT:
        record["payload_omitted"] = "over_limit"
    else:
        record["payload_b64"] = base64.b64encode(payload).decode("ascii")
    return record


def _native_record(item: RawNativeItem) -> dict[str, Any]:
    record = {key: getattr(item, key) for key in ("identity", "store", "key", "native_type", "native_version",
                                                  "payload_encoding")}
    record.update(_payload_record(item.payload, len(item.payload), hashlib.sha256(item.payload).hexdigest()))
    return record


def _nontag_record(item: NonTagItem) -> dict[str, Any]:
    record = {key: getattr(item, key) for key in ("identity", "kind", "locator", "detail")}
    record.update(_payload_record(item.payload, item.payload_size, item.payload_sha256))
    return record


def _evidence_value(value: Any, plan: MetadataPlan, exports: Mapping[bytes, str]) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Artwork):
        return {"kind": "picture", "file": exports.get(value.data), "size": len(value.data),
                "sha256": hashlib.sha256(value.data).hexdigest()}
    if isinstance(value, InventoryRef):
        index = {item.identity: item for item in getattr(plan, value.scope)}
        items = [_nontag_record(index[identity]) if value.scope == "non_tag_items" else _native_record(index[identity])
                 for identity in value.identities]
        return {"kind": "inventory_ref", "scope": value.scope, "identities": list(value.identities), "items": items}
    raise TypeError(f"not a roll-up value: {type(value).__name__}")


def _manifest_value(name: str, value: Any) -> Any:
    """A valid schema-v1 manifest value, or ``None`` to omit it (MD41: never coerced)."""
    if name in model.TEXT_FIELDS or name == model.EXTRA:
        field = name if name in model.TEXT_FIELDS else model.EXTRA
        try:
            return value if isinstance(value, str) and value and model.canonical_text(value, field) == value else None
        except model.InvalidValue:
            return None
    if name == "date":
        try:
            return model.project_date(value) if isinstance(value, str) else None
        except model.InvalidValue:
            return None
    if name in model.INTEGER_FIELDS:
        return value if type(value) is int and value > 0 else None
    if name == "compilation":
        return value if type(value) is bool else None
    return None


def _manifest(plan: MetadataPlan, audio_name: str, clears: Sequence[str], cover_file: str | None) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in model.CANONICAL_FIELDS:
        if name == "artwork":
            continue
        if f"field:{name}" in clears:
            fields[name] = None
        elif name in plan.resolved:
            value = _manifest_value(name, plan.resolved[name].value)
            if value is not None:
                fields[name] = value
    if cover_file is not None:
        fields["artwork"] = cover_file
    elif "field:artwork" in clears:
        fields["artwork"] = None
    cleared = {}
    for key in clears:
        if key.startswith("extra:") and model.extra_name_problem(key[6:]) is None:
            cleared.setdefault(key[6:].casefold(), key[6:])
    for name in cleared.values():
        fields[name] = None
    for name, value in plan.extras.items():
        if model.extra_name_problem(name) is None and name.casefold() not in cleared:
            extra = _manifest_value(model.EXTRA, value.value)
            if extra is not None:
                fields[name] = extra
    return {"version": 1, "files": {audio_name: fields}}


def _picture_for(plan: MetadataPlan, key: str | None) -> Artwork:
    for status in plan.statuses:
        if status.key == key:
            for value in (status.requested, status.effective):
                if isinstance(value, Artwork):
                    return value
    raise ValueError(f"no picture for sidecar member key {key!r}")


def stage_members(plan: MetadataPlan, *, audio_name: str, clears: Sequence[str] = (),
                  tool_version: str | None = None) -> FrozenMap:
    """Bytes of every requested sidecar member, keyed by ``member_id`` (MD25, MD36, MD41). Writes nothing."""
    if not audio_name or "/" in audio_name or audio_name in (".", ".."):
        raise ValueError("audio_name must be a plain file name")
    if tool_version is None:
        tool_version = __version__
    members: dict[str, bytes] = {}
    exports: dict[bytes, str] = {}
    cover_file = None
    for request in plan.sidecar_requests:
        if request.purpose == "tags_sidecar":
            continue
        picture = _picture_for(plan, request.inventory_key)
        members[request.member_id] = picture.data
        exports.setdefault(picture.data, audio_name + request.suffix)
        if request.purpose == "cover_export":
            cover_file = audio_name + request.suffix
    for request in plan.sidecar_requests:
        if request.purpose != "tags_sidecar":
            continue
        document = {
            "sidecar_version": SIDECAR_VERSION,
            "tool": {"name": "norma", "version": tool_version},
            "audio": audio_name,
            "plan_revision": plan.plan_revision,
            "manifest": _manifest(plan, audio_name, clears, cover_file),
            "losses": [_loss_record(status, plan, exports) for status in plan.statuses
                       if status.status == "dropped_format_limit"],
            "native_losses": [_native_loss(row, plan) for row in plan.native_statuses
                              if row.status == "dropped_format_limit"],
            "items": [_nontag_record(item) for item in plan.non_tag_items
                      if any(status.key == item.identity and status.status == "dropped_format_limit"
                             for status in plan.statuses)],
        }
        members[request.member_id] = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    return FrozenMap({request.member_id: members[request.member_id] for request in plan.sidecar_requests})


def _loss_record(status: FieldStatus, plan: MetadataPlan, exports: Mapping[bytes, str]) -> dict[str, Any]:
    return {"key": status.key, "loss_kind": status.loss_kind,
            "requested": _evidence_value(status.requested, plan, exports),
            "effective": _evidence_value(status.effective, plan, exports), "detail": status.detail,
            "source_native_ids": list(status.source_native_ids),
            "expected_native_ids": list(status.expected_native_ids)}


def _native_loss(row: Any, plan: MetadataPlan) -> dict[str, Any]:
    index = {item.identity: item for item in plan.source_native}
    record = {"key": row.key, "loss_kind": row.loss_kind, "detail": row.detail,
              "expected_ids": list(row.expected_ids)}
    record["items"] = [_native_record(index[identity]) for identity in row.source_ids]
    return record
