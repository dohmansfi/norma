"""Tag-store readers: one source file into a C-06 ``SourceTags`` (WS-06, I/O edge).

Behaviour: the source-reading rules (MD3–MD7, MD32, MD37) and
the format tables. Every store MD3 lists for the container is read by
its own reader into separate native occurrences; nothing is read through
ffprobe's flattened tags. Reading never writes to the source.

- ``id3v2``: the tag is located by a stdlib walk (MP3 offset 0, a FLAC's leading
  header, the AIFF/WAV ``ID3 `` chunk), split into stored frames, and each frame
  is parsed alone with mutagen (``translate=False``) to give MD32 ``dn-json-1``
  fields. Fixed-point, lossy-parse and unparseable frames keep the stored frame
  as ``dn-raw-1``.
- ``mp4``: mutagen parses each stored ``ilst`` atom in isolation; a stdlib atom
  walk supplies stored boundaries/order, raw evidence and the chapter sources.
- ``vorbis``, ``riff-info``, ``aiff-text``, ``apev2``, ``id3v1``: stdlib walks of
  the stored bytes, so invalid text is never replaced on reading.

``mutagen`` is the only third-party import (ARCHITECTURE § 3). No mutagen
object crosses the C-06 boundary.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import io
import json
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from mutagen.id3 import ID3, ID3TimeStamp, TimeStampTextFrame
from mutagen.id3 import TCON as _TCON
from mutagen.mp4 import Atoms, MP4Cover, MP4FreeForm, MP4Tags

from norma.errors import FileFailure
from norma.metadata import model
from norma.models import (
    Artwork, Conflict, NonTagItem, ProcessingMark, RawNativeItem, SourceTags, TagValue,
)

RAW = "dn-raw-1"
JSON = "dn-json-1"
STAGE = "tags_read"
PAYLOAD_LIMIT = 1_048_576
"""MD7/MD25 evidence limit: a larger non-tag payload is kept as size and SHA-256 only."""

NOT_CARRIED = "not_carried_v1"

_FORCED_RAW_FRAMES = frozenset({"RVA2", "EQU2", "RVAD"})
"""MD32's closed list of fixed-point ID3 frames, always kept as the stored frame."""

_WAV_STRUCTURE = frozenset({b"fmt ", b"data", b"fact", b"ds64"})
_AIFF_STRUCTURE = frozenset({b"COMM", b"SSND", b"FVER"})
_AIFF_TEXT = frozenset({b"NAME", b"AUTH", b"(c) ", b"ANNO"})
_ID3_CHUNKS = frozenset({b"ID3 ", b"id3 "})
_CHAPTER_SOURCES = ("mp4-track", "mp4-chpl", "id3", "vorbis", "flac-cuesheet")


class _Unrepresentable(Exception):
    """A parsed value MD32's ``dn-json-1`` cannot hold losslessly."""


@dataclass
class _Item:
    store: str
    key: str
    native_key: str
    native_type: str
    native_version: str | None
    encoding: str
    payload: bytes
    locator: str
    value: Any = None
    """Projection material: a tuple of decoded strings, an MP4 pair, a bool or an Artwork; None if unusable."""
    chapter: dict[str, Any] | None = None
    identity: str = ""
    foreign_text: tuple[tuple[int, ...], str] | None = None
    """CR-043: an MP4 text occurrence stored with a non-UTF-8 data type: (data types, text read under them)."""


@dataclass
class _Read:
    items: list[_Item] = field(default_factory=list)
    stores: list[str] = field(default_factory=list)
    nontag: list[tuple[str, str, bytes | None, int, str]] = field(default_factory=list)
    """``(kind:locator id, locator, payload, size, sha256)`` for chunk items, in stored order."""
    chapters: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def present(self, store: str) -> None:
        if store not in self.stores:
            self.stores.append(store)


# --- Encodings (MD32) ---------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    """MD32 ``dn-json-1`` serialisation: sorted keys, no whitespace, integers only, UTF-8."""
    _check_integers(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _check_integers(value: Any) -> None:
    if isinstance(value, float):
        raise _Unrepresentable("non-integer number")
    if isinstance(value, dict):
        for item in value.values():
            _check_integers(item)
    elif isinstance(value, list):
        for item in value:
            _check_integers(item)


def _b64(data: bytes) -> dict[str, str]:
    return {"b64": base64.b64encode(data).decode("ascii")}


# --- File access ----------------------------------------------------------------------


def _failure(exc: OSError) -> FileFailure:
    if exc.errno in (errno.EACCES, errno.EPERM):
        return FileFailure("permission_denied", STAGE, str(exc))
    return FileFailure("source_changed", STAGE, f"source could not be read: {exc}")


def _read_at(handle: BinaryIO, offset: int, size: int) -> bytes:
    handle.seek(offset)
    return handle.read(size)


def _hash_range(handle: BinaryIO, offset: int, size: int) -> tuple[bytes | None, int, str]:
    """Payload bytes (or ``None`` over the limit), the stored size and SHA-256."""
    handle.seek(offset)
    if size <= PAYLOAD_LIMIT:
        data = handle.read(size)
        return data, len(data), hashlib.sha256(data).hexdigest()
    digest = hashlib.sha256()
    remaining, total = size, 0
    while remaining:
        block = handle.read(min(remaining, 1 << 20))
        if not block:
            break
        digest.update(block)
        total += len(block)
        remaining -= len(block)
    return None, total, digest.hexdigest()


# --- ID3v2 ----------------------------------------------------------------------------


def _syncsafe(data: bytes) -> int:
    return (data[0] << 21) | (data[1] << 14) | (data[2] << 7) | data[3]


def _id3_tag_size(header: bytes) -> int | None:
    """Total size of an ID3v2 tag from its 10-byte header, footer included; ``None`` if not a header."""
    if len(header) < 10 or header[:3] != b"ID3" or header[3] not in (2, 3, 4) or header[4] == 0xFF:
        return None
    if any(byte & 0x80 for byte in header[6:10]):
        return None
    footer = 10 if header[3] == 4 and header[5] & 0x10 else 0
    return 10 + _syncsafe(header[6:10]) + footer


@dataclass(frozen=True)
class _RawFrame:
    frame_id: str
    flags: int
    body: bytes
    stored: bytes
    offset: int


def _id3_raw_frames(tag: bytes) -> tuple[int, list[_RawFrame]]:
    """Stored frames of one tag after reversing whole-tag unsynchronisation (MD32)."""
    major, flags = tag[3], tag[5]
    size = _syncsafe(tag[6:10])
    body = tag[10:10 + size]
    if major in (2, 3) and flags & 0x80:
        body = body.replace(b"\xff\x00", b"\xff")
    position = 0
    if flags & 0x40:
        if major == 3 and len(body) >= 4:
            position = 4 + struct.unpack(">I", body[:4])[0]
        elif major == 4 and len(body) >= 4:
            position = _syncsafe(body[:4])
        else:
            return major, []  # ID3v2.2 compression: the tag cannot be parsed
    header_size, id_size = (6, 3) if major == 2 else (10, 4)
    frames = []
    while position + header_size <= len(body):
        raw_id = body[position:position + id_size]
        if not raw_id or raw_id[0] == 0 or not all(48 <= c <= 57 or 65 <= c <= 90 for c in raw_id):
            break
        if major == 2:
            frame_size = int.from_bytes(body[position + 3:position + 6], "big")
            frame_flags = 0
        else:
            size_bytes = body[position + 4:position + 8]
            if major == 4 and not any(byte & 0x80 for byte in size_bytes):
                frame_size = _syncsafe(size_bytes)
            else:
                frame_size = struct.unpack(">I", size_bytes)[0]
            frame_flags = struct.unpack(">H", body[position + 8:position + 10])[0]
        end = position + header_size + frame_size
        if end > len(body):
            break
        frames.append(_RawFrame(raw_id.decode("ascii"), frame_flags, body[position + header_size:end],
                                body[position:end], position))
        position = end
    return major, frames


def _single_frame_tag(major: int, stored: bytes) -> bytes:
    size = len(stored)
    return (b"ID3" + bytes([major, 0, 0])
            + bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F]) + stored)


def _parse_frame(major: int, raw: _RawFrame) -> Any | None:
    """The frame parsed alone by mutagen, without version translation; ``None`` if it cannot be."""
    try:
        tags = ID3(io.BytesIO(_single_frame_tag(major, raw.stored)), translate=False)
    except Exception:  # noqa: BLE001 - any parser failure keeps the stored frame
        return None
    frames = list(tags.values())
    return frames[0] if len(frames) == 1 and not tags.unknown_frames else None


def _decode_id3_strings(body: bytes) -> list[str] | None:
    """The stored strings of a plain text-frame body, or ``None`` if they do not decode."""
    if not body:
        return None
    encoding, data = body[0], body[1:]
    try:
        if encoding in (0, 3):
            parts = data.split(b"\x00")
            codec = "latin-1" if encoding == 0 else "utf-8"
            strings = [part.decode(codec) for part in parts]
        elif encoding in (1, 2):
            pieces, start = [], 0
            for index in range(0, len(data) - 1, 2):
                if data[index:index + 2] == b"\x00\x00":
                    pieces.append(data[start:index])
                    start = index + 2
            pieces.append(data[start:])
            codec = "utf-16" if encoding == 1 else "utf-16-be"
            strings = [piece.decode(codec) if piece else "" for piece in pieces]
        else:
            return None
    except UnicodeDecodeError:
        return None
    if len(strings) > 1 and strings[-1] == "":
        strings.pop()
    return strings


def _id3_value(value: Any) -> Any:
    if isinstance(value, bool):
        raise _Unrepresentable("boolean")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return _b64(bytes(value))
    if isinstance(value, ID3TimeStamp):
        return {"ts": value.text}
    if isinstance(value, (list, tuple)):
        return [_id3_value(item) for item in value]
    if hasattr(value, "values") and hasattr(value, "getall"):  # embedded frames (CHAP, CTOC)
        return {"frames": [_id3_frame_object(frame, type(frame).__name__) for frame in value.values()]}
    raise _Unrepresentable(type(value).__name__)


def _id3_frame_object(frame: Any, frame_id: str, strings: list[str] | None = None) -> dict[str, Any]:
    fields = []
    for spec in list(frame._framespec) + list(frame._optionalspec):  # noqa: SLF001 - MD32 field order
        if not hasattr(frame, spec.name):
            continue
        value = getattr(frame, spec.name)
        if strings is not None and spec.name == "text":
            fields.append([spec.name, [{"ts": text} for text in strings]])
        else:
            fields.append([spec.name, _id3_value(value)])
    return {"frame": frame_id, "fields": fields}


def _id3_native_key(frame_id: str, frame: Any | None) -> str:
    """MD4 refined native key for repeatable frames."""
    if frame is None:
        return frame_id
    kind = type(frame).__name__
    if kind in ("TXXX", "WXXX", "APIC", "GEOB", "RVA2", "EQU2", "PIC", "GEO", "TXX", "WXX"):
        return f"{frame_id}:{getattr(frame, 'desc', '')}"
    if kind in ("COMM", "USLT", "COM", "ULT"):
        return f"{frame_id}:{getattr(frame, 'desc', '')}:{getattr(frame, 'lang', '')}"
    if kind == "PRIV":
        return f"{frame_id}:{frame.owner}"
    if kind in ("CHAP", "CTOC"):
        return f"{frame_id}:{frame.element_id}"
    return frame_id


_PIC_MIME = {"PNG": "image/png", "JPG": "image/jpeg"}


def _id3_projection(frame: Any, strings: list[str] | None) -> Any:
    kind = type(frame).__name__
    if kind in ("APIC", "PIC"):
        mime = getattr(frame, "mime", None) or _PIC_MIME.get(getattr(frame, "format", "").upper(), "")
        return Artwork(bytes(frame.data), mime or _sniff_mime(frame.data), None, None, int(frame.type),
                       str(frame.desc))
    if isinstance(frame, _TCON):
        return tuple(frame.genres)
    if strings is not None:
        return tuple(strings)
    text = getattr(frame, "text", None)
    if isinstance(text, list) and all(isinstance(item, str) for item in text):
        return tuple(text)
    return None


def _id3_chapter(frame: Any, identity_holder: _Item) -> dict[str, Any] | None:
    if type(frame).__name__ != "CHAP":
        return None
    title = None
    for sub in frame.sub_frames.values():
        if type(sub).__name__ == "TIT2":
            title = model.MULTI_VALUE_SEPARATOR.join(str(text) for text in sub.text)
            break
    end = None if frame.end_time == 0xFFFFFFFF else [int(frame.end_time), 1000]
    return {"start": [int(frame.start_time), 1000], "end": end, "title": title, "native": identity_holder}


def _read_id3_tag(tag: bytes, offset: int, read: _Read, where: str) -> None:
    """Inventory one ID3v2 tag's frames as stored, in stored order (MD3, MD32)."""
    read.present("id3v2")
    major, frames = _id3_raw_frames(tag)
    version = f"2.{major}"
    for raw in frames:
        frame = _parse_frame(major, raw)
        native_key = _id3_native_key(raw.frame_id, frame)
        locator = f"{where}+{10 + raw.offset}"
        item = _Item("id3v2", native_key, native_key, raw.frame_id, version, RAW, raw.stored, locator)
        if frame is not None:
            strings = None
            if isinstance(frame, TimeStampTextFrame):
                strings = _decode_id3_strings(raw.body) if raw.flags == 0 else None
                if strings is not None and len(strings) != len(frame.text):
                    strings = None
            try:
                if raw.frame_id in _FORCED_RAW_FRAMES:
                    raise _Unrepresentable("fixed-point frame")
                if isinstance(frame, TimeStampTextFrame) and strings is None:
                    raise _Unrepresentable("timestamp text not kept by the parse")
                item.payload = canonical_json(_id3_frame_object(frame, raw.frame_id, strings))
                item.encoding = JSON
            except (_Unrepresentable, UnicodeEncodeError, ValueError):
                item.payload, item.encoding = raw.stored, RAW
            item.value = _id3_projection(frame, strings)
            chapter = _id3_chapter(frame, item)
            if chapter is not None:
                item.chapter = chapter
                read.chapters.setdefault("id3", []).append(chapter)
        read.items.append(item)


# --- Vorbis comments and FLAC -----------------------------------------------------------


def _read_vorbis_block(block: bytes, offset: int, read: _Read) -> None:
    """Walk a VORBIS_COMMENT block; values stay the stored bytes (MD32)."""
    read.present("vorbis")
    if len(block) < 8:
        return
    vendor = struct.unpack("<I", block[:4])[0]
    position = 4 + vendor
    if position + 4 > len(block):
        return
    count = struct.unpack("<I", block[position:position + 4])[0]
    position += 4
    for _ in range(count):
        if position + 4 > len(block):
            break
        length = struct.unpack("<I", block[position:position + 4])[0]
        comment = block[position + 4:position + 4 + length]
        locator = f"VORBIS_COMMENT@{offset}+{position}"
        position += 4 + length
        name, separator, value = comment.partition(b"=")
        key = name.decode("ascii", "replace") if separator else comment.decode("ascii", "replace")
        payload = value if separator else b""
        try:
            text: Any = (payload.decode("utf-8"),) if separator else None
        except UnicodeDecodeError:
            text = None
        read.items.append(_Item("vorbis", key, key.upper(), "VORBIS_COMMENT", None, RAW, payload, locator, text))


def parse_flac_picture(body: bytes) -> Artwork | None:
    """A METADATA_BLOCK_PICTURE body as an ``Artwork``, or ``None`` if malformed."""
    try:
        position = 0
        picture_type, mime_length = struct.unpack(">II", body[0:8])
        position = 8
        mime = body[position:position + mime_length].decode("ascii")
        position += mime_length
        description_length = struct.unpack(">I", body[position:position + 4])[0]
        position += 4
        description = body[position:position + description_length].decode("utf-8")
        position += description_length
        width, height, _depth, _colours, data_length = struct.unpack(">IIIII", body[position:position + 20])
        position += 20
        data = body[position:position + data_length]
        if len(data) != data_length:
            return None
    except (struct.error, UnicodeDecodeError):
        return None
    return Artwork(data, mime or _sniff_mime(data), width or None, height or None, picture_type, description)


def _read_cuesheet(block: bytes, sample_rate: int | None) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    if len(block) < 396 or not sample_rate:
        return chapters
    is_cd = bool(block[136] & 0x80)
    count = block[395]
    position = 396
    for _ in range(count):
        if position + 36 > len(block):
            break
        offset = struct.unpack(">Q", block[position:position + 8])[0]
        number = block[position + 8]
        indices = block[position + 35]
        position += 36 + 12 * indices
        if number == (170 if is_cd else 255):
            continue
        chapters.append({"start": [offset, sample_rate], "end": None, "title": None, "native": None})
    return chapters


def _read_flac(handle: BinaryIO, size: int, read: _Read) -> None:
    offset = 0
    header = _read_at(handle, 0, 10)
    tag_size = _id3_tag_size(header)
    if tag_size is not None:
        _read_id3_tag(_read_at(handle, 0, tag_size), 0, read, "ID3@0")
        offset = tag_size
    if _read_at(handle, offset, 4) != b"fLaC":
        return
    offset += 4
    sample_rate: int | None = None
    while offset + 4 <= size:
        block_header = _read_at(handle, offset, 4)
        last, kind = bool(block_header[0] & 0x80), block_header[0] & 0x7F
        length = int.from_bytes(block_header[1:4], "big")
        data_offset = offset + 4
        if kind == 0:
            info = _read_at(handle, data_offset, min(length, 34))
            if len(info) >= 13:
                sample_rate = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
        elif kind == 4:
            _read_vorbis_block(_read_at(handle, data_offset, length), data_offset, read)
        elif kind == 6:
            body = _read_at(handle, data_offset, length)
            read.present("vorbis")
            read.items.append(_Item("vorbis", "PICTURE", "PICTURE", "PICTURE", None, RAW, body,
                                    f"PICTURE@{data_offset}", parse_flac_picture(body)))
        elif kind == 2:
            # APPLICATION (MD7, IC-2 F-1): foreign-application data; the body starts with the 4-byte application ID.
            payload, stored_size, digest = _hash_range(handle, data_offset, length)
            read.nontag.append(("APPLICATION", f"APPLICATION@{offset}", payload, stored_size, digest))
        elif kind == 5:
            chapters = _read_cuesheet(_read_at(handle, data_offset, length), sample_rate)
            read.chapters["flac-cuesheet"] = chapters
        offset = data_offset + length
        if last:
            break


# --- RIFF and AIFF chunk walks ----------------------------------------------------------


def _chunks(handle: BinaryIO, start: int, end: int, order: str,
            sizes: dict[bytes, int] | None = None) -> Iterator[tuple[bytes, int, int]]:
    """``(chunk id, data offset, data size)`` for each chunk; stops at the first malformed one.

    ``sizes`` supplies RF64 64-bit sizes from ``ds64`` for chunks stored with 0xFFFFFFFF.
    """
    position = start
    fmt = "<I" if order == "little" else ">I"
    while position + 8 <= end:
        header = _read_at(handle, position, 8)
        if len(header) < 8:
            return
        chunk_id, chunk_size = header[:4], struct.unpack(fmt, header[4:])[0]
        if chunk_size == 0xFFFFFFFF and sizes is not None and chunk_id in sizes:
            chunk_size = sizes[chunk_id]
        if chunk_id == b"ds64" and sizes is not None and chunk_size >= 16:
            ds64 = _read_at(handle, position + 8, 16)
            if len(ds64) == 16:
                sizes[b"data"] = struct.unpack("<Q", ds64[8:16])[0]
        yield chunk_id, position + 8, chunk_size
        position += 8 + chunk_size + (chunk_size & 1)


def _nontag_chunk(handle: BinaryIO, read: _Read, chunk_id: bytes, offset: int, size: int) -> None:
    name = chunk_id.decode("latin-1")
    payload, stored_size, digest = _hash_range(handle, offset, size)
    read.nontag.append((name, f"{name}@{offset - 8}", payload, stored_size, digest))


def _text_item(store: str, chunk_id: bytes, data: bytes, locator: str) -> _Item:
    key = chunk_id.decode("latin-1")
    try:
        text: Any = (data.rstrip(b"\x00").decode("utf-8"),)
    except UnicodeDecodeError:
        text = None
    return _Item(store, key, key, key, None, RAW, data, locator, text)


def _id3_chunk(handle: BinaryIO, read: _Read, chunk_id: bytes, offset: int, size: int) -> None:
    data = _read_at(handle, offset, size)
    tag_size = _id3_tag_size(data[:10])
    if tag_size is None:
        _nontag_chunk(handle, read, chunk_id, offset, size)
        return
    _read_id3_tag(data[:tag_size], offset, read, f"{chunk_id.decode('latin-1')}@{offset - 8}")


def _read_wav(handle: BinaryIO, size: int, read: _Read) -> None:
    riff = _read_at(handle, 0, 12)
    if len(riff) < 12 or riff[:4] not in (b"RIFF", b"RF64", b"BW64") or riff[8:12] != b"WAVE":
        return
    for chunk_id, offset, chunk_size in _chunks(handle, 12, size, "little", sizes={}):
        if chunk_id in _WAV_STRUCTURE:
            continue
        if chunk_id in _ID3_CHUNKS:
            _id3_chunk(handle, read, chunk_id, offset, chunk_size)
        elif chunk_id == b"LIST" and _read_at(handle, offset, 4) == b"INFO":
            read.present("riff-info")
            data = _read_at(handle, offset, chunk_size)
            position = 4
            while position + 8 <= len(data):
                item_id = data[position:position + 4]
                item_size = struct.unpack("<I", data[position + 4:position + 8])[0]
                body = data[position + 8:position + 8 + item_size]
                read.items.append(_text_item("riff-info", item_id, body, f"LIST/INFO@{offset - 8}+{position}"))
                position += 8 + item_size + (item_size & 1)
        else:
            _nontag_chunk(handle, read, chunk_id, offset, chunk_size)


def _read_aiff(handle: BinaryIO, size: int, read: _Read) -> None:
    form = _read_at(handle, 0, 12)
    if len(form) < 12 or form[:4] != b"FORM" or form[8:12] not in (b"AIFF", b"AIFC"):
        return
    for chunk_id, offset, chunk_size in _chunks(handle, 12, size, "big"):
        if chunk_id in _AIFF_STRUCTURE:
            continue
        if chunk_id in _ID3_CHUNKS:
            _id3_chunk(handle, read, chunk_id, offset, chunk_size)
            continue
        if chunk_id in _AIFF_TEXT:
            read.present("aiff-text")
            data = _read_at(handle, offset, chunk_size)
            read.items.append(_text_item("aiff-text", chunk_id, data, f"{chunk_id.decode('latin-1')}@{offset - 8}"))
            continue
        _nontag_chunk(handle, read, chunk_id, offset, chunk_size)


# --- MP3: ID3v2, APEv2, ID3v1 -----------------------------------------------------------

_ID3V1_LAYOUT = (("title", 3, 33), ("artist", 33, 63), ("album", 63, 93), ("year", 93, 97))


def _read_id3v1(tag: bytes, offset: int, read: _Read) -> None:
    read.present("id3v1")
    fields = list(_ID3V1_LAYOUT)
    track: int | None = None
    if tag[125] == 0 and tag[126] != 0:
        fields.append(("comment", 97, 125))
        track = tag[126]
    else:
        fields.append(("comment", 97, 127))
    for name, start, end in fields:
        data = tag[start:end]
        text = data.rstrip(b"\x00 ").decode("latin-1").rstrip("\x00 ")
        if text.strip():
            read.items.append(_Item("id3v1", name, name, name, None, RAW, data, f"TAG@{offset}+{start}", (text,)))
    if track:
        read.items.append(_Item("id3v1", "track", "track", "track", None, RAW, tag[126:127],
                                f"TAG@{offset}+126", (str(track),)))
    if tag[127] != 0xFF:
        genres = _TCON.GENRES
        name = genres[tag[127]] if tag[127] < len(genres) else None
        read.items.append(_Item("id3v1", "genre", "genre", "genre", None, RAW, tag[127:128],
                                f"TAG@{offset}+127", (name,) if name else None))


_APE_TYPES = ("text", "binary", "external", "binary")


def _read_apev2(handle: BinaryIO, end: int, read: _Read) -> int:
    """Read an APEv2 tag whose footer ends at ``end``; returns the tag's start offset or ``end``."""
    if end < 32:
        return end
    footer = _read_at(handle, end - 32, 32)
    if footer[:8] != b"APETAGEX":
        return end
    _version, tag_size, count, flags = struct.unpack("<IIII", footer[8:24])
    items_start = end - tag_size
    if items_start < 0:
        return end
    read.present("apev2")
    data = _read_at(handle, items_start, tag_size - 32)
    position = 0
    for _ in range(count):
        if position + 9 > len(data):
            break
        value_size, item_flags = struct.unpack("<II", data[position:position + 8])
        key_end = data.find(b"\x00", position + 8)
        if key_end < 0:
            break
        key = data[position + 8:key_end].decode("ascii", "replace")
        value = data[key_end + 1:key_end + 1 + value_size]
        locator = f"APETAGEX@{items_start}+{position}"
        position = key_end + 1 + value_size
        kind = _APE_TYPES[(item_flags >> 1) & 3]
        payload = canonical_json({"flags": item_flags, "value": _b64(value)})
        text: Any = None
        if kind == "text":
            try:
                text = tuple(value.decode("utf-8").split("\x00"))
            except UnicodeDecodeError:
                text = None
        read.items.append(_Item("apev2", key, key, kind, None, JSON, payload, locator, text))
    has_header = bool(flags & 0x80000000)
    return items_start - (32 if has_header else 0)


def _id3v1_at(handle: BinaryIO, offset: int) -> tuple[bytes, int] | None:
    if offset < 0:
        return None
    tail = _read_at(handle, offset, 128)
    return (tail, offset) if len(tail) == 128 and tail[:3] == b"TAG" else None


def _read_mp3(handle: BinaryIO, size: int, read: _Read) -> None:
    """ID3v2 at offset 0 (and any tag directly after it); APEv2 and ID3v1 at the end, in either order."""
    offset = 0
    while (tag_size := _id3_tag_size(_read_at(handle, offset, 10))) is not None and offset + tag_size <= size:
        _read_id3_tag(_read_at(handle, offset, tag_size), offset, read, f"ID3@{offset}")
        offset += tag_size
    ape_start = _read_apev2(handle, size, read)
    if ape_start != size:
        id3v1 = _id3v1_at(handle, ape_start - 128)
    else:
        id3v1 = _id3v1_at(handle, size - 128)
        if id3v1 is not None:
            _read_apev2(handle, size - 128, read)
    if id3v1 is not None:
        _read_id3v1(*id3v1, read)


# --- MP4 ------------------------------------------------------------------------------------


def _atoms(handle: BinaryIO, start: int, end: int) -> Iterator[tuple[bytes, int, int, int]]:
    """``(name, atom offset, header size, atom size)`` for each child atom."""
    position = start
    while position + 8 <= end:
        header = _read_at(handle, position, 16)
        if len(header) < 8:
            return
        size, name = struct.unpack(">I4s", header[:8])
        header_size = 8
        if size == 1:
            if len(header) < 16:
                return
            size, header_size = struct.unpack(">Q", header[8:16])[0], 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            return
        yield name, position, header_size, size
        position += size


def _child(handle: BinaryIO, start: int, end: int, name: bytes) -> tuple[int, int] | None:
    for child, offset, header, size in _atoms(handle, start, end):
        if child == name:
            return offset + header, offset + size
    return None


def _path(handle: BinaryIO, start: int, end: int, *names: bytes) -> tuple[int, int] | None:
    span: tuple[int, int] | None = (start, end)
    for name in names:
        assert span is not None
        span = _child(handle, span[0], span[1], name)
        if span is None:
            return None
        if name == b"meta":
            span = (span[0] + 4, span[1])  # full atom: version and flags
    return span


def _ilst_key(handle: BinaryIO, name: bytes, offset: int, header: int, size: int) -> str:
    if name != b"----":
        return name.decode("latin-1")
    mean = item_name = b""
    for child, child_offset, child_header, child_size in _atoms(handle, offset + header, offset + size):
        if child in (b"mean", b"name"):
            value = _read_at(handle, child_offset + child_header + 4, child_size - child_header - 4)
            if child == b"mean":
                mean = value
            else:
                item_name = value
    return "----:" + mean.decode("utf-8", "replace") + ":" + item_name.decode("utf-8", "replace")


def _data_types(handle: BinaryIO, offset: int, header: int, size: int) -> list[int]:
    types = []
    for child, child_offset, child_header, _ in _atoms(handle, offset + header, offset + size):
        if child == b"data":
            types.append(struct.unpack(">I", _read_at(handle, child_offset + child_header, 4))[0] & 0xFFFFFF)
    return types


_MP4_TEXT_FIELDS = frozenset(model.TEXT_FIELDS | {"date"})
_UTF16_TYPES = frozenset({2, 5})


def _foreign_text(handle: BinaryIO, key: str, offset: int, header: int, size: int) -> tuple[tuple[int, ...], str] | None:
    """CR-043: the data types and text of an MP4 text atom stored under a non-UTF-8 data type.

    Text atoms (title, artist, date …) must be type 1; iTunes freeforms (extras) type 0 or 1. Each
    data child is read under its declared type (2 and 5 UTF-16 BE, otherwise UTF-8), invalid bytes
    replaced: the text is diagnostic display for ``internal_conflicts`` only, never a canonical value,
    and the stored atom stays the occurrence's raw evidence. ``None`` for any other atom.
    """
    if key.startswith("----:"):
        if model.native_extra("mp4", key) is None:
            return None
        allowed: tuple[int, ...] = (0, 1)
    elif set(model.native_fields("mp4", key)) & _MP4_TEXT_FIELDS:
        allowed = (1,)
    else:
        return None
    types: list[int] = []
    texts: list[str] = []
    for child, child_offset, child_header, child_size in _atoms(handle, offset + header, offset + size):
        if child != b"data" or child_size < child_header + 8:
            continue
        kind = struct.unpack(">I", _read_at(handle, child_offset + child_header, 4))[0] & 0xFFFFFF
        value = _read_at(handle, child_offset + child_header + 8, child_size - child_header - 8)
        types.append(kind)
        texts.append(value.decode("utf-16-be" if kind in _UTF16_TYPES else "utf-8", "replace"))
    if not types or all(kind in allowed for kind in types):
        return None
    return tuple(types), model.join_text([model.normalise_text(text) for text in texts])


def _mp4_value(value: Any) -> Any:
    if isinstance(value, MP4Cover):
        return {"cover": base64.b64encode(bytes(value)).decode("ascii"), "imageformat": int(value.imageformat)}
    if isinstance(value, MP4FreeForm):
        return {"freeform": base64.b64encode(bytes(value)).decode("ascii"),
                "dataformat": int(value.dataformat), "version": int(value.version)}
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(part, int) for part in value):
        return {"pair": [int(value[0]), int(value[1])]}
    if isinstance(value, (bytes, bytearray)):
        return _b64(bytes(value))
    raise _Unrepresentable(type(value).__name__)


_COVER_MIME = {13: "image/jpeg", 14: "image/png", 27: "image/bmp"}


def _mp4_projection(key: str, values: list[Any]) -> Any:
    if key == "covr":
        cover = values[0]
        data = bytes(cover)
        return Artwork(data, _COVER_MIME.get(int(cover.imageformat), _sniff_mime(data)), None, None, 3, "")
    if all(isinstance(value, bool) for value in values):
        return values[0] if len(values) == 1 else None
    if all(isinstance(value, tuple) for value in values):
        return values[0]
    if all(isinstance(value, str) for value in values):
        return tuple(values)
    if all(isinstance(value, MP4FreeForm) for value in values):
        if any(int(value.dataformat) not in (0, 1) for value in values):
            return None
        try:
            return tuple(bytes(value).decode("utf-8") for value in values)
        except UnicodeDecodeError:
            return None
    return None


def _read_chpl(handle: BinaryIO, span: tuple[int, int]) -> list[dict[str, Any]]:
    data = _read_at(handle, span[0], span[1] - span[0])
    chapters: list[dict[str, Any]] = []
    if len(data) < 9:
        return chapters
    position = 9
    for _ in range(data[8]):
        if position + 9 > len(data):
            break
        start = struct.unpack(">Q", data[position:position + 8])[0]
        length = data[position + 8]
        raw_title = data[position + 9:position + 9 + length]
        position += 9 + length
        try:
            title: str | None = raw_title.decode("utf-8")
        except UnicodeDecodeError:
            title = None
        chapters.append({"start": [start, 10_000_000], "end": None, "title": title, "native": None})
    return chapters


def _full_atom_entries(handle: BinaryIO, span: tuple[int, int], width: int) -> list[tuple[int, ...]]:
    data = _read_at(handle, span[0], span[1] - span[0])
    if len(data) < 8:
        return []
    count = struct.unpack(">I", data[4:8])[0]
    entries = []
    for index in range(count):
        start = 8 + index * width
        if start + width > len(data):
            break
        entries.append(struct.unpack(">" + "I" * (width // 4), data[start:start + width]))
    return entries


def _chapter_track(handle: BinaryIO, moov: tuple[int, int]) -> list[dict[str, Any]]:
    """Chapters from a QuickTime/MP4 chapter text track referenced by ``tref/chap``."""
    tracks: dict[int, tuple[int, int]] = {}
    referenced: list[int] = []
    for name, offset, header, size in _atoms(handle, moov[0], moov[1]):
        if name != b"trak":
            continue
        span = (offset + header, offset + size)
        tkhd = _child(handle, *span, b"tkhd")
        if tkhd is None:
            continue
        head = _read_at(handle, tkhd[0], 24)
        track_id = struct.unpack(">I", head[20:24] if head[0] == 1 else head[12:16])[0]
        tracks[track_id] = span
        chap = _path(handle, *span, b"tref", b"chap")
        if chap is not None:
            data = _read_at(handle, chap[0], chap[1] - chap[0])
            referenced.extend(struct.unpack(">" + "I" * (len(data) // 4), data[:len(data) // 4 * 4]))
    chapters: list[dict[str, Any]] = []
    for track_id in referenced:
        span = tracks.get(track_id)
        if span is None:
            continue
        mdhd = _path(handle, *span, b"mdia", b"mdhd")
        stbl = _path(handle, *span, b"mdia", b"minf", b"stbl")
        if mdhd is None or stbl is None:
            continue
        head = _read_at(handle, mdhd[0], 32)
        timescale = struct.unpack(">I", head[20:24] if head[0] == 1 else head[12:16])[0]
        stts = _child(handle, *stbl, b"stts")
        stsz = _child(handle, *stbl, b"stsz")
        stsc = _child(handle, *stbl, b"stsc")
        stco = _child(handle, *stbl, b"stco")
        co64 = _child(handle, *stbl, b"co64")
        if None in (stts, stsz, stsc) or (stco is None and co64 is None) or not timescale:
            continue
        durations = [delta for count, delta in _full_atom_entries(handle, stts, 8) for _ in range(count)]
        size_data = _read_at(handle, stsz[0], stsz[1] - stsz[0])
        uniform, sample_count = struct.unpack(">II", size_data[4:12])
        sizes = [uniform] * sample_count if uniform else list(
            struct.unpack(">" + "I" * sample_count, size_data[12:12 + 4 * sample_count]))
        if co64 is not None:
            raw = _read_at(handle, co64[0], co64[1] - co64[0])
            count = struct.unpack(">I", raw[4:8])[0]
            offsets = list(struct.unpack(">" + "Q" * count, raw[8:8 + 8 * count]))
        else:
            offsets = [entry[0] for entry in _full_atom_entries(handle, stco, 4)]
        runs = _full_atom_entries(handle, stsc, 12)
        sample_offsets: list[int] = []
        for chunk_index, chunk_offset in enumerate(offsets, start=1):
            per_chunk = next((run[1] for run in reversed(runs) if run[0] <= chunk_index), 0)
            position = chunk_offset
            for _ in range(per_chunk):
                if len(sample_offsets) >= len(sizes):
                    break
                sample_offsets.append(position)
                position += sizes[len(sample_offsets) - 1]
        start = 0
        for index, sample_offset in enumerate(sample_offsets):
            duration = durations[index] if index < len(durations) else 0
            sample = _read_at(handle, sample_offset, sizes[index])
            title = None
            if len(sample) >= 2:
                length = struct.unpack(">H", sample[:2])[0]
                text = sample[2:2 + length]
                try:
                    title = text.decode("utf-16") if text[:2] in (b"\xfe\xff", b"\xff\xfe") else text.decode("utf-8")
                except UnicodeDecodeError:
                    title = None
            chapters.append({"start": [start, timescale], "end": [start + duration, timescale],
                             "title": title, "native": None})
            start += duration
    return chapters


def _parse_mp4_atom(stored: bytes) -> MP4Tags:
    """Parse one occurrence without mutagen merging other same-key atoms (MD4).

    The minimal in-memory tag tree uses the public MP4Tags.load interface;
    no source audio or container property is needed to parse an ilst value.
    The original atom is retained separately for MD32's raw fall-back.
    """
    def wrap(name: bytes, body: bytes) -> bytes:
        return struct.pack(">I4s", len(body) + 8, name) + body

    tree = wrap(b"moov", wrap(b"udta", wrap(b"meta", bytes(4) + wrap(b"ilst", stored))))
    buffer = io.BytesIO(tree)
    tags = MP4Tags()
    tags.load(Atoms(buffer), buffer)
    return tags


def _read_mp4(handle: BinaryIO, size: int, read: _Read) -> None:
    moov = None
    for name, offset, header, atom_size in _atoms(handle, 0, size):
        if name == b"moov":
            moov = (offset + header, offset + atom_size)
            break
    if moov is None:
        return
    track_chapters = _chapter_track(handle, moov)
    if track_chapters:
        read.chapters["mp4-track"] = track_chapters
    chpl = _path(handle, *moov, b"udta", b"chpl")
    if chpl is not None:
        read.chapters["mp4-chpl"] = _read_chpl(handle, chpl)
    ilst = _path(handle, *moov, b"udta", b"meta", b"ilst")
    if ilst is None:
        return
    read.present("mp4")
    for name, offset, header, atom_size in _atoms(handle, ilst[0], ilst[1]):
        key = _ilst_key(handle, name, offset, header, atom_size)
        locator = f"ilst@{offset}"
        native_type = name.decode("latin-1")
        stored = _read_at(handle, offset, atom_size)
        foreign = _foreign_text(handle, key, offset, header, atom_size)
        try:
            tags = _parse_mp4_atom(stored)
        except Exception:  # noqa: BLE001 - only this unparsable occurrence stays raw
            tags = {}
        if key not in tags:
            read.items.append(_Item("mp4", key, key, native_type, None, RAW, stored, locator, foreign_text=foreign))
            continue
        value = tags[key]
        scalar = not isinstance(value, list)
        values = [value] if scalar else list(value)
        text_types_ok = not any(isinstance(item, str) for item in values) or all(
            kind == 1 for kind in _data_types(handle, offset, header, atom_size))
        groups = [[item] for item in values] if key == "covr" else [values]
        for group in groups:
            item = _Item("mp4", key, key, native_type, None, RAW, stored, locator, _mp4_projection(key, group),
                         foreign_text=foreign)
            try:
                if not text_types_ok:
                    raise _Unrepresentable("text atom not stored as UTF-8")
                item.payload = canonical_json({"atom": key, "scalar": scalar,
                                               "values": [_mp4_value(entry) for entry in group]})
                item.encoding = JSON
            except (_Unrepresentable, UnicodeEncodeError, ValueError):
                pass  # the stored atom stays the dn-raw-1 payload
            read.items.append(item)


# --- Shared helpers ------------------------------------------------------------------------


def _sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:2] == b"BM":
        return "image/bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _assign_identities(items: list[_Item]) -> None:
    """``store:<store>:<native key>``, with ``#n`` only where a key repeats in its store (MD4)."""
    counts: dict[tuple[str, str], int] = {}
    for item in items:
        counts[(item.store, item.native_key)] = counts.get((item.store, item.native_key), 0) + 1
    seen: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item.store, item.native_key)
        seen[key] = seen.get(key, 0) + 1
        item.identity = model.native_identity(item.store, item.native_key,
                                              seen[key] if counts[key] > 1 else None)


# --- Canonical projection (MD3, MD5) ----------------------------------------------------------

_INVALID = object()


@dataclass
class _Candidate:
    value: Any  # a MetadataValue, or _INVALID
    identities: tuple[str, ...]
    store: str
    invalid_ids: tuple[str, ...] = ()


def _texts(item: _Item) -> tuple[str, ...] | None:
    value = item.value
    if isinstance(value, tuple) and all(isinstance(part, str) for part in value):
        return value
    return None


def _text_candidate(items: list[_Item], field_name: str, store: str) -> _Candidate | None:
    """Several occurrences and multi-strings are joined with ``; `` (MD5)."""
    valid: list[str] = []
    valid_ids: list[str] = []
    invalid_ids: list[str] = []
    for item in items:
        texts = _texts(item)
        if texts is None:
            invalid_ids.append(item.identity)
            continue
        try:
            values = [model.canonical_text(text, field_name) for text in texts]
        except model.InvalidValue:
            invalid_ids.append(item.identity)
            continue
        if any(values):
            valid.extend(value for value in values if value)
            valid_ids.append(item.identity)
    if valid:
        joined = model.join_text(valid)
        try:
            return _Candidate(model.canonical_text(joined, field_name), tuple(valid_ids), store, tuple(invalid_ids))
        except model.InvalidValue:
            return _Candidate(_INVALID, tuple(valid_ids + invalid_ids), store)
    if invalid_ids:
        return _Candidate(_INVALID, tuple(invalid_ids), store)
    return None


def _first_texts(item: _Item) -> list[str] | None:
    texts = _texts(item)
    return list(texts) if texts else None


class _StoreView:
    """The items of one store grouped by the canonical field they feed."""

    def __init__(self, store: str, items: list[_Item]) -> None:
        self.store = store
        self.items = items
        self.by_field: dict[str, list[_Item]] = {}
        for item in items:
            for name in model.native_fields(store, item.native_key):
                self.by_field.setdefault(name, []).append(item)

    def frames(self, *ids: str) -> list[_Item]:
        return [item for item in self.items if item.native_type in ids]


def _typed_candidates(store: str, items: list[_Item], parse: Any) -> list[_Candidate]:
    """One candidate per stored string; invalid strings become invalid candidates."""
    out: list[_Candidate] = []
    for item in items:
        texts = _first_texts(item)
        if texts is None:
            out.append(_Candidate(_INVALID, (item.identity,), store))
            continue
        for text in texts:
            if not text.strip():
                continue
            try:
                out.append(_Candidate(parse(text), (item.identity,), store))
            except model.InvalidValue:
                out.append(_Candidate(_INVALID, (item.identity,), store))
    return out


def _date_candidates(view: _StoreView) -> list[_Candidate]:
    store = view.store
    if store != "id3v2":
        return _typed_candidates(store, view.by_field.get("date", []), model.source_date)
    out = _typed_candidates(store, view.frames("TDRC"), model.source_date)
    for year_id, day_id in (("TYER", "TDAT"), ("TYE", "TDA")):
        days = view.frames(day_id)
        day_texts = _first_texts(days[0]) if days else None
        for item in view.frames(year_id):
            texts = _first_texts(item)
            if not texts:
                out.append(_Candidate(_INVALID, (item.identity,), store))
                continue
            try:
                date = model.id3v23_date(texts[0], day_texts[0] if day_texts else None)
                ids = (item.identity, days[0].identity) if day_texts else (item.identity,)
            except model.InvalidValue:
                try:
                    date = model.id3v23_date(texts[0], None)
                    ids = (item.identity,)
                except model.InvalidValue:
                    out.append(_Candidate(_INVALID, (item.identity,), store))
                    continue
            out.append(_Candidate(date, ids, store))
    return out


def _pair_candidates(view: _StoreView, number_field: str, total_field: str,
                     pair_keys: tuple[str, ...], total_keys: tuple[str, ...]) -> dict[str, list[_Candidate]]:
    """Number and total candidates (MD5).

    Totals rank: dedicated total keys in the given order (Vorbis ``TRACKTOTAL``
    above ``TOTALTRACKS``), then the ``N`` of a pair. A 0 or empty component is
    absent; a pair with ``n > N`` keeps ``n`` and makes ``N`` invalid.
    """
    store = view.store
    numbers: list[_Candidate] = []
    pair_totals: list[_Candidate] = []
    for item in view.items:
        if item.native_key.upper() not in pair_keys:
            continue
        pieces: list[tuple[int | None, int | None, bool]] = []
        if isinstance(item.value, tuple) and len(item.value) == 2 and \
                all(isinstance(part, int) and not isinstance(part, bool) for part in item.value):
            pieces.append(model.check_pair(item.value[0] or None, item.value[1] or None))
        else:
            texts = _first_texts(item)
            if texts is None:
                numbers.append(_Candidate(_INVALID, (item.identity,), store))
            for text in texts or ():
                if not text.strip():
                    continue
                try:
                    pieces.append(model.parse_number_pair(text))
                except model.InvalidValue:
                    numbers.append(_Candidate(_INVALID, (item.identity,), store))
        for number, total, total_invalid in pieces:
            if number is not None:
                numbers.append(_Candidate(number, (item.identity,), store))
            if total is not None:
                pair_totals.append(_Candidate(total, (item.identity,), store))
            elif total_invalid:
                pair_totals.append(_Candidate(_INVALID, (item.identity,), store))
    totals: list[_Candidate] = []
    for total_key in total_keys:
        items = [item for item in view.items if item.native_key.upper() == total_key]
        totals.extend(candidate for candidate in _typed_candidates(store, items, model.parse_integer)
                      if candidate.value is not None)
    return {number_field: numbers, total_field: totals + pair_totals}


_PAIR_KEYS = {
    "id3v2": (("TRCK", "TRK"), ("TPOS", "TPA")),
    "mp4": (("TRKN",), ("DISK",)),
    "apev2": (("TRACK",), ("DISC",)),
    "vorbis": (("TRACKNUMBER",), ("DISCNUMBER",)),
}
_TOTAL_KEYS = {"vorbis": (("TRACKTOTAL", "TOTALTRACKS"), ("DISCTOTAL", "TOTALDISCS"))}


def _field_candidates(view: _StoreView) -> dict[str, list[_Candidate]]:
    """Every candidate per canonical field for one store, in the store's own ranking."""
    store = view.store
    out: dict[str, list[_Candidate]] = {}
    for name in sorted(model.TEXT_FIELDS):
        items = view.by_field.get(name, [])
        candidate = _text_candidate(items, name, store) if items else None
        if candidate is not None:
            out[name] = [candidate]
    out["date"] = _date_candidates(view)
    if store in _PAIR_KEYS:
        track_keys, disc_keys = _PAIR_KEYS[store]
        track_totals, disc_totals = _TOTAL_KEYS.get(store, ((), ()))
        out.update(_pair_candidates(view, "track", "track_total", track_keys, track_totals))
        out.update(_pair_candidates(view, "disc", "disc_total", disc_keys, disc_totals))
    elif store in ("riff-info", "id3v1"):
        # IPRT and the ID3v1.1 track byte carry a number only (formats § 8)
        out["track"] = [candidate for candidate in _typed_candidates(
            store, view.by_field.get("track", []), lambda text: model.parse_number_pair(text)[0])
            if candidate.value is not None]
    flags: list[_Candidate] = []
    for item in view.by_field.get("compilation", []):
        if isinstance(item.value, bool):
            flags.append(_Candidate(item.value, (item.identity,), store))
        else:
            flags.extend(_typed_candidates(store, [item], model.parse_compilation))
    out["compilation"] = flags
    return {name: candidates for name, candidates in out.items() if candidates}


def _id3v1_truncation(key: str, candidate: _Candidate, winner: Any) -> bool:
    """MD3: an ID3v1 value equal to the winning value cut to the v1 field width is no conflict.

    [judgement] The four-byte v1 year equal to the winning date's year is the same truncation.
    """
    if candidate.store != "id3v1" or not isinstance(winner, str) or not isinstance(candidate.value, str):
        return False
    if key == "field:date":
        return winner[:4] == candidate.value
    encoded = winner.encode("latin-1", "replace")
    return any(encoded[:width].decode("latin-1").strip() == candidate.value for width in (30, 28))


def _resolve(key: str, candidates: list[_Candidate], conflicts: list[Conflict]) -> _Candidate | None:
    """The first valid candidate in priority order; differing valid values are internal conflicts."""
    winner = next((candidate for candidate in candidates if candidate.value is not _INVALID), None)
    if winner is None:
        return None
    for candidate in candidates:
        if candidate is winner or candidate.value is _INVALID:
            continue
        if model.values_equal(candidate.value, winner.value) or _id3v1_truncation(key, candidate, winner.value):
            continue
        conflicts.append(Conflict(key, candidate.value, winner.value, "source:" + candidate.identities[0],
                                  "source", "internal:" + winner.identities[0]))
    return winner


def _extra_candidates(view: _StoreView) -> dict[str, tuple[str, list[_Candidate]]]:
    groups: dict[str, tuple[str, list[_Item]]] = {}
    for item in view.items:
        name = model.native_extra(view.store, item.key)
        if name is None:
            continue
        if view.store == "apev2" and item.native_type != "text":
            continue
        if view.store == "mp4" and item.native_type != "----":
            continue
        folded = name.casefold()
        if folded not in groups:
            groups[folded] = (name, [])
        groups[folded][1].append(item)
    out = {}
    for folded, (name, items) in groups.items():
        candidate = _text_candidate(items, model.EXTRA, view.store)
        if candidate is not None:
            out[folded] = (name, [candidate])
    return out


def _project(order: tuple[str, ...], items: list[_Item]) -> tuple[dict[str, TagValue], dict[str, TagValue],
                                                                  tuple[TagValue, ...], tuple[Conflict, ...]]:
    views = [_StoreView(store, [item for item in items if item.store == store]) for store in order]
    conflicts: list[Conflict] = []
    fields: dict[str, TagValue] = {}
    per_store = [_field_candidates(view) for view in views]
    for name in model.CANONICAL_FIELDS:
        if name == "artwork":
            continue
        candidates = [candidate for table in per_store for candidate in table.get(name, [])]
        winner = _resolve(f"field:{name}", candidates, conflicts)
        if winner is not None:
            fields[name] = TagValue(winner.value, winner.store, ",".join(winner.identities))
    extras: dict[str, TagValue] = {}
    extra_tables = [_extra_candidates(view) for view in views]
    names: dict[str, str] = {}
    for table in extra_tables:
        for folded, (name, _) in table.items():
            names.setdefault(folded, name)
    for folded, name in names.items():
        candidates = [candidate for table in extra_tables for candidate in table.get(folded, ("", []))[1]]
        winner = _resolve(f"extra:{name}", candidates, conflicts)
        if winner is not None:
            extras[name] = TagValue(winner.value, winner.store, ",".join(winner.identities))
    _foreign_conflicts(items, fields, extras, conflicts)
    # CR-082: joining valid occurrences must not hide rejected evidence.
    item_index = {item.identity: item for item in items}
    tables = [(f"field:{name}", value, [candidate for table in per_store for candidate in table.get(name, [])])
              for name, value in fields.items()]
    tables += [(f"extra:{name}", value,
                [candidate for table in extra_tables for candidate in table.get(name.casefold(), ("", []))[1]])
               for name, value in extras.items()]
    for key, winner, candidates in tables:
        rejected = dict.fromkeys(identity for candidate in candidates
                                 for identity in (candidate.identities if candidate.value is _INVALID else candidate.invalid_ids))
        for identity in rejected:
            item = item_index[identity]
            if item.foreign_text is not None:
                continue  # CR-043 already records this exact occurrence.
            texts = _texts(item)
            display = model.join_text(texts) if texts is not None else item.payload.decode("utf-8", "replace")
            conflicts.append(Conflict(key, display, winner.value, "source:" + identity,
                                      "source", "invalid_source_value"))
    artwork: tuple[TagValue, ...] = ()
    for view in views:
        pictures = tuple(TagValue(item.value, view.store, item.identity)
                         for item in view.by_field.get("artwork", []) if isinstance(item.value, Artwork))
        if pictures:
            artwork = pictures
            break
    return fields, extras, artwork, tuple(conflicts)


def _foreign_conflicts(items: list[_Item], fields: dict[str, TagValue], extras: dict[str, TagValue],
                       conflicts: list[Conflict]) -> None:
    """CR-043: an MP4 text occurrence left out of a projected value because of its data type is a conflict.

    The occurrence stays raw evidence in the inventory; without a projected value for its field
    there is nothing to disagree with, and the later roll-up reports an invalid source value (MD5).
    """
    for item in items:
        if item.foreign_text is None:
            continue
        types, text = item.foreign_text
        names = [name for name in model.native_fields("mp4", item.native_key) if name in fields]
        winners = [(f"field:{name}", fields[name]) for name in names]
        extra = model.native_extra("mp4", item.native_key)
        if extra is not None:
            winners += [(f"extra:{name}", value) for name, value in extras.items()
                        if name.casefold() == extra.casefold()]
        for key, winner in winners:
            if item.identity in winner.locator.split(","):
                continue
            conflicts.append(Conflict(key, text, winner.value, "source:" + item.identity, "source",
                                      "mp4_data_type:" + "/".join(str(kind) for kind in types)))


# --- Non-tag items (MD7) ------------------------------------------------------------------------


def _nontag_items(read: _Read) -> list[NonTagItem]:
    items: list[NonTagItem] = []
    counts: dict[str, int] = {}
    for name, _, _, _, _ in read.nontag:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    for name, locator, payload, size, digest in read.nontag:
        seen[name] = seen.get(name, 0) + 1
        identity = model.nontag_identity("chunk", name, seen[name] if counts[name] > 1 else None)
        items.append(NonTagItem(identity, "chunk", locator, NOT_CARRIED,
                                payload if size <= PAYLOAD_LIMIT else None, size, digest))
    vorbis = _vorbis_chapters(read)
    if vorbis:
        read.chapters["vorbis"] = vorbis
    sources = []
    for source in _CHAPTER_SOURCES:
        chapters = read.chapters.get(source)
        if not chapters:
            continue
        entries = []
        for chapter in chapters:
            native = chapter["native"]
            entries.append({**chapter, "native": native.identity if isinstance(native, _Item) else native})
        sources.append({"source": source, "chapters": entries})
    if sources:
        payload = canonical_json({"sources": sources})
        locator = ",".join(source["source"] for source in sources)
        items.append(NonTagItem(model.nontag_identity("chapters"), "chapters", locator, NOT_CARRIED,
                                payload if len(payload) <= PAYLOAD_LIMIT else None, len(payload),
                                hashlib.sha256(payload).hexdigest()))
    return items


def _vorbis_chapters(read: _Read) -> list[dict[str, Any]]:
    names: dict[str, str | None] = {}
    starts: list[tuple[str, _Item]] = []
    for item in read.items:
        if item.store != "vorbis" or not model.is_chapter_key("vorbis", item.native_key):
            continue
        texts = _texts(item)
        upper = item.native_key.upper()
        if upper.endswith("NAME"):
            names.setdefault(upper[:-4], model.MULTI_VALUE_SEPARATOR.join(texts) if texts else None)
        elif not upper.endswith("URL"):
            starts.append((upper, item))
    chapters = []
    for upper, item in starts:
        texts = _texts(item)
        chapters.append({"start": {"text": texts[0] if texts else
                                   item.payload.decode("utf-8", "replace")},
                         "end": None, "title": names.get(upper), "native": item})
    return chapters


# --- Public entry points -----------------------------------------------------------------------

_READERS = {"mp4": _read_mp4, "flac": _read_flac, "mp3": _read_mp3, "aiff": _read_aiff, "wav": _read_wav}


def read_source_tags(path: Path, container: str) -> SourceTags:
    """Read every MD3 store of ``path`` into a C-06 ``SourceTags``.

    ``container`` is the C-02 canonical container name (``mp4``, ``flac``,
    ``mp3``, ``aiff``, ``wav``); a container without an MD3 row raises
    ``ValueError``. Access denial or a vanished source raises ``FileFailure``
    (``permission_denied`` / ``source_changed``). The file is opened read-only
    and never modified.
    """
    family = model.store_family(container)
    read = _Read()
    try:
        with open(path, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            _READERS[family](handle, size, read)
    except OSError as exc:
        raise _failure(exc) from None
    _assign_identities(read.items)
    order = model.CONTAINER_STORES[family]
    fields, extras, artwork, conflicts = _project(order, read.items)
    inventory = tuple(
        RawNativeItem(item.identity, item.store, item.key, item.native_type, item.native_version,
                      item.encoding, item.payload, item.locator)
        for store in order for item in read.items if item.store == store)
    stores = {store: tuple(item.identity for item in inventory if item.store == store)
              for store in order if store in read.stores}
    return SourceTags(fields, extras, artwork, tuple(_nontag_items(read)), conflicts, inventory, stores)


def source_field_ids(source: SourceTags) -> dict[str, tuple[str, ...]]:
    """Preparation's component-specific evidence index for pure planning.

    A stored zero/empty component is absent, not invalid evidence. Decode
    combined numbers and validate legacy canonical/extra projections here at
    the MD32 edge; no core payload parser (CR-098).
    """
    result: dict[str, list[str]] = {}
    spellings = {name.casefold(): name for name in source.extras}
    for native in source.native_inventory:
        if native.store in ("id3v1", "apev2", "aiff-text"):
            # CR-098: key eligibility alone does not prove a legacy value projects.
            value = None
            if native.store == "apev2" and native.native_type == "text":
                document = json.loads(native.payload)
                try:
                    value = tuple(base64.b64decode(document["value"]["b64"], validate=True).decode("utf-8").split("\x00"))
                except UnicodeDecodeError:
                    pass
            elif native.store == "aiff-text":
                value = _text_item(native.store, native.key.encode("latin-1"), native.payload, native.locator).value
            elif native.store == "id3v1":
                if native.key == "track":
                    value = (str(native.payload[0]),) if native.payload else None
                elif native.key == "genre":
                    value = (_TCON.GENRES[native.payload[0]],) if native.payload and native.payload[0] < len(_TCON.GENRES) else None
                else:
                    value = (native.payload.rstrip(b"\x00 ").decode("latin-1"),)
            item = _Item(native.store, native.key, native.key, native.native_type,
                         native.native_version, native.payload_encoding, native.payload,
                         native.locator, value=value, identity=native.identity)
            view = _StoreView(native.store, [item])
            for name, candidates in _field_candidates(view).items():
                if any(candidate.value is not _INVALID for candidate in candidates):
                    result.setdefault("field:" + name, []).append(native.identity)
            for folded, (name, candidates) in _extra_candidates(view).items():
                if any(candidate.value is not _INVALID for candidate in candidates):
                    name = spellings.setdefault(folded, name)
                    result.setdefault("extra:" + name, []).append(native.identity)
            continue
        names = model.native_fields(native.store, native.key)
        if len(names) > 1 and names[0] in ("track", "disc"):
            value = None
            if native.payload_encoding == JSON:
                document = json.loads(native.payload)
                if native.store == "id3v2":
                    value = tuple(str(text) for text in dict(document["fields"]).get("text", ()))
                elif native.store == "mp4":
                    values = document.get("values", ())
                    if values and isinstance(values[0], dict) and "pair" in values[0]:
                        value = tuple(values[0]["pair"])
                elif native.store == "apev2" and native.native_type == "text":
                    try:
                        value = tuple(base64.b64decode(document["value"]["b64"], validate=True).decode("utf-8").split("\x00"))
                    except UnicodeDecodeError:
                        pass
            elif native.store == "vorbis":
                try:
                    value = (native.payload.decode("utf-8"),)
                except UnicodeDecodeError:
                    pass

            item = _Item(native.store, native.key, native.key, native.native_type,
                         native.native_version, native.payload_encoding, native.payload,
                         native.locator, value=value, identity=native.identity)
            candidates = _pair_candidates(_StoreView(native.store, [item]), names[0], names[1],
                                          (native.key.upper(),), ())
            names = tuple(name for name, values in candidates.items() if values)
        for name in names:
            if name != "artwork":
                result.setdefault("field:" + name, []).append(native.identity)
        extra = model.native_extra(native.store, native.key)
        if extra is not None:
            name = spellings.setdefault(extra.casefold(), extra)
            result.setdefault("extra:" + name, []).append(native.identity)
    return {key: tuple(ids) for key, ids in result.items()}


def native_items_from_bytes(store: str, data: bytes) -> tuple[RawNativeItem, ...]:
    """The native items of one in-memory store render, encoded exactly as read-back encodes them (MD29, MD32).

    ``store`` is ``id3v2`` (``data`` is one ID3v2 tag), ``mp4`` (an MP4 container) or ``vorbis``
    (a FLAC stream). The write edge uses this so that expected and read-back payloads share one
    encoder (MD22). Only items of ``store`` are returned, with MD4 identities.
    """
    read = _Read()
    handle = io.BytesIO(data)
    if store == "id3v2":
        if _id3_tag_size(data[:10]) is None:
            raise ValueError("not an ID3v2 tag")
        _read_id3_tag(data, 0, read, "ID3@0")
    elif store == "mp4":
        _read_mp4(handle, len(data), read)
    elif store == "vorbis":
        _read_flac(handle, len(data), read)
    else:
        raise ValueError(f"no in-memory reader for store {store!r}")
    items = [item for item in read.items if item.store == store]
    _assign_identities(items)
    return tuple(RawNativeItem(item.identity, item.store, item.key, item.native_type, item.native_version,
                               item.encoding, item.payload, item.locator) for item in items)


def decode_mark_value(item: RawNativeItem) -> list[str | None]:
    """The text values of a native occurrence holding the reserved mark name (MD37).

    ``None`` marks a value that cannot be decoded as text (a present, malformed mark).
    """
    try:
        if item.payload_encoding == RAW:
            if item.store == "vorbis":
                return [item.payload.decode("utf-8")]
            return [None]
        document = json.loads(item.payload)
        if item.store == "id3v2":
            fields = dict((name, value) for name, value in document["fields"])
            return [model.MULTI_VALUE_SEPARATOR.join(fields["text"])]
        if item.store == "mp4":
            out: list[str | None] = []
            for value in document["values"]:
                raw = base64.b64decode(value["freeform"]) if isinstance(value, dict) and "freeform" in value \
                    else value.encode("utf-8") if isinstance(value, str) else None
                out.append(raw.decode("utf-8") if raw is not None else None)
            return out
        if item.store == "apev2":
            return [base64.b64decode(document["value"]["b64"]).decode("utf-8")]
    except (UnicodeDecodeError, KeyError, TypeError, ValueError):
        return [None]
    return [None]


def read_processing_mark(tags: SourceTags) -> ProcessingMark:
    """Parse the processing mark from a source inventory at ``tags_read`` (MD26, MD37)."""
    occurrences: list[tuple[str, str | None]] = []
    for item in tags.native_inventory:
        if model.is_mark_key(item.store, item.key):
            occurrences.extend((item.identity, value) for value in decode_mark_value(item))
    return model.parse_processing_mark(occurrences)
