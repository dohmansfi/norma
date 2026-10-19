"""ID3v2, APEv2 and ID3v1 readers (TST-M-01 readers slice; MD3–MD5, MD32, MD37).

Tags are written by mutagen in the test or built by hand; mutagen's whole-tag
parse and raw byte searches are the independent cross-checks.
"""

from __future__ import annotations

import base64
import json
import struct
from collections import Counter
from pathlib import Path

import pytest
from mutagen.apev2 import APEv2
from mutagen.aiff import AIFF
from mutagen.id3 import (
    APIC, CHAP, COMM, CTOC, GEOB, ID3, POPM, PRIV, TCMP, TCOM, TCON, TDRC, TIPL, TIT2, TPE1, TPE2, TPOS, TRCK,
    TXXX, USLT, WXXX, CTOCFlags, Frames,
)
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from norma.metadata.read import JSON, RAW, read_processing_mark, read_source_tags

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(32))
FIXTURE_MARK = (
    "v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=limit;gain=+5.80;limited=2.60;out_i=-16.02;"
    "out_tp=-1.04;origin=pcm_s24le;ffmpeg=N-125892;date=2026-10-05"
)
OPENERS = {"mp3": MP3, "aiff": AIFF, "wav": WAVE}


def tag_file(path: Path, kind: str, frames: list, version: int = 4) -> None:
    if kind == "flac":
        tags = ID3()
        for frame in frames:
            tags.add(frame)
        tags.save(path, v2_version=version)
        return
    audio = OPENERS[kind](path)
    if audio.tags is None:
        audio.add_tags()
    for frame in frames:
        audio.tags.add(frame)
    if version == 3:
        audio.tags.update_to_v23()
    audio.save(v2_version=version)


def by_identity(tags) -> dict:
    return {item.identity: item for item in tags.native_inventory}


def json_payload(item) -> dict:
    assert item.payload_encoding == JSON
    return json.loads(item.payload)


# --- Stores per container -----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mp3", "aiff", "wav", "flac"])
def test_id3_tag_is_read_in_every_container(kind: str, audio, build) -> None:
    path = audio(kind)
    tag_file(path, kind, [TIT2(encoding=3, text=["Night Bus"]), TPE1(encoding=1, text=["Test Artist"]),
                          TXXX(encoding=3, desc="BPM", text=["120"])])
    before = path.read_bytes()
    tags = read_source_tags(path, build.container(kind))
    assert path.read_bytes() == before  # reading never writes
    assert "id3v2" in tags.stores
    ids = tags.stores["id3v2"]
    assert set(ids) == {"store:id3v2:TIT2", "store:id3v2:TPE1", "store:id3v2:TXXX:BPM"}
    assert tags.fields["title"].value == "Night Bus" and tags.fields["title"].system == "id3v2"
    assert tags.fields["artist"].value == "Test Artist"
    assert tags.extras["BPM"].value == "120"
    # independent whole-tag parse agrees on the stored frames
    independent = ID3(path, translate=False) if kind in ("mp3", "flac") else OPENERS[kind](path).tags
    assert Counter(frame.FrameID for frame in independent.values()) == \
        Counter(item.native_type for item in tags.native_inventory if item.store == "id3v2")
    assert all(item.native_version == "2.4" for item in tags.native_inventory)


def test_flac_leading_id3_ranks_below_vorbis(audio) -> None:
    from mutagen.flac import FLAC
    path = audio("flac")
    flac = FLAC(path)
    flac["TITLE"] = "Vorbis title"
    flac.save()
    tag_file(path, "flac", [TIT2(encoding=3, text=["ID3 title"]), TCOM(encoding=3, text=["Composer"])])
    tags = read_source_tags(path, "flac")
    assert tuple(tags.stores) == ("vorbis", "id3v2")
    assert tags.fields["title"].value == "Vorbis title"
    assert tags.fields["composer"].value == "Composer"  # filled from the lower store (MD3)
    assert [(c.key, c.old_value, c.new_value, c.old_source) for c in tags.internal_conflicts] == \
        [("field:title", "ID3 title", "Vorbis title", "source:store:id3v2:TIT2")]


# --- Versions and dates --------------------------------------------------------------------


def test_id3v23_records_stored_frames_and_joins_tyer_tdat(audio) -> None:
    path = audio("mp3")
    tag_file(path, "mp3", [TIT2(encoding=3, text=["x"]), TDRC(encoding=3, text=["2026-10-05"])], version=3)
    tags = read_source_tags(path, "mp3")
    items = by_identity(tags)
    assert {"store:id3v2:TYER", "store:id3v2:TDAT"} <= set(items)
    assert "store:id3v2:TDRC" not in items  # the stored v2.3 frames, never translated
    assert {item.native_version for item in tags.native_inventory} == {"2.3"}
    assert tags.fields["date"].value == "2026-10-05"
    assert set(tags.fields["date"].locator.split(",")) == {"store:id3v2:TYER", "store:id3v2:TDAT"}


def test_tdrc_timestamp_keeps_stored_text(audio) -> None:
    path = audio("mp3")
    tag_file(path, "mp3", [TDRC(encoding=3, text=["2026-10-05T07:00"])])
    tags = read_source_tags(path, "mp3")
    item = by_identity(tags)["store:id3v2:TDRC"]
    assert json_payload(item)["fields"] == [["encoding", 3], ["text", [{"ts": "2026-10-05T07:00"}]]]
    assert tags.fields["date"].value == "2026-10-05T07:00"


def test_invalid_timestamp_is_evidence_not_a_date(audio, build) -> None:
    path = audio("mp3")
    path.write_bytes(build.id3_tag(4, [build.id3_frame("TDRC", b"\x03Spring 2026")]) + path.read_bytes())
    tags = read_source_tags(path, "mp3")
    assert "date" not in tags.fields
    item = by_identity(tags)["store:id3v2:TDRC"]
    assert json_payload(item)["fields"][1] == ["text", [{"ts": "Spring 2026"}]]  # no lossy re-format


def test_tdrc_and_tyer_disagreeing_is_an_internal_conflict(audio, build) -> None:
    path = audio("mp3")
    frames = [build.id3_frame("TDRC", b"\x032026-10-05"), build.id3_frame("TYER", b"\x002025")]
    path.write_bytes(build.id3_tag(4, frames) + path.read_bytes())
    tags = read_source_tags(path, "mp3")
    assert tags.fields["date"].value == "2026-10-05"
    assert [(c.key, c.old_value, c.new_value) for c in tags.internal_conflicts] == \
        [("field:date", "2025", "2026-10-05")]


def test_id3v22_tag_is_read_with_stored_ids(audio, build) -> None:
    path = audio("mp3")
    frames = [build.id3_frame("TT2", b"\x00Old title", 2), build.id3_frame("TP1", b"\x00Old artist", 2),
              build.id3_frame("COM", b"\x00eng\x00a comment", 2), build.id3_frame("TYE", b"\x002024", 2)]
    path.write_bytes(build.id3_tag(2, frames) + path.read_bytes())
    tags = read_source_tags(path, "mp3")
    assert [item.native_type for item in tags.native_inventory] == ["TT2", "TP1", "COM", "TYE"]
    assert {item.native_version for item in tags.native_inventory} == {"2.2"}
    assert "store:id3v2:COM::eng" in by_identity(tags)
    assert json_payload(by_identity(tags)["store:id3v2:TT2"])["frame"] == "TT2"
    assert (tags.fields["title"].value, tags.fields["artist"].value) == ("Old title", "Old artist")
    assert (tags.fields["comment"].value, tags.fields["date"].value) == ("a comment", "2024")


def test_whole_tag_unsynchronisation_is_reversed(audio, build) -> None:
    path = audio("mp3")
    frame = build.id3_frame("TIT2", b"\x00\xff\xff ok", 3)
    tag = build.id3_tag(3, [frame])
    body = tag[10:].replace(b"\xff", b"\xff\x00")
    path.write_bytes(b"ID3\x03\x00\x80" + build.syncsafe(len(body)) + body + path.read_bytes())
    tags = read_source_tags(path, "mp3")
    assert tags.fields["title"].value == "\xff\xff ok"


# --- Native keys, duplicates and fall-backs (MD4, MD32) ------------------------------------------


def test_refined_native_keys(audio) -> None:
    path = audio("mp3")
    tag_file(path, "mp3", [
        COMM(encoding=3, lang="eng", desc="", text=["note"]), COMM(encoding=3, lang="eng", desc="iTunNORM", text=["x"]),
        USLT(encoding=3, lang="ita", desc="v1", text="lyrics"),
        APIC(encoding=3, mime="image/png", type=3, desc="", data=PNG),
        PRIV(owner="com.example", data=b"\x00\x01"), GEOB(encoding=3, mime="a/b", filename="f", desc="g", data=b"z"),
        WXXX(encoding=3, desc="site", url="https://example.invalid"),
        TXXX(encoding=3, desc="REPLAYGAIN_TRACK_GAIN", text=["-3 dB"]),
    ])
    ids = set(read_source_tags(path, "mp3").stores["id3v2"])
    assert ids == {"store:id3v2:COMM::eng", "store:id3v2:COMM:iTunNORM:eng", "store:id3v2:USLT:v1:ita",
                   "store:id3v2:APIC:", "store:id3v2:PRIV:com.example", "store:id3v2:GEOB:g",
                   "store:id3v2:WXXX:site", "store:id3v2:TXXX:REPLAYGAIN_TRACK_GAIN"}


def test_repeated_frames_get_occurrence_numbers_and_join(audio, build) -> None:
    path = audio("mp3")
    frames = [build.id3_frame("TIT2", b"\x03First"), build.id3_frame("TIT2", b"\x03Second")]
    path.write_bytes(build.id3_tag(4, frames) + path.read_bytes())
    tags = read_source_tags(path, "mp3")
    assert tags.stores["id3v2"] == ("store:id3v2:TIT2#1", "store:id3v2:TIT2#2")
    assert tags.fields["title"].value == "First; Second"
    assert tags.fields["title"].locator == "store:id3v2:TIT2#1,store:id3v2:TIT2#2"


def test_fixed_point_and_unparseable_frames_keep_the_stored_frame(audio, build) -> None:
    path = audio("mp3")
    rva2 = build.id3_frame("RVA2", b"track\x00" + bytes([1]) + struct.pack(">h", -1664) + bytes([16]) + b"\x70\x00")
    equ2 = build.id3_frame("EQU2", b"\x01eq\x00" + struct.pack(">Hh", 200, 1280) + struct.pack(">Hh", 200, 512))
    unknown = build.id3_frame("XYZW", b"opaque bytes")
    bad_encoding = build.id3_frame("TPE1", b"\x09not text")
    title = build.id3_frame("TIT2", b"\x03ok")
    path.write_bytes(build.id3_tag(4, [rva2, equ2, unknown, bad_encoding, title]) + path.read_bytes())
    tags = read_source_tags(path, "mp3")
    items = by_identity(tags)
    for identity, stored in (("store:id3v2:RVA2:track", rva2), ("store:id3v2:EQU2:eq", equ2),
                             ("store:id3v2:XYZW", unknown), ("store:id3v2:TPE1", bad_encoding)):
        item = items[identity]
        assert (item.payload_encoding, item.payload) == (RAW, stored)  # the complete stored frame
    assert items["store:id3v2:TIT2"].payload_encoding == JSON
    assert "artist" not in tags.fields  # an unparseable frame is evidence, never a value
    assert tags.stores["id3v2"] == ("store:id3v2:RVA2:track", "store:id3v2:EQU2:eq", "store:id3v2:XYZW",
                                    "store:id3v2:TPE1", "store:id3v2:TIT2")


def test_rvad_is_always_raw(audio, build) -> None:
    path = audio("mp3")
    rvad = build.id3_frame("RVAD", b"\x03\x10" + bytes(8), 3)
    path.write_bytes(build.id3_tag(3, [rvad]) + path.read_bytes())
    item = read_source_tags(path, "mp3").native_inventory[0]
    assert (item.native_type, item.payload_encoding, item.payload) == ("RVAD", RAW, rvad)


# --- MD32 lossless round trip ---------------------------------------------------------------------


def _decode_value(value):
    if isinstance(value, dict) and "b64" in value:
        return base64.b64decode(value["b64"])
    if isinstance(value, dict) and "ts" in value:
        return value["ts"]
    if isinstance(value, dict) and "frames" in value:
        return [_frame_from_payload(frame) for frame in value["frames"]]
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _frame_from_payload(document: dict):
    """Independent decoder: rebuild a mutagen frame from a dn-json-1 payload alone."""
    return Frames[document["frame"]](**{name: _decode_value(value) for name, value in document["fields"]})


@pytest.mark.parametrize("version", [3, 4])
def test_dn_json_payloads_round_trip_losslessly(version: int, audio) -> None:
    rich = [
        TIT2(encoding=1, text=["One", "Two"]), TPE1(encoding=3, text=["Perché 東京 🎵"]),
        TXXX(encoding=3, desc="BPM", text=["120"]), COMM(encoding=3, lang="eng", desc="", text=["a\nb"]),
        APIC(encoding=3, mime="image/png", type=3, desc="front", data=PNG), PRIV(owner="com.example", data=b"\x00\xff"),
        GEOB(encoding=3, mime="application/x", filename="f.bin", desc="blob", data=b"\x01\x02"),
        USLT(encoding=3, lang="eng", desc="", text="words"), TIPL(encoding=3, people=[["mix", "D"]]),
        TDRC(encoding=3, text=["2026-10-05T07:00:00"]), POPM(email="a@example.invalid", rating=200, count=3),
        CHAP(element_id="ch0", start_time=0, end_time=1000, start_offset=0xFFFFFFFF, end_offset=0xFFFFFFFF,
             sub_frames=[TIT2(encoding=3, text=["Intro"])]),
        CTOC(element_id="toc", flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED, child_element_ids=["ch0"], sub_frames=[]),
    ]
    first = audio("mp3")
    tag_file(first, "mp3", rich, version=version)
    source = read_source_tags(first, "mp3")
    rebuilt = ID3()
    for item in source.native_inventory:
        rebuilt.add(_frame_from_payload(json_payload(item)))
    second = audio("mp3")
    rebuilt.save(second, v2_version=version)
    again = read_source_tags(second, "mp3")
    left = sorted((i.identity, i.native_type, i.native_version, i.payload) for i in source.native_inventory)
    right = sorted((i.identity, i.native_type, i.native_version, i.payload) for i in again.native_inventory)
    assert left == right
    assert all(item.payload_encoding == JSON for item in source.native_inventory)


def test_reading_is_deterministic(audio) -> None:
    path = audio("mp3")
    tag_file(path, "mp3", [TIT2(encoding=3, text=["x"]), APIC(encoding=3, mime="image/png", type=3, desc="", data=PNG)])
    assert read_source_tags(path, "mp3") == read_source_tags(path, "mp3")


# --- Canonical projection (MD5) ---------------------------------------------------------------------


@pytest.mark.parametrize(("trck", "track", "total"), [
    ("3/12", 3, 12), ("03", 3, None), ("0/12", None, 12), ("/12", None, 12), ("13/12", 13, None), ("A1", None, None),
])
def test_track_pairs(trck: str, track, total, audio) -> None:
    path = audio("mp3")
    tag_file(path, "mp3", [TRCK(encoding=3, text=[trck]), TPOS(encoding=3, text=["1/2"])])
    tags = read_source_tags(path, "mp3")
    assert (getattr(tags.fields.get("track"), "value", None), getattr(tags.fields.get("track_total"), "value", None)) \
        == (track, total)
    assert (tags.fields["disc"].value, tags.fields["disc_total"].value) == (1, 2)
    assert "store:id3v2:TRCK" in by_identity(tags)  # invalid or partial values remain evidence


def test_typed_and_text_fields(audio) -> None:
    path = audio("aiff")
    tag_file(path, "aiff", [
        TPE2(encoding=3, text=["Norma fixtures"]), TCOM(encoding=3, text=["Composer"]), TCMP(encoding=3, text=["1"]),
        TCON(encoding=3, text=["Demo"]), COMM(encoding=3, lang="eng", desc="", text=["Line one\r\nLine two"]),
        COMM(encoding=3, lang="eng", desc="iTunNORM", text=[" 000003E8"]),
        APIC(encoding=3, mime="image/png", type=3, desc="cover", data=PNG),
        APIC(encoding=3, mime="image/png", type=4, desc="back", data=PNG + b"!"),
        TXXX(encoding=3, desc="NORMA", text=[FIXTURE_MARK]),
        TXXX(encoding=3, desc="REPLAYGAIN_TRACK_GAIN", text=["-3"]),
    ])
    tags = read_source_tags(path, "aiff")
    assert tags.fields["album_artist"].value == "Norma fixtures"
    assert tags.fields["composer"].value == "Composer"
    assert tags.fields["compilation"].value is True
    assert tags.fields["genre"].value == "Demo"
    assert tags.fields["comment"].value == "Line one\nLine two"
    assert tags.extras == {}  # reserved names are never extras (MS3)
    pictures = [tag.value for tag in tags.artwork]
    assert [(p.mime, p.picture_type, p.description, p.data) for p in pictures] == \
        [("image/png", 3, "cover", PNG), ("image/png", 4, "back", PNG + b"!")]
    assert [tag.locator for tag in tags.artwork] == ["store:id3v2:APIC:cover", "store:id3v2:APIC:back"]
    assert read_processing_mark(tags).validity == "valid"


def test_numeric_genre_reference_is_read_through_mutagen(audio) -> None:
    path = audio("mp3")
    tag_file(path, "mp3", [TCON(encoding=3, text=["(17)"])])
    assert read_source_tags(path, "mp3").fields["genre"].value == "Rock"


def test_empty_and_invalid_text_are_not_values(audio, build) -> None:
    path = audio("mp3")
    frames = [build.id3_frame("TIT2", b"\x03   "), build.id3_frame("TPE1", b"\x03bell\x07"),
              build.id3_frame("TCMP", b"\x03maybe"), build.id3_frame("TALB", b"\x03Album")]
    path.write_bytes(build.id3_tag(4, frames) + path.read_bytes())
    tags = read_source_tags(path, "mp3")
    assert set(tags.fields) == {"album"}
    assert len(tags.native_inventory) == 4  # every occurrence stays as evidence


# --- APEv2 and ID3v1 ----------------------------------------------------------------------------------


def ape_tag(items: list[tuple[str, bytes, int]]) -> bytes:
    body = b"".join(struct.pack("<II", len(value), flags) + key.encode() + b"\x00" + value
                    for key, value, flags in items)
    footer = b"APETAGEX" + struct.pack("<IIII", 2000, len(body) + 32, len(items), 0) + bytes(8)
    return body + footer


def id3v1(title: bytes, artist: bytes = b"", year: bytes = b"", track: int = 0, genre: int = 255) -> bytes:
    comment = bytes(28) + (b"\x00" + bytes([track]) if track else bytes(2))
    return (b"TAG" + title.ljust(30, b"\x00") + artist.ljust(30, b"\x00") + bytes(30) + year.ljust(4, b"\x00")
            + comment + bytes([genre]))


def test_mp3_legacy_stores_priority_and_truncation(audio) -> None:
    path = audio("mp3")
    long_title = "A very long title that exceeds thirty bytes"
    tag_file(path, "mp3", [TIT2(encoding=3, text=[long_title]), TPE1(encoding=3, text=["Artist"]),
                           TDRC(encoding=3, text=["2026-10-05"])])
    ape = APEv2()
    ape["Title"], ape["Artist"], ape["Composer"], ape["Mood"] = "Ape title", "Artist", "Ape composer", "calm"
    ape.save(path)
    with open(path, "ab") as handle:
        handle.write(id3v1(long_title.encode()[:30], b"Artist", b"2026", track=3, genre=17))
    tags = read_source_tags(path, "mp3")
    assert tuple(tags.stores) == ("id3v2", "apev2", "id3v1")
    assert set(APEv2(path).keys()) == {item.key for item in tags.native_inventory if item.store == "apev2"}
    assert tags.fields["title"].value == long_title
    assert tags.fields["composer"].value == "Ape composer"
    assert tags.fields["track"].value == 3 and tags.fields["track"].system == "id3v1"
    assert tags.fields["genre"].value == "Rock"
    assert tags.extras["Mood"].value == "calm"
    assert [(c.key, c.old_source) for c in tags.internal_conflicts] == [("field:title", "source:store:apev2:Title")]
    title = by_identity(tags)["store:id3v1:title"]
    assert (title.payload_encoding, len(title.payload)) == (RAW, 30)
    ape_item = by_identity(tags)["store:apev2:Title"]
    assert json.loads(ape_item.payload) == {"flags": 0, "value": {"b64": base64.b64encode(b"Ape title").decode()}}


@pytest.mark.parametrize("ape_first", [True, False])
def test_apev2_and_id3v1_in_either_order(ape_first: bool, audio) -> None:
    path = audio("mp3")
    ape = ape_tag([("Title", b"Ape", 0), ("Cover Art (Front)", b"front\x00" + PNG, 2),
                   ("NORMA", FIXTURE_MARK.encode(), 0)])
    v1 = id3v1(b"V1 title")
    with open(path, "ab") as handle:
        handle.write(ape + v1 if ape_first else v1 + ape)
    tags = read_source_tags(path, "mp3")
    assert set(tags.stores) == {"id3v2", "apev2", "id3v1"}
    assert tags.stores["id3v2"] == ()  # the encoder's empty tag is a present, empty store
    items = by_identity(tags)
    assert items["store:apev2:Cover Art (Front)"].native_type == "binary"
    assert tags.fields["title"].value == "Ape"
    assert read_processing_mark(tags).validity == "valid"


def test_marks_that_disagree_across_stores_are_malformed(audio) -> None:
    path = audio("mp3")
    tag_file(path, "mp3", [TXXX(encoding=3, desc="NORMA", text=[FIXTURE_MARK])])
    with open(path, "ab") as handle:
        handle.write(ape_tag([("Norma", FIXTURE_MARK.replace("-16.02", "-16.03").encode(), 0)]))
    mark = read_processing_mark(read_source_tags(path, "mp3"))
    assert (mark.validity, mark.locator) == ("malformed", "store:id3v2:TXXX:NORMA")
    assert mark.diagnostics[0].detail == "malformed:mark_inconsistent"
