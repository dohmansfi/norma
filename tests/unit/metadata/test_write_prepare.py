"""MD29/MD31 preparation: expected native items, conversion links and the INFO subset (TST-M-01, TST-M-02).

The decisive cross-check: items prepared in memory equal what ``read.py`` reads
from a real file that mutagen tagged independently with the same values, so the
expected inventory and read-back share one encoding (MD22, MD32).
"""

from __future__ import annotations

import io
import json

import pytest
from mutagen.flac import FLAC, Picture
from mutagen.id3 import (
    COMM, ID3, PRIV, RVA2, SIGN, TBPM, TCON, TDOR, TDRC, TIPL, TIT2, TMOO, TPE1, TRCK, TSOP, TXXX, USLT,
)
from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm

from norma.metadata import mapping, write
from norma.metadata.read import JSON, RAW, native_items_from_bytes, read_source_tags
from norma.models import Artwork, RawNativeItem

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(32))


def signature(item: RawNativeItem) -> tuple:
    return item.store, item.key, item.native_type, item.native_version, item.payload_encoding, item.payload


def id3_source(frames: list, version: int = 4) -> tuple[RawNativeItem, ...]:
    """Source evidence as ``read.py`` records it, from a tag mutagen saved independently."""
    tags = ID3()
    for frame in frames:
        tags.add(frame)
    if version == 3:
        tags.update_to_v23()
    buffer = io.BytesIO()
    tags.save(buffer, v2_version=version, padding=lambda info: 0)
    return native_items_from_bytes("id3v2", buffer.getvalue())


# --- Agreement with independently written files ---------------------------------------------------


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_prepared_id3_items_equal_an_independent_mutagen_write(version: str, audio) -> None:
    fields = {"title": "Night Bus", "artist": "Anna", "date": "2026-10-05", "track": 3, "track_total": 12,
              "genre": "Rock", "comment": "note"}
    recipes = mapping.native_recipes("id3v2", fields, extras={"BPM": "120"}, id3_version=version)
    prepared = write.prepare_store("id3v2", recipes, id3_version=version)
    path = audio("mp3")
    tags = ID3()
    encoding = 1 if version == "2.3" else 3
    tags.add(TIT2(encoding=encoding, text=["Night Bus"]))
    tags.add(TPE1(encoding=encoding, text=["Anna"]))
    tags.add(TRCK(encoding=encoding, text=["3/12"]))
    tags.add(TCON(encoding=encoding, text=["Rock"]))
    tags.add(COMM(encoding=encoding, lang="eng", desc="", text=["note"]))
    tags.add(TXXX(encoding=encoding, desc="BPM", text=["120"]))
    tags.add(TDRC(encoding=encoding, text=["2026-10-05"]))
    if version == "2.3":
        tags.update_to_v23()
    tags.save(path, v2_version=int(version[-1]))
    independent = [item for item in read_source_tags(path, "mp3").native_inventory if item.store == "id3v2"]
    assert sorted(map(signature, prepared.items)) == sorted(map(signature, independent))


def test_prepared_mp4_and_vorbis_items_equal_independent_writes(audio) -> None:
    fields = {"title": "Night Bus", "track_total": 12, "compilation": True}
    picture = Artwork(PNG, "image/png", None, None, 3, "")
    mp4 = write.prepare_store("mp4", mapping.native_recipes("mp4", fields, extras={"BPM": "120"}, pictures=(picture,)))
    path = audio("alac")
    tags = MP4(path)
    tags["©nam"] = ["Night Bus"]
    tags["trkn"] = [(0, 12)]
    tags["cpil"] = True
    tags["----:com.apple.iTunes:BPM"] = [MP4FreeForm(b"120", dataformat=1)]
    tags["covr"] = [MP4Cover(PNG, imageformat=MP4Cover.FORMAT_PNG)]
    tags.save()
    assert sorted(map(signature, mp4.items)) == sorted(map(signature, read_source_tags(path, "mp4").native_inventory))
    vorbis = write.prepare_store("vorbis", mapping.native_recipes("vorbis", fields, extras={"Mood": "calm"},
                                                                  pictures=(picture,)))
    path = audio("flac")
    flac = FLAC(path)
    flac["TITLE"] = "Night Bus"
    flac["TRACKTOTAL"] = "12"
    flac["COMPILATION"] = "1"
    flac["Mood"] = "calm"
    block = Picture()
    block.type, block.mime, block.data = 3, "image/png", PNG
    flac.add_picture(block)
    flac.save()
    assert sorted(map(signature, vorbis.items)) == \
        sorted(map(signature, read_source_tags(path, "flac").native_inventory))
    assert {item.key for item in vorbis.items} >= {"Mood", "TITLE"}  # stored spelling kept


# --- Links -------------------------------------------------------------------------------------------


def test_each_recipe_links_to_the_occurrence_it_produced() -> None:
    fields = {"title": "T", "date": "2026-10-05", "track": 1}
    recipes = mapping.native_recipes("id3v2", fields, pictures=(Artwork(PNG, "image/png", None, None, 3, ""),))
    prepared = write.prepare_store("id3v2", recipes)
    by_id = {item.identity: item for item in prepared.items}
    assert [recipe["key"] for recipe in recipes] == ["TIT2", "TYER", "TDAT", "TRCK", "APIC:"]
    assert [by_id[ids[0]].key for ids in prepared.recipe_ids] == ["TIT2", "TYER", "TDAT", "TRCK", "APIC:"]
    assert all(len(ids) == 1 for ids in prepared.recipe_ids)
    assert len(prepared.items) == 5 and prepared.passthrough_ids == {}


def test_id3_passthrough_is_converted_one_frame_at_a_time() -> None:
    source = id3_source([TIPL(encoding=3, people=[["producer", "Anna"]]), TMOO(encoding=3, text=["calm"]),
                         PRIV(owner="com.example", data=b"\x00\x01"), TBPM(encoding=3, text=["120"]),
                         TDOR(encoding=3, text=["2020-05"]), TSOP(encoding=3, text=["Composer"]),
                         USLT(encoding=3, lang="eng", desc="", text="words"), SIGN(group=1, sig=b"xy")])
    passthrough = [item for item in source if item.payload_encoding == JSON]
    to_v23 = write.prepare_store("id3v2", (), passthrough, id3_version="2.3")
    keys = {identity: [next(i.key for i in to_v23.items if i.identity == e) for e in ids]
            for identity, ids in to_v23.passthrough_ids.items()}
    assert keys == {"store:id3v2:TIPL": ["IPLS"], "store:id3v2:TMOO": [], "store:id3v2:TSOP": [],
                    "store:id3v2:PRIV:com.example": ["PRIV:com.example"], "store:id3v2:TBPM": ["TBPM"],
                    "store:id3v2:TDOR": ["TORY"], "store:id3v2:USLT::eng": ["USLT::eng"],
                    "store:id3v2:SIGN": []}  # mutagen's update_to_v23 keeps a keyed SIGN; Norma removes it
    assert all(item.native_version == "2.3" for item in to_v23.items)
    assert not {item.native_type for item in to_v23.items} & write.V24_ONLY_FRAMES
    priv = next(item for item in to_v23.items if item.key == "PRIV:com.example")
    assert priv.payload == next(item for item in source if item.key == "PRIV:com.example").payload
    same = write.prepare_store("id3v2", (), passthrough, id3_version="2.4")
    by_source = {item.identity: item for item in passthrough}
    for identity, ids in same.passthrough_ids.items():
        (expected,) = [item for item in same.items if item.identity in ids]
        assert signature(expected) == signature(by_source[identity])  # same version: unchanged evidence


def test_v22_and_v23_sources_upgrade_to_the_target() -> None:
    v23 = id3_source([TBPM(encoding=1, text=["99"])], version=3)
    prepared = write.prepare_store("id3v2", (), v23, id3_version="2.4")
    (item,) = prepared.items
    assert item.native_version == "2.4" and json.loads(item.payload)["frame"] == "TBPM"
    v22 = RawNativeItem("store:id3v2:TBP", "id3v2", "TBP", "TBP", "2.2", JSON,
                        b'{"fields":[["encoding",0],["text",["88"]]],"frame":"TBP"}', "ID3@0+10")
    prepared = write.prepare_store("id3v2", (), (v22,), id3_version="2.3")
    (item,) = prepared.items
    assert item.key == "TBPM" and item.native_version == "2.3" and prepared.passthrough_ids == {v22.identity: (item.identity,)}


def test_fall_back_and_cross_family_items_are_never_passed_through() -> None:
    raw = [item for item in id3_source([RVA2(desc="track", channel=1, gain=-3.25, peak=0.875)])]
    assert raw[0].payload_encoding == RAW
    with pytest.raises(ValueError, match="evidence"):
        write.prepare_store("id3v2", (), raw)
    vorbis = RawNativeItem("store:vorbis:MOOD", "vorbis", "MOOD", "VORBIS_COMMENT", None, RAW, b"calm", "x")
    with pytest.raises(ValueError, match="family"):
        write.prepare_store("id3v2", (), (vorbis,))


def test_duplicate_passthrough_keys_are_dropped_not_merged() -> None:
    first = id3_source([TBPM(encoding=3, text=["120"])])[0]
    second = RawNativeItem("store:id3v2:TBPM#2", "id3v2", "TBPM", "TBPM", "2.4", JSON,
                           b'{"fields":[["encoding",3],["text",["121"]]],"frame":"TBPM"}', "x")
    prepared = write.prepare_store("id3v2", (), (first, second), id3_version="2.4")
    assert prepared.passthrough_ids[first.identity] and prepared.passthrough_ids[second.identity] == ()
    assert [json.loads(item.payload)["fields"][1][1] for item in prepared.items] == [["120"]]
    recipes = mapping.native_recipes("id3v2", {"title": "T"})
    with pytest.raises(ValueError, match="share one native key"):
        write.prepare_store("id3v2", tuple(recipes) * 2)


def test_vorbis_passthrough_that_cannot_be_written_gets_no_link() -> None:
    good = RawNativeItem("store:vorbis:MOOD", "vorbis", "Mood", "VORBIS_COMMENT", None, RAW, "ruhig—".encode(), "a")
    invalid = RawNativeItem("store:vorbis:BAD", "vorbis", "BAD", "VORBIS_COMMENT", None, RAW, b"\xff\xfe", "b")
    prepared = write.prepare_store("vorbis", (), (good, invalid))
    assert prepared.passthrough_ids[invalid.identity] == ()
    (item,) = prepared.items
    assert item.key == "Mood" and item.payload == "ruhig—".encode() and prepared.passthrough_ids[good.identity]


def test_mp4_passthrough_atoms_are_kept_equal(audio) -> None:
    path = audio("alac")
    tags = MP4(path)
    tags["tmpo"] = [120]
    tags["©lyr"] = ["la la"]
    tags["soar"] = ["Composer"]
    tags.save()
    source = read_source_tags(path, "mp4").native_inventory
    prepared = write.prepare_store("mp4", (), source)
    assert sorted(map(signature, prepared.items)) == sorted(map(signature, source))
    assert all(len(ids) == 1 for ids in prepared.passthrough_ids.values())


# --- RIFF INFO (MD29, MD30) ----------------------------------------------------------------------------


def test_info_items_and_render_pairs() -> None:
    fields = {"title": "Perché 東京", "date": "2026-10", "track": 3, "comment": "c", "composer": "not in INFO"}
    recipes = mapping.native_recipes("riff-info", fields)
    copyright_item = RawNativeItem("store:riff-info:ICOP", "riff-info", "ICOP", "ICOP", None, RAW, b"(c) D\x00", "x")
    other = RawNativeItem("store:riff-info:IENG", "riff-info", "IENG", "IENG", None, RAW, b"E\x00", "y")
    prepared = write.prepare_store("riff-info", recipes, (copyright_item, other))
    assert {item.key: item.payload for item in prepared.items} == {
        "INAM": "Perché 東京".encode() + b"\x00", "ICRD": b"2026-10\x00", "IPRT": b"3\x00", "ICMT": b"c\x00",
        "ICOP": b"(c) D\x00"}
    assert prepared.passthrough_ids == {copyright_item.identity: ("store:riff-info:ICOP",), other.identity: ()}
    assert write.riff_info_metadata(prepared.items) == (
        ("title", "Perché 東京"), ("date", "2026-10"), ("track", "3"), ("comment", "c"), ("copyright", "(c) D"))
    with pytest.raises(ValueError):
        write.riff_info_metadata((other,))
    again = RawNativeItem("store:riff-info:ICOP#2", "riff-info", "ICOP", "ICOP", None, RAW, b"(c) E\x00", "z")
    repeated = write.prepare_store("riff-info", (), (copyright_item, again))
    assert [item.payload for item in repeated.items] == [b"(c) D\x00"] and repeated.passthrough_ids[again.identity] == ()


def test_render_writes_exactly_the_planned_info_items(encode) -> None:
    prepared = write.prepare_store("riff-info", mapping.native_recipes(
        "riff-info", {"title": "Perché 東京 🎵", "artist": "A", "date": "2026-10-05", "track": 12, "genre": "G"}))
    args = [arg for key, value in write.riff_info_metadata(prepared.items) for arg in ("-metadata", f"{key}={value}")]
    path = encode("wav", "info.wav", "-c:a", "pcm_s24le", *args)
    found = read_source_tags(path, "wav").native_inventory
    assert sorted(map(signature, found)) == sorted(map(signature, prepared.items))


def test_unknown_store_and_version_are_refused() -> None:
    with pytest.raises(ValueError):
        write.prepare_store("apev2", ())
    with pytest.raises(ValueError):
        write.prepare_store("id3v2", (), id3_version="2.2")
    with pytest.raises(ValueError):
        native_items_from_bytes("id3v2", b"not a tag")
