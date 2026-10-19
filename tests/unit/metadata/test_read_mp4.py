"""MP4 ``ilst`` atoms and MP4 chapter sources (TST-M-01 readers slice; MD3, MD5, MD7, MD32, MD38)."""

from __future__ import annotations

import base64
import json
import struct

import pytest
from mutagen.mp4 import MP4, AtomDataType, MP4Cover, MP4FreeForm

from norma.metadata.read import JSON, RAW, read_processing_mark, read_source_tags

JPEG = b"\xff\xd8\xff\xe0" + bytes(20)
PNG = b"\x89PNG\r\n\x1a\n" + bytes(20)
FIXTURE_MARK = (
    "v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=limit;gain=+5.80;limited=2.60;out_i=-16.02;"
    "out_tp=-1.04;origin=pcm_s24le;ffmpeg=N-125892;date=2026-10-05"
)


def by_identity(tags) -> dict:
    return {item.identity: item for item in tags.native_inventory}


def ilst_names(data: bytes) -> list[bytes]:
    """Independent walk to moov/udta/meta/ilst: the stored child atom names, in order."""
    def children(start: int, end: int):
        position = start
        while position + 8 <= end:
            size, name = struct.unpack(">I4s", data[position:position + 8])
            yield name, position + 8, position + size
            position += size

    span = (0, len(data))
    for name in (b"moov", b"udta", b"meta", b"ilst"):
        span = next((s, e) for n, s, e in children(*span) if n == name)
        if name == b"meta":
            span = (span[0] + 4, span[1])
    return [name for name, _, _ in children(*span)]


@pytest.mark.parametrize("kind", ["alac", "aac"])
def test_mp4_atoms_written_by_mutagen(kind: str, audio) -> None:
    path = audio(kind)
    mp4 = MP4(path)
    mp4["©nam"] = ["Night Bus"]
    mp4["©ART"] = ["Anna", "Test Artist"]
    mp4["aART"], mp4["©alb"], mp4["©gen"], mp4["©wrt"] = ["Norma fixtures"], ["Demo set"], ["Demo"], ["Composer"]
    mp4["©cmt"] = ["Line one\nLine two"]
    mp4["©day"] = ["2026-10-05T07:00:00Z"]
    mp4["trkn"], mp4["disk"], mp4["cpil"], mp4["tmpo"] = [(0, 12)], [(1, 2)], True, [120]
    mp4["covr"] = [MP4Cover(JPEG, imageformat=MP4Cover.FORMAT_JPEG), MP4Cover(PNG, imageformat=MP4Cover.FORMAT_PNG)]
    mp4["----:com.apple.iTunes:BPM"] = [MP4FreeForm(b"120")]
    mp4["----:com.apple.iTunes:iTunSMPB"] = [MP4FreeForm(b" 00000000 00000840")]
    mp4["----:com.apple.iTunes:BLOB"] = [MP4FreeForm(b"\x00\x01", dataformat=AtomDataType.IMPLICIT)]
    mp4["----:com.apple.iTunes:NORMA"] = [MP4FreeForm(FIXTURE_MARK.encode())]
    mp4.save()
    tags = read_source_tags(path, "mp4")
    assert tuple(tags.stores) == ("mp4",)
    values = {name: tag.value for name, tag in tags.fields.items()}
    assert values == {"title": "Night Bus", "artist": "Anna; Test Artist", "album_artist": "Norma fixtures",
                      "album": "Demo set", "date": "2026-10-05T07:00:00Z", "track_total": 12, "disc": 1,
                      "disc_total": 2, "genre": "Demo", "composer": "Composer", "comment": "Line one\nLine two",
                      "compilation": True}
    assert {name: tag.value for name, tag in tags.extras.items()} == {"BPM": "120"}
    items = by_identity(tags)
    assert json.loads(items["store:mp4:trkn"].payload) == {"atom": "trkn", "scalar": False,
                                                           "values": [{"pair": [0, 12]}]}  # stored 0 kept (MD32)
    assert json.loads(items["store:mp4:cpil"].payload) == {"atom": "cpil", "scalar": True, "values": [True]}
    assert json.loads(items["store:mp4:----:com.apple.iTunes:BLOB"].payload)["values"] == [
        {"freeform": base64.b64encode(b"\x00\x01").decode(), "dataformat": 0, "version": 0}]
    covers = [items["store:mp4:covr#1"], items["store:mp4:covr#2"]]
    assert [json.loads(item.payload)["values"][0]["imageformat"] for item in covers] == [13, 14]
    assert [(tag.value.mime, tag.value.data, tag.value.picture_type) for tag in tags.artwork] == \
        [("image/jpeg", JPEG, 3), ("image/png", PNG, 3)]
    assert {item.payload_encoding for item in tags.native_inventory} == {JSON}
    # independent: the stored ilst atoms are all inventoried, in stored order (covr once per value)
    names = [name.decode("latin-1") for name in ilst_names(path.read_bytes())]
    inventoried = [item.native_type for item in tags.native_inventory]
    assert [name for i, name in enumerate(inventoried) if not (name == "covr" and inventoried[i - 1] == "covr")] \
        == names
    assert read_processing_mark(tags).validity == "valid"


def test_mp4_payloads_rebuild_the_same_atoms(audio) -> None:
    first = audio("alac")
    mp4 = MP4(first)
    mp4["©nam"], mp4["trkn"], mp4["cpil"], mp4["tmpo"] = ["x"], [(3, 12)], False, [90]
    mp4["covr"] = [MP4Cover(PNG, imageformat=MP4Cover.FORMAT_PNG)]
    mp4["----:com.apple.iTunes:MOOD"] = [MP4FreeForm(b"calm")]
    mp4.save()
    source = read_source_tags(first, "mp4")
    second = audio("alac")
    rebuilt = MP4(second)
    for item in source.native_inventory:
        document = json.loads(item.payload)
        values = []
        for value in document["values"]:
            if isinstance(value, dict) and "pair" in value:
                values.append(tuple(value["pair"]))
            elif isinstance(value, dict) and "cover" in value:
                values.append(MP4Cover(base64.b64decode(value["cover"]), imageformat=value["imageformat"]))
            elif isinstance(value, dict) and "freeform" in value:
                values.append(MP4FreeForm(base64.b64decode(value["freeform"]), dataformat=value["dataformat"],
                                          version=value["version"]))
            else:
                values.append(value)
        key = document["atom"]
        rebuilt[key] = values[0] if document["scalar"] else (rebuilt.get(key, []) + values if key == "covr" else values)
    rebuilt.save()
    again = read_source_tags(second, "mp4")
    def comparable(tags):
        return sorted((i.identity, i.payload) for i in tags.native_inventory if i.native_type != "©too")
    assert comparable(again) == comparable(source)


def test_mp4_chapters_from_ffmpeg_feed_one_aggregate(audio, tmp_path, encode) -> None:
    chapters = tmp_path / "chapters.txt"
    chapters.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=100\ntitle=One\n"
                        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=100\nEND=200\ntitle=Two\n", encoding="utf-8")
    path = tmp_path / "chapters.m4a"
    source = audio("alac")
    from norma.adapters import ffmpeg
    executable = ffmpeg.locate_ffmpeg()
    ffmpeg.check_result(ffmpeg.run_process(ffmpeg.ffmpeg_argv(
        executable, "-y", "-i", str(source), "-i", str(chapters), "-map", "0:a", "-map_chapters", "1",
        "-c", "copy", str(path)), executable=executable, deadline_s=60))
    tags = read_source_tags(path, "mp4")
    assert [item.identity for item in tags.non_tag_items] == ["item:chapters"]
    document = json.loads(tags.non_tag_items[0].payload)
    sources = {source["source"]: source["chapters"] for source in document["sources"]}
    assert list(sources) == ["mp4-track", "mp4-chpl"]
    assert [chapter["title"] for chapter in sources["mp4-track"]] == ["One", "Two"]
    start, scale = sources["mp4-track"][1]["start"]
    assert start / scale == pytest.approx(0.1)
    assert [chapter["title"] for chapter in sources["mp4-chpl"]] == ["One", "Two"]
    assert sources["mp4-chpl"][1]["start"] == [1_000_000, 10_000_000]
    assert tags.non_tag_items[0].locator == "mp4-track,mp4-chpl"


def test_unparsed_atoms_keep_the_stored_atom(audio) -> None:
    path = audio("alac")
    mp4 = MP4(path)
    mp4["©nam"] = ["x"]
    mp4.save()
    data = bytearray(path.read_bytes())
    # mark ©nam's data as UTF-16 (type 2): a UTF-8 dn-json-1 string could not keep that stored type
    index = data.index(b"\xa9nam")
    data_index = data.index(b"data", index)
    data[data_index + 4:data_index + 8] = b"\x00\x00\x00\x02"
    path.write_bytes(bytes(data))
    tags = read_source_tags(path, "mp4")
    item = by_identity(tags)["store:mp4:©nam"]
    assert item.payload_encoding == RAW
    assert item.payload[4:8] == b"\xa9nam" and item.payload in path.read_bytes()


def test_untagged_mp4_has_no_ilst_store(audio) -> None:
    tags = read_source_tags(audio("aac"), "mp4")
    assert "mp4" not in tags.stores or tags.stores["mp4"] == ()
