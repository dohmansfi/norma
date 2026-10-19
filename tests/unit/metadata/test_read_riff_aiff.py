"""RIFF INFO, AIFF text chunks and non-tag chunks (TST-M-01 readers slice; MD3, MD7; formats M4, M5)."""

from __future__ import annotations

import hashlib
import struct

import pytest
from mutagen.aiff import AIFF
from mutagen.id3 import TIT2, TPE1
from mutagen.wave import WAVE

from norma.metadata.read import PAYLOAD_LIMIT, RAW, read_source_tags


def by_identity(tags) -> dict:
    return {item.identity: item for item in tags.native_inventory}


def riff_walk(data: bytes) -> list[bytes]:
    """Independent top-level chunk list of a RIFF or FORM file."""
    order = "<I" if data[:4] == b"RIFF" else ">I"
    ids, position = [], 12
    while position + 8 <= len(data):
        size = struct.unpack(order, data[position + 4:position + 8])[0]
        ids.append(data[position:position + 4])
        position += 8 + size + (size & 1)
    return ids


def test_ffmpeg_info_subset_is_read_as_riff_info(encode) -> None:
    path = encode("wav", "info.wav", "-metadata", "title=Perché 東京", "-metadata", "artist=Test Artist",
                  "-metadata", "date=2026-10", "-metadata", "track=3", "-metadata", "comment=note",
                  "-metadata", "copyright=© 2026")
    tags = read_source_tags(path, "wav")
    assert tuple(tags.stores) == ("riff-info",)
    items = by_identity(tags)
    assert items["store:riff-info:INAM"].payload == "Perché 東京".encode() + b"\x00"  # chunk data, pad byte excluded
    values = {name: tag.value for name, tag in tags.fields.items()}
    assert values == {"title": "Perché 東京", "artist": "Test Artist", "date": "2026-10", "track": 3, "comment": "note"}
    assert tags.extras["ICOP"].value == "© 2026"  # MD19: INFO IDs with a valid extra name
    assert {item.payload_encoding for item in tags.native_inventory} == {RAW}
    assert b"LIST" in riff_walk(path.read_bytes())


def test_info_and_id3_stores_stay_separate_with_conflicts(encode) -> None:
    path = encode("wav", "conflict.wav", "-metadata", "title=INFO title", "-metadata", "artist=Same")
    wave = WAVE(path)
    wave.add_tags()
    wave.tags.add(TIT2(encoding=3, text=["ID3 title"]))
    wave.tags.add(TPE1(encoding=3, text=["Same"]))
    wave.save(v2_version=3)
    tags = read_source_tags(path, "wav")
    assert tuple(tags.stores) == ("id3v2", "riff-info")
    assert tags.fields["title"].value == "ID3 title" and tags.fields["title"].system == "id3v2"
    assert [(c.key, c.old_value, c.new_value, c.old_source, c.winning_source) for c in tags.internal_conflicts] == \
        [("field:title", "INFO title", "ID3 title", "source:store:riff-info:INAM", "source")]
    assert {"store:riff-info:INAM", "store:id3v2:TIT2"} <= set(by_identity(tags))
    chunk_ids = riff_walk(path.read_bytes())
    assert b"LIST" in chunk_ids and (b"id3 " in chunk_ids or b"ID3 " in chunk_ids)


def test_hand_built_chunks_become_non_tag_items(audio, build) -> None:
    path = audio("wav")
    bext = b"Norma fixture".ljust(602, b"\x00")
    big = bytes(PAYLOAD_LIMIT + 1)
    build.append_chunks(path, [
        build.riff_chunk(b"bext", bext), build.riff_chunk(b"cue ", struct.pack("<I", 0)),
        build.riff_chunk(b"iXML", b"<BWFXML/>"), build.riff_chunk(b"iXML", b"<second/>"),
        build.riff_chunk(b"LIST", b"adtl" + build.riff_chunk(b"labl", b"\x01\x00\x00\x00x\x00")),
        build.info_list([(b"INAM", b"odd\x00"), (b"ICMT", b"bad \xff utf8\x00"), (b"ISFT", b"Lavf\x00")]),
        build.riff_chunk(b"junk", big),
    ])
    tags = read_source_tags(path, "wav")
    nontag = {item.identity: item for item in tags.non_tag_items}
    assert list(nontag) == ["item:chunk:bext", "item:chunk:cue ", "item:chunk:iXML#1", "item:chunk:iXML#2",
                            "item:chunk:LIST", "item:chunk:junk"]
    assert nontag["item:chunk:bext"].payload == bext
    assert nontag["item:chunk:bext"].locator.startswith("bext@")
    assert {item.kind for item in nontag.values()} == {"chunk"}
    assert {item.detail for item in nontag.values()} == {"not_carried_v1"}
    over = nontag["item:chunk:junk"]
    assert (over.payload, over.payload_size, over.payload_sha256) == (None, len(big), hashlib.sha256(big).hexdigest())
    assert nontag["item:chunk:LIST"].payload[:4] == b"adtl"
    assert tags.fields["title"].value == "odd"
    assert "comment" not in tags.fields  # INFO bytes that are not UTF-8 are invalid evidence (formats M5)
    assert by_identity(tags)["store:riff-info:ICMT"].payload == b"bad \xff utf8\x00"
    assert "ISFT" not in tags.extras  # an MD6 tool tag is never an extra


def test_rf64_data_size_comes_from_ds64(encode) -> None:
    path = encode("wav", "large.wav", "-rf64", "always", "-metadata", "title=RF64")
    assert path.read_bytes()[:4] == b"RF64"
    tags = read_source_tags(path, "wav")
    assert tags.fields["title"].value == "RF64"
    assert tags.non_tag_items == ()  # ds64, fmt and data are structure


def test_aiff_text_chunks_and_id3_chunk(audio, build) -> None:
    path = audio("aiff")
    aiff = AIFF(path)
    aiff.add_tags()
    aiff.tags.add(TIT2(encoding=3, text=["ID3 title"]))
    aiff.save(v2_version=3)
    build.append_chunks(path, [
        build.riff_chunk(b"NAME", b"Text title", "big"), build.riff_chunk(b"AUTH", b"Author", "big"),
        build.riff_chunk(b"(c) ", b"2026 Test Artist", "big"), build.riff_chunk(b"ANNO", b"first", "big"),
        build.riff_chunk(b"ANNO", b"second", "big"), build.riff_chunk(b"MARK", b"\x00\x00", "big"),
    ])
    tags = read_source_tags(path, "aiff")
    assert tuple(tags.stores) == ("id3v2", "aiff-text")
    assert tags.stores["aiff-text"] == ("store:aiff-text:NAME", "store:aiff-text:AUTH", "store:aiff-text:(c) ",
                                        "store:aiff-text:ANNO#1", "store:aiff-text:ANNO#2")
    assert tags.fields["title"].value == "ID3 title"
    assert tags.fields["artist"].value == "Author"
    assert tags.fields["comment"].value == "first; second"
    assert [c.old_source for c in tags.internal_conflicts] == ["source:store:aiff-text:NAME"]
    assert [item.identity for item in tags.non_tag_items] == ["item:chunk:MARK"]
    assert b"MARK" in riff_walk(path.read_bytes())


@pytest.mark.parametrize("kind", ["wav", "aiff"])
def test_untagged_pcm_has_no_stores(kind: str, audio, build) -> None:
    tags = read_source_tags(audio(kind), build.container(kind))
    assert (dict(tags.stores), tags.fields, tags.native_inventory, tags.non_tag_items) == ({}, {}, (), ())
