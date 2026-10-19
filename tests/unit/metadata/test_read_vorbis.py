"""Vorbis comment, FLAC PICTURE and CUESHEET readers (TST-M-01 readers slice; MD3, MD5, MD7, MD32)."""

from __future__ import annotations

import hashlib
import json
import struct

from mutagen.flac import FLAC, Picture

from norma.metadata.read import RAW, parse_flac_picture, read_processing_mark, read_source_tags

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(40))


def by_identity(tags) -> dict:
    return {item.identity: item for item in tags.native_inventory}


def test_vorbis_comments_written_by_mutagen(audio) -> None:
    path = audio("flac")
    flac = FLAC(path)
    for key, value in {"TITLE": "Perché 東京 🎵", "ARTIST": ["Anna", "Test Artist"], "ALBUMARTIST": "Norma fixtures",
                       "ALBUM": "Demo set", "DATE": "2026-10", "TRACKNUMBER": "03", "TRACKTOTAL": "12",
                       "DISCNUMBER": "1/2", "GENRE": "Demo", "COMPOSER": "Composer", "COMMENT": "a\nb",
                       "COMPILATION": "true", "Mood": "calm", "REPLAYGAIN_TRACK_GAIN": "-3.00 dB",
                       "ENCODER": "old"}.items():
        flac[key] = value
    picture = Picture()
    picture.type, picture.mime, picture.desc, picture.width, picture.height, picture.depth, picture.data = \
        3, "image/png", "front", 64, 48, 24, PNG
    flac.add_picture(picture)
    flac.save()
    tags = read_source_tags(path, "flac")
    assert tuple(tags.stores) == ("vorbis",)
    values = {name: tag.value for name, tag in tags.fields.items()}
    assert values == {"title": "Perché 東京 🎵", "artist": "Anna; Test Artist", "album_artist": "Norma fixtures",
                      "album": "Demo set", "date": "2026-10", "track": 3, "track_total": 12, "disc": 1,
                      "disc_total": 2, "genre": "Demo", "composer": "Composer", "comment": "a\nb",
                      "compilation": True}
    assert {name: tag.value for name, tag in tags.extras.items()} == {"Mood": "calm"}
    items = by_identity(tags)
    assert "store:vorbis:ARTIST#1" in items and "store:vorbis:ARTIST#2" in items
    assert items["store:vorbis:MOOD"].key == "Mood"  # identity upper-cases, key keeps the stored spelling
    assert items["store:vorbis:TITLE"].payload == "Perché 東京 🎵".encode()
    assert {item.payload_encoding for item in tags.native_inventory} == {RAW}
    # independent: mutagen's own comment list matches the inventory, in order
    comments = [(key.upper(), value.encode()) for key, value in FLAC(path).tags]
    assert comments == [(item.key.upper(), item.payload) for item in tags.native_inventory
                        if item.native_type == "VORBIS_COMMENT"]
    art = tags.artwork[0].value
    assert (art.data, art.mime, art.width_px, art.height_px, art.picture_type, art.description) == \
        (PNG, "image/png", 64, 48, 3, "front")
    picture_item = items["store:vorbis:PICTURE"]
    assert picture_item.payload == FLAC(path).pictures[0].write()  # the PICTURE body as stored


def test_invalid_utf8_stays_raw_and_is_not_projected(audio, build) -> None:
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    comments = [b"TITLE=caf\xe9", b"ARTIST=fine", b"NOEQUALS", b"DATE=Spring 2026", b"TRACKNUMBER=A1"]
    blocks = [block for block in blocks if block[0] != 4] + [(4, build.vorbis_block(comments))]
    build.write_flac_blocks(path, prefix, blocks, frames)
    tags = read_source_tags(path, "flac")
    items = by_identity(tags)
    assert items["store:vorbis:TITLE"].payload == b"caf\xe9"  # never replaced on reading
    assert set(tags.fields) == {"artist"}
    assert items["store:vorbis:NOEQUALS"].payload == b""
    assert len(tags.native_inventory) == 5


def test_vorbis_total_ranking_and_conflicts(audio, build) -> None:
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    comments = [b"TRACKNUMBER=3/10", b"TOTALTRACKS=11", b"TRACKTOTAL=12", b"DISCNUMBER=5/2"]
    blocks = [block for block in blocks if block[0] != 4] + [(4, build.vorbis_block(comments))]
    build.write_flac_blocks(path, prefix, blocks, frames)
    tags = read_source_tags(path, "flac")
    assert (tags.fields["track"].value, tags.fields["track_total"].value) == (3, 12)
    assert tags.fields["track_total"].locator == "store:vorbis:TRACKTOTAL"
    assert sorted((c.key, c.old_value, c.old_source) for c in tags.internal_conflicts) == [
        ("field:track_total", 10, "source:store:vorbis:TRACKNUMBER"),
        ("field:track_total", 11, "source:store:vorbis:TOTALTRACKS"),
    ]
    assert tags.fields["disc"].value == 5 and "disc_total" not in tags.fields  # n > N keeps n (MD5)


def test_track_total_alone_and_zero_numbers(audio, build) -> None:
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    comments = [b"TRACKNUMBER=0", b"TRACKTOTAL=12", b"DISCTOTAL=0"]
    blocks = [block for block in blocks if block[0] != 4] + [(4, build.vorbis_block(comments))]
    build.write_flac_blocks(path, prefix, blocks, frames)
    tags = read_source_tags(path, "flac")
    assert {name: tag.value for name, tag in tags.fields.items()} == {"track_total": 12}


def test_flac_cuesheet_and_vorbis_chapters_feed_one_aggregate(audio, build) -> None:
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    track = lambda offset, number: struct.pack(">QB", offset, number) + bytes(12) + b"\x00" + bytes(13) + b"\x01" \
        + struct.pack(">QB", 0, 1) + bytes(3)  # noqa: E731
    cuesheet = bytes(128) + struct.pack(">Q", 0) + b"\x00" + bytes(258) + bytes([3]) + track(0, 1) + track(4800, 2) \
        + track(9600, 255)[:36]
    comments = [b"CHAPTER001=00:00:00.000", b"CHAPTER001NAME=Intro", b"CHAPTER002=00:00:00.100"]
    blocks = [b for b in blocks if b[0] != 4] + [(4, build.vorbis_block(comments)), (5, cuesheet)]
    build.write_flac_blocks(path, prefix, blocks, frames)
    tags = read_source_tags(path, "flac")
    assert [item.identity for item in tags.non_tag_items] == ["item:chapters"]
    chapters = tags.non_tag_items[0]
    assert chapters.kind == "chapters" and chapters.detail == "not_carried_v1"
    assert chapters.payload_sha256 == hashlib.sha256(chapters.payload).hexdigest()
    document = json.loads(chapters.payload)
    assert [source["source"] for source in document["sources"]] == ["vorbis", "flac-cuesheet"]
    assert document["sources"][0]["chapters"] == [
        {"start": {"text": "00:00:00.000"}, "end": None, "title": "Intro", "native": "store:vorbis:CHAPTER001"},
        {"start": {"text": "00:00:00.100"}, "end": None, "title": None, "native": "store:vorbis:CHAPTER002"},
    ]
    assert [chapter["start"] for chapter in document["sources"][1]["chapters"]] == [[0, 48000], [4800, 48000]]
    assert tags.extras == {}  # chapter comments are never extras
    assert chapters.payload == json.dumps(document, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False).encode()


def test_vorbis_mark(audio) -> None:
    path = audio("flac")
    flac = FLAC(path)
    flac["norma"] = "v=1;target=-14;tp=-1;gain=0;mode=linear"
    flac.save()
    mark = read_processing_mark(read_source_tags(path, "flac"))
    assert (mark.validity, mark.locator) == ("malformed", "store:vorbis:NORMA")


def test_malformed_picture_body_is_evidence_only(audio, build) -> None:
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    build.write_flac_blocks(path, prefix, blocks + [(6, b"\x00\x00\x00\x03junk")], frames)
    tags = read_source_tags(path, "flac")
    assert tags.artwork == ()
    assert by_identity(tags)["store:vorbis:PICTURE"].payload == b"\x00\x00\x00\x03junk"
    assert parse_flac_picture(b"") is None
