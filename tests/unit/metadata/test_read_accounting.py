"""CR-082: rejected occurrences and exact component evidence from the edge."""

import hashlib

import pytest

from norma.metadata import read, resolve, write
from norma.models import ManifestCell


def make_plan(tags, *, target="flac", strict=True):
    resolution = resolve.resolve_opinions(tags, stem="fixture", no_title_fallback=True, strict_conflicts=strict)
    requests = resolve.preparation_requests(tags, resolution, target=target)
    prepared = tuple(write.prepare_store(store, request["recipes"], request["passthrough"])
                     for store, request in requests.items())
    return resolve.build_metadata_plan(tags, resolution, target=target, prepared_stores=prepared,
                                       source_field_ids=read.source_field_ids(tags))


@pytest.mark.parametrize("name,invalid", [("TITLE", b"bad\xff"), ("TITLE", b"bad\x01"),
    ("TITLE", b"bad\nline"), ("TITLE", b"x" * 1001), ("Mood", b"bad\xff"), ("Mood", b"bad\x01")])
def test_cr082_invalid_among_valid_is_one_internal_conflict_and_one_native_fate(audio, build, name, invalid):
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    comments = [name.encode() + b"=good", name.encode() + b"=" + invalid]
    build.write_flac_blocks(path, prefix, [b for b in blocks if b[0] != 4] + [(4, build.vorbis_block(comments))], frames)
    before = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
    tags = read.read_source_tags(path, "flac")
    projected = tags.fields if name == "TITLE" else tags.extras
    assert next(iter(projected.values())).value == "good"
    assert len(tags.internal_conflicts) == 1
    conflict = tags.internal_conflicts[0]
    assert conflict.detail == "invalid_source_value" and conflict.old_source.endswith("#2")
    result = make_plan(tags)
    rejected = conflict.old_source
    rows = [row for row in result.native_statuses if row.key == rejected]
    assert len(rows) == 1 and rows[0].status == "changed" and rows[0].detail == "invalid_source_value"
    assert result.conflicts == ()  # Internal conflicts never trip strict mode.
    assert len(result.statuses) == 1 and result.loss_policy_result == "ok"
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("name,value,fields", [("TRACKNUMBER", b"0", ()), ("TRACKNUMBER", b"/12", ("track_total",)),
    ("TRACKNUMBER", b"3", ("track",)), ("TRACKNUMBER", b"3/12", ("track", "track_total")),
    ("DISCNUMBER", b"0/2", ("disc_total",)), ("TRACKNUMBER", b"13/12", ("track", "track_total")),
    ("TRACKNUMBER", b"A1", ("track",))])
def test_exact_component_indices_never_invent_absent_zero(audio, build, name, value, fields):
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    build.write_flac_blocks(path, prefix, [b for b in blocks if b[0] != 4] + [(4, build.vorbis_block([name.encode() + b"=" + value]))], frames)
    tags = read.read_source_tags(path, "flac")
    index = read.source_field_ids(tags)
    assert set(index) == {"field:" + field for field in fields}
    assert all(ids == ("store:vorbis:" + name,) for ids in index.values())


@pytest.mark.parametrize("text,fields", [("3", ("track",)), ("0/12", ("track_total",)),
    ("3/12", ("track", "track_total")), ("A1", ("track",))])
def test_id3_component_indices(audio, build, text, fields):
    path = audio("mp3")
    original = path.read_bytes()
    # Remove the empty render ID3 tag before adding the independently built tag.
    length = 10 + read._syncsafe(original[6:10]) if original[:3] == b"ID3" else 0
    path.write_bytes(build.id3_tag(4, [build.id3_frame("TRCK", b"\x03" + text.encode())]) + original[length:])
    tags = read.read_source_tags(path, "mp3")
    assert set(read.source_field_ids(tags)) == {"field:" + field for field in fields}


def test_mp4_total_only_component_indices(audio):
    from mutagen.mp4 import MP4
    path = audio("alac")
    tags = MP4(path)
    tags["trkn"] = [(0, 12)]
    tags.save()
    src = read.read_source_tags(path, "mp4")
    assert read.source_field_ids(src) == {"field:track_total": ("store:mp4:trkn",)}
    result = make_plan(src, target="alac")
    assert set(result.resolved) == {"track_total"} and result.effective["track_total"].value == 12


def test_invalid_only_field_is_raw_evidence_not_a_canonical_value(audio, build):
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    build.write_flac_blocks(path, prefix, [b for b in blocks if b[0] != 4] + [(4, build.vorbis_block([b"TITLE=bad\xff"]))], frames)
    src = read.read_source_tags(path, "flac")
    result = make_plan(src)
    assert result.resolved == {} and result.effective == {}
    row = result.statuses[0]
    assert row.key == "field:title" and row.status == "dropped_format_limit" and row.detail == "invalid_source_value"
    assert row.requested.scope == "source_native"


def test_valid_explicit_replaces_invalid_evidence_not_a_loss(audio, build):
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    build.write_flac_blocks(path, prefix, [b for b in blocks if b[0] != 4] + [(4, build.vorbis_block([b"TRACKNUMBER=A1"]))], frames)
    src = read.read_source_tags(path, "flac")
    resolution = resolve.resolve_opinions(src, stem="fixture", no_title_fallback=True, manifest={"track": ManifestCell("set", 3)})
    request = resolve.preparation_requests(src, resolution, target="flac")["vorbis"]
    prepared = write.prepare_store("vorbis", request["recipes"])
    result = resolve.build_metadata_plan(src, resolution, target="flac", prepared_stores=(prepared,), source_field_ids=read.source_field_ids(src))
    assert result.loss_policy_result == "ok" and result.native_statuses[0].detail == "superseded"
