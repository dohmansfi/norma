"""IC-4 CR-098 to CR-100 and MD32 duplicate-order regressions (TK-0615)."""
from __future__ import annotations

import json
import struct

import pytest
from mutagen.apev2 import APEv2
from mutagen.flac import FLAC
from mutagen.id3 import ID3
from mutagen.aiff import AIFF
from mutagen.wave import WAVE

from norma.errors import FileFailure
from norma.metadata import read, write
from norma.models import ManifestCell
from tests.unit.metadata.test_resolve import plan
from tests.unit.metadata.test_write import candidate, expected_for, plan_for


@pytest.mark.parametrize("kind,key,value", [
    ("mp3", "Cover Art (Front)", b"cover.jpg\0\xff\xd8\xff"),
    ("mp3", "bad name!", "unprojected text"),
    ("mp3", "BPM", b"binary not an extra"),
    ("mp3", "Title", b"binary not a title"),
    ("mp3", "Year", "not a date"),
    ("aiff", "(c) ", b"Aiff copyright"),
])
@pytest.mark.parametrize("policy", ["warn", "sidecar", "error"])
def test_unprojected_legacy_occurrence_obeys_every_loss_policy(audio, kind, key, value, policy):
    path = audio(kind)
    if kind == "mp3":
        tags = APEv2()
        tags[key] = value
        tags.save(path)
        identity = "store:apev2:" + key
    else:
        data = path.read_bytes()
        data += key.encode("ascii") + struct.pack(">I", len(value)) + value + (b"\0" if len(value) & 1 else b"")
        path.write_bytes(data[:4] + struct.pack(">I", len(data) - 8) + data[8:])
        identity = "store:aiff-text:" + key
    source = read.read_source_tags(path, kind)
    if policy == "error":
        with pytest.raises(FileFailure) as caught:
            plan(source, target="flac", on_metadata_loss=policy)
        assert caught.value.reason == "metadata_loss_error"
        assert caught.value.stage == "metadata_planned"
        assert identity.removeprefix("store:") in caught.value.detail
        return
    result, _ = plan(source, target="flac", on_metadata_loss=policy)
    rollup = next(row for row in result.statuses if identity in row.source_native_ids)
    native = next(row for row in result.native_statuses if row.key == "source:" + identity)
    assert (rollup.status, rollup.loss_kind, rollup.detail) == (
        "dropped_format_limit", "whole_value", "legacy_store_not_carried")
    assert (native.status, native.loss_kind, native.detail) == (
        "dropped_format_limit", "whole_value", "legacy_store_not_carried")
    assert native.expected_ids == () and rollup.expected_native_ids == ()
    assert result.loss_policy_result == policy
    assert any(d.code == "metadata_loss_warn" for d in result.diagnostics)
    if policy == "sidecar":
        sidecar = json.loads(write.stage_members(result, audio_name="take.flac")["tags"])
        assert any(row["key"] == rollup.key for row in sidecar["losses"])
        assert sidecar["losses"][0]["requested"]["items"][0]["identity"] == identity


@pytest.mark.parametrize("key,value", [("Title", "APE title"), ("BPM", "120"), ("Year", "2026")])
def test_projected_legacy_occurrences_still_removed_by_design(audio, key, value):
    path = audio("mp3")
    tags = APEv2()
    tags[key] = value
    tags.save(path)
    result, _ = plan(read.read_source_tags(path, "mp3"), target="flac", on_metadata_loss="error")
    row = next(row for row in result.native_statuses if row.key == "source:store:apev2:" + key)
    assert row.status == "removed_by_design" and row.expected_ids
    assert result.loss_policy_result == "ok"


@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("target", ["mp3", "aiff", "wav"])
@pytest.mark.parametrize("genre", ["(17)Rock", "(RX)", "((x", "(Live)", "Ambient (Live)", "256"])
def test_literal_genre_text_roundtrips_on_id3_targets(audio, encode, version, target, genre):
    fields = {"genre": genre}
    expected = expected_for(target, fields=fields, extras={}, pictures=(), mark=None, id3_version=version)
    path = candidate(audio, encode, target, target, expected)
    result = write.write_tags(path, plan_for(target, expected, fields=fields, extras={}, pictures=(),
                                           mark_state="omitted", id3_version=version),
                              target=target, id3_version=version).result
    assert result.matched and result.read_back.fields["genre"].value == genre
    tags = ID3(path, translate=False) if target == "mp3" else {"aiff": AIFF, "wav": WAVE}[target](path, translate=False).tags
    assert tags["TCON"].genres == [genre]


@pytest.mark.parametrize("genre", ["17", "017", "255", "RX", "CR"])
@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("origin", ["cli", "flac"])
def test_unrepresentable_id3_genre_refused_in_preparation(audio, genre, version, origin):
    path = audio("flac")
    if origin == "flac":
        tags = FLAC(path)
        tags["GENRE"] = genre
        tags.save()
    source = read.read_source_tags(path, "flac")
    with pytest.raises(FileFailure) as caught:
        plan(source, target="mp3", id3_version=version,
             cli={"genre": ManifestCell("set", genre)} if origin == "cli" else {})
    assert caught.value.reason == "metadata_loss_error"
    assert caught.value.stage == "metadata_prepared"
    assert "genre: ID3 TCON cannot represent this text exactly" in caught.value.detail


@pytest.mark.parametrize("genre", ["(17)Rock", "(RX)", "((x"])
def test_flac_literal_genre_converts_to_mp3_exactly(audio, genre):
    source = audio("flac")
    tags = FLAC(source)
    tags["GENRE"] = genre
    tags.save()
    intended, _ = plan(read.read_source_tags(source, "flac"), target="mp3", no_mark_processed=True)
    target = audio("mp3")
    result = write.write_tags(target, intended, target="mp3").result
    assert result.matched and result.read_back.fields["genre"].value == genre


def test_legacy_loss_explicitly_removed_by_no_source_tags(audio):
    path = audio("mp3")
    tags = APEv2()
    tags["Cover Art (Front)"] = b"picture bytes"
    tags.save(path)
    result, _ = plan(read.read_source_tags(path, "mp3"), no_source_tags=True, on_metadata_loss="error")
    assert result.loss_policy_result == "ok"
    assert next(row for row in result.native_statuses if row.key == "source:store:apev2:Cover Art (Front)").status == "removed_by_request"


def test_readback_rejects_reordered_duplicate_occurrences_even_when_projection_matches(audio):
    path = audio("flac")
    tags = FLAC(path)
    tags.clear()
    tags["Mood"] = [" calm", "calm "]
    tags.save()
    expected = read.read_source_tags(path, "flac").native_inventory
    intended = plan_for("flac", expected, fields={}, extras={"Mood": "calm; calm"}, pictures=(),
                        mark_state="omitted")
    assert write.write_tags(path, intended, target="flac").result.matched
    tags = FLAC(path)
    tags["Mood"] = ["calm ", " calm"]
    tags.save()
    # Both occurrences project to 'calm'; only native duplicate order changed.
    _, mismatches = write.verify_tags(path, intended, target="flac")
    assert "extra:Mood" not in mismatches
    assert {item.identity for item in expected} <= set(mismatches)
