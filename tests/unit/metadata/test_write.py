"""Tag writing and read-back on real files (TST-M-01, TST-M-02; MD22, MD29–MD31, MD39).

Sources are generated untagged by FFmpeg in ``tmp_path`` (``conftest.audio``); WAV
INFO is written by FFmpeg ``-metadata`` exactly as the render does (R8). Each
written candidate is cross-checked with an independent mutagen whole-file parse
or a raw byte walk, not only with Norma's own reader.
"""

from __future__ import annotations

import builtins
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from mutagen.aiff import AIFF
from mutagen.apev2 import APEv2
from mutagen.flac import FLAC
from mutagen.id3 import ID3, TIT2, TSSE
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

from norma.errors import FileFailure
from norma.metadata import losscheck, mapping, model, write
from norma.metadata.read import read_processing_mark, read_source_tags
from norma.models import Artwork, MetadataPlan, RawNativeItem, ResolvedValue, SidecarRequest

JPEG = b"\xff\xd8\xff\xe0" + bytes(range(48))
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
MARK = ("v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=limit;gain=+5.80;limited=2.60;out_i=-16.02;"
        "out_tp=-1.04;origin=pcm_s24le;ffmpeg=N-125892;date=2026-10-05")
FIELDS: dict[str, object] = {
    "title": "Night Bus — Perché 東京", "artist": "Test Artist", "album_artist": "Test Artist", "album": "Demos",
    "date": "2026-10-05", "track": 3, "track_total": 12, "disc": 1, "disc_total": 2, "genre": "Electronic",
    "composer": "T. Composer", "comment": "first line\nsecond line", "compilation": False,
}
EXTRAS = {"BPM": "120", "Mood": "calm"}
PICTURES = (Artwork(JPEG, "image/jpeg", None, None, 3, ""), Artwork(PNG, "image/png", None, None, 4, "back"))

# (target, conftest kind, ID3 version)
TARGETS = [("mp3", "mp3", "2.3"), ("mp3", "mp3", "2.4"), ("aiff", "aiff", "2.3"), ("aiff", "aiff", "2.4"),
           ("wav", "wav", "2.3"), ("wav", "wav", "2.4"), ("flac", "flac", "2.3"), ("alac", "alac", "2.3"),
           ("aac", "aac", "2.3")]


def expected_for(target: str, *, fields: Mapping[str, object] = FIELDS, extras: Mapping[str, str] = EXTRAS,
                 pictures: Sequence[Artwork] = PICTURES, mark: str | None = MARK, id3_version: str = "2.3",
                 wav_tags: str = "info+id3") -> tuple[RawNativeItem, ...]:
    items: list[RawNativeItem] = []
    for store in mapping.target_stores(target, wav_tags):
        recipes = mapping.native_recipes(store, fields, extras=extras, pictures=pictures,
                                         mark=mark if store != "riff-info" else None, id3_version=id3_version)
        items += write.prepare_store(store, recipes, id3_version=id3_version).items
    return tuple(items)


def plan_for(target: str, expected: Sequence[RawNativeItem], *, fields: Mapping[str, object] = FIELDS,
             extras: Mapping[str, str] = EXTRAS, pictures: Sequence[Artwork] = PICTURES, id3_version: str = "2.3",
             wav_tags: str = "info+id3", mark_state: str = "final", **changes) -> MetadataPlan:
    store = mapping.primary_store(target, wav_tags)
    effective = mapping.project_fields(store, fields, id3_version=id3_version)
    stored_extras = extras if store in ("id3v2", "mp4", "vorbis") else {}
    plan = MetadataPlan(
        plan_revision=1, audio_basis="encode", conditional=False, source_native=(), requested_native=(),
        expected_native=tuple(expected),
        resolved={name: ResolvedValue(value, "cli", None) for name, value in fields.items()},
        effective={name: ResolvedValue(value, "cli", None) for name, value in effective.items()},
        extras={name: ResolvedValue(value, "cli", None) for name, value in extras.items()},
        effective_extras={name: ResolvedValue(value, "cli", None) for name, value in stored_extras.items()},
        artwork=None, effective_artwork=mapping.project_artwork(store, pictures), statuses=(),
        native_statuses=losscheck.native_statuses((), tuple(expected), (), audio_basis="encode"), conflicts=(),
        non_tag_items=(), sidecar_requests=(), mark_state=mark_state, loss_policy_result="ok")
    return replace(plan, **changes) if changes else plan


def candidate(audio, encode, target: str, kind: str, expected: Sequence[RawNativeItem], name: str = "") -> Path:
    """A render-like untagged candidate; a WAV gets FFmpeg's INFO subset from the plan (R8)."""
    if target == "wav":
        info = [arg for key, value in write.riff_info_metadata(expected) for arg in ("-metadata", f"{key}={value}")]
        return encode("wav", name or f"candidate-{len(info)}.wav", "-c:a", "pcm_s16le", *info)
    return audio(kind)


def signature(item: RawNativeItem) -> tuple:
    return item.store, item.key, item.native_type, item.native_version, item.payload_encoding, item.payload


# --- Round trips per target -----------------------------------------------------------------------


@pytest.mark.parametrize(("target", "kind", "version"), TARGETS)
def test_every_planned_value_reads_back_exactly(target: str, kind: str, version: str, audio, encode) -> None:
    expected = expected_for(target, id3_version=version)
    plan = plan_for(target, expected, id3_version=version)
    path = candidate(audio, encode, target, kind, expected)
    outcome = write.write_tags(path, plan, target=target, id3_version=version)
    result = outcome.result
    assert result.matched and result.mismatches == () and result.diagnostics == ()
    assert result.plan_revision == 1 and result.candidate_path == path and result.staged_member_ids == ()
    # every planned native value, type and version, with duplicate order, and nothing else
    assert sorted(map(signature, result.read_back.native_inventory)) == sorted(map(signature, expected))
    assert result.read_back.non_tag_items == ()
    store = mapping.primary_store(target)
    for name, value in mapping.project_fields(store, FIELDS, id3_version=version).items():
        assert model.values_equal(result.read_back.fields[name].value, value), name
    assert {name: tag.value for name, tag in result.read_back.extras.items()} == EXTRAS
    assert sorted((tag.value.data, tag.value.picture_type) for tag in result.read_back.artwork) == \
        sorted((picture.data, picture.picture_type) for picture in mapping.project_artwork(store, PICTURES))
    assert read_processing_mark(result.read_back).validity == "valid"


@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize(("target", "kind"), [("mp3", "mp3"), ("aiff", "aiff"), ("wav", "wav")])
def test_id3_targets_cross_checked_by_an_independent_parse(target: str, kind: str, version: str, audio, encode) -> None:
    expected = expected_for(target, id3_version=version)
    path = candidate(audio, encode, target, kind, expected)
    assert write.write_tags(path, plan_for(target, expected, id3_version=version), target=target,
                            id3_version=version).result.matched
    tags = ID3(path, translate=False) if kind == "mp3" else {"aiff": AIFF, "wav": WAVE}[kind](path, translate=False).tags
    assert tags.version[:2] == (2, int(version[-1]))
    ids = {frame.FrameID for frame in tags.values()}
    assert ids.isdisjoint(write.V24_ONLY_FRAMES) if version == "2.3" else "TDRC" in ids
    assert str(tags["TIT2"]) == FIELDS["title"] and str(tags["TPE1"]) == "Test Artist"
    assert str(tags["TRCK"]) == "3/12" and str(tags["TPOS"]) == "1/2"
    assert tags.getall("TXXX:NORMA")[0].text == [MARK]
    assert tags.getall("COMM::eng")[0].text == [FIELDS["comment"]]
    assert sorted((frame.type, frame.data) for frame in tags.getall("APIC")) == [(3, JPEG), (4, PNG)]
    if version == "2.3":
        assert str(tags["TYER"]) == "2026" and str(tags["TDAT"]) == "0510"
        assert all(frame.encoding == 1 for frame in tags.values() if hasattr(frame, "encoding"))  # UTF-16 (MD29)
    else:
        assert str(tags["TDRC"]) == "2026-10-05"


def test_mp3_has_one_tag_at_the_target_version_and_no_legacy_stores(audio) -> None:
    expected = expected_for("mp3")
    path = audio("mp3")
    assert path.read_bytes()[3] == 4  # the render's empty ID3v2.4 tag (MD39)
    assert write.write_tags(path, plan_for("mp3", expected), target="mp3").result.matched
    data = path.read_bytes()
    size = 10 + ((data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9])
    assert data[:4] == b"ID3\x03" and data[size] == 0xFF and data[size + 1] & 0xE0 == 0xE0
    assert data.count(b"ID3\x03") == 1 and b"ID3\x04" not in data
    assert data[-128:-125] != b"TAG" and b"APETAGEX" not in data  # no ID3v1, no APEv2
    with pytest.raises(Exception):
        APEv2(path)


@pytest.mark.parametrize("target", ["mp3", "aiff"])
def test_empty_expected_store_still_rewrites_the_tag_at_the_target_version(target: str, audio) -> None:
    path = audio(target)
    plan = plan_for(target, (), fields={}, extras={}, pictures=(), mark_state="omitted")
    result = write.write_tags(path, plan, target=target).result
    assert result.matched and result.read_back.native_inventory == ()
    tags = ID3(path, translate=False) if target == "mp3" else AIFF(path).tags
    assert tags.version[:2] == (2, 3) and len(tags) == 0


def test_flac_keeps_vendor_and_writes_native_picture_blocks(audio, build) -> None:
    expected = expected_for("flac")
    path = audio("flac")
    vendor = FLAC(path).tags.vendor
    assert write.write_tags(path, plan_for("flac", expected), target="flac").result.matched
    flac = FLAC(path)
    assert flac.tags.vendor == vendor
    assert flac["TRACKNUMBER"] == ["3"] and flac["TRACKTOTAL"] == ["12"] and flac["COMPILATION"] == ["0"]
    assert flac["NORMA"] == [MARK] and "METADATA_BLOCK_PICTURE" not in flac
    assert [(picture.type, picture.data) for picture in flac.pictures] == [(3, JPEG), (4, PNG)]
    _, blocks, _ = build.flac_blocks(path)
    assert [kind for kind, _ in blocks].count(4) == 1 and [kind for kind, _ in blocks].count(6) == 2


@pytest.mark.parametrize("kind", ["alac", "aac"])
def test_mp4_atoms_types_and_front_cover_only(kind: str, audio) -> None:
    expected = expected_for(kind)
    path = audio(kind)
    assert write.write_tags(path, plan_for(kind, expected), target=kind).result.matched
    tags = MP4(path).tags
    assert tags["trkn"] == [(3, 12)] and tags["disk"] == [(1, 2)] and tags["cpil"] is False
    assert [bytes(cover) for cover in tags["covr"]] == [JPEG]  # covr has no picture type (MD23)
    assert tags["covr"][0].imageformat == 13
    freeform = tags["----:com.apple.iTunes:NORMA"][0]
    assert bytes(freeform) == MARK.encode() and freeform.dataformat == 1


def test_mp4_total_without_number_is_stored_with_a_zero_placeholder(audio) -> None:
    fields = {"title": "T", "track_total": 12}
    expected = expected_for("alac", fields=fields, extras={}, pictures=(), mark=None)
    plan = plan_for("alac", expected, fields=fields, extras={}, pictures=(), mark_state="omitted")
    path = audio("alac")
    result = write.write_tags(path, plan, target="alac").result
    assert result.matched and MP4(path).tags["trkn"] == [(0, 12)]
    assert "track" not in result.read_back.fields and result.read_back.fields["track_total"].value == 12


def test_wav_info_subset_comes_from_the_render_and_id3_from_the_writer(audio, encode) -> None:
    expected = expected_for("wav")
    info = [item for item in expected if item.store == "riff-info"]
    assert {item.key for item in info} == {"INAM", "IART", "IPRD", "ICRD", "IPRT", "IGNR", "ICMT"}
    assert all(item.payload.endswith(b"\x00") for item in info)
    path = candidate(audio, encode, "wav", "wav", expected)
    before = {item.identity: item.payload for item in read_source_tags(path, "wav").native_inventory}
    assert before == {item.identity: item.payload for item in info}  # FFmpeg wrote exactly the planned INFO
    result = write.write_tags(path, plan_for("wav", expected), target="wav").result
    assert result.matched
    assert result.read_back.fields["date"].value == "2026-10-05"


@pytest.mark.parametrize("mode", ["info", "none"])
def test_wav_without_id3_mode_writes_no_id3_chunk(mode: str, audio, encode) -> None:
    fields = {"title": "Night Bus", "date": "2026-10"}
    expected = expected_for("wav", fields=fields, extras={}, pictures=(), mark=None, wav_tags=mode)
    plan = plan_for("wav", expected, fields=fields, extras={}, pictures=(), wav_tags=mode, mark_state="not_storable")
    path = candidate(audio, encode, "wav", "wav", expected, name=f"{mode}.wav")
    before = path.read_bytes()
    result = write.write_tags(path, plan, target="wav", wav_tags=mode).result
    assert result.matched and path.read_bytes() == before
    assert b"ID3" not in before and (b"INFO" in before) == (mode == "info")


def test_several_pictures_with_one_description_keep_their_order(audio) -> None:
    pictures = tuple(Artwork(PNG + bytes([n]) * n, "image/png", None, None, 3 if n == 1 else 6, "")
                     for n in (1, 5, 2))
    expected = expected_for("mp3", fields={"title": "T"}, extras={}, pictures=pictures, mark=None)
    assert [item.identity for item in expected if item.native_type == "APIC"] == \
        ["store:id3v2:APIC:#1", "store:id3v2:APIC:#2", "store:id3v2:APIC:#3"]
    plan = plan_for("mp3", expected, fields={"title": "T"}, extras={}, pictures=pictures, mark_state="omitted")
    path = audio("mp3")
    result = write.write_tags(path, plan, target="mp3").result
    assert result.matched
    found = [item for item in result.read_back.native_inventory if item.native_type == "APIC"]
    assert [signature(item) for item in found] == [signature(item) for item in expected if item.native_type == "APIC"]


# --- Read-back catches a wrong write --------------------------------------------------------------


WRONG = [("mp3", "mp3", "id3_frame"), ("aiff", "aiff", "id3_frame"), ("flac", "flac", "vorbis_entry"),
         ("alac", "alac", "mp4_entry")]


@pytest.mark.parametrize(("target", "kind", "decoder"), WRONG)
def test_a_deliberately_wrong_write_is_caught(target: str, kind: str, decoder: str, audio, monkeypatch) -> None:
    expected = expected_for(target)
    original = getattr(write, decoder)

    def wrong(item: RawNativeItem):
        value = original(item)
        if item.key in ("TIT2", "TITLE", "©nam"):
            if decoder == "id3_frame":
                value.text = ["Wrong Title"]
            elif decoder == "vorbis_entry":
                value = (value[0], "Wrong Title")
            else:
                value = (value[0], ["Wrong Title"])
        return value

    monkeypatch.setattr(write, decoder, wrong)
    path = audio(kind)
    result = write.write_tags(path, plan_for(target, expected), target=target).result
    title_id = next(item.identity for item in expected if item.key in ("TIT2", "TITLE", "©nam"))
    assert not result.matched and title_id in result.mismatches and "field:title" in result.mismatches
    assert result.diagnostics[0].code == "verification_tags" and result.diagnostics[0].stage == "content_verified"


def test_unexpected_frames_and_legacy_stores_fail_read_back(audio) -> None:
    expected = expected_for("mp3")
    plan = plan_for("mp3", expected)
    path = audio("mp3")
    assert write.write_tags(path, plan, target="mp3").result.matched
    tags = ID3(path, translate=False)
    tags.add(TSSE(encoding=3, text=["Lavf"]))
    tags.save(path, v2_version=3)
    _, mismatches = write.verify_tags(path, plan, target="mp3")
    assert mismatches == ("store:id3v2:TSSE",)  # A8d: an addition is never allowed
    assert write.write_tags(path, plan, target="mp3").result.matched
    ape = APEv2()
    ape["Encoder"] = "x"
    ape.save(path)
    _, mismatches = write.verify_tags(path, plan, target="mp3")
    assert "store:apev2:Encoder" in mismatches and "store:apev2" in mismatches
    with open(path, "ab") as handle:
        handle.write(b"TAG" + bytes(125))
    _, mismatches = write.verify_tags(path, plan, target="mp3")
    assert "store:id3v1" in mismatches


def test_a_second_or_wrong_version_id3_tag_fails_structure(audio) -> None:
    expected = expected_for("mp3", fields={"title": "Night Bus"}, extras={}, pictures=(), mark=None)
    plan = plan_for("mp3", expected, fields={"title": "Night Bus"}, extras={}, pictures=(), mark_state="omitted")
    path = audio("mp3")
    assert write.write_tags(path, plan, target="mp3").result.matched
    data = path.read_bytes()
    empty_v24 = b"ID3\x04\x00\x00\x00\x00\x00\x0a" + bytes(10)
    size = 10 + ((data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9])
    path.write_bytes(data[:size] + empty_v24 + data[size:])  # A8c: the render's tag survives behind ours
    assert write.verify_tags(path, plan, target="mp3")[1] == ("store:id3v2",)
    tags = ID3()
    tags.add(TIT2(encoding=3, text=["Night Bus"]))
    path.write_bytes(data[size:])
    tags.save(path, v2_version=4)  # right frames, wrong version
    assert "store:id3v2" in write.verify_tags(path, plan, target="mp3")[1]


def test_planned_mark_must_read_back_valid(audio) -> None:
    broken = MARK.replace("v=1;", "")
    expected = expected_for("mp3", fields={"title": "T"}, extras={}, pictures=(), mark=broken)
    plan = plan_for("mp3", expected, fields={"title": "T"}, extras={}, pictures=())
    result = write.write_tags(audio("mp3"), plan, target="mp3").result
    assert not result.matched and result.mismatches == ("extra:NORMA",)


def test_a_planned_loss_that_does_not_happen_is_a_mismatch(audio) -> None:
    expected = expected_for("mp3", fields={"title": "T", "composer": "C"}, extras={}, pictures=(), mark=None)
    plan = plan_for("mp3", expected, fields={"title": "T"}, extras={}, pictures=(), mark_state="omitted")
    plan = replace(plan, native_statuses=losscheck.native_statuses((), expected, (), audio_basis="encode"))
    result = write.write_tags(audio("mp3"), plan, target="mp3").result
    assert result.mismatches == ("field:composer",)


def test_effective_artwork_is_compared_by_bytes(audio) -> None:
    expected = expected_for("flac", fields={"title": "T"}, extras={}, pictures=PICTURES[:1], mark=None)
    plan = plan_for("flac", expected, fields={"title": "T"}, extras={}, pictures=PICTURES[:1], mark_state="omitted")
    plan = replace(plan, effective_artwork=(Artwork(JPEG + b"x", "image/jpeg", None, None, 3, ""),))
    assert write.write_tags(audio("flac"), plan, target="flac").result.mismatches == ("field:artwork",)


# --- Plans the writer refuses -----------------------------------------------------------------------


def test_refused_plans_leave_the_candidate_untouched(audio) -> None:
    expected = expected_for("mp3")
    path = audio("mp3")
    before = path.read_bytes()
    plan = plan_for("mp3", expected)
    v24 = expected_for("mp3", id3_version="2.4")
    cases = [
        replace(plan, conditional=True),
        replace(plan, mark_state="pending"),
        replace(plan, mark_state="omitted"),  # a mark item without a final mark
        plan_for("mp3", v24, id3_version="2.4"),  # ID3v2.4 items written as ID3v2.3
        plan_for("flac", expected_for("flac")),  # Vorbis items in an MP3
    ]
    for bad in cases:
        with pytest.raises(ValueError):
            write.write_tags(path, bad, target="mp3", id3_version="2.3")
    tdrc = [item for item in v24 if item.native_type == "TDRC"]
    smuggled = tuple(item for item in expected if item.native_type not in ("TYER", "TDAT")) + tuple(
        replace(item, native_version="2.3") for item in tdrc)
    with pytest.raises(ValueError, match="ID3v2.4-only"):
        write.write_tags(path, plan_for("mp3", smuggled), target="mp3")
    raw = replace(expected[0], payload_encoding="dn-raw-1", payload=b"TIT2\x00\x00\x00\x02\x00\x00\x00T")
    with pytest.raises(ValueError, match="evidence"):
        write.write_tags(path, plan_for("mp3", (raw,) + expected[1:]), target="mp3")
    assert path.read_bytes() == before


def test_source_and_protected_paths_are_never_written(audio, tmp_path, monkeypatch) -> None:
    expected = expected_for("mp3")
    plan = plan_for("mp3", expected)
    source = audio("mp3")
    alias = tmp_path / "alias.mp3"
    os.link(source, alias)
    link = tmp_path / "link.mp3"
    link.symlink_to(source)
    before = source.read_bytes()
    for target_path in (source, alias, link):
        with pytest.raises(ValueError):
            write.write_tags(target_path, plan, target="mp3", protected=(source,))
    with pytest.raises(ValueError, match="absolute"):
        write.write_tags(Path(source.name), plan, target="mp3")
    assert source.read_bytes() == before
    opened: list[tuple[str, str]] = []
    real_open = builtins.open

    def spy(file, mode="r", *args, **kwargs):
        opened.append((os.fspath(file) if isinstance(file, (str, os.PathLike)) else repr(file), mode))
        return real_open(file, mode, *args, **kwargs)

    path = audio("mp3")
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", spy)
        assert write.write_tags(path, plan, target="mp3", protected=(source,)).result.matched
    writes = {name for name, mode in opened if any(flag in mode for flag in "wa+")}
    assert writes == {str(path)}
    assert all(name != str(source) for name, _ in opened) and source.read_bytes() == before


def test_unwritable_candidate_is_a_file_failure(audio) -> None:
    expected = expected_for("mp3")
    path = audio("mp3")
    path.chmod(0o444)
    try:
        with pytest.raises(FileFailure) as caught:
            write.write_tags(path, plan_for("mp3", expected), target="mp3")
    finally:
        path.chmod(0o644)
    assert caught.value.reason == "permission_denied" and caught.value.stage == "tags_written"


def test_garbage_candidate_is_a_tag_failure_not_a_crash(tmp_path) -> None:
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"not an mp4 file at all" * 4)
    expected = expected_for("alac")
    with pytest.raises(FileFailure) as caught:
        write.write_tags(path, plan_for("alac", expected), target="alac")
    assert caught.value.reason == "verification_tags"


def test_plan_revision_is_carried_and_staging_needs_a_name(audio) -> None:
    expected = expected_for("flac", fields={"title": "T"}, extras={}, pictures=(), mark=None)
    plan = plan_for("flac", expected, fields={"title": "T"}, extras={}, pictures=(), mark_state="omitted",
                    plan_revision=3)
    result = write.write_tags(audio("flac"), plan, target="flac").result
    assert result.matched and result.plan_revision == 3
    assert expected[0].payload_encoding == "dn-raw-1" and expected[0].payload == b"T"
    staged_plan = replace(plan, sidecar_requests=(SidecarRequest("tags", "tags_sidecar", None, ".tags.json"),))
    with pytest.raises(ValueError, match="audio file name"):
        write.write_tags(audio("flac"), staged_plan, target="flac")
