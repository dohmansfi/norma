"""Read-only explicit inspection references (MD27/MD37; TST-A-08/M-01)."""

import builtins
import hashlib
import json
from dataclasses import replace
from pathlib import PurePosixPath

import pytest
from mutagen.flac import FLAC
from mutagen.id3 import ID3, TDRC, TXXX

from norma.metadata import manifest, read, write
from norma.models import InputItem, ManifestCell, SourceTags, TagValue

MARK = ("v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=linear;gain=+0.00;out_i=-16.00;"
        "out_tp=-1.00;origin=flac;ffmpeg=N-125892;date=2026-10-05")


def tagged(audio, **values):
    path = audio("flac")
    store = FLAC(path)
    store["NORMA"] = MARK
    for key, value in values.items():
        store[key] = value
    store.save()
    return path, read.read_source_tags(path, "flac")


def test_missing_mark_fails_even_without_references():
    tags = SourceTags({}, {}, (), (), (), (), {})
    result = write.check_inspection_tags(tags)
    assert not result.matched
    assert result.mismatches == ("extra:NORMA",)
    assert result.diagnostics[0].code == "verification_tags"
    assert result.diagnostics[0].message == "processing mark missing"


@pytest.mark.parametrize("mark", [MARK, "broken", "v=2;future=yes"])
def test_present_marks_warn_only_when_invalid(audio, mark):
    _, tags = tagged(audio, NORMA=mark, TITLE="unrequested")
    result = write.check_inspection_tags(tags)
    assert result.matched
    assert [d.code for d in result.diagnostics] == ([] if mark == MARK else ["processing_mark_invalid"])


def test_only_explicit_cells_checked_with_typed_false_and_casefolded_extras(audio):
    _, tags = tagged(audio, TITLE="Café", TRACKNUMBER="03/12", DISCNUMBER="01/02",
                     COMPILATION="false", Mood=" calm\r\nnight ", ALBUM="unrequested")
    refs = {"title": ManifestCell("set", " Cafe\u0301 "), "track": ManifestCell("set", 3),
            "track_total": ManifestCell("set", 12), "disc_total": ManifestCell("set", 2),
            "compilation": ManifestCell("set", False), "mOoD": ManifestCell("set", "calm\nnight"),
            "comment": ManifestCell("clear", None)}
    assert write.check_inspection_tags(tags, refs).matched
    refs["compilation"] = ManifestCell("clear", None)
    result = write.check_inspection_tags(tags, refs)
    assert result.mismatches == ("field:compilation",)
    assert [d.code for d in result.diagnostics] == ["verification_tags"]


def test_missing_mark_and_explicit_mismatches_keep_both_causes(audio):
    path, _ = tagged(audio, TITLE="wrong")
    store = FLAC(path)
    del store["NORMA"]
    store.save()
    tags = read.read_source_tags(path, "flac")
    result = write.check_inspection_tags(tags, {"title": ManifestCell("set", "expected")})
    assert result.mismatches == ("extra:NORMA", "field:title")
    assert len(result.diagnostics) == 1
    assert "processing mark missing" in result.diagnostics[0].message
    assert "field:title" in result.diagnostics[0].message


@pytest.mark.parametrize("version,source,expected,matched", [(3, "2026-10", "2026", True),
    (4, "2026-10", "2026-10", True), (4, "2026", "2026", False)])
def test_date_precision_uses_observed_id3_version(audio, version, source, expected, matched):
    path = audio("mp3")
    store = ID3()
    store.add(TDRC(encoding=3, text=[source]))
    store.add(TXXX(encoding=3, desc="NORMA", text=[MARK]))
    if version == 3:
        store.update_to_v23()
    store.save(path, v2_version=version)
    tags = read.read_source_tags(path, "mp3")
    assert tags.fields["date"].value == expected
    result = write.check_inspection_tags(tags, {"date": ManifestCell("set", "2026-10")})
    assert result.matched is matched


@pytest.mark.parametrize("field,value", [("title", "wrong"), ("track_total", 13), ("Mood", "wrong"),
                                         ("compilation", True), ("album", "missing")])
def test_invalid_mark_does_not_mask_explicit_mismatches(audio, field, value):
    _, tags = tagged(audio, NORMA="v=2", TITLE="right", TRACKNUMBER="3/12", Mood="calm", COMPILATION="false")
    result = write.check_inspection_tags(tags, {field: ManifestCell("set", value)})
    assert not result.matched
    assert [d.code for d in result.diagnostics] == ["verification_tags", "processing_mark_invalid"]


def test_total_only_assertion_does_not_require_number_reference(audio):
    _, tags = tagged(audio, TRACKNUMBER="03/12", DISCNUMBER="01/02")
    assert write.check_inspection_tags(tags, {"track_total": ManifestCell("set", 12)}).matched


def test_typed_false_is_not_integer_zero(audio):
    _, tags = tagged(audio, COMPILATION="false")
    tags = replace(tags, fields={"compilation": TagValue(0, "vorbis", "COMPILATION")})
    assert not write.check_inspection_tags(tags, {"compilation": ManifestCell("set", False)}).matched


def test_checker_performs_no_file_reads_writes_or_planning(audio, monkeypatch):
    path, tags = tagged(audio, TITLE="reference", Mood="calm")
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    def forbidden(*args, **kwargs):
        pytest.fail("inspection must not perform I/O or prepare a write plan")
    with monkeypatch.context() as patcher:
        patcher.setattr(builtins, "open", forbidden)
        patcher.setattr(write, "prepare_store", forbidden)
        patcher.setattr(write, "verify_tags", forbidden)
        patcher.setattr(write, "write_tags", forbidden)
        assert write.check_inspection_tags(tags, {"title": ManifestCell("set", "reference")}).matched
        assert not write.check_inspection_tags(tags, {"Mood": ManifestCell("clear", None)}).matched
    assert before == (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)


@pytest.mark.parametrize("form", ["csv", "json"])
def test_parsed_manifest_references_preserve_clear_false_extras_and_defaults(audio, tmp_path, form):
    path, tags = tagged(audio, TITLE="row", ARTIST="default", COMPILATION="false", TRACKNUMBER="3/12")
    if form == "csv":
        raw = ("file,title,artist,compilation,track,track_total,comment,Mood\n"
               "*,ignored,default,,,,,\n"
               f"{path.name},row,,false,3,12,[clear],[clear]\n").encode()
    else:
        raw = json.dumps({"version": 1, "defaults": {"artist": "default", "title": "ignored"},
            "files": {path.name: {"title": "row", "compilation": False, "track": 3, "track_total": 12,
                                  "comment": None, "extras": {"Mood": None}}}}).encode()
    data = manifest.parse_manifest(raw, tmp_path / ("refs." + form))
    item = InputItem(path, tmp_path, PurePosixPath(path.name), (path.stat().st_dev, path.stat().st_ino), 0, True)
    match = manifest.match_manifest(data, [item]).inputs[0]
    assert not match.failures
    assert match.row is not None
    refs = {**data.defaults, **match.row.cells}
    assert refs["compilation"].value is False
    assert refs["Mood"].operation == "clear"
    assert write.check_inspection_tags(tags, refs).matched


def test_whole_manifest_and_matched_row_errors_stay_distinct(audio, tmp_path):
    with pytest.raises(manifest.ManifestError) as failure:
        manifest.parse_manifest(b'{"version":2,"files":{}}', tmp_path / "bad.json")
    assert failure.value.detail == "version"
    path, _ = tagged(audio)
    raw = json.dumps({"version": 1, "files": {path.name: {"track": False}}}).encode()
    data = manifest.parse_manifest(raw, tmp_path / "row.json")
    item = InputItem(path, tmp_path, PurePosixPath(path.name), (path.stat().st_dev, path.stat().st_ino), 0, True)
    match = manifest.match_manifest(data, [item]).inputs[0]
    assert match.row is None
    assert [issue.detail for issue in match.failures] == ["number_invalid"]
