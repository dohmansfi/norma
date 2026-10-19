"""S1 acceptance: ``norma tags show`` lists every tag system (TST-M-01 readers, TST-S-05 in part).

Tag stores come from the F-13 catalogue (ID3v2, RIFF INFO, Vorbis comments, MP4 atoms). Three
legacy stores have no catalogue member; they are built here with mutagen or a hand-written AIFF
chunk in the test's own temporary folder and are labelled as local constructions.
"""

from __future__ import annotations

import base64
import json
import shutil
import struct
from pathlib import Path

import pytest
import soundfile as sf

from norma.metadata.model import STORES

ARTWORK_TYPE_FRONT_COVER = 3
CANONICAL = {
    "title": None,  # filled from the recipe module (TITLE)
    "artist": "Test Artist", "album_artist": "Norma fixtures", "album": "Demo set", "date": "2026-10-05",
    "track": 3, "track_total": 12, "disc": 1, "disc_total": 2, "genre": "Demo", "composer": "Composer",
    "comment": None,  # filled from the recipe module (COMMENT)
    "compilation": True,
}
TAGGED = {  # F-13 variant -> (file name, tag systems it must list)
    "wav24": ("tagged-wav24.wav", {"id3v2", "riff-info"}),
    "aiff24": ("tagged-aiff24.aif", {"id3v2"}),
    "flac24": ("tagged-flac24.flac", {"vorbis"}),
    "alac24": ("tagged-alac24.m4a", {"mp4"}),
    "mp3-v0": ("tagged-mp3-v0.mp3", {"id3v2"}),
}


def parse(run, path):
    """Group one file's records by kind: STORE/FIELD/EXTRA/NATIVE/CONFLICT by key, ARTWORK by index."""
    records: dict[str, dict] = {}
    for line in run.lines:
        if len(line) == 4 and line[1] == str(path) and line[0].isupper() and line[0] not in {"OK", "WARN", "SKIP", "FAIL"}:
            records.setdefault(line[0], {})[line[2]] = json.loads(line[3])
    return records


def canonical(fixture_module):
    return {**CANONICAL, "title": fixture_module.TITLE, "comment": fixture_module.COMMENT}


@pytest.mark.parametrize("variant", list(TAGGED))
def test_m01_canonical_object_is_read_identically_from_each_tag_system(norma, fixture_path, fixture_module, variant):
    """TST-M-01 (readers): the F-13 object reads back exact, with its native system and locator."""
    name, systems = TAGGED[variant]
    path = fixture_path("F-13", variant)
    assert path.name == name
    run = norma("tags", "show", path)
    found = parse(run, path)
    assert run.code == 0
    assert run.files[0][:3] == ["OK", "1/1", str(path)]
    assert set(found["STORE"]) == systems
    assert {key: item["value"] for key, item in found["FIELD"].items()} == canonical(fixture_module)
    assert {item["system"] for item in found["FIELD"].values()} <= systems
    assert found["EXTRA"]["DEMO_NOTE"]["value"] == "seedless"
    assert list(found["ARTWORK"]) == ["1"]
    picture = found["ARTWORK"]["1"]["value"]
    assert (picture["mime"], picture["picture_type"]) == ("image/png", ARTWORK_TYPE_FRONT_COVER)
    assert found["MARK"]["absent"]["validity"] == "absent"
    # The raw inventory is complete enough to rebuild each canonical field's source.
    for item in found["FIELD"].values():
        for locator in item["locator"].split(","):
            assert locator in found["NATIVE"], locator
    for identity, native in found["NATIVE"].items():
        assert native["payload_base64"] and identity.startswith("store:")
    assert "CONFLICT" not in found


def test_m01_conflicting_wav_keeps_both_stores_and_reports_the_title_conflict(norma, fixture_path, fixture_module):
    """F-13/conflict: ID3 wins the canonical title; RIFF INFO's own title stays in the raw inventory."""
    path = fixture_path("F-13", "conflict")
    run = norma("tags", "show", path)
    found = parse(run, path)
    assert run.code == 0
    assert set(found["STORE"]) == {"id3v2", "riff-info"}
    assert found["FIELD"]["title"]["value"] == fixture_module.TITLE
    assert found["FIELD"]["title"]["system"] == "id3v2"
    assert list(found["CONFLICT"]) == ["1"]
    assert found["CONFLICT"]["1"]["key"] == "field:title"
    info = found["NATIVE"]["store:riff-info:INAM"]
    assert (info["store"], info["key"]) == ("riff-info", "INAM")
    assert base64.b64decode(info["payload_base64"]).rstrip(b"\0") == b"INFO title"


def test_m01_untagged_audio_lists_no_store_and_an_absent_mark(norma, fixture_path):
    path = fixture_path("F-01")
    run = norma("tags", "show", path)
    found = parse(run, path)
    assert run.code == 0
    assert set(found) == {"MARK"}
    assert found["MARK"]["absent"]["diagnostics"] == []


@pytest.mark.parametrize(("variant", "validity"), [("mark-valid", "valid"), ("mark-invalid", "malformed")])
def test_m01_processing_mark_is_classified(norma, fixture_path, variant, validity):
    path = fixture_path("F-18", variant)
    found = parse(norma("tags", "show", path), path)
    assert list(found["MARK"]) == [validity]
    mark = found["MARK"][validity]
    assert mark["locator"] == "store:vorbis:NORMA"
    if validity == "valid":
        assert (mark["target_lufs"], mark["mode"], mark["origin_codec"]) == (-16.0, "limit", "pcm_s24le")
    else:
        assert [item["code"] for item in mark["diagnostics"]] == ["processing_mark_invalid"]


def build_legacy_stores(tmp_path: Path, fixture_path) -> dict[str, Path]:
    """Local constructions for the stores the catalogue lacks; no Norma code writes them."""
    from mutagen.apev2 import APEv2
    from mutagen.id3 import ID3, TIT2

    folder = tmp_path / "legacy"
    folder.mkdir()
    plain = fixture_path("F-12", "mp3-no-id3")
    v1, ape = folder / "id3v1.mp3", folder / "apev2.mp3"
    shutil.copyfile(plain, v1)
    shutil.copyfile(plain, ape)
    tags = ID3()
    tags.add(TIT2(encoding=0, text=["V1 title"]))
    tags.save(v1, v2_version=3, v1=2)
    apev2 = APEv2()
    apev2["Title"] = "APE title"
    apev2.save(ape)
    text = folder / "aiff-text.aif"
    samples, rate = sf.read(fixture_path("F-01"))
    sf.write(text, samples, rate, subtype="PCM_24", format="AIFF")
    chunk = b"NAME" + struct.pack(">I", 10) + b"AIFF title"
    data = text.read_bytes() + chunk
    text.write_bytes(data[:4] + struct.pack(">I", len(data) - 8) + data[8:])
    return {"id3v1": v1, "apev2": ape, "aiff-text": text}


def test_m01_every_tag_system_is_listed_by_some_file(norma, fixture_path, tmp_path):
    """S1 acceptance: ``tags show`` lists every tag system the reader knows (model.STORES)."""
    legacy = build_legacy_stores(tmp_path, fixture_path)
    listed: dict[str, set[Path]] = {}
    paths = [fixture_path("F-13", variant) for variant in TAGGED] + list(legacy.values())
    run = norma("tags", "show", *paths)
    assert run.code == 0
    assert [row[0] for row in run.files] == ["OK"] * len(paths)
    for path in paths:
        for store in parse(run, path).get("STORE", {}):
            listed.setdefault(store, set()).add(path)
    assert set(listed) == set(STORES)
    expected_titles = {"id3v1": "V1 title", "apev2": "APE title", "aiff-text": "AIFF title"}
    for store, path in legacy.items():
        found = parse(run, path)
        assert path in listed[store]
        assert found["FIELD"]["title"]["value"] == expected_titles[store]
        if store == "id3v1":  # the file also holds the ID3v2 copy; the v1 record is listed beside it
            assert found["STORE"]["id3v1"] == ["store:id3v1:title"]
        else:
            assert found["FIELD"]["title"]["system"] == store
    assert run.summary["ok"] == str(len(paths)) and run.summary["failed"] == "0"


@pytest.mark.parametrize(("variant", "container"), [("caf-pcm", "caf"), ("adts", "adts")])
def test_m01_caf_and_raw_adts_warn_that_their_tags_cannot_be_read(norma, fixture_path, demos, variant, container):
    """CR-081 / ADR-0020 / D-058: these two containers have no tag store in v1 (metadata spec 3.1).

    The file is a typed WARN ``metadata_loss_warn`` (D-060), not an internal failure: the run exits 0
    and the batch carries on. Nothing is claimed read, so only the two readable files list stores.
    """
    limited = fixture_path("F-12", variant)
    root = demos({"a-tagged.wav": fixture_path("F-13", "wav24"), f"b-limited{limited.suffix}": limited,
                  "c-tagged.flac": fixture_path("F-13", "flac24")})
    run = norma("tags", "show", root)
    assert run.code == 0, run.stderr
    assert [row[0] for row in run.files] == ["OK", "WARN", "OK"]
    assert run.files[1][10].startswith(f"metadata_loss_warn: {container} tags cannot be read in this version")
    assert str(root / f"b-limited{limited.suffix}") not in {line[1] for line in run.records("STORE")}
    assert run.records("STORE") and {line[2] for line in run.records("STORE")} == {"id3v2", "riff-info", "vorbis"}
    assert (run.summary["ok"], run.summary["warn"], run.summary["failed"]) == ("2", "1", "0")
    assert f"{container} tags cannot be read" in run.stderr


def test_s05_tags_show_reports_bad_files_without_stopping(norma, fixture_path, demos):
    root = demos({"a.wav": fixture_path("F-13", "wav24"), "b-not-audio.wav": fixture_path("F-11"),
                  "c-truncated.wav": fixture_path("F-10")})
    run = norma("tags", "show", root)
    assert run.code == 1
    assert [row[0] for row in run.files] == ["OK", "SKIP", "FAIL"]
    assert run.files[1][10].startswith("not_audio: ")
    assert run.files[2][10].startswith("truncated: ")
    assert {line[1] for line in run.lines if line[0] in {"STORE", "FIELD"}} == {str(root / "a.wav")}


def test_c07_tags_show_needs_no_loudness_work_and_takes_recursive_folders(norma, fixture_path):
    """The F-15 tree has two tagged files named alike; both are shown, in discovery order."""
    tree = fixture_path("F-15")
    run = norma("tags", "show", "-r", tree)
    assert run.code == 0
    assert len(run.files) == 9
    titles = {line[1].rsplit("/", 1)[1]: json.loads(line[3])["value"]
              for line in run.records("FIELD") if line[2] == "title"}
    assert titles == {"Song.wav": "shared", "song.WAV": "shared"}
    assert all(row[3:6] == ["", "", ""] for row in run.files)
