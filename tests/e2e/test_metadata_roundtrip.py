"""TST-M-01/M-02: six-target native read-back, loss fates and bundles."""
from __future__ import annotations

import base64
import collections
import io
import json
import shutil
from pathlib import Path

import pytest
from mutagen.id3 import APIC, GEOB, ID3, PRIV, USLT, TIT2, TXXX
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.wave import WAVE

from tests.e2e.test_metadata_support import (
    F13, TARGETS, assert_accounted, assert_no_render, branch_rows, chunks,
    container, details, document, ffmpeg_trap, native_ids, normalise, plan,
    readback, sha, tag_wav,
)


def check_effective(run, output, *, info=False):
    fields, extras, pictures = readback(output, info=info)
    for _, branch, _ in details(run, "PLAN"):
        for row in branch_rows(run, "TAG", branch):
            key, value = row["key"], row["effective"]
            if key.startswith("field:"):
                assert fields.get(key[6:]) == value, (key, row, fields)
            elif key.startswith("extra:"):
                assert extras.get(key[6:]) == value, (key, row, extras)
            elif key.startswith("picture:") and value is not None:
                assert pictures and value["picture_type"] >= 0, row
    return fields, extras, pictures


@pytest.mark.parametrize("source_kind", F13)
@pytest.mark.parametrize("target", TARGETS)
def test_m01_f13_every_source_to_six_targets(norma, tools, fixture_path, fixture_module,
                                             demos, tmp_path, source_kind, target):
    source = demos({"take" + fixture_path("F-13", source_kind).suffix:
                    fixture_path("F-13", source_kind)})
    source, = source.iterdir()
    before = tools.snapshot(source.parent)
    preview = plan(norma, source, "-f", target, "--no-mark-processed", "--allow-lossy-transcode")
    assert_accounted(preview, source)
    run, row, output = normalise(norma, source, tmp_path / "run", "-f", target,
                                "--allow-lossy-transcode", "--no-mark-processed")
    assert row["execution"] == "committed" and output.is_file()
    fields, extras, pictures = check_effective(preview, output)
    assert fields == dict(title=fixture_module.TITLE, artist="Test Artist", album_artist="Norma fixtures",
                         album="Demo set", date="2026-10-05", track=3, track_total=12,
                         disc=1, disc_total=2, genre="Demo", composer="Composer",
                         comment=fixture_module.COMMENT, compilation=True)
    assert extras == {"DEMO_NOTE": "seedless"}
    assert pictures == [fixture_module.cover_png()]
    assert row["dropped"] == []
    assert tools.snapshot(source.parent) == before
    if target in ("wav", "aiff", "mp3"):
        if target == "mp3":
            tags = ID3(output, translate=False)
            assert output.read_bytes()[:5] == b"ID3\x03\x00"
        else:
            body, = [b for key, b in chunks(output.read_bytes(), big=target == "aiff") if key.lower() == b"id3 "]
            assert body[:5] == b"ID3\x03\x00"
            tags = ID3(io.BytesIO(body), translate=False)
        assert tags["TYER"].text == ["2026"] and tags["TDAT"].text == ["0510"]
        assert tags["TRCK"].text == ["3/12"] and tags["TPOS"].text == ["1/2"]
        assert tags["COMM::eng"].lang == "eng"
        assert tags["APIC:"].type == 3 and tags["APIC:"].mime == "image/png"
    elif target in ("alac", "aac"):
        tags = MP4(output).tags
        assert tags["trkn"] == [(3, 12)] and tags["disk"] == [(1, 2)]
        assert tags["cpil"] is True and tags["covr"][0].imageformat == 14
    else:
        tags = FLAC(output)
        assert tags["TRACKNUMBER"] == ["3"] and tags["TRACKTOTAL"] == ["12"]
        assert tags.pictures[0].type == 3 and tags.pictures[0].data == fixture_module.cover_png()


@pytest.mark.parametrize("mode", ["info+id3", "info", "none"])
def test_m01_wav_store_modes_have_only_the_specified_subset(norma, tools, fixture_path,
                                                           demos, tmp_path, mode):
    source = demos({"take.flac": fixture_path("F-13", "flac24")}) / "take.flac"
    preview = plan(norma, source, "-f", "wav", "--wav-tags", mode, "--no-mark-processed")
    assert_accounted(preview, source)
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", "wav", "--wav-tags", mode,
                              "--no-mark-processed")
    fields, extras, pictures = check_effective(preview, output, info=mode == "info")
    chunk_names = [key for key, _ in chunks(output.read_bytes())]
    assert (b"id3 " in [key.lower() for key in chunk_names]) == (mode == "info+id3")
    assert (b"LIST" in chunk_names) == (mode != "none")
    if mode == "none":
        assert fields == extras == {} and pictures == []
        assert row["status"] == "warn" and row["dropped"]
    elif mode == "info":
        assert set(fields) == {"title", "artist", "album", "date", "track", "genre", "comment"}
        assert extras == {} and pictures == []
        assert row["status"] == "warn"
    else:
        assert len(fields) == 13 and extras == {"DEMO_NOTE": "seedless"} and len(pictures) == 1
        assert row["dropped"] == []


# Absent, number-only, total-only, paired and cleared are independent requests.
PAIR_CASES = [({}, {}, []), ({"track": 3, "disc": 2}, {"track": 3, "disc": 2}, []),
              ({"track_total": 12, "disc_total": 4}, {"track_total": 12, "disc_total": 4}, []),
              ({"track": 3, "track_total": 12, "disc": 2, "disc_total": 4},
               {"track": 3, "track_total": 12, "disc": 2, "disc_total": 4}, []),
              ({"track": None, "track_total": None, "disc": None, "disc_total": None}, {}, [])]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("opinion,expected,unused", PAIR_CASES,
                         ids=["absent", "number", "total", "pair", "clear"])
def test_m01_number_pairs_and_false_are_typed(norma, tools, tmp_path, target, opinion, expected, unused):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    if any(v is None for v in opinion.values()):
        tag_wav(source, track="3/12", disc="2/4")
    manifest = tmp_path / "values.json"
    manifest.write_text(json.dumps({"version": 1, "files": {source.name: dict(opinion, compilation=False)}}))
    preview = plan(norma, source, "-f", target, "--manifest", manifest, "--no-mark-processed")
    assert_accounted(preview, source)
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", target,
                              "--manifest", manifest, "--no-mark-processed")
    fields, _, _ = check_effective(preview, output)
    expected = dict(expected)
    if target in ("wav", "aiff", "mp3") and opinion and not any(k in opinion for k in ("track", "disc")):
        expected = {}
        assert {loss["key"] for loss in row["dropped"]} == {"field:track_total", "field:disc_total"}
    assert {key: fields[key] for key in ("track", "track_total", "disc", "disc_total") if key in fields} == expected
    assert fields["compilation"] is False and type(fields["compilation"]) is bool
    if target in ("alac", "aac") and "track_total" in expected and "track" not in expected:
        assert MP4(output)["trkn"] == [(0, 12)] and MP4(output)["disk"] == [(0, 4)]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("date", ["2026", "2026-10", "2026-10-05"])
@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_m01_date_precision_is_predicted_not_silently_lost(norma, tools, tmp_path, target, date, version):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    preview = plan(norma, source, "-f", target, "--date", date, "--id3-version", version,
                   "--no-mark-processed")
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", target, "--date", date,
                              "--id3-version", version, "--no-mark-processed")
    fields, _, _ = check_effective(preview, output)
    loses_month = version == "2.3" and target in ("wav", "aiff", "mp3") and len(date) == 7
    assert fields["date"] == ("2026" if loses_month else date)
    assert [loss["loss_kind"] for loss in row["dropped"]] == (["precision"] if loses_month else [])
    if target in ("wav", "aiff", "mp3"):
        body = output.read_bytes() if target == "mp3" else next(
            b for k, b in chunks(output.read_bytes(), big=target == "aiff") if k.lower() == b"id3 ")
        assert body[:4] == b"ID3" + bytes([int(version[-1])])


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("target", ["wav", "aiff", "mp3", "flac"])
def test_m01_binary_and_multivalue_native_items_keep_types_or_report_loss(
        norma, tools, tmp_path, version, target):
    source = tools.tone(tmp_path / "native.wav", seconds=3)
    audio = WAVE(source)
    audio.add_tags()
    frames = [PRIV(owner="com.example", data=b"\x00\xffprivate"),
              GEOB(encoding=1, mime="application/octet-stream", filename="a.bin", desc="binary", data=b"\xff\x00blob"),
              USLT(encoding=1, lang="ita", desc="lyric", text="Perché\n東京"),
              TIT2(encoding=1, text=["One", "Two"]), TXXX(encoding=1, desc="MOOD", text=["calm", "bright"])]
    for frame in frames:
        audio.tags.add(frame)
    audio.save(v2_version=version)
    before = tools.snapshot(tmp_path)
    preview = plan(norma, source, "-f", target, "--no-mark-processed")
    assert_accounted(preview, source)
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", target, "--no-mark-processed")
    fields, extras, _ = check_effective(preview, output)
    assert fields["title"] == ("One/Two" if version == 3 else "One; Two")
    assert extras["MOOD"] == ("calm/bright" if version == 3 else "calm; bright")
    if target == "flac":
        assert {loss["key"] for loss in row["dropped"]} == {
            "item:native:id3v2:PRIV:com.example", "item:native:id3v2:GEOB:binary", "item:native:id3v2:USLT:lyric:ita"}
    else:
        stored = container(output).tags
        assert next(f for f in stored.getall("PRIV") if f.owner == "com.example").data == b"\x00\xffprivate"
        assert stored["GEOB:binary"].data == b"\xff\x00blob"
        assert stored["USLT:lyric:ita"].text == "Perché\n東京"
        assert row["dropped"] == []
    assert tools.snapshot(tmp_path)["native.wav"] == before["native.wav"]


@pytest.mark.parametrize("policy", ["warn", "sidecar", "error"])
def test_m02_all_six_fates_and_loss_policy(norma, tools, fixture_module, tmp_path, monkeypatch, policy):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    tag_wav(source, title="old", artist="kept", comment="clear me", REPLAYGAIN_TRACK_GAIN="-3 dB")
    audio = WAVE(source)
    audio.tags.add(PRIV(owner="com.example", data=b"\x00\xffportable"))
    audio.save(v2_version=3)
    options = ["-f", "flac", "--title", "new", "--clear-tag", "comment", "--album", "added",
               "--on-metadata-loss", policy, "--no-mark-processed"]
    if policy != "error":
        preview = plan(norma, source, *options)
        assert_accounted(preview, source)
        statuses = branch_rows(preview, "TAG", details(preview, "PLAN")[0][1])
        assert {r["status"] for r in statuses} == {"kept", "changed", "added", "removed_by_request",
                                                   "removed_by_design", "dropped_format_limit"}
    calls = ffmpeg_trap(tmp_path, monkeypatch, delegate=True)
    _, row, output = normalise(norma, source, tmp_path / "run", *options, code=int(policy == "error"))
    if policy == "error":
        assert row["reason"] == "metadata_loss_error" and row["published_paths"] == []
        assert output is None or not output.exists()
        assert_no_render(calls, source)
    else:
        check_effective(preview, output)
        assert len(row["dropped"]) == 1 and row["dropped"][0]["key"] == "item:native:id3v2:PRIV:com.example"
        assert row["reason"] == "metadata_loss_warn"
        assert bool(row["sidecar"]) == (policy == "sidecar")
        if policy == "sidecar":
            sidecar = document(Path(row["sidecar"]))
            assert sidecar["sidecar_version"] == 1 and sidecar["audio"] == output.name
            assert set(sidecar["manifest"]) == {"version", "files"}
            assert sidecar["manifest"]["files"][output.name]["comment"] is None
            assert "PRIV" not in json.dumps(sidecar["manifest"])
            loss, = sidecar["native_losses"]
            item, = loss["items"]
            payload = base64.b64decode(item["payload_b64"])
            assert sha(payload) == item["payload_sha256"] and len(payload) == item["payload_size"]
            assert base64.b64encode(b"\x00\xffportable").decode() in payload.decode()
            assert set(row["published_paths"]) == {str(output), row["sidecar"]}
            subset = output.parent / "subset.json"
            subset.write_text(json.dumps(sidecar["manifest"]))
            reapplied = plan(norma, output, "--manifest", subset, "--no-mark-processed")
            assert all(p["resolved"]["title"]["value"] == "new" for _, _, p in details(reapplied, "PLAN"))


@pytest.mark.parametrize("policy", ["warn", "sidecar", "error"])
def test_m02_dropped_picture_exports_are_published_as_one_complete_bundle(
        norma, tools, fixture_module, tmp_path, monkeypatch, policy):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    audio = WAVE(source)
    audio.add_tags()
    front = fixture_module.cover_png()
    other = front + b"other-picture-marker"
    audio.tags.add(APIC(mime="image/png", type=3, desc="front", data=front))
    audio.tags.add(APIC(mime="image/png", type=4, desc="back", data=other))
    audio.save(v2_version=3)
    # INFO cannot carry either picture, so both cover and numbered export are required.
    opts = ["--wav-tags", "info", "--on-metadata-loss", policy]
    calls = ffmpeg_trap(tmp_path, monkeypatch, delegate=True)
    _, row, output = normalise(norma, source, tmp_path / "run", *opts, code=int(policy == "error"))
    if policy == "error":
        assert row["reason"] == "metadata_loss_error" and (output is None or not output.exists())
        assert row["published_paths"] == []
        assert_no_render(calls, source)
    elif policy == "warn":
        assert row["sidecar"] is None and row["published_paths"] == [str(output)]
        assert readback(output, info=True)[2] == []
    else:
        sidecar = document(Path(row["sidecar"]))
        cover = Path(str(output) + ".cover.png")
        picture = Path(str(output) + ".picture-2.png")
        assert cover.read_bytes() == front and picture.read_bytes() == other
        assert set(row["published_paths"]) == {str(p) for p in (output, cover, picture, Path(row["sidecar"]))}
        assert sidecar["manifest"]["files"][output.name]["artwork"] == cover.name
        assert len([loss for loss in sidecar["losses"] if loss["key"].startswith("picture:")]) == 2


@pytest.mark.parametrize("target", TARGETS)
def test_f04_processing_mark_skip_force_and_omit(norma, tools, tmp_path, target):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    _, row, marked = normalise(norma, source, tmp_path / "first", "-f", target)
    mark = readback(marked)[1]["NORMA"]
    pairs = dict(part.split("=", 1) for part in mark.split(";"))
    assert mark.startswith("v=1;tool=norma-") and len(mark) <= 255 and " " not in mark
    assert pairs["mode"] == row["mode"] and float(pairs["out_i"]) == pytest.approx(row["out_lufs"], abs=.005)
    before = tools.snapshot(marked.parent)
    run, skipped, _ = normalise(norma, marked, tmp_path / "skip", "--artist", "Ignored")
    assert skipped["reason"] == "already_processed" and skipped["published_paths"] == []
    assert tools.snapshot(marked.parent) == before
    _, forced, output = normalise(norma, marked, tmp_path / "forced", "--force", "--artist", "Forced")
    assert forced["execution"] == "committed" and readback(output)[0]["artist"] == "Forced"
    assert "NORMA" in readback(output)[1]
    _, _, unmarked = normalise(norma, source, tmp_path / "omit", "-f", target, "--no-mark-processed")
    assert "NORMA" not in readback(unmarked)[1]


@pytest.mark.parametrize("variant", ["mark-valid", "mark-invalid"])
def test_f04_presence_of_even_a_malformed_mark_skips(norma, fixture_path, demos, tmp_path, variant):
    source = demos({"take.flac": fixture_path("F-18", variant)}) / "take.flac"
    _, row, _ = normalise(norma, source, tmp_path / "skip")
    assert row["reason"] == "already_processed" and row["published_paths"] == []
    assert any(w["code"] == "processing_mark_invalid" for w in row["warnings"]) == (variant == "mark-invalid")


@pytest.mark.parametrize("source_kind", F13)
def test_m02_f18_stale_items_are_removed_from_real_outputs(norma, fixture_path, demos, tmp_path, source_kind):
    cached = fixture_path("F-18", source_kind)
    source = demos({"take" + cached.suffix: cached}) / ("take" + cached.suffix)
    preview = plan(norma, source, "--no-mark-processed")
    assert_accounted(preview, source)
    for _, branch, _ in details(preview, "PLAN"):
        stale = [row for row in branch_rows(preview, "NATIVE_FATE", branch)
                 if any(key in row["key"].upper() for key in ("REPLAYGAIN_", "ITUNNORM", "ENCODER", "TSSE", "©TOO"))]
        assert len(stale) == 4 and all(row["status"] == "removed_by_design" for row in stale)
    _, row, output = normalise(norma, source, tmp_path / "run", "--no-mark-processed")
    check_effective(preview, output)
    assert not any(key in identity.upper() for identity in native_ids(output)
                   for key in ("REPLAYGAIN_", "ITUNNORM", "ENCODER", "TSSE", "©TOO"))
    assert row["dropped"] == [] and row["sidecar"] is None
