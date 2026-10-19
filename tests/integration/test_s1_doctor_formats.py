"""S1 acceptance: ``norma doctor`` and ``norma formats`` (TST-C-01, TST-C-07 in part)."""

from __future__ import annotations

import json
from pathlib import Path

LOSSLESS = ("wav", "aiff", "flac", "alac")
LOSSY = ("mp3", "aac")


def test_c01_doctor_reports_the_installed_toolchain(norma, norma_environment, tmp_path):
    """TST-C-01: FFmpeg, ffprobe, filters, encoders, mutagen, sips and config are all reported."""
    run = norma("doctor")
    rows = run.rows()
    assert run.code == 0, run.stdout
    for name in ("versions", "platform", "ffmpeg", "ffprobe", "filters", "alimiter options", "encoders",
                 "native alac", "mutagen", "sips", "config"):
        assert name in rows, (name, sorted(rows))
    assert not [line for line in run.lines if line[0] == "FAIL"], run.stdout
    for name in ("ffmpeg", "ffprobe", "filters", "alimiter options", "encoders", "native alac", "mutagen", "sips"):
        assert rows[name][0] == "OK", (name, rows[name])
    path, version = rows["ffmpeg"][1].split(" · ")
    assert path == norma_environment["ffmpeg_path"]
    assert version in norma_environment["ffmpeg_version"]
    assert {"ebur128", "alimiter", "loudnorm"} <= set(rows["filters"][1].split(", "))
    assert {"pcm_s24le", "flac", "alac", "libmp3lame"} <= set(rows["encoders"][1].split(", "))
    assert rows["native alac"][1].endswith("alac_at is never substituted")
    # The sandbox HOME is the one read: no preference file there, so built-in defaults apply.
    assert rows["config"][1] == f"{tmp_path / 'home' / '.config' / 'norma' / 'config.toml'}: missing; using built-in defaults"


def test_c01_lossy_gaps_are_stated_not_invented(norma):
    """Capability gaps are explicitly conservative: a WARN row names its consequence."""
    evidence = [line for line in norma("doctor").lines if line[1:2] == ["lossy evidence"]]
    assert evidence
    encoders = {line[2].split(" ", 1)[0] for line in evidence}
    assert encoders == {"libmp3lame", "aac_at"}
    for status, _, detail in evidence:
        assert status in {"OK", "WARN"}
        if status == "WARN":
            assert "automatic output would fail unsupported_target" in detail


def test_c01_missing_ffmpeg_exits_3(norma, without_ffmpeg):
    """TST-C-01 / S1 acceptance: FFmpeg absent from PATH is an environment failure."""
    run = norma("doctor", path=without_ffmpeg)
    rows = run.rows()
    assert run.code == 3
    assert rows["ffmpeg"][0] == "FAIL"
    assert rows["ffmpeg"][1].startswith("FFmpeg not found")
    for name in ("ffprobe", "filters", "alimiter options", "encoders", "native alac"):
        assert rows[name][0] == "FAIL", name
    assert rows["mutagen"][0] == "OK"
    assert run.stderr.strip() == "Run: norma doctor"


def test_c01_explicit_unusable_ffmpeg_exits_3(norma, tmp_path):
    run = norma("doctor", "--ffmpeg", tmp_path / "nowhere" / "ffmpeg")
    assert run.code == 3
    assert run.rows()["--ffmpeg"][0] == "FAIL"
    assert run.rows()["ffmpeg"][0] == "FAIL"


def test_c01_unreadable_config_exits_2_but_still_reports(norma, tmp_path):
    """A bad preference file is a usage failure (2), reported beside the healthy tools."""
    config = tmp_path / "bad.toml"
    config.write_text("this is = [not toml\n", encoding="utf-8")
    run = norma("doctor", "--config", config)
    assert run.code == 2
    assert run.rows()["config"][0] == "FAIL"
    assert run.rows()["ffmpeg"][0] == "OK"


def test_c01_analyse_without_ffmpeg_exits_3_and_prints_no_rows(norma, without_ffmpeg, tmp_path):
    run = norma("analyse", tmp_path, path=without_ffmpeg)
    assert run.code == 3
    assert run.stdout == ""
    assert run.stderr.splitlines()[0].startswith("norma: error: FFmpeg not found")
    assert run.stderr.splitlines()[-1] == "Run: norma doctor"


def test_c01_formats_lists_every_output_format_with_its_tag_stores(norma):
    run = norma("formats")
    rows = [line for line in run.lines if line[0] == "OK"]
    assert run.code == 0
    assert [row[1] for row in rows] == [*LOSSLESS, *LOSSY]
    by_name = {row[1]: row for row in rows}
    for name, row in by_name.items():
        assert len(row) == 7, row
        matrix = json.loads(row[5])
        assert matrix and all(isinstance(fields, dict) and fields for fields in matrix.values())
        assert "policy only" in row[6]
    assert set(json.loads(by_name["wav"][5])) == {"riff_info", "id3v23"}
    assert set(json.loads(by_name["flac"][5])) == {"flac"}
    assert set(json.loads(by_name["alac"][5])) == {"mp4"}
    assert "alac" in by_name["alac"][2]
    assert "libmp3lame" in by_name["mp3"][2] and "aac_at" in by_name["aac"][2]
    for name in LOSSY:
        assert "run norma doctor" in by_name[name][6]
    for name in LOSSLESS:
        assert "run norma doctor" not in by_name[name][6]


def test_c01_formats_needs_no_ffmpeg_and_is_quietable(norma, without_ffmpeg):
    """Pure policy tables: no tool is located; ``-q`` hides the stderr note only."""
    noisy = norma("formats", path=without_ffmpeg)
    quiet = norma("formats", "-q", path=without_ffmpeg)
    assert noisy.code == quiet.code == 0
    assert noisy.stdout == quiet.stdout
    assert noisy.stderr.startswith("note: v1 accepts mono/stereo only")
    assert quiet.stderr == ""


def test_c07_formats_refuses_ffmpeg_option(norma):
    run = norma("formats", "--ffmpeg", "/usr/local/bin/ffmpeg")
    assert run.code == 2
    assert run.stdout == ""
    assert run.stderr.splitlines()[0] == "norma: error: --ffmpeg is not used by formats"
