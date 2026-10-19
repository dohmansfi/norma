"""Catalogue coverage closure and declared no-store/corrupt-input boundaries."""
from __future__ import annotations

import collections
from pathlib import Path

import pytest

from tests.e2e.test_metadata_support import (
    CATALOGUE, CATALOGUE_AUDIO, NO_STORE_VARIANTS, assert_accounted, details,
    assert_no_render, ffmpeg_trap, native_ids, normalise, plan,
)


F15_MEMBERS = (
    "space name.wav", "Perché.wav", "unicode/Perche\u0301.wav", "Test Artist's demo.wav",
    "東京 🎵.wav", "a/Song.wav", "b/song.WAV", "Artist/Album/03 - Perché.wav",
    "Artist/Album/004_Not_a_match.wav", ".hidden.wav", "._sidecar.wav",
    "Normalised/output.wav", ".norma-originals/backup.wav", "alias.wav", "hardlink.wav",
)


def test_m02_catalogue_matrix_has_no_silently_omitted_audio_rows():
    """Every catalogue row is exercised or explicitly classified, never a skipped pass."""
    actual = collections.Counter(CATALOGUE_AUDIO)
    for row in CATALOGUE:
        key = row["id"], row.get("variant")
        if row["id"] == "F-15":
            assert row["container"] == "directory"  # members are exercised below
        elif row["id"] == "F-10":
            assert row.get("variant") in (None, "aiff", "payload")
        elif row.get("variant") in NO_STORE_VARIANTS and row["id"] == "F-12":
            assert row["container"] in ("adts", "caf")
        elif row["sample_rate_hz"] == 0:
            assert row["id"] in ("F-11", "F-16", "F-17")
        else:
            assert actual.pop(key) == 1, key
    assert actual == {}


@pytest.mark.parametrize("relative", F15_MEMBERS)
def test_m02_f15_every_member_including_links_and_excluded_folder_copies_is_accounted(
        norma, tools, fixture_path, demos, tmp_path, monkeypatch, relative):
    tree = fixture_path("F-15")
    cached = tree / relative
    # Copy bytes, not the link: no shared fixture or linked inode is ever modified.
    source = demos({"take.wav": cached}) / "take.wav"
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = plan(norma, source)
    assert_accounted(run, source)
    assert native_ids(source) == ({"store:id3v2:TIT2"} if relative in ("a/Song.wav", "b/song.WAV") else set())
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


@pytest.mark.parametrize("variant", sorted(NO_STORE_VARIANTS))
@pytest.mark.parametrize("policy", ["warn", "sidecar"])
def test_m02_catalogue_unknown_store_is_disclosed_not_claimed_tag_free(
        norma, tools, fixture_path, demos, tmp_path, monkeypatch, variant, policy):
    cached = fixture_path("F-12", variant)
    source = demos({"take" + cached.suffix: cached}) / ("take" + cached.suffix)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "--on-metadata-loss", policy, "-v")
    assert run.code == int(policy == "error"), run.stdout + run.stderr
    assert run.files[0][0] == ("FAIL" if policy == "error" else "WARN")
    assert ("metadata_loss_error" if policy == "error" else "metadata_loss_warn") in run.stdout
    assert "no tag store" in run.stdout + run.stderr
    assert not details(run, "PLAN") and not details(run, "NATIVE_FATE")
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


@pytest.mark.parametrize("variant", sorted(NO_STORE_VARIANTS))
def test_m02_no_store_error_policy_plan_predicts_the_real_refusal(
        norma, tools, fixture_path, demos, tmp_path, monkeypatch, variant):
    cached = fixture_path("F-12", variant)
    source = demos({"take" + cached.suffix: cached}) / ("take" + cached.suffix)
    calls = ffmpeg_trap(tmp_path, monkeypatch, delegate=True)
    _, row, output = normalise(norma, source, tmp_path / "run", "--on-metadata-loss", "error", code=1)
    assert row["reason"] == "metadata_loss_error" and row["published_paths"] == []
    assert output is None or not output.exists()
    assert_no_render(calls, source)
    calls.write_text("")
    before = tools.snapshot(tmp_path)
    preview = norma("tags", "plan", source, "--config", "/dev/null", "--on-metadata-loss", "error")
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before
    assert preview.code == 1 and preview.files[0][0] == "FAIL" and "metadata_loss_error" in preview.stdout


@pytest.mark.parametrize("variant", [None, "aiff"])
def test_m02_structurally_truncated_catalogue_files_never_claim_a_complete_inventory(
        norma, tools, fixture_path, demos, tmp_path, monkeypatch, variant):
    cached = fixture_path("F-10", variant)
    source = demos({"take" + cached.suffix: cached}) / ("take" + cached.suffix)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "-v")
    assert run.code == 1 and "truncated" in run.stdout
    assert not details(run, "PLAN") and not details(run, "NATIVE_FATE")
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


def test_m02_payload_corruption_is_not_misrepresented_as_missing_metadata(
        norma, tools, fixture_path, demos, tmp_path, monkeypatch):
    # Audio corruption is a normalise/decode failure; tags plan does not decode audio.
    source = demos({"take.flac": fixture_path("F-10", "payload")}) / "take.flac"
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = plan(norma, source)
    assert_accounted(run, source)
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before
