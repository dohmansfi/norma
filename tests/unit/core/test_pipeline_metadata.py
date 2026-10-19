"""TK-0427: S4 metadata through the pipeline (ARCHITECTURE §§ 3–5; metadata MD1–MD41; D-066 lifted).

Real-tool tests generate small tagged sources under tmp_path with the installed FFmpeg, run
``normalise_items`` through ``batch.execute_batch`` and check every output independently:
``metadata/read.py`` reads each written store back, mutagen reads the file as a second,
independent parser, sources are compared by SHA-256 and times, and a ledger of tmp_path shows
what a refusal, a dry run or ``tags plan`` left behind. Pure helpers (the MD26 mark text and
the copy-skip rule) are checked without tools.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
import threading
import zlib
from dataclasses import replace
from pathlib import Path

import mutagen
import pytest

from norma import batch, pipeline
from norma.adapters import ffmpeg
from norma.errors import EnvironmentError_, UsageError, tool_loss_reason
from norma.models import Diagnostic, FieldStatus, GainPlan, LoudnessMeasurement, RunOptions

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
BITEXACT = ("-fflags", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1")
TONE = "sine=frequency=997:sample_rate=48000:duration={seconds}"
CONTAINERS = {"wav": "wav", "flac": "flac", "mp3": "mp3", "m4a": "mp4", "aif": "aiff", "aiff": "aiff"}

real = pytest.mark.real_ffmpeg


@pytest.fixture(autouse=True)
def path_based_rmtree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evidence folder is a ``TemporaryDirectory``; see test_pipeline_doctor.py for why."""
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


@pytest.fixture(autouse=True)
def no_process_left() -> None:
    yield
    assert ffmpeg._ACTIVE == {}, "a tool call outlived its test"


# --- Helpers ----------------------------------------------------------------------------------


def generate(path: Path, *args: str, seconds: float = 3, level_db: float = -25,
             tags: dict[str, str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: list[str] = []
    for key, value in (tags or {}).items():
        meta += ["-metadata", f"{key}={value}"]
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", TONE.format(seconds=seconds), "-af", f"volume={level_db}dB", "-ac", "2",
                    *(meta or BITEXACT), *args, str(path)], check=True, timeout=60)
    return path


TAGS = {"title": "Night Bus", "artist": "Test Artist", "album": "Demos", "date": "2026-10"}


def png(path: Path, width: int = 8, height: int = 8) -> Path:
    """A tiny valid RGB PNG, written with the standard library."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = b"".join(b"\0" + b"".join(bytes((x * 30 % 256, y * 30 % 256, 128)) for x in range(width))
                    for y in range(height))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def options(tmp_path: Path, *inputs: Path, **changes) -> RunOptions:
    return replace(RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=tuple(inputs),
                              jobs=2), **changes)


def execute(request: pipeline.BatchRequest) -> batch.BatchResult:
    control = batch.BatchControl(commit_lock=request.commit_lock, cancel_event=request.cancel_event,
                                 force_hook=request.force_hook)
    return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                               jobs=request.jobs, control=control, handle_sigint=False)


def normalise(run_options: RunOptions, **kwargs) -> pipeline.NormaliseRun:
    tools = pipeline.locate_normalise_tools(run_options)
    items = pipeline.discover_inputs(run_options).items
    return pipeline.normalise_items(items, run_options, tools,
                                    execute=None if run_options.dry_run else execute, **kwargs)


def by_name(run: pipeline.NormaliseRun) -> dict[str, pipeline.Normalised]:
    return {record.outcome.input.source_path.name: record for record in run.records}


def fingerprint(path: Path) -> tuple[str, int, int]:
    info = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), info.st_mtime_ns, info.st_size


def ledger(root: Path) -> dict[str, tuple[str, int, int]]:
    return {str(p.relative_to(root)): fingerprint(p) for p in sorted(root.rglob("*")) if p.is_file()}


def residue(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.name.startswith(".norma-"))


def read_back(path: Path):
    return pipeline.read_source_tags(path, CONTAINERS[path.suffix.lstrip(".").lower()])


def renders_forbidden(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("render")
        raise AssertionError("no render may start for this file")

    monkeypatch.setattr(pipeline, "render", forbidden)
    return calls


def status_of(record: pipeline.Normalised, key: str) -> FieldStatus:
    return next(s for s in record.outcome.metadata.statuses if s.key == key)


# --- Pure helpers ---------------------------------------------------------------------------


def gain_plan(**changes) -> GainPlan:
    base = GainPlan(action="limit", requested_peak_mode="limit", target_lufs=-16.0, ceiling_dbtp=-1.0,
                    tolerance_lu=0.5, max_limit_db=3.0, gain_db=5.8, predicted_true_peak_dbtp=-1.0,
                    limiter_reduction_db=2.6, effective_limiter_bound_db=2.9, limiter_threshold_dbtp=-1.3,
                    expected_integrated_lufs=-16.0, shortfall_lu=0.0, fallback_from=None, loudness_corrections=0,
                    peak_corrections=0, reduce_fallbacks=0, reason_code="ok")
    return replace(base, **changes)


def measurement(integrated: float = -16.02, peak: float = -1.04) -> LoudnessMeasurement:
    return LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=integrated, true_peak_dbtp=peak, true_peak_linear=0.9,
        lra_lu=3.0, threshold_lufs=None, integrated_valid=True, peak_valid=True, lra_valid=True,
        threshold_valid=False, integrated_resolution_lu=0.1, peak_resolution_linear=1e-6, dual_mono=False,
        meter="ebur128", value_source="metadata", ffmpeg_version="N-125892-g406c5a37aa",
        decoded_duration_s=3.0, decoded_frames=144000, sample_rate_hz=48000, decoded_nonzero=True,
        terminal_peak_covered=True)


def test_md26_mark_text_matches_the_spec_example() -> None:
    text = pipeline.processing_mark_value(gain_plan(), measurement(), origin="pcm_s24le",
                                          ffmpeg_version="N-125892-g406c5a37aa", run_date="2026-10-05",
                                          tool_version="0.1.0")
    assert text == ("v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=limit;gain=+5.80;limited=2.60;out_i=-16.02;"
                    "out_tp=-1.04;origin=pcm_s24le;ffmpeg=N-125892;date=2026-10-05")


@pytest.mark.parametrize("value, signed, strip, expected", [
    (-16.0, False, True, "-16"), (-14.5, False, True, "-14.5"), (-14.25, False, True, "-14.25"),
    (-16.125, False, False, "-16.12"), (-16.135, False, False, "-16.14"), (5.8, True, False, "+5.80"),
    (-0.0, True, False, "+0.00"), (-0.004, True, False, "+0.00"), (-3.0, True, False, "-3.00"),
])
def test_md26_rounding_is_half_even_on_the_shortest_decimal(value, signed, strip, expected) -> None:
    """Metadata § 13 probe P10's values: Decimal(repr(x)), two places, ROUND_HALF_EVEN, no −0."""
    assert pipeline._mark_number(value, signed=signed, strip=strip) == expected


def test_md26_limited_appears_only_for_limit_and_copy_gain_is_zero() -> None:
    from types import SimpleNamespace

    copy = SimpleNamespace(action="copy", target_lufs=-16.0, ceiling_dbtp=-1.0, gain_db=None, limiter_reduction_db=0.0)
    text = pipeline.processing_mark_value(copy, measurement(), origin="flac",  # type: ignore[arg-type]
                                          ffmpeg_version="7.1", run_date="2026-10-05")
    assert "limited=" not in text and ";mode=copy;gain=+0.00;" in text and ";ffmpeg=7.1;" in text


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None])
def test_md26_a_non_finite_value_is_a_bug_never_written(bad) -> None:
    from types import SimpleNamespace

    broken = SimpleNamespace(integrated_lufs=bad, true_peak_dbtp=-1.0)
    with pytest.raises(ValueError, match="finite"):
        pipeline.processing_mark_value(gain_plan(), broken, origin="flac",  # type: ignore[arg-type]
                                       ffmpeg_version="7.1", run_date="2026-10-05")


def test_cr079_artwork_stage_has_a_tool_loss_reason() -> None:
    assert tool_loss_reason("metadata_prepared") == "artwork_invalid"


def test_cr081_unreadable_containers_get_a_typed_warning() -> None:
    warning = pipeline.unreadable_tags("caf")
    assert isinstance(warning, Diagnostic) and warning.code == "metadata_loss_warn"
    assert warning.stage == "tags_read" and "caf tags cannot be read" in warning.message
    assert pipeline.UNREADABLE_TAG_CONTAINERS == {"caf", "adts"}


def test_number_tracks_count_per_folder_in_discovery_order(tmp_path: Path) -> None:
    from pathlib import PurePosixPath

    from norma.models import InputItem

    def item(index: int, folder: str, name: str) -> InputItem:
        path = tmp_path / folder / name
        return InputItem(path, tmp_path, PurePosixPath(folder, name), (1, index), index, False)

    items = [item(0, "a", "x.wav"), item(1, "b", "y.wav"), item(2, "a", "z.wav")]
    assert pipeline._track_numbers(items) == {0: (1, 2), 2: (2, 2), 1: (1, 1)}


# --- normalise: tags carried per target (TST-M-01, TST-M-02) -------------------------------------

SOURCES = {"a.wav": ("-c:a", "pcm_s24le"), "b.flac": ("-c:a", "flac"),
           "c.mp3": ("-c:a", "libmp3lame", "-q:a", "2"), "d.m4a": ("-c:a", "alac")}
TARGET_EXT = {"wav": ".wav", "flac": ".flac", "mp3": ".mp3", "alac": ".m4a", "aac": ".m4a", "aiff": ".aiff"}


@real
@pytest.mark.parametrize("target", ["same", "wav", "flac", "mp3", "alac", "aac", "aiff"])
def test_tagged_sources_carry_their_tags_to_every_target(tmp_path: Path, target: str) -> None:
    folder = tmp_path / "in"
    for name, args in SOURCES.items():
        generate(folder / name, *args, tags=TAGS)
    before = ledger(folder)
    run = by_name(normalise(options(tmp_path, folder, format=target, out_dir=tmp_path / "out")))
    assert set(run) == set(SOURCES)
    for name, record in run.items():
        outcome = record.outcome
        assert outcome.execution == "committed" and outcome.status in ("ok", "warn"), (name, record.primary)
        plan, written = outcome.metadata, outcome.tag_write
        assert written is not None and written.matched and written.plan_revision == plan.plan_revision
        assert plan.mark_state == "final" and outcome.output.plan_revision == plan.plan_revision
        (path,) = outcome.published_paths
        tags = read_back(path)
        assert {key: tags.fields[key].value for key in ("title", "artist", "album")} == {
            "title": "Night Bus", "artist": "Test Artist", "album": "Demos"}
        mark = pipeline.read_processing_mark(tags)
        assert mark.validity == "valid" and mark.target_lufs == -16.0 and mark.mode == outcome.gain.action
        # ID3v2.3 stores YYYY only (MD23, A1): a precision warning, one roll-up, never silent.
        id3 = path.suffix in (".mp3", ".aiff", ".aif", ".wav")
        date = status_of(record, "field:date")
        if id3:
            assert tags.fields["date"].value == "2026"
            assert (date.status, date.loss_kind, date.requested, date.effective) == (
                "dropped_format_limit", "precision", "2026-10", "2026")
            assert "metadata_precision_loss" in {w.code for w in outcome.warnings}
        else:
            assert tags.fields["date"].value == "2026-10" and date.status == "kept"
        independent = mutagen.File(path)  # a second parser, not Norma's reader
        assert independent is not None and independent.tags is not None and len(independent.tags) >= 4
    assert ledger(folder) == before  # TST-S-03: sources unchanged
    assert residue(tmp_path) == []


@real
def test_wav_info_mode_writes_the_info_subset_through_the_render(tmp_path: Path) -> None:
    """R8/MD30: under ``--wav-tags info`` the render writes INFO; no ID3 chunk, no mark (not storable)."""
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    record = by_name(normalise(options(tmp_path, source, format="wav", wav_tags="info")))["take.flac"]
    outcome = record.outcome
    assert outcome.execution == "committed" and outcome.tag_write.matched, record.primary
    assert outcome.metadata.mark_state == "not_storable"
    tags = read_back(outcome.published_paths[0])
    assert set(tags.stores) == {"riff-info"} and tags.fields["title"].value == "Night Bus"
    assert pipeline.read_processing_mark(tags).validity == "absent"


@real
def test_explicit_opinions_override_source_values_and_report_conflicts(tmp_path: Path) -> None:
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    run_options = options(tmp_path, source, artist="Anna Rossi", genre="Demo", clear_tag=("album",))
    record = by_name(normalise(run_options))["take.flac"]
    tags = read_back(record.outcome.published_paths[0])
    assert tags.fields["artist"].value == "Anna Rossi" and tags.fields["genre"].value == "Demo"
    assert "album" not in tags.fields  # R1: cleared
    plan = record.outcome.metadata
    assert [(c.key, c.old_value, c.new_value) for c in plan.conflicts] == [("field:artist", "Test Artist", "Anna Rossi")]
    assert status_of(record, "field:album").status == "removed_by_request"
    assert status_of(record, "field:genre").status == "added"


@real
def test_strict_conflicts_fail_before_measurement_and_render(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "in"
    generate(folder / "take.flac", "-c:a", "flac", tags=TAGS)
    generate(folder / "other.flac", "-c:a", "flac", tags={"title": "Other"})
    calls: list[Path] = []
    real_render = pipeline.render

    def counted(source, *args, **kwargs):
        calls.append(source)
        return real_render(source, *args, **kwargs)

    monkeypatch.setattr(pipeline, "render", counted)
    run = by_name(normalise(options(tmp_path, folder, artist="Anna Rossi", strict_conflicts=True)))
    record = run["take.flac"]
    assert (record.outcome.status, record.outcome.reason_code, record.outcome.execution) == (
        "failed", "metadata_conflict_strict", "not_written")
    assert record.primary.stage == "metadata_planned" and record.outcome.measurement_in is None
    assert run["other.flac"].outcome.execution == "committed"  # no source artist: no conflict
    assert calls == [folder / "other.flac"]
    assert not (folder / "Normalised" / "take.flac").exists()


@real
def test_loss_policy_error_fails_before_rendering_and_warn_lists_one_row_per_loss(tmp_path: Path,
                                                                                  monkeypatch) -> None:
    """WAV ``info`` cannot hold composer or album artist (MD23): two whole-value roll-ups."""
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac",
                      tags={**TAGS, "composer": "Test Composer", "album_artist": "Test Artist G"})
    warn = by_name(normalise(options(tmp_path, source, format="wav", wav_tags="info",
                                     out_dir=tmp_path / "warn")))["take.flac"]
    assert (warn.outcome.status, warn.outcome.reason_code) == ("warn", "metadata_loss_warn")
    dropped = {s.key for s in warn.outcome.metadata.statuses if s.status == "dropped_format_limit"}
    assert dropped == {"field:composer", "field:album_artist"}
    loss = next(w for w in warn.outcome.warnings if w.code == "metadata_loss_warn")
    assert set(loss.detail.split(",")) == dropped
    renders_forbidden(monkeypatch)
    error = by_name(normalise(options(tmp_path, source, format="wav", wav_tags="info", out_dir=tmp_path / "err",
                                      on_metadata_loss="error")))["take.flac"]
    assert (error.outcome.status, error.outcome.reason_code, error.outcome.execution) == (
        "failed", "metadata_loss_error", "not_written")
    assert error.primary.stage == "metadata_planned" and error.outcome.verification is None
    assert not (tmp_path / "err").exists() or not any((tmp_path / "err").rglob("*.wav"))


@real
def test_sidecar_policy_publishes_the_actual_reserved_members(tmp_path: Path) -> None:
    """CR-084: no metadata-loss refusal once the members can be staged."""
    import json

    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags={**TAGS, "composer": "Test Composer"})
    record = by_name(normalise(options(tmp_path, source, format="wav", wav_tags="info",
                                       on_metadata_loss="sidecar")))["take.flac"]
    outcome = record.outcome
    assert outcome.execution == "committed", record.primary
    assert (outcome.status, outcome.reason_code) == ("warn", "metadata_loss_warn")
    assert set(outcome.published_paths) == {member.final_path for member in outcome.output.members}
    sidecar = next(path for path in outcome.published_paths if path.suffix == ".json")
    document = json.loads(sidecar.read_text())
    assert document["plan_revision"] == outcome.metadata.plan_revision
    assert document["manifest"]["files"]["take.wav"]["composer"] == "Test Composer"
    assert residue(tmp_path) == []
    clean = generate(tmp_path / "in2" / "take.flac", "-c:a", "flac", tags=TAGS)  # no loss: no sidecar needed
    ok = by_name(normalise(options(tmp_path, clean, on_metadata_loss="sidecar")))["take.flac"]
    assert ok.outcome.execution == "committed" and not ok.outcome.metadata.sidecar_requests


# --- The processing mark (MD26, MD37; B-002) ----------------------------------------------------


@real
def test_marked_outputs_are_already_processed_unless_forced(tmp_path: Path) -> None:
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    first = by_name(normalise(options(tmp_path, source)))["take.flac"]
    output = first.outcome.published_paths[0]
    again = by_name(normalise(options(tmp_path, output, out_dir=tmp_path / "again")))["take.flac"]
    assert (again.outcome.status, again.outcome.reason_code, again.outcome.execution) == (
        "skipped", "already_processed", "not_written")
    assert again.primary.stage == "tags_read" and again.outcome.measurement_in is None
    # A new target makes work; at the same target the forced copy would need no output (MD34).
    same = by_name(normalise(options(tmp_path, output, out_dir=tmp_path / "same", force=True)))["take.flac"]
    assert (same.outcome.status, same.outcome.reason_code) == ("skipped", "already_within_tolerance")
    forced = by_name(normalise(options(tmp_path, output, out_dir=tmp_path / "forced", force=True,
                                       target=-18.0)))["take.flac"]
    assert forced.outcome.execution == "committed", forced.primary
    rows = {row.key: row.status for row in forced.outcome.metadata.native_statuses}
    old = next(key for key in rows if key.startswith("source:") and "NORMA" in key)
    new = next(key for key in rows if key.startswith("output:") and "NORMA" in key)
    assert (rows[old], rows[new]) == ("removed_by_design", "added")
    tags = read_back(forced.outcome.published_paths[0])
    assert [item.key for item in tags.native_inventory].count("NORMA") == 1
    assert pipeline.read_processing_mark(tags).validity == "valid"


@real
def test_a_malformed_mark_still_counts_as_present_and_warns_when_forced(tmp_path: Path) -> None:
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags={"title": "T", "NORMA": "v=1;broken"})
    skipped = by_name(normalise(options(tmp_path, source)))["take.flac"]
    assert skipped.outcome.reason_code == "already_processed"
    assert "processing_mark_invalid" in {w.code for w in skipped.outcome.warnings}
    forced = by_name(normalise(options(tmp_path, source, force=True)))["take.flac"]
    assert forced.outcome.execution == "committed"
    assert (forced.outcome.status, forced.outcome.reason_code) == ("warn", "processing_mark_invalid")


@real
def test_no_mark_processed_writes_no_mark(tmp_path: Path) -> None:
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    record = by_name(normalise(options(tmp_path, source, no_mark_processed=True)))["take.flac"]
    assert record.outcome.metadata.mark_state == "omitted" and record.outcome.metadata.plan_revision == 1
    assert pipeline.read_processing_mark(read_back(record.outcome.published_paths[0])).validity == "absent"


# --- Copy within tolerance (audio P4; MD34: the mark never creates work) ------------------------


@real
def test_a_stream_copy_with_no_requested_change_needs_no_output(tmp_path: Path) -> None:
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", level_db=5.1, tags=TAGS)
    probe_level = by_name(normalise(options(tmp_path, source, dry_run=True)))["take.flac"].outcome
    if probe_level.gain is None or probe_level.gain.action != "copy":
        pytest.skip(f"fixture level did not land within tolerance: {probe_level.gain}")
    skipped = by_name(normalise(options(tmp_path, source)))["take.flac"]
    assert (skipped.outcome.status, skipped.outcome.reason_code) == ("skipped", "already_within_tolerance")
    assert skipped.outcome.output is None and not (source.parent / "Normalised").exists()
    changed = by_name(normalise(options(tmp_path, source, artist="Anna Rossi")))["take.flac"]
    assert changed.outcome.execution == "committed" and changed.outcome.target.stream_copy
    assert changed.outcome.reason_code == "already_within_tolerance" and changed.outcome.status == "ok"
    tags = read_back(changed.outcome.published_paths[0])
    assert tags.fields["artist"].value == "Anna Rossi"
    assert pipeline.read_processing_mark(tags).mode == "copy"


# --- Manifest, inference, numbering (MD8, MD9; C-11) ----------------------------------------------


@real
def test_manifest_rows_and_filename_inference_feed_the_plan(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    generate(folder / "01 Glasshouse.wav", "-c:a", "pcm_s16le")
    generate(folder / "02 Bounce.wav", "-c:a", "pcm_s16le")
    manifest = folder / "demos.csv"
    manifest.write_text("file,title,artist,album\n*,,Test Artist,Demos 2026\n02 Bounce.wav,Bounce 7,,\n",
                        encoding="utf-8")
    run = by_name(normalise(options(tmp_path, folder, manifest=manifest, infer_from_filename="{track} {title}",
                                    out_dir=tmp_path / "out")))
    first = read_back(run["01 Glasshouse.wav"].outcome.published_paths[0])
    second = read_back(run["02 Bounce.wav"].outcome.published_paths[0])
    assert (first.fields["title"].value, first.fields["track"].value) == ("Glasshouse", 1)
    assert (second.fields["title"].value, second.fields["track"].value) == ("Bounce 7", 2)
    for tags in (first, second):
        assert (tags.fields["artist"].value, tags.fields["album"].value) == ("Test Artist", "Demos 2026")
    assert status_of(run["02 Bounce.wav"], "field:title").provenance == "manifest"
    assert status_of(run["01 Glasshouse.wav"], "field:artist").provenance == "manifest_default"


@real
def test_an_invalid_manifest_row_fails_only_its_file(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "in"
    generate(folder / "a.wav", "-c:a", "pcm_s16le")
    generate(folder / "b.wav", "-c:a", "pcm_s16le")
    manifest = folder / "m.csv"
    manifest.write_text("file,date\na.wav,Spring\nb.wav,2026\n", encoding="utf-8")
    run = by_name(normalise(options(tmp_path, folder, manifest=manifest)))
    assert (run["a.wav"].outcome.status, run["a.wav"].outcome.reason_code) == ("failed", "manifest_row_invalid")
    assert run["a.wav"].outcome.measurement_in is None
    assert run["b.wav"].outcome.execution == "committed"
    assert read_back(run["b.wav"].outcome.published_paths[0]).fields["date"].value == "2026"


def test_bad_run_level_metadata_inputs_are_usage_errors(tmp_path: Path) -> None:
    items: tuple = ()
    base = RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=tmp_path)
    with pytest.raises(UsageError):
        pipeline.MetadataInputs(items, replace(base, infer_from_filename="{nope}"))
    with pytest.raises(UsageError):
        pipeline.MetadataInputs(items, replace(base, artwork=tmp_path / "missing.png"))
    with pytest.raises(UsageError):
        pipeline.MetadataInputs(items, replace(base, manifest=tmp_path / "missing.csv"))
    with pytest.raises(UsageError):
        pipeline.MetadataInputs(items, replace(base, tag=(("compilation", "perhaps"),)))
    with pytest.raises(UsageError):
        pipeline.MetadataInputs(items, replace(base, date="Spring"))


# --- Artwork (MD24; TST-M-06) -------------------------------------------------------------------


@real
@pytest.mark.parametrize("target", ["flac", "alac", "mp3"])
def test_supplied_artwork_is_prepared_and_embedded(tmp_path: Path, target: str) -> None:
    if shutil.which("sips") is None:
        pytest.skip("macOS sips required")
    cover = png(tmp_path / "art" / "cover.png", 12, 10)
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s24le", tags=TAGS)
    record = by_name(normalise(options(tmp_path, source, format=target, artwork=cover)))["take.wav"]
    outcome = record.outcome
    assert outcome.execution == "committed", record.primary
    tags = read_back(outcome.published_paths[0])
    assert len(tags.artwork) == 1
    picture = tags.artwork[0].value
    assert picture.data == cover.read_bytes() and picture.picture_type == 3 and picture.mime == "image/png"
    assert status_of(record, "picture:new").status == "added"


@real
def test_sips_missing_when_artwork_is_needed_is_an_environment_error(tmp_path: Path, monkeypatch) -> None:
    cover = png(tmp_path / "art" / "cover.png")
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s24le")

    def missing():
        raise EnvironmentError_("required tool not found on PATH: sips")

    monkeypatch.setattr(pipeline, "locate_sips_tool", missing)
    real_render = pipeline.render
    renders_forbidden(monkeypatch)
    before = ledger(tmp_path)
    with pytest.raises(EnvironmentError_, match="sips"):
        normalise(options(tmp_path, source, artwork=cover))
    assert ledger(tmp_path) == before  # exit 3 before any write
    # Without an image to prepare, sips is never needed.
    monkeypatch.setattr(pipeline, "render", real_render)
    assert by_name(normalise(options(tmp_path, source)))["take.wav"].outcome.execution == "committed"


@real
def test_an_unusable_manifest_image_is_artwork_invalid_for_that_file(tmp_path: Path) -> None:
    if shutil.which("sips") is None:
        pytest.skip("macOS sips required")
    folder = tmp_path / "in"
    generate(folder / "a.wav", "-c:a", "pcm_s16le")
    generate(folder / "b.wav", "-c:a", "pcm_s16le")
    (folder / "broken.png").write_bytes(b"not an image at all")
    manifest = folder / "m.csv"
    manifest.write_text("file,artwork\na.wav,broken.png\n", encoding="utf-8")
    run = by_name(normalise(options(tmp_path, folder, manifest=manifest)))
    assert (run["a.wav"].outcome.status, run["a.wav"].outcome.reason_code) == ("failed", "artwork_invalid")
    assert run["a.wav"].primary.stage == "metadata_prepared" and run["a.wav"].outcome.measurement_in is None
    assert run["b.wav"].outcome.execution == "committed"


# --- CAF and raw ADTS (CR-081) --------------------------------------------------------------------


@real
def test_caf_has_a_typed_non_failure_outcome(tmp_path: Path) -> None:
    source = generate(tmp_path / "in" / "take.caf", "-c:a", "pcm_s24le")
    run_options = options(tmp_path, source)
    record = by_name(normalise(run_options))["take.caf"]
    assert (record.outcome.status, record.outcome.reason_code) == ("warn", "metadata_loss_warn"), record.primary
    assert record.outcome.execution == "committed"
    refused = by_name(normalise(options(tmp_path, source, on_metadata_loss="error",
                                        out_dir=tmp_path / "err")))["take.caf"]
    assert (refused.outcome.status, refused.outcome.reason_code) == ("failed", "metadata_loss_error")
    tools = pipeline.locate_tools(run_options, need_ffmpeg=False)
    shown = pipeline.inspect_tags(pipeline.discover_inputs(run_options).items, replace(run_options,
                                  command="tags_show"), tools)
    (inspection,) = shown.inspections
    assert (inspection.outcome.status, inspection.outcome.reason_code) == ("warn", "metadata_loss_warn")
    assert inspection.source_tags is None and inspection.primary is None


# --- Dry run ------------------------------------------------------------------------------------


@real
def test_dry_run_shows_the_final_initial_plan_and_writes_nothing(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    generate(folder / "take.flac", "-c:a", "flac", tags=TAGS)
    before = ledger(tmp_path)
    record = by_name(normalise(options(tmp_path, folder, format="mp3", dry_run=True)))["take.flac"]
    plan = record.outcome.metadata
    assert record.outcome.execution == "plan" and plan.mark_state == "pending" and not plan.conditional
    assert status_of(record, "field:date").loss_kind == "precision"
    assert ledger(tmp_path) == before


# --- tags plan (MD27, MD35; TST-M-07) -------------------------------------------------------------


def plan_tags(tmp_path: Path, *inputs: Path, **changes) -> pipeline.ReadOnlyRun:
    run_options = replace(RunOptions(command="tags_plan", ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=inputs),
                          **changes)
    tools = pipeline.locate_tools(run_options, need_ffmpeg=False)
    return pipeline.plan_tags_items(pipeline.discover_inputs(run_options).items, run_options, tools)


@pytest.fixture()
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    started: list[tuple] = []

    def forbidden(argv, *args, **kwargs):
        started.append(tuple(argv))
        raise AssertionError(f"tags plan started FFmpeg: {argv}")

    monkeypatch.setattr(ffmpeg, "run_process", forbidden)
    return started


@real
def test_tags_plan_starts_no_ffmpeg_and_writes_nothing(tmp_path: Path, no_ffmpeg) -> None:
    folder = tmp_path / "in"
    generate(folder / "take.wav", "-c:a", "pcm_s24le", tags=TAGS)
    generate(folder / "take.flac", "-c:a", "flac", tags=TAGS)
    before = ledger(tmp_path)
    run = plan_tags(tmp_path, folder)
    by = {i.outcome.input.source_path.name: i for i in run.inspections}
    assert no_ffmpeg == [] and ledger(tmp_path) == before
    same = by["take.wav"].outcome  # a same-format WAV may copy: two documentary alternatives (MD35)
    assert same.execution == "plan" and same.metadata is None and same.output is None
    assert [c.assumption for c in same.conditional_plans] == ["stream_copy", "encode"]
    assert all(c.metadata.conditional and c.metadata.mark_state == "pending" for c in same.conditional_plans)
    assert same.conditional_plans[0].output.audio_member.final_path == folder / "Normalised" / "take.wav"


@real
def test_tags_plan_with_a_format_change_shows_one_encode_plan(tmp_path: Path, no_ffmpeg) -> None:
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    (inspection,) = plan_tags(tmp_path, source, format="mp3").inspections
    outcome = inspection.outcome
    assert outcome.metadata is not None, (inspection.primary, inspection.diagnostics, outcome)
    assert outcome.conditional_plans == () and outcome.metadata.audio_basis == "encode"
    assert not outcome.metadata.conditional
    assert (outcome.status, outcome.reason_code) == ("warn", "metadata_precision_loss")
    assert outcome.output.audio_member.final_path.name == "take.mp3"


@real
def test_tags_plan_fails_a_file_only_when_every_branch_fails(tmp_path: Path, no_ffmpeg) -> None:
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags={**TAGS, "composer": "Test Composer"})
    (inspection,) = plan_tags(tmp_path, source, format="wav", wav_tags="info",
                              on_metadata_loss="error").inspections
    assert (inspection.outcome.status, inspection.outcome.reason_code) == ("failed", "metadata_loss_error")
    assert inspection.primary.message.startswith("stream_copy:") or inspection.primary.message.startswith("encode:")
    marked = generate(tmp_path / "in2" / "take.flac", "-c:a", "flac", tags={"NORMA": "v=1;broken"})
    (skipped,) = plan_tags(tmp_path, marked).inspections
    assert skipped.outcome.reason_code == "already_processed" and skipped.outcome.execution == "plan"


@real
def test_tags_plan_strict_conflict_and_cancellation(tmp_path: Path, no_ffmpeg) -> None:
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    (inspection,) = plan_tags(tmp_path, source, artist="Anna", strict_conflicts=True).inspections
    assert inspection.outcome.reason_code == "metadata_conflict_strict"
    run_options = RunOptions(command="tags_plan", ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=(source,))
    cancel = threading.Event()
    cancel.set()
    run = pipeline.plan_tags_items(pipeline.discover_inputs(run_options).items, run_options,
                                   pipeline.locate_tools(run_options, need_ffmpeg=False), cancel_event=cancel)
    assert run.cancelled and run.inspections == () and run.not_started == 1


@real
@pytest.mark.parametrize("refused_basis", ["stream_copy", "encode"])
def test_tags_plan_retains_one_refused_alternative(tmp_path: Path, monkeypatch, no_ffmpeg,
                                                  refused_basis: str) -> None:
    from norma.errors import FileFailure

    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le")
    before = ledger(tmp_path)
    original = pipeline.plan_metadata

    def selective(*args, **kwargs):
        if kwargs["audio_basis"] == refused_basis:
            raise FileFailure("metadata_loss_error", "metadata_planned", f"{refused_basis}-only loss")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "plan_metadata", selective)
    (inspection,) = plan_tags(tmp_path, source).inspections
    outcome = inspection.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == ("ok", "ok", "plan")
    assert outcome.metadata is None and outcome.output is None and outcome.published_paths == ()
    assert [branch.assumption for branch in outcome.conditional_plans] == ["stream_copy", "encode"]
    for branch in outcome.conditional_plans:
        if branch.assumption == refused_basis:
            assert branch.metadata is None and branch.output is None
            assert (branch.refusal.reason_code, branch.refusal.stage, branch.refusal.detail) == (
                "metadata_loss_error", "metadata_planned", f"{refused_basis}-only loss")
        else:
            assert branch.refusal is None and branch.metadata.conditional
            assert branch.output.audio_member.temporary_path is None
    assert inspection.primary is None and len(inspection.diagnostics) == 1
    assert inspection.diagnostics[0].message.startswith(f"{refused_basis}:")
    assert no_ffmpeg == [] and ledger(tmp_path) == before


@real
@pytest.mark.parametrize("copy_stage,encode_stage,primary_basis", [
    ("metadata_planned", "metadata_planned", "stream_copy"),
    ("path_planned", "metadata_planned", "encode"),
    ("metadata_planned", "path_planned", "stream_copy"),
])
def test_tags_plan_all_refusals_select_earliest_state_and_keep_diagnostics(
        tmp_path: Path, monkeypatch, no_ffmpeg, copy_stage, encode_stage, primary_basis) -> None:
    from norma.errors import FileFailure

    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le")
    before = ledger(tmp_path)
    stages = {"stream_copy": copy_stage, "encode": encode_stage}

    def refused(*args, **kwargs):
        basis = kwargs["audio_basis"]
        stage = stages[basis]
        reason = "metadata_loss_error" if stage == "metadata_planned" else "unsafe_output"
        secondary = Diagnostic("internal", f"{basis} extra evidence", stage, f"{basis} extra")
        raise FileFailure(reason, stage, f"{basis} refusal", diagnostics=(secondary,))

    monkeypatch.setattr(pipeline, "plan_metadata", refused)
    (inspection,) = plan_tags(tmp_path, source).inspections
    outcome = inspection.outcome
    assert outcome.status == "failed" and outcome.execution == "plan"
    assert inspection.primary.message.startswith(f"{primary_basis}:")
    assert inspection.primary.stage == "metadata_planned"
    assert outcome.reason_code == "metadata_loss_error"
    assert [branch.assumption for branch in outcome.conditional_plans] == ["stream_copy", "encode"]
    assert all(branch.metadata is None and branch.output is None for branch in outcome.conditional_plans)
    assert [branch.refusal.stage for branch in outcome.conditional_plans] == [copy_stage, encode_stage]
    diagnostics = (inspection.primary, *inspection.diagnostics)
    assert len(diagnostics) == 4
    assert {diagnostic.detail for diagnostic in diagnostics} == {
        "stream_copy refusal", "stream_copy extra", "encode refusal", "encode extra"}
    assert outcome.published_paths == () and no_ffmpeg == [] and ledger(tmp_path) == before


# --- Read-back failures (MD22; safety § 5.2) -------------------------------------------------------


@real
@pytest.mark.parametrize("keep_failed", [False, True])
def test_a_tag_read_back_mismatch_fails_verification_tags_and_publishes_nothing(tmp_path: Path, monkeypatch,
                                                                                keep_failed: bool) -> None:
    from norma.metadata import write

    real_verify = write.verify_tags

    def mismatching(path, plan, **kwargs):
        read_back, _ = real_verify(path, plan, **kwargs)
        return read_back, ("store:id3v2:TIT2",)

    monkeypatch.setattr(write, "verify_tags", mismatching)
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    record = by_name(normalise(options(tmp_path, source, format="mp3", keep_failed=keep_failed)))["take.flac"]
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == ("failed", "verification_tags", "not_written")
    assert record.primary.stage == "content_verified" and "store:id3v2:TIT2" in record.primary.message
    assert outcome.tag_write is not None and not outcome.tag_write.matched
    assert outcome.published_paths == () and not (source.parent / "Normalised" / "take.mp3").exists()
    if keep_failed:  # final tag verification failed: the candidate may be kept (safety § 5.2)
        assert outcome.retained_failed_path is not None and outcome.retained_failed_path.is_file()
    else:
        assert outcome.retained_failed_path is None and residue(tmp_path) == []


@real
def test_a_tag_write_that_changed_the_audio_fails_content_verification(tmp_path: Path, monkeypatch) -> None:
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(pipeline, "decoded_digest", lambda *args, **kwargs: next(digests))
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    record = by_name(normalise(options(tmp_path, source)))["take.flac"]
    assert (record.outcome.reason_code, record.primary.stage) == ("verification_tags", "content_verified")
    assert "changed the decoded audio" in record.primary.message
    assert residue(tmp_path) == [] and not (source.parent / "Normalised" / "take.flac").exists()
