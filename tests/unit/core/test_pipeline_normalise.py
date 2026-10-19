"""TK-0412: the ``normalise`` pipeline (ARCHITECTURE § 4; safety §§ 2.6, 5, 6; D-065, D-066).

Real-tool tests generate small untagged inputs under tmp_path with the installed FFmpeg,
run the whole chain through ``batch.execute_batch`` and check outputs independently: the
output is re-measured with the project's meter, sources are compared by SHA-256 and times,
and a ledger of tmp_path shows what a dry run or a refusal left behind. Interrupt tests
replace only the render step with a long tool call writing into the real allocated
candidate, so the first and second interrupts land while a process is in flight.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from norma import batch, pipeline, safety
from norma.adapters import ffmpeg
from norma.adapters.ffprobe import locate_ffprobe, probe
from norma.audio.measure import measure
from norma.errors import UsageError
from norma.models import (
    InputItem,
    ProbeResult,
    RunOptions,
    SourceTags,
)
from norma.safety import RunOutput

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
REAL_FFPROBE = Path("/usr/local/bin/ffprobe")
BITEXACT = ("-fflags", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1")
TONE = "sine=frequency=997:sample_rate={rate}:duration={seconds}"

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


def generate(path: Path, *args: str, rate: int = 48000, seconds: float = 4, level_db: float = -25,
             channels: int = 2, tagged: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = ("-metadata", "title=Demo Take", "-metadata", "artist=Test Artist") if tagged else BITEXACT
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", TONE.format(rate=rate, seconds=seconds), "-af", f"volume={level_db}dB",
                    "-ac", str(channels), *meta, *args, str(path)], check=True, timeout=60)
    return path


def options(tmp_path: Path, *inputs: Path, **changes) -> RunOptions:
    return replace(RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=tuple(inputs),
                              jobs=2), **changes)


def executor(controls: list | None = None):
    def execute(request: pipeline.BatchRequest) -> batch.BatchResult:
        control = batch.BatchControl(commit_lock=request.commit_lock, cancel_event=request.cancel_event,
                                     force_hook=request.force_hook)
        if controls is not None:
            controls.append(control)
        return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                                   jobs=request.jobs, control=control, handle_sigint=False)

    return execute


def normalise(run_options: RunOptions, *, execute=None, **kwargs) -> pipeline.NormaliseRun:
    tools = pipeline.locate_normalise_tools(run_options)
    items = pipeline.discover_inputs(run_options).items
    if execute is None and not run_options.dry_run:
        execute = executor()
    return pipeline.normalise_items(items, run_options, tools, execute=execute, **kwargs)


def by_name(run: pipeline.NormaliseRun) -> dict[str, pipeline.Normalised]:
    return {record.outcome.input.source_path.name: record for record in run.records}


def fingerprint(path: Path) -> tuple[str, int, int]:
    info = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), info.st_size, info.st_mtime_ns


def ledger(root: Path, *, skip: Path | None = None) -> dict[str, tuple]:
    entries: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        if skip is not None and path.is_relative_to(skip):
            continue
        info = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        entries[path.relative_to(root).as_posix()] = (info.st_mode, info.st_size, info.st_mtime_ns, digest)
    return entries


def residue(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob(".norma-*"))


def measured_lufs(path: Path) -> float:
    prober = locate_ffprobe(REAL_FFPROBE)
    executable = ffmpeg.locate_ffmpeg(REAL_FFMPEG)
    result = measure(path, probe(prober, path), executable)
    assert result.integrated_lufs is not None
    return result.integrated_lufs


# --- Synthetic records (no tools) -------------------------------------------------------------


def synthetic_probe(**changes) -> ProbeResult:
    return replace(ProbeResult(
        codec="pcm_s24le", codec_class="lossless", container="wav", audio_stream_index=0, ignored_streams=0,
        sample_rate_hz=48000, bit_depth=24, sample_float=False, sample_format="s32", channels=2,
        channel_layout="stereo", duration_s=4.0, duration_basis="stream", bitrate_bps=2304000,
        has_artwork=False, chapters=0, ffprobe_tags={}, ffprobe_version="synthetic"), **changes)


def tags(**changes) -> SourceTags:
    return replace(SourceTags(fields={}, extras={}, artwork=(), non_tag_items=(), internal_conflicts=(),
                              native_inventory=(), stores={}), **changes)


def test_candidate_formats_cover_what_resolution_can_choose() -> None:
    base = RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=Path("/tmp"))
    assert pipeline._candidate_formats(synthetic_probe(), base) == {"wav"}
    assert pipeline._candidate_formats(synthetic_probe(), replace(base, format="apple")) == {"wav", "alac"}
    assert pipeline._candidate_formats(synthetic_probe(codec="pcm_s16le", container="mov"), base) >= {"alac"}
    mp3 = synthetic_probe(codec="mp3", codec_class="lossy", container="mp3", bit_depth=None)
    assert pipeline._candidate_formats(mp3, replace(base, lossy_sources="skip")) == set()
    assert pipeline._candidate_formats(mp3, replace(base, lossy_sources="lossless")) >= {"flac", "alac", "mp3"}
    assert pipeline._candidate_formats(synthetic_probe(channels=6), base) == set()


def test_writing_run_needs_an_executor(tmp_path: Path) -> None:
    item = InputItem(tmp_path / "x.wav", tmp_path, PurePosixPath("x.wav"), (1, 2), 0, True)
    tools = pipeline.NormaliseTools.__new__(pipeline.NormaliseTools)
    with pytest.raises(ValueError, match="execute"):
        pipeline.normalise_items([item], options(tmp_path), tools)


@pytest.mark.parametrize("changes, refused", [({"in_place": True}, "--in-place")])
def test_run_level_refusals_come_before_any_file(tmp_path: Path, changes: dict, refused: str) -> None:
    item = InputItem(tmp_path / "x.wav", tmp_path, PurePosixPath("x.wav"), (1, 2), 0, True)
    tools = pipeline.NormaliseTools.__new__(pipeline.NormaliseTools)
    with pytest.raises(UsageError, match=refused):
        pipeline.normalise_items([item], options(tmp_path, dry_run=True, **changes), tools)


def test_mutagen_is_not_needed_to_load_the_pipeline() -> None:
    """CR-040: with mutagen unimportable, ``doctor`` still loads and reports it (E-03)."""
    code = (
        "import sys\n"
        "sys.modules['mutagen'] = None\n"
        "import norma.pipeline as p\n"
        "assert 'norma.metadata.read' not in sys.modules\n"
        "check = p._inspect_mutagen()\n"
        "assert not check.ok and check.problem.code == 'not_importable', check\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_force_hook_kills_both_adapters_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """CR-045, CR-047: the second interrupt reaches FFmpeg and ffprobe, even if the first hook raises."""
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "force_kill", lambda: calls.append("ffmpeg") or 0)
    monkeypatch.setattr(pipeline, "force_kill_ffprobe", lambda: calls.append("ffprobe") or 0)
    pipeline.force_kill_tools()
    assert calls == ["ffmpeg", "ffprobe"]

    def broken() -> int:
        calls.append("ffmpeg raised")
        raise OSError("gone")

    monkeypatch.setattr(pipeline, "force_kill", broken)
    with pytest.raises(OSError):
        pipeline.force_kill_tools()
    assert calls[-2:] == ["ffmpeg raised", "ffprobe"]


# --- Real tools: the acceptance path ----------------------------------------------------------


@real
def test_space_shortfall_fails_before_allocation_or_render(tmp_path: Path, monkeypatch) -> None:
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s24le")
    before = fingerprint(source)
    estimates: list[tuple[int, float | None, object, safety.SpaceEstimate]] = []
    registries = []
    estimate_space = safety.estimate_space
    registry_type = safety.RunRegistry

    def estimate(size, duration, target, **kwargs):
        result = estimate_space(size, duration, target, **kwargs)
        estimates.append((size, duration, target, result))
        return result

    def registry(*args, **kwargs):
        result = registry_type(*args, **kwargs, free_space=lambda folder: 0)
        registries.append(result)
        return result

    def unexpected(*args, **kwargs):
        pytest.fail("a space shortfall must stop before candidate allocation or render")

    monkeypatch.setattr(safety, "estimate_space", estimate)
    monkeypatch.setattr(safety, "RunRegistry", registry)
    monkeypatch.setattr(registry_type, "allocate", unexpected)
    monkeypatch.setattr(pipeline, "render", unexpected)
    run = normalise(options(tmp_path, source))
    assert len(run.records) == 1 and run.run_errors == ()
    record = run.records[0]
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == ("failed", "disk_full", "not_written")
    assert record.primary is not None and outcome.probe is not None
    assert record.primary.stage == "allocated"
    assert len(estimates) == 1
    size, duration, target, space = estimates[0]
    assert (size, duration, target) == (before[1], outcome.probe.duration_s, outcome.target)
    assert space.required_bytes > 0
    assert len(registries) == 1
    assert registries[0].records() == () and registries[0].reserved_bytes() == 0
    assert registries[0].created_directories == ()
    assert not (source.parent / "Normalised").exists() and residue(tmp_path) == []
    assert fingerprint(source) == before


@pytest.fixture
def lossless(tmp_path: Path) -> Path:
    folder = tmp_path / "in"
    generate(folder / "01 wav24.wav", "-c:a", "pcm_s24le")
    generate(folder / "02 flac16.flac", "-c:a", "flac", "-sample_fmt", "s16", rate=44100, level_db=-30)
    generate(folder / "03 aiff16 mono.aif", "-c:a", "pcm_s16be", rate=44100, channels=1, level_db=-20)
    generate(folder / "04 alac16.m4a", "-c:a", "alac", "-sample_fmt", "s16p", rate=44100, level_db=-28)
    return folder


@real
def test_untagged_lossless_inputs_are_normalised_to_target(lossless: Path, tmp_path: Path) -> None:
    sources = sorted(p for p in lossless.iterdir() if p.is_file())
    before = {p.name: fingerprint(p) for p in sources}
    run = normalise(options(tmp_path, lossless))
    assert not run.cancelled and run.not_started == 0 and run.run_errors == ()
    assert [r.outcome.input.source_path.name for r in run.records] == [p.name for p in sources]
    prober = locate_ffprobe(REAL_FFPROBE)
    for record in run.records:
        outcome = record.outcome
        source = outcome.input.source_path
        expected = lossless / "Normalised" / source.name
        assert (outcome.status, outcome.reason_code, outcome.execution) == ("ok", "ok", "committed"), record
        assert outcome.published_paths == (expected,) and expected.is_file()
        assert record.primary is None and outcome.gain.action == "linear"
        assert outcome.verification.loudness_ok and outcome.verification.peak_ok
        assert abs(outcome.measurement_out.integrated_lufs + 16.0) <= 0.5
        assert abs(measured_lufs(expected) + 16.0) <= 0.5  # independent re-measurement
        written, original = probe(prober, expected), outcome.probe
        assert (written.codec, written.sample_rate_hz, written.channels, written.bit_depth) == (
            original.codec, original.sample_rate_hz, original.channels, original.bit_depth)
        assert outcome.target.format == {"wav": "wav", "flac": "flac", "aiff": "aiff", "mp4": "alac"}[
            original.container]
    assert {p.name: fingerprint(p) for p in sources} == before  # TST-S-03: sources unchanged
    assert residue(tmp_path) == []


@real
def test_tagged_input_is_normalised_with_its_tags_carried(tmp_path: Path) -> None:
    """S4 lifts the D-066 refusal: tags are read, planned, written and read back (TK-0427)."""
    folder = tmp_path / "in"
    generate(folder / "clean.wav", "-c:a", "pcm_s16le")
    generate(folder / "tagged.flac", "-c:a", "flac", tagged=True)
    generate(folder / "tagged.wav", "-c:a", "pcm_s16le", tagged=True)
    sources = sorted(folder.iterdir())
    before = {p.name: fingerprint(p) for p in sources}
    run = by_name(normalise(options(tmp_path, folder)))
    for name in ("clean.wav", "tagged.flac", "tagged.wav"):
        outcome = run[name].outcome
        assert (outcome.status, outcome.reason_code, outcome.execution) == ("ok", "ok", "committed"), run[name]
        assert outcome.tag_write is not None and outcome.tag_write.matched
        assert outcome.metadata.mark_state == "final" and outcome.metadata.plan_revision == 2
        written = pipeline.read_source_tags(folder / "Normalised" / name, "flac" if name.endswith("flac") else "wav")
        mark = pipeline.read_processing_mark(written)
        assert mark.validity == "valid" and mark.target_lufs == -16.0
        if name.startswith("tagged"):
            assert written.fields["title"].value == "Demo Take" and written.fields["artist"].value == "Test Artist"
        else:
            assert written.fields["title"].value == "clean"  # MD10 title fallback
    assert {p.name: fingerprint(p) for p in sources} == before
    assert residue(tmp_path) == []


@real
def test_structurally_broken_file_fails_alone_in_a_mixed_batch(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    generate(folder / "a good.wav", "-c:a", "pcm_s24le")
    broken = generate(folder / "b broken.wav", "-c:a", "pcm_s24le")
    data = broken.read_bytes()
    broken.write_bytes(data[: len(data) // 2])  # the data chunk now declares more than is present
    generate(folder / "c good.flac", "-c:a", "flac")
    run = by_name(normalise(options(tmp_path, folder)))
    assert (run["b broken.wav"].outcome.status, run["b broken.wav"].outcome.reason_code) == ("failed", "truncated")
    for name in ("a good.wav", "c good.flac"):
        assert run[name].outcome.execution == "committed"
        assert abs(measured_lufs(folder / "Normalised" / name) + 16.0) <= 0.5
    assert not (folder / "Normalised" / "b broken.wav").exists()
    assert residue(tmp_path) == []


@real
def test_dry_run_plans_everything_and_writes_nothing(lossless: Path, tmp_path: Path) -> None:
    home = tmp_path / "home"
    before = ledger(tmp_path, skip=home)
    run = normalise(options(tmp_path, lossless, dry_run=True))
    assert ledger(tmp_path, skip=home) == before  # TST-C-03: no output, folder, candidate or report
    assert list((home / "tmp").iterdir()) == []  # the C-04 evidence folder was removed
    assert run.dry_run and run.write_set is not None and run.write_set.validated
    for record in run.records:
        outcome = record.outcome
        assert (outcome.status, outcome.reason_code, outcome.execution) == ("ok", "ok", "plan")
        assert outcome.output.audio_member.final_path == lossless / "Normalised" / outcome.input.source_path.name
        assert outcome.output.audio_member.temporary_path is None and outcome.published_paths == ()
        assert outcome.gain.action == "linear" and outcome.verification is None


@real
def test_dry_run_explicit_report_is_its_only_write(lossless: Path, tmp_path: Path) -> None:
    report = tmp_path / "report.csv"
    run = normalise(options(tmp_path, lossless, dry_run=True), run_outputs=[RunOutput("report", report)])
    assert not (lossless / "Normalised").exists()
    assert run.outputs.publish("report", b"input,status\n") == report
    assert report.read_bytes() == b"input,status\n" and residue(tmp_path) == []


@real
def test_report_aliasing_a_planned_output_is_a_usage_error(lossless: Path, tmp_path: Path) -> None:
    def outputs(plans):
        first = plans[min(plans)].audio_member.final_path
        return [RunOutput("report", first, parent_must_exist=False)]

    with pytest.raises(UsageError, match="also planned"):
        normalise(options(tmp_path, lossless), run_outputs=outputs)
    assert not (lossless / "Normalised").exists() and residue(tmp_path) == []


@real
def test_existing_outputs_skip_on_a_second_run(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    generate(folder / "take.wav", "-c:a", "pcm_s16le")
    first = normalise(options(tmp_path, folder)).records[0].outcome
    output = first.published_paths[0]
    kept = fingerprint(output)
    second = normalise(options(tmp_path, folder)).records[0].outcome
    assert (second.status, second.reason_code, second.execution) == ("skipped", "output_exists", "not_written")
    assert fingerprint(output) == kept and residue(tmp_path) == []


@real
def test_within_tolerance_copy_needs_no_output_unless_converted(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    source = generate(folder / "take.wav", "-c:a", "pcm_s24le")
    level = measured_lufs(source)
    same = normalise(options(tmp_path, folder, target=level)).records[0]
    assert (same.outcome.status, same.outcome.reason_code, same.outcome.execution) == (
        "skipped", "already_within_tolerance", "not_written")
    assert same.outcome.target.stream_copy and not (folder / "Normalised").exists()

    converted = normalise(options(tmp_path, folder, target=level, format="flac")).records[0].outcome
    assert (converted.status, converted.reason_code, converted.execution) == (
        "ok", "already_within_tolerance", "committed")
    assert converted.gain.action == "copy" and converted.target.format == "flac"
    assert abs(measured_lufs(folder / "Normalised" / "take.flac") - level) <= 0.5


@real
def test_cancel_before_planning_starts_nothing(lossless: Path, tmp_path: Path) -> None:
    before = ledger(tmp_path, skip=tmp_path / "home")
    event = threading.Event()
    event.set()
    run = normalise(options(tmp_path, lossless), cancel_event=event)
    assert run.records == () and run.not_started == 4 and run.cancelled and run.outputs is None
    assert ledger(tmp_path, skip=tmp_path / "home") == before


# --- Real tools: interrupts while a process writes the candidate (TST-S-04) ---------------------

STUBBORN_WRITER = """
import signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1], "wb") as handle:
    handle.write(b"RIFF partial render")
    handle.flush()
    time.sleep(60)
"""


def interrupting_render(monkeypatch, controls: list, argv_for, *, interrupts: int, on_file: str):
    """Replace render for ``on_file`` only: a long tool call writes the candidate, then we interrupt."""
    real_render = pipeline.render
    seen: dict = {}

    def fake(source, identity, probe_result, plan, target, candidate, env, evidence, **kwargs):
        if source.name != on_file:
            return real_render(source, identity, probe_result, plan, target, candidate, env, evidence, **kwargs)
        seen["candidate"] = candidate

        def interrupt() -> None:
            end = time.monotonic() + 10
            while not ffmpeg._ACTIVE and time.monotonic() < end:
                time.sleep(0.01)
            while candidate.stat().st_size == 0 and time.monotonic() < end:
                time.sleep(0.01)
            seen["at"] = time.monotonic()
            for _ in range(interrupts):
                controls[0].interrupt()

        thread = threading.Thread(target=interrupt)
        thread.start()
        try:
            ffmpeg.run_process(argv_for(candidate), deadline_s=60.0, cancel_event=kwargs["cancel_event"])
        finally:
            seen["ended"] = time.monotonic()
            thread.join()
        raise AssertionError("the interrupted call returned")

    monkeypatch.setattr(pipeline, "render", fake)
    return seen


@pytest.fixture
def three(tmp_path: Path) -> Path:
    folder = tmp_path / "in"
    for name in ("a.wav", "b.wav", "c.wav"):
        generate(folder / name, "-c:a", "pcm_s16le", seconds=2)
    return folder


@real
def test_first_interrupt_settles_the_running_file_and_leaves_no_candidate(three: Path, tmp_path: Path,
                                                                        monkeypatch) -> None:
    controls: list = []

    def ffmpeg_writer(candidate: Path) -> tuple[str, ...]:
        return ffmpeg.ffmpeg_argv(REAL_FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-re", "-i",
                                  "sine=duration=600", "-f", "wav", str(candidate))

    seen = interrupting_render(monkeypatch, controls, ffmpeg_writer, interrupts=1, on_file="b.wav")
    run = normalise(options(tmp_path, three, jobs=1), execute=executor(controls))
    rows = by_name(run)
    assert rows["a.wav"].outcome.execution == "committed"
    b = rows["b.wav"].outcome
    assert (b.status, b.reason_code, b.execution) == ("skipped", "cancelled", "not_written")
    assert "c.wav" not in rows and run.not_started == 1 and run.cancelled
    assert not seen["candidate"].exists() and residue(tmp_path) == []
    assert sorted(p.name for p in (three / "Normalised").iterdir()) == ["a.wav"]
    assert not controls[0].force_event.is_set()


@real
def test_second_interrupt_kills_at_once_and_leaves_no_candidate(three: Path, tmp_path: Path, make_tool,
                                                               monkeypatch) -> None:
    controls: list = []
    tool = make_tool("stubborn-writer", STUBBORN_WRITER)
    seen = interrupting_render(monkeypatch, controls, lambda candidate: (str(tool), str(candidate)),
                               interrupts=2, on_file="a.wav")
    run = normalise(options(tmp_path, three, jobs=1), execute=executor(controls))
    rows = by_name(run)
    a = rows["a.wav"].outcome
    assert (a.status, a.reason_code, a.execution) == ("skipped", "cancelled", "not_written")
    assert run.not_started == 2 and run.cancelled and controls[0].force_event.is_set()
    assert seen["ended"] - seen["at"] < ffmpeg.TERMINATE_GRACE_S  # SIGTERM-immune writer killed at once
    assert not seen["candidate"].exists() and residue(tmp_path) == []
    assert not (three / "Normalised").exists()  # the run-created empty folder was removed too
