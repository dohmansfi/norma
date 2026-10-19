"""TK-0410: the S1 read-only pipeline slice (ARCHITECTURE § 4 analyse and tags show; D-057).

Most tests replace the lower modules with fakes at the pipeline's import names, so they
exercise sequencing, outcome settlement and cancellation only. The ``real_ffmpeg`` set
generates small inputs under tmp_path with the installed FFmpeg, runs the real chain and
checks with a whole-tmp_path identity ledger that a run writes nothing.
"""

from __future__ import annotations

import ast
import hashlib
import math
import subprocess
import threading
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from norma import pipeline
from norma.errors import EnvironmentError_, FileFailure, SkipFile, UsageError, exit_contribution
from norma.models import (
    Diagnostic,
    FileOutcome,
    InputItem,
    LoudnessMeasurement,
    ProbeResult,
    ProcessingMark,
    RawNativeItem,
    RunOptions,
    SourceTags,
    TagValue,
)

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
REAL_FFPROBE = Path("/usr/local/bin/ffprobe")


# --- Synthetic records -----------------------------------------------------------------------


def item(index: int = 0, name: str = "take.wav") -> InputItem:
    # Immutable C-01 values only; fakes never open these paths.
    return InputItem(Path(f"/synthetic/{name}"), Path("/synthetic"), PurePosixPath(name),
                     (1, 100 + index), index, True)


def items(count: int) -> list[InputItem]:
    return [item(i, f"take-{i}.wav") for i in range(count)]


def probe_result(**changes) -> ProbeResult:
    return replace(ProbeResult(
        codec="pcm_s24le", codec_class="lossless", container="wav", audio_stream_index=0,
        ignored_streams=0, sample_rate_hz=48000, bit_depth=24, sample_float=False, sample_format="s32",
        channels=2, channel_layout="stereo", duration_s=20.0, duration_basis="stream", bitrate_bps=2304000,
        has_artwork=False, chapters=0, ffprobe_tags={}, ffprobe_version="synthetic",
    ), **changes)


def measurement(integrated=-23.0, peak=-10.0, **changes) -> LoudnessMeasurement:
    return replace(LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=integrated, true_peak_dbtp=peak,
        true_peak_linear=10 ** (peak / 20), lra_lu=2.0, threshold_lufs=None,
        integrated_valid=True, peak_valid=True, lra_valid=True, threshold_valid=False,
        integrated_resolution_lu=.001, peak_resolution_linear=.001, dual_mono=False, meter="ebur128",
        value_source="metadata", ffmpeg_version="synthetic", decoded_duration_s=20.0,
        decoded_frames=960000, sample_rate_hz=48000, decoded_nonzero=True, terminal_peak_covered=True,
    ), **changes)


SILENT = measurement(
    signal_state="silent", integrated_lufs=None, true_peak_dbtp=-math.inf, true_peak_linear=0.0,
    lra_lu=None, integrated_valid=False, peak_valid=True, lra_valid=False, decoded_nonzero=False,
    diagnostics=(Diagnostic("silent", "all decoded samples are zero", "measure"),),
)
UNMEASURABLE = measurement(
    signal_state="unmeasurable", integrated_lufs=None, lra_lu=None, integrated_valid=False, lra_valid=False,
    diagnostics=(Diagnostic("unmeasurable", "nonzero signal has no gated integrated loudness", "measure"),),
)


def options(**changes) -> RunOptions:
    values = {"command": "analyse", "ffmpeg": Path("/synthetic/bin/ffmpeg"), "cwd": Path("/synthetic"), "jobs": 1}
    return RunOptions(**(values | changes))


TOOLS = pipeline.ReadOnlyTools(prober=object(), ffmpeg=object())  # type: ignore[arg-type]


def empty_tags(**changes) -> SourceTags:
    return replace(SourceTags(fields={}, extras={}, artwork=(), non_tag_items=(), internal_conflicts=(),
                              native_inventory=(), stores={}), **changes)


class Fakes:
    """Record every lower-module call; per-file behaviour comes from the dictionaries."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, str]] = []
        self.probes: dict[str, object] = {}
        self.measurements: dict[str, object] = {}
        self.tags: dict[str, object] = {}
        self.measure_kwargs: list[dict] = []
        monkeypatch.setattr(pipeline, "classify", self.classify)
        monkeypatch.setattr(pipeline, "measure", self.measure)
        monkeypatch.setattr(pipeline, "read_source_tags", self.read_source_tags)

    @staticmethod
    def _result(table: dict[str, object], key: str, default: object):
        value = table.get(key, default)
        if callable(value) and not isinstance(value, type):
            value = value()
        if isinstance(value, BaseException):
            raise value
        return value

    def classify(self, entry, prober, *, cancel_event=None, **kwargs):
        assert prober is TOOLS.prober and not kwargs  # no processed-mark guard in read-only commands
        self.calls.append(("classify", entry.source_path.name))
        return self._result(self.probes, entry.source_path.name, probe_result())

    def measure(self, path, probe, executable, **kwargs):
        assert executable is TOOLS.ffmpeg
        self.calls.append(("measure", path.name))
        self.measure_kwargs.append(kwargs)
        return self._result(self.measurements, path.name, measurement())

    def read_source_tags(self, path, container):
        self.calls.append(("read_source_tags", path.name))
        return self._result(self.tags, path.name, empty_tags())


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    return Fakes(monkeypatch)


def only(run: pipeline.ReadOnlyRun) -> pipeline.Inspection:
    assert len(run.inspections) == 1 and not run.cancelled and run.not_started == 0
    return run.inspections[0]


def assert_read_only(outcome: FileOutcome) -> None:
    assert outcome.execution == "inspect"
    assert outcome.published_paths == () and outcome.retained_failed_path is None
    assert outcome.output is None and outcome.target is None and outcome.metadata is None
    assert outcome.measurement_out is None and outcome.verification is None and outcome.tag_write is None
    assert outcome.conditional_plans == () and outcome.elapsed_s >= 0
    exit_contribution(outcome.status, outcome.reason_code)  # primary is legal for its status


# --- analyse ---------------------------------------------------------------------------------


def test_analyse_linear_plan_is_ok_inspect(fakes: Fakes) -> None:
    inspection = only(pipeline.analyse_items([item()], options(), TOOLS))
    outcome = inspection.outcome
    assert_read_only(outcome)
    assert (outcome.status, outcome.reason_code) == ("ok", "ok")
    assert outcome.gain is not None and outcome.gain.action == "linear"
    assert outcome.gain.gain_db == pytest.approx(7.0)
    assert outcome.probe == probe_result() and outcome.measurement_in == measurement()
    assert outcome.source_codec == "pcm_s24le" and outcome.origin_codec is None
    assert outcome.warnings == () and inspection.primary is None
    assert inspection.source_tags is None and inspection.mark is None
    assert fakes.calls == [("classify", "take.wav"), ("measure", "take.wav")]


def test_analyse_uses_options_for_the_pure_plan_and_meter(fakes: Fakes) -> None:
    outcome = only(pipeline.analyse_items([item()], options(target=-14.0, true_peak=-2.0, no_dual_mono=True),
                                          TOOLS)).outcome
    assert outcome.gain.target_lufs == -14.0 and outcome.gain.ceiling_dbtp == -2.0
    assert outcome.gain.gain_db == pytest.approx(9.0)
    assert outcome.gain.action == "limit"  # predicted peak -1 dBTP is 1 dB over the -2 dBTP ceiling
    assert fakes.measure_kwargs[0]["no_dual_mono"] is True


def test_analyse_predicted_reduce_shortfall_is_warn(fakes: Fakes) -> None:
    fakes.measurements["take.wav"] = measurement(-23.0, -4.0)
    outcome = only(pipeline.analyse_items([item()], options(peak_mode="reduce"), TOOLS)).outcome
    assert (outcome.status, outcome.reason_code) == ("warn", "reduce_shortfall")
    assert outcome.gain.action == "reduce" and outcome.gain.shortfall_lu == pytest.approx(4.0)
    assert [d.code for d in outcome.warnings] == ["reduce_shortfall"]
    assert outcome.measurement_out is None  # a prediction, never a measured output


def test_analyse_within_tolerance_copy_plan_is_ok(fakes: Fakes) -> None:
    fakes.measurements["take.wav"] = measurement(-16.1, -6.0)
    outcome = only(pipeline.analyse_items([item()], options(), TOOLS)).outcome
    assert (outcome.status, outcome.reason_code) == ("ok", "already_within_tolerance")
    assert outcome.gain.action == "copy"


def test_analyse_warning_from_measurement_promotes_to_warn_by_precedence(fakes: Fakes) -> None:
    short = Diagnostic("too_short", "measurable programme is shorter than 3 s", "measure")
    fakes.measurements["take.wav"] = measurement(-23.0, -4.0, decoded_duration_s=2.0, diagnostics=(short,))
    outcome = only(pipeline.analyse_items([item()], options(peak_mode="reduce"), TOOLS)).outcome
    assert (outcome.status, outcome.reason_code) == ("warn", "reduce_shortfall")  # C-13 precedence
    assert [d.code for d in outcome.warnings] == ["too_short", "reduce_shortfall"]

    fakes.measurements["take.wav"] = measurement(-16.0, -6.0, decoded_duration_s=2.0, diagnostics=(short,))
    outcome = only(pipeline.analyse_items([item()], options(), TOOLS)).outcome
    assert (outcome.status, outcome.reason_code) == ("warn", "too_short")


@pytest.mark.parametrize("record, reason", [(SILENT, "silent"), (UNMEASURABLE, "unmeasurable")])
def test_analyse_silent_and_unmeasurable_are_skips_with_a_skip_plan(fakes: Fakes, record, reason) -> None:
    fakes.measurements["take.wav"] = record
    inspection = only(pipeline.analyse_items([item()], options(), TOOLS))
    outcome = inspection.outcome
    assert_read_only(outcome)
    assert (outcome.status, outcome.reason_code) == ("skipped", reason)
    assert outcome.gain.action == "skip" and outcome.gain.gain_db is None
    assert inspection.primary.code == reason and inspection.primary.stage == "measure"
    assert outcome.warnings == ()


@pytest.mark.parametrize("source, action, reason", [
    (measurement(), "dynamic", "ok"),
    (measurement(-16.1, -6.0), "copy", "already_within_tolerance"),
    (SILENT, "skip", "silent"),
    (UNMEASURABLE, "skip", "unmeasurable"),
])
def test_analyse_dynamic_preserves_preview_copy_and_skip_rows(fakes: Fakes, monkeypatch,
                                                            source, action: str, reason: str) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("analyse must not calibrate or render")

    monkeypatch.setattr(pipeline, "calibrate_dynamic", forbidden)
    monkeypatch.setattr(pipeline, "render", forbidden)
    fakes.measurements["take.wav"] = source
    outcome = only(pipeline.analyse_items([item()], options(peak_mode="dynamic"), TOOLS)).outcome
    assert_read_only(outcome)
    assert outcome.gain is not None
    assert outcome.gain.action == action and outcome.reason_code == reason
    assert outcome.gain.dynamic_parameters is None
    assert outcome.gain.dynamic_preview == (action == "dynamic")
    assert fakes.calls == [("classify", "take.wav"), ("measure", "take.wav")]


def test_analyse_requires_a_located_ffmpeg(fakes: Fakes) -> None:
    with pytest.raises(ValueError, match="FFmpeg"):
        pipeline.analyse_items([item()], options(), pipeline.ReadOnlyTools(prober=TOOLS.prober))
    assert fakes.calls == []


def test_per_file_skips_and_failures_never_stop_the_run(fakes: Fakes) -> None:
    fakes.probes["take-1.wav"] = SkipFile("not_audio", "signature", "no recognised audio or container signature")
    fakes.probes["take-2.wav"] = FileFailure("probe_error", "probe", "ffprobe exited 1")
    fakes.measurements["take-3.wav"] = FileFailure("decode_error", "measure", "corrupt packet")
    fakes.probes["take-4.wav"] = SkipFile("unsupported_channels", "probe", "6 audio channels")
    seen: list[str] = []
    run = pipeline.analyse_items(items(6), options(), TOOLS,
                                 on_inspection=lambda i: seen.append(i.outcome.input.source_path.name))
    assert not run.cancelled and run.not_started == 0
    assert [(o.status, o.reason_code) for o in run.outcomes] == [
        ("ok", "ok"), ("skipped", "not_audio"), ("failed", "probe_error"),
        ("failed", "decode_error"), ("skipped", "unsupported_channels"), ("ok", "ok"),
    ]
    assert seen == [f"take-{i}.wav" for i in range(6)]
    assert [o.input.input_index for o in run.outcomes] == list(range(6))
    for outcome in run.outcomes:
        assert_read_only(outcome)
    failed_probe, failed_decode = run.inspections[2], run.inspections[3]
    assert failed_probe.outcome.probe is None and failed_probe.primary.detail == "ffprobe exited 1"
    assert failed_decode.outcome.probe == probe_result() and failed_decode.outcome.measurement_in is None
    assert failed_decode.outcome.gain is None and failed_decode.primary.stage == "measure"
    assert ("measure", "take-1.wav") not in fakes.calls and ("measure", "take-4.wav") not in fakes.calls


def test_unexpected_exception_is_internal_and_contained(fakes: Fakes) -> None:
    fakes.measurements["take-0.wav"] = RuntimeError("parser bug")
    run = pipeline.analyse_items(items(2), options(), TOOLS)
    first, second = run.inspections
    assert (first.outcome.status, first.outcome.reason_code) == ("failed", "internal")
    assert first.primary.stage == "measure" and "RuntimeError: parser bug" in first.primary.detail
    assert "Traceback" in first.primary.detail
    assert (second.outcome.status, second.outcome.reason_code) == ("ok", "ok")


def test_failure_diagnostics_and_earlier_warnings_are_kept(fakes: Fakes) -> None:
    extra = Diagnostic("too_short", "short", "measure")
    fakes.measurements["take.wav"] = FileFailure("measurement_error", "measure", "no coverage", (extra,))
    inspection = only(pipeline.analyse_items([item()], options(), TOOLS))
    assert (inspection.outcome.status, inspection.outcome.reason_code) == ("failed", "measurement_error")
    assert inspection.outcome.warnings == (extra,)  # warnings never replace the primary failure
    assert inspection.primary.code == "measurement_error"


# --- Cancellation ----------------------------------------------------------------------------


def test_cancel_before_the_run_starts_nothing(fakes: Fakes) -> None:
    event = threading.Event()
    event.set()
    run = pipeline.analyse_items(items(3), options(), TOOLS, cancel_event=event)
    assert run.inspections == () and run.not_started == 3 and run.cancelled
    assert fakes.calls == []


def test_cancel_during_a_file_settles_it_cancelled_and_stops(fakes: Fakes) -> None:
    event = threading.Event()

    def interrupted():
        event.set()
        return pipeline.Cancelled()  # what the adapters raise once the event is seen

    fakes.measurements["take-1.wav"] = interrupted
    run = pipeline.analyse_items(items(4), options(), TOOLS, cancel_event=event)
    assert run.cancelled and run.not_started == 2
    assert [(o.status, o.reason_code) for o in run.outcomes] == [("ok", "ok"), ("skipped", "cancelled")]
    cancelled = run.inspections[1]
    assert cancelled.primary.code == "cancelled" and cancelled.primary.stage == "measure"
    assert exit_contribution("skipped", "cancelled") == 130
    assert ("classify", "take-2.wav") not in fakes.calls


def test_cancel_between_stages_runs_no_later_stage(fakes: Fakes) -> None:
    event = threading.Event()

    def probed():
        event.set()
        return probe_result()

    fakes.probes["take-0.wav"] = probed
    run = pipeline.analyse_items(items(2), options(), TOOLS, cancel_event=event)
    assert [(o.status, o.reason_code) for o in run.outcomes] == [("skipped", "cancelled")]
    assert run.outcomes[0].probe == probe_result()  # what was gathered before cancellation stays
    assert run.not_started == 1 and fakes.calls == [("classify", "take-0.wav")]


def test_cancel_on_the_last_file_leaves_nothing_unstarted(fakes: Fakes) -> None:
    event = threading.Event()
    fakes.tags["take-1.wav"] = lambda: (event.set(), pipeline.Cancelled())[1]
    run = pipeline.inspect_tags(items(2), options(command="tags_show"), TOOLS, cancel_event=event)
    assert run.cancelled and run.not_started == 0
    assert run.outcomes[-1].reason_code == "cancelled"


def test_keyboard_interrupt_is_left_to_the_interface(fakes: Fakes) -> None:
    fakes.measurements["take.wav"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        pipeline.analyse_items([item()], options(), TOOLS)


# --- tags show -------------------------------------------------------------------------------


def mark_item(value: bytes) -> RawNativeItem:
    return RawNativeItem("store:vorbis:NORMA", "vorbis", "NORMA", "comment", None, "dn-raw-1", value,
                         "vorbis comment #0")


def test_tags_show_returns_source_tags_and_absent_mark(fakes: Fakes, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "measure", lambda *a, **k: pytest.fail("tags show must not run FFmpeg"))
    tags = empty_tags(fields={"title": TagValue("Demo", "vorbis", "TITLE")})
    fakes.tags["take.wav"] = tags
    inspection = only(pipeline.inspect_tags([item()], options(command="tags_show"),
                                            pipeline.ReadOnlyTools(prober=TOOLS.prober)))
    assert_read_only(inspection.outcome)
    assert (inspection.outcome.status, inspection.outcome.reason_code) == ("ok", "ok")
    assert inspection.source_tags is tags
    assert inspection.mark == ProcessingMark("absent", None, None, None, None)
    assert inspection.outcome.gain is None and inspection.outcome.measurement_in is None
    assert fakes.calls == [("classify", "take.wav"), ("read_source_tags", "take.wav")]


def test_tags_show_malformed_mark_is_warn_processing_mark_invalid(fakes: Fakes) -> None:
    raw = mark_item(b"garbage")
    fakes.tags["take.wav"] = empty_tags(native_inventory=(raw,), stores={"vorbis": (raw.identity,)})
    inspection = only(pipeline.inspect_tags([item()], options(command="tags_show"), TOOLS))
    assert (inspection.outcome.status, inspection.outcome.reason_code) == ("warn", "processing_mark_invalid")
    assert inspection.mark.validity == "malformed"
    assert [d.code for d in inspection.outcome.warnings] == ["processing_mark_invalid"]


def test_tags_show_valid_mark_is_ok_and_not_a_processed_skip(fakes: Fakes) -> None:
    raw = mark_item(b"v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=limit;gain=+5.80;limited=2.60;"
                    b"out_i=-16.02;out_tp=-1.04;origin=pcm_s24le;ffmpeg=N-125892;date=2026-10-05")
    fakes.tags["take.wav"] = empty_tags(native_inventory=(raw,), stores={"vorbis": (raw.identity,)})
    inspection = only(pipeline.inspect_tags([item()], options(command="tags_show"), TOOLS))
    assert (inspection.outcome.status, inspection.outcome.reason_code) == ("ok", "ok")
    assert inspection.mark.validity == "valid" and inspection.mark.target_lufs == -16.0


def test_tags_show_read_failures_are_per_file(fakes: Fakes) -> None:
    fakes.tags["take-0.wav"] = FileFailure("permission_denied", "tags_read", "Permission denied")
    fakes.tags["take-1.wav"] = ValueError("no MD3 store row for container 'caf'")
    run = pipeline.inspect_tags(items(3), options(command="tags_show"), TOOLS)
    assert [(o.status, o.reason_code) for o in run.outcomes] == [
        ("failed", "permission_denied"), ("failed", "internal"), ("ok", "ok"),
    ]
    assert run.inspections[0].source_tags is None and run.inspections[0].mark is None
    assert run.inspections[1].primary.stage == "tags_read"
    assert "no MD3 store row" in run.inspections[1].primary.message


# --- Preflight pass-throughs -----------------------------------------------------------------


def test_locate_tools_without_ffprobe_is_environment_error(make_tool) -> None:
    ffmpeg = make_tool("ffmpeg", 'print("ffmpeg version fake-1")')
    with pytest.raises(EnvironmentError_, match="ffprobe"):
        pipeline.locate_tools(options(ffmpeg=ffmpeg), need_ffmpeg=False)


def test_locate_tools_requires_measurement_filters_only_for_analyse(make_tool) -> None:
    ffmpeg = make_tool("ffmpeg", 'print("ffmpeg version fake-1")')
    make_tool("ffprobe", 'print("ffprobe version fake-1")')
    tools = pipeline.locate_tools(options(ffmpeg=ffmpeg), need_ffmpeg=False)
    assert tools.ffmpeg is None and tools.prober.path == ffmpeg.parent / "ffprobe"
    assert tools.prober.version == "fake-1"
    with pytest.raises(EnvironmentError_, match="filter ebur128.*filter astats"):
        pipeline.locate_tools(options(ffmpeg=ffmpeg), need_ffmpeg=True)


def test_analyse_filters_cover_every_filter_measure_runs() -> None:
    # CR-038: locate_tools requires the adapter's measurement set, which names every filter measure.py runs.
    source = (Path(pipeline.__file__).parent / "audio" / "measure.py").read_text(encoding="utf-8")
    for name in ("ebur128", "ametadata", "apad", "aformat", "astats"):
        assert f"{name}=" in source and name in pipeline.MEASURE_FILTERS


def test_discover_inputs_passes_options_through(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    (folder / "sub").mkdir(parents=True)
    for name in ("b.wav", "a.wav", "sub/c.wav", "notes.txt"):
        (folder / name).write_bytes(b"")
    flat = pipeline.discover_inputs(options(inputs=(folder,)))
    assert [i.relative_path.as_posix() for i in flat.items] == ["a.wav", "b.wav"]
    deep = pipeline.discover_inputs(options(inputs=(folder,), recursive=True), suffix="b")
    assert [i.relative_path.as_posix() for i in deep.items] == ["a.wav", "sub/c.wav"]
    with pytest.raises(UsageError):
        pipeline.discover_inputs(options(inputs=(tmp_path / "missing.wav",)))


def test_pipeline_layering() -> None:
    path = Path(pipeline.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules |= {f"{node.module}.{alias.name}" for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "norma" for alias in node.names}
    modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    # tempfile is allowed only for the C-04 evidence folders (doctor TK-0411, normalise TK-0412).
    assert not {"subprocess", "mutagen", "os", "shutil"} & {m.split(".")[0] for m in modules}
    # S2 (TK-0412): pipeline calls safety for candidates and publication (ARCHITECTURE § 4);
    # it still never imports batch (batch is driven by the interface, § 3) or the interface.
    assert "norma.safety" in modules
    assert not {"norma.batch", "norma.cli", "norma.config", "norma.report"} & modules
    calls = {getattr(node.func, "id", getattr(node.func, "attr", None))
             for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not {"open", "write_text", "write_bytes", "mkdir", "unlink", "rename", "replace"} & calls


# --- Real tools on generated inputs ----------------------------------------------------------


def _generate(path: Path, *args: str) -> None:
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", *args, str(path)],
                   check=True, timeout=60)


def _ledger(root: Path) -> dict[str, tuple]:
    entries: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        entries[path.relative_to(root).as_posix()] = (info.st_mode, info.st_size, info.st_mtime_ns, digest)
    return entries


@pytest.fixture
def generated(tmp_path: Path) -> Path:
    folder = tmp_path / "in"
    folder.mkdir()
    _generate(folder / "01 tone.wav", "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=48000:duration=5",
              "-af", "volume=-20dB", "-ac", "2", "-c:a", "pcm_s24le")
    _generate(folder / "02 silence.wav", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "4",
              "-c:a", "pcm_s16le")
    _generate(folder / "03 six.wav", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
              "-af", "volume=-20dB", "-ac", "6", "-c:a", "pcm_s16le")
    _generate(folder / "04 tagged.flac", "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=44100:duration=4",
              "-af", "volume=-12dB", "-metadata", "title=Demo Take", "-metadata", "artist=Test Artist",
              "-metadata", "NORMA=not-a-mark", "-c:a", "flac")
    (folder / "05 notes.wav").write_text("not audio\n", encoding="utf-8")
    return folder


@pytest.mark.real_ffmpeg
def test_real_analyse_and_tags_show_write_nothing(generated: Path, tmp_path: Path) -> None:
    run_options = options(ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=(generated,))
    analyse_tools = pipeline.locate_tools(run_options, need_ffmpeg=True)
    show_tools = pipeline.locate_tools(run_options, need_ffmpeg=False)
    assert analyse_tools.prober.path == REAL_FFPROBE and show_tools.ffmpeg is None
    discovered = pipeline.discover_inputs(run_options)
    assert [i.source_path.name for i in discovered.items] == [
        "01 tone.wav", "02 silence.wav", "03 six.wav", "04 tagged.flac", "05 notes.wav"]

    before = _ledger(tmp_path)
    analysed = pipeline.analyse_items(discovered.items, run_options, analyse_tools)
    shown = pipeline.inspect_tags(discovered.items, replace(run_options, command="tags_show"), show_tools)
    assert _ledger(tmp_path) == before  # no file created, changed or touched anywhere under tmp_path

    rows = [(o.status, o.reason_code) for o in analysed.outcomes]
    assert rows == [("ok", "ok"), ("skipped", "silent"), ("skipped", "unsupported_channels"),
                    ("ok", "ok"), ("skipped", "not_audio")]
    tone = analysed.outcomes[0]
    assert tone.measurement_in.signal_state == "measurable" and tone.gain.action == "linear"
    assert tone.gain.gain_db == pytest.approx(-16.0 - tone.measurement_in.integrated_lufs)
    assert tone.probe.codec == tone.source_codec == "pcm_s24le"
    assert tone.measurement_in.ffmpeg_version == analyse_tools.ffmpeg.version
    for outcome in analysed.outcomes + shown.outcomes:
        assert_read_only(outcome)

    rows = [(o.status, o.reason_code) for o in shown.outcomes]
    assert rows == [("ok", "ok"), ("ok", "ok"), ("skipped", "unsupported_channels"),
                    ("warn", "processing_mark_invalid"), ("skipped", "not_audio")]
    tagged = shown.inspections[3]
    assert tagged.source_tags.fields["title"].value == "Demo Take"
    assert tagged.source_tags.fields["artist"].value == "Test Artist"
    assert tagged.mark.validity == "malformed"
    assert all(i.source_tags is None for i in analysed.inspections)


@pytest.mark.real_ffmpeg
def test_real_cancel_before_start(generated: Path, tmp_path: Path) -> None:
    run_options = options(ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=(generated,))
    tools = pipeline.locate_tools(run_options, need_ffmpeg=True)
    found = pipeline.discover_inputs(run_options).items
    event = threading.Event()
    event.set()
    run = pipeline.analyse_items(found, run_options, tools, cancel_event=event)
    assert run.inspections == () and run.not_started == len(found) == 5 and run.cancelled
