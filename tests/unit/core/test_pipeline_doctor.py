"""TK-0411: ``pipeline.inspect_environment``, the application entry point for ``doctor`` (D-059).

Fake tools (executable Python scripts under tmp_path) cover a PATH without FFmpeg, an FFmpeg
that does not run, listings that lack a filter, an encoder or an alimiter option, and failing
queries. The ``real_ffmpeg`` test runs the installed tools. Every test checks that no temporary
folder survives a call (the hermetic fixture points TMPDIR into tmp_path).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import shutil
import tempfile
from pathlib import Path

import pytest

from norma import pipeline
from norma.adapters import ffmpeg as adapter
from norma.errors import ToolTimeout
from norma.formats.capabilities import TARGETS

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
REAL_FFPROBE = Path("/usr/local/bin/ffprobe")
ONE_ROW = {"evidence_rates_hz": (44_100,), "evidence_layouts": (("stereo", 2),)}

ALL_FILTERS = ("ebur128", "ametadata", "apad", "aformat", "astats", "volume", "aresample", "alimiter", "loudnorm")
ALL_ENCODERS = ("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_s16be", "pcm_s24be", "pcm_s32be",
                "flac", "alac", "alac_at", "libmp3lame", "aac_at")


def listing(kind: str, names: tuple[str, ...]) -> str:
    rows = "\n".join(f" A..... {name:<12} fake {kind}" for name in names)
    return f"{kind.capitalize()}:\n A.... = Audio\n ------\n{rows}\n"


def alimiter_help(names: tuple[str, ...]) -> str:
    rows = "\n".join(f"   {name:<17} <boolean>    ..F.A....T. fake option" for name in names)
    return f"Filter alimiter\nalimiter AVOptions:\n{rows}\n"


@pytest.fixture
def fake_ffmpeg(make_tool):
    """A fake ffmpeg (and, unless told otherwise, ffprobe) answering version, listings and encodes.

    An encode exits 0 without writing anything, so evidence collection runs but proves nothing.
    """

    def factory(*, encoders=ALL_ENCODERS, filters=ALL_FILTERS, alimiter=("level_in", "level", "latency"),
                fail_listing=False, runs=True, ffprobe=True) -> Path:
        body = (
            f"enc = {listing('encoders', encoders)!r}\nfil = {listing('filters', filters)!r}\n"
            f"lim = {alimiter_help(alimiter)!r}\nfail = {fail_listing!r}\nruns = {runs!r}\n"
            "args = sys.argv[1:]\n"
            "if not runs: sys.exit(1)\n"
            "if '-version' in args: print('ffmpeg version fake-7 Copyright')\n"
            "elif '-encoders' in args: sys.exit(1) if fail else print(enc)\n"
            "elif '-filters' in args: print(fil)\n"
            "elif 'filter=alimiter' in args: print(lim)\n"
            "elif 'lavfi' in args: sys.exit(0)\n"
            "else: sys.exit(2)\n"
        )
        path = make_tool("ffmpeg", body)
        if ffprobe:
            make_tool("ffprobe", 'print("ffprobe version fake-7 Copyright")')
        return path

    return factory


@pytest.fixture
def temp_root() -> Path:
    root = Path(tempfile.gettempdir())
    assert list(root.iterdir()) == []
    return root


@pytest.fixture(autouse=True)
def path_based_rmtree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the audit guard assess the evidence folder's removal.

    ``TemporaryDirectory.cleanup`` uses ``shutil.rmtree``, which on macOS deletes relative to
    directory descriptors; the hermetic guard (tests/conftest.py) refuses every fd-relative
    mutation because it cannot see the path. The path-based variant performs the same
    removal with absolute paths the guard checks against tmp_path. Production is unchanged.
    """
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


def assert_no_temporary_folder(root: Path) -> None:
    assert list(root.iterdir()) == [], "a temporary evidence folder survived the call"


def assert_common_facts(result: pipeline.EnvironmentInspection) -> None:
    assert result.runtime.python_version and result.runtime.python_executable.is_absolute()
    assert result.runtime.norma_version
    assert result.platform.system and result.platform.machine and result.platform.basis == adapter.platform_basis()
    assert result.mutagen.ok and result.mutagen.value == importlib.metadata.version("mutagen")


# --- Records and constants -------------------------------------------------------------------


def test_check_needs_a_value_or_a_problem() -> None:
    with pytest.raises(ValueError):
        pipeline.Check()
    problem = pipeline.Problem("missing", "lacks x", ("x",))
    assert pipeline.Check("v").ok and not pipeline.Check(problem=problem).ok
    assert not pipeline.Check(pipeline.Facilities(frozenset(), ("x",)), problem).ok
    assert pipeline.Facilities(frozenset({"b"}), ("a", "b", "c")).missing == ("a", "c")


def test_required_facilities_come_from_their_owners() -> None:
    assert pipeline.DOCTOR_ENCODERS == ("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_s16be",
                                        "pcm_s24be", "pcm_s32be", "flac", "alac", "libmp3lame", "aac_at")
    assert "alac_at" not in pipeline.DOCTOR_ENCODERS  # native alac only (ARCHITECTURE § 9)
    assert set(pipeline.DOCTOR_FILTERS) == set(ALL_FILTERS)
    assert pipeline.LOSSY_TARGETS == tuple(name for name, spec in TARGETS.items() if spec.family == "lossy")
    assert pipeline.EVIDENCE_LAYOUTS == (("mono", 1), ("stereo", 2))  # v1 is mono/stereo only (ADR-0020)


def test_measure_filters_include_the_sample_zero_scan() -> None:
    # CR-038: the adapter's set now names every filter audio/measure.py runs.
    source = (Path(pipeline.__file__).parent / "audio" / "measure.py").read_text(encoding="utf-8")
    for name in ("ebur128", "ametadata", "apad", "aformat", "astats"):
        assert f"{name}=" in source and name in adapter.MEASURE_FILTERS
    assert not hasattr(pipeline, "ANALYSE_FILTERS")


# --- Missing or broken tools never raise ------------------------------------------------------


def test_path_without_ffmpeg(monkeypatch: pytest.MonkeyPatch, temp_root: Path) -> None:
    # TST-C-01 shape (Releases § 5): PATH=/usr/bin:/bin has no FFmpeg on this Mac.
    if any(Path(folder, "ffmpeg").exists() for folder in ("/usr/bin", "/bin")):
        pytest.skip("this system has FFmpeg in /usr/bin or /bin")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    result = pipeline.inspect_environment("ffmpeg")
    assert result.ffmpeg.problem.code == "not_found" and result.ffmpeg.value is None
    assert "ffmpeg" in result.ffmpeg.problem.message
    assert result.ffprobe.problem.code == "not_checked"
    for check in (result.encoders, result.filters, result.alimiter_options, result.capability_evidence):
        assert check.problem.code == "not_checked" and check.value is None
    assert_common_facts(result)
    if Path("/usr/bin/sips").exists():
        assert result.sips.value == Path("/usr/bin/sips")
    assert_no_temporary_folder(temp_root)


@pytest.mark.parametrize("given", ["absolute", "relative"])
def test_given_ffmpeg_path_that_does_not_exist(tmp_path: Path, given: str, temp_root: Path) -> None:
    path = tmp_path / "nowhere" / "ffmpeg" if given == "absolute" else Path("nowhere/ffmpeg")
    result = pipeline.inspect_environment(path)
    assert result.ffmpeg.problem.code == "not_found"
    # ffprobe is looked for beside an absolute --ffmpeg even when ffmpeg itself is missing.
    assert result.ffprobe.problem.code == ("not_found" if given == "absolute" else "not_checked")
    assert result.encoders.problem.code == "not_checked"
    assert_no_temporary_folder(temp_root)


def test_ffmpeg_that_does_not_run(fake_ffmpeg, temp_root: Path) -> None:
    path = fake_ffmpeg(runs=False)
    result = pipeline.inspect_environment(path)
    assert result.ffmpeg.problem.code == "unusable" and str(path) in result.ffmpeg.problem.message
    assert result.ffprobe.value == pipeline.ToolInfo(path.parent / "ffprobe", "fake-7")
    assert result.encoders.problem.code == result.capability_evidence.problem.code == "not_checked"
    assert_no_temporary_folder(temp_root)


def test_ffprobe_missing_beside_ffmpeg(fake_ffmpeg, temp_root: Path) -> None:
    path = fake_ffmpeg(ffprobe=False)
    result = pipeline.inspect_environment(path, **ONE_ROW)
    assert result.ffmpeg.ok and result.ffmpeg.value == pipeline.ToolInfo(path, "fake-7")
    assert result.ffprobe.problem.code == "not_found"
    assert str(path.parent) in result.ffprobe.problem.message
    assert_no_temporary_folder(temp_root)


def test_bare_name_is_found_on_path_and_ffprobe_beside_it(fake_ffmpeg, monkeypatch, temp_root: Path) -> None:
    path = fake_ffmpeg()
    monkeypatch.setenv("PATH", f"{path.parent}:/usr/bin:/bin")
    result = pipeline.inspect_environment(Path("ffmpeg"), **ONE_ROW)
    assert result.ffmpeg.value.path == path and result.ffprobe.value.path == path.parent / "ffprobe"
    assert_no_temporary_folder(temp_root)


def test_listing_failure_is_a_typed_problem(fake_ffmpeg, temp_root: Path) -> None:
    result = pipeline.inspect_environment(fake_ffmpeg(fail_listing=True))
    assert result.ffmpeg.ok
    assert result.encoders.problem.code == result.filters.problem.code == "query_failed"
    assert result.alimiter_options.problem.code == "not_checked"
    assert result.capability_evidence.problem.code == "not_checked"
    assert_no_temporary_folder(temp_root)


# --- Missing facilities -----------------------------------------------------------------------


def test_ffmpeg_missing_a_filter(fake_ffmpeg, temp_root: Path) -> None:
    filters = tuple(name for name in ALL_FILTERS if name not in ("loudnorm", "astats"))
    result = pipeline.inspect_environment(fake_ffmpeg(filters=filters), **ONE_ROW)
    assert result.filters.problem.code == "missing"
    assert result.filters.problem.missing == ("astats", "loudnorm")
    assert result.filters.value.present == frozenset(filters)
    assert "astats" in result.filters.problem.message and "loudnorm" in result.filters.problem.message
    assert result.encoders.ok and result.alimiter_options.ok
    assert_no_temporary_folder(temp_root)


def test_alimiter_without_latency_and_without_the_filter(fake_ffmpeg, temp_root: Path) -> None:
    result = pipeline.inspect_environment(fake_ffmpeg(alimiter=("level",)), **ONE_ROW)
    assert result.alimiter_options.problem.code == "missing"
    assert result.alimiter_options.problem.missing == ("latency",)
    filters = tuple(name for name in ALL_FILTERS if name != "alimiter")
    result = pipeline.inspect_environment(fake_ffmpeg(filters=filters), **ONE_ROW)
    assert result.filters.problem.missing == ("alimiter",)
    assert result.alimiter_options.problem.code == "not_checked"
    assert_no_temporary_folder(temp_root)


def test_missing_encoders_are_named_and_mp3_evidence_is_not_collected(fake_ffmpeg, temp_root: Path) -> None:
    encoders = tuple(name for name in ALL_ENCODERS if name not in ("alac", "libmp3lame"))
    result = pipeline.inspect_environment(fake_ffmpeg(encoders=encoders))
    assert result.encoders.problem.code == "missing"
    assert result.encoders.problem.missing == ("alac", "libmp3lame")  # alac_at never stands in for alac
    rows = result.capability_evidence.value
    assert len(rows) == 8  # 2 lossy targets x 2 rates x 2 layouts
    for row in (r for r in rows if r.target == "mp3"):
        assert row.problem.code == "missing" and row.problem.missing == ("libmp3lame",)
        assert row.records == () and not row.supported
    assert_no_temporary_folder(temp_root)


def test_unconfirmed_evidence_is_an_explicit_gap(fake_ffmpeg, temp_root: Path) -> None:
    # The fake encode writes nothing: the collector runs, confirms nothing, and every choice is a gap.
    result = pipeline.inspect_environment(fake_ffmpeg(), **ONE_ROW)
    assert result.capability_evidence.ok
    mp3, aac = result.capability_evidence.value
    assert (mp3.target, mp3.encoder, mp3.sample_rate_hz, mp3.channels, mp3.channel_layout) == (
        "mp3", "libmp3lame", 44_100, 2, "stereo")
    assert mp3.problem is None and mp3.bases == ("installed_query",)
    assert [record.bitrate_mode for record in mp3.records] == ["cbr", "vbr"]
    assert all(not record.nominal_bitrates_bps and not record.vbr_qualities for record in mp3.records)
    assert not mp3.supported and [c.bitrate_mode for c in mp3.capabilities] == ["cbr", "vbr"]
    assert any("V0" in gap for gap in mp3.gaps)
    # TK-0419 registers TK-0418's AAC collector: it runs, the fake encode confirms no nominal, so a gap.
    assert aac.problem is None and aac.bases == ("installed_query",)
    assert [(r.encoder, r.bitrate_mode, r.nominal_bitrates_bps) for r in aac.records] == [("aac_at", "cbr", ())]
    assert not aac.supported and aac.automatic.reason == "unsupported_target"
    assert any("nominal setting cannot be confirmed" in gap for gap in aac.gaps)
    assert_no_temporary_folder(temp_root)


# --- Temporary folder ---------------------------------------------------------------------------


def test_folder_removed_when_a_collector_fails(fake_ffmpeg, monkeypatch, temp_root: Path) -> None:
    seen: list[Path] = []

    def failing(environment, rate, channels, layout, work_dir: Path, **_):
        work_dir.mkdir(parents=True)
        (work_dir / "partial.mp3").write_bytes(b"\xff\xfb")
        seen.append(work_dir)
        raise ToolTimeout("ffmpeg exceeded its 60 s deadline", deadline_s=60.0)

    monkeypatch.setitem(pipeline._EVIDENCE_COLLECTORS, "libmp3lame", failing)
    result = pipeline.inspect_environment(fake_ffmpeg(), **ONE_ROW)
    mp3 = result.capability_evidence.value[0]
    assert mp3.problem.code == "query_failed" and "deadline" in mp3.problem.message and not mp3.supported
    assert seen and seen[0].is_relative_to(temp_root) and not seen[0].exists()
    assert_no_temporary_folder(temp_root)


def test_folder_removed_on_interrupt(fake_ffmpeg, monkeypatch, temp_root: Path) -> None:
    def interrupted(environment, rate, channels, layout, work_dir: Path, **_):
        work_dir.mkdir(parents=True)
        raise KeyboardInterrupt

    monkeypatch.setitem(pipeline._EVIDENCE_COLLECTORS, "libmp3lame", interrupted)
    with pytest.raises(KeyboardInterrupt):  # the interface owns interruption
        pipeline.inspect_environment(fake_ffmpeg(), **ONE_ROW)
    assert_no_temporary_folder(temp_root)


def test_cleanup_failure_is_reported_not_raised(fake_ffmpeg, monkeypatch, temp_root: Path) -> None:
    class Stubborn(tempfile.TemporaryDirectory):
        def cleanup(self) -> None:
            super().cleanup()
            raise PermissionError(13, "Operation not permitted")

    monkeypatch.setattr(pipeline.tempfile, "TemporaryDirectory", Stubborn)
    result = pipeline.inspect_environment(fake_ffmpeg(), **ONE_ROW)
    assert result.capability_evidence.problem.code == "cleanup_failed"
    assert len(result.capability_evidence.value) == 2


def test_unwritable_temporary_folder_is_a_typed_problem(fake_ffmpeg, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(pipeline.tempfile, "TemporaryDirectory", refuse)
    result = pipeline.inspect_environment(fake_ffmpeg(), **ONE_ROW)
    assert result.capability_evidence.problem.code == "query_failed" and result.capability_evidence.value is None


# --- mutagen and sips -----------------------------------------------------------------------------


def test_mutagen_not_installed_or_not_findable(fake_ffmpeg, monkeypatch) -> None:
    def absent(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(pipeline, "distribution_version", absent)
    result = pipeline.inspect_environment(fake_ffmpeg(), **ONE_ROW)
    assert result.mutagen.problem.code == "not_importable" and result.mutagen.value is None
    monkeypatch.setattr(pipeline, "distribution_version", lambda name: "9.9")
    monkeypatch.setattr(pipeline, "find_spec", lambda name: None)
    result = pipeline.inspect_environment(fake_ffmpeg(), **ONE_ROW)
    assert result.mutagen.problem.code == "not_importable" and "9.9" in result.mutagen.problem.message


def test_sips_not_on_path(fake_ffmpeg, monkeypatch, tmp_path: Path) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    result = pipeline.inspect_environment(fake_ffmpeg(), **ONE_ROW)
    assert result.sips.problem.code == "not_found" and "sips" in result.sips.problem.message


# --- Layering -------------------------------------------------------------------------------------


def test_doctor_uses_tempfile_only_through_temporary_directory() -> None:
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding="utf-8"))
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tempfile"}
    assert used == {"TemporaryDirectory"}
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "tempfile" not in imported


# --- Real tools -----------------------------------------------------------------------------------


def _names(root: Path) -> dict[str, str | None]:
    return {path.relative_to(root).as_posix(): (hashlib.sha256(path.read_bytes()).hexdigest()
                                                if path.is_file() else None)
            for path in sorted(root.rglob("*"))}


@pytest.mark.real_ffmpeg
def test_real_tools(tmp_path: Path, temp_root: Path) -> None:
    before = _names(tmp_path)
    result = pipeline.inspect_environment(REAL_FFMPEG)
    assert _names(tmp_path) == before  # nothing written, temporary encodes included
    assert_no_temporary_folder(temp_root)
    assert_common_facts(result)
    assert result.ffmpeg.value.path == REAL_FFMPEG and result.ffmpeg.value.version
    assert result.ffprobe.value == pipeline.ToolInfo(REAL_FFPROBE, result.ffmpeg.value.version)
    for check in (result.encoders, result.filters, result.alimiter_options):
        assert check.ok and check.value.missing == ()
    assert result.sips.ok and result.sips.value.name == "sips"
    assert result.capability_evidence.ok
    rows = result.capability_evidence.value
    assert [(r.target, r.sample_rate_hz, r.channel_layout) for r in rows] == [
        (target, rate, layout) for target in ("mp3", "aac") for rate in (44_100, 48_000)
        for layout in ("mono", "stereo")]
    for row in rows:
        if row.target == "mp3":
            assert row.problem is None
            assert "installed_query" in row.bases
            assert set(row.bases) <= {"installed_query", "build_matched_evidence"}
            cbr = next(c for c in row.capabilities if c.bitrate_mode == "cbr")
            assert cbr.supported and 128_000 in cbr.nominal_bitrates_bps
            assert all(r.executable_path == REAL_FFMPEG for r in row.records)
            # V0 is confirmed by headers or matched primary source plus encode/decode (F15).
            assert row.supported == (0 in row.automatic.vbr_qualities)
            assert row.supported or any("V0" in gap for gap in row.gaps)
        else:  # TK-0419: TK-0418's AAC collector is registered; stereo CBR is evidenced on this Mac.
            assert row.problem is None and [r.encoder for r in row.records] == ["aac_at"]
            assert all(r.executable_path == REAL_FFMPEG for r in row.records)
            if row.channel_layout == "stereo":
                assert row.supported and 256_000 in row.automatic.nominal_bitrates_bps


def test_a_failing_stats_probe_is_made_once_and_every_mp3_row_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """CR-070 (N-5): a probe that raises is recorded once, not repeated by the collector per MP3 row."""
    from pathlib import Path as _Path

    from norma.adapters import ffmpeg as adapter

    environment = adapter.FFmpegEnvironment(adapter.Executable(_Path("/fake/ffmpeg"), (1, 2), "N-test"),
                                            frozenset({"libmp3lame"}), frozenset(), "test platform")
    probes: list[object] = []

    def failing(env):
        probes.append(env)
        raise ValueError("probe failed")

    monkeypatch.setattr(pipeline, "probe_mp3_encoder_stats", failing)
    monkeypatch.setattr(adapter, "probe_mp3_encoder_stats", failing)
    check = pipeline._inspect_evidence(environment, (44100, 48000), (("stereo", 2), ("mono", 1)))
    rows = [row for row in check.value if row.target == "mp3"]
    assert len(rows) == 4 and len(probes) == 1
    for row in rows:
        assert not row.automatic.supported and not any(c.supported for c in row.capabilities)
