"""TK-0413: the report and log seam in the pipeline (D-068; safety §§ 2.6, 2.7, 6; cli § 7.3).

The interface may not import ``safety`` (ARCHITECTURE § 3), so the pipeline builds run-output
requests and runs the barrier for ``analyse --report``. Synthetic tests use real files under
tmp_path for the inputs, config and stand-in tool paths, and replace only the probe step, so
the refusal tests can prove no file was started. A whole-tmp_path ledger shows that nothing
but the published report (and, for ``normalise``, its outputs and log) was written.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from norma import batch, pipeline, safety
from norma.adapters import ffmpeg
from norma.errors import SkipFile, UsageError
from norma.models import InputItem, RunOptions

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
BITEXACT = ("-fflags", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1")

real = pytest.mark.real_ffmpeg


@pytest.fixture(autouse=True)
def path_based_rmtree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The C-04 evidence folder is a ``TemporaryDirectory``; see test_pipeline_doctor.py."""
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


@pytest.fixture(autouse=True)
def no_process_left() -> None:
    yield
    assert ffmpeg._ACTIVE == {}, "a tool call outlived its test"


# --- Helpers ----------------------------------------------------------------------------------


def ledger(root: Path) -> dict[str, tuple]:
    entries: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not path.is_symlink() else None
        entries[path.relative_to(root).as_posix()] = (info.st_mode, info.st_size, info.st_mtime_ns, digest)
    return entries


def added(before: dict[str, tuple], after: dict[str, tuple]) -> set[str]:
    """Names that are new or changed; a removed name is reported too, prefixed with ``-``."""
    changed = {name for name, entry in after.items() if before.get(name) != entry}
    return changed | {f"-{name}" for name in before if name not in after}


def residue(root: Path) -> list[Path]:
    return sorted(root.rglob(".norma-*"))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return Path(os.path.realpath(tmp_path))


@pytest.fixture
def world(root: Path) -> SimpleNamespace:
    """Two inputs, a config file and two stand-in tool files, all real; plus an empty report folder."""
    inputs = root / "in"
    inputs.mkdir()
    sources = [inputs / "take one.wav", inputs / "take two.wav"]
    for number, source in enumerate(sources):
        source.write_bytes(b"RIFF" + bytes([number]) * 64)
    items = [InputItem(source, inputs, PurePosixPath(source.name), (os.stat(source).st_dev, os.stat(source).st_ino),
                       index, True) for index, source in enumerate(sources)]
    tools_folder = root / "tools"
    tools_folder.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (tools_folder / name).write_bytes(b"#!/bin/sh\n")
    config = root / "norma.toml"
    config.write_text("target = -16\n", encoding="utf-8")
    reports = root / "reports"
    reports.mkdir()
    tools = pipeline.ReadOnlyTools(prober=SimpleNamespace(path=tools_folder / "ffprobe"),  # type: ignore[arg-type]
                                   ffmpeg=SimpleNamespace(path=tools_folder / "ffmpeg"))  # type: ignore[arg-type]
    return SimpleNamespace(root=root, sources=sources, items=items, tools=tools, config=config, reports=reports)


def analyse_options(world: SimpleNamespace, **changes) -> RunOptions:
    return replace(RunOptions(command="analyse", ffmpeg=world.tools.ffmpeg.path, cwd=world.root,
                              inputs=tuple(world.sources), jobs=1, config=world.config), **changes)


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the probe: record each started file and end it ``skipped/not_audio`` at once."""
    started: list[Path] = []

    def classify(item, prober, *, cancel_event=None, **kwargs):
        started.append(item.source_path)
        raise SkipFile("not_audio", "probe", "synthetic stand-in")

    monkeypatch.setattr(pipeline, "classify", classify)
    return started


# --- Requests ---------------------------------------------------------------------------------


def test_requests_are_the_safety_boundary_type(root: Path) -> None:
    explicit = pipeline.report_request(root / "levels.csv")
    automatic = pipeline.report_request(str(root / "norma-report.csv"), automatic=True)
    log = pipeline.log_request(root / "norma.log")
    assert pipeline.RunOutputRequest is safety.RunOutput and pipeline.RunOutputError is safety.RunOutputError
    assert (explicit.purpose, explicit.path, explicit.parent_must_exist) == ("report", root / "levels.csv", True)
    assert (automatic.purpose, automatic.path, automatic.parent_must_exist) == (
        "report", root / "norma-report.csv", False)
    assert (log.purpose, log.path, log.parent_must_exist) == ("log", root / "norma.log", True)


@pytest.mark.parametrize("factory", [pipeline.report_request, pipeline.log_request])
def test_requests_need_an_absolute_path(factory) -> None:
    with pytest.raises(ValueError, match="absolute"):
        factory(Path("levels.csv"))


@pytest.mark.parametrize("request_for", [lambda w: pipeline.log_request(w.reports / "norma.log"),
                                         lambda w: pipeline.report_request(w.reports / "r.csv", automatic=True)])
def test_read_only_report_takes_only_an_explicit_report(world, probes, request_for) -> None:
    with pytest.raises(ValueError, match="explicit report"):
        pipeline.analyse_items(world.items, analyse_options(world), world.tools, report=request_for(world))
    assert probes == []


# --- analyse --report: the barrier before any file, publication after the run -----------------


def test_analyse_report_is_checked_first_and_is_the_only_write(world, probes) -> None:
    target = world.reports / "levels.csv"
    before = ledger(world.root)
    order: list[str] = []
    real_preflight = safety.preflight

    def preflight(*args, **kwargs):
        order.append("barrier")
        assert probes == [], "a file was started before the report was checked"
        return real_preflight(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(safety, "preflight", preflight)
        run = pipeline.analyse_items(world.items, analyse_options(world), world.tools,
                                     report=pipeline.report_request(target))
    assert order == ["barrier"] and probes == world.sources
    assert [o.reason_code for o in run.outcomes] == ["not_audio", "not_audio"]
    assert ledger(world.root) == before  # the run itself wrote nothing

    write_set = run.outputs.write_set
    assert write_set.validated and [(d.purpose, d.final_path, d.open_mode) for d in write_set.destinations] == [
        ("report", target, "publish")]
    protected = {(read.path, read.purpose) for read in write_set.protected_reads}
    assert protected == {*((s, "input") for s in world.sources), (world.config, "config"),
                         (world.tools.ffmpeg.path, "tool"), (world.tools.prober.path, "tool")}

    assert run.outputs.publish("report", b"input,status\n") == target
    assert target.read_bytes() == b"input,status\n"
    assert added(before, ledger(world.root)) == {"reports", "reports/levels.csv"}  # folder mtime + report
    assert residue(world.root) == []


def test_analyse_without_report_has_no_outputs(world, probes) -> None:
    before = ledger(world.root)
    run = pipeline.analyse_items(world.items, analyse_options(world), world.tools)
    assert run.outputs is None and ledger(world.root) == before


def test_cancelled_analyse_still_publishes_its_checked_report(world, probes) -> None:
    event = threading.Event()
    event.set()
    target = world.reports / "levels.json"
    run = pipeline.analyse_items(world.items, analyse_options(world), world.tools, cancel_event=event,
                                 report=pipeline.report_request(target))
    assert run.cancelled and run.not_started == 2 and probes == []
    assert run.outputs.publish("report", b"{}") == target and target.read_bytes() == b"{}"


def test_existing_report_needs_overwrite(world, probes) -> None:
    target = world.reports / "levels.csv"
    target.write_bytes(b"last week\n")
    before = ledger(world.root)
    with pytest.raises(UsageError, match="already exists; use --overwrite"):
        pipeline.analyse_items(world.items, analyse_options(world), world.tools,
                               report=pipeline.report_request(target))
    assert probes == [] and ledger(world.root) == before

    run = pipeline.analyse_items(world.items, analyse_options(world, overwrite=True), world.tools,
                                 report=pipeline.report_request(target))
    assert run.outputs.publish("report", b"this week\n") == target
    assert target.read_bytes() == b"this week\n" and residue(world.root) == []


def alias_cases():
    def same_name_other_case(w):
        return w.sources[0].with_name(w.sources[0].name.upper())

    def hard_link_to_input(w):
        link = w.reports / "levels.csv"
        os.link(w.sources[1], link)
        return link

    def hard_link_to_config(w):
        link = w.reports / "levels.csv"
        os.link(w.config, link)
        return link

    def the_tool(w):
        return w.tools.prober.path

    def symlink(w):
        link = w.reports / "levels.csv"
        link.symlink_to(w.root / "elsewhere.csv")
        return link

    def a_folder(w):
        folder = w.reports / "levels.csv"
        folder.mkdir()
        return folder

    def missing_parent(w):
        return w.root / "no such folder" / "levels.csv"

    return [
        pytest.param(same_name_other_case, "is the input this run reads", id="input-by-name"),
        pytest.param(hard_link_to_input, "another name for the input", id="input-hard-link"),
        pytest.param(hard_link_to_config, "another name for the config", id="config-hard-link"),
        pytest.param(the_tool, "is the tool this run reads", id="tool"),
        pytest.param(symlink, "exists as a symlink", id="symlink"),
        pytest.param(a_folder, "exists as a directory", id="folder"),
        pytest.param(missing_parent, "does not exist", id="missing-parent"),
    ]


@pytest.mark.parametrize("overwrite", [False, True], ids=["plain", "overwrite"])
@pytest.mark.parametrize("make, refused", alias_cases())
def test_unsafe_report_is_a_usage_error_before_any_file(world, probes, make, refused, overwrite) -> None:
    target = make(world)
    before = ledger(world.root)
    with pytest.raises(UsageError, match=refused) as caught:
        pipeline.analyse_items(world.items, analyse_options(world, overwrite=overwrite), world.tools,
                               report=pipeline.report_request(target))
    assert caught.value.exit_code == 2
    assert probes == [] and ledger(world.root) == before
    assert not (world.root / "no such folder").exists()


def test_report_folder_removed_after_the_check_is_not_recreated(world, probes) -> None:
    target = world.reports / "levels.csv"
    run = pipeline.analyse_items(world.items, analyse_options(world), world.tools,
                                 report=pipeline.report_request(target))
    world.reports.rmdir()
    with pytest.raises(pipeline.RunOutputError, match="changed since it was checked"):
        run.outputs.publish("report", b"x")
    assert not world.reports.exists() and residue(world.root) == []


def test_report_folder_replaced_after_the_check_is_refused(world, probes) -> None:
    target = world.reports / "levels.csv"
    run = pipeline.analyse_items(world.items, analyse_options(world), world.tools,
                                 report=pipeline.report_request(target))
    world.reports.rename(world.root / "old reports")
    world.reports.mkdir()
    with pytest.raises(pipeline.RunOutputError, match="changed since it was checked"):
        run.outputs.publish("report", b"x")
    assert list(world.reports.iterdir()) == [] and residue(world.root) == []


def test_report_parent_removed_between_pipeline_and_safety_checks_is_not_recreated(world, probes,
                                                                                 monkeypatch) -> None:
    target = world.reports / "levels.csv"
    run = pipeline.analyse_items(world.items, analyse_options(world), world.tools,
                                 report=pipeline.report_request(target))
    publish = safety.publish_run_output
    assert run.outputs is not None

    def remove_parent(*args, **kwargs):
        world.reports.rmdir()  # after the pipeline's recheck, before safety stages anything
        return publish(*args, **kwargs)

    monkeypatch.setattr(safety, "publish_run_output", remove_parent)
    with pytest.raises(pipeline.RunOutputError, match="changed since it was checked"):
        run.outputs.publish("report", b"x")
    assert not world.reports.exists() and residue(world.root) == []


@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "automatic"])
@pytest.mark.parametrize("overwrite", [False, True], ids=["plain", "overwrite"])
def test_report_publication_passes_the_pin_and_only_explicit_overwrite(world, monkeypatch,
                                                                     automatic, overwrite) -> None:
    target = world.reports / "levels.csv"
    request = pipeline.report_request(target, automatic=automatic)
    checked = safety.preflight({}, (request,), (), overwrite=overwrite)
    registry = safety.RunRegistry("report-wiring")
    outputs = pipeline.RunOutputs(checked, (), registry, overwrite=overwrite, requests=(request,))
    calls = []

    def publish(registry_arg, destination, data, **kwargs):
        calls.append((registry_arg, destination, data, kwargs))
        return target

    monkeypatch.setattr(safety, "publish_run_output", publish)
    assert outputs.publish("report", b"rows") == target
    assert len(calls) == 1
    registry_arg, destination, data, kwargs = calls[0]
    assert registry_arg is registry and data == b"rows"
    assert kwargs["pin"] is checked.pins[destination.destination_id]
    assert kwargs["overwrite"] is (overwrite and not automatic)
    assert kwargs["ordinal_retry"] is automatic and kwargs["create_parent"] is automatic
    assert not hasattr(request, "overwrite")  # request carries no replacement authority
    assert registry.close() == ()


def test_report_created_by_another_program_is_kept(world, probes) -> None:
    target = world.reports / "levels.csv"
    run = pipeline.analyse_items(world.items, analyse_options(world), world.tools,
                                 report=pipeline.report_request(target))
    target.write_bytes(b"someone else's\n")
    with pytest.raises(pipeline.RunOutputError, match="appeared while the run was working"):
        run.outputs.publish("report", b"ours\n")
    assert target.read_bytes() == b"someone else's\n" and residue(world.root) == []


def test_tags_show_takes_no_report(world) -> None:
    import inspect

    assert "report" not in inspect.signature(pipeline.inspect_tags).parameters


# --- normalise: requests built by the factory, real tools --------------------------------------


def generate(path: Path, *, level_db: float = -25) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=997:sample_rate=48000:duration=4", "-af", f"volume={level_db}dB", "-ac", "2",
                    *BITEXACT, "-c:a", "pcm_s16le", str(path)], check=True, timeout=60)
    return path


def normalise_options(root: Path, *inputs: Path, **changes) -> RunOptions:
    return replace(RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=root, inputs=tuple(inputs), jobs=1),
                   **changes)


def logging_executor(lines: list[bytes]):
    """Run the batch the way the interface will, opening the log only once every file is planned."""

    def execute(request: pipeline.BatchRequest) -> batch.BatchResult:
        fd = request.outputs.open_log()
        try:
            for line in lines:
                os.write(fd, line)
        finally:
            os.close(fd)
        control = batch.BatchControl(commit_lock=request.commit_lock, cancel_event=request.cancel_event,
                                     force_hook=request.force_hook)
        return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                                   jobs=request.jobs, control=control, handle_sigint=False)

    return execute


def normalise(run_options: RunOptions, **kwargs) -> pipeline.NormaliseRun:
    tools = pipeline.locate_normalise_tools(run_options)
    items = pipeline.discover_inputs(run_options).items
    return pipeline.normalise_items(items, run_options, tools, **kwargs)


@pytest.fixture
def take(root: Path) -> Path:
    return generate(root / "in" / "take.wav")


@real
def test_normalise_dry_run_publishes_only_the_explicit_report(root: Path, take: Path) -> None:
    target = root / "plan.csv"
    before = ledger(root)
    run = normalise(normalise_options(root, take, dry_run=True),
                    run_outputs=[pipeline.report_request(target)])
    assert ledger(root).keys() == before.keys()  # dry run: no output folder, candidate or report yet
    assert run.outputs.publish("report", b"plan\n") == target
    assert {name for name in added(before, ledger(root)) if not name.startswith("home")} == {"plan.csv"}
    assert residue(root) == []


@real
@pytest.mark.parametrize("overwrite", [False, True], ids=["plain", "overwrite"])
def test_normalise_automatic_report_and_log_through_the_factory(root: Path, take: Path, overwrite: bool) -> None:
    log = root / "norma.log"
    log.write_bytes(b"earlier run\n")

    def outputs(plans):
        folder = {plan.audio_member.final_path.parent for plan in plans.values()}.pop()
        return [pipeline.report_request(folder / "norma-report.csv", automatic=True), pipeline.log_request(log)]

    before = ledger(root)
    run = normalise(normalise_options(root, take, overwrite=overwrite),
                    execute=logging_executor([b"this run\n"]), run_outputs=outputs)
    outcome = run.records[0].outcome
    assert (outcome.status, outcome.execution) == ("ok", "committed")
    report_folder = root / "in" / "Normalised"
    (report_folder / "norma-report.csv").write_bytes(b"taken\n")  # the name is taken at publication
    written = run.outputs.publish("report", b"rows\n")
    assert written == report_folder / "norma-report-2.csv" and written.read_bytes() == b"rows\n"
    assert (report_folder / "norma-report.csv").read_bytes() == b"taken\n"
    assert log.read_bytes() == b"earlier run\nthis run\n"  # appended, never truncated
    assert {name for name in added(before, ledger(root)) if not name.startswith("home")} == {
        "in", "in/Normalised", "in/Normalised/take.wav", "in/Normalised/norma-report.csv",
        "in/Normalised/norma-report-2.csv", "norma.log"}
    assert residue(root) == []


@real
@pytest.mark.parametrize("make, refused", [
    pytest.param(lambda root, take: (root / "levels.csv").write_bytes(b"old") and root / "levels.csv",
                 "already exists", id="exists"),
    pytest.param(lambda root, take: os.link(take, root / "levels.csv") or root / "levels.csv",
                 "another name for the input", id="input-hard-link"),
    pytest.param(lambda root, take: root / "missing" / "levels.csv", "does not exist", id="missing-parent"),
])
def test_normalise_unsafe_explicit_report_writes_nothing(root: Path, take: Path, make, refused: str) -> None:
    target = make(root, take)
    before = ledger(root)
    with pytest.raises(UsageError, match=refused):
        normalise(normalise_options(root, take), execute=logging_executor([]),
                  run_outputs=[pipeline.report_request(target)])
    after = ledger(root)
    assert {name for name in added(before, after) if not name.startswith("home")} == set()
    assert not (root / "in" / "Normalised").exists() and residue(root) == []


@real
def test_real_analyse_report_is_the_only_write(root: Path, take: Path) -> None:
    run_options = replace(RunOptions(command="analyse", ffmpeg=REAL_FFMPEG, cwd=root, inputs=(take,), jobs=1))
    tools = pipeline.locate_tools(run_options, need_ffmpeg=True)
    items = pipeline.discover_inputs(run_options).items
    target = root / "levels.csv"
    before = ledger(root)
    run = pipeline.analyse_items(items, run_options, tools, report=pipeline.report_request(target))
    assert run.outcomes[0].status == "ok" and run.outcomes[0].measurement_in is not None
    assert ledger(root) == before
    assert run.outputs.publish("report", b"rows\n") == target
    assert added(before, ledger(root)) == {"levels.csv"}
    assert residue(root) == []
