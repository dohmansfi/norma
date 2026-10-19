"""TST-S-02/TST-S-03 — the run write set, protected reads and retry reservations (safety §§ 2.6, 3, 4.1, 5.3).

Every tree is built inside tmp_path; preflight is checked to write nothing.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import pytest

from norma import safety
from norma.errors import FileFailure, SkipFile, UsageError
from norma.models import InputItem, OutputPlan, RunOptions, SidecarRequest, TargetFormat
from norma.paths import NamingOptions, comparison_key, plan_output
from norma.safety import ReservationBook, RunOutput, preflight, protect, protect_inputs


def target(extension: str = ".wav", fmt: str = "wav") -> TargetFormat:
    return TargetFormat(format=fmt, extension=extension, encoder="pcm_s24le", bit_depth=24,  # type: ignore[arg-type]
                        sample_float=False, sample_rate_hz=48000, channels=2, channel_layout="stereo",
                        dither="none", bitrate_mode="none", requested_bitrate_bps=None,
                        effective_bitrate_bps=None, vbr_quality=None, bitrate_source="not_applicable",
                        capability_basis="ev:test", stream_copy=False, decision="as_requested")


def make_item(path: Path, index: int = 0) -> InputItem:
    info = os.stat(path)
    return InputItem(path, path.parent, PurePosixPath(path.name), (info.st_dev, info.st_ino), index, True)


@pytest.fixture
def area(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "work"
    root.mkdir()
    return root


def source(area: Path, name: str, data: bytes = b"RIFF") -> Path:
    path = area / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def tree_state(root: Path) -> list[tuple[str, int, int, int]]:
    rows = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        rows.append((str(path), info.st_mode, info.st_ino, info.st_mtime_ns))
    return rows


TAGS = [SidecarRequest("tags", "tags_sidecar", None, ".tags.json")]


def plan_for(item: InputItem, *, overwrite: bool = False, requests=(), **naming) -> OutputPlan:  # type: ignore[no-untyped-def]
    return plan_output(item, target(), NamingOptions(**naming), plan_revision=0, sidecar_requests=requests,
                       overwrite=overwrite)


# --- In-place refusal (§ 4.1) -----------------------------------------------------------------


@pytest.mark.parametrize("in_place, no_backup", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("extra", [{}, {"dry_run": True}, {"yes": True, "force": True, "overwrite": True}])
def test_in_place_and_no_backup_are_refused_whatever_else_is_set(tmp_path: Path, in_place: bool,
                                                                  no_backup: bool, extra: dict) -> None:
    options = RunOptions(command="normalise", ffmpeg=Path("/usr/local/bin/ffmpeg"), cwd=tmp_path,
                         in_place=in_place, no_backup=no_backup, **extra)
    with pytest.raises(UsageError) as caught:
        safety.refuse_in_place(options)
    assert caught.value.exit_code == 2


def test_ordinary_options_pass_the_in_place_check(tmp_path: Path) -> None:
    safety.refuse_in_place(RunOptions(command="normalise", ffmpeg=Path("/usr/local/bin/ffmpeg"), cwd=tmp_path))


# --- A clean write set ------------------------------------------------------------------------


def test_clean_plans_give_a_validated_write_set_and_write_nothing(area: Path) -> None:
    first, second = make_item(source(area, "a.wav"), 0), make_item(source(area, "b.wav"), 1)
    config = source(area, "norma.toml", b"x")
    plans = {0: plan_for(first, requests=TAGS), 1: plan_for(second)}
    reads = protect_inputs([first, second]) + (protect(config, "config"),)
    before = tree_state(area)
    checked = preflight(plans, [RunOutput("report", area / "report.csv"), RunOutput("log", area / "run.log")],
                        reads, overwrite=False)
    assert tree_state(area) == before  # no folder, temp, log or report
    assert checked.refusals == {}
    assert checked.write_set.validated and checked.eligible() == (0, 1)
    ids = [d.destination_id for d in checked.write_set.destinations]
    assert ids == ["report", "log", "0:audio", "0:tags", "1:audio"]
    modes = {d.destination_id: d.open_mode for d in checked.write_set.destinations}
    assert modes["log"] == "append" and set(modes.values()) == {"append", "publish"}
    assert checked.write_set.protected_reads == reads
    assert all(plan.collision == "none" for plan in checked.plans.values())
    assert not checked.pins["0:audio"].exists  # Normalised/ is created only after the barrier


# --- Bundle refusals (§ 3 table) ----------------------------------------------------------------


def test_existing_member_without_overwrite_skips_the_whole_bundle(area: Path) -> None:
    item = make_item(source(area, "a.wav"))
    out = area / "Normalised"
    out.mkdir()
    (out / "a.wav.tags.json").write_bytes(b"old")
    checked = preflight({0: plan_for(item, requests=TAGS)}, [], protect_inputs([item]), overwrite=False)
    refusal = checked.refusals[0]
    assert isinstance(refusal, SkipFile) and refusal.reason == "output_exists"
    assert checked.plans[0].collision == "exists_skip"
    assert (out / "a.wav.tags.json").read_bytes() == b"old"


def test_existing_member_with_overwrite_is_authorised(area: Path) -> None:
    item = make_item(source(area, "a.wav"))
    (area / "Normalised").mkdir()
    (area / "Normalised" / "a.wav").write_bytes(b"old")
    checked = preflight({0: plan_for(item, overwrite=True)}, [], protect_inputs([item]), overwrite=True)
    assert checked.refusals == {} and checked.plans[0].collision == "exists_overwrite"
    assert checked.occupants["0:audio"].kind == "regular"


@pytest.mark.parametrize("kind", ["dangling", "link", "directory", "fifo"])
def test_unsafe_entry_types_are_refused_even_with_overwrite(area: Path, kind: str) -> None:
    item = make_item(source(area, "a.wav"))
    out = area / "Normalised"
    out.mkdir()
    final = out / "a.wav"
    if kind == "dangling":
        final.symlink_to(area / "missing.wav")
    elif kind == "link":
        other = source(area, "other.bin", b"keep")
        final.symlink_to(other)
    elif kind == "directory":
        final.mkdir()
    else:
        os.mkfifo(final)
    checked = preflight({0: plan_for(item, overwrite=True)}, [], protect_inputs([item]), overwrite=True)
    assert checked.refusals[0].reason == "unsafe_output"


def test_destination_aliasing_an_input_by_key_is_unsafe_output(area: Path) -> None:
    item = make_item(source(area, "Take.wav"))
    alias = area / "take.WAV"  # same key; on this volume possibly the same file
    plan = OutputPlan(0, (safety.BundleMember("audio", "audio", None, alias, None, comparison_key(alias)),),
                      "none", item.identity, True)
    checked = preflight({0: plan}, [], protect_inputs([item]), overwrite=True)
    assert checked.refusals[0].reason == "unsafe_output"
    assert (area / "Take.wav").read_bytes() == b"RIFF"


def test_existing_destination_hard_linked_to_a_protected_read_is_unsafe_output(area: Path) -> None:
    item = make_item(source(area, "a.wav"))
    config = source(area, "norma.toml", b"cfg")
    (area / "Normalised").mkdir()
    os.link(config, area / "Normalised" / "a.wav")
    checked = preflight({0: plan_for(item, overwrite=True)}, [],
                        protect_inputs([item]) + (protect(config, "config"),), overwrite=True)
    assert checked.refusals[0].reason == "unsafe_output"
    assert "config" in (checked.refusals[0].detail or "")


def test_skipped_inputs_are_still_protected(area: Path) -> None:
    eligible = make_item(source(area, "a.wav"), 0)
    skipped = make_item(source(area, "Normalised/a.wav", b"skipped input"), 1)  # say, already processed
    checked = preflight({0: plan_for(eligible, overwrite=True)}, [], protect_inputs([eligible, skipped]),
                        overwrite=True)
    assert checked.refusals[0].reason == "unsafe_output"


def test_initial_clashes_fail_every_participant_and_keep_their_keys(area: Path) -> None:
    first = make_item(source(area, "Song.wav"), 0)
    second = make_item(source(area, "song.aiff"), 1)
    third = make_item(source(area, "other.wav"), 2)
    checked = preflight({i.input_index: plan_for(i) for i in (first, second, third)}, [],
                        protect_inputs([first, second, third]), overwrite=False)
    assert {i: r.reason for i, r in checked.refusals.items()} == {0: "name_clash", 1: "name_clash"}
    assert checked.plans[0].collision == "clash_between_inputs"
    assert checked.eligible() == (2,)
    keys = [d.comparison_key for d in checked.write_set.destinations]
    assert keys.count(comparison_key(area / "Normalised" / "song.wav")) == 2


def test_unsafe_output_wins_over_name_clash_for_the_same_file(area: Path) -> None:
    first = make_item(source(area, "a.wav"), 0)
    second = make_item(source(area, "A.aiff"), 1)
    out = area / "Normalised"
    out.mkdir()
    (out / "a.wav").symlink_to(area / "nowhere")
    checked = preflight({0: plan_for(first), 1: plan_for(second)}, [], protect_inputs([first, second]),
                        overwrite=False)
    assert checked.refusals[0].reason == "unsafe_output"
    assert checked.refusals[1].reason == "unsafe_output"


def test_ancestry_through_a_link_or_a_file_is_unsafe_output(area: Path) -> None:
    item = make_item(source(area, "a.wav"))
    real = area / "real"
    real.mkdir()
    (area / "via").symlink_to(real, target_is_directory=True)
    checked = preflight({0: plan_for(item, out_dir=area / "via")}, [], protect_inputs([item]), overwrite=False)
    assert checked.refusals[0].reason == "unsafe_output"
    source(area, "blocker", b"file")
    checked = preflight({0: plan_for(item, out_dir=area / "blocker" / "x")}, [], protect_inputs([item]),
                        overwrite=False)
    assert checked.refusals[0].reason == "unsafe_output"


# --- Run-level destinations (§ 2.6, § 6) ------------------------------------------------------


def run_refusal(area: Path, outputs: list[RunOutput], *, extra_reads=(), plans=None,  # type: ignore[no-untyped-def]
                overwrite: bool = False) -> str:
    item = make_item(area / "take.wav") if (area / "take.wav").exists() else make_item(source(area, "take.wav"))
    before = tree_state(area)
    with pytest.raises(UsageError) as caught:
        preflight(plans if plans is not None else {0: plan_for(item)}, outputs,
                  protect_inputs([item]) + tuple(extra_reads), overwrite=overwrite)
    assert caught.value.exit_code == 2
    assert tree_state(area) == before
    return str(caught.value)


def test_log_equal_to_an_input_exits_2_and_the_input_is_untouched(area: Path) -> None:
    source(area, "take.wav", b"source bytes")
    message = run_refusal(area, [RunOutput("log", area / "take.wav")])
    assert "input" in message
    assert (area / "take.wav").read_bytes() == b"source bytes"


def test_log_case_alias_of_an_input_exits_2(area: Path) -> None:
    source(area, "take.wav")
    run_refusal(area, [RunOutput("log", area / "TAKE.WAV")])


@pytest.mark.parametrize("target_name", ["manifest.csv", "ffmpeg"])
def test_log_hard_link_to_a_protected_read_exits_2(area: Path, target_name: str) -> None:
    read = source(area, target_name, b"protected")
    os.link(read, area / "run.log")
    run_refusal(area, [RunOutput("log", area / "run.log")], extra_reads=[protect(read, "manifest")])
    assert read.read_bytes() == b"protected"


def test_log_symlink_exits_2(area: Path) -> None:
    (area / "run.log").symlink_to(source(area, "anything.txt"))
    run_refusal(area, [RunOutput("log", area / "run.log")])


def test_log_with_several_links_exits_2(area: Path) -> None:
    log = source(area, "run.log", b"history")
    os.link(log, area / "copy.log")
    assert "links" in run_refusal(area, [RunOutput("log", log)])


def test_existing_single_link_log_is_accepted_for_append(area: Path) -> None:
    source(area, "run.log", b"history")
    item = make_item(source(area, "take.wav"))
    checked = preflight({0: plan_for(item)}, [RunOutput("log", area / "run.log")], protect_inputs([item]),
                        overwrite=False)
    assert checked.occupants["log"].kind == "regular"


def test_report_equal_to_a_planned_member_exits_2_before_any_file_result(area: Path) -> None:
    source(area, "take.wav")
    (area / "Normalised").mkdir()
    message = run_refusal(area, [RunOutput("report", area / "Normalised" / "take.wav.tags.json")],
                          plans={0: plan_for(make_item(area / "take.wav"), requests=TAGS)})
    assert "tags" in message


def test_existing_report_needs_overwrite_and_template_alias_of_config_exits_2(area: Path) -> None:
    source(area, "report.csv", b"old")
    run_refusal(area, [RunOutput("report", area / "report.csv")])
    item = make_item(source(area, "take.wav"))
    checked = preflight({0: plan_for(item)}, [RunOutput("report", area / "report.csv")], protect_inputs([item]),
                        overwrite=True)
    assert checked.occupants["report"].kind == "regular"
    config = source(area, "norma.toml", b"cfg")
    os.link(config, area / "take.csv")
    run_refusal(area, [RunOutput("template_export", area / "take.csv")], extra_reads=[protect(config, "config")],
                overwrite=True)
    assert config.read_bytes() == b"cfg"


def test_missing_parent_exits_2_except_for_the_automatic_report(area: Path) -> None:
    run_refusal(area, [RunOutput("report", area / "missing" / "r.csv")])
    run_refusal(area, [RunOutput("template_export", area / "missing" / "t.csv")])
    item = make_item(area / "take.wav")
    checked = preflight({0: plan_for(item)}, [RunOutput("report", area / "missing" / "r.csv", parent_must_exist=False)],
                        protect_inputs([item]), overwrite=False)
    assert not checked.pins["report"].exists


def test_two_run_outputs_with_one_name_exit_2(area: Path) -> None:
    run_refusal(area, [RunOutput("report", area / "r.csv"), RunOutput("log", area / "R.CSV")])


def test_run_output_parent_is_canonicalised_through_folder_links(area: Path) -> None:
    real = area / "reports"
    real.mkdir()
    (area / "alias").symlink_to(real, target_is_directory=True)
    item = make_item(source(area, "take.wav"))
    checked = preflight({0: plan_for(item)}, [RunOutput("report", area / "alias" / "r.csv")],
                        protect_inputs([item]), overwrite=False)
    report = next(d for d in checked.write_set.destinations if d.purpose == "report")
    assert report.final_path == real / "r.csv"


# --- Retry reservations (§ 5.3) -----------------------------------------------------------------


@pytest.fixture
def book_setup(area: Path):  # type: ignore[no-untyped-def]
    items = [make_item(source(area, f"{name}.wav"), i) for i, name in enumerate(("a", "b", "c"))]
    plans = {0: plan_for(items[0]), 1: plan_for(items[1], requests=TAGS), 2: plan_for(items[2])}
    reads = protect_inputs(items)
    checked = preflight(plans, [RunOutput("report", area / "report.csv")], reads, overwrite=False)
    return items, plans, reads, ReservationBook(checked, reads)


def retry(item: InputItem, suffix: str, member: str = "tags", purpose: str = "tags_sidecar") -> OutputPlan:
    return plan_for(item, requests=[SidecarRequest(member, purpose, None, suffix)])  # type: ignore[arg-type]


def test_retry_with_an_unchanged_member_set_needs_no_new_reservation(book_setup) -> None:  # type: ignore[no-untyped-def]
    items, plans, _reads, book = book_setup
    assert book.reserve_retry(1, plans[1]) is plans[1]


def test_retry_clashing_with_another_files_reservation_fails_only_the_retrying_file(area: Path,
                                                                                   book_setup) -> None:  # type: ignore[no-untyped-def]
    items, _plans, _reads, book = book_setup
    # a.wav's new sidecar would be b.wav's tags member: b keeps it.
    clash = plan_output(items[0], target(), NamingOptions(), plan_revision=1,
                        sidecar_requests=[SidecarRequest("tags", "tags_sidecar", None, ".x")])
    clash = OutputPlan(1, (clash.audio_member, safety.BundleMember(
        "tags", "tags_sidecar", None, area / "Normalised" / "b.wav.tags.json", None,
        comparison_key(area / "Normalised" / "b.wav.tags.json"))), "none", items[0].identity, False)
    with pytest.raises(FileFailure) as caught:
        book.reserve_retry(0, clash)
    assert caught.value.reason == "name_clash" and "input 1 (tags)" in (caught.value.detail or "")
    assert book.holder(comparison_key(area / "Normalised" / "b.wav.tags.json")) == (1, "tags")


def test_own_retained_key_under_a_different_member_is_a_clash(area: Path, book_setup) -> None:  # type: ignore[no-untyped-def]
    items, _plans, _reads, book = book_setup
    renamed = retry(items[1], ".tags.json", member="cover", purpose="cover_export")
    with pytest.raises(FileFailure) as caught:
        book.reserve_retry(1, renamed)
    assert caught.value.reason == "name_clash"


def test_retry_keys_are_held_for_the_whole_run(area: Path, book_setup) -> None:  # type: ignore[no-untyped-def]
    items, _plans, _reads, book = book_setup
    book.reserve_retry(0, retry(items[0], ".sidecar.json"))
    # Even if file 0 later drops that member or fails, file 2 cannot take the name.
    taker = OutputPlan(1, (plan_for(items[2]).audio_member, safety.BundleMember(
        "tags", "tags_sidecar", None, area / "Normalised" / "A.WAV.SIDECAR.JSON", None,
        comparison_key(area / "Normalised" / "A.WAV.SIDECAR.JSON"))), "none", items[2].identity, False)
    with pytest.raises(FileFailure) as caught:
        book.reserve_retry(2, taker)
    assert caught.value.reason == "name_clash"


def test_retry_member_on_a_run_output_or_protected_read_is_unsafe_output(area: Path, book_setup) -> None:  # type: ignore[no-untyped-def]
    items, _plans, _reads, book = book_setup
    report = area / "report.csv"
    plan = OutputPlan(1, (plan_for(items[0]).audio_member, safety.BundleMember(
        "tags", "tags_sidecar", None, report, None, comparison_key(report))), "none", items[0].identity, False)
    with pytest.raises(FileFailure) as caught:
        book.reserve_retry(0, plan)
    assert caught.value.reason == "unsafe_output"
    plan = OutputPlan(1, (plan_for(items[0]).audio_member, safety.BundleMember(
        "tags", "tags_sidecar", None, items[2].source_path, None, comparison_key(items[2].source_path))),
        "none", items[0].identity, False)
    with pytest.raises(FileFailure) as caught:
        book.reserve_retry(0, plan)
    assert caught.value.reason == "unsafe_output"  # input c is a protected read


def test_retry_member_already_on_disk_skips_output_exists(area: Path, book_setup) -> None:  # type: ignore[no-untyped-def]
    items, _plans, _reads, book = book_setup
    (area / "Normalised").mkdir(exist_ok=True)
    (area / "Normalised" / "a.wav.new.json").write_bytes(b"someone else's")
    with pytest.raises(SkipFile) as caught:
        book.reserve_retry(0, retry(items[0], ".new.json"))
    assert caught.value.reason == "output_exists"
    assert (area / "Normalised" / "a.wav.new.json").read_bytes() == b"someone else's"
