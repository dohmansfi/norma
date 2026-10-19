"""TK-0416: IC-2 F-6 and CR-058 in the pipeline (D-072; cli § 7.3, safety § 6).

The automatic report folder is chosen by ``automatic_report_folder`` from the planned audio
folders (skipped and failed plans count) with the ``-o`` folder as the fallback. The
verification pass receives pass 1's ``decoded_frames`` so the source is decoded once.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma import batch, pipeline
from norma.audio import measure as meter
from norma.models import RunOptions

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")

real = pytest.mark.real_ffmpeg


def plans(*folders: Path) -> dict[int, SimpleNamespace]:
    return {index: SimpleNamespace(audio_member=SimpleNamespace(final_path=folder / f"f{index}.wav"))
            for index, folder in enumerate(folders)}


def run_options(tmp_path: Path, **changes) -> RunOptions:
    return replace(RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=tmp_path / "start"), **changes)


# --- F-6: the folder choice ---------------------------------------------------------------------


def test_sole_planned_folder_wins_over_out_dir(tmp_path: Path) -> None:
    folder = tmp_path / "out"
    assert pipeline.automatic_report_folder(plans(folder, folder), run_options(tmp_path, out_dir=folder)) == folder
    other = tmp_path / "in" / "Normalised"
    assert pipeline.automatic_report_folder(plans(other), run_options(tmp_path)) == other


def test_no_planned_destination_uses_out_dir_when_given(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert pipeline.automatic_report_folder({}, run_options(tmp_path, out_dir=out)) == out.resolve()


def test_no_planned_destination_without_out_dir_uses_the_start_folder(tmp_path: Path) -> None:
    assert pipeline.automatic_report_folder({}, run_options(tmp_path)) == tmp_path / "start"


def test_several_folders_use_the_start_folder_with_or_without_out_dir(tmp_path: Path) -> None:
    # IC-2 recheck R-3 / C-7: the -o folder is only the fallback for *no* planned folder.
    two = plans(tmp_path / "a", tmp_path / "b")
    assert pipeline.automatic_report_folder(two, run_options(tmp_path)) == tmp_path / "start"
    out = tmp_path / "out"
    assert pipeline.automatic_report_folder(two, run_options(tmp_path, out_dir=out)) == tmp_path / "start"


def test_recursive_out_dir_subfolders_report_to_the_start_folder(tmp_path: Path) -> None:
    out = tmp_path / "out"
    options = run_options(tmp_path, out_dir=out, recursive=True)
    assert pipeline.automatic_report_folder(plans(out / "one", out / "two"), options) == tmp_path / "start"


def test_out_dir_is_canonical_through_symlinks(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (tmp_path / "link").symlink_to(real_dir)
    folder = pipeline.automatic_report_folder({}, run_options(tmp_path, out_dir=tmp_path / "link" / "new"))
    assert folder == real_dir.resolve() / "new"


# --- CR-058: verify receives the source frame count ----------------------------------------------


@real
def test_verify_receives_pass_one_frames_and_the_source_is_not_recounted(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "in" / "a.wav"
    source.parent.mkdir()
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=997:sample_rate=48000:duration=3", "-af", "volume=-25dB", "-ac", "2",
                    "-fflags", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1", "-c:a", "pcm_s24le",
                    str(source)], check=True, timeout=60)
    seen: list[int | None] = []
    real_verify = pipeline.verify_candidate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("source_frames"))
        return real_verify(*args, **kwargs)

    def no_recount(*args, **kwargs):
        raise AssertionError("the source was decoded a second time")

    monkeypatch.setattr(pipeline, "verify_candidate", spy)
    monkeypatch.setattr(meter, "count_decoded_frames", no_recount)
    options = run_options(tmp_path, ffmpeg=REAL_FFMPEG, inputs=(source.parent,), jobs=1)
    tools = pipeline.locate_normalise_tools(options)
    items = pipeline.discover_inputs(options).items

    def execute(request: pipeline.BatchRequest) -> batch.BatchResult:
        control = batch.BatchControl(commit_lock=request.commit_lock, cancel_event=request.cancel_event,
                                     force_hook=request.force_hook)
        return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                                   jobs=request.jobs, control=control, handle_sigint=False)

    run = pipeline.normalise_items(items, options, tools, execute=execute)
    outcome = run.records[0].outcome
    assert (outcome.status, outcome.execution) == ("ok", "committed")
    assert seen == [outcome.measurement_in.decoded_frames] and seen[0] == 3 * 48000
