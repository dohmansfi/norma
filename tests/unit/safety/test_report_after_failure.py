"""TST-S-05 — the automatic report survives a run whose files all failed (safety § 6, cli § 7.3, CR-051).

End-of-run cleanup removes the empty folders the run created (§ 5.2). The automatic report is
published after that cleanup and may create its folder again even if every file failed; the
removed folder's identity must not refuse it. A folder someone else replaced is still refused,
and never removed. Every tree is built inside tmp_path; third parties are simulated directly.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import pytest

from norma import safety
from norma.errors import FileFailure
from norma.models import InputItem
from norma.safety import Preflight, RunOutput, RunOutputError, RunRegistry, preflight, protect_inputs

BASE = "norma-report-2026-10-11T031500.csv"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path.resolve() / "short"
    folder.mkdir()
    return folder


def checked_report(root: Path, folder: Path) -> tuple[Preflight, tuple]:
    source = root / "short.wav"
    source.write_bytes(b"RIFF source")
    info = source.stat()
    reads = protect_inputs([InputItem(source, root, PurePosixPath("short.wav"), (info.st_dev, info.st_ino), 0, True)])
    return preflight({}, [RunOutput("report", folder / BASE, parent_must_exist=False)], reads, overwrite=False), reads


def publish(registry: RunRegistry, checked: Preflight, reads: tuple) -> Path:
    destination = checked.write_set.destinations[0]
    return safety.publish_run_output(registry, destination, b"rows\n", overwrite=False, protected=reads,
                                     reserved_keys=(destination.comparison_key,), ordinal_retry=True,
                                     create_parent=True, pin=checked.pins.get(destination.destination_id))


def failed_file(registry: RunRegistry, checked: Preflight, folder: Path) -> None:
    """A file allocates in ``folder`` (creating it), fails, is settled; the run then closes."""
    registry.ensure_folder(folder, checked.pins["report"])
    candidate = registry.write_receipt(0, folder, {"phase": "prepared"})  # any registered private entry
    assert candidate.exists()
    safety.settle(registry, 0, FileFailure("verification_loudness", "verified", "1 s file"))
    assert registry.close() == ()


@pytest.mark.parametrize("depth", [("Normalised",), ("Normalised", "Album")])
def test_the_report_recreates_a_folder_the_run_removed_after_every_file_failed(root: Path, depth: tuple) -> None:
    folder = root.joinpath(*depth)
    checked, reads = checked_report(root, folder)
    registry = RunRegistry("run1")
    failed_file(registry, checked, folder)
    assert not (root / "Normalised").exists()  # the empty run-created folders are gone
    assert registry.created_directories == ()

    written = publish(registry, checked, reads)

    assert written == folder / BASE and written.read_bytes() == b"rows\n"
    assert registry.created_directories == tuple(root.joinpath(*depth[:n]) for n in range(1, len(depth) + 1))
    registry.check_folder(folder)  # the new folder is now the one this run allocated in
    assert not [p for p in folder.iterdir() if p.name.startswith(safety.PRIVATE_PREFIX)]


def test_a_removed_folder_is_no_longer_a_commit_target(root: Path) -> None:
    folder = root / "Normalised"
    checked, _reads = checked_report(root, folder)
    registry = RunRegistry("run1")
    failed_file(registry, checked, folder)
    with pytest.raises(FileFailure) as caught:
        registry.check_folder(folder)
    assert caught.value.reason == "unsafe_output"


def test_a_folder_replaced_by_someone_else_is_still_refused_and_left_alone(root: Path) -> None:
    folder = root / "Normalised"
    checked, reads = checked_report(root, folder)
    registry = RunRegistry("run1")
    registry.ensure_folder(folder, checked.pins["report"])
    os.rmdir(folder)
    folder.mkdir()  # a third party's folder under the same name, a new identity
    theirs = folder.stat().st_ino

    assert registry.close() == ()
    assert folder.is_dir() and folder.stat().st_ino == theirs  # never removed: it is not ours
    with pytest.raises(RunOutputError, match="was replaced"):
        publish(registry, checked, reads)
    assert list(folder.iterdir()) == []


def test_a_replacement_holding_files_is_refused_too(root: Path) -> None:
    folder = root / "Normalised"
    checked, reads = checked_report(root, folder)
    registry = RunRegistry("run1")
    registry.ensure_folder(folder, checked.pins["report"])
    os.rmdir(folder)
    folder.mkdir()
    (folder / "theirs.txt").write_bytes(b"x")

    assert registry.close() == ()
    with pytest.raises(RunOutputError, match="was replaced"):
        publish(registry, checked, reads)
    assert sorted(p.name for p in folder.iterdir()) == ["theirs.txt"]


def test_a_folder_someone_else_removed_keeps_its_identity_and_is_refused(root: Path) -> None:
    """[judgement] Only a removal this run made frees the name; a vanished folder stays suspect."""
    folder = root / "Normalised"
    checked, reads = checked_report(root, folder)
    registry = RunRegistry("run1")
    registry.ensure_folder(folder, checked.pins["report"])
    os.rmdir(folder)  # a third party removes it before the run's own cleanup

    assert registry.close() == ()
    with pytest.raises(RunOutputError, match="was replaced"):
        publish(registry, checked, reads)


def test_a_folder_still_holding_output_is_kept_and_reused(root: Path) -> None:
    folder = root / "Normalised"
    checked, reads = checked_report(root, folder)
    registry = RunRegistry("run1")
    registry.ensure_folder(folder, checked.pins["report"])
    (folder / "song.wav").write_bytes(b"RIFF published")
    before = folder.stat().st_ino

    assert registry.close() == ()
    written = publish(registry, checked, reads)
    assert written.parent == folder and folder.stat().st_ino == before
