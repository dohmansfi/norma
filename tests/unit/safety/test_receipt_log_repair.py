"""TST-S-03/S-04 — IC-2 recheck repairs R-1 and R-2 (TK-0511, D-075; safety §§ 5.2, 6).

R-1: an I/O failure while the commit receipt is written or synchronised leaves no receipt and,
at the end of the run, no run-created folder (Appendix B C-1, C-2). R-2: opening the log
rechecks the preflight pin of its folder, so a folder replaced since the barrier is refused
with nothing appended (Appendix B C-5). Faults are injected at the named boundary only;
everything is synthetic inside tmp_path and the source is checked unchanged.
"""

from __future__ import annotations

import errno
import hashlib
import os
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from norma import pipeline, safety
from norma.errors import UsageError
from norma.models import InputItem, SidecarRequest, TargetFormat
from norma.paths import NamingOptions, plan_output
from norma.safety import RunOutput, RunRegistry, commit_bundle, preflight, protect_inputs

TAGS = SidecarRequest("tags", "tags_sidecar", None, ".tags.json")


def wav_target() -> TargetFormat:
    return TargetFormat(format="wav", extension=".wav", encoder="pcm_s16le", bit_depth=16, sample_float=False,
                        sample_rate_hz=44100, channels=2, channel_layout="stereo", dither="none",
                        bitrate_mode="none", requested_bitrate_bps=None, effective_bitrate_bps=None,
                        vbr_quality=None, bitrate_source="not_applicable", capability_basis="ev:test",
                        stream_copy=False, decision="as_requested")


def fingerprint(path: Path) -> tuple[str, int, int, int]:
    info = path.lstat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), info.st_mtime_ns, info.st_ino, info.st_mode


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path.resolve() / "demos"
    folder.mkdir()
    return folder


# --- R-1: receipt I/O failure (C-1, C-2) ---------------------------------------------------------


class Bundle:
    """One source, a planned bundle in a run-created folder, and written candidates."""

    def __init__(self, root: Path, build: SimpleNamespace, *, requests=(TAGS,)) -> None:  # type: ignore[no-untyped-def]
        self.root = root
        self.source = root / "take.wav"
        self.source.write_bytes(build.wav(50))
        info = os.stat(self.source)
        self.item = InputItem(self.source, root, PurePosixPath("take.wav"), (info.st_dev, info.st_ino), 0, True)
        self.before = fingerprint(self.source)
        self.reads = protect_inputs([self.item])
        plan = plan_output(self.item, wav_target(), NamingOptions(), plan_revision=1, sidecar_requests=requests)
        checked = preflight({0: plan}, [], self.reads, overwrite=False)
        assert checked.refusals == {}
        self.registry = RunRegistry("run1")
        self.snapshot = safety.snapshot_source(self.item)
        self.plan = self.registry.allocate_plan(0, checked.plans[0], pins=checked.pins)
        for member in self.plan.members:
            assert member.temporary_path is not None
            member.temporary_path.write_bytes(build.wav(20) if member.purpose == "audio" else b"{}")
        assert self.registry.created_directories == (self.folder,)

    @property
    def folder(self) -> Path:
        return self.root / "Normalised"

    def commit(self) -> safety.CommitResult:
        return commit_bundle(self.registry, self.item, self.plan, snapshot=self.snapshot, protected=self.reads,
                             metadata_revision=self.plan.plan_revision,
                             staged_member_ids=[m.member_id for m in self.plan.members if m.member_id != "audio"])

    def leftovers(self) -> list[str]:
        return sorted(p.name for p in self.folder.iterdir()) if self.folder.exists() else []


def receipt_fault(call: str, code: int) -> Callable[..., Path]:
    """Appendix B's wrapper: ``os.<call>`` fails only while the receipt is written, after creation."""
    original = RunRegistry.write_receipt

    def write_receipt(registry: RunRegistry, *args, **kwargs) -> Path:  # type: ignore[no-untyped-def]
        with patch.object(safety.os, call, side_effect=OSError(code, os.strerror(code))):
            return original(registry, *args, **kwargs)
    return write_receipt


@pytest.mark.parametrize("call, code, reason", [
    ("fsync", errno.EIO, "commit_error"),     # C-1: the receipt is created and written, fsync fails
    ("write", errno.ENOSPC, "disk_full"),     # C-2: the receipt is created, its write fails
])
def test_receipt_io_failure_leaves_no_receipt_and_no_run_folder(root: Path, build: SimpleNamespace,
                                                                call: str, code: int, reason: str) -> None:
    bundle = Bundle(root, build)
    with patch.object(RunRegistry, "write_receipt", receipt_fault(call, code)):
        result = bundle.commit()
    assert (result.status, result.reason, result.execution) == ("failed", reason, "not_written")
    assert result.published_paths == () and result.retained_paths == ()
    assert [d.code for d in result.diagnostics] == [reason]  # no cleanup_error: nothing was left
    assert bundle.leftovers() == [] and bundle.registry.records() == ()
    assert bundle.registry.close() == ()
    assert not bundle.folder.exists()  # the empty run-created folder is removed at the end of the run
    assert fingerprint(bundle.source) == bundle.before


@pytest.mark.parametrize("call, code", [("fsync", errno.EIO), ("write", errno.ENOSPC)])
def test_write_receipt_removes_its_own_file_and_reraises(root: Path, call: str, code: int) -> None:
    registry = RunRegistry("run1")
    with patch.object(safety.os, call, side_effect=OSError(code, "injected")), \
            pytest.raises(OSError) as caught:
        registry.write_receipt(0, root, {"phase": "prepared"})
    assert caught.value.errno == code
    assert registry.records() == () and not list(root.iterdir())


def test_a_receipt_that_cannot_be_removed_stays_registered_for_cleanup(root: Path) -> None:
    registry = RunRegistry("run1")
    with patch.object(safety.os, "fsync", side_effect=OSError(errno.EIO, "injected")), \
            patch.object(RunRegistry, "remove", side_effect=OSError(errno.EBUSY, "busy")), \
            pytest.raises(OSError) as caught:
        registry.write_receipt(0, root, {"phase": "prepared"})
    assert caught.value.errno == errno.EIO  # the original failure stays primary
    (record,) = registry.records()
    assert record.kind == "receipt" and record.path.exists()
    assert registry.close() == () and not list(root.iterdir())


def test_receipt_is_registered_before_its_content_is_written(root: Path) -> None:
    registry = RunRegistry("run1")
    seen: list[tuple] = []
    real_write = os.write

    def write(fd: int, data) -> int:  # type: ignore[no-untyped-def]
        seen.append(registry.records())
        return real_write(fd, data)

    with patch.object(safety.os, "write", side_effect=write):
        path = registry.write_receipt(0, root, {"phase": "prepared"})
    assert seen and all(len(records) == 1 and records[0].path == path for records in seen)


def test_short_writes_still_store_the_whole_receipt(root: Path) -> None:
    registry = RunRegistry("run1")
    real_write = os.write

    def one_byte(fd: int, data) -> int:  # type: ignore[no-untyped-def]
        return real_write(fd, bytes(data[:1]))

    content = {"phase": "prepared", "members": ["a" * 40]}
    with patch.object(safety.os, "write", side_effect=one_byte):
        path = registry.write_receipt(0, root, content)
        registry.update_receipt(path, {**content, "phase": "publishing"})
    assert path.read_text(encoding="utf-8") == ('{"members": ["' + "a" * 40 + '"], "phase": "publishing"}')
    assert registry.close() == () and not path.exists()


def test_a_write_that_accepts_no_bytes_fails_instead_of_looping(root: Path) -> None:
    registry = RunRegistry("run1")
    with patch.object(safety.os, "write", return_value=0), pytest.raises(OSError) as caught:
        registry.write_receipt(0, root, {"phase": "prepared"})
    assert caught.value.errno == errno.EIO
    assert registry.records() == () and not list(root.iterdir())


# --- R-2: the log folder's preflight pin (C-5) ----------------------------------------------------


def checked_log(root: Path, name: str = "logs/run.log") -> tuple[safety.Preflight, tuple]:
    source = root / "take.wav"
    source.write_bytes(b"RIFF")
    info = source.stat()
    reads = protect_inputs([InputItem(source, root, PurePosixPath("take.wav"), (info.st_dev, info.st_ino), 0, True)])
    return preflight({}, [RunOutput("log", root / name)], reads, overwrite=False), reads


def swap_folder(root: Path, *, seed: bytes | None = b"substitute\n") -> None:
    """C-5: rename the checked folder away and put a new regular folder at the same path."""
    (root / "logs").rename(root / "logs-old")
    (root / "logs").mkdir()
    if seed is not None:
        (root / "logs" / "run.log").write_bytes(seed)


def test_preflight_records_the_log_folder_pin_on_its_occupant(root: Path) -> None:
    (root / "logs").mkdir()
    checked, _reads = checked_log(root)
    assert checked.occupants["log"].parent == checked.pins["log"]
    assert checked.pins["log"].exists and checked.pins["log"].parent == root / "logs"


@pytest.mark.parametrize("existing", [False, True])
def test_log_in_a_replaced_folder_is_refused_with_nothing_appended(root: Path, existing: bool) -> None:
    (root / "logs").mkdir()
    if existing:
        (root / "logs" / "run.log").write_bytes(b"history\n")
    checked, reads = checked_log(root)
    swap_folder(root)
    with pytest.raises(UsageError, match="folder .* changed after it was checked; nothing was written"):
        safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    assert (root / "logs" / "run.log").read_bytes() == b"substitute\n"
    old = root / "logs-old" / "run.log"
    assert old.read_bytes() == b"history\n" if existing else not old.exists()
    assert (root / "take.wav").read_bytes() == b"RIFF"


def test_new_log_in_a_replaced_empty_folder_is_not_created(root: Path) -> None:
    (root / "logs").mkdir()
    checked, reads = checked_log(root)
    swap_folder(root, seed=None)
    with pytest.raises(UsageError):
        safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    assert list((root / "logs").iterdir()) == []  # refused before the open: nothing created


def test_log_folder_replaced_by_a_link_to_itself_is_refused(root: Path) -> None:
    (root / "real").mkdir()
    (root / "logs").mkdir()
    checked, reads = checked_log(root)
    (root / "logs").rmdir()
    (root / "logs").symlink_to(root / "real")
    with pytest.raises(UsageError):
        safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    assert list((root / "real").iterdir()) == []


def test_removed_log_folder_is_refused(root: Path) -> None:
    (root / "logs").mkdir()
    checked, reads = checked_log(root)
    (root / "logs").rmdir()
    with pytest.raises(UsageError):
        safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    assert not (root / "logs").exists()


def test_folder_swapped_during_the_open_is_refused_after_it(root: Path) -> None:
    """The second recheck: the folder is replaced between the first check and the open."""
    (root / "logs").mkdir()
    (root / "logs" / "run.log").write_bytes(b"history\n")
    checked, reads = checked_log(root)
    real_open = os.open

    def racing_open(path, flags, mode=0o777, **kwargs):  # type: ignore[no-untyped-def]
        if Path(os.fsdecode(path)) == root / "logs" / "run.log":
            fd = real_open(path, flags, mode, **kwargs)
            swap_folder(root)
            return fd
        return real_open(path, flags, mode, **kwargs)

    with patch.object(safety.os, "open", side_effect=racing_open), pytest.raises(UsageError):
        safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    assert (root / "logs-old" / "run.log").read_bytes() == b"history\n"
    assert (root / "logs" / "run.log").read_bytes() == b"substitute\n"


def test_explicit_pin_takes_precedence_over_the_recorded_one(root: Path) -> None:
    (root / "logs").mkdir()
    checked, reads = checked_log(root)
    destination = checked.write_set.destinations[0]
    stale = safety.ParentPin(root / "logs", root / "logs", (0, 0))
    with pytest.raises(UsageError):
        safety.open_log(destination, checked.occupants["log"], reads, pin=stale)
    fd = safety.open_log(destination, checked.occupants["log"], reads, pin=checked.pins["log"])
    os.close(fd)


def test_unchanged_log_folder_still_opens_and_appends(root: Path) -> None:
    (root / "logs").mkdir()
    (root / "logs" / "run.log").write_bytes(b"history\n")
    checked, reads = checked_log(root)
    fd = safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    try:
        os.write(fd, b"line\n")
    finally:
        os.close(fd)
    assert (root / "logs" / "run.log").read_bytes() == b"history\nline\n"


def test_the_pipeline_caller_carries_the_pin_to_open_log(root: Path) -> None:
    """``pipeline.RunOutputs.open_log`` passes preflight's occupant unchanged, so C-5 is refused."""
    (root / "logs").mkdir()
    checked, reads = checked_log(root)
    outputs = pipeline.RunOutputs(checked, reads, RunRegistry("run1"), overwrite=False,
                                  requests=(pipeline.log_request(root / "logs" / "run.log"),))
    swap_folder(root)
    with pytest.raises(UsageError):
        outputs.open_log()
    assert (root / "logs" / "run.log").read_bytes() == b"substitute\n"
