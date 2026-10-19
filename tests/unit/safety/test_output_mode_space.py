"""TST-S-03/S-05 — final output mode, source mtime and the space reservation (safety § 5.1; CR-046).

Everything is synthetic inside tmp_path. Free space is injected through ``RunRegistry``'s
``free_space`` seam, so no test depends on the real volume's fill level; the real ``statvfs``
reader is checked separately. Every commit test checks that the source is unchanged.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import threading
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from norma import safety
from norma.errors import FileFailure
from norma.models import InputItem, OutputPlan, SidecarRequest, TargetFormat
from norma.paths import NamingOptions, plan_output
from norma.safety import RunOutput, RunRegistry, SpaceEstimate, commit_bundle, estimate_space, preflight

TAGS = SidecarRequest("tags", "tags_sidecar", None, ".tags.json")
SOURCE_MTIME_NS = 1_600_000_000_123_456_789  # odd nanoseconds: a truncating volume would show
OLD_OUTPUT_MTIME_NS = 1_500_000_000_987_654_321
PLENTY = 1 << 50


def target(fmt: str = "wav", *, bit_depth: int | None = 16, rate: int = 44100, channels: int = 2,
           extension: str = ".wav") -> TargetFormat:
    return TargetFormat(format=fmt, extension=extension, encoder="pcm_s16le", bit_depth=bit_depth,  # type: ignore[arg-type]
                        sample_float=False, sample_rate_hz=rate, channels=channels, channel_layout="stereo",
                        dither="none", bitrate_mode="none", requested_bitrate_bps=None, effective_bitrate_bps=None,
                        vbr_quality=None, bitrate_source="not_applicable", capability_basis="ev:test",
                        stream_copy=False, decision="as_requested")


def fingerprint(path: Path) -> tuple[str, int, int, int, int]:
    info = path.lstat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), info.st_mtime_ns, info.st_ino, info.st_mode, info.st_size


@pytest.fixture
def umask() -> Iterator[Callable[[int], None]]:
    """Set the process umask for one test and always restore it."""
    original = os.umask(0o022)
    os.umask(original)

    def set_mask(value: int) -> None:
        os.umask(value)

    try:
        yield set_mask
    finally:
        os.umask(original)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path.resolve() / "demos"
    folder.mkdir()
    return folder


class Rig:
    """One source with a distinctive mtime, its planned bundle and a registry."""

    def __init__(self, root: Path, build: SimpleNamespace, *, overwrite: bool = False, requests=(TAGS,),
                 free: Callable[[Path], int] | None = None,
                 fault: Callable[[str, str | None], None] | None = None) -> None:
        self.root = root
        self.build = build
        self.source = root / "take.wav"
        self.source.write_bytes(build.wav(50))
        os.utime(self.source, ns=(SOURCE_MTIME_NS, SOURCE_MTIME_NS))
        info = os.stat(self.source)
        self.item = InputItem(self.source, root, PurePosixPath("take.wav"), (info.st_dev, info.st_ino), 0, True)
        self.reads = safety.protect_inputs([self.item])
        self.plan = plan_output(self.item, target(), NamingOptions(), plan_revision=0, sidecar_requests=requests,
                                overwrite=overwrite)
        self.overwrite = overwrite
        self.registry = RunRegistry("run1", fault_hook=fault, free_space=free or (lambda _path: PLENTY))
        self.snapshot = safety.snapshot_source(self.item)
        self.before = fingerprint(self.source)

    @property
    def folder(self) -> Path:
        return self.root / "Normalised"

    def checked_plan(self) -> OutputPlan:
        checked = preflight({0: self.plan}, [], self.reads, overwrite=self.overwrite)
        assert checked.refusals == {}, checked.refusals
        self.pins = checked.pins
        return checked.plans[0]

    def prepare(self, *, space: SpaceEstimate | None = None) -> OutputPlan:
        self.allocated = self.registry.allocate_plan(0, self.checked_plan(), pins=self.pins, space=space)
        for member in self.allocated.members:
            assert member.temporary_path is not None
            payload = self.build.wav(20) if member.purpose == "audio" else f"new {member.member_id}".encode()
            member.temporary_path.write_bytes(payload)
        return self.allocated

    def commit(self) -> safety.CommitResult:
        plan = self.allocated
        return commit_bundle(self.registry, self.item, plan, snapshot=self.snapshot, protected=self.reads,
                             metadata_revision=plan.plan_revision,
                             staged_member_ids=[m.member_id for m in plan.members if m.member_id != "audio"])

    def listing(self) -> list[str]:
        return sorted(p.name for p in self.folder.iterdir()) if self.folder.exists() else []

    def source_unchanged(self) -> bool:
        return fingerprint(self.source) == self.before


@pytest.fixture
def rig(root: Path, build: SimpleNamespace) -> Callable[..., Rig]:
    def make(**kwargs) -> Rig:  # type: ignore[no-untyped-def]
        return Rig(root, build, **kwargs)
    return make


def mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


# --- The umask and the final mode ------------------------------------------------------------------


def test_process_umask_reads_without_changing_it(umask) -> None:  # type: ignore[no-untyped-def]
    umask(0o027)
    assert safety.process_umask() == 0o027
    assert safety.process_umask() == 0o027  # still set: the read restored it
    assert safety.output_mode_for(0o022) == 0o644
    assert safety.output_mode_for(0o077) == 0o600
    assert safety.output_mode_for(0o002) == 0o664
    assert safety.output_mode_for(0o7777) == 0


@pytest.mark.parametrize(("mask", "expected"), [(0o022, 0o644), (0o077, 0o600), (0o002, 0o664)])
def test_published_outputs_take_the_umask_mode_and_candidates_stay_private(  # type: ignore[no-untyped-def]
        rig, umask, mask: int, expected: int) -> None:
    umask(mask)
    r = rig()
    assert r.registry.output_mode == expected
    plan = r.prepare()
    for member in plan.members:
        assert member.temporary_path is not None and mode(member.temporary_path) == 0o600
    result = r.commit()
    assert result.execution == "committed", result.diagnostics
    assert r.listing() == ["take.wav", "take.wav.tags.json"]
    assert mode(r.folder / "take.wav") == expected
    assert mode(r.folder / "take.wav.tags.json") == expected
    assert r.source_unchanged()


def test_the_mode_is_captured_when_the_registry_is_made(rig, umask) -> None:  # type: ignore[no-untyped-def]
    umask(0o022)
    r = rig()
    r.prepare()
    umask(0o077)  # a later change does not reach this run's outputs
    assert r.commit().execution == "committed"
    assert mode(r.folder / "take.wav") == 0o644


def test_explicit_umask_argument_overrides_the_process_mask(umask) -> None:  # type: ignore[no-untyped-def]
    umask(0o077)
    assert RunRegistry("run1", umask=0o022).output_mode == 0o644


# --- The source mtime on the audio candidate --------------------------------------------------------


def test_audio_output_carries_the_source_mtime_to_the_nanosecond(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    r.prepare()
    assert r.commit().execution == "committed"
    assert (r.folder / "take.wav").stat().st_mtime_ns == SOURCE_MTIME_NS  # APFS stores nanoseconds
    assert (r.folder / "take.wav.tags.json").stat().st_mtime_ns != SOURCE_MTIME_NS  # sidecars are new files
    assert r.source_unchanged()  # content, mtime, inode, mode and size


def test_hard_link_route_keeps_mode_and_mtime(rig, umask, monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
                                              allow_linked_names: object) -> None:
    monkeypatch.setattr(safety, "_RENAMEX", None)
    umask(0o022)
    r = rig()
    r.prepare()
    assert r.commit().execution == "committed"
    final = r.folder / "take.wav"
    assert (mode(final), final.stat().st_mtime_ns, final.stat().st_nlink) == (0o644, SOURCE_MTIME_NS, 1)
    assert r.source_unchanged()


def test_overwrite_gives_the_replacement_the_final_mode_and_source_mtime(  # type: ignore[no-untyped-def]
        rig, umask, allow_linked_names: object) -> None:
    umask(0o022)
    r = rig(overwrite=True)
    r.folder.mkdir()
    (r.folder / "take.wav").write_bytes(b"old audio")
    (r.folder / "take.wav.tags.json").write_bytes(b"old tags")
    r.prepare()
    assert r.commit().execution == "committed"
    final = r.folder / "take.wav"
    assert final.read_bytes() == r.build.wav(20)
    assert (mode(final), final.stat().st_mtime_ns) == (0o644, SOURCE_MTIME_NS)
    assert r.source_unchanged()


def test_a_coarser_volume_granularity_is_accepted(rig, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    real_utime = os.utime

    def seconds_only(path, *args, ns=None, **kwargs):  # type: ignore[no-untyped-def]
        if ns is not None:
            ns = tuple(value - value % 1_000_000_000 for value in ns)
        return real_utime(path, *args, ns=ns, **kwargs)

    r = rig()
    r.prepare()
    monkeypatch.setattr(safety.os, "utime", seconds_only)
    assert r.commit().execution == "committed"
    assert (r.folder / "take.wav").stat().st_mtime_ns == SOURCE_MTIME_NS - SOURCE_MTIME_NS % 1_000_000_000


def test_times_match_accepts_only_truncation() -> None:
    wanted = SOURCE_MTIME_NS
    assert safety._times_match(wanted, wanted)
    assert safety._times_match(wanted - wanted % 1_000, wanted)
    assert safety._times_match(wanted - wanted % 1_000_000_000, wanted)
    assert not safety._times_match(wanted + 1, wanted)
    assert not safety._times_match(wanted - wanted % 1_000_000_000 + 1_000_000_000, wanted)  # rounded up
    assert not safety._times_match(wanted - 1, wanted)


# --- Failure to apply either value is a specific failure, never success ------------------------------


def _fails_on_candidates(code: int) -> Callable[..., None]:
    def fail(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if Path(path).name.startswith(".norma-run1-0-cand-"):
            raise OSError(code, os.strerror(code), str(path))
        raise AssertionError(f"unexpected call on {path}")
    return fail


@pytest.mark.parametrize(("call", "code", "reason"), [
    ("chmod", errno.EPERM, "permission_denied"),
    ("chmod", errno.EIO, "commit_error"),
    ("utime", errno.EACCES, "permission_denied"),
    ("utime", errno.ENOSPC, "disk_full"),
    ("utime", errno.EIO, "commit_error"),
])
def test_failure_to_set_mode_or_mtime_publishes_nothing(  # type: ignore[no-untyped-def]
        rig, monkeypatch: pytest.MonkeyPatch, call: str, code: int, reason: str) -> None:
    r = rig()
    r.prepare()
    monkeypatch.setattr(safety.os, call, _fails_on_candidates(code))
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", reason, "not_written")
    assert result.published_paths == ()
    assert r.listing() == []  # no final, no candidate, no receipt
    assert r.registry.records(0) == ()
    assert r.source_unchanged()


def test_an_mtime_the_volume_did_not_store_is_a_commit_error(rig, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    r.prepare()
    monkeypatch.setattr(safety.os, "utime", lambda *args, **kwargs: None)  # silently ignored
    result = r.commit()
    assert (result.reason, result.execution) == ("commit_error", "not_written")
    assert "modification time" in (result.diagnostics[0].detail or result.diagnostics[0].message)
    assert r.listing() == []
    assert r.source_unchanged()


def test_a_mode_the_volume_did_not_store_is_a_commit_error(rig, monkeypatch: pytest.MonkeyPatch, umask) -> None:  # type: ignore[no-untyped-def]
    umask(0o022)
    r = rig()
    r.prepare()
    monkeypatch.setattr(safety.os, "chmod", lambda *args, **kwargs: None)
    result = r.commit()
    assert (result.reason, result.execution) == ("commit_error", "not_written")
    assert r.listing() == []
    assert r.source_unchanged()


def test_the_source_is_never_given_the_new_mode_or_times(rig, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    touched: list[Path] = []
    real_chmod, real_utime = os.chmod, os.utime

    def chmod(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        touched.append(Path(path))
        return real_chmod(path, *args, **kwargs)

    def utime(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        touched.append(Path(path))
        return real_utime(path, *args, **kwargs)

    r = rig()
    r.prepare()
    monkeypatch.setattr(safety.os, "chmod", chmod)
    monkeypatch.setattr(safety.os, "utime", utime)
    assert r.commit().execution == "committed"
    assert touched and all(p.name.startswith(".norma-run1-0-cand-") for p in touched)
    assert r.source not in touched and r.source_unchanged()


# --- Run outputs and preimage copies ----------------------------------------------------------------


def test_a_published_report_takes_the_umask_mode(root: Path, umask) -> None:  # type: ignore[no-untyped-def]
    umask(0o022)
    source = root / "take.wav"
    source.write_bytes(b"RIFF")
    info = source.stat()
    item = InputItem(source, root, PurePosixPath("take.wav"), (info.st_dev, info.st_ino), 0, True)
    reads = safety.protect_inputs([item])
    checked = preflight({}, [RunOutput("report", root / "r.csv")], reads, overwrite=False)
    path = safety.publish_run_output(RunRegistry("run1"), checked.write_set.destinations[0], b"rows",
                                     overwrite=False, protected=reads)
    assert path.read_bytes() == b"rows" and mode(path) == 0o644
    assert sorted(p.name for p in root.iterdir()) == ["r.csv", "take.wav"]


def test_a_restored_copy_preimage_gives_back_the_old_mode_and_times(  # type: ignore[no-untyped-def]
        rig, monkeypatch: pytest.MonkeyPatch) -> None:
    real_link = os.link

    def refuse(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        if Path(dst).name.startswith(".norma-run1-0-pre-"):
            raise OSError(errno.ENOTSUP, "Operation not supported")
        return real_link(src, dst, *args, **kwargs)

    def fail(phase: str, member: str | None) -> None:
        if (phase, member) == ("published", "audio"):
            raise OSError(errno.EIO, "read-back failed")

    monkeypatch.setattr(safety.os, "link", refuse)
    r = rig(overwrite=True, fault=fail)
    r.folder.mkdir()
    old = r.folder / "take.wav"
    old.write_bytes(b"old audio")
    old.chmod(0o640)
    os.utime(old, ns=(OLD_OUTPUT_MTIME_NS, OLD_OUTPUT_MTIME_NS))
    (r.folder / "take.wav.tags.json").write_bytes(b"old tags")
    r.prepare()
    result = r.commit()
    assert result.execution == "not_written"
    assert old.read_bytes() == b"old audio"
    assert (mode(old), old.stat().st_mtime_ns) == (0o640, OLD_OUTPUT_MTIME_NS)
    assert r.listing() == ["take.wav", "take.wav.tags.json"]
    assert r.source_unchanged()


# --- The space estimate -----------------------------------------------------------------------------


def test_estimate_follows_the_section_5_1_formula() -> None:
    # 10 s of 16-bit stereo 44.1 kHz: 441,000 frames × 2 × 2 = 1,764,000 B, plus the allowance.
    estimate = estimate_space(1_000, 10.0, target())
    assert estimate.audio_bytes == 1_764_000 + 1_000_000
    assert estimate.required_bytes == 3_316_800 + 50_000_000  # ceil(1.2 × 2,764,000) + margin
    # A larger source wins over the PCM bound.
    assert estimate_space(9_000_000, 10.0, target()).audio_bytes == 10_000_000
    # Compressed output counts at least four bytes per sample.
    assert estimate_space(0, 10.0, target("flac", extension=".flac")).audio_bytes == 441_000 * 2 * 4 + 1_000_000
    # 24-bit PCM stores three bytes; sidecars and preimage copies join the 1.2 factor.
    sized = estimate_space(0, 1.0, target(bit_depth=24, rate=48000), sidecar_bytes=10, preimage_bytes=20)
    assert sized.audio_bytes == 48_000 * 2 * 3 + 1_000_000
    assert sized.required_bytes == -(-(sized.audio_bytes + 30) * 12 // 10) + 50_000_000


def test_ceil_is_exact_for_fractional_frames() -> None:
    assert estimate_space(0, 0.00001, target(rate=44100, channels=1)).audio_bytes == 1 * 2 + 1_000_000
    assert SpaceEstimate(1).required_bytes == 2 + 50_000_000  # ceil(1.2)


@pytest.mark.parametrize("duration", [None, float("nan"), -1.0])
def test_unknown_duration_uses_the_unknown_duration_bound(duration: float | None) -> None:
    assert estimate_space(0, duration, target()).audio_bytes == 3_600 * 44_100 * 2 * 2 + 1_000_000


@pytest.mark.parametrize(("rate", "channels"), [(0, 2), (44100, 0)])
def test_unknown_rate_or_channels_is_probe_error_not_zero_space(rate: int, channels: int) -> None:
    shape = SimpleNamespace(sample_rate_hz=rate, channels=channels, bit_depth=16, format="wav")
    with pytest.raises(FileFailure) as caught:
        estimate_space(0, 10.0, shape)  # type: ignore[arg-type]
    assert caught.value.reason == "probe_error"


def test_estimate_rejects_negative_parts() -> None:
    with pytest.raises(ValueError):
        SpaceEstimate(-1)


def test_free_space_reader_uses_statvfs(tmp_path: Path) -> None:
    info = os.statvfs(tmp_path)
    assert safety.free_space_bytes(tmp_path) == info.f_bavail * info.f_frsize > 0


# --- Reservation before a render ----------------------------------------------------------------------


def test_shortfall_refuses_with_disk_full_before_any_candidate(rig) -> None:  # type: ignore[no-untyped-def]
    estimate = estimate_space(0, 10.0, target())
    r = rig(free=lambda _path: estimate.required_bytes - 1)
    with pytest.raises(FileFailure) as caught:
        r.prepare(space=estimate)
    assert (caught.value.reason, caught.value.stage) == ("disk_full", safety.ALLOCATE_STAGE)
    assert not r.folder.exists()  # not even the output folder was created
    assert r.registry.records() == () and r.registry.reserved_bytes() == 0
    settled = safety.settle(r.registry, 0, caught.value)
    assert (settled.status, settled.reason, settled.execution) == ("failed", "disk_full", "not_written")
    assert r.source_unchanged()


def test_exactly_enough_space_allocates_and_reserves(rig) -> None:  # type: ignore[no-untyped-def]
    estimate = estimate_space(0, 10.0, target())
    seen: list[Path] = []

    def free(path: Path) -> int:
        seen.append(path)
        return estimate.required_bytes

    r = rig(free=free)
    r.prepare(space=estimate)
    assert seen == [r.root]  # the nearest existing ancestor of the not-yet-created folder
    assert r.registry.reserved_bytes(0) == estimate.required_bytes
    assert r.commit().execution == "committed"
    assert r.registry.reserved_bytes() == 0  # released once the bundle is published
    assert r.source_unchanged()


def test_reservation_is_released_when_a_file_is_settled(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    r.prepare(space=SpaceEstimate(1_000))
    assert r.registry.reserved_bytes(0) > 0
    safety.settle(r.registry, 0, FileFailure("render_error", "render", "synthetic"))
    assert r.registry.reserved_bytes() == 0 and r.listing() == []


def test_other_files_reservations_count_against_the_volume(root: Path) -> None:
    need = SpaceEstimate(1_000_000)
    registry = RunRegistry("run1", free_space=lambda _path: need.required_bytes * 3 // 2)
    registry.reserve_space(0, root, need)
    with pytest.raises(FileFailure) as caught:
        registry.reserve_space(1, root, need)
    assert caught.value.reason == "disk_full"
    registry.reserve_space(0, root, need)  # a retry of the same file replaces, never doubles, its hold
    assert registry.reserved_bytes() == need.required_bytes
    registry.cleanup(0)
    registry.reserve_space(1, root, need)  # input 0's bytes are no longer held
    assert registry.reserved_bytes() == need.required_bytes
    registry.close()
    assert registry.reserved_bytes() == 0


def test_reservations_are_checked_under_one_lock(root: Path) -> None:
    need = SpaceEstimate(1_000_000)
    registry = RunRegistry("run1", free_space=lambda _path: need.required_bytes * 3)
    start = threading.Barrier(8)
    outcomes: list[str] = []
    guard = threading.Lock()

    def worker(index: int) -> None:
        start.wait()
        try:
            registry.reserve_space(index, root, need)
            result = "ok"
        except FileFailure as exc:
            result = exc.reason
        with guard:
            outcomes.append(result)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["disk_full"] * 5 + ["ok"] * 3
    assert registry.reserved_bytes() == need.required_bytes * 3


def test_unreadable_free_space_is_a_specific_failure(root: Path) -> None:
    def denied(_path: Path) -> int:
        raise PermissionError(errno.EACCES, "denied")

    registry = RunRegistry("run1", free_space=denied)
    with pytest.raises(FileFailure) as caught:
        registry.reserve_space(0, root / "Normalised", SpaceEstimate(1))
    assert caught.value.reason == "permission_denied"


def test_a_copy_preimage_that_would_not_fit_is_disk_full_with_nothing_published(  # type: ignore[no-untyped-def]
        rig, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError(errno.ENOTSUP, "Operation not supported")

    estimate = SpaceEstimate(1_000)
    budget = {"free": estimate.required_bytes}
    r = rig(overwrite=True, free=lambda _path: budget["free"])
    r.folder.mkdir()
    (r.folder / "take.wav").write_bytes(b"old audio")
    (r.folder / "take.wav.tags.json").write_bytes(b"old tags")
    r.prepare(space=estimate)
    budget["free"] = 1_000  # the volume filled up while the file rendered
    monkeypatch.setattr(safety.os, "link", refuse)
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", "disk_full", "not_written")
    assert (r.folder / "take.wav").read_bytes() == b"old audio"
    assert r.listing() == ["take.wav", "take.wav.tags.json"]
    assert r.registry.reserved_bytes() == 0
    assert r.source_unchanged()


# --- CR-048: an explicit report's parent is rechecked, never re-created --------------------------------


def _report_in(root: Path, sub: str, *, parent_must_exist: bool = True):  # type: ignore[no-untyped-def]
    source = root / "take.wav"
    source.write_bytes(b"RIFF")
    info = source.stat()
    item = InputItem(source, root, PurePosixPath("take.wav"), (info.st_dev, info.st_ino), 0, True)
    reads = safety.protect_inputs([item])
    output = RunOutput("report", root / sub / "r.csv", parent_must_exist=parent_must_exist)
    checked = preflight({}, [output], reads, overwrite=False)
    return checked, reads


@pytest.mark.parametrize("with_pin", [True, False])
def test_a_report_parent_removed_after_preflight_is_not_re_created(root: Path, with_pin: bool) -> None:
    (root / "reports").mkdir()
    checked, reads = _report_in(root, "reports")
    (root / "reports").rmdir()
    registry = RunRegistry("run1")
    with pytest.raises(safety.RunOutputError):
        safety.publish_run_output(registry, checked.write_set.destinations[0], b"rows", overwrite=False,
                                  protected=reads, pin=checked.pins["report"] if with_pin else None)
    assert not (root / "reports").exists()
    assert registry.created_directories == () and registry.records() == ()


def test_a_report_parent_replaced_after_preflight_is_refused_by_its_pin(root: Path) -> None:
    (root / "reports").mkdir()
    checked, reads = _report_in(root, "reports")
    (root / "reports").rename(root / "old-reports")
    (root / "reports").mkdir()  # same name, different folder
    with pytest.raises(safety.RunOutputError):
        safety.publish_run_output(RunRegistry("run1"), checked.write_set.destinations[0], b"rows",
                                  overwrite=False, protected=reads, pin=checked.pins["report"])
    assert list((root / "reports").iterdir()) == [] and list((root / "old-reports").iterdir()) == []


def test_a_pinned_report_parent_still_present_publishes(root: Path) -> None:
    (root / "reports").mkdir()
    checked, reads = _report_in(root, "reports")
    path = safety.publish_run_output(RunRegistry("run1"), checked.write_set.destinations[0], b"rows",
                                     overwrite=False, protected=reads, pin=checked.pins["report"])
    assert path == root / "reports" / "r.csv" and path.read_bytes() == b"rows"


def test_the_automatic_report_may_still_create_its_parent_under_its_pin(root: Path) -> None:
    checked, reads = _report_in(root, "Normalised", parent_must_exist=False)
    registry = RunRegistry("run1")
    path = safety.publish_run_output(registry, checked.write_set.destinations[0], b"rows", overwrite=False,
                                     protected=reads, create_parent=True, pin=checked.pins["report"])
    assert path.read_bytes() == b"rows" and registry.created_directories == (root / "Normalised",)
