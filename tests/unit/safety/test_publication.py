"""TST-S-02/S-03/S-04/S-05 — candidates, verified publication, rollback and cleanup (safety §§ 5–8).

All racers, preimages and faults are synthetic inside tmp_path and injected at the registry's
controlled hooks, never by timing. Every test checks that the source is unchanged.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from norma import safety
from norma.errors import FileFailure, SkipFile, UsageError
from norma.models import InputItem, OutputPlan, SidecarRequest, TargetFormat
from norma.paths import NamingOptions, comparison_key, plan_output
from norma.safety import RunOutput, RunRegistry, commit_bundle, preflight, protect_inputs

TAGS = SidecarRequest("tags", "tags_sidecar", None, ".tags.json")
COVER = SidecarRequest("cover", "cover_export", "picture:new", ".cover.jpg")


def wav_target() -> TargetFormat:
    return TargetFormat(format="wav", extension=".wav", encoder="pcm_s16le", bit_depth=16, sample_float=False,
                        sample_rate_hz=44100, channels=2, channel_layout="stereo", dither="none",
                        bitrate_mode="none", requested_bitrate_bps=None, effective_bitrate_bps=None,
                        vbr_quality=None, bitrate_source="not_applicable", capability_basis="ev:test",
                        stream_copy=False, decision="as_requested")


def fingerprint(path: Path) -> tuple[str, int, int, int]:
    info = path.lstat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), info.st_mtime_ns, info.st_ino, info.st_mode


class Rig:
    """One source, its planned bundle, a registry and an allocated, written candidate set."""

    def __init__(self, root: Path, build: SimpleNamespace, *, requests=(TAGS,), overwrite: bool = False,
                 fault: Callable[[str, str | None], None] | None = None) -> None:
        self.root = root
        self.build = build
        self.source = root / "take.wav"
        self.source.write_bytes(build.wav(50))
        info = os.stat(self.source)
        self.item = InputItem(self.source, root, PurePosixPath("take.wav"), (info.st_dev, info.st_ino), 0, True)
        self.before = fingerprint(self.source)
        self.reads = protect_inputs([self.item])
        self.plan = plan_output(self.item, wav_target(), NamingOptions(), plan_revision=3,
                                sidecar_requests=requests, overwrite=overwrite)
        self.overwrite = overwrite
        self.registry = RunRegistry("run1", fault_hook=fault)
        self.snapshot = safety.snapshot_source(self.item)

    @property
    def folder(self) -> Path:
        return self.root / "Normalised"

    def prepare(self) -> OutputPlan:
        checked = preflight({0: self.plan}, [], self.reads, overwrite=self.overwrite)
        assert checked.refusals == {}, checked.refusals
        self.allocated = self.registry.allocate_plan(0, checked.plans[0], pins=checked.pins)
        for member in self.allocated.members:
            assert member.temporary_path is not None
            payload = self.build.wav(20) if member.purpose == "audio" else f"new {member.member_id}".encode()
            member.temporary_path.write_bytes(payload)
        return self.allocated

    def commit(self, plan: OutputPlan | None = None, **changes) -> safety.CommitResult:  # type: ignore[no-untyped-def]
        plan = plan or self.allocated
        arguments = dict(snapshot=self.snapshot, protected=self.reads, metadata_revision=plan.plan_revision,
                         staged_member_ids=[m.member_id for m in plan.members if m.member_id != "audio"])
        arguments.update(changes)
        return commit_bundle(self.registry, self.item, plan, **arguments)

    def listing(self) -> list[str]:
        return sorted(p.name for p in self.folder.iterdir()) if self.folder.exists() else []

    def source_unchanged(self) -> bool:
        return fingerprint(self.source) == self.before


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path.resolve() / "demos"
    folder.mkdir()
    return folder


@pytest.fixture
def rig(root: Path, build: SimpleNamespace) -> Callable[..., Rig]:
    def make(**kwargs) -> Rig:  # type: ignore[no-untyped-def]
        return Rig(root, build, **kwargs)
    return make


# --- Allocation (§ 5.1) -------------------------------------------------------------------------


def test_allocation_is_private_registered_and_beside_the_final_name(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig(requests=(TAGS, COVER))
    plan = r.prepare()
    assert r.registry.created_directories == (r.folder,)
    for member in plan.members:
        temp = member.temporary_path
        assert temp is not None and temp.parent == member.final_path.parent
        assert temp.name.startswith(".norma-run1-0-cand-") and temp.suffix == member.final_path.suffix
        assert stat.S_IMODE(temp.stat().st_mode) == 0o600
    records = r.registry.records(0)
    assert sorted(rec.member_id for rec in records) == ["audio", "cover", "tags"]
    assert {rec.kind for rec in records} == {"candidate"}
    assert r.source_unchanged()


def test_close_removes_only_registered_entries_and_empty_run_folders(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    r.prepare()
    foreign = r.folder / ".norma-otherrun-0-cand-abc.wav"  # stale residue of another run
    foreign.write_bytes(b"not ours")
    assert r.registry.close() == ()
    assert r.listing() == [foreign.name]  # never deleted by glob; the folder is kept as it is not empty
    foreign.unlink()
    r.folder.rmdir()
    r2 = rig()
    r2.prepare()
    assert r2.registry.close() == ()
    assert not r2.folder.exists()
    assert r2.source_unchanged()


def test_allocation_refuses_a_folder_replaced_since_preflight(rig, root: Path) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    checked = preflight({0: r.plan}, [], r.reads, overwrite=False)
    (root / "Normalised").mkdir()
    (root / "Normalised").rmdir()
    os.rename(root, root.with_name("moved"))
    root.mkdir()
    with pytest.raises(FileFailure) as caught:
        r.registry.allocate_plan(0, checked.plans[0], pins=checked.pins)
    assert caught.value.reason == "unsafe_output"


def test_allocation_maps_disk_full(rig, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    def full(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(safety.tempfile, "mkstemp", full)
    with pytest.raises(FileFailure) as caught:
        r.registry.allocate(0, r.plan.audio_member)
    assert caught.value.reason == "disk_full"


# --- Structural checks before publication (§ 5.2 gate, § 8) -------------------------------------


def test_structure_accepts_a_complete_candidate(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    plan = r.prepare()
    assert r.registry.check_structure(0, plan.audio_member) == "wav"
    assert r.registry.check_structure(0, next(m for m in plan.members if m.member_id == "tags")) is None


@pytest.mark.parametrize("payload", ["empty", "truncated", "wrong_kind", "not_audio"])
def test_structure_refuses_bad_candidates_as_render_error(rig, build: SimpleNamespace, payload: str) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    plan = r.prepare()
    temp = plan.audio_member.temporary_path
    assert temp is not None
    data = {"empty": b"", "truncated": build.wav(100)[:-50], "wrong_kind": build.flac, "not_audio": b"hello"}[payload]
    with open(temp, "r+b") as handle:  # same inode, as the encoder writes it
        handle.truncate(0)
        handle.write(data)
    with pytest.raises(FileFailure) as caught:
        r.registry.check_structure(0, plan.audio_member)
    assert caught.value.reason == "render_error"


def test_substituted_candidate_is_unsafe_output(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    plan = r.prepare()
    temp = plan.audio_member.temporary_path
    assert temp is not None
    temp.unlink()
    temp.write_bytes(b"impostor")
    with pytest.raises(FileFailure) as caught:
        r.registry.check_candidate(plan.audio_member)
    assert caught.value.reason == "unsafe_output"


# --- Source stability (§ 5.1) ----------------------------------------------------------------


def test_same_size_change_with_restored_mtime_is_source_changed(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    info = r.source.stat()
    data = bytearray(r.source.read_bytes())
    data[-1] ^= 0xFF
    with open(r.source, "r+b") as handle:
        handle.write(bytes(data))
    os.utime(r.source, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(FileFailure) as caught:
        safety.recheck_source(r.item, r.snapshot, rehash=True)
    assert caught.value.reason == "source_changed"


def test_vanished_or_replaced_source_is_source_changed(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    data = r.source.read_bytes()
    r.source.unlink()
    with pytest.raises(FileFailure) as caught:
        safety.recheck_source(r.item, r.snapshot)
    assert caught.value.reason == "source_changed"
    r.source.write_bytes(data)  # same bytes, new inode
    with pytest.raises(FileFailure) as caught:
        safety.snapshot_source(r.item)
    assert caught.value.reason == "source_changed"


# --- Commit (§ 5.2) ------------------------------------------------------------------------------


def test_commit_publishes_sidecars_first_audio_last_and_leaves_no_residue(rig) -> None:  # type: ignore[no-untyped-def]
    order: list[tuple[str, str | None]] = []
    r = rig(requests=(TAGS, COVER), fault=lambda phase, member: order.append((phase, member)))
    plan = r.prepare()
    candidate_bytes = {m.member_id: m.temporary_path.read_bytes() for m in plan.members}  # type: ignore[union-attr]
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("ok", "ok", "committed")
    assert result.published_paths == (r.folder / "take.wav.cover.jpg", r.folder / "take.wav.tags.json",
                                       r.folder / "take.wav")
    assert [m for phase, m in order if phase == "publish"] == ["cover", "tags", "audio"]
    assert order.index(("receipt", None)) < order.index(("publish", "cover"))
    for member in plan.members:
        assert member.final_path.read_bytes() == candidate_bytes[member.member_id]
    assert r.listing() == ["take.wav", "take.wav.cover.jpg", "take.wav.tags.json"]
    assert r.registry.records() == ()
    assert r.source_unchanged()


def test_commit_refuses_a_revision_or_staged_member_mismatch(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig(requests=(TAGS, COVER))
    r.prepare()
    for changes in ({"metadata_revision": 4}, {"staged_member_ids": ["tags"]},
                    {"staged_member_ids": ["tags", "cover", "cover"]}, {"staged_member_ids": ["tags", "x", "cover"]}):
        result = r.commit(**changes)
        assert (result.status, result.reason, result.execution) == ("failed", "commit_error", "not_written")
        assert result.published_paths == () and r.listing() == []
        r.prepare()
    assert r.source_unchanged()


def test_commit_after_cancellation_settles_as_cancelled_and_publishes_nothing(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    r.prepare()
    flag = threading.Event()
    flag.set()
    result = r.commit(cancel_event=flag)
    assert (result.status, result.reason, result.execution) == ("skipped", "cancelled", "not_written")
    assert r.listing() == [] and r.registry.records() == ()


def test_racer_created_after_planning_is_never_clobbered(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    r.prepare()
    (r.folder / "take.wav").write_bytes(b"racer")
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("skipped", "output_exists", "not_written")
    assert (r.folder / "take.wav").read_bytes() == b"racer"
    assert r.listing() == ["take.wav"] and r.registry.records() == ()
    assert r.source_unchanged()


def test_racer_at_audio_publication_rolls_back_the_published_sidecar(rig) -> None:  # type: ignore[no-untyped-def]
    holder: dict[str, Rig] = {}

    def racer(phase: str, member: str | None) -> None:
        if (phase, member) == ("publish", "audio"):
            (holder["rig"].folder / "take.wav").write_bytes(b"racer")

    r = rig(fault=racer)
    holder["rig"] = r
    r.prepare()
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("skipped", "output_exists", "not_written")
    assert r.listing() == ["take.wav"]  # the sidecar was removed by identity; the racer kept its bytes
    assert (r.folder / "take.wav").read_bytes() == b"racer"
    assert r.registry.records() == ()


@pytest.mark.parametrize("occupant", ["symlink", "config_link", "directory"])
def test_unsafe_commit_time_occupant_is_unsafe_output(rig, occupant: str) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    config = r.root / "norma.toml"
    config.write_bytes(b"cfg")
    reads = r.reads + (safety.protect(config, "config"),)
    r.prepare()
    final = r.folder / "take.wav"
    if occupant == "symlink":
        final.symlink_to(r.source)
    elif occupant == "config_link":
        os.link(config, final)
    else:
        final.mkdir()
    result = r.commit(protected=reads)
    assert config.read_bytes() == b"cfg"
    assert (result.status, result.reason, result.execution) == ("failed", "unsafe_output", "not_written")
    assert r.source.read_bytes() == r.build.wav(50)


@pytest.mark.parametrize("change", ["append", "hard_link"])
def test_source_change_before_commit_publishes_nothing(rig, change: str) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    r.prepare()
    if change == "append":
        with open(r.source, "ab") as handle:
            handle.write(b"x")
    else:  # a new link changes the source's ctime: a detected change, even at the final name
        os.link(r.source, r.folder / "take.wav")
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", "source_changed", "not_written")
    assert r.listing() == ([] if change == "append" else ["take.wav"])


def test_changed_output_folder_is_unsafe_output(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    r.prepare()
    os.rename(r.folder, r.root / "elsewhere")
    r.folder.mkdir()
    result = r.commit()
    assert (result.reason, result.execution) == ("unsafe_output", "not_written")
    assert r.listing() == []


def test_receipt_failure_publishes_nothing(rig) -> None:  # type: ignore[no-untyped-def]
    def fail(phase: str, member: str | None) -> None:
        if phase == "receipt":
            raise OSError(errno.EIO, "I/O error")

    r = rig(fault=fail)
    r.prepare()
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", "commit_error", "not_written")
    assert r.listing() == [] and r.registry.records() == ()


@pytest.mark.parametrize("code, reason", [(errno.ENOSPC, "disk_full"), (errno.EACCES, "permission_denied"),
                                          (errno.EIO, "commit_error")])
def test_publication_fault_after_a_sidecar_rolls_back_with_the_specific_reason(rig, code: int, reason: str) -> None:  # type: ignore[no-untyped-def]
    def fail(phase: str, member: str | None) -> None:
        if (phase, member) == ("publish", "audio"):
            raise OSError(code, os.strerror(code))

    r = rig(fault=fail)
    r.prepare()
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", reason, "not_written")
    assert r.listing() == [] and r.registry.records() == ()
    assert r.source_unchanged()


def test_failed_rollback_reports_the_real_survivors(rig) -> None:  # type: ignore[no-untyped-def]
    def fail(phase: str, member: str | None) -> None:
        if (phase, member) == ("publish", "audio"):
            raise OSError(errno.EIO, "I/O error")
        if (phase, member) == ("rollback", "tags"):
            raise OSError(errno.EIO, "cannot remove")

    r = rig(fault=fail)
    r.prepare()
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", "commit_error", "publication_failed")
    assert result.published_paths == (r.folder / "take.wav.tags.json",)
    assert [d.code for d in result.diagnostics] == ["commit_error", "commit_error"]
    assert any(path.name.startswith(".norma-run1-0-receipt-") for path in result.retained_paths)
    assert (r.folder / "take.wav.tags.json").exists() and not (r.folder / "take.wav").exists()


def test_cleanup_failure_after_publication_is_publication_failed_with_paths(rig) -> None:  # type: ignore[no-untyped-def]
    def fail(phase: str, member: str | None) -> None:
        if phase == "cleanup":
            raise OSError(errno.EIO, "cannot clean")

    r = rig()
    r.registry.fault_hook = fail
    r.prepare()
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", "cleanup_error", "publication_failed")
    assert set(result.published_paths) == {r.folder / "take.wav", r.folder / "take.wav.tags.json"}
    assert any(p.name.startswith(".norma-run1-0-receipt-") for p in result.retained_paths)
    assert (r.folder / "take.wav").exists()  # final audio is never deleted to hide a cleanup failure


def test_unexpected_exception_rolls_back_as_internal(rig) -> None:  # type: ignore[no-untyped-def]
    def fail(phase: str, member: str | None) -> None:
        if (phase, member) == ("published", "audio"):
            raise RuntimeError("bug")

    r = rig(fault=fail)
    r.prepare()
    result = r.commit()
    assert (result.reason, result.execution) == ("internal", "not_written")
    assert r.listing() == []


def test_hard_link_publication_route_when_exclusive_rename_is_unavailable(rig, monkeypatch: pytest.MonkeyPatch,
                                                                           allow_linked_names: object) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(safety, "_RENAMEX", None)
    r = rig()
    r.prepare()
    result = r.commit()
    assert result.execution == "committed"
    assert r.listing() == ["take.wav", "take.wav.tags.json"]
    assert (r.folder / "take.wav").stat().st_nlink == 1


def test_hard_link_route_still_refuses_an_existing_name(rig, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(safety, "_RENAMEX", None)
    r = rig()
    plan = r.prepare()
    (r.folder / "take.wav").write_bytes(b"racer")
    result = r.commit(plan)
    assert result.reason == "output_exists" and (r.folder / "take.wav").read_bytes() == b"racer"


def test_exclusive_rename_rejects_an_existing_destination(root: Path) -> None:
    (root / "a").write_bytes(b"candidate")
    (root / "b").write_bytes(b"racer")
    with pytest.raises(FileExistsError):
        safety.exclusive_rename(root / "a", root / "b")
    assert (root / "b").read_bytes() == b"racer" and (root / "a").read_bytes() == b"candidate"
    safety.exclusive_rename(root / "a", root / "c")
    assert (root / "c").read_bytes() == b"candidate" and not (root / "a").exists()


# --- Overwrite with preimages (§ 5.2; ADR-0016) -----------------------------------------------------


def existing_outputs(r: Rig) -> dict[str, tuple[int, bytes]]:
    r.folder.mkdir(exist_ok=True)
    (r.folder / "take.wav").write_bytes(b"old audio")
    (r.folder / "take.wav.tags.json").write_bytes(b"old tags")
    return {name: ((r.folder / name).stat().st_ino, (r.folder / name).read_bytes())
            for name in ("take.wav", "take.wav.tags.json")}


def test_overwrite_replaces_through_held_preimages_without_touching_old_inodes(rig, allow_linked_names: object) -> None:  # type: ignore[no-untyped-def]
    seen: list[list[str]] = []
    r = rig(overwrite=True)
    old = existing_outputs(r)
    r.registry.fault_hook = lambda phase, member: seen.append(r.listing()) if phase == "publish" else None
    plan = r.prepare()
    with open(r.folder / "take.wav", "rb") as reader:  # an old descriptor keeps reading old bytes
        result = r.commit(plan)
        assert reader.read() == b"old audio"
    assert result.execution == "committed"
    assert (r.folder / "take.wav").read_bytes() == r.build.wav(20)
    assert (r.folder / "take.wav").stat().st_ino != old["take.wav"][0]
    assert sum(name.startswith(".norma-run1-0-pre-") for name in seen[0]) == 2  # both held before any publish
    assert r.listing() == ["take.wav", "take.wav.tags.json"]
    assert r.source_unchanged()


def test_failure_after_overwrite_restores_the_old_outputs(rig, allow_linked_names: object) -> None:  # type: ignore[no-untyped-def]
    def fail(phase: str, member: str | None) -> None:
        if (phase, member) == ("published", "audio"):
            raise OSError(errno.EIO, "read-back failed")

    r = rig(overwrite=True, fault=fail)
    old = existing_outputs(r)
    r.prepare()
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", "commit_error", "not_written")
    for name, (inode, data) in old.items():
        assert (r.folder / name).read_bytes() == data and (r.folder / name).stat().st_ino == inode
    assert r.listing() == ["take.wav", "take.wav.tags.json"]


def test_copy_preimage_when_the_volume_refuses_a_hard_link(rig, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
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
    old = existing_outputs(r)
    r.prepare()
    result = r.commit()
    assert result.execution == "not_written"
    for name, (_inode, data) in old.items():
        assert (r.folder / name).read_bytes() == data  # restored from the checked copy
    assert r.listing() == ["take.wav", "take.wav.tags.json"]


def test_no_preimage_possible_is_backup_error_with_nothing_published(rig, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError(errno.ENOTSUP, "Operation not supported")

    def no_copy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError(errno.EIO, "copy failed")

    monkeypatch.setattr(safety.os, "link", refuse)
    monkeypatch.setattr(RunRegistry, "_copy_preimage", no_copy)
    r = rig(overwrite=True)
    old = existing_outputs(r)
    r.prepare()
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", "backup_error", "not_written")
    for name, (inode, data) in old.items():
        assert (r.folder / name).read_bytes() == data and (r.folder / name).stat().st_ino == inode


def test_failed_restore_keeps_the_trigger_primary_and_the_preimage(rig, allow_linked_names: object) -> None:  # type: ignore[no-untyped-def]
    def fail(phase: str, member: str | None) -> None:
        if (phase, member) == ("published", "audio"):
            raise OSError(errno.EIO, "read-back failed")
        if (phase, member) == ("rollback", "audio"):
            raise OSError(errno.EIO, "restore failed")

    r = rig(overwrite=True, fault=fail)
    existing_outputs(r)
    r.prepare()
    result = r.commit()
    assert (result.status, result.reason, result.execution) == ("failed", "commit_error", "publication_failed")
    assert result.published_paths == (r.folder / "take.wav",)
    assert [d.code for d in result.diagnostics] == ["commit_error", "backup_error"]
    preimages = [p for p in result.retained_paths if p.name.startswith(".norma-run1-0-pre-")]
    assert len(preimages) == 1 and preimages[0].read_bytes() == b"old audio"
    assert (r.folder / "take.wav.tags.json").read_bytes() == b"old tags"  # restored


def test_third_party_replacement_is_never_rolled_back(rig) -> None:  # type: ignore[no-untyped-def]
    holder: dict[str, Rig] = {}

    def fail(phase: str, member: str | None) -> None:
        r = holder["rig"]
        if (phase, member) == ("publish", "audio"):
            (r.folder / "take.wav.tags.json").unlink()
            (r.folder / "take.wav.tags.json").write_bytes(b"third party")
            raise OSError(errno.EIO, "boom")

    r = rig(fault=fail)
    holder["rig"] = r
    r.prepare()
    result = r.commit()
    assert result.execution == "not_written"
    assert (r.folder / "take.wav.tags.json").read_bytes() == b"third party"


# --- Settling unpublished files (§ 5.2 cleanup/retention) ----------------------------------------


def test_settle_removes_candidates_and_retains_only_a_named_failed_candidate(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    plan = r.prepare()
    keep = plan.audio_member.temporary_path
    result = safety.settle(r.registry, 0, FileFailure("verification_peak", "audio_verified"), keep_failed=keep)
    assert (result.status, result.reason, result.execution) == ("failed", "verification_peak", "not_written")
    assert result.retained_failed_path == keep and keep is not None and keep.exists()
    assert r.listing() == [keep.name]


def test_cancellation_overrides_keep_failed(rig) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    plan = r.prepare()
    result = safety.settle(r.registry, 0, SkipFile("cancelled", "rendered"),
                           keep_failed=plan.audio_member.temporary_path)
    assert result.retained_failed_path is None and r.listing() == []


def test_cleanup_failure_before_publication(rig, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    r = rig()
    plan = r.prepare()
    temp = plan.audio_member.temporary_path
    assert temp is not None
    temp.unlink()
    temp.write_bytes(b"someone else's")  # not our inode any more: never removed
    failed = safety.settle(r.registry, 0, FileFailure("render_error", "rendered"))
    assert (failed.status, failed.reason, failed.execution) == ("failed", "render_error", "not_written")
    assert [d.code for d in failed.diagnostics] == ["render_error", "cleanup_error"]
    assert temp.read_bytes() == b"someone else's"
    r2 = rig()
    plan = r2.prepare()
    temp = plan.audio_member.temporary_path
    assert temp is not None
    temp.unlink()
    temp.write_bytes(b"x")
    skipped = safety.settle(r2.registry, 0, SkipFile("cancelled", "rendered"))
    assert (skipped.status, skipped.reason, skipped.execution) == ("failed", "cleanup_error", "not_written")


# --- Run outputs (§ 6) -------------------------------------------------------------------------


def run_destination(root: Path, purpose: str, name: str, *, overwrite: bool = False):  # type: ignore[no-untyped-def]
    source = root / "take.wav"
    source.write_bytes(b"RIFF")
    info = source.stat()
    item = InputItem(source, root, PurePosixPath("take.wav"), (info.st_dev, info.st_ino), 0, True)
    reads = protect_inputs([item])
    checked = preflight({}, [RunOutput(purpose, root / name)], reads, overwrite=overwrite)  # type: ignore[arg-type]
    return checked, reads


def test_report_publication_is_private_then_no_clobber(root: Path) -> None:
    checked, reads = run_destination(root, "report", "r.csv")
    registry = RunRegistry("run1")
    path = safety.publish_run_output(registry, checked.write_set.destinations[0], b"a,b\n", overwrite=False,
                                     protected=reads)
    assert path == root / "r.csv" and path.read_bytes() == b"a,b\n"
    assert sorted(p.name for p in root.iterdir()) == ["r.csv", "take.wav"] and registry.records() == ()


def test_report_racer_without_overwrite_is_a_run_error_and_kept(root: Path) -> None:
    checked, reads = run_destination(root, "report", "r.csv")
    (root / "r.csv").write_bytes(b"racer")
    registry = RunRegistry("run1")
    with pytest.raises(safety.RunOutputError) as caught:
        safety.publish_run_output(registry, checked.write_set.destinations[0], b"new", overwrite=False,
                                  protected=reads)
    assert caught.value.exit_code == 1
    assert (root / "r.csv").read_bytes() == b"racer" and registry.records() == ()
    assert sorted(p.name for p in root.iterdir()) == ["r.csv", "take.wav"]


def test_report_overwrite_replaces_a_safe_regular_file(root: Path) -> None:
    (root / "r.csv").write_bytes(b"old")
    checked, reads = run_destination(root, "report", "r.csv", overwrite=True)
    path = safety.publish_run_output(RunRegistry("run1"), checked.write_set.destinations[0], b"new",
                                     overwrite=True, protected=reads)
    assert path.read_bytes() == b"new"


def test_automatic_report_takes_an_ordinal_when_its_name_is_taken(root: Path) -> None:
    checked, reads = run_destination(root, "report", "norma-report.csv")
    (root / "norma-report.csv").write_bytes(b"racer")
    (root / "norma-report-2.csv").write_bytes(b"racer 2")
    path = safety.publish_run_output(RunRegistry("run1"), checked.write_set.destinations[0], b"new",
                                     overwrite=True, protected=reads, ordinal_retry=True,
                                     reserved_keys=[comparison_key(root / "norma-report-3.csv")])
    assert path == root / "norma-report-4.csv" and path.read_bytes() == b"new"  # first ordinal -2 (D-069)
    assert (root / "norma-report.csv").read_bytes() == b"racer"  # never replaced, even with overwrite
    assert (root / "norma-report-2.csv").read_bytes() == b"racer 2"
    assert not (root / "norma-report-1.csv").exists() and not (root / "norma-report-3.csv").exists()


def test_log_opens_append_only_and_private(root: Path) -> None:
    checked, reads = run_destination(root, "log", "run.log")
    fd = safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    try:
        os.write(fd, b"line\n")
    finally:
        os.close(fd)
    assert stat.S_IMODE((root / "run.log").stat().st_mode) == 0o600
    checked, reads = run_destination(root, "log", "run.log")
    fd = safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    os.write(fd, b"more\n")
    os.close(fd)
    assert (root / "run.log").read_bytes() == b"line\nmore\n"


def test_log_swapped_for_a_symlink_or_another_file_is_refused_unwritten(root: Path) -> None:
    (root / "run.log").write_bytes(b"history")
    checked, reads = run_destination(root, "log", "run.log")
    (root / "run.log").unlink()
    (root / "run.log").symlink_to(root / "take.wav")
    with pytest.raises(UsageError):
        safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    (root / "run.log").unlink()
    (root / "run.log").write_bytes(b"other")
    with pytest.raises(UsageError):
        safety.open_log(checked.write_set.destinations[0], checked.occupants["log"], reads)
    assert (root / "run.log").read_bytes() == b"other" and (root / "take.wav").read_bytes() == b"RIFF"
