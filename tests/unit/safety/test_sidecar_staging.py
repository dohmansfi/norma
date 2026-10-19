"""CR-084 — public sidecar staging, bundle rollback and actual prepared-byte space (S4).

Synthetic containers and payloads only; attacks and faults stay inside tmp_path.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma import safety
from norma.errors import FileFailure
from norma.models import SidecarRequest
from tests.unit.safety.test_publication import COVER, TAGS, Rig, wav_target

PICTURE = SidecarRequest("picture-2", "picture_export", "picture:2", ".picture-2.png")
PAYLOADS = {"tags": '{"title": "Città"}\n'.encode(), "cover": b"jpeg prepared bytes", "picture-2": b"png prepared bytes"}


@pytest.fixture
def rig(tmp_path: Path, build: SimpleNamespace):  # type: ignore[no-untyped-def]
    root = tmp_path.resolve() / "demos"
    root.mkdir()

    def make(**kwargs):  # type: ignore[no-untyped-def]
        return Rig(root, build, requests=(TAGS, COVER, PICTURE), **kwargs)

    return make


def prepare(r: Rig):  # type: ignore[no-untyped-def]
    checked = safety.preflight({0: r.plan}, [], r.reads, overwrite=r.overwrite)
    assert checked.refusals == {}
    r.allocated = r.registry.allocate_plan(0, checked.plans[0], pins=checked.pins)
    r.allocated.audio_member.temporary_path.write_bytes(r.build.wav(20))
    return r.allocated


def stage(r: Rig):  # type: ignore[no-untyped-def]
    plan = prepare(r)
    for member in plan.members:
        if member.purpose != "audio":
            assert r.registry.stage_member(0, member, PAYLOADS[member.member_id]) is member
    return plan


def tags(plan):  # type: ignore[no-untyped-def]
    return next(m for m in plan.members if m.member_id == "tags")


def test_staged_bytes_are_private_flushed_and_committed_with_all_exports(rig, monkeypatch):  # type: ignore[no-untyped-def]
    r = rig()
    r.registry.output_mode = 0o644
    synced = []
    real_fsync = os.fsync

    def fsync(fd):  # type: ignore[no-untyped-def]
        synced.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(safety.os, "fsync", fsync)
    plan = stage(r)
    for member in plan.members:
        assert not member.final_path.exists()
        assert stat.S_IMODE(member.temporary_path.stat().st_mode) == 0o600
        if member.purpose != "audio":
            assert member.temporary_path.read_bytes() == PAYLOADS[member.member_id]
            assert member.temporary_path.stat().st_ino in synced
    result = r.commit()
    assert result.execution == "committed"
    assert result.published_paths == tuple(sorted((m.final_path for m in plan.members if m.purpose != "audio"))) + (plan.audio_member.final_path,)
    for member in plan.members:
        assert stat.S_IMODE(member.final_path.stat().st_mode) == 0o644
        if member.purpose != "audio":
            assert member.final_path.read_bytes() == PAYLOADS[member.member_id]
            assert member.final_path.stat().st_mtime_ns != r.snapshot.mtime_ns
    assert plan.audio_member.final_path.stat().st_mtime_ns == r.snapshot.mtime_ns
    assert r.registry.records() == () and r.source_unchanged()


def test_short_writes_are_completed(rig, monkeypatch):  # type: ignore[no-untyped-def]
    r = rig()
    member = tags(prepare(r))
    real_write = os.write
    calls = []

    def short(fd, data):  # type: ignore[no-untyped-def]
        calls.append(len(data))
        return real_write(fd, data[:2])

    monkeypatch.setattr(safety.os, "write", short)
    r.registry.stage_member(0, member, PAYLOADS["tags"])
    assert len(calls) > 1
    assert member.temporary_path.read_bytes() == PAYLOADS["tags"]
    assert r.registry.close() == () and r.source_unchanged()


@pytest.mark.parametrize("operation", ["open", "ftruncate", "write", "fsync"])
@pytest.mark.parametrize("code, reason", [(errno.ENOSPC, "disk_full"), (errno.EDQUOT, "disk_full"), (errno.EACCES, "permission_denied"), (errno.EIO, "commit_error")])
def test_staging_io_failure_cleans_member_and_settlement_cleans_bundle(rig, monkeypatch, operation, code, reason):  # type: ignore[no-untyped-def]
    r = rig()
    plan = prepare(r)
    member = tags(plan)

    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError(code, os.strerror(code))

    with monkeypatch.context() as patch:
        patch.setattr(safety.os, operation, fail)
        with pytest.raises(FileFailure) as caught:
            r.registry.stage_member(0, member, PAYLOADS["tags"])
    assert caught.value.reason == reason and caught.value.stage == "tags_written"
    assert not member.temporary_path.exists()
    result = safety.settle(r.registry, 0, caught.value)
    assert result.execution == "not_written" and result.published_paths == ()
    assert r.listing() == [] and r.registry.records() == () and r.source_unchanged()


def test_zero_length_write_is_an_io_failure_not_an_infinite_loop(rig, monkeypatch):  # type: ignore[no-untyped-def]
    r = rig()
    member = tags(prepare(r))
    monkeypatch.setattr(safety.os, "write", lambda *args: 0)
    with pytest.raises(FileFailure, match="accepted no bytes"):
        r.registry.stage_member(0, member, b"data")
    assert not member.temporary_path.exists()
    assert r.registry.close() == ()


@pytest.mark.parametrize("attack", ["symlink", "hard_link", "substitute"])
def test_substituted_or_linked_staging_candidate_never_writes_foreign_bytes(rig, attack, allow_linked_names):  # type: ignore[no-untyped-def]
    r = rig()
    member = tags(prepare(r))
    path = member.temporary_path
    foreign = r.root / "foreign"
    foreign.write_bytes(b"protected foreign bytes")
    if attack == "hard_link":
        os.link(path, r.root / "extra-link")
    else:
        path.unlink()
        if attack == "symlink":
            path.symlink_to(foreign)
        else:
            os.link(foreign, path)
    with pytest.raises(FileFailure) as caught:
        r.registry.stage_member(0, member, b"new")
    assert caught.value.reason == "unsafe_output"
    assert foreign.read_bytes() == b"protected foreign bytes" and r.source_unchanged()
    r.registry.close()


def test_open_time_substitution_is_checked_before_truncation(rig, monkeypatch):  # type: ignore[no-untyped-def]
    r = rig()
    member = tags(prepare(r))
    foreign = r.root / "foreign"
    foreign.write_bytes(b"do not truncate")
    real_open = os.open
    flags_seen = []

    def racer(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path == member.temporary_path:
            flags_seen.append(flags)
            assert flags & os.O_NOFOLLOW and not flags & os.O_TRUNC
            return real_open(foreign, flags, *args, **kwargs)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(safety.os, "open", racer)
    with pytest.raises(FileFailure) as caught:
        r.registry.stage_member(0, member, b"new")
    assert caught.value.reason == "unsafe_output" and flags_seen
    assert foreign.read_bytes() == b"do not truncate" and r.source_unchanged()
    assert r.registry.close() == ()


@pytest.mark.parametrize("change", ["input", "id", "final", "unallocated", "other_candidate"])
def test_stage_refuses_wrong_member_provenance(rig, change):  # type: ignore[no-untyped-def]
    r = rig()
    plan = prepare(r)
    member = tags(plan)
    original = member.temporary_path.read_bytes()
    index = 1 if change == "input" else 0
    if change == "id":
        member = replace(member, member_id="cover")
    elif change == "final":
        member = replace(member, final_path=r.folder / "other.tags.json")
    elif change == "unallocated":
        member = tags(r.plan)
    elif change == "other_candidate":
        member = replace(member, temporary_path=plan.audio_member.temporary_path)
    with pytest.raises(FileFailure) as caught:
        r.registry.stage_member(index, member, b"new")
    assert caught.value.reason == "unsafe_output"
    assert tags(plan).temporary_path.read_bytes() == original
    assert r.registry.close() == () and r.source_unchanged()


def test_stage_refuses_audio_and_non_bytes(rig):  # type: ignore[no-untyped-def]
    r = rig()
    plan = prepare(r)
    with pytest.raises(ValueError):
        r.registry.stage_member(0, plan.audio_member, b"new")
    with pytest.raises(TypeError):
        r.registry.stage_member(0, tags(plan), "text")
    assert r.registry.close() == () and r.source_unchanged()


def test_replaced_staging_folder_is_refused(rig):  # type: ignore[no-untyped-def]
    r = rig()
    member = tags(prepare(r))
    moved = r.root / "moved"
    os.rename(r.folder, moved)
    r.folder.mkdir()
    with pytest.raises(FileFailure) as caught:
        r.registry.stage_member(0, member, b"new")
    assert caught.value.reason == "unsafe_output"
    assert r.listing() == [] and r.source_unchanged()


@pytest.mark.parametrize("failed_member", ["cover", "picture-2", "tags", "audio"])
def test_failure_after_any_member_rolls_back_entire_staged_bundle(rig, failed_member):  # type: ignore[no-untyped-def]
    def fail(phase, member):  # type: ignore[no-untyped-def]
        if (phase, member) == ("published", failed_member):
            raise OSError(errno.ENOSPC, "full")

    r = rig(fault=fail)
    stage(r)
    result = r.commit()
    assert (result.reason, result.execution, result.published_paths) == ("disk_full", "not_written", ())
    assert r.listing() == [] and r.registry.records() == () and r.source_unchanged()


@pytest.mark.parametrize("member_id", ["cover", "picture-2", "tags", "audio"])
@pytest.mark.parametrize("interrupt", ["keyboard", "event"])
def test_interrupt_after_any_publication_rolls_back_staged_bundle(rig, member_id, interrupt):  # type: ignore[no-untyped-def]
    event = threading.Event()

    def cancel(phase, member):  # type: ignore[no-untyped-def]
        if (phase, member) == ("published", member_id):
            if interrupt == "keyboard":
                raise KeyboardInterrupt
            event.set()

    r = rig(fault=cancel)
    stage(r)
    result = r.commit(cancel_event=event)
    assert (result.status, result.reason, result.execution) == ("skipped", "cancelled", "not_written")
    assert result.published_paths == () and r.listing() == []
    assert r.registry.records() == () and r.source_unchanged()


def test_interrupt_during_staging_removes_incomplete_bytes(rig, monkeypatch):  # type: ignore[no-untyped-def]
    r = rig()
    member = tags(prepare(r))

    def interrupt(*args):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt

    monkeypatch.setattr(safety.os, "fsync", interrupt)
    with pytest.raises(KeyboardInterrupt):
        r.registry.stage_member(0, member, b"new")
    assert not member.temporary_path.exists()
    assert r.registry.close() == () and r.source_unchanged()


@pytest.mark.parametrize("attack", ["collision", "symlink", "protected_link"])
def test_sidecar_publication_attack_refuses_entire_bundle(rig, attack):  # type: ignore[no-untyped-def]
    r = rig()
    member = tags(stage(r))
    protected = r.root / "manifest.json"
    if attack == "collision":
        member.final_path.write_bytes(b"racer")
    elif attack == "symlink":
        member.final_path.symlink_to(r.source)
    else:
        protected.write_bytes(b"protected")
        r.reads += (safety.protect(protected, "manifest"),)
        os.link(protected, member.final_path)
    result = r.commit()
    assert result.reason == ("output_exists" if attack == "collision" else "unsafe_output")
    assert result.execution == "not_written" and result.published_paths == ()
    assert r.listing() == [member.final_path.name] and r.source_unchanged()
    if attack == "collision":
        assert member.final_path.read_bytes() == b"racer"
    elif attack == "protected_link":
        assert protected.read_bytes() == b"protected"


def test_space_includes_actual_json_and_all_picture_bytes_before_allocation(rig):  # type: ignore[no-untyped-def]
    r = rig()
    lengths = sum(len(data) for data in PAYLOADS.values())
    base = safety.estimate_space(r.snapshot.size, 1.0, wav_target())
    estimate = safety.estimate_space(r.snapshot.size, 1.0, wav_target(), sidecar_bytes=lengths)
    assert estimate.sidecar_bytes == lengths
    assert estimate.required_bytes == (6 * (estimate.audio_bytes + lengths) + 4) // 5 + safety.SPACE_MARGIN_BYTES
    r.registry = safety.RunRegistry("run1", free_space=lambda _: base.required_bytes)
    checked = safety.preflight({0: r.plan}, [], r.reads, overwrite=False)
    with pytest.raises(FileFailure) as caught:
        r.registry.allocate_plan(0, checked.plans[0], pins=checked.pins, space=estimate)
    assert caught.value.reason == "disk_full"
    assert not r.folder.exists() and r.registry.records() == () and r.registry.reserved_bytes() == 0
    assert r.source_unchanged()


def test_receipt_lists_all_staged_members_and_actual_publication_phases(rig, monkeypatch):  # type: ignore[no-untyped-def]
    r = rig()
    plan = stage(r)
    seen = []
    original = r.registry.update_receipt

    def update(path, content):  # type: ignore[no-untyped-def]
        original(path, content)
        seen.append(json.loads(path.read_bytes()))

    monkeypatch.setattr(r.registry, "update_receipt", update)
    result = r.commit()
    assert result.execution == "committed"
    assert {entry["member_id"] for entry in seen[0]["members"]} == {m.member_id for m in plan.members}
    assert all(not entry["published"] for entry in seen[0]["members"])
    assert all(entry["published"] for entry in seen[-1]["members"])
    for entry in seen[-1]["members"]:
        info = Path(entry["final"]).stat()
        assert entry["candidate_identity"] == [info.st_dev, info.st_ino]
    assert r.registry.records() == ()


@pytest.mark.parametrize("interrupt", [False, True])
def test_staged_overwrite_failure_restores_every_old_member(rig, allow_linked_names, interrupt):  # type: ignore[no-untyped-def]
    def fail(phase, member):  # type: ignore[no-untyped-def]
        if (phase, member) == ("published", "audio"):
            if interrupt:
                raise KeyboardInterrupt
            raise OSError(errno.EIO, "publication failed")

    r = rig(overwrite=True, fault=fail)
    r.folder.mkdir()
    old = {}
    for member in r.plan.members:
        payload = f"old {member.member_id}".encode()
        member.final_path.write_bytes(payload)
        old[member.final_path] = (member.final_path.stat().st_ino, payload)
    stage(r)
    result = r.commit()
    assert result.reason == ("cancelled" if interrupt else "commit_error")
    assert result.execution == "not_written" and result.published_paths == ()
    for path, (inode, payload) in old.items():
        assert path.stat().st_ino == inode and path.read_bytes() == payload
    assert r.registry.records() == () and r.source_unchanged()


def test_failed_staged_rollback_names_only_current_run_survivor(rig):  # type: ignore[no-untyped-def]
    def fail(phase, member):  # type: ignore[no-untyped-def]
        if (phase, member) in (("publish", "audio"), ("rollback", "picture-2")):
            raise OSError(errno.EIO, "failed")

    r = rig(fault=fail)
    plan = stage(r)
    survivor = next(m.final_path for m in plan.members if m.member_id == "picture-2")
    result = r.commit()
    assert (result.reason, result.execution) == ("commit_error", "publication_failed")
    assert result.published_paths == (survivor,)
    assert survivor.read_bytes() == PAYLOADS["picture-2"]
    assert not plan.audio_member.final_path.exists() and not tags(plan).final_path.exists()
    assert any(path.name.startswith(".norma-run1-0-receipt-") for path in result.retained_paths)
    assert r.source_unchanged()
