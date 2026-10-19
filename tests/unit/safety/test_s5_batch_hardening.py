"""S5 deterministic scheduler/settlement checks; synthetic workers and bytes only."""

from dataclasses import replace
from pathlib import Path, PurePosixPath
import errno
import signal
import threading

import pytest

from norma import batch, safety
from norma.errors import Cancelled, FileFailure, SkipFile
from norma.models import BundleMember, FileOutcome, InputItem, OutputPlan


def item(index):
    return InputItem(Path(f"/planned/{index}.wav"), Path("/planned"),
                     PurePosixPath(f"{index}.wav"), (1, index + 1), index, True)


def outcome(source, **changes):
    result = FileOutcome(
        status="ok", reason_code="ok", input=source, probe=None, target=None,
        measurement_in=None, measurement_out=None, gain=None, verification=None,
        verification_reference=None, source_codec=None, origin_codec=None,
        metadata=None, tag_write=None, output=None, conditional_plans=(), warnings=(),
        elapsed_s=0.0, ffmpeg_stderr_tail=None, execution="not_written",
        published_paths=(), retained_failed_path=None,
    )
    return replace(result, **changes)


def settle(source, error, elapsed, control):
    assert isinstance(error, Cancelled)
    return outcome(source, status="skipped", reason_code="cancelled", elapsed_s=elapsed)


@pytest.mark.parametrize("jobs", [1, 3])
@pytest.mark.parametrize("failure", [RuntimeError("executor refused submission"), SystemExit(9), KeyboardInterrupt()])
def test_submission_failure_cancels_and_settles_active_worker_before_propagation(monkeypatch, jobs, failure):
    entered = threading.Event()
    settled = threading.Event()
    saw_cancellation = []
    real_pool = batch.ThreadPoolExecutor

    class Pool(real_pool):
        def submit(self, fn, /, *args, **kwargs):
            source = args[0]
            if source.input_index == 1:
                assert entered.wait(2)
                raise failure
            return super().submit(fn, *args, **kwargs)

    def worker(source, control):
        entered.set()
        saw_cancellation.append(control.cancel_event.wait(2))
        raise Cancelled()

    def settler(*args):
        result = settle(*args)
        settled.set()
        return result

    monkeypatch.setattr(batch, "ThreadPoolExecutor", Pool)
    with pytest.raises(type(failure)) as caught:
        batch.execute_batch([item(i) for i in range(5)], worker, settle_error=settler, jobs=jobs)
    assert caught.value is failure
    assert saw_cancellation == [True]
    assert settled.is_set()


class Bundle:
    """Closed synthetic candidates; no audio/tag verification is claimed here."""

    def __init__(self, root, *, overwrite=False):
        root = root.resolve()
        source = root / "source.bin"
        source.write_bytes(b"unchanged source")
        info = source.stat()
        self.item = InputItem(source, root, PurePosixPath(source.name),
                              (info.st_dev, info.st_ino), 0, True)
        self.snapshot = safety.snapshot_source(self.item)
        self.protected = safety.protect_inputs([self.item])
        self.registry = safety.RunRegistry("s5", umask=0o022)
        self.folder = root / "output"
        members = tuple(BundleMember(name, purpose, None, self.folder / filename,
                                     None, str(self.folder / filename))
                        for name, purpose, filename in (
                            ("tags", "tags_sidecar", "take.bin.tags.json"),
                            ("audio", "audio", "take.bin")))
        plan = OutputPlan(1, members, "none", self.item.identity, overwrite)
        self.old = {}
        if overwrite:
            self.folder.mkdir()
            for member in plan.members:
                self.old[member.final_path] = f"old {member.member_id}".encode()
                member.final_path.write_bytes(self.old[member.final_path])
        self.plan = self.registry.allocate_plan(0, plan)
        self.payloads = {}
        for member in self.plan.members:
            data = f"complete {member.member_id}".encode()
            self.payloads[member.final_path] = data
            assert member.temporary_path is not None
            member.temporary_path.write_bytes(data)

    def commit(self, event=None, **diagnostic_options):
        return safety.commit_bundle(self.registry, self.item, self.plan,
                                    snapshot=self.snapshot, protected=self.protected,
                                    metadata_revision=1, staged_member_ids=("tags",),
                                    cancel_event=event, **diagnostic_options)

    def assert_source_unchanged(self):
        safety.recheck_source(self.item, self.snapshot, rehash=True)
        assert self.item.source_path.read_bytes() == b"unchanged source"


def test_cancelled_failed_rollback_lists_candidate_that_cleanup_cannot_remove(tmp_path, monkeypatch):
    bundle = Bundle(tmp_path)
    event = threading.Event()
    audio = bundle.plan.audio_member.temporary_path
    assert audio is not None
    tags = bundle.plan.members[0].final_path
    remove = bundle.registry.remove

    def fault(phase, member):
        if (phase, member) == ("published", "tags"):
            event.set()
        if phase == "rollback":
            raise OSError(errno.EIO, "rollback refused")

    def fail_cleanup(path):
        if path == audio:
            raise OSError(errno.EACCES, "candidate cleanup refused")
        remove(path)

    bundle.registry.fault_hook = fault
    monkeypatch.setattr(bundle.registry, "remove", fail_cleanup)
    result = bundle.commit(event)
    assert (result.status, result.reason, result.execution) == ("failed", "commit_error", "publication_failed")
    assert result.published_paths == (tags,)
    assert audio.exists()
    assert audio in result.retained_paths
    assert set(result.retained_paths) == set(bundle.folder.iterdir()) - {tags}
    assert [d.code for d in result.diagnostics] == ["cancelled", "commit_error", "cleanup_error"]
    bundle.assert_source_unchanged()


@pytest.mark.parametrize("jobs", [1, 3])
def test_interrupt_bounds_queue_and_admission_with_one_row_per_started_file(monkeypatch, jobs):
    real_pool = batch.ThreadPoolExecutor
    submitted = []
    entered = []
    settled = []
    rows = []
    progress = []
    all_active = threading.Event()
    lock = threading.Lock()
    coordinator = threading.get_ident()
    control = batch.BatchControl()

    class Pool(real_pool):
        def submit(self, fn, /, *args, **kwargs):
            submitted.append(args[0].input_index)
            return super().submit(fn, *args, **kwargs)

    def worker(source, state):
        with lock:
            entered.append(source.input_index)
            if len(entered) == jobs:
                all_active.set()
        assert state.cancel_event.wait(2)
        raise Cancelled()

    def settler(source, error, elapsed, state):
        with lock:
            settled.append(source.input_index)
        return settle(source, error, elapsed, state)

    def notify(event):
        assert threading.get_ident() == coordinator
        progress.append(event)
        if event.kind == "started" and not control.interrupted:
            assert all_active.wait(2)
            assert len(submitted) == 2 * jobs
            control.interrupt()

    def done(row):
        assert threading.get_ident() == coordinator
        rows.append(row)

    monkeypatch.setattr(batch, "ThreadPoolExecutor", Pool)
    preflight = outcome(item(30), status="skipped", reason_code="not_audio")
    result = batch.execute_batch([*(item(i) for i in range(30)), preflight], worker,
                                 settle_error=settler, jobs=jobs, control=control,
                                 on_progress=notify, on_outcome=done)
    assert len(submitted) == 2 * jobs
    assert len(entered) == jobs and sorted(entered) == sorted(settled)
    assert [r.input.input_index for r in result.outcomes] == sorted([*entered, 30])
    assert len(rows) == jobs + 1 and len({r.input.input_index for r in rows}) == jobs + 1
    assert result.outcomes[-1] is preflight
    assert all(r.reason_code == "cancelled" for r in result.outcomes[:-1])
    assert result.not_started == 30 - jobs and result.cancelled
    assert sum(e.kind == "started" for e in progress) == jobs
    assert progress[-1].started == jobs and progress[-1].finished == jobs + 1


def test_completion_order_callbacks_and_input_order_result_without_sleep():
    release_first = threading.Event()
    rows = []
    coordinator = threading.get_ident()

    def worker(source, control):
        if source.input_index == 0:
            assert release_first.wait(2)
        return outcome(source)

    def done(row):
        assert threading.get_ident() == coordinator
        rows.append(row.input.input_index)
        if row.input.input_index == 1:
            release_first.set()

    result = batch.execute_batch([item(0), item(1)], worker, settle_error=settle,
                                 jobs=2, on_outcome=done)
    assert rows == [1, 0]
    assert [r.input.input_index for r in result.outcomes] == [0, 1]
    assert result.not_started == 0 and not result.cancelled


@pytest.mark.parametrize("callback", ["on_progress", "on_outcome"])
def test_callback_failure_stops_admission_and_drains_started_files(callback):
    other_entered = threading.Event()
    drained = threading.Event()
    entered = []
    coordinator = threading.get_ident()
    failure = RuntimeError("coordinator callback failed")

    def worker(source, control):
        entered.append(source.input_index)
        if source.input_index == 0:
            assert other_entered.wait(2)
            return outcome(source)
        other_entered.set()
        assert control.cancel_event.wait(2)
        drained.set()
        raise Cancelled()

    def broken(event):
        assert threading.get_ident() == coordinator
        assert other_entered.wait(2)
        raise failure

    with pytest.raises(RuntimeError) as caught:
        batch.execute_batch([item(0), item(1)], worker, settle_error=settle,
                            jobs=2, on_progress=broken if callback == "on_progress" else None,
                            on_outcome=broken if callback == "on_outcome" else None)
    assert caught.value is failure
    assert drained.is_set() and sorted(entered) == [0, 1]


@pytest.mark.parametrize("force_fails", [False, True])
def test_force_hook_remains_armed_during_executor_teardown(monkeypatch, force_fails):
    real_pool = batch.ThreadPoolExecutor
    previous = signal.getsignal(signal.SIGINT)
    settled = threading.Event()
    calls = []
    failure = RuntimeError("force hook failed")

    def force():
        calls.append("force")
        if force_fails:
            raise failure

    control = batch.BatchControl(force_hook=force)

    class Pool(real_pool):
        def __exit__(self, *args):
            assert settled.is_set()
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler) and handler is not previous
            # First interrupt came from the coordinator. Both repeats happen
            # after worker settlement but before executor shutdown completes.
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
            assert control.force_event.is_set() and calls == ["force"]
            return super().__exit__(*args)

    def worker(source, state):
        assert state.cancel_event.wait(2)
        raise Cancelled()

    def settler(*args):
        result = settle(*args)
        settled.set()
        return result

    def notify(event):
        if event.kind == "started":
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(batch, "ThreadPoolExecutor", Pool)
    try:
        if force_fails:
            with pytest.raises(RuntimeError) as caught:
                batch.execute_batch([item(0), item(1)], worker, settle_error=settler,
                                    jobs=1, control=control, on_progress=notify)
            assert caught.value is failure
        else:
            result = batch.execute_batch([item(0), item(1)], worker, settle_error=settler,
                                         jobs=1, control=control, on_progress=notify)
            assert result.cancelled and result.not_started == 1
            assert result.outcomes[0].reason_code == "cancelled"
    finally:
        assert signal.getsignal(signal.SIGINT) is previous


@pytest.mark.parametrize("phase,member,overwrite", [
    (phase, member, overwrite)
    for overwrite in (False, True)
    for phase, member in (
        ("gate", None), ("receipt", None), ("preimage", "tags"),
        ("finalise", "audio"), ("publish", "tags"), ("published", "tags"),
        ("publish", "audio"), ("published", "audio"))
    if overwrite or phase != "preimage"
])
def test_event_cancellation_rolls_back_at_each_publication_phase(
        tmp_path, allow_linked_names, phase, member, overwrite):
    target = (phase, member)
    bundle = Bundle(tmp_path, overwrite=overwrite)
    event = threading.Event()
    reached = []

    def fault(at, name):
        if (at, name) == target:
            reached.append(target)
            event.set()

    bundle.registry.fault_hook = fault
    result = bundle.commit(event)
    assert reached == [target]
    assert (result.status, result.reason, result.execution) == ("skipped", "cancelled", "not_written")
    assert result.published_paths == result.retained_paths == ()
    assert bundle.registry.records() == ()
    assert set(bundle.folder.iterdir()) == set(bundle.old)
    for path, data in bundle.old.items():
        assert path.read_bytes() == data
    bundle.assert_source_unchanged()


@pytest.mark.parametrize("jobs", [1, 3])
@pytest.mark.parametrize("phase,member", [
    ("preimage", "tags"), ("published", "tags"),
    ("published", "audio"), ("cleanup", None), ("rollback", "tags"),
])
def test_sigint_defers_cancellation_across_publication_or_rollback_critical_section(
        tmp_path, allow_linked_names, jobs, phase, member):
    bundle = Bundle(tmp_path, overwrite=True)
    entered_phase = threading.Event()
    release = threading.Event()
    calls = []
    rows = []
    control = batch.BatchControl(commit_lock=bundle.registry.commit_lock,
                                 force_hook=lambda: calls.append("force"))
    previous = signal.getsignal(signal.SIGINT)

    def fault(at, name):
        if phase == "rollback" and (at, name) == ("publish", "audio"):
            raise OSError(errno.EIO, "publication failed before audio")
        if (at, name) == (phase, member):
            entered_phase.set()
            assert release.wait(2)
            assert not control.cancel_event.is_set()

    bundle.registry.fault_hook = fault

    def worker(source, state):
        if source.input_index != 0:
            assert state.cancel_event.wait(2)
            raise Cancelled()
        result = bundle.commit(state.cancel_event)
        return outcome(source, status=result.status, reason_code=result.reason,
                       execution=result.execution, published_paths=result.published_paths,
                       warnings=result.diagnostics)

    def notify(event):
        if event.kind == "started" and event.input.input_index == 0:
            try:
                assert entered_phase.wait(2)
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                signal.raise_signal(signal.SIGINT)
                assert control.interrupted and not control.cancel_event.is_set()
                signal.raise_signal(signal.SIGINT)
                signal.raise_signal(signal.SIGINT)
                assert control.force_event.is_set() and calls == ["force"]
            finally:
                release.set()

    result = batch.execute_batch([bundle.item, *(item(i) for i in range(1, 12))], worker,
                                 settle_error=settle, jobs=jobs, control=control,
                                 on_progress=notify, on_outcome=rows.append)
    assert signal.getsignal(signal.SIGINT) is previous
    assert result.cancelled and control.cancel_event.is_set() and calls == ["force"]
    assert len(rows) == len(result.outcomes)
    assert len({r.input.input_index for r in rows}) == len(rows)
    assert len(rows) + result.not_started == 12
    first = result.outcomes[0]
    if phase == "rollback":
        assert (first.status, first.reason_code, first.execution) == ("failed", "commit_error", "not_written")
        assert first.published_paths == ()
        for path, data in bundle.old.items():
            assert path.read_bytes() == data
    else:
        assert (first.status, first.reason_code, first.execution) == ("ok", "ok", "committed")
        assert set(first.published_paths) == set(bundle.payloads)
        for path, data in bundle.payloads.items():
            assert path.read_bytes() == data
    assert all(r.reason_code == "cancelled" for r in result.outcomes[1:])
    assert bundle.registry.records() == ()
    assert set(bundle.folder.iterdir()) == set(bundle.payloads)
    bundle.assert_source_unchanged()


def test_cancelled_restore_failure_retains_exact_survivor_preimage_and_receipt(
        tmp_path, allow_linked_names):
    bundle = Bundle(tmp_path, overwrite=True)
    event = threading.Event()
    audio = bundle.plan.audio_member.final_path
    tags = bundle.plan.members[0].final_path

    def fault(phase, member):
        if (phase, member) == ("published", "audio"):
            event.set()
        if (phase, member) == ("rollback", "audio"):
            raise OSError(errno.EIO, "restore refused")

    bundle.registry.fault_hook = fault
    result = bundle.commit(event)
    assert (result.status, result.reason, result.execution) == ("failed", "commit_error", "publication_failed")
    assert result.published_paths == (audio,)
    assert audio.read_bytes() == bundle.payloads[audio]
    assert tags.read_bytes() == bundle.old[tags]
    assert [d.code for d in result.diagnostics] == ["cancelled", "backup_error"]
    retained = {p: p.read_bytes() for p in result.retained_paths}
    assert len(retained) == 2
    assert sum(data == bundle.old[audio] for data in retained.values()) == 1
    assert set(bundle.folder.iterdir()) == {*bundle.payloads, *retained}
    assert bundle.registry.close() == ()  # retained recovery evidence must outlive teardown
    assert all(path.read_bytes() == data for path, data in retained.items())
    bundle.assert_source_unchanged()


def test_cancellation_during_failed_postpublication_cleanup_has_interruption_diagnostic(tmp_path):
    bundle = Bundle(tmp_path)
    event = threading.Event()

    def fault(phase, member):
        if phase == "cleanup":
            event.set()
            raise OSError(errno.EACCES, "cleanup refused")

    bundle.registry.fault_hook = fault
    result = bundle.commit(event)
    assert (result.status, result.reason, result.execution) == ("failed", "cleanup_error", "publication_failed")
    assert set(result.published_paths) == set(bundle.payloads)
    assert [d.code for d in result.diagnostics] == ["cleanup_error", "cancelled"]
    assert set(result.retained_paths) == set(bundle.folder.iterdir()) - set(bundle.payloads)
    for path, data in bundle.payloads.items():
        assert path.read_bytes() == data
    bundle.assert_source_unchanged()


@pytest.mark.parametrize("primary", [SkipFile("cancelled", "render"), FileFailure("render_error", "render")])
def test_failed_cleanup_before_publication_never_claims_success(tmp_path, monkeypatch, primary):
    bundle = Bundle(tmp_path)
    audio = bundle.plan.audio_member.temporary_path
    assert audio is not None
    original = bundle.registry.remove

    def remove(path):
        if path == audio:
            raise OSError(errno.EACCES, "cleanup refused")
        original(path)

    monkeypatch.setattr(bundle.registry, "remove", remove)
    result = safety.settle(bundle.registry, 0, primary,
                           keep_failed=audio if primary.reason == "cancelled" else None)
    expected = "cleanup_error" if primary.reason == "cancelled" else "render_error"
    assert (result.status, result.reason, result.execution) == ("failed", expected, "not_written")
    assert result.retained_failed_path is None  # cancellation overrides keep-failed
    assert result.retained_paths == (audio,)
    assert [d.code for d in result.diagnostics] == [primary.reason, "cleanup_error"]
    assert audio.exists() and result.published_paths == ()
    assert not any(path.exists() for path in bundle.payloads)
    bundle.assert_source_unchanged()


@pytest.mark.parametrize("jobs", [1, 3])
@pytest.mark.parametrize("stage", ["pre-allocation", "render", "tags", "verification"])
def test_sigint_settles_synthetic_active_stages_before_deleting_candidates(tmp_path, jobs, stage):
    registry = safety.RunRegistry("stages", umask=0o022)
    control = batch.BatchControl(commit_lock=registry.commit_lock)
    lock = threading.Lock()
    active = []
    stopped = set()
    settled = []
    reached = threading.Event()
    original = registry.remove

    def remove(path):
        record = next(r for r in registry.records() if r.path == path)
        assert record.input_index in stopped  # stand-in writer must stop before deletion
        original(path)

    registry.remove = remove

    def worker(source, state):
        if stage != "pre-allocation":
            final = tmp_path.resolve() / "output" / f"{source.input_index}.bin"
            member = BundleMember("audio", "audio", None, final, None, str(final))
            candidate = registry.allocate(source.input_index, member)
            assert candidate.temporary_path is not None
            candidate.temporary_path.write_bytes(f"incomplete {stage}".encode())
        with lock:
            active.append(source.input_index)
            if len(active) == jobs:
                reached.set()
        assert state.cancel_event.wait(2)
        with lock:
            stopped.add(source.input_index)
        raise Cancelled()

    def settler(source, error, elapsed, state):
        result = safety.settle(registry, source.input_index, SkipFile("cancelled", stage))
        with lock:
            settled.append(source.input_index)
        return outcome(source, status=result.status, reason_code=result.reason,
                       execution=result.execution, published_paths=result.published_paths)

    def notify(event):
        if event.kind == "started" and not control.interrupted:
            assert reached.wait(2)
            signal.raise_signal(signal.SIGINT)

    result = batch.execute_batch([item(i) for i in range(20)], worker, settle_error=settler,
                                 jobs=jobs, control=control, on_progress=notify)
    assert len(active) == jobs and sorted(active) == sorted(settled)
    assert all(r.reason_code == "cancelled" and r.execution == "not_written" for r in result.outcomes)
    assert len(result.outcomes) == jobs and result.not_started == 20 - jobs and result.cancelled
    assert registry.records() == () and registry.close() == ()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("code,reason", [(errno.ENOSPC, "disk_full"), (errno.EIO, "commit_error")])
def test_cancellation_during_failed_rollback_keeps_original_failure_primary(
        tmp_path, allow_linked_names, code, reason):
    bundle = Bundle(tmp_path, overwrite=True)
    event = threading.Event()
    tags = bundle.plan.members[0].final_path
    audio = bundle.plan.audio_member.final_path

    def fault(phase, member):
        if (phase, member) == ("publish", "audio"):
            raise OSError(code, "publication refused")
        if (phase, member) == ("rollback", "tags"):
            event.set()
            raise OSError(errno.EIO, "restore refused")

    bundle.registry.fault_hook = fault
    result = bundle.commit(event)
    assert (result.status, result.reason, result.execution) == ("failed", reason, "publication_failed")
    assert [d.code for d in result.diagnostics] == [reason, "backup_error", "cancelled"]
    assert result.published_paths == (tags,)
    assert tags.read_bytes() == bundle.payloads[tags]
    assert audio.read_bytes() == bundle.old[audio]
    assert len(result.retained_paths) == 2
    assert set(bundle.folder.iterdir()) == {*bundle.payloads, *result.retained_paths}
    bundle.assert_source_unchanged()


def test_cancellation_during_successful_cleanup_preserves_completed_bundle(tmp_path):
    bundle = Bundle(tmp_path)
    event = threading.Event()

    def fault(phase, member):
        if phase == "cleanup":
            event.set()

    bundle.registry.fault_hook = fault
    result = bundle.commit(event)
    assert event.is_set()
    assert (result.status, result.reason, result.execution) == ("ok", "ok", "committed")
    assert set(result.published_paths) == set(bundle.payloads)
    assert result.retained_paths == result.diagnostics == ()
    assert bundle.registry.records() == ()
    for path, data in bundle.payloads.items():
        assert path.read_bytes() == data
    bundle.assert_source_unchanged()


@pytest.mark.parametrize("jobs", [1, 3])
def test_deferred_sigint_during_failed_cleanup_preserves_interruption_diagnostic(tmp_path, jobs):
    bundle = Bundle(tmp_path)
    reached = threading.Event()
    release = threading.Event()
    control = batch.BatchControl(commit_lock=bundle.registry.commit_lock)

    def fault(phase, member):
        if phase == "cleanup":
            reached.set()
            assert release.wait(2)
            raise OSError(errno.EACCES, "cleanup refused")

    def worker(source, state):
        result = bundle.commit(state.cancel_event, interruption_requested=lambda: state.interrupted)
        return outcome(source, status=result.status, reason_code=result.reason,
                       execution=result.execution, published_paths=result.published_paths,
                       warnings=result.diagnostics)

    def notify(event):
        if event.kind == "started":
            try:
                assert reached.wait(2)
                signal.raise_signal(signal.SIGINT)
                assert control.interrupted and not control.cancel_event.is_set()
            finally:
                release.set()

    bundle.registry.fault_hook = fault
    result = batch.execute_batch([bundle.item], worker, settle_error=settle, jobs=jobs,
                                 control=control, on_progress=notify)
    row = result.outcomes[0]
    assert result.cancelled
    assert (row.status, row.reason_code, row.execution) == ("failed", "cleanup_error", "publication_failed")
    assert set(row.published_paths) == set(bundle.payloads)
    bundle.assert_source_unchanged()
    for path, data in bundle.payloads.items():
        assert path.read_bytes() == data
    assert [d.code for d in row.warnings] == ["cleanup_error", "cancelled"]


@pytest.mark.parametrize("requested", [None, False, True])
@pytest.mark.parametrize("phase", ["gate", "cleanup", "rollback"])
def test_pending_interruption_only_changes_failed_diagnostics(
        tmp_path, allow_linked_names, monkeypatch, requested, phase):
    bundle = Bundle(tmp_path, overwrite=phase == "rollback")
    event = threading.Event()
    calls = []
    settled = []
    original = safety._finish if phase == "cleanup" else safety._rollback

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        settled.append(result)
        return result

    monkeypatch.setattr(safety, "_finish" if phase == "cleanup" else "_rollback", capture)

    def pending():
        assert not event.is_set()
        calls.append("pending")
        return requested

    def fault(at, member):
        if (phase == "gate" and at == "gate") or (phase == "rollback" and (at, member) == ("publish", "audio")):
            raise OSError(errno.ENOSPC, "publication refused")
        if (phase == "cleanup" and at == "cleanup") or (phase == "rollback" and (at, member) == ("rollback", "tags")):
            raise OSError(errno.EACCES, "settlement refused")

    bundle.registry.fault_hook = fault
    options = {} if requested is None else {"interruption_requested": pending}
    result = bundle.commit(event, **options)
    assert len(settled) == 1
    assert replace(result, diagnostics=settled[0].diagnostics) == settled[0]
    assert not event.is_set()
    assert calls == ([] if requested is None else ["pending"])
    assert result.status == "failed"
    assert result.reason == ("cleanup_error" if phase == "cleanup" else "disk_full")
    assert result.execution == ("not_written" if phase == "gate" else "publication_failed")
    expected = [result.reason] + (["backup_error"] if phase == "rollback" else [])
    assert [d.code for d in result.diagnostics] == expected + (["cancelled"] if requested else [])
    published = (set(bundle.payloads) if phase == "cleanup" else
                 {bundle.plan.members[0].final_path} if phase == "rollback" else set())
    assert set(result.published_paths) == published
    if phase == "gate":
        assert result.retained_paths == ()
        assert bundle.registry.records() == ()
        assert not any(path.exists() for path in bundle.payloads)
    else:
        assert set(result.retained_paths) == set(bundle.folder.iterdir()) - set(bundle.old) - published
    assert result.retained_failed_path is None
    if phase == "rollback":
        audio = bundle.plan.audio_member.final_path
        assert audio.read_bytes() == bundle.old[audio]
    for path in published:
        assert path.read_bytes() == bundle.payloads[path]
    bundle.assert_source_unchanged()


@pytest.mark.parametrize("requested", [False, True])
def test_success_never_consults_pending_interruption(tmp_path, requested):
    bundle = Bundle(tmp_path)
    calls = []
    event = threading.Event()

    def pending():
        calls.append("pending")
        return requested

    result = bundle.commit(event, interruption_requested=pending)
    assert calls == [] and not event.is_set()
    assert (result.status, result.reason, result.execution) == ("ok", "ok", "committed")
    assert set(result.published_paths) == set(bundle.payloads)
    assert result.diagnostics == result.retained_paths == ()
    assert bundle.registry.records() == ()
    for path, data in bundle.payloads.items():
        assert path.read_bytes() == data
    bundle.assert_source_unchanged()


@pytest.mark.parametrize("event_visible", [False, True])
def test_pending_interruption_never_duplicates_existing_cancelled_diagnostic(
        tmp_path, allow_linked_names, event_visible):
    bundle = Bundle(tmp_path, overwrite=True)
    event = threading.Event()
    calls = []

    def pending():
        calls.append("pending")
        return True

    def fault(at, member):
        if (at, member) == ("published", "tags"):
            if event_visible:
                event.set()
            raise SkipFile("cancelled", "committing", "publication interrupted")
        if (at, member) == ("rollback", "tags"):
            raise OSError(errno.EACCES, "restore refused")

    bundle.registry.fault_hook = fault
    result = bundle.commit(event, interruption_requested=pending)
    assert (result.status, result.reason, result.execution) == ("failed", "commit_error", "publication_failed")
    assert [d.code for d in result.diagnostics] == ["cancelled", "backup_error"]
    assert calls == [] and event.is_set() == event_visible
    assert result.published_paths == (bundle.plan.members[0].final_path,)
    assert len(result.retained_paths) == 2
    bundle.assert_source_unchanged()


def test_event_visible_interruption_does_not_consult_pending_predicate(tmp_path):
    bundle = Bundle(tmp_path)
    event = threading.Event()

    def pending():
        pytest.fail("event-visible cancellation already supplies the diagnostic")

    def fault(at, member):
        if at == "cleanup":
            event.set()
            raise OSError(errno.EACCES, "cleanup refused")

    bundle.registry.fault_hook = fault
    result = bundle.commit(event, interruption_requested=pending)
    assert (result.status, result.reason, result.execution) == ("failed", "cleanup_error", "publication_failed")
    assert [d.code for d in result.diagnostics] == ["cleanup_error", "cancelled"]
    assert set(result.published_paths) == set(bundle.payloads)
    bundle.assert_source_unchanged()


def test_prepublication_cancellation_does_not_consult_pending_predicate(tmp_path):
    bundle = Bundle(tmp_path)
    event = threading.Event()
    event.set()

    def pending():
        pytest.fail("the diagnostic predicate cannot change the cancellation gate")

    result = bundle.commit(event, interruption_requested=pending)
    assert (result.status, result.reason, result.execution) == ("skipped", "cancelled", "not_written")
    assert result.published_paths == result.retained_paths == ()
    assert bundle.registry.records() == ()
    assert not any(path.exists() for path in bundle.payloads)
    bundle.assert_source_unchanged()
