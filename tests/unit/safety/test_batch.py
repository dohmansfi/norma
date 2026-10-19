"""Hermetic batch scheduling, interrupt and publication-lock checks (TST-S-04)."""

from dataclasses import replace
from pathlib import Path, PurePosixPath
import random
import signal
import threading
import time

import pytest

from norma.batch import BatchControl, execute_batch
from norma.errors import Cancelled, FileFailure, FileOutcomeError, SkipFile
from norma.models import Diagnostic, FileOutcome, InputItem, ReasonCode, Status
from norma.safety import RunRegistry


def item(index):
    return InputItem(Path(f"/planned/{index}.wav"), Path("/planned"),
                     PurePosixPath(f"{index}.wav"), (1, index + 1), index, True)


def outcome(source, status: Status = "ok", reason: ReasonCode = "ok", **changes):
    result = FileOutcome(
        status=status, reason_code=reason, input=source, probe=None, target=None,
        measurement_in=None, measurement_out=None, gain=None, verification=None,
        verification_reference=None, source_codec=None, origin_codec=None,
        metadata=None, tag_write=None, output=None, conditional_plans=(), warnings=(),
        elapsed_s=0.0, ffmpeg_stderr_tail=None, execution="not_written",
        published_paths=(), retained_failed_path=None,
    )
    return replace(result, **changes)


def settle(source, error, elapsed, control):
    if isinstance(error, (Cancelled, KeyboardInterrupt)):
        return outcome(source, "skipped", "cancelled", elapsed_s=elapsed)
    if isinstance(error, FileOutcomeError):
        return outcome(source, error.status, error.reason, warnings=error.diagnostics,
                       elapsed_s=elapsed)
    return outcome(source, "failed", "internal", elapsed_s=elapsed,
                   warnings=(Diagnostic("internal", str(error), "worker"),))


def execute(entries, worker, **kwargs):
    return execute_batch(entries, worker, settle_error=settle, **kwargs)


@pytest.mark.parametrize("jobs", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 7, 29])
def test_order_and_worker_bound(jobs, seed):
    delays = [random.Random(seed + i).uniform(0.001, 0.01) for i in range(25)]
    lock = threading.Lock()
    active = peak = 0
    callbacks = []
    progress = []
    coordinator = threading.get_ident()

    def worker(source, control):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(delays[source.input_index])
        with lock:
            active -= 1
        return outcome(source)

    def done(result):
        assert threading.get_ident() == coordinator
        callbacks.append(result)

    result = execute([item(i) for i in range(25)], worker, jobs=jobs,
                     on_outcome=done, on_progress=progress.append)
    assert [row.input.input_index for row in result.outcomes] == list(range(25))
    assert len(callbacks) == 25
    assert peak <= jobs
    assert peak == jobs
    assert progress[-1].started == progress[-1].finished == 25
    assert sum(event.kind == "started" for event in progress) == 25
    assert result.not_started == 0 and not result.cancelled


def test_completion_callbacks_not_input_order():
    second_finished = threading.Event()
    rows = []

    def worker(source, control):
        if source.input_index == 0:
            assert second_finished.wait(2)
            time.sleep(0.02)
        else:
            second_finished.set()
        return outcome(source)

    result = execute([item(0), item(1)], worker, jobs=2,
                     on_outcome=lambda row: rows.append(row.input.input_index))
    assert rows == [1, 0]
    assert [row.input.input_index for row in result.outcomes] == [0, 1]


def test_all_file_planning_barrier_and_preflight_rows():
    planned = []
    worked = []
    rows = []

    def plans():
        for i in range(5):
            planned.append(i)
            yield outcome(item(i), "failed", "probe_error") if i == 2 else item(i)
        planned.append("barrier")

    def worker(source, control):
        assert planned[-1] == "barrier"
        worked.append(source.input_index)
        return outcome(source)

    result = execute(plans(), worker, jobs=2, on_outcome=rows.append)
    assert sorted(worked) == [0, 1, 3, 4]
    assert result.outcomes[2].reason_code == "probe_error"
    assert len(rows) == 5 and result.not_started == 0


def test_planning_exception_starts_nothing():
    worked = []

    def plans():
        yield item(0)
        raise RuntimeError("barrier failed")

    with pytest.raises(RuntimeError, match="barrier failed"):
        execute(plans(), lambda source, control: worked.append(source))
    assert not worked


@pytest.mark.parametrize("error,reason", [
    (RuntimeError("bug"), "internal"),
    (FileFailure("render_error", "render"), "render_error"),
    (SkipFile("output_exists", "allocation"), "output_exists"),
    (Cancelled(), "cancelled"),
])
def test_worker_exceptions_are_settled_per_item(error, reason):
    def worker(source, control):
        if source.input_index == 1:
            raise error
        return outcome(source)

    result = execute([item(i) for i in range(3)], worker, jobs=1)
    assert len(result.outcomes) == 3
    assert result.outcomes[1].reason_code == reason
    assert result.outcomes[2].status == "ok"


def test_first_sigint_cancels_queued_and_preserves_settled_outcomes():
    control = BatchControl()
    rows = []
    worked = []

    def worker(source, state):
        worked.append(source.input_index)
        if source.input_index == 0:
            signal.raise_signal(signal.SIGINT)
            assert state.cancel_event.wait(2)
            raise Cancelled()
        return outcome(source)

    entries = [item(i) for i in range(20)] + [outcome(item(20), "failed", "probe_error")]
    result = execute(entries, worker, jobs=1, control=control, on_outcome=rows.append)
    assert worked == [0]
    assert [row.input.input_index for row in result.outcomes] == [0, 20]
    assert result.outcomes[0].reason_code == "cancelled"
    assert result.not_started == 19 and result.cancelled
    assert len(rows) == 2


def test_repeated_sigint_during_teardown_forces_and_drains():
    killed = threading.Event()
    reaped = threading.Event()
    calls = []
    def force():
        calls.append("force")
        killed.set()

    control = BatchControl(force_hook=force)

    def worker(source, state):
        signal.raise_signal(signal.SIGINT)
        assert state.cancel_event.wait(2)
        signal.raise_signal(signal.SIGINT)
        assert killed.wait(2)
        assert state.force_event.is_set()
        reaped.set()
        raise Cancelled()

    result = execute([item(0), item(1)], worker, jobs=1, control=control)
    assert calls == ["force"] and reaped.is_set()
    assert result.cancelled and result.not_started == 1
    assert result.outcomes[0].reason_code == "cancelled"


def test_interrupt_does_not_split_publication_or_block_force_hook():
    registry = RunRegistry("batch-test")
    published = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    forced = threading.Event()
    control = BatchControl(commit_lock=registry.commit_lock, force_hook=forced.set)

    def worker(source, state):
        with registry.commit_lock:
            assert not state.cancel_event.is_set()
            entered.set()
            assert release.wait(2)
            assert not state.cancel_event.is_set()
            published.set()
        return outcome(source, execution="committed", published_paths=(Path("/published.wav"),))

    def progress(event):
        if event.kind == "started":
            assert entered.wait(2)
            # Simulate two actual handler calls while publication owns the lock.
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            handler(signal.SIGINT, None)
            assert not control.cancel_event.is_set()
            handler(signal.SIGINT, None)
            assert forced.is_set() and control.force_event.is_set()
            release.set()

    result = execute([item(0), item(1)], worker, jobs=1, control=control, on_progress=progress)
    assert published.is_set() and control.cancel_event.is_set()
    assert result.outcomes[0].execution == "committed"
    assert result.outcomes[0].published_paths == (Path("/published.wav"),)
    assert result.not_started == 1 and result.cancelled


def test_unresolved_publication_remains_truthful_on_cancel():
    control = BatchControl()

    def worker(source, state):
        state.interrupt()
        raise Cancelled()

    def unresolved(source, error, elapsed, state):
        return outcome(source, "failed", "commit_error", execution="publication_failed",
                       published_paths=(Path("/surviving.tags.json"),),
                       warnings=(Diagnostic("cancelled", "rollback interrupted", "commit"),))

    result = execute_batch([item(0), item(1)], worker, settle_error=unresolved,
                           jobs=1, control=control)
    assert result.outcomes[0].reason_code == "commit_error"
    assert result.outcomes[0].execution == "publication_failed"
    assert result.outcomes[0].published_paths == (Path("/surviving.tags.json"),)
    assert result.not_started == 1 and result.cancelled


def test_pre_cancelled_run_keeps_preflight_rows():
    event = threading.Event()
    event.set()
    worked = []
    result = execute([item(0), outcome(item(1), "skipped", "not_audio"), item(2)],
                     lambda source, state: worked.append(source),
                     control=BatchControl(cancel_event=event))
    assert not worked
    assert result.not_started == 2 and result.cancelled
    assert result.outcomes[0].input.input_index == 1


@pytest.mark.parametrize("jobs", [True, 0, -1, 1.5, "2"])
def test_invalid_jobs(jobs):
    with pytest.raises(ValueError, match="jobs"):
        execute([], lambda source, state: outcome(source), jobs=jobs)


@pytest.mark.parametrize("indices", [[1, 0], [0, 0]])
def test_invalid_order(indices):
    with pytest.raises(ValueError, match="input_index"):
        execute([item(i) for i in indices], lambda source, state: outcome(source))


@pytest.mark.parametrize("cpus,expected", [(None, 1), (1, 1), (2, 2), (32, 4)])
def test_default_jobs(monkeypatch, cpus, expected):
    import norma.batch as batch

    real_pool = batch.ThreadPoolExecutor
    sizes = []

    def pool(*, max_workers, **kwargs):
        sizes.append(max_workers)
        return real_pool(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(batch.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(batch, "ThreadPoolExecutor", pool)
    assert execute([], lambda source, state: outcome(source)).outcomes == ()
    assert sizes == [expected]


def test_signal_handler_restored_on_success_and_settler_failure():
    previous = signal.getsignal(signal.SIGINT)
    execute([], lambda source, state: outcome(source))
    assert signal.getsignal(signal.SIGINT) is previous

    def worker(source, state):
        raise RuntimeError("worker failed")

    def broken_settler(*args):
        raise RuntimeError("settlement failed")

    with pytest.raises(RuntimeError, match="settlement failed"):
        execute_batch([item(0)], worker, settle_error=broken_settler)
    assert signal.getsignal(signal.SIGINT) is previous


def test_progress_failure_still_drains_worker():
    settled = threading.Event()

    def worker(source, state):
        assert state.cancel_event.wait(2)
        settled.set()
        raise Cancelled()

    def broken_progress(event):
        raise RuntimeError("presentation failed")

    with pytest.raises(RuntimeError, match="presentation failed"):
        execute([item(0), item(1)], worker, jobs=1, on_progress=broken_progress)
    assert settled.is_set()


def test_bad_worker_outcome_goes_through_settlement():
    result = execute([item(0)], lambda source, state: outcome(item(1)))
    assert result.outcomes[0].reason_code == "internal"


def test_force_hook_failure_is_raised_only_after_drain():
    settled = threading.Event()

    def broken_hook():
        raise RuntimeError("kill request failed")

    control = BatchControl(force_hook=broken_hook)

    def worker(source, state):
        state.interrupt()
        state.interrupt()
        settled.set()
        raise Cancelled()

    with pytest.raises(RuntimeError, match="kill request failed"):
        execute([item(0)], worker, control=control)
    assert settled.is_set()


@pytest.mark.parametrize("jobs", [1, 3])
def test_submission_is_bounded_not_entire_run(monkeypatch, jobs):
    import norma.batch as batch

    real_pool = batch.ThreadPoolExecutor
    submitted = []
    first_window = threading.Event()
    control = BatchControl()

    class Pool(real_pool):
        def submit(self, fn, *args, **kwargs):
            submitted.append(args[0].input_index)
            future = super().submit(fn, *args, **kwargs)
            if len(submitted) == 2 * jobs:
                first_window.set()
            return future

    monkeypatch.setattr(batch, "ThreadPoolExecutor", Pool)

    def worker(source, state):
        assert first_window.wait(2)
        assert len(submitted) == 2 * jobs
        assert state.cancel_event.wait(2)
        raise Cancelled()

    def progress(event):
        if event.kind == "started":
            assert first_window.is_set()
            control.interrupt()

    result = execute([item(i) for i in range(100)], worker, jobs=jobs,
                     control=control, on_progress=progress)
    assert len(submitted) == 2 * jobs
    assert len(result.outcomes) <= jobs
    assert len(result.outcomes) + result.not_started == 100


def test_settler_failure_drains_other_active_files():
    barrier = threading.Barrier(2)
    drained = threading.Event()

    def worker(source, state):
        barrier.wait(timeout=2)
        if source.input_index == 0:
            raise RuntimeError("worker failed")
        assert state.cancel_event.wait(2)
        drained.set()
        raise Cancelled()

    def settler(source, error, elapsed, state):
        if source.input_index == 0:
            raise RuntimeError("settlement failed")
        return settle(source, error, elapsed, state)

    with pytest.raises(RuntimeError, match="settlement failed"):
        execute_batch([item(0), item(1)], worker, jobs=2, settle_error=settler)
    assert drained.is_set()


def test_custom_handler_is_untouched_with_sigint_opt_out():
    previous = signal.getsignal(signal.SIGINT)

    def handler(*args):
        pass

    signal.signal(signal.SIGINT, handler)
    try:
        execute([item(0)], lambda source, state: outcome(source), handle_sigint=False)
        assert signal.getsignal(signal.SIGINT) is handler
    finally:
        signal.signal(signal.SIGINT, previous)


def test_non_main_coordinator_can_opt_out():
    results = []

    def coordinator():
        with pytest.raises(ValueError, match="main thread"):
            execute([], lambda source, state: outcome(source))
        results.append(execute([item(0)], lambda source, state: outcome(source),
                               handle_sigint=False))

    thread = threading.Thread(target=coordinator)
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(results) == 1 and len(results[0].outcomes) == 1


def test_worker_keyboard_interrupt_is_settled_and_stops_admission():
    def worker(source, state):
        raise KeyboardInterrupt()

    result = execute([item(0), item(1)], worker, jobs=1)
    assert result.cancelled and result.not_started == 1
    assert result.outcomes[0].reason_code == "cancelled"


def test_third_interrupt_does_not_repeat_force_hook():
    calls = []
    control = BatchControl(force_hook=lambda: calls.append("force"))
    for _ in range(3):
        control.interrupt()
    assert calls == ["force"]
    assert control.cancel_event.is_set() and control.force_event.is_set()
