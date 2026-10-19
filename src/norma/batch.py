"""Bounded file scheduling and cancellation (safety § 6; D-065).

The caller completes all-file planning and reservation before calling ``execute_batch``.
Its ordered entries are eligible C-01 items or already settled C-09 outcomes.
Workers know the plans through their callable; batch makes no audio decisions or
writes. The mandatory error settler must stop/reap writers and call safety's
settlement/rollback hooks before returning a truthful outcome. Batch never guesses
that a file which raised has no surviving publication.
"""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from types import FrameType
from typing import Literal, cast

from norma.errors import Cancelled, check_primary
from norma.models import FileOutcome, InputItem


class BatchControl:
    """Run-scoped cancellation shared with workers, safety and tool adapters.

    Pass ``RunRegistry.commit_lock`` for writing runs. The pipeline must hold
    this same lock while checking cancellation and publishing a complete bundle.
    ``interrupt`` is safe to call from a SIGINT handler: it never waits for the
    commit lock. The coordinator retries a deferred first cancellation; admission
    also checks ``interrupted``, so no new file starts during that deferral.

    The force hook must be quick, non-blocking and safe in a signal handler. It
    requests adapter-owned process-group killing, never performs publication or
    cleanup itself. Force remains armed throughout worker teardown (D-065).
    """

    def __init__(self, *, commit_lock: threading.RLock | None = None,
                 cancel_event: threading.Event | None = None,
                 force_event: threading.Event | None = None,
                 force_hook: Callable[[], None] | None = None) -> None:
        self.commit_lock = commit_lock if commit_lock is not None else threading.RLock()
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.force_event = force_event if force_event is not None else threading.Event()
        self.force_hook = force_hook
        self._interrupts = 0
        self._hook_error: Exception | None = None

    @property
    def interrupted(self) -> bool:
        return self._interrupts > 0 or self.cancel_event.is_set()

    def interrupt(self) -> None:
        """Request graceful cancellation first, immediate adapter escalation next."""
        self._interrupts += 1
        if self._interrupts == 2:
            self.force_event.set()
            if self.force_hook is not None:
                try:
                    self.force_hook()
                except Exception as exc:
                    # Do not abandon running workers from inside a signal handler.
                    self._hook_error = exc
        self.flush_cancellation()

    def flush_cancellation(self) -> None:
        """Set the first-interrupt flag only outside a publication critical section."""
        if self._interrupts and not self.cancel_event.is_set():
            if self.commit_lock.acquire(blocking=False):
                try:
                    self.cancel_event.set()
                finally:
                    self.commit_lock.release()


@dataclass(frozen=True)
class BatchProgress:
    """Coordinator-only event; finished includes preflight-settled entries.

    Started counts worker entries, not submissions. No guessed audio percentage
    is emitted here; duration-aware tool progress belongs to the per-file caller.
    """

    kind: Literal["started", "finished"]
    input: InputItem
    started: int
    finished: int
    total: int
    outcome: FileOutcome | None = None


@dataclass(frozen=True)
class BatchResult:
    """Final input-ordered rows and unstarted accounting for C-09 projection."""

    outcomes: tuple[FileOutcome, ...]
    not_started: int
    cancelled: bool


Worker = Callable[[InputItem, BatchControl], FileOutcome]
ErrorSettler = Callable[[InputItem, BaseException, float, BatchControl], FileOutcome]


def _validate_outcome(item: InputItem, outcome: FileOutcome) -> FileOutcome:
    if not isinstance(outcome, FileOutcome) or outcome.input != item:
        raise ValueError("worker/settler outcome must belong to its input")
    check_primary(outcome.status, outcome.reason_code)
    return outcome


def execute_batch(entries: Iterable[InputItem | FileOutcome], worker: Worker, *,
                  settle_error: ErrorSettler, jobs: int | None = None,
                  control: BatchControl | None = None,
                  on_progress: Callable[[BatchProgress], None] | None = None,
                  on_outcome: Callable[[FileOutcome], None] | None = None,
                  handle_sigint: bool = True) -> BatchResult:
    """Execute a fully planned run, with at most jobs active and jobs pending.

    Materialises and validates *every* entry before creating the executor; callers
    may supply a planning generator whose final step is the all-file write barrier.
    No worker starts if that generator raises. Preflight rows and completion-order
    callbacks run on the coordinator thread; final rows follow input_index.

    ``settle_error`` handles escaped Exception, Cancelled and worker-local
    KeyboardInterrupt. It must preserve committed/publication_failed outcomes and
    diagnose required cleanup; failure of the settler is propagated after all
    workers are drained, never replaced by a fabricated cancelled/internal row.
    Other BaseException instances likewise propagate after drain. Callbacks which
    fail stop admission and are propagated after settlement. Tool deadlines,
    two-second graceful termination, killing and reaping are adapter-owned.

    SIGINT handling is opt-out for callers with their own handler or a non-main
    coordinator; they call control.interrupt themselves. The previous handler is
    restored only after workers have finished teardown. No process is abandoned.
    """
    if jobs is None:
        jobs = max(1, min(4, os.cpu_count() or 1))
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if handle_sigint and threading.current_thread() is not threading.main_thread():
        raise ValueError("SIGINT handling requires the main thread")
    planned = tuple(entries)
    items = tuple(entry.input if isinstance(entry, FileOutcome) else entry for entry in planned)
    if not all(isinstance(item, InputItem) for item in items):
        raise TypeError("entries must be InputItem or FileOutcome records")
    indices = tuple(item.input_index for item in items)
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("entries must have unique, increasing input_index values")
    for entry in planned:
        if isinstance(entry, FileOutcome):
            _validate_outcome(entry.input, entry)
    state = control if control is not None else BatchControl()
    events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
    outcomes: dict[int, FileOutcome] = {}
    started = finished = 0
    error: BaseException | None = None
    pending: set[Future[FileOutcome | None]] = set()
    cursor = 0

    def run(item: InputItem) -> FileOutcome | None:
        # Admission and publication use the same lock as first cancellation.
        with state.commit_lock:
            if state.interrupted:
                return None
            events.put(("started", item))
        began = time.monotonic()
        try:
            return _validate_outcome(item, worker(item, state))
        except (Exception, Cancelled, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                state.interrupt()
            return _validate_outcome(item, settle_error(item, exc, time.monotonic() - began, state))

    def emit(kind: Literal["started", "finished"], item: InputItem,
             outcome: FileOutcome | None = None) -> None:
        nonlocal error
        if error is not None:
            return
        try:
            if on_progress is not None:
                on_progress(BatchProgress(kind, item, started, finished, len(items), outcome))
            if outcome is not None and on_outcome is not None:
                on_outcome(outcome)
        except BaseException as exc:
            error = exc
            state.interrupt()

    def sigint_handler(signum: int, frame: FrameType | None) -> None:
        state.interrupt()

    previous = signal.getsignal(signal.SIGINT) if handle_sigint else None
    if handle_sigint:
        signal.signal(signal.SIGINT, sigint_handler)
    try:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="norma") as pool:
            while cursor < len(planned) or pending:
                state.flush_cancellation()
                if state.interrupted:
                    for future in pending:
                        future.cancel()
                    # Preflight outcomes remain real even when no work starts.
                    for entry in planned[cursor:]:
                        if isinstance(entry, FileOutcome):
                            outcomes[entry.input.input_index] = entry
                            finished += 1
                            emit("finished", entry.input, entry)
                    cursor = len(planned)
                while cursor < len(planned) and len(pending) < 2 * jobs and not state.interrupted:
                    entry = planned[cursor]
                    cursor += 1
                    if isinstance(entry, FileOutcome):
                        outcomes[entry.input.input_index] = entry
                        finished += 1
                        emit("finished", entry.input, entry)
                    else:
                        try:
                            future = pool.submit(run, entry)
                        except BaseException as exc:
                            # Shutdown waits for workers: request their settlement first,
                            # then drain existing futures with the handler still armed.
                            error = exc
                            state.interrupt()
                            break
                        pending.add(future)
                        future.add_done_callback(lambda done: events.put(("done", done)))
                if not pending:
                    continue
                try:
                    kind, value = events.get(timeout=0.02)
                except queue.Empty:
                    continue
                if kind == "started":
                    started += 1
                    emit("started", cast(InputItem, value))
                    continue
                future = cast(Future[FileOutcome | None], value)
                pending.remove(future)
                if future.cancelled():
                    continue
                try:
                    outcome = future.result()
                except BaseException as exc:
                    if error is None:
                        error = exc
                    state.interrupt()
                    continue
                if outcome is not None:
                    outcomes[outcome.input.input_index] = outcome
                    finished += 1
                    emit("finished", outcome.input, outcome)
        state.flush_cancellation()
        if error is not None:
            raise error
        if state._hook_error is not None:
            raise state._hook_error
        return BatchResult(tuple(outcomes[index] for index in sorted(outcomes)),
                           len(items) - len(outcomes), state.interrupted)
    finally:
        if handle_sigint:
            signal.signal(signal.SIGINT, previous)
