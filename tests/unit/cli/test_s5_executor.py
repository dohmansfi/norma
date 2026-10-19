"""Exercise the CLI's real read-only pipeline/batch seam with synthetic I/O edges."""
import json
import signal
import threading
from dataclasses import replace
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from norma import cli, pipeline
from norma.models import InputItem, ProbeResult, SourceTags
from norma.errors import Cancelled


@pytest.mark.parametrize('command', ['analyse', 'verify'])
@pytest.mark.parametrize('phase', ['before', 'handoff', 'straddling', 'worker'])
@pytest.mark.parametrize('interrupts', [1, 2, 3])
def test_interrupt_history_survives_executor_handoff(rig, command, phase, interrupts, monkeypatch, capsys):
    from norma import batch

    old = signal.getsignal(signal.SIGINT)
    before = {p: p.read_bytes() for p in rig.paths[0].parent.rglob('*') if p.is_file()}
    forced, requests, controls, cached, delivered = [], [], [], [], []
    monkeypatch.setattr(pipeline, 'force_kill_tools', lambda: forced.append('force'))
    operation = getattr(pipeline, command + '_items')
    real_execute = batch.execute_batch
    sent = 0

    def interrupt():
        nonlocal sent
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        sent += 1
        if requests:
            assert requests[0].cancel_event.is_set()
        assert forced == (['force'] if sent >= 2 else [])

    def observed(*args, **kwargs):
        callback = kwargs['on_inspection']
        executor = kwargs['execute']
        def on_inspection(record):
            assert threading.current_thread() is threading.main_thread()
            delivered.append(record)
            callback(record)
        def execute(request):
            requests.append(request)
            return executor(request)
        kwargs.update(on_inspection=on_inspection, execute=execute)
        if phase == 'before':
            for _ in range(interrupts):
                interrupt()
            assert kwargs['cancel_event'].is_set()
        elif phase == 'straddling':
            interrupt()
            assert kwargs['cancel_event'].is_set()
        return operation(*args, **kwargs)

    def execute_batch(*args, **kwargs):
        control, request = kwargs['control'], requests[0]
        controls.append(control)
        assert control.commit_lock is request.commit_lock
        assert control.cancel_event is request.cancel_event
        assert control.force_hook is request.force_hook
        assert kwargs['handle_sigint'] is False
        callback = kwargs['on_outcome']
        def on_outcome(outcome):
            cached.append(request.record_for(outcome))
            callback(outcome)
        kwargs['on_outcome'] = on_outcome
        if phase in ('handoff', 'straddling'):
            while sent < interrupts:
                interrupt()
        result = real_execute(*args, **kwargs)
        assert result.cancelled and result.not_started == (3 if phase == 'worker' else 4)
        assert control.force_event.is_set() == (interrupts >= 2)
        return result

    def measure(*args, **kwargs):
        while sent < interrupts:
            interrupt()
        assert kwargs['cancel_event'].is_set()
        raise Cancelled()

    monkeypatch.setattr(pipeline, command + '_items', observed)
    monkeypatch.setattr(batch, 'execute_batch', execute_batch)
    if phase == 'worker':
        monkeypatch.setattr(pipeline, 'measure', measure)
        monkeypatch.setattr(pipeline, 'measure_candidate', measure)
    assert cli.main([command, str(rig.paths[0]), '--config', '/dev/null', '-j', '1']) == 130
    captured = capsys.readouterr()
    assert len(controls) == 1 and sent == interrupts
    assert forced == (['force'] if interrupts >= 2 else [])
    assert signal.getsignal(signal.SIGINT) is old
    assert len(delivered) == (1 if phase == 'worker' else 0)
    assert len(cached) == len(delivered)
    assert all(actual is expected for actual, expected in zip(delivered, cached))
    if phase == 'worker':
        assert delivered[0].outcome.status == 'skipped'
        assert delivered[0].outcome.reason_code == 'cancelled'
        assert captured.out.startswith('SKIP\t')
    assert f'Interrupted: 0 finished, {len(delivered)} cancelled, 0 failed, {4 - len(delivered)} not started.' in captured.err
    assert f'\tnot_started={4 - len(delivered)}\t' in captured.out.splitlines()[-1]
    assert {p: p.read_bytes() for p in rig.paths[0].parent.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('command', ['analyse', 'verify'])
@pytest.mark.parametrize('phase', ['before', 'handoff'])
@pytest.mark.parametrize('interrupts', [0, 1, 2, 3])
def test_independent_cancellation_event_is_not_an_interrupt(rig, command, phase, interrupts, monkeypatch, capsys):
    from norma import batch

    old = signal.getsignal(signal.SIGINT)
    forced, events = [], []
    monkeypatch.setattr(pipeline, 'force_kill_tools', lambda: forced.append('force'))
    operation = getattr(pipeline, command + '_items')
    real_execute = batch.execute_batch
    def observed(*args, **kwargs):
        events.append(kwargs['cancel_event'])
        if phase == 'before':
            events[0].set()
        return operation(*args, **kwargs)
    def execute_batch(*args, **kwargs):
        control = kwargs['control']
        assert control.cancel_event is events[0]
        events[0].set()
        assert not control.force_event.is_set() and not forced
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        for count in range(1, interrupts + 1):
            handler(signal.SIGINT, None)
            assert forced == (['force'] if count >= 2 else [])
            assert control.force_event.is_set() == (count >= 2)
        return real_execute(*args, **kwargs)
    monkeypatch.setattr(pipeline, command + '_items', observed)
    monkeypatch.setattr(batch, 'execute_batch', execute_batch)
    before = {p: p.read_bytes() for p in rig.paths[0].parent.rglob('*') if p.is_file()}
    assert cli.main([command, str(rig.paths[0]), '--config', '/dev/null', '-j', '1']) == 130
    captured = capsys.readouterr()
    assert captured.out.startswith('SUMMARY\t') and '\tnot_started=4\t' in captured.out
    assert 'Interrupted: 0 finished, 0 cancelled, 0 failed, 4 not started.' in captured.err
    assert signal.getsignal(signal.SIGINT) is old
    assert forced == (['force'] if interrupts >= 2 else [])
    assert {p: p.read_bytes() for p in rig.paths[0].parent.rglob('*') if p.is_file()} == before


@pytest.fixture
def rig(tmp_path, monkeypatch, outcome):
    paths = [tmp_path / f'take-{i}.wav' for i in range(4)]
    for p in paths:
        p.write_bytes(b'synthetic read identity')
    tool = tmp_path / 'ffmpeg'
    tool.write_bytes(b'synthetic executable identity')
    items = tuple(InputItem(p, tmp_path, PurePosixPath(p.name), (p.stat().st_dev, p.stat().st_ino), i, True)
                  for i, p in enumerate(paths))
    executable = SimpleNamespace(path=tool, version='synthetic')
    tools = pipeline.ReadOnlyTools(executable, executable)
    monkeypatch.setattr(cli.shutil, 'which', lambda *a: str(tool))
    monkeypatch.setattr(pipeline, 'locate_tools', lambda *a, **k: tools)
    monkeypatch.setattr(pipeline, 'discover_inputs', lambda *a: SimpleNamespace(items=items, hints=(), excluded={},
                      duplicates=0, ignored_links=0, directory_links=0))
    probe = ProbeResult(codec='pcm_s24le', codec_class='lossless', container='wav', audio_stream_index=0,
        ignored_streams=0, sample_rate_hz=48000, bit_depth=24, sample_float=False, sample_format='s32',
        channels=2, channel_layout='stereo', duration_s=20, duration_basis='stream', bitrate_bps=None,
        has_artwork=False, chapters=0, ffprobe_tags={}, ffprobe_version='synthetic')
    monkeypatch.setattr(pipeline, 'classify', lambda *a, **k: probe)
    monkeypatch.setattr(pipeline, 'read_source_tags', lambda *a: SourceTags({}, {}, (), (), (), (), {}))
    measured = replace(outcome().measurement_in, integrated_lufs=-16)
    monkeypatch.setattr(pipeline, 'measure', lambda *a, **k: measured)
    monkeypatch.setattr(pipeline, 'measure_candidate', lambda *a, **k: measured)
    return SimpleNamespace(paths=paths, items=items, measured=measured, tool=tool)


@pytest.mark.parametrize('command', ['analyse', 'verify'])
@pytest.mark.parametrize('interrupts', [1, 2])
def test_cli_interrupts_drain_with_exact_counts_restore_handler_and_force_once(rig, command, interrupts, monkeypatch, capsys):
    old = signal.getsignal(signal.SIGINT)
    forced = []
    monkeypatch.setattr(pipeline, 'force_kill_tools', lambda: forced.append('force'))
    def interrupted(*args, **kwargs):
        handler = signal.getsignal(signal.SIGINT)
        for _ in range(interrupts):
            handler(signal.SIGINT, None)
        assert kwargs['cancel_event'].is_set()
        raise Cancelled()
    monkeypatch.setattr(pipeline, 'measure', interrupted)
    monkeypatch.setattr(pipeline, 'measure_candidate', interrupted)
    assert cli.main([command, str(rig.paths[0]), '--config', '/dev/null', '-j', '1']) == 130
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 2 and lines[0].startswith('SKIP\t')
    assert '\tnot_started=3\t' in lines[-1] and lines[-1].startswith('SUMMARY\t')
    assert 'Interrupted: 0 finished, 1 cancelled, 0 failed, 3 not started.' in captured.err
    assert forced == (['force'] if interrupts == 2 else [])
    assert signal.getsignal(signal.SIGINT) is old


@pytest.mark.parametrize('command', ['analyse', 'verify'])
@pytest.mark.parametrize('terminal', [False, True])
def test_cli_executor_runs_bounded_jobs_with_coordinator_rows_and_sorted_report(rig, command, terminal, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: terminal)
    monkeypatch.setenv('NO_COLOR', '')
    barrier = threading.Barrier(2, timeout=5)
    second = threading.Event()
    active = maximum = 0
    lock = threading.Lock()
    def measure(path, *args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        barrier.wait()
        if path == rig.paths[0]:
            assert second.wait(5)

        with lock:
            active -= 1
        return rig.measured
    monkeypatch.setattr(pipeline, 'measure', measure)
    monkeypatch.setattr(pipeline, 'measure_candidate', measure)
    real_render = cli.report.render_tsv
    real_table = cli.report.render_table
    def table(outcomes, *args, **kwargs):
        assert threading.current_thread() is threading.main_thread()
        assert outcomes[0].input.input_index == 1
        return real_table(outcomes, *args, **kwargs)
    def coordinator(*args, **kwargs):
        assert threading.current_thread() is threading.main_thread()
        if args[0].input.input_index == 1:
            second.set()
        return real_render(*args, **kwargs)
    monkeypatch.setattr(cli.report, 'render_tsv', coordinator)
    monkeypatch.setattr(cli.report, 'render_table', table)
    if terminal:
        # Terminal rows are buffered; the callback itself is still coordinator-owned.
        operation = getattr(pipeline, command + '_items')
        def observed(*args, **kwargs):
            callback = kwargs['on_inspection']
            def on_inspection(record):
                assert threading.current_thread() is threading.main_thread()
                if record.outcome.input.input_index == 1:
                    second.set()
                callback(record)
            kwargs['on_inspection'] = on_inspection
            return operation(*args, **kwargs)
        monkeypatch.setattr(pipeline, command + '_items', observed)
    path = tmp_path / 'result.json'
    assert cli.main([command, str(rig.paths[0]), '--config', '/dev/null', '-j', '2', '--report', str(path)]) == 0
    captured = capsys.readouterr()
    assert maximum == 2
    if terminal:
        assert captured.out.splitlines()[1].split()[1] == '2/4'
    else:
        assert captured.out.splitlines()[0].split('\t')[1] == '2/4'
    document = json.loads(path.read_text())
    assert [row['input_index'] for row in document['outcomes']] == [0, 1, 2, 3]
    assert all(row['execution'] == 'inspect' and row['published_paths'] == [] for row in document['outcomes'])


@pytest.mark.parametrize('racer', [False, True])
def test_verify_report_failure_preserves_rows_and_truthful_final_summary(rig, tmp_path, monkeypatch, capsys, racer):
    path = tmp_path / 'report.json'
    def fail(self, purpose, data):
        assert purpose == 'report'
        if racer:
            path.write_bytes(b'another process owns this report')
        raise pipeline.RunOutputError('report appeared while the run was working' if racer else 'permission failure')
    monkeypatch.setattr(pipeline.RunOutputs, 'publish', fail)
    assert cli.main(['verify', str(rig.paths[0]), '--config', '/dev/null', '--report', str(path)]) == 1
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 5 and all(line.startswith('OK\t') for line in lines[:-1])
    assert '\trun_errors=1\treport=\t' in lines[-1]
    assert 'norma: error:' in captured.err
    assert path.exists() == racer
    if racer:
        assert path.read_bytes() == b'another process owns this report'


@pytest.mark.parametrize('terminal', [False, True])
def test_verify_failure_summary_is_last_with_terminal_repeat_and_no_colour(rig, monkeypatch, capsys, terminal):
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: terminal)
    monkeypatch.setenv('NO_COLOR', '')
    monkeypatch.setattr(pipeline, 'measure_candidate', lambda *a, **k: replace(rig.measured, integrated_lufs=-23))
    assert cli.main(['verify', str(rig.paths[0]), '--config', '/dev/null']) == 1
    captured = capsys.readouterr()
    assert '\x1b' not in captured.out and 'target -16 (default)' in captured.out
    if terminal:
        assert captured.out.startswith('STATUS') and '4 FAIL' in captured.out.splitlines()[-1]
        assert sum(line.startswith('FAIL ') for line in captured.out.splitlines()) == 8
    else:
        assert captured.out.splitlines()[-1].startswith('SUMMARY\t')


@pytest.mark.parametrize('kind', ['missing_sips', 'bad_artwork'])
def test_cli_artwork_errors_keep_preflight_vs_per_file_boundary(rig, tmp_path, monkeypatch, capsys, kind):
    from norma.errors import EnvironmentError_, FileFailure
    from norma.metadata import artwork
    manifest = tmp_path / 'refs.json'
    manifest.write_text(json.dumps({'version': 1, 'defaults': {'artwork': 'missing.png'},
                                    'files': {p.name: {} for p in rig.paths}}))
    def missing():
        raise EnvironmentError_('sips missing')
    def bad(*a, **k):
        raise FileFailure('artwork_invalid', 'content_verified', 'missing image')
    monkeypatch.setattr(pipeline, 'locate_sips_tool', missing if kind == 'missing_sips' else lambda: SimpleNamespace(path=rig.tool))
    monkeypatch.setattr(artwork, 'read_artwork_reference', bad)
    path = tmp_path / 'report.json'
    assert cli.main(['verify', str(rig.paths[0]), '--config', '/dev/null', '--check-tags',
                     '--manifest', str(manifest), '--report', str(path)]) == (3 if kind == 'missing_sips' else 1)
    captured = capsys.readouterr()
    if kind == 'missing_sips':
        assert not captured.out and not path.exists()
    else:
        assert captured.out.startswith('FAIL\t') and path.exists()
        assert {row['reason'] for row in json.loads(path.read_text())['outcomes']} == {'artwork_invalid'}
