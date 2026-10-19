"""TK-0315: real CLI signals with injected preflight faults, never real user audio."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import pytest


NATIVE_FAULT = '''
import json, signal, sys
from norma import cli, pipeline
case = json.loads(sys.argv[1])
old = signal.getsignal(signal.SIGINT)
forced = []
pipeline.force_kill_tools = lambda: forced.append('force')
def failing(*args, **kwargs):
    for _ in range(case['signals']):
        signal.raise_signal(signal.SIGINT)
        assert kwargs['cancel_event'].is_set(), 'live CLI did not record SIGINT'
    raise OSError('injected genuine capability evidence read failure')
pipeline.collect_lossless_evidence = failing
code = cli.main(sys.argv[2:])
assert signal.getsignal(signal.SIGINT) is old, 'SIGINT handler not restored'
assert forced == (['force'] if case['signals'] >= 2 else [])
raise SystemExit(code)
'''


def snapshot(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            if p.is_file() else None for p in root.rglob('*')}


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('signals', [0, 1, 2, 3])
def test_native_capability_error_after_received_sigint_is_exit_130(tmp_path, norma_environment, dry_run, signals):
    executable = Path(norma_environment['ffmpeg_path'])
    if not executable.is_file() or not executable.with_name('ffprobe').is_file():
        pytest.skip('explicit native FFmpeg/ffprobe pair required')
    source = tmp_path / 'source.wav'
    with wave.open(str(source), 'wb') as writer:
        writer.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
        writer.writeframes(b'\0\0\0\0' * 4800)
    before = snapshot(tmp_path)
    child = subprocess.run([sys.executable, '-c', NATIVE_FAULT, json.dumps({'signals': signals}), 'normalise', str(source),
        '--config', os.devnull, '--format', 'wav', '--ffmpeg', str(executable),
        '-o', str(tmp_path / 'out'), '--report', str(tmp_path / 'report.json'),
        '--log', str(tmp_path / 'run.log')] + (['--dry-run', '--quiet'] if dry_run else []),
        capture_output=True, text=True, timeout=30)
    assert child.stdout == ''
    assert 'norma: error: cannot collect FFmpeg capability evidence: injected genuine capability evidence read failure' in child.stderr
    assert 'Run: norma doctor' in child.stderr
    assert 'Traceback' not in child.stderr
    assert snapshot(tmp_path) == before
    assert child.stderr.count('Interrupted before processing finished.') == bool(signals)
    assert child.returncode == (130 if signals else 3), child.stderr


# Real main/parser/config/BatchControl/handler; synthetic startup and error seams.
# No worker or native encoder is admitted by this ordering-focused harness.
SYNTHETIC_FAULT = '''
import contextlib, io, json, signal, sys, threading
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from norma import batch, cli, pipeline
from norma.errors import UsageError, EnvironmentError_
from norma.models import InputItem
case = json.loads(sys.argv[1])
source = Path(sys.argv[2])
args = sys.argv[3:]
def previous(signum, frame):
    raise AssertionError('signal escaped CLI-owned lifetime')
signal.signal(signal.SIGINT, previous)
forced, controls, calls = [], [], []
original_control = batch.BatchControl
class ObservedControl(original_control):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        controls.append(self)
    def interrupt(self):
        calls.append('interrupt')
        super().interrupt()
batch.BatchControl = ObservedControl
pipeline.force_kill_tools = lambda: forced.append('preflight')
tool = SimpleNamespace(path=source, version='synthetic')
cli.shutil.which = lambda *a: str(source)
pipeline.locate_tools = lambda *a, **k: pipeline.ReadOnlyTools(tool, tool)
pipeline.locate_normalise_tools = lambda *a: SimpleNamespace(executable=tool, prober=tool)
item = InputItem(source, source.parent, PurePosixPath(source.name), (1, 2), 0, True)
pipeline.discover_inputs = lambda *a: SimpleNamespace(items=(item,), hints=(), excluded={},
    duplicates=0, ignored_links=0, directory_links=0)
error = {'usage': UsageError, 'environment': EnvironmentError_, 'unexpected': RuntimeError}[case['error']](
    'injected original preflight diagnostic')
sent = 0
def send(count):
    global sent
    for _ in range(count):
        signal.raise_signal(signal.SIGINT)
        sent += 1
        assert controls[0].cancel_event.is_set()
def fail():
    send(case['signals'] - sent)
    raise error
def execute_batch(*a, **kwargs):
    control = kwargs['control']
    assert len(controls) == 1 and control is controls[0]
    assert kwargs['handle_sigint'] is False
    assert control.commit_lock is request.commit_lock
    assert control.cancel_event is request.cancel_event
    assert control.force_hook is request.force_hook
    fail()
batch.execute_batch = execute_batch
request = None
def operation(*a, **kwargs):
    global request
    event = kwargs['cancel_event']
    assert event is controls[-1].cancel_event
    if case.get('event_only'):
        event.set()
    if case.get('internal_stop'):
        controls[0].interrupt()
    if case['phase'] == 'before':
        fail()
    if case['phase'] == 'straddling':
        send(1)
    request = SimpleNamespace(commit_lock=threading.RLock(), cancel_event=event,
        force_hook=lambda: forced.append('executor'), outputs=SimpleNamespace(open_log=fail))
    def forbidden(*a):
        raise AssertionError('unexpected worker, settlement or row')
    common = ((), forbidden, forbidden, 1, request.commit_lock, event, request.force_hook, forbidden)
    request = (pipeline.BatchRequest(*common, request.outputs) if a[1].command == 'normalise'
               else pipeline.ReadOnlyBatchRequest(*common, forbidden))
    return kwargs['execute'](request)
for name in ('normalise_items', 'analyse_items', 'verify_items', 'inspect_tags',
             'plan_tags_items', 'template_tags_items'):
    setattr(pipeline, name, operation)
try:
    code = cli.main(args)
except RuntimeError as caught:
    assert case['error'] == 'unexpected' and caught is error
    print('UNEXPECTED PRESERVED', file=sys.stderr)
    code = 87
assert signal.getsignal(signal.SIGINT) is previous
assert len(controls) == 1
assert len(calls) == case['signals'] + bool(case.get('internal_stop'))
expected_owner = 'preflight' if case['phase'] == 'before' else 'executor'
assert forced == ([expected_owner] if case['signals'] >= 2 else [])
if case['error'] != 'unexpected':
    assert error.exit_code == (2 if case['error'] == 'usage' else 3)
# The same real command on a second invocation must forget the previous signal.
case.update(signals=0, phase='before', event_only=False, internal_stop=False)
with contextlib.redirect_stderr(io.StringIO()) as later:
    try:
        later_code = cli.main(args)
    except RuntimeError as caught:
        assert case['error'] == 'unexpected' and caught is error
    else:
        assert later_code == error.exit_code
assert 'Interrupted' not in later.getvalue()
assert len(controls) == 2
assert signal.getsignal(signal.SIGINT) is previous
# Received-signal state must not leak to a subsequent invocation.
assert cli.main(['normalise', '--in-place', '--help']) == 2
assert signal.getsignal(signal.SIGINT) is previous
raise SystemExit(code)
'''


COMMANDS = ['normalise', 'dry_run', 'analyse', 'verify', 'tags_show', 'tags_plan', 'tags_template']
SEAMS = [(command, phase) for command in COMMANDS
         for phase in (['before', 'handoff', 'straddling'] if command in ('normalise', 'analyse', 'verify')
                       else ['before'])] + [('normalise', 'open_log')]


def run_fault(tmp_path, command, case, *, quiet=False):
    source = tmp_path / 'source.wav'
    source.write_bytes(b'synthetic source identity, never decoded')
    args = ['normalise', '--dry-run'] if command == 'dry_run' else command.split('_')
    args += [str(source), '--config', os.devnull]
    if command == 'tags_template':
        args += ['-o', str(tmp_path / 'template.csv')]
    elif command != 'tags_show':
        args += ['--report', str(tmp_path / 'report.json')]
    if command in ('normalise', 'dry_run'):
        args += ['-o', str(tmp_path / 'out')]
        if case['phase'] == 'open_log':
            args += ['--log', str(tmp_path / 'run.log')]
    if quiet:
        args += ['--quiet']
    before = snapshot(tmp_path)
    child = subprocess.run([sys.executable, '-c', SYNTHETIC_FAULT, json.dumps(case), str(source), *args],
        capture_output=True, text=True, timeout=15)
    assert snapshot(tmp_path) == before
    assert child.stdout == ''
    assert 'Traceback' not in child.stderr, child.stderr
    return child


@pytest.mark.parametrize('command,phase,signals', [
    (command, phase, signals) for command, phase in SEAMS for signals in (0, 1, 2, 3)
    if phase != 'straddling' or signals > 0])
@pytest.mark.parametrize('error', ['usage', 'environment'])
def test_expected_preflight_error_keeps_diagnostic_and_signal_priority(tmp_path, command, phase, signals, error):
    child = run_fault(tmp_path, command, dict(error=error, phase=phase, signals=signals), quiet=True)
    assert child.returncode == (130 if signals else 2 if error == 'usage' else 3), child.stderr
    assert child.stderr.count('norma: error: injected original preflight diagnostic') == 1
    expected_command = 'normalise' if command == 'dry_run' else command.replace('_', ' ')
    assert ('Run: norma doctor' if error == 'environment' else f'Try: norma {expected_command} --help') in child.stderr
    assert child.stderr.count('Interrupted before processing finished.') == bool(signals)
    assert 'SUMMARY' not in child.stderr and 'not started' not in child.stderr


@pytest.mark.parametrize('command', COMMANDS)
@pytest.mark.parametrize('stop', ['event_only', 'internal_stop'])
@pytest.mark.parametrize('error', ['usage', 'environment'])
def test_non_signal_stop_does_not_override_genuine_error(tmp_path, command, stop, error):
    child = run_fault(tmp_path, command, dict(error=error, phase='before', signals=0, **{stop: True}))
    assert child.returncode == (2 if error == 'usage' else 3), child.stderr
    assert 'norma: error: injected original preflight diagnostic' in child.stderr
    assert 'Interrupted' not in child.stderr


@pytest.mark.parametrize('command,phase', SEAMS)
@pytest.mark.parametrize('signals', [1, 2, 3])
def test_unrelated_unexpected_exception_is_not_swallowed(tmp_path, command, phase, signals):
    child = run_fault(tmp_path, command, dict(error='unexpected', phase=phase, signals=signals))
    assert child.returncode == 87, child.stderr
    assert 'UNEXPECTED PRESERVED' in child.stderr
    assert 'Interrupted' not in child.stderr


@pytest.mark.parametrize('option', ['--in-place', '--no-backup'])
@pytest.mark.parametrize('command', [['normalise'], ['analyse'], ['verify'], ['tags', 'plan']])
@pytest.mark.parametrize('display', ['--help', '--version'])
def test_early_safety_refusal_does_not_install_controller_or_load_pipeline(monkeypatch, capsys, option, command, display):
    import signal
    from norma import cli

    def forbidden(*args, **kwargs):
        pytest.fail('early forbidden option reached startup or handler installation')
    old = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(cli, '_pipeline', forbidden)
    monkeypatch.setattr(cli, 'load_config', forbidden)
    monkeypatch.setattr(signal, 'signal', forbidden)
    assert cli.main([*command, option, display]) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and f'{option} is not available' in captured.err
    assert signal.getsignal(signal.SIGINT) is old
