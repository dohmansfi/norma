"""TK-0435: evidence cancellation. Stubs are synthetic unless explicitly labelled native."""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path, PurePosixPath

import pytest

from norma import pipeline
from norma.adapters import ffmpeg as f
from norma.adapters.ffprobe import Prober
from norma.errors import Cancelled, EnvironmentError_, SkipFile, ToolMissing
from norma.models import InputItem, ProbeResult, RunOptions


@pytest.fixture
def env():
    return f.FFmpegEnvironment(f.Executable(Path('/fake/ffmpeg'), (1, 2), 'synthetic'),
                               frozenset({'pcm_s16le', 'flac', 'aac_at', 'libmp3lame'}),
                               frozenset(), 'synthetic')


def probe():
    return ProbeResult('pcm_s16le', 'lossless', 'wav', 0, 0, 48000, 16, False, 's16',
                       2, 'stereo', 1.0, 'stream', 1536000, False, 0, {}, 'synthetic')


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


@pytest.mark.parametrize('dry_run', [True, False])
@pytest.mark.parametrize('forced', [True, False])
def test_planning_evidence_interruption_preserves_settled_records_and_writes_nothing(
        tmp_path, monkeypatch, env, dry_run, forced):
    sources = [tmp_path / name for name in ('settled.wav', 'pending.wav')]
    for source in sources:
        source.write_bytes(b'synthetic source, never decoded')
    before = {p: fingerprint(p) for p in sources}
    entries = set(tmp_path.rglob('*'))
    items = tuple(InputItem(p, tmp_path, PurePosixPath(p.name), f.executable_identity(p), i, True)
                  for i, p in enumerate(sources))
    event = threading.Event()
    options = RunOptions(command='normalise', ffmpeg=env.executable.path, cwd=tmp_path,
                         dry_run=dry_run, inputs=tuple(sources))
    tools = pipeline.NormaliseTools(Prober(Path('/fake/ffprobe'), (3, 4), 'synthetic'), env)
    monkeypatch.setattr(pipeline, 'classify', lambda *a, **kw: probe())
    settled = []

    def read_tags(self, job):
        if job.item.input_index == 0:
            raise SkipFile('already_processed', 'tags_read', 'synthetic settled skip')

    original_records = pipeline._NormaliseRun._records

    def records(self):
        result = original_records(self)
        settled.extend(result)
        return result

    def evidence(*args, **kwargs):
        if not forced:
            assert kwargs['cancel_event'] is event
            event.set()
            return ()
        raise Cancelled()

    def forbidden(*args, **kwargs):
        pytest.fail('cancelled planning reached measurement, write barrier or execution')

    monkeypatch.setattr(pipeline._NormaliseRun, '_read_tags', read_tags)
    monkeypatch.setattr(pipeline._NormaliseRun, '_records', records)
    monkeypatch.setattr(pipeline, 'collect_evidence', evidence)
    monkeypatch.setattr(pipeline, 'preflight_targets', forbidden)
    monkeypatch.setattr(pipeline, 'measure', forbidden)
    monkeypatch.setattr(pipeline._NormaliseRun, '_preflight', forbidden)
    result = pipeline.normalise_items(items, options, tools, cancel_event=event, execute=forbidden)
    assert result.cancelled and result.dry_run is dry_run
    assert result.not_started == 1 and len(result.records) == 1
    assert result.records[0] is settled[0]
    outcome = result.records[0].outcome
    assert outcome.reason_code == 'already_processed' and outcome.published_paths == ()
    assert outcome.execution == ('plan' if dry_run else 'not_written')
    assert result.outputs is None and result.write_set is None and result.run_errors == ()
    assert {p: fingerprint(p) for p in sources} == before
    assert set(tmp_path.rglob('*')) == entries


def test_aac_cancellation_in_nominal_query_admits_no_encode(tmp_path, monkeypatch, env):
    event = threading.Event()

    def nominal(*args, **kwargs):
        assert kwargs['cancel_event'] is event
        event.set()
        return (128000, 256000)

    monkeypatch.setattr(f, '_aac_nominal_settings', nominal)
    monkeypatch.setattr(f, 'run_process', lambda *a, **kw: pytest.fail('encode after cancellation'))
    work = tmp_path / 'evidence'
    with pytest.raises(Cancelled):
        f.collect_aac_evidence(env, 48000, 2, 'stereo', work, cancel_event=event)
    assert not work.exists()


@pytest.mark.parametrize('entry', ['pipeline', 'aac', 'mp3', 'v0', 'lossless', 'stats', 'encode', 'decode'])
def test_already_set_event_admits_no_query_or_directory(tmp_path, monkeypatch, env, entry):
    event = threading.Event()
    event.set()
    work = tmp_path / 'not-created'
    monkeypatch.setattr(f, 'run_process', lambda *a, **kw: pytest.fail('process admitted'))
    monkeypatch.setattr(f, '_aac_nominal_settings', lambda *a, **kw: pytest.fail('query admitted'))
    calls = {
        'pipeline': lambda: pipeline.collect_evidence(env, [probe()], RunOptions(
            command='normalise', ffmpeg=env.executable.path, cwd=tmp_path), cancel_event=event),
        'aac': lambda: f.collect_aac_evidence(env, 48000, 2, 'stereo', work, cancel_event=event),
        'mp3': lambda: f.collect_mp3_evidence(env, 48000, 2, 'stereo', work, cancel_event=event),
        'v0': lambda: f.collect_mp3_v0_evidence(env, 48000, 2, 'stereo', work, cancel_event=event),
        'lossless': lambda: f.collect_lossless_evidence(env, 'flac', 48000, 2, 'stereo', work,
                                                       cancel_event=event),
        'stats': lambda: f.probe_mp3_encoder_stats(env, cancel_event=event),
        'encode': lambda: f._encode_outputs(env, (), [(work / 'x.flac', ())], cancel_event=event),
        'decode': lambda: f._decoded_evidence_ok(env, work / 'x.m4a', 48000, 2, cancel_event=event),
    }
    with pytest.raises(Cancelled):
        calls[entry]()
    assert not work.exists()


@pytest.mark.parametrize('entry', ['lossless', 'mp3', 'stats', 'encode'])
def test_event_inside_encode_stops_fallback_and_readback(tmp_path, monkeypatch, env, entry):
    event = threading.Event()
    calls = []

    def process(argv, **kwargs):
        calls.append(tuple(argv))
        assert kwargs['cancel_event'] is event
        event.set()
        return f.ProcessResult(tuple(argv), 1, b'', (), .01, False)

    monkeypatch.setattr(f, 'run_process', process)
    monkeypatch.setattr(f, '_read_pcm_header', lambda *a: pytest.fail('header read after cancel'))
    monkeypatch.setattr(f, 'read_mp3_frames', lambda *a: pytest.fail('frame read after cancel'))
    calls_by_entry = {
        'lossless': lambda: f.collect_lossless_evidence(env, 'flac', 48000, 2, 'stereo', tmp_path,
                                                       cancel_event=event),
        'mp3': lambda: f.collect_mp3_evidence(env, 48000, 2, 'stereo', tmp_path,
                                             encoder_stats=True, cancel_event=event),
        'stats': lambda: f.probe_mp3_encoder_stats(env, cancel_event=event),
        'encode': lambda: f._encode_outputs(env, (), [(tmp_path / 'x.flac', ()),
                                                     (tmp_path / 'y.flac', ())], cancel_event=event),
    }
    with pytest.raises(Cancelled):
        calls_by_entry[entry]()
    assert len(calls) == 1


@pytest.mark.parametrize('phase', ['encode', 'packets', 'packet_parse', 'decode', 'decode_parse'])
def test_aac_event_stops_each_readback_and_next_bitrate(tmp_path, monkeypatch, env, phase):
    event = threading.Event()
    calls = []
    monkeypatch.setattr(f, '_aac_nominal_settings', lambda *a, **kw: (128000, 256000))

    def process(argv, **kwargs):
        assert kwargs['cancel_event'] is event
        stage = 'encode' if '-b:a' in argv else ('packets' if 'adts' in argv else 'decode')
        calls.append(stage)
        assert calls == ['encode', 'packets', 'decode'][:len(calls)]
        if stage == 'encode':
            Path(argv[-1]).write_bytes(b'synthetic encode')
        if stage == phase:
            event.set()
        return f.ProcessResult(tuple(argv), 0, b'synthetic bytes parsed by stubs', (), .01, False)

    def packets(*args):
        if phase == 'packet_parse':
            event.set()

    def wav(*args):
        if phase == 'decode_parse':
            event.set()
        return f.PcmHeader(2, 48000, 16, False)

    monkeypatch.setattr(f, 'run_process', process)
    monkeypatch.setattr(f, '_check_adts', packets)
    monkeypatch.setattr(f, 'read_wav_header', wav)
    with pytest.raises(Cancelled):
        f.collect_aac_evidence(env, 48000, 2, 'stereo', tmp_path, cancel_event=event)
    assert calls == (['encode'] if phase == 'encode' else
                     ['encode', 'packets'] if phase in ('packets', 'packet_parse') else
                     ['encode', 'packets', 'decode'])
    assert not list(tmp_path.glob('norma-*'))


@pytest.mark.parametrize('phase', ['lossless', 'stats', 'mp3', 'aac'])
@pytest.mark.parametrize('forced', [False, True])
def test_pipeline_stops_between_collectors_and_removes_evidence_scratch(
        tmp_path, monkeypatch, env, phase, forced):
    event = threading.Event()
    called = []
    paths = []
    monkeypatch.setattr(pipeline, '_candidate_formats', lambda *a: {'wav', 'mp3', 'aac'})

    def query(name, args, kwargs):
        called.append(name)
        assert kwargs['cancel_event'] is event
        if name != 'stats':
            work = args[-1]
            work.mkdir(parents=True)
            path = work / 'synthetic-partial'
            path.write_bytes(b'partial evidence')
            paths.append(path)
        if name == phase:
            if forced:
                raise Cancelled()
            event.set()

    def lossless(*args, **kwargs):
        query('lossless', args, kwargs)
        return object()  # never resolved or presented as successful capability evidence

    def stats(*args, **kwargs):
        query('stats', args, kwargs)
        return True

    def mp3(*args, **kwargs):
        query('mp3', args, kwargs)
        assert kwargs['encoder_stats'] is True
        return ()

    def aac(*args, **kwargs):
        query('aac', args, kwargs)
        return object()

    monkeypatch.setattr(pipeline, 'collect_lossless_evidence', lossless)
    monkeypatch.setattr(pipeline, 'probe_mp3_encoder_stats', stats)
    monkeypatch.setattr(pipeline, 'collect_mp3_evidence', mp3)
    monkeypatch.setattr(pipeline, 'collect_aac_evidence', aac)
    options = RunOptions(command='normalise', ffmpeg=env.executable.path, cwd=tmp_path)
    with pytest.raises(Cancelled):
        pipeline.collect_evidence(env, [probe()], options, cancel_event=event)
    order = ['lossless', 'stats', 'mp3', 'aac']
    assert called == order[:order.index(phase) + 1]
    assert all(not path.exists() and not path.parent.parent.exists() for path in paths)
    assert not list(tmp_path.rglob('norma-evidence-*'))


@pytest.mark.parametrize('error', [ToolMissing('synthetic missing'), OSError('synthetic I/O'),
                                   ValueError('synthetic invalid')])
@pytest.mark.parametrize('set_event', [False, True])
def test_genuine_environment_errors_are_not_masked_by_cancellation(
        tmp_path, monkeypatch, env, error, set_event):
    event = threading.Event()

    def fail(*args, **kwargs):
        if set_event:
            event.set()
        raise error

    monkeypatch.setattr(pipeline, 'collect_lossless_evidence', fail)
    options = RunOptions(command='normalise', ffmpeg=env.executable.path, cwd=tmp_path)
    with pytest.raises(EnvironmentError_, match=str(error)) as caught:
        pipeline.collect_evidence(env, [probe()], options, cancel_event=event)
    assert caught.value.exit_code == 3
    assert not list(tmp_path.rglob('norma-evidence-*'))


@pytest.mark.parametrize('entry', ['lossless', 'mp3', 'stats'])
def test_cancellation_during_local_parsing_never_returns_evidence(tmp_path, monkeypatch, env, entry):
    event = threading.Event()
    parsed = []

    def encode(environment, source, outputs, **kwargs):
        assert kwargs['cancel_event'] is event
        for path, _ in outputs:
            path.write_bytes(b'synthetic headers')
        return {p: True for p, _ in outputs}, ()

    def header(*args):
        parsed.append(entry)
        event.set()
        return f.PcmHeader(2, 48000, 16, False)

    def frames(*args):
        parsed.append(entry)
        event.set()
        return (f.Mp3Frame(0, '1', 128000, 48000, 0, 384),)

    def stats(*args):
        parsed.append(entry)
        event.set()
        return True

    monkeypatch.setattr(f, '_encode_outputs', encode)
    monkeypatch.setattr(f, '_read_pcm_header', header)
    monkeypatch.setattr(f, 'read_mp3_frames', frames)
    monkeypatch.setattr(f, '_usable_mp3_stats', stats)
    monkeypatch.setattr(f, 'run_process', lambda argv, **kw: f.ProcessResult(tuple(argv), 0, b'pre', (), .01, False))
    with pytest.raises(Cancelled):
        if entry == 'lossless':
            f.collect_lossless_evidence(env, 'flac', 48000, 2, 'stereo', tmp_path, cancel_event=event)
        elif entry == 'mp3':
            f.collect_mp3_evidence(env, 48000, 2, 'stereo', tmp_path, encoder_stats=True, cancel_event=event)
        else:
            f.probe_mp3_encoder_stats(env, cancel_event=event)
    assert parsed == [entry]


def test_encode_fallback_stops_after_cancelled_single(tmp_path, monkeypatch, env):
    event = threading.Event()
    calls = []

    def process(argv, **kwargs):
        assert kwargs['cancel_event'] is event
        calls.append(tuple(argv))
        if len(calls) == 2:
            event.set()
        return f.ProcessResult(tuple(argv), 1, b'', (), .01, False)

    monkeypatch.setattr(f, 'run_process', process)
    with pytest.raises(Cancelled):
        f._encode_outputs(env, (), [(tmp_path / 'a.flac', ()), (tmp_path / 'b.flac', ())],
                          cancel_event=event)
    assert len(calls) == 2  # failed batch, first fallback, never the second fallback


@pytest.mark.parametrize('wrapper', [False, True])
def test_mp3_stats_cancellation_never_reaches_encode(tmp_path, monkeypatch, env, wrapper):
    event = threading.Event()

    def stats(environment, *, cancel_event=None):
        assert cancel_event is event
        event.set()
        return False  # must not turn interruption into an unsupported-target evidence gap

    monkeypatch.setattr(f, 'probe_mp3_encoder_stats', stats)
    monkeypatch.setattr(f, '_encode_outputs', lambda *a, **kw: pytest.fail('encode after stats cancelled'))
    with pytest.raises(Cancelled):
        collector = f.collect_mp3_v0_evidence if wrapper else f.collect_mp3_evidence
        collector(env, 48000, 2, 'stereo', tmp_path, cancel_event=event)
    assert not list(tmp_path.glob('norma-*'))


@pytest.mark.parametrize('matched', [False, True])
def test_mp3_v0_decode_receives_event_and_cleans_private_scratch(tmp_path, monkeypatch, env, matched):
    """Synthetic frame/Xing/setting proof; real collector and decode helper call graph."""
    event = threading.Event()
    paths = []

    def encode(environment, source, outputs, **kwargs):
        assert kwargs['cancel_event'] is event
        for path, _ in outputs:
            path.write_bytes(b'synthetic mp3 headers')
            paths.append(path)
        return {p: True for p, _ in outputs}, ()

    def decode(environment, path, rate, channels, *, cancel_event=None):
        assert cancel_event is event and path in paths
        raise Cancelled()

    monkeypatch.setattr(f, 'probe_mp3_encoder_stats', lambda *a, **kw: True)
    monkeypatch.setattr(f, '_encode_outputs', encode)
    monkeypatch.setattr(f, 'read_mp3_frames', lambda *a: (f.Mp3Frame(0, '1', 128000, 48000, 0, 384),))
    monkeypatch.setattr(f, 'read_xing_tag', lambda *a: f.XingTag(b'Xing', 0 if matched else 100, None))
    monkeypatch.setattr(f, '_matched_lossy_build', lambda *a: matched)
    monkeypatch.setattr(f, '_decoded_evidence_ok', decode)
    with pytest.raises(Cancelled):
        f.collect_mp3_v0_evidence(env, 48000, 2, 'stereo', tmp_path, cancel_event=event)
    assert paths and all(not p.exists() and not p.parent.exists() for p in paths)


@pytest.mark.parametrize('error', [RuntimeError('unrelated bug'), EnvironmentError_('genuine environment')])
def test_planning_does_not_swallow_unrelated_exception_when_event_is_set(tmp_path, monkeypatch, env, error):
    event = threading.Event()
    options = RunOptions(command='normalise', ffmpeg=env.executable.path, cwd=tmp_path, dry_run=True)
    tools = pipeline.NormaliseTools(Prober(Path('/fake/ffprobe'), (3, 4), 'synthetic'), env)

    def fail(*args, **kwargs):
        event.set()
        raise error

    monkeypatch.setattr(pipeline._NormaliseRun, '_probe_and_read_tags', lambda self: [])
    monkeypatch.setattr(pipeline, 'collect_evidence', fail)
    with pytest.raises(type(error)) as caught:
        pipeline.normalise_items((), options, tools, cancel_event=event)
    assert caught.value is error


@pytest.mark.parametrize('cancel_at', ['new', 'mode_set', 'mode_get', 'bitrate_set', 'bitrate_get'])
@pytest.mark.parametrize('failed', [False, True])
def test_audiotoolbox_event_stops_queries_but_always_disposes(monkeypatch, cancel_at, failed):
    """Synthetic ctypes calls, not a claim about installed AudioToolbox support."""
    from types import SimpleNamespace

    event = threading.Event()
    calls = []
    disposed = []
    nominal = [0]

    def visit(name):
        calls.append(name)
        if name == cancel_at:
            event.set()
            return int(failed)
        return 0

    def new(source, target, converter):
        converter._obj.value = 123
        return visit('new')

    def set_property(converter, prop, size, value):
        if prop == int.from_bytes(b'brat', 'big'):
            nominal[0] = value._obj.value
            return visit('bitrate_set')
        return visit('mode_set')

    def get_property(converter, prop, size, value):
        bitrate = prop == int.from_bytes(b'brat', 'big')
        value._obj.value = nominal[0] if bitrate else 0
        return visit('bitrate_get' if bitrate else 'mode_get')

    class Function:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    lib = SimpleNamespace(AudioConverterNew=Function(new),
                          AudioConverterSetProperty=Function(set_property),
                          AudioConverterGetProperty=Function(get_property),
                          AudioConverterDispose=Function(lambda c: disposed.append(c.value)))
    monkeypatch.setattr(f.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(f.ctypes, 'CDLL', lambda path: lib)
    with pytest.raises(Cancelled):
        f._aac_nominal_settings(48000, 2, (128000, 256000), cancel_event=event)
    order = ['new', 'mode_set', 'mode_get', 'bitrate_set', 'bitrate_get']
    assert calls == order[:order.index(cancel_at) + 1]
    assert disposed == [123]


def wait_for(predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.01)
    return predicate()


@pytest.mark.real_ffmpeg
@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('forced', [False, True])
def test_native_owned_cli_first_and_forced_signals_during_evidence(
        tmp_path, make_tool, dry_run, forced):
    """Real CLI/probe/runner, synthetic blocked evidence encode; never signals pytest."""
    real_ffmpeg = shutil.which('ffmpeg')
    real_ffprobe = shutil.which('ffprobe')
    assert real_ffmpeg and real_ffprobe
    source = tmp_path / 'source.wav'
    with wave.open(str(source), 'wb') as output:
        output.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
        output.writeframes(b'\0' * (48000 * 4))
    before = fingerprint(source)
    ready = tmp_path / 'ready'
    terminated = tmp_path / 'terminated'
    scratch = tmp_path / 'evidence-scratch'
    scratch.mkdir()
    shim = make_tool('ffmpeg', f'''
import signal
from pathlib import Path
if '-y' in sys.argv:
    def terminate(signum, frame):
        Path({str(terminated)!r}).write_text('SIGTERM received')
        if not {forced!r}:
            raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    Path({str(ready)!r}).write_text(str(os.getpid()))
    while True:
        time.sleep(.01)
os.execv({real_ffmpeg!r}, [{real_ffmpeg!r}, *sys.argv[1:]])
''')
    make_tool('ffprobe', f'os.execv({real_ffprobe!r}, [{real_ffprobe!r}, *sys.argv[1:]])')
    environment = dict(os.environ, PATH=f'{shim.parent}{os.pathsep}{os.environ["PATH"]}',
                       TMPDIR=str(scratch), NO_COLOR='1')
    command = [sys.executable, '-m', 'norma', 'normalise', str(source), '-o', str(tmp_path / 'out'),
               '-j', '1', '--log', str(tmp_path / 'run.log')]
    if dry_run:
        command.append('--dry-run')
    entries = set(tmp_path.rglob('*'))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               cwd=tmp_path, env=environment)
    child_pid = None
    try:
        assert wait_for(lambda: (ready.exists() and ready.read_text().strip().isdigit())
                        or process.poll() is not None)
        assert process.poll() is None
        child_pid = int(ready.read_text())
        process.send_signal(signal.SIGINT)
        assert wait_for(terminated.exists), 'first Event never terminated the active evidence process'
        if forced:
            assert process.poll() is None
            process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, (stdout, stderr)
        assert 'Traceback' not in stderr
        assert stdout.count('SUMMARY') == 1 and 'not_started=1' in stdout
        assert not (tmp_path / 'out').exists()
        assert not list(scratch.iterdir())
        assert fingerprint(source) == before
        assert not list(tmp_path.rglob('.norma-*'))
        assert set(tmp_path.rglob('*')) == entries | {ready, terminated}
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
