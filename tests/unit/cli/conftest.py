"""Synthetic immutable records for projection/dispatch tests; not audio evidence."""
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from norma import pipeline
from norma.models import FileOutcome, InputItem, LoudnessMeasurement, RunOptions, RunReport


@pytest.fixture
def outcome():
    def factory(index=0, name='take.wav', **changes):
        path = Path.cwd() / name
        item = InputItem(path, path.parent, PurePosixPath(path.name), (1, index + 10), index, True)
        measurement = LoudnessMeasurement(signal_state='measurable', integrated_lufs=-23.0, true_peak_dbtp=-10.0,
            true_peak_linear=0.31622776601683794, lra_lu=2.0, threshold_lufs=None, integrated_valid=True,
            peak_valid=True, lra_valid=True, threshold_valid=False, integrated_resolution_lu=0.001,
            peak_resolution_linear=0.001, dual_mono=False, meter='ebur128', value_source='metadata',
            ffmpeg_version='synthetic', decoded_duration_s=20, decoded_frames=960000, sample_rate_hz=48000,
            decoded_nonzero=True, terminal_peak_covered=True)
        return replace(FileOutcome(status='ok', reason_code='ok', input=item, probe=None, target=None,
            measurement_in=measurement, measurement_out=None, gain=None, verification=None,
            verification_reference=None, source_codec='pcm_s24le', origin_codec=None, metadata=None,
            tag_write=None, output=None, conditional_plans=(), warnings=(), elapsed_s=0.25,
            ffmpeg_stderr_tail=None, execution='inspect', published_paths=(), retained_failed_path=None), **changes)
    return factory


@pytest.fixture
def options():
    return RunOptions('analyse', Path('/tool/ffmpeg'), Path.cwd(), value_sources={'target': 'config', 'true_peak': 'default', 'tolerance': 'default'})


@pytest.fixture
def envelope(options):
    def factory(outcomes=(), **changes):
        counts = {status: sum(o.status == status for o in outcomes) for status in ('ok', 'warn', 'skipped', 'failed')}
        return replace(RunReport('synthetic-run', {'norma': 'synthetic'}, options, tuple(outcomes), counts,
                                 0, 0, None, 0.5, ()), **changes)
    return factory
