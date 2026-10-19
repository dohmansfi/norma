"""IC-3 F-2: the AAC collector asks low-rate questions and keeps only confirmed answers."""

from __future__ import annotations

import pytest

from norma.adapters import ffmpeg as f


def test_low_rate_questions_are_asked_but_not_answers():
    questions = f.AAC_CBR_CANDIDATES_BPS
    assert 48_000 in questions and 32_000 in questions
    assert list(questions) == sorted(set(questions))
    assert all(type(bps) is int and bps > 0 for bps in questions)
    assert {64_000, 128_000, 256_000, 320_000} <= set(questions)


@pytest.fixture
def env():
    env = f.probe_environment(f.locate_ffmpeg('/usr/local/bin/ffmpeg'))
    if 'aac_at' not in env.encoders:
        pytest.skip('build lacks aac_at')
    return env


@pytest.mark.real_ffmpeg
def test_real_stereo_8khz_confirms_48k(env, tmp_path):
    work = tmp_path / 'work'
    record = f.collect_aac_evidence(env, 8000, 2, 'stereo', work)
    assert 48_000 in record.nominal_bitrates_bps, record.source_locator
    assert record.nominal_confirmation == 'encoder_diagnostic'
    assert record.basis == 'installed_query'
    assert 64_000 not in record.nominal_bitrates_bps
    assert max(record.nominal_bitrates_bps) == 48_000
    assert f.revalidate_evidence(env, record)
    assert list(work.iterdir()) == []


@pytest.mark.real_ffmpeg
@pytest.mark.parametrize('rate', [44100, 48000])
def test_real_existing_rates_keep_their_stereo_settings(env, tmp_path, rate):
    record = f.collect_aac_evidence(env, rate, 2, 'stereo', tmp_path)
    expected = (64, 72, 80, 96, 112, 128, 144, 160, 192, 224, 256, 288, 320)
    assert record.nominal_bitrates_bps == tuple(k * 1000 for k in expected), record.source_locator
    assert record.nominal_confirmation == 'encoder_diagnostic'


@pytest.mark.real_ffmpeg
def test_real_unsupported_low_rate_is_not_retained(env, tmp_path):
    record = f.collect_aac_evidence(env, 8000, 2, 'stereo', tmp_path, cbr_candidates_bps=(64_000,))
    assert record.nominal_bitrates_bps == ()
    assert record.nominal_confirmation == 'not_applicable'
