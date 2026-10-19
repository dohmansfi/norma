"""Real-tool tests against /usr/local/bin/ffmpeg and ffprobe (marker ``real_ffmpeg``).

Every generated file lives under tmp_path. These record behaviour of the
installed build; they are evidence foundation (TST-F-01), not acceptance of a
complete capability matrix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from norma.adapters import ffmpeg as f
from norma.adapters import ffprobe as p
from norma.errors import SkipFile

pytestmark = pytest.mark.real_ffmpeg

FFMPEG = Path("/usr/local/bin/ffmpeg")
FFPROBE = Path("/usr/local/bin/ffprobe")


@pytest.fixture(scope="module")
def env() -> f.FFmpegEnvironment:
    return f.probe_environment(f.locate_ffmpeg(FFMPEG))


@pytest.fixture(scope="module")
def prober() -> p.Prober:
    return p.locate_ffprobe(FFPROBE)


def generate(env: f.FFmpegEnvironment, out: Path, *encode: str, rate: int = 44100, layout: str = "stereo",
             duration: float = 0.5) -> Path:
    graph = f"sine=frequency=997:sample_rate={rate}:duration={duration},aformat=channel_layouts={layout}"
    argv = f.ffmpeg_argv(env.executable, "-loglevel", "error", "-protocol_whitelist", "file", "-f", "lavfi",
                         "-i", graph, *encode, "-y", str(out))
    f.check_result(f.run_process(argv, deadline_s=f.tool_deadline(duration), executable=env.executable))
    return out


def test_environment_has_every_planned_facility(env: f.FFmpegEnvironment) -> None:
    assert env.executable.version and env.executable.path == FFMPEG
    f.check_requirements(env, encoders=("flac", "alac", "libmp3lame", "aac_at", "pcm_s16le", "pcm_s24le",
                                        "pcm_f32le", "pcm_s16be", "pcm_s24be"),
                         filters=f.MEASURE_FILTERS + f.RENDER_FILTERS + f.DYNAMIC_FILTERS,
                         alimiter_options=f.ALIMITER_OPTIONS)


def test_mp3_evidence_44100_stereo(env: f.FFmpegEnvironment, tmp_path: Path) -> None:
    cbr, vbr = f.collect_mp3_evidence(env, 44100, 2, "stereo", tmp_path / "mp3")
    assert cbr.nominal_confirmation == "validated_setting"
    assert 128000 in cbr.nominal_bitrates_bps
    assert set(cbr.nominal_bitrates_bps) <= set(f.MP3_CBR_CANDIDATES_BPS)
    assert cbr.executable_identity == env.executable.identity and cbr.basis == "installed_query"
    # Either the quality is confirmed from written headers or the record is an explicit gap.
    assert (vbr.nominal_confirmation == "validated_setting") == bool(vbr.vbr_qualities)
    assert all(path.is_relative_to(tmp_path) for path in tmp_path.rglob("*"))


def test_mp3_clamped_request_is_not_evidenced(env: f.FFmpegEnvironment, tmp_path: Path) -> None:
    cbr, _ = f.collect_mp3_evidence(env, 44100, 2, "stereo", tmp_path, cbr_candidates_bps=(100000, 128000),
                                    vbr_candidates=())
    assert 100000 not in cbr.nominal_bitrates_bps and 128000 in cbr.nominal_bitrates_bps


def test_mp3_multichannel_downmix_is_not_evidenced(env: f.FFmpegEnvironment, tmp_path: Path) -> None:
    cbr, vbr = f.collect_mp3_evidence(env, 48000, 6, "5.1", tmp_path, cbr_candidates_bps=(128000,),
                                      vbr_candidates=(0,))
    assert cbr.nominal_bitrates_bps == () and cbr.nominal_confirmation == "not_applicable"
    assert vbr.vbr_qualities == ()


def test_lossless_evidence(env: f.FFmpegEnvironment, tmp_path: Path) -> None:
    flac = f.collect_lossless_evidence(env, "flac", 96000, 2, "stereo", tmp_path / "flac")
    assert flac.supported_bit_depths == (16, 24)
    alac = f.collect_lossless_evidence(env, "alac", 96000, 2, "stereo", tmp_path / "alac")
    assert alac.supported_bit_depths == (16, 24)
    wav = f.collect_lossless_evidence(env, "pcm_f32le", 48000, 2, "stereo", tmp_path / "wav")
    assert wav.supports_float
    assert f.revalidate_evidence(env, flac)


def test_probe_generated_files(env: f.FFmpegEnvironment, prober: p.Prober, tmp_path: Path) -> None:
    flac = p.probe(prober, generate(env, tmp_path / "a.flac", "-c:a", "flac", "-sample_fmt", "s32",
                                    "-bits_per_raw_sample", "24", rate=96000))
    assert (flac.codec, flac.codec_class, flac.container, flac.bit_depth, flac.sample_rate_hz) == (
        "flac", "lossless", "flac", 24, 96000)
    mp3 = p.probe(prober, generate(env, tmp_path / "a.mp3", "-c:a", "libmp3lame", "-b:a", "128k"))
    assert (mp3.codec_class, mp3.bit_depth, mp3.bitrate_bps) == ("lossy", None, 128000)
    m4a = p.probe(prober, generate(env, tmp_path / "a.m4a", "-c:a", "alac", layout="mono"))
    assert (m4a.container, m4a.codec, m4a.channels, m4a.channel_layout) == ("mp4", "alac", 1, "mono")
    mulaw = p.probe(prober, generate(env, tmp_path / "u.wav", "-c:a", "pcm_mulaw", rate=8000, layout="mono"))
    assert mulaw.codec_class == "unsupported"
    assert mulaw.ffprobe_version == prober.version


def test_hostile_names_are_single_argv_values(env: f.FFmpegEnvironment, prober: p.Prober, tmp_path: Path) -> None:
    folder = tmp_path / "-i $(touch pwned) ; `x`"
    folder.mkdir()
    out = generate(env, folder / "-y http:..evil.wav", "-c:a", "pcm_s16le")
    result = p.probe(prober, out)
    assert result.codec == "pcm_s16le"
    assert not (tmp_path / "pwned").exists() and not Path("pwned").exists()


def test_probe_video_only_is_not_audio(env: f.FFmpegEnvironment, prober: p.Prober, tmp_path: Path) -> None:
    out = tmp_path / "v.mp4"
    argv = f.ffmpeg_argv(env.executable, "-loglevel", "error", "-f", "lavfi", "-i",
                         "color=c=black:s=16x16:d=0.2", "-c:v", "mpeg4", "-y", str(out))
    f.check_result(f.run_process(argv, deadline_s=60, executable=env.executable))
    with pytest.raises(SkipFile) as caught:
        p.probe(prober, out)
    assert caught.value.reason == "not_audio"
