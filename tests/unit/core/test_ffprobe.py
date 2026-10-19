"""ffprobe adapter: JSON → C-02 mapping and C-13 failure classes (fake ffprobe)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from norma.adapters import ffprobe as p
from norma.errors import EnvironmentError_, FileFailure, SkipFile


def stream(**changes) -> dict:
    base = {"index": 0, "codec_name": "pcm_s24le", "codec_type": "audio", "sample_fmt": "s32",
            "sample_rate": "48000", "channels": 2, "channel_layout": "stereo", "bits_per_sample": 24,
            "duration": "10.000000", "bit_rate": "2304000", "tags": {"title": "Take 3"}}
    base.update(changes)
    return base


def document(*streams: dict, fmt: dict | None = None, chapters: list | None = None) -> dict:
    return {"streams": list(streams), "chapters": chapters or [],
            "format": fmt or {"format_name": "wav", "duration": "10.000000", "bit_rate": "9999999",
                              "tags": {"encoder": "Lavf"}}}


def test_codec_classes_follow_formats_f1() -> None:
    for codec in ("pcm_s16le", "pcm_s24be", "pcm_f32le", "pcm_s32le", "pcm_f64be", "pcm_u8", "flac", "alac"):
        assert p.codec_class(codec) == "lossless", codec
    for codec in ("mp3", "aac"):
        assert p.codec_class(codec) == "lossy"
    for codec in ("pcm_alaw", "pcm_mulaw", "opus", "vorbis", "ac3", "pcm_dvd", ""):
        assert p.codec_class(codec) == "unsupported", codec


def test_canonical_container() -> None:
    assert p.canonical_container("mov,mp4,m4a,3gp,3g2,mj2") == "mp4"
    assert p.canonical_container("aac") == "adts"
    assert p.canonical_container("wav") == "wav"
    assert p.canonical_container("matroska,webm") == "matroska"


def test_parse_wav_probe() -> None:
    result = p.parse_probe(document(stream()), version="N-1")
    assert (result.codec, result.codec_class, result.container) == ("pcm_s24le", "lossless", "wav")
    assert (result.sample_rate_hz, result.bit_depth, result.channels, result.channel_layout) == (48000, 24, 2, "stereo")
    assert (result.duration_s, result.duration_basis, result.bitrate_bps) == (10.0, "stream", 2304000)
    assert result.ffprobe_tags == {"encoder": "Lavf", "stream:title": "Take 3"}
    assert result.ffprobe_version == "N-1" and not result.sample_float and not result.has_artwork


def test_first_audio_stream_selected_artwork_and_ignored_streams_counted() -> None:
    art = {"index": 0, "codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}}
    first = stream(index=1, codec_name="aac", sample_fmt="fltp", bits_per_sample=0, bit_rate="256000")
    second = stream(index=2, codec_name="aac", sample_fmt="fltp")
    result = p.parse_probe(document(art, first, second, fmt={"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
                                    chapters=[{"id": 0}, {"id": 1}]), version="N-1")
    assert result.audio_stream_index == 1 and result.ignored_streams == 1
    assert result.has_artwork and result.chapters == 2 and result.container == "mp4"
    assert result.codec_class == "lossy" and result.bit_depth is None and result.sample_float


def test_bitrate_never_taken_from_container_and_duration_fallback() -> None:
    result = p.parse_probe(document(stream(bit_rate=None, duration=None)), version="N-1")
    assert result.bitrate_bps is None
    assert (result.duration_s, result.duration_basis) == (10.0, "container")
    unknown = p.parse_probe(document(stream(duration="N/A"), fmt={"format_name": "flac"}), version="N-1")
    assert (unknown.duration_s, unknown.duration_basis) == (None, "unknown")


def test_layout_not_inferred_for_unspecified_multichannel() -> None:
    assert p.parse_probe(document(stream(channels=6, channel_layout="6 channels")), version="N").channel_layout is None
    assert p.parse_probe(document(stream(channels=6, channel_layout="5.1")), version="N").channel_layout == "5.1"
    assert p.parse_probe(document(stream(channel_layout=None)), version="N").channel_layout is None


def test_flac_depth_from_raw_sample_bits() -> None:
    result = p.parse_probe(document(stream(codec_name="flac", bits_per_sample=0, bits_per_raw_sample="24"),
                                    fmt={"format_name": "flac"}), version="N")
    assert result.bit_depth == 24


def test_unsupported_codec_is_still_representable() -> None:
    result = p.parse_probe(document(stream(codec_name="pcm_mulaw", bits_per_sample=8)), version="N")
    assert result.codec_class == "unsupported" and result.bit_depth is None


def test_no_audio_stream_is_not_audio_skip() -> None:
    video = {"index": 0, "codec_type": "video", "codec_name": "h264"}
    with pytest.raises(SkipFile) as caught:
        p.parse_probe(document(video), version="N")
    assert caught.value.reason == "not_audio"


@pytest.mark.parametrize("bad", [dict(sample_rate="0"), dict(channels=0), dict(codec_name=None), dict(index=None)])
def test_unusable_stream_is_probe_error(bad: dict) -> None:
    with pytest.raises(FileFailure) as caught:
        p.parse_probe(document(stream(**bad)), version="N")
    assert caught.value.reason == "probe_error"


def test_missing_sections_are_probe_error() -> None:
    with pytest.raises(FileFailure):
        p.parse_probe({"streams": []}, version="N")


FAKE = (
    "args = sys.argv[1:]\n"
    "if '-version' in args:\n"
    "    print('ffprobe version N-125892-g406c5a37aa-x Copyright'); sys.exit(0)\n"
    "assert args[args.index('-protocol_whitelist') + 1] == 'file', args\n"
    "src = args[args.index('-i') + 1]\n"
    "assert os.path.isabs(src), src\n"
    "mode = open(src, 'rb').read().decode()\n"
    "if mode == 'ok':\n"
    "    print(json.dumps(DOC))\n"
    "elif mode == 'fail':\n"
    "    print('Invalid data found when processing input', file=sys.stderr); sys.exit(1)\n"
    "elif mode == 'garbage':\n"
    "    print('{not json')\n"
    "elif mode == 'slow':\n"
    "    time.sleep(30)\n"
)


@pytest.fixture
def prober(make_tool) -> p.Prober:
    tool = make_tool("ffprobe", f"DOC = {json.dumps(document(stream()))!r}\nDOC = json.loads(DOC)\n{FAKE}")
    return p.locate_ffprobe(tool)


def source(tmp_path: Path, content: str) -> Path:
    path = tmp_path / f"{content}.wav"
    path.write_text(content, encoding="utf-8")
    return path


def test_locate_ffprobe(prober: p.Prober, tmp_path: Path) -> None:
    assert prober.version == "N-125892-g406c5a37aa-x"
    with pytest.raises(EnvironmentError_):
        p.locate_ffprobe("ffprobe", search_path=str(tmp_path / "nothing"))
    with pytest.raises(EnvironmentError_):
        p.locate_ffprobe(tmp_path / "missing" / "ffprobe")


def test_probe_argv_rules(prober: p.Prober) -> None:
    argv = p.probe_argv(prober, Path("/demo/a.wav"))
    assert argv[0] == str(prober.path) and argv[-2:] == ("-i", "/demo/a.wav")
    assert argv[argv.index("-protocol_whitelist") + 1] == "file"
    with pytest.raises(ValueError):
        p.probe_argv(prober, Path("a.wav"))


def test_probe_success(prober: p.Prober, tmp_path: Path) -> None:
    result = p.probe(prober, source(tmp_path, "ok"))
    assert result.codec == "pcm_s24le" and result.ffprobe_version == prober.version


@pytest.mark.parametrize("mode", ["fail", "garbage"])
def test_probe_failures_are_probe_error(prober: p.Prober, tmp_path: Path, mode: str) -> None:
    with pytest.raises(FileFailure) as caught:
        p.probe(prober, source(tmp_path, mode))
    assert caught.value.reason == "probe_error"


def test_probe_timeout(prober: p.Prober, tmp_path: Path) -> None:
    with pytest.raises(FileFailure) as caught:
        p.probe(prober, source(tmp_path, "slow"), deadline_s=0.5)
    assert caught.value.reason == "timeout"


def test_probe_source_errors(prober: p.Prober, tmp_path: Path) -> None:
    with pytest.raises(FileFailure) as caught:
        p.probe(prober, tmp_path / "vanished.wav")
    assert caught.value.reason == "source_changed"
    locked = source(tmp_path, "ok")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running with privileges that bypass file modes")
        with pytest.raises(FileFailure) as caught:
            p.probe(prober, locked)
        assert caught.value.reason == "permission_denied"
    finally:
        locked.chmod(0o600)


def test_probe_tool_lost_after_preflight_is_probe_error(prober: p.Prober, tmp_path: Path) -> None:
    changed = p.Prober(prober.path, (0, 0), prober.version)
    with pytest.raises(FileFailure) as caught:
        p.probe(changed, source(tmp_path, "ok"))
    assert caught.value.reason == "probe_error"
