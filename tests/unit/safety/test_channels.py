"""ADR-0020 (D-056): discovery skips inputs with more than two channels (TK-0505, stage S1).

Files are hand-written in ``tmp_path`` (WAV headers carry the channel count) with a scripted
ffprobe; one test also runs the installed ffprobe when present. No fixture catalogue is used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from norma.adapters.ffprobe import locate_ffprobe
from norma.discovery import classify, discover
from norma.errors import SkipFile

REAL_FFPROBE = Path("/usr/local/bin/ffprobe")

LAYOUTS = [(3, "3.0"), (4, "quad"), (6, "5.1"), (8, "7.1")]


def item_for(path: Path):
    (item,) = discover([path]).items
    return item


@pytest.mark.parametrize(("channels", "layout"), LAYOUTS)
def test_more_than_two_channels_is_skipped_unsupported_channels(
    tmp_path: Path, build, fake_probe, channels: int, layout: str
) -> None:
    path = tmp_path / "multi.wav"
    path.write_bytes(build.wav(channels=channels))
    fake_probe.answer("multi.wav", build.document(build.stream(channels=channels, channel_layout=layout)))
    with pytest.raises(SkipFile) as caught:
        classify(item_for(path), fake_probe.prober)
    assert caught.value.reason == "unsupported_channels" and caught.value.stage == "probe"
    assert str(channels) in caught.value.detail


@pytest.mark.parametrize(("channels", "layout"), [(1, "mono"), (2, "stereo")])
def test_mono_and_stereo_still_pass(tmp_path: Path, build, fake_probe, channels: int, layout: str) -> None:
    path = tmp_path / "ok.wav"
    path.write_bytes(build.wav(channels=channels))
    fake_probe.answer("ok.wav", build.document(build.stream(channels=channels, channel_layout=layout)))
    assert classify(item_for(path), fake_probe.prober).channels == channels


def test_skip_is_decided_before_the_mark_test_and_force_does_not_bypass_it(
    tmp_path: Path, build, fake_probe
) -> None:
    path = tmp_path / "surround.wav"
    path.write_bytes(build.wav(channels=6))
    fake_probe.answer("surround.wav", build.document(build.stream(channels=6, channel_layout="5.1")))
    calls: list[object] = []
    for force in (False, True):
        with pytest.raises(SkipFile) as caught:
            classify(item_for(path), fake_probe.prober, force=force,
                     is_processed=lambda item: calls.append(item) or True)
        assert caught.value.reason == "unsupported_channels"
    assert not calls


def test_unsupported_codec_keeps_precedence(tmp_path: Path, build, fake_probe) -> None:
    path = tmp_path / "odd.m4a"
    path.write_bytes(build.mp4)
    fake_probe.answer("odd.m4a", build.document(
        build.stream(codec_name="pcm_alaw", sample_fmt="s16", channels=6, channel_layout="5.1"),
        format_name="mov,mp4,m4a,3gp,3g2,mj2"))
    with pytest.raises(SkipFile) as caught:
        classify(item_for(path), fake_probe.prober)
    assert caught.value.reason == "unsupported_codec"


def test_the_skip_changes_nothing_on_disk(tmp_path: Path, build, fake_probe) -> None:
    path = tmp_path / "surround.wav"
    data = build.wav(channels=3)
    path.write_bytes(data)
    fake_probe.answer("surround.wav", build.document(build.stream(channels=3, channel_layout="3.0")))
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(SkipFile):
        classify(item_for(path), fake_probe.prober)
    assert path.read_bytes() == data and sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.skipif(not REAL_FFPROBE.is_file(), reason="installed ffprobe not found")
def test_real_ffprobe_sees_three_and_six_channels(tmp_path: Path, build) -> None:
    prober = locate_ffprobe(REAL_FFPROBE)
    for channels in (3, 6):
        path = tmp_path / f"c{channels}.wav"
        path.write_bytes(build.wav(frames=4410, channels=channels))
        with pytest.raises(SkipFile) as caught:
            classify(item_for(path), prober)
        assert caught.value.reason == "unsupported_channels"
    for channels in (1, 2):
        path = tmp_path / f"c{channels}.wav"
        path.write_bytes(build.wav(frames=4410, channels=channels))
        assert classify(item_for(path), prober).channels == channels
