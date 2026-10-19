"""CR-067: MP3 evidence is withheld when FFmpeg lacks the encoder statistics CR-063 reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from norma.adapters import ffmpeg as f
from norma.audio.render import MP3_ENCODER_STATS
from norma.formats.capabilities import ToolBasis, automatic_mp3_capability, lossy_capability

GOOD_STATS = b"pre 0 0 1/44100 1152\npre 1 1152 1/44100 1152\npost 0 0 1/44100\npost 1 1105 1/44100\n"


@pytest.fixture
def env() -> f.FFmpegEnvironment:
    return f.FFmpegEnvironment(f.Executable(Path("/fake/ffmpeg"), (1, 2), "N-test"),
                               frozenset({"libmp3lame"}), frozenset(), "test platform")


def install_fake(monkeypatch, *, code=0, stdout=GOOD_STATS, truncated=False):
    calls: list[tuple[str, ...]] = []

    def run(argv, **kwargs):
        calls.append(tuple(argv))
        return f.ProcessResult(tuple(argv), code, stdout, ("Unrecognized option",) if code else (), 0.01, truncated)

    monkeypatch.setattr(f, "run_process", run)
    return calls


def test_probe_options_are_the_renders_options() -> None:
    assert f.MP3_STATS_PROBE_ARGS == MP3_ENCODER_STATS


def test_probe_passes_the_render_options_and_accepts_well_formed_lines(env, monkeypatch) -> None:
    calls = install_fake(monkeypatch)
    assert f.probe_mp3_encoder_stats(env) is True
    (argv,) = calls
    start = argv.index("-stats_enc_pre")
    assert argv[start:start + len(MP3_ENCODER_STATS)] == MP3_ENCODER_STATS
    assert argv[argv.index("-c:a") + 1] == "libmp3lame" and argv[-3:] == ("-f", "null", "-")


@pytest.mark.parametrize("kwargs", [
    {"code": 8, "stdout": b""},                       # a build without the options: "Unrecognized option"
    {"stdout": b""},                                  # accepted but silent
    {"stdout": b"pre 0 0 1/44100 1152\n"},            # no post line
    {"stdout": b"post 0 0 1/44100\n"},                # no pre line
    {"stdout": b"pre x y\npost a\n"},                 # malformed lines
    {"truncated": True},
], ids=["unrecognised", "silent", "no-post", "no-pre", "malformed", "truncated"])
def test_probe_rejects_a_build_without_usable_statistics(env, monkeypatch, kwargs) -> None:
    install_fake(monkeypatch, **kwargs)
    assert f.probe_mp3_encoder_stats(env) is False


GOOD = ["pre 0 0 1/44100 1152", "pre 1 1152 1/44100 1152", "post 0 -1105 1/44100", "post 1 47 1/44100"]


@pytest.mark.parametrize("lines", [
    ["pre 0 bananas 1/44100 1152", "post 0 0 1/44100"],                      # non-numeric frame timestamp
    ["pre 0 0 1/44100 1152", "post 0 nonsense 1/44100"],                      # non-numeric packet timestamp
    ["pre 0 0 garbage 1152", "post 0 0 1/44100"],                             # malformed time base
    ["pre 0 0 1/48000 1152", "post 0 0 1/48000"],                             # wrong time base for 44.1 kHz
    ["pre 0 0 1/44100 1152", "post 0 0 1/44100", "post 1 x 1/44100"],         # later bad line
    ["pre 0 0 1/44100 1152", "pre 1 1152 bananas 1152", "post 0 0 1/44100"],  # later bad time base
    ["pre 0 0 1/44100 1152", "pre 1 999 1/44100 1152", "post 0 0 1/44100"],   # timestamp inconsistent with samples
    ["pre 0 0 1/44100 1152", "pre 2 1152 1/44100 1152", "post 0 0 1/44100"],  # frame numbers skip
    ["pre 0 0 1/44100 1152", "post 0 0 1/44100", "post 0 5 1/44100"],         # packet numbers repeat
    ["pre 0 0 1/44100 1152", "post 0 5 1/44100", "post 1 5 1/44100"],         # packet timestamps do not increase
    ["pre 0 0 1/44100 -1", "post 0 0 1/44100"],                               # negative sample count
    ["pre 0 0 1/44100 0", "post 0 0 1/44100"],                                # no samples at all
    ["pre 0 0 1/44100 1152 extra", "post 0 0 1/44100"],                       # extra field
    ["pre"],                                                                 # CR-072: bare keyword
    ["pre 0 0 1/44100 1152", "post"],                                        # CR-072: bare post
    ["pre 0 0 1/44100", "post 0 0 1/44100"],                                 # CR-072: pre without sample count
    ["pre 0 0 1/44100 1152", "post 0"],                                     # CR-072: post cut after the number
], ids=lambda lines: str(len(lines)) + ":" + lines[-1][:24])
def test_probe_rejects_malformed_or_inconsistent_lines(env, monkeypatch, lines) -> None:
    install_fake(monkeypatch, stdout=("\n".join(lines) + "\n").encode())
    assert f.probe_mp3_encoder_stats(env) is False


@pytest.mark.parametrize("stdout", [b"pre\n", b"pre 0 0 1/44100 1152\\npost 0\\n", b"\\n"],
                         ids=["bare-pre", "short-post", "empty"])
def test_probe_returns_false_for_truncated_records_without_raising(env, monkeypatch, stdout) -> None:
    """CR-072: record shape is checked before indexing; no IndexError escapes."""
    install_fake(monkeypatch, stdout=stdout)
    assert f.probe_mp3_encoder_stats(env) is False


def test_probe_accepts_consistent_lines_among_other_output(env, monkeypatch) -> None:
    install_fake(monkeypatch, stdout=("\n".join(["note", *GOOD, ""])).encode())
    assert f.probe_mp3_encoder_stats(env) is True


def test_without_the_options_mp3_evidence_is_unsupported_and_makes_no_encode(env, monkeypatch, tmp_path) -> None:
    calls = install_fake(monkeypatch, code=8, stdout=b"")
    monkeypatch.setattr(f, "_encode_outputs", lambda *a, cancel_event=None: pytest.fail("no evidence encode without statistics"))
    cbr, vbr = f.collect_mp3_evidence(env, 44100, 2, "stereo", tmp_path / "ev")
    assert len(calls) == 1  # the probe alone
    for record in (cbr, vbr):
        assert record.nominal_bitrates_bps == () and record.vbr_qualities == ()
        assert record.nominal_confirmation == "not_applicable" and "-stats_enc" in record.source_locator
        assert not record.supports_settings
    assert (cbr.encoder, cbr.bitrate_mode, vbr.bitrate_mode) == ("libmp3lame", "cbr", "vbr")
    assert not (tmp_path / "ev").exists()
    basis = ToolBasis(cbr.executable_path, cbr.executable_identity, cbr.ffmpeg_version, cbr.platform_basis)
    automatic = automatic_mp3_capability([cbr, vbr], basis, sample_rate_hz=44100, channels=2, channel_layout="stereo")
    assert not automatic.supported and automatic.gaps
    explicit = lossy_capability("mp3", "cbr", [cbr, vbr], basis, sample_rate_hz=44100, channels=2,
                                channel_layout="stereo")
    assert not explicit.supported


def test_v0_helper_is_unsupported_too(env, monkeypatch, tmp_path) -> None:
    install_fake(monkeypatch, code=8, stdout=b"")
    record = f.collect_mp3_v0_evidence(env, 48000, 2, "stereo", tmp_path)
    assert record.vbr_qualities == () and record.nominal_confirmation == "not_applicable"


def test_a_given_answer_is_not_probed_again(env, monkeypatch, tmp_path) -> None:
    calls = install_fake(monkeypatch, code=8, stdout=b"")
    f.collect_mp3_evidence(env, 44100, 2, "stereo", tmp_path, encoder_stats=False)
    assert calls == []


def test_a_supporting_build_still_collects(env, monkeypatch, tmp_path) -> None:
    install_fake(monkeypatch)
    seen = []

    def encode(env, source, outputs, *, cancel_event=None):
        seen.append(len(outputs))
        return {path: False for path, _ in outputs}, ()

    monkeypatch.setattr(f, "_encode_outputs", encode)
    f.collect_mp3_evidence(env, 44100, 2, "stereo", tmp_path)
    assert seen and seen[0] > 0


@pytest.mark.real_ffmpeg
def test_real_ffmpeg_supports_the_statistics_and_keeps_its_mp3_evidence(tmp_path) -> None:
    real = f.probe_environment(f.locate_ffmpeg("/usr/local/bin/ffmpeg"))
    if "libmp3lame" not in real.encoders:
        pytest.skip("build lacks libmp3lame")
    assert f.probe_mp3_encoder_stats(real) is True
    cbr, _ = f.collect_mp3_evidence(real, 44100, 2, "stereo", tmp_path)
    assert 128_000 in cbr.nominal_bitrates_bps, cbr.source_locator


@pytest.mark.real_ffmpeg
@pytest.mark.parametrize("stdout", [b"pre\\n", b"pre 0 0 1/44100 1152\\npost 0\\n"], ids=["bare-pre", "short-post"])
def test_doctor_shows_mp3_unavailable_for_truncated_probe_lines_and_exits_normally(
        monkeypatch, capsys, stdout) -> None:
    """CR-072: only the statistics probe's stdout is replaced; the public doctor must not crash."""
    import dataclasses

    from norma import cli

    real_run = f.run_process

    def run(argv, **kwargs):
        result = real_run(argv, **kwargs)
        return dataclasses.replace(result, stdout=stdout) if "-stats_enc_pre" in argv else result

    monkeypatch.setattr(f, "run_process", run)
    code = cli.main(["doctor", "--ffmpeg", "/usr/local/bin/ffmpeg", "--config", "/dev/null"])
    out = capsys.readouterr().out
    assert code in (0, 1), out  # a normal exit, not an uncaught exception
    mp3_rows = [line for line in out.splitlines() if "lossy evidence" in line and "libmp3lame" in line]
    assert mp3_rows and all(line.lstrip().startswith("WARN") for line in mp3_rows), out
