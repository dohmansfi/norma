"""TST-A-01/02/03: recorded frame streams and fail-closed parser behaviour."""

import math
from pathlib import Path

import pytest

from norma.adapters import ffmpeg
from norma.audio.measure import MeterParser, conservative_peak_dbtp, measurement_argv
from norma.errors import FileFailure


def record(index=0, integrated="-23.012", lra="1.234", peaks=("0.100", "0.200")):
    # Actual ametadata stderr framing inspected on N-125892, 2026-10-09.
    prefix = "[Parsed_ametadata_1 @ 0x6000031bc160] "
    return [prefix + f"frame:{index} pts:{index * 4800} pts_time:{index / 10}",
            prefix + f"lavfi.r128.I={integrated}", prefix + f"lavfi.r128.LRA={lra}",
            *[prefix + f"lavfi.r128.true_peaks_ch{n}={value}" for n, value in enumerate(peaks)],
            prefix + "lavfi.r128.true_peak=999.000"]


def parse(lines, channels=2):
    parser = MeterParser(channels)
    for line in lines:
        parser.feed(line)
    parser.finish()
    return parser


def test_last_valid_gated_i_lra_and_max_every_channel_frame():
    parser = parse(record(peaks=("0.800", "0.100"))
                   + record(1, "-23.987", "2.123", ("0.200", "0.900"))
                   + record(2, "-70.000", "3.456", ("0.100", "0.200"))
                   + ["Summary:", "I: -5.0 LUFS", "Peak: 9.0 dBFS"])
    assert parser.integrated_lufs == -23.987
    assert parser.lra_lu == 3.456
    assert parser.peak == .9
    assert parser.records == 3


@pytest.mark.parametrize("integrated", ["-70.000", "-inf"])
def test_unavailable_integrated_is_not_zero(integrated):
    assert parse(record(integrated=integrated)).integrated_lufs is None


@pytest.mark.parametrize("field,value", [
    ("I", "nan"), ("I", "inf"), ("I", "bad"), ("LRA", "nan"),
    ("LRA", "-1"), ("true_peaks_ch0", "nan"), ("true_peaks_ch0", "inf"),
    ("true_peaks_ch0", "-0.1"),
])
def test_invalid_fields_fail(field, value):
    lines = [line.split("=", 1)[0] + "=" + value if f"lavfi.r128.{field}=" in line else line
             for line in record()]
    with pytest.raises(FileFailure, match="measurement_error"):
        parse(lines)


@pytest.mark.parametrize("field", ["I", "LRA", "true_peaks_ch1"])
def test_missing_fields_fail(field):
    with pytest.raises(FileFailure, match="measurement_error"):
        parse([line for line in record() if f"lavfi.r128.{field}=" not in line])


@pytest.mark.parametrize("lines", [
    ["lavfi.r128.I=-23"], record() + ["lavfi.r128.I=-24"],
    record() + record(2), ["frame:bad"], record(peaks=(".1", ".2", ".3")),
])
def test_broken_record_sequence_fails(lines):
    with pytest.raises(FileFailure, match="measurement_error"):
        parse(lines)


def test_empty_stream_never_uses_summary():
    parser = parse(["I: -23.0 LUFS", "Peak: -1.0 dBFS"])
    assert parser.records == 0 and parser.peak is None


def test_m1_argv_and_m6_no_transform_before_meter():
    executable = ffmpeg.Executable(Path("/ffmpeg"), (1, 2), "test")
    path = Path("/a name;$(no).wav")
    argv = measurement_argv(executable, path, dual_mono=True)
    assert argv == ("/ffmpeg", "-nostdin", "-hide_banner", "-xerror", "-nostats", "-protocol_whitelist",
                    "file", "-i", str(path), "-map", "0:a:0", "-af",
                    "ebur128=peak=true:metadata=1:dualmono=true:panlaw=-3.0103,ametadata=mode=print",
                    "-f", "null", "-")
    assert measurement_argv(executable, path, dual_mono=False, padded=True)[12].startswith("apad=pad_dur=0.1,")
    with pytest.raises(ValueError):
        measurement_argv(executable, Path("relative.wav"), dual_mono=False)


@pytest.mark.parametrize("resolution", [None, 0, -.001, math.nan, math.inf])
def test_unknown_resolution_cannot_establish_safety(resolution):
    assert conservative_peak_dbtp(.716, resolution) is None


def test_m7_quantiser_bin_and_zero_decisions():
    assert 20 * math.log10(.716) < -2.9 < conservative_peak_dbtp(.716, .001)
    assert conservative_peak_dbtp(0, .001) == 20 * math.log10(.0005)
    assert conservative_peak_dbtp(-1, .001) is None
