"""Static format capabilities and their intersection with C-04 evidence (TST-F-01, TST-C-01 data).

Every evidence record here is a fake, injected immutable record built for the
unit decision (test plan); none is a measured capability. The
AAC choices used below are illustrative synthetic values, not a claim about
any encoder.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest

import norma.formats.capabilities as cap
from norma.adapters.ffprobe import codec_class
from norma.models import LOSSLESS_FORMATS, CapabilityEvidence, FrozenMap, TargetFormatName

BASIS = cap.ToolBasis(Path("/usr/local/bin/ffmpeg"), (1, 2), "N-test", "macOS test arm64")


def evidence(encoder: str, *, mode: str = "none", rate: int = 48000, channels: int = 2,
             layout: str | None = "stereo", depths: tuple[int, ...] = (), supports_float: bool = False,
             bitrates: tuple[int, ...] = (), qualities: tuple[int, ...] = (),
             confirmation: str | None = None, evidence_id: str | None = None) -> CapabilityEvidence:
    if confirmation is None:
        confirmation = "not_applicable" if mode == "none" else "validated_setting"
    return CapabilityEvidence(
        evidence_id=evidence_id or f"ev:{encoder}:{mode}:{rate}:{channels}:{layout}",
        executable_path=BASIS.executable_path,
        executable_identity=BASIS.executable_identity,
        ffmpeg_version=BASIS.ffmpeg_version,
        platform_basis=BASIS.platform_basis,
        encoder=encoder,
        sample_rate_hz=rate,
        channels=channels,
        channel_layout=layout,
        bitrate_mode=mode,  # type: ignore[arg-type]
        supported_bit_depths=depths,
        supports_float=supports_float,
        nominal_bitrates_bps=bitrates,
        vbr_qualities=qualities,
        basis="installed_query",
        source_locator="fake record for a unit decision",
        nominal_confirmation=confirmation,  # type: ignore[arg-type]
    )


def where(rate: int = 48000, channels: int = 2, layout: str | None = "stereo") -> dict[str, Any]:
    return {"sample_rate_hz": rate, "channels": channels, "channel_layout": layout}


# --- § 2 Inputs -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["take.wav", "Take.WAV", "a.aif", "a.AIFF", "a.flac", "a.Mp3", "a.m4a",
                                  "a.aac", "a.mp4", "a.caf", "/abs/dir.x/b.FLAC"])
def test_recognised_extensions_case_insensitive(name: str) -> None:
    assert cap.is_recognised_extension(name)
    assert cap.is_recognised_extension(Path(name))


@pytest.mark.parametrize("name", ["song.m4p", "a.ogg", "a.opus", "README", "a.wav.txt", ".wav", "wav"])
def test_unrecognised_extensions(name: str) -> None:
    assert not cap.is_recognised_extension(name)


def test_input_table_is_formats_section_2() -> None:
    assert cap.RECOGNISED_EXTENSIONS == {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".aac", ".mp4", ".caf"}
    assert dict(cap.INPUT_CONTAINERS) == {".wav": "wav", ".aif": "aiff", ".aiff": "aiff", ".flac": "flac",
                                          ".m4a": "mp4", ".mp4": "mp4", ".caf": "caf", ".mp3": "mp3",
                                          ".aac": "adts"}


# --- § 3 Output choices -----------------------------------------------------------------------


def test_format_choices() -> None:
    assert cap.FORMAT_CHOICES == ("same", "wav", "flac", "alac", "aiff", "mp3", "aac", "apple")
    for choice in cap.FORMAT_CHOICES:
        assert cap.canonical_format_choice(choice) == choice
    assert cap.canonical_format_choice("aif") == "aiff"
    assert cap.canonical_format_choice("m4a") is None and "ambiguous" in cap.REFUSED_FORMAT_CHOICES["m4a"]
    assert cap.canonical_format_choice("ogg") is None
    assert cap.BUILTIN_FORMAT_CHOICE == "same"


@pytest.mark.parametrize(("target", "extension"), [("wav", ".wav"), ("flac", ".flac"), ("alac", ".m4a"),
                                                   ("aac", ".m4a"), ("mp3", ".mp3"), ("aiff", ".aif")])
def test_output_extensions(target: TargetFormatName, extension: str) -> None:
    assert cap.output_extension(target) == extension == cap.TARGETS[target].extension


def test_aiff_keeps_source_spelling_under_same() -> None:
    assert cap.output_extension("aiff", same_source_suffix=".aiff") == ".aiff"
    assert cap.output_extension("aiff", same_source_suffix=".AIF") == ".aif"
    assert cap.output_extension("aiff", same_source_suffix=".wav") == ".aif"
    assert cap.output_extension("flac", same_source_suffix=".aiff") == ".flac"


@pytest.mark.parametrize(("codec", "container", "expected"), [
    ("pcm_s16le", "wav", "wav"), ("pcm_f32le", "wav", "wav"), ("pcm_s24be", "aiff", "aiff"),
    ("pcm_f32be", "aiff", "aiff"), ("pcm_s24le", "caf", "wav"), ("flac", "flac", "flac"),
    ("alac", "mp4", "alac"), ("alac", "caf", "alac"), ("aac", "mp4", "aac"), ("aac", "adts", "aac"),
    ("aac", "caf", "aac"), ("mp3", "mp3", "mp3"),
    ("pcm_mulaw", "wav", None), ("pcm_alaw", "wav", None), ("opus", "ogg", None), ("pcm_s16le", "mp4", None),
])
def test_same_target_format(codec: str, container: str, expected: str | None) -> None:
    # codec_class comes from the WS-04 probe; the table never re-classifies codecs.
    assert cap.same_target_format(codec, codec_class(codec), container) == expected


def test_same_extension_for_raw_aac_and_caf() -> None:
    assert cap.output_extension(cap.same_target_format("aac", "lossy", "adts")) == ".m4a"  # type: ignore[arg-type]
    assert cap.output_extension(cap.same_target_format("alac", "lossless", "caf")) == ".m4a"  # type: ignore[arg-type]


# --- § 4 Static policy ----------------------------------------------------------------------------


def test_targets_cover_every_c04_format() -> None:
    assert set(cap.TARGETS) == set(get_args(TargetFormatName)) == set(cap.OUTPUT_EXTENSIONS)
    assert {name for name, spec in cap.TARGETS.items() if spec.family == "lossless"} == LOSSLESS_FORMATS


def test_encoders_and_depths_are_formats_section_4() -> None:
    t = cap.TARGETS
    assert t["wav"].encoders == ("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le")
    assert t["wav"].offered_bit_depths == (16, 24, 32) and t["wav"].offers_float
    assert t["aiff"].encoders == ("pcm_s16be", "pcm_s24be", "pcm_s32be")
    assert t["aiff"].offered_bit_depths == (16, 24, 32) and not t["aiff"].offers_float
    for name in ("flac", "alac"):
        assert t[name].encoders == (name,) and t[name].offered_bit_depths == (16, 24)
        assert not t[name].offers_float and t[name].max_explicit_rate_hz == 192_000
    assert t["mp3"].encoders == ("libmp3lame",) and t["mp3"].bitrate_modes == ("cbr", "vbr")
    assert t["mp3"].vbr_qualities == tuple(range(10))
    assert t["aac"].encoders == ("aac_at",) and t["aac"].bitrate_modes == ("cbr",) and not t["aac"].vbr_qualities
    for name in ("mp3", "aac"):
        assert t[name].max_rate_hz == 48_000 and not t[name].representations
    for name in LOSSLESS_FORMATS:
        assert t[name].bitrate_modes == ("none",) and t[name].max_rate_hz is None


def test_never_alac_at() -> None:
    assert "alac_at" in cap.NEVER_ENCODERS
    assert not {e for spec in cap.TARGETS.values() for e in spec.encoders} & cap.NEVER_ENCODERS


def test_tables_are_immutable() -> None:
    with pytest.raises(TypeError):
        cap.TARGETS["wav"] = cap.TARGETS["flac"]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.TARGETS["wav"].max_rate_hz = 1  # type: ignore[misc]
    assert isinstance(cap.TAG_MATRIX["mp4"], FrozenMap)


@pytest.mark.parametrize(("target", "depth", "is_float", "lossy", "expected"), [
    ("wav", 16, False, False, (16, False)), ("wav", 24, False, False, (24, False)),
    ("wav", 32, False, False, (32, False)), ("wav", 32, True, False, (32, True)),
    ("wav", None, True, False, (32, True)),
    ("aiff", 16, False, False, (16, False)), ("aiff", 24, False, False, (24, False)),
    ("aiff", 32, False, False, (24, False)), ("aiff", 32, True, False, (24, False)),
    ("flac", 16, False, False, (16, False)), ("flac", 24, False, False, (24, False)),
    ("flac", 32, False, False, (24, False)), ("flac", 64, True, False, (24, False)),
    ("alac", 24, False, False, (24, False)), ("alac", 32, True, False, (24, False)),
    # F6: a lossy source stored losslessly defaults to 16-bit, whatever the target.
    ("wav", None, False, True, (16, False)), ("alac", None, False, True, (16, False)),
    ("flac", None, True, True, (16, False)),
    # No stated rule: never invented.
    ("wav", 20, False, False, None), ("flac", 8, False, False, None), ("aiff", None, False, False, None),
])
def test_automatic_representation(target: TargetFormatName, depth: int | None, is_float: bool, lossy: bool,
                                  expected: tuple[int, bool] | None) -> None:
    assert cap.automatic_representation(target, source_bit_depth=depth, source_float=is_float,
                                        lossy_source=lossy) == expected


def test_lossy_targets_have_no_pcm_depth() -> None:
    for target in ("mp3", "aac"):
        with pytest.raises(ValueError):
            cap.automatic_representation(target, source_bit_depth=16, source_float=False,  # type: ignore[arg-type]
                                         lossy_source=False)


@pytest.mark.parametrize(("source", "expected"), [
    (8000, 8000), (22050, 22050), (44100, 44100), (48000, 48000), (88200, 44100), (176400, 44100),
    (96000, 48000), (192000, 48000), (64000, None), (352800, None), (384000, None),
])
def test_automatic_lossy_rate(source: int, expected: int | None) -> None:
    assert cap.automatic_lossy_rate(source) == expected


def test_automatic_lossy_rate_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        cap.automatic_lossy_rate(0)


# --- Evidence intersection: lossless -------------------------------------------------------------


def test_wav_claims_each_evidenced_representation() -> None:
    records = (evidence("pcm_s16le", depths=(16,)), evidence("pcm_s24le", depths=(24,)),
               evidence("pcm_f32le", supports_float=True))
    result = cap.lossless_capability("wav", records, BASIS, **where())
    assert result.supported and result.reason is None and result.bitrate_mode == "none"
    assert [(r.representation.bit_depth, r.representation.sample_float) for r in result.representations] == [
        (16, False), (24, False), (32, True)]
    assert result.representation(24).evidence_id == "ev:pcm_s24le:none:48000:2:stereo"  # type: ignore[union-attr]
    assert result.representation(32, True).representation.encoder == "pcm_f32le"  # type: ignore[union-attr]
    assert result.representation(32) is None
    assert any("pcm_s32le: no matching evidence" in gap for gap in result.gaps)


def test_aiff_never_claims_float_even_with_float_evidence() -> None:
    records = (evidence("pcm_s24be", depths=(24,)), evidence("pcm_f32be", supports_float=True))
    result = cap.lossless_capability("aiff", records, BASIS, **where())
    assert [r.representation.bit_depth for r in result.representations] == [24]
    assert result.representation(32, True) is None


@pytest.mark.parametrize("target", ["flac", "alac"])
def test_flac_alac_intersect_depths(target: TargetFormatName) -> None:
    generous = evidence(target, rate=96000, depths=(16, 24, 32), supports_float=True)
    result = cap.lossless_capability(target, (generous,), BASIS, **where(96000))
    assert [r.representation.bit_depth for r in result.representations] == [16, 24]
    assert not any(r.representation.sample_float for r in result.representations)
    partial = cap.lossless_capability(target, (evidence(target, rate=96000, depths=(16,)),), BASIS, **where(96000))
    assert [r.representation.bit_depth for r in partial.representations] == [16]
    assert any("24-bit not evidenced" in gap for gap in partial.gaps)
    empty = cap.lossless_capability(target, (evidence(target, rate=96000),), BASIS, **where(96000))
    assert not empty.supported and empty.reason == "unsupported_target"


def test_alac_at_evidence_never_serves_alac() -> None:
    result = cap.lossless_capability("alac", (evidence("alac_at", depths=(16, 24)),), BASIS, **where())
    assert not result.supported and result.reason == "unsupported_target"


DIMENSIONS = {
    "executable_path": {"executable_path": Path("/opt/other/ffmpeg")},
    "executable_identity": {"executable_identity": (1, 3)},
    "ffmpeg_version": {"ffmpeg_version": "N-other"},
    "platform_basis": {"platform_basis": "macOS other arm64"},
    "encoder": {"encoder": "pcm_s24le"},
    "sample_rate_hz": {"sample_rate_hz": 44100},
    "channels": {"channels": 1, "channel_layout": "mono"},
    "channel_layout": {"channel_layout": None},
    "bitrate_mode": {"bitrate_mode": "cbr", "nominal_bitrates_bps": (128000,),
                     "nominal_confirmation": "validated_setting"},
}


@pytest.mark.parametrize("dimension", sorted(DIMENSIONS))
def test_every_evidence_dimension_must_match(dimension: str) -> None:
    good = evidence("flac", depths=(16, 24))
    assert cap.lossless_capability("flac", (good,), BASIS, **where()).supported
    changed = replace(good, **DIMENSIONS[dimension])
    result = cap.lossless_capability("flac", (changed,), BASIS, **where())
    assert not result.supported and result.reason == "unsupported_target"
    assert result.representations == ()


def test_unknown_layout_never_borrows_stereo() -> None:
    stereo = evidence("flac", depths=(16, 24))
    assert not cap.lossless_capability("flac", (stereo,), BASIS, **where(layout=None)).supported
    six = evidence("flac", channels=6, layout="5.1", depths=(16, 24))
    assert not cap.lossless_capability("flac", (six,), BASIS, **where(channels=6, layout="5.1(side)")).supported
    assert cap.lossless_capability("flac", (six,), BASIS, **where(channels=6, layout="5.1")).supported


def test_ambiguous_records_claim_nothing() -> None:
    records = (evidence("flac", depths=(16, 24), evidence_id="a"), evidence("flac", depths=(16,), evidence_id="b"))
    result = cap.lossless_capability("flac", records, BASIS, **where())
    assert not result.supported and "ambiguous" in result.gaps[0]


def test_no_evidence_at_all() -> None:
    for target in sorted(LOSSLESS_FORMATS):
        result = cap.lossless_capability(target, (), BASIS, **where())  # type: ignore[arg-type]
        assert not result.supported and result.reason == "unsupported_target" and result.gaps


def test_query_validation() -> None:
    with pytest.raises(ValueError):
        cap.lossless_capability("mp3", (), BASIS, **where())
    with pytest.raises(ValueError):
        cap.lossy_capability("flac", "cbr", (), BASIS, **where())
    with pytest.raises(ValueError):
        cap.lossless_capability("ogg", (), BASIS, **where())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cap.lossless_capability("flac", (), BASIS, **where(rate=0))
    with pytest.raises(ValueError):
        cap.lossy_capability("aac", "cbr", (), BASIS, **where(channels=0))


# --- Evidence intersection: lossy ------------------------------------------------------------------

MP3_CBR = (64000, 96000, 128000, 192000, 256000, 320000)


def test_mp3_cbr_uses_exact_evidenced_choices() -> None:
    record = evidence("libmp3lame", mode="cbr", rate=44100, bitrates=(320000, 128000, 64000))
    result = cap.lossy_capability("mp3", "cbr", (record,), BASIS, **where(44100))
    assert result.supported and result.nominal_bitrates_bps == (64000, 128000, 320000)
    assert result.evidence_for_setting(bitrate_bps=128000) == record.evidence_id
    # Never rounded or clamped (F11): an interior or out-of-list value is unsupported.
    assert result.evidence_for_setting(bitrate_bps=100000) is None
    assert result.evidence_for_setting(bitrate_bps=384000) is None
    assert result.evidence_for_setting(vbr_quality=0) is None
    assert result.evidence_for_setting() is None
    assert result.evidence_for_setting(bitrate_bps=128000, vbr_quality=0) is None


def test_mp3_n3_gap_cbr_validated_v0_unevidenced() -> None:
    """The N-3 state on the installed FFmpeg (TK-0406): CBR validated, VBR an explicit gap."""
    cbr = evidence("libmp3lame", mode="cbr", rate=44100, bitrates=MP3_CBR)
    vbr = evidence("libmp3lame", mode="vbr", rate=44100, confirmation="not_applicable")
    records = (cbr, vbr)
    automatic = cap.automatic_mp3_capability(records, BASIS, **where(44100))
    assert not automatic.supported and automatic.reason == "unsupported_target"
    assert automatic.vbr_qualities == () and automatic.evidence_id is None
    assert any("not_applicable" in gap for gap in automatic.gaps)
    assert any("no automatic CBR alternative" in gap for gap in automatic.gaps)
    explicit_vbr = cap.lossy_capability("mp3", "vbr", records, BASIS, **where(44100))
    assert not explicit_vbr.supported and explicit_vbr.evidence_for_setting(vbr_quality=0) is None
    # Explicit CBR stays available exactly as evidenced.
    assert cap.lossy_capability("mp3", "cbr", records, BASIS, **where(44100)).nominal_bitrates_bps == MP3_CBR


def test_mp3_missing_records_are_gaps_never_invented() -> None:
    for mode in ("cbr", "vbr"):
        result = cap.lossy_capability("mp3", mode, (), BASIS, **where(44100))  # type: ignore[arg-type]
        assert not result.supported and result.reason == "unsupported_target"
        assert result.nominal_bitrates_bps == () and result.vbr_qualities == ()
    assert not cap.automatic_mp3_capability((), BASIS, **where(44100)).supported
    # Evidence at 44.1 kHz does not serve 48 kHz.
    cbr = evidence("libmp3lame", mode="cbr", rate=44100, bitrates=MP3_CBR)
    assert not cap.lossy_capability("mp3", "cbr", (cbr,), BASIS, **where(48000)).supported


def test_mp3_cbr_unconfirmed_or_empty_is_a_gap() -> None:
    unconfirmed = evidence("libmp3lame", mode="cbr", bitrates=MP3_CBR, confirmation="not_applicable")
    result = cap.lossy_capability("mp3", "cbr", (unconfirmed,), BASIS, **where())
    assert not result.supported and "F12" in result.gaps[0]
    empty = evidence("libmp3lame", mode="cbr", bitrates=())
    result = cap.lossy_capability("mp3", "cbr", (empty,), BASIS, **where())
    assert not result.supported and "empty" in result.gaps[0]


def test_automatic_mp3_requires_v0() -> None:
    partial = evidence("libmp3lame", mode="vbr", qualities=(2, 3))
    vbr = cap.lossy_capability("mp3", "vbr", (partial,), BASIS, **where())
    assert vbr.supported and vbr.vbr_qualities == (2, 3)
    assert any("V0" in gap for gap in vbr.gaps)
    assert not cap.automatic_mp3_capability((partial,), BASIS, **where()).supported
    full = evidence("libmp3lame", mode="vbr", qualities=tuple(range(10)), confirmation="encoder_diagnostic")
    automatic = cap.automatic_mp3_capability((full,), BASIS, **where())
    assert automatic.supported and automatic.vbr_qualities == (0,) and automatic.evidence_id == full.evidence_id
    assert cap.lossy_capability("mp3", "vbr", (full,), BASIS, **where()).gaps == ()


def test_aac_is_cbr_only_and_exact() -> None:
    # Synthetic choices for the unit decision only, not a claimed encoder table.
    record = evidence("aac_at", mode="cbr", bitrates=(128000, 256000, 320000))
    result = cap.lossy_capability("aac", "cbr", (record,), BASIS, **where())
    assert result.supported and result.evidence_for_setting(bitrate_bps=256000) == record.evidence_id
    assert result.evidence_for_setting(bitrate_bps=300000) is None
    vbr = cap.lossy_capability("aac", "vbr", (record,), BASIS, **where())
    assert not vbr.supported and "offers no vbr" in vbr.gaps[0]
    # AAC never comes from a different encoder.
    other = evidence("aac", mode="cbr", bitrates=(256000,))
    assert not cap.lossy_capability("aac", "cbr", (other,), BASIS, **where()).supported
    # A mono record does not serve stereo.
    mono = evidence("aac_at", mode="cbr", channels=1, layout="mono", bitrates=(128000,))
    assert not cap.lossy_capability("aac", "cbr", (mono,), BASIS, **where()).supported


@pytest.mark.parametrize("target", ["mp3", "aac"])
def test_lossy_rate_cap_beats_evidence(target: TargetFormatName) -> None:
    encoder = cap.TARGETS[target].encoders[0]
    high = evidence(encoder, mode="cbr", rate=96000, bitrates=(256000,))
    result = cap.lossy_capability(target, "cbr", (high,), BASIS, **where(96000))
    assert not result.supported and "48000 Hz lossy output cap" in result.gaps[0]


def test_capability_records_are_frozen() -> None:
    result = cap.lossy_capability("mp3", "cbr", (), BASIS, **where())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.evidence_id = "x"  # type: ignore[misc]
    assert isinstance(cap.Capability("mp3", "cbr", 1, 1, None, gaps=["a"]).gaps, tuple)  # type: ignore[arg-type]


# --- § 8 Tag matrix --------------------------------------------------------------------------------


def test_matrix_is_complete() -> None:
    assert set(cap.TAG_MATRIX) == set(cap.TAG_STORES)
    for store in cap.TAG_STORES:
        assert tuple(cap.TAG_MATRIX[store]) == cap.TAG_FIELDS
    assert len(cap.TAG_FIELDS) == 15 and cap.TAG_FIELDS[-1] == "extras"


@pytest.mark.parametrize(("field", "mp4", "flac", "id3", "info"), [
    ("title", "©nam", "TITLE", "TIT2", "INAM"),
    ("album_artist", "aART", "ALBUMARTIST", "TPE2", None),
    ("date", "©day", "DATE", "TYER+TDAT", "ICRD"),
    ("track", "trkn[0]", "TRACKNUMBER", "TRCK", "IPRT"),
    ("track_total", "trkn[1]", "TRACKTOTAL", "TRCK", None),
    ("disc_total", "disk[1]", "DISCTOTAL", "TPOS", None),
    ("comment", "©cmt", "COMMENT", "COMM:eng", "ICMT"),
    ("artwork", "covr", "PICTURE:3", "APIC:3", None),
    ("compilation", "cpil", "COMPILATION", "TCMP", None),
    ("extras", "----:com.apple.iTunes:{key}", "{key}", "TXXX:{key}", None),
])
def test_matrix_rows(field: str, mp4: str, flac: str, id3: str, info: str | None) -> None:
    m = cap.TAG_MATRIX
    assert (m["mp4"][field], m["flac"][field], m["id3v23"][field], m["riff_info"][field]) == (mp4, flac, id3, info)


def test_totals_without_numbers() -> None:
    assert dict(cap.TOTAL_WITHOUT_NUMBER) == {"mp4": "placeholder_zero", "flac": "separate_field",
                                              "id3v23": "not_stored", "riff_info": "not_stored"}


def test_tag_stores_per_target_and_wav_mode() -> None:
    assert cap.tag_stores("alac") == cap.tag_stores("aac") == ("mp4",)
    assert cap.tag_stores("flac") == ("flac",)
    assert cap.tag_stores("mp3") == cap.tag_stores("aiff") == ("id3v23",)
    assert cap.tag_stores("wav") == ("riff_info", "id3v23")
    assert cap.tag_stores("wav", "info") == ("riff_info",)
    assert cap.tag_stores("wav", "none") == ()
    with pytest.raises(ValueError):
        cap.tag_stores("wav", "id3")
    with pytest.raises(ValueError):
        cap.tag_stores("ogg")  # type: ignore[arg-type]


def test_storable_fields() -> None:
    every = frozenset(cap.TAG_FIELDS)
    for target in ("alac", "aac", "flac", "mp3", "aiff"):
        assert cap.storable_fields(target) == every  # type: ignore[arg-type]
    assert cap.storable_fields("wav") == every
    assert cap.storable_fields("wav", "info") == {"title", "artist", "album", "date", "track", "genre", "comment"}
    assert cap.storable_fields("wav", "none") == frozenset()


# --- Purity and layering (ARCHITECTURE § 3) ------------------------------------------------------

SRC = Path(cap.__file__).resolve().parent


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add("." * node.level + (node.module or ""))
    return found


def test_capabilities_is_pure() -> None:
    imports = _imports(SRC / "capabilities.py")
    assert {name for name in imports if name.startswith(("norma", "."))} == {"norma.models"}
    assert {name.split(".")[0] for name in imports} <= {"__future__", "collections", "dataclasses", "pathlib",
                                                         "typing", "norma"}
    assert _imports(SRC / "__init__.py") == set()


def test_formats_package_is_inert_on_import() -> None:
    code = ("import sys\nbefore = set(sys.modules)\nimport norma.formats\nnew = set(sys.modules) - before\n"
            "print(sorted(n for n in new if n.startswith('norma.formats.') or n.split('.')[0] in "
            "{'subprocess', 'mutagen'} or n.startswith('norma.adapters')))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                            check=False, shell=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
