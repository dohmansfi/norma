"""C-01..C-11 records: field sets equal CONTRACTS v1, and stated invariants hold."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path, PurePosixPath

import pytest

from norma import models as m

# Field names transcribed from the contracts v1 (frozen, D-040).
CONTRACT_FIELDS: dict[type, tuple[str, ...]] = {
    m.Diagnostic: ("code", "message", "stage", "detail"),
    m.InputItem: ("source_path", "root", "relative_path", "identity", "input_index", "explicit"),
    m.ProbeResult: ("codec", "codec_class", "container", "audio_stream_index", "ignored_streams", "sample_rate_hz",
                    "bit_depth", "sample_float", "sample_format", "channels", "channel_layout", "duration_s",
                    "duration_basis", "bitrate_bps", "has_artwork", "chapters", "ffprobe_tags", "ffprobe_version"),
    m.LoudnessMeasurement: ("signal_state", "integrated_lufs", "true_peak_dbtp", "true_peak_linear", "lra_lu",
                            "threshold_lufs", "integrated_valid", "peak_valid", "lra_valid", "threshold_valid",
                            "integrated_resolution_lu", "peak_resolution_linear", "dual_mono", "meter",
                            "value_source", "ffmpeg_version", "decoded_duration_s", "decoded_frames",
                            "sample_rate_hz", "decoded_nonzero", "terminal_peak_covered", "diagnostics"),
    m.CapabilityEvidence: ("evidence_id", "executable_path", "executable_identity", "ffmpeg_version",
                           "platform_basis", "encoder", "sample_rate_hz", "channels", "channel_layout",
                           "bitrate_mode", "supported_bit_depths", "supports_float", "nominal_bitrates_bps",
                           "vbr_qualities", "basis", "source_locator", "nominal_confirmation"),
    m.TargetFormat: ("format", "extension", "encoder", "bit_depth", "sample_float", "sample_rate_hz", "channels",
                     "channel_layout", "dither", "bitrate_mode", "requested_bitrate_bps", "effective_bitrate_bps",
                     "vbr_quality", "bitrate_source", "capability_basis", "stream_copy", "decision", "notes",
                     "diagnostics"),
    m.GainPlan: ("action", "requested_peak_mode", "target_lufs", "ceiling_dbtp", "tolerance_lu", "max_limit_db",
                 "gain_db", "predicted_true_peak_dbtp", "limiter_reduction_db", "effective_limiter_bound_db",
                 "limiter_threshold_dbtp", "expected_integrated_lufs", "shortfall_lu", "fallback_from",
                 "loudness_corrections", "peak_corrections", "reduce_fallbacks", "reason_code", "diagnostics",
                 "dynamic_parameters"),
    m.RenderResult: ("candidate_path", "attempt", "plan", "target", "decoded_source_identity", "elapsed_s",
                     "stderr_tail", "encoder_effective_bitrate_bps", "encoder_settings_valid",
                     "encoder_setting_evidence", "normalization_type"),
    m.Verification: ("measurement", "expected_lufs", "loudness_error_lu", "measured_shortfall_lu", "peak_excess_db",
                     "loudness_ok", "peak_ok", "measurable", "diagnostics"),
    m.Artwork: ("data", "mime", "width_px", "height_px", "picture_type", "description"),
    m.InventoryRef: ("scope", "identities"),
    m.TagValue: ("value", "system", "locator"),
    m.ResolvedValue: ("value", "provenance", "locator"),
    m.RawNativeItem: ("identity", "store", "key", "native_type", "native_version", "payload_encoding", "payload",
                      "locator"),
    m.Conflict: ("key", "old_value", "new_value", "old_source", "winning_source", "detail"),
    m.NonTagItem: ("identity", "kind", "locator", "detail", "payload", "payload_size", "payload_sha256"),
    m.SourceTags: ("fields", "extras", "artwork", "non_tag_items", "internal_conflicts", "native_inventory",
                   "stores"),
    m.ArtworkPlan: ("pictures", "provenance", "source_locator", "diagnostics"),
    m.FieldStatus: ("key", "status", "requested", "effective", "provenance", "loss_kind", "source_native_ids",
                    "expected_native_ids", "detail"),
    m.NativeItemStatus: ("key", "status", "source_ids", "expected_ids", "loss_kind", "detail"),
    m.SidecarRequest: ("member_id", "purpose", "inventory_key", "suffix"),
    m.MetadataPlan: ("plan_revision", "audio_basis", "conditional", "source_native", "requested_native",
                     "expected_native", "resolved", "effective", "extras", "effective_extras", "artwork",
                     "effective_artwork", "statuses", "native_statuses", "conflicts", "non_tag_items",
                     "sidecar_requests", "mark_state", "loss_policy_result", "diagnostics"),
    m.TagWriteResult: ("candidate_path", "plan_revision", "read_back", "matched", "mismatches", "staged_member_ids",
                       "diagnostics"),
    m.BundleMember: ("member_id", "purpose", "inventory_key", "final_path", "temporary_path", "comparison_key"),
    m.OutputPlan: ("plan_revision", "members", "collision", "source_identity", "overwrite_authorised"),
    m.WriteDestination: ("destination_id", "purpose", "input_index", "final_path", "comparison_key", "open_mode"),
    m.ReadDependency: ("path", "identity", "purpose"),
    m.RunWriteSet: ("revision", "destinations", "protected_reads", "validated"),
    m.ProcessingMark: ("validity", "target_lufs", "mode", "origin_codec", "locator", "diagnostics"),
    m.VerificationReference: ("target_lufs", "target_origin", "ceiling_dbtp", "ceiling_origin", "tolerance_lu",
                              "tolerance_origin", "mark"),
    m.ConditionalRefusal: ("reason_code", "stage", "detail"),
    m.ConditionalPlan: ("assumption", "metadata", "output", "refusal"),
    m.FileOutcome: ("status", "reason_code", "input", "probe", "target", "measurement_in", "measurement_out", "gain",
                    "verification", "verification_reference", "source_codec", "origin_codec", "metadata",
                    "tag_write", "output", "conditional_plans", "warnings", "elapsed_s", "ffmpeg_stderr_tail",
                    "execution", "published_paths", "retained_failed_path"),
    m.RunReport: ("run_id", "versions", "options", "outcomes", "counts", "not_started", "ignored", "report_path",
                  "elapsed_s", "run_errors"),
    m.ManifestCell: ("operation", "value"),
    m.ManifestRow: ("match_key", "lexical_key", "comparison_key", "cells", "locator"),
    m.ManifestIssue: ("locator", "field", "detail", "affected_indices"),
    m.ManifestData: ("path", "version", "defaults", "rows", "issues"),
}

RUN_OPTION_FIELDS = (
    "command", "inputs", "target", "true_peak", "peak_mode", "max_limit", "tolerance", "skip_within",
    "no_dual_mono", "out_dir", "suffix", "rename", "in_place", "no_backup", "yes", "overwrite", "format",
    "bit_depth", "sample_rate", "dither", "bitrate", "flac_level", "lossy_sources", "allow_lossy_transcode",
    "force", "keep_failed", "no_source_tags", "prefer", "artist", "album_artist", "genre", "composer", "title",
    "album", "date", "comment", "track", "track_total", "disc", "disc_total", "artwork", "folder_artwork",
    "remove_artwork", "tag", "clear_tag", "number_tracks", "manifest", "infer_from_filename", "infer_mode",
    "infer_underscores", "no_title_fallback", "on_metadata_loss", "strict_conflicts", "wav_tags", "id3_version",
    "artwork_max_size", "no_mark_processed", "recursive", "dry_run", "jobs", "report", "config", "verbose",
    "quiet", "check_tags", "log", "ffmpeg", "template_output", "cwd", "value_sources", "explicit_fields",
    "dropped_config_fields",
)


@pytest.mark.parametrize("record", list(CONTRACT_FIELDS), ids=lambda record: record.__name__)
def test_record_fields_equal_contract(record: type) -> None:
    assert m.field_names(record) == CONTRACT_FIELDS[record]


def test_run_options_fields_equal_contract_c10() -> None:
    assert len(RUN_OPTION_FIELDS) == 72
    assert set(m.RUN_OPTION_FIELDS) == set(RUN_OPTION_FIELDS)
    assert len(m.RUN_OPTION_FIELDS) == 72


def test_report_columns_and_summary_keys() -> None:
    assert len(m.REPORT_COLUMNS) == 53 and len(set(m.REPORT_COLUMNS)) == 53
    assert m.REPORT_COLUMNS[:3] == ("input", "status", "reason")
    assert m.REPORT_COLUMNS[-1] == "conditional_plans"
    assert m.SUMMARY_KEYS == ("run_id", "ok", "warn", "skipped", "failed", "not_started", "ignored", "run_errors",
                              "report", "elapsed_s")


def test_reason_codes_closed_set() -> None:
    assert len(m.REASON_CODES) == 47 and len(set(m.REASON_CODES)) == 47


# --- Builders ----------------------------------------------------------------------------

ABS = Path("/demo")


def input_item(**changes) -> m.InputItem:
    base = dict(source_path=ABS / "a.wav", root=ABS, relative_path=PurePosixPath("a.wav"), identity=(1, 2),
                input_index=0, explicit=False)
    base.update(changes)
    return m.InputItem(**base)


def probe(**changes) -> m.ProbeResult:
    base = dict(codec="pcm_s24le", codec_class="lossless", container="wav", audio_stream_index=0, ignored_streams=0,
                sample_rate_hz=48000, bit_depth=24, sample_float=False, sample_format="s32", channels=2,
                channel_layout="stereo", duration_s=1.0, duration_basis="stream", bitrate_bps=2304000,
                has_artwork=False, chapters=0, ffprobe_tags={}, ffprobe_version="N-1")
    base.update(changes)
    return m.ProbeResult(**base)


def measurement(**changes) -> m.LoudnessMeasurement:
    base = dict(signal_state="measurable", integrated_lufs=-20.0, true_peak_dbtp=-3.0, true_peak_linear=0.708,
                lra_lu=4.0, threshold_lufs=None, integrated_valid=True, peak_valid=True, lra_valid=True,
                threshold_valid=False, integrated_resolution_lu=0.001, peak_resolution_linear=0.001, dual_mono=False,
                meter="ebur128", value_source="metadata", ffmpeg_version="N-1", decoded_duration_s=10.0,
                decoded_frames=480000, sample_rate_hz=48000, decoded_nonzero=True, terminal_peak_covered=True)
    base.update(changes)
    return m.LoudnessMeasurement(**base)


def target(**changes) -> m.TargetFormat:
    base = dict(format="flac", extension=".flac", encoder="flac", bit_depth=24, sample_float=False,
                sample_rate_hz=48000, channels=2, channel_layout="stereo", dither="none", bitrate_mode="none",
                requested_bitrate_bps=None, effective_bitrate_bps=None, vbr_quality=None,
                bitrate_source="not_applicable", capability_basis="ev:flac", stream_copy=False,
                decision="as_requested")
    base.update(changes)
    return m.TargetFormat(**base)


def gain(**changes) -> m.GainPlan:
    base = dict(action="linear", requested_peak_mode="limit", target_lufs=-16.0, ceiling_dbtp=-1.0, tolerance_lu=0.5,
                max_limit_db=3.0, gain_db=4.0, predicted_true_peak_dbtp=-2.0, limiter_reduction_db=0.0,
                effective_limiter_bound_db=0.0, limiter_threshold_dbtp=None, expected_integrated_lufs=-16.0,
                shortfall_lu=None, fallback_from=None, loudness_corrections=0, peak_corrections=0,
                reduce_fallbacks=0, reason_code="ok")
    base.update(changes)
    return m.GainPlan(**base)


CALIBRATION = {"input_i": -20.4, "input_tp": -3.1, "input_lra": 6.2, "input_thresh": -30.9, "target_offset": 0.3}
"""A complete loudnorm calibration (audio § 4.3); values are arbitrary finite numbers."""


def dynamic_gain(**changes) -> m.GainPlan:
    base = dict(action="dynamic", requested_peak_mode="dynamic", limiter_threshold_dbtp=-1.5,
                dynamic_parameters=CALIBRATION)
    base.update(changes)
    return gain(**base)


def native(identity: str) -> m.RawNativeItem:
    return m.RawNativeItem(identity=identity, store="id3v2", key="PRIV", native_type="PRIV", native_version="2.3",
                           payload_encoding="raw", payload=b"\x00\x01", locator="0")


def metadata_plan(**changes) -> m.MetadataPlan:
    base = dict(plan_revision=1, audio_basis="encode", conditional=False, source_native=(), requested_native=(),
                expected_native=(), resolved={}, effective={}, extras={}, effective_extras={}, artwork=None,
                effective_artwork=(), statuses=(), native_statuses=(), conflicts=(), non_tag_items=(),
                sidecar_requests=(), mark_state="final", loss_policy_result="ok")
    base.update(changes)
    return m.MetadataPlan(**base)


def output_plan(**changes) -> m.OutputPlan:
    audio = m.BundleMember(member_id="audio", purpose="audio", inventory_key=None, final_path=ABS / "out/a.flac",
                           temporary_path=None, comparison_key="a.flac")
    base = dict(plan_revision=1, members=(audio,), collision="none", source_identity=(1, 2),
                overwrite_authorised=False)
    base.update(changes)
    return m.OutputPlan(**base)


def outcome(**changes) -> m.FileOutcome:
    base = dict(status="ok", reason_code="ok", input=input_item(), probe=None, target=None, measurement_in=None,
                measurement_out=None, gain=None, verification=None, verification_reference=None, source_codec=None,
                origin_codec=None, metadata=None, tag_write=None, output=None, conditional_plans=(), warnings=(),
                elapsed_s=0.5, ffmpeg_stderr_tail=None, execution="committed", published_paths=(ABS / "out/a.flac",),
                retained_failed_path=None)
    base.update(changes)
    return m.FileOutcome(**base)


# --- Immutability ----------------------------------------------------------------------------


def test_records_are_frozen_and_collections_recursively_frozen() -> None:
    result = probe(ffprobe_tags={"title": "Demo"})
    with pytest.raises(FrozenInstanceError):
        result.codec = "flac"  # type: ignore[misc]
    assert isinstance(result.ffprobe_tags, m.FrozenMap)
    with pytest.raises(TypeError):
        result.ffprobe_tags["title"] = "x"  # type: ignore[index]
    plan = gain(action="dynamic", requested_peak_mode="dynamic", limiter_threshold_dbtp=-1.5,
                dynamic_parameters=CALIBRATION)
    assert isinstance(plan.dynamic_parameters, m.FrozenMap)
    hash(plan.dynamic_parameters)
    tags = m.SourceTags(fields={}, extras={}, artwork=[], non_tag_items=[], internal_conflicts=[],
                        native_inventory=[native("a"), native("b")], stores={"id3v2": ["a", "b"]})
    assert tags.stores["id3v2"] == ("a", "b") and isinstance(tags.native_inventory, tuple)
    item = m.TargetFormat(**{**target().__dict__, "notes": ["a", "b"]})
    assert item.notes == ("a", "b")


def test_frozen_map_equality_and_hash() -> None:
    a, b = m.FrozenMap({"x": 1}), m.FrozenMap({"x": 1})
    assert a == b == {"x": 1} and hash(a) == hash(b)


# --- C-01 / C-02 ------------------------------------------------------------------------------


def test_input_item_invariants() -> None:
    assert input_item().relative_path == PurePosixPath("a.wav")
    with pytest.raises(ValueError):
        input_item(relative_path=PurePosixPath("../a.wav"))
    with pytest.raises(ValueError):
        input_item(source_path=Path("a.wav"))
    with pytest.raises(ValueError):
        input_item(input_index=-1)
    with pytest.raises(ValueError):
        input_item(identity=(1,))


def test_probe_result_invariants() -> None:
    probe(duration_s=None, duration_basis="unknown", bitrate_bps=None, bit_depth=None)
    for bad in (dict(sample_rate_hz=0), dict(channels=0), dict(bit_depth=0), dict(container="mov,mp4"),
                dict(duration_s=None), dict(codec_class="drm"), dict(bitrate_bps=0), dict(ignored_streams=-1)):
        with pytest.raises(ValueError):
            probe(**bad)


# --- C-03 -------------------------------------------------------------------------------------


def test_measurement_states() -> None:
    measurement()
    measurement(signal_state="silent", integrated_lufs=None, integrated_valid=False, true_peak_linear=0.0,
                true_peak_dbtp=-math.inf, lra_lu=None, lra_valid=False, decoded_nonzero=False)
    measurement(signal_state="unmeasurable", integrated_lufs=None, integrated_valid=False)
    with pytest.raises(ValueError):  # threshold is never supplied by ebur128
        measurement(threshold_lufs=-30.0, threshold_valid=True)
    with pytest.raises(ValueError):  # validity must match availability
        measurement(integrated_valid=False)
    with pytest.raises(ValueError):  # measurable needs terminal peak coverage
        measurement(terminal_peak_covered=False)
    with pytest.raises(ValueError):  # silence needs a decoded zero check
        measurement(signal_state="silent", integrated_lufs=None, integrated_valid=False, true_peak_linear=0.0,
                    true_peak_dbtp=-math.inf, decoded_nonzero=None)
    with pytest.raises(ValueError):  # -inf only for proven silence
        measurement(signal_state="unmeasurable", integrated_lufs=None, integrated_valid=False,
                    true_peak_linear=0.0, true_peak_dbtp=-math.inf)
    with pytest.raises(ValueError):
        measurement(integrated_lufs=math.nan)


# --- C-04 -------------------------------------------------------------------------------------


def evidence(**changes) -> m.CapabilityEvidence:
    base = dict(evidence_id="ev:1", executable_path=Path("/usr/local/bin/ffmpeg"), executable_identity=(1, 2),
                ffmpeg_version="N-1", platform_basis="macOS 15.6.1 arm64", encoder="libmp3lame",
                sample_rate_hz=44100, channels=2, channel_layout="stereo", bitrate_mode="cbr",
                supported_bit_depths=(), supports_float=False, nominal_bitrates_bps=(128000, 256000),
                vbr_qualities=(), basis="installed_query", source_locator="q",
                nominal_confirmation="validated_setting")
    base.update(changes)
    return m.CapabilityEvidence(**base)


def test_capability_evidence_invariants() -> None:
    assert evidence().supports_settings
    gap = evidence(bitrate_mode="vbr", nominal_bitrates_bps=(), nominal_confirmation="not_applicable")
    assert gap.vbr_qualities == () and not gap.supports_settings
    for bad in (dict(bitrate_mode="none"), dict(vbr_qualities=(0,)), dict(nominal_bitrates_bps=(0,)),
                dict(basis="guess"), dict(executable_path=Path("ffmpeg")), dict(nominal_bitrates_bps=(1, 1))):
        with pytest.raises(ValueError):
            evidence(**bad)
    with pytest.raises(ValueError):
        evidence(bitrate_mode="vbr", nominal_bitrates_bps=(), vbr_qualities=(10,))


def test_target_format_lossless_and_copy_records() -> None:
    target()
    with pytest.raises(ValueError):  # every lossless target is not_applicable (N-4)
        target(bitrate_source="auto")
    with pytest.raises(ValueError):
        target(format="alac", extension=".m4a", encoder="alac_at")
    copy = target(format="mp3", extension=".mp3", encoder="stream_copy", bit_depth=None,
                  capability_basis="not_applicable", stream_copy=True, notes=("ignored 256k from command line",))
    assert copy.stream_copy and copy.notes
    with pytest.raises(ValueError):  # a copy carries no applied setting (CR-025)
        target(format="mp3", extension=".mp3", encoder="stream_copy", bit_depth=None, bitrate_mode="cbr",
               requested_bitrate_bps=256000, effective_bitrate_bps=256000, bitrate_source="explicit",
               capability_basis="not_applicable", stream_copy=True)
    with pytest.raises(ValueError):  # copy uses the copy sentinels
        target(format="mp3", extension=".mp3", encoder="libmp3lame", bit_depth=None, stream_copy=True,
               capability_basis="not_applicable")
    with pytest.raises(ValueError):  # encode needs an evidence id
        target(capability_basis="not_applicable")
    with pytest.raises(ValueError):
        target(extension="flac")


def test_target_format_lossy_modes() -> None:
    lossy = dict(format="mp3", extension=".mp3", encoder="libmp3lame", bit_depth=None, capability_basis="ev:mp3")
    target(**lossy, bitrate_mode="cbr", requested_bitrate_bps=256000, effective_bitrate_bps=256000,
           bitrate_source="explicit")
    target(**lossy, bitrate_mode="vbr", vbr_quality=0, bitrate_source="auto")
    with pytest.raises(ValueError):  # CBR needs both nominal fields
        target(**lossy, bitrate_mode="cbr", requested_bitrate_bps=256000, bitrate_source="explicit")
    with pytest.raises(ValueError):  # VBR has no fictitious nominal bitrate
        target(**lossy, bitrate_mode="vbr", vbr_quality=0, effective_bitrate_bps=245000, bitrate_source="auto")
    with pytest.raises(ValueError):  # quality is MP3-only
        target(format="aac", extension=".m4a", encoder="aac_at", bit_depth=None, capability_basis="ev:aac",
               bitrate_mode="vbr", vbr_quality=0, bitrate_source="auto")


# --- C-05 -------------------------------------------------------------------------------------


def test_gain_plan_invariants() -> None:
    gain()
    gain(action="skip", gain_db=None, predicted_true_peak_dbtp=None, expected_integrated_lufs=None,
         reason_code="silent")
    gain(action="limit", limiter_reduction_db=2.0, effective_limiter_bound_db=2.3, limiter_threshold_dbtp=-1.3)
    with pytest.raises(ValueError):  # skip has no gain or predictions
        gain(action="skip", reason_code="silent")
    with pytest.raises(ValueError):  # ordinary limiter policy bound within max_limit
        gain(action="limit", limiter_reduction_db=3.5, effective_limiter_bound_db=3.8, limiter_threshold_dbtp=-1.3)
    with pytest.raises(ValueError):  # bounds are zero without a limiter
        gain(limiter_reduction_db=0.5)
    with pytest.raises(ValueError):
        gain(loudness_corrections=3)
    with pytest.raises(ValueError):  # dynamic parameters only for explicit dynamic mode
        gain(dynamic_parameters=CALIBRATION)
    with pytest.raises(ValueError):
        gain(reason_code="nope")


def test_gain_plan_dynamic_calibration_is_complete() -> None:
    """C-05: explicit dynamic mode carries exactly the five finite § 4.3 values (IC-0 finding 3)."""
    plan = dynamic_gain()
    assert set(plan.dynamic_parameters) == set(m.DYNAMIC_CALIBRATION_KEYS)
    assert m.DYNAMIC_CALIBRATION_KEYS == ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    dynamic_gain(dynamic_parameters=dict(reversed(list(CALIBRATION.items()))))  # order is immaterial
    # Requested dynamic mode that needs no render (copy/skip) carries no calibration.
    gain(action="copy", requested_peak_mode="dynamic", gain_db=0.0)
    partial = {key: value for key, value in CALIBRATION.items() if key != "input_thresh"}
    for bad in ({}, {"offset": 0.0}, partial, {**CALIBRATION, "output_i": -16.0},
                {**{k: v for k, v in CALIBRATION.items() if k != "target_offset"}, "offset": 0.3}):
        with pytest.raises(ValueError, match="exactly"):
            dynamic_gain(dynamic_parameters=bad)
    for value in (math.nan, math.inf, -math.inf, None, "1.0", True):
        with pytest.raises(ValueError):
            dynamic_gain(dynamic_parameters={**CALIBRATION, "input_tp": value})
    with pytest.raises(ValueError, match="calibration"):  # without calibration only the v1.3 preview is valid
        dynamic_gain(dynamic_parameters=None)


def preview(**changes) -> m.GainPlan:
    """C-05 v1.3 (D-080): ``analyse``/dry-run's dynamic preview for I_in −20.4 at t −16, c −1."""
    base = dict(action="dynamic", requested_peak_mode="dynamic", gain_db=4.4, predicted_true_peak_dbtp=None,
                limiter_threshold_dbtp=-1.0, expected_integrated_lufs=-16.0, shortfall_lu=0.0,
                dynamic_parameters=None)
    base.update(changes)
    return gain(**base)


def test_gain_plan_dynamic_preview_is_the_fixed_read_only_shape() -> None:
    """C-05 v1.3: action=dynamic with no calibration, never rendered; every field is fixed by D-080."""
    plan = preview()
    assert plan.dynamic_preview and plan.dynamic_parameters is None
    assert not dynamic_gain().dynamic_preview and not gain().dynamic_preview
    for bad, rule in (
        ({"gain_db": None}, "nominal gain"),
        ({"expected_integrated_lufs": -16.5}, "target loudness"),
        ({"expected_integrated_lufs": None}, "target loudness"),
        ({"predicted_true_peak_dbtp": -0.5}, "no true peak"),
        ({"limiter_reduction_db": 0.5}, "limiter bounds"),
        ({"effective_limiter_bound_db": 0.3}, "limiter bounds"),
        ({"limiter_threshold_dbtp": -1.5}, "dynamic ceiling"),
        ({"limiter_threshold_dbtp": None}, "dynamic ceiling"),
        ({"shortfall_lu": None}, "no shortfall"),
        ({"shortfall_lu": 0.4}, "no shortfall"),
        ({"peak_corrections": 1}, "retry budget"),
        ({"loudness_corrections": 1}, "retry budget"),
        ({"reduce_fallbacks": 1}, "retry budget"),
        ({"fallback_from": "limit"}, "retry budget"),
    ):
        with pytest.raises(ValueError, match=rule):
            preview(**bad)
    with pytest.raises(ValueError, match="explicit dynamic mode"):
        preview(requested_peak_mode="limit")
    with pytest.raises(ValueError, match="explicit dynamic mode"):  # also for a calibrated plan
        dynamic_gain(requested_peak_mode="reduce", dynamic_parameters=None)


def test_a_dynamic_preview_is_never_rendered() -> None:
    """C-05 v1.3: a RenderResult holds only a calibrated dynamic plan."""
    common = dict(candidate_path=ABS / "out/.a.tmp.flac", attempt=1, target=target(),
                  decoded_source_identity=(1, 2), elapsed_s=1.2, stderr_tail=None,
                  encoder_effective_bitrate_bps=None, encoder_settings_valid=True,
                  encoder_setting_evidence=None, normalization_type="dynamic")
    m.RenderResult(plan=dynamic_gain(), **common)
    with pytest.raises(ValueError, match="never rendered"):
        m.RenderResult(plan=preview(), **common)


def test_render_result_and_verification() -> None:
    result = m.RenderResult(candidate_path=ABS / "out/.a.tmp.flac", attempt=1, plan=gain(), target=target(),
                            decoded_source_identity=(1, 2), elapsed_s=1.2, stderr_tail=None,
                            encoder_effective_bitrate_bps=None, encoder_settings_valid=True,
                            encoder_setting_evidence=None, normalization_type=None)
    with pytest.raises(ValueError):
        replace(result, attempt=6)
    m.Verification(measurement=measurement(), expected_lufs=-16.0, loudness_error_lu=0.1, measured_shortfall_lu=0.0,
                   peak_excess_db=-0.5, loudness_ok=True, peak_ok=True, measurable=True)
    with pytest.raises(ValueError):
        m.Verification(measurement=measurement(), expected_lufs=-16.0, loudness_error_lu=0.1,
                       measured_shortfall_lu=-0.1, peak_excess_db=-0.5, loudness_ok=True, peak_ok=True,
                       measurable=True)


# --- C-06 / C-07 ------------------------------------------------------------------------------


def test_non_tag_item_payload_evidence() -> None:
    payload = b"chapters"
    digest = hashlib.sha256(payload).hexdigest()
    m.NonTagItem("item:chapters", "chapters", "moov", "", payload, len(payload), digest)
    m.NonTagItem("item:chapters", "chapters", "moov", "omitted", None, 2_000_000, "ab" * 32)
    with pytest.raises(ValueError):
        m.NonTagItem("item:chapters", "chapters", "moov", "", payload, len(payload) + 1, digest)
    with pytest.raises(ValueError):
        m.NonTagItem("item:chapters", "chapters", "moov", "", payload, len(payload), "0" * 64)


def test_source_tags_store_index_and_artwork() -> None:
    art = m.Artwork(b"\x89PNG", "image/png", 32, 32, 3, "")
    tags = m.SourceTags(fields={"title": m.TagValue("Demo", "id3v2", "TIT2")}, extras={},
                        artwork=(m.TagValue(art, "id3v2", "APIC"),), non_tag_items=(), internal_conflicts=(),
                        native_inventory=(native("store:id3v2:PRIV:a"),),
                        stores={"id3v2": ("store:id3v2:PRIV:a",)})
    assert tags.stores["id3v2"] == ("store:id3v2:PRIV:a",)
    with pytest.raises(ValueError):  # every native item is indexed exactly once
        m.SourceTags(fields={}, extras={}, artwork=(), non_tag_items=(), internal_conflicts=(),
                     native_inventory=(native("store:id3v2:PRIV:a"),), stores={})
    with pytest.raises(ValueError):  # artwork is not duplicated under fields
        m.SourceTags(fields={"cover": m.TagValue(art, "id3v2", "APIC")}, extras={}, artwork=(), non_tag_items=(),
                     internal_conflicts=(), native_inventory=(), stores={})


def test_metadata_plan_reference_scopes() -> None:
    ident = "store:id3v2:PRIV:com.example"
    status = m.FieldStatus(key="item:native:id3v2:PRIV:com.example", status="dropped_format_limit",
                           requested=m.InventoryRef("source_native", (ident,)), effective=None, provenance="source",
                           loss_kind="whole_value", source_native_ids=(ident,), expected_native_ids=(),
                           detail="no_cross_family_mapping")
    fate = m.NativeItemStatus(f"source:{ident}", "dropped_format_limit", (ident,), (), "whole_value",
                              "no_cross_family_mapping")
    plan = metadata_plan(source_native=(native(ident),), statuses=(status,), native_statuses=(fate,),
                         loss_policy_result="warn")
    assert plan.statuses[0].requested.identities == (ident,)
    with pytest.raises(ValueError):  # unresolved reference
        metadata_plan(statuses=(status,))
    with pytest.raises(ValueError):  # effective references use expected_native only
        m.FieldStatus(key="item:x", status="kept", requested=None, effective=m.InventoryRef("source_native", (ident,)),
                      provenance=None, loss_kind="none", source_native_ids=(), expected_native_ids=(), detail="")
    with pytest.raises(ValueError):
        m.InventoryRef("source_native", ())
    with pytest.raises(ValueError):
        m.FieldStatus(key="title", status="kept", requested="x", effective="x", provenance="cli", loss_kind="none",
                      source_native_ids=(), expected_native_ids=(), detail="")


def fate(identity: str, status: str = "kept", expected: tuple[str, ...] | None = None,
         loss_kind: str = "none") -> m.NativeItemStatus:
    expected_ids = (identity,) if expected is None else expected
    return m.NativeItemStatus(f"source:{identity}", status, (identity,), expected_ids, loss_kind, "")


def addition(identity: str) -> m.NativeItemStatus:
    return m.NativeItemStatus(f"output:{identity}", "added", (), (identity,), "none", "")


PRIV = "store:id3v2:PRIV:com.example"
TYER = "store:id3v2:TYER"
RVA2 = "store:id3v2:RVA2"


def test_metadata_plan_native_accounting_accepts_contract_traces() -> None:
    """The C-07 worked rows: kept, dropped, removed by design, added and a two-into-one conversion."""
    metadata_plan()  # nothing to account for
    metadata_plan(source_native=(native(PRIV),), requested_native=(native(PRIV),), expected_native=(native(PRIV),),
                  native_statuses=(fate(PRIV),))
    metadata_plan(source_native=(native(PRIV),), native_statuses=(fate(PRIV, "dropped_format_limit", (),
                                                                        "whole_value"),),
                  loss_policy_result="warn")
    metadata_plan(source_native=(native(RVA2),), native_statuses=(fate(RVA2, "removed_by_design", ()),))
    date = m.FieldStatus(key="field:date", status="dropped_format_limit", requested="2026-10", effective="2026",
                         provenance="cli", loss_kind="precision", source_native_ids=(), expected_native_ids=(TYER,),
                         detail="")
    metadata_plan(expected_native=(native(TYER),), native_statuses=(addition(TYER),), statuses=(date,),
                  loss_policy_result="warn")
    # Two source occurrences converted into one output occurrence: both fates link it; nothing is added.
    tdat, tdrc = "store:id3v2:TDAT", "store:id3v2:TDRC"
    metadata_plan(source_native=(native(TYER), native(tdat)), expected_native=(native(tdrc),),
                  native_statuses=(fate(TYER, "changed", (tdrc,)), fate(tdat, "changed", (tdrc,))))


def test_metadata_plan_refuses_missing_and_duplicate_fates() -> None:
    """IC-0 finding 2(a) and its duplicate-fate twin."""
    with pytest.raises(ValueError, match="exactly one fate"):  # IC-0 counterexample (a): no fate at all
        metadata_plan(source_native=(native("source-a"),))
    with pytest.raises(ValueError, match="exactly one fate"):  # one of two occurrences unaccounted
        metadata_plan(source_native=(native("a"), native("b")), native_statuses=(fate("a", "removed_by_request", ()),))
    claims_both = m.NativeItemStatus("source:a", "removed_by_request", ("a", "b"), (), "none", "")
    with pytest.raises(ValueError, match="exactly one fate"):  # a's row also claims b, which has its own row
        metadata_plan(source_native=(native("a"), native("b")),
                      native_statuses=(claims_both, fate("b", "removed_by_request", ())))
    with pytest.raises(ValueError, match="exactly one fate"):  # b is covered only through a's row
        metadata_plan(source_native=(native("a"), native("b")), native_statuses=(claims_both,))
    with pytest.raises(ValueError, match="never added"):
        metadata_plan(source_native=(native("a"),), expected_native=(native("a"),),
                      native_statuses=(fate("a", "added"),))
    with pytest.raises(ValueError, match="name the accounted occurrence"):
        metadata_plan(source_native=(native("a"), native("b")),
                      native_statuses=(m.NativeItemStatus("source:a", "removed_by_request", ("b",), (), "none", ""),))


def test_metadata_plan_refuses_dangling_links() -> None:
    """IC-0 finding 2(b) and the other scoped link fields."""
    with pytest.raises(ValueError):  # IC-0 counterexample (b): a fate for a ghost occurrence
        metadata_plan(native_statuses=(m.NativeItemStatus("source:ghost", "kept", ("ghost",), (), "none", ""),))
    with pytest.raises(ValueError, match="expected_ids must resolve"):  # kept into an output that is not planned
        metadata_plan(source_native=(native(PRIV),), native_statuses=(fate(PRIV),))
    with pytest.raises(ValueError, match="source_ids must resolve"):  # scope matters: an output ID is no source ID
        metadata_plan(expected_native=(native("x"),),
                      native_statuses=(m.NativeItemStatus("output:x", "added", ("x",), ("x",), "none", ""),))
    with pytest.raises(ValueError, match="source_native_ids must resolve"):
        metadata_plan(statuses=(m.FieldStatus("field:title", "kept", "A", "A", "source", "none", ("ghost",), (),
                                              ""),))
    with pytest.raises(ValueError, match="expected_native_ids must resolve"):
        metadata_plan(statuses=(m.FieldStatus("field:title", "kept", "A", "A", "source", "none", (), ("ghost",),
                                              ""),))
    with pytest.raises(ValueError, match="must be in expected_native"):  # the row key itself dangles
        metadata_plan(expected_native=(native("x"),),
                      native_statuses=(m.NativeItemStatus("output:ghost", "added", (), ("x",), "none", ""),))
    with pytest.raises(ValueError, match="must be in source_native"):
        metadata_plan(source_native=(native("a"),),
                      native_statuses=(m.NativeItemStatus("source:ghost", "kept", ("a",), (), "none", ""),))


def test_metadata_plan_new_output_occurrences_need_one_added_row() -> None:
    with pytest.raises(ValueError, match="derived from a source fate or have one added row"):  # neither
        metadata_plan(expected_native=(native(TYER),))
    with pytest.raises(ValueError, match="derived from a source fate or have one added row"):  # both
        metadata_plan(source_native=(native(PRIV),), expected_native=(native(PRIV),),
                      native_statuses=(fate(PRIV), addition(PRIV)))
    with pytest.raises(ValueError, match="records a new occurrence as added"):
        metadata_plan(expected_native=(native(TYER),),
                      native_statuses=(m.NativeItemStatus(f"output:{TYER}", "kept", (), (TYER,), "none", ""),))
    with pytest.raises(ValueError, match="name the added occurrence"):
        metadata_plan(expected_native=(native(TYER), native("other")),
                      native_statuses=(m.NativeItemStatus(f"output:{TYER}", "added", (), ("other",), "none", ""),
                                       addition("other")))


def test_tag_write_result_matched_flag() -> None:
    tags = m.SourceTags({}, {}, (), (), (), (), {})
    m.TagWriteResult(ABS / "c.flac", 1, tags, True, (), ("audio",))
    with pytest.raises(ValueError):
        m.TagWriteResult(ABS / "c.flac", 1, tags, True, ("field:title",), ("audio",))


# --- C-08 -------------------------------------------------------------------------------------


def test_output_plan_and_write_set() -> None:
    plan = output_plan()
    assert plan.audio_member.member_id == "audio"
    with pytest.raises(ValueError):
        output_plan(members=())
    with pytest.raises(ValueError):  # temps are same-directory
        m.BundleMember("audio", "audio", None, ABS / "out/a.flac", ABS / "other/.a.tmp", "a.flac")
    m.WriteDestination("log", "log", None, ABS / "norma.log", "norma.log", "append")
    with pytest.raises(ValueError):  # append is for the normalise log only
        m.WriteDestination("r", "report", None, ABS / "r.csv", "r.csv", "append")
    with pytest.raises(ValueError):  # run-level writes have no input index
        m.WriteDestination("r", "report", 0, ABS / "r.csv", "r.csv", "publish")
    with pytest.raises(ValueError):
        m.WriteDestination("a", "audio", None, ABS / "a.flac", "a.flac", "publish")


# --- C-09 -------------------------------------------------------------------------------------


def test_file_outcome_publication_truthfulness() -> None:
    outcome()
    outcome(status="skipped", reason_code="not_audio", execution="not_written", published_paths=())
    with pytest.raises(ValueError):  # no published path without publication
        outcome(execution="plan")
    with pytest.raises(ValueError):
        outcome(elapsed_s=-1.0)
    branch = m.ConditionalPlan("stream_copy", metadata_plan(conditional=True, audio_basis="stream_copy"),
                               output_plan())
    outcome(execution="plan", published_paths=(), conditional_plans=(branch,))
    with pytest.raises(ValueError):  # unresolved alternatives leave metadata/output None
        outcome(execution="plan", published_paths=(), conditional_plans=(branch,), output=output_plan())
    with pytest.raises(ValueError):
        m.ConditionalPlan("encode", metadata_plan(conditional=True, audio_basis="stream_copy"), output_plan())


def test_conditional_refusal_and_plan_invariants() -> None:
    refusal = m.ConditionalRefusal("metadata_loss_error", "metadata_planned", "encode-only loss")
    branch = m.ConditionalPlan("encode", None, None, refusal=refusal)
    assert branch.refusal == refusal and branch.metadata is None and branch.output is None
    with pytest.raises(FrozenInstanceError):
        refusal.detail = "changed"
    with pytest.raises(ValueError):
        m.ConditionalRefusal("invented", "metadata_planned", None)
    with pytest.raises(ValueError):
        m.ConditionalRefusal("metadata_loss_error", None, None)
    with pytest.raises(ValueError):
        m.ConditionalRefusal("metadata_loss_error", "metadata_planned", [])
    for metadata, output in ((metadata_plan(conditional=True), None),
                             (None, output_plan()),
                             (metadata_plan(conditional=True), output_plan())):
        with pytest.raises(ValueError):
            m.ConditionalPlan("encode", metadata, output, refusal=refusal)
    for metadata, output in ((None, None), (metadata_plan(conditional=True), None), (None, output_plan())):
        with pytest.raises(ValueError):
            m.ConditionalPlan("encode", metadata, output)
    with pytest.raises(ValueError):
        m.ConditionalPlan("encode", None, None, refusal="not a refusal")
    assert m.ConditionalPlan("encode", metadata_plan(conditional=True), output_plan()).refusal is None


def test_processing_mark_and_reference() -> None:
    valid = m.ProcessingMark("valid", -16.0, "limit", "pcm_s24le", "TXXX:NORMA")
    m.VerificationReference(-16.0, "mark", -1.0, "default", 0.5, "default", valid)
    malformed = m.ProcessingMark("malformed", None, None, None, "TXXX:NORMA")
    with pytest.raises(ValueError):  # a malformed mark is never a numeric reference
        m.VerificationReference(-16.0, "mark", -1.0, "default", 0.5, "default", malformed)
    with pytest.raises(ValueError):
        m.ProcessingMark("malformed", -16.0, None, None, None)


def test_run_report_counts() -> None:
    options = m.RunOptions(command="normalise", ffmpeg=Path("/usr/local/bin/ffmpeg"), cwd=ABS)
    counts = {"ok": 1, "warn": 0, "skipped": 0, "failed": 0}
    m.RunReport("r1", {"ffmpeg": "N-1"}, options, (outcome(),), counts, 0, 0, None, 1.0, ())
    with pytest.raises(ValueError):
        m.RunReport("r1", {}, options, (outcome(),), {**counts, "ok": 2}, 0, 0, None, 1.0, ())


# --- C-10 / C-11 ------------------------------------------------------------------------------


def test_run_options_defaults_and_invariants() -> None:
    options = m.RunOptions(command="analyse", ffmpeg=Path("/usr/local/bin/ffmpeg"), cwd=ABS)
    assert (options.target, options.true_peak, options.max_limit, options.tolerance, options.skip_within) == (
        -16.0, -1.0, 3.0, 0.5, 0.3)
    assert options.format == "same" and options.bitrate == "auto" and options.flac_level == 8
    assert 1 <= options.jobs <= 4 and options.id3_version == "2.3" and options.artwork_max_size == 3000
    assert isinstance(options.value_sources, m.FrozenMap) and options.explicit_fields == frozenset()
    m.RunOptions(command="normalise", ffmpeg=Path("/f"), cwd=ABS, in_place=True)  # truthful refusal is the CLI's
    for bad in (dict(skip_within=0.6), dict(jobs=33), dict(jobs=0), dict(target=math.inf), dict(format="ogg"),
                dict(cwd=Path("rel")), dict(template_output=Path("rel.csv")), dict(sample_rate=0),
                dict(value_sources={"target": "env"}), dict(track=0)):
        with pytest.raises(ValueError):
            m.RunOptions(**{"command": "normalise", "ffmpeg": Path("/f"), "cwd": ABS, **bad})
    m.RunOptions(command="tags_template", ffmpeg=Path("/f"), cwd=ABS, template_output="-")


def test_manifest_records() -> None:
    row = m.ManifestRow("../a.wav", PurePosixPath("../a.wav"), "../a.wav",
                        {"title": m.ManifestCell("set", "Demo"), "genre": m.ManifestCell("clear", None)}, "row 2")
    data = m.ManifestData(ABS / "manifest.json", 1, {}, (row,), ())
    assert data.rows[0].cells["genre"].operation == "clear"
    with pytest.raises(ValueError):
        m.ManifestCell("clear", "x")
    with pytest.raises(ValueError):
        m.ManifestData(ABS / "manifest.json", True, {}, (), ())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        m.ManifestRow("/abs.wav", PurePosixPath("abs.wav"), "abs.wav", {}, "row 3")
