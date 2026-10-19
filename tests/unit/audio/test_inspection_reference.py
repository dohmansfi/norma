"""Pure V9 reference and inspection policy (TK-0119, TST-A-08)."""

import math
from pathlib import Path
from typing import Literal

import pytest

from norma.audio import verify
from norma.models import Diagnostic, GainPlan, LoudnessMeasurement, MarkMode, ProcessingMark, RunOptions


def options(**changes):
    return RunOptions(command="verify", ffmpeg=Path("/unused/ffmpeg"), cwd=Path("/unused"), **changes)


def mark(validity: Literal["absent", "valid", "malformed", "unsupported"] = "valid",
         target: float | None = -18.123, mode: MarkMode | None = "linear", diagnostics=()):
    return ProcessingMark(validity, target if validity == "valid" else None,
                          mode if validity == "valid" else None, "pcm_s24le", "synthetic", diagnostics)


def test_valid_mark_overrides_config_target_without_changing_run_options():
    run = options(target=-20.0, true_peak=-3.0, tolerance=.75,
                  value_sources={"target": "config", "true_peak": "cli", "tolerance": "config"})
    parsed = mark()
    reference = verify.select_reference(run, parsed)
    assert (reference.target_lufs, reference.target_origin) == (-18.123, "mark")
    assert (reference.ceiling_dbtp, reference.ceiling_origin) == (-3.0, "cli")
    assert (reference.tolerance_lu, reference.tolerance_origin) == (.75, "config")
    assert reference.mark is parsed
    assert run.target == -20.0


def measurement(integrated=-16.0, linear=.5, *, resolution=.001,
                state: Literal["measurable", "silent", "unmeasurable"] = "measurable"):
    peak = None if linear is None else (20 * math.log10(linear) if linear > 0 else -math.inf)
    if state != "measurable":
        integrated = None
    return LoudnessMeasurement(
        signal_state=state, integrated_lufs=integrated, true_peak_dbtp=peak, true_peak_linear=linear,
        lra_lu=None, threshold_lufs=None, integrated_valid=integrated is not None, peak_valid=peak is not None,
        lra_valid=False, threshold_valid=False, integrated_resolution_lu=.001, peak_resolution_linear=resolution,
        dual_mono=False, meter="ebur128", value_source="metadata", ffmpeg_version="synthetic",
        decoded_duration_s=30.0, decoded_frames=1440000, sample_rate_hz=48000,
        decoded_nonzero=state != "silent", terminal_peak_covered=True,
    )


def test_read_only_reduce_uses_upper_bound_and_reports_measured_shortfall():
    reference = verify.select_reference(options(), mark(target=-16.0, mode="reduce"))
    quiet = verify.assess_inspection(measurement(-20.25), reference)
    assert quiet.loudness_ok and quiet.peak_ok and quiet.measurable
    assert quiet.measured_shortfall_lu == 4.25
    assert quiet.expected_lufs is None and quiet.loudness_error_lu is None
    assert [d.code for d in quiet.diagnostics] == ["reduce_shortfall"]
    assert verify.verification_failure(quiet) is None
    loud = verify.assess_inspection(measurement(-15.499), reference)
    assert not loud.loudness_ok
    failure = verify.verification_failure(loud)
    assert failure is not None and failure.reason == "verification_loudness"


@pytest.mark.parametrize("validity", ["absent", "valid", "malformed", "unsupported"])
@pytest.mark.parametrize("origin", [None, "default", "config", "cli"])
def test_target_precedence_for_every_mark_state_and_origin(validity, origin):
    run = options(target=-19.25, value_sources={} if origin is None else {"target": origin})
    parsed = mark(validity)
    reference = verify.select_reference(run, parsed)
    from_mark = validity == "valid" and origin != "cli"
    assert reference.target_lufs == (-18.123 if from_mark else -19.25)
    assert reference.target_origin == ("mark" if from_mark else origin or "default")
    assert reference.mark is parsed


@pytest.mark.parametrize("validity", ["absent", "valid", "malformed", "unsupported"])
@pytest.mark.parametrize("ceiling_origin", ["cli", "config", "default"])
@pytest.mark.parametrize("tolerance_origin", ["cli", "config", "default"])
def test_ceiling_and_tolerance_never_come_from_mark(validity, ceiling_origin, tolerance_origin):
    run = options(true_peak=-3.25, tolerance=.875,
                  value_sources={"true_peak": ceiling_origin, "tolerance": tolerance_origin})
    reference = verify.select_reference(run, mark(validity))
    assert (reference.ceiling_dbtp, reference.ceiling_origin) == (-3.25, ceiling_origin)
    assert (reference.tolerance_lu, reference.tolerance_origin) == (.875, tolerance_origin)


def test_defaults_and_missing_sources():
    reference = verify.select_reference(options(), mark("absent"))
    assert (reference.target_lufs, reference.target_origin) == (-16.0, "default")
    assert (reference.ceiling_dbtp, reference.ceiling_origin) == (-1.0, "default")
    assert (reference.tolerance_lu, reference.tolerance_origin) == (.5, "default")


@pytest.mark.parametrize("mode", ["copy", "linear", "limit", "dynamic", None])
@pytest.mark.parametrize("integrated,ok", [(-16.501, False), (-16.5, True), (-16.0, True),
                                         (-15.5, True), (-15.499, False), (-25.0, False)])
def test_ordinary_modes_use_inclusive_tolerance(mode, integrated, ok):
    reference = verify.select_reference(options(peak_mode="reduce"), mark(target=-16.0, mode=mode))
    result = verify.assess_inspection(measurement(integrated), reference)
    assert result.loudness_ok is ok and result.peak_ok and result.measurable
    assert result.expected_lufs == -16.0
    assert result.loudness_error_lu == pytest.approx(integrated + 16)
    assert result.measured_shortfall_lu == pytest.approx(max(0, -16 - integrated))
    assert not any(d.code == "reduce_shortfall" for d in result.diagnostics)
    assert (verify.verification_failure(result) is None) is ok


@pytest.mark.parametrize("validity", ["absent", "malformed", "unsupported"])
def test_nonvalid_marks_never_enable_reduce_even_when_run_requests_it(validity):
    reference = verify.select_reference(options(peak_mode="reduce"), mark(validity))
    result = verify.assess_inspection(measurement(-25.0), reference)
    assert not result.loudness_ok
    assert not any(d.code == "reduce_shortfall" for d in result.diagnostics)


@pytest.mark.parametrize("validity", ["malformed", "unsupported"])
@pytest.mark.parametrize("origin", ["cli", "config", "default"])
def test_invalid_mark_detail_survives_target_override(validity, origin):
    diagnostic = Diagnostic("processing_mark_invalid", "parsed detail", "tags_read", "synthetic:bad")
    parsed = mark(validity, diagnostics=(diagnostic,))
    reference = verify.select_reference(options(target=-17.0, value_sources={"target": origin}), parsed)
    result = verify.assess_inspection(measurement(-17.0), reference)
    assert reference.target_origin == origin and reference.mark is parsed
    assert result.diagnostics == (diagnostic,)
    assert result.diagnostics[0] is diagnostic
    assert verify.verification_failure(result) is None


@pytest.mark.parametrize("validity", ["malformed", "unsupported"])
def test_invalid_mark_still_warns_without_parser_detail(validity):
    result = verify.assess_inspection(measurement(), verify.select_reference(options(), mark(validity)))
    assert [d.code for d in result.diagnostics] == ["processing_mark_invalid"]


@pytest.mark.parametrize("target_origin", ["cli", "config", "default"])
@pytest.mark.parametrize("integrated,ok,shortfall", [(-30.0, True, 14.0), (-16.0, True, 0.0),
                                                 (-15.5, True, 0.0), (-15.499, False, 0.0)])
def test_reduce_mark_policy_survives_cli_override(target_origin, integrated, ok, shortfall):
    # A CLI target changes the reference, not the parsed mode.
    run = options(target=-16.0, value_sources={"target": target_origin})
    reference = verify.select_reference(run, mark(target=-16.0, mode="reduce"))
    result = verify.assess_inspection(measurement(integrated), reference)
    assert result.loudness_ok is ok
    assert result.measured_shortfall_lu == shortfall
    assert any(d.code == "reduce_shortfall" for d in result.diagnostics) is (ok and shortfall > 0)


@pytest.mark.parametrize("mode", ["linear", "reduce"])
@pytest.mark.parametrize("offset,ok", [(-1e-6, True), (0.0, True), (1e-6, False)])
def test_m7_rounding_boundary_keeps_printed_peak(mode, offset, ok):
    linear = 10 ** ((-1.0 + verify.PEAK_ALLOWANCE_DB + offset) / 20) - .0005
    record = measurement(linear=linear)
    result = verify.assess_inspection(record, verify.select_reference(options(), mark(target=-16.0, mode=mode)))
    assert result.peak_ok is ok
    assert result.measurement is record
    assert result.peak_excess_db == pytest.approx(20 * math.log10(linear) + 1)


@pytest.mark.parametrize("mode", ["linear", "reduce"])
def test_m7_printed_pass_can_be_conservatively_unsafe(mode):
    reference = verify.select_reference(options(true_peak=-3.0), mark(target=-16.0, mode=mode))
    result = verify.assess_inspection(measurement(linear=.716), reference)
    assert result.peak_excess_db is not None and result.peak_excess_db <= .1 and not result.peak_ok
    failure = verify.verification_failure(result)
    assert failure is not None and failure.reason == "verification_peak"
    assert failure.detail is not None and "-2.8957" in failure.detail


@pytest.mark.parametrize("mode", ["linear", "reduce"])
@pytest.mark.parametrize("resolution", [None, 0.0, -.001])
def test_missing_or_nonpositive_resolution_is_unsafe(mode, resolution):
    reference = verify.select_reference(options(), mark(target=-16.0, mode=mode))
    result = verify.assess_inspection(measurement(-16.0, linear=.1, resolution=resolution), reference)
    assert not result.peak_ok and result.loudness_ok
    failure = verify.verification_failure(result)
    assert failure is not None and failure.reason == "verification_peak"


@pytest.mark.parametrize("mode", ["linear", "reduce"])
def test_peak_failure_precedes_loudness_and_does_not_warn_reduce_shortfall(mode):
    reference = verify.select_reference(options(), mark(target=-16.0, mode=mode))
    result = verify.assess_inspection(measurement(-10.0, linear=1.0), reference)
    assert not result.peak_ok and not result.loudness_ok
    failure = verify.verification_failure(result)
    assert failure is not None and (failure.reason, failure.stage) == ("verification_peak", "verify")
    assert {d.code for d in result.diagnostics} == {"verification_peak", "verification_loudness"}
    quiet = verify.assess_inspection(measurement(-25.0, linear=1.0), reference)
    assert not any(d.code == "reduce_shortfall" for d in quiet.diagnostics)


@pytest.mark.parametrize("mode", ["linear", "reduce"])
@pytest.mark.parametrize("state,linear", [("silent", 0.0), ("unmeasurable", None),
                                         ("unmeasurable", .5), ("unmeasurable", 1.0)])
def test_nonmeasurable_input_never_passes_or_invents_shortfall(mode, state, linear):
    reference = verify.select_reference(options(), mark(target=-16.0, mode=mode))
    result = verify.assess_inspection(measurement(state=state, linear=linear), reference)
    assert not result.measurable and not result.loudness_ok
    assert result.loudness_error_lu is None and result.measured_shortfall_lu is None
    failure = verify.verification_failure(result)
    assert failure is not None and failure.reason == "verification_loudness"
    assert not any(d.code == "reduce_shortfall" for d in result.diagnostics)
    if state == "silent":
        assert result.peak_excess_db is None


def test_mark_target_is_used_as_parsed_not_rounded_again():
    target = -18.123
    reference = verify.select_reference(options(tolerance=.375), mark(target=target))
    inside = verify.assess_inspection(measurement(target + .375), reference)
    outside = verify.assess_inspection(measurement(target + .375001), reference)
    assert inside.loudness_ok and not outside.loudness_ok


def test_cli_reference_changes_reduce_acceptance_and_measured_shortfall():
    reference = verify.select_reference(options(target=-16.0, value_sources={"target": "cli"}),
                                        mark(target=-20.0, mode="reduce"))
    result = verify.assess_inspection(measurement(-17.0), reference)
    assert result.loudness_ok and result.measured_shortfall_lu == 1.0
    assert [d.code for d in result.diagnostics] == ["reduce_shortfall"]


@pytest.mark.parametrize("mode", ["linear", "reduce"])
@pytest.mark.parametrize("integrated,ordinary_ok,reduce_ok", [(-16.001, False, True),
                                                            (-16.0, True, True), (-15.999, False, False)])
def test_zero_tolerance_is_not_replaced_with_default(mode, integrated, ordinary_ok, reduce_ok):
    reference = verify.select_reference(options(tolerance=0.0, skip_within=0.0), mark(target=-16.0, mode=mode))
    result = verify.assess_inspection(measurement(integrated), reference)
    assert result.loudness_ok is (reduce_ok if mode == "reduce" else ordinary_ok)


def test_reduce_warning_uses_nonzero_shortfall_before_display_rounding():
    result = verify.assess_inspection(measurement(-16.0001),
                                     verify.select_reference(options(), mark(target=-16.0, mode="reduce")))
    assert result.measured_shortfall_lu == pytest.approx(.0001)
    assert [d.code for d in result.diagnostics] == ["reduce_shortfall"]


def test_valid_mark_without_parsed_target_is_a_caller_error_not_a_silent_fallback():
    with pytest.raises(ValueError, match="valid parsed mark must provide a target"):
        verify.select_reference(options(), mark(target=None))


def test_candidate_reduce_semantics_remain_distinct_from_inspection():
    plan = GainPlan(
        action="reduce", requested_peak_mode="reduce", target_lufs=-16.0, ceiling_dbtp=-1.0,
        tolerance_lu=.5, max_limit_db=3.0, gain_db=0.0, predicted_true_peak_dbtp=-6.0,
        limiter_reduction_db=0.0, effective_limiter_bound_db=0.0, limiter_threshold_dbtp=None,
        expected_integrated_lufs=-19.0, shortfall_lu=3.0, fallback_from=None,
        loudness_corrections=0, peak_corrections=0, reduce_fallbacks=0, reason_code="ok",
    )
    record = measurement(-10.0)
    candidate = verify.assess(record, plan)
    inspection = verify.assess_inspection(record, verify.select_reference(options(), mark(target=-16.0, mode="reduce")))
    assert candidate.loudness_ok and verify.verification_failure(candidate) is None
    assert candidate.expected_lufs == -19.0 and candidate.loudness_error_lu == 9.0
    assert not inspection.loudness_ok


def test_public_policy_does_not_measure_or_run_tools(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pure inspection policy called a tool")

    monkeypatch.setattr(verify.meter, "measure", forbidden)
    monkeypatch.setattr(verify.ffmpeg, "run_process", forbidden)
    reference = verify.select_reference(options(), mark(target=-16.0))
    result = verify.assess_inspection(measurement(), reference)
    assert verify.verification_failure(result) is None
