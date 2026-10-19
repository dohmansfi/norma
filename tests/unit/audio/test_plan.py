"""Hermetic C-05 initial planning tests; synthetic metadata, not audio renders."""

import ast
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from norma.audio.plan import needs_dynamic_calibration, plan_gain
from norma.models import Diagnostic, LoudnessMeasurement, RunOptions


def measurement(integrated=-23.0, peak=-10.0, *, resolution=.001, **changes):
    return replace(LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=integrated, true_peak_dbtp=peak,
        true_peak_linear=10 ** (peak / 20), lra_lu=0.0, threshold_lufs=None,
        integrated_valid=True, peak_valid=True, lra_valid=True, threshold_valid=False,
        integrated_resolution_lu=.001, peak_resolution_linear=resolution,
        dual_mono=False, meter="ebur128", value_source="metadata",
        ffmpeg_version="synthetic test metadata", decoded_duration_s=20.0,
        decoded_frames=960000, sample_rate_hz=48000, decoded_nonzero=True,
        terminal_peak_covered=True,
    ), **changes)


def options(**changes):
    # Paths are immutable C-10 values only: these files are never accessed.
    return RunOptions(command="analyse", ffmpeg=Path("/unused/ffmpeg"),
                      cwd=Path("/unused"), jobs=1, **changes)


@pytest.mark.parametrize("mode,integrated,peak,cap,action,gain,expected,shortfall", [
    ("limit", -16.0, -5.0, 3.0, "copy", 0.0, -16.0, 0.0),
    ("reduce", -16.0, -5.0, 3.0, "copy", 0.0, -16.0, 0.0),
    ("limit", -23.0, -10.0, 3.0, "linear", 7.0, -16.0, 0.0),
    ("reduce", -23.0, -10.0, 3.0, "linear", 7.0, -16.0, 0.0),
    ("limit", -23.0, -5.0, 3.0, "limit", 7.0, -16.0, 0.0),
    ("limit", -23.0, -4.0, 3.0, "reduce", 3.0, -20.0, 4.0),
    ("limit", -23.0, -5.0, 0.0, "reduce", 4.0, -19.0, 3.0),
    ("reduce", -23.0, -5.0, 3.0, "reduce", 4.0, -19.0, 3.0),
    ("limit", -12.0, 1.0, 3.0, "linear", -4.0, -16.0, 0.0),
    ("limit", -15.0, 2.0, 3.0, "limit", -1.0, -16.0, 0.0),
    ("reduce", -15.0, 2.0, 3.0, "reduce", -3.0, -18.0, 2.0),
])
def test_p2_supported_decision_rows(mode, integrated, peak, cap, action, gain, expected, shortfall):
    result = plan_gain(measurement(integrated, peak), options(peak_mode=mode, max_limit=cap))
    assert result.action == action
    assert result.requested_peak_mode == mode
    assert result.gain_db == pytest.approx(gain)
    assert result.predicted_true_peak_dbtp == pytest.approx(peak + gain)
    assert result.expected_integrated_lufs == pytest.approx(expected)
    assert result.shortfall_lu == pytest.approx(shortfall)
    reason = ("already_within_tolerance" if action == "copy" else
              "reduce_shortfall" if shortfall > 0 and action == "reduce" else "ok")
    assert result.reason_code == reason
    if action != "limit":
        assert result.limiter_threshold_dbtp is None
        assert result.limiter_reduction_db == result.effective_limiter_bound_db == 0
    assert result.fallback_from is None
    assert result.loudness_corrections == result.peak_corrections == result.reduce_fallbacks == 0
    assert result.dynamic_parameters is None


@pytest.mark.parametrize("reduction,cap,action", [
    (3.0, 3.0, "limit"), (3.001, 3.0, "reduce"),
    (2.999, 3.0, "limit"), (.001, 0.0, "reduce"),
])
def test_r4_cap_uses_policy_bound_not_fixed_margin(reduction, cap, action):
    source = measurement(-23.0, reduction - 8.0)
    result = plan_gain(source, options(max_limit=cap))
    assert result.action == action
    if action == "limit":
        assert result.limiter_reduction_db == pytest.approx(reduction)
        assert result.effective_limiter_bound_db == pytest.approx(reduction + .3)
        assert result.limiter_threshold_dbtp == pytest.approx(-1.3)


@pytest.mark.parametrize("mode,cap", [("limit", 3.0), ("limit", 0.0), ("reduce", 3.0)])
def test_predicted_peak_exactly_at_ceiling_is_linear(mode, cap):
    result = plan_gain(measurement(-23, -8), options(peak_mode=mode, max_limit=cap))
    assert result.action == "linear"
    assert result.predicted_true_peak_dbtp == -1


@pytest.mark.parametrize("gain,action", [(0.0, "copy"), (.25, "copy"), (-.25, "copy"),
                                        (.251, "linear"), (-.251, "linear")])
def test_copy_gain_boundary_and_predictions(gain, action):
    source = measurement(-16 - gain, -5)
    result = plan_gain(source, options(skip_within=.25))
    assert result.action == action
    assert result.gain_db == (0 if action == "copy" else pytest.approx(gain))
    assert result.expected_integrated_lufs == pytest.approx(source.integrated_lufs + result.gain_db)
    assert result.predicted_true_peak_dbtp == pytest.approx(source.true_peak_dbtp + result.gain_db)
    assert result.shortfall_lu == pytest.approx(max(0, -16 - result.expected_integrated_lufs))


@pytest.mark.parametrize("mode", ["limit", "reduce"])
def test_zero_skip_disables_even_exact_target_shortcut(mode):
    result = plan_gain(measurement(-16, -5), options(skip_within=0, peak_mode=mode))
    assert result.action == "linear" and result.gain_db == 0
    assert result.reason_code == "ok"


@pytest.mark.parametrize("resolution", [None, 0.0, -.001])
def test_unknown_or_invalid_resolution_never_copies(resolution):
    result = plan_gain(measurement(-16, -5, resolution=resolution), options())
    assert result.action == "linear"
    assert result.predicted_true_peak_dbtp == -5


def test_m7_upper_bound_blocks_central_value_safe_copy():
    # The central value fits c, but the upper end of its quantiser bin does not.
    amplitude = .891
    source = measurement(-16, 20 * math.log10(amplitude), true_peak_linear=amplitude)
    assert source.true_peak_dbtp < -1 < 20 * math.log10(amplitude + .0005)
    result = plan_gain(source, options())
    assert result.action == "linear"  # central predictions still fit, per P2
    assert result.predicted_true_peak_dbtp == source.true_peak_dbtp
    assert result.reason_code == "ok"


@pytest.mark.parametrize("offset,action", [(0.0, "copy"), (-.001, "linear")])
def test_m7_copy_ceiling_boundary(offset, action):
    source = measurement(-16, -1, true_peak_linear=.891)
    ceiling = 20 * math.log10(.8915) + offset
    result = plan_gain(source, options(true_peak=ceiling))
    assert result.action == action


def test_prediction_and_cap_do_not_use_conservative_bound():
    source = measurement(-23, -5, resolution=.2)
    result = plan_gain(source, options())
    assert result.action == "limit"
    assert result.predicted_true_peak_dbtp == 2
    assert result.limiter_reduction_db == 3
    assert result.effective_limiter_bound_db == pytest.approx(3.3)


@pytest.mark.parametrize("state", ["silent", "unmeasurable"])
def test_m4_skips_have_no_predictions_even_for_short_programme(state):
    diagnostic = Diagnostic(state, "synthetic unavailable loudness", "measure")
    source = measurement(
        signal_state=state, integrated_lufs=None, integrated_valid=False,
        true_peak_linear=0.0, true_peak_dbtp=-math.inf if state == "silent" else None,
        peak_valid=state == "silent", decoded_nonzero=state != "silent",
        decoded_duration_s=.1, diagnostics=(diagnostic,),
    )
    result = plan_gain(source, options())
    assert result.action == "skip" and result.reason_code == state
    assert result.gain_db is result.predicted_true_peak_dbtp is None
    assert result.expected_integrated_lufs is result.shortfall_lu is None
    assert result.limiter_threshold_dbtp is None
    assert result.limiter_reduction_db == result.effective_limiter_bound_db == 0
    assert result.diagnostics == (diagnostic,)


def test_short_measurable_programme_keeps_warning_and_is_planned():
    warning = Diagnostic("too_short", "synthetic short programme", "measure")
    source = measurement(decoded_duration_s=.45, diagnostics=(warning,))
    result = plan_gain(source, options())
    assert result.action == "linear"
    assert result.diagnostics == (warning,)


def test_reduce_shortfall_is_prediction_and_preserves_other_diagnostics():
    warning = Diagnostic("too_short", "synthetic short programme", "measure")
    source = measurement(-23, -5, diagnostics=(warning,))
    result = plan_gain(source, options(peak_mode="reduce"))
    assert result.shortfall_lu == 3
    assert result.reason_code == "reduce_shortfall"
    assert result.diagnostics[0] == warning
    assert result.diagnostics[1].code == "reduce_shortfall"
    assert result.diagnostics[1].stage == "plan"
    assert "predicted" in result.diagnostics[1].message
    assert source.diagnostics == (warning,)


def test_effective_options_are_used_without_rounding_or_mono_readjustment():
    source = measurement(-22.123, -10.123, dual_mono=True)
    selected = options(target=-18.234, true_peak=-2.345, max_limit=1.234,
                       tolerance=.25, skip_within=.2)
    result = plan_gain(source, selected)
    assert result.action == "linear"
    assert result.gain_db == pytest.approx(3.889)
    assert result.expected_integrated_lufs == pytest.approx(-18.234)
    assert result.predicted_true_peak_dbtp == pytest.approx(-6.234)
    assert result.target_lufs == selected.target
    assert result.ceiling_dbtp == selected.true_peak
    assert result.tolerance_lu == selected.tolerance
    assert result.max_limit_db == selected.max_limit
    assert plan_gain(replace(source, dual_mono=False), replace(selected, no_dual_mono=True)) == result


@pytest.mark.parametrize("state", ["silent", "unmeasurable"])
def test_dynamic_policy_still_skips_silent_and_unmeasurable(state):
    source = measurement(-16, -5)
    if state == "silent":
        source = replace(source, signal_state=state, integrated_lufs=None, integrated_valid=False,
                         true_peak_linear=0, true_peak_dbtp=-math.inf, decoded_nonzero=False)
    else:
        source = replace(source, signal_state=state, integrated_lufs=None, integrated_valid=False)
    selected = options(peak_mode="dynamic")
    assert not needs_dynamic_calibration(source, selected)
    result = plan_gain(source, selected)
    assert (result.action, result.reason_code, result.dynamic_parameters) == ("skip", state, None)


def test_dynamic_needs_calibration_when_selected_and_copy_row_wins_first():
    selected = options(peak_mode="dynamic")
    assert needs_dynamic_calibration(measurement(-23, -5), selected)
    with pytest.raises(ValueError, match="calibration"):
        plan_gain(measurement(-23, -5), selected)
    within = measurement(-16.1, -5)
    assert not needs_dynamic_calibration(within, selected)
    assert plan_gain(within, selected).action == "copy"  # P2 row 1 precedes the dynamic row


def test_planning_is_deterministic_and_does_not_mutate_inputs():
    source, selected = measurement(), options()
    source_before, options_before = replace(source), replace(selected)
    first = plan_gain(source, selected)
    assert first == plan_gain(source, selected)
    assert source == source_before and selected == options_before
    with pytest.raises(FrozenInstanceError):
        first.gain_db = 99


def test_module_imports_only_pure_dependencies():
    import norma.audio.plan as module

    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names}
    names.update(node.module for node in imports if isinstance(node, ast.ImportFrom))
    assert names <= {"__future__", "collections.abc", "dataclasses", "math", "types", "typing", "norma.errors",
                     "norma.models"}
