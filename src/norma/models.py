"""Shared runtime records for contracts C-01 to C-11 (contracts v1).

Every record is a frozen dataclass. Collections are tuples or ``FrozenMap``
instances, frozen recursively when the record is built, so nothing mutable
crosses a contract boundary. ``__post_init__`` enforces the invariants that
CONTRACTS states for each record; a violation raises ``ValueError``.

This module imports no other Norma module (ARCHITECTURE § 3). C-12 is test
data and C-13's table lives in ``norma.errors``; only the closed set of reason
codes is declared here, because ``Diagnostic`` needs it.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Union, get_args

# --- Common types --------------------------------------------------------------

Status = Literal["ok", "warn", "skipped", "failed"]

ReasonCode = Literal[
    "ok",
    "already_within_tolerance",
    "already_processed",
    "not_audio",
    "unsupported_codec",
    "unsupported_channels",
    "silent",
    "unmeasurable",
    "lossy_source_skipped",
    "output_exists",
    "cancelled",
    "too_short",
    "reduce_shortfall",
    "processing_mark_invalid",
    "lossy_transcode_blocked",
    "lossy_origin",
    "lossy_reencoded",
    "aac_bitrate_adjusted",
    "format_adjusted",
    "metadata_loss_warn",
    "metadata_precision_loss",
    "manifest_row_ambiguous",
    "truncated",
    "decode_error",
    "probe_error",
    "measurement_error",
    "render_error",
    "unsupported_target",
    "encoder_settings_mismatch",
    "permission_denied",
    "source_changed",
    "unsafe_output",
    "music_media_protected",
    "name_clash",
    "disk_full",
    "timeout",
    "verification_loudness",
    "verification_peak",
    "verification_tags",
    "metadata_loss_error",
    "metadata_conflict_strict",
    "artwork_invalid",
    "manifest_row_invalid",
    "backup_error",
    "commit_error",
    "cleanup_error",
    "internal",
]

REASON_CODES: tuple[str, ...] = get_args(ReasonCode)

Provenance = Literal[
    "cli", "config", "manifest", "manifest_default", "filename", "source", "fallback", "generated"
]

FieldKey = str
ExtraKey = str

Identity = tuple[int, int]
"""``(st_dev, st_ino)`` of a filesystem object."""


# --- Immutable helpers ---------------------------------------------------------


class FrozenMap(Mapping[str, Any]):
    """A read-only, hashable mapping; the defensive mapping CONTRACTS requires."""

    __slots__ = ("_data", "_hash")

    def __init__(self, data: Mapping[Any, Any] | None = None) -> None:
        items = {} if data is None else dict(data)
        self._data = {key: _freeze(value) for key, value in items.items()}
        self._hash: int | None = None

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"FrozenMap({self._data!r})"


def _freeze(value: Any) -> Any:
    """Return a recursively immutable equivalent of ``value``."""
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _set(obj: object, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _freeze_fields(obj: object, *names: str) -> None:
    for name in names:
        _set(obj, name, _freeze(getattr(obj, name)))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_literal(value: Any, literal: Any, name: str) -> None:
    allowed = get_args(literal)
    _check(value in allowed, f"{name}={value!r} is not one of {allowed}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_count(value: Any, name: str) -> None:
    _check(_is_int(value) and value >= 0, f"{name} must be a non-negative integer, got {value!r}")


def _check_positive_int(value: Any, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    _check(_is_int(value) and value > 0, f"{name} must be a positive integer, got {value!r}")


def _check_finite(value: Any, name: str, *, optional: bool = True) -> None:
    if value is None:
        _check(optional, f"{name} is required")
        return
    _check(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"{name} must be a finite number, got {value!r}",
    )


def _check_positive_float(value: Any, name: str) -> None:
    if value is None:
        return
    _check_finite(value, name)
    _check(value > 0, f"{name} must be positive when available, got {value!r}")


def _check_identity(value: Any, name: str) -> None:
    _check(
        isinstance(value, tuple) and len(value) == 2 and all(_is_int(part) for part in value),
        f"{name} must be a (st_dev, st_ino) pair of integers, got {value!r}",
    )


def _check_abs_path(value: Any, name: str, *, optional: bool = False) -> None:
    if value is None:
        _check(optional, f"{name} is required")
        return
    _check(isinstance(value, Path) and value.is_absolute(), f"{name} must be an absolute Path, got {value!r}")


def _check_unique(values: tuple[Any, ...], name: str) -> None:
    _check(len(set(values)) == len(values), f"{name} contains duplicates: {values!r}")


def field_names(record_type: type) -> tuple[str, ...]:
    """Field names of a contract record, in declaration order."""
    return tuple(item.name for item in fields(record_type))


@dataclass(frozen=True)
class Diagnostic:
    """A reason-coded diagnostic; it carries no exit decision on its own."""

    code: ReasonCode
    message: str
    stage: str
    detail: str | None = None

    def __post_init__(self) -> None:
        _check_literal(self.code, ReasonCode, "Diagnostic.code")


def _check_diagnostics(value: tuple[Any, ...], name: str) -> None:
    _check(all(isinstance(item, Diagnostic) for item in value), f"{name} must hold Diagnostic records")


# --- C-01 InputItem ------------------------------------------------------------


@dataclass(frozen=True)
class InputItem:
    source_path: Path
    root: Path
    relative_path: PurePosixPath
    identity: Identity
    input_index: int
    explicit: bool

    def __post_init__(self) -> None:
        _check_abs_path(self.source_path, "source_path")
        _check_abs_path(self.root, "root")
        _check(isinstance(self.relative_path, PurePosixPath), "relative_path must be a PurePosixPath")
        _check(not self.relative_path.is_absolute(), "relative_path must be relative")
        _check(".." not in self.relative_path.parts, "relative_path must not contain '..'")
        _set(self, "identity", tuple(self.identity))
        _check_identity(self.identity, "identity")
        _check_count(self.input_index, "input_index")


# --- C-02 ProbeResult ----------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    codec: str
    codec_class: Literal["lossless", "lossy", "unsupported"]
    container: str
    audio_stream_index: int
    ignored_streams: int
    sample_rate_hz: int
    bit_depth: int | None
    sample_float: bool
    sample_format: str
    channels: int
    channel_layout: str | None
    duration_s: float | None
    duration_basis: Literal["stream", "container", "unknown"]
    bitrate_bps: int | None
    has_artwork: bool
    chapters: int
    ffprobe_tags: Mapping[str, str]
    ffprobe_version: str

    def __post_init__(self) -> None:
        _freeze_fields(self, "ffprobe_tags")
        _check_literal(self.codec_class, Literal["lossless", "lossy", "unsupported"], "codec_class")
        _check_literal(self.duration_basis, Literal["stream", "container", "unknown"], "duration_basis")
        _check(bool(self.container) and "," not in self.container, "container must be one canonical name")
        _check_count(self.audio_stream_index, "audio_stream_index")
        _check_count(self.ignored_streams, "ignored_streams")
        _check_count(self.chapters, "chapters")
        _check_positive_int(self.sample_rate_hz, "sample_rate_hz")
        _check_positive_int(self.channels, "channels")
        _check_positive_int(self.bit_depth, "bit_depth", optional=True)
        _check_positive_int(self.bitrate_bps, "bitrate_bps", optional=True)
        _check_positive_float(self.duration_s, "duration_s")
        _check(
            (self.duration_s is None) == (self.duration_basis == "unknown"),
            "duration_basis must be 'unknown' exactly when duration_s is None",
        )
        _check(
            all(isinstance(k, str) and isinstance(v, str) for k, v in self.ffprobe_tags.items()),
            "ffprobe_tags must map str to str",
        )


# --- C-03 LoudnessMeasurement --------------------------------------------------


@dataclass(frozen=True)
class LoudnessMeasurement:
    signal_state: Literal["measurable", "silent", "unmeasurable"]
    integrated_lufs: float | None
    true_peak_dbtp: float | None
    true_peak_linear: float | None
    lra_lu: float | None
    threshold_lufs: float | None
    integrated_valid: bool
    peak_valid: bool
    lra_valid: bool
    threshold_valid: bool
    integrated_resolution_lu: float | None
    peak_resolution_linear: float | None
    dual_mono: bool
    meter: Literal["ebur128"]
    value_source: Literal["metadata"]
    ffmpeg_version: str
    decoded_duration_s: float
    decoded_frames: int | None
    sample_rate_hz: int
    decoded_nonzero: bool | None
    terminal_peak_covered: bool
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "diagnostics")
        _check_diagnostics(self.diagnostics, "diagnostics")
        _check_literal(self.signal_state, Literal["measurable", "silent", "unmeasurable"], "signal_state")
        _check_literal(self.meter, Literal["ebur128"], "meter")
        _check_literal(self.value_source, Literal["metadata"], "value_source")
        for name in ("integrated_lufs", "true_peak_linear", "lra_lu", "threshold_lufs",
                     "integrated_resolution_lu", "peak_resolution_linear"):
            _check_finite(getattr(self, name), name)
        if self.true_peak_dbtp is not None and self.true_peak_dbtp != -math.inf:
            _check_finite(self.true_peak_dbtp, "true_peak_dbtp")
        if self.true_peak_linear is not None:
            _check(self.true_peak_linear >= 0, "true_peak_linear must be non-negative")
        _check(self.integrated_valid == (self.integrated_lufs is not None),
               "integrated_valid must match integrated_lufs availability")
        _check(self.peak_valid == (self.true_peak_linear is not None and self.true_peak_dbtp is not None),
               "peak_valid must match peak availability")
        _check(self.lra_valid == (self.lra_lu is not None), "lra_valid must match lra_lu availability")
        # Ebur128 supplies no threshold metadata (C-03).
        _check(self.threshold_lufs is None and not self.threshold_valid,
               "threshold_lufs must be None and threshold_valid false for ebur128")
        _check_finite(self.decoded_duration_s, "decoded_duration_s", optional=False)
        _check(self.decoded_duration_s >= 0, "decoded_duration_s must be non-negative")
        if self.decoded_frames is not None:
            _check_count(self.decoded_frames, "decoded_frames")
        _check_positive_int(self.sample_rate_hz, "sample_rate_hz")
        if self.true_peak_dbtp == -math.inf:
            _check(self.signal_state == "silent", "-inf true peak is allowed only for a silent signal")
        if self.signal_state == "measurable":
            _check(self.integrated_valid and self.peak_valid and self.terminal_peak_covered,
                   "measurable requires valid integrated loudness, valid peak and terminal peak coverage")
            _check(self.true_peak_dbtp is not None and math.isfinite(self.true_peak_dbtp),
                   "measurable requires a finite true peak")
        elif self.signal_state == "silent":
            _check(self.decoded_nonzero is False, "silent requires decoded_nonzero=False")
            _check(self.peak_valid and self.true_peak_linear == 0 and self.true_peak_dbtp == -math.inf,
                   "silent requires a valid zero linear peak and -inf dBTP")


# --- C-04 CapabilityEvidence and TargetFormat ----------------------------------

BitrateMode = Literal["none", "cbr", "vbr"]


@dataclass(frozen=True)
class CapabilityEvidence:
    evidence_id: str
    executable_path: Path
    executable_identity: Identity
    ffmpeg_version: str
    platform_basis: str
    encoder: str
    sample_rate_hz: int
    channels: int
    channel_layout: str | None
    bitrate_mode: BitrateMode
    supported_bit_depths: tuple[int, ...]
    supports_float: bool
    nominal_bitrates_bps: tuple[int, ...]
    vbr_qualities: tuple[int, ...]
    basis: Literal["installed_query", "build_matched_evidence"]
    source_locator: str
    nominal_confirmation: Literal["validated_setting", "encoder_diagnostic", "not_applicable"]

    def __post_init__(self) -> None:
        _freeze_fields(self, "executable_identity", "supported_bit_depths", "nominal_bitrates_bps",
                       "vbr_qualities")
        _check(bool(self.evidence_id), "evidence_id must be non-empty")
        _check_abs_path(self.executable_path, "executable_path")
        _check_identity(self.executable_identity, "executable_identity")
        _check_positive_int(self.sample_rate_hz, "sample_rate_hz")
        _check_positive_int(self.channels, "channels")
        _check_literal(self.bitrate_mode, BitrateMode, "bitrate_mode")
        _check_literal(self.basis, Literal["installed_query", "build_matched_evidence"], "basis")
        _check_literal(self.nominal_confirmation,
                       Literal["validated_setting", "encoder_diagnostic", "not_applicable"],
                       "nominal_confirmation")
        for depth in self.supported_bit_depths:
            _check_positive_int(depth, "supported_bit_depths item")
        for rate in self.nominal_bitrates_bps:
            _check_positive_int(rate, "nominal_bitrates_bps item")
        for quality in self.vbr_qualities:
            _check(_is_int(quality) and 0 <= quality <= 9, f"vbr quality must be 0..9, got {quality!r}")
        _check_unique(self.nominal_bitrates_bps, "nominal_bitrates_bps")
        _check_unique(self.vbr_qualities, "vbr_qualities")
        if self.bitrate_mode == "none":
            _check(not self.nominal_bitrates_bps and not self.vbr_qualities,
                   "a bitrate_mode=none record carries no bitrate or quality choices")
        elif self.bitrate_mode == "cbr":
            _check(not self.vbr_qualities, "a CBR record carries no VBR qualities")
        else:
            _check(not self.nominal_bitrates_bps, "a VBR record carries no nominal bitrates")

    @property
    def supports_settings(self) -> bool:
        """False when a CBR/VBR record cannot support its target (formats F12)."""
        return self.bitrate_mode == "none" or self.nominal_confirmation != "not_applicable"


TargetFormatName = Literal["wav", "aiff", "flac", "alac", "mp3", "aac"]
LOSSLESS_FORMATS: frozenset[str] = frozenset({"wav", "aiff", "flac", "alac"})
STREAM_COPY_ENCODER = "stream_copy"
NOT_APPLICABLE_BASIS = "not_applicable"


@dataclass(frozen=True)
class TargetFormat:
    format: TargetFormatName
    extension: str
    encoder: str
    bit_depth: int | None
    sample_float: bool
    sample_rate_hz: int
    channels: int
    channel_layout: str | None
    dither: Literal["triangular", "none"]
    bitrate_mode: BitrateMode
    requested_bitrate_bps: int | None
    effective_bitrate_bps: int | None
    vbr_quality: int | None
    bitrate_source: Literal["auto", "explicit", "not_applicable"]
    capability_basis: str
    stream_copy: bool
    decision: Literal["as_requested", "kept_source_codec", "lossy_origin", "skipped_lossy"]
    notes: tuple[str, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "notes", "diagnostics")
        _check_diagnostics(self.diagnostics, "diagnostics")
        _check_literal(self.format, TargetFormatName, "format")
        _check_literal(self.dither, Literal["triangular", "none"], "dither")
        _check_literal(self.bitrate_mode, BitrateMode, "bitrate_mode")
        _check_literal(self.bitrate_source, Literal["auto", "explicit", "not_applicable"], "bitrate_source")
        _check_literal(self.decision,
                       Literal["as_requested", "kept_source_codec", "lossy_origin", "skipped_lossy"],
                       "decision")
        _check(self.extension.startswith(".") and len(self.extension) > 1
               and "/" not in self.extension and os.sep not in self.extension,
               "extension must start with '.' and contain no separators")
        _check(bool(self.encoder), "encoder must be non-empty")
        _check(bool(self.capability_basis), "capability_basis must be non-empty")
        _check_positive_int(self.bit_depth, "bit_depth", optional=True)
        _check_positive_int(self.sample_rate_hz, "sample_rate_hz")
        _check_positive_int(self.channels, "channels")
        _check_positive_int(self.requested_bitrate_bps, "requested_bitrate_bps", optional=True)
        _check_positive_int(self.effective_bitrate_bps, "effective_bitrate_bps", optional=True)
        if self.vbr_quality is not None:
            _check(_is_int(self.vbr_quality) and 0 <= self.vbr_quality <= 9, "vbr_quality must be 0..9")
            _check(self.format == "mp3", "vbr_quality applies to MP3 only")
        if self.format == "alac":
            _check(self.encoder in ("alac", STREAM_COPY_ENCODER), "ALAC uses native 'alac', never alac_at")
        no_setting = (self.requested_bitrate_bps is None and self.effective_bitrate_bps is None
                      and self.vbr_quality is None and self.bitrate_mode == "none"
                      and self.bitrate_source == "not_applicable")
        if self.format in LOSSLESS_FORMATS:
            _check(no_setting, "a lossless target has no bitrate/quality, mode none, source not_applicable")
        if self.stream_copy:
            _check(no_setting, "a stream-copy record carries no applied encoder setting (CR-025)")
            _check(self.encoder == STREAM_COPY_ENCODER and self.capability_basis == NOT_APPLICABLE_BASIS,
                   "a stream-copy record uses encoder 'stream_copy' and basis 'not_applicable'")
            return
        _check(self.encoder != STREAM_COPY_ENCODER, "encoder 'stream_copy' requires stream_copy=True")
        if self.decision != "skipped_lossy":
            _check(self.capability_basis != NOT_APPLICABLE_BASIS,
                   "a non-copy record needs a matched evidence_id")
        if self.bitrate_mode == "cbr":
            _check(self.requested_bitrate_bps is not None and self.effective_bitrate_bps is not None
                   and self.vbr_quality is None, "CBR needs both nominal bitrates and no VBR quality")
        elif self.bitrate_mode == "vbr":
            _check(self.vbr_quality is not None and self.requested_bitrate_bps is None
                   and self.effective_bitrate_bps is None,
                   "VBR needs a quality and no fictitious nominal bitrate")


# --- C-05 GainPlan, RenderResult, Verification ---------------------------------

GainAction = Literal["copy", "linear", "limit", "reduce", "dynamic", "skip"]

DYNAMIC_CALIBRATION_KEYS: tuple[str, ...] = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
"""The five loudnorm analysis values a dynamic render consumes (the dynamic render rules).

C-05 requires ``dynamic_parameters`` to be finite and complete: exactly these
keys, spelled as loudnorm's JSON prints them, and nothing else."""


@dataclass(frozen=True)
class GainPlan:
    action: GainAction
    requested_peak_mode: Literal["limit", "reduce", "dynamic"]
    target_lufs: float
    ceiling_dbtp: float
    tolerance_lu: float
    max_limit_db: float
    gain_db: float | None
    predicted_true_peak_dbtp: float | None
    limiter_reduction_db: float
    effective_limiter_bound_db: float
    limiter_threshold_dbtp: float | None
    expected_integrated_lufs: float | None
    shortfall_lu: float | None
    fallback_from: Literal["copy", "linear", "limit"] | None
    loudness_corrections: int
    peak_corrections: int
    reduce_fallbacks: int
    reason_code: ReasonCode
    diagnostics: tuple[Diagnostic, ...] = ()
    dynamic_parameters: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        _freeze_fields(self, "diagnostics", "dynamic_parameters")
        _check_diagnostics(self.diagnostics, "diagnostics")
        _check_literal(self.action, GainAction, "action")
        _check_literal(self.requested_peak_mode, Literal["limit", "reduce", "dynamic"], "requested_peak_mode")
        _check_literal(self.reason_code, ReasonCode, "reason_code")
        if self.fallback_from is not None:
            _check_literal(self.fallback_from, Literal["copy", "linear", "limit"], "fallback_from")
        for name in ("target_lufs", "ceiling_dbtp", "tolerance_lu", "max_limit_db",
                     "limiter_reduction_db", "effective_limiter_bound_db"):
            _check_finite(getattr(self, name), name, optional=False)
        for name in ("gain_db", "predicted_true_peak_dbtp", "limiter_threshold_dbtp",
                     "expected_integrated_lufs", "shortfall_lu"):
            _check_finite(getattr(self, name), name)
        _check(_is_int(self.loudness_corrections) and 0 <= self.loudness_corrections <= 2,
               "loudness_corrections must be 0..2")
        _check(_is_int(self.peak_corrections) and 0 <= self.peak_corrections <= 1, "peak_corrections must be 0..1")
        _check(_is_int(self.reduce_fallbacks) and 0 <= self.reduce_fallbacks <= 1, "reduce_fallbacks must be 0..1")
        _check(self.limiter_reduction_db >= 0 and self.effective_limiter_bound_db >= 0,
               "limiter bounds must be non-negative")
        if self.shortfall_lu is not None:
            _check(self.shortfall_lu >= 0, "shortfall_lu must be non-negative")
        if self.action == "skip":
            _check(self.gain_db is None and self.predicted_true_peak_dbtp is None
                   and self.expected_integrated_lufs is None and self.shortfall_lu is None,
                   "a skip plan has no gain or predictions")
        if self.action == "limit":
            _check(self.limiter_reduction_db <= self.max_limit_db,
                   "an ordinary limiter plan's policy bound must not exceed max_limit_db")
        if self.action not in ("limit", "dynamic"):
            _check(self.limiter_reduction_db == 0 and self.effective_limiter_bound_db == 0,
                   "both limiter bounds are zero without a limiter")
            _check(self.limiter_threshold_dbtp is None, "limiter_threshold_dbtp applies to limit/dynamic only")
        if self.dynamic_parameters is not None:
            _check(self.requested_peak_mode == "dynamic", "dynamic_parameters require explicit dynamic mode")
            keys = tuple(self.dynamic_parameters)
            _check(len(keys) == len(DYNAMIC_CALIBRATION_KEYS) and set(keys) == set(DYNAMIC_CALIBRATION_KEYS),
                   f"dynamic_parameters must hold exactly {DYNAMIC_CALIBRATION_KEYS} (audio § 4.3), got {keys!r}")
            for key, value in self.dynamic_parameters.items():
                _check_finite(value, f"dynamic_parameters[{key!r}]", optional=False)
        if self.action == "dynamic":
            _check(self.requested_peak_mode == "dynamic", "a dynamic plan needs explicit dynamic mode")
            if self.dynamic_parameters is None:
                self._check_dynamic_preview()

    @property
    def dynamic_preview(self) -> bool:
        """C-05 v1.3 (D-080): a read-only dynamic plan without calibration, never rendered."""
        return self.action == "dynamic" and self.dynamic_parameters is None

    def _check_dynamic_preview(self) -> None:
        """C-05 v1.3 (D-080): the preview made only by ``analyse`` and ``normalise --dry-run``.

        Nominal gain ``t − I_in``, expected loudness ``t``, predicted peak unknown, both
        limiter bounds zero, ``limiter_threshold_dbtp`` the dynamic ceiling, shortfall zero,
        and no retry budget used. A calibrated plan is still required for any render
        (``RenderResult``).
        """
        rules = (
            (self.gain_db is not None, "it carries the nominal gain t − I_in"),
            (self.expected_integrated_lufs == self.target_lufs, "it expects the target loudness"),
            (self.predicted_true_peak_dbtp is None, "it predicts no true peak"),
            (self.limiter_reduction_db == 0 and self.effective_limiter_bound_db == 0, "both limiter bounds are zero"),
            (self.limiter_threshold_dbtp == self.ceiling_dbtp, "limiter_threshold_dbtp is the dynamic ceiling"),
            (self.shortfall_lu == 0, "it predicts no shortfall"),
            (self.fallback_from is None and self.loudness_corrections == 0 and self.peak_corrections == 0
             and self.reduce_fallbacks == 0, "it is an initial plan with no retry budget used"),
        )
        for holds, rule in rules:
            _check(holds, "a dynamic plan without its loudnorm calibration is only the read-only preview "
                          f"(C-05 v1.3, D-080), in which {rule}")


@dataclass(frozen=True)
class RenderResult:
    candidate_path: Path
    attempt: int
    plan: GainPlan
    target: TargetFormat
    decoded_source_identity: Identity
    elapsed_s: float
    stderr_tail: str | None
    encoder_effective_bitrate_bps: int | None
    encoder_settings_valid: bool
    encoder_setting_evidence: str | None
    normalization_type: Literal["linear", "dynamic"] | None

    def __post_init__(self) -> None:
        _check_abs_path(self.candidate_path, "candidate_path")
        _check(_is_int(self.attempt) and 1 <= self.attempt <= 5, "attempt must be 1..5")
        _check(isinstance(self.plan, GainPlan) and isinstance(self.target, TargetFormat),
               "plan/target must be GainPlan/TargetFormat")
        _check(not self.plan.dynamic_preview,
               "a dynamic preview plan is never rendered; a render needs the calibrated plan (C-05 v1.3)")
        _set(self, "decoded_source_identity", tuple(self.decoded_source_identity))
        _check_identity(self.decoded_source_identity, "decoded_source_identity")
        _check_finite(self.elapsed_s, "elapsed_s", optional=False)
        _check(self.elapsed_s >= 0, "elapsed_s must be non-negative")
        _check_positive_int(self.encoder_effective_bitrate_bps, "encoder_effective_bitrate_bps", optional=True)
        if self.normalization_type is not None:
            _check_literal(self.normalization_type, Literal["linear", "dynamic"], "normalization_type")


@dataclass(frozen=True)
class Verification:
    measurement: LoudnessMeasurement
    expected_lufs: float | None
    loudness_error_lu: float | None
    measured_shortfall_lu: float | None
    peak_excess_db: float | None
    loudness_ok: bool
    peak_ok: bool
    measurable: bool
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "diagnostics")
        _check_diagnostics(self.diagnostics, "diagnostics")
        for name in ("expected_lufs", "loudness_error_lu", "measured_shortfall_lu", "peak_excess_db"):
            _check_finite(getattr(self, name), name)
        if self.measured_shortfall_lu is not None:
            _check(self.measured_shortfall_lu >= 0, "measured_shortfall_lu must be non-negative")


# --- C-06 SourceTags and metadata value records --------------------------------


@dataclass(frozen=True)
class Artwork:
    data: bytes
    mime: str
    width_px: int | None
    height_px: int | None
    picture_type: int
    description: str

    def __post_init__(self) -> None:
        _freeze_fields(self, "data")
        _check(isinstance(self.data, bytes), "Artwork.data must be bytes")
        _check_positive_int(self.width_px, "width_px", optional=True)
        _check_positive_int(self.height_px, "height_px", optional=True)
        _check_count(self.picture_type, "picture_type")


MetadataValue = Union[str, int, bool, Artwork]

InventoryScope = Literal["source_native", "requested_native", "expected_native", "non_tag_items"]


@dataclass(frozen=True)
class InventoryRef:
    scope: InventoryScope
    identities: tuple[str, ...]

    def __post_init__(self) -> None:
        _freeze_fields(self, "identities")
        _check_literal(self.scope, InventoryScope, "scope")
        _check(len(self.identities) > 0, "InventoryRef.identities must be non-empty")
        _check_unique(self.identities, "InventoryRef.identities")


RollupValue = Union[MetadataValue, InventoryRef]


def _check_metadata_value(value: Any, name: str) -> None:
    _check(isinstance(value, (str, int, bool, Artwork)), f"{name} must be a MetadataValue, got {type(value)}")


@dataclass(frozen=True)
class TagValue:
    value: MetadataValue
    system: str
    locator: str

    def __post_init__(self) -> None:
        _check_metadata_value(self.value, "TagValue.value")


@dataclass(frozen=True)
class ResolvedValue:
    value: MetadataValue
    provenance: Provenance
    locator: str | None

    def __post_init__(self) -> None:
        _check_metadata_value(self.value, "ResolvedValue.value")
        _check_literal(self.provenance, Provenance, "provenance")


@dataclass(frozen=True)
class RawNativeItem:
    identity: str
    store: str
    key: str
    native_type: str
    native_version: str | None
    payload_encoding: str
    payload: bytes
    locator: str

    def __post_init__(self) -> None:
        _freeze_fields(self, "payload")
        _check(bool(self.identity), "RawNativeItem.identity must be non-empty")
        _check(isinstance(self.payload, bytes), "RawNativeItem.payload must be bytes")


@dataclass(frozen=True)
class Conflict:
    key: str
    old_value: MetadataValue
    new_value: MetadataValue
    old_source: str
    winning_source: Provenance
    detail: str

    def __post_init__(self) -> None:
        _check_metadata_value(self.old_value, "Conflict.old_value")
        _check_metadata_value(self.new_value, "Conflict.new_value")
        _check_literal(self.winning_source, Provenance, "winning_source")


@dataclass(frozen=True)
class NonTagItem:
    identity: str
    kind: str
    locator: str
    detail: str
    payload: bytes | None
    payload_size: int
    payload_sha256: str

    def __post_init__(self) -> None:
        _freeze_fields(self, "payload")
        _check(bool(self.identity), "NonTagItem.identity must be non-empty")
        _check_count(self.payload_size, "payload_size")
        if self.payload is not None:
            _check(isinstance(self.payload, bytes), "NonTagItem.payload must be bytes")
            _check(len(self.payload) == self.payload_size, "payload_size must match the payload")
            _check(hashlib.sha256(self.payload).hexdigest() == self.payload_sha256,
                   "payload_sha256 must match the payload")


@dataclass(frozen=True)
class SourceTags:
    fields: Mapping[FieldKey, TagValue]
    extras: Mapping[ExtraKey, TagValue]
    artwork: tuple[TagValue, ...]
    non_tag_items: tuple[NonTagItem, ...]
    internal_conflicts: tuple[Conflict, ...]
    native_inventory: tuple[RawNativeItem, ...]
    stores: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        _freeze_fields(self, "fields", "extras", "artwork", "non_tag_items", "internal_conflicts",
                       "native_inventory", "stores")
        _check(all(isinstance(item.value, Artwork) for item in self.artwork),
               "SourceTags.artwork holds Artwork values only")
        _check(not any(isinstance(item.value, Artwork) for item in self.fields.values()),
               "artwork is not duplicated under fields")
        _check_unique(tuple(item.identity for item in self.non_tag_items), "non_tag_items identities")
        native_ids = tuple(item.identity for item in self.native_inventory)
        _check_unique(native_ids, "native_inventory identities")
        indexed = tuple(identity for ids in self.stores.values() for identity in ids)
        _check(sorted(indexed) == sorted(native_ids),
               "every native item belongs to exactly one indexed store occurrence")


# --- C-07 MetadataPlan and tag execution ---------------------------------------

Fate = Literal["kept", "changed", "added", "removed_by_request", "removed_by_design", "dropped_format_limit"]
LossKind = Literal["none", "whole_value", "precision"]


@dataclass(frozen=True)
class ArtworkPlan:
    pictures: tuple[Artwork, ...]
    provenance: Provenance
    source_locator: str | None
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "pictures", "diagnostics")
        _check(all(isinstance(item, Artwork) for item in self.pictures), "pictures must be Artwork")
        _check_literal(self.provenance, Provenance, "provenance")


def _check_rollup(value: Any, name: str) -> None:
    if value is None:
        return
    _check(isinstance(value, InventoryRef) or isinstance(value, (str, int, bool, Artwork)),
           f"{name} must be a RollupValue or None")


@dataclass(frozen=True)
class FieldStatus:
    key: str
    status: Fate
    requested: RollupValue | None
    effective: RollupValue | None
    provenance: Provenance | None
    loss_kind: LossKind
    source_native_ids: tuple[str, ...]
    expected_native_ids: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        _freeze_fields(self, "source_native_ids", "expected_native_ids")
        _check(self.key.startswith(("field:", "extra:", "picture:", "item:")),
               "FieldStatus.key must start with field:, extra:, picture: or item:")
        _check_literal(self.status, Fate, "status")
        _check_literal(self.loss_kind, LossKind, "loss_kind")
        if self.provenance is not None:
            _check_literal(self.provenance, Provenance, "provenance")
        _check_rollup(self.requested, "requested")
        _check_rollup(self.effective, "effective")
        if isinstance(self.effective, InventoryRef):
            _check(self.effective.scope == "expected_native", "an effective reference must use expected_native")


@dataclass(frozen=True)
class NativeItemStatus:
    key: str
    status: Fate
    source_ids: tuple[str, ...]
    expected_ids: tuple[str, ...]
    loss_kind: LossKind
    detail: str

    def __post_init__(self) -> None:
        _freeze_fields(self, "source_ids", "expected_ids")
        _check(self.key.startswith(("source:", "output:")), "NativeItemStatus.key must start with source: or output:")
        _check_literal(self.status, Fate, "status")
        _check_literal(self.loss_kind, LossKind, "loss_kind")


@dataclass(frozen=True)
class SidecarRequest:
    member_id: str
    purpose: Literal["tags_sidecar", "cover_export", "picture_export"]
    inventory_key: str | None
    suffix: str

    def __post_init__(self) -> None:
        _check(bool(self.member_id), "member_id must be non-empty")
        _check_literal(self.purpose, Literal["tags_sidecar", "cover_export", "picture_export"], "purpose")


@dataclass(frozen=True)
class MetadataPlan:
    plan_revision: int
    audio_basis: Literal["stream_copy", "encode"]
    conditional: bool
    source_native: tuple[RawNativeItem, ...]
    requested_native: tuple[RawNativeItem, ...]
    expected_native: tuple[RawNativeItem, ...]
    resolved: Mapping[FieldKey, ResolvedValue]
    effective: Mapping[FieldKey, ResolvedValue]
    extras: Mapping[ExtraKey, ResolvedValue]
    effective_extras: Mapping[ExtraKey, ResolvedValue]
    artwork: ArtworkPlan | None
    effective_artwork: tuple[Artwork, ...]
    statuses: tuple[FieldStatus, ...]
    native_statuses: tuple[NativeItemStatus, ...]
    conflicts: tuple[Conflict, ...]
    non_tag_items: tuple[NonTagItem, ...]
    sidecar_requests: tuple[SidecarRequest, ...]
    mark_state: Literal["pending", "final", "omitted", "not_storable"]
    loss_policy_result: Literal["ok", "warn", "sidecar", "error"]
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "source_native", "requested_native", "expected_native", "resolved", "effective",
                       "extras", "effective_extras", "effective_artwork", "statuses", "native_statuses",
                       "conflicts", "non_tag_items", "sidecar_requests", "diagnostics")
        _check_count(self.plan_revision, "plan_revision")
        _check_literal(self.audio_basis, Literal["stream_copy", "encode"], "audio_basis")
        _check_literal(self.mark_state, Literal["pending", "final", "omitted", "not_storable"], "mark_state")
        _check_literal(self.loss_policy_result, Literal["ok", "warn", "sidecar", "error"], "loss_policy_result")
        inventories = {
            "source_native": tuple(item.identity for item in self.source_native),
            "requested_native": tuple(item.identity for item in self.requested_native),
            "expected_native": tuple(item.identity for item in self.expected_native),
            "non_tag_items": tuple(item.identity for item in self.non_tag_items),
        }
        for scope, identities in inventories.items():
            _check_unique(identities, f"{scope} identities")
        for status in self.statuses:
            for value in (status.requested, status.effective):
                if isinstance(value, InventoryRef):
                    known = set(inventories[value.scope])
                    _check(all(identity in known for identity in value.identities),
                           f"{status.key}: reference IDs must resolve in {value.scope}")
        _check_unique(tuple(status.key for status in self.statuses), "FieldStatus keys")
        _check_unique(tuple(status.key for status in self.native_statuses), "NativeItemStatus keys")
        _check_unique(tuple(request.member_id for request in self.sidecar_requests), "sidecar member IDs")
        _check_native_accounting(self, frozenset(inventories["source_native"]),
                                 frozenset(inventories["expected_native"]))


def _check_native_accounting(plan: MetadataPlan, source: frozenset[str], expected: frozenset[str]) -> None:
    """C-07 native accounting: scoped links, one fate per source occurrence, one added row per new output.

    - Every ``source_native_ids``/``source_ids`` link resolves in ``source_native``
      and every ``expected_native_ids``/``expected_ids`` link in ``expected_native``.
    - Each source occurrence has exactly one fate: the row ``source:<identity>``,
      whose ``source_ids`` name that occurrence; no other row names it.
      A source occurrence is never ``added``.
    - Each ``output:<identity>`` row is ``added``, names an ``expected_native``
      occurrence in ``expected_ids`` and has no source link.
    - Each expected occurrence is either derived (linked from a source row's
      ``expected_ids``) or new with exactly one ``added`` row, never both.
    """
    for status in plan.statuses:
        _check(all(identity in source for identity in status.source_native_ids),
               f"{status.key}: source_native_ids must resolve in source_native")
        _check(all(identity in expected for identity in status.expected_native_ids),
               f"{status.key}: expected_native_ids must resolve in expected_native")
    fates: dict[str, list[str]] = {identity: [] for identity in source}
    derived: set[str] = set()
    added: set[str] = set()
    for row in plan.native_statuses:
        _check(all(identity in source for identity in row.source_ids),
               f"{row.key}: source_ids must resolve in source_native")
        _check(all(identity in expected for identity in row.expected_ids),
               f"{row.key}: expected_ids must resolve in expected_native")
        for identity in row.source_ids:
            fates[identity].append(row.key)
        prefix, _, identity = row.key.partition(":")
        if prefix == "source":
            _check(identity in source, f"{row.key}: the accounted occurrence must be in source_native")
            _check(identity in row.source_ids, f"{row.key}: source_ids must name the accounted occurrence")
            _check(row.status != "added", f"{row.key}: a source occurrence is never added")
            derived.update(row.expected_ids)
        else:
            _check(identity in expected, f"{row.key}: the added occurrence must be in expected_native")
            _check(row.status == "added", f"{row.key}: an output row records a new occurrence as added")
            _check(identity in row.expected_ids, f"{row.key}: expected_ids must name the added occurrence")
            _check(not row.source_ids, f"{row.key}: a new output occurrence has no source fate")
            added.add(identity)
    for identity in sorted(source):
        _check(fates[identity] == [f"source:{identity}"],
               f"source occurrence {identity!r} needs exactly one fate, found {fates[identity]!r}")
    for identity in sorted(expected):
        _check((identity in derived) != (identity in added),
               f"expected occurrence {identity!r} must be derived from a source fate or have one added row, "
               "not neither or both")


@dataclass(frozen=True)
class TagWriteResult:
    candidate_path: Path
    plan_revision: int
    read_back: SourceTags
    matched: bool
    mismatches: tuple[str, ...]
    staged_member_ids: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "mismatches", "staged_member_ids", "diagnostics")
        _check_abs_path(self.candidate_path, "candidate_path")
        _check_count(self.plan_revision, "plan_revision")
        _check(self.matched == (not self.mismatches), "matched must be true exactly when there are no mismatches")


# --- C-08 OutputPlan and RunWriteSet -------------------------------------------

BundlePurpose = Literal["audio", "tags_sidecar", "cover_export", "picture_export"]
WritePurpose = Literal["audio", "tags_sidecar", "cover_export", "picture_export", "log", "report",
                       "template_export"]
RUN_LEVEL_PURPOSES: frozenset[str] = frozenset({"log", "report", "template_export"})


@dataclass(frozen=True)
class BundleMember:
    member_id: str
    purpose: BundlePurpose
    inventory_key: str | None
    final_path: Path
    temporary_path: Path | None
    comparison_key: str

    def __post_init__(self) -> None:
        _check(bool(self.member_id), "member_id must be non-empty")
        _check_literal(self.purpose, BundlePurpose, "purpose")
        _check_abs_path(self.final_path, "final_path")
        _check_abs_path(self.temporary_path, "temporary_path", optional=True)
        if self.temporary_path is not None:
            _check(self.temporary_path.parent == self.final_path.parent,
                   "temporary_path must be in the final path's directory")


@dataclass(frozen=True)
class OutputPlan:
    plan_revision: int
    members: tuple[BundleMember, ...]
    collision: Literal["none", "exists_skip", "exists_overwrite", "clash_between_inputs"]
    source_identity: Identity
    overwrite_authorised: bool

    def __post_init__(self) -> None:
        _freeze_fields(self, "members", "source_identity")
        _check_count(self.plan_revision, "plan_revision")
        _check_literal(self.collision,
                       Literal["none", "exists_skip", "exists_overwrite", "clash_between_inputs"], "collision")
        _check_identity(self.source_identity, "source_identity")
        _check(sum(1 for member in self.members if member.purpose == "audio") == 1,
               "an OutputPlan has exactly one audio member")
        _check_unique(tuple(member.member_id for member in self.members), "member IDs")

    @property
    def audio_member(self) -> BundleMember:
        return next(member for member in self.members if member.purpose == "audio")


@dataclass(frozen=True)
class WriteDestination:
    destination_id: str
    purpose: WritePurpose
    input_index: int | None
    final_path: Path
    comparison_key: str
    open_mode: Literal["publish", "append"]

    def __post_init__(self) -> None:
        _check_literal(self.purpose, WritePurpose, "purpose")
        _check_literal(self.open_mode, Literal["publish", "append"], "open_mode")
        _check_abs_path(self.final_path, "final_path")
        if self.purpose in RUN_LEVEL_PURPOSES:
            _check(self.input_index is None, "run-level writes have no input_index")
        else:
            _check_count(self.input_index, "input_index")
        if self.open_mode == "append":
            _check(self.purpose == "log", "append is only for the normalise log")


@dataclass(frozen=True)
class ReadDependency:
    path: Path
    identity: Identity
    purpose: str

    def __post_init__(self) -> None:
        _set(self, "identity", tuple(self.identity))
        _check_abs_path(self.path, "path")
        _check_identity(self.identity, "identity")


@dataclass(frozen=True)
class RunWriteSet:
    revision: int
    destinations: tuple[WriteDestination, ...]
    protected_reads: tuple[ReadDependency, ...]
    validated: bool

    def __post_init__(self) -> None:
        _freeze_fields(self, "destinations", "protected_reads")
        _check_count(self.revision, "revision")
        _check_unique(tuple(item.destination_id for item in self.destinations), "destination IDs")


# --- C-09 FileOutcome and report projection ------------------------------------

MarkMode = Literal["copy", "linear", "limit", "reduce", "dynamic"]
Execution = Literal["inspect", "plan", "committed", "not_written", "publication_failed"]


@dataclass(frozen=True)
class ProcessingMark:
    validity: Literal["absent", "valid", "malformed", "unsupported"]
    target_lufs: float | None
    mode: MarkMode | None
    origin_codec: str | None
    locator: str | None
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "diagnostics")
        _check_literal(self.validity, Literal["absent", "valid", "malformed", "unsupported"], "validity")
        if self.mode is not None:
            _check_literal(self.mode, MarkMode, "mode")
        _check_finite(self.target_lufs, "target_lufs")
        if self.validity != "valid":
            _check(self.target_lufs is None and self.mode is None,
                   "only a valid mark provides a numeric target or mode")


@dataclass(frozen=True)
class VerificationReference:
    target_lufs: float
    target_origin: Literal["cli", "mark", "config", "default"]
    ceiling_dbtp: float
    ceiling_origin: Literal["cli", "config", "default"]
    tolerance_lu: float
    tolerance_origin: Literal["cli", "config", "default"]
    mark: ProcessingMark

    def __post_init__(self) -> None:
        for name in ("target_lufs", "ceiling_dbtp", "tolerance_lu"):
            _check_finite(getattr(self, name), name, optional=False)
        _check_literal(self.target_origin, Literal["cli", "mark", "config", "default"], "target_origin")
        _check_literal(self.ceiling_origin, Literal["cli", "config", "default"], "ceiling_origin")
        _check_literal(self.tolerance_origin, Literal["cli", "config", "default"], "tolerance_origin")
        if self.target_origin == "mark":
            _check(self.mark.validity == "valid", "a mark target needs a valid mark")


@dataclass(frozen=True)
class ConditionalRefusal:
    """C-09 v1.4 (D-092): a documentary alternative that cannot be planned."""

    reason_code: ReasonCode
    stage: str
    detail: str | None

    def __post_init__(self) -> None:
        _check_literal(self.reason_code, ReasonCode, "ConditionalRefusal.reason_code")
        _check(isinstance(self.stage, str), "ConditionalRefusal.stage must be str")
        _check(self.detail is None or isinstance(self.detail, str),
               "ConditionalRefusal.detail must be str or None")


@dataclass(frozen=True)
class ConditionalPlan:
    assumption: Literal["stream_copy", "encode"]
    metadata: MetadataPlan | None
    output: OutputPlan | None
    refusal: ConditionalRefusal | None = None

    def __post_init__(self) -> None:
        _check_literal(self.assumption, Literal["stream_copy", "encode"], "assumption")
        if self.refusal is not None:
            _check(isinstance(self.refusal, ConditionalRefusal), "refusal must be ConditionalRefusal or None")
            _check(self.metadata is None and self.output is None,
                   "a refused conditional branch has no metadata or output plan")
            return
        if not isinstance(self.metadata, MetadataPlan) or not isinstance(self.output, OutputPlan):
            raise ValueError("a non-refused conditional branch requires metadata and output plans")
        _check(self.metadata.conditional and self.metadata.audio_basis == self.assumption,
               "a conditional branch's metadata must be conditional with the matching audio_basis")
        _check(all(member.temporary_path is None for member in self.output.members),
               "conditional output plans allocate no temporary paths")


@dataclass(frozen=True)
class FileOutcome:
    status: Status
    reason_code: ReasonCode
    input: InputItem
    probe: ProbeResult | None
    target: TargetFormat | None
    measurement_in: LoudnessMeasurement | None
    measurement_out: LoudnessMeasurement | None
    gain: GainPlan | None
    verification: Verification | None
    verification_reference: VerificationReference | None
    source_codec: str | None
    origin_codec: str | None
    metadata: MetadataPlan | None
    tag_write: TagWriteResult | None
    output: OutputPlan | None
    conditional_plans: tuple[ConditionalPlan, ...]
    warnings: tuple[Diagnostic, ...]
    elapsed_s: float
    ffmpeg_stderr_tail: str | None
    execution: Execution
    published_paths: tuple[Path, ...]
    retained_failed_path: Path | None

    def __post_init__(self) -> None:
        _freeze_fields(self, "conditional_plans", "warnings", "published_paths")
        _check_literal(self.status, Status, "status")
        _check_literal(self.reason_code, ReasonCode, "reason_code")
        _check_literal(self.execution, Execution, "execution")
        _check_diagnostics(self.warnings, "warnings")
        _check_finite(self.elapsed_s, "elapsed_s", optional=False)
        _check(self.elapsed_s >= 0, "elapsed_s must be non-negative")
        for path in self.published_paths:
            _check_abs_path(path, "published_paths item")
        _check_abs_path(self.retained_failed_path, "retained_failed_path", optional=True)
        if self.execution in ("inspect", "plan", "not_written"):
            _check(not self.published_paths, f"execution={self.execution} has no published paths")
        if self.conditional_plans:
            _check(self.metadata is None and self.output is None,
                   "unresolved conditional alternatives leave metadata and output None")


REPORT_COLUMNS: tuple[str, ...] = (
    "input", "status", "reason", "out_path", "out_format", "out_bit_depth", "out_rate",
    "in_lufs", "in_dbtp", "in_lra", "gain_db", "mode", "limited_db", "out_lufs", "out_dbtp",
    "tag_sources", "conflicts", "dropped", "sidecar", "elapsed_s",
    "execution", "input_index", "in_signal_state", "out_signal_state",
    "target_lufs", "ceiling_dbtp", "expected_lufs", "predicted_shortfall_lu",
    "measured_shortfall_lu", "loudness_corrections", "peak_corrections", "reduce_fallbacks",
    "limiter_threshold_dbtp", "effective_limiter_bound_db", "requested_bitrate_bps", "effective_bitrate_bps",
    "bitrate_mode", "vbr_quality", "encoder", "ffmpeg_version", "warnings",
    "retained_failed_path", "published_paths", "in_peak_resolution_linear",
    "out_peak_resolution_linear", "target_origin", "ceiling_origin", "tolerance_lu",
    "tolerance_origin", "mark_validity", "source_codec", "origin_codec", "conditional_plans",
)
"""C-09 flat report columns shared by CSV and JSON, in contract order."""

SUMMARY_KEYS: tuple[str, ...] = (
    "run_id", "ok", "warn", "skipped", "failed", "not_started", "ignored", "run_errors", "report", "elapsed_s",
)
"""C-09 SUMMARY line keys, in contract order (CR-019 / D-033)."""


# --- C-10 RunOptions -----------------------------------------------------------

Command = Literal["analyse", "normalise", "verify", "tags_show", "tags_plan", "tags_template", "formats", "doctor"]
OptionSource = Literal["cli", "config", "default"]

# Keep the CLI option matrix (N/A/P) and audio/plan.py's _copy_row
# consumer aligned. tags_plan retains validation parity although it ignores
# both values at runtime (ADR-0021); config must import this single policy set.
SKIP_WITHIN_COMMANDS = frozenset({"normalise", "analyse", "tags_plan"})


def _default_jobs() -> int:
    return max(1, min(4, os.cpu_count() or 1))


@dataclass(frozen=True)
class RunOptions:
    command: Command
    ffmpeg: Path
    cwd: Path
    inputs: tuple[Path, ...] = ()
    target: float = -16.0
    true_peak: float = -1.0
    peak_mode: Literal["limit", "reduce", "dynamic"] = "limit"
    max_limit: float = 3.0
    tolerance: float = 0.5
    skip_within: float = 0.3
    no_dual_mono: bool = False
    out_dir: Path | None = None
    suffix: str | None = None
    rename: str | None = None
    in_place: bool = False
    no_backup: bool = False
    yes: bool = False
    overwrite: bool = False
    format: Literal["same", "wav", "aiff", "flac", "alac", "mp3", "aac", "apple"] = "same"
    bit_depth: Literal["same", "16", "24", "32", "float"] = "same"
    sample_rate: int | Literal["same"] = "same"
    dither: Literal["auto", "triangular", "none"] = "auto"
    bitrate: str = "auto"
    flac_level: int = 8
    lossy_sources: Literal["reencode", "lossless", "skip"] = "reencode"
    allow_lossy_transcode: bool = False
    force: bool = False
    keep_failed: bool = False
    no_source_tags: bool = False
    prefer: Literal["explicit", "source"] = "explicit"
    artist: str | None = None
    album_artist: str | None = None
    genre: str | None = None
    composer: str | None = None
    title: str | None = None
    album: str | None = None
    date: str | None = None
    comment: str | None = None
    track: int | None = None
    track_total: int | None = None
    disc: int | None = None
    disc_total: int | None = None
    artwork: Path | None = None
    folder_artwork: bool = False
    remove_artwork: bool = False
    tag: tuple[tuple[str, str], ...] = ()
    clear_tag: tuple[str, ...] = ()
    number_tracks: bool = False
    manifest: Path | None = None
    infer_from_filename: str | None = None
    infer_mode: Literal["fill", "override"] = "fill"
    infer_underscores: bool = False
    no_title_fallback: bool = False
    on_metadata_loss: Literal["warn", "sidecar", "error"] = "warn"
    strict_conflicts: bool = False
    wav_tags: Literal["info+id3", "info", "none"] = "info+id3"
    id3_version: Literal["2.3", "2.4"] = "2.3"
    artwork_max_size: int = 3000
    no_mark_processed: bool = False
    recursive: bool = False
    dry_run: bool = False
    jobs: int = field(default_factory=_default_jobs)
    report: Path | None = None
    config: Path | None = None
    verbose: bool = False
    quiet: bool = False
    check_tags: bool = False
    log: Path | None = None
    template_output: Path | Literal["-"] | None = None
    value_sources: Mapping[str, OptionSource] = field(default_factory=FrozenMap)
    explicit_fields: frozenset[str] = frozenset()
    dropped_config_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "inputs", "tag", "clear_tag", "value_sources", "dropped_config_fields")
        _set(self, "explicit_fields", frozenset(self.explicit_fields))
        _check_literal(self.command, Command, "command")
        _check_literal(self.peak_mode, Literal["limit", "reduce", "dynamic"], "peak_mode")
        _check_literal(self.format, Literal["same", "wav", "aiff", "flac", "alac", "mp3", "aac", "apple"], "format")
        _check_literal(self.bit_depth, Literal["same", "16", "24", "32", "float"], "bit_depth")
        _check_literal(self.dither, Literal["auto", "triangular", "none"], "dither")
        _check_literal(self.lossy_sources, Literal["reencode", "lossless", "skip"], "lossy_sources")
        _check_literal(self.prefer, Literal["explicit", "source"], "prefer")
        _check_literal(self.infer_mode, Literal["fill", "override"], "infer_mode")
        _check_literal(self.on_metadata_loss, Literal["warn", "sidecar", "error"], "on_metadata_loss")
        _check_literal(self.wav_tags, Literal["info+id3", "info", "none"], "wav_tags")
        _check_literal(self.id3_version, Literal["2.3", "2.4"], "id3_version")
        if self.sample_rate != "same":
            _check_positive_int(self.sample_rate, "sample_rate")
        for name in ("target", "true_peak", "max_limit", "tolerance", "skip_within"):
            _check_finite(getattr(self, name), name, optional=False)
        if self.command in SKIP_WITHIN_COMMANDS:
            _check(self.skip_within <= self.tolerance, "skip_within must not exceed tolerance")
        _check(_is_int(self.jobs) and 1 <= self.jobs <= 32, "jobs must be 1..32")
        _check(_is_int(self.flac_level), "flac_level must be an integer")
        _check_positive_int(self.artwork_max_size, "artwork_max_size")
        for name in ("track", "track_total", "disc", "disc_total"):
            _check_positive_int(getattr(self, name), name, optional=True)
        _check_abs_path(self.ffmpeg, "ffmpeg")
        _check_abs_path(self.cwd, "cwd")
        for path in self.inputs:
            _check_abs_path(path, "inputs item")
        for name in ("out_dir", "artwork", "manifest", "report", "config", "log"):
            _check_abs_path(getattr(self, name), name, optional=True)
        if self.template_output not in (None, "-"):
            _check_abs_path(self.template_output, "template_output")
        _check(all(source in get_args(OptionSource) for source in self.value_sources.values()),
               "value_sources values must be cli, config or default")


RUN_OPTION_FIELDS: tuple[str, ...] = field_names(RunOptions)


@dataclass(frozen=True)
class RunReport:
    """C-09 JSON run envelope."""

    run_id: str
    versions: Mapping[str, str]
    options: RunOptions
    outcomes: tuple[FileOutcome, ...]
    counts: Mapping[Status, int]
    not_started: int
    ignored: int
    report_path: Path | None
    elapsed_s: float
    run_errors: tuple[str, ...]

    def __post_init__(self) -> None:
        _freeze_fields(self, "versions", "outcomes", "counts", "run_errors")
        _check_count(self.not_started, "not_started")
        _check_count(self.ignored, "ignored")
        _check_abs_path(self.report_path, "report_path", optional=True)
        _check_finite(self.elapsed_s, "elapsed_s", optional=False)
        expected = {status: 0 for status in get_args(Status)}
        for outcome in self.outcomes:
            expected[outcome.status] += 1
        _check(dict(self.counts) == expected, "counts must equal the outcome status counts")


# --- C-11 Manifest v1 ----------------------------------------------------------


@dataclass(frozen=True)
class ManifestCell:
    operation: Literal["set", "clear"]
    value: MetadataValue | Path | None

    def __post_init__(self) -> None:
        _check_literal(self.operation, Literal["set", "clear"], "operation")
        if self.operation == "clear":
            _check(self.value is None, "a clear cell has no value")
        else:
            _check(self.value is not None, "a set cell has a value")


@dataclass(frozen=True)
class ManifestRow:
    match_key: str
    lexical_key: PurePosixPath
    comparison_key: str
    cells: Mapping[str, ManifestCell]
    locator: str

    def __post_init__(self) -> None:
        _freeze_fields(self, "cells")
        _check(bool(self.match_key), "match_key must be non-empty")
        _check(isinstance(self.lexical_key, PurePosixPath) and not self.lexical_key.is_absolute(),
               "lexical_key must be a relative PurePosixPath")
        _check(not self.match_key.startswith(("/", "~")), "selectors are not absolute or '~'")


@dataclass(frozen=True)
class ManifestIssue:
    locator: str
    field: str | None
    detail: str
    affected_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _freeze_fields(self, "affected_indices")
        for index in self.affected_indices:
            _check_count(index, "affected_indices item")


@dataclass(frozen=True)
class ManifestData:
    path: Path
    version: Literal[1]
    defaults: Mapping[str, ManifestCell]
    rows: tuple[ManifestRow, ...]
    issues: tuple[ManifestIssue, ...]

    def __post_init__(self) -> None:
        _freeze_fields(self, "defaults", "rows", "issues")
        _check_abs_path(self.path, "path")
        _check(_is_int(self.version) and self.version == 1, "manifest version must be exactly 1")
