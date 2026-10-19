"""C-13 reasons, statuses, exits and streams, plus the exception families.

The reason table is closed (contract C-13): each code has the
terminal status it takes when primary, its exit contribution and the streams
it appears on. Exceptions follow ARCHITECTURE § 7. This module imports only
``norma.models`` and the standard library.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from norma.models import REASON_CODES, Diagnostic, FrozenMap, ReasonCode, Status

# --- Exit codes (C-13) -----------------------------------------------------------

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_ENVIRONMENT = 3
EXIT_INTERRUPTED = 130

EXIT_PRECEDENCE: tuple[int, ...] = (EXIT_INTERRUPTED, EXIT_ENVIRONMENT, EXIT_USAGE, EXIT_FAILED, EXIT_OK)
"""Highest first: 130 > 3 > 2 > 1 > 0."""


def combine_exit_codes(codes: Iterable[int]) -> int:
    """The run exit code for a set of contributions, by C-13 precedence."""
    seen = set(codes)
    unknown = seen - set(EXIT_PRECEDENCE)
    if unknown:
        raise ValueError(f"unknown exit codes: {sorted(unknown)}")
    for code in EXIT_PRECEDENCE:
        if code in seen:
            return code
    return EXIT_OK


# --- Streams (C-13) ------------------------------------------------------------

Stream = Literal["stdout", "stderr"]

STDOUT_CARRIES: tuple[str, ...] = ("rows", "tags_formats_doctor_data", "summary")
STDERR_CARRIES: tuple[str, ...] = (
    "progress", "warnings", "usage_errors", "environment_errors", "run_errors", "verbose_stderr_tails",
)

# --- The reason table ------------------------------------------------------------

ReasonKind = Literal["ok", "skip", "warning", "failure"]


@dataclass(frozen=True)
class ReasonSpec:
    """One closed C-13 row."""

    code: ReasonCode
    kind: ReasonKind
    statuses: tuple[Status, ...]
    """Terminal status(es) when primary; already_within_tolerance has two contexts."""
    exit_code: int
    """Exit contribution when primary."""
    streams: tuple[Stream, ...]
    """The outcome row goes to stdout; warning codes are also reported on stderr."""
    active: bool = True
    """False only for music_media_protected, unreachable in v1 (D-028)."""


def _row(code: ReasonCode, kind: ReasonKind, statuses: tuple[Status, ...], exit_code: int,
         *, active: bool = True) -> ReasonSpec:
    streams: tuple[Stream, ...] = ("stdout", "stderr") if kind == "warning" else ("stdout",)
    return ReasonSpec(code, kind, statuses, exit_code, streams, active)


REASONS: FrozenMap = FrozenMap({spec.code: spec for spec in (
    _row("ok", "ok", ("ok",), EXIT_OK),
    _row("already_within_tolerance", "ok", ("ok", "skipped"), EXIT_OK),
    _row("already_processed", "skip", ("skipped",), EXIT_OK),
    _row("not_audio", "skip", ("skipped",), EXIT_OK),
    _row("unsupported_codec", "skip", ("skipped",), EXIT_OK),
    _row("unsupported_channels", "skip", ("skipped",), EXIT_OK),
    _row("silent", "skip", ("skipped",), EXIT_OK),
    _row("unmeasurable", "skip", ("skipped",), EXIT_OK),
    _row("lossy_source_skipped", "skip", ("skipped",), EXIT_OK),
    _row("output_exists", "skip", ("skipped",), EXIT_OK),
    _row("cancelled", "skip", ("skipped",), EXIT_INTERRUPTED),
    _row("too_short", "warning", ("warn",), EXIT_OK),
    _row("reduce_shortfall", "warning", ("warn",), EXIT_OK),
    _row("processing_mark_invalid", "warning", ("warn",), EXIT_OK),
    _row("lossy_transcode_blocked", "warning", ("warn",), EXIT_OK),
    _row("lossy_origin", "warning", ("warn",), EXIT_OK),
    _row("lossy_reencoded", "warning", ("warn",), EXIT_OK),
    _row("aac_bitrate_adjusted", "warning", ("warn",), EXIT_OK),
    _row("format_adjusted", "warning", ("warn",), EXIT_OK),
    _row("metadata_loss_warn", "warning", ("warn",), EXIT_OK),
    _row("metadata_precision_loss", "warning", ("warn",), EXIT_OK),
    _row("manifest_row_ambiguous", "warning", ("warn",), EXIT_OK),
    _row("truncated", "failure", ("failed",), EXIT_FAILED),
    _row("decode_error", "failure", ("failed",), EXIT_FAILED),
    _row("probe_error", "failure", ("failed",), EXIT_FAILED),
    _row("measurement_error", "failure", ("failed",), EXIT_FAILED),
    _row("render_error", "failure", ("failed",), EXIT_FAILED),
    _row("unsupported_target", "failure", ("failed",), EXIT_FAILED),
    _row("encoder_settings_mismatch", "failure", ("failed",), EXIT_FAILED),
    _row("permission_denied", "failure", ("failed",), EXIT_FAILED),
    _row("source_changed", "failure", ("failed",), EXIT_FAILED),
    _row("unsafe_output", "failure", ("failed",), EXIT_FAILED),
    _row("music_media_protected", "failure", ("failed",), EXIT_FAILED, active=False),
    _row("name_clash", "failure", ("failed",), EXIT_FAILED),
    _row("disk_full", "failure", ("failed",), EXIT_FAILED),
    _row("timeout", "failure", ("failed",), EXIT_FAILED),
    _row("verification_loudness", "failure", ("failed",), EXIT_FAILED),
    _row("verification_peak", "failure", ("failed",), EXIT_FAILED),
    _row("verification_tags", "failure", ("failed",), EXIT_FAILED),
    _row("metadata_loss_error", "failure", ("failed",), EXIT_FAILED),
    _row("metadata_conflict_strict", "failure", ("failed",), EXIT_FAILED),
    _row("artwork_invalid", "failure", ("failed",), EXIT_FAILED),
    _row("manifest_row_invalid", "failure", ("failed",), EXIT_FAILED),
    _row("backup_error", "failure", ("failed",), EXIT_FAILED),
    _row("commit_error", "failure", ("failed",), EXIT_FAILED),
    _row("cleanup_error", "failure", ("failed",), EXIT_FAILED),
    _row("internal", "failure", ("failed",), EXIT_FAILED),
)})

if tuple(REASONS) != REASON_CODES:  # pragma: no cover - guards the two declarations against drift
    raise RuntimeError("errors.REASONS and models.ReasonCode disagree")

WARNING_PRECEDENCE: tuple[ReasonCode, ...] = (
    "reduce_shortfall",
    "aac_bitrate_adjusted",
    "metadata_precision_loss",
    "metadata_loss_warn",
    "lossy_transcode_blocked",
    "lossy_origin",
    "lossy_reencoded",
    "format_adjusted",
    "manifest_row_ambiguous",
    "processing_mark_invalid",
    "too_short",
)
"""Primary-warning precedence (C-13), first wins."""


def reason_spec(code: str) -> ReasonSpec:
    try:
        return REASONS[code]
    except KeyError:
        raise ValueError(f"{code!r} is not a C-13 reason code") from None


def check_primary(status: Status, reason: ReasonCode) -> None:
    """Raise ValueError when ``reason`` cannot be primary for ``status``."""
    spec = reason_spec(reason)
    if status not in spec.statuses:
        raise ValueError(f"reason {reason!r} is primary only for {spec.statuses}, not {status!r}")


def exit_contribution(status: Status, reason: ReasonCode) -> int:
    """The exit contribution of one file outcome."""
    check_primary(status, reason)
    return reason_spec(reason).exit_code


def primary_warning(codes: Iterable[str]) -> ReasonCode | None:
    """The first warning by C-13 precedence, or None."""
    present = set(codes)
    for code in WARNING_PRECEDENCE:
        if code in present:
            return code
    return None


def settle_primary(status: Status, reason: ReasonCode,
                   warnings: Iterable[Diagnostic]) -> tuple[Status, ReasonCode]:
    """Promote an otherwise OK result carrying warnings to WARN (C-13).

    Failures and skips keep their primary reason; warnings stay diagnostics.
    """
    check_primary(status, reason)
    if status != "ok":
        return status, reason
    first = primary_warning(item.code for item in warnings)
    if first is None:
        return status, reason
    return "warn", first


# Late executable loss/change after preflight fails only the affected file, by
# stage (C-13; ARCHITECTURE § 7). It is never a late exit 3.
TOOL_LOSS_REASONS: FrozenMap = FrozenMap({
    "signature": "probe_error",
    "probe": "probe_error",
    "decode": "decode_error",
    "measure": "decode_error",
    "verify": "decode_error",
    "render": "render_error",
    # CR-079: ``sips`` lost or changed after preflight while artwork is prepared (metadata
    # MD24; ``metadata/artwork.py`` stage). C-13 has no image-tool reason, and the file's
    # artwork cannot be prepared, so it is ``artwork_invalid`` with the tool-loss detail.
    "metadata_prepared": "artwork_invalid",
})


def tool_loss_reason(stage: str) -> ReasonCode:
    try:
        return TOOL_LOSS_REASONS[stage]
    except KeyError:
        raise ValueError(f"no tool-loss mapping for stage {stage!r}") from None


# --- Exceptions (ARCHITECTURE § 7) -------------------------------------------------


class NormaError(Exception):
    """Base for every expected Norma error."""


class UsageError(NormaError):
    """Usage/config/whole-manifest/explicit capability error: exit 2 before writes."""

    exit_code = EXIT_USAGE


class EnvironmentError_(NormaError):  # noqa: N801 - name fixed by ARCHITECTURE § 7
    """Missing or unusable required environment in run preflight: exit 3."""

    exit_code = EXIT_ENVIRONMENT


DoctorError = EnvironmentError_


class FileOutcomeError(NormaError):
    """A per-file terminal result carried as an exception; subclasses fix the status family."""

    status: Status

    def __init__(self, reason: ReasonCode, stage: str, detail: str | None = None,
                 diagnostics: tuple[Diagnostic, ...] = ()) -> None:
        spec = reason_spec(reason)
        if self.status not in spec.statuses:
            raise ValueError(f"{type(self).__name__} cannot carry reason {reason!r} ({spec.statuses})")
        self.reason: ReasonCode = reason
        self.stage = stage
        self.detail = detail
        self.diagnostics = tuple(diagnostics)
        super().__init__(f"{reason} at {stage}" + (f": {detail}" if detail else ""))

    def diagnostic(self) -> Diagnostic:
        return Diagnostic(self.reason, str(self), self.stage, self.detail)


class SkipFile(FileOutcomeError):
    status = "skipped"


class FileFailure(FileOutcomeError):
    status = "failed"


class Cancelled(BaseException):  # noqa: N818 - control signal, not an error
    """Cancellation control signal; not caught by workers' ``except Exception``."""


class ToolError(NormaError):
    """An external tool call that did not succeed; edge modules map it by stage."""

    def __init__(self, message: str, *, argv: tuple[str, ...] = (), stderr_tail: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.argv = tuple(argv)
        self.stderr_tail = tuple(stderr_tail)

    def to_failure(self, reason: ReasonCode, stage: str) -> FileFailure:
        """The C-13 failure for this error at ``stage``; ``reason`` covers ordinary tool failure."""
        detail = str(self)
        if self.stderr_tail:
            detail += "\n" + "\n".join(self.stderr_tail)
        return FileFailure(reason, stage, detail)


class ToolTimeout(ToolError):
    """The per-call deadline expired; the process group was terminated and reaped."""

    def __init__(self, message: str, *, deadline_s: float, **kwargs: tuple[str, ...]) -> None:
        super().__init__(message, **kwargs)
        self.deadline_s = deadline_s

    def to_failure(self, reason: ReasonCode, stage: str) -> FileFailure:
        return super().to_failure("timeout", stage)


class ToolMissing(ToolError):
    """The executable disappeared or changed identity after preflight."""

    def to_failure(self, reason: ReasonCode, stage: str) -> FileFailure:
        return super().to_failure(tool_loss_reason(stage), stage)


class ToolFailed(ToolError):
    """The tool ran and exited non-zero."""

    def __init__(self, message: str, *, returncode: int, **kwargs: tuple[str, ...]) -> None:
        super().__init__(message, **kwargs)
        self.returncode = returncode
