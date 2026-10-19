"""Per-file orchestration: the read-only commands, ``doctor`` and ``normalise`` (S1–S5).

Sequences the owners' public functions along ARCHITECTURE § 4. Every started
file gets one C-09 ``FileOutcome``; a per-file skip or failure never stops the
run, and files never started are counted, never given invented outcomes.

Read-only commands (S1, D-057), ``execution="inspect"``:

- ``analyse``: discovered → signature_checked/probed → measured → gain_planned;
- ``verify`` (S5): … → tags_read → per-file reference → measured → optional tag checks;
- ``tags show``: discovered → signature_checked/probed → tags_read;
- ``tags plan`` (S4, TK-0427, ``execution="plan"``): … → tags_read → a provisional format
  choice → metadata_prepared → C-07/C-08 plans, with MD35's conditional alternatives.

Analyse and verify accept a ReadOnlyBatchRequest executor for bounded jobs; direct
callers without it retain sequential execution. They write nothing: no output,
temporary or log file is opened. The exception is an explicitly requested report
(safety § 2.7): analyse_items and verify_items check it against every input,
the config and the tools before the first file (verify includes its manifest
and effective artwork references). Neither publishes it; the interface publishes rendered bytes after
the run through ``ReadOnlyRun.outputs`` (see *Run outputs* below).
Cancellation is a ``threading.Event`` set by the caller (for example from a
Ctrl-C handler): the file in progress ends ``skipped/cancelled``.
``KeyboardInterrupt`` is not caught here; the interface owns it.

``normalise`` (S2, TK-0412; S3, TK-0419; S4 metadata, TK-0427) runs the write path of § 4
through ``normalise_items``: all-file planning (signature/probe, tags and the processing
mark, C-04 preflight, metadata inputs and ``sips`` preflight, metadata_prepared,
measurement, explicit ``dynamic`` calibration, gain plan, final C-04, the C-07 metadata
plan, C-08 paths), the WS-05 run write set, then one worker per
eligible file. The worker drives WS-01's bounded retry ladder
(``verify.run_ladder``, ADR-0014): each attempt allocates a registered
candidate, renders from the original source, checks its structure and verifies
it; a rejected candidate is removed before the next allocation, and each retry
first passes V3a's planning states (final C-04, a new C-07 revision when the basis,
target or action changes (MD34), a retry reservation when the C-08 bundle changes). The
accepted candidate gets its final mark revision, its tags written and read back by WS-06,
and a decoded-sample identity check of the tagged bytes (``content_verified``); then it is
published or rolled back. Every action (``copy``, ``linear``, ``limit``,
``reduce``, calibrated ``dynamic``) and every evidenced target (lossless, MP3,
AAC, stream copy) renders; the caller's batch executor runs the workers.
``pipeline`` never imports ``batch`` (§ 3): the caller passes an ``execute``
callable and receives a ``BatchRequest`` holding the entries, the worker, the
error settler, the registry's commit lock and the force-kill hook (CR-045,
CR-047, D-065). ``--dry-run`` stops after the write set and
writes nothing beyond the C-04 evidence encodes described below; it runs no
calibration (audio V8), using the read-only dynamic preview plan
(C-05 v1.3, D-080), as ``analyse`` does; its metadata plan keeps the mark pending (MD26).
The D-066 tagged-input refusal is lifted in S4: non-tag items and unstorable values are
accounted losses under ``--on-metadata-loss`` (MD7, MD21), never silent.

No FFmpeg filter string or tag object is built or inspected here (§ 3), and no
file is opened, created or removed here: candidates, folders and publication
are ``safety``'s, renders ``audio/render``'s. The only files this module causes
outside the run's destinations are the capability collectors' tiny encodes
(``doctor``, D-059, and ``normalise`` C-04 evidence), inside one system
temporary folder removed before the call continues, and ``metadata/artwork.py``'s private
``sips`` scratch copies (MD24), inside one system temporary folder removed at the end of
the run. Tags are written into the private candidate by ``metadata/write.py``.

Interface code dispatches through this module rather than calling adapters or
domain-edge modules itself (§ 3), so tool location, discovery, the force-kill
hook, report publication and the log are exposed here as thin pass-throughs.

``doctor`` gets one entry point, ``inspect_environment`` (TK-0411, D-059): it
gathers every fact the interface prints for the ``doctor`` rows (Releases § 5,
cli § 2) as a frozen ``EnvironmentInspection``. Each check holds a value, a typed
``Problem`` or both; a missing or broken tool never raises, and OK/WARN/FAIL and
exit 3 stay the interface's decisions. ``mutagen`` is imported only when tags
are read (CR-040), so ``doctor`` loads and reports E-03 without it.
"""

from __future__ import annotations

import platform
import secrets
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as evolve
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

import norma
from norma import safety
from norma.adapters.ffmpeg import (
    ALIMITER_OPTIONS,
    DYNAMIC_FILTERS,
    MEASURE_FILTERS,
    RENDER_FILTERS,
    Executable,
    FFmpegEnvironment,
    check_requirements,
    collect_aac_evidence,
    collect_lossless_evidence,
    collect_mp3_evidence,
    filter_options,
    force_kill,
    locate_ffmpeg,
    platform_basis,
    probe_environment,
    probe_mp3_encoder_stats,
    resolve_executable,
)
from norma.adapters.ffprobe import Prober, locate_ffprobe
from norma.adapters.ffprobe import force_kill as force_kill_ffprobe
from norma.adapters.ffprobe import probe as probe_file
from norma.audio.measure import calibrate_dynamic, measure
from norma.audio.plan import DynamicCalibration, needs_dynamic_calibration, plan_dynamic_preview, plan_gain
from norma.audio.render import render
from norma.audio.verify import (
    RetryKind, assess_inspection, decoded_digest, measure_candidate, run_ladder,
    select_reference, verify_candidate,
)
from norma.discovery import DiscoveryResult, classify, discover
from norma.errors import (
    Cancelled,
    EnvironmentError_,
    FileFailure,
    FileOutcomeError,
    NormaError,
    SkipFile,
    ToolError,
    UsageError,
    check_primary,
    reason_spec,
    settle_primary,
)
from norma.formats.capabilities import (
    TARGETS,
    Capability,
    ToolBasis,
    automatic_lossy_rate,
    automatic_mp3_capability,
    lossy_capability,
    same_target_format,
)
from norma.formats.resolve import effective_channel_layout, finalise_target, preflight_targets
from norma.models import (
    LOSSLESS_FORMATS,
    CapabilityEvidence,
    ConditionalPlan,
    ConditionalRefusal,
    Diagnostic,
    Execution,

    FileOutcome,
    GainPlan,
    InputItem,
    LoudnessMeasurement,
    ManifestCell,
    MetadataPlan,

    OutputPlan,
    ProbeResult,
    ProcessingMark,
    RawNativeItem,
    ReadDependency,
    ReasonCode,
    RenderResult,
    RunOptions,
    RunWriteSet,
    SourceTags,
    Status,
    TagWriteResult,
    TargetFormat,
    Verification,
    WriteDestination,
)
from norma.paths import NamingOptions, comparison_key, plan_output, rebind


# --- Tag reading, imported lazily (CR-040) ----------------------------------------------------


def read_source_tags(path: Path, container: str) -> SourceTags:
    """``metadata.read.read_source_tags``, imported on first use.

    ``metadata.read`` imports mutagen at load time; importing it lazily keeps
    ``import norma.pipeline`` (and so ``doctor``) working when mutagen is missing,
    which ``doctor`` must report as E-03 rather than crash on (CR-040).
    """
    from norma.metadata.read import read_source_tags as read  # noqa: PLC0415 - CR-040

    return read(path, container)


def read_processing_mark(tags: SourceTags) -> ProcessingMark:
    """``metadata.read.read_processing_mark``, imported on first use (CR-040)."""
    from norma.metadata.read import read_processing_mark as read  # noqa: PLC0415 - CR-040

    return read(tags)


# --- Records returned to the interface -------------------------------------------------------


@dataclass(frozen=True)
class ReadOnlyTools:
    """Executables resolved once; ``ffmpeg`` is needed by ``analyse`` and ``verify``."""

    prober: Prober
    ffmpeg: Executable | None = None


@dataclass(frozen=True)
class InspectionReference:
    """Effective explicit tag assertion; actual evidence is in Inspection.source_tags.

    tag_mismatches uses WS-06's field:/extra: keys and extra:NORMA for absence.
    Typed expected values, clears and origins are retained for cli § 10.2.
    """

    name: str
    cell: ManifestCell
    origin: Literal["cli", "manifest", "manifest_default"]
    locator: str


@dataclass(frozen=True)
class Inspection:
    """One started file: its C-09 outcome plus the read-only records the interface presents.

    ``primary`` is the diagnostic behind a skip or failure (stage and detail), which C-09
    ``FileOutcome`` has no field for; it is None for ok/warn outcomes, whose warnings are in
    ``outcome.warnings``. ``source_tags`` (C-06) and ``mark`` (C-09 ``ProcessingMark``) are
    filled by ``tags show`` once tags were read, and are None otherwise. ``tags plan`` fills
    them too, and adds ``diagnostics`` (a branch's conditional refusal, C-09) and ``notes``
    (resolution, inference and artwork notes, MD14, MD24).
    """

    outcome: FileOutcome
    primary: Diagnostic | None = None
    source_tags: SourceTags | None = None
    mark: ProcessingMark | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    notes: tuple[str, ...] = ()
    tag_references: tuple[InspectionReference, ...] = ()
    tag_mismatches: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadOnlyRun:
    """Input-ordered outcomes and what cancellation left undone.

    ``outputs`` publishes an explicit report checked before the first file;
    it is None when no report was requested (and always for ``tags show``).
    """

    inspections: tuple[Inspection, ...]
    not_started: int
    cancelled: bool
    outputs: RunOutputs | None = None
    notes: tuple[str, ...] = ()
    """Run-level notes (``tags plan``: unmatched manifest rows, likely typos)."""

    @property
    def outcomes(self) -> tuple[FileOutcome, ...]:
        return tuple(inspection.outcome for inspection in self.inspections)


OnInspection = Callable[[Inspection], None]


@dataclass(frozen=True)
class ReadOnlyBatchRequest:
    """Read-only inversion-of-control seam; no writes or scheduler in pipeline.

    The executor drains workers, returns input-ordered outcomes and delivers
    on_outcome on its coordinator (never a worker). For analyse and verify::

        def execute(request):
            control = batch.BatchControl(commit_lock=request.commit_lock,
                                         cancel_event=request.cancel_event,
                                         force_hook=request.force_hook)
            return batch.execute_batch(request.entries, request.worker,
                settle_error=request.settle_error, jobs=request.jobs, control=control,
                on_outcome=request.on_outcome)

    record_for returns the exact cached Inspection, including secondary checks.
    All references and explicit-report barriers precede this request. Direct
    callers may omit execute to retain the original sequential behaviour.
    """

    entries: tuple[InputItem, ...]
    worker: Callable[[InputItem, Any], FileOutcome]
    settle_error: Callable[[InputItem, BaseException, float, Any], FileOutcome]
    jobs: int
    commit_lock: Any
    cancel_event: threading.Event
    force_hook: Callable[[], None]
    record_for: Callable[[FileOutcome], Inspection]
    on_outcome: Callable[[FileOutcome], None]


ReadOnlyExecute = Callable[[ReadOnlyBatchRequest], "BatchResultLike"]


# --- Run preflight pass-throughs -------------------------------------------------------------


def locate_tools(options: RunOptions, *, need_ffmpeg: bool) -> ReadOnlyTools:
    """Resolve ffprobe from the folder of C-10 ``ffmpeg`` (CLI § 3.5) and, if asked, FFmpeg.

    With ``need_ffmpeg`` the FFmpeg filters that measurement runs are required too. Any
    missing or unusable tool or facility raises ``EnvironmentError_`` (exit 3, before any
    file is touched). Facilities a read-only command does not use are not required
    (ARCHITECTURE § 9).
    """
    prober = locate_ffprobe(options.ffmpeg.parent / "ffprobe")
    if not need_ffmpeg:
        return ReadOnlyTools(prober)
    executable = locate_ffmpeg(options.ffmpeg)
    check_requirements(probe_environment(executable), filters=MEASURE_FILTERS)
    return ReadOnlyTools(prober, executable)


def discover_inputs(options: RunOptions, *, suffix: str = "", output_dir: Path | None = None) -> DiscoveryResult:
    """Expand C-10 ``inputs`` into C-01 items (safety § 2; ``UsageError`` for bad arguments).

    ``suffix`` and ``output_dir`` are the folder-expansion exclusions that apply to the
    command; which ones apply is the interface's command matrix, so none is assumed here.
    """
    return discover(options.inputs, recursive=options.recursive, suffix=suffix, output_dir=output_dir)


# --- doctor: environment inspection (TK-0411) ------------------------------------------------

ProblemCode = Literal["not_found", "unusable", "missing", "query_failed", "not_checked", "not_importable",
                      "no_procedure", "cleanup_failed"]
LossyTarget = Literal["mp3", "aac"]


@dataclass(frozen=True)
class Problem:
    """What a ``doctor`` check found wrong. A run-level fact, never a C-13 reason code.

    - ``not_found``: the tool is not at the given path or on PATH (E-01, E-05; ffprobe beside ffmpeg);
    - ``unusable``: the tool was found but did not run or gave no version;
    - ``missing``: the listing ran but lacks the facilities named in ``missing`` (E-02);
    - ``query_failed``: a listing, option or evidence query could not run to completion;
    - ``not_checked``: an earlier check this one depends on failed, so it was not attempted;
    - ``not_importable``: a Python dependency cannot be found (E-03);
    - ``no_procedure``: no installed-query evidence procedure exists for this encoder yet, so its
      targets are an explicit, conservative C-04 gap;
    - ``cleanup_failed``: the temporary evidence folder could not be removed.
    """

    code: ProblemCode
    message: str
    missing: tuple[str, ...] = ()


T = TypeVar("T")


@dataclass(frozen=True)
class Check(Generic[T]):
    """One ``doctor`` fact: a value, a problem, or both (a listing that lacks something)."""

    value: T | None = None
    problem: Problem | None = None

    def __post_init__(self) -> None:
        if self.value is None and self.problem is None:
            raise ValueError("a check needs a value, a problem or both")

    @property
    def ok(self) -> bool:
        return self.problem is None


@dataclass(frozen=True)
class ToolInfo:
    """A located tool that ran, as found (not symlink-resolved), and its version token."""

    path: Path
    version: str


@dataclass(frozen=True)
class Facilities:
    """What a listing reported, against what Norma requires of it."""

    present: frozenset[str]
    required: tuple[str, ...]

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(name for name in self.required if name not in self.present)


@dataclass(frozen=True)
class RuntimeInfo:
    norma_version: str
    python_version: str
    python_executable: Path


@dataclass(frozen=True)
class PlatformInfo:
    """``system`` and ``machine`` from the platform module; ``basis`` is the C-04 platform basis."""

    system: str
    release: str
    machine: str
    basis: str


@dataclass(frozen=True)
class LossyEvidenceRow:
    """C-04 evidence for one lossy target at one rate and channel layout, and what it allows.

    ``capabilities`` holds one evidence intersection per bitrate mode the target offers;
    ``automatic`` is what an automatic output there needs (MP3: evidenced V0, F15; AAC: CBR).
    When ``automatic.supported`` is False, an automatic output there would fail
    ``unsupported_target``. ``problem`` says why no evidence was collected, if so.
    """

    target: LossyTarget
    encoder: str
    sample_rate_hz: int
    channels: int
    channel_layout: str
    records: tuple[CapabilityEvidence, ...]
    capabilities: tuple[Capability, ...]
    automatic: Capability
    problem: Problem | None = None

    @property
    def supported(self) -> bool:
        return self.automatic.supported

    @property
    def bases(self) -> tuple[str, ...]:
        return tuple(sorted({record.basis for record in self.records}))

    @property
    def gaps(self) -> tuple[str, ...]:
        found: list[str] = []
        for capability in (*self.capabilities, self.automatic):
            found.extend(gap for gap in capability.gaps if gap not in found)
        return tuple(found)


@dataclass(frozen=True)
class EnvironmentInspection:
    """Everything the ``doctor`` rows need (Releases § 5), without the interface touching adapters.

    Rows 12 (config file) and 13 (whether ``--ffmpeg`` was given) are the interface's own
    facts; row 13's existence and executability are ``ffmpeg``'s ``not_found`` problem.
    """

    runtime: RuntimeInfo
    platform: PlatformInfo
    ffmpeg: Check[ToolInfo]
    ffprobe: Check[ToolInfo]
    encoders: Check[Facilities]
    filters: Check[Facilities]
    alimiter_options: Check[Facilities]
    mutagen: Check[str]
    sips: Check[Path]
    capability_evidence: Check[tuple[LossyEvidenceRow, ...]]


DOCTOR_FILTERS: tuple[str, ...] = tuple(dict.fromkeys(MEASURE_FILTERS + RENDER_FILTERS + DYNAMIC_FILTERS))
"""Every filter a Norma command runs (ARCHITECTURE § 9); ``doctor`` reports the full set."""
DOCTOR_ENCODERS: tuple[str, ...] = tuple(dict.fromkeys(
    encoder for spec in TARGETS.values() for encoder in spec.encoders))
"""Every encoder an output target uses (formats § 4 policy table): PCM, flac, native alac, libmp3lame, aac_at."""
LOSSY_TARGETS: tuple[LossyTarget, ...] = ("mp3", "aac")
"""The lossy targets of the formats § 4 policy table, each needing C-04 evidence."""
EVIDENCE_RATES_HZ: tuple[int, ...] = (44_100, 48_000)
"""Rates probed for lossy evidence: the two rates every lossy output lands on (F4/F5)."""
EVIDENCE_LAYOUTS: tuple[tuple[str, int], ...] = (("mono", 1), ("stereo", 2))
"""Layouts probed for lossy evidence: v1 handles mono and stereo only (ADR-0020)."""

def _collect_aac_rows(environment: FFmpegEnvironment, rate: int, channels: int, layout: str | None,
                     work: Path) -> tuple[CapabilityEvidence, ...]:
    """``collect_aac_evidence`` (TK-0418) shaped like the MP3 collector: a tuple of records."""
    return (collect_aac_evidence(environment, rate, channels, layout, work),)


_EVIDENCE_COLLECTORS = {"libmp3lame": collect_mp3_evidence, "aac_at": _collect_aac_rows}
"""Installed-query evidence procedures by encoder; an encoder without one is an explicit gap."""
_QUERY_ERRORS = (NormaError, OSError, ValueError)


def inspect_environment(ffmpeg: Path | str = "ffmpeg", *, evidence_rates_hz: Sequence[int] = EVIDENCE_RATES_HZ,
                        evidence_layouts: Sequence[tuple[str, int]] = EVIDENCE_LAYOUTS) -> EnvironmentInspection:
    """``doctor``: inspect tools, facilities and capability evidence; never raise for a missing tool.

    ``ffmpeg`` is what the user gave (``--ffmpeg`` or the config key, which must be absolute)
    or the bare name ``ffmpeg``, looked up on PATH as the cli § 3.5 default. It is not a
    ``RunOptions``: C-10 ``ffmpeg`` is an already resolved absolute path, which cannot exist
    when FFmpeg is missing, and that is exactly the case ``doctor`` must report. ffprobe is
    taken from the folder FFmpeg was found in (cli § 3.5). Lossy capability evidence is
    collected with the adapter's installed queries in one system temporary folder, removed
    before returning.
    """
    ffmpeg_check, executable, ffmpeg_path = _inspect_ffmpeg(ffmpeg)
    environment, encoders, filters = _inspect_features(executable)
    return EnvironmentInspection(
        runtime=RuntimeInfo(norma.__version__, platform.python_version(), Path(sys.executable)),
        platform=PlatformInfo(platform.system(), platform.mac_ver()[0] or platform.release(), platform.machine(),
                              platform_basis()),
        ffmpeg=ffmpeg_check,
        ffprobe=_inspect_ffprobe(ffmpeg_path, Path(ffmpeg)),
        encoders=encoders,
        filters=filters,
        alimiter_options=_inspect_alimiter(environment),
        mutagen=_inspect_mutagen(),
        sips=_inspect_sips(),
        capability_evidence=_inspect_evidence(environment, tuple(evidence_rates_hz), tuple(evidence_layouts)),
    )


def _not_checked(why: str) -> Check:
    return Check(problem=Problem("not_checked", why))


def _inspect_ffmpeg(name: Path | str) -> tuple[Check[ToolInfo], Executable | None, Path | None]:
    try:
        path = resolve_executable(name)
    except _QUERY_ERRORS as exc:
        return Check(problem=Problem("not_found", str(exc))), None, None
    try:
        executable = locate_ffmpeg(path)
    except _QUERY_ERRORS as exc:
        return Check(problem=Problem("unusable", str(exc))), None, path
    return Check(ToolInfo(executable.path, executable.version)), executable, path


def _inspect_ffprobe(ffmpeg_path: Path | None, given: Path) -> Check[ToolInfo]:
    folder = ffmpeg_path.parent if ffmpeg_path is not None else (given.parent if given.is_absolute() else None)
    if folder is None:
        return _not_checked("ffprobe is taken from the folder of ffmpeg, which was not found")
    path = folder / "ffprobe"
    try:
        resolve_executable(path)
    except _QUERY_ERRORS as exc:
        return Check(problem=Problem("not_found", f"ffprobe not found beside ffmpeg in {folder}: {exc}"))
    try:
        prober = locate_ffprobe(path)
    except _QUERY_ERRORS as exc:
        return Check(problem=Problem("unusable", str(exc)))
    return Check(ToolInfo(prober.path, prober.version))


def _facilities(present: frozenset[str], required: tuple[str, ...], kind: str) -> Check[Facilities]:
    value = Facilities(frozenset(present), required)
    missing = value.missing
    if not missing:
        return Check(value)
    return Check(value, Problem("missing", f"FFmpeg lacks required {kind}: {', '.join(missing)}", missing))


def _inspect_features(executable: Executable | None,
                      ) -> tuple[FFmpegEnvironment | None, Check[Facilities], Check[Facilities]]:
    if executable is None:
        skipped = _not_checked("FFmpeg was not found or did not run")
        return None, skipped, skipped
    try:
        environment = probe_environment(executable)
    except _QUERY_ERRORS as exc:
        failed: Check[Facilities] = Check(problem=Problem("query_failed", str(exc)))
        return None, failed, failed
    return (environment, _facilities(environment.encoders, DOCTOR_ENCODERS, "encoders"),
            _facilities(environment.filters, DOCTOR_FILTERS, "filters"))


def _inspect_alimiter(environment: FFmpegEnvironment | None) -> Check[Facilities]:
    if environment is None:
        return _not_checked("FFmpeg's filters could not be listed")
    if "alimiter" not in environment.filters:
        return _not_checked("the alimiter filter is absent")
    try:
        present = filter_options(environment.executable, "alimiter")
    except _QUERY_ERRORS as exc:
        return Check(problem=Problem("query_failed", f"cannot list alimiter options: {exc}"))
    return _facilities(present, ALIMITER_OPTIONS, "alimiter options")


def _inspect_mutagen() -> Check[str]:
    """Version from distribution metadata; mutagen itself is imported only by metadata read/write."""
    try:
        found = distribution_version("mutagen")
    except PackageNotFoundError:
        return Check(problem=Problem("not_importable", "mutagen is not installed"))
    try:
        spec = find_spec("mutagen")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return Check(problem=Problem("not_importable", f"mutagen {found} is recorded as installed but cannot be "
                                                       "found on the import path"))
    return Check(found)


def _inspect_sips() -> Check[Path]:
    try:
        return Check(resolve_executable("sips"))
    except _QUERY_ERRORS as exc:
        return Check(problem=Problem("not_found", str(exc)))


def _inspect_evidence(environment: FFmpegEnvironment | None, rates: tuple[int, ...],
                      layouts: tuple[tuple[str, int], ...]) -> Check[tuple[LossyEvidenceRow, ...]]:
    if environment is None:
        return _not_checked("FFmpeg's encoders could not be listed")
    try:
        folder = tempfile.TemporaryDirectory(prefix="norma-doctor-")
    except OSError as exc:
        return Check(problem=Problem("query_failed", f"cannot create a temporary folder for evidence: {exc}"))
    executable = environment.executable
    basis = ToolBasis(executable.path, executable.identity, executable.version, environment.platform_basis)
    rows: list[LossyEvidenceRow] = []
    cleanup: Problem | None = None
    stats: bool | None = None  # CR-067/CR-070: probed once, a failure too (recorded as unavailable)
    if "libmp3lame" in environment.encoders:
        try:
            stats = probe_mp3_encoder_stats(environment)
        except _QUERY_ERRORS:
            stats = False  # CR-070: a failed probe is recorded once; every MP3 row stays unavailable
    try:
        for target in LOSSY_TARGETS:
            for rate in rates:
                for layout, channels in layouts:
                    rows.append(_evidence_row(environment, basis, target, rate, channels, layout, Path(folder.name), stats))
    finally:
        try:
            folder.cleanup()
        except OSError as exc:
            cleanup = Problem("cleanup_failed", f"cannot remove the temporary evidence folder {folder.name}: {exc}")
    return Check(tuple(rows), cleanup)


def _evidence_row(environment: FFmpegEnvironment, basis: ToolBasis, target: LossyTarget, rate: int, channels: int,
                  layout: str, work: Path, mp3_stats: bool | None = None) -> LossyEvidenceRow:
    spec = TARGETS[target]
    encoder = spec.encoders[0]
    records: tuple[CapabilityEvidence, ...] = ()
    problem: Problem | None = None
    collector = _EVIDENCE_COLLECTORS.get(encoder)
    if encoder not in environment.encoders:
        problem = Problem("missing", f"FFmpeg lacks encoder {encoder}", (encoder,))
    elif collector is None:
        problem = Problem("no_procedure", f"no installed-query evidence procedure for {encoder} exists in this "
                                          f"version, so {target} outputs are refused conservatively (C-04, ADR-0015)")
    else:
        try:
            folder = work / f"{encoder}-{rate}-{layout}"
            extra = {"encoder_stats": mp3_stats} if encoder == "libmp3lame" and mp3_stats is not None else {}
            records = tuple(collector(environment, rate, channels, layout, folder, **extra))
        except _QUERY_ERRORS as exc:
            problem = Problem("query_failed", f"{encoder} evidence at {rate} Hz {layout} could not be collected: {exc}")
    capabilities = tuple(lossy_capability(target, mode, records, basis, sample_rate_hz=rate, channels=channels,
                                          channel_layout=layout) for mode in spec.bitrate_modes)
    if target == "mp3":
        automatic = automatic_mp3_capability(records, basis, sample_rate_hz=rate, channels=channels,
                                             channel_layout=layout)
    else:  # automatic AAC is a CBR setting (ADR-0015)
        automatic = next(capability for capability in capabilities if capability.bitrate_mode == "cbr")
    return LossyEvidenceRow(target, encoder, rate, channels, layout, records, capabilities, automatic, problem)


# --- Run outputs: report and log requests (TK-0413, D-068) ----------------------------------

RunOutputRequest = safety.RunOutput
"""A run-level destination request: purpose, absolute path, whether its folder must exist.

The interface builds these with ``report_request`` and ``log_request`` (it may not import
``safety``, ARCHITECTURE § 3). A request carries no overwrite authority: explicit reports
use C-10 ``RunOptions.overwrite`` at the barrier and publication, while automatic reports
always take an ordinal and never replace a taken name (D-069, CR-049).
"""

RunOutputError = safety.RunOutputError
"""A report could not be published after the run: a run error, exit 1, no file row (N-07)."""


def report_request(path: Path | str, *, automatic: bool = False) -> RunOutputRequest:
    """A ``report`` request for ``path``, which must be absolute (``ValueError`` otherwise).

    ``automatic`` is the ``normalise`` automatic report (cli § 7.3, D-018): its folder may be
    created after the barrier and a taken name gets an ordinal at publication, irrespective
    of ``RunOptions.overwrite`` (D-069). An explicit
    ``--report`` (the default) needs an existing folder; a missing one is usage exit 2.
    The ``.csv``/``.json`` extension rule is the interface's argument check, not made here.
    """
    return RunOutputRequest("report", Path(path), parent_must_exist=not automatic)


def log_request(path: Path | str) -> RunOutputRequest:
    """A ``normalise`` ``--log`` request: appended to, existing folder required (safety § 6)."""
    return RunOutputRequest("log", Path(path))


class RunOutputs:
    """Run-level writes the interface needs and may not reach itself (§ 3): report and log.

    Both go through the checked run write set (safety § 6): only destinations the barrier
    reserved can be written, with the same protected reads. A report published after the run
    uses the run's own registry, so its private candidate is registered and removed like any
    other.

    ``publish`` takes its defaults from the request: an automatic report may create its folder
    and takes an ordinal when its name is taken; an explicit report's folder must still be the
    folder the barrier pinned, so a folder removed or replaced since is a ``RunOutputError``
    and is never re-created.
    """

    def __init__(self, checked: safety.Preflight, protected: tuple[ReadDependency, ...],
                 registry: safety.RunRegistry, *, overwrite: bool,
                 requests: Sequence[RunOutputRequest] = ()) -> None:
        self._checked = checked
        self._protected = protected
        self._registry = registry
        self._overwrite = overwrite
        self._requests = {request.purpose: request for request in requests}

    @property
    def write_set(self) -> RunWriteSet:
        return self._checked.write_set

    def destination(self, purpose: str) -> WriteDestination | None:
        return next((d for d in self._checked.write_set.destinations
                     if d.input_index is None and d.purpose == purpose), None)

    def publish(self, purpose: Literal["report", "template_export"], data: bytes, *,
                ordinal_retry: bool | None = None, create_parent: bool | None = None) -> Path:
        """Publish a reserved report or template file; ``RunOutputError`` (exit 1) on failure.

        Returns the path actually written, which for an automatic report may carry an ordinal.
        """
        destination = self.destination(purpose)
        if destination is None:
            raise ValueError(f"no {purpose} destination was reserved for this run")
        request = self._requests.get(purpose)
        automatic = request is not None and not request.parent_must_exist
        ordinal_retry = automatic if ordinal_retry is None else ordinal_retry
        create_parent = automatic if create_parent is None else create_parent
        if not create_parent:
            self._recheck_parent(destination)
        reserved = tuple(d.comparison_key for d in self._checked.write_set.destinations)
        return safety.publish_run_output(self._registry, destination, data,
                                         overwrite=self._overwrite and not automatic,
                                         protected=self._protected, reserved_keys=reserved,
                                         ordinal_retry=ordinal_retry, create_parent=create_parent,
                                         pin=self._checked.pins.get(destination.destination_id))

    def _recheck_parent(self, destination: WriteDestination) -> None:
        """The pinned folder still exists and is the same folder (safety § 2.6 (b), § 6)."""
        pin = self._checked.pins.get(destination.destination_id)
        final = destination.final_path
        try:
            current = safety.recheck_pin(pin) if pin is not None else safety.pin_parent(final)
        except safety.UnsafeAncestry as exc:
            raise RunOutputError(f"cannot write {final}: {exc}") from None
        if not current.exists:
            raise RunOutputError(f"cannot write {final}: its folder {final.parent} no longer exists")

    def open_log(self) -> int:
        """Open the reserved ``normalise`` log for appending (``UsageError`` before any byte)."""
        destination = self.destination("log")
        if destination is None:
            raise ValueError("no log destination was reserved for this run")
        occupant = self._checked.occupants.get(destination.destination_id, safety.Occupant("absent", None, 0))
        return safety.open_log(destination, occupant, self._protected)


def _protected_reads(items: Sequence[InputItem], options: RunOptions,
                     executables: Sequence[Path]) -> tuple[ReadDependency, ...]:
    """Every C-01 input, the tools the run invokes and the config actually read (safety § 2.6)."""
    protected = list(safety.protect_inputs(items))
    protected += [safety.protect(path, "tool") for path in dict.fromkeys(executables)]
    if options.config is not None and options.config.is_file():
        protected.append(safety.protect(options.config, "config"))
    return tuple(protected)


def check_read_only_report(items: Sequence[InputItem], options: RunOptions, tools: ReadOnlyTools,
                           request: RunOutputRequest, *, run_id: str | None = None) -> RunOutputs:
    """The barrier for a read-only command's explicit ``--report`` (safety §§ 2.6, 2.7, 6).

    ``items`` are every discovered input, whatever their later outcome; they, the config and
    the located tools are protected. No file is probed, measured or planned: there is no
    tagged-input refusal, format resolution or output planning here, and nothing is written.
    An existing report without ``options.overwrite``, a missing folder, an alias of a
    protected read, a link, folder or special file is ``UsageError`` (exit 2, no rows).
    A ``log`` request is refused (``ValueError``): read-only commands ignore it with N-08.
    """
    if request.purpose != "report" or not request.parent_must_exist:
        raise ValueError("a read-only command takes only an explicit report (report_request(path))")
    executables = [tool.path for tool in (tools.ffmpeg, tools.prober) if tool is not None]
    protected = _protected_reads(items, options, executables)
    return _check_plan_report(options, request, protected, {}, run_id=run_id)


def _check_plan_report(options: RunOptions, request: RunOutputRequest,
                       protected: tuple[ReadDependency, ...], plans: Mapping[int, OutputPlan], *,
                       run_id: str | None = None) -> RunOutputs:
    """Reserve a report against all read dependencies and documentary bundle members."""
    if request.purpose != "report" or not request.parent_must_exist:
        raise ValueError("a read-only command takes only an explicit report (report_request(path))")
    checked = safety.preflight(plans, (request,), protected, overwrite=options.overwrite)
    registry = safety.RunRegistry(run_id or new_run_id())
    return RunOutputs(checked, protected, registry, overwrite=options.overwrite, requests=(request,))


# --- Commands --------------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateRun:
    """Checked ``tags template`` result for the interface (MD28, CR-089).

    ``text`` is CSV for stdout or the requested file form. ``path`` is set only
    after publication; ``run_errors`` names a failed publication, never a file row.
    No metadata or safety edge object is needed by the interface.
    """

    inspections: tuple[Inspection, ...]
    not_started: int
    cancelled: bool
    text: str = ""
    path: Path | None = None
    run_errors: tuple[str, ...] = ()

    @property
    def outcomes(self) -> tuple[FileOutcome, ...]:
        return tuple(inspection.outcome for inspection in self.inspections)


def template_tags_items(items: Sequence[InputItem], options: RunOptions, tools: ReadOnlyTools, *,
                        cancel_event: threading.Event | None = None,
                        on_inspection: OnInspection | None = None,
                        run_id: str | None = None) -> TemplateRun:
    """Read source tags, optionally infer, then export a checked manifest (MD28).

    Uses only source values and filename inference, never field options, config
    metadata or manifests. A file destination needs an existing folder and passes
    the full write barrier before any callback. ``-`` returns CSV relative to cwd
    and invokes no write API. No FFmpeg, measurement or artwork preparation occurs.
    """
    from norma.metadata import infer, manifest, resolve  # noqa: PLC0415

    destination = options.template_output
    if destination is None:
        raise UsageError("tags template requires -o FILE.csv|FILE.json|-")
    pattern = None
    if options.infer_from_filename is not None:
        try:
            pattern = infer.compile_pattern(options.infer_from_filename)
        except infer.InvalidPattern as exc:
            raise UsageError(f"--infer-from-filename: {exc}") from None
    outputs = None
    registry = None
    folder, form = options.cwd, "csv"
    if destination != "-":
        path = Path(destination)
        form = path.suffix.lower().lstrip(".")
        if form not in ("csv", "json"):
            raise UsageError("tags template -o needs .csv or .json, or - for stdout")
        folder = path.parent
        request = RunOutputRequest("template_export", path)
        executables = [tool.path for tool in (tools.prober, tools.ffmpeg) if tool is not None]
        protected = _protected_reads(items, options, executables)
        checked = safety.preflight({}, (request,), protected, overwrite=options.overwrite)
        registry = safety.RunRegistry(run_id or new_run_id())
        outputs = RunOutputs(checked, protected, registry, overwrite=options.overwrite, requests=(request,))
    entries: list[Any] = []

    def one(item: InputItem) -> Inspection:
        inspection = _inspect_tags_one(item, tools.prober, cancel_event)
        if inspection.outcome.status not in ("ok", "warn") or inspection.source_tags is None:
            return inspection
        try:
            _check_cancel(cancel_event)
            inferred = (infer.infer(pattern, item.relative_path, underscores=options.infer_underscores)
                        if pattern is not None else None)
            resolution = resolve.resolve_opinions(
                inspection.source_tags, stem=item.source_path.stem,
                filename=inferred.cells if inferred is not None else {},
                infer_mode=options.infer_mode, no_title_fallback=True)
            fields = {name: value.value for name, value in resolution["resolved"].items()}
            extras = {name: value.value for name, value in resolution["extras"].items()}
            entries.append(manifest.TemplateEntry(item.source_path, fields, extras))
            return evolve(inspection, notes=tuple(resolution["notes"]) +
                          (tuple(inferred.notes) if inferred is not None else ()))
        except (Cancelled, Exception) as exc:  # noqa: BLE001 - per-file containment
            error = _as_failure(exc, "tags_read")
            return evolve(inspection, primary=error.diagnostic(), outcome=evolve(
                inspection.outcome, status=error.status, reason_code=error.reason))

    try:
        run = _run(items, one, cancel_event, None)
        text = ""
        written = None
        errors: tuple[str, ...] = ()
        if entries and not run.cancelled:
            text = manifest.build_template(entries, folder, form)
            if outputs is not None:
                try:
                    written = outputs.publish("template_export", text.encode("utf-8"))
                except RunOutputError as exc:
                    errors = (str(exc),)
        if on_inspection is not None:
            for inspection in run.inspections:
                on_inspection(inspection)
        if registry is not None:
            errors += tuple(registry.close())
            registry = None
        return TemplateRun(run.inspections, run.not_started, run.cancelled, text, written, errors)
    finally:
        if registry is not None:
            registry.close()


def analyse_items(items: Sequence[InputItem], options: RunOptions, tools: ReadOnlyTools, *,
                  cancel_event: threading.Event | None = None,
                  on_inspection: OnInspection | None = None,
                  report: RunOutputRequest | None = None,
                  execute: ReadOnlyExecute | None = None) -> ReadOnlyRun:
    """``analyse``: probe, measure and plan each item with the same pure C-05 plan as ``normalise``.

    Explicit dynamic uses the uncalibrated preview plan (C-05 v1.3, D-080).
    Target and ceiling feed only the prediction; nothing is rendered or written (audio V8).
    ``report`` is an explicit ``--report``: it passes ``check_read_only_report`` before the
    first file (``UsageError`` otherwise) and is returned unpublished in ``ReadOnlyRun.outputs``.
    Pass ``execute`` as in ReadOnlyBatchRequest's example for bounded ``options.jobs``;
    the existing direct-call default remains sequential.

    Typical use for ``analyse --report`` (the interface renders the bytes; ARCHITECTURE § 3)::

        tools = pipeline.locate_tools(options, need_ffmpeg=True)
        found = pipeline.discover_inputs(options)
        request = pipeline.report_request(options.report) if options.report else None
        run = pipeline.analyse_items(found.items, options, tools, report=request)  # exit 2 here
        if run.outputs is not None:
            try:
                written = run.outputs.publish("report", report.render_csv(envelope).encode())  # real path
            except pipeline.RunOutputError as exc:
                ...  # N-07: name the intended path, still print the summary, exit at least 1
    """
    if tools.ffmpeg is None:
        raise ValueError("analyse needs a located FFmpeg (locate_tools(..., need_ffmpeg=True))")
    executable = tools.ffmpeg
    outputs = check_read_only_report(items, options, tools, report) if report is not None else None

    def one(item: InputItem, event: threading.Event | None, state: _File) -> Inspection:
        return _analyse_one(item, options, tools.prober, executable, event, state=state)

    run = _run_read_only(items, one, options.jobs, cancel_event, on_inspection, execute)
    return ReadOnlyRun(run.inspections, run.not_started, run.cancelled, outputs)


def verify_items(items: Sequence[InputItem], options: RunOptions, tools: ReadOnlyTools, *,
                 cancel_event: threading.Event | None = None,
                 on_inspection: OnInspection | None = None,
                 report: RunOutputRequest | None = None,
                 execute: ReadOnlyExecute | None = None) -> ReadOnlyRun:
    """Inspect unchanged outputs against WS-01's per-file reference (audio V9).

    No gain/output/metadata plan, capability encode, repair, log or automatic report.
    Only an explicit report is reserved, never published here. Pass ``execute``
    using ReadOnlyBatchRequest's example for bounded ``options.jobs`` execution;
    omitting it preserves sequential direct calls, as in analyse_items.
    The inspected pathname is ``outcome.input.source_path``, not an OutputPlan;
    output measurements and VerificationReference are populated without publication.
    """
    if tools.ffmpeg is None:
        raise ValueError("verify needs a located FFmpeg (locate_tools(..., need_ffmpeg=True))")
    event = cancel_event if cancel_event is not None else threading.Event()
    references = _InspectionInputs(items, options)
    references.read_artwork(event)
    outputs = None
    if report is not None:
        references.check_report(report)
        executables = [tool.path for tool in (tools.ffmpeg, tools.prober, references.sips) if tool is not None]
        protected = _protected_reads(items, options, executables) + references.protected
        outputs = _check_plan_report(options, report, protected, {})
    run = _run_read_only(items, lambda item, event, state: _verify_one(
        item, options, tools, event, references, state=state),
        options.jobs, event, on_inspection, execute)
    return evolve(run, outputs=outputs, notes=tuple(references.notes))


class _InspectionInputs:
    """Explicit-only references, independent of production preparation settings."""

    def __init__(self, items: Sequence[InputItem], options: RunOptions) -> None:
        from norma.metadata import manifest  # noqa: PLC0415

        self.notes: list[str] = []
        self.protected: tuple[ReadDependency, ...] = ()
        self.references: dict[int, tuple[InspectionReference, ...]] = {}
        self.errors: dict[int, FileOutcomeError] = {}
        self.warnings: dict[int, tuple[Diagnostic, ...]] = {}
        self.sips: Any = None
        self.artwork: dict[Path, Any] = {}
        if not options.check_tags:
            return
        data = manifest.load_manifest(options.manifest) if options.manifest is not None else None
        matching = manifest.match_manifest(data, items) if data is not None else None
        if data is not None and matching is not None:
            self.protected = (safety.protect(data.path, "metadata"),)
            self.notes.extend(manifest.format_issue(issue) for issue in data.issues
                              if issue.detail in manifest.WARNING_DETAILS)
            self.notes.extend(f"manifest row {issue.locator} matches no input (unmatched row)"
                              for issue in matching.unmatched)
        for item in items:
            refs: dict[str, InspectionReference] = {}
            layers = []
            if data is not None and matching is not None:
                match = matching.inputs[item.input_index]
                self.warnings[item.input_index] = tuple(Diagnostic(
                    "manifest_row_ambiguous", "manifest row matches several inputs and was not applied",
                    "metadata_prepared", manifest.format_issue(issue)) for issue in match.warnings)
                if match.failures:
                    self.errors[item.input_index] = FileFailure(
                        "manifest_row_invalid", "metadata_prepared",
                        "; ".join(manifest.format_issue(issue) for issue in match.failures))
                layers.append((data.defaults, "manifest_default", f"{data.path.name}:defaults"))
                if match.row is not None:
                    layers.append((match.row.cells, "manifest", match.row.locator))
            cli = {name: ManifestCell("set", getattr(options, name)) for name in _CLI_FIELDS
                   if getattr(options, name) is not None and
                   (name in options.explicit_fields or options.value_sources.get(name) == "cli")}
            layers.append((cli, "cli", ""))
            for cells, origin, locator in layers:
                for name, cell in cells.items():
                    if name == "artwork" and cell.operation == "set":
                        assert data is not None and isinstance(cell.value, (str, Path))
                        path = Path(cell.value)
                        cell = ManifestCell("set", safety.canonical_destination(
                            path if path.is_absolute() else data.path.parent / path))
                    option = name.removesuffix("_total") if name in ("track_total", "disc_total") else name
                    refs[name.casefold()] = InspectionReference(
                        name, cell, origin, "--" + "-".join(option.split("_")) if origin == "cli" else locator)
            self.references[item.input_index] = tuple(refs.values())

    def read_artwork(self, event: threading.Event) -> None:
        """Gather every effective image read before the whole-report barrier.

        Immutable evidence is shared, with content rechecked per consumer. Even
        invalid existing images remain protected; absent paths get a path-only
        report check, never an invented device/inode identity.
        """
        from norma.metadata import artwork  # noqa: PLC0415

        for index, refs in self.references.items():
            if event.is_set():
                break
            if index in self.errors:
                continue
            for ref in refs:
                if ref.name != "artwork" or ref.cell.operation != "set":
                    continue
                if self.sips is None:
                    self.sips = locate_sips_tool()  # E-05 is preflight-only, before any row
                path = ref.cell.value
                assert isinstance(path, Path)
                if path not in self.artwork:
                    try:
                        self.protected += (safety.protect(path, "artwork"),)
                    except OSError:
                        pass  # owner translates inaccessible/missing images below
                    try:
                        evidence = artwork.read_artwork_reference(path, self.sips, cancel_event=event)
                        self.artwork[path] = evidence
                        self.protected += (evidence.stamp.read_dependency(),)
                    except Cancelled:
                        event.set()
                        return
                    except Exception as exc:  # noqa: BLE001 - file-local reference failure
                        self.artwork[path] = _as_failure(exc, "content_verified")
                if isinstance(self.artwork[path], FileOutcomeError):
                    self.errors[index] = self.artwork[path]

    def check_report(self, request: RunOutputRequest) -> None:
        """Also protect lexical image names when no identity could be read."""
        destination = comparison_key(safety.canonical_destination(request.path))
        for refs in self.references.values():
            for ref in refs:
                if ref.name == "artwork" and ref.cell.operation == "set":
                    assert isinstance(ref.cell.value, Path)
                    if destination == comparison_key(safety.canonical_destination(ref.cell.value)):
                        raise UsageError(f"report aliases artwork reference {ref.cell.value}")


def _verify_one(item: InputItem, options: RunOptions, tools: ReadOnlyTools,
                cancel_event: threading.Event | None, inputs: _InspectionInputs, *,
                state: _File | None = None) -> Inspection:
    state = state if state is not None else _File(item)
    reference = verification = None
    diagnostics: list[Diagnostic] = []
    refs = inputs.references.get(item.input_index, ())
    mismatches: tuple[str, ...] = ()
    try:
        _check_cancel(cancel_event)
        snapshot = safety.snapshot_source(item, stage="probe")
        state.probe = classify(item, tools.prober, cancel_event=cancel_event)
        _check_cancel(cancel_event)
        state.stage = "tags_read"
        state.source_tags = (_EMPTY_TAGS if state.probe.container in UNREADABLE_TAG_CONTAINERS
                             else read_source_tags(item.source_path, state.probe.container))
        state.mark = read_processing_mark(state.source_tags)
        reference = select_reference(options, state.mark)
        diagnostics.extend(state.mark.diagnostics)
        diagnostics.extend(inputs.warnings.get(item.input_index, ()))
        if item.input_index in inputs.errors:
            raise inputs.errors[item.input_index]
        _check_cancel(cancel_event)
        state.stage = "verify"
        assert tools.ffmpeg is not None
        verification = assess_inspection(measure_candidate(
            item.source_path, state.probe, tools.ffmpeg,
            no_dual_mono=options.no_dual_mono, cancel_event=cancel_event), reference)
        diagnostics.extend(verification.measurement.diagnostics)
        diagnostics.extend(verification.diagnostics)
        _check_cancel(cancel_event)
        if options.check_tags:
            from norma.metadata import artwork  # noqa: PLC0415

            image = None
            for ref in refs:
                if ref.name == "artwork" and ref.cell.operation == "set":
                    assert isinstance(ref.cell.value, Path)
                    image = inputs.artwork[ref.cell.value]
            if image is not None:
                artwork.recheck_artwork_reference(image)
            checked = _write_edge().check_inspection_tags(
                state.source_tags, {r.name: r.cell for r in refs}, artwork_reference=image)
            mismatches = checked.mismatches
            diagnostics.extend(checked.diagnostics)
        _check_cancel(cancel_event)
        safety.recheck_source(item, snapshot, rehash=True, stage="verify")
        # C-13 inspection precedence: peak, loudness, then tags. Keep every cause.
        failure = next((d for code in ("verification_peak", "verification_loudness", "verification_tags")
                        for d in diagnostics if d.code == code), None)
        warnings = state.warnings(diagnostics)
        if failure is not None:
            result = state.finish("failed", failure.code, primary=failure, warnings=warnings)
        else:
            status, reason = settle_primary("ok", "ok", warnings)
            result = state.finish(status, reason, warnings=warnings)
    except (Cancelled, Exception) as exc:  # noqa: BLE001 - per-file containment
        if isinstance(exc, FileOutcomeError):
            diagnostics.extend((*exc.diagnostics, exc.diagnostic()))
        result = state.settle_error(exc)
    if result.primary is not None:
        diagnostics.append(result.primary)
    return evolve(result, diagnostics=tuple(dict.fromkeys(diagnostics)), tag_references=refs,
                  tag_mismatches=mismatches, outcome=evolve(
        result.outcome, measurement_out=verification.measurement if verification is not None else None,
        verification=verification, verification_reference=reference,
        origin_codec=state.mark.origin_codec if state.mark is not None else None,
        warnings=state.warnings((*diagnostics, *result.outcome.warnings))))


def inspect_tags(items: Sequence[InputItem], options: RunOptions, tools: ReadOnlyTools, *,
                 cancel_event: threading.Event | None = None,
                 on_inspection: OnInspection | None = None) -> ReadOnlyRun:
    """``tags show``: probe each item and read every tag store and its processing mark.

    ``options`` is accepted for a uniform dispatch signature; reading uses no option. Runs
    ffprobe only, never FFmpeg, and writes nothing (metadata MD27).
    """
    del options

    def one(item: InputItem) -> Inspection:
        return _inspect_tags_one(item, tools.prober, cancel_event)

    return _run(items, one, cancel_event, on_inspection)


def plan_tags_items(items: Sequence[InputItem], options: RunOptions, tools: ReadOnlyTools, *,
                    cancel_event: threading.Event | None = None,
                    on_inspection: OnInspection | None = None,
                    report: RunOutputRequest | None = None) -> ReadOnlyRun:
    """``tags plan`` (ARCHITECTURE § 5, metadata MD27, MD35; TST-M-07): read-only metadata plans.

    States: signature/probe, ``tags_read``, a provisional format choice, ``metadata_prepared``,
    then C-07/C-08 plans with ``execution="plan"``. Runs ffprobe and, only when a supplied or
    MP4-bound image must be prepared, ``sips``; never FFmpeg (no measurement, calibration,
    render or capability-evidence encode), and writes nothing except an explicit ``--report``
    (checked after planning, before any callback, published through ``ReadOnlyRun.outputs``).

    Without measurement, copy eligibility is unknown: when a stream copy is possible the
    file gets two documentary alternatives (``stream_copy``, ``encode``) in
    ``outcome.conditional_plans`` with ``metadata``/``output`` None; otherwise one plan with
    ``audio_basis=encode``. A branch that would fail carries ``ConditionalPlan.refusal``
    and no metadata/output plan, with its diagnostics also retained on ``Inspection``.
    When every branch fails the file is failed at the earliest state (stream_copy first on a tie).
    ``sips`` missing when an image must be prepared is ``EnvironmentError_`` (exit 3) before any row.
    """
    meta = MetadataInputs(items, options, cancel_event=cancel_event)
    naming = NamingOptions(
        out_dir=safety.canonical_directory(options.out_dir) if options.out_dir is not None else None,
        suffix=options.suffix, rename=options.rename, target_lufs=options.target)
    try:
        states = _tags_plan_prepass(items, options, tools.prober, meta, cancel_event)
        done: list[Inspection] = []
        cancelled = len(states) < len(items)
        for item, state, choice in states:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            inspection = (state if isinstance(state, Inspection)
                          else _plan_tags_one(item, state, choice, options, meta, naming, cancel_event))
            done.append(inspection)
            if report is None and on_inspection is not None:
                on_inspection(inspection)
            if inspection.outcome.reason_code == "cancelled":
                cancelled = True
                break
        outputs = None
        if report is not None:
            plans: dict[int, OutputPlan] = {}
            for inspection in done:
                outcome = inspection.outcome
                branches = ([outcome.output] if outcome.output is not None else []) + [
                    branch.output for branch in outcome.conditional_plans if branch.output is not None]
                if branches:
                    # Alternatives may share names. Reserve their union, not just the first
                    # branch, without turning documentary per-file refusals into new outcomes.
                    members = {member.comparison_key: member for plan in branches for member in plan.members}
                    plans[outcome.input.input_index] = evolve(branches[0], members=tuple(members.values()))
            executables = [tool.path for tool in (tools.ffmpeg, tools.prober, meta.sips) if tool is not None]
            protected = _protected_reads(items, options, executables) + meta.protected
            outputs = _check_plan_report(options, report, protected, plans)
        if report is not None and on_inspection is not None:
            for inspection in done:
                on_inspection(inspection)
        return ReadOnlyRun(tuple(done), len(items) - len(done), cancelled, outputs, tuple(meta.notes))
    finally:
        meta.close()


@dataclass(frozen=True)
class ProvisionalFormat:
    """``tags plan``'s evidence-free format choice: enough to name stores and the output (no C-04)."""

    format: str
    extension: str
    copy_possible: bool
    stream_copy: bool = False


def provisional_format(probe: ProbeResult, options: RunOptions, *, source_suffix: str | None) -> ProvisionalFormat:
    """The target format WS-02 would choose, without capability evidence (no FFmpeg run).

    Uses WS-02's public evidence-free chooser. ``copy_possible`` is conservative: true when
    the codec could stay unchanged and no explicit rate or depth forces an encode.
    """
    from norma.formats import capabilities, resolve as formats  # noqa: PLC0415

    target, decision = formats.provisional_target(probe, options)
    if decision == "skipped_lossy":
        raise SkipFile("lossy_source_skipped", "format", f"lossy {probe.codec} source skipped (--lossy-sources skip)")
    extension = capabilities.output_extension(target, same_source_suffix=source_suffix
                                              if options.format == "same" else None)
    codec_same = (probe.codec == target or (target in ("wav", "aiff") and probe.codec.startswith("pcm_")
                                            and probe.container == target))
    copy_possible = (codec_same and decision != "lossy_origin" and options.sample_rate == "same"
                     and options.bit_depth == "same")
    return ProvisionalFormat(target, extension, copy_possible)


def _tags_plan_prepass(items: Sequence[InputItem], options: RunOptions, prober: Prober, meta: MetadataInputs,
                       cancel_event: threading.Event | None) -> list[tuple[InputItem, Any, Any]]:
    """Probe, read tags and gather opinions for every file, then locate ``sips`` if needed (E-05).

    A file that ends here (skip, failure, CAF warning) is an ``Inspection``; others carry their
    working state to the planning pass.
    """
    states: list[tuple[InputItem, Any, Any]] = []
    needs_sips = False
    for item in items:
        if cancel_event is not None and cancel_event.is_set():
            break
        state = _File(item)
        choice = None
        try:
            state.probe = classify(item, prober, cancel_event=cancel_event)
            _check_cancel(cancel_event)
            state.stage = "tags_read"
            if state.probe.container in UNREADABLE_TAG_CONTAINERS:  # CR-081
                warning = unreadable_tags(state.probe.container)
                if options.on_metadata_loss == "error":  # CR-097: predict normalise's refusal
                    raise FileFailure("metadata_loss_error", "tags_read", warning.message, (warning,))
                states.append((item, _tags_plan_finish(state, "warn", "metadata_loss_warn",
                                                       warnings=(warning,)), None))
                continue
            state.source_tags = read_source_tags(item.source_path, state.probe.container)
            state.mark = read_processing_mark(state.source_tags)
            if state.mark.validity != "absent" and not options.force:
                raise SkipFile("already_processed", "tags_read", "the file carries a Norma processing mark; "
                                                                 "use --force to plan it again")
            state.stage = "format"
            choice = provisional_format(state.probe, options, source_suffix=item.source_path.suffix)
            state.stage = "metadata_prepared"
            state.opinions = meta.opinions(item)
            needs_sips = needs_sips or meta.needs_sips(state.opinions, state.source_tags, choice.format)
        except (Cancelled, Exception) as exc:  # noqa: BLE001 - per-file containment (§ 7)
            states.append((item, _tags_plan_settle_error(state, exc), None))
            continue
        states.append((item, state, choice))
    if needs_sips:
        meta.ensure_preparer()  # EnvironmentError_ (exit 3) before any row
    return states


def _tags_plan_finish(state: _File, status: Status, reason: ReasonCode, *, primary: Diagnostic | None = None,
                      warnings: tuple[Diagnostic, ...] = (), metadata: MetadataPlan | None = None,
                      output: OutputPlan | None = None, conditional: tuple[ConditionalPlan, ...] = (),
                      diagnostics: tuple[Diagnostic, ...] = (), notes: tuple[str, ...] = ()) -> Inspection:
    check_primary(status, reason)
    outcome = FileOutcome(
        status=status, reason_code=reason, input=state.item, probe=state.probe, target=None,
        measurement_in=None, measurement_out=None, gain=None, verification=None, verification_reference=None,
        source_codec=state.probe.codec if state.probe is not None else None, origin_codec=None,
        metadata=metadata, tag_write=None, output=output, conditional_plans=conditional, warnings=warnings,
        elapsed_s=max(0.0, time.monotonic() - state.started), ffmpeg_stderr_tail=None, execution="plan",
        published_paths=(), retained_failed_path=None)
    return Inspection(outcome, primary, state.source_tags, state.mark, diagnostics, notes)


def _tags_plan_settle_error(state: _File, exc: BaseException) -> Inspection:
    inspection = state.settle_error(exc)
    outcome = evolve(inspection.outcome, execution="plan")
    return evolve(inspection, outcome=outcome)


def _plan_tags_one(item: InputItem, state: _File, choice: ProvisionalFormat, options: RunOptions,
                   meta: MetadataInputs, naming: NamingOptions,
                   cancel_event: threading.Event | None) -> Inspection:
    """One ``tags plan`` file after the prepass: resolve, then plan each feasible branch (MD35)."""
    notes: tuple[str, ...] = ()
    try:
        _check_cancel(cancel_event)
        state.stage = "metadata_prepared"
        assert state.source_tags is not None and state.opinions is not None
        resolution, _stamps, notes = meta.resolve(item, state.opinions, state.source_tags, choice.format)
        state.stage = "metadata_planned"
        bases = ("stream_copy", "encode") if choice.copy_possible else ("encode",)
        branches: list[tuple[str, MetadataPlan, OutputPlan]] = []
        conditional: list[ConditionalPlan] = []
        refusals: list[Diagnostic] = []
        branch_failures: list[Diagnostic] = []
        for basis in bases:
            try:
                plan = plan_metadata(state.source_tags, resolution, choice, options, revision=1,  # type: ignore[arg-type]
                                     conditional=len(bases) > 1, audio_basis=basis)
                values = {name: value.value for name, value in plan.resolved.items()}
                output = plan_output(item, choice, naming, plan_revision=plan.plan_revision,  # type: ignore[arg-type]
                                     sidecar_requests=plan.sidecar_requests, values=values,
                                     overwrite=options.overwrite)
            except FileOutcomeError as exc:
                diagnostic = exc.diagnostic()
                failure = evolve(diagnostic, message=f"{basis}: {diagnostic.message}")
                branch_failures.append(failure)
                refusals.append(failure)
                refusals.extend(evolve(d, message=f"{basis}: {d.message}") for d in exc.diagnostics)
                if len(bases) > 1:
                    conditional.append(ConditionalPlan(basis, None, None,  # type: ignore[arg-type]
                        refusal=ConditionalRefusal(exc.reason, exc.stage, exc.detail)))
                continue
            branches.append((basis, plan, output))
            if len(bases) > 1:
                conditional.append(ConditionalPlan(basis, plan, output))  # type: ignore[arg-type]
    except (Cancelled, Exception) as exc:  # noqa: BLE001 - per-file containment (§ 7)
        inspection = _tags_plan_settle_error(state, exc)
        return evolve(inspection, notes=notes)
    extra = list(state.opinions.warnings) if state.opinions is not None else []

    if not branches:
        # C-09 / ARCHITECTURE § 4: metadata preparation/planning precedes path planning.
        # The loop order supplies stream_copy first on a tie; secondary diagnostics do
        # not compete with the terminal failure of their branch for primary selection.
        stages = ("metadata_prepared", "metadata_planned", "path_planned")
        first = min(branch_failures, key=lambda d: stages.index(d.stage) if d.stage in stages else len(stages))
        return _tags_plan_finish(state, "failed", first.code, primary=first,  # type: ignore[arg-type]
                                 warnings=state.warnings(extra), conditional=tuple(conditional),
                                 diagnostics=tuple(d for d in refusals if d is not first), notes=notes)
    for _basis, plan, _output in branches:
        extra.extend(plan.diagnostics)
    warnings = state.warnings(extra)
    status, reason = settle_primary("ok", "ok", warnings)
    if len(bases) == 1:
        _basis, plan, output = branches[0]
        return _tags_plan_finish(state, status, reason, warnings=warnings, metadata=plan, output=output, notes=notes)
    return _tags_plan_finish(state, status, reason, warnings=warnings, conditional=tuple(conditional),
                             diagnostics=tuple(refusals), notes=notes)


# --- Sequencing ------------------------------------------------------------------------------


def _run_read_only(items: Sequence[InputItem],
                   one: Callable[[InputItem, threading.Event | None, _File], Inspection],
                   jobs: int, cancel_event: threading.Event | None,
                   on_inspection: OnInspection | None, execute: ReadOnlyExecute | None) -> ReadOnlyRun:
    """Cache owner-layered records, leaving bounded admission/drain to the executor."""
    if execute is None:
        run = _run(items, lambda item: one(item, cancel_event, _File(item)), cancel_event, on_inspection)
        return evolve(run, cancelled=run.cancelled or (cancel_event is not None and cancel_event.is_set()))
    event = cancel_event if cancel_event is not None else threading.Event()
    records: dict[int, Inspection] = {}
    states: dict[int, _File] = {}
    lock = threading.RLock()
    delivered: set[int] = set()

    def remember(record: Inspection) -> FileOutcome:
        with lock:
            records[record.outcome.input.input_index] = record
        return record.outcome

    def record_for(outcome: FileOutcome) -> Inspection:
        with lock:
            record = records[outcome.input.input_index]
            if record.outcome is not outcome:
                raise ValueError("executor must return the pipeline's original outcome")
            return record

    def worker(item: InputItem, control: Any) -> FileOutcome:
        state = _File(item)
        with lock:
            states[item.input_index] = state
        record = one(item, control.cancel_event, state)
        if record.outcome.reason_code == "cancelled":
            control.cancel_event.set()
        return remember(record)

    def settle_error(item: InputItem, exc: BaseException, elapsed: float, control: Any) -> FileOutcome:
        with lock:
            state = states.get(item.input_index)
        if state is None:
            state = _File(item)
        error = Cancelled() if isinstance(exc, KeyboardInterrupt) else exc
        if isinstance(error, Cancelled):
            control.cancel_event.set()
        record = state.settle_error(error)
        return remember(evolve(record, outcome=evolve(record.outcome, elapsed_s=elapsed)))

    def on_outcome(outcome: FileOutcome) -> None:
        index = outcome.input.input_index
        if index not in delivered:
            delivered.add(index)
            if on_inspection is not None:
                on_inspection(record_for(outcome))

    result = execute(ReadOnlyBatchRequest(tuple(items), worker, settle_error, jobs, lock,
                                         event, force_kill_tools, record_for, on_outcome))
    ordered = tuple(sorted(result.outcomes, key=lambda outcome: outcome.input.input_index))
    for outcome in ordered:
        on_outcome(outcome)  # executors may omit optional completion callbacks
    return ReadOnlyRun(tuple(record_for(outcome) for outcome in ordered), result.not_started, result.cancelled)


def _run(items: Sequence[InputItem], one: Callable[[InputItem], Inspection],
         cancel_event: threading.Event | None, on_inspection: OnInspection | None) -> ReadOnlyRun:
    done: list[Inspection] = []
    for position, item in enumerate(items):
        if cancel_event is not None and cancel_event.is_set():
            return ReadOnlyRun(tuple(done), len(items) - position, True)
        inspection = one(item)
        done.append(inspection)
        if on_inspection is not None:
            on_inspection(inspection)
        if inspection.outcome.reason_code == "cancelled":
            return ReadOnlyRun(tuple(done), len(items) - position - 1, True)
    return ReadOnlyRun(tuple(done), 0, False)


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()


class _File:
    """Records gathered for one file so far; turns into exactly one terminal outcome."""

    def __init__(self, item: InputItem) -> None:
        self.item = item
        self.started = time.monotonic()
        self.stage = "probe"
        self.probe: ProbeResult | None = None
        self.measurement: LoudnessMeasurement | None = None
        self.gain: GainPlan | None = None
        self.source_tags: SourceTags | None = None
        self.mark: ProcessingMark | None = None
        self.opinions: FileOpinions | None = None

    def warnings(self, extra: Sequence[Diagnostic] = ()) -> tuple[Diagnostic, ...]:
        """Warning-family diagnostics from every record gathered, in pipeline order, once each."""
        found: list[Diagnostic] = []
        sources: list[Sequence[Diagnostic]] = []
        if self.gain is not None:
            sources.append(self.gain.diagnostics)  # already includes the measurement's
        elif self.measurement is not None:
            sources.append(self.measurement.diagnostics)
        if self.mark is not None:
            sources.append(self.mark.diagnostics)
        sources.append(extra)
        for diagnostics in sources:
            for diagnostic in diagnostics:
                if reason_spec(diagnostic.code).kind == "warning" and diagnostic not in found:
                    found.append(diagnostic)
        return tuple(found)

    def finish(self, status: Status, reason: ReasonCode, *, primary: Diagnostic | None = None,
               warnings: tuple[Diagnostic, ...] = ()) -> Inspection:
        check_primary(status, reason)
        outcome = FileOutcome(
            status=status, reason_code=reason, input=self.item, probe=self.probe, target=None,
            measurement_in=self.measurement, measurement_out=None, gain=self.gain, verification=None,
            verification_reference=None, source_codec=self.probe.codec if self.probe is not None else None,
            origin_codec=None, metadata=None, tag_write=None, output=None, conditional_plans=(),
            warnings=warnings, elapsed_s=max(0.0, time.monotonic() - self.started),
            ffmpeg_stderr_tail=None, execution="inspect", published_paths=(), retained_failed_path=None,
        )
        return Inspection(outcome, primary, self.source_tags, self.mark)

    def settle_ok(self, reason: ReasonCode = "ok") -> Inspection:
        warnings = self.warnings()
        status, primary_reason = settle_primary("ok", reason, warnings)
        return self.finish(status, primary_reason, warnings=warnings)

    def settle_error(self, exc: BaseException) -> Inspection:
        """The terminal outcome for an exception raised while the file was in progress."""
        if isinstance(exc, Cancelled):
            primary = Diagnostic("cancelled", "cancelled before the file finished", self.stage)
            return self.finish("skipped", "cancelled", primary=primary, warnings=self.warnings())
        if isinstance(exc, FileOutcomeError):
            return self.finish(exc.status, exc.reason, primary=exc.diagnostic(),
                               warnings=self.warnings(exc.diagnostics))
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        primary = Diagnostic("internal", f"unexpected {type(exc).__name__}: {exc}", self.stage, detail)
        return self.finish("failed", "internal", primary=primary, warnings=self.warnings())


def _analyse_one(item: InputItem, options: RunOptions, prober: Prober, executable: Executable,
                 cancel_event: threading.Event | None, *, state: _File | None = None) -> Inspection:
    state = state if state is not None else _File(item)
    try:
        _check_cancel(cancel_event)
        state.probe = classify(item, prober, cancel_event=cancel_event)
        _check_cancel(cancel_event)
        state.stage = "measure"
        state.measurement = measure(item.source_path, state.probe, executable,
                                    no_dual_mono=options.no_dual_mono, cancel_event=cancel_event)
        _check_cancel(cancel_event)
        state.stage = "plan"
        state.gain = (plan_dynamic_preview(state.measurement, options) if options.peak_mode == "dynamic"
                      else plan_gain(state.measurement, options))
    except (Cancelled, Exception) as exc:  # noqa: BLE001 - per-file containment (ARCHITECTURE § 7)
        return state.settle_error(exc)

    plan = state.gain
    if plan.action == "skip":  # silent or unmeasurable input: SKIP with the meter's diagnostic
        primary = next((d for d in plan.diagnostics if d.code == plan.reason_code),
                       Diagnostic(plan.reason_code, "no gain plan for this signal", "plan"))
        return state.finish("skipped", plan.reason_code, primary=primary, warnings=state.warnings())
    # analyse writes nothing, so a within-tolerance copy plan is an ok plan (C-13 "ok for copy/plan").
    return state.settle_ok("already_within_tolerance" if plan.reason_code == "already_within_tolerance" else "ok")


def _inspect_tags_one(item: InputItem, prober: Prober, cancel_event: threading.Event | None) -> Inspection:
    state = _File(item)
    try:
        _check_cancel(cancel_event)
        state.probe = classify(item, prober, cancel_event=cancel_event)
        _check_cancel(cancel_event)
        state.stage = "tags_read"
        if state.probe.container in UNREADABLE_TAG_CONTAINERS:  # CR-081: no tag store (CR-036)
            return state.finish("warn", "metadata_loss_warn", warnings=(unreadable_tags(state.probe.container),))
        state.source_tags = read_source_tags(item.source_path, state.probe.container)
        state.mark = read_processing_mark(state.source_tags)
    except (Cancelled, Exception) as exc:  # noqa: BLE001 - per-file containment (ARCHITECTURE § 7)
        return state.settle_error(exc)
    return state.settle_ok()


# --- normalise: the write path (S2 TK-0412, S3 TK-0419) ---------------------------------------

NORMALISE_FILTERS: tuple[str, ...] = tuple(dict.fromkeys(MEASURE_FILTERS + ("volume", "aresample")))
"""Filters every ``normalise`` runs: the meter scans plus the gain and resampler of copy, linear and reduce."""


def normalise_filters(peak_mode: Literal["limit", "reduce", "dynamic"]) -> tuple[str, ...]:
    """Filters a ``normalise`` run with this peak policy can use (ARCHITECTURE § 9).

    Policy ``limit`` may limit (initially or as a peak correction), so it needs ``alimiter``
    with its ``level``/``latency`` options; explicit ``dynamic`` needs ``loudnorm``; ``reduce``
    never limits. A facility the policy cannot use is not required.
    """
    extra = {"limit": ("alimiter",), "dynamic": DYNAMIC_FILTERS, "reduce": ()}[peak_mode]
    return tuple(dict.fromkeys(NORMALISE_FILTERS + extra))

VERIFICATION_FAILURES: frozenset[str] = frozenset({"verification_loudness", "verification_peak", "verification_tags"})
"""Failures whose candidate ``--keep-failed`` may retain: final audio or tag verification (safety § 5.2,
§ 6 cleanup); never cancellation."""


@dataclass(frozen=True)
class NormaliseTools:
    """Executables and FFmpeg facts resolved once in ``normalise`` run preflight."""

    prober: Prober
    environment: FFmpegEnvironment

    @property
    def executable(self) -> Executable:
        return self.environment.executable

    @property
    def basis(self) -> ToolBasis:
        executable = self.environment.executable
        return ToolBasis(executable.path, executable.identity, executable.version, self.environment.platform_basis)


def locate_normalise_tools(options: RunOptions) -> NormaliseTools:
    """ffprobe beside C-10 ``ffmpeg``, FFmpeg itself, its listings and the policy's filters.

    Any missing or unusable tool, filter or ``alimiter`` option raises ``EnvironmentError_``
    (exit 3) before any file is touched. Encoders are checked per run by
    ``formats.resolve.preflight_targets``.
    """
    prober = locate_ffprobe(options.ffmpeg.parent / "ffprobe")
    environment = probe_environment(locate_ffmpeg(options.ffmpeg))
    check_requirements(environment, filters=normalise_filters(options.peak_mode),
                       alimiter_options=ALIMITER_OPTIONS if options.peak_mode == "limit" else ())
    return NormaliseTools(prober, environment)


def force_kill_tools() -> None:
    """The second-interrupt force hook for ``batch.BatchControl`` (CR-045, CR-047, D-065).

    Kills every FFmpeg and every ffprobe process group in progress at once (both adapters'
    hooks, TK-0418); quick, non-blocking and safe in a signal handler. FFmpeg goes first: it
    writes candidates. Exposed here because the interface may not import adapters (§ 3).
    """
    try:
        force_kill()
    finally:
        force_kill_ffprobe()


@dataclass(frozen=True)
class Normalised:
    """One file's C-09 outcome plus what C-09 has no field for (D-066 (1), as ``Inspection``).

    ``primary`` is the diagnostic behind a skip or failure (stage and detail; for ``internal``
    the traceback); ``diagnostics`` holds secondary ones, such as a rollback or cleanup failure
    after the primary failure; ``notes`` are the C-04 target notes and publication notes. All
    are None or empty for an ok result, whose warnings are in ``outcome.warnings``.
    """

    outcome: FileOutcome
    primary: Diagnostic | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    notes: tuple[str, ...] = ()


class BatchControlLike(Protocol):
    """What a worker needs from ``batch.BatchControl``, named without importing ``batch``."""

    cancel_event: threading.Event

    @property
    def interrupted(self) -> bool:
        """Pending interruption, including while publication defers cancel_event."""
        ...


class BatchResultLike(Protocol):
    """What ``normalise_items`` needs back from ``batch.execute_batch``."""

    outcomes: tuple[FileOutcome, ...]
    not_started: int
    cancelled: bool


@dataclass(frozen=True)
class BatchRequest:
    """Everything the interface's batch executor needs for one ``normalise`` run.

    Typical use (the interface owns ``batch``; ARCHITECTURE § 3)::

        def execute(request):
            control = batch.BatchControl(commit_lock=request.commit_lock,
                                         cancel_event=request.cancel_event,
                                         force_hook=request.force_hook)
            return batch.execute_batch(request.entries, request.worker,
                                       settle_error=request.settle_error,
                                       jobs=request.jobs, control=control)

    ``entries`` are in input order: eligible C-01 items and the C-09 outcomes already
    settled in planning. ``record_for`` gives the pipeline-local record of any outcome.
    """

    entries: tuple[InputItem | FileOutcome, ...]
    worker: Callable[[InputItem, Any], FileOutcome]
    settle_error: Callable[[InputItem, BaseException, float, Any], FileOutcome]
    jobs: int
    commit_lock: Any
    cancel_event: threading.Event
    force_hook: Callable[[], None]
    record_for: Callable[[FileOutcome], Normalised]
    outputs: RunOutputs


Execute = Callable[[BatchRequest], BatchResultLike]
RunOutputsSpec = Sequence[RunOutputRequest] | Callable[[Mapping[int, OutputPlan]], Sequence[RunOutputRequest]]


@dataclass(frozen=True)
class NormaliseRun:
    """The ordered records of one ``normalise`` (or dry run), and its run-level results.

    ``records`` follow input order and hold every file with an outcome; ``not_started``
    counts files never begun (C-09). ``notes`` are run-level notes (for example an ignored
    ``--bitrate``); ``run_errors`` are run-level failures after the files settled, such as
    private residue that end-of-run cleanup could not remove. ``outputs`` publishes the
    report and opens the log; it is None when the run stopped before its write set existed.
    """

    records: tuple[Normalised, ...]
    not_started: int
    cancelled: bool
    dry_run: bool
    notes: tuple[str, ...] = ()
    run_errors: tuple[str, ...] = ()
    outputs: RunOutputs | None = None

    @property
    def outcomes(self) -> tuple[FileOutcome, ...]:
        return tuple(record.outcome for record in self.records)

    @property
    def write_set(self) -> RunWriteSet | None:
        return self.outputs.write_set if self.outputs is not None else None


def new_run_id() -> str:
    """A run token for private names: UTC time plus randomness, no separators or spaces."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)


def automatic_report_folder(plans: Mapping[int, OutputPlan], options: RunOptions) -> Path:
    """The folder of the automatic ``normalise`` report (cli § 7.3, safety § 6, D-018, D-072).

    The sole planned audio output folder when there is exactly one (skipped and failed plans
    count). With several folders, the folder the command was started in (``options.cwd``), ``-o``
    or not. Only when no audio destination is planned at all, the ``-o`` folder when one was
    given, else ``options.cwd``.
    """
    folders = {plan.audio_member.final_path.parent for plan in plans.values()}
    if len(folders) == 1:
        return next(iter(folders))
    if folders:
        return options.cwd
    return safety.canonical_directory(options.out_dir) if options.out_dir is not None else options.cwd


def normalise_items(items: Sequence[InputItem], options: RunOptions, tools: NormaliseTools, *,
                    execute: Execute | None = None, run_outputs: RunOutputsSpec = (),
                    cancel_event: threading.Event | None = None, run_id: str | None = None,
                    fault_hook: safety.FaultHook | None = None) -> NormaliseRun:
    """``normalise`` the C-01 ``items``: plan every file, check the write set, then execute.

    Run-level refusals raise before any write: ``UsageError`` (in-place,
    explicit target settings, run-output conflicts;
    exit 2) and ``EnvironmentError_`` (required encoders or C-04 evidence; exit 3).
    ``run_outputs`` are the interface's log, report and template destinations, or a callable
    that receives the planned C-08 bundles by input index and returns them (the automatic
    report's folder depends on them, D-018).

    With ``options.dry_run`` nothing is executed: eligible files end ``execution="plan"``
    and ``execute`` is not needed. Otherwise ``execute`` runs the ``BatchRequest`` (normally
    through ``batch.execute_batch``) and must return its result. ``cancel_event`` is the run's
    cancellation flag, for planning and execution alike; ``fault_hook`` is safety's test seam.

    Typical use with ``--log`` and a report (the interface renders bytes; ARCHITECTURE § 3)::

        def outputs(plans):  # planned C-08 bundles by input index
            wanted = []
            if options.report is not None:
                wanted.append(pipeline.report_request(options.report))
            elif not options.dry_run:  # D-018, D-072
                folder = pipeline.automatic_report_folder(plans, options)
                wanted.append(pipeline.report_request(folder / name, automatic=True))
            if options.log is not None and not options.dry_run:  # N-08 in a dry run
                wanted.append(pipeline.log_request(options.log))
            return wanted

        run = pipeline.normalise_items(items, options, tools, execute=execute,
                                       run_outputs=outputs)  # exit 2 or 3 here, before any write
        # the log can be opened only after every file is planned: inside ``execute`` via
        # ``request.outputs.open_log()`` (UsageError, exit 2, before its first byte)
        if run.outputs is not None and run.outputs.destination("report") is not None:
            written = run.outputs.publish("report", data)  # RunOutputError → N-07, exit at least 1

    ``run.outputs`` is None when cancellation stopped planning, so no report is published.
    """
    safety.refuse_in_place(options)
    if not options.dry_run and execute is None:
        raise ValueError("a normalise run that writes needs an execute callable (batch.execute_batch)")
    naming = NamingOptions(
        out_dir=safety.canonical_directory(options.out_dir) if options.out_dir is not None else None,
        suffix=options.suffix, rename=options.rename, target_lufs=options.target)
    run = _NormaliseRun(items, options, tools, naming, cancel_event or threading.Event(),
                        run_id or new_run_id(), fault_hook)
    return run.run(execute, run_outputs)


# --- normalise: planning helpers --------------------------------------------------------------


def _candidate_formats(probe: ProbeResult, options: RunOptions) -> set[str]:
    """A superset of the targets ``formats.resolve`` may choose for ``probe``, for evidence.

    Collecting a little more evidence than is used is harmless; collecting less would turn a
    resolvable target into ``unsupported_target``. A skipped lossy source needs none.
    """
    if probe.codec_class == "unsupported" or probe.channels > 2:
        return set()
    lossy = probe.codec_class == "lossy"
    if lossy and options.lossy_sources == "skip":
        return set()
    found: set[str] = set()
    same = same_target_format(probe.codec, probe.codec_class, probe.container)
    if same is not None:
        found.add(same)
    if probe.codec.startswith("pcm_") and probe.container in ("mp4", "mov"):
        found.add("alac")  # D-064
    if options.format in TARGETS:
        found.add(options.format)
    if options.format == "apple":
        found.add("alac")
    if lossy and options.lossy_sources == "lossless":
        found.update({"flac", "alac"})
    return found


def collect_evidence(environment: FFmpegEnvironment, probes: Sequence[ProbeResult],
                     options: RunOptions, *,
                     cancel_event: threading.Event | None = None) -> tuple[CapabilityEvidence, ...]:
    """Run-local C-04 evidence for every target the probed inputs may resolve to (ADR-0015).

    Installed-query encodes are made in one system temporary folder that is removed before
    returning. Encoders FFmpeg does not list are left to ``preflight_targets`` (exit 3 when
    required); a collector that cannot run is an environment error, exit 3 before any write.
    MP3 (``libmp3lame``) and AAC (``aac_at``, TK-0418) are collected at the automatic lossy
    rate (F4/F5) or the explicit one; settings they cannot confirm stay conservative gaps.
    Every query uses the D-081 effective layout: an unmasked mono or stereo input is queried as
    ``mono``/``stereo``; any other layout, stated or absent, is queried as probed.
    """
    _check_cancel(cancel_event)
    lossless: dict[tuple[str, int, int, str | None], None] = {}
    mp3: dict[tuple[int, int, str | None], None] = {}
    aac: dict[tuple[int, int, str | None], None] = {}
    for probe in probes:
        _check_cancel(cancel_event)
        # D-081: collect at the layout resolution will query; C-02 itself stays as probed.
        layout = effective_channel_layout(probe.channels, probe.channel_layout)[0]
        for target in sorted(_candidate_formats(probe, options)):
            spec = TARGETS[target]
            explicit = options.sample_rate if options.sample_rate != "same" else None
            if spec.family == "lossless":
                rate = explicit or probe.sample_rate_hz
                for encoder in spec.encoders:
                    if encoder in environment.encoders:
                        lossless[(encoder, rate, probe.channels, layout)] = None
            elif target in ("mp3", "aac") and spec.encoders[0] in environment.encoders:
                rate = explicit or automatic_lossy_rate(probe.sample_rate_hz)
                if rate is not None:
                    (mp3 if target == "mp3" else aac)[(rate, probe.channels, layout)] = None
    _check_cancel(cancel_event)
    if not lossless and not mp3 and not aac:
        return ()
    records: list[CapabilityEvidence] = []
    try:
        with tempfile.TemporaryDirectory(prefix="norma-evidence-") as folder:
            work = Path(folder)
            for number, (encoder, rate, channels, layout) in enumerate(lossless):
                _check_cancel(cancel_event)
                records.append(collect_lossless_evidence(environment, encoder, rate, channels, layout,
                                                         work / f"lossless-{number}", cancel_event=cancel_event))
            _check_cancel(cancel_event)
            stats = (probe_mp3_encoder_stats(environment, cancel_event=cancel_event)
                     if mp3 else None)  # CR-067: once per run
            for number, (rate, channels, layout) in enumerate(mp3):
                _check_cancel(cancel_event)
                records.extend(collect_mp3_evidence(environment, rate, channels, layout, work / f"mp3-{number}",
                                                    encoder_stats=stats, cancel_event=cancel_event))
            for number, (rate, channels, layout) in enumerate(aac):
                _check_cancel(cancel_event)
                records.append(collect_aac_evidence(environment, rate, channels, layout, work / f"aac-{number}",
                                                    cancel_event=cancel_event))
    except (ToolError, OSError, ValueError) as exc:
        raise EnvironmentError_(f"cannot collect FFmpeg capability evidence: {exc}") from None
    _check_cancel(cancel_event)
    return tuple(records)


def _reencoded(probe: ProbeResult, target: TargetFormat) -> tuple[Diagnostic, ...]:
    """F14, C-13: an accepted same-codec lossy encode of a lossy source is one more generation.

    Reported once, for the accepted candidate only (``formats.resolve`` leaves this to the
    caller); a stream copy, a lossless output and an allowed codec switch report nothing here.
    """
    if (probe.codec_class != "lossy" or target.stream_copy or target.format in LOSSLESS_FORMATS
            or target.format != probe.codec):
        return ()
    setting = (f"V{target.vbr_quality}" if target.bitrate_mode == "vbr"
               else f"{(target.effective_bitrate_bps or 0) // 1000}k")
    return (Diagnostic("lossy_reencoded", f"re-encoded once as {target.format} ({setting}) to change its level "
                                          "— one more lossy generation", "render"),)


def _as_failure(exc: BaseException, stage: str) -> FileOutcomeError:
    """The C-13 refusal for an exception that ended a file (ARCHITECTURE § 7)."""
    if isinstance(exc, (Cancelled, KeyboardInterrupt)):
        return SkipFile("cancelled", stage, "interrupted before the file finished")
    if isinstance(exc, FileOutcomeError):
        return exc
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return FileFailure("internal", stage, f"unexpected {type(exc).__name__}: {exc}\n{detail}")


# --- Metadata through the pipeline (S4, TK-0427) ----------------------------------------------
#
# Metadata maps onto the § 4 states (metadata MD1): ``tags_read`` (C-06 and the mark, MD37),
# ``metadata_prepared`` (manifest C-11, filename inference MD9, artwork MD24), ``metadata_planned``
# after the final C-04/C-05 (WS-06's pure resolve and loss accounting, MD18–MD21), then after the
# accepted candidate ``mark_finalised`` (MD26, MD34), ``tags_written`` and ``content_verified``
# (MD22). The pipeline only sequences WS-06's public functions: it builds no tag object and
# decodes no payload; ``metadata/write.py`` and ``metadata/read.py`` (mutagen) are imported lazily.

UNREADABLE_TAG_CONTAINERS: frozenset[str] = frozenset({"caf", "adts"})
"""CR-036/CR-081: containers with no MD3 tag-store row; their tags cannot be read in v1."""


_EMPTY_TAGS = SourceTags(fields={}, extras={}, artwork=(), non_tag_items=(), internal_conflicts=(),
                         native_inventory=(), stores={})
_CLI_FIELDS: tuple[str, ...] = ("title", "artist", "album_artist", "album", "date", "track", "track_total",
                                "disc", "disc_total", "genre", "composer", "comment")
_STORABLE_COVERS = ("image/jpeg", "image/png")


def _write_edge() -> Any:
    """``metadata.write``, imported on first use: it imports mutagen (CR-040)."""
    from norma.metadata import write  # noqa: PLC0415 - CR-040

    return write


def _read_edge() -> Any:
    """``metadata.read``, imported on first use (CR-040)."""
    from norma.metadata import read  # noqa: PLC0415 - CR-040

    return read


def unreadable_tags(container: str) -> Diagnostic:
    """CR-081: the typed non-failure outcome for a container with no tag store (CAF, raw ADTS).

    C-13's existing ``metadata_loss_warn``: the tags cannot be read in v1, so ``normalise``
    cannot carry any, and the file is never claimed tag-free. No new reason is needed.
    """
    return Diagnostic("metadata_loss_warn",
                      f"{container} tags cannot be read in this version (no tag store is known for this "
                      "container), so any it carries are not copied to an output", "tags_read",
                      f"no_tag_store:{container}")


def _mark_number(value: float | None, *, signed: bool = False, strip: bool = False) -> str:
    """Compatibility pass-through to WS-06's MD26 number formatter."""
    from norma.metadata.model import format_mark_number  # noqa: PLC0415

    return format_mark_number(value, signed=signed, strip=strip)


def ffmpeg_version_token(version: str) -> str:
    """Compatibility pass-through to WS-06's MD26 version token."""
    from norma.metadata.model import ffmpeg_version_token as token  # noqa: PLC0415

    return token(version)


def processing_mark_value(gain: GainPlan, measurement: LoudnessMeasurement, *, origin: str, ffmpeg_version: str,
                          run_date: str, tool_version: str | None = None) -> str:
    """The MD26 mark text from the last verified audio plan and measurement.

    ``limited`` is C-09 ``limited_db`` (the policy bound), present only for ``mode=limit``.
    A non-finite value is a bug (``ValueError``, ``internal``), never written.
    """
    from norma.metadata.model import processing_mark_value as value  # noqa: PLC0415

    return value(gain, measurement, origin=origin, ffmpeg_version=ffmpeg_version, run_date=run_date,
                 tool_version=tool_version or norma.__version__)


def finalise_mark(plan: MetadataPlan, mark: str, *, target: str, wav_tags: str, id3_version: str) -> MetadataPlan:
    """``mark_finalised`` (MD26, MD34): a new revision whose only change is the mark.

    The mark item is prepared by WS-06's own recipe and store render (``mapping.native_recipes``
    with ``mark``, ``write.prepare_store``), so it carries the exact payload read-back produces.
    Its native row is ``output:…`` ``added`` and its roll-up ``extra:NORMA`` ``added``; user values,
    fates, the loss-policy result and sidecar requests are unchanged. A plan whose mark is
    ``omitted`` or ``not_storable`` is returned unchanged.
    """
    if plan.mark_state != "pending":
        return plan
    from norma.metadata import mapping, model  # noqa: PLC0415 - pure WS-06 helpers

    write = _write_edge()

    items: list[RawNativeItem] = []
    for store in mapping.target_stores(target, wav_tags):
        if mapping.extra_key(store, model.MARK_NAME, processing_mark=True) is None:
            continue
        recipes = mapping.native_recipes(store, {}, mark=mark, id3_version=id3_version)
        items.extend(write.prepare_store(store, recipes, id3_version=id3_version).items)
    return model.finalise_mark(plan, mark, items)


def metadata_requests_output(plan: MetadataPlan) -> bool:
    """Audio P4: whether the metadata plan asks for an output a stream copy would not otherwise need.

    True when an explicit or inferred opinion changes, adds or clears something, or a source item
    is removed on request. Source-derived accounting (``kept``, joined values, removals by design,
    the title fallback, losses) and the pending mark never create work (MD34).
    """
    for status in plan.statuses:
        if status.key == "extra:NORMA":
            continue
        if status.status == "removed_by_request":
            return True
        if status.status in ("changed", "added", "dropped_format_limit") and status.provenance not in (
                None, "source", "fallback") and status.detail != "pair_inconsistent":
            return True
    return False


def clear_keys(resolution: Mapping[str, Any]) -> tuple[str, ...]:
    """Roll-up keys whose winning opinion is a clear (MD41 needs them for the sidecar ``manifest``)."""
    return tuple(resolution.get("clear_keys", ()))


@dataclass(frozen=True)
class FileOpinions:
    """One file's explicit and inferred opinions (C-11 cells), before artwork preparation.

    ``artwork`` is the winning explicit artwork opinion (E1–E4) still to prepare, with the layer
    it ranks in: ``cli``, ``manifest``, ``manifest_default`` or ``folder``.
    """

    cli: Mapping[str, ManifestCell]
    generated: Mapping[str, ManifestCell]
    manifest: Mapping[str, ManifestCell]
    manifest_default: Mapping[str, ManifestCell]
    config: Mapping[str, ManifestCell]
    filename: Mapping[str, ManifestCell]
    locators: Mapping[str, str]
    artwork: Any
    artwork_layer: str | None
    folder_provenance: str
    warnings: tuple[Diagnostic, ...] = ()
    notes: tuple[str, ...] = ()


class MetadataInputs:
    """Run-level metadata inputs, loaded once before the first file (C-10, C-11; MD8–MD10, MD24).

    A bad manifest, filename pattern, explicit ``--artwork`` path or command-line metadata value
    is a ``UsageError`` (exit 2) here, before any file is touched. ``sips`` is located only when
    some file needs artwork preparation (E-05, exit 3 before any write).
    """

    def __init__(self, items: Sequence[InputItem], options: RunOptions, *,
                 cancel_event: threading.Event | None = None) -> None:
        from norma.metadata import artwork, infer, manifest  # noqa: PLC0415 - edge helpers, no mutagen

        self.options = options
        self.cancel_event = cancel_event
        self.notes: list[str] = []
        self.cli, self.config = _cli_layers(options)
        self.cli_artwork = (artwork.check_cli_artwork(options.artwork, cwd=options.cwd)
                            if options.artwork is not None else None)
        self.manifest = None
        self.matching = None
        if options.manifest is not None:
            path = options.manifest if options.manifest.is_absolute() else options.cwd / options.manifest
            self.manifest = manifest.load_manifest(path)
            self.matching = manifest.match_manifest(self.manifest, items)
            for issue in self.matching.unmatched:
                self.notes.append(f"manifest row {issue.locator} matches no input (unmatched row)")
            for issue in self.manifest.issues:
                if issue.detail in manifest.WARNING_DETAILS:
                    self.notes.append(manifest.format_issue(issue))
        self.pattern = None
        if options.infer_from_filename is not None:
            try:
                self.pattern = infer.compile_pattern(options.infer_from_filename)
            except infer.InvalidPattern as exc:
                raise UsageError(f"--infer-from-filename: {exc}") from None
        self.numbers = _track_numbers(items) if options.number_tracks else {}
        self._check_cli_values()
        self.sips: Any = None
        self._workdir: tempfile.TemporaryDirectory[str] | None = None
        self.preparer: Any = None
        self._artwork_reads: list[ReadDependency] = []

    def _check_cli_values(self) -> None:
        """Validate the command-line and config opinions once (exit 2), not once per file."""
        from norma.metadata import resolve  # noqa: PLC0415 - pure WS-06

        try:
            resolve.resolve_opinions(_EMPTY_TAGS, stem="x", cli=self.cli, config=self.config,
                                     prefer=self.options.prefer, no_source_tags=self.options.no_source_tags)
        except UsageError:
            raise
        except (ValueError, TypeError) as exc:
            raise UsageError(f"invalid metadata option: {exc}") from None

    @property
    def protected(self) -> tuple[ReadDependency, ...]:
        """The manifest, CLI artwork and every supplied image read are dependencies (C-08)."""
        paths = [path for path in (self.manifest.path if self.manifest is not None else None, self.cli_artwork)
                 if path is not None]
        return tuple(dict.fromkeys([*(safety.protect(path, "metadata") for path in paths),
                                    *self._artwork_reads]))

    def opinions(self, item: InputItem) -> FileOpinions:
        """Gather one file's opinions; ``manifest_row_invalid`` refuses only this file."""
        from norma.metadata import artwork, infer, manifest  # noqa: PLC0415

        options = self.options
        warnings: list[Diagnostic] = []
        notes: list[str] = []
        row_cells: Mapping[str, ManifestCell] = {}
        default_cells: Mapping[str, ManifestCell] = {}
        locators: dict[str, str] = {}
        if self.manifest is not None and self.matching is not None:
            match = self.matching.inputs[item.input_index]
            if match.failures:
                raise FileFailure("manifest_row_invalid", "metadata_prepared",
                                  "; ".join(manifest.format_issue(issue) for issue in match.failures))
            for issue in match.warnings:
                warnings.append(Diagnostic("manifest_row_ambiguous", f"manifest row {issue.locator} matches "
                                           "several inputs and was not applied", "metadata_prepared",
                                           manifest.format_issue(issue)))
            if match.row is not None:
                row_cells = match.row.cells
                locators["manifest"] = match.row.locator
            default_cells = self.manifest.defaults
            locators["manifest_default"] = f"{self.manifest.path.name}:defaults"
        filename: dict[str, ManifestCell] = {}
        if self.pattern is not None:
            inferred = infer.infer(self.pattern, item.relative_path, underscores=options.infer_underscores)
            filename = dict(inferred.cells)
            notes.extend(inferred.notes)
            locators["filename"] = str(item.relative_path)
        generated = {}
        if item.input_index in self.numbers:
            number, total = self.numbers[item.input_index]
            generated = {"track": ManifestCell("set", number), "track_total": ManifestCell("set", total)}
        folder_provenance = "config" if options.value_sources.get("folder_artwork") == "config" else "cli"
        folder = artwork.find_folder_artwork(item.source_path.parent) if options.folder_artwork else None
        remove = options.remove_artwork or "artwork" in options.clear_tag
        explicit = artwork.explicit_artwork(
            remove=remove, cli_path=self.cli_artwork, row_cell=row_cells.get("artwork"),
            default_cell=default_cells.get("artwork"),
            manifest_dir=self.manifest.path.parent if self.manifest is not None else None,
            folder_path=folder, folder_provenance=folder_provenance)
        if explicit is None:
            layer = None
        elif remove or (self.cli_artwork is not None and explicit.path == self.cli_artwork):
            layer = "cli"
        elif explicit.provenance in ("manifest", "manifest_default"):
            layer = explicit.provenance
        else:
            layer = "folder"
        return FileOpinions(
            cli=self.cli, generated=generated,
            manifest={k: v for k, v in row_cells.items() if k != "artwork"},
            manifest_default={k: v for k, v in default_cells.items() if k != "artwork"},
            config=self.config, filename=filename, locators=locators, artwork=explicit, artwork_layer=layer,
            folder_provenance=folder_provenance, warnings=tuple(warnings), notes=tuple(notes))

    # -- artwork preparation (MD24)
    def needs_sips(self, opinions: FileOpinions, tags: SourceTags, target_format: str | None) -> bool:
        """Whether this file's resolution prepares an image: a supplied set opinion that can win,
        or an MP4 target with a source front cover ``covr`` cannot type."""
        if self._uses_supplied(opinions, tags):
            return True
        if target_format in ("alac", "aac") and not self.options.no_source_tags:
            return any(tag.value.picture_type == 3 and _sniff(tag.value.data) is None for tag in tags.artwork)
        return False

    def _uses_supplied(self, opinions: FileOpinions, tags: SourceTags) -> bool:
        explicit = opinions.artwork
        if explicit is None or explicit.operation != "set":
            return False
        source_cover = any(tag.value.picture_type == 3 for tag in tags.artwork)
        # --prefer source: a source front cover wins (R6), so the supplied image is never used.
        return not (self.options.prefer == "source" and source_cover and not self.options.no_source_tags)

    def ensure_preparer(self) -> Any:
        """Locate ``sips`` (exit 3 when missing, E-05) and create the shared preparer once."""
        if self.preparer is None:
            from norma.metadata import artwork  # noqa: PLC0415

            self.sips = locate_sips_tool()
            self._workdir = tempfile.TemporaryDirectory(prefix="norma-artwork-")
            self.preparer = artwork.ArtworkPreparer(self.sips, Path(self._workdir.name),
                                                    max_size=self.options.artwork_max_size,
                                                    cancel_event=self.cancel_event)
        return self.preparer

    def close(self) -> None:
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None

    def resolve(self, item: InputItem, opinions: FileOpinions, tags: SourceTags,
                target_format: str | None) -> tuple[Mapping[str, Any], tuple[Any, ...], tuple[str, ...]]:
        """``metadata_prepared`` for one file: prepare artwork, then resolve (R1–R10, MD15, MD16).

        Returns the resolution, the supplied images' stamps (rechecked before use) and notes.
        ``--strict-conflicts`` fails here, before any render (``metadata_conflict_strict``).
        """
        from norma.metadata import resolve  # noqa: PLC0415

        options = self.options
        layers: dict[str, dict[str, ManifestCell]] = {
            "cli": dict(opinions.cli), "manifest": dict(opinions.manifest),
            "manifest_default": dict(opinions.manifest_default)}
        folder_cell = None
        stamps: tuple[Any, ...] = ()
        notes = list(opinions.notes)
        explicit = opinions.artwork
        if explicit is not None and explicit.operation == "clear":
            layers[opinions.artwork_layer or "cli"]["artwork"] = ManifestCell("clear", None)
        elif self._uses_supplied(opinions, tags):
            assert explicit is not None and explicit.path is not None
            # Keep the dependency even if preparation or later strict resolution fails.
            try:
                self._artwork_reads.append(safety.protect(explicit.path, "metadata"))
            except OSError as exc:
                # CR-094: stat precedes the artwork edge's own missing/unreadable check.
                raise FileFailure("artwork_invalid", "metadata_prepared", f"{explicit.path}: {exc}") from None
            image = _guard_tool(lambda: self.ensure_preparer().prepare(explicit.path))
            stamps = (image.stamp,) if image.stamp is not None else ()
            self._artwork_reads.extend(stamp.read_dependency() for stamp in stamps)
            notes.extend(image.notes)
            cell = ManifestCell("set", image.artwork)
            if opinions.artwork_layer == "folder":
                folder_cell = cell
            else:
                layers[opinions.artwork_layer or "cli"]["artwork"] = cell
        pictures = tuple(tag.value for tag in tags.artwork)
        if target_format in ("alac", "aac") and any(
                p.picture_type == 3 and _sniff(p.data) is None for p in pictures) and not options.no_source_tags:
            preparer = self.ensure_preparer()
            pictures = tuple(_guard_tool(lambda p=p: preparer.convert_embedded(p)) if p.picture_type == 3 else p
                             for p in pictures)
        resolution = resolve.resolve_opinions(
            tags, stem=item.source_path.stem, cli=layers["cli"], generated=opinions.generated,
            manifest=layers["manifest"], manifest_default=layers["manifest_default"], config=opinions.config,
            folder_artwork=folder_cell, folder_artwork_provenance=opinions.folder_provenance,
            filename=opinions.filename, locators=opinions.locators, prefer=options.prefer,
            infer_mode=options.infer_mode, no_source_tags=options.no_source_tags,
            no_title_fallback=options.no_title_fallback, strict_conflicts=options.strict_conflicts,
            source_pictures=pictures)
        notes.extend(resolution["notes"])
        return resolution, stamps, tuple(notes)

    def recheck(self, stamps: Sequence[Any]) -> None:
        """Every supplied image is rechecked before its bytes are written (``source_changed``)."""
        if stamps and self.preparer is not None:
            for stamp in stamps:
                self.preparer.recheck(stamp)


def _sniff(data: bytes) -> str | None:
    from norma.metadata.artwork import sniff_storable  # noqa: PLC0415

    return sniff_storable(data)


def locate_sips_tool() -> Any:
    """``adapters.sips.locate_sips`` (``EnvironmentError_``, exit 3, when missing: E-05)."""
    from norma.adapters.sips import locate_sips  # noqa: PLC0415

    return locate_sips()


def _guard_tool(call: Callable[[], T]) -> T:
    """CR-079: a ``sips`` lost or broken after preflight fails the file by its stage's tool-loss reason."""
    try:
        return call()
    except ToolError as exc:
        from norma.errors import tool_loss_reason  # noqa: PLC0415

        raise FileFailure(tool_loss_reason("metadata_prepared"), "metadata_prepared",
                          f"sips is no longer usable: {exc}") from None


def _cli_layers(options: RunOptions) -> tuple[dict[str, ManifestCell], dict[str, ManifestCell]]:
    """C-10 field options, ``--tag`` and ``--clear-tag`` as E1 cells; config-sourced fields as E4 (MD13)."""
    from norma.metadata import model  # noqa: PLC0415

    cli: dict[str, ManifestCell] = {}
    config: dict[str, ManifestCell] = {}
    for name in _CLI_FIELDS:
        value = getattr(options, name)
        if value is None:
            continue
        layer = config if options.value_sources.get(name) == "config" else cli
        layer[name] = ManifestCell("set", value)
    for key, value in options.tag:
        if key.casefold() == "compilation":
            try:
                cli["compilation"] = ManifestCell("set", model.parse_compilation(value))
            except ValueError as exc:
                raise UsageError(f"--tag compilation={value}: {exc}") from None
        else:
            cli[key] = ManifestCell("set", value)
    for key in options.clear_tag:
        if key == "artwork":
            continue  # a clear artwork opinion, handled with --remove-artwork (MD8)
        cli["compilation" if key.casefold() == "compilation" else key] = ManifestCell("clear", None)
    return cli, config


def _track_numbers(items: Sequence[InputItem]) -> dict[int, tuple[int, int]]:
    """``--number-tracks`` (MD8): 1…N per input folder in C-01 discovery order."""
    folders: dict[Path, list[int]] = {}
    for item in items:
        folders.setdefault(item.source_path.parent, []).append(item.input_index)
    return {index: (number, len(indices)) for indices in folders.values()
            for number, index in enumerate(indices, 1)}


def plan_metadata(tags: SourceTags, resolution: Mapping[str, Any], target: TargetFormat, options: RunOptions, *,
                  revision: int, conditional: bool = False, audio_basis: str | None = None) -> MetadataPlan:
    """``metadata_planned``: one C-07 revision for ``target`` (MD29, MD33, MD34).

    WS-06's ``preparation_requests`` names each store's recipes and passthrough items; each store
    is prepared at the edge by ``write.prepare_store`` and the pure ``build_metadata_plan``
    consumes those items and their explicit links (CR-083). An ``error`` loss policy raises
    ``metadata_loss_error`` here, before any render.
    """
    from norma.metadata import resolve  # noqa: PLC0415

    write, read = _write_edge(), _read_edge()
    basis = audio_basis or ("stream_copy" if target.stream_copy else "encode")
    resolution = resolve.with_audio_basis(resolution, basis)
    requests = resolve.preparation_requests(tags, resolution, target=target.format, wav_tags=options.wav_tags,
                                            id3_version=options.id3_version)
    prepared = [write.prepare_store(store, request["recipes"], request["passthrough"],
                                    id3_version=options.id3_version) for store, request in requests.items()]
    return resolve.build_metadata_plan(
        tags, resolution, target=target.format, prepared_stores=prepared,
        source_field_ids=read.source_field_ids(tags), wav_tags=options.wav_tags, id3_version=options.id3_version,
        on_metadata_loss=options.on_metadata_loss, plan_revision=revision, conditional=conditional,
        no_mark_processed=options.no_mark_processed)


def wav_info_pairs(plan: MetadataPlan | None, target: TargetFormat) -> tuple[tuple[str, str], ...]:
    """The render's RIFF INFO ``-metadata`` pairs for a WAV target (R8, MD30); none otherwise."""
    if plan is None or target.format != "wav":
        return ()
    return tuple(_write_edge().riff_info_metadata(plan.expected_native))


class _Job:
    """Records gathered for one file across planning and its worker; one terminal record."""

    def __init__(self, item: InputItem) -> None:
        self.item = item
        self.stage = "probe"
        self._spent = 0.0
        self._resumed: float | None = None
        self.probe: ProbeResult | None = None
        self.snapshot: safety.SourceSnapshot | None = None
        self.measurement: LoudnessMeasurement | None = None
        self.gain: GainPlan | None = None
        self.target: TargetFormat | None = None
        self.output: OutputPlan | None = None
        self.allocated: OutputPlan | None = None
        self.result: RenderResult | None = None
        self.verification: Verification | None = None
        self.accepted: tuple[Diagnostic, ...] | None = None
        """The ladder's acceptance warnings (V3 step 3, V10), set once a candidate is accepted."""
        self.record: Normalised | None = None
        # metadata (S4): C-06, the mark, opinions, resolution, the current C-07 revision, read-back
        self.tags: SourceTags | None = None
        self.mark: ProcessingMark | None = None
        self.provisional: TargetFormat | None = None
        self.opinions: FileOpinions | None = None
        self.resolution: Mapping[str, Any] | None = None
        self.stamps: tuple[Any, ...] = ()
        self.metadata: MetadataPlan | None = None
        self.tag_write: TagWriteResult | None = None
        self.meta_warnings: list[Diagnostic] = []
        self.meta_notes: tuple[str, ...] = ()

    # -- time
    def resume(self) -> None:
        self._resumed = time.monotonic()

    def pause(self) -> None:
        if self._resumed is not None:
            self._spent += time.monotonic() - self._resumed
            self._resumed = None

    def elapsed(self) -> float:
        running = time.monotonic() - self._resumed if self._resumed is not None else 0.0
        return max(0.0, self._spent + running)

    # -- outcomes
    def warnings(self) -> tuple[Diagnostic, ...]:
        """Warning-family diagnostics from every record gathered, in pipeline order, once each.

        Once a candidate is accepted, its measured ``reduce_shortfall`` (the ladder's
        acceptance warning) replaces the plan's predicted one: a committed file reports
        only the measured shortfall (audio V10).
        """
        sources: list[Sequence[Diagnostic]] = []
        if self.gain is not None:  # already includes the measurement's
            planned = self.gain.diagnostics
            if self.accepted is not None:
                planned = tuple(d for d in planned if d.code != "reduce_shortfall")
            sources.append(planned)
        elif self.measurement is not None:
            sources.append(self.measurement.diagnostics)
        if self.target is not None:
            sources.append(self.target.diagnostics)
        if self.mark is not None:
            sources.append(self.mark.diagnostics)  # processing_mark_invalid (MD37)
        sources.append(self.meta_warnings)
        if self.metadata is not None:
            sources.append(self.metadata.diagnostics)  # MD21 loss warnings
        if self.accepted is not None:
            sources.append(self.accepted)
        found: list[Diagnostic] = []
        for diagnostics in sources:
            for diagnostic in diagnostics:
                if reason_spec(diagnostic.code).kind == "warning" and diagnostic not in found:
                    found.append(diagnostic)
        return tuple(found)

    def finish(self, status: Status, reason: ReasonCode, execution: Execution, *,
               primary: Diagnostic | None = None, diagnostics: tuple[Diagnostic, ...] = (),
               notes: tuple[str, ...] = (), published: tuple[Path, ...] = (),
               retained: Path | None = None) -> Normalised:
        check_primary(status, reason)
        self.pause()
        outcome = FileOutcome(
            status=status, reason_code=reason, input=self.item, probe=self.probe, target=self.target,
            measurement_in=self.measurement,
            measurement_out=self.verification.measurement if self.verification is not None else None,
            gain=self.gain, verification=self.verification, verification_reference=None,
            source_codec=self.probe.codec if self.probe is not None else None, origin_codec=None,
            metadata=self.metadata, tag_write=self.tag_write, output=self.output, conditional_plans=(),
            warnings=self.warnings(), elapsed_s=self.elapsed(),
            ffmpeg_stderr_tail=self.result.stderr_tail if self.result is not None else None,
            execution=execution, published_paths=published, retained_failed_path=retained,
        )
        target_notes = self.target.notes if self.target is not None else ()
        self.record = Normalised(outcome, primary, diagnostics,
                                 tuple(target_notes) + self.meta_notes + tuple(notes))
        return self.record

    def finish_ok(self, execution: Execution, *, notes: tuple[str, ...] = (),
                  published: tuple[Path, ...] = ()) -> Normalised:
        """An accepted file: ``already_within_tolerance`` for a copy, else ok; warnings promote."""
        assert self.gain is not None
        base: ReasonCode = "already_within_tolerance" if self.gain.action == "copy" else "ok"
        status, reason = settle_primary("ok", base, self.warnings())
        return self.finish(status, reason, execution, notes=notes, published=published)

    def finish_error(self, exc: BaseException, execution: Execution) -> Normalised:
        """A file ended in planning, before anything of it was allocated."""
        error = _as_failure(exc, self.stage)
        return self.finish(error.status, error.reason, execution, primary=error.diagnostic(),
                           diagnostics=error.diagnostics)

    def finish_commit(self, result: safety.CommitResult) -> Normalised:
        """The truthful C-09 record for what publication, rollback or cleanup actually did."""
        if result.status == "ok" and result.execution == "committed":
            return self.finish_ok("committed", notes=result.notes, published=result.published_paths)
        primary = result.diagnostics[0] if result.diagnostics else None
        return self.finish(result.status, result.reason, result.execution, primary=primary,
                           diagnostics=tuple(result.diagnostics[1:]), notes=result.notes,
                           published=result.published_paths, retained=result.retained_failed_path)


class _NormaliseRun:
    """One ``normalise`` run: planning in the coordinator, then per-file workers."""

    def __init__(self, items: Sequence[InputItem], options: RunOptions, tools: NormaliseTools,
                 naming: NamingOptions, cancel_event: threading.Event, run_id: str,
                 fault_hook: safety.FaultHook | None) -> None:
        self.items = tuple(items)
        self.options = options
        self.tools = tools
        self.naming = naming
        self.cancel_event = cancel_event
        self.run_id = run_id
        self.fault_hook = fault_hook
        self.execution: Execution = "plan" if options.dry_run else "not_written"
        self.jobs: dict[int, _Job] = {item.input_index: _Job(item) for item in self.items}
        if len(self.jobs) != len(self.items):
            raise ValueError("items must have unique input_index values")
        self.evidence: tuple[CapabilityEvidence, ...] = ()
        self.protected: tuple[ReadDependency, ...] = ()
        self.registry: safety.RunRegistry | None = None
        self.checked: safety.Preflight | None = None
        self.book: safety.ReservationBook | None = None
        self.meta: MetadataInputs | None = None
        self.run_date = time.strftime("%Y-%m-%d")  # MD26 ``date``: run start, local

    # -- orchestration
    def run(self, execute: Execute | None, run_outputs: RunOutputsSpec) -> NormaliseRun:
        self.meta = MetadataInputs(self.items, self.options, cancel_event=self.cancel_event)  # exit 2 here
        try:
            return self._run(execute, run_outputs)
        finally:
            self.meta.close()

    def _run(self, execute: Execute | None, run_outputs: RunOutputsSpec) -> NormaliseRun:
        assert self.meta is not None
        live = self._probe_and_read_tags()
        if live is None:
            return self._stopped(tuple(self.meta.notes))
        try:
            self._check_cancel()
            notes = self._resolve_targets(live) + tuple(self.meta.notes)
        except Cancelled:
            return self._stopped(tuple(self.meta.notes))
        self._gather_opinions(live)
        live = self._measure_and_plan(live)
        if live is None:
            return self._stopped(notes)
        outputs = self._preflight(live, run_outputs)
        checked = self.checked
        assert checked is not None
        eligible = [job for job in live if job.item.input_index not in checked.refusals]
        for job in live:
            refusal = checked.refusals.get(job.item.input_index)
            job.output = checked.plans[job.item.input_index]
            if refusal is not None:
                job.finish_error(refusal, self.execution)
        if self.options.dry_run:
            for job in eligible:
                job.finish_ok("plan")
            return NormaliseRun(self._records(), 0, False, True, notes, (), outputs)
        assert execute is not None
        return self._execute(execute, eligible, notes, outputs)

    def _records(self) -> tuple[Normalised, ...]:
        return tuple(job.record for job in self.jobs.values() if job.record is not None)

    def _stopped(self, notes: tuple[str, ...] = ()) -> NormaliseRun:
        """Cancelled during planning: nothing was written; unplanned files are not started."""
        records = self._records()
        return NormaliseRun(records, len(self.items) - len(records), True, self.options.dry_run, notes)

    def _cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def _check_cancel(self) -> None:
        if self._cancelled():
            raise Cancelled()

    # -- planning, phase 1: signature, probe, tags and the processing mark (MD37, B-002)
    def _probe_and_read_tags(self) -> list[_Job] | None:
        live: list[_Job] = []
        for job in self.jobs.values():
            if self._cancelled():
                return None
            job.resume()
            try:
                job.probe = classify(job.item, self.tools.prober, cancel_event=self.cancel_event)
                self._check_cancel()
                job.stage = "tags_read"
                self._read_tags(job)
            except (Cancelled, Exception) as exc:  # noqa: BLE001 - per-file containment (§ 7)
                job.finish_error(exc, self.execution)
                if isinstance(exc, Cancelled):
                    return None
                continue
            job.pause()
            live.append(job)
        return live

    def _read_tags(self, job: _Job) -> None:
        """``tags_read``: C-06 and the mark; a marked file is ``already_processed`` unless ``--force``.

        CAF and raw ADTS have no tag store (CR-036): CR-081's typed outcome is a
        ``metadata_loss_warn`` (no tag can be carried), or ``metadata_loss_error`` under
        ``--on-metadata-loss error``, never an internal failure.
        """
        assert job.probe is not None
        container = job.probe.container
        if container in UNREADABLE_TAG_CONTAINERS:
            warning = unreadable_tags(container)
            if self.options.on_metadata_loss == "error":
                raise FileFailure("metadata_loss_error", "tags_read", warning.message, (warning,))
            job.tags = _EMPTY_TAGS
            job.meta_warnings.append(warning)
        else:
            job.tags = read_source_tags(job.item.source_path, container)
        job.mark = read_processing_mark(job.tags)
        if job.mark.validity != "absent" and not self.options.force:  # MD26 detection, formats F17
            state = "" if job.mark.validity == "valid" else f" ({job.mark.validity})"
            raise SkipFile("already_processed", "tags_read",
                           f"the file carries a Norma processing mark{state}; use --force to process it again")

    # -- planning, phase 2: C-04 evidence and all-input target preflight (exit 2/3 before writes)
    def _resolve_targets(self, live: list[_Job]) -> tuple[str, ...]:
        probes = [job.probe for job in live if job.probe is not None]
        self.evidence = collect_evidence(self.tools.environment, probes, self.options,
                                         cancel_event=self.cancel_event)
        self._check_cancel()
        results, notes = preflight_targets(
            [job.probe for job in live], self.options, self.evidence, self.tools.basis,
            source_suffixes=[job.item.source_path.suffix for job in live],
            input_paths=[job.item.source_path for job in live],
            available_encoders=self.tools.environment.encoders)
        for job, result in zip(live, results, strict=True):
            job.stage = "format"
            if isinstance(result, FileOutcomeError):
                job.finish_error(result, self.execution)
            elif isinstance(result, TargetFormat) and result.decision != "skipped_lossy":
                job.provisional = result
            elif result is not None and result.decision == "skipped_lossy":
                job.target = result
                job.finish_error(SkipFile("lossy_source_skipped", "format",
                                          f"lossy {job.probe.codec} source skipped (--lossy-sources skip)"),
                                 self.execution)
        return notes

    # -- planning, phase 2b: opinions and the sips preflight (MD8–MD10, MD24; E-05 exit 3)
    def _gather_opinions(self, live: list[_Job]) -> None:
        meta = self.meta
        assert meta is not None
        needs_sips = False
        for job in live:
            if job.record is not None:
                continue
            job.resume()
            job.stage = "metadata_prepared"
            try:
                job.opinions = meta.opinions(job.item)
            except Exception as exc:  # noqa: BLE001 - per-file containment (§ 7)
                job.finish_error(exc, self.execution)
                continue
            job.pause()
            job.meta_warnings.extend(job.opinions.warnings)
            assert job.tags is not None
            target = job.provisional.format if job.provisional is not None else None
            needs_sips = needs_sips or meta.needs_sips(job.opinions, job.tags, target)
        if needs_sips:
            meta.ensure_preparer()  # EnvironmentError_ (exit 3) before any write

    # -- planning, phase 3: snapshot, metadata_prepared, measurement, plan, final C-04, C-08 path
    def _measure_and_plan(self, live: list[_Job]) -> list[_Job] | None:
        planned: list[_Job] = []
        for job in live:
            if job.record is not None:
                continue
            if self._cancelled():
                return None
            job.resume()
            try:
                self._plan_one(job)
            except (Cancelled, Exception) as exc:  # noqa: BLE001 - per-file containment (§ 7)
                job.finish_error(exc, self.execution)
                if isinstance(exc, Cancelled):
                    return None
                continue
            if job.record is None:
                job.pause()
                planned.append(job)
        return planned

    def _plan_one(self, job: _Job) -> None:
        item, probe, options = job.item, job.probe, self.options
        assert probe is not None
        if not options.dry_run:  # § 5.1 source stability: captured before processing
            job.stage = "source"
            job.snapshot = safety.snapshot_source(item)
        job.stage = "metadata_prepared"
        assert self.meta is not None and job.opinions is not None and job.tags is not None
        job.resolution, job.stamps, job.meta_notes = self.meta.resolve(
            item, job.opinions, job.tags, job.provisional.format if job.provisional is not None else None)
        job.stage = "measure"
        job.measurement = measure(item.source_path, probe, self.tools.executable,
                                  no_dual_mono=options.no_dual_mono, cancel_event=self.cancel_event)
        self._check_cancel()
        calibration: DynamicCalibration | None = None
        if not options.dry_run and needs_dynamic_calibration(job.measurement, options):  # R5: at ceiling c
            calibration = calibrate_dynamic(item.source_path, probe, self.tools.executable,
                                            target_lufs=options.target, ceiling_dbtp=options.true_peak,
                                            no_dual_mono=options.no_dual_mono, cancel_event=self.cancel_event)
            self._check_cancel()
        job.stage = "plan"
        job.gain = (plan_dynamic_preview(job.measurement, options)
                    if options.dry_run and options.peak_mode == "dynamic"
                    else plan_gain(job.measurement, options, calibration=calibration))
        if job.gain.action == "skip":  # silent or unmeasurable: SKIP with the meter's diagnostic
            primary = next((d for d in job.gain.diagnostics if d.code == job.gain.reason_code),
                           Diagnostic(job.gain.reason_code, "no gain plan for this signal", "plan"))
            job.finish("skipped", job.gain.reason_code, self.execution, primary=primary)
            return
        job.stage = "format"
        job.target = finalise_target(probe, options, self.evidence, self.tools.basis, action=job.gain.action,
                                     source_suffix=item.source_path.suffix)
        job.stage = "metadata_planned"
        job.metadata = self._plan_metadata(job, job.target, revision=1)
        if job.gain.action == "copy" and job.target.stream_copy and not metadata_requests_output(job.metadata):
            # audio P4: unchanged representation and no requested metadata change; the mark never
            # creates work (MD34), so no output is needed.
            primary = Diagnostic("already_within_tolerance", "within tolerance and peak-safe; no output needed",
                                 "plan")
            job.metadata = None
            job.finish("skipped", "already_within_tolerance", self.execution, primary=primary)
            return
        job.stage = "path_planned"
        job.output = self._plan_output(job, job.target)

    def _plan_metadata(self, job: _Job, target: TargetFormat, *, revision: int) -> MetadataPlan:
        """One C-07 revision, including the sidecar members reserved in C-08."""
        assert job.tags is not None and job.resolution is not None
        return plan_metadata(job.tags, job.resolution, target, self.options, revision=revision)

    def _plan_output(self, job: _Job, target: TargetFormat) -> OutputPlan:
        """C-08 bundle bound to the current C-07 revision, with its sidecar members (MD36)."""
        plan = job.metadata
        assert plan is not None
        values = {name: value.value for name, value in plan.resolved.items()}
        return plan_output(job.item, target, self.naming, plan_revision=plan.plan_revision,
                           sidecar_requests=plan.sidecar_requests, values=values, overwrite=self.options.overwrite)

    # -- the write barrier (safety § 2.6)
    def _preflight(self, live: list[_Job], run_outputs: RunOutputsSpec) -> RunOutputs:
        plans = {job.item.input_index: job.output for job in live if job.output is not None}
        outputs = tuple(run_outputs(dict(plans)) if callable(run_outputs) else run_outputs)
        executables = (self.tools.executable.path, self.tools.prober.path)
        assert self.meta is not None
        stamps = [stamp.read_dependency() for job in live for stamp in job.stamps]
        if self.meta.sips is not None:
            executables += (self.meta.sips.path,)
        self.protected = _protected_reads(self.items, self.options, executables) + self.meta.protected + tuple(
            dict.fromkeys(stamps))
        self.checked = safety.preflight(plans, outputs, self.protected, overwrite=self.options.overwrite)
        self.registry = safety.RunRegistry(self.run_id, fault_hook=self.fault_hook)
        self.book = safety.ReservationBook(self.checked, self.protected)  # retries' new members (§ 5.3)
        return RunOutputs(self.checked, self.protected, self.registry, overwrite=self.options.overwrite,
                          requests=outputs)

    # -- execution
    def _execute(self, execute: Execute, eligible: list[_Job], notes: tuple[str, ...],
                 outputs: RunOutputs) -> NormaliseRun:
        assert self.registry is not None
        eligible_ids = {job.item.input_index for job in eligible}
        entries: list[InputItem | FileOutcome] = []
        for index, job in self.jobs.items():
            if index in eligible_ids:
                entries.append(job.item)
            elif job.record is not None:
                entries.append(job.record.outcome)
        request = BatchRequest(
            entries=tuple(entries), worker=self._worker, settle_error=self._settle_error,
            jobs=self.options.jobs, commit_lock=self.registry.commit_lock, cancel_event=self.cancel_event,
            force_hook=force_kill_tools, record_for=self._record_for, outputs=outputs)
        try:
            result = execute(request)
        finally:
            problems = self.registry.close()
        records = tuple(self._record_for(outcome) for outcome in result.outcomes)
        return NormaliseRun(records, result.not_started, result.cancelled, False, notes, tuple(problems), outputs)

    def _record_for(self, outcome: FileOutcome) -> Normalised:
        job = self.jobs.get(outcome.input.input_index)
        if job is not None and job.record is not None and job.record.outcome is outcome:
            return job.record
        return Normalised(outcome)

    def _worker(self, item: InputItem, control: BatchControlLike) -> FileOutcome:
        """Run the bounded retry ladder for one planned file, then publish the accepted candidate.

        Each attempt allocates, renders from the original source, checks and verifies one
        candidate (§ 4, safety § 5); ``verify.run_ladder`` decides acceptance, the one next
        retry or the terminal failure (ADR-0014: at most five renders, budgets never reset).
        A rejected candidate is removed before the next allocation, so at most one candidate
        of the file exists at a time; the last one is what ``--keep-failed`` may retain.
        """
        job = self.jobs[item.input_index]
        registry = self.registry
        assert registry is not None and self.checked is not None
        assert job.output is not None and job.snapshot is not None and job.target is not None
        assert job.gain is not None and job.probe is not None and job.measurement is not None
        cancel = control.cancel_event
        job.resume()
        job.stage = "allocated"
        if cancel.is_set():
            raise Cancelled()

        def attempt(plan: GainPlan, number: int) -> tuple[RenderResult, Verification]:
            return self._attempt(job, plan, number, cancel)

        def before_retry(previous: GainPlan, following: GainPlan, kind: RetryKind) -> None:
            self._before_retry(job, previous, following, cancel)

        def calibrate(ceiling: float) -> DynamicCalibration:  # V6: a fresh analysis of the original source
            assert job.probe is not None and job.gain is not None
            job.stage = "measure"
            return calibrate_dynamic(item.source_path, job.probe, self.tools.executable,
                                     target_lufs=job.gain.target_lufs, ceiling_dbtp=ceiling,
                                     no_dual_mono=self.options.no_dual_mono, cancel_event=cancel)

        ladder = run_ladder(job.measurement, job.gain, attempt, before_retry, calibrate=calibrate)
        job.gain, job.result, job.verification = ladder.plan, ladder.result, ladder.verification
        job.accepted = ladder.warnings + _reencoded(job.probe, ladder.result.target)
        assert job.allocated is not None
        staged = self._write_tags(job, cancel)
        job.stage = "committing"
        result = safety.commit_bundle(registry, item, job.allocated, snapshot=job.snapshot,
                                      protected=self.protected, metadata_revision=job.allocated.plan_revision,
                                      staged_member_ids=staged, cancel_event=cancel,
                                      interruption_requested=lambda: control.interrupted)
        return job.finish_commit(result).outcome

    def _write_tags(self, job: _Job, cancel: threading.Event) -> tuple[str, ...]:
        """``mark_finalised`` → ``tags_written`` → ``content_verified`` on the accepted candidate.

        The mark takes the last verified values (MD26) in a new C-07 revision, to which the C-08
        bundle is rebound (§ 5.3). WS-06 writes the expected stores into the private candidate and
        reads every store back (MD22); a mismatch is ``verification_tags``. The tagged bytes are
        then decoded and measured again: the tag write must leave the verified audio identical
        (ARCHITECTURE § 4 ``content_verified``), or the file fails ``verification_tags``.
        """
        item, registry, plan = job.item, self.registry, job.metadata
        assert registry is not None and plan is not None and job.allocated is not None and job.target is not None
        assert job.verification is not None and job.gain is not None and job.probe is not None
        if cancel.is_set():
            raise Cancelled()
        job.stage = "mark_finalised"
        if plan.mark_state == "pending":
            origin = job.probe.codec
            if job.mark is not None and job.mark.validity == "valid" and job.mark.origin_codec:
                origin = job.mark.origin_codec  # --force on a Norma output keeps its recorded origin
            mark = processing_mark_value(job.gain, job.verification.measurement, origin=origin,
                                         ffmpeg_version=self.tools.executable.version, run_date=self.run_date)
            plan = finalise_mark(plan, mark, target=job.target.format, wav_tags=self.options.wav_tags,
                                 id3_version=self.options.id3_version)
            job.metadata = plan
            job.allocated = rebind(job.allocated, plan.plan_revision)
            job.output = rebind(job.output, plan.plan_revision) if job.output is not None else None
        job.stage = "tags_written"
        assert self.meta is not None
        self.meta.recheck(job.stamps)
        candidate = job.allocated.audio_member.temporary_path
        assert candidate is not None and job.resolution is not None
        protected = (item.source_path, *(dependency.path for dependency in self.protected))
        tagged = job.target.format != "wav" or self.options.wav_tags == "info+id3"  # mutagen writes the candidate
        before = self._digest(job, candidate, cancel) if tagged else None
        written = _write_edge().write_tags(
            candidate, plan, target=job.target.format, wav_tags=self.options.wav_tags,
            id3_version=self.options.id3_version, protected=protected,
            audio_name=job.allocated.audio_member.final_path.name, clears=clear_keys(job.resolution))
        job.tag_write = written.result
        if not written.result.matched:
            raise FileFailure("verification_tags", "content_verified", "tag read-back does not match the plan: "
                              + ", ".join(written.result.mismatches), tuple(written.result.diagnostics))
        if cancel.is_set():
            raise Cancelled()
        job.stage = "content_verified"
        registry.check_structure(item.input_index, job.allocated.audio_member)
        if before is not None:
            after = self._digest(job, candidate, cancel)
            if after != before:  # ARCHITECTURE § 4: the final tagged bytes' audio is the verified audio
                raise FileFailure("verification_tags", "content_verified",
                                  f"writing tags changed the decoded audio ({before[:12]} → {after[:12]})")
        staged: list[str] = []
        members = {member.member_id: member for member in job.allocated.members if member.purpose != "audio"}
        if set(written.result.staged_member_ids) != set(members) or set(written.staged) != set(members):
            raise FileFailure("verification_tags", "content_verified", "sidecar bytes do not match the reserved bundle")
        for member_id in written.result.staged_member_ids:
            if cancel.is_set():
                raise Cancelled()
            registry.stage_member(item.input_index, members[member_id], written.staged[member_id])
            staged.append(member_id)
        return tuple(staged)

    def _digest(self, job: _Job, candidate: Path, cancel: threading.Event) -> str:
        """Decoded-sample SHA-256 of the candidate (audio/verify), before and after the tag write."""
        duration = job.probe.duration_s if job.probe is not None else None
        return decoded_digest(candidate, self.tools.executable, duration_s=duration, cancel_event=cancel)

    def _attempt(self, job: _Job, plan: GainPlan, number: int, cancel: threading.Event,
                 ) -> tuple[RenderResult, Verification]:
        """One ladder attempt: dispose of the rejected candidate, allocate, render, check, verify."""
        item, registry = job.item, self.registry
        assert registry is not None and self.checked is not None and job.snapshot is not None
        assert job.output is not None and job.target is not None and job.probe is not None
        assert job.measurement is not None
        if cancel.is_set():
            raise Cancelled()
        job.gain = plan
        self._dispose(job)
        job.result = job.verification = None
        job.stage = "allocated"
        assert self.book is not None  # holds the preflight pins plus every retry reservation's
        pins = {safety.destination_id(item.input_index, member.member_id): pin for member in job.output.members
                if (pin := self.book.pin(item.input_index, member.member_id)) is not None}
        assert job.metadata is not None and job.resolution is not None
        # MD26 adds a revision, but no loss or export. MD41 serialises the final
        # revision in JSON, so estimate those exact bytes before allocating anything.
        sidecar_plan = (evolve(job.metadata, plan_revision=job.metadata.plan_revision + 1)
                        if job.metadata.mark_state == "pending" else job.metadata)
        sidecars = _write_edge().stage_members(sidecar_plan, audio_name=job.output.audio_member.final_path.name,
                                              clears=clear_keys(job.resolution))
        job.allocated = registry.allocate_plan(item.input_index, job.output, pins=pins,
                                               worker=threading.current_thread().name,
                                               space=safety.estimate_space(job.snapshot.size,
                                                                           job.probe.duration_s, job.target,
                                                                           sidecar_bytes=sum(map(len, sidecars.values()))))
        candidate = job.allocated.audio_member.temporary_path
        assert candidate is not None
        job.stage = "render"
        safety.recheck_source(item, job.snapshot, stage="render")
        job.result = render(item.source_path, item.identity, job.probe, plan, job.target, candidate,
                            self.tools.environment, self.evidence, attempt=number,
                            flac_level=self.options.flac_level, wav_info=wav_info_pairs(job.metadata, job.target),
                            no_dual_mono=self.options.no_dual_mono,
                            cancel_event=cancel, source_frames=job.measurement.decoded_frames)
        job.stage = "content_verified"
        registry.check_structure(item.input_index, job.allocated.audio_member)
        job.stage = "verify"
        candidate_probe = self._probe_candidate(candidate, cancel)
        job.verification = verify_candidate(job.result, item.source_path, job.probe, candidate_probe,
                                            self.tools.executable, no_dual_mono=self.options.no_dual_mono,
                                            source_frames=job.measurement.decoded_frames,
                                            cancel_event=cancel)
        return job.result, job.verification

    def _dispose(self, job: _Job) -> None:
        """Remove a rejected candidate before the next allocation; it never becomes a source (V3)."""
        if job.allocated is None or self.registry is None:
            return
        job.stage = safety.CLEANUP_STAGE
        paths = [member.temporary_path for member in job.allocated.members if member.temporary_path is not None]
        problems = self.registry.cleanup_paths(paths)
        job.allocated = None
        if problems:
            raise FileFailure("cleanup_error", safety.CLEANUP_STAGE,
                              "a rejected candidate could not be removed: " + "; ".join(problems))

    def _before_retry(self, job: _Job, previous: GainPlan, following: GainPlan, cancel: threading.Event) -> None:
        """V3a before every retry: ``gain_planned`` → ``metadata_planned`` → ``writes_reserved``.

        The C-04 record is finalised again for the retry's action (C-04: after every
        action-changing retry). A retry that changes the audio basis, the C-04 target or the
        C-05 action makes a new C-07 revision from scratch (MD34; its loss policy may refuse
        the file before the new render); other retries keep the revision. The C-08 bundle is
        rebound to the revision, and members it did not hold before are reserved for the rest
        of the run (safety § 5.3), refusing only this file on a conflict.
        """
        item, probe = job.item, job.probe
        assert probe is not None and job.output is not None and job.target is not None
        assert job.metadata is not None
        if cancel.is_set():
            raise Cancelled()
        job.stage = "gain_planned"
        target = finalise_target(probe, self.options, self.evidence, self.tools.basis, action=following.action,
                                 source_suffix=item.source_path.suffix)
        job.stage = "metadata_planned"
        if target != job.target or following.action != previous.action:
            job.metadata = self._plan_metadata(job, target, revision=job.metadata.plan_revision + 1)
        job.stage = safety.RESERVE_STAGE
        if job.metadata.plan_revision != job.output.plan_revision:
            output = self._plan_output(job, target)
            held = {member.member_id: member.comparison_key for member in job.output.members}
            if all(held.get(member.member_id) == member.comparison_key for member in output.members):
                output = OutputPlan(output.plan_revision, output.members, job.output.collision,
                                    output.source_identity, output.overwrite_authorised)
            else:
                assert self.book is not None
                output = self.book.reserve_retry(item.input_index, output)
            job.output = output
        job.target = target

    def _probe_candidate(self, candidate: Path, cancel: threading.Event) -> ProbeResult:
        """C-02 for the candidate; one that cannot be probed is a defective render."""
        try:
            return probe_file(self.tools.prober, candidate, cancel_event=cancel)
        except FileOutcomeError as exc:
            reason: ReasonCode = "timeout" if exc.reason == "timeout" else "render_error"
            raise FileFailure(reason, "verify", f"the candidate cannot be probed: {exc}") from None

    def _settle_error(self, item: InputItem, exc: BaseException, elapsed: float,
                      control: BatchControlLike) -> FileOutcome:
        """Discard what the file left and return its truthful outcome (safety § 5.2, § 6).

        Writers were already reaped by the adapters before the exception reached here.
        ``--keep-failed`` retains only a candidate that failed verification, never on
        cancellation; a cleanup failure keeps the earlier failure primary (C-13).
        """
        del elapsed, control
        job = self.jobs[item.input_index]
        assert self.registry is not None
        if job.record is not None and job.record.outcome.execution in ("committed", "publication_failed"):
            return job.record.outcome  # never claim a publication did not happen
        error = _as_failure(exc, job.stage)
        keep = None
        if (self.options.keep_failed and error.reason in VERIFICATION_FAILURES and job.allocated is not None):
            keep = job.allocated.audio_member.temporary_path
        result = safety.settle(self.registry, item.input_index, error, keep_failed=keep)
        primary = result.diagnostics[0] if result.diagnostics else error.diagnostic()
        secondary = tuple(result.diagnostics[1:]) + tuple(error.diagnostics)
        return job.finish(result.status, result.reason, result.execution, primary=primary,
                          diagnostics=secondary, published=result.published_paths,
                          retained=result.retained_failed_path).outcome
