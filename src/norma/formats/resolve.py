"""Pure C-04 resolution, all-input preflight and immutable copy finalisation.

The shell supplies C-02 probes, frozen C-10 options, run-local evidence and a
revalidated ToolBasis. This module performs no I/O. Missing evidence is not
proof of a missing encoder: the shell checks tools before calling this module,
or supplies its checked available_encoders set to preflight_targets.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Literal, NoReturn

from norma.errors import EnvironmentError_, FileFailure, SkipFile, UsageError
from norma.models import (
    LOSSLESS_FORMATS, NOT_APPLICABLE_BASIS, STREAM_COPY_ENCODER,
    CapabilityEvidence, Diagnostic, ProbeResult, ReasonCode, RunOptions, TargetFormat,
    TargetFormatName,
)
from norma.formats import capabilities as cap

_STAGE = "format"
_Decision = Literal["as_requested", "kept_source_codec", "lossy_origin", "skipped_lossy"]


def effective_channel_layout(channels: int, layout: str | None) -> tuple[str | None, bool]:
    """D-081: assume standard mono/stereo only when the input has no layout.

    Return (effective layout, assumed). Keep C-02 and stated layouts unchanged;
    the shell also uses this rule before collecting run-local evidence.
    """
    if layout is None and channels in (1, 2):
        return ("mono" if channels == 1 else "stereo"), True
    return layout, False


def _source_note(options: RunOptions, field: str) -> str:
    source = options.value_sources.get(field)
    if source == "config":
        return " (from the config)"
    if source == "cli" or field in options.explicit_fields:
        return " (from the command line)"
    return ""


def _ignored_streams_note(count: int) -> str:
    noun = "audio stream" if count == 1 else "audio streams"
    return f"{count} extra {noun} not carried over; only the first audio stream is processed"


def _warn(code: ReasonCode, message: str) -> Diagnostic:
    return Diagnostic(code, message, _STAGE)


def _same(probe: ProbeResult) -> TargetFormatName | None:
    # D-064 adapts the S0 table without widening this task's file allocation.
    if (probe.codec_class == "lossless" and probe.codec.startswith("pcm_")
            and probe.container in ("mp4", "mov")):
        return "alac"
    return cap.same_target_format(probe.codec, probe.codec_class, probe.container)


def _choose(probe: ProbeResult, options: RunOptions) -> tuple[TargetFormatName, _Decision]:
    if probe.codec_class == "unsupported":
        raise SkipFile("unsupported_codec", _STAGE, probe.codec)
    if probe.channels > 2:
        raise SkipFile("unsupported_channels", _STAGE, str(probe.channels))
    same = _same(probe)
    if probe.codec_class == "lossy" and options.lossy_sources == "skip":
        assert same is not None
        return same, "skipped_lossy"
    if probe.codec_class == "lossy" and options.lossy_sources == "lossless":
        target = options.format if options.format in LOSSLESS_FORMATS else (
            "alac" if options.format == "apple" else "flac")
    elif options.format == "same":
        target = same
    elif options.format == "apple":
        target = "alac" if probe.codec_class == "lossless" else same
    else:
        target = options.format
    if target is None:
        raise FileFailure("unsupported_target", _STAGE, "no same-format policy for this source")
    if probe.codec_class == "lossy":
        if target in LOSSLESS_FORMATS:
            return target, "lossy_origin"  # type: ignore[return-value]
        if target != probe.codec and not options.allow_lossy_transcode:
            return same, "kept_source_codec"  # type: ignore[return-value]
    return target, "as_requested"  # type: ignore[return-value]


def provisional_target(probe: ProbeResult, options: RunOptions) -> tuple[TargetFormatName, str]:
    """Choose a provisional target name and decision label without evidence or I/O.

    Apply the same source/format policy and refusals as _choose, including
    SkipFile for unsupported codecs or more than two channels. This is not
    capability validation: the final target still comes from full resolution
    with evidence (resolve_target and, after audio planning, finalise_target).
    """
    return _choose(probe, options)


def _representation(target: TargetFormatName, probe: ProbeResult,
                    options: RunOptions) -> tuple[int, bool] | None:
    if options.bit_depth != "same":
        requested = (32, True) if options.bit_depth == "float" else (int(options.bit_depth), False)
        if not any((r.bit_depth, r.sample_float) == requested for r in cap.TARGETS[target].representations):
            raise UsageError(f"--bit-depth {options.bit_depth} is not offered for {target}")
        return requested
    depth = probe.bit_depth
    if not probe.sample_float:
        depth = {8: 16, 20: 24, None: 24}.get(depth, depth)
    return cap.automatic_representation(target, source_bit_depth=depth,
                                        source_float=probe.sample_float,
                                        lossy_source=probe.codec_class == "lossy")


def _gap(detail: str, *, explicit: bool, diagnostics: list[Diagnostic]) -> NoReturn:
    if explicit:
        raise UsageError(detail)
    raise FileFailure("unsupported_target", _STAGE, detail, tuple(diagnostics))


def resolve_target(probe: ProbeResult, options: RunOptions,
                   records: Iterable[CapabilityEvidence], basis: cap.ToolBasis, *,
                   source_suffix: str | None = None) -> TargetFormat:
    """Resolve one provisional C-04 record (stream_copy=False).

    UsageError means an applicable explicit setting must stop the whole run;
    FileFailure means an automatic per-file target is unsupported. SkipFile
    protects unsupported codecs/channels. A skipped lossy source returns the
    C-04 skipped_lossy sentinel; it uses no evidence or encoding options.
    source_suffix is lexical input data, needed only for AIFF under same.
    """
    records = tuple(records)
    target, decision = _choose(probe, options)
    notes: list[str] = []
    diagnostics: list[Diagnostic] = []
    extension = cap.output_extension(target, same_source_suffix=(
        source_suffix if options.format == "same" else None))
    if decision == "skipped_lossy":
        return TargetFormat(target, extension, cap.TARGETS[target].encoders[0],
                            None, False, probe.sample_rate_hz, probe.channels, probe.channel_layout,
                            "none", "none", None, None, None, "not_applicable",
                            NOT_APPLICABLE_BASIS, False, "skipped_lossy")
    layout, assumed_layout = effective_channel_layout(probe.channels, probe.channel_layout)
    # D-081: evidence is matched at the effective layout. A record collected at the
    # input's own unstated layout is exact evidence for it too (no borrowing), so it
    # stays usable when no effective-layout record evidences the setting; the
    # assumption note is then dropped, because nothing was assumed.
    layout_note_at = len(notes)
    if decision == "kept_source_codec":
        message = f"lossy transcode to {options.format} blocked; keeping {probe.codec}"
        notes.append(message)
        diagnostics.append(_warn("lossy_transcode_blocked", message))
    elif decision == "lossy_origin":
        message = f"lossy source {probe.codec} stored losslessly as {target}; lost fidelity is not restored"
        notes.append(message)
        diagnostics.append(_warn("lossy_origin", message))
    if (options.format == "same" and probe.codec.startswith("pcm_")
            and probe.container in ("mp4", "mov")):
        message = f"PCM in {probe.container.upper()} resolved to native ALAC .m4a (D-064)"
        notes.append(message)
        diagnostics.append(_warn("format_adjusted", message))

    if probe.ignored_streams > 0:
        notes.append(_ignored_streams_note(probe.ignored_streams))

    lossless = target in LOSSLESS_FORMATS
    if not lossless and options.bit_depth != "same":
        if options.format in ("mp3", "aac"):
            raise UsageError(f"--bit-depth {options.bit_depth} does not apply to effective {target}")
        notes.append(f"--bit-depth {options.bit_depth}{_source_note(options, 'bit_depth')} ignored for {target}")
    if not lossless and options.dither != "auto":
        notes.append(f"--dither {options.dither}{_source_note(options, 'dither')} ignored for {target}")
    if lossless and options.bitrate != "auto":
        notes.append(f"--bitrate {options.bitrate}{_source_note(options, 'bitrate')} ignored for {target}")
    if (target != "flac" and ("flac_level" in options.explicit_fields
                             or options.value_sources.get("flac_level") == "config")):
        notes.append(f"--flac-level {options.flac_level}{_source_note(options, 'flac_level')} ignored for {target}")

    explicit_rate = options.sample_rate != "same"
    rate = options.sample_rate if explicit_rate else (
        probe.sample_rate_hz if lossless else cap.automatic_lossy_rate(probe.sample_rate_hz))
    spec = cap.TARGETS[target]
    if rate is None:
        _gap(f"no automatic rate-family policy for {probe.sample_rate_hz} Hz {target}",
             explicit=not lossless and options.bitrate != "auto", diagnostics=diagnostics)
    assert isinstance(rate, int)
    if explicit_rate and spec.max_explicit_rate_hz is not None and rate > spec.max_explicit_rate_hz:
        raise UsageError(f"--sample-rate {rate} exceeds {target}'s {spec.max_explicit_rate_hz} Hz policy cap")
    if not explicit_rate and rate != probe.sample_rate_hz:
        message = f"sample rate adjusted automatically from {probe.sample_rate_hz} to {rate} Hz"
        notes.append(message)
        diagnostics.append(_warn("format_adjusted", message))

    depth = None
    sample_float = False
    dither = "none"
    mode = "none"
    requested = effective = quality = None
    bitrate_source = "not_applicable"
    if lossless:
        representation = _representation(target, probe, options)
        explicit_depth = options.bit_depth != "same"
        if representation is None:
            _gap(f"no automatic representation policy for {probe.bit_depth}-bit {target}",
                 explicit=explicit_rate, diagnostics=diagnostics)
        assert representation is not None
        depth, sample_float = representation
        if (not explicit_depth and probe.codec_class == "lossless"
                and (depth, sample_float) != (probe.bit_depth, probe.sample_float)):
            old = "unknown depth" if probe.bit_depth is None else f"{probe.bit_depth}-bit"
            if probe.sample_float:
                old += " float"
            message = f"representation adjusted automatically from {old} to {depth}-bit{' float' if sample_float else ' integer'}"
            notes.append(message)
            diagnostics.append(_warn("format_adjusted", message))
        capability = cap.lossless_capability(target, records, basis, sample_rate_hz=rate,
                                             channels=probe.channels, channel_layout=layout)
        evidenced = capability.representation(depth, sample_float)
        if evidenced is None and assumed_layout:
            own = cap.lossless_capability(target, records, basis, sample_rate_hz=rate,
                                             channels=probe.channels, channel_layout=probe.channel_layout)
            evidenced = own.representation(depth, sample_float)
            if evidenced is not None:
                layout, assumed_layout = probe.channel_layout, False
        if evidenced is None:
            _gap(f"{target} {depth}-bit{' float' if sample_float else ''} at {rate} Hz, "
                 f"{probe.channels} ch / {layout}: no matching representation evidence; "
                 + "; ".join(capability.gaps), explicit=explicit_depth or explicit_rate,
                 diagnostics=diagnostics)
        assert evidenced is not None
        encoder = evidenced.representation.encoder
        evidence_id = evidenced.evidence_id
        if probe.codec_class == "lossy" and not explicit_depth:
            notes.append(f"lossy-source automatic representation: {depth}-bit integer (D-007)")
        if not sample_float:
            dither = ("triangular" if depth <= 16 else "none") if options.dither == "auto" else options.dither
        if options.dither != "auto":
            notes.append(f"--dither {options.dither}{_source_note(options, 'dither')} requested"
                         + ("; ignored for float output" if sample_float else ""))
    else:
        explicit_bitrate = options.bitrate != "auto"
        mode = "vbr" if (target == "mp3" and options.bitrate == "auto") or options.bitrate.startswith("V") else "cbr"
        encoder = spec.encoders[0]
        bitrate_source = "explicit" if explicit_bitrate else "auto"

        def lossy_setting(at_layout: str | None) -> tuple[cap.Capability, int | None, int | None, int | None,
                                                         str | None]:
            capability = cap.lossy_capability(target, mode, records, basis, sample_rate_hz=rate,
                                              channels=probe.channels, channel_layout=at_layout)
            if mode == "vbr":
                quality = int(options.bitrate[1:]) if explicit_bitrate else cap.MP3_AUTOMATIC_VBR_QUALITY
                return capability, None, None, quality, capability.evidence_for_setting(vbr_quality=quality)
            # F10: only a lossy source's bitrate is meaningful; a PCM/FLAC/ALAC stream rate is not (B-006).
            source_bitrate = (probe.bitrate_bps or 0) if probe.codec_class == "lossy" else 0
            requested = int(options.bitrate[:-1]) * 1000 if explicit_bitrate else max(source_bitrate, 256000)
            choices = capability.nominal_bitrates_bps
            effective = requested if explicit_bitrate else (
                next((value for value in choices if value >= requested), choices[-1]) if choices else None)
            evidence_id = capability.evidence_for_setting(bitrate_bps=effective) if effective is not None else None
            return capability, requested, effective, None, evidence_id

        capability, requested, effective, quality, evidence_id = lossy_setting(layout)
        if evidence_id is None and assumed_layout:
            own = lossy_setting(probe.channel_layout)
            if own[4] is not None:
                capability, requested, effective, quality, evidence_id = own
                layout, assumed_layout = probe.channel_layout, False
        if evidence_id is None:
            _gap(f"{target} --bitrate {options.bitrate} at {rate} Hz, {probe.channels} ch / "
                 f"{layout}: setting not evidenced; " + "; ".join(capability.gaps),
                 explicit=explicit_bitrate or explicit_rate, diagnostics=diagnostics)
        if target == "aac" and not explicit_bitrate and requested != effective:
            message = f"AAC automatic nominal bitrate adjusted from {requested} to {effective} bps"
            notes.append(message)
            diagnostics.append(_warn("aac_bitrate_adjusted", message))
        if explicit_bitrate and ((target == "aac" and requested is not None and requested < 256000)
                                 or (target == "mp3" and quality is not None and quality > 0)):
            notes.append(f"--bitrate {options.bitrate}{_source_note(options, 'bitrate')} overrides automatic "
                         + ("AAC floor" if target == "aac" else "MP3 V0 quality"))
    if assumed_layout:
        notes.insert(layout_note_at, f"channel layout not stated; assumed {layout}")
    return TargetFormat(target, extension, encoder, depth, sample_float, rate, probe.channels,
                        layout, dither, mode, requested, effective, quality,
                        bitrate_source, evidence_id, False, decision, tuple(notes), tuple(diagnostics))


def preflight_targets(probes: Iterable[ProbeResult | None], options: RunOptions,
                      records: Iterable[CapabilityEvidence], basis: cap.ToolBasis, *,
                      source_suffixes: Iterable[str | None] | None = None,
                      input_paths: Iterable[Path] | None = None,
                      available_encoders: frozenset[str] | None = None,
                      ) -> tuple[tuple[TargetFormat | FileFailure | SkipFile | None, ...], tuple[str, ...]]:
    """Validate every resolvable input before writing; aggregate all usage refusals.

    Entries preserve input order. None denotes an input already terminal before
    encoding (for example already_processed or probe failure); it neither uses
    nor invalidates options. Automatic failures remain per-file exceptions for
    the caller to project. A skipped_lossy record similarly requires no render.
    Checked missing encoders raise EnvironmentError_ (exit 3), never a fabricated
    evidence record. Missing tools/unusable executable checks belong to the shell.
    Supply discovered C-01 source paths as input_paths for file-named refusals:
    C-10 inputs may be folder arguments and must not be guessed to match probes.
    Without those paths, refusals identify the supplied probe's ordinal.
    """
    probes, records = tuple(probes), tuple(records)
    suffixes = tuple(source_suffixes) if source_suffixes is not None else (None,) * len(probes)
    if len(suffixes) != len(probes):
        raise ValueError("source_suffixes must have one entry per probe")
    paths = tuple(input_paths) if input_paths is not None else None
    if paths is not None and len(paths) != len(probes):
        raise ValueError("input_paths must have one entry per probe")
    if available_encoders is not None:
        required: set[str] = set()
        for probe in probes:
            if probe is None:
                continue
            try:
                target, decision = _choose(probe, options)
            except (SkipFile, FileFailure):
                continue
            if decision == "skipped_lossy":
                continue
            spec = cap.TARGETS[target]
            if target in LOSSLESS_FORMATS:
                try:
                    representation = _representation(target, probe, options)
                except UsageError:
                    # Aggregate usage errors below instead of stopping at this input.
                    continue
                if representation is not None:
                    required.update(r.encoder for r in spec.representations
                                    if (r.bit_depth, r.sample_float) == representation)
            else:
                required.update(spec.encoders)
        missing = required - available_encoders
        if missing:
            raise EnvironmentError_("required encoders missing or unusable: " + ", ".join(sorted(missing)))
    results: list[TargetFormat | FileFailure | SkipFile | None] = []
    usage: list[str] = []
    effective_lossy = False
    for index, (probe, suffix) in enumerate(zip(probes, suffixes)):
        if probe is None:
            results.append(None)
            continue
        try:
            target, decision = _choose(probe, options)
            effective_lossy |= target not in LOSSLESS_FORMATS and decision != "skipped_lossy"
            results.append(resolve_target(probe, options, records, basis, source_suffix=suffix))
        except UsageError as error:
            name = str(paths[index]) if paths is not None else f"input[{index}]"
            usage.append(f"{name}: {error}")
        except (FileFailure, SkipFile) as error:
            results.append(error)
    if usage:
        raise UsageError("invalid explicit target settings before any write:\n" + "\n".join(usage))
    notes = ()
    if options.bitrate != "auto" and not effective_lossy:
        config_note = " (from the config)" if options.value_sources.get("bitrate") == "config" else ""
        notes = (f"--bitrate {options.bitrate}{config_note} ignored — no file in this run uses a lossy-output bitrate setting",)
    return tuple(results), notes


def finalise_target(probe: ProbeResult, options: RunOptions,
                    records: Iterable[CapabilityEvidence], basis: cap.ToolBasis, *,
                    action: Literal["copy", "linear", "limit", "reduce", "dynamic", "skip"],
                    source_suffix: str | None = None) -> TargetFormat:
    """Re-resolve from original inputs for each audio action, including retries.

    copy is eligible only with unchanged codec, rate, depth/float and channels;
    container and bitrate requests do not defeat eligibility. The caller owns
    actual render completion and its lossy_reencoded diagnostic, not this plan.
    skip is a no-output audio plan and creates no copy/ignored-bitrate note.
    """
    if action not in ("copy", "linear", "limit", "reduce", "dynamic", "skip"):
        raise ValueError(f"unknown audio action {action!r}")
    target = resolve_target(probe, options, records, basis, source_suffix=source_suffix)
    if target.decision == "skipped_lossy":
        return target
    if action == "skip" and probe.ignored_streams > 0:
        # No output is written, so nothing is "not carried over".
        ignored = _ignored_streams_note(probe.ignored_streams)
        target = replace(target, notes=tuple(n for n in target.notes if n != ignored))
    codec_unchanged = probe.codec == target.format if target.format not in ("wav", "aiff") else probe.codec == target.encoder
    representation_unchanged = target.format not in LOSSLESS_FORMATS or (
        target.bit_depth == probe.bit_depth and target.sample_float == probe.sample_float)
    # The target carries the effective layout (D-081) or, with stated-layout
    # evidence, the source's own; neither alone changes the stream.
    source_layouts = (probe.channel_layout, effective_channel_layout(probe.channels, probe.channel_layout)[0])
    if not (action == "copy" and codec_unchanged and representation_unchanged
            and target.sample_rate_hz == probe.sample_rate_hz
            and target.channels == probe.channels and target.channel_layout in source_layouts):
        return target
    # An unchanged copy applies neither dither nor automatic/explicit bitrate.
    diagnostics = tuple(d for d in target.diagnostics if d.code != "aac_bitrate_adjusted")
    notes = tuple(note for note in target.notes if not (
        note.startswith("AAC automatic nominal bitrate adjusted")
        or "overrides automatic" in note or note.startswith("--dither ")))
    if options.dither != "auto":
        notes += (f"--dither {options.dither}{_source_note(options, 'dither')} ignored — "
                  "an unchanged stream copy performs no quantisation",)
    if options.bitrate != "auto" and target.format not in LOSSLESS_FORMATS:
        notes += (f"--bitrate {options.bitrate}{_source_note(options, 'bitrate')} not applied — "
                  "the encoded audio was copied unchanged, so no encoder setting was used",)
    return replace(target, stream_copy=True, encoder=STREAM_COPY_ENCODER,
                   capability_basis=NOT_APPLICABLE_BASIS, dither="none", bitrate_mode="none",
                   requested_bitrate_bps=None, effective_bitrate_bps=None, vbr_quality=None,
                   bitrate_source="not_applicable", notes=notes, diagnostics=diagnostics)
