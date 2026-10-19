"""Static format capabilities intersected with C-04 capability evidence (WS-02).

Pure domain module (ARCHITECTURE § 3): it imports only ``norma.models`` and the
standard library, runs no process, reads no file and queries nothing. The
tables below declare what Norma's *policy* offers for each target
(the format tables). They are not encoder capabilities: a
setting is claimed only when the static table offers it **and** a matching
run-local ``CapabilityEvidence`` record (C-04, produced at the WS-04 tool edge)
evidences it. Anything else is an explicit conservative gap whose reason is
``unsupported_target`` (formats F7, F9, F12; C-04). No AAC, MP3 or rate table
is invented here.

Target resolution (formats F2, F8, F10–F16, ``formats/resolve.py``) consumes
the data and evidence intersection supplied here. AAC and MP3 support comes
only from matching run-local records, including the S3 lossy collectors;
there is no unconditional AAC gap or build-specific table in this module.
Whether a gap becomes exit 2 (explicit setting) or a per-file
``unsupported_target`` (automatic choice) is the resolver's decision (F7).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from norma.models import (
    BitrateMode,
    CapabilityEvidence,
    FrozenMap,
    Identity,
    ReasonCode,
    TargetFormatName,
)

# --- § 2 Inputs ---------------------------------------------------------------------------

INPUT_CONTAINERS: FrozenMap = FrozenMap({
    # recognised final suffix (formats F1) -> container the probe is expected to decide
    ".wav": "wav",
    ".aif": "aiff",
    ".aiff": "aiff",
    ".flac": "flac",
    ".m4a": "mp4",
    ".mp4": "mp4",
    ".caf": "caf",
    ".mp3": "mp3",
    ".aac": "adts",
})
"""Recognised input suffixes (formats § 2 table). A suffix is never proof of audio:
WS-05's signature check and the WS-04 probe decide; this is only folder-expansion data."""

RECOGNISED_EXTENSIONS: frozenset[str] = frozenset(INPUT_CONTAINERS)


def is_recognised_extension(name: str | Path) -> bool:
    """True when the final suffix is recognised, compared case-insensitively (F1)."""
    return Path(name).suffix.casefold() in RECOGNISED_EXTENSIONS


# --- § 3 Output format choices ----------------------------------------------------------------

FORMAT_CHOICES: tuple[str, ...] = ("same", "wav", "flac", "alac", "aiff", "mp3", "aac", "apple")
FORMAT_ALIASES: FrozenMap = FrozenMap({"aif": "aiff"})
REFUSED_FORMAT_CHOICES: FrozenMap = FrozenMap({"m4a": "ambiguous: .m4a holds both ALAC and AAC"})
BUILTIN_FORMAT_CHOICE = "same"
"""Built-in default (D-007); ``apple`` is a user config choice, never the built-in default."""


def canonical_format_choice(token: str) -> str | None:
    """Return the canonical ``-f`` choice, or None for a refused or unknown token (F2).

    The usage error (exit 2) and its wording belong to WS-03; this is only the data.
    """
    if token in REFUSED_FORMAT_CHOICES:
        return None
    token = FORMAT_ALIASES.get(token, token)
    return token if token in FORMAT_CHOICES else None


OUTPUT_EXTENSIONS: FrozenMap = FrozenMap({
    "wav": ".wav", "flac": ".flac", "alac": ".m4a", "aac": ".m4a", "mp3": ".mp3", "aiff": ".aif",
})
AIFF_EXTENSIONS: tuple[str, ...] = (".aif", ".aiff")


def output_extension(target: TargetFormatName, *, same_source_suffix: str | None = None) -> str:
    """Output extension (F3). Under ``same``, AIFF keeps the source's ``.aif``/``.aiff`` spelling."""
    extension = OUTPUT_EXTENSIONS[target]
    if target == "aiff" and same_source_suffix is not None:
        spelling = same_source_suffix.casefold()
        if spelling in AIFF_EXTENSIONS:
            return spelling
    return extension


_PCM_TARGET_BY_CONTAINER: FrozenMap = FrozenMap({"wav": "wav", "aiff": "aiff", "caf": "wav"})
_CODEC_TARGET: FrozenMap = FrozenMap({"flac": "flac", "alac": "alac", "mp3": "mp3", "aac": "aac"})


def same_target_format(codec: str, codec_class: str, container: str) -> TargetFormatName | None:
    """Target format for ``-f same``: the source codec in its default container (F2, F3).

    Takes C-02's ``codec``, ``codec_class`` and canonical ``container`` (the WS-04
    probe classifies codecs; this module does not repeat it). Linear PCM stays WAV
    or AIFF and PCM in CAF goes to WAV; ALAC and AAC (including raw ADTS and CAF)
    go to ``.m4a``. Returns None for an unsupported codec, or for PCM in a
    container with no stated default, rather than guess.
    """
    if codec_class == "unsupported":
        return None
    if codec in _CODEC_TARGET:
        return _CODEC_TARGET[codec]  # type: ignore[no-any-return]
    if codec_class == "lossless" and codec.startswith("pcm_"):
        return _PCM_TARGET_BY_CONTAINER.get(container)  # type: ignore[no-any-return]
    return None


# --- § 4 Depth, rate and encoder policy --------------------------------------------------------


@dataclass(frozen=True)
class Representation:
    """One output sample representation; float is 32-bit IEEE float (pcm_f32le)."""

    bit_depth: int
    sample_float: bool
    encoder: str


@dataclass(frozen=True)
class TargetCapability:
    """What Norma's policy offers for one target (formats § 4 table). Not encoder support."""

    format: TargetFormatName
    family: Literal["lossless", "lossy"]
    extension: str
    representations: tuple[Representation, ...]
    """Offered lossless representations, keyed to the encoder that writes them."""
    bitrate_modes: tuple[BitrateMode, ...]
    vbr_qualities: tuple[int, ...]
    max_rate_hz: int | None
    """Policy cap on the output rate: 48 kHz for lossy targets (F4/F5); None otherwise."""
    max_explicit_rate_hz: int | None
    """Highest offered explicit ``--sample-rate`` (FLAC/ALAC up to 192 kHz, F4); still evidence-bound."""

    @property
    def encoders(self) -> tuple[str, ...]:
        seen: list[str] = []
        for item in self.representations:
            if item.encoder not in seen:
                seen.append(item.encoder)
        return tuple(seen) or (_LOSSY_ENCODERS[self.format],)

    @property
    def offered_bit_depths(self) -> tuple[int, ...]:
        return tuple(sorted({r.bit_depth for r in self.representations if not r.sample_float}))

    @property
    def offers_float(self) -> bool:
        return any(r.sample_float for r in self.representations)


_LOSSY_ENCODERS: FrozenMap = FrozenMap({"mp3": "libmp3lame", "aac": "aac_at"})
LOSSY_MAX_RATE_HZ = 48_000
LOSSLESS_MAX_EXPLICIT_RATE_HZ = 192_000
MP3_VBR_QUALITIES: tuple[int, ...] = tuple(range(10))
MP3_AUTOMATIC_VBR_QUALITY = 0
"""Automatic MP3 is LAME V0 when evidenced, otherwise ``unsupported_target`` (F15, D-029)."""
NEVER_ENCODERS: frozenset[str] = frozenset({"alac_at"})
"""``alac_at`` wrote 16-bit from 24-bit input with exit 0 (formats § 9 C3); never a Norma encoder."""


def _rep(depth: int, encoder: str, *, sample_float: bool = False) -> Representation:
    return Representation(depth, sample_float, encoder)


TARGETS: FrozenMap = FrozenMap({
    "wav": TargetCapability(
        "wav", "lossless", ".wav",
        (_rep(16, "pcm_s16le"), _rep(24, "pcm_s24le"), _rep(32, "pcm_s32le"),
         _rep(32, "pcm_f32le", sample_float=True)),
        ("none",), (), None, None),
    "aiff": TargetCapability(
        "aiff", "lossless", ".aif",
        (_rep(16, "pcm_s16be"), _rep(24, "pcm_s24be"), _rep(32, "pcm_s32be")),
        ("none",), (), None, None),
    "flac": TargetCapability(
        "flac", "lossless", ".flac", (_rep(16, "flac"), _rep(24, "flac")),
        ("none",), (), None, LOSSLESS_MAX_EXPLICIT_RATE_HZ),
    "alac": TargetCapability(
        "alac", "lossless", ".m4a", (_rep(16, "alac"), _rep(24, "alac")),
        ("none",), (), None, LOSSLESS_MAX_EXPLICIT_RATE_HZ),
    "mp3": TargetCapability(
        "mp3", "lossy", ".mp3", (), ("cbr", "vbr"), MP3_VBR_QUALITIES, LOSSY_MAX_RATE_HZ, LOSSY_MAX_RATE_HZ),
    "aac": TargetCapability(
        "aac", "lossy", ".m4a", (), ("cbr",), (), LOSSY_MAX_RATE_HZ, LOSSY_MAX_RATE_HZ),
})
"""Every C-04 target format. AAC is AAC-LC CBR through ``aac_at`` only; MP3 is ``libmp3lame``."""

LOSSY_SOURCE_LOSSLESS_DEPTH = 16
"""A lossy source stored losslessly defaults to 16-bit with triangular dither (F6, D-007)."""

HIGH_RATE_FAMILY: FrozenMap = FrozenMap({88_200: 44_100, 176_400: 44_100, 96_000: 48_000, 192_000: 48_000})
"""Automatic high-rate lossy conversion keeps the rate family (F5, ADR-0006)."""


def automatic_lossy_rate(source_rate_hz: int) -> int | None:
    """Automatic MP3/AAC output rate (F4/F5): preserve ≤48 kHz, map the four high rates.

    Any other rate above 48 kHz returns None: it needs evidenced rate-family
    resolution or ``unsupported_target``, never an invented rate. Never upsamples.
    The returned rate is still subject to capability evidence.
    """
    if source_rate_hz <= 0:
        raise ValueError("sample rate must be positive")
    if source_rate_hz <= LOSSY_MAX_RATE_HZ:
        return source_rate_hz
    return HIGH_RATE_FAMILY.get(source_rate_hz)  # type: ignore[no-any-return]


def automatic_representation(target: TargetFormatName, *, source_bit_depth: int | None, source_float: bool,
                             lossy_source: bool) -> tuple[int, bool] | None:
    """Automatic ``(bit_depth, sample_float)`` for a lossless target (F4, F6, ADR-0006).

    - Lossy source: 16-bit integer (F6; dither is decided elsewhere).
    - WAV: preserve a supported representation (16/24/32 integer, 32-bit float).
    - AIFF, FLAC, ALAC: preserve 16/24; float or 32-bit become 24-bit.

    Returns None when no rule applies — an unknown source depth, or a depth the
    target does not offer (for example 8 or 20 bits): the static policy states no
    conversion for it, so none is invented here. Lossy targets have no PCM depth;
    asking for one is a programming error.
    """
    spec: TargetCapability = TARGETS[target]
    if spec.family == "lossy":
        raise ValueError(f"{target} has no PCM output depth")
    if lossy_source:
        return (LOSSY_SOURCE_LOSSLESS_DEPTH, False)
    if source_float:
        return (32, True) if spec.offers_float else (24, False)
    if source_bit_depth is None:
        return None
    if target == "wav":
        return (source_bit_depth, False) if source_bit_depth in spec.offered_bit_depths else None
    if source_bit_depth in (16, 24):
        return (source_bit_depth, False)
    if source_bit_depth == 32:
        return (24, False)
    return None


# --- C-04 evidence intersection -----------------------------------------------------------------


@dataclass(frozen=True)
class ToolBasis:
    """The run's revalidated FFmpeg identity, against which evidence must match (F9)."""

    executable_path: Path
    executable_identity: Identity
    ffmpeg_version: str
    platform_basis: str

    def matches(self, record: CapabilityEvidence) -> bool:
        return (record.executable_path == self.executable_path
                and tuple(record.executable_identity) == tuple(self.executable_identity)
                and record.ffmpeg_version == self.ffmpeg_version
                and record.platform_basis == self.platform_basis)


@dataclass(frozen=True)
class EvidencedRepresentation:
    representation: Representation
    evidence_id: str


GAP_REASON: ReasonCode = "unsupported_target"


@dataclass(frozen=True)
class Capability:
    """Static policy ∩ evidence for one target, mode, rate and layout.

    ``gaps`` names every offered item that was not claimed and why; a capability
    with no claimed choice is unsupported and carries ``reason=unsupported_target``.
    """

    target: TargetFormatName
    bitrate_mode: BitrateMode
    sample_rate_hz: int
    channels: int
    channel_layout: str | None
    representations: tuple[EvidencedRepresentation, ...] = ()
    nominal_bitrates_bps: tuple[int, ...] = ()
    vbr_qualities: tuple[int, ...] = ()
    evidence_id: str | None = None
    """The single matched record for a lossy target; lossless ids live per representation."""
    gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("representations", "nominal_bitrates_bps", "vbr_qualities", "gaps"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def supported(self) -> bool:
        if self.bitrate_mode == "none":
            return bool(self.representations)
        if self.evidence_id is None:
            return False
        return bool(self.nominal_bitrates_bps if self.bitrate_mode == "cbr" else self.vbr_qualities)

    @property
    def reason(self) -> ReasonCode | None:
        return None if self.supported else GAP_REASON

    def representation(self, bit_depth: int, sample_float: bool = False) -> EvidencedRepresentation | None:
        for item in self.representations:
            if item.representation.bit_depth == bit_depth and item.representation.sample_float == sample_float:
                return item
        return None

    def evidence_for_setting(self, *, bitrate_bps: int | None = None, vbr_quality: int | None = None) -> str | None:
        """Evidence ID for an exact nominal bitrate or VBR quality; never rounds or clamps (F11)."""
        if not self.supported or (bitrate_bps is None) == (vbr_quality is None):
            return None
        if bitrate_bps is not None:
            return self.evidence_id if bitrate_bps in self.nominal_bitrates_bps else None
        return self.evidence_id if vbr_quality in self.vbr_qualities else None


def _matching(records: Iterable[CapabilityEvidence], basis: ToolBasis, encoder: str, mode: BitrateMode,
              sample_rate_hz: int, channels: int, channel_layout: str | None) -> list[CapabilityEvidence]:
    """Match every F9 dimension exactly, using the caller's effective layout.

    D-081's assumption belongs to resolve.effective_channel_layout: the shell
    collects evidence at that layout and the resolver queries it (then, failing
    that, the input's own absent layout, which is exact). Unknown layouts for
    other channel counts still never borrow a known layout here.
    """
    return [record for record in records
            if basis.matches(record)
            and record.encoder == encoder
            and record.bitrate_mode == mode
            and record.sample_rate_hz == sample_rate_hz
            and record.channels == channels
            and record.channel_layout == channel_layout]


def _check_query(target: str, sample_rate_hz: int, channels: int) -> TargetCapability:
    if target not in TARGETS:
        raise ValueError(f"unknown target format {target!r}")
    if not (isinstance(sample_rate_hz, int) and sample_rate_hz > 0 and isinstance(channels, int) and channels > 0):
        raise ValueError("sample rate and channels must be positive integers")
    return TARGETS[target]  # type: ignore[no-any-return]


def lossless_capability(target: TargetFormatName, records: Iterable[CapabilityEvidence], basis: ToolBasis, *,
                        sample_rate_hz: int, channels: int, channel_layout: str | None) -> Capability:
    """Offered representations of a lossless target that matching evidence confirms (F4, F9).

    WAV and AIFF have one encoder per representation (``pcm_s24le`` and so on); FLAC
    and ALAC confirm each depth from one record. Float is claimed only where the
    policy offers it (WAV). Two matching records for one encoder are ambiguous and
    claim nothing for that encoder.
    """
    spec = _check_query(target, sample_rate_hz, channels)
    if spec.family != "lossless":
        raise ValueError(f"{target} is not a lossless target")
    records = tuple(records)
    claimed: list[EvidencedRepresentation] = []
    gaps: list[str] = []
    for encoder in spec.encoders:
        offered = [r for r in spec.representations if r.encoder == encoder]
        found = _matching(records, basis, encoder, "none", sample_rate_hz, channels, channel_layout)
        label = ", ".join(_label(r) for r in offered)
        if not found:
            gaps.append(f"{encoder}: no matching evidence record for {label}")
            continue
        if len(found) > 1:
            ids = ", ".join(sorted(record.evidence_id for record in found))
            gaps.append(f"{encoder}: ambiguous evidence ({ids}); nothing claimed")
            continue
        record = found[0]
        for item in offered:
            evidenced = (record.supports_float if item.sample_float
                         else item.bit_depth in record.supported_bit_depths)
            if evidenced:
                claimed.append(EvidencedRepresentation(item, record.evidence_id))
            else:
                gaps.append(f"{encoder}: {_label(item)} not evidenced by {record.evidence_id}")
    return Capability(target, "none", sample_rate_hz, channels, channel_layout,
                      representations=tuple(claimed), gaps=tuple(gaps))


def _label(item: Representation) -> str:
    return f"{item.bit_depth}-bit{' float' if item.sample_float else ''}"


def lossy_capability(target: TargetFormatName, bitrate_mode: BitrateMode, records: Iterable[CapabilityEvidence],
                     basis: ToolBasis, *, sample_rate_hz: int, channels: int,
                     channel_layout: str | None) -> Capability:
    """Nominal bitrates or VBR qualities of an MP3/AAC target that one matching record evidences.

    Conservative gaps (C-04, F7, F9, F12): a mode the policy does not offer (VBR
    on AAC), a rate above the 48 kHz lossy cap, no matching record, more than one,
    ``nominal_confirmation=not_applicable`` or an empty choice list. Choices are
    the record's discrete values, intersected with the static VBR range; nothing
    is interpolated or added.
    """
    spec = _check_query(target, sample_rate_hz, channels)
    if spec.family != "lossy":
        raise ValueError(f"{target} is not a lossy target")
    gap = _Gap(target, bitrate_mode, sample_rate_hz, channels, channel_layout)
    if bitrate_mode not in spec.bitrate_modes:
        return gap(f"{target} offers no {bitrate_mode} setting")
    assert spec.max_rate_hz is not None
    if sample_rate_hz > spec.max_rate_hz:
        return gap(f"{sample_rate_hz} Hz is above the {spec.max_rate_hz} Hz lossy output cap (F4/F5)")
    encoder = spec.encoders[0]
    found = _matching(tuple(records), basis, encoder, bitrate_mode, sample_rate_hz, channels, channel_layout)
    where = f"{encoder} {bitrate_mode} at {sample_rate_hz} Hz, {channels} ch, layout {channel_layout}"
    if not found:
        return gap(f"no matching evidence record for {where}")
    if len(found) > 1:
        return gap(f"ambiguous evidence for {where}: " + ", ".join(sorted(r.evidence_id for r in found)))
    record = found[0]
    if record.nominal_confirmation == "not_applicable":
        return gap(f"{record.evidence_id}: nominal setting cannot be confirmed (nominal_confirmation="
                   "not_applicable, F12)")
    if bitrate_mode == "cbr":
        bitrates = tuple(sorted(record.nominal_bitrates_bps))
        if not bitrates:
            return gap(f"{record.evidence_id}: empty nominal bitrate list")
        return Capability(target, "cbr", sample_rate_hz, channels, channel_layout,
                          nominal_bitrates_bps=bitrates, evidence_id=record.evidence_id)
    qualities = tuple(q for q in spec.vbr_qualities if q in record.vbr_qualities)
    if not qualities:
        return gap(f"{record.evidence_id}: no evidenced VBR quality")
    missing = tuple(q for q in spec.vbr_qualities if q not in qualities)
    gaps = (f"{record.evidence_id}: V{', V'.join(map(str, missing))} not evidenced",) if missing else ()
    return Capability(target, "vbr", sample_rate_hz, channels, channel_layout,
                      vbr_qualities=qualities, evidence_id=record.evidence_id, gaps=gaps)


def automatic_mp3_capability(records: Iterable[CapabilityEvidence], basis: ToolBasis, *, sample_rate_hz: int,
                             channels: int, channel_layout: str | None) -> Capability:
    """Automatic MP3: evidenced LAME V0 at this rate and layout, or a gap (F15, D-029).

    There is no automatic CBR alternative: a CBR record never rescues a missing V0.
    """
    vbr = lossy_capability("mp3", "vbr", records, basis, sample_rate_hz=sample_rate_hz, channels=channels,
                           channel_layout=channel_layout)
    if vbr.evidence_for_setting(vbr_quality=MP3_AUTOMATIC_VBR_QUALITY) is not None:
        return Capability("mp3", "vbr", sample_rate_hz, channels, channel_layout,
                          vbr_qualities=(MP3_AUTOMATIC_VBR_QUALITY,), evidence_id=vbr.evidence_id)
    return Capability("mp3", "vbr", sample_rate_hz, channels, channel_layout,
                      gaps=vbr.gaps + ("automatic MP3 needs evidenced V0; no automatic CBR alternative (F15)",))


@dataclass(frozen=True)
class _Gap:
    target: TargetFormatName
    bitrate_mode: BitrateMode
    sample_rate_hz: int
    channels: int
    channel_layout: str | None

    def __call__(self, why: str) -> Capability:
        return Capability(self.target, self.bitrate_mode, self.sample_rate_hz, self.channels,
                          self.channel_layout, gaps=(why,))


# --- § 8 Tag capability matrix -----------------------------------------------------------------

TagStore = Literal["mp4", "flac", "id3v23", "riff_info"]
TAG_STORES: tuple[TagStore, ...] = ("mp4", "flac", "id3v23", "riff_info")
TAG_FIELDS: tuple[str, ...] = (
    "title", "artist", "album_artist", "album", "date", "track", "track_total", "disc", "disc_total",
    "genre", "composer", "comment", "artwork", "compilation", "extras",
)
"""The 14 canonical fields (metadata-schema § 2.1) plus the extras row of formats § 8."""

_MATRIX_ROWS: dict[str, tuple[str | None, str | None, str | None, str | None]] = {
    # field: (MP4, FLAC, ID3v2.3, RIFF INFO); None = no supported mapping (formats M2)
    "title": ("©nam", "TITLE", "TIT2", "INAM"),
    "artist": ("©ART", "ARTIST", "TPE1", "IART"),
    "album_artist": ("aART", "ALBUMARTIST", "TPE2", None),
    "album": ("©alb", "ALBUM", "TALB", "IPRD"),
    "date": ("©day", "DATE", "TYER+TDAT", "ICRD"),
    "track": ("trkn[0]", "TRACKNUMBER", "TRCK", "IPRT"),
    "track_total": ("trkn[1]", "TRACKTOTAL", "TRCK", None),
    "disc": ("disk[0]", "DISCNUMBER", "TPOS", None),
    "disc_total": ("disk[1]", "DISCTOTAL", "TPOS", None),
    "genre": ("©gen", "GENRE", "TCON", "IGNR"),
    "composer": ("©wrt", "COMPOSER", "TCOM", None),
    "comment": ("©cmt", "COMMENT", "COMM:eng", "ICMT"),
    "artwork": ("covr", "PICTURE:3", "APIC:3", None),
    "compilation": ("cpil", "COMPILATION", "TCMP", None),
    "extras": ("----:com.apple.iTunes:{key}", "{key}", "TXXX:{key}", None),
}

TAG_MATRIX: FrozenMap = FrozenMap({
    store: FrozenMap({name: row[index] for name, row in _MATRIX_ROWS.items()})
    for index, store in enumerate(TAG_STORES)
})
"""Native storage per store and canonical field (formats § 8, M2). Storage, not Music display;
field fates are WS-06's (metadata § 7). Mutagen writes every store except RIFF INFO,
which FFmpeg writes (ADR-0004)."""

TOTAL_WITHOUT_NUMBER: FrozenMap = FrozenMap({
    "mp4": "placeholder_zero",   # (0, N): stored, read back as absent (M3, MD38)
    "flac": "separate_field",    # TRACKTOTAL / DISCTOTAL comments, a convention (M3)
    "id3v23": "not_stored",      # no valid ID3v2.3 form; WS-06 dropped_format_limit (MD38)
    "riff_info": "not_stored",   # no total mapping at all
})
"""How each store holds a track or disc total that has no number (formats M3)."""

TARGET_TAG_STORES: FrozenMap = FrozenMap({
    "alac": ("mp4",), "aac": ("mp4",), "flac": ("flac",), "mp3": ("id3v23",), "aiff": ("id3v23",),
})
WAV_TAG_MODES: FrozenMap = FrozenMap({
    "info+id3": ("riff_info", "id3v23"),
    "info": ("riff_info",),
    "none": (),
})
DEFAULT_WAV_TAG_MODE = "info+id3"


def tag_stores(target: TargetFormatName, wav_tags: str = DEFAULT_WAV_TAG_MODE) -> tuple[TagStore, ...]:
    """Tag stores written for a target (formats § 8, M4). ``wav_tags`` applies to WAV only."""
    if target == "wav":
        if wav_tags not in WAV_TAG_MODES:
            raise ValueError(f"unknown WAV tag mode {wav_tags!r}")
        return WAV_TAG_MODES[wav_tags]  # type: ignore[no-any-return]
    if target not in TARGET_TAG_STORES:
        raise ValueError(f"unknown target format {target!r}")
    return TARGET_TAG_STORES[target]  # type: ignore[no-any-return]


def storable_fields(target: TargetFormatName, wav_tags: str = DEFAULT_WAV_TAG_MODE) -> frozenset[str]:
    """Fields with a native mapping in at least one store written for the target."""
    return frozenset(name for store in tag_stores(target, wav_tags) for name, key in TAG_MATRIX[store].items()
                     if key is not None)
