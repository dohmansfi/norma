"""Pure output naming, comparison keys and collision groups (C-08; safety spec § 3).

Everything here is a decision over values: no filesystem access, no clock, no
randomness. ``safety`` performs the checks that need the filesystem (existing
entries, identities, ancestry) and canonicalises output folders before they
reach this module, so the paths given here are already canonical.

Modes (see the output-path modes):

* default (neither ``-o`` nor ``-s``, recursive or not):
  ``<canonical source parent>/Normalised/<stem><ext>``;
* ``-o DIR`` (flat for explicit files and top-level entries, mirrored for
  recursive entries): ``DIR/<relative parent from retained root>/<stem><ext>``;
* ``-s SUFFIX``: ``<canonical source parent>/<stem><expanded suffix><ext>``;
* ``--rename PATTERN``: the stem comes from resolved metadata within the
  chosen location; a suffix, if selected, follows it.

Sidecar members append their MD25 suffix to the complete audio filename.
Comparison keys are NFC/case-folded absolute names: a deliberate, disclosed
over-refusal on case-sensitive volumes (ADR-0016).
"""

from __future__ import annotations

import string
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from norma.errors import FileFailure, UsageError
from norma.models import (
    BundleMember,
    InputItem,
    OutputPlan,
    SidecarRequest,
    TargetFormat,
)

PATHS_STAGE = "path_planned"

DEFAULT_FOLDER = "Normalised"
STEM_MAX_BYTES = 200
"""Generated stems (rename/suffix) are cut to this many UTF-8 bytes, suffix included (§ 3)."""
NAME_MAX_BYTES = 255
"""The complete component limit checked here; APFS and HFS+ both allow 255 UTF-8 bytes per name."""

RENAME_FIELDS: frozenset[str] = frozenset(
    {"title", "artist", "album_artist", "album", "track", "disc", "date", "year", "genre"}
)
SUFFIX_FIELDS: frozenset[str] = frozenset({"target"})

AUDIO_MEMBER_ID = "audio"
_PURPOSE_FOR_ID_PREFIX = {"tags": "tags_sidecar", "cover": "cover_export", "picture-": "picture_export"}


# --- Comparison keys (§ 3 "Comparison — policy and identity") ----------------------------


def comparison_key(path: Path | str) -> str:
    """The conservative clash key: NFC, case-folded, NFC again (case folding can denormalise).

    ``Song.wav`` and ``song.WAV`` share a key, so do composed and decomposed ``Perché``, and
    ``Straße`` folds to ``strasse``; ``Perché`` and ``Perche`` never do, because accents are kept.
    The key is policy for comparison only; it is never used as a pathname.
    """
    text = str(path)
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", text).casefold())


# --- Pattern handling (§ 3 "Rename/suffix"; cli § 3.2) -------------------------------------


def _is_control(char: str) -> bool:
    return ord(char) < 32 or ord(char) == 127 or unicodedata.category(char) == "Cc"


@dataclass(frozen=True)
class Pattern:
    """A parsed ``--rename`` or ``-s`` pattern: literal text and placeholder pieces in order."""

    source: str
    pieces: tuple[tuple[str, str | None, str], ...]  # literal, placeholder name or None, format spec


def parse_pattern(pattern: str, *, kind: str) -> Pattern:
    """Parse a rename (``kind='rename'``) or suffix (``kind='suffix'``) pattern.

    Unsupported syntax, an unknown placeholder, ``/``, ``:`` or a control character in the
    pattern itself, and a rename pattern without a placeholder are usage errors, exit 2.
    """
    if kind not in ("rename", "suffix"):
        raise ValueError(f"unknown pattern kind {kind!r}")
    option = "--rename" if kind == "rename" else "--suffix"
    permitted = RENAME_FIELDS if kind == "rename" else SUFFIX_FIELDS

    def bad(why: str) -> UsageError:
        return UsageError(f'{option} "{pattern}" is not valid — {why}; placeholders are '
                          + " ".join("{" + name + "}" for name in sorted(permitted)))

    if not isinstance(pattern, str) or not pattern:
        raise bad("it is empty")
    if any(_is_control(char) for char in pattern):
        raise bad("it contains a control character")
    try:
        parsed = list(string.Formatter().parse(pattern))
    except ValueError:
        raise bad("its braces are unbalanced") from None
    pieces: list[tuple[str, str | None, str]] = []
    has_field = False
    for literal, name, spec, conversion in parsed:
        if "/" in literal or ":" in literal:
            raise bad("/ and : are not allowed in the pattern itself")
        if name is not None:
            spec = spec or ""
            allowed_spec = spec == "" or (kind == "rename" and name == "track" and spec == "02")
            if name not in permitted or conversion or not allowed_spec:
                raise bad(f"{{{name}{':' + spec if spec else ''}}} is not a placeholder")
            has_field = True
        pieces.append((literal, name, spec or ""))
    if kind == "rename" and not has_field:
        raise bad("it needs at least one metadata placeholder")
    if kind == "suffix" and len(pattern) > 32:
        raise bad("use 1 to 32 characters")
    return Pattern(pattern, tuple(pieces))


def sanitise_value(text: str) -> str:
    """Replace ``/``, ``:`` and control characters in a substituted value with ``-`` (§ 3)."""
    return "".join("-" if char in "/:" or _is_control(char) else char for char in text)


def format_target(target_lufs: float) -> str:
    """``{target}`` expansion: ``-16`` for -16.0, ``-14.5`` for -14.5 (cli § 3.2)."""
    text = f"{target_lufs:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def expand_suffix(pattern: str, target_lufs: float) -> str:
    """The expanded ``-s`` suffix; the pattern was validated by ``parse_pattern``."""
    parsed = parse_pattern(pattern, kind="suffix")
    out: list[str] = []
    for literal, name, _spec in parsed.pieces:
        out.append(literal)
        if name == "target":
            out.append(format_target(target_lufs))
    return unicodedata.normalize("NFC", "".join(out))


def _unsafe(detail: str) -> FileFailure:
    return FileFailure("unsafe_output", PATHS_STAGE, detail)


def _placeholder_value(name: str, spec: str, values: Mapping[str, object]) -> str | None:
    if name == "year":
        date = values.get("year")
        if date is None:
            date = values.get("date")
        if date is None:
            return None
        text = str(date).strip()
        return text[:4] if len(text) >= 4 and text[:4].isdigit() else None
    value = values.get(name)
    if value is None or isinstance(value, bool):
        return None
    if name in ("track", "disc"):
        try:
            number = int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        return f"{number:02d}" if spec == "02" else str(number)
    text = str(value)
    return text if text.strip() else None


def render_rename(pattern: str, values: Mapping[str, object]) -> str:
    """The resolved-metadata stem for ``--rename`` before truncation.

    ``values`` maps canonical field names (``title``, ``artist``, ``album_artist``, ``album``,
    ``track``, ``disc``, ``date``, ``genre``) to resolved values; ``year`` derives from ``date``.
    A missing required value fails the file ``unsafe_output``.
    """
    parsed = parse_pattern(pattern, kind="rename")
    out: list[str] = []
    for literal, name, spec in parsed.pieces:
        out.append(literal)
        if name is not None:
            value = _placeholder_value(name, spec, values)
            if value is None:
                raise _unsafe(f"--rename needs {{{name}}}, which this file's tags leave empty")
            out.append(sanitise_value(value))
    return "".join(out)


def truncate_utf8(text: str, limit: int) -> str:
    """The longest prefix of ``text`` whose UTF-8 encoding fits ``limit`` bytes (whole characters)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def generated_stem(stem: str, suffix: str) -> str:
    """NFC stem plus suffix, the stem cut on character boundaries so the whole fits 200 bytes."""
    stem = unicodedata.normalize("NFC", stem)
    suffix = unicodedata.normalize("NFC", suffix)
    room = STEM_MAX_BYTES - len(suffix.encode("utf-8"))
    if room <= 0:
        raise _unsafe(f"the expanded suffix {suffix!r} leaves no room for a name within {STEM_MAX_BYTES} bytes")
    cut = truncate_utf8(stem, room)
    # Truncation must not leave a dangling combining mark that NFC would attach to nothing.
    cut = unicodedata.normalize("NFC", cut)
    return cut + suffix


def check_component(name: str) -> str:
    """Refuse a final filename that no supported volume can hold, or that is not a single name."""
    if not name or name in (".", "..") or name.strip(" .") == "":
        raise _unsafe(f"the output name {name!r} is empty or not a usable file name")
    if "/" in name or "\x00" in name or any(_is_control(char) for char in name):
        raise _unsafe(f"the output name {name!r} contains a path separator or control character")
    if len(name.encode("utf-8")) > NAME_MAX_BYTES:
        raise _unsafe(f"the output name is {len(name.encode('utf-8'))} bytes; the limit is {NAME_MAX_BYTES}")
    return name


# --- Output plans (§ 3; C-08) -----------------------------------------------------------


@dataclass(frozen=True)
class NamingOptions:
    """The C-10 naming inputs after configuration precedence, validated once per run.

    ``out_dir`` must already be canonical (``safety.canonical_directory``). ``suffix`` is the
    raw ``-s`` pattern; ``rename`` the raw ``--rename`` pattern.
    """

    out_dir: Path | None = None
    suffix: str | None = None
    rename: str | None = None
    target_lufs: float = -16.0

    def __post_init__(self) -> None:
        if self.out_dir is not None and self.suffix is not None:
            raise UsageError("-o and -s cannot be used together")
        if self.out_dir is not None and not self.out_dir.is_absolute():
            raise ValueError("out_dir must be an absolute, canonical path")
        if self.suffix is not None:
            parse_pattern(self.suffix, kind="suffix")
        if self.rename is not None:
            parse_pattern(self.rename, kind="rename")

    @property
    def mode(self) -> str:
        return "out_dir" if self.out_dir is not None else "suffix" if self.suffix is not None else "default"


def audio_location(item: InputItem, options: NamingOptions) -> Path:
    """The folder that receives the audio member (§ 3 mode table)."""
    if options.out_dir is not None:
        return options.out_dir.joinpath(*item.relative_path.parent.parts)
    if options.suffix is not None:
        return item.source_path.parent
    return item.source_path.parent / DEFAULT_FOLDER


def audio_name(item: InputItem, target: TargetFormat, options: NamingOptions,
               values: Mapping[str, object] | None = None) -> str:
    """The audio member's file name: stem (source or rename), then suffix, then target extension."""
    suffix = expand_suffix(options.suffix, options.target_lufs) if options.suffix is not None else ""
    if options.rename is not None:
        stem = generated_stem(render_rename(options.rename, values or {}), suffix)
    elif suffix:
        stem = generated_stem(Path(item.source_path.name).stem, suffix)
    else:
        stem = Path(item.source_path.name).stem
    if not stem or stem in (".", "..") or stem.strip(" .") == "":
        raise _unsafe(f"the output stem {stem!r} is empty or unusable")
    return check_component(stem + target.extension)


def _check_request(request: SidecarRequest) -> None:
    if request.member_id == AUDIO_MEMBER_ID:
        raise ValueError("the 'audio' member ID is reserved for the audio member")
    for prefix, purpose in _PURPOSE_FOR_ID_PREFIX.items():
        if request.member_id == prefix or (prefix.endswith("-") and request.member_id.startswith(prefix)):
            if request.purpose != purpose:
                raise ValueError(f"member {request.member_id!r} must have purpose {purpose!r} (MD36)")
            break
    if not request.suffix or "/" in request.suffix or any(_is_control(c) for c in request.suffix):
        raise ValueError(f"sidecar suffix {request.suffix!r} for {request.member_id!r} is not a usable suffix")


def plan_output(item: InputItem, target: TargetFormat, options: NamingOptions, *,
                plan_revision: int,
                sidecar_requests: Sequence[SidecarRequest] = (),
                values: Mapping[str, object] | None = None,
                overwrite: bool = False) -> OutputPlan:
    """The complete C-08 bundle for one input, without temporary paths.

    Raises ``FileFailure('unsafe_output')`` for an unrepresentable resolved name. Collision
    state starts ``none``; ``safety`` and ``find_clashes`` set it from the whole run.
    """
    folder = audio_location(item, options)
    name = audio_name(item, target, options, values)
    audio_path = folder / name
    members = [BundleMember(AUDIO_MEMBER_ID, "audio", None, audio_path, None, comparison_key(audio_path))]
    for request in sidecar_requests:
        _check_request(request)
        sidecar_name = check_component(name + unicodedata.normalize("NFC", request.suffix))
        path = folder / sidecar_name
        members.append(BundleMember(request.member_id, request.purpose, request.inventory_key, path, None,
                                    comparison_key(path)))
    return OutputPlan(plan_revision, tuple(members), "none", item.identity, overwrite)


def rebind(plan: OutputPlan, plan_revision: int) -> OutputPlan:
    """The same members bound to a new C-07 revision (mark finalisation, § 5.3)."""
    return replace(plan, plan_revision=plan_revision)


# --- Collision groups (§ 3 table rows 2 and the bundle namespace) ---------------------------


@dataclass(frozen=True)
class Clash:
    """One bundle member of ``input_index`` sharing its key with ``others`` ((index, member_id) pairs)."""

    input_index: int
    member_id: str
    path: Path
    others: tuple[tuple[int, str], ...]


def find_clashes(plans: Mapping[int, OutputPlan]) -> dict[int, tuple[Clash, ...]]:
    """Every initial participant of a coinciding key, within one bundle or across bundles.

    Returns input index → its clashing members. All participants fail ``name_clash`` (§ 3).
    """
    groups: dict[str, list[tuple[int, BundleMember]]] = {}
    for index, plan in plans.items():
        for member in plan.members:
            groups.setdefault(member.comparison_key, []).append((index, member))
    found: dict[int, list[Clash]] = {}
    for holders in groups.values():
        if len(holders) < 2:
            continue
        for index, member in holders:
            others = tuple((i, m.member_id) for i, m in holders if (i, m.member_id) != (index, member.member_id))
            found.setdefault(index, []).append(Clash(index, member.member_id, member.final_path, others))
    return {index: tuple(clashes) for index, clashes in sorted(found.items())}


def clash_failure(clashes: Sequence[Clash]) -> FileFailure:
    """The ``name_clash`` failure for one file's clashing members."""
    first = clashes[0]
    other = ", ".join(f"input {i} ({m})" for i, m in first.others)
    detail = f"{first.path} ({first.member_id}) would also be written by {other}"
    if len(clashes) > 1:
        detail += f"; {len(clashes) - 1} further member(s) clash"
    return FileFailure("name_clash", PATHS_STAGE, detail)
