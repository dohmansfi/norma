"""Flat TOML preferences and provenance-preserving C-10 merging (cli § 6)."""
from __future__ import annotations

import difflib
import math
import re
import string
import tomllib
import unicodedata
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from collections.abc import Mapping
from typing import Any, NoReturn

from norma.errors import UsageError
from norma.models import SKIP_WITHIN_COMMANDS, FrozenMap, RunOptions

CONFIG_KEYS = frozenset('target true_peak peak_mode max_limit tolerance skip_within no_dual_mono out_dir suffix rename format bit_depth sample_rate dither bitrate flac_level lossy_sources allow_lossy_transcode no_source_tags prefer artist album_artist genre composer folder_artwork infer_mode infer_underscores no_title_fallback on_metadata_loss strict_conflicts wav_tags id3_version artwork_max_size no_mark_processed recursive jobs log ffmpeg'.split())
CONTEXT_KEYS = frozenset(('value_sources', 'explicit_fields', 'dropped_config_fields', 'cwd'))
NON_CONFIG_KEYS = (frozenset(f.name for f in fields(RunOptions)) - CONFIG_KEYS) | {'help', 'version'}
RANGES = {'target': (-40, -5), 'true_peak': (-9, 0), 'max_limit': (0, 12),
          'tolerance': (0.05, 3), 'skip_within': (0, 3), 'jobs': (1, 32),
          'flac_level': (0, 8), 'artwork_max_size': (300, 10000)}
ENUMS = {'peak_mode': ('limit', 'reduce', 'dynamic'), 'format': ('same', 'wav', 'flac', 'alac', 'aiff', 'mp3', 'aac', 'apple'),
         'bit_depth': ('same', '16', '24', '32', 'float'), 'dither': ('auto', 'triangular', 'none'),
         'lossy_sources': ('reencode', 'lossless', 'skip'), 'prefer': ('explicit', 'source'),
         'infer_mode': ('fill', 'override'), 'on_metadata_loss': ('warn', 'sidecar', 'error'),
         'wav_tags': ('info+id3', 'info', 'none'), 'id3_version': ('2.3', '2.4')}
BOOL_KEYS = frozenset('no_dual_mono allow_lossy_transcode no_source_tags folder_artwork infer_underscores no_title_fallback strict_conflicts no_mark_processed recursive'.split())
INT_KEYS = frozenset(('jobs', 'flac_level', 'artwork_max_size'))
PATH_KEYS = frozenset(('out_dir', 'log', 'ffmpeg', 'report', 'artwork', 'manifest'))
ANALYSE_KEYS = frozenset('target true_peak peak_mode max_limit tolerance skip_within no_dual_mono recursive jobs log ffmpeg'.split())
COMMAND_KEYS = {'analyse': ANALYSE_KEYS, 'tags_show': frozenset(('recursive', 'ffmpeg')),
                'verify': frozenset('target true_peak tolerance no_dual_mono recursive jobs log ffmpeg'.split()),
                'tags_template': frozenset(('recursive', 'ffmpeg', 'infer_mode', 'infer_underscores')),
                'doctor': frozenset(('ffmpeg',)), 'formats': frozenset()}


@dataclass(frozen=True)
class ConfigFile:
    path: Path
    exists: bool
    values: Mapping[str, object]
    lines: Mapping[str, int]


def absolute_path(value: str | Path, cwd: Path | None = None) -> Path:
    text = str(value)
    if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*://', text):
        raise UsageError(f'{text} is not a local path')
    path = Path(text).expanduser()
    return path if path.is_absolute() else (cwd or Path.cwd()) / path


def load_config(path: str | Path | None = None) -> ConfigFile:
    explicit = path is not None
    selected = absolute_path(path) if explicit else Path.home() / '.config/norma/config.toml'
    if selected == Path('/dev/null'):
        return ConfigFile(selected, True, FrozenMap(), FrozenMap())
    try:
        text = selected.read_text(encoding='utf-8')
    except FileNotFoundError:
        if explicit:
            raise UsageError(f'{selected} does not exist') from None
        return ConfigFile(selected, False, FrozenMap(), FrozenMap())
    except (OSError, UnicodeError) as exc:
        raise UsageError(f'{selected}: cannot read config ({exc})') from None
    try:
        values = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise UsageError(f'{selected}: {exc}') from None
    lines = {}
    # TOML owns parsing; this scanner supplies diagnostic locations, including quoted keys.
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'\s*(?:"([^"]+)"|\'([^\']+)\'|([A-Za-z0-9_-]+))\s*=', line)
        if match:
            lines.setdefault(next(part for part in match.groups() if part is not None), number)
    for key, value in values.items():
        location = f'{selected}:{lines.get(key, 1)}'
        if isinstance(value, dict) or (isinstance(value, list) and any(isinstance(item, dict) for item in value)):
            raise UsageError(f'{location}: config must be flat TOML — no tables or sections')
        if key not in CONFIG_KEYS:
            if key in NON_CONFIG_KEYS:
                tail = f'give it on the command line as --{key.replace("_", "-")}'
                if key in ('in_place', 'no_backup'):
                    tail = f'--{key.replace("_", "-")} is not available in this version'
                raise UsageError(f'{location}: "{key}" is not a config option — it changes files or applies to one run only; {tail}')
            near = difflib.get_close_matches(key, sorted(CONFIG_KEYS), n=1)
            hint = f' — did you mean "{near[0]}"?' if near else ''
            raise UsageError(f'{location}: unknown config key "{key}"{hint}')

    return ConfigFile(selected, True, FrozenMap(values), FrozenMap(lines))


def defaults() -> dict[str, Any]:
    result = {}
    for item in fields(RunOptions):
        if item.default is not MISSING:
            result[item.name] = item.default
        elif item.default_factory is not MISSING:
            result[item.name] = item.default_factory()
    return result


def validate_value(key: str, value: Any) -> Any:
    """Validate a preference's type/range and canonicalise documented aliases only."""
    def bad(why: str) -> NoReturn:
        raise ValueError(why)
    if key in BOOL_KEYS:
        if type(value) is not bool:
            bad('use true or false')
    elif key in RANGES:
        if (type(value) is not int if key in INT_KEYS else type(value) not in (int, float)):
            bad('use an integer' if key in INT_KEYS else 'use a finite number')
        low, high = RANGES[key]
        if isinstance(value, float) and not math.isfinite(value):
            bad('use a finite number')
        if not low <= value <= high:
            bad(f'use {low} to {high}')
        if key not in INT_KEYS:
            value = float(value)
    elif key in ENUMS:
        if key in ('bit_depth', 'id3_version') and type(value) in (int, float):
            value = str(value)
        if key == 'format':
            if value == 'm4a':
                bad('"m4a" is ambiguous — say alac (lossless), aac (lossy), or apple')
            if value == 'aif':
                value = 'aiff'
        if value not in ENUMS[key] or not isinstance(value, str):
            bad('choose ' + ' | '.join(ENUMS[key]))
    elif key == 'sample_rate':
        aliases = {'44.1k': 44100, '48k': 48000, '88.2k': 88200, '96k': 96000, '176.4k': 176400, '192k': 192000}
        if isinstance(value, str):
            value = aliases.get(value, value)
            if isinstance(value, str) and value.isdigit():
                value = int(value)
        if value != 'same' and (type(value) is not int or value not in aliases.values()):
            bad('choose same | 44100 | 48000 | 88200 | 96000 | 176400 | 192000')
    elif key == 'bitrate':
        if not isinstance(value, str) or not (value == 'auto' or re.fullmatch(r'V[0-9]', value) or
                (re.fullmatch(r'[0-9]+k', value) and 64 <= int(value[:-1]) <= 320)):
            bad('choose auto | V0–V9 | 64k–320k; settings must also be evidenced per file')
    elif key in PATH_KEYS:
        if not isinstance(value, (str, Path)) or not str(value):
            bad('use a non-empty local path')
        value = absolute_path(value)
    elif key in ('artist', 'album_artist', 'genre', 'composer', 'title', 'album', 'comment'):
        from norma.metadata.model import canonical_text
        if not isinstance(value, str):
            bad('use text')
        value = canonical_text(value, key)
        if not value and key != 'comment':
            bad('use text of 1 to 1000 characters after trimming')
    elif key == 'date':
        from norma.metadata.model import canonical_date
        value = canonical_date(value)
    elif key in ('track', 'disc'):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9]+(?:/[0-9]+)?', value):
            bad('use N or N/M, integers at least 1, with N no greater than M')
        numbers = tuple(int(part) for part in value.split('/'))
        if not all(numbers) or (len(numbers) == 2 and numbers[0] > numbers[1]):
            bad('use N or N/M, integers at least 1, with N no greater than M')
        value = numbers
    elif key == 'infer_from_filename':
        from norma.metadata.infer import compile_pattern
        if not re.search(r'\{[^{}]+\}', value):
            bad('include at least one filename placeholder')
        compile_pattern(value)
    elif key in ('suffix', 'rename'):
        if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
            bad('use a non-empty pattern without control characters')
        permitted = {'target'} if key == 'suffix' else {'title', 'artist', 'album_artist', 'album', 'track', 'disc', 'date', 'year', 'genre'}
        try:
            pieces = list(string.Formatter().parse(value))
        except ValueError:
            bad('unbalanced pattern braces')
        has_field = False
        for literal, name, spec, conversion in pieces:
            if '/' in literal or ':' in literal:
                bad('no / or : in the pattern itself')
            if name is not None:
                has_field = True
                if name not in permitted or conversion or (spec and not (key == 'rename' and name == 'track' and spec == '02')):
                    bad('placeholders are ' + ', '.join(sorted(permitted)))
        if key == 'suffix' and len(value) > 32:
            bad('use 1 to 32 characters')
        if key == 'rename' and not has_field:
            bad('include at least one metadata placeholder')
    return value


def merge_values(command: str, cli: Mapping[str, Any], config: ConfigFile) -> dict[str, Any]:
    """Merge before validation so explicit group choices displace excluded preferences."""
    if command == 'verify' and not cli.get('check_tags'):
        references = 'artist album_artist genre composer title album comment date track disc manifest'.split()
        supplied = next((key for key in references if key in cli), None)
        if supplied is not None:
            raise UsageError(f'--{supplied.replace("_", "-")} is only checked by verify with --check-tags')
    permitted = COMMAND_KEYS.get(command, CONFIG_KEYS)
    prefs = dict(config.values)
    dropped = []
    groups = [('out_dir', 'suffix'), ('artwork', 'folder_artwork', 'remove_artwork'), ('verbose', 'quiet')]
    for group in groups:
        chosen = [k for k in group if k in cli and cli[k] is not False and cli[k] is not None]
        if len(chosen) > 1:
            raise UsageError(f'--{chosen[0].replace("_", "-")} cannot be used with --{chosen[1].replace("_", "-")} — pick one')
        if chosen:
            for key in group:
                if key != chosen[0] and key in prefs:
                    dropped.append(key)
                    del prefs[key]
    if cli.get('prefer') == 'source' and cli.get('no_source_tags'):
        raise UsageError('--prefer source cannot be used with --no-source-tags — pick one')
    for chosen, excluded in (('prefer', 'no_source_tags'), ('no_source_tags', 'prefer')):
        active = cli.get(chosen) == 'source' if chosen == 'prefer' else cli.get(chosen) is True
        if active and excluded in prefs:
            dropped.append(excluded)
            del prefs[excluded]
    for a, b in (('out_dir', 'suffix'), ('prefer', 'no_source_tags')):
        if a not in permitted or b not in permitted:
            continue
        conflict = (prefs.get(a) == 'source' and prefs.get(b) is True) if a == 'prefer' else prefs.get(a) is not None and prefs.get(b) is not None
        if conflict:
            raise UsageError(f'{config.path}: {a} and {b} are both set — keep one, or choose on the command line')
    validated = {}
    for key, value in prefs.items():
        try:
            validated[key] = validate_value(key, value)
        except ValueError as exc:
            raise UsageError(f'{config.path}:{config.lines.get(key, 1)}: {key} = {value!r} is not valid — {exc}') from None
    result = defaults()
    sources = {item.name: 'default' for item in fields(RunOptions) if item.name not in CONTEXT_KEYS}
    permitted = COMMAND_KEYS.get(command, CONFIG_KEYS)
    for key, value in validated.items():
        if key in permitted:
            result[key] = value
            sources[key] = 'config'
    for key, value in cli.items():
        if key in CONFIG_KEYS or key in ('report', 'title', 'album', 'comment', 'date', 'track', 'disc',
                                        'artwork', 'manifest', 'infer_from_filename'):
            try:
                value = validate_value(key, value)
            except (ValueError, TypeError) as exc:
                raise UsageError(f'{value!r} is not valid for --{key.replace("_", "-")} — {exc}') from None
        result[key] = value
        sources[key] = 'cli'
    if command in ('normalise', 'tags_plan', 'verify'):
        _metadata_options(result, cli, sources)
    if command == 'tags_template':
        destination = result['template_output']
        if destination is None:
            raise UsageError('tags template requires -o FILE.csv|FILE.json|-')
        if destination != '-':
            destination = absolute_path(destination)
            if destination.suffix not in ('.csv', '.json'):
                raise UsageError('tags template -o needs .csv or .json, or - for stdout')
            result['template_output'] = destination
        if not result['infer_from_filename']:
            for key in ('infer_mode', 'infer_underscores'):
                if key in cli:
                    raise UsageError(f'--{key.replace("_", "-")} needs --infer-from-filename')
    if command in SKIP_WITHIN_COMMANDS and result['skip_within'] > result['tolerance']:
        raise UsageError('--skip-within must not exceed the effective --tolerance')
    result.update(command=command, config=config.path, value_sources=sources,
                  explicit_fields=frozenset(cli), dropped_config_fields=tuple(dropped))
    sources['command'] = 'cli'
    sources['config'] = 'cli' if 'config' in cli else 'default'
    return result


def _metadata_options(result, cli, sources):
    """Convert the CLI spelling to C-10 without changing metadata precedence."""
    from norma.metadata import model

    for key in ('track', 'disc'):
        if key in cli:
            numbers = result[key]
            result[key] = numbers[0]
            if len(numbers) == 2:
                result[key + '_total'] = numbers[1]
                sources[key + '_total'] = 'cli'
    if 'track' in cli and cli.get('number_tracks'):
        raise UsageError('--track cannot be used with --number-tracks — pick one')
    if not result['infer_from_filename']:
        for key in ('infer_mode', 'infer_underscores'):
            if key in cli:
                raise UsageError(f'--{key.replace("_", "-")} needs --infer-from-filename')
    pairs = []
    for entry in cli.get('tag', ()):
        key, sep, raw = entry.partition('=')
        if not sep:
            raise UsageError('--tag needs KEY=VALUE')
        if model.is_reserved_name(key):
            raise UsageError(f'"{key}" is reserved for tags Norma writes or removes itself (NORMA, ITUNNORM, ITUNSMPB, ENCODER, REPLAYGAIN_…, R128_…) — choose another name; to leave the processing mark out, use --no-mark-processed')
        if key.casefold() == 'compilation':
            if raw not in ('true', 'false'):
                raise UsageError('--tag compilation needs true or false')
            key, value = 'compilation', raw
        else:
            if model.extra_name_problem(key):
                raise UsageError(f'--tag {key}: use an extra name, or compilation=true|false; canonical fields have their own options')
            try:
                value = model.canonical_text(raw, model.EXTRA)
            except ValueError as exc:
                raise UsageError(f'--tag {key}: {exc}') from None
        if not value:
            raise UsageError(f'--tag {key}= has no value — to remove a tag, use --clear-tag {key}')
        pairs.append((key, value))
    result['tag'] = tuple(pairs)
    clears = []
    for key in cli.get('clear_tag', ()):
        folded = key.casefold()
        if folded == 'artwork':
            raise UsageError('--clear-tag artwork: use --remove-artwork')
        if folded in model.CANONICAL_FIELDS:
            key = folded
        elif model.extra_name_problem(key):
            raise UsageError(f'--clear-tag {key}: not a legal field or extra name')
        conflicts = (key in cli or any(name.casefold() == key.casefold() for name, _ in pairs)
                     or (key in ('track', 'track_total') and cli.get('number_tracks'))
                     or (key == 'track_total' and 'track' in cli and result['track_total'] is not None)
                     or (key == 'disc_total' and 'disc' in cli and result['disc_total'] is not None))
        if conflicts:
            raise UsageError(f'--clear-tag {key} cannot be used with a value for the same field — pick one')
        clears.append(key)
    result['clear_tag'] = tuple(clears)

    if result['manifest'] is not None and result['manifest'].suffix not in ('.csv', '.json'):
        raise UsageError('--manifest must name a .csv or .json file')


def build_options(command: str, cli: Mapping[str, Any], config: ConfigFile, ffmpeg: Path) -> RunOptions:
    values = merge_values(command, cli, config)
    for key in ('artwork', 'manifest'):
        path = values[key]
        if path is not None and not path.exists():
            raise UsageError(f'{path} does not exist')
    values['ffmpeg'] = ffmpeg
    values['cwd'] = Path.cwd()
    values['inputs'] = tuple(absolute_path(p) for p in values.get('inputs', ()))
    if values['report'] is not None and values['report'].suffix not in ('.csv', '.json'):
        raise UsageError('--report must name a .csv or .json file')
    return RunOptions(**values)
