"""Norma's interface: parsing, application dispatch and truthful run presentation."""
from __future__ import annotations

import argparse
import base64
import difflib
import importlib
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path

from norma import __version__
from norma.config import CONFIG_KEYS, absolute_path, build_options, defaults, load_config, merge_values
from norma.errors import EnvironmentError_, UsageError, combine_exit_codes, exit_contribution
from norma.models import RunReport, TargetFormatName
from typing import cast
from norma import report

PURPOSE = 'Norma — measure and normalise the loudness of music demos; write Apple Music-ready copies.'
EPILOG = 'Configuration: ~/.config/norma/config.toml; CLI > config > defaults.\nExit codes: 0 success (warnings/skips allowed); 1 file failure;\n            2 usage/config; 3 environment; 130 interrupted.'
COMMANDS = ('analyse', 'normalise', 'verify', 'tags', 'formats', 'doctor', 'help')
ALIASES = {'analyze': 'analyse', 'normalize': 'normalise'}


class _InterruptedPreflight(Exception):
    """Carry an unchanged expected error past the CLI-owned SIGINT lifetime."""

    def __init__(self, error: UsageError | EnvironmentError_) -> None:
        self.error = error
        super().__init__(str(error))


class Parser(argparse.ArgumentParser):
    _children: dict[str, Parser]
    examples: str | None = None

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('allow_abbrev', False)
        super().__init__(*args, **kwargs)

    def format_help(self):
        formatter = self._get_formatter()
        formatter.add_text(self.description)
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.examples)
        for group in self._action_groups:
            formatter.start_section(group.title)
            formatter.add_text(group.description)
            formatter.add_arguments(group._group_actions)
            formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()

    def error(self, message):
        match = re.search(r'invalid choice: \'([^\']+)\' \(choose from (.*)\)', message)
        if match:
            choices = [value.strip(" '\"") for value in match[2].split(',')]
            choices = [value for value in choices if value not in ALIASES]
            near = difflib.get_close_matches(match[1], choices, n=1)
            message = f'unknown command "{match[1]}"' + (f' — did you mean "{near[0]}"?' if near else '')
        elif message.startswith('unrecognized arguments:'):
            token = message.split(':', 1)[1].strip().split()[0]
            def options(parser):
                found = [option for action in parser._actions for option in action.option_strings]
                for action in parser._actions:
                    if isinstance(action, argparse._SubParsersAction):
                        for child in action.choices.values():
                            found.extend(options(child))
                return found
            choices = options(self)
            near = difflib.get_close_matches(token.split('=')[0], choices, n=1)
            message = f'unknown option "{token}"' + (f' — did you mean "{near[0]}"?' if near else '')
        raise UsageError(message)


def _common(parser, *, ffmpeg=True):
    parser.add_argument('--config', metavar='FILE', help='Select flat TOML preferences [default: ~/.config/norma/config.toml].')
    if ffmpeg:
        parser.add_argument('--ffmpeg', metavar='PATH', help='FFmpeg executable; ffprobe comes from the same folder [default: PATH].')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action='store_true', help='Show config sources and diagnostic detail [default: off].')
    group.add_argument('-q', '--quiet', action='store_true', help='Hide stderr notes, hints and warnings, not errors or rows [default: off].')
    parser.add_argument('--version', action='version', version=f'norma {__version__}')
    parser.add_argument('--in-place', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-backup', action='store_true', help=argparse.SUPPRESS)


def make_parser(config=None):
    """Build inert 80-column help; load config only when main is called."""
    def factory(**kwargs):
        return Parser(argument_default=argparse.SUPPRESS,
                      formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, width=80), **kwargs)
    parser = factory(prog='norma', description=PURPOSE, epilog=EPILOG)
    parser.examples = 'Start here:\n  norma analyse ~/Music/Demos -r\n  norma tags show demo.wav\n  norma doctor\nUse norma help CMD for more.'
    _common(parser)
    subs = parser.add_subparsers(dest='command', parser_class=Parser, metavar='COMMAND')
    children = {}
    specs = [('analyse', 'Measure loudness and predict gain; changes no audio.'),
             ('normalise', 'Write normalised, tagged copies; never change originals.'), ('verify', 'Check loudness and tags.'),
             ('tags', 'Look at tags: show | plan | template.'),
             ('formats', 'What each output format can store.'), ('doctor', 'Check FFmpeg and other tools.')]
    for name, description in specs:
        child = subs.add_parser(name, aliases=[k for k, v in ALIASES.items() if v == name],
                                description=description, help=description, argument_default=argparse.SUPPRESS,
                                formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, width=80),
                                epilog=EPILOG)
        _common(child, ffmpeg=name != 'formats')
        children[name] = child
    analysis = children['analyse']
    analysis.examples = 'Examples:\n  norma analyse demo.wav\n  norma analyse Demos -r -t -14'
    loudness = analysis.add_argument_group('Loudness')
    builtins = defaults()
    def help_default(key, text):
        tail = f'default: {builtins[key]}'
        if config is not None and key in config.values and config.values[key] != builtins[key]:
            tail += f'; your config: {config.values[key]}'
        return f'{text} [{tail}].'
    for key, short, metavar, text in (
        ('target', '-t', 'LUFS', 'Target for predicted gain, −40 to −5 LUFS'),
        ('true_peak', '-p', 'dBTP', 'Peak ceiling, −9 to 0 dBTP'),
        ('max_limit', None, 'dB', 'Predicted limiter policy cap, 0 to 12 dB'),
        ('skip_within', None, 'LU', 'Copy tolerance, 0 to 3 LU and not above effective tolerance')):
        names = ([short] if short else []) + ['--' + key.replace('_', '-')]
        loudness.add_argument(*names, type=float, metavar=metavar, help=help_default(key, text))
    loudness.add_argument('--peak-mode', metavar='MODE', help=help_default('peak_mode', 'Peak policy: limit | reduce | dynamic'))
    loudness.add_argument('--no-dual-mono', action='store_true', help='Measure mono as one channel [default: off].')
    run = analysis.add_argument_group('Run')
    run.add_argument('-r', '--recursive', action='store_true', help=help_default('recursive', 'Read subfolders'))
    run.add_argument('-j', '--jobs', type=int, metavar='N', help=help_default('jobs', 'Concurrent files, 1 to 32'))
    run.add_argument('--report', metavar='FILE', help='Explicit CSV or JSON report destination.')
    run.add_argument('--overwrite', action='store_true', help='Allow replacing an earlier report [default: off].')
    run.add_argument('--log', metavar='FILE', help='Accepted but ignored by analyse; no log is written.')
    analysis.add_argument('inputs', nargs='*', metavar='PATH')
    tags = children['tags']
    tag_subs = tags.add_subparsers(dest='tag_command', parser_class=Parser, metavar='COMMAND')
    for name in ('show', 'plan', 'template'):
        child = tag_subs.add_parser(name, description=f'{name.capitalize()} source tags.',
                                    argument_default=argparse.SUPPRESS,
                                    formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, width=80), epilog=EPILOG)
        _common(child)
        child.add_argument('-r', '--recursive', action='store_true', help='Read subfolders [default: off].')
        child.add_argument('inputs', nargs='*', metavar='PATH')
    normalise = children['normalise']
    normalise.examples = 'Writes new copies; never changes the originals.\n--dry-run measures every file, so takes about as long as analyse.\nExamples:\n  norma normalise demo.wav\n  norma normalise Demos -r -o Demos-16 --dry-run'
    # Reuse argument declarations, not argparse Action instances shared between parsers.
    for key, short, metavar, text in (
        ('target', '-t', 'LUFS', 'Target, −40 to −5 LUFS'),
        ('true_peak', '-p', 'dBTP', 'Peak ceiling, −9 to 0 dBTP'),
        ('max_limit', None, 'dB', 'Predicted limiter policy cap, 0 to 12 dB'),
        ('tolerance', None, 'LU', 'Loudness acceptance tolerance, 0.05 to 3 LU'),
        ('skip_within', None, 'LU', 'Copy tolerance, 0 to 3 LU and not above tolerance')):
        normalise.add_argument(*(([short] if short else []) + ['--' + key.replace('_', '-')]),
                               type=float, metavar=metavar, help=help_default(key, text))
    normalise.add_argument('--peak-mode', metavar='MODE', help=help_default('peak_mode', 'Peak policy: limit | reduce | dynamic'))
    location = normalise.add_mutually_exclusive_group()
    location.add_argument('-o', '--out-dir', '--output-dir', dest='out_dir', metavar='DIR', help='Output folder [default: Normalised/ beside each input].')
    location.add_argument('-s', '--suffix', metavar='TEXT', help='Write beside inputs with this suffix; {target} is expanded.')
    for key, short, text in (
        ('format', '-f', 'same | wav | aiff | flac | alac | mp3 | aac | apple'),
        ('bit_depth', None, 'same | 16 | 24 | 32 | float'),
        ('sample_rate', None, 'same | 44100 | 48000 | 88200 | 96000 | 176400 | 192000'),
        ('dither', None, 'auto | triangular | none'), ('bitrate', None, 'auto | V0–V9 | 64k–320k'),
        ('lossy_sources', None, 'reencode | lossless | skip')):
        normalise.add_argument(*(([short] if short else []) + ['--' + key.replace('_', '-')]),
                               help=help_default(key, text))
    normalise.add_argument('--flac-level', type=int, help=help_default('flac_level', 'Compression level, 0 to 8'))
    normalise.add_argument('-j', '--jobs', type=int, metavar='N', help=help_default('jobs', 'Concurrent files, 1 to 32'))
    normalise.add_argument('-n', '--dry-run', action='store_true', help='Plan; write nothing except an explicit report.')
    normalise.add_argument('-y', '--yes', action='store_true', help='Accepted for scripts; no effect, no source replacement authority.')
    for key, text in (
        ('recursive', 'Read subfolders'), ('no_dual_mono', 'Measure mono as one channel'),
        ('overwrite', 'Replace earlier outputs/reports, never inputs'), ('force', 'Ignore the processing-mark guard'),
        ('keep_failed', 'Keep a verification-failed candidate; ignored in a dry run'),
        ('allow_lossy_transcode', 'Allow lossy codec changes')):
        names = ['--' + key.replace('_', '-')]
        if key == 'recursive':
            names.insert(0, '-r')
        normalise.add_argument(*names, action='store_true', help=text + ' [default: off].')
    normalise.add_argument('--report', metavar='FILE', help='Explicit .csv/.json report; parent must exist. Otherwise automatic CSV except in a dry run.')
    normalise.add_argument('--log', metavar='FILE', help='Append checked run log; parent must exist; ignored in a dry run.')
    metadata = normalise.add_argument_group('Metadata')
    for key in ('artist', 'album_artist', 'genre', 'composer', 'title', 'album', 'comment', 'date', 'track', 'disc'):
        metadata.add_argument('--' + key.replace('_', '-'), metavar='VALUE', help=f'Set {key.replace("_", " ")} [default: source].')
    for key, text in (('prefer', 'Precedence: explicit | source'), ('infer_mode', 'Filename inference: fill | override'),
                      ('on_metadata_loss', 'Loss policy: warn | sidecar | error'), ('wav_tags', 'WAV stores: info+id3 | info | none'),
                      ('id3_version', 'ID3 version: 2.3 | 2.4')):
        metadata.add_argument('--' + key.replace('_', '-'), help=help_default(key, text))
    for key in ('no_source_tags', 'folder_artwork', 'remove_artwork', 'number_tracks', 'infer_underscores',
                'no_title_fallback', 'strict_conflicts', 'no_mark_processed'):
        metadata.add_argument('--' + key.replace('_', '-'), action='store_true', help=help_default(key, key.replace('_', ' ')))
    for key in ('artwork', 'manifest', 'infer_from_filename', 'rename'):
        metadata.add_argument('--' + key.replace('_', '-'), metavar='VALUE', help=f'{key.replace("_", " ")} [default: none].')
    metadata.add_argument('--artwork-max-size', type=int, metavar='PX', help=help_default('artwork_max_size', 'Maximum artwork dimension, 300 to 10000'))
    metadata.add_argument('--tag', action='append', metavar='KEY=VALUE', help='Set an extra, or compilation=true|false; repeatable.')
    metadata.add_argument('--clear-tag', action='append', metavar='FIELD', help='Clear a field or extra; repeatable.')
    # Reuse declarations, never share argparse actions between commands.
    plan = tag_subs.choices['plan']
    existing = {option for action in plan._actions for option in action.option_strings}
    for action in normalise._actions:
        if not action.option_strings or any(option in existing for option in action.option_strings):
            continue
        kwargs = dict(dest=action.dest, help=action.help, default=argparse.SUPPRESS)
        if isinstance(action, argparse._StoreTrueAction):
            kwargs['action'] = 'store_true'
        elif isinstance(action, argparse._AppendAction):
            kwargs['action'] = 'append'
        else:
            kwargs.update(type=action.type, metavar=action.metavar)
        plan.add_argument(*action.option_strings, **kwargs)
    plan.description = 'Preview tags, conflicts and losses; never measure or render audio.'
    template = tag_subs.choices['template']
    template.description = 'Export source tags and optional filename inference as an editable manifest.'
    template.examples = 'Examples:\n  norma tags template Demos -r -o metadata.csv\n  norma tags template demo.wav -o -'
    template.add_argument('-o', dest='template_output', metavar='FILE', help='Required .csv/.json destination, or - for CSV stdout; parent must exist.')
    template.add_argument('--overwrite', action='store_true', help='Replace an earlier template, never a source [default: off].')
    template.add_argument('--infer-from-filename', metavar='PATTERN', help='Infer from {title} {artist} {album} {genre} {track} {disc} {date} {year} {*} [default: none].')
    template.add_argument('--infer-mode', help=help_default('infer_mode', 'Filename inference: fill | override'))
    template.add_argument('--infer-underscores', action='store_true', help=help_default('infer_underscores', 'Read underscores as spaces'))
    # Group display belongs to each parser; reuse no Action across parsers.
    group_keys = {
        'Loudness': 'target true_peak peak_mode max_limit tolerance skip_within no_dual_mono',
        'Output': 'out_dir suffix rename overwrite',
        'Format': 'format bit_depth sample_rate dither bitrate flac_level lossy_sources allow_lossy_transcode wav_tags id3_version',
        'Metadata': 'artist album_artist genre composer title album comment date track disc prefer infer_mode on_metadata_loss no_source_tags folder_artwork remove_artwork number_tracks infer_underscores no_title_fallback strict_conflicts no_mark_processed artwork manifest infer_from_filename artwork_max_size tag clear_tag force',
        'Run': 'recursive jobs dry_run yes keep_failed report log',
    }
    for child in (normalise, plan):
        groups = {title: next((group for group in child._action_groups if group.title == title), None)
                  or child.add_argument_group(title) for title in group_keys}
        for title, keys in group_keys.items():
            for action in child._actions:
                if action.dest in keys.split():
                    for group in child._action_groups:
                        if action in group._group_actions:
                            group._group_actions.remove(action)
                    groups[title]._group_actions.append(action)
        child._action_groups = [group for group in child._action_groups if group.title not in groups] + list(groups.values())
    normalise.add_argument('inputs', nargs='*', metavar='PATH')
    verify = children['verify']
    verify.description = 'Measure unchanged files against a target and optional explicit tag references; never repair.'
    verify.examples = 'Examples:\n  norma verify demo.wav\n  norma verify Demos -r --check-tags --manifest metadata.csv'
    verify_keys = set('target true_peak tolerance no_dual_mono recursive jobs report overwrite log artist album_artist genre composer title album comment date track disc manifest'.split())
    for action in normalise._actions:
        if action.dest not in verify_keys:
            continue
        kwargs = dict(dest=action.dest, help=action.help, default=argparse.SUPPRESS)
        if isinstance(action, argparse._StoreTrueAction):
            kwargs['action'] = 'store_true'
        else:
            kwargs.update(type=action.type, metavar=action.metavar)
        if action.dest == 'log':
            kwargs['help'] = 'Accepted but ignored by verify; no log is written.'
        elif action.dest == 'report':
            kwargs['help'] = 'Explicit checked CSV or JSON report; no automatic report.'
        elif action.dest in 'artist album_artist genre composer title album comment date track disc manifest'.split():
            kwargs['help'] = 'Explicit reference for --check-tags; never changes tags.'
        verify.add_argument(*action.option_strings, **kwargs)
    verify.add_argument('--check-tags', action='store_true', help='Check mark presence and explicit field/manifest references [default: off].')
    verify.add_argument('inputs', nargs='*', metavar='PATH')
    parser._children = children
    return parser


def _pipeline():
    try:
        return importlib.import_module('norma.pipeline')
    except ImportError as exc:
        if (exc.name or '').split('.')[0] == 'mutagen':
            raise EnvironmentError_('mutagen is not installed — Norma needs it to read and write tags') from None
        raise


def _doctor(pipeline, ffmpeg, config, config_error, explicit_ffmpeg):
    snapshot = pipeline.inspect_environment(ffmpeg)
    failures = []
    def row(status, name, detail):
        print('\t'.join(report.escape_cell(v) for v in (status, name, detail)))
    def check(name, item, *, required=True, message=None):
        if item.problem is not None:
            status = 'FAIL' if required else 'WARN'
            if required:
                failures.append(3)
            row(status, name, message or item.problem.message)
        else:
            value = item.value
            detail = (f'{value.path} · {value.version}' if hasattr(value, 'version') else
                      ', '.join(value.required) if hasattr(value, 'required') else str(value))
            row('OK', name, detail)
    runtime = snapshot.runtime
    row('OK', 'versions', f'norma {runtime.norma_version} · Python {runtime.python_version} · {runtime.python_executable}')
    platform = snapshot.platform
    supported = platform.system == 'Darwin' and platform.machine == 'arm64' and platform.release.split('.')[0].isdigit() and int(platform.release.split('.')[0]) >= 15
    row('OK' if supported else 'WARN', 'platform', f'{platform.system} {platform.release} on {platform.machine}' +
        ('' if supported else ' — Norma is tested only on macOS 15 on Apple Silicon'))
    missing = snapshot.ffmpeg.problem is not None and snapshot.ffmpeg.problem.code == 'not_found'
    check('ffmpeg', snapshot.ffmpeg, message='FFmpeg not found — install it and make sure it is on your PATH, or give its location with --ffmpeg' if missing else None)
    check('ffprobe', snapshot.ffprobe)
    version = snapshot.ffmpeg.value.version if snapshot.ffmpeg.value else 'not found'
    # CR-042: compare the build token only; the version string may carry -g<hash>-<vendor> tails.
    build = re.match(r'(N-[0-9]+)(?=-|$)', version)
    row('OK' if build is not None and build[1] == 'N-125892' else 'WARN', 'tested FFmpeg', f'{version}; tested: N-125892 — reports record the version used')
    check('filters', snapshot.filters)
    check('alimiter options', snapshot.alimiter_options)
    check('encoders', snapshot.encoders)
    native = snapshot.encoders.value is not None and 'alac' in snapshot.encoders.value.present
    row('OK' if native else 'FAIL', 'native alac', 'alac is used; alac_at is never substituted' if native else 'native alac is absent; alac_at is never substituted')
    if not native:
        failures.append(3)
    check('mutagen', snapshot.mutagen, message='mutagen is not installed — Norma needs it to read and write tags' if snapshot.mutagen.problem else None)
    # Missing sips does not disable read-only audio inspection (Releases § 5 gives WS-03 this choice).
    check('sips', snapshot.sips, required=False)
    evidence = snapshot.capability_evidence
    if evidence.problem:
        row('WARN', 'capability evidence', evidence.problem.message)
    for item in evidence.value or ():
        choices = '; '.join(f'{cap.bitrate_mode}: {cap.nominal_bitrates_bps if cap.bitrate_mode == "cbr" else cap.vbr_qualities}' for cap in item.capabilities)
        detail = f'{item.encoder} {item.sample_rate_hz} Hz {item.channel_layout}; basis: {", ".join(item.bases) or "none"}; {choices}'
        if not item.supported:
            detail += ' — automatic output would fail unsupported_target; ' + (item.problem.message if item.problem else '; '.join(item.gaps))
        row('OK' if item.supported else 'WARN', 'lossy evidence', detail)
    if config_error:
        row('FAIL', 'config', str(config_error))
        failures.append(2)
    else:
        row('OK', 'config', f'{config.path}: ' + ('exists and parses' if config.exists else 'missing; using built-in defaults'))
    if explicit_ffmpeg:
        row('FAIL' if snapshot.ffmpeg.problem else 'OK', '--ffmpeg', str(ffmpeg))
    code = combine_exit_codes(failures)
    if code:
        print('Run: norma doctor' if code == 3 else 'Try: norma doctor --help', file=sys.stderr)
    return code


def _formats(*, quiet=False):
    # Pure policy tables, not an installed encoder query or support promise.
    from norma.formats.capabilities import TARGETS, TAG_MATRIX, tag_stores
    for name, target in TARGETS.items():
        depths = ','.join(map(str, target.offered_bit_depths)) + (',float' if target.offers_float else '')
        rate = f'≤{target.max_rate_hz} Hz' if target.max_rate_hz else 'source rate; explicit up to 192000 Hz'
        stores = tag_stores(cast(TargetFormatName, name))
        tags = {store: dict(TAG_MATRIX[store]) for store in stores}
        note = 'policy only; installed settings need executable-matched evidence'
        if target.family == 'lossy':
            note += '; automatic MP3 V0/AAC support is not established here — run norma doctor'
        print('\t'.join(report.escape_cell(v) for v in ('OK', name, ','.join(target.encoders), depths, rate,
                   json.dumps(tags, ensure_ascii=False, separators=(',', ':')), note)))
    if not quiet:
        print('note: v1 accepts mono/stereo only; Music import/display is untested; FLAC is not documented by Apple for Music import', file=sys.stderr)
    return 0


def _max_limit_note(command, options):
    """N-01 for `--max-limit` the user typed with a peak mode that has no limiter cap (cli § 3.1)."""
    if options.value_sources.get('max_limit') == 'cli' and options.peak_mode in ('reduce', 'dynamic'):
        return (f'note: {command} ignores --max-limit — --max-limit has no effect with --peak-mode {options.peak_mode}, '
                'which never uses a limiter cap')
    return None


FLAC_NOTE = 'FLAC is not among the formats Apple documents for Music import — for Music use -f apple or -f alac'


def _flac_note(outcomes):
    """N-05, once per run: any planned or published output is FLAC, fallback outputs included."""
    for outcome in outcomes:
        if outcome.target is not None and outcome.target.format == 'flac' and outcome.output is not None \
                and outcome.execution in ('committed', 'plan', 'publication_failed'):
            return FLAC_NOTE
    return None


def _show_tags(inspection):
    outcome = inspection.outcome
    name = str(outcome.input.source_path)
    tags = inspection.source_tags
    if tags is None:
        return
    def data(kind, key, value):
        print('\t'.join(report.escape_cell(v) for v in (kind, name, key, json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))))
    for store, identities in tags.stores.items():
        data('STORE', store, list(identities))
    for category, values in (('FIELD', tags.fields), ('EXTRA', tags.extras)):
        for key, value in values.items():
            data(category, key, report.portable(value))
    for index, picture in enumerate(tags.artwork):
        data('ARTWORK', str(index + 1), report.portable(picture))
    for item in tags.native_inventory:
        data('NATIVE', item.identity, dict(store=item.store, key=item.key, native_type=item.native_type,
            native_version=item.native_version, payload_encoding=item.payload_encoding, locator=item.locator,
            payload_base64=base64.b64encode(item.payload).decode('ascii')))
    for item in tags.non_tag_items:
        data('NON_TAG', item.identity, {key: getattr(item, key) for key in ('kind', 'locator', 'detail', 'payload_size', 'payload_sha256')})
    for index, conflict in enumerate(tags.internal_conflicts):
        data('CONFLICT', str(index + 1), report.portable(conflict))
    if inspection.mark is not None:
        data('MARK', inspection.mark.validity, report.portable(inspection.mark))


def _inspect(pipeline, command, cli, config):
    from norma import batch

    merged = merge_values(command, cli, config)
    if not cli.get('inputs'):
        raise UsageError('no input given — name a file or folder, for example: norma analyse ~/Music/Demos -r')
    selected = merged.get('ffmpeg', 'ffmpeg')
    found = shutil.which(str(selected))
    if found is None:
        raise EnvironmentError_('FFmpeg not found — install it and make sure it is on your PATH, or give its location with --ffmpeg')
    options = build_options(command, cli, config, absolute_path(found))
    started = time.monotonic()
    tools = pipeline.locate_tools(options, need_ffmpeg=command in ('analyse', 'verify'))
    discovery = pipeline.discover_inputs(options)
    _check_metadata_inputs(options, discovery.items)
    if not options.quiet:
        for hint in discovery.hints:
            print('note: ' + report.escape_cell(hint), file=sys.stderr)
        if options.log is not None:
            print(f'note: {command.replace("_", " ")} writes no log file; --log ignored', file=sys.stderr)
        if command != 'tags_plan' and _max_limit_note(command, options):
            print(_max_limit_note(command, options), file=sys.stderr)
        if command == 'tags_plan':
            ignored = ('target', 'true_peak', 'peak_mode', 'max_limit', 'tolerance', 'skip_within',
                       'no_dual_mono', 'dry_run', 'yes', 'keep_failed')
            typed = [f'--{key.replace("_", "-")}' for key in ignored if key in options.explicit_fields]
            if typed:
                print('note: tags plan ignores ' + ', '.join(typed) + ' — no audio is processed by tags plan', file=sys.stderr)
        if command not in ('tags_plan', 'tags_template') and options.overwrite and options.report is None:
            print(f'note: {command} ignores --overwrite — no file report is requested', file=sys.stderr)
    if options.verbose:
        for key, source in options.value_sources.items():
            if source == 'config':
                print(f'config {key} = {getattr(options, key)}', file=sys.stderr)
        for key in options.dropped_config_fields:
            print(f'config {key} not used: an exclusive option was given on the command line', file=sys.stderr)
    cancel = threading.Event()
    control = batch.BatchControl(cancel_event=cancel, force_hook=pipeline.force_kill_tools)
    def execute(request):
        # Keep the run's signal history; replaying cancellation can force twice
        # or count an independently set Event as an interrupt. No worker has
        # started yet, so bind the request's public coordination seams in place.
        control.commit_lock = request.commit_lock
        control.cancel_event = request.cancel_event
        control.force_hook = request.force_hook
        return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
            jobs=request.jobs, control=control, on_outcome=request.on_outcome, handle_sigint=False)

    inspections = []
    verification_notes = {}
    total = len(discovery.items)
    terminal = sys.stdout.isatty()
    template_stdout = command == 'tags_template' and options.template_output == '-'
    row_stream = sys.stderr if template_stdout else sys.stdout
    def on_inspection(item):
        inspections.append(item)
        details = report.verification_details(item) if command == 'verify' else ()
        verification_notes[item.outcome.input.input_index] = details
        if not terminal:
            print(report.render_tsv(item.outcome, total, primary=item.primary, notes=details), file=row_stream)
            if command == 'tags_show':
                _show_tags(item)
            elif command == 'tags_plan':
                detail = report.render_tag_plan(item.outcome, verbose=options.verbose, options=options, diagnostics=item.diagnostics)
                if detail:
                    print(detail)
        if command == 'verify' and item.outcome.reason_code != 'verification_tags':
            for detail in details:
                print(f'check: {report.escape_cell(item.outcome.input.source_path)}: {report.escape_cell(detail)}', file=sys.stderr)
        if not options.quiet:
            for diagnostic in item.outcome.warnings:
                message = (report.diagnostic_note(replace(item.outcome, reason_code=diagnostic.code), diagnostic)
                           if command == 'verify' else diagnostic.message)
                print(f'warning: {report.escape_cell(item.outcome.input.source_path)}: {report.escape_cell(message)}', file=sys.stderr)
        if options.verbose:
            if command == 'verify':
                for diagnostic in item.diagnostics:
                    if diagnostic != item.primary and diagnostic not in item.outcome.warnings:
                        print(f'diagnostic: {report.escape_cell(diagnostic.code)}: {report.escape_cell(diagnostic.message)}', file=sys.stderr)
            if item.primary and item.primary.detail:
                print('\n'.join(item.primary.detail.splitlines()[-20:]), file=sys.stderr)
            if item.outcome.gain:
                gain = item.outcome.gain
                print(f'predicted limiter policy bound {gain.limiter_reduction_db:.2f} dB; threshold bound {gain.effective_limiter_bound_db:.2f} dB', file=sys.stderr)
        if sys.stderr.isatty() and not options.quiet:
            print(f'{len(inspections)}/{total} {report.escape_cell(item.outcome.input.source_path)}', file=sys.stderr)
    received_sigint = False
    def interrupt(signum, frame):
        nonlocal received_sigint
        # An independently set Event or batch-internal stop is not a CLI signal.
        received_sigint = True
        control.interrupt()
    old_handler = None
    if threading.current_thread() is threading.main_thread():
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, interrupt)
    try:
        try:
            operation = {'analyse': pipeline.analyse_items, 'verify': pipeline.verify_items, 'tags_show': pipeline.inspect_tags,
                         'tags_plan': pipeline.plan_tags_items, 'tags_template': pipeline.template_tags_items}[command]
            kwargs = dict(cancel_event=cancel, on_inspection=on_inspection)
            if command in ('analyse', 'verify'):
                kwargs['execute'] = execute
            if command in ('analyse', 'verify', 'tags_plan') and options.report is not None:
                kwargs['report'] = pipeline.report_request(options.report)
            result = operation(discovery.items, options, tools, **kwargs)
        except (UsageError, EnvironmentError_) as exc:
            if received_sigint:
                raise _InterruptedPreflight(exc) from exc
            raise
        # Inspection owns its selected report through settlement, just as a
        # writing run does. A first SIGINT must not abandon a registered report
        # candidate via the previous handler (safety § 6).
        if terminal:
            print(report.render_table(tuple(item.outcome for item in inspections), total, options.cwd,
                  primaries={i.outcome.input.input_index: i.primary for i in result.inspections},
                  notes=verification_notes, colour=not template_stdout), file=row_stream)
            if command == 'tags_show':
                for item in result.inspections:
                    _show_tags(item)
            elif command == 'tags_plan':
                for item in result.inspections:
                    detail = report.render_tag_plan(item.outcome, verbose=options.verbose, options=options, diagnostics=item.diagnostics)
                    if detail:
                        print(detail)
        if not options.quiet:
            for note in getattr(result, 'notes', ()):
                print('note: ' + report.escape_cell(note), file=sys.stderr)
            for item in result.inspections:
                for note in item.notes:
                    print('note: ' + report.escape_cell(note), file=sys.stderr)
        counts = Counter(o.status for o in result.outcomes)
        envelope = RunReport(run_id=uuid.uuid4().hex, versions={'norma': __version__, 'ffmpeg': tools.ffmpeg.version if tools.ffmpeg else '',
                            'ffprobe': tools.prober.version}, options=options, outcomes=result.outcomes,
                            counts={status: counts[status] for status in report.STATUS_WORDS}, not_started=result.not_started,
                            ignored=sum(discovery.excluded.values()) + discovery.duplicates + discovery.ignored_links + discovery.directory_links,
                            report_path=None, elapsed_s=time.monotonic() - started, run_errors=tuple(getattr(result, 'run_errors', ())))
        if command == 'tags_template':
            for error in result.run_errors:
                print('norma: error: could not write the template — ' + report.escape_cell(error), file=sys.stderr)
            if template_stdout:
                sys.stdout.write(result.text)
            elif result.path is not None and not options.quiet:
                print('note: template: ' + report.escape_cell(result.path), file=sys.stderr)
        else:
            envelope = _publish_report(pipeline, result.outputs, envelope)
        if terminal:
            for item in result.inspections:
                if item.outcome.status == 'failed':
                    print(f'FAIL {report.escape_cell(item.outcome.input.source_path)}: {report.diagnostic_note(item.outcome, item.primary, notes=verification_notes.get(item.outcome.input.input_index, ()))}', file=row_stream)
        print(report.render_summary(envelope, terminal=terminal), file=row_stream)
        code = combine_exit_codes([130 if result.cancelled else 0, 1 if envelope.run_errors else 0,
                                   *(exit_contribution(o.status, o.reason_code) for o in result.outcomes)])
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    # Read the complete handler history after restoration, including signals
    # during summary rendering or restoration itself; never mutate outcomes.
    if result.cancelled or received_sigint:
        cancelled = sum(o.reason_code == 'cancelled' for o in result.outcomes)
        print(f'Interrupted: {len(result.outcomes) - cancelled - counts["failed"]} finished, {cancelled} cancelled, {counts["failed"]} failed, {result.not_started} not started.', file=sys.stderr)
    return combine_exit_codes([130 if received_sigint else 0, code])


def _publish_report(pipeline, outputs, envelope):
    """Only a checked application output writes; never imply publication on failure."""
    destination = outputs.destination('report') if outputs is not None else None
    if destination is None:
        return envelope
    candidate = replace(envelope, report_path=destination.final_path)
    text = report.render_json(candidate) if destination.final_path.suffix == '.json' else report.render_csv(candidate)
    try:
        written = outputs.publish('report', text.encode('utf-8'))
    except pipeline.RunOutputError as exc:
        # N-07 (cli § 10.3), including its racer form: a file that appeared is kept.
        if 'appeared while the run was working' in str(exc):
            message = (f'{destination.final_path} appeared while the run was working and was kept — '
                       'the report was not written; the files above are as their rows say')
        else:
            message = f'could not write the report {destination.final_path} ({exc}) — the files above are as their rows say'
        print('norma: error: ' + report.escape_cell(message), file=sys.stderr)
        return replace(envelope, run_errors=envelope.run_errors + (message,))
    return replace(envelope, report_path=written)


def _check_metadata_inputs(options, items):
    if (options.title is not None or options.track is not None) and len(items) != 1:
        raise UsageError(f'--title and --track need exactly one input file, but {len(items)} were found — use --manifest FILE or --infer-from-filename PATTERN for several files (see: norma tags template --help)')


def _normalise(pipeline, cli, config):
    from norma import batch

    merged = merge_values('normalise', cli, config)

    if not cli.get('inputs'):
        raise UsageError('no input given — name a file or folder, for example: norma normalise Demos -r')
    options = build_options('normalise', cli, config, absolute_path(merged.get('ffmpeg', 'ffmpeg')))
    found = shutil.which(str(merged.get('ffmpeg', 'ffmpeg')))
    if found is None:
        raise EnvironmentError_('FFmpeg not found — install it and make sure it is on your PATH, or give its location with --ffmpeg')
    options = replace(options, ffmpeg=absolute_path(found))
    started = time.monotonic()
    tools = pipeline.locate_normalise_tools(options)
    discovery = pipeline.discover_inputs(options)
    _check_metadata_inputs(options, discovery.items)
    versions = {'norma': __version__, 'ffmpeg': tools.executable.version, 'ffprobe': tools.prober.version}
    run_id = pipeline.new_run_id()
    automatic_name = time.strftime('norma-report-%Y-%m-%dT%H%M%S.csv')
    terminal = sys.stdout.isatty()
    cancel = threading.Event()
    control = batch.BatchControl(cancel_event=cancel, force_hook=pipeline.force_kill_tools)
    records = []
    run_errors = []
    log_fd = None

    def log_line(text):
        nonlocal log_fd
        if log_fd is None:
            return
        try:
            data = (text + '\n').encode('utf-8')
            while data:
                count = os.write(log_fd, data)
                if count == 0:
                    raise OSError('short log write')
                data = data[count:]
        except OSError as exc:
            message = f'cannot append log {options.log}: {exc}'
            run_errors.append(message)
            print('norma: error: ' + report.escape_cell(message), file=sys.stderr)
            os.close(log_fd)
            log_fd = None

    def show(record):
        records.append(record)
        outcome = record.outcome
        row = report.render_tsv(outcome, len(discovery.items), primary=record.primary, notes=record.notes)
        if not terminal:
            print(row)
        log_line(row)
        if not options.quiet:
            for diagnostic in outcome.warnings:
                print(f'warning: {report.escape_cell(outcome.input.source_path)}: {report.escape_cell(diagnostic.message)}', file=sys.stderr)
        # C-04 notes (copy, ignored audio streams, ...) reach stderr unless -q; the row's note cell stays as § 7.1 defines it.
        if not options.quiet:
            for note in record.notes:
                print(('    ' if terminal else '') + 'note: ' + report.escape_cell(note), file=sys.stderr)
        if options.verbose:
            if record.primary and record.primary.detail:
                print('\n'.join(record.primary.detail.splitlines()[-20:]), file=sys.stderr)
            for path in outcome.published_paths:
                if outcome.output is not None and path != outcome.output.audio_member.final_path:
                    print('    published: ' + report.escape_cell(path), file=sys.stderr)
            if outcome.gain:
                gain = outcome.gain
                print(f'predicted limiter policy bound {gain.limiter_reduction_db:.2f} dB; threshold bound {gain.effective_limiter_bound_db:.2f} dB', file=sys.stderr)
            if outcome.target:
                print(f'capability basis: {outcome.target.capability_basis}', file=sys.stderr)
        # Log records all available plan/evidence notes, independent of verbosity.
        for note in record.notes:
            log_line('note: ' + report.escape_cell(note))
        if outcome.target:
            log_line('capability basis: ' + report.escape_cell(outcome.target.capability_basis))
        diagnostics = ((record.primary,) if record.primary is not None else ()) + record.diagnostics
        for diagnostic in diagnostics:
            log_line('diagnostic: ' + report.escape_cell(diagnostic.message))
            if diagnostic.detail:
                log_line('detail: ' + report.escape_cell(diagnostic.detail))

    def progress(event):
        if not options.quiet and (sys.stderr.isatty() or options.verbose):
            print(f'{event.started} started, {event.finished}/{event.total} finished: '
                  + report.escape_cell(event.input.source_path), file=sys.stderr)

    def outputs(plans):
        wanted = []
        if options.report is not None:
            wanted.append(pipeline.report_request(options.report))
        elif not options.dry_run:
            # CR-061: one rule (D-018, D-072), owned by the pipeline.
            wanted.append(pipeline.report_request(pipeline.automatic_report_folder(plans, options) / automatic_name, automatic=True))
        if options.log is not None and not options.dry_run:
            wanted.append(pipeline.log_request(options.log))
        return wanted

    def execute(request):
        nonlocal log_fd
        control.commit_lock = request.commit_lock
        control.force_hook = request.force_hook
        if options.log is not None:
            log_fd = request.outputs.open_log()  # UsageError before its first byte or any row.
            log_line(json.dumps(dict(run_id=run_id, versions=versions, options=report.portable(options)),
                                ensure_ascii=False, separators=(',', ':'), allow_nan=False))
        return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                                   jobs=request.jobs, control=control,
                                   on_outcome=lambda outcome: show(request.record_for(outcome)),
                                   on_progress=progress, handle_sigint=False)

    if not options.quiet:
        for hint in discovery.hints:
            print('note: ' + report.escape_cell(hint), file=sys.stderr)

        # N-01 and N-08 appear only for options the user typed (cli § 3 'Ignored when').
        typed = {key for key, source in options.value_sources.items() if source == 'cli'}
        if options.no_mark_processed:
            print('note: --no-mark-processed leaves out the processing mark — later runs cannot recognise these files; verify has no recorded target and verify --check-tags fails them (processing mark missing)', file=sys.stderr)
        if options.yes:
            print('note: normalise ignores --yes — --yes has no effect: Norma asks no questions', file=sys.stderr)
        if options.dry_run and options.keep_failed:
            print('note: normalise --dry-run ignores --keep-failed — nothing is verified or kept in a dry run', file=sys.stderr)
        if _max_limit_note('normalise', options):
            print(_max_limit_note('normalise', options), file=sys.stderr)
        if options.dry_run and options.log is not None and 'log' in typed:
            print('note: normalise --dry-run writes no log file; --log ignored', file=sys.stderr)
    if options.verbose:
        for key, source in options.value_sources.items():
            if source == 'config':
                print(f'config {key} = {getattr(options, key)}', file=sys.stderr)
        for key in options.dropped_config_fields:
            print(f'config {key} not used: an exclusive option was given on the command line', file=sys.stderr)
    received_sigint = False
    def interrupt(signum, frame):
        nonlocal received_sigint
        received_sigint = True
        control.interrupt()
    old_handler = None
    if threading.current_thread() is threading.main_thread():
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, interrupt)
    try:
        try:
            result = pipeline.normalise_items(discovery.items, options, tools, execute=execute,
                                              run_outputs=outputs, cancel_event=cancel, run_id=run_id)
        except (UsageError, EnvironmentError_) as exc:
            if received_sigint:
                raise _InterruptedPreflight(exc) from exc
            raise
        if options.dry_run or result.outputs is None:
            for record in result.records:
                show(record)
        counts = Counter(o.status for o in result.outcomes)
        for note in result.notes:
            if not options.quiet:
                print('note: ' + report.escape_cell(note), file=sys.stderr)
            log_line('note: ' + report.escape_cell(note))
        flac_note = _flac_note(result.outcomes)
        if flac_note is not None:
            if not options.quiet:
                print('note: ' + flac_note, file=sys.stderr)
            log_line('note: ' + flac_note)
        for error in result.run_errors:
            print('norma: error: ' + report.escape_cell(error), file=sys.stderr)
        envelope = RunReport(run_id, versions, options, result.outcomes,
            {status: counts[status] for status in report.STATUS_WORDS}, result.not_started,
            sum(discovery.excluded.values()) + discovery.duplicates + discovery.ignored_links + discovery.directory_links,
            None, time.monotonic() - started, tuple(result.run_errors) + tuple(run_errors))
        envelope = _publish_report(pipeline, result.outputs, envelope)
        for error in envelope.run_errors:
            log_line('error: ' + report.escape_cell(error))
        log_line(report.render_summary(envelope))
        # A final log error must contribute to the final summary/exit, too.
        envelope = replace(envelope, run_errors=tuple(result.run_errors) + tuple(run_errors) +
                           tuple(e for e in envelope.run_errors if e not in result.run_errors and e not in run_errors))
        if terminal:
            print(report.render_table(result.outcomes, len(discovery.items), options.cwd,
                primaries={r.outcome.input.input_index: r.primary for r in records},
                notes={r.outcome.input.input_index: r.notes for r in records}, colour=True))
            for record in records:
                if record.outcome.status == 'failed':
                    print(f'FAIL {report.escape_cell(record.outcome.input.source_path)}: {report.diagnostic_note(record.outcome, record.primary, notes=record.notes)}')
        if result.cancelled:
            cancelled = sum(o.reason_code == 'cancelled' for o in result.outcomes)
            failed = counts['failed']
            print(f'Interrupted: {len(result.outcomes) - cancelled - failed} finished, {cancelled} cancelled, {failed} failed, {result.not_started} not started.', file=sys.stderr)
            if not terminal:
                for record in records:
                    if record.outcome.execution == 'publication_failed':
                        print(report.diagnostic_note(record.outcome, record.primary, notes=record.notes), file=sys.stderr)
        print(report.render_summary(envelope, terminal=terminal))
        code = combine_exit_codes([130 if result.cancelled else 0, 1 if envelope.run_errors else 0,
                                  *(exit_contribution(o.status, o.reason_code) for o in result.outcomes)])
    finally:
        try:
            if log_fd is not None:
                os.close(log_fd)
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)
    # Finalisation still belongs to our handler. Read its signal history only
    # after restoring the previous owner, not in a return evaluated before finally.
    if received_sigint and not result.cancelled:
        cancelled = sum(o.reason_code == 'cancelled' for o in result.outcomes)
        failed = counts['failed']
        print(f'Interrupted: {len(result.outcomes) - cancelled - failed} finished, {cancelled} cancelled, {failed} failed, {result.not_started} not started.', file=sys.stderr)
    return combine_exit_codes([130 if received_sigint else 0, code])


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = 'analyse'
    try:
        # Safety refusals take precedence over help/version and all work (D-028).
        for token in args[:args.index('--')] if '--' in args else args:
            if token.split('=')[0] in ('--in-place', '--no-backup'):
                raise UsageError(f'{token.split("=")[0]} is not available in this version — Norma never replaces your originals; it writes new copies into Normalised/ beside them, or where -o or -s says')
        if args and args[0] == 'help':
            args = args[1:] + ['--help']
        help_config = None
        if any(token in ('-h', '--help') for token in args):
            config_arg = next((token.split('=', 1)[1] for token in args if token.startswith('--config=')), None)
            if '--config' in args:
                position = args.index('--config')
                if position + 1 < len(args):
                    config_arg = args[position + 1]
            help_config = load_config(config_arg)
        parser = make_parser(help_config)
        try:
            namespace, unknown = parser.parse_known_args(args)
        except SystemExit as exc:
            return int(exc.code or 0)
        cli = vars(namespace)
        parsed_command = cli.pop('command', None)
        command = ALIASES.get(parsed_command, parsed_command)
        if command is None:
            if unknown:
                parser.error('unrecognized arguments: ' + ' '.join(unknown))
            parser.print_help()
            return 0
        if command == 'tags':
            subcommand = cli.pop('tag_command', None)
            if subcommand is None:
                if unknown:
                    parser.error('unrecognized arguments: ' + ' '.join(unknown))
                parser._children['tags'].print_help()
                return 0
            command = 'tags_' + subcommand

        # argparse's nargs='*' leaves trailing paths after an intervening option.
        # Admit those paths only in path-taking commands, never an unknown option.
        if command in ('analyse', 'normalise', 'verify', 'tags_show', 'tags_plan', 'tags_template') and unknown and all(not token.startswith('-') for token in unknown):
            cli['inputs'] = cli.get('inputs', []) + unknown
            unknown = []
        if unknown:
            option = unknown[0].split('=')[0]
            if option.startswith('--') and option[2:].replace('-', '_') in CONFIG_KEYS:
                raise UsageError(f'{option} is not used by {command.replace("_", " ")}')
            parser.error('unrecognized arguments: ' + ' '.join(unknown))
        if command == 'formats' and 'ffmpeg' in cli:
            raise UsageError('--ffmpeg is not used by formats')

        config_error = None
        try:
            config = load_config(cli.get('config'))
            merged = merge_values(command, cli, config)
        except UsageError as exc:
            if command != 'doctor':
                raise
            config_error = exc
            config = None
            merged = dict(cli)
        pipeline = _pipeline() if command != 'formats' else None
        if command == 'doctor':
            return _doctor(pipeline, merged.get('ffmpeg', 'ffmpeg'), config, config_error, 'ffmpeg' in cli)
        if command == 'formats':
            return _formats(quiet=cli.get('quiet', False))
        if command == 'normalise':
            return _normalise(pipeline, cli, config)
        return _inspect(pipeline, command, cli, config)
    except (UsageError, EnvironmentError_, _InterruptedPreflight) as caught:
        interrupted = isinstance(caught, _InterruptedPreflight)
        exc = caught.error if isinstance(caught, _InterruptedPreflight) else caught
        print(f'norma: error: {exc}', file=sys.stderr)
        print('Run: norma doctor' if exc.exit_code == 3 else f'Try: norma {(command or "analyse").replace("_", " ")} --help', file=sys.stderr)
        if interrupted:
            # No producer result exists: do not invent rows, counts or a report.
            print('Interrupted before processing finished.', file=sys.stderr)
        return combine_exit_codes([130 if interrupted else 0, exc.exit_code])
    except KeyboardInterrupt:
        print('Interrupted before processing finished.', file=sys.stderr)
        return 130
