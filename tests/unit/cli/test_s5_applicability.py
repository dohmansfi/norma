"""ADR-0021 consumer agreement against independent command-applicability policy."""
from pathlib import Path

import pytest

from norma import config
from norma.errors import UsageError
from norma.models import RunOptions


@pytest.mark.parametrize('command,constrained', [('normalise', True), ('analyse', True),
    ('tags_plan', True), ('verify', False), ('tags_show', False), ('tags_template', False),
    ('formats', False), ('doctor', False)])
@pytest.mark.parametrize('skip,tolerance', [(0, .05), (.05, .05), (.3, .299), (.3, .3), (3, 3), (3, .05)])
def test_resolved_pairs_obey_independent_policy_across_commands(command, constrained, skip, tolerance):
    # merge_values takes parsed spellings. This exercises effective config values, not
    # unsupported flags; verify's skip is deliberately ignored before the inequality.
    prefs = config.ConfigFile(Path('/synthetic/config.toml'), True,
                              {'skip_within': skip, 'tolerance': tolerance}, {})
    cli = {'template_output': '-'} if command == 'tags_template' else {}
    effective_skip = skip if command in ('normalise', 'analyse', 'tags_plan') else .3
    effective_tolerance = tolerance if command in ('normalise', 'analyse', 'tags_plan', 'verify') else .5
    rejected = constrained and effective_skip > effective_tolerance
    if rejected:
        with pytest.raises(UsageError, match='--skip-within must not exceed the effective --tolerance'):
            config.build_options(command, cli, prefs, Path('/synthetic/ffmpeg'))
    else:
        options = config.build_options(command, cli, prefs, Path('/synthetic/ffmpeg'))
        assert (options.skip_within, options.tolerance) == (effective_skip, effective_tolerance)
        assert options.value_sources['skip_within'] == ('config' if constrained else 'default')
        assert options.value_sources['tolerance'] == ('config' if command in ('normalise', 'analyse', 'tags_plan', 'verify') else 'default')


@pytest.mark.parametrize('command', ['normalise', 'analyse', 'tags_plan', 'verify'])
@pytest.mark.parametrize('cli_values,expected,origins', [
    ({}, (.3, .5), ('config', 'config')),
    ({'tolerance': .1}, (.3, .1), ('config', 'cli')),
    ({'skip_within': .05}, (.05, .5), ('cli', 'config')),
    ({'skip_within': .05, 'tolerance': .1}, (.05, .1), ('cli', 'cli'))])
def test_cli_over_config_precedence_is_checked_after_resolution(command, cli_values, expected, origins):
    if command == 'verify' and 'skip_within' in cli_values:
        return  # Parser refusal is covered separately; do not invent a parsed invocation.
    prefs = config.ConfigFile(Path('/synthetic/config.toml'), True, {'skip_within': .3, 'tolerance': .5}, {})
    if command != 'verify' and expected[0] > expected[1]:
        with pytest.raises(UsageError):
            config.build_options(command, cli_values, prefs, Path('/synthetic/ffmpeg'))
    else:
        options = config.build_options(command, cli_values, prefs, Path('/synthetic/ffmpeg'))
        assert (options.skip_within, options.tolerance) == expected
        assert tuple(options.value_sources[key] for key in ('skip_within', 'tolerance')) == (
            ('default', origins[1]) if command == 'verify' else origins)


@pytest.mark.parametrize('tolerance', [-1, 0, .049, 3.001, float('inf'), float('-inf'), float('nan')])
def test_verify_rejects_range_and_non_finite_tolerance(tolerance):
    prefs = config.ConfigFile(Path('/synthetic/config.toml'), False, {}, {})
    with pytest.raises(UsageError):
        config.build_options('verify', {'tolerance': tolerance}, prefs, Path('/synthetic/ffmpeg'))


@pytest.mark.parametrize('command', ['normalise', 'analyse', 'tags_plan', 'verify', 'tags_show', 'tags_template', 'formats', 'doctor'])
@pytest.mark.parametrize('key', ['skip_within', 'tolerance'])
@pytest.mark.parametrize('value', [float('inf'), float('-inf'), float('nan')])
def test_model_finiteness_remains_universal(command, key, value):
    with pytest.raises(ValueError):
        RunOptions(command, Path('/synthetic/ffmpeg'), Path.cwd(), **{key: value})


def test_normalise_dry_run_retains_inequality():
    prefs = config.ConfigFile(Path('/synthetic/config.toml'), False, {}, {})
    with pytest.raises(UsageError):
        config.build_options('normalise', {'dry_run': True, 'tolerance': .1}, prefs, Path('/synthetic/ffmpeg'))


def test_verify_ignored_config_skip_keeps_default_provenance_and_json_value():
    from norma import report
    prefs = config.ConfigFile(Path('/synthetic/config.toml'), True, {'skip_within': 3, 'tolerance': .05}, {})
    options = config.build_options('verify', {}, prefs, Path('/synthetic/ffmpeg'))
    projected = report.portable(options)
    assert (projected['skip_within'], projected['tolerance']) == (.3, .05)
    assert projected['value_sources']['skip_within'] == 'default'
    assert projected['value_sources']['tolerance'] == 'config'
