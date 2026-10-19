"""C-10 preferences, types, aliases, exclusive groups and value provenance."""
from pathlib import Path

import pytest

from norma.config import CONFIG_KEYS, ConfigFile, build_options, load_config, merge_values, validate_value
from norma.errors import UsageError


VALID = dict(target=-14, true_peak=-2, peak_mode='reduce', max_limit=4, tolerance=1, skip_within=0.2,
             no_dual_mono=True, out_dir='~/out', suffix=' [{target}]', rename='{track:02} {title}',
             format='aif', bit_depth=24, sample_rate='48k', dither='triangular', bitrate='V2', flac_level=5,
             lossy_sources='lossless', allow_lossy_transcode=True, no_source_tags=True, prefer='source',
             artist=' Artist ', album_artist='Band', genre='Demo', composer='Composer', folder_artwork=True,
             infer_mode='override', infer_underscores=True, no_title_fallback=True, on_metadata_loss='sidecar',
             strict_conflicts=True, wav_tags='info', id3_version=2.4, artwork_max_size=900,
             no_mark_processed=True, recursive=True, jobs=2, log='~/log', ffmpeg='/tool/ffmpeg')


def config(tmp_path, values):
    return ConfigFile(tmp_path / 'config.toml', True, values, {key: 1 for key in values})


def test_whitelist_has_a_test_for_every_key():
    assert set(VALID) == CONFIG_KEYS


@pytest.mark.parametrize('key', sorted(VALID))
def test_each_config_key_validates(key):
    validate_value(key, VALID[key])


@pytest.mark.parametrize('key', sorted(VALID))
def test_each_config_key_refuses_wrong_type(key):
    with pytest.raises((ValueError, UsageError)):
        validate_value(key, ['not a scalar'])


@pytest.mark.parametrize('key', ('target', 'true_peak', 'max_limit', 'tolerance', 'skip_within'))
@pytest.mark.parametrize('value', (True, float('inf'), float('nan'), '0'))
def test_numeric_preferences_are_finite_numbers_not_booleans(key, value):
    with pytest.raises(ValueError):
        validate_value(key, value)


@pytest.mark.parametrize('key,value', [('jobs', 0), ('jobs', 33), ('jobs', 2.0), ('flac_level', -1),
    ('artwork_max_size', 10001), ('target', -4), ('true_peak', 1), ('skip_within', -1),
    ('format', 'm4a'), ('bitrate', '321k'), ('bitrate', 'v0'), ('sample_rate', True), ('sample_rate', '48001'),
    ('suffix', '{unknown}'), ('suffix', 'bad/name'), ('suffix', 'x' * 33), ('rename', 'plain'),
    ('rename', '{title!r}'), ('rename', '{title:02}'), ('rename', '{track:03}'), ('rename', '{title'),
    ('artist', ' '), ('ffmpeg', 'https://example.invalid/tool')])
def test_invalid_preference(key, value):
    with pytest.raises((ValueError, UsageError)):
        validate_value(key, value)


def test_default_absence_and_dev_null(tmp_path):
    absent = load_config()
    assert not absent.exists and not absent.values
    assert load_config('/dev/null').values == {}
    with pytest.raises(UsageError, match='does not exist'):
        load_config(tmp_path / 'missing.toml')


@pytest.mark.parametrize('text,expected', [
    ('targte = -14\n', 'did you mean "target"'), ('overwrite = true\n', 'not a config option'),
    ('in_place = true\n', 'not available'), ('help = true\n', 'not a config option'),
    ('[loudness]\ntarget = -14\n', 'flat TOML'), ('target =\n', 'Invalid value')])
def test_invalid_config_location(tmp_path, text, expected):
    path = tmp_path / 'config.toml'
    path.write_text(text)
    with pytest.raises(UsageError, match=expected):
        loaded = load_config(path)
        merge_values('analyse', {}, loaded)


def test_load_and_merge_with_provenance(tmp_path):
    path = tmp_path / 'prefs.toml'
    path.write_text('target = -14.0\nformat = "apple"\nrecursive = true\n')
    options = build_options('analyse', {'target': -18.0, 'inputs': ('take.wav',), 'config': str(path)},
                            load_config(path), Path('/tool/ffmpeg'))
    assert options.target == -18
    assert options.value_sources['target'] == 'cli'
    assert options.recursive and options.value_sources['recursive'] == 'config'
    assert options.format == 'same' and options.value_sources['format'] == 'default'
    assert options.inputs == (Path.cwd() / 'take.wav',)
    assert options.config == path and options.ffmpeg.is_absolute()
    assert options.explicit_fields == {'target', 'inputs', 'config'}


@pytest.mark.parametrize('cli,prefs,dropped', [
    ({'suffix': ' v2'}, {'out_dir': '~/out'}, ('out_dir',)),
    ({'out_dir': '~/out'}, {'suffix': '{bad}'}, ('suffix',)),
    ({'artwork': '/picture.png'}, {'folder_artwork': True}, ('folder_artwork',)),
    ({'remove_artwork': True}, {'folder_artwork': True}, ('folder_artwork',)),
    ({'prefer': 'source'}, {'no_source_tags': True}, ('no_source_tags',)),
    ({'no_source_tags': True}, {'prefer': 'source'}, ('prefer',))])
def test_cli_group_choice_drops_exact_config_fields_before_validation(tmp_path, cli, prefs, dropped):
    merged = merge_values('normalise', cli, config(tmp_path, prefs))
    assert merged['dropped_config_fields'] == dropped
    for key in dropped:
        assert merged['value_sources'][key] == 'default'


@pytest.mark.parametrize('prefs', [{'out_dir': '~/out', 'suffix': ' v2'}, {'prefer': 'source', 'no_source_tags': True}])
def test_surviving_config_conflicts(tmp_path, prefs):
    with pytest.raises(UsageError, match='both set'):
        merge_values('normalise', {}, config(tmp_path, prefs))


@pytest.mark.parametrize('cli', [{'out_dir': '/out', 'suffix': ' v2'}, {'prefer': 'source', 'no_source_tags': True},
                               {'verbose': True, 'quiet': True}, {'artwork': '/art', 'remove_artwork': True}])
def test_cli_conflicts(tmp_path, cli):
    with pytest.raises(UsageError, match='cannot be used'):
        merge_values('normalise', cli, config(tmp_path, {}))


def test_skip_within_checked_after_tolerance_override(tmp_path):
    with pytest.raises(UsageError, match='effective'):
        merge_values('analyse', {'skip_within': 1}, config(tmp_path, {}))
    merged = merge_values('analyse', {'skip_within': 1}, config(tmp_path, {'tolerance': 2}))
    assert merged['tolerance'] == 2


def test_aliases_and_path_expansion():
    assert validate_value('format', 'aif') == 'aiff'
    assert validate_value('sample_rate', '44.1k') == 44100
    assert validate_value('bit_depth', 24) == '24'
    assert validate_value('id3_version', 2.3) == '2.3'
    assert validate_value('artist', ' e\u0301 ') == 'é'
    assert validate_value('out_dir', '~/out') == Path.home() / 'out'


def test_multiline_text_is_not_a_table(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('genre = """Demo\n[not a table]\n"""\n')
    loaded = load_config(path)
    assert '[not a table]' in loaded.values['genre']
    with pytest.raises(UsageError, match='line break'):
        merge_values('normalise', {}, loaded)


def test_nonexclusive_prefer_explicit_keeps_no_source_tags(tmp_path):
    values = merge_values('normalise', {'prefer': 'explicit'}, config(tmp_path, {'no_source_tags': True}))
    assert values['no_source_tags'] is True
    assert values['dropped_config_fields'] == ()


def test_large_integer_is_a_usage_error_not_overflow(tmp_path):
    with pytest.raises(UsageError, match='not valid'):
        merge_values('analyse', {}, config(tmp_path, {'target': 10 ** 1000}))
