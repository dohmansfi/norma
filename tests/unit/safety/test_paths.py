"""TST-S-02 — pure output naming, comparison keys and collision groups (safety § 3; C-08).

Everything here is pure: paths are built by hand and never touched on disk.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path, PurePosixPath

import pytest

from norma import paths
from norma.errors import FileFailure, UsageError
from norma.models import InputItem, SidecarRequest, TargetFormat
from norma.paths import NamingOptions, comparison_key, find_clashes, plan_output

ROOT = Path("/Volumes/Work/Demos")


def item(relative: str, index: int = 0, *, root: Path = ROOT, explicit: bool = False) -> InputItem:
    rel = PurePosixPath(relative)
    return InputItem(root.joinpath(*rel.parts), root, rel, (1, 100 + index), index, explicit)


def target(extension: str = ".flac", fmt: str = "flac") -> TargetFormat:
    return TargetFormat(format=fmt, extension=extension, encoder=fmt, bit_depth=24, sample_float=False,  # type: ignore[arg-type]
                        sample_rate_hz=48000, channels=2, channel_layout="stereo", dither="none",
                        bitrate_mode="none", requested_bitrate_bps=None, effective_bitrate_bps=None,
                        vbr_quality=None, bitrate_source="not_applicable", capability_basis="ev:test",
                        stream_copy=False, decision="as_requested")


def audio(plan) -> Path:  # type: ignore[no-untyped-def]
    return plan.audio_member.final_path


# --- Locations (§ 3 mode table) -----------------------------------------------------------


def test_default_writes_beside_the_source_in_normalised() -> None:
    plan = plan_output(item("song.wav"), target(), NamingOptions(), plan_revision=0)
    assert audio(plan) == ROOT / "Normalised" / "song.flac"
    assert plan.collision == "none" and plan.audio_member.temporary_path is None
    assert plan.source_identity == (1, 100)


def test_recursive_default_keeps_normalised_beside_each_source_not_a_mirrored_tree() -> None:
    plan = plan_output(item("Album/Disc 1/song.wav"), target(), NamingOptions(), plan_revision=0)
    assert audio(plan) == ROOT / "Album" / "Disc 1" / "Normalised" / "song.flac"


def test_out_dir_mirrors_the_relative_parent_from_the_retained_root() -> None:
    out = Path("/Volumes/Out")
    plan = plan_output(item("Album/song.wav"), target(), NamingOptions(out_dir=out), plan_revision=0)
    assert audio(plan) == out / "Album" / "song.flac"


def test_out_dir_is_flat_for_an_explicit_file_and_top_level_entries() -> None:
    out = Path("/Volumes/Out")
    explicit = item("take.wav", explicit=True)
    assert audio(plan_output(explicit, target(), NamingOptions(out_dir=out), plan_revision=0)) == out / "take.flac"


def test_multiple_roots_with_out_dir_share_one_tree_and_clash_without_invented_prefixes() -> None:
    out = Path("/Volumes/Out")
    first = item("song.wav", 0, root=Path("/A"))
    second = item("song.aiff", 1, root=Path("/B"))
    plans = {i.input_index: plan_output(i, target(), NamingOptions(out_dir=out), plan_revision=0)
             for i in (first, second)}
    assert audio(plans[0]) == audio(plans[1]) == out / "song.flac"
    assert set(find_clashes(plans)) == {0, 1}


def test_suffix_stays_beside_the_source_and_expands_target() -> None:
    options = NamingOptions(suffix=" [{target}]", target_lufs=-16.0)
    assert audio(plan_output(item("Album/song.wav"), target(), options, plan_revision=0)) == \
        ROOT / "Album" / "song [-16].flac"
    options = NamingOptions(suffix="_{target}LUFS", target_lufs=-14.5)
    assert audio(plan_output(item("song.wav"), target(), options, plan_revision=0)).name == "song_-14.5LUFS.flac"


@pytest.mark.parametrize("value, text", [(-16.0, "-16"), (-14.5, "-14.5"), (-23.25, "-23.25"), (0.0, "0")])
def test_target_expansion(value: float, text: str) -> None:
    assert paths.format_target(value) == text


def test_out_dir_and_suffix_together_are_a_usage_error() -> None:
    with pytest.raises(UsageError):
        NamingOptions(out_dir=Path("/x"), suffix="-n")


# --- Rename (§ 3; cli § 3.2) ------------------------------------------------------------------

VALUES = {"title": "Song", "artist": "Test Artist", "album_artist": "D. G.", "album": "Demos", "track": 3,
          "disc": 1, "date": "2026-10-05", "genre": "Pop"}


def test_rename_uses_every_placeholder_and_changes_only_the_stem() -> None:
    pattern = "{track:02} {track} {disc} {title} {artist} {album_artist} {album} {date} {year} {genre}"
    plan = plan_output(item("Album/x.wav"), target(), NamingOptions(rename=pattern), plan_revision=0,
                       values=VALUES)
    assert audio(plan) == ROOT / "Album" / "Normalised" / "03 3 1 Song Test Artist D. G. Demos 2026-10-05 2026 Pop.flac"


def test_rename_then_suffix_in_that_order() -> None:
    options = NamingOptions(rename="{title}", suffix=" ({target})", target_lufs=-16)
    assert audio(plan_output(item("x.wav"), target(), options, plan_revision=0, values=VALUES)).name == \
        "Song (-16).flac"


def test_substituted_separators_and_controls_become_hyphens() -> None:
    values = {**VALUES, "title": "AC/DC: Live\tTake\x7f"}
    plan = plan_output(item("x.wav"), target(), NamingOptions(rename="{title}"), plan_revision=0, values=values)
    assert audio(plan).name == "AC-DC- Live-Take-.flac"


@pytest.mark.parametrize("values", [
    {**VALUES, "title": None}, {**VALUES, "title": "   "}, {k: v for k, v in VALUES.items() if k != "title"},
])
def test_missing_substitution_is_unsafe_output(values: dict) -> None:
    with pytest.raises(FileFailure) as caught:
        plan_output(item("x.wav"), target(), NamingOptions(rename="{track} {title}"), plan_revision=0, values=values)
    assert caught.value.reason == "unsafe_output"


@pytest.mark.parametrize("title", [".", "..", " . "])
def test_dot_stems_are_unsafe_output(title: str) -> None:
    with pytest.raises(FileFailure) as caught:
        plan_output(item("x.wav"), target(), NamingOptions(rename="{title}"), plan_revision=0,
                    values={**VALUES, "title": title})
    assert caught.value.reason == "unsafe_output"


def test_year_derives_from_date_and_a_bad_date_is_missing() -> None:
    assert paths.render_rename("{year}", {"date": "2026"}) == "2026"
    with pytest.raises(FileFailure):
        paths.render_rename("{year}", {"date": "Spring"})
    with pytest.raises(FileFailure):
        paths.render_rename("{track}", {"track": 0})  # 0 is absent (metadata § 3.3)


@pytest.mark.parametrize("pattern", [
    "{nope}", "{title!r}", "{track:03}", "{title:02}", "a/{title}", "{title}:x", "{title", "plain", "",
    "{title}\x01",
])
def test_invalid_rename_patterns_are_usage_errors(pattern: str) -> None:
    with pytest.raises(UsageError):
        NamingOptions(rename=pattern)


@pytest.mark.parametrize("pattern", ["{title}", "x" * 33, "a/b", "{target:02}"])
def test_invalid_suffix_patterns_are_usage_errors(pattern: str) -> None:
    with pytest.raises(UsageError):
        NamingOptions(suffix=pattern)


def test_generated_stems_are_cut_to_200_bytes_on_character_boundaries_with_the_suffix_kept() -> None:
    title = "€" * 100  # three bytes each: 300 bytes
    options = NamingOptions(rename="{title}", suffix=" [{target}]", target_lufs=-16)
    name = audio(plan_output(item("x.wav"), target(), options, plan_revision=0, values={"title": title})).name
    stem = name[: -len(".flac")]
    assert stem.endswith(" [-16]")
    assert len(stem.encode("utf-8")) <= 200
    assert stem.encode("utf-8").decode("utf-8") == stem  # whole characters only
    assert len(stem.encode("utf-8")) == 198  # 194 bytes of room hold 64 whole euro signs


def test_generated_stems_are_nfc() -> None:
    nfd = unicodedata.normalize("NFD", "Perché")
    name = paths.generated_stem(nfd, "")
    assert name == unicodedata.normalize("NFC", "Perché")


def test_plain_stems_keep_native_spelling() -> None:
    nfd = unicodedata.normalize("NFD", "Perché.wav")
    plan = plan_output(item(nfd), target(), NamingOptions(), plan_revision=0)
    assert audio(plan).name == unicodedata.normalize("NFD", "Perché") + ".flac"
    assert plan.audio_member.comparison_key == comparison_key(ROOT / "Normalised" / "Perché.flac")


def test_a_component_over_255_bytes_is_unsafe_output() -> None:
    long = "a" * 250 + ".wav"
    with pytest.raises(FileFailure) as caught:
        plan_output(item(long), target(), NamingOptions(), plan_revision=0,
                    sidecar_requests=[SidecarRequest("tags", "tags_sidecar", None, ".tags.json")])
    assert caught.value.reason == "unsafe_output"


def test_suffix_with_no_room_is_unsafe_output() -> None:
    with pytest.raises(FileFailure):
        paths.generated_stem("x", "y" * 200)


# --- Bundle members (§ 3 "Bundle namespace"; MD36) --------------------------------------------

REQUESTS = (
    SidecarRequest("tags", "tags_sidecar", None, ".tags.json"),
    SidecarRequest("cover", "cover_export", "picture:new", ".cover.jpg"),
    SidecarRequest("picture-2", "picture_export", "picture:2", ".picture-2.png"),
)


def test_bundle_members_append_their_suffix_to_the_complete_audio_name() -> None:
    plan = plan_output(item("x.wav"), target(), NamingOptions(), plan_revision=4, sidecar_requests=REQUESTS)
    folder = ROOT / "Normalised"
    assert [(m.member_id, m.purpose, m.inventory_key, m.final_path) for m in plan.members] == [
        ("audio", "audio", None, folder / "x.flac"),
        ("tags", "tags_sidecar", None, folder / "x.flac.tags.json"),
        ("cover", "cover_export", "picture:new", folder / "x.flac.cover.jpg"),
        ("picture-2", "picture_export", "picture:2", folder / "x.flac.picture-2.png"),
    ]
    assert plan.plan_revision == 4
    assert all(m.comparison_key == comparison_key(m.final_path) for m in plan.members)


@pytest.mark.parametrize("request_", [
    SidecarRequest("audio", "tags_sidecar", None, ".x"),
    SidecarRequest("tags", "cover_export", None, ".x"),
    SidecarRequest("picture-1", "cover_export", "picture:1", ".x"),
    SidecarRequest("tags", "tags_sidecar", None, "/x"),
])
def test_member_ids_purposes_and_suffixes_are_checked(request_: SidecarRequest) -> None:
    with pytest.raises(ValueError):
        plan_output(item("x.wav"), target(), NamingOptions(), plan_revision=0, sidecar_requests=[request_])


def test_rebind_changes_only_the_revision() -> None:
    plan = plan_output(item("x.wav"), target(), NamingOptions(), plan_revision=1, sidecar_requests=REQUESTS)
    again = paths.rebind(plan, 2)
    assert again.plan_revision == 2 and again.members == plan.members


# --- Keys and clash groups ------------------------------------------------------------------


@pytest.mark.parametrize("a, b, same", [
    ("/x/Song.m4a", "/x/song.M4A", True),
    ("/x/" + unicodedata.normalize("NFC", "Perché"), "/x/" + unicodedata.normalize("NFD", "Perché"), True),
    ("/x/Straße.wav", "/x/STRASSE.WAV", True),
    ("/x/Perché", "/x/Perche", False),
    ("/x/a.wav", "/y/a.wav", False),
])
def test_conservative_comparison_key(a: str, b: str, same: bool) -> None:
    assert (comparison_key(a) == comparison_key(b)) is same


def test_case_only_collision_fails_every_participant_and_names_the_other() -> None:
    plans = {0: plan_output(item("Song.wav", 0), target(), NamingOptions(), plan_revision=0),
             1: plan_output(item("song.aiff", 1), target(), NamingOptions(), plan_revision=0),
             2: plan_output(item("other.wav", 2), target(), NamingOptions(), plan_revision=0)}
    clashes = find_clashes(plans)
    assert set(clashes) == {0, 1}
    assert clashes[0][0].others == ((1, "audio"),)
    failure = paths.clash_failure(clashes[1])
    assert failure.reason == "name_clash" and "input 0 (audio)" in (failure.detail or "")


def test_sidecar_of_one_file_against_audio_of_another_clashes() -> None:
    requests = [SidecarRequest("tags", "tags_sidecar", None, ".x.flac")]
    plans = {0: plan_output(item("Song.wav", 0), target(), NamingOptions(), plan_revision=0,
                            sidecar_requests=requests),
             1: plan_output(item("Song.flac.x.wav", 1), target(), NamingOptions(), plan_revision=0)}
    clashes = find_clashes(plans)
    assert [(c.input_index, c.member_id) for i in sorted(clashes) for c in clashes[i]] == [(0, "tags"), (1, "audio")]


def test_rename_collisions_after_cleaning_are_clashes_not_numbered_names() -> None:
    options = NamingOptions(rename="{title}")
    plans = {0: plan_output(item("a.wav", 0), target(), options, plan_revision=0, values={"title": "A/B"}),
             1: plan_output(item("b.wav", 1), target(), options, plan_revision=0, values={"title": "A:B"})}
    assert audio(plans[0]) == audio(plans[1])
    assert set(find_clashes(plans)) == {0, 1}


def test_accented_and_unaccented_stems_do_not_clash() -> None:
    plans = {0: plan_output(item("Perché.wav", 0), target(), NamingOptions(), plan_revision=0),
             1: plan_output(item("Perche.wav", 1), target(), NamingOptions(), plan_revision=0)}
    assert find_clashes(plans) == {}
