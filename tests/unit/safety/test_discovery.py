"""TST-S-01 — discovery: expansion, filtering, ordering, links, roots, identity (safety § 2.1-2.4).

Every tree is built in tmp_path from empty or hand-written files; no fixture catalogue is used.
"""

from __future__ import annotations

import os
import stat
import unicodedata
from pathlib import Path

import pytest

from norma import discovery
from norma.discovery import EXCLUSION_KINDS, discover, natural_key
from norma.errors import UsageError


def rel(result: discovery.DiscoveryResult) -> list[str]:
    return [item.relative_path.as_posix() for item in result.items]


def snapshot(root: Path) -> list[tuple[str, int, int]]:
    rows = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        rows.append((str(path), info.st_mode, info.st_mtime_ns))
    return rows


# --- Ordering (§ 2.4) ----------------------------------------------------------------


def test_numeric_runs_sort_numerically() -> None:
    names = ["10.wav", "2.wav", "1.wav", "a10.wav", "a2.wav", "a.wav"]
    assert sorted(names, key=natural_key) == ["1.wav", "2.wav", "10.wav", "a2.wav", "a10.wav", "a.wav"]


def test_case_and_normalisation_are_folded_before_bytes_break_ties() -> None:
    nfc, nfd = unicodedata.normalize("NFC", "Café.wav"), unicodedata.normalize("NFD", "Café.wav")
    assert nfc != nfd and natural_key(nfc)[0] == natural_key(nfd)[0]
    assert sorted([nfd, nfc], key=natural_key) == sorted([nfc, nfd], key=natural_key)
    upper, lower = "TAKE.wav", "take.wav"
    assert natural_key(upper)[0] == natural_key(lower)[0]
    assert sorted([lower, upper], key=natural_key) == [upper, lower]  # UTF-8 bytes: "T" before "t"
    assert natural_key("Straße.wav")[0] == natural_key("STRASSE.wav")[0]


def test_equal_numeric_value_ties_fall_to_bytes() -> None:
    assert natural_key("02.wav")[0] == natural_key("2.wav")[0]
    assert sorted(["2.wav", "02.wav"], key=natural_key) == ["02.wav", "2.wav"]


def test_folder_order_is_depth_first_in_natural_order(tree: Path, put) -> None:
    for name in ("10.wav", "2.wav", "b/1.wav", "b/10.wav", "b/2.wav", "a.wav", "B2.wav"):
        put(tree / name)
    result = discover([tree], recursive=True)
    # The folder "b" sorts as the text run "b", which precedes "B2.wav" ("b", 2, ".wav").
    assert rel(result) == ["2.wav", "10.wav", "a.wav", "b/1.wav", "b/2.wav", "b/10.wav", "B2.wav"]
    assert [item.input_index for item in result.items] == list(range(7))


def test_real_case_and_normalisation_clashes_keep_both_in_stable_order(tree: Path, put) -> None:
    nfc, nfd = unicodedata.normalize("NFC", "é.wav"), unicodedata.normalize("NFD", "é.wav")
    put(tree / nfc)
    put(tree / nfd)
    if len(list(tree.iterdir())) < 2:
        pytest.skip("this volume treats NFC and NFD spellings as one name")
    first = rel(discover([tree]))
    assert len(first) == 2 and first == rel(discover([tree]))
    assert first == sorted([nfc, nfd], key=lambda n: os.fsencode(n))  # tie on folded key → bytes


# --- Arguments, filters and counts (§ 2.1, § 2.2) --------------------------------------


def test_explicit_file_bypasses_extension_filter_and_uses_its_parent(tree: Path, put) -> None:
    note = put(tree / "notes.txt", b"hello")
    (item,) = discover([note]).items
    assert item.explicit and item.root == tree.resolve() and item.relative_path.as_posix() == "notes.txt"
    assert item.source_path == note.resolve()


def test_folder_expansion_filters_extensions_case_insensitively(tree: Path, put) -> None:
    for name in ("a.WAV", "b.Aiff", "c.flac", "d.mp3", "e.M4A", "f.aac", "g.mp4", "h.caf", "i.aif"):
        put(tree / name)
    for name in ("x.txt", "y.m4p", "z", "w.wav.bak", "v.ogg"):
        put(tree / name)
    result = discover([tree])
    assert sorted(rel(result)) == sorted(["a.WAV", "b.Aiff", "c.flac", "d.mp3", "e.M4A", "f.aac", "g.mp4",
                                          "h.caf", "i.aif"])
    assert result.excluded["extension"] == 5
    assert set(result.excluded) == set(EXCLUSION_KINDS)
    assert all(not item.explicit for item in result.items)


def test_folders_are_one_level_deep_unless_recursive(tree: Path, put) -> None:
    put(tree / "top.wav")
    put(tree / "sub" / "deep" / "inner.wav")
    assert rel(discover([tree])) == ["top.wav"]
    assert rel(discover([tree], recursive=True)) == ["sub/deep/inner.wav", "top.wav"]


def test_hidden_and_appledouble_entries_are_excluded_and_counted(tree: Path, put) -> None:
    put(tree / "keep.wav")
    put(tree / ".hidden.wav")
    put(tree / "._keep.wav")
    put(tree / ".DS_Store")
    put(tree / ".secret" / "x.wav")
    result = discover([tree], recursive=True)
    assert rel(result) == ["keep.wav"]
    assert result.excluded["hidden"] == 4  # the hidden folder counts once and is not entered


def test_explicit_hidden_file_is_kept(tree: Path, put) -> None:
    hidden = put(tree / ".odd.wav")
    assert rel(discover([hidden])) == [".odd.wav"]


def test_generated_folders_are_not_traversed(tree: Path, put) -> None:
    put(tree / "song.wav")
    put(tree / "Normalised" / "song.wav")
    put(tree / "album" / "Normalised" / "deep.wav")
    put(tree / ".norma-originals" / "2026-10-09" / "old.wav")
    put(tree / "other" / "normalised" / "case.wav")  # the folder name is compared case-folded
    result = discover([tree], recursive=True)
    assert rel(result) == ["song.wav"]
    assert result.excluded["generated"] == 3 and result.excluded["hidden"] == 1


def test_explicit_normalised_folder_is_inspected_at_top_level_only(tree: Path, put) -> None:
    put(tree / "Normalised" / "top.wav")
    put(tree / "Normalised" / "Normalised" / "nested.wav")
    result = discover([tree / "Normalised"], recursive=True)
    assert rel(result) == ["top.wav"]
    assert result.excluded["generated"] == 1


def test_configured_output_subtree_is_excluded(tree: Path, put) -> None:
    put(tree / "a.wav")
    put(tree / "out" / "b.wav")
    put(tree / "out" / "deeper" / "c.wav")
    result = discover([tree], recursive=True, output_dir=tree / "out")
    assert rel(result) == ["a.wav"]
    assert result.excluded["generated"] == 1
    # a link to a file inside the output subtree is excluded too
    (tree / "alias.wav").symlink_to(tree / "out" / "b.wav")
    assert rel(discover([tree], output_dir=tree / "out")) == ["a.wav"]


def test_active_suffix_at_end_of_stem_is_excluded(tree: Path, put) -> None:
    for name in ("song.wav", "song_norm.wav", "norm_song.wav", "SONG_NORM.flac", "song_norm.2.wav"):
        put(tree / name)
    result = discover([tree], suffix="_norm")
    assert sorted(rel(result)) == ["norm_song.wav", "song.wav", "song_norm.2.wav"]
    assert result.excluded["suffix"] == 2
    assert len(discover([tree], suffix="").items) == 5


def test_explicit_file_bypasses_convenience_exclusions(tree: Path, put) -> None:
    gen = put(tree / "Normalised" / "x.wav")
    tagged = put(tree / "song_norm.wav")
    assert len(discover([gen, tagged], suffix="_norm").items) == 2


def test_special_files_are_counted_and_never_opened(tree: Path, put) -> None:
    put(tree / "ok.wav")
    os.mkfifo(tree / "pipe.wav")
    result = discover([tree])
    assert rel(result) == ["ok.wav"] and result.excluded["special"] == 1


# --- Links and roots (§ 2.3) -----------------------------------------------------------


def test_file_symlink_is_resolved_to_its_canonical_target(tree: Path, put) -> None:
    real = put(tree / "real" / "take.wav")
    (tree / "link.wav").symlink_to(real)
    result = discover([tree], recursive=True)
    # "link.wav" sorts first; the target is the same file as real/take.wav
    (first,) = result.items
    assert first.source_path == real.resolve() and first.root == tree.resolve()
    assert first.relative_path.as_posix() == "real/take.wav"
    assert result.duplicates == 1 and first.root / first.relative_path == first.source_path


def test_escaping_file_link_uses_target_parent_as_root(tmp_path: Path, tree: Path, put) -> None:
    outside = put(tmp_path / "elsewhere" / "far.wav")
    (tree / "near.wav").symlink_to(outside)
    (item,) = discover([tree]).items
    assert item.source_path == outside.resolve()
    assert item.root == outside.resolve().parent and item.relative_path.as_posix() == "far.wav"
    assert ".." not in item.relative_path.parts


def test_directory_links_inside_a_folder_are_not_followed(tmp_path: Path, tree: Path, put) -> None:
    put(tmp_path / "other" / "x.wav")
    (tree / "dirlink").symlink_to(tmp_path / "other", target_is_directory=True)
    (tree / "loop").symlink_to(tree, target_is_directory=True)
    put(tree / "a.wav")
    result = discover([tree], recursive=True)
    assert rel(result) == ["a.wav"] and result.directory_links == 2 and result.hints


def test_explicit_directory_link_becomes_the_canonical_root(tmp_path: Path, put) -> None:
    real = tmp_path / "real"
    put(real / "a.wav")
    link = tmp_path / "shortcut"
    link.symlink_to(real, target_is_directory=True)
    (item,) = discover([link]).items
    assert item.root == real.resolve() and item.source_path == (real / "a.wav").resolve()


def test_unresolved_and_looping_links_in_a_folder_are_ignored_with_a_count(tree: Path, put) -> None:
    put(tree / "ok.wav")
    (tree / "dangling.wav").symlink_to(tree / "missing.wav")
    (tree / "self.wav").symlink_to(tree / "self.wav")
    result = discover([tree])
    assert rel(result) == ["ok.wav"] and result.ignored_links == 2
    assert any("unresolved" in hint for hint in result.hints)


def test_hard_links_and_repeats_are_deduplicated_by_identity(tree: Path, put) -> None:
    first = put(tree / "a.wav")
    os.link(first, tree / "b.wav")
    put(tree / "c.wav")
    result = discover([tree, first, tree])
    assert rel(result) == ["a.wav", "c.wav"] and result.duplicates == 5
    assert [item.input_index for item in result.items] == [0, 1]
    assert result.items[0].identity == (first.stat().st_dev, first.stat().st_ino)


def test_overlap_keeps_the_first_root_and_argument_order(tree: Path, put) -> None:
    inner = put(tree / "sub" / "x.wav")
    put(tree / "y.wav")
    explicit_first = discover([inner, tree], recursive=True)
    assert [(i.explicit, i.root, i.relative_path.as_posix()) for i in explicit_first.items] == [
        (True, inner.resolve().parent, "x.wav"), (False, tree.resolve(), "y.wav")]
    folder_first = discover([tree, inner], recursive=True)
    assert [(i.explicit, i.root) for i in folder_first.items] == [(False, tree.resolve())] * 2


def test_multiple_roots_keep_argument_order(tmp_path: Path, put) -> None:
    put(tmp_path / "r1" / "z.wav")
    put(tmp_path / "r2" / "a.wav")
    result = discover([tmp_path / "r1", tmp_path / "r2"])
    assert [(i.root.name, i.relative_path.as_posix()) for i in result.items] == [("r1", "z.wav"), ("r2", "a.wav")]


def test_item_invariants_hold_for_every_kind_of_input(tmp_path: Path, tree: Path, put) -> None:
    put(tree / "a" / "b.wav")
    outside = put(tmp_path / "out" / "o.wav")
    (tree / "esc.wav").symlink_to(outside)
    for item in discover([tree, outside], recursive=True).items:
        assert item.source_path.is_absolute() and item.root.is_absolute()
        assert item.root / item.relative_path == item.source_path
        info = item.source_path.stat()
        assert item.identity == (info.st_dev, info.st_ino) and stat.S_ISREG(info.st_mode)


# --- Explicit-argument failures (§ 2.1) ----------------------------------------------


def test_bad_explicit_arguments_raise_one_usage_error_naming_all(tmp_path: Path, tree: Path, put) -> None:
    put(tree / "ok.wav")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(UsageError) as caught:
        discover([tree, tmp_path / "missing.wav", tmp_path / "dangling", tmp_path / "loop", tmp_path / "fifo"])
    message = str(caught.value)
    for fragment in ("missing.wav", "dangling", "loop", "fifo", "broken symbolic link", "not a regular file"):
        assert fragment in message


def test_unreadable_explicit_folder_is_a_usage_error(tree: Path, put) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can read everything")
    locked = tree / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        with pytest.raises(UsageError):
            discover([locked])
    finally:
        locked.chmod(0o700)


def test_unreadable_nested_folder_is_reported_and_others_are_still_found(tree: Path, put) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can read everything")
    put(tree / "a.wav")
    locked = tree / "locked"
    locked.mkdir()
    put(tree / "z" / "b.wav")
    locked.chmod(0)
    try:
        result = discover([tree], recursive=True)
    finally:
        locked.chmod(0o700)
    assert rel(result) == ["a.wav", "z/b.wav"] and result.unreadable == (tree.resolve() / "locked",)


def test_empty_arguments_and_empty_folder_discover_nothing(tree: Path) -> None:
    assert discover([]).items == () and discover([tree]).items == ()


# --- Read-only (§ 2.7) -----------------------------------------------------------------


def test_discovery_writes_nothing_and_never_touches_contents(tmp_path: Path, tree: Path, put) -> None:
    put(tree / "a.wav", b"RIFF")
    put(tree / "sub" / "b.flac", b"fLaC")
    os.link(tree / "a.wav", tree / "hard.wav")
    before = snapshot(tmp_path)
    contents = {p: p.read_bytes() for p in tree.rglob("*") if p.is_file()}
    discover([tree], recursive=True)
    assert snapshot(tmp_path) == before
    assert {p: p.read_bytes() for p in tree.rglob("*") if p.is_file()} == contents
