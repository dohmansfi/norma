"""The hermetic guard's two deliberate rules (CR-041, CR-044; TK-0707).

The guard (tests/conftest.py) is an audit hook, so a refusal is raised *before* the operation runs.
That is what makes it safe to aim the negative cases below at places outside the sandbox: nothing
there is ever touched. The error is an ``AssertionError`` subclass; the tests match on its message.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


# --- CR-041: descriptors are judged by the path they name -----------------------------------


def test_cr041_descriptor_based_rmtree_works_inside_the_sandbox(tmp_path):
    """``shutil.rmtree`` on macOS removes relative to open directory descriptors."""
    assert shutil._use_fd_functions  # the code path CR-041 is about
    tree = tmp_path / "scratch"
    (tree / "a" / "b").mkdir(parents=True)
    (tree / "a" / "b" / "file.bin").write_bytes(b"x")
    (tree / "a" / "other.bin").write_bytes(b"y")
    shutil.rmtree(tree)
    assert not tree.exists()


def test_cr041_a_temporary_directory_cleans_itself_up(tmp_path):
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "inner").mkdir()
        (Path(folder) / "inner" / "f").write_text("x")
    assert not Path(folder).exists() and Path(folder).parent == tmp_path / "home" / "tmp"


def test_cr041_a_name_relative_to_a_descriptor_outside_the_sandbox_is_refused():
    descriptor = os.open("/usr", os.O_RDONLY)
    try:
        for attempt in (
            lambda: os.mkdir("norma-never", dir_fd=descriptor),
            lambda: os.rmdir("norma-never", dir_fd=descriptor),
            lambda: os.unlink("norma-never", dir_fd=descriptor),
            lambda: os.open("norma-never", os.O_WRONLY | os.O_CREAT, dir_fd=descriptor),
        ):
            with pytest.raises(AssertionError, match="outside test temporary directory"):
                attempt()
    finally:
        os.close(descriptor)
    assert not Path("/usr/norma-never").exists()


def test_cr041_the_music_refusal_still_applies_through_a_descriptor(tmp_path):
    home = tmp_path / "home"
    descriptor = os.open(home, os.O_RDONLY)
    try:
        with pytest.raises(AssertionError, match="Music library access refused"):
            os.mkdir("Music", dir_fd=descriptor)
    finally:
        os.close(descriptor)
    with pytest.raises(AssertionError, match="Music library access refused"):
        os.mkdir(home / "Music")
    assert not (home / "Music").exists()


def test_cr041_a_descriptor_that_names_no_path_is_still_refused(tmp_path):
    read_end, write_end = os.pipe()
    try:
        with pytest.raises(AssertionError, match="unassessed file descriptor"):
            os.truncate(write_end, 0)
    finally:
        os.close(read_end)
        os.close(write_end)


# --- CR-044: hard-linked names ----------------------------------------------------------------


def linked_pair(tmp_path: Path) -> tuple[Path, Path]:
    original = tmp_path / "original.bin"
    original.write_bytes(b"payload")
    alias = tmp_path / "alias.bin"
    os.link(original, alias)  # allowed: the source still has a single link when it is made
    assert original.stat().st_nlink == 2
    return original, alias


def test_cr044_a_multiply_linked_name_cannot_be_renamed_or_removed_by_default(tmp_path):
    original, alias = linked_pair(tmp_path)
    with pytest.raises(AssertionError, match="hard-linked write target refused"):
        os.rename(alias, tmp_path / "moved.bin")
    with pytest.raises(AssertionError, match="hard-linked write target refused"):
        os.remove(alias)
    assert alias.exists() and original.read_bytes() == b"payload"


def test_cr044_the_shared_fixture_lifts_only_the_link_count_rule(tmp_path, allow_linked_names):
    original, alias = linked_pair(tmp_path)
    os.rename(alias, tmp_path / "moved.bin")
    os.remove(tmp_path / "moved.bin")
    assert original.read_bytes() == b"payload" and original.stat().st_nlink == 1
    # Containment is untouched: a linked name still cannot be moved out of the sandbox ...
    os.link(original, tmp_path / "again.bin")
    with pytest.raises(AssertionError, match="outside test temporary directory"):
        os.rename(tmp_path / "again.bin", "/usr/norma-never")
    # ... nor may a Music path be reached.
    with pytest.raises(AssertionError, match="Music library access refused"):
        os.rename(tmp_path / "again.bin", tmp_path / "home" / "Music")
    assert not Path("/usr/norma-never").exists()


def test_cr044_the_allowance_ends_with_the_test_that_asked_for_it(tmp_path):
    _, alias = linked_pair(tmp_path)
    with pytest.raises(AssertionError, match="hard-linked write target refused"):
        os.remove(alias)
