"""metadata/artwork.py sources, ranks and plans (TK-0609, S4; MD24, MD11, MD12; TST-M-06).

Explicit-source ordering, ``--artwork`` and folder candidate discovery, and the
``ArtworkPlan`` builders for supplied, cleared and embedded pictures.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from norma.adapters import sips
from norma.errors import EXIT_USAGE, FileFailure, UsageError
from norma.metadata import artwork
from norma.models import Artwork, ArtworkPlan, ManifestCell

_spec = importlib.util.spec_from_file_location("artwork_images", Path(__file__).with_name("artwork_images.py"))
assert _spec is not None and _spec.loader is not None
images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(images)

needs_sips = pytest.mark.skipif(not images.HAVE_SIPS, reason="macOS sips not available")


def cell_set(path: str) -> ManifestCell:
    return ManifestCell("set", Path(path))


CLEAR = ManifestCell("clear", None)


# --- --artwork ------------------------------------------------------------------------------------


def test_missing_explicit_artwork_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="does not exist") as caught:
        artwork.check_cli_artwork("nope.png", cwd=tmp_path)
    assert caught.value.exit_code == EXIT_USAGE == 2


def test_a_directory_is_not_artwork(tmp_path: Path) -> None:
    (tmp_path / "cover.png").mkdir()
    with pytest.raises(UsageError):
        artwork.check_cli_artwork("cover.png", cwd=tmp_path)


def test_relative_artwork_is_taken_from_the_given_working_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.png").write_bytes(b"x")
    assert artwork.check_cli_artwork("sub/c.png", cwd=tmp_path) == tmp_path / "sub" / "c.png"


def test_relative_artwork_defaults_to_the_process_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "c.png").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert artwork.check_cli_artwork("c.png") == Path(os.path.abspath(tmp_path / "c.png"))


def test_the_content_of_an_existing_artwork_is_not_checked_here(tmp_path: Path) -> None:
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    assert artwork.check_cli_artwork(tmp_path / "bad.jpg") == tmp_path / "bad.jpg"


# --- --folder-artwork ----------------------------------------------------------------------------


def test_folder_candidates_are_tried_in_the_documented_order(tmp_path: Path) -> None:
    for name in ("folder.jpg", "cover.png", "cover.jpg", "other.jpg"):
        (tmp_path / name).write_bytes(b"x")
    assert artwork.find_folder_artwork(tmp_path) == tmp_path / "cover.jpg"
    (tmp_path / "cover.jpg").unlink()
    assert artwork.find_folder_artwork(tmp_path) == tmp_path / "cover.png"
    (tmp_path / "cover.png").unlink()
    assert artwork.find_folder_artwork(tmp_path) == tmp_path / "folder.jpg"
    (tmp_path / "folder.jpg").unlink()
    assert artwork.find_folder_artwork(tmp_path) is None


@pytest.mark.parametrize("name", ["COVER.JPG", "Cover.Png", "FOLDER.jpg"])
def test_folder_names_match_case_insensitively(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"x")
    assert artwork.find_folder_artwork(tmp_path) == tmp_path / name


@pytest.mark.parametrize("name", ["cover.jpeg", "front.jpg", "cover.gif", "album.png", "cover.jpg.bak", ".cover.jpg"])
def test_other_names_are_no_opinion(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"x")
    assert artwork.find_folder_artwork(tmp_path) is None


def test_a_folder_without_candidates_or_a_missing_folder_is_no_opinion(tmp_path: Path) -> None:
    assert artwork.find_folder_artwork(tmp_path) is None
    assert artwork.find_folder_artwork(tmp_path / "missing") is None


def test_a_directory_named_like_a_candidate_is_not_one(tmp_path: Path) -> None:
    (tmp_path / "cover.jpg").mkdir()
    (tmp_path / "cover.png").write_bytes(b"x")
    assert artwork.find_folder_artwork(tmp_path) == tmp_path / "cover.png"


def test_only_the_files_own_folder_is_searched(tmp_path: Path) -> None:
    (tmp_path / "cover.jpg").write_bytes(b"x")
    (tmp_path / "album").mkdir()
    assert artwork.find_folder_artwork(tmp_path / "album") is None


# --- ranks --------------------------------------------------------------------------------------


def explicit(**kwargs: object) -> artwork.ExplicitArtwork | None:
    return artwork.explicit_artwork(**kwargs)  # type: ignore[arg-type]


def test_nothing_given_is_no_opinion() -> None:
    assert explicit() is None


def test_remove_artwork_is_a_clear_at_the_command_line_rank(tmp_path: Path) -> None:
    result = explicit(remove=True, row_cell=cell_set("c.png"), manifest_dir=tmp_path, folder_path=tmp_path / "cover.jpg")
    assert result == artwork.ExplicitArtwork("clear", None, "cli")


def test_the_command_line_image_beats_the_manifest_and_the_folder(tmp_path: Path) -> None:
    result = explicit(cli_path=tmp_path / "cli.png", row_cell=cell_set("m.png"), default_cell=cell_set("d.png"),
                      manifest_dir=tmp_path, folder_path=tmp_path / "cover.jpg")
    assert result == artwork.ExplicitArtwork("set", tmp_path / "cli.png", "cli")


def test_the_manifest_row_beats_the_defaults_and_the_folder(tmp_path: Path) -> None:
    result = explicit(row_cell=cell_set("row.png"), default_cell=cell_set("d.png"), manifest_dir=tmp_path,
                      folder_path=tmp_path / "cover.jpg")
    assert result == artwork.ExplicitArtwork("set", tmp_path / "row.png", "manifest")


def test_a_row_clear_beats_a_defaults_image(tmp_path: Path) -> None:
    assert explicit(row_cell=CLEAR, default_cell=cell_set("d.png"), manifest_dir=tmp_path) == \
        artwork.ExplicitArtwork("clear", None, "manifest")


def test_a_defaults_clear_beats_the_folder(tmp_path: Path) -> None:
    assert explicit(default_cell=CLEAR, folder_path=tmp_path / "cover.jpg") == \
        artwork.ExplicitArtwork("clear", None, "manifest_default")


def test_the_defaults_image_beats_the_folder(tmp_path: Path) -> None:
    result = explicit(default_cell=cell_set("d.png"), manifest_dir=tmp_path, folder_path=tmp_path / "cover.jpg")
    assert result == artwork.ExplicitArtwork("set", tmp_path / "d.png", "manifest_default")


@pytest.mark.parametrize("provenance", ["config", "cli"])
def test_the_folder_image_is_rank_e4_with_its_flags_own_provenance(tmp_path: Path, provenance: str) -> None:
    result = explicit(folder_path=tmp_path / "cover.jpg", folder_provenance=provenance)
    assert result == artwork.ExplicitArtwork("set", tmp_path / "cover.jpg", provenance)  # type: ignore[arg-type]


def test_manifest_paths_are_relative_to_the_manifest_folder_and_normalised(tmp_path: Path) -> None:
    result = explicit(row_cell=cell_set("art/../covers/a.png"), manifest_dir=tmp_path / "manifests")
    assert result is not None and result.path == tmp_path / "manifests" / "covers" / "a.png"


def test_an_absolute_manifest_path_is_kept(tmp_path: Path) -> None:
    result = explicit(row_cell=cell_set(str(tmp_path / "abs.png")))
    assert result is not None and result.path == tmp_path / "abs.png"


def test_a_relative_manifest_path_needs_its_folder() -> None:
    with pytest.raises(ValueError, match="manifest folder"):
        explicit(row_cell=cell_set("a.png"))


def test_a_manifest_artwork_cell_must_hold_a_path() -> None:
    with pytest.raises(ValueError, match="path"):
        explicit(row_cell=ManifestCell("set", "text"), manifest_dir=Path("/x"))


# --- plans: supplied, cleared, none ---------------------------------------------------------------


@pytest.fixture
def preparer(tmp_path: Path) -> artwork.ArtworkPreparer:
    if not images.HAVE_SIPS:
        pytest.skip("macOS sips not available")
    work = tmp_path / "work"
    work.mkdir()
    return artwork.ArtworkPreparer(sips.locate_sips("/usr/bin/sips"), work)


def png(tmp_path: Path, name: str = "cover.png", width: int = 64, height: int = 64) -> Path:
    path = tmp_path / name
    path.write_bytes(images.png_bytes(width, height))
    return path


def pic(data: bytes, mime: str, kind: int, description: str = "") -> Artwork:
    return Artwork(data, mime, None, None, kind, description)


def test_no_opinion_has_no_plan(preparer: artwork.ArtworkPreparer) -> None:
    assert artwork.plan_explicit(preparer, None) == artwork.ArtworkResult(None)


def test_clear_is_an_explicit_plan_without_pictures(preparer: artwork.ArtworkPreparer) -> None:
    result = artwork.plan_explicit(preparer, artwork.ExplicitArtwork("clear", None, "manifest"),
                                   source_pictures=(pic(b"x", "image/png", 3),))
    assert result.plan == ArtworkPlan((), "manifest", None) and result.stamps == () and result.notes == ()
    assert artwork.clear_plan("cli") == ArtworkPlan((), "cli", None)


@needs_sips
def test_a_supplied_file_becomes_one_front_cover(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = png(tmp_path)
    result = artwork.plan_explicit(preparer, artwork.ExplicitArtwork("set", path, "cli"))
    assert result.plan is not None
    assert result.plan.pictures == (Artwork(path.read_bytes(), "image/png", 64, 64, 3, ""),)
    assert (result.plan.provenance, result.plan.source_locator) == ("cli", str(path))
    assert result.plan.diagnostics == ()
    assert [s.path for s in result.stamps] == [Path(os.path.realpath(path))]


@needs_sips
def test_the_plan_carries_the_non_square_note(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    result = artwork.plan_explicit(preparer, artwork.ExplicitArtwork("set", png(tmp_path, width=64, height=32), "cli"))
    assert result.notes == ("artwork is not square (64x32)",) and result.plan is not None and result.plan.diagnostics == ()


@needs_sips
def test_supplied_cover_replaces_source_front_covers_and_keeps_other_pictures(tmp_path: Path,
                                                                              preparer: artwork.ArtworkPreparer) -> None:
    source = (pic(b"front", "image/png", 3), pic(b"back", "image/jpeg", 4, "back"), pic(b"disc", "image/png", 6))
    result = artwork.supplied_plan(preparer, png(tmp_path), "manifest", source_pictures=source)
    assert result.plan is not None
    assert [(p.picture_type, p.data) for p in result.plan.pictures][1:] == [(4, b"back"), (6, b"disc")]
    assert result.plan.pictures[0].picture_type == 3 and result.plan.pictures[0].data != b"front"


@needs_sips
def test_failures_propagate_for_each_file(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image\n")
    for _ in range(2):
        with pytest.raises(FileFailure) as caught:
            artwork.plan_explicit(preparer, artwork.ExplicitArtwork("set", bad, "manifest"))
        assert caught.value.reason == "artwork_invalid"


@needs_sips
def test_recheck_result_covers_every_stamp(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = png(tmp_path)
    result = artwork.plan_explicit(preparer, artwork.ExplicitArtwork("set", path, "cli"))
    artwork.recheck_result(preparer, result)
    path.write_bytes(images.png_bytes(65, 65))
    with pytest.raises(FileFailure) as caught:
        artwork.recheck_result(preparer, result)
    assert caught.value.reason == "source_changed"
    artwork.recheck_result(preparer, artwork.ArtworkResult(None))


@needs_sips
def test_a_batch_shares_one_preparation(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = png(tmp_path)
    plans = [artwork.plan_explicit(preparer, artwork.ExplicitArtwork("set", path, "manifest")).plan for _ in range(5)]
    assert preparer.preparations == 1 and all(p == plans[0] for p in plans)


# --- plans: embedded -----------------------------------------------------------------------------


def test_no_embedded_pictures_is_no_opinion(preparer: artwork.ArtworkPreparer) -> None:
    assert artwork.embedded_plan(preparer, (), target_mp4=True) == artwork.ArtworkResult(None)


@needs_sips
def test_embedded_pictures_keep_their_bytes_and_order_for_non_mp4(tmp_path: Path,
                                                                  preparer: artwork.ArtworkPreparer) -> None:
    gif = images.convert(png(tmp_path), tmp_path / "c.gif", "-s", "format", "gif")
    pictures = (pic(gif.read_bytes(), "image/gif", 3, "front"), pic(b"\xff\xd8\xff-jpeg", "image/jpeg", 4),
                pic(png(tmp_path).read_bytes(), "image/png", 0))
    result = artwork.embedded_plan(preparer, pictures, target_mp4=False)
    assert result.plan == ArtworkPlan(pictures, "source", "embedded") and result.stamps == ()
    assert preparer.preparations == 0


@needs_sips
def test_embedded_jpeg_and_png_front_covers_are_not_re_encoded_for_mp4(tmp_path: Path,
                                                                       preparer: artwork.ArtworkPreparer) -> None:
    large = images.convert(png(tmp_path), tmp_path / "l.jpg", "-s", "format", "jpeg", "-z", 4000, 4000)
    pictures = (pic(large.read_bytes(), "image/jpeg", 3), pic(png(tmp_path).read_bytes(), "image/png", 4))
    result = artwork.embedded_plan(preparer, pictures, target_mp4=True)
    assert result.plan is not None and result.plan.pictures == pictures  # not scaled either: bytes are kept
    assert preparer.preparations == 0


@needs_sips
def test_an_embedded_gif_front_cover_is_converted_for_mp4_and_keeps_type_and_description(
        tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    gif = images.convert(png(tmp_path, width=64, height=32), tmp_path / "c.gif", "-s", "format", "gif")
    source = pic(gif.read_bytes(), "image/gif", 3, "front")
    result = artwork.embedded_plan(preparer, (source,), target_mp4=True)
    assert result.plan is not None and result.plan.provenance == "source"
    (converted,) = result.plan.pictures
    assert converted.data != source.data and converted.data.startswith(b"\xff\xd8\xff")
    assert (converted.mime, converted.width_px, converted.height_px) == ("image/jpeg", 64, 32)
    assert (converted.picture_type, converted.description) == (3, "front")


@needs_sips
def test_embedded_conversion_scales_like_a_supplied_file(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    big = images.convert(png(tmp_path), tmp_path / "b.bmp", "-s", "format", "bmp", "-z", 3500, 3500)
    (converted,) = artwork.embedded_plan(preparer, (pic(big.read_bytes(), "image/bmp", 3),), target_mp4=True).plan.pictures  # type: ignore[union-attr]
    assert (converted.width_px, converted.height_px) == (3000, 3000)


@needs_sips
def test_non_front_embedded_pictures_are_left_alone_for_mp4(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    back = pic(b"GIF89a-not-storable", "image/gif", 4)
    result = artwork.embedded_plan(preparer, (back,), target_mp4=True)
    assert result.plan is not None and result.plan.pictures == (back,) and preparer.preparations == 0


@needs_sips
def test_an_unreadable_embedded_front_cover_is_artwork_invalid_for_mp4(preparer: artwork.ArtworkPreparer) -> None:
    for data in (b"", b"garbage that is not an image"):
        with pytest.raises(FileFailure) as caught:
            artwork.embedded_plan(preparer, (pic(data, "image/gif", 3),), target_mp4=True)
        assert caught.value.reason == "artwork_invalid" and caught.value.stage == artwork.STAGE


def test_oversize_embedded_front_cover_is_refused_before_sips(tmp_path: Path) -> None:
    work = tmp_path / "w"
    work.mkdir()
    never = images.fake_with(tmp_path, f"touch '{tmp_path / 'ran'}'\nexit 1\n", "never")
    preparer = artwork.ArtworkPreparer(sips.locate_sips(never), work)
    with pytest.raises(FileFailure, match="too large to decode"):
        artwork.embedded_plan(preparer, (pic(b"G" * (artwork.MAX_BYTES + 1), "image/gif", 3),), target_mp4=True)
    assert not (tmp_path / "ran").exists()


@pytest.mark.parametrize("data, expected", [
    (b"\xff\xd8\xff\xe0rest", "image/jpeg"), (b"\x89PNG\r\n\x1a\nrest", "image/png"),
    (b"GIF89a", None), (b"RIFFxxxxWEBP", None), (b"", None), (b"\xff\xd8", None),
])
def test_signature_sniffing(data: bytes, expected: str | None) -> None:
    assert artwork.sniff_storable(data) == expected


# --- constants and layering -----------------------------------------------------------------------


def test_fixed_limits_and_names() -> None:
    assert (artwork.MAX_BYTES, artwork.MAX_SIDE_PX, artwork.JPEG_QUALITY) == (50 * 1024 * 1024, 16384, 90)
    assert artwork.FOLDER_NAMES == ("cover.jpg", "cover.png", "folder.jpg")


def test_artwork_imports_only_what_the_layering_allows() -> None:
    import ast

    tree = ast.parse(Path(artwork.__file__).read_text(encoding="utf-8"))
    roots = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    roots |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {"subprocess", "mutagen", "socket", "urllib"} & {r.split(".")[0] for r in roots}
    assert {r for r in roots if r.startswith("norma")} <= {"norma.adapters", "norma.errors", "norma.models"}


def test_importing_artwork_runs_no_process_and_loads_no_tag_library() -> None:
    code = ("import sys\nimport norma.metadata.artwork\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'mutagen'))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
