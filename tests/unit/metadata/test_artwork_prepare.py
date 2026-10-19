"""metadata/artwork.py preparation: formats, colour, scaling, limits, failure classes, cache, recheck (TK-0609, S4).

TST-M-06, F-16. Real sips builds and checks images; fakes cover timeouts, a lost
executable and sips lying about its output. Everything lives in ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

import pytest

from norma.adapters import sips
from norma import errors
from norma.errors import FileFailure
from norma.metadata import artwork
from norma.models import Artwork

_spec = importlib.util.spec_from_file_location("artwork_images", Path(__file__).with_name("artwork_images.py"))
assert _spec is not None and _spec.loader is not None
images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(images)

pytestmark = pytest.mark.skipif(not images.HAVE_SIPS, reason="macOS sips not available")


@pytest.fixture
def tool() -> sips.Sips:
    return sips.locate_sips("/usr/bin/sips")


@pytest.fixture
def work(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def preparer(tool: sips.Sips, work: Path) -> artwork.ArtworkPreparer:
    return artwork.ArtworkPreparer(tool, work)


def write_png(tmp_path: Path, name: str = "cover.png", width: int = 64, height: int = 64, colour: int = 2,
              pixel: tuple[int, ...] = (40, 120, 200)) -> Path:
    path = tmp_path / name
    path.write_bytes(images.png_bytes(width, height, colour, pixel))
    return path


def info(tool: sips.Sips, data: bytes, tmp_path: Path) -> sips.ImageInfo:
    probe = tmp_path / "probe-copy"
    probe.write_bytes(data)
    try:
        return sips.inspect(tool, probe)
    finally:
        probe.unlink()


def jpeg_of(source: Path, dest: Path, *extra: object) -> Path:
    return images.convert(source, dest, *extra, "-s", "format", "jpeg", "-s", "formatOptions", 90)


def failure_of(preparer: artwork.ArtworkPreparer, path: Path) -> FileFailure:
    with pytest.raises(FileFailure) as caught:
        preparer.prepare(path)
    return caught.value


# --- kept as supplied ---------------------------------------------------------------------------


def test_rgb_png_keeps_its_exact_bytes(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path)
    result = preparer.prepare(path)
    assert result.artwork == Artwork(path.read_bytes(), "image/png", 64, 64, 3, "")
    assert not result.converted and not result.scaled and result.notes == ()


def test_grey_png_is_kept(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path, "grey.png", 32, 32, 0, (100,))
    result = preparer.prepare(path)
    assert result.artwork.data == path.read_bytes() and result.artwork.mime == "image/png" and not result.converted


def test_rgba_png_is_kept_as_png(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path, "alpha.png", 32, 32, 6, (40, 120, 200, 128))
    assert preparer.prepare(path).artwork.data == path.read_bytes()


def test_rgb_jpeg_is_kept(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = jpeg_of(write_png(tmp_path), tmp_path / "cover.jpg")
    result = preparer.prepare(path)
    assert result.artwork.data == path.read_bytes() and result.artwork.mime == "image/jpeg" and not result.converted


def test_grey_jpeg_is_kept(tmp_path: Path, tool: sips.Sips, preparer: artwork.ArtworkPreparer) -> None:
    path = images.convert(write_png(tmp_path), tmp_path / "grey.jpg", "-m", images.GRAY_PROFILE, "-s", "format", "jpeg")
    assert sips.inspect(tool, path).space == "Gray"
    result = preparer.prepare(path)
    assert result.artwork.data == path.read_bytes() and not result.converted


def test_a_misleading_extension_does_not_decide_the_format(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = jpeg_of(write_png(tmp_path), tmp_path / "actually.jpg")
    liar = tmp_path / "liar.png"
    liar.write_bytes(path.read_bytes())
    assert preparer.prepare(liar).artwork.mime == "image/jpeg"


def test_picture_type_and_description_are_applied_without_touching_the_cache(tmp_path: Path,
                                                                              preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path)
    back = preparer.prepare(path, picture_type=4, description="back")
    front = preparer.prepare(path)
    assert (back.artwork.picture_type, back.artwork.description) == (4, "back")
    assert (front.artwork.picture_type, front.artwork.description) == (3, "")
    assert preparer.preparations == 1


# --- converted to RGB JPEG quality 90 -------------------------------------------------------------


@pytest.mark.parametrize("cut", [33, 60, -12, -1])
def test_truncated_png_is_not_kept_even_if_sips_accepts_it(tmp_path, preparer, cut):
    path = write_png(tmp_path, width=64, height=32)
    path.write_bytes(path.read_bytes()[:cut])
    failure = failure_of(preparer, path)
    assert failure.reason == "artwork_invalid" and failure.stage == "metadata_prepared"
    assert not list(preparer.workdir.iterdir())


def test_png_bad_crc_is_not_kept(tmp_path, preparer):
    path = write_png(tmp_path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1  # IEND CRC, without changing the inspected image properties
    path.write_bytes(data)
    failure = failure_of(preparer, path)
    assert failure.reason == "artwork_invalid" and "CRC" in failure.detail


def test_truncated_jpeg_is_not_kept(tmp_path, preparer):
    path = jpeg_of(write_png(tmp_path), tmp_path / "cover.jpg")
    data = path.read_bytes()
    assert data.endswith(b"\xff\xd9")
    path.write_bytes(data[:-2])
    failure = failure_of(preparer, path)
    assert failure.reason == "artwork_invalid" and "EOI" in failure.detail


def test_jpeg_eoi_inside_an_app_segment_is_not_the_end(tmp_path, preparer):
    path = jpeg_of(write_png(tmp_path), tmp_path / "cover.jpg")
    data = path.read_bytes()
    # A complete APP15 segment containing EOI does not finish a truncated scan.
    path.write_bytes(data[:2] + b"\xff\xef\x00\x04\xff\xd9" + data[2:-2])
    failure = failure_of(preparer, path)
    assert failure.reason == "artwork_invalid"


def test_embedded_truncated_picture_is_still_kept_as_source_evidence(preparer):
    picture = Artwork(images.png_bytes(64, 32)[:60], "image/png", 64, 32, 3, "")
    assert preparer.convert_embedded(picture) is picture


@pytest.mark.parametrize("fmt", ["gif", "bmp", "tiff", "heic"])
def test_other_formats_become_rgb_jpeg(tmp_path: Path, tool: sips.Sips, preparer: artwork.ArtworkPreparer,
                                       fmt: str) -> None:
    source = images.convert(write_png(tmp_path), tmp_path / f"cover.{fmt}", "-s", "format", fmt)
    result = preparer.prepare(source)
    assert result.converted and not result.scaled
    assert info(tool, result.artwork.data, tmp_path) == sips.ImageInfo("jpeg", 64, 64, "RGB")
    assert (result.artwork.mime, result.artwork.width_px, result.artwork.height_px) == ("image/jpeg", 64, 64)


def test_cmyk_jpeg_becomes_rgb_jpeg(tmp_path: Path, tool: sips.Sips, preparer: artwork.ArtworkPreparer) -> None:
    source = images.convert(write_png(tmp_path), tmp_path / "cmyk.jpg", "-m", images.CMYK_PROFILE, "-s", "format",
                            "jpeg", "-s", "formatOptions", 90)
    assert sips.inspect(tool, source).space == "CMYK"
    result = preparer.prepare(source)
    assert result.converted and result.artwork.mime == "image/jpeg"
    assert info(tool, result.artwork.data, tmp_path).space == "RGB"


def test_grey_gif_or_tiff_is_converted_to_rgb(tmp_path: Path, tool: sips.Sips, preparer: artwork.ArtworkPreparer) -> None:
    grey = write_png(tmp_path, "g.png", 32, 32, 0, (100,))
    source = images.convert(grey, tmp_path / "g.tiff", "-s", "format", "tiff")
    result = preparer.prepare(source)
    assert info(tool, result.artwork.data, tmp_path) == sips.ImageInfo("jpeg", 32, 32, "RGB")


def test_conversion_quality_is_90(tmp_path: Path, tool: sips.Sips, preparer: artwork.ArtworkPreparer) -> None:
    source = images.convert(write_png(tmp_path, "c.png", 128, 128), tmp_path / "c.bmp", "-s", "format", "bmp")
    result = preparer.prepare(source)
    reference = images.convert(source, tmp_path / "ref.jpg", "-s", "format", "jpeg", "-s", "formatOptions", 90)
    assert result.artwork.data == reference.read_bytes()


# --- scaling --------------------------------------------------------------------------------------


def test_large_jpeg_is_scaled_to_the_default_3000(tmp_path: Path, tool: sips.Sips,
                                                  preparer: artwork.ArtworkPreparer) -> None:
    large = jpeg_of(write_png(tmp_path), tmp_path / "large.jpg", "-z", 6000, 6000)
    result = preparer.prepare(large)
    assert result.scaled and not result.converted
    assert (result.artwork.width_px, result.artwork.height_px) == (3000, 3000)
    assert info(tool, result.artwork.data, tmp_path) == sips.ImageInfo("jpeg", 3000, 3000, "RGB")


def test_scaled_png_stays_png(tmp_path: Path, tool: sips.Sips, work: Path) -> None:
    path = write_png(tmp_path, "big.png", 800, 400)
    result = artwork.ArtworkPreparer(tool, work, max_size=300).prepare(path)
    assert result.scaled and not result.converted and result.artwork.mime == "image/png"
    assert (result.artwork.width_px, result.artwork.height_px) == (300, 150)
    assert info(tool, result.artwork.data, tmp_path).format == "png"


def test_scaled_grey_png_stays_grey_png(tmp_path: Path, tool: sips.Sips, work: Path) -> None:
    path = write_png(tmp_path, "grey.png", 600, 600, 0, (90,))
    result = artwork.ArtworkPreparer(tool, work, max_size=300).prepare(path)
    assert info(tool, result.artwork.data, tmp_path) == sips.ImageInfo("png", 300, 300, "Gray")


@pytest.mark.parametrize("size, longest", [(301, 300), (300, None), (299, None)])
def test_the_scale_boundary(tmp_path: Path, tool: sips.Sips, work: Path, size: int, longest: int | None) -> None:
    path = write_png(tmp_path, "edge.png", size, 10)
    result = artwork.ArtworkPreparer(tool, work, max_size=300).prepare(path)
    assert result.scaled is (longest is not None)
    assert max(result.artwork.width_px or 0, result.artwork.height_px or 0) == (longest or size)
    if longest is None:
        assert result.artwork.data == path.read_bytes()


def test_a_small_image_is_never_enlarged(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path, "tiny.png", 8, 8)
    result = preparer.prepare(path)
    assert (result.artwork.width_px, result.artwork.height_px) == (8, 8) and result.artwork.data == path.read_bytes()


@pytest.mark.parametrize("width, height", [(1000, 3001), (3001, 1000), (777, 4001), (4000, 4000)])
def test_aspect_ratio_is_kept_within_one_pixel(tmp_path: Path, tool: sips.Sips, preparer: artwork.ArtworkPreparer,
                                               width: int, height: int) -> None:
    path = write_png(tmp_path, "r.png", width, height)
    art = preparer.prepare(path).artwork
    factor = 3000 / max(width, height)
    assert max(art.width_px or 0, art.height_px or 0) == 3000
    assert abs((art.width_px or 0) - width * factor) <= 1 and abs((art.height_px or 0) - height * factor) <= 1
    assert info(tool, art.data, tmp_path).width_px == art.width_px


def test_conversion_and_scaling_combine(tmp_path: Path, tool: sips.Sips, work: Path) -> None:
    source = images.convert(write_png(tmp_path, "w.png", 640, 320), tmp_path / "w.bmp", "-s", "format", "bmp")
    result = artwork.ArtworkPreparer(tool, work, max_size=320).prepare(source)
    assert result.converted and result.scaled
    assert info(tool, result.artwork.data, tmp_path) == sips.ImageInfo("jpeg", 320, 160, "RGB")


def test_non_square_gets_a_note_and_square_does_not(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    assert preparer.prepare(write_png(tmp_path, "wide.png", 64, 32)).notes == ("artwork is not square (64x32)",)
    assert preparer.prepare(write_png(tmp_path, "sq.png", 32, 32)).notes == ()


def test_the_note_names_the_source_size_even_when_scaled(tmp_path: Path, tool: sips.Sips, work: Path) -> None:
    result = artwork.ArtworkPreparer(tool, work, max_size=300).prepare(write_png(tmp_path, "w.png", 1000, 500))
    assert result.notes == ("artwork is not square (1000x500)",)


# --- limits --------------------------------------------------------------------------------------


def test_exactly_the_byte_limit_is_accepted_if_decodable(tmp_path: Path, tool: sips.Sips, work: Path) -> None:
    path = write_png(tmp_path, "limit.png", 32, 32)
    images.pad_to(path, artwork.MAX_BYTES)
    assert path.stat().st_size == 52_428_800
    result = artwork.ArtworkPreparer(tool, work).prepare(path)
    assert result.artwork.mime == "image/png" and len(result.artwork.data) == 52_428_800


def test_one_byte_over_the_limit_is_refused_before_sips_runs(tmp_path: Path, work: Path) -> None:
    path = write_png(tmp_path, "over.png", 32, 32)
    images.pad_to(path, artwork.MAX_BYTES + 1)
    never = images.fake_with(tmp_path, f"touch '{tmp_path / 'ran'}'; exit 1\n", "never-sips")
    preparer = artwork.ArtworkPreparer(sips.locate_sips(never), work)
    failure = failure_of(preparer, path)
    assert failure.reason == "artwork_invalid" and "too large to decode" in (failure.detail or "")
    assert not (tmp_path / "ran").exists() and preparer.preparations == 0


def test_exactly_16384_pixels_is_accepted(tmp_path: Path, tool: sips.Sips, work: Path) -> None:
    path = write_png(tmp_path, "wide.png", 16384, 16)
    result = artwork.ArtworkPreparer(tool, work, max_size=10000).prepare(path)
    assert result.scaled and result.artwork.width_px == 10000


def test_16385_pixels_is_refused(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    for name, (width, height) in {"wide.png": (16385, 2), "tall.png": (2, 16385)}.items():
        failure = failure_of(preparer, write_png(tmp_path, name, width, height))
        assert failure.reason == "artwork_invalid" and "too large to decode" in (failure.detail or "")


# --- failure classes ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, data", [
    ("corrupt.jpg", b"not an image\n"),
    ("empty.png", b""),
    ("plain", b"plain text, no extension\n"),
    ("truncated.png", images.png_bytes(64, 64)[:40]),
], ids=["text-as-jpg", "empty", "untyped-text", "truncated-png"])
def test_non_images_are_artwork_invalid(tmp_path: Path, preparer: artwork.ArtworkPreparer, name: str,
                                        data: bytes) -> None:
    path = tmp_path / name
    path.write_bytes(data)
    failure = failure_of(preparer, path)
    assert (failure.reason, failure.stage) == ("artwork_invalid", artwork.STAGE) and name in (failure.detail or "")


def test_a_crashing_sips_gives_a_short_detail(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = tmp_path / "plain"
    path.write_bytes(b"plain\n")
    assert len(failure_of(preparer, path).detail or "") < 600


def test_missing_directory_and_unreadable_are_artwork_invalid(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    assert "does not exist" in (failure_of(preparer, tmp_path / "missing.png").detail or "")
    (tmp_path / "dir.png").mkdir()
    assert "not a regular file" in (failure_of(preparer, tmp_path / "dir.png").detail or "")
    locked = write_png(tmp_path, "locked.png")
    locked.chmod(0)
    try:
        assert failure_of(preparer, locked).reason == "artwork_invalid"
    finally:
        locked.chmod(0o600)


def test_a_dangling_symlink_is_artwork_invalid(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    (tmp_path / "link.png").symlink_to(tmp_path / "nowhere.png")
    assert failure_of(preparer, tmp_path / "link.png").reason == "artwork_invalid"


def test_f16_recipes(tmp_path: Path, tool: sips.Sips, preparer: artwork.ArtworkPreparer) -> None:
    """TESTING § 4.7: HEIC, large and CMYK variants end as RGB JPEG at most 3000 px; corrupt is invalid."""
    cover = write_png(tmp_path)
    heic = images.convert(cover, tmp_path / "cover.heic", "-s", "format", "heic", "-s", "formatOptions", 90)
    large = jpeg_of(cover, tmp_path / "large.jpg", "-z", 6000, 6000)
    cmyk = images.convert(cover, tmp_path / "cmyk.jpg", "-m", images.CMYK_PROFILE, "-s", "format", "jpeg",
                          "-s", "formatOptions", 90)
    for path, side in ((heic, 64), (large, 3000), (cmyk, 64)):
        art = preparer.prepare(path).artwork
        assert info(tool, art.data, tmp_path) == sips.ImageInfo("jpeg", side, side, "RGB")
    assert preparer.prepare(cover).artwork.data == cover.read_bytes()
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image\n")
    assert failure_of(preparer, corrupt).reason == "artwork_invalid"


# --- tool failures -------------------------------------------------------------------------------


def test_sips_timeout_is_a_timeout_failure_and_is_not_cached(tmp_path: Path, work: Path) -> None:
    slow = sips.locate_sips(images.fake_with(tmp_path, "sleep 60\n"))
    preparer = artwork.ArtworkPreparer(slow, work, deadline_s=0.3)
    path = write_png(tmp_path)
    for expected in (1, 2):
        failure = failure_of(preparer, path)
        assert (failure.reason, failure.stage) == ("timeout", artwork.STAGE)
        assert preparer.preparations == expected


def test_the_default_deadline_is_the_adr_0018_image_inspection_bound(tool: sips.Sips, work: Path) -> None:
    assert artwork.ArtworkPreparer(tool, work).deadline_s == 60.0 == sips.INSPECT_DEADLINE_S


def test_sips_lost_after_preflight_is_a_per_file_failure(tmp_path: Path, work: Path) -> None:
    fake = images.fake_with(tmp_path, "exit 0\n")
    preparer = artwork.ArtworkPreparer(sips.locate_sips(fake), work)
    fake.unlink()
    failure = failure_of(preparer, write_png(tmp_path))
    assert failure.reason == "artwork_invalid" and "no longer available" in (failure.detail or "")


def test_sips_lost_after_preflight_uses_the_tool_loss_reason_for_the_stage(tmp_path: Path, work: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """CR-086: the reason is ``errors.tool_loss_reason(STAGE)``, not a literal in artwork.py."""
    asked: list[str] = []

    def reason(stage: str) -> str:
        asked.append(stage)
        return "decode_error"  # a value no literal in artwork.py would produce

    monkeypatch.setattr(artwork, "tool_loss_reason", reason)
    fake = images.fake_with(tmp_path, "exit 0\n")
    preparer = artwork.ArtworkPreparer(sips.locate_sips(fake), work)
    fake.unlink()
    failure = failure_of(preparer, write_png(tmp_path))
    assert asked == ["metadata_prepared"] and failure.reason == "decode_error"
    assert failure.stage == "metadata_prepared" and "no longer available" in (failure.detail or "")


def test_the_real_tool_loss_mapping_for_the_artwork_stage_is_artwork_invalid() -> None:
    assert errors.tool_loss_reason(artwork.STAGE) == "artwork_invalid"


def test_a_lost_sips_is_not_reported_as_a_timeout_or_an_ordinary_tool_failure(tmp_path: Path) -> None:
    failure = artwork._tool_failure(tmp_path / "c.png", errors.ToolMissing("sips vanished"))
    assert (failure.reason, failure.stage) == ("artwork_invalid", "metadata_prepared")
    failure = artwork._tool_failure(tmp_path / "c.png", errors.ToolTimeout("slow", deadline_s=60.0))
    assert failure.reason == "timeout"


def test_sips_producing_the_wrong_size_is_refused(tmp_path: Path, work: Path) -> None:
    # Every inspection says 4000x4000, so a 3000 scale is expected; the "converted" file also reports 4000.
    script = """
        for last; do :; done
        case "$*" in *--out*) printf x > "$last";; esac
        printf '  format: png\\n  pixelWidth: 4000\\n  pixelHeight: 4000\\n  space: RGB\\n'
    """
    preparer = artwork.ArtworkPreparer(sips.locate_sips(images.fake_with(tmp_path, script)), work)
    failure = failure_of(preparer, write_png(tmp_path))
    assert failure.reason == "artwork_invalid" and "expected about 3000x3000" in (failure.detail or "")


def test_sips_producing_the_wrong_format_is_refused(tmp_path: Path, work: Path) -> None:
    script = f"""
        case "$*" in
          *--out*) for last; do :; done; printf x > "$last"
                   printf '  format: png\\n  pixelWidth: 64\\n  pixelHeight: 64\\n  space: RGB\\n'; exit 0;;
        esac
        printf '  format: gif\\n  pixelWidth: 64\\n  pixelHeight: 64\\n  space: RGB\\n'
    """
    fake = images.fake_with(tmp_path, script)
    preparer = artwork.ArtworkPreparer(sips.locate_sips(fake), work)
    failure = failure_of(preparer, write_png(tmp_path))
    assert failure.reason == "artwork_invalid"


def test_scratch_copies_are_removed(tmp_path: Path, work: Path, tool: sips.Sips) -> None:
    preparer = artwork.ArtworkPreparer(tool, work, max_size=300)
    preparer.prepare(write_png(tmp_path, "a.png", 600, 600))
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope\n")
    failure_of(preparer, bad)
    assert list(work.iterdir()) == []


def test_sips_works_on_a_private_copy_not_the_supplied_path(tmp_path: Path, work: Path) -> None:
    log = tmp_path / "args.log"
    fake = images.fake_with(tmp_path, f'echo "$@" >> \'{log}\'\nexit 1\n')
    preparer = artwork.ArtworkPreparer(sips.locate_sips(fake), work)
    path = write_png(tmp_path)
    failure_of(preparer, path)
    assert str(path) not in log.read_text() and str(work) in log.read_text()


def test_constructor_refuses_bad_arguments(tmp_path: Path, tool: sips.Sips) -> None:
    with pytest.raises(ValueError):
        artwork.ArtworkPreparer(tool, tmp_path / "missing")
    with pytest.raises(ValueError):
        artwork.ArtworkPreparer(tool, tmp_path, max_size=0)


# --- identity: cache and recheck ---------------------------------------------------------------


def test_one_preparation_per_distinct_identity(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path)
    (tmp_path / "hard.png").hardlink_to(path)
    (tmp_path / "soft.png").symlink_to(path)
    results = [preparer.prepare(tmp_path / name) for name in ("cover.png", "hard.png", "soft.png", "cover.png")]
    assert preparer.preparations == 1 and len({r.artwork.data for r in results}) == 1
    preparer.prepare(write_png(tmp_path, "other.png", 32, 32))
    assert preparer.preparations == 2


def test_a_copy_with_equal_bytes_is_a_distinct_identity(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path)
    copy = tmp_path / "copy.png"
    copy.write_bytes(path.read_bytes())
    preparer.prepare(path)
    preparer.prepare(copy)
    assert preparer.preparations == 2


def test_the_cache_is_per_max_size(tmp_path: Path, tool: sips.Sips, work: Path) -> None:
    path = write_png(tmp_path, "m.png", 600, 600)
    small = artwork.ArtworkPreparer(tool, work, max_size=300)
    large = artwork.ArtworkPreparer(tool, work, max_size=500)
    assert small.prepare(path).artwork.width_px == 300 and large.prepare(path).artwork.width_px == 500


def test_an_invalid_file_is_judged_once_and_fails_every_time(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image\n")
    first, second = failure_of(preparer, bad), failure_of(preparer, bad)
    assert preparer.preparations == 1 and first is not second
    assert (first.reason, first.detail) == (second.reason, second.detail)


def test_concurrent_requests_prepare_once(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path, "t.png", 400, 400)
    seen: list[bytes] = []
    errors: list[BaseException] = []

    def work_item() -> None:
        try:
            seen.append(preparer.prepare(path).artwork.data)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work_item) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(seen) == 8 and preparer.preparations == 1


def test_stamp_records_identity_size_mtime_and_hash(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    import hashlib

    path = write_png(tmp_path)
    stamp = preparer.prepare(path).stamp
    assert stamp is not None
    info_ = os.stat(path)
    assert (stamp.path, stamp.identity, stamp.size, stamp.mtime_ns) == (
        Path(os.path.realpath(path)), (info_.st_dev, info_.st_ino), info_.st_size, info_.st_mtime_ns)
    assert stamp.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    dependency = stamp.read_dependency()
    assert (dependency.path, dependency.identity, dependency.purpose) == (stamp.path, stamp.identity, "artwork")


def test_recheck_accepts_an_unchanged_file(tmp_path: Path, preparer: artwork.ArtworkPreparer) -> None:
    preparer.recheck(preparer.prepare(write_png(tmp_path)).stamp)
    preparer.recheck(None)


def test_recheck_detects_same_size_content_change_with_restored_mtime(tmp_path: Path,
                                                                      preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path)
    prepared = preparer.prepare(path)
    before = os.stat(path)
    altered = bytearray(path.read_bytes())
    altered[-20] ^= 0xFF
    path.write_bytes(bytes(altered))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(FileFailure) as caught:
        preparer.recheck(prepared.stamp)
    assert caught.value.reason == "source_changed"
    with pytest.raises(FileFailure) as again:  # a cache hit is a use too
        preparer.prepare(path)
    assert again.value.reason == "source_changed"


@pytest.mark.parametrize("change", ["append", "touch", "replace", "delete"])
def test_recheck_detects_every_other_change(tmp_path: Path, preparer: artwork.ArtworkPreparer, change: str) -> None:
    path = write_png(tmp_path)
    prepared = preparer.prepare(path)
    if change == "append":
        with open(path, "ab") as handle:
            handle.write(b"\0")
    elif change == "touch":
        os.utime(path, ns=(1, 1))
    elif change == "replace":
        other = tmp_path / "other.png"
        other.write_bytes(path.read_bytes())
        os.utime(other, ns=(prepared.stamp.mtime_ns, prepared.stamp.mtime_ns))  # type: ignore[union-attr]
        other.replace(path)
    else:
        path.unlink()
    with pytest.raises(FileFailure) as caught:
        preparer.recheck(prepared.stamp)
    assert (caught.value.reason, caught.value.stage) == ("source_changed", artwork.STAGE)


def test_a_file_changed_after_preparation_is_prepared_afresh_next_time(tmp_path: Path,
                                                                       preparer: artwork.ArtworkPreparer) -> None:
    path = write_png(tmp_path, "c.png", 32, 32)
    first = preparer.prepare(path).artwork
    path.write_bytes(images.png_bytes(48, 48))
    assert preparer.prepare(path).artwork != first and preparer.preparations == 2
