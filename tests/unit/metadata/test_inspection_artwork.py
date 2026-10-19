"""Byte-exact inspection artwork, D-101/MD27 (TST-M-01/M-06)."""

import builtins
import hashlib
import io
import os
import struct
import zlib

import pytest

from norma.adapters import sips
from norma.errors import FileFailure
from norma.metadata import artwork, write
from norma.models import Artwork, ManifestCell, RawNativeItem, SourceTags, TagValue


def png(width=1, height=1, *, level=-1):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0\xff\0\0", level)) + chunk(b"IEND", b""))


def test_reference_read_keeps_exact_bytes_and_identity_without_writes(tmp_path, monkeypatch):
    path = tmp_path / "cover.png"
    data = png()
    path.write_bytes(data)
    before = path.stat()
    tool = sips.locate_sips()
    files_before = set(tmp_path.rglob("*"))
    def forbidden(*args, **kwargs):
        pytest.fail("no preparation, conversion or temporary file is authorised")
    monkeypatch.setattr(sips, "convert", forbidden)
    monkeypatch.setattr(artwork.ArtworkPreparer, "prepare", forbidden)
    monkeypatch.setattr(artwork.tempfile, "mkdtemp", forbidden)
    reference = artwork.read_artwork_reference(path, tool)
    assert reference.data == data
    assert reference.stamp.sha256 == hashlib.sha256(data).hexdigest()
    assert reference.stamp.read_dependency().path == path
    artwork.recheck_artwork_reference(reference)
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.read_bytes() == data
    assert set(tmp_path.rglob("*")) == files_before


def tags_with(pictures):
    mark = RawNativeItem("vorbis:NORMA#1", "vorbis", "NORMA", "comment", None, "dn-raw-1", b"v=2", "NORMA")
    return SourceTags({}, {}, tuple(TagValue(Artwork(data, "image/png", None, None, kind, ""),
                                             "vorbis", f"picture:{i}")
                                    for i, (kind, data) in enumerate(pictures)), (), (), (mark,),
                      {"vorbis": (mark.identity,)})


@pytest.mark.parametrize("pictures,matched", [
    ([(3, "exact")], True), ([], False), ([(4, "exact")], False),
    ([(3, "exact"), (4, "other")], True), ([(3, "exact"), (3, "exact")], False),
    ([(3, "exact"), (3, "other")], False), ([(3, "transformed")], False),
])
def test_byte_exact_single_front_cover_only(tmp_path, pictures, matched):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    ref = artwork.read_artwork_reference(path, sips.locate_sips())
    tags = tags_with([(kind, ref.data if value == "exact" else png(level=0))
                      for kind, value in pictures])
    result = write.check_inspection_tags(tags, {"artwork": ManifestCell("set", path)}, artwork_reference=ref)
    assert result.matched is matched
    assert result.mismatches == (() if matched else ("field:artwork",))
    assert [d.code for d in result.diagnostics] == (["processing_mark_invalid"] if matched else
                                                  ["verification_tags", "processing_mark_invalid"])


@pytest.mark.parametrize("pictures", [[], [(3, b"cover")], [(4, b"back")], [(0, b"other"), (4, b"back")]])
def test_artwork_clear_asserts_no_pictures_of_any_type(pictures):
    result = write.check_inspection_tags(tags_with(pictures), {"artwork": ManifestCell("clear", None)})
    assert result.matched is (not pictures)
    assert result.mismatches == (() if not pictures else ("field:artwork",))


@pytest.mark.parametrize("kind", ["missing", "empty", "nonimage", "directory", "truncated", "oversize"])
def test_invalid_references_have_artwork_invalid_class(tmp_path, kind):
    path = tmp_path / "cover"
    if kind == "directory":
        path.mkdir()
    elif kind == "oversize":
        with path.open("wb") as handle:
            handle.truncate(artwork.MAX_BYTES + 1)
    elif kind != "missing":
        path.write_bytes({"empty": b"", "nonimage": b"not an image", "truncated": png()[:-12]}[kind])
    with pytest.raises(FileFailure) as failure:
        artwork.read_artwork_reference(path, sips.locate_sips())
    assert failure.value.reason == "artwork_invalid"
    assert failure.value.stage == "content_verified"


def test_unreadable_reference_is_artwork_invalid_not_permission_denied(tmp_path, monkeypatch):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    tool = sips.locate_sips()
    real_open = builtins.open
    def denied(file, *args, **kwargs):
        if file == path:
            raise PermissionError("injected unreadable image")
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", denied)
    with pytest.raises(FileFailure) as failure:
        artwork.read_artwork_reference(path, tool)
    assert failure.value.reason == "artwork_invalid"


@pytest.mark.parametrize("side,accepted", [(16384, True), (16385, False)])
@pytest.mark.parametrize("dimension", ["width", "height"])
def test_pixel_limits_without_conversion(tmp_path, monkeypatch, side, accepted, dimension):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    tool = sips.locate_sips()
    # Deterministic adapter boundary: do not allocate giant decoded images.
    monkeypatch.setattr(sips, "inspect", lambda *a, **kw: sips.ImageInfo(
        "png", side if dimension == "width" else 1, side if dimension == "height" else 1, "RGB"))
    if accepted:
        assert artwork.read_artwork_reference(path, tool).data == png()
    else:
        with pytest.raises(FileFailure) as failure:
            artwork.read_artwork_reference(path, tool)
        assert failure.value.reason == "artwork_invalid"


@pytest.mark.parametrize("change", ["replace", "content", "mtime", "delete", "symlink"])
def test_protected_reference_recheck_reports_source_changed(tmp_path, change):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    ref = artwork.read_artwork_reference(path, sips.locate_sips())
    if change in ("replace", "symlink"):
        other = tmp_path / "other.png"
        other.write_bytes(ref.data)
        path.unlink()
        if change == "replace":
            other.rename(path)
        else:
            path.symlink_to(other)
    elif change == "content":
        path.write_bytes(bytes([ref.data[0] ^ 1]) + ref.data[1:])
        os.utime(path, ns=(path.stat().st_atime_ns, ref.stamp.mtime_ns))
    elif change == "mtime":
        os.utime(path, ns=(path.stat().st_atime_ns, ref.stamp.mtime_ns + 1000000))
    else:
        path.unlink()
    with pytest.raises(FileFailure) as failure:
        artwork.recheck_artwork_reference(ref)
    assert failure.value.reason == "source_changed"
    assert failure.value.stage == "content_verified"


def test_change_during_sips_inspection_is_source_changed(tmp_path, monkeypatch):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    tool = sips.locate_sips()
    def inspect(*args, **kwargs):
        other = tmp_path / "replacement.png"
        other.write_bytes(png())
        other.replace(path)
        return sips.ImageInfo("png", 1, 1, "RGB")
    monkeypatch.setattr(sips, "inspect", inspect)
    with pytest.raises(FileFailure) as failure:
        artwork.read_artwork_reference(path, tool)
    assert failure.value.reason == "source_changed"


def test_no_write_opens_even_for_reference_inspection(tmp_path, monkeypatch):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    tool = sips.locate_sips()
    def guarded(original):
        def open_(file, mode="r", *args, **kwargs):
            assert not any(flag in mode for flag in "wax+"), "write open in inspection"
            return original(file, mode, *args, **kwargs)
        return open_
    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))
    ref = artwork.read_artwork_reference(path, tool)
    assert write.check_inspection_tags(tags_with([(3, ref.data)]),
        {"artwork": ManifestCell("set", path)}, artwork_reference=ref).matched


def test_unvalidated_or_wrong_artwork_reference_is_programming_error(tmp_path):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    ref = artwork.read_artwork_reference(path, sips.locate_sips())
    tags = tags_with([(3, ref.data)])
    with pytest.raises(ValueError, match="validated reference"):
        write.check_inspection_tags(tags, {"artwork": ManifestCell("set", path)})
    with pytest.raises(ValueError, match="validated reference"):
        write.check_inspection_tags(tags, {"artwork": ManifestCell("set", tmp_path / "wrong")},
                                    artwork_reference=ref)


def test_initial_reference_removed_between_stat_and_open_is_source_changed(tmp_path, monkeypatch):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    tool = sips.locate_sips()
    real_open = builtins.open
    def removed(file, *args, **kwargs):
        if file == path:
            path.unlink()
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", removed)
    with pytest.raises(FileFailure) as failure:
        artwork.read_artwork_reference(path, tool)
    assert failure.value.reason == "source_changed"


def test_exact_byte_limit_is_accepted_and_larger_file_never_inspected(tmp_path, monkeypatch):
    path = tmp_path / "cover.png"
    path.write_bytes(png())
    with path.open("ab") as handle:
        handle.truncate(artwork.MAX_BYTES)
    tool = sips.locate_sips()
    calls = []
    def inspect(*args, **kwargs):
        calls.append(args)
        return sips.ImageInfo("png", 1, 1, "RGB")
    monkeypatch.setattr(sips, "inspect", inspect)
    ref = artwork.read_artwork_reference(path, tool)
    assert len(ref.data) == artwork.MAX_BYTES
    assert len(calls) == 1
    with path.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(FileFailure) as failure:
        artwork.read_artwork_reference(path, tool)
    assert failure.value.reason == "artwork_invalid"
    assert len(calls) == 1
