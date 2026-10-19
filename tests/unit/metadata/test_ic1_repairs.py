"""IC-1 F-1/F-2 regressions, independently built native evidence (TK-0606)."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys

import pytest
from mutagen.flac import FLAC
from mutagen.mp4 import MP4

from norma.metadata import model
from norma.metadata.read import JSON, RAW, read_source_tags


@pytest.mark.parametrize("value", [
    "2026-00", "2026-01-00", "2026-00-00", "2026-04-31", "2026-02-29",
])
def test_model_zero_calendar_components_invalid(value: str) -> None:
    for parse in (model.canonical_date, model.source_date, model.project_date):
        with pytest.raises(model.InvalidValue, match="real calendar date"):
            parse(value)


@pytest.mark.parametrize("date", ["2026-00-01", "2026-01-00", "2026-00-00", "2026-04-31", "2026-02-29"])
def test_source_timestamps_reject_impossible_calendar_dates(date: str) -> None:
    value = date + "T23:59:59Z"
    assert not model.is_source_timestamp(value)
    assert not model.time_of_day_dropped(value)
    for parse in (model.source_date, model.project_date):
        with pytest.raises(model.InvalidValue):
            parse(value)


@pytest.mark.parametrize("value", ["2024", "2024-02", "2024-02-29", "2024-02-29T23:59:59Z"])
def test_valid_partial_and_leap_dates_still_project(value: str) -> None:
    assert model.source_date(value) == value
    assert model.project_date(value) == value.split("T")[0]


@pytest.mark.parametrize("value", [
    "2026-00", "2026-01-00", "2026-00-00", "2026-04-31", "2026-02-29",
    "2026-00-00T23:59:59Z",
])
def test_read_invalid_source_date_does_not_project(audio, value: str) -> None:
    path = audio("flac")
    tags = FLAC(path)
    tags["DATE"] = value
    tags.save()
    before = path.read_bytes()
    got = read_source_tags(path, "flac")
    assert "date" not in got.fields
    item = next(item for item in got.native_inventory if item.key.casefold() == "date")
    assert item.payload_encoding == RAW and item.payload == value.encode()
    assert path.read_bytes() == before


def test_cli_zero_calendar_source_is_not_printed_as_valid(audio) -> None:
    # Appendix A's child-CLI probe lives here without editing another lane's tests.
    path = audio("flac")
    tags = FLAC(path)
    tags["DATE"] = "2026-00-00"
    tags.save()
    before = path.read_bytes()
    got = subprocess.run([sys.executable, "-B", "-m", "norma", "tags", "show", "--config", "/dev/null", str(path)],
                         capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    assert not any(line.startswith("FIELD\t") and "\tdate\t" in line for line in got.stdout.splitlines())
    assert "store:vorbis:DATE" in got.stdout
    assert path.read_bytes() == before


def atom(name: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", len(body) + 8, name) + body


def data_child(value: bytes, kind: int = 1, version: int = 0) -> bytes:
    return atom(b"data", struct.pack(">II", (version << 24) | kind, 0) + value)


def freeform(value: bytes, kind: int = 1, version: int = 0) -> bytes:
    return atom(b"----", atom(b"mean", bytes(4) + b"com.apple.iTunes")
                + atom(b"name", bytes(4) + b"MOOD") + data_child(value, kind, version))


def replace_ilst(data: bytes, atoms: list[bytes]) -> bytes:
    """Independent byte walk; FFmpeg's moov follows mdat so offsets do not move."""
    count = []

    def walk(data: bytes) -> bytes:
        out = bytearray()
        position = 0
        while position < len(data):
            size, name = struct.unpack(">I4s", data[position:position + 8])
            assert size >= 8 and position + size <= len(data)
            body = data[position + 8:position + size]
            if name == b"ilst":
                body = b"".join(atoms)
                count.append(1)
            elif name in (b"moov", b"udta"):
                body = walk(body)
            elif name == b"meta":
                body = body[:4] + walk(body[4:])
            out.extend(atom(name, body))
            position += size
        return bytes(out)

    names = []
    position = 0
    while position < len(data):
        size, name = struct.unpack(">I4s", data[position:position + 8])
        names.append(name)
        position += size
    assert names.index(b"mdat") < names.index(b"moov")
    result = walk(data)
    assert count == [1]
    return result


def tagged_mp4(audio, atoms: list[bytes]):
    path = audio("alac")
    tags = MP4(path)
    tags["©nam"] = ["seed"]
    tags.save()
    path.write_bytes(replace_ilst(path.read_bytes(), atoms))
    return path


def test_read_repeated_mp4_atoms_keep_independent_occurrences(audio, build) -> None:
    stored = [atom(b"\xa9nam", data_child(value)) for value in (b"A", b"B")]
    path = audio("alac")
    decoded_before = hashlib.sha256(build.fixtures.decode(path).tobytes()).hexdigest()
    tags = MP4(path)
    tags["©nam"] = ["seed"]
    tags.save()
    path.write_bytes(replace_ilst(path.read_bytes(), stored))
    assert MP4(path).tags["©nam"] == ["A", "B"]
    assert hashlib.sha256(build.fixtures.decode(path).tobytes()).hexdigest() == decoded_before
    before = path.read_bytes()
    got = read_source_tags(path, "mp4")
    items = list(got.native_inventory)
    assert [item.identity for item in items] == ["store:mp4:©nam#1", "store:mp4:©nam#2"]
    assert [json.loads(item.payload)["values"] for item in items] == [["A"], ["B"]]
    assert [item.native_type for item in items] == ["©nam", "©nam"]
    assert len({item.locator for item in items}) == 2
    assert all(item.payload_encoding == JSON for item in items)
    assert got.fields["title"].value == "A; B"
    assert path.read_bytes() == before


def test_one_mp4_atom_with_several_data_children_is_one_occurrence(audio) -> None:
    path = tagged_mp4(audio, [atom(b"\xa9nam", data_child(b"A") + data_child(b"B"))])
    got = read_source_tags(path, "mp4")
    assert [item.identity for item in got.native_inventory] == ["store:mp4:©nam"]
    assert json.loads(got.native_inventory[0].payload)["values"] == ["A", "B"]
    assert got.fields["title"].value == "A; B"


def test_repeated_freeforms_keep_values_types_and_versions_per_atom(audio) -> None:
    path = tagged_mp4(audio, [freeform(b"calm"), freeform(b"\x00\xff", kind=0, version=1), freeform(b"bright")])
    got = read_source_tags(path, "mp4")
    items = list(got.native_inventory)
    key = "store:mp4:----:com.apple.iTunes:MOOD"
    assert [item.identity for item in items] == [key + "#1", key + "#2", key + "#3"]
    assert [item.native_type for item in items] == ["----"] * 3
    assert len({item.locator for item in items}) == 3
    values = [json.loads(item.payload)["values"] for item in items]
    assert values == [[{"freeform": "Y2FsbQ==", "dataformat": 1, "version": 0}],
                      [{"freeform": "AP8=", "dataformat": 0, "version": 1}],
                      [{"freeform": "YnJpZ2h0", "dataformat": 1, "version": 0}]]
    assert got.extras["MOOD"].value == "calm; bright"


@pytest.mark.parametrize("name", [b"\xa9nam", b"zzzz"])
def test_repeated_unparsed_atoms_do_not_borrow_another_atoms_value(audio, name: bytes) -> None:
    stored = [atom(name, data_child(b"A")), atom(name, data_child(b"B", kind=2)),
              atom(name, data_child(b"\xff")), atom(name, b"malformed"), atom(name, data_child(b"C"))]
    path = tagged_mp4(audio, stored)
    got = read_source_tags(path, "mp4")
    items = list(got.native_inventory)
    key = "store:mp4:" + name.decode("latin-1")
    assert [item.identity for item in items] == [key + f"#{n}" for n in range(1, 6)]
    assert [item.payload_encoding for item in items] == [JSON, RAW, RAW, RAW, JSON]
    assert [item.payload for item in items[1:4]] == stored[1:4]
    assert [json.loads(items[n].payload)["values"] for n in (0, 4)] == [["A"], ["C"]]
    assert len({item.locator for item in items}) == 5
    if name == b"\xa9nam":
        assert got.fields["title"].value == "A; C"


def test_mp4_text_with_a_non_utf8_data_type_is_an_internal_conflict(audio) -> None:
    # CR-043 (IC-1 recheck R-2): the occurrence is still left out of the joined value, but not silently.
    stored = [atom(b"\xa9nam", data_child(b"A")), atom(b"\xa9nam", data_child("Bé".encode("utf-16-be"), kind=2)),
              atom(b"\xa9nam", data_child(b"C")), atom(b"\xa9ART", data_child(b"Anna")),
              freeform(b"calm"), freeform("ruhig".encode("utf-16-be"), kind=2), freeform(b"bright", kind=0)]
    got = read_source_tags(tagged_mp4(audio, stored), "mp4")
    assert got.fields["title"].value == "A; C" and got.fields["artist"].value == "Anna"
    assert got.extras["MOOD"].value == "calm; bright"  # type 0 stays an accepted freeform type
    conflicts = {conflict.key: conflict for conflict in got.internal_conflicts}
    assert set(conflicts) == {"field:title", "extra:MOOD"}
    title = conflicts["field:title"]
    assert (title.old_value, title.new_value) == ("Bé", "A; C")
    assert title.old_source == "source:store:mp4:©nam#2" and title.detail == "mp4_data_type:2"
    mood = conflicts["extra:MOOD"]
    assert (mood.old_value, mood.new_value, mood.old_source) == (
        "ruhig", "calm; bright", "source:store:mp4:----:com.apple.iTunes:MOOD#2")
    raw = {item.identity: item for item in got.native_inventory}
    assert raw["store:mp4:©nam#2"].payload == stored[1]  # raw evidence unchanged (MD32)


def test_mp4_text_with_a_foreign_type_alone_is_no_conflict(audio) -> None:
    got = read_source_tags(tagged_mp4(audio, [atom(b"\xa9nam", data_child(b"B", kind=2))]), "mp4")
    assert "title" not in got.fields and got.internal_conflicts == ()
    assert [item.identity for item in got.native_inventory] == ["store:mp4:©nam"]


def test_repeated_covr_atoms_keep_each_cover_value_separately(audio) -> None:
    jpeg, png = b"\xff\xd8\xff" + bytes(10), b"\x89PNG\r\n\x1a\n" + bytes(10)
    path = tagged_mp4(audio, [atom(b"covr", data_child(jpeg, 13) + data_child(png, 14)),
                              atom(b"covr", data_child(jpeg + b"B", 13))])
    got = read_source_tags(path, "mp4")
    items = list(got.native_inventory)
    assert [item.identity for item in items] == ["store:mp4:covr#1", "store:mp4:covr#2", "store:mp4:covr#3"]
    assert [json.loads(item.payload)["values"][0]["imageformat"] for item in items] == [13, 14, 13]
    assert all(len(json.loads(item.payload)["values"]) == 1 for item in items)
    assert items[0].locator == items[1].locator and items[2].locator != items[0].locator
    assert [tag.value.data for tag in got.artwork] == [jpeg, png, jpeg + b"B"]
