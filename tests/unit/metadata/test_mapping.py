"""TST-M-01/M-02: independent tables for every MD23 store/field row."""

from __future__ import annotations

import pytest

from norma.metadata import mapping
from norma.models import Artwork

FIELDS = {"title": "Café", "artist": "Artist", "album_artist": "Various", "album": "Demo",
          "date": "2026-10-05", "track": 3, "track_total": 12, "disc": 1, "disc_total": 2,
          "genre": "Rock", "composer": "Composer", "comment": "First\nsecond", "compilation": False}
KEYS = {
    "mp4": ("©nam", "©ART", "aART", "©alb", "©day", "trkn", "trkn", "disk", "disk", "©gen", "©wrt", "©cmt", "cpil"),
    "vorbis": ("TITLE", "ARTIST", "ALBUMARTIST", "ALBUM", "DATE", "TRACKNUMBER", "TRACKTOTAL", "DISCNUMBER", "DISCTOTAL", "GENRE", "COMPOSER", "COMMENT", "COMPILATION"),
    "id3v2": ("TIT2", "TPE1", "TPE2", "TALB", "TYER", "TRCK", "TRCK", "TPOS", "TPOS", "TCON", "TCOM", "COMM::eng", "TCMP"),
    "riff-info": ("INAM", "IART", None, "IPRD", "ICRD", "IPRT", None, None, None, "IGNR", None, "ICMT", None),
}
TARGETS = [("alac", "info+id3", ("mp4",)), ("aac", "info+id3", ("mp4",)),
           ("flac", "info+id3", ("vorbis",)), ("mp3", "info+id3", ("id3v2",)),
           ("aiff", "info+id3", ("id3v2",)), ("wav", "info+id3", ("id3v2", "riff-info")),
           ("wav", "info", ("riff-info",)), ("wav", "none", ())]


@pytest.mark.parametrize("target,mode,stores", TARGETS)
def test_target_store_priority(target, mode, stores):
    assert mapping.target_stores(target, mode) == stores
    assert mapping.primary_store(target, mode) == (stores[0] if stores else None)


@pytest.mark.parametrize("store", KEYS)
@pytest.mark.parametrize("field", FIELDS)
def test_every_scalar_field_row(store, field):
    key = KEYS[store][list(FIELDS).index(field)]
    keys = mapping.native_keys(store, field, value=FIELDS[field])
    assert keys == (() if key is None else ("TYER", "TDAT") if field == "date" and store == "id3v2" else (key,))
    projected = mapping.project_fields(store, FIELDS)
    assert (field in projected) == (key is not None)
    if key is not None:
        assert projected[field] == FIELDS[field]
        assert field in mapping.canonical_fields(store, key)


@pytest.mark.parametrize("store", KEYS)
def test_native_value_types_versions_and_text_encoding(store):
    recipes = mapping.native_recipes(store, FIELDS, extras={"BPM": "120"}, mark="v=1")
    indexed = {recipe["key"]: recipe for recipe in recipes}
    for recipe in recipes:
        assert recipe["native_version"] == ("2.3" if store == "id3v2" else None)
        assert recipe["text_encoding"] == ("utf-16" if store == "id3v2" else "utf-8")
        with pytest.raises(TypeError):
            recipe["value"] = "changed"
    if store == "mp4":
        assert indexed["cpil"]["value"] is False
        assert indexed["trkn"]["value"] == ((3, 12),)
        assert indexed["disk"]["value"] == ((1, 2),)
        assert indexed["©nam"]["value"] == ("Café",)
        assert indexed["----:com.apple.iTunes:BPM"]["value"] == ("120",)
    elif store == "id3v2":
        assert indexed["TYER"]["value"] == ("2026",)
        assert indexed["TDAT"]["value"] == ("0510",)
        assert indexed["TCMP"]["value"] == ("0",)
        assert indexed["TRCK"]["value"] == ("3/12",)
        assert indexed["COMM::eng"]["value"] == ("First\nsecond",)
    elif store == "vorbis":
        assert indexed["COMPILATION"]["value"] == "0"
        assert indexed["TITLE"]["native_type"] == "VORBIS_COMMENT"
        assert indexed["TRACKTOTAL"]["value"] == "12"
    else:
        assert indexed["INAM"]["value"] == "Café"
        assert all(not k.startswith("TXXX") for k in indexed)


@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("date", ["2026", "2026-10", "2026-10-05", "2026-10-05T07:00:00Z"])
def test_date_versions_and_precision(version, date):
    projected = mapping.project_fields("id3v2", {"date": date}, id3_version=version)
    expected = "2026-10-05" if "T" in date else "2026" if date == "2026-10" and version == "2.3" else date
    assert projected["date"] == expected
    details = mapping.loss_details("id3v2", {"date": date}, id3_version=version)
    assert bool(details) == (date != expected)
    if details:
        assert details["date"] == ("time_of_day_dropped" if "T" in date else "id3_month_dropped")
    recipes = mapping.native_recipes("id3v2", {"date": date}, id3_version=version)
    assert recipes[0]["key"] == ("TYER" if version == "2.3" else "TDRC")
    assert recipes[0]["text_encoding"] == ("utf-16" if version == "2.3" else "utf-8")


@pytest.mark.parametrize("number,total", [("track", "track_total"), ("disc", "disc_total")])
@pytest.mark.parametrize("store", KEYS)
def test_totals_without_number(store, number, total):
    fields = {total: 12}
    projected = mapping.project_fields(store, fields)
    assert number not in projected
    assert (total in projected) == (store in ("mp4", "vorbis"))
    recipes = mapping.native_recipes(store, fields)
    if store == "mp4":
        assert recipes[0]["value"] == ((0, 12),)
    elif store == "vorbis":
        assert recipes[0]["value"] == "12"
    else:
        assert recipes == ()
        assert mapping.loss_details(store, fields)[total] == ("total_without_number" if store == "id3v2" else "no_supported_mapping")


@pytest.mark.parametrize("target,mode,stores", TARGETS)
@pytest.mark.parametrize("field", FIELDS)
def test_all_target_rows_at_primary_store(target, mode, stores, field):
    store = mapping.primary_store(target, mode)
    effective = mapping.project_fields(store, FIELDS)
    expected = store is not None and KEYS[store][list(FIELDS).index(field)] is not None
    assert (field in effective) == expected


def test_wav_secondary_month_does_not_hide_primary_loss():
    assert mapping.project_fields("id3v2", {"date": "2026-10"})["date"] == "2026"
    assert mapping.project_fields("riff-info", {"date": "2026-10"})["date"] == "2026-10"
    assert mapping.loss_details("id3v2", {"date": "2026-10"}) == {"date": "id3_month_dropped"}


@pytest.mark.parametrize("store", KEYS)
@pytest.mark.parametrize("picture_type", [3, 4])
def test_picture_storage_rows(store, picture_type):
    art = Artwork(b"prepared", "image/png", 1, 1, picture_type, "cover")
    pictures = mapping.project_artwork(store, (art,))
    expected = store in ("id3v2", "vorbis") or store == "mp4" and picture_type == 3
    assert pictures == ((art,) if expected else ())
    recipes = mapping.native_recipes(store, {}, pictures=(art,))
    assert len(recipes) == int(expected)
    if recipes:
        assert recipes[0]["value"] is art
        assert recipes[0]["key"] == {"id3v2": "APIC:cover", "vorbis": "PICTURE", "mp4": "covr"}[store]


def test_embedded_unsupported_format_only_requires_mp4_preparation():
    art = Artwork(b"gif", "image/gif", 1, 1, 3, "")
    assert mapping.project_artwork("id3v2", (art,)) == (art,)
    assert mapping.project_artwork("vorbis", (art,)) == (art,)
    with pytest.raises(ValueError, match="MP4 cover"):
        mapping.project_artwork("mp4", (art,))


@pytest.mark.parametrize("store,key", [("id3v2", "TXXX:MOOD"), ("vorbis", "MOOD"), ("mp4", "----:com.apple.iTunes:MOOD"), ("riff-info", None)])
def test_extra_and_mark_rows(store, key):
    assert mapping.extra_key(store, "MOOD") == key
    if key:
        assert mapping.canonical_extra(store, key) == "MOOD"
    mark_key = mapping.extra_key(store, "NORMA", processing_mark=True)
    assert (mark_key is None) == (store == "riff-info")
    if mark_key:
        from norma.metadata.model import is_mark_key
        assert is_mark_key(store, mark_key)


@pytest.mark.parametrize("fields", [{"track": True}, {"compilation": 0}, {"track": 0}, {"date": "Spring 2026"},
                                    {"title": " x "}, {"track": 13, "track_total": 12}, {"unknown": "x"}])
def test_invalid_values_are_not_coerced(fields):
    with pytest.raises(ValueError):
        mapping.native_recipes("id3v2", fields)


@pytest.mark.parametrize("name", ["title", "REPLAYGAIN_TRACK_GAIN", "NORMA", "illegal-name"])
def test_invalid_extra_names(name):
    with pytest.raises(ValueError):
        mapping.extra_key("mp4", name)


def test_bad_store_version_target_and_mode():
    with pytest.raises(ValueError):
        mapping.native_keys("apev2", "title")
    with pytest.raises(ValueError):
        mapping.native_keys("id3v2", "title", id3_version="2.2")
    with pytest.raises(ValueError):
        mapping.target_stores("caf")
    with pytest.raises(ValueError):
        mapping.target_stores("wav", "both")
    assert mapping.project_fields(None, FIELDS) == {}
    assert mapping.project_artwork(None, ()) == ()


def test_native_attributes_and_extra_spelling():
    id3 = {r["key"]: r for r in mapping.native_recipes("id3v2", {"comment": "x"}, extras={"Mood": "calm"})}
    assert id3["COMM::eng"]["attributes"] == {"encoding": 1, "lang": "eng", "desc": ""}
    assert id3["TXXX:Mood"]["attributes"] == {"encoding": 1, "desc": "Mood"}
    vorbis = mapping.native_recipes("vorbis", {}, extras={"Mood": "calm"})
    assert vorbis[0]["key"] == "Mood"
    art = Artwork(b"jpeg", "image/jpeg", 1, 1, 3, "")
    mp4 = {r["key"]: r for r in mapping.native_recipes("mp4", {}, extras={"Mood": "calm"}, pictures=(art,))}
    assert mp4["----:com.apple.iTunes:Mood"]["attributes"] == {"dataformat": 1, "version": 0}
    assert mp4["covr"]["attributes"] == {"imageformat": 13}
    with pytest.raises(ValueError, match="duplicates"):
        mapping.native_recipes("vorbis", {}, extras={"Mood": "a", "MOOD": "b"})


def test_native_recipes_preserve_picture_order_and_multiplicity():
    a = Artwork(b"a", "image/png", 1, 1, 3, "front")
    b = Artwork(b"b", "image/png", 1, 1, 4, "back")
    assert [r["value"] for r in mapping.native_recipes("vorbis", {}, pictures=(a, b, a))] == [a, b, a]


def test_info_passthrough_keys_are_closed_and_not_canonical_extras():
    assert mapping.INFO_PASSTHROUGH == {"ICOP": "copyright", "ILNG": "language", "ITCH": "encoded_by", "ISMP": "timecode"}
    assert mapping.extra_key("riff-info", "ICOP") is None
