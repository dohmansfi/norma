"""Pure canonical/native mappings (MD20, MD23, MD29, MD38; formats M2–M6).

Recipes are immutable implementation-local mappings, not native evidence or
new contract records. The write edge encodes them using MD32 and performs
mutagen's version conversion on passthrough copies. This module never parses
raw evidence, reads files or imports mutagen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from norma.metadata import model
from norma.models import Artwork, FrozenMap, MetadataValue

_KEYS = FrozenMap({
    "mp4": {"title": "©nam", "artist": "©ART", "album_artist": "aART", "album": "©alb",
            "date": "©day", "track": "trkn", "track_total": "trkn", "disc": "disk",
            "disc_total": "disk", "genre": "©gen", "composer": "©wrt", "comment": "©cmt",
            "artwork": "covr", "compilation": "cpil"},
    "vorbis": {"title": "TITLE", "artist": "ARTIST", "album_artist": "ALBUMARTIST", "album": "ALBUM",
               "date": "DATE", "track": "TRACKNUMBER", "track_total": "TRACKTOTAL", "disc": "DISCNUMBER",
               "disc_total": "DISCTOTAL", "genre": "GENRE", "composer": "COMPOSER", "comment": "COMMENT",
               "artwork": "PICTURE", "compilation": "COMPILATION"},
    "id3v2": {"title": "TIT2", "artist": "TPE1", "album_artist": "TPE2", "album": "TALB",
              "date": "TYER", "track": "TRCK", "track_total": "TRCK", "disc": "TPOS",
              "disc_total": "TPOS", "genre": "TCON", "composer": "TCOM", "comment": "COMM::eng",
              "artwork": "APIC", "compilation": "TCMP"},
    "riff-info": {"title": "INAM", "artist": "IART", "album": "IPRD", "date": "ICRD",
                  "track": "IPRT", "genre": "IGNR", "comment": "ICMT"},
})
INFO_PASSTHROUGH = FrozenMap({"ICOP": "copyright", "ILNG": "language", "ITCH": "encoded_by", "ISMP": "timecode"})


def _validate(store: str, id3_version: str) -> None:
    if store not in _KEYS:
        raise ValueError(f"not a target store: {store!r}")
    if id3_version not in ("2.3", "2.4"):
        raise ValueError(f"not a target ID3 version: {id3_version!r}")


def target_stores(target: str, wav_tags: str = "info+id3") -> tuple[str, ...]:
    """Primary first; target is a C-04 format name, not a source container."""
    if wav_tags not in ("info+id3", "info", "none"):
        raise ValueError(f"unknown WAV mode: {wav_tags!r}")
    if target == "wav":
        return {"info+id3": ("id3v2", "riff-info"), "info": ("riff-info",), "none": ()}[wav_tags]
    if target in ("alac", "aac"):
        return ("mp4",)
    if target == "flac":
        return ("vorbis",)
    if target in ("mp3", "aiff"):
        return ("id3v2",)
    raise ValueError(f"unknown target format: {target!r}")


def primary_store(target: str, wav_tags: str = "info+id3") -> str | None:
    stores = target_stores(target, wav_tags)
    return stores[0] if stores else None


def native_keys(store: str, field: str, *, id3_version: str = "2.3", value: MetadataValue | None = None) -> tuple[str, ...]:
    """Keys carrying a field; a full v2.3 date uses TYER and TDAT."""
    _validate(store, id3_version)
    if field not in model.CANONICAL_FIELDS:
        raise ValueError(f"not a canonical field: {field!r}")
    key = _KEYS[store].get(field)
    if key is None:
        return ()
    if field == "date" and store == "id3v2":
        if id3_version == "2.4":
            return ("TDRC",)
        if isinstance(value, str) and len(model.project_date(value)) == 10:
            return ("TYER", "TDAT")
    return (key,)


def extra_key(store: str, name: str, *, processing_mark: bool = False) -> str | None:
    """Canonical extras; INFO's supported native passthrough is a separate path."""
    _validate(store, "2.3")
    if processing_mark:
        if name != model.MARK_NAME:
            raise ValueError("only NORMA is a processing mark")
    elif model.extra_name_problem(name) is not None:
        raise ValueError(f"invalid extra name: {name!r}")
    if store == "id3v2":
        return f"TXXX:{name}"
    if store == "mp4":
        return f"----:com.apple.iTunes:{name}"
    if store == "vorbis":
        return name
    return None


def _validate_fields(fields: Mapping[str, MetadataValue]) -> None:
    for field, value in fields.items():
        if field in model.TEXT_FIELDS:
            if not isinstance(value, str) or model.canonical_text(value, field) != value or not value:
                raise ValueError(f"{field}: expected validated canonical text")
        elif field == "date":
            if not isinstance(value, str) or model.source_date(value) != value:
                raise ValueError("date: expected a valid date or source timestamp")
        elif field in model.INTEGER_FIELDS:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field}: expected a positive integer")
        elif field == "compilation":
            if type(value) is not bool:
                raise ValueError("compilation: expected a boolean")
        else:
            raise ValueError(f"not a scalar canonical field: {field!r}")
    for number, total in (("track", "track_total"), ("disc", "disc_total")):
        if number in fields and total in fields:
            n, count = fields[number], fields[total]
            assert isinstance(n, int) and isinstance(count, int)
            if n > count:
                raise ValueError(f"{number} exceeds {total}; resolve the pair before mapping")


def project_fields(store: str | None, fields: Mapping[str, MetadataValue], *, id3_version: str = "2.3") -> FrozenMap:
    """Expected primary-store canonical values, not the winning requests."""
    _validate_fields(fields)
    if store is None:
        return FrozenMap()
    _validate(store, id3_version)
    result = {}
    for field, value in fields.items():
        if not native_keys(store, field, id3_version=id3_version, value=value):
            continue
        if field.endswith("_total") and store == "id3v2" and field.removesuffix("_total") not in fields:
            continue
        if field == "date":
            assert isinstance(value, str)
            value = model.project_date(value)
            if store == "id3v2" and id3_version == "2.3" and len(value) == 7:
                value = value[:4]
        result[field] = value
    return FrozenMap(result)


def loss_details(store: str | None, fields: Mapping[str, MetadataValue], *, id3_version: str = "2.3") -> FrozenMap:
    """Documented storage/projection losses, keyed by scalar field name."""
    effective = project_fields(store, fields, id3_version=id3_version)
    details = {}
    for field, value in fields.items():
        if field not in effective:
            details[field] = ("total_without_number" if store == "id3v2" and field.endswith("_total")
                              and field.removesuffix("_total") not in fields else "no_supported_mapping")
        elif field == "date" and value != effective[field]:
            assert isinstance(value, str)
            details[field] = "time_of_day_dropped" if model.is_source_timestamp(value) else "id3_month_dropped"
    return FrozenMap(details)


def project_artwork(store: str | None, pictures: Sequence[Artwork]) -> tuple[Artwork, ...]:
    """MP4 keeps only front covers; INFO/no store cannot carry pictures."""
    if store is not None:
        _validate(store, "2.3")
    if store in (None, "riff-info"):
        return ()
    for picture in pictures:
        if store == "mp4" and picture.picture_type == 3 and picture.mime not in ("image/jpeg", "image/png"):
            raise ValueError("MP4 cover must be prepared as JPEG or PNG at the edge")
    return tuple(picture for picture in pictures if store != "mp4" or picture.picture_type == 3)


def native_recipes(store: str, fields: Mapping[str, MetadataValue], *, extras: Mapping[str, str] | None = None,
                   pictures: Sequence[Artwork] = (), mark: str | None = None, id3_version: str = "2.3") -> tuple[FrozenMap, ...]:
    """Immutable write recipes: key, type, version, text encoding and native value.

    ID3 text values are one-element tuples; MP4 text/freeforms are tuples,
    pairs are tuples of pairs, cpil is scalar bool. Picture values are Artwork
    records (the edge serialises APIC/PICTURE/covr, with its native type).
    These are not MD32 payloads: the edge must encode and compare evidence.
    """
    effective = project_fields(store, fields, id3_version=id3_version)
    recipes = []

    def add(key: str, value: object) -> None:
        attributes = {}
        native_type = key.split(":", 1)[0] if store == "id3v2" else key
        if store == "id3v2":
            attributes["encoding"] = 1 if id3_version == "2.3" else 3
            if native_type == "COMM":
                attributes.update(lang="eng", desc="")
            elif native_type == "TXXX":
                attributes["desc"] = key.partition(":")[2]
        elif store == "mp4" and key.startswith("----:"):
            attributes.update(dataformat=1, version=0)
        elif store == "mp4" and key == "covr":
            assert isinstance(value, Artwork)
            attributes["imageformat"] = 13 if value.mime == "image/jpeg" else 14
        if store == "vorbis" and key != "PICTURE":
            native_type = "VORBIS_COMMENT"
        recipes.append(FrozenMap({"key": key, "native_type": native_type,
                                  "native_version": id3_version if store == "id3v2" else None,
                                  "text_encoding": ("utf-16" if id3_version == "2.3" else "utf-8")
                                  if store == "id3v2" else "utf-8", "attributes": attributes, "value": value}))

    handled = set()
    for field, value in effective.items():
        if field in handled:
            continue
        keys = native_keys(store, field, id3_version=id3_version, value=value)
        if field in ("track", "track_total", "disc", "disc_total") and store in ("mp4", "id3v2"):
            number = field.removesuffix("_total")
            total = number + "_total"
            n, count = effective.get(number), effective.get(total)
            handled.update((number, total))
            if store == "mp4":
                add(keys[0], ((n or 0, count or 0),))
            else:
                add(keys[0], (str(n) + (f"/{count}" if count is not None else ""),))
        elif field == "date" and store == "id3v2":
            if id3_version == "2.4":
                add("TDRC", (value,))
            else:
                add("TYER", (value[:4],))
                if len(value) == 10:
                    add("TDAT", (value[8:10] + value[5:7],))
        else:
            if store == "mp4":
                native = value if field == "compilation" else (value,)
            else:
                native = ("1" if value else "0") if field == "compilation" else str(value)
                if store == "id3v2":
                    native = (native,)
            add(keys[0], native)
    extra_names = set()
    for name, value in (extras or {}).items():
        if name.casefold() in extra_names:
            raise ValueError("resolve case-insensitive extra duplicates before mapping")
        extra_names.add(name.casefold())
        if not isinstance(value, str) or not value or model.canonical_text(value, model.EXTRA) != value:
            raise ValueError(f"{name}: expected validated extra text")
        key = extra_key(store, name)
        if key is not None:
            add(key, (value,) if store in ("mp4", "id3v2") else value)
    if mark is not None:
        key = extra_key(store, model.MARK_NAME, processing_mark=True)
        if key is not None:
            add(key, (mark,) if store in ("mp4", "id3v2") else mark)
    for picture in project_artwork(store, pictures):
        key = "APIC:" + picture.description if store == "id3v2" else _KEYS[store]["artwork"]
        add(key, picture)
    return tuple(recipes)


def canonical_fields(store: str, key: str) -> tuple[str, ...]:
    """Native → canonical key projection, including all source/legacy stores."""
    return model.native_fields(store, key)


def canonical_extra(store: str, key: str) -> str | None:
    return model.native_extra(store, key)
