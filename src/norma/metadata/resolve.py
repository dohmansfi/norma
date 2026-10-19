"""Pure metadata resolution and C-07 planning (MD11–MD21, CR-082/083).

Call resolve_opinions, then preparation_requests. The I/O edge passes each
request to write.prepare_store; build_metadata_plan consumes those immutable
items and explicit links. No mutagen, payload decoding, files or subprocesses
belong here. Returned FrozenMaps are implementation-local compositions of the
existing C-06/C-11 records, not new boundary records.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Protocol

from norma.errors import FileFailure, UsageError
from norma.metadata import losscheck, mapping, model
from norma.models import (
    Artwork, ArtworkPlan, Conflict, FrozenMap, ManifestCell, MetadataPlan,
    InventoryRef, RawNativeItem, ResolvedValue, SidecarRequest, SourceTags,
)


class PreparedStore(Protocol):
    """Structural interface to write.prepare_store; no edge import at runtime."""

    @property
    def store(self) -> str: ...

    @property
    def items(self) -> tuple[RawNativeItem, ...]: ...

    @property
    def recipe_ids(self) -> tuple[tuple[str, ...], ...]: ...

    @property
    def passthrough_ids(self) -> Mapping[str, tuple[str, ...]]: ...


def _key(name: str) -> str:
    if name in model.CANONICAL_FIELDS:
        return "field:" + name
    extra = name[6:] if name.startswith("extra:") else name
    if model.extra_name_problem(extra) is None:
        return "extra:" + extra
    raise ValueError(f"not a metadata opinion key: {name!r}")


def _validated(name: str, value: object) -> object:
    if name.startswith("extra:") or name in model.TEXT_FIELDS:
        if not isinstance(value, str):
            raise ValueError(f"{name}: expected text")
        value = model.canonical_text(value, model.EXTRA if name.startswith("extra:") else name)
        if not value:
            raise ValueError(f"{name}: empty set is not an opinion")
    elif name == "date":
        if not isinstance(value, str):
            raise ValueError("date: expected text")
        value = model.canonical_date(value)
    elif name in model.INTEGER_FIELDS:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name}: expected positive integer")
    elif name == "compilation":
        if type(value) is not bool:
            raise ValueError("compilation: expected boolean")
    elif name == "artwork":
        if not isinstance(value, Artwork) or value.picture_type != 3:
            raise ValueError("artwork: supply a prepared front cover, not a path")
    return value


def _equal(left: object, right: object) -> bool:
    if isinstance(left, Artwork) and isinstance(right, Artwork):
        return hashlib.sha256(left.data).digest() == hashlib.sha256(right.data).digest()
    return model.values_equal(left, right)


def resolve_opinions(source: SourceTags, *, stem: str,
                     cli: Mapping[str, ManifestCell] | None = None,
                     generated: Mapping[str, ManifestCell] | None = None,
                     manifest: Mapping[str, ManifestCell] | None = None,
                     manifest_default: Mapping[str, ManifestCell] | None = None,
                     config: Mapping[str, ManifestCell] | None = None,
                     folder_artwork: ManifestCell | None = None,
                     folder_artwork_provenance: str = "cli",
                     filename: Mapping[str, ManifestCell | ResolvedValue] | None = None,
                     locators: Mapping[str, str] | None = None,
                     prefer: str = "explicit", infer_mode: str = "fill",
                     no_source_tags: bool = False, no_title_fallback: bool = False,
                     strict_conflicts: bool = False,
                     source_pictures: Sequence[Artwork] | None = None) -> FrozenMap:
    """R1–R10: resolve independent typed opinions and MD15 pairs.

    Canonical opinion keys are bare field names; extras keep their spelling
    (extra:NAME is also accepted). Inference supplies C-11 set cells or C-06
    ResolvedValues. Artwork set cells must already hold
    prepared Artwork bytes. source_pictures optionally contains edge-converted
    embedded pictures in exactly the original source order (MD24).
    Informational layering/fill notes stay strings, never invented reason codes.
    """
    if prefer not in ("source", "explicit") or infer_mode not in ("fill", "override"):
        raise ValueError("unknown metadata preference or inference mode")
    if no_source_tags and prefer == "source":
        raise UsageError("--no-source-tags conflicts with --prefer source")
    if folder_artwork_provenance not in ("cli", "config"):
        raise ValueError("folder artwork provenance must be cli or config")
    locators = locators or {}
    layers = [("cli", 0, cli or {}), ("generated", 0, generated or {}),
              ("manifest", 1, manifest or {}), ("manifest_default", 2, manifest_default or {}),
              ("config", 3, config or {})]
    if folder_artwork is not None:
        layers.append((folder_artwork_provenance, 3, {"artwork": folder_artwork}))
    opinions: dict[str, list[tuple[int, str, ManifestCell, str]]] = {}
    spellings: dict[str, str] = {}
    for provenance, rank, cells in layers:
        local = {}
        for name, cell in cells.items():
            key = _key(name)
            folded = key.casefold() if key.startswith("extra:") else key
            if folded in local:
                raise ValueError("duplicate case-insensitive extra opinion")
            local[folded] = cell
            spellings.setdefault(folded, key)
            if not isinstance(cell, ManifestCell):
                raise TypeError("opinions must use ManifestCell")
            if cell.operation == "set":
                cell = ManifestCell("set", _validated(key if key.startswith("extra:") else name, cell.value))
            bucket = opinions.setdefault(folded, [])
            if any(old[0] == rank for old in bucket):
                raise UsageError(f"duplicate set/clear opinions at the same rank: {name}")
            bucket.append((rank, provenance, cell, locators.get(provenance, provenance)))
        for number, total in (("track", "track_total"), ("disc", "disc_total")):
            a, b = cells.get(number), cells.get(total)
            if a and b and a.operation == b.operation == "set" and a.value > b.value:
                raise ValueError(f"{provenance}: {number} exceeds {total}; validate input pairs first")
    inferred = {}
    for name, value in (filename or {}).items():
        key = _key(name)
        if name == "artwork" or key.startswith("extra:"):
            raise ValueError("filename inference supplies canonical scalar fields only")
        if isinstance(value, ManifestCell) and value.operation != "set":
            raise ValueError("filename inference cannot clear a field")
        locator = value.locator if isinstance(value, ResolvedValue) else locators.get("filename", "filename")
        inferred[key] = ResolvedValue(_validated(name, value.value), "filename", locator)
    old = {"field:" + name: value for name, value in source.fields.items()}
    for name, value in source.extras.items():
        key = "extra:" + name
        old[key.casefold()] = value
        spellings.setdefault(key.casefold(), key)
    if source.artwork:
        cover = next((value for value in source.artwork if value.value.picture_type == 3), None)
        if cover is not None:
            old["field:artwork"] = cover
    resolved, extras, clears, conflicts, notes, ranks = {}, {}, [], [], [], {}
    clear_provenance = {}
    names = set(opinions) | set(old) | set(inferred) | {"field:title"}
    artwork_value = None
    for folded in sorted(names):
        key = spellings.get(folded, folded)
        explicit = sorted(opinions.get(folded, ()), key=lambda opinion: opinion[0])
        e = explicit[0] if explicit else None
        s = old.get(folded) if not no_source_tags else None
        if key == "field:date" and s is not None:
            s = replace(s, value=model.project_date(s.value))
        i = inferred.get(folded)
        for lower in explicit[1:]:
            if e[2].operation != lower[2].operation:
                notes.append(f"{key}: {e[3]} overrides {lower[2].operation} in {lower[3]}")
        winner = None
        if e and e[2].operation == "clear":  # R1
            clears.append(key)
            clear_provenance[key] = e[1]
            continue
        if prefer == "explicit":
            if e:  # R2
                winner = ResolvedValue(e[2].value, e[1], e[3])
            elif i is not None and infer_mode == "override":  # R3
                winner = i
            elif s is not None:  # R4
                winner = ResolvedValue(s.value, "source", s.locator)
            elif i is not None:  # R5
                winner = i
        else:
            if s is not None:  # R6
                winner = ResolvedValue(s.value, "source", s.locator)
            elif e:  # R7
                winner = ResolvedValue(e[2].value, e[1], e[3])
            elif i is not None:  # R8
                winner = i
        if winner is None and key == "field:title" and not no_title_fallback:  # R9
            winner = ResolvedValue(unicodedata.normalize("NFC", stem), "fallback", stem)
        if winner is None:  # R10
            continue
        category = ("source" if winner.provenance == "source" else "filename" if winner.provenance == "filename"
                    else "fallback" if winner.provenance == "fallback" else "explicit")
        rank = (e[0] if category == "explicit" else 4 if category == "filename" and infer_mode == "override"
                else 5 if category == "source" else 6 if category == "filename" else 7)
        if prefer == "source" and category == "source":
            rank = -1
        ranks[key] = rank
        comparisons = []
        if e and category != "explicit":
            comparisons.append((e[2].value, e[3]))
        if s is not None and category != "source":
            comparisons.append((s.value, "source:" + s.locator))
        if i is not None and category != "filename":
            if infer_mode == "override":
                comparisons.append((i.value, i.locator or "filename"))
            else:
                notes.append(f"{key}: inferred {i.value!r} not used")
        if category != "fallback":
            for value, locator in comparisons:
                if value is not None and not _equal(value, winner.value):
                    conflicts.append(Conflict(key, value, winner.value, locator, winner.provenance, "superseded"))
        if key == "field:date" and winner.provenance == "source":
            winner = replace(winner, value=model.project_date(winner.value))
        if key == "field:artwork":
            artwork_value = winner
        elif key.startswith("extra:"):
            extras[key[6:]] = winner
        else:
            resolved[key[6:]] = winner
    discarded = {}
    for number, total in (("track", "track_total"), ("disc", "disc_total")):
        if number in resolved and total in resolved and resolved[number].value > resolved[total].value:
            a, b = "field:" + number, "field:" + total
            if ranks[a] == ranks[b]:
                raise ValueError("equal-rank inconsistent pair; input validation is missing")
            loser, keeper = (number, total) if ranks[a] > ranks[b] else (total, number)
            value = resolved.pop(loser)
            discarded["field:" + loser] = value
            conflicts.append(Conflict("field:" + loser, value.value, resolved[keeper].value,
                                      value.locator or value.provenance, resolved[keeper].provenance, "pair_inconsistent"))
    if strict_conflicts and conflicts:
        raise FileFailure("metadata_conflict_strict", "metadata_planned",
                          ",".join(dict.fromkeys(conflict.key for conflict in conflicts)))
    pictures = tuple(source_pictures) if source_pictures is not None else tuple(value.value for value in source.artwork)
    if len(pictures) != len(source.artwork):
        raise ValueError("prepared source pictures must retain source order and multiplicity")
    if "field:artwork" in clears:
        artwork = None
    elif artwork_value is not None and artwork_value.provenance != "source":
        pictures = (artwork_value.value,) + (() if no_source_tags else tuple(p for p in pictures if p.picture_type != 3))
        artwork = ArtworkPlan(pictures, artwork_value.provenance, artwork_value.locator)
    elif pictures and not no_source_tags:
        artwork = ArtworkPlan(pictures, "source", source.artwork[0].locator)
    else:
        artwork = None
    return FrozenMap({"resolved": resolved, "extras": extras, "artwork": artwork, "clear_keys": tuple(clears),
                      "clear_provenance": clear_provenance,
                      "conflicts": tuple(conflicts), "notes": tuple(notes), "pair_discarded": discarded,
                      "no_source_tags": no_source_tags,
                      "source_pictures": tuple(source_pictures) if source_pictures is not None else
                      tuple(value.value for value in source.artwork)})


def preparation_requests(source: SourceTags, resolution: Mapping, *, target: str,
                         wav_tags: str = "info+id3", id3_version: str = "2.3") -> FrozenMap:
    """Native recipes and verbatim passthrough for each target store (primary first).

    The edge must call write.prepare_store with exactly these recipes/items.
    Removed-by-design items are filtered by build_metadata_plan's audio basis;
    this function takes the basis through resolution['audio_basis'] when set,
    otherwise encode. Use with_audio_basis to prepare copy/retry alternatives.
    """
    basis = resolution.get("audio_basis", "encode")
    stores = mapping.target_stores(target, wav_tags)
    fields = {name: value.value for name, value in resolution["resolved"].items()}
    extras = {name: value.value for name, value in resolution["extras"].items()}
    artwork = resolution["artwork"]
    pictures = artwork.pictures if artwork else ()
    requests = {}
    for store in stores:
        passthrough = tuple(item for item in source.native_inventory
                            if item.store == store and not resolution["no_source_tags"]
                            and not losscheck.removed_by_design(item, basis)
                            and not model.native_fields(item.store, item.key)
                            and not model.native_extra(item.store, item.key)
                            and not model.is_chapter_key(item.store, item.key)
                            and not (item.store in ("id3v2", "mp4") and item.payload_encoding == "dn-raw-1"))
        recipes = mapping.native_recipes(store, fields, extras=extras, pictures=pictures, id3_version=id3_version)
        if store == "riff-info":
            # MD19's closed INFO subset is native source metadata, not general
            # --tag support. MD5's canonical joined text is already prepared;
            # never decode source payloads here to reconstruct it.
            for name, value in resolution["extras"].items():
                original = next((tag for key, tag in source.extras.items() if key.casefold() == name.casefold()), None)
                if (value.provenance == "source" and original is not None and original.system == "riff-info"
                        and name.upper() in mapping.INFO_PASSTHROUGH):
                    native_key = name.upper()
                    recipes += (FrozenMap({"key": native_key, "native_type": native_key, "native_version": None,
                                           "text_encoding": "utf-8", "attributes": {}, "value": value.value}),)
        requests[store] = FrozenMap({"recipes": recipes, "passthrough": passthrough})
    return FrozenMap(requests)


def with_audio_basis(resolution: Mapping, audio_basis: str) -> FrozenMap:
    if audio_basis not in ("encode", "stream_copy"):
        raise ValueError("unknown audio basis")
    return FrozenMap({**resolution, "audio_basis": audio_basis})


def build_metadata_plan(source: SourceTags, resolution: Mapping, *, target: str,
                        prepared_stores: Sequence[PreparedStore],
                        source_field_ids: Mapping[str, Sequence[str]],
                        wav_tags: str = "info+id3", id3_version: str = "2.3",
                        on_metadata_loss: str = "warn", plan_revision: int = 1,
                        conditional: bool = False, no_mark_processed: bool = False,
                        native_precision_details: Mapping[str, str] | None = None) -> MetadataPlan:
    """Compose C-07 using explicit edge recipe/passthrough links, never guessed IDs.

    source_field_ids is read.source_field_ids(source). Unknown, missing or
    duplicate prepared links are internal planning errors, not silent losses.
    An error loss policy raises FileFailure before any allocation/render.
    """
    basis = resolution.get("audio_basis", "encode")
    requests = preparation_requests(source, resolution, target=target, wav_tags=wav_tags, id3_version=id3_version)
    source_index = losscheck.inventory_index(source.native_inventory)
    for key, ids in source_field_ids.items():
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source field evidence identity")
        for identity in ids:
            if identity not in source_index:
                raise ValueError("unresolved source field evidence")
            item = source_index[identity]
            fields = model.native_fields(item.store, item.key)
            extra = model.native_extra(item.store, item.key)
            if not (key.startswith("field:") and key[6:] in fields or
                    key.startswith("extra:") and extra is not None and key[6:].casefold() == extra.casefold()):
                raise ValueError("source field evidence does not match its native key")
    for name, value in source.fields.items():
        known_ids = set(value.locator.split(",")) & set(source_index)
        if not known_ids <= set(source_field_ids.get("field:" + name, ())):
            raise ValueError("source field evidence omits projected occurrences")
    stores = tuple(requests)
    if tuple(prepared.store for prepared in prepared_stores) != stores:
        raise ValueError("prepared stores must match the target stores in primary order")
    resolved, extras = resolution["resolved"], resolution["extras"]
    clears, no_source = resolution["clear_keys"], resolution["no_source_tags"]
    clear_provenance = resolution["clear_provenance"]
    primary = stores[0] if stores else None
    fields = {name: value.value for name, value in resolved.items()}
    projected = mapping.project_fields(primary, fields, id3_version=id3_version)
    effective = {name: replace(resolved[name], value=value) for name, value in projected.items()}
    effective_extras = dict(extras) if primary not in (None, "riff-info") else {}
    if primary == "riff-info":
        info_names = {recipe["key"].casefold() for recipe in requests[primary]["recipes"]}
        effective_extras = {name: value for name, value in extras.items() if name.casefold() in info_names}
    artwork = resolution["artwork"]
    pictures = artwork.pictures if artwork else ()
    effective_artwork = mapping.project_artwork(primary, pictures)
    expected = tuple(item for prepared in prepared_stores for item in prepared.items)
    expected_index = losscheck.inventory_index(expected)
    requested = tuple(item for request in requests.values() for item in request["passthrough"])
    expected_fields: dict[str, list[str]] = {}
    picture_ids: dict[int, tuple[str, ...]] = {}
    links = {}
    for prepared in prepared_stores:
        request = requests[prepared.store]
        if len(prepared.recipe_ids) != len(request["recipes"]):
            raise ValueError("prepared recipe links must cover every recipe")
        if set(prepared.passthrough_ids) != {item.identity for item in request["passthrough"]}:
            raise ValueError("prepared passthrough links must cover every occurrence")
        links.update(prepared.passthrough_ids)
        derived = []
        pic_position = 0
        for recipe, ids in zip(request["recipes"], prepared.recipe_ids):
            if not ids or any(identity not in expected_index for identity in ids):
                raise ValueError("unresolved prepared recipe link")
            if any(expected_index[identity].store != prepared.store for identity in ids):
                raise ValueError("prepared links cross stores")
            derived.extend(ids)
            if isinstance(recipe["value"], Artwork):
                if prepared.store == primary:
                    picture_ids[pic_position] = tuple(ids)
                    pic_position += 1
                continue
            names = model.native_fields(prepared.store, recipe["key"])
            for name in names:
                if name in mapping.project_fields(prepared.store, fields, id3_version=id3_version):
                    expected_fields.setdefault("field:" + name, []).extend(ids)
            extra = model.native_extra(prepared.store, recipe["key"])
            if extra is not None:
                name = next((name for name in extras if name.casefold() == extra.casefold()), extra)
                expected_fields.setdefault("extra:" + name, []).extend(ids)
        for ids in prepared.passthrough_ids.values():
            if any(identity not in expected_index or expected_index[identity].store != prepared.store for identity in ids):
                raise ValueError("unresolved prepared passthrough link")
            derived.extend(ids)
        if len(derived) != len(set(derived)) or set(derived) != {item.identity for item in prepared.items}:
            raise ValueError("prepared occurrence must have exactly one recipe/passthrough producer")
    expected_fields = {key: list(dict.fromkeys(ids)) for key, ids in expected_fields.items()}
    # Preserve winning spelling for case-insensitive extra source evidence.
    spellings = {name.casefold(): name for name in extras}
    spellings.update({key[6:].casefold(): key[6:] for key in clears if key.startswith("extra:")})
    evidence = {("extra:" + spellings.get(key[6:].casefold(), key[6:]) if key.startswith("extra:") else key): tuple(ids)
                for key, ids in source_field_ids.items()}
    legacy_projected_ids = tuple(identity for ids in evidence.values() for identity in ids
                                 if source_index[identity].store in ("id3v1", "apev2", "aiff-text"))
    details = {"field:" + key: value for key, value in mapping.loss_details(primary, fields, id3_version=id3_version).items()}
    precision = {key: value for key, value in details.items() if key[6:] in effective}
    scalar = list(losscheck.scalar_statuses(source, resolved, effective, extras=extras, effective_extras=effective_extras,
                                            expected_native=expected, source_field_ids=evidence,
                                            expected_field_ids=expected_fields, clear_keys=clears,
                                            no_source_tags=no_source, precision_details=precision, loss_details=details))
    # R1 has no ResolvedValue: preserve its winning explicit layer in C-07
    # without inventing a requested/effective value or changing loss fates.
    scalar = [replace(row, provenance=clear_provenance[row.key]) if row.key in clear_provenance else row
              for row in scalar]
    for key, value in resolution["pair_discarded"].items():
        index = next((i for i, row in enumerate(scalar) if row.key == key), None)
        row = losscheck.rollup_status(key, source=source.fields[key[6:]].value if key[6:] in source.fields else None,
                                     requested=None, effective=None, provenance=value.provenance,
                                     source_ids=evidence.get(key, ()), detail="pair_inconsistent")
        row = replace(row, status="changed", detail="pair_inconsistent")
        if index is None:
            scalar.append(row)
        else:
            scalar[index] = row
    statuses = scalar + list(losscheck.other_statuses(source, target_stores=stores, audio_basis=basis,
                                                     requested_native=requested, expected_native=expected, links=links,
                                                     no_source_tags=no_source, precision_details=native_precision_details,
                                                     legacy_projected_ids=legacy_projected_ids))
    # Each source picture keeps its stable source identity; a supplied front
    # cover is a separate addition, never derived from a replaced source cover.
    output_pictures = list(enumerate(effective_artwork))
    available = list(output_pictures)
    supplied = artwork is not None and artwork.provenance != "source"
    picture_provenance = clear_provenance.get("field:artwork",
                                             artwork.provenance if artwork is not None and supplied else "source")
    for n, tag in enumerate(source.artwork, 1):
        old_picture = tag.value
        ids = tuple(tag.locator.split(","))
        replacement = supplied and old_picture.picture_type == 3
        current = None
        expected_ids = ()
        if not no_source and "field:artwork" not in clears and not replacement:
            prepared_picture = resolution["source_pictures"][n - 1]
            candidate = next(((pos, p) for pos, p in available if p == prepared_picture), None)
            if candidate:
                available.remove(candidate)
                pos, current = candidate
                expected_ids = picture_ids.get(pos, ())
        row = losscheck.rollup_status("picture:" + str(n), source=old_picture,
                                     requested=None if replacement or no_source or "field:artwork" in clears else
                                     current if current is not None else old_picture,
                                     effective=current, provenance=picture_provenance,
                                     source_ids=ids, expected_ids=expected_ids,
                                     by_request=no_source or "field:artwork" in clears,
                                     detail="superseded" if replacement else "no_supported_mapping" if current is None else "")
        if replacement:
            row = replace(row, status="changed", detail="superseded")
        statuses.append(row)
        for identity in ids:
            links[identity] = expected_ids
    if supplied:
        picture = pictures[0]
        pos = next((pos for pos, p in output_pictures if p is picture or p == picture), None)
        statuses.append(losscheck.rollup_status("picture:new", source=None, requested=picture,
                                                effective=picture if pos is not None else None,
                                                provenance=artwork.provenance,
                                                expected_ids=picture_ids.get(pos, ()), detail="no_supported_mapping"))
    # Malformed pictures have no Artwork; they remain present raw evidence.
    pictured_ids = {identity for tag in source.artwork for identity in tag.locator.split(",")}
    for item in source.native_inventory:
        if "artwork" in model.native_fields(item.store, item.key) and item.identity not in pictured_ids:
            ref = InventoryRef("source_native", (item.identity,))
            statuses.append(losscheck.rollup_status("item:native:" + item.identity.removeprefix("store:"),
                                                    source=ref, requested=None if no_source or "field:artwork" in clears else ref,
                                                    effective=None, provenance=clear_provenance.get("field:artwork", "source"),
                                                    source_ids=(item.identity,),
                                                    inventories={"source_native": source.native_inventory},
                                                    by_request=no_source or "field:artwork" in clears, detail="invalid_source_value"))
    # A stored empty/zero pair has no canonical component but is still a
    # native occurrence. Keep its evidence and account its absent output once.
    covered = {identity for row in statuses for identity in row.source_native_ids}
    for item in source.native_inventory:
        if item.identity not in covered and not losscheck.removed_by_design(item, basis,
                                                                          legacy_projected_ids=legacy_projected_ids):
            ref = InventoryRef("source_native", (item.identity,))
            statuses.append(losscheck.rollup_status("item:native:" + item.identity.removeprefix("store:"),
                                                    source=ref, requested=None if no_source else ref, effective=None,
                                                    provenance="source", source_ids=(item.identity,),
                                                    inventories={"source_native": source.native_inventory},
                                                    by_request=no_source, detail="empty_source_value"))
    changed, superseded, native_precision = {}, set(), dict(native_precision_details or {})
    for key, ids in evidence.items():
        name = key[6:]
        desired = extras.get(name) if key.startswith("extra:") else resolved.get(name)
        old_values = source.extras if key.startswith("extra:") else source.fields
        old_value = next((v for k, v in old_values.items() if k.casefold() == name.casefold()), None)
        if desired is not None and desired.provenance != "source" and (old_value is None or not _equal(desired.value, old_value.value)):
            superseded.update(ids)
        if key in resolution["pair_discarded"]:
            changed.update({identity: "pair_inconsistent" for identity in ids})
        elif desired is not None and desired.provenance == "source" and old_value is not None:
            if len(old_value.locator.split(",")) > 1 or isinstance(old_value.value, str) and "; " in old_value.value:
                changed.update({identity: "multi_value_joined" for identity in ids})
            if key in precision:
                for identity in ids:
                    outputs = tuple(expected_index[i] for i in expected_fields.get(key, ()))
                    if outputs and all(mapping.project_fields(item.store, fields, id3_version=id3_version).get(name) != desired.value
                                       for item in outputs):
                        native_precision[identity] = precision[key]
            if key == "field:date" and isinstance(old_value.value, str) and model.is_source_timestamp(old_value.value):
                native_precision.update({identity: "time_of_day_dropped" for identity in ids})
    for row in scalar:
        if row.detail == "invalid_source_value":
            changed.update({identity: "invalid_source_value" for identity in row.source_native_ids})
    for conflict in source.internal_conflicts:
        if conflict.detail == "invalid_source_value" or conflict.detail.startswith("mp4_data_type:"):
            identity = conflict.old_source.removeprefix("source:")
            changed[identity] = "invalid_source_value"
        elif conflict.detail.startswith("internal:"):
            superseded.add(conflict.old_source.removeprefix("source:"))
    for row in statuses:
        if row.key.startswith("picture:") and row.detail == "superseded":
            superseded.update(row.source_native_ids)
        elif row.key.startswith("picture:") and row.status == "changed":
            changed.update({identity: "artwork_prepared" for identity in row.source_native_ids})
    native_rows = losscheck.native_statuses(source.native_inventory, expected, statuses, audio_basis=basis,
                                           links=links, no_source_tags=no_source, superseded_ids=tuple(superseded),
                                           changed_details=changed, precision_details=native_precision,
                                           legacy_projected_ids=legacy_projected_ids)
    policy, diagnostics = losscheck.loss_policy(statuses, on_metadata_loss)
    if policy == "error":
        raise FileFailure("metadata_loss_error", "metadata_planned", diagnostics[0].detail, diagnostics)
    sidecars = _sidecar_requests(statuses) if policy == "sidecar" else ()
    mark_state = "not_storable" if primary in (None, "riff-info") else "omitted" if no_mark_processed else "pending"
    return MetadataPlan(plan_revision, basis, conditional, source.native_inventory, requested, expected,
                        resolved, effective, extras, effective_extras, artwork, effective_artwork,
                        tuple(statuses), native_rows, resolution["conflicts"], source.non_tag_items,
                        sidecars, mark_state, policy, diagnostics)


def _sidecar_requests(statuses) -> tuple[SidecarRequest, ...]:
    requests = [SidecarRequest("tags", "tags_sidecar", None, ".tags.json")]
    cover = False
    for row in statuses:
        if row.status != "dropped_format_limit" or not isinstance(row.requested, Artwork):
            continue
        picture = row.requested
        extension = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif", "image/bmp": "bmp",
                     "image/webp": "webp", "image/heic": "heic", "image/tiff": "tif"}.get(picture.mime, "bin")
        if picture.picture_type == 3 and not cover and extension in ("jpg", "png"):
            requests.append(SidecarRequest("cover", "cover_export", row.key, ".cover." + extension))
            cover = True
        else:
            number = row.key.removeprefix("picture:")
            requests.append(SidecarRequest("picture-" + number, "picture_export", row.key,
                                           ".picture-" + number + "." + extension))
    return tuple(requests)
