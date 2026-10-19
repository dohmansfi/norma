"""Pure C-07 fate/precision accounting (MD6, MD7, MD18–MD21, MD32, MD40).

Resolution and native conversion are inputs, not guessed here. The edge
supplies immutable converted inventories, explicit occurrence links and any
established precision/reshaping relationships. Raw payloads are only compared,
never decoded. The resolver can compose these functions into MetadataPlan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from norma.metadata import model
from norma.models import (
    Artwork, Diagnostic, FieldStatus, FrozenMap, InventoryRef, MetadataValue,
    NativeItemStatus, NonTagItem, Provenance, RawNativeItem, ReasonCode, ResolvedValue,
    RollupValue, SourceTags,
)

_NATIVE_VALUE_ATTRS = ("store", "key", "native_type", "native_version", "payload_encoding", "payload")
_FILLER = frozenset({"JUNK", "junk", "PAD ", "FLLR", "PEAK"})


def native_equal(left: RawNativeItem, right: RawNativeItem) -> bool:
    """MD32 exact native value equality, ignoring identity and locator."""
    return all(getattr(left, name) == getattr(right, name) for name in _NATIVE_VALUE_ATTRS)


def inventory_index(items: Sequence[RawNativeItem | NonTagItem]) -> FrozenMap:
    result = {item.identity: item for item in items}
    if len(result) != len(items):
        raise ValueError("duplicate inventory identity")
    return FrozenMap(result)


def rollup_equal(left: RollupValue | None, right: RollupValue | None,
                 inventories: Mapping[str, Sequence[RawNativeItem | NonTagItem]]) -> bool:
    """Dereference within this revision; no identity/scope shortcut (MD40)."""
    def dereference(value: InventoryRef) -> tuple[RawNativeItem | NonTagItem, ...]:
        if value.scope not in inventories:
            raise ValueError(f"missing inventory scope: {value.scope}")
        index = inventory_index(inventories[value.scope])
        try:
            return tuple(index[identity] for identity in value.identities)
        except KeyError as exc:
            raise ValueError(f"unresolved inventory reference: {exc.args[0]}") from None

    # Even comparisons with None/type mismatch validate any reference.
    a = dereference(left) if isinstance(left, InventoryRef) else None
    b = dereference(right) if isinstance(right, InventoryRef) else None
    if isinstance(left, InventoryRef) and isinstance(right, InventoryRef):
        assert a is not None and b is not None
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, RawNativeItem) and isinstance(y, RawNativeItem):
                if not native_equal(x, y):
                    return False
            elif isinstance(x, NonTagItem) and isinstance(y, NonTagItem):
                # Size/hash of omitted bytes is evidence identity, not semantic equality.
                if x.payload is None or y.payload is None or x.kind != y.kind or x.payload != y.payload:
                    return False
            else:
                return False
        return True
    if isinstance(left, InventoryRef) or isinstance(right, InventoryRef):
        return False
    if isinstance(left, Artwork) and isinstance(right, Artwork):
        return left.data == right.data
    return model.values_equal(left, right)


def removed_by_design(item: RawNativeItem, audio_basis: str, *,
                      legacy_projected_ids: Sequence[str] = ()) -> bool:
    """MD6's closed native list, before request removal; gapless is basis-bound."""
    if audio_basis not in ("stream_copy", "encode"):
        raise ValueError("unknown audio basis")
    if item.store in ("id3v1", "apev2", "aiff-text") and item.identity in legacy_projected_ids:
        return True
    frame, _, rest = item.key.partition(":")
    frame = frame.upper()
    folded = item.key.casefold()
    name = rest.rsplit(":", 1)[-1] if item.store == "mp4" else rest
    if item.store == "id3v2" and frame in ("COMM", "COM"):
        name = rest.rsplit(":", 1)[0]
    if ((item.store == "id3v2" and frame in ("TXXX", "TXX", "COMM", "COM"))
            or (item.store == "mp4" and frame == "----")):
        folded = name.casefold()
    if model.is_mark_key(item.store, item.key):
        return True
    if folded.startswith(("replaygain_", "r128_")) or folded == "itunnorm":
        return True
    if folded == "itunsmpb":
        return audio_basis == "encode"
    return ((item.store == "id3v2" and frame in ("RVA2", "RVAD", "TSSE"))
            or (item.store == "mp4" and item.key.casefold() == "©too")
            or (item.store == "vorbis" and folded == "encoder")
            or (item.store == "riff-info" and folded == "isft")
            or (item.store == "apev2" and folded == "encoder"))


def rollup_status(key: str, *, source: RollupValue | None, requested: RollupValue | None,
                  effective: RollupValue | None, provenance: Provenance | None = None,
                  source_ids: Sequence[str] = (), expected_ids: Sequence[str] = (),
                  inventories: Mapping[str, Sequence[RawNativeItem | NonTagItem]] | None = None,
                  by_design: bool = False, by_request: bool = False,
                  precision_detail: str | None = None, detail: str = "") -> FieldStatus:
    """MD18 roll-up precedence, loss against request, never superseded source.

    precision_detail is an established MD40 projection from preparation; byte
    differences alone never establish precision. Absent effective is always
    whole-value loss, even if a projection detail was supplied.
    """
    inventories = inventories or {}
    for value in (source, requested, effective):
        if isinstance(value, InventoryRef):
            rollup_equal(value, value, inventories)
    if precision_detail and (requested is None or effective is None
                             or rollup_equal(requested, effective, inventories)):
        if effective is not None:
            raise ValueError("precision requires a present unequal request and effective value")
    loss = "none"
    if by_design:
        status = "removed_by_design"
    elif by_request and source is not None and effective is None:
        status = "removed_by_request"
    elif requested is not None and effective is None:
        status, loss = "dropped_format_limit", "whole_value"
    elif requested is not None and precision_detail:
        status, loss, detail = "dropped_format_limit", "precision", precision_detail
    elif source is not None:
        status = "kept" if rollup_equal(source, effective, inventories) else "changed"
    else:
        status = "added"
    return FieldStatus(key, status, requested, effective, provenance, loss,
                       tuple(source_ids), tuple(expected_ids), detail)


def scalar_statuses(source: SourceTags, resolved: Mapping[str, ResolvedValue],
                    effective: Mapping[str, ResolvedValue], *,
                    extras: Mapping[str, ResolvedValue] | None = None,
                    effective_extras: Mapping[str, ResolvedValue] | None = None,
                    expected_native: Sequence[RawNativeItem] = (),
                    source_field_ids: Mapping[str, Sequence[str]] | None = None,
                    expected_field_ids: Mapping[str, Sequence[str]] | None = None,
                    clear_keys: Sequence[str] = (), no_source_tags: bool = False,
                    precision_details: Mapping[str, str] | None = None,
                    loss_details: Mapping[str, str] | None = None) -> tuple[FieldStatus, ...]:
    """One row per scalar field/extra with evidence or an opinion.

    clear_keys and detail keys are roll-up keys (field:title, extra:BPM).
    Pictures and passthrough/non-tag items use separate functions below.
    source_field_ids is preparation's exact per-roll-up evidence index. It is
    required for absent components in combined number frames: the pure core
    cannot distinguish an invalid value from an absent 0 without decoding.
    """
    precision_details, loss_details = precision_details or {}, loss_details or {}
    if source_field_ids is None:
        for item in source.native_inventory:
            fields = model.native_fields(item.store, item.key)
            if len(fields) > 1 and any(name not in source.fields for name in fields):
                raise ValueError("combined number evidence requires source_field_ids from preparation")
    if expected_field_ids is None:
        for item in expected_native:
            fields = model.native_fields(item.store, item.key)
            if len(fields) > 1 and any(name not in effective for name in fields):
                raise ValueError("combined number output requires expected_field_ids from preparation")
    inventories = {"source_native": source.native_inventory, "expected_native": expected_native}
    rows = []
    for prefix, old, desired, stored in (("field:", source.fields, resolved, effective),
                                       ("extra:", source.extras, extras or {}, effective_extras or {})):
        spellings = {}
        if prefix == "extra:":
            # Resolution supplies the spelling to write. Compare old evidence
            # case-insensitively without adding a second roll-up for that name.
            for key in clear_keys:
                if key.startswith("extra:"):
                    spellings.setdefault(key[6:].casefold(), key[6:])
            for values in (desired, old, stored):
                folded = [name.casefold() for name in values]
                if len(set(folded)) != len(folded):
                    raise ValueError("duplicate case-insensitive extra identity")
                for name in values:
                    spellings.setdefault(name.casefold(), name)
            old = {spellings[name.casefold()]: value for name, value in old.items()}
            stored = {spellings[name.casefold()]: value for name, value in stored.items()}
        def matching(item: RawNativeItem, name: str) -> bool:
            if prefix == "field:":
                return name in model.native_fields(item.store, item.key)
            extra = model.native_extra(item.store, item.key)
            return extra is not None and extra.casefold() == name.casefold()

        evidence_names = set()
        for item in source.native_inventory:
            if prefix == "field:":
                if source_field_ids is None:
                    evidence_names.update(name for name in model.native_fields(item.store, item.key) if name != "artwork")
            else:
                name = model.native_extra(item.store, item.key)
                if name is not None and source_field_ids is None:
                    # Use the source projection's spelling when available.
                    evidence_names.add(spellings.setdefault(name.casefold(), name))
        if source_field_ids is not None:
            evidence_names.update(key[len(prefix):] for key, ids in source_field_ids.items() if key.startswith(prefix) and ids)
        clear_names = {key[len(prefix):] for key in clear_keys if key.startswith(prefix)}
        if prefix == "extra:":
            clear_names = {spellings.setdefault(name.casefold(), name) for name in clear_names}
            evidence_names = {spellings.setdefault(name.casefold(), name) for name in evidence_names}
        names = set(old) | set(desired) | evidence_names | clear_names
        for name in sorted(names):
            key = prefix + name
            source_ids = (tuple(dict.fromkeys(identity for label, ids in source_field_ids.items()
                                              if label == key or prefix == "extra:" and label.casefold() == key.casefold()
                                              for identity in ids)) if source_field_ids is not None
                          else tuple(item.identity for item in source.native_inventory if matching(item, name)))
            expected_ids = (tuple(dict.fromkeys(identity for label, ids in expected_field_ids.items()
                                                if label == key or prefix == "extra:" and label.casefold() == key.casefold()
                                                for identity in ids)) if expected_field_ids is not None
                            else tuple(item.identity for item in expected_native if matching(item, name)))
            old_value = old[name].value if name in old else InventoryRef("source_native", source_ids) if source_ids else None
            request = desired[name].value if name in desired else None
            current = stored[name].value if name in stored else None
            removal = name in clear_names or (no_source_tags and name not in desired)
            if request is None and old_value is not None and not removal:
                request = old_value
            # SourceTags retains timestamps verbatim (MD5); resolved carries date part.
            precision = precision_details.get(key)
            if key == "field:date" and not no_source_tags and name in desired and desired[name].provenance == "source" and isinstance(old_value, str):
                if model.is_source_timestamp(old_value):
                    request = old_value
                    if current is not None:
                        precision = "time_of_day_dropped"
            detail = loss_details.get(key, "")
            if isinstance(request, InventoryRef) and current is None and not removal:
                detail = "invalid_source_value"
            comparison_source = None if no_source_tags and name in desired else old_value
            rows.append(rollup_status(key, source=comparison_source, requested=request, effective=current,
                                      provenance=desired[name].provenance if name in desired else "source" if old_value is not None else None,
                                      source_ids=source_ids, expected_ids=expected_ids, inventories=inventories,
                                      by_request=removal, precision_detail=precision, detail=detail))
    return tuple(rows)


def other_statuses(source: SourceTags, *, target_stores: Sequence[str], audio_basis: str,
                   requested_native: Sequence[RawNativeItem] = (), expected_native: Sequence[RawNativeItem] = (),
                   links: Mapping[str, Sequence[str]] | None = None, no_source_tags: bool = False,
                   precision_details: Mapping[str, str] | None = None,
                   legacy_projected_ids: Sequence[str] = ()) -> tuple[FieldStatus, ...]:
    """MD19 passthrough/non-canonical rows and MD7's one aggregate chapter row.

    Canonical fields/extras/pictures are accounted for elsewhere. Preparation
    supplies links for conversions (including changed native keys). No guessed
    cross-family conversion or raw fallback passthrough is admitted.
    """
    links, precision_details = links or {}, precision_details or {}
    requested_index = inventory_index(requested_native)
    expected_index = inventory_index(expected_native)
    inventories = {"source_native": source.native_inventory, "requested_native": requested_native,
                   "expected_native": expected_native, "non_tag_items": source.non_tag_items}
    rows = []
    chapters = tuple(item.identity for item in source.native_inventory if model.is_chapter_key(item.store, item.key))
    if chapters and not any(item.identity == "item:chapters" for item in source.non_tag_items):
        raise ValueError("native chapters require the aggregate item:chapters evidence")
    for item in source.native_inventory:
        legacy = item.store in ("id3v1", "apev2", "aiff-text")
        design = removed_by_design(item, audio_basis, legacy_projected_ids=legacy_projected_ids)
        if model.is_chapter_key(item.store, item.key):
            continue
        if ((not legacy or item.identity in legacy_projected_ids)
                and (not design or legacy)
                and (model.native_fields(item.store, item.key) or model.native_extra(item.store, item.key))):
            continue
        key = "item:native:" + item.identity.removeprefix("store:")
        requested = InventoryRef("source_native", (item.identity,))
        ids = tuple(links.get(item.identity, ()))
        detail = ""
        fallback = item.store in ("id3v2", "mp4") and item.payload_encoding == "dn-raw-1"
        if design:
            ids = ()
        elif no_source_tags:
            ids, requested = (), None
        elif legacy:
            ids, detail = (), "legacy_store_not_carried"
        elif fallback:
            if item.identity in requested_index or ids:
                raise ValueError("raw fallback evidence cannot be passed through")
            detail = "raw_evidence_only"
        elif item.identity in requested_index:
            if item.store not in target_stores:
                raise ValueError("non-canonical passthrough must stay in the same family")
            if not native_equal(item, requested_index[item.identity]):
                raise ValueError("requested passthrough must keep the source native value")
            requested = InventoryRef("requested_native", (item.identity,))
            if not ids:
                detail = "native_conversion_removed"
        elif ids:
            raise ValueError("native links without requested passthrough evidence")
        else:
            detail = "no_cross_family_mapping" if item.store not in target_stores else "not_carried_v1"
        if any(identity not in expected_index for identity in ids):
            raise ValueError("unresolved expected native link")
        effective = InventoryRef("expected_native", ids) if ids else None
        if effective is not None and not rollup_equal(InventoryRef("source_native", (item.identity,)), effective, inventories):
            detail = "id3_version_converted" if item.store == "id3v2" and any(expected_index[i].native_version != item.native_version for i in ids) else "native_value_changed"
        rows.append(rollup_status(key, source=InventoryRef("source_native", (item.identity,)), requested=requested,
                                  effective=effective, provenance="source", source_ids=(item.identity,), expected_ids=ids,
                                  inventories=inventories, by_design=design, by_request=no_source_tags,
                                  precision_detail=precision_details.get(item.identity), detail=detail))
    for item in source.non_tag_items:
        design = item.kind == "chunk" and item.identity.removeprefix("item:chunk:").split("#", 1)[0] in _FILLER
        request = InventoryRef("non_tag_items", (item.identity,))
        rows.append(rollup_status(item.identity, source=request, requested=None if no_source_tags and not design else request,
                                  effective=None, provenance="source", source_ids=chapters if item.identity == "item:chapters" else (),
                                  inventories=inventories, by_design=design, by_request=no_source_tags,
                                  detail="removed_by_design" if design else "not_carried_v1"))
    return tuple(rows)


def native_statuses(source_native: Sequence[RawNativeItem], expected_native: Sequence[RawNativeItem],
                    statuses: Sequence[FieldStatus], *, audio_basis: str,
                    links: Mapping[str, Sequence[str]] | None = None,
                    no_source_tags: bool = False, superseded_ids: Sequence[str] = (),
                    changed_details: Mapping[str, str] | None = None,
                    precision_details: Mapping[str, str] | None = None,
                    legacy_projected_ids: Sequence[str] = ()) -> tuple[NativeItemStatus, ...]:
    """Exactly one source fate and one added row per underived output (MD18).

    A single occurrence can carry two number components or feed several rows.
    Their links are unioned in expected-inventory order. Explicit links override
    that union, e.g. supplied artwork has no source counterpart. Superseded and
    multi-string/reshaping facts come from edge preparation/resolution, never
    from decoding bytes in this pure module.
    """
    links, changed_details, precision_details = links or {}, changed_details or {}, precision_details or {}
    source_index, expected_index = inventory_index(source_native), inventory_index(expected_native)
    if len({row.key for row in statuses}) != len(statuses):
        raise ValueError("duplicate roll-up identity")
    for identity in set(links) | set(superseded_ids) | set(changed_details) | set(precision_details):
        if identity not in source_index:
            raise ValueError(f"unknown source occurrence: {identity}")
    for row in statuses:
        if any(i not in source_index for i in row.source_native_ids) or any(i not in expected_index for i in row.expected_native_ids):
            raise ValueError("unresolved roll-up link")
    rows, derived = [], set()
    for item in source_native:
        related = tuple(row for row in statuses if item.identity in row.source_native_ids)
        candidates = set(links.get(item.identity, ())) if item.identity in links else {i for row in related for i in row.expected_native_ids}
        if any(i not in expected_index for i in candidates):
            raise ValueError("unresolved expected native link")
        design = removed_by_design(item, audio_basis, legacy_projected_ids=legacy_projected_ids)
        if not related and not design:
            raise ValueError("source occurrence needs a roll-up")
        ids = tuple(i for i in expected_index if i in candidates)
        detail, loss = changed_details.get(item.identity, ""), "none"
        if design:
            status = "removed_by_design"
        elif no_source_tags or related and all(row.status == "removed_by_request" for row in related):
            status, ids = "removed_by_request", ()
        elif item.identity in superseded_ids:
            status, detail = "changed", "superseded"
        elif detail == "pair_inconsistent":
            status = "changed"
        elif not ids:
            status, loss = "dropped_format_limit", "whole_value"
            detail = next((row.detail for row in related if row.detail), "no_supported_mapping")
        elif item.identity in precision_details:
            status, loss = "dropped_format_limit", "precision"
            detail = precision_details[item.identity]
        elif detail or any(row.status in ("changed", "removed_by_request") for row in related):
            status = "changed"
        elif related and any(not row.key.startswith("item:native:") for row in related):
            # Canonical equality, not native encoding/version equality.
            status = "kept"
        elif len(ids) == 1 and native_equal(item, expected_index[ids[0]]):
            status = "kept"
        else:
            status = "changed"
            if item.store == "id3v2" and any(expected_index[i].native_version != item.native_version for i in ids):
                detail = "id3_version_converted"
        rows.append(NativeItemStatus("source:" + item.identity, status, (item.identity,), ids, loss, detail))
        derived.update(ids)
    rows.extend(NativeItemStatus("output:" + item.identity, "added", (), (item.identity,), "none", "")
                for item in expected_native if item.identity not in derived)
    return tuple(rows)


def loss_policy(statuses: Sequence[FieldStatus], policy: str) -> tuple[str, tuple[Diagnostic, ...]]:
    """MD21 reads roll-ups only, returning a C-07 policy result/diagnostics.

    Sidecar reservation/serialisation is the caller's job, not a write here.
    Both loss warnings survive; precision comes first under C-13 precedence.
    """
    if policy not in ("warn", "sidecar", "error"):
        raise ValueError("unknown metadata loss policy")
    if len({row.key for row in statuses}) != len(statuses):
        raise ValueError("duplicate roll-up identity")
    losses = tuple(row for row in statuses if row.status == "dropped_format_limit")
    if not losses:
        return "ok", ()
    if any(row.loss_kind == "none" for row in losses):
        raise ValueError("dropped roll-up must specify its loss kind")
    if policy == "error":
        return "error", (Diagnostic("metadata_loss_error", "metadata loss prevents rendering", "metadata_planned",
                                    ",".join(row.key for row in losses)),)
    diagnostics = []
    warning_codes: tuple[tuple[str, ReasonCode], ...] = (("precision", "metadata_precision_loss"), ("whole_value", "metadata_loss_warn"))
    for kind, code in warning_codes:
        keys = tuple(row.key for row in losses if row.loss_kind == kind)
        if keys:
            diagnostics.append(Diagnostic(code, "metadata " + kind.replace("_", " ") + " loss", "metadata_planned", ",".join(keys)))
    return policy, tuple(diagnostics)
