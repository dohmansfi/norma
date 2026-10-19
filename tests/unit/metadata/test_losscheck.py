"""TST-M-02: MD18/MD40 fates, exact occurrence coverage and policy inputs."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from norma.metadata import losscheck as lc
from norma.models import (
    Artwork, FieldStatus, InventoryRef, MetadataPlan, NonTagItem, RawNativeItem,
    ResolvedValue, SourceTags, TagValue,
)


def native(store="id3v2", key="TIT2", payload=b"evidence", version="2.3", encoding="dn-json-1", identity=None):
    return RawNativeItem(identity or f"store:{store}:{key}", store, key, key.split(":")[0],
                         version if store == "id3v2" else None, encoding, payload, "test")


def source(items=(), fields=None, extras=None, nontag=()):
    return SourceTags(fields or {}, extras or {}, (), nontag, (), items,
                      {store: tuple(i.identity for i in items if i.store == store) for store in dict.fromkeys(i.store for i in items)})


def rv(value, provenance="source"):
    return ResolvedValue(value, provenance, None)


def plan(items, expected, statuses, native_rows, *, requested=(), nontag=()):
    result, diagnostics = lc.loss_policy(statuses, "warn")
    return MetadataPlan(1, "encode", False, items, requested, expected, {}, {}, {}, {}, None, (),
                        statuses, native_rows, (), nontag, (), "omitted", result, diagnostics)


@pytest.mark.parametrize("old,requested,effective,options,fate,loss", [
    ("x", "x", "x", {}, "kept", "none"),
    ("x", "y", "y", {}, "changed", "none"),
    (None, "y", "y", {}, "added", "none"),
    ("x", None, None, {"by_request": True}, "removed_by_request", "none"),
    ("x", "x", None, {"by_design": True, "by_request": True}, "removed_by_design", "none"),
    ("x", "x", None, {}, "dropped_format_limit", "whole_value"),
    (None, "2026-10", "2026", {"precision_detail": "id3_month_dropped"}, "dropped_format_limit", "precision"),
    ("2026-03", "2026", "2026", {}, "changed", "none"),
    ("2026-03", "2026-10", "2026", {"precision_detail": "id3_month_dropped"}, "dropped_format_limit", "precision"),
    (False, False, False, {}, "kept", "none"),
    (False, False, None, {}, "dropped_format_limit", "whole_value"),
    (None, False, False, {}, "added", "none"),
])
def test_rollup_precedence_and_request_loss(old, requested, effective, options, fate, loss):
    row = lc.rollup_status("field:date", source=old, requested=requested, effective=effective, **options)
    assert (row.status, row.loss_kind) == (fate, loss)


def test_no_byte_size_or_unequal_string_implies_precision():
    row = lc.rollup_status("field:title", source="long", requested="long", effective="short")
    assert (row.status, row.loss_kind) == ("changed", "none")
    with pytest.raises(ValueError, match="precision requires"):
        lc.rollup_status("field:title", source=None, requested="same", effective="same", precision_detail="invented")
    row = lc.rollup_status("field:title", source=None, requested="lost", effective=None, precision_detail="established")
    assert row.loss_kind == "whole_value"


def test_reference_equality_dereferences_values_not_ids_or_scopes():
    a = native(key="PRIV:owner")
    b = replace(a, identity="renamed", locator="elsewhere")
    inventories = {"source_native": (a,), "expected_native": (b,)}
    left, right = InventoryRef("source_native", (a.identity,)), InventoryRef("expected_native", (b.identity,))
    assert lc.rollup_equal(left, right, inventories)
    for attribute, value in [("store", "mp4"), ("key", "other"), ("native_type", "other"),
                             ("native_version", "2.4"), ("payload_encoding", "dn-raw-1"), ("payload", b"other")]:
        assert not lc.rollup_equal(left, right, {**inventories, "expected_native": (replace(b, **{attribute: value}),)})
    assert not lc.rollup_equal(left, "evidence", inventories)
    assert not lc.rollup_equal(True, 1, inventories)
    with pytest.raises(ValueError, match="unresolved"):
        lc.rollup_equal(InventoryRef("source_native", ("missing",)), None, inventories)
    with pytest.raises(ValueError, match="missing inventory"):
        lc.rollup_equal(left, None, {})
    with pytest.raises(ValueError, match="duplicate"):
        lc.inventory_index((a, a))


def test_duplicate_sequence_order_and_multiplicity_are_values():
    a, b = native(payload=b"a", identity="a"), native(payload=b"b", identity="b")
    inv = {"source_native": (a, b), "expected_native": (a, b)}
    left = InventoryRef("source_native", ("a", "b"))
    assert not lc.rollup_equal(left, InventoryRef("expected_native", ("b", "a")), inv)
    assert not lc.rollup_equal(left, InventoryRef("expected_native", ("a",)), inv)


def test_omitted_non_tag_hash_is_not_semantic_equality():
    item = NonTagItem("item:chunk:bext", "chunk", "bext@0", "", None, 2000000, "hash")
    ref = InventoryRef("non_tag_items", (item.identity,))
    assert not lc.rollup_equal(ref, ref, {"non_tag_items": (item,)})


@pytest.mark.parametrize("store,key", [("vorbis", "replaygain_track_gain"), ("vorbis", "R128_TRACK_GAIN"),
    ("id3v2", "RVA2:track"), ("id3v2", "RVAD"), ("id3v2", "COMM:iTunNORM:eng"),
    ("mp4", "----:com.apple.iTunes:iTunNORM"), ("id3v2", "TSSE"), ("mp4", "©too"),
    ("vorbis", "ENCODER"), ("riff-info", "ISFT"), ("apev2", "Unknown"),
    ("id3v1", "title"), ("aiff-text", "NAME"), ("id3v2", "TXXX:NORMA")])
def test_design_precedes_no_source_tags(store, key):
    item = native(store, key)
    assert lc.removed_by_design(item, "encode", legacy_projected_ids=(item.identity,))
    rows = lc.native_statuses((item,), (), (), audio_basis="encode", no_source_tags=True,
                             legacy_projected_ids=(item.identity,))
    assert rows[0].status == "removed_by_design"
    assert rows[0].loss_kind == "none"


@pytest.mark.parametrize("store,key", [("mp4", "----:com.apple.iTunes:iTunSMPB"),
    ("id3v2", "COMM:iTunSMPB:eng"), ("id3v2", "TXXX:iTunSMPB")])
def test_gapless_replans_from_copy_to_encode(store, key):
    item = native(store, key)
    for basis, fate in (("stream_copy", "kept"), ("encode", "removed_by_design")):
        carried = basis == "stream_copy"
        statuses = lc.other_statuses(source((item,)), target_stores=(store,), audio_basis=basis,
                                     requested_native=(item,) if carried else (), expected_native=(item,) if carried else (),
                                     links={item.identity: (item.identity,)} if carried else {})
        assert len(statuses) == 1 and statuses[0].status == fate
        assert lc.loss_policy(statuses, "error") == ("ok", ())


@pytest.mark.parametrize("target,expected_version,expected_payload,precision,fate,loss", [
    ("id3v2", "2.3", b"evidence", None, "kept", "none"),
    ("id3v2", "2.4", b"evidence", None, "changed", "none"),
    ("id3v2", "2.3", b"reshaped", "id3_reshape", "dropped_format_limit", "precision"),
    ("vorbis", None, None, None, "dropped_format_limit", "whole_value"),
])
def test_native_passthrough_preparation(target, expected_version, expected_payload, precision, fate, loss):
    item = native(key="PRIV:com.example")
    expected = (replace(item, native_version=expected_version, payload=expected_payload),) if expected_payload else ()
    requested = (item,) if expected else ()
    links = {item.identity: (item.identity,)} if expected else {}
    statuses = lc.other_statuses(source((item,)), target_stores=(target,), audio_basis="encode",
                                 requested_native=requested, expected_native=expected, links=links,
                                 precision_details={item.identity: precision} if precision else {})
    rows = lc.native_statuses((item,), expected, statuses, audio_basis="encode", links=links,
                              precision_details={item.identity: precision} if precision else {})
    assert (statuses[0].status, statuses[0].loss_kind) == (fate, loss)
    assert (rows[0].status, rows[0].loss_kind) == (fate, loss)
    candidate = plan((item,), expected, statuses, rows, requested=requested)
    assert candidate.source_native == (item,)
    if expected_version == "2.4":
        assert statuses[0].detail == "id3_version_converted"


def test_raw_evidence_only_and_design_exception():
    equ = native(key="EQU2:eq", encoding="dn-raw-1")
    rva = native(key="RVA2:track", encoding="dn-raw-1")
    statuses = lc.other_statuses(source((equ, rva)), target_stores=("id3v2",), audio_basis="encode")
    assert [(r.status, r.detail) for r in statuses] == [("dropped_format_limit", "raw_evidence_only"), ("removed_by_design", "")]
    rows = lc.native_statuses((equ, rva), (), statuses, audio_basis="encode")
    plan((equ, rva), (), statuses, rows)
    with pytest.raises(ValueError, match="raw fallback"):
        lc.other_statuses(source((equ,)), target_stores=("id3v2",), audio_basis="encode", requested_native=(equ,))


def test_invalid_typed_source_replaced_or_lost_without_invented_total():
    item = native(key="TRCK", payload=b"A1")
    for replacement in (None, 3):
        desired = {"track": rv(replacement, "manifest")} if replacement is not None else {}
        expected = (replace(item, payload=b"3"),) if desired else ()
        statuses = lc.scalar_statuses(source((item,)), desired, desired, expected_native=expected,
                                     source_field_ids={"field:track": (item.identity,)},
                                     expected_field_ids={"field:track": (item.identity,)} if expected else {})
        assert len(statuses) == 1
        assert statuses[0].status == ("changed" if desired else "dropped_format_limit")
        assert statuses[0].detail == ("" if desired else "invalid_source_value")
        assert isinstance(statuses[0].requested, int if desired else InventoryRef)
        rows = lc.native_statuses((item,), expected, statuses, audio_basis="encode",
                                  superseded_ids=(item.identity,) if desired else ())
        assert rows[0].status == statuses[0].status
        if desired:
            assert rows[0].detail == "superseded"
        plan((item,), expected, statuses, rows)
    with pytest.raises(ValueError, match="source_field_ids"):
        lc.scalar_statuses(source((item,)), {}, {})


def test_source_number_without_total_does_not_report_phantom_loss():
    item = native(key="TRCK", payload=b"3")
    old = source((item,), {"track": TagValue(3, "id3v2", item.identity)})
    desired = {"track": rv(3)}
    statuses = lc.scalar_statuses(old, desired, desired, expected_native=(item,),
                                 source_field_ids={"field:track": (item.identity,)},
                                 expected_field_ids={"field:track": (item.identity,)})
    assert [(r.key, r.status) for r in statuses] == [("field:track", "kept")]
    assert lc.loss_policy(statuses, "error") == ("ok", ())


def test_timestamp_request_preserved_only_when_source_wins():
    item = native("mp4", "©day", payload=b"2026-10-05T07:00:00Z")
    old = source((item,), {"date": TagValue("2026-10-05T07:00:00Z", "mp4", item.identity)})
    for provenance, fate in (("source", "dropped_format_limit"), ("manifest", "changed")):
        desired = {"date": rv("2026-10-05", provenance)}
        expected = (replace(item, payload=b"2026-10-05"),)
        statuses = lc.scalar_statuses(old, desired, desired, expected_native=expected)
        assert statuses[0].status == fate
        assert statuses[0].requested == (old.fields["date"].value if provenance == "source" else "2026-10-05")
        rows = lc.native_statuses((item,), expected, statuses, audio_basis="encode",
                                  superseded_ids=(item.identity,) if provenance != "source" else (),
                                  precision_details={item.identity: "time_of_day_dropped"} if provenance == "source" else {})
        assert rows[0].status == fate
        plan((item,), expected, statuses, rows)


def test_multi_occurrences_join_into_one_rollup_with_one_fate_each():
    items = (native("vorbis", "ARTIST", b"Alice", identity="store:vorbis:ARTIST#1"),
             native("vorbis", "ARTIST", b"Bob", identity="store:vorbis:ARTIST#2"))
    expected = (native("vorbis", "ARTIST", b"Alice; Bob"),)
    old = source(items, {"artist": TagValue("Alice; Bob", "vorbis", ",".join(i.identity for i in items))})
    desired = {"artist": rv("Alice; Bob")}
    statuses = lc.scalar_statuses(old, desired, desired, expected_native=expected)
    assert len(statuses) == 1 and statuses[0].status == "kept"
    rows = lc.native_statuses(items, expected, statuses, audio_basis="encode",
                              changed_details={i.identity: "multi_value_joined" for i in items})
    assert len(rows) == 2
    assert all(r.status == "changed" and r.detail == "multi_value_joined" for r in rows)
    plan(items, expected, statuses, rows)


def test_superseded_source_is_not_precision_loss_but_winning_request_can_be():
    item = native("vorbis", "DATE", b"2026-03")
    old = source((item,), {"date": TagValue("2026-03", "vorbis", item.identity)})
    expected = (native(key="TYER", payload=b"2026"),)
    for request, loss in (("2026", "none"), ("2026-10", "precision")):
        statuses = lc.scalar_statuses(old, {"date": rv(request, "manifest")}, {"date": rv("2026", "manifest")},
                                     expected_native=expected,
                                     precision_details={"field:date": "id3_month_dropped"} if loss == "precision" else {})
        rows = lc.native_statuses((item,), expected, statuses, audio_basis="encode", superseded_ids=(item.identity,))
        assert statuses[0].loss_kind == loss
        assert rows[0].status == "changed" and rows[0].loss_kind == "none" and rows[0].detail == "superseded"
        plan((item,), expected, statuses, rows)


def test_new_native_addition_and_precision_rollup_are_not_double_counted():
    expected = (native(key="TYER", payload=b"2026"), native("riff-info", "ICRD", b"2026-10\0", encoding="dn-raw-1"))
    statuses = lc.scalar_statuses(source(), {"date": rv("2026-10", "cli")}, {"date": rv("2026", "cli")},
                                 expected_native=expected, precision_details={"field:date": "id3_month_dropped"})
    rows = lc.native_statuses((), expected, statuses, audio_basis="encode")
    assert len(statuses) == 1 and len(rows) == 2
    assert all(r.status == "added" for r in rows)
    assert statuses[0].expected_native_ids == tuple(i.identity for i in expected)
    plan((), expected, statuses, rows)


def test_chapters_have_only_one_rollup_and_separate_native_fates():
    items = (native(key="CHAP:a"), native(key="CHAP:b"), native(key="CTOC:toc"))
    data = b'{"sources":[]}'
    chapters = NonTagItem("item:chapters", "chapters", "test", "", data, len(data), hashlib.sha256(data).hexdigest())
    old = source(items, nontag=(chapters,))
    statuses = lc.other_statuses(old, target_stores=("id3v2",), audio_basis="encode")
    assert len(statuses) == 1 and statuses[0].key == "item:chapters"
    assert statuses[0].source_native_ids == tuple(i.identity for i in items)
    rows = lc.native_statuses(items, (), statuses, audio_basis="encode")
    assert len(rows) == 3 and all(r.status == "dropped_format_limit" and r.loss_kind == "whole_value" for r in rows)
    plan(items, (), statuses, rows, nontag=(chapters,))
    with pytest.raises(ValueError, match="aggregate"):
        lc.other_statuses(source(items), target_stores=("id3v2",), audio_basis="encode")


@pytest.mark.parametrize("chunk,design", [("JUNK", True), ("junk", True), ("PAD ", True), ("FLLR", True),
    ("PEAK", True), ("peak", False), ("PAD_", False), ("bext", False), ("APPLICATION", False)])
def test_non_tag_closed_design_list(chunk, design):
    item = NonTagItem("item:chunk:" + chunk + "#2", "chunk", "test", "", b"", 0, hashlib.sha256(b"").hexdigest())
    statuses = lc.other_statuses(source(nontag=(item,)), target_stores=("id3v2",), audio_basis="encode")
    assert statuses[0].status == ("removed_by_design" if design else "dropped_format_limit")
    assert isinstance(statuses[0].requested, InventoryRef)
    assert statuses[0].effective is None
    assert lc.loss_policy(statuses, "error")[0] == ("ok" if design else "error")
    plan((), (), statuses, (), nontag=(item,))


def test_request_removal_and_new_output_not_linked_to_ignored_source():
    item = native()
    expected = (replace(item, payload=b"new"),)
    status = lc.rollup_status("field:title", source="old", requested="new", effective="new",
                             source_ids=(item.identity,), expected_ids=(item.identity,))
    rows = lc.native_statuses((item,), expected, (status,), audio_basis="encode", no_source_tags=True)
    assert [(r.status, r.expected_ids) for r in rows] == [("removed_by_request", ()), ("added", (item.identity,))]
    plan((item,), expected, (status,), rows)


def test_picture_fates_use_bytes_and_supplied_image_is_native_addition():
    a = Artwork(b"a", "image/png", 10, 10, 3, "")
    b = Artwork(b"b", "image/png", 10, 10, 3, "")
    assert lc.rollup_equal(a, replace(a, description="different"), {})
    for effective, fate in ((a, "kept"), (b, "changed"), (None, "dropped_format_limit")):
        row = lc.rollup_status("picture:1", source=a, requested=a, effective=effective)
        assert row.status == fate
    item, output = native(key="APIC:old"), native(key="APIC:new")
    old_row = lc.rollup_status("picture:1", source=a, requested=b, effective=b, source_ids=(item.identity,))
    new_row = lc.rollup_status("picture:new", source=None, requested=b, effective=b, expected_ids=(output.identity,))
    rows = lc.native_statuses((item,), (output,), (old_row, new_row), audio_basis="encode",
                              superseded_ids=(item.identity,), links={item.identity: ()})
    assert [r.status for r in rows] == ["changed", "added"]
    plan((item,), (output,), (old_row, new_row), rows)


@pytest.mark.parametrize("policy", ["warn", "sidecar", "error"])
def test_policy_only_rollups_and_precision_precedence(policy):
    precision = lc.rollup_status("field:date", source=None, requested="2026-10", effective="2026", precision_detail="id3_month_dropped")
    whole = lc.rollup_status("field:composer", source=None, requested="Alice", effective=None)
    result, diagnostics = lc.loss_policy((whole, precision), policy)
    assert result == policy
    assert [d.code for d in diagnostics] == (["metadata_loss_error"] if policy == "error" else ["metadata_precision_loss", "metadata_loss_warn"])
    assert all(d.stage == "metadata_planned" for d in diagnostics)
    design = lc.rollup_status("item:native:id3v2:RVA2", source="x", requested="x", effective=None, by_design=True)
    assert lc.loss_policy((design,), policy) == ("ok", ())
    with pytest.raises(ValueError, match="duplicate"):
        lc.loss_policy((whole, whole), policy)


def test_bad_links_and_unknown_occurrence_hints_are_rejected():
    item = native()
    with pytest.raises(ValueError, match="unknown source"):
        lc.native_statuses((item,), (), (), audio_basis="encode", superseded_ids=("missing",))
    with pytest.raises(ValueError, match="unresolved expected"):
        lc.native_statuses((item,), (), (), audio_basis="encode", links={item.identity: ("missing",)})
    row = lc.rollup_status("field:title", source="x", requested="x", effective="x", source_ids=("missing",))
    with pytest.raises(ValueError, match="unresolved roll-up"):
        lc.native_statuses((item,), (), (row,), audio_basis="encode")


def test_primary_precision_does_not_imply_every_native_store_loses_precision():
    item = native("riff-info", "ICRD", b"2026-10\0", encoding="dn-raw-1")
    year = native(key="TYER", payload=b"2026")
    old = source((item,), {"date": TagValue("2026-10", "riff-info", item.identity)})
    rows = lc.scalar_statuses(old, {"date": rv("2026-10")}, {"date": rv("2026")},
                             expected_native=(year, item), precision_details={"field:date": "id3_month_dropped"})
    native_rows = lc.native_statuses((item,), (year, item), rows, audio_basis="encode")
    assert rows[0].loss_kind == "precision"
    assert native_rows[0].status == "kept" and native_rows[0].loss_kind == "none"
    plan((item,), (year, item), rows, native_rows)


def test_extra_case_variants_have_only_one_rollup():
    item = native("vorbis", "MOOD", b"calm", encoding="dn-raw-1")
    old = source((item,), extras={"MOOD": TagValue("calm", "vorbis", item.identity)})
    desired = {"Mood": rv("calm")}
    rows = lc.scalar_statuses(old, {}, {}, extras=desired, effective_extras=desired, expected_native=(item,))
    assert [(r.key, r.status) for r in rows] == [("extra:Mood", "kept")]
    native_rows = lc.native_statuses((item,), (item,), rows, audio_basis="encode")
    plan((item,), (item,), rows, native_rows)


def test_clear_request_and_false_compilation_are_present_evidence():
    title, compilation = native(), native(key="TCMP", payload=b"0")
    old = source((title, compilation), {"title": TagValue("x", "id3v2", title.identity),
                                      "compilation": TagValue(False, "id3v2", compilation.identity)})
    rows = lc.scalar_statuses(old, {"compilation": rv(False)}, {"compilation": rv(False)},
                             expected_native=(compilation,), clear_keys=("field:title",))
    assert {r.key: r.status for r in rows} == {"field:compilation": "kept", "field:title": "removed_by_request"}
    native_rows = lc.native_statuses(old.native_inventory, (compilation,), rows, audio_basis="encode")
    assert [r.status for r in native_rows] == ["removed_by_request", "kept"]
    assert lc.loss_policy(rows, "error") == ("ok", ())
    plan(old.native_inventory, (compilation,), rows, native_rows)


def test_total_without_number_is_rollup_loss_even_with_no_native_item():
    rows = lc.scalar_statuses(source(), {"track_total": rv(12, "cli")}, {},
                             loss_details={"field:track_total": "total_without_number"})
    assert len(rows) == 1 and rows[0].loss_kind == "whole_value"
    assert rows[0].detail == "total_without_number" and rows[0].expected_native_ids == ()
    native_rows = lc.native_statuses((), (), rows, audio_basis="encode")
    assert native_rows == ()
    plan((), (), rows, native_rows)


def test_new_processing_mark_has_exactly_one_added_native_row():
    mark = native(key="TXXX:NORMA")
    rows = lc.native_statuses((), (mark,), (), audio_basis="encode")
    assert len(rows) == 1 and rows[0].key == "output:" + mark.identity
    plan((), (mark,), (), rows)


def test_no_source_tags_does_not_hide_loss_of_newly_supplied_value():
    item = native(key="TCOM")
    old = source((item,), {"composer": TagValue("old", "id3v2", item.identity)})
    desired = {"composer": rv("new", "cli")}
    lost = lc.scalar_statuses(old, desired, {}, no_source_tags=True)
    assert lost[0].status == "dropped_format_limit" and lost[0].loss_kind == "whole_value"
    assert lc.loss_policy(lost, "error")[0] == "error"
    added = lc.scalar_statuses(old, desired, desired, expected_native=(item,), no_source_tags=True)
    assert added[0].status == "added"
    rows = lc.native_statuses((item,), (item,), added, audio_basis="encode", no_source_tags=True)
    assert [r.status for r in rows] == ["removed_by_request", "added"]
    plan((item,), (item,), added, rows)


def test_passthrough_request_is_verbatim_source_evidence():
    item = native(key="PRIV:owner")
    with pytest.raises(ValueError, match="source native value"):
        lc.other_statuses(source((item,)), target_stores=("id3v2",), audio_basis="encode",
                          requested_native=(replace(item, payload=b"invented"),))


def test_same_family_conversion_can_drop_a_frame_whole():
    item = native(key="TMOO", version="2.4")
    statuses = lc.other_statuses(source((item,)), target_stores=("id3v2",), audio_basis="encode", requested_native=(item,))
    assert statuses[0].requested == InventoryRef("requested_native", (item.identity,))
    assert statuses[0].status == "dropped_format_limit" and statuses[0].loss_kind == "whole_value"
    rows = lc.native_statuses((item,), (), statuses, audio_basis="encode")
    plan((item,), (), statuses, rows, requested=(item,))
