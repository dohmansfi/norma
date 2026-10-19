"""CR-083: lost passthrough occurrences must consume empty preparation links."""

import base64
from dataclasses import replace

import pytest

from norma.metadata import losscheck, read, resolve, write
from norma.models import RawNativeItem, SourceTags, TagValue


def source(items):
    return SourceTags({}, {}, (), (), (), items,
                      {store: tuple(item.identity for item in items if item.store == store)
                       for store in dict.fromkeys(item.store for item in items)})


def item(store, key, payload, version=None, native_type=None):
    return RawNativeItem("store:" + store + ":" + key, store, key, native_type or key.split(":")[0], version,
                         "dn-json-1" if store in ("id3v2", "mp4") else "dn-raw-1", payload, "test")


def priv(data, identity):
    document = {"frame": "PRIV", "fields": [["owner", "com.example"], ["data", {"b64": base64.b64encode(data).decode()}]]}
    return replace(item("id3v2", "PRIV:com.example", read.canonical_json(document), "2.3"), identity=identity)


@pytest.mark.parametrize("store,target", [("id3v2", "mp3"), ("mp4", "alac")])
def test_duplicate_same_family_passthrough_has_exactly_one_fate_per_occurrence(store, target):
    if store == "id3v2":
        items = (priv(b"first", "store:id3v2:PRIV:com.example#1"), priv(b"first", "store:id3v2:PRIV:com.example#2"))
    else:
        items = tuple(replace(item("mp4", "©lyr", read.canonical_json({"atom": "©lyr", "scalar": False, "values": [text]})),
                              identity=f"store:mp4:©lyr#{n}") for n, text in enumerate(("first", "second"), 1))
    src = source(items)
    resolution = resolve.resolve_opinions(src, stem="X", no_title_fallback=True)
    request = resolve.preparation_requests(src, resolution, target=target)[store]
    prepared = write.prepare_store(store, request["recipes"], request["passthrough"])
    assert prepared.passthrough_ids[items[1].identity] == ()
    result = resolve.build_metadata_plan(src, resolution, target=target, prepared_stores=(prepared,), source_field_ids={})
    assert result.requested_native == items and len(result.expected_native) == 1
    assert len(result.native_statuses) == 2 and result.native_statuses[1].status == "dropped_format_limit"
    assert result.native_statuses[1].expected_ids == () and result.native_statuses[1].loss_kind == "whole_value"
    assert result.native_statuses[1].detail == "native_conversion_removed"
    assert len(result.statuses) == 2 and result.statuses[1].status == "dropped_format_limit"
    assert result.loss_policy_result == "warn" and [d.code for d in result.diagnostics] == ["metadata_loss_warn"]
    assert result.statuses[1].requested.scope == "requested_native"
    # Without consuming the edge's empty link, this would silently derive the
    # second occurrence from the first, or fabricate a second expected item.
    with pytest.raises(ValueError, match="exactly one.*producer"):
        wrong = replace(prepared, passthrough_ids={identity: (prepared.items[0].identity,) for identity in prepared.passthrough_ids})
        resolve.build_metadata_plan(src, resolution, target=target, prepared_stores=(wrong,), source_field_ids={})


@pytest.mark.parametrize("policy,expected", [("warn", "warn"), ("sidecar", "sidecar"), ("error", "error")])
def test_empty_passthrough_link_is_policy_input_once(policy, expected):
    a, b = priv(b"a", "store:id3v2:PRIV:com.example#1"), priv(b"a", "store:id3v2:PRIV:com.example#2")
    prepared = write.prepare_store("id3v2", passthrough=(a, b))
    rows = losscheck.other_statuses(source((a, b)), target_stores=("id3v2",), audio_basis="encode",
                                   requested_native=(a, b), expected_native=prepared.items, links=prepared.passthrough_ids)
    result, diagnostics = losscheck.loss_policy(rows, policy)
    assert result == expected and len(diagnostics) == 1
    assert diagnostics[0].detail == rows[1].key
    natives = losscheck.native_statuses((a, b), prepared.items, rows, audio_basis="encode", links=prepared.passthrough_ids)
    assert [row.key for row in natives] == ["source:" + a.identity, "source:" + b.identity]


def test_conversion_removed_frame_has_explicit_loss_not_kept():
    frame = {"frame": "TMOO", "fields": [["encoding", 3], ["text", ["calm"]]]}
    native = item("id3v2", "TMOO", read.canonical_json(frame), "2.4")
    src = source((native,))
    resolution = resolve.resolve_opinions(src, stem="X", no_title_fallback=True)
    request = resolve.preparation_requests(src, resolution, target="mp3")["id3v2"]
    prepared = write.prepare_store("id3v2", request["recipes"], request["passthrough"])
    result = resolve.build_metadata_plan(src, resolution, target="mp3", prepared_stores=(prepared,), source_field_ids={})
    assert result.expected_native == () and result.native_statuses[0].status == "dropped_format_limit"


def test_gapless_copy_to_encode_rebuilds_links_and_references():
    doc = {"frame": "COMM", "fields": [["encoding", 1], ["lang", "eng"], ["desc", "iTunSMPB"], ["text", ["gapless"]]]}
    native = item("id3v2", "COMM:iTunSMPB:eng", read.canonical_json(doc), "2.3")
    src = source((native,))
    initial = resolve.resolve_opinions(src, stem="X", no_title_fallback=True)
    plans = []
    for basis in ("stream_copy", "encode"):
        resolution = resolve.with_audio_basis(initial, basis)
        request = resolve.preparation_requests(src, resolution, target="mp3")["id3v2"]
        prepared = write.prepare_store("id3v2", request["recipes"], request["passthrough"])
        plans.append(resolve.build_metadata_plan(src, resolution, target="mp3", prepared_stores=(prepared,), source_field_ids={},
                                                 plan_revision=len(plans) + 1, conditional=True))
    assert plans[0].native_statuses[0].status == "kept" and plans[0].expected_native
    assert plans[1].native_statuses[0].status == "removed_by_design" and plans[1].expected_native == ()
    assert plans[0].plan_revision == 1 and plans[1].plan_revision == 2
    assert all(plan.loss_policy_result == "ok" and plan.conditional for plan in plans)


def test_pair_repair_with_no_expected_occurrence_is_changed_not_storage_loss():
    native = item("vorbis", "TRACKTOTAL", b"12", native_type="VORBIS_COMMENT")
    row = losscheck.rollup_status("field:track_total", source=12, requested=None, effective=None,
                                 source_ids=(native.identity,), detail="pair_inconsistent")
    row = replace(row, status="changed")
    result = losscheck.native_statuses((native,), (), (row,), audio_basis="encode",
                                       changed_details={native.identity: "pair_inconsistent"})
    assert result[0].status == "changed" and result[0].loss_kind == "none"


def test_scalar_clear_keeps_case_different_preparation_evidence():
    native = item("vorbis", "Mood", b"old", native_type="VORBIS_COMMENT")
    src = replace(source((native,)), extras={"Mood": TagValue("old", "vorbis", native.identity)})
    rows = losscheck.scalar_statuses(src, {}, {}, clear_keys=("extra:MOOD",),
                                     source_field_ids={"extra:Mood": (native.identity,)})
    assert len(rows) == 1 and rows[0].key == "extra:MOOD" and rows[0].source_native_ids == (native.identity,)
    assert rows[0].status == "removed_by_request"
