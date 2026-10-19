"""Sidecar members staged by ``member_id`` (MD25, MD36, MD41; TST-M-02, TST-S-02).

The writer hands bytes back and never writes a sidecar or picture file itself:
WS-05 reserves and publishes them (C-08).
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace

import pytest

from norma.metadata import mapping, write
from norma.models import (
    Artwork, FieldStatus, InventoryRef, MetadataPlan, NativeItemStatus, NonTagItem, RawNativeItem, ResolvedValue,
    SidecarRequest,
)

JPEG = b"\xff\xd8\xff\xe0" + bytes(range(40))
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(40))
AUDIO = "01 Night Bus.m4a"
STAMP = "2026-10-05T07:00:00Z"
BIG = 1_048_577


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def alac_plan() -> MetadataPlan:
    """A7 plus a cross-family PRIV (A11), a dropped back cover, a bext chunk and an over-limit chunk."""
    priv = RawNativeItem("store:id3v2:PRIV:com.example", "id3v2", "PRIV:com.example", "PRIV", "2.4", "dn-json-1",
                         b'{"fields":[["owner","com.example"],["data",{"b64":"AAE="}]],"frame":"PRIV"}', "ID3@0+10")
    day = RawNativeItem("store:mp4:©day", "mp4", "©day", "©day", None, "dn-json-1",
                        json.dumps({"atom": "©day", "scalar": False, "values": [STAMP]}).encode(), "ilst@1")
    fields = {"title": "Night Bus", "date": "2026-10-05"}
    expected = write.prepare_store("mp4", mapping.native_recipes(
        "mp4", fields, extras={"BPM": "120"}, pictures=(Artwork(JPEG, "image/jpeg", None, None, 3, ""),))).items
    by_key = {item.key: item.identity for item in expected}
    bext = NonTagItem("item:chunk:bext", "chunk", "bext@36", "not_carried_v1", b"BWF", 3, _sha(b"BWF"))
    huge = NonTagItem("item:chunk:iXML", "chunk", "iXML@99", "not_carried_v1", None, BIG, "ab" * 32)
    back = Artwork(PNG, "image/png", None, None, 4, "back")
    statuses = (
        FieldStatus("field:title", "added", "Night Bus", "Night Bus", "cli", "none", (), (by_key["©nam"],), ""),
        FieldStatus("field:date", "dropped_format_limit", STAMP, "2026-10-05", "source", "precision",
                    (day.identity,), (by_key["©day"],), "time_of_day_dropped"),
        FieldStatus("field:comment", "removed_by_request", None, None, "cli", "none", (), (), ""),
        FieldStatus("item:native:id3v2:PRIV:com.example", "dropped_format_limit",
                    InventoryRef("source_native", (priv.identity,)), None, "source", "whole_value",
                    (priv.identity,), (), "no_cross_family_mapping"),
        FieldStatus("picture:1", "kept", Artwork(JPEG, "image/jpeg", None, None, 3, ""),
                    Artwork(JPEG, "image/jpeg", None, None, 3, ""), "source", "none", (), (by_key["covr"],), ""),
        FieldStatus("picture:2", "dropped_format_limit", back, None, "source", "whole_value", (), (),
                    "no_supported_mapping"),
        FieldStatus("item:chunk:bext", "dropped_format_limit", InventoryRef("non_tag_items", ("item:chunk:bext",)),
                    None, "source", "whole_value", (), (), "not_carried_v1"),
        FieldStatus("item:chunk:iXML", "dropped_format_limit", InventoryRef("non_tag_items", ("item:chunk:iXML",)),
                    None, "source", "whole_value", (), (), "not_carried_v1"),
    )
    native = (
        NativeItemStatus("source:" + priv.identity, "dropped_format_limit", (priv.identity,), (), "whole_value",
                         "no_cross_family_mapping"),
        NativeItemStatus("source:" + day.identity, "dropped_format_limit", (day.identity,), (by_key["©day"],),
                         "precision", "time_of_day_dropped"),
        *(NativeItemStatus("output:" + item.identity, "added", (), (item.identity,), "none", "")
          for item in expected if item.key != "©day"),
    )
    requests = (SidecarRequest("tags", "tags_sidecar", None, ".tags.json"),
                SidecarRequest("picture-2", "picture_export", "picture:2", ".picture-2.png"))
    return MetadataPlan(
        plan_revision=2, audio_basis="encode", conditional=False, source_native=(priv, day), requested_native=(),
        expected_native=expected,
        resolved={"title": ResolvedValue("Night Bus", "cli", None), "date": ResolvedValue(STAMP, "source", "x")},
        effective={"title": ResolvedValue("Night Bus", "cli", None), "date": ResolvedValue("2026-10-05", "source", None)},
        extras={"BPM": ResolvedValue("120", "manifest", None), "東京": ResolvedValue("x", "cli", None)},
        effective_extras={"BPM": ResolvedValue("120", "manifest", None)}, artwork=None,
        effective_artwork=(Artwork(JPEG, "image/jpeg", None, None, 3, ""),), statuses=statuses,
        native_statuses=native, conflicts=(), non_tag_items=(bext, huge), sidecar_requests=requests,
        mark_state="omitted", loss_policy_result="sidecar")


def test_tags_sidecar_holds_the_canonical_subset_and_dereferenced_evidence() -> None:
    plan = alac_plan()
    staged = write.stage_members(plan, audio_name=AUDIO, clears=("field:comment", "extra:Mood"), tool_version="9.9")
    assert list(staged) == ["tags", "picture-2"]
    assert staged["picture-2"] == PNG
    raw = staged["tags"]
    document = json.loads(raw)
    assert set(document) == {"sidecar_version", "tool", "audio", "plan_revision", "manifest", "losses",
                             "native_losses", "items"}
    assert document["sidecar_version"] == 1 and document["tool"] == {"name": "norma", "version": "9.9"}
    assert document["audio"] == AUDIO and document["plan_revision"] == 2
    # MD41: timestamp projected, clears null, retained cover omitted, illegal extra name omitted
    assert document["manifest"] == {"version": 1, "files": {AUDIO: {
        "title": "Night Bus", "date": "2026-10-05", "comment": None, "Mood": None, "BPM": "120"}}}
    losses = {entry["key"]: entry for entry in document["losses"]}
    assert set(losses) == {"field:date", "item:native:id3v2:PRIV:com.example", "picture:2", "item:chunk:bext",
                           "item:chunk:iXML"}
    assert losses["field:date"]["requested"] == STAMP and losses["field:date"]["effective"] == "2026-10-05"
    assert losses["field:date"]["loss_kind"] == "precision" and losses["field:date"]["detail"] == "time_of_day_dropped"
    ref = losses["item:native:id3v2:PRIV:com.example"]["requested"]
    assert ref["kind"] == "inventory_ref" and ref["scope"] == "source_native"
    (item,) = ref["items"]
    priv = plan.source_native[0]
    assert base64.b64decode(item["payload_b64"]) == priv.payload and item["payload_sha256"] == _sha(priv.payload)
    assert item["native_version"] == "2.4" and item["payload_encoding"] == "dn-json-1"
    assert losses["picture:2"]["requested"] == {"kind": "picture", "file": AUDIO + ".picture-2.png",
                                                "size": len(PNG), "sha256": _sha(PNG)}
    assert losses["item:chunk:iXML"]["requested"]["items"][0]["payload_omitted"] == "over_limit"
    assert "payload_b64" not in losses["item:chunk:iXML"]["requested"]["items"][0]
    natives = {entry["key"]: entry for entry in document["native_losses"]}
    assert set(natives) == {"source:store:id3v2:PRIV:com.example", "source:store:mp4:©day"}
    day = natives["source:store:mp4:©day"]["items"][0]
    assert STAMP in base64.b64decode(day["payload_b64"]).decode()  # evidence keeps the timestamp
    items = {entry["identity"]: entry for entry in document["items"]}
    assert base64.b64decode(items["item:chunk:bext"]["payload_b64"]) == b"BWF"
    assert items["item:chunk:iXML"]["payload_size"] == BIG and items["item:chunk:iXML"]["payload_omitted"] == "over_limit"
    assert raw.decode("utf-8") and "\\u" not in raw.decode("utf-8")  # ensure_ascii=False


def test_a_dropped_front_cover_is_exported_and_named_in_the_manifest() -> None:
    plan = alac_plan()
    cover = Artwork(JPEG, "image/jpeg", 600, 600, 3, "")
    statuses = tuple(status for status in plan.statuses if status.key != "picture:1") + (
        FieldStatus("picture:new", "dropped_format_limit", cover, None, "cli", "whole_value", (), (),
                    "no_supported_mapping"),)
    plan = replace(plan, statuses=statuses, sidecar_requests=plan.sidecar_requests + (
        SidecarRequest("cover", "cover_export", "picture:new", ".cover.jpg"),))
    staged = write.stage_members(plan, audio_name="x.wav", tool_version="1")
    assert list(staged) == ["tags", "picture-2", "cover"] and staged["cover"] == JPEG
    document = json.loads(staged["tags"])
    assert document["manifest"]["files"]["x.wav"]["artwork"] == "x.wav.cover.jpg"
    loss = next(entry for entry in document["losses"] if entry["key"] == "picture:new")
    assert loss["requested"]["file"] == "x.wav.cover.jpg"


def test_staging_refuses_bad_names_and_missing_pictures() -> None:
    plan = alac_plan()
    for name in ("", "a/b.m4a", ".."):
        with pytest.raises(ValueError):
            write.stage_members(plan, audio_name=name)
    missing = replace(plan, sidecar_requests=(SidecarRequest("picture-9", "picture_export", "picture:9", ".x"),))
    with pytest.raises(ValueError, match="no picture"):
        write.stage_members(missing, audio_name=AUDIO)


def test_write_tags_stages_members_only_when_read_back_matches(audio, tmp_path) -> None:
    plan = alac_plan()
    path = audio("alac")
    before = sorted(item.name for item in tmp_path.iterdir())
    outcome = write.write_tags(path, plan, target="alac", audio_name=AUDIO, tool_version="9.9")
    assert outcome.result.matched and outcome.result.staged_member_ids == ("tags", "picture-2")
    assert outcome.staged["picture-2"] == PNG and json.loads(outcome.staged["tags"])["audio"] == AUDIO
    assert sorted(item.name for item in tmp_path.iterdir()) == before  # nothing staged on disk
    wrong = replace(plan, effective={"title": ResolvedValue("Other", "cli", None)})
    outcome = write.write_tags(audio("alac"), wrong, target="alac", audio_name=AUDIO)
    assert not outcome.result.matched and outcome.result.staged_member_ids == () and outcome.staged == {}
