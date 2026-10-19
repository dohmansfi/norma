"""TST-M-03: R1–R10, W1–W14, pair/clear semantics and C-07 links."""

from dataclasses import replace
from typing import Any

import pytest

from norma.errors import FileFailure, UsageError
from norma.metadata import mapping, read, resolve, write
from norma.models import Artwork, ManifestCell, ResolvedValue, SourceTags, TagValue


def cell(value):
    return ManifestCell("clear", None) if value is None else ManifestCell("set", value)


def source(fields=None, extras=None, items=(), artwork=(), conflicts=()):
    fields = fields or {}
    values = {name: TagValue(value, "id3v2", next((item.identity for item in items
                                                if name in mapping.canonical_fields(item.store, item.key)), "test:" + name))
              for name, value in fields.items()}
    return SourceTags(values, {name: TagValue(value, "id3v2", "test:" + name) for name, value in (extras or {}).items()},
                      tuple(TagValue(picture, "id3v2", item.identity) for picture, item in artwork), (), conflicts, items,
                      {store: tuple(item.identity for item in items if item.store == store)
                       for store in dict.fromkeys(item.store for item in items)})


def plan(src, *, target="mp3", audio_basis="encode", on_metadata_loss="warn", wav_tags="info+id3",
         id3_version="2.3", no_mark_processed=False, **kwargs):
    resolution = resolve.with_audio_basis(resolve.resolve_opinions(src, stem="03 Glasshouse", **kwargs), audio_basis)
    requests = resolve.preparation_requests(src, resolution, target=target, wav_tags=wav_tags, id3_version=id3_version)
    prepared = tuple(write.prepare_store(store, request["recipes"], request["passthrough"], id3_version=id3_version)
                     for store, request in requests.items())
    return resolve.build_metadata_plan(src, resolution, target=target, wav_tags=wav_tags, id3_version=id3_version,
                                       prepared_stores=prepared, source_field_ids=read.source_field_ids(src),
                                       on_metadata_loss=on_metadata_loss, no_mark_processed=no_mark_processed), resolution


@pytest.mark.parametrize("row,prefer,old,explicit,inferred,mode,field,expected,provenance", [
    ("R1", "source", "old", None, "inferred", "override", "title", None, None),
    ("R2", "explicit", "old", "explicit", "inferred", "override", "title", "explicit", "cli"),
    ("R3", "explicit", "old", "absent", "inferred", "override", "title", "inferred", "filename"),
    ("R4", "explicit", "old", "absent", "inferred", "fill", "title", "old", "source"),
    ("R5", "explicit", None, "absent", "inferred", "fill", "title", "inferred", "filename"),
    ("R6", "source", "old", "explicit", "inferred", "override", "title", "old", "source"),
    ("R7", "source", None, "explicit", "inferred", "override", "title", "explicit", "cli"),
    ("R8", "source", None, "absent", "inferred", "fill", "title", "inferred", "filename"),
    ("R9", "explicit", None, "absent", None, "fill", "title", "03 Glasshouse", "fallback"),
    ("R10", "explicit", None, "absent", None, "fill", "artist", None, None),
])
def test_every_decision_row(row, prefer, old, explicit, inferred, mode, field, expected, provenance):
    result = resolve.resolve_opinions(source({field: old} if old is not None else {}), stem="03 Glasshouse",
                                      cli={} if explicit == "absent" else {field: cell(explicit)},
                                      filename={field: ResolvedValue(inferred, "filename", "pattern")} if inferred else {},
                                      prefer=prefer, infer_mode=mode)
    value = result["resolved"].get(field)
    assert ((value.value, value.provenance) if value else (None, None)) == (expected, provenance)


def artist_item(value="Dave G", version="2.3"):
    recipes = mapping.native_recipes("id3v2", {"artist": value}, id3_version=version)
    return write.prepare_store("id3v2", recipes, id3_version=version).items[0]


@pytest.mark.parametrize("w,old,kwargs,expected,provenance,count,fate", [
    ("W1", "Dave G", {"manifest_default": {"artist": cell("Test Artist")}}, "Test Artist", "manifest_default", 1, "changed"),
    ("W2", "Dave G", {"manifest_default": {"artist": cell("Test Artist")}, "prefer": "source"}, "Dave G", "source", 1, "kept"),
    ("W9", "Dave G", {"config": {"artist": cell("Test Artist")}, "manifest": {"artist": cell("Anna Rossi")}}, "Anna Rossi", "manifest", 1, "changed"),
    ("W12", "Anna Rossi", {"config": {"artist": cell("Test Artist")}, "manifest": {"artist": cell("Anna Rossi")}, "strict_conflicts": True}, "Anna Rossi", "manifest", 0, "kept"),
    ("W13", "Anna Rossi", {"config": {"artist": cell("Test Artist")}}, "Test Artist", "config", 1, "changed"),
    ("W14", "Anna Rossi", {"config": {"artist": cell("Test Artist")}, "cli": {"artist": cell("Anna Rossi")}, "strict_conflicts": True}, "Anna Rossi", "cli", 0, "kept"),
])
def test_artist_worked_examples(w, old, kwargs, expected, provenance, count, fate):
    item = artist_item(old)
    result, _ = plan(source({"artist": old}, items=(item,)), **kwargs)
    assert (result.resolved["artist"].value, result.resolved["artist"].provenance) == (expected, provenance)
    assert len(result.conflicts) == count
    assert next(row for row in result.statuses if row.key == "field:artist").status == fate
    assert next(row for row in result.native_statuses if row.key == "source:" + item.identity).status == fate


def test_w3_clear_beats_prefer_source_and_never_refills():
    item = write.prepare_store("id3v2", mapping.native_recipes("id3v2", {"comment": "old note"})).items[0]
    result, _ = plan(source({"comment": "old note"}, items=(item,)), manifest={"comment": cell(None)}, prefer="source")
    assert "comment" not in result.resolved and not result.conflicts
    assert next(row for row in result.native_statuses if row.key == "source:" + item.identity).status == "removed_by_request"


@pytest.mark.parametrize("layer", ["cli", "generated", "manifest", "manifest_default", "config"])
@pytest.mark.parametrize("prefer", ["explicit", "source"])
@pytest.mark.parametrize("kind", ["scalar", "extra", "picture"])
def test_clear_status_keeps_winning_layer_for_scalars_extras_and_pictures(layer, prefer, kind):
    picture = Artwork(b"cover", "image/jpeg", 1, 1, 3, "front")
    fields = {"comment": "old note"} if kind == "scalar" else {}
    extras = {"Mood": "old"} if kind == "extra" else {}
    pictures = (picture,) if kind == "picture" else ()
    items = write.prepare_store("id3v2", mapping.native_recipes("id3v2", fields, extras=extras,
                                                               pictures=pictures)).items
    src = source(fields, extras, items=items, artwork=((picture, items[0]),) if pictures else ())
    name = {"scalar": "comment", "extra": "MOOD", "picture": "artwork"}[kind]
    key = {"scalar": "field:comment", "extra": "extra:MOOD", "picture": "picture:1"}[kind]
    opinions: dict[str, Any] = {layer: {name: cell(None)}}
    result, resolution = plan(src, **opinions, prefer=prefer, no_title_fallback=True)
    row = next(row for row in result.statuses if row.key == key)
    assert (row.key, row.status, row.requested, row.effective, row.provenance, row.loss_kind) == (
        key, "removed_by_request", None, None, layer, "none")
    assert row.source_native_ids == (items[0].identity,) and row.expected_native_ids == ()
    assert result.expected_native == () and not result.conflicts and result.loss_policy_result == "ok"
    assert result.native_statuses[0].status == "removed_by_request"
    assert resolution["clear_provenance"] == {"extra:MOOD" if kind == "extra" else "field:" + name: layer}


def test_clear_provenance_follows_highest_explicit_layer_not_lower_clear_or_source():
    result, _ = plan(source({"comment": "old"}), manifest={"comment": cell(None)},
                      manifest_default={"comment": cell(None)}, config={"comment": cell("config")}, prefer="source")
    row = next(row for row in result.statuses if row.key == "field:comment")
    assert row.provenance == "manifest" and row.status == "removed_by_request"
    kept, resolution = plan(source({"comment": "old"}), cli={"comment": cell("kept")},
                              manifest={"comment": cell(None)})
    assert kept.resolved["comment"].value == "kept" and resolution["clear_provenance"] == {}


def test_clear_of_absent_field_has_provenance_without_creating_a_value():
    result, _ = plan(source(), manifest_default={"comment": cell(None)}, no_title_fallback=True)
    row, = result.statuses
    assert row.provenance == "manifest_default" and row.requested is None and row.effective is None
    assert result.resolved == {} and result.effective == {} and result.expected_native == ()


def test_no_source_tags_removal_is_not_an_explicit_clear():
    result, resolution = plan(source({"comment": "old"}), no_source_tags=True, no_title_fallback=True)
    row, = result.statuses
    assert row.status == "removed_by_request" and row.provenance == "source"
    assert resolution["clear_provenance"] == {}


def test_clear_provenance_also_covers_unparseable_native_pictures():
    picture = Artwork(b"cover", "image/jpeg", 1, 1, 3, "front")
    item = write.prepare_store("id3v2", mapping.native_recipes("id3v2", {}, pictures=(picture,))).items[0]
    # The reader can inventory a malformed APIC without projecting an Artwork.
    result, _ = plan(source(items=(item,)), manifest={"artwork": cell(None)}, no_title_fallback=True)
    native_picture = next(row for row in result.statuses if row.key.startswith("item:native:"))
    assert native_picture.provenance == "manifest" and native_picture.status == "removed_by_request"
    assert native_picture.requested is None and native_picture.effective is None
    assert native_picture.source_native_ids == (item.identity,)
    assert result.native_statuses[0].status == "removed_by_request" and result.loss_policy_result == "ok"


@pytest.mark.parametrize("w,title,mode,expected,conflicts", [("W4", None, "fill", "Glasshouse", 0),
    ("W5", "Bounce 7", "fill", "Bounce 7", 0), ("W6", "Bounce 7", "override", "Glasshouse", 1)])
def test_filename_worked_examples(w, title, mode, expected, conflicts):
    items = write.prepare_store("id3v2", mapping.native_recipes("id3v2", {"title": title})).items if title else ()
    result, resolution = plan(source({"title": title} if title else {}, items=items),
                              filename={"title": ResolvedValue("Glasshouse", "filename", "pattern"),
                                        "track": ResolvedValue(3, "filename", "pattern")}, infer_mode=mode)
    assert result.resolved["title"].value == expected and result.resolved["track"].value == 3
    assert len(result.conflicts) == conflicts
    if w == "W5":
        assert any("not used" in note for note in resolution["notes"])


def test_w7_explicit_layering_set_over_clear():
    result, resolution = plan(source(), manifest_default={"comment": cell("Demos 2026")},
                              manifest={"comment": cell(None)}, cli={"comment": cell("Mastered")})
    assert result.resolved["comment"].value == "Mastered"
    assert not result.conflicts and any("overrides clear" in note for note in resolution["notes"])


def test_w8_pair_total_discarded_without_format_loss():
    items = write.prepare_store("id3v2", mapping.native_recipes("id3v2", {"track": 3, "track_total": 12})).items
    result, _ = plan(source({"track": 3, "track_total": 12}, items=items), manifest={"track": cell(13)})
    assert result.resolved["track"].value == 13 and "track_total" not in result.resolved
    assert any(c.key == "field:track_total" and c.detail == "pair_inconsistent" for c in result.conflicts)
    assert next(row for row in result.statuses if row.key == "field:track_total").detail == "pair_inconsistent"
    assert next(row for row in result.native_statuses if row.key == "source:" + items[0].identity).status == "changed"
    assert result.loss_policy_result == "ok"


def test_w10_no_source_design_precedes_request():
    title = write.prepare_store("id3v2", mapping.native_recipes("id3v2", {"title": "Bounce 7"})).items[0]
    gain = replace(title, identity="store:id3v2:TXXX:REPLAYGAIN_TRACK_GAIN", key="TXXX:REPLAYGAIN_TRACK_GAIN")
    result, _ = plan(source({"title": "Bounce 7"}, items=(title, gain)), no_source_tags=True)
    assert result.resolved["title"].provenance == "fallback" and not result.conflicts
    assert [row.status for row in result.native_statuses[:2]] == ["removed_by_request", "removed_by_design"]


def test_w11_duplicate_cli_opinion_is_usage_error():
    with pytest.raises(UsageError):
        resolve.resolve_opinions(source(), stem="X", cli={"title": cell("X")}, generated={"title": cell(None)})


def test_w13_strict_conflict_fails_before_preparation(monkeypatch):
    monkeypatch.setattr(write, "prepare_store", lambda *a, **k: pytest.fail("strict failure must precede preparation/render"))
    with pytest.raises(FileFailure) as exc:
        plan(source({"artist": "Anna Rossi"}), config={"artist": cell("Test Artist")}, strict_conflicts=True)
    assert exc.value.reason == "metadata_conflict_strict" and exc.value.stage == "metadata_planned"


@pytest.mark.parametrize("number,total", [("track", "track_total"), ("disc", "disc_total")])
@pytest.mark.parametrize("prefer", ["explicit", "source"])
def test_pair_discards_lower_rank_in_both_directions(number, total, prefer):
    result = resolve.resolve_opinions(source({number: 13}), stem="X", manifest={total: cell(12)}, prefer=prefer)
    assert (number in result["resolved"], total in result["resolved"]) == ((False, True) if prefer == "explicit" else (True, False))
    assert len(result["pair_discarded"]) == 1 and result["conflicts"][-1].detail == "pair_inconsistent"


@pytest.mark.parametrize("prefer", ["explicit", "source"])
def test_cleared_title_blocks_fallback_and_inference(prefer):
    result = resolve.resolve_opinions(source(), stem="X", manifest={"title": cell(None)},
                                      filename={"title": ResolvedValue("inferred", "filename", "pattern")}, prefer=prefer)
    assert "title" not in result["resolved"] and result["clear_keys"] == ("field:title",)


def test_no_title_fallback_is_r10():
    result, _ = plan(source(), no_title_fallback=True)
    assert result.resolved == {} and result.expected_native == ()


def test_compilation_false_is_a_present_value():
    result, _ = plan(source({"compilation": False}), target="flac", cli={"compilation": cell(True)}, prefer="source")
    assert result.effective["compilation"].value is False
    assert len(result.conflicts) == 1 and result.conflicts[0].old_value is True


def test_category_conflicts_best_explicit_only_and_override_filename():
    result = resolve.resolve_opinions(source({"artist": "source"}), stem="X", config={"artist": cell("config")},
                                      manifest={"artist": cell("manifest")}, cli={"artist": cell("cli")},
                                      filename={"artist": ResolvedValue("filename", "filename", "pattern")}, infer_mode="override")
    assert [c.old_value for c in result["conflicts"]] == ["source", "filename"]


def test_extra_spelling_casefold_clear_and_normalised_text():
    result, _ = plan(source(extras={"Mood": "old"}), cli={"extra:MOOD": cell("  cafe\u0301  ")})
    assert result.extras["MOOD"].value == "café"
    assert len([row for row in result.statuses if row.key.startswith("extra:")]) == 1
    cleared, _ = plan(source(extras={"Mood": "old"}), cli={"extra:MOOD": cell(None)})
    assert cleared.extras == {} and next(row for row in cleared.statuses if row.key == "extra:MOOD").status == "removed_by_request"


def test_case_different_extra_clear_removes_the_native_occurrence():
    items = write.prepare_store("id3v2", mapping.native_recipes("id3v2", {}, extras={"Mood": "old"})).items
    cleared, _ = plan(source(extras={"Mood": "old"}, items=items), cli={"MOOD": cell(None)}, no_title_fallback=True)
    row = next(row for row in cleared.statuses if row.key == "extra:MOOD")
    assert row.source_native_ids == (items[0].identity,)
    assert len(cleared.statuses) == 1 and len(cleared.native_statuses) == 1
    assert cleared.native_statuses[0].status == "removed_by_request" and cleared.loss_policy_result == "ok"


@pytest.mark.parametrize("bad", [True, 0, -1, "3"])
def test_invalid_constructed_integer_opinions_rejected(bad):
    with pytest.raises(ValueError):
        resolve.resolve_opinions(source(), stem="X", manifest={"track": cell(bad)})


def test_input_pair_validation_and_no_source_preference_validation():
    with pytest.raises(ValueError, match="validate input pairs"):
        resolve.resolve_opinions(source(), stem="X", cli={"track": cell(13), "track_total": cell(12)})
    with pytest.raises(UsageError):
        resolve.resolve_opinions(source(), stem="X", no_source_tags=True, prefer="source")


@pytest.mark.parametrize("target,wav_tags", [("mp3", "info+id3"), ("aiff", "info+id3"), ("flac", "info+id3"),
    ("alac", "info+id3"), ("aac", "info+id3"), ("wav", "info+id3"), ("wav", "info"), ("wav", "none")])
def test_complete_plan_all_targets_and_native_accounting(target, wav_tags):
    result, _ = plan(source(), target=target, wav_tags=wav_tags, cli={"date": cell("2026-10"), "compilation": cell(False),
                                                                 "track_total": cell(12), "extra:BPM": cell("120")})
    assert len(result.native_statuses) == len(result.expected_native)
    assert all(row.status == "added" for row in result.native_statuses)
    assert ("track_total" in result.effective) == (target in ("alac", "aac", "flac"))
    assert result.resolved["compilation"].value is False


def test_losing_source_precision_is_not_loss_and_winning_precision_is():
    items = write.prepare_store("vorbis", mapping.native_recipes("vorbis", {"date": "2026-03"})).items
    src = source({"date": "2026-03"}, items=items)
    changed, _ = plan(src, manifest={"date": cell("2026")})
    assert changed.loss_policy_result == "ok"
    precision, _ = plan(src, manifest={"date": cell("2026-10")})
    assert precision.loss_policy_result == "warn"
    assert next(row for row in precision.statuses if row.key == "field:date").loss_kind == "precision"
    assert precision.native_statuses[0].detail == "superseded"


def test_loss_error_fails_during_planning():
    with pytest.raises(FileFailure) as exc:
        plan(source(), target="wav", wav_tags="none", on_metadata_loss="error", cli={"artist": cell("X")})
    assert exc.value.reason == "metadata_loss_error"


def test_source_timestamp_stays_in_request_not_resolved():
    text = "2026-10-05T07:00:00Z"
    item = write.prepare_store("mp4", mapping.native_recipes("mp4", {"date": "2026-10-05"})).items[0]
    result, _ = plan(source({"date": text}, items=(item,)), target="alac")
    assert result.resolved["date"].value == "2026-10-05"
    date = next(row for row in result.statuses if row.key == "field:date")
    assert (date.requested, date.effective, date.detail) == (text, "2026-10-05", "time_of_day_dropped")


def test_clear_number_never_clears_independent_total():
    items = write.prepare_store("mp4", mapping.native_recipes("mp4", {"track": 3, "track_total": 12})).items
    result, _ = plan(source({"track": 3, "track_total": 12}, items=items), target="alac", cli={"track": cell(None)})
    assert "track" not in result.effective and result.effective["track_total"].value == 12


def test_generated_rank_and_folder_artwork_rank():
    cover = Artwork(b"cover", "image/jpeg", 1, 1, 3, "")
    result = resolve.resolve_opinions(source(), stem="X", generated={"track": cell(1), "track_total": cell(5)},
                                      manifest={"track": cell(2), "artwork": cell(cover)}, folder_artwork=cell(None))
    assert result["resolved"]["track"].provenance == "generated"
    assert result["artwork"].provenance == "manifest"


def test_supplied_cover_replaces_only_front_covers():
    front = Artwork(b"front", "image/jpeg", 1, 1, 3, "front")
    back = Artwork(b"back", "image/png", 1, 1, 4, "back")
    new = Artwork(b"new", "image/jpeg", 1, 1, 3, "new")
    items = write.prepare_store("id3v2", mapping.native_recipes("id3v2", {}, pictures=(front, back))).items
    by_key = {item.key: item for item in items}
    src = source(items=items, artwork=((front, by_key["APIC:front"]), (back, by_key["APIC:back"])))
    result, _ = plan(src, cli={"artwork": cell(new)}, no_title_fallback=True)
    assert result.effective_artwork == (new, back)
    pictures = {row.key: row for row in result.statuses}
    assert pictures["picture:1"].status == "changed" and pictures["picture:2"].status == "kept"
    assert pictures["picture:new"].status == "added"
    assert any(row.key.startswith("output:") and "APIC:new" in row.key for row in result.native_statuses)


def test_picture_clear_and_sidecar_requests():
    picture = Artwork(b"cover", "image/jpeg", 1, 1, 3, "")
    result, _ = plan(source(), target="wav", wav_tags="none", cli={"artwork": cell(picture)}, on_metadata_loss="sidecar")
    assert [(r.member_id, r.purpose, r.inventory_key, r.suffix) for r in result.sidecar_requests] == [
        ("tags", "tags_sidecar", None, ".tags.json"), ("cover", "cover_export", "picture:new", ".cover.jpg")]


def test_prepared_link_corruption_is_never_silent():
    src = source()
    resolution = resolve.resolve_opinions(src, stem="X")
    request = resolve.preparation_requests(src, resolution, target="mp3")["id3v2"]
    prepared = write.prepare_store("id3v2", request["recipes"])
    with pytest.raises(ValueError, match="unresolved prepared recipe"):
        resolve.build_metadata_plan(src, resolution, target="mp3", prepared_stores=(replace(prepared, recipe_ids=(("missing",),)),), source_field_ids={})
    with pytest.raises(ValueError, match="primary order"):
        resolve.build_metadata_plan(src, resolution, target="mp3", prepared_stores=(), source_field_ids={})


@pytest.mark.parametrize("target", ["mp3", "aiff", "flac", "alac", "aac"])
@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_actual_plan_write_readback(audio, target, version):
    candidate = audio(target)
    result, _ = plan(source(), target=target, id3_version=version, no_mark_processed=True,
                      cli={"title": cell("Perché 東京"), "artist": cell("Anna"), "date": cell("2026-10"),
                           "track": cell(3), "track_total": cell(12), "compilation": cell(False), "extra:MOOD": cell("calm")})
    outcome = write.write_tags(candidate, result, target=target, id3_version=version)
    assert outcome.result.matched, outcome.result.mismatches
    assert outcome.result.plan_revision == result.plan_revision
    assert outcome.result.read_back.fields["compilation"].value is False
    assert outcome.result.read_back.fields["date"].value == result.effective["date"].value


def test_converted_embedded_cover_uses_exact_source_index():
    one = Artwork(b"one", "image/gif", 1, 1, 3, "one")
    two = Artwork(b"two", "image/gif", 1, 1, 3, "two")
    prepared_one = replace(one, data=b"one-jpeg", mime="image/jpeg")
    prepared_two = replace(two, data=b"two-jpeg", mime="image/jpeg")
    items = write.prepare_store("id3v2", mapping.native_recipes("id3v2", {}, pictures=(one, two))).items
    by_key = {item.key: item for item in items}
    src = source(items=items, artwork=((one, by_key["APIC:one"]), (two, by_key["APIC:two"])))
    result, _ = plan(src, target="alac", source_pictures=(prepared_one, prepared_two), no_title_fallback=True)
    assert result.effective_artwork == (prepared_one, prepared_two)
    pictures = [row for row in result.statuses if row.key.startswith("picture:")]
    assert [row.effective.data for row in pictures] == [b"one-jpeg", b"two-jpeg"]
    assert all(row.status == "changed" for row in result.native_statuses)


def test_source_timestamp_date_projection_is_conflict_comparison_value():
    resolution = resolve.resolve_opinions(source({"date": "2026-10-05T07:00:00Z"}), stem="X",
                                           cli={"date": cell("2026-10-05")}, strict_conflicts=True)
    assert resolution["conflicts"] == ()


def test_missing_projected_source_evidence_is_rejected():
    item = artist_item("Anna")
    src = source({"artist": "Anna"}, items=(item,))
    resolution = resolve.resolve_opinions(src, stem="X")
    request = resolve.preparation_requests(src, resolution, target="mp3")["id3v2"]
    prepared = write.prepare_store("id3v2", request["recipes"])
    with pytest.raises(ValueError, match="omits projected"):
        resolve.build_metadata_plan(src, resolution, target="mp3", prepared_stores=(prepared,), source_field_ids={})


def test_empty_native_pair_is_accounted_without_a_zero_field():
    items = read.native_items_from_bytes("id3v2", b"ID3\x04\x00\x00\x00\x00\x00\x0cTRCK\x00\x00\x00\x02\x00\x00\x030")
    src = source(items=items)
    result, _ = plan(src, no_title_fallback=True)
    assert result.resolved == {} and result.effective == {}
    assert len(result.native_statuses) == 1
    assert result.native_statuses[0].detail == "empty_source_value"


@pytest.mark.parametrize("wav_tags", ["info", "info+id3"])
def test_source_info_native_subset_is_kept_not_falsely_lost(audio, build, wav_tags):
    path = audio("wav")
    build.append_chunks(path, [build.info_list([(b"ICOP", b"Copyright source\x00"), (b"ILNG", b"eng\x00")])])
    tags = read.read_source_tags(path, "wav")
    result, _ = plan(tags, target="wav", wav_tags=wav_tags, no_title_fallback=True)
    assert {name: value.value for name, value in result.effective_extras.items()} == {"ICOP": "Copyright source", "ILNG": "eng"}
    assert result.loss_policy_result == "ok"
    assert {item.key for item in result.expected_native if item.store == "riff-info"} == {"ICOP", "ILNG"}


def test_explicit_info_extra_is_not_general_info_tag_support():
    result, _ = plan(source(), target="wav", wav_tags="info", cli={"extra:ICOP": cell("new")}, no_title_fallback=True)
    assert result.effective_extras == {} and result.statuses[0].status == "dropped_format_limit"


def test_strict_pair_conflict_fails_and_equal_rank_pair_is_invalid():
    with pytest.raises(FileFailure) as exc:
        resolve.resolve_opinions(source({"track": 3, "track_total": 12}), stem="X",
                                  manifest={"track": cell(13)}, strict_conflicts=True)
    assert exc.value.reason == "metadata_conflict_strict"
    with pytest.raises(ValueError, match="equal-rank"):
        resolve.resolve_opinions(source({"track": 13, "track_total": 12}), stem="X")


def test_tk0611_real_manifest_and_inference_opinions_interoperate(tmp_path):
    from pathlib import PurePosixPath
    from norma.metadata import infer, manifest

    data = manifest.parse_manifest(b'{"version":1,"defaults":{"extras":{"Mood":"default"}},'
                                   b'"files":{"03 Glasshouse.wav":{"extras":{"MOOD":"row"}}}}', tmp_path / "tags.json")
    inferred = infer.infer(infer.compile_pattern("{track} {title}"), PurePosixPath("03 Glasshouse.wav"))
    result, _ = plan(source(extras={"mood": "source"}), manifest=data.rows[0].cells, manifest_default=data.defaults,
                      filename=inferred.cells, locators={"manifest": data.rows[0].locator})
    assert result.extras["MOOD"].value == "row" and len(result.extras) == 1
    assert result.resolved["title"].value == "Glasshouse" and result.resolved["track"].value == 3
    assert len([c for c in result.conflicts if c.key.startswith("extra:")]) == 1
    row = manifest.parse_manifest(b'{"version":1,"files":{"x.wav":{"extras":{"mOoD":null}}}}', tmp_path / "clear.json")
    cleared, _ = plan(source(extras={"mood": "source"}), manifest=row.rows[0].cells)
    assert cleared.extras == {} and any(status.key == "extra:mOoD" and status.status == "removed_by_request"
                                       for status in cleared.statuses)
