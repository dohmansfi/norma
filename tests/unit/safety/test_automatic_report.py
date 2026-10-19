"""TST-S-02/TST-S-05 — automatic report collisions take an ordinal (safety § 6, cli § 7.3, CR-049, D-069).

An automatic ``normalise`` report never refuses and never replaces: at preflight a taken base
name is no conflict; at publication a taken name gets ``-2``, then ``-3`` …, each new name
rechecked against the write set and protected reads and published exclusively. ``--overwrite``
never applies to it. Explicit reports keep their rules. Every tree is built inside tmp_path;
racers are injected at the publication primitive, never by timing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

import pytest

from norma import safety
from norma.errors import UsageError
from norma.models import InputItem, ReadDependency
from norma.paths import comparison_key
from norma.safety import Preflight, RunOutput, RunRegistry, preflight, protect, protect_inputs

BASE = "norma-report-2026-10-10T143012.csv"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path.resolve() / "demos"
    folder.mkdir()
    return folder


def tree_state(root: Path) -> dict[str, tuple[int, int, int, str]]:
    """Name → (mode, inode, mtime, content hash) for every entry, links not followed."""
    rows: dict[str, tuple[int, int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not path.is_symlink() else ""
        rows[str(path.relative_to(root))] = (info.st_mode, info.st_ino, info.st_mtime_ns, digest)
    return rows


def take(root: Path, name: str = "take.wav", data: bytes = b"RIFF source") -> InputItem:
    source = root / name
    source.write_bytes(data)
    info = source.stat()
    return InputItem(source, root, PurePosixPath(name), (info.st_dev, info.st_ino), 0, True)


def automatic(root: Path, name: str = BASE) -> RunOutput:
    return RunOutput("report", root / name, parent_must_exist=False)


def check(root: Path, outputs: list[RunOutput], *, overwrite: bool = False,
          items: list[InputItem] | None = None,
          extra: tuple[ReadDependency, ...] = ()) -> tuple[Preflight, tuple[ReadDependency, ...]]:
    reads = protect_inputs(items if items is not None else [take(root)]) + extra
    return preflight({}, outputs, reads, overwrite=overwrite), reads


def publish(checked: Preflight, reads: tuple[ReadDependency, ...], data: bytes = b"rows\n", *,
            overwrite: bool = False, ordinal_retry: bool = True, registry: RunRegistry | None = None,
            reserved: tuple[str, ...] | None = None) -> Path:
    destination = checked.write_set.destinations[0]
    keys = tuple(d.comparison_key for d in checked.write_set.destinations) if reserved is None else reserved
    return safety.publish_run_output(registry or RunRegistry("run1"), destination, data, overwrite=overwrite,
                                     protected=reads, reserved_keys=keys, ordinal_retry=ordinal_retry,
                                     create_parent=ordinal_retry, pin=checked.pins.get(destination.destination_id))


def racer(monkeypatch: pytest.MonkeyPatch, names: dict[str, bytes]) -> list[str]:
    """Create ``names`` (file name → bytes) the moment publication first tries each one."""
    real = safety.publish_no_clobber
    tried: list[str] = []

    def wrapped(candidate: Path, final: Path) -> bool:
        tried.append(final.name)
        if final.name in names and not final.exists():
            final.write_bytes(names[final.name])  # the racer wins between the check and the publish
        return real(candidate, final)

    monkeypatch.setattr(safety, "publish_no_clobber", wrapped)
    return tried


# --- The marker ---------------------------------------------------------------------------------


def test_only_a_report_whose_folder_may_be_created_is_automatic(root: Path) -> None:
    assert automatic(root).automatic
    assert not RunOutput("report", root / "r.csv").automatic
    assert not RunOutput("template_export", root / "t.csv", parent_must_exist=False).automatic
    assert not RunOutput("log", root / "norma.log", parent_must_exist=False).automatic


# --- Preflight: a taken base name is no conflict ---------------------------------------------------


@pytest.mark.parametrize("overwrite", [False, True])
def test_preflight_accepts_a_taken_automatic_base_name_and_writes_nothing(root: Path, overwrite: bool) -> None:
    item = take(root)
    (root / BASE).write_bytes(b"earlier report\n")
    before = tree_state(root)
    checked, _reads = check(root, [automatic(root)], overwrite=overwrite, items=[item])
    assert checked.occupants["report"].kind == "regular"
    assert [d.final_path for d in checked.write_set.destinations] == [root / BASE]
    assert tree_state(root) == before


@pytest.mark.parametrize("kind", ["symlink", "directory", "special", "input_alias"])
def test_preflight_accepts_any_occupant_of_the_automatic_base_name(root: Path, kind: str,
                                                                   request: pytest.FixtureRequest) -> None:
    if kind == "input_alias":
        request.getfixturevalue("allow_linked_names")  # the suite guard refuses renames onto linked names
    item = take(root)
    final = root / BASE
    if kind == "symlink":
        final.symlink_to(item.source_path)
    elif kind == "directory":
        final.mkdir()
    elif kind == "special":
        os.mkfifo(final)
    else:
        os.link(item.source_path, final)  # another name for the input this run reads
    before = tree_state(root)
    checked, reads = check(root, [automatic(root)], items=[item])
    assert checked.occupants["report"].kind == ("regular" if kind == "input_alias" else kind)
    written = publish(checked, reads)
    assert written == root / BASE.replace(".csv", "-2.csv") and written.read_bytes() == b"rows\n"
    after = tree_state(root)
    assert {name: after[name] for name in before} == before  # the occupant and the input are untouched
    assert set(after) - set(before) == {written.name}


def test_preflight_accepts_an_automatic_base_name_equal_to_a_protected_read(root: Path) -> None:
    item = take(root)
    config = root / "norma-report.csv"
    config.write_bytes(b"cfg")
    checked, reads = check(root, [automatic(root, "NORMA-REPORT.csv")], items=[item],
                           extra=(protect(config, "config"),))
    written = publish(checked, reads)
    assert written == root / "NORMA-REPORT-2.csv" and config.read_bytes() == b"cfg"


def test_an_automatic_report_inside_the_write_set_is_still_refused(root: Path) -> None:
    item = take(root)
    before = tree_state(root)
    with pytest.raises(UsageError) as caught:
        check(root, [automatic(root), RunOutput("log", root / BASE.upper())], items=[item])
    assert caught.value.exit_code == 2 and "also the report" in str(caught.value)
    assert tree_state(root) == before


def test_an_automatic_report_whose_ancestry_is_unsafe_is_still_refused(root: Path) -> None:
    item = take(root)
    (root / "blocker").write_bytes(b"a file where a folder would go")
    with pytest.raises(UsageError) as caught:
        check(root, [RunOutput("report", root / "blocker" / "Normalised" / BASE, parent_must_exist=False)],
              items=[item])
    assert caught.value.exit_code == 2


# --- Explicit reports keep today's rules -------------------------------------------------------------


def test_an_existing_explicit_report_still_needs_overwrite(root: Path) -> None:
    item = take(root)
    (root / "levels.csv").write_bytes(b"old")
    with pytest.raises(UsageError) as caught:
        check(root, [RunOutput("report", root / "levels.csv")], items=[item])
    assert "--overwrite" in str(caught.value)
    checked, reads = check(root, [RunOutput("report", root / "levels.csv")], overwrite=True, items=[item])
    written = publish(checked, reads, b"new", overwrite=True, ordinal_retry=False)
    assert written == root / "levels.csv" and written.read_bytes() == b"new"


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_an_explicit_report_over_an_unsafe_occupant_is_still_refused(root: Path, kind: str) -> None:
    item = take(root)
    final = root / "levels.csv"
    final.symlink_to(item.source_path) if kind == "symlink" else final.mkdir()
    with pytest.raises(UsageError):
        check(root, [RunOutput("report", final)], overwrite=True, items=[item])


def test_an_explicit_report_racer_without_overwrite_is_kept_and_is_a_run_error(
        root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checked, reads = check(root, [RunOutput("report", root / "levels.csv")])
    racer(monkeypatch, {"levels.csv": b"racer"})
    registry = RunRegistry("run1")
    with pytest.raises(safety.RunOutputError) as caught:
        publish(checked, reads, ordinal_retry=False, registry=registry)
    assert caught.value.exit_code == 1
    assert (root / "levels.csv").read_bytes() == b"racer" and registry.records() == ()
    assert sorted(p.name for p in root.iterdir()) == ["levels.csv", "take.wav"]


def test_an_explicit_report_racer_with_overwrite_is_still_replaced(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checked, reads = check(root, [RunOutput("report", root / "levels.csv")], overwrite=True)
    racer(monkeypatch, {"levels.csv": b"racer"})
    written = publish(checked, reads, b"new", overwrite=True, ordinal_retry=False)
    assert written == root / "levels.csv" and written.read_bytes() == b"new"
    assert sorted(p.name for p in root.iterdir()) == ["levels.csv", "take.wav"]


# --- Publication: ordinals from -2, never a replacement -----------------------------------------------


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_taken_base_name_gets_ordinal_2(root: Path, overwrite: bool) -> None:
    (root / BASE).write_bytes(b"earlier report\n")
    checked, reads = check(root, [automatic(root)], overwrite=overwrite)
    registry = RunRegistry("run1")
    written = publish(checked, reads, overwrite=overwrite, registry=registry)
    assert written == root / "norma-report-2026-10-10T143012-2.csv" and written.read_bytes() == b"rows\n"
    assert (root / BASE).read_bytes() == b"earlier report\n"  # with --overwrite too: never replaced
    assert registry.records() == ()
    assert {p.name for p in root.iterdir()} == {BASE, written.name, "take.wav"}


@pytest.mark.parametrize("overwrite", [False, True])
def test_ordinal_2_taken_gives_ordinal_3_and_leaves_both_untouched(root: Path, overwrite: bool) -> None:
    (root / BASE).write_bytes(b"first\n")
    second = root / BASE.replace(".csv", "-2.csv")
    second.write_bytes(b"second\n")
    checked, reads = check(root, [automatic(root)], overwrite=overwrite)
    before = tree_state(root)
    written = publish(checked, reads, overwrite=overwrite)
    assert written == root / BASE.replace(".csv", "-3.csv") and written.read_bytes() == b"rows\n"
    after = tree_state(root)
    assert {name: after[name] for name in before} == before
    assert not (root / BASE.replace(".csv", "-1.csv")).exists()  # there is no -1


def test_a_free_base_name_is_used_as_is(root: Path) -> None:
    checked, reads = check(root, [automatic(root)], overwrite=True)
    assert publish(checked, reads, overwrite=True) == root / BASE


def test_a_base_name_taken_after_preflight_gets_an_ordinal_and_the_racer_is_kept(
        root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checked, reads = check(root, [automatic(root)], overwrite=True)
    assert checked.occupants["report"].kind == "absent"
    tried = racer(monkeypatch, {BASE: b"racer\n"})
    written = publish(checked, reads, overwrite=True)
    assert tried == [BASE, BASE.replace(".csv", "-2.csv")]
    assert written.name == BASE.replace(".csv", "-2.csv") and (root / BASE).read_bytes() == b"racer\n"


def test_an_ordinal_taken_between_its_check_and_its_publish_moves_on(
        root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (root / BASE).write_bytes(b"first\n")
    checked, reads = check(root, [automatic(root)], overwrite=True)
    racer(monkeypatch, {BASE.replace(".csv", "-2.csv"): b"racer 2\n", BASE.replace(".csv", "-3.csv"): b"racer 3\n"})
    registry = RunRegistry("run1")
    written = publish(checked, reads, overwrite=True, registry=registry)
    assert written.name == BASE.replace(".csv", "-4.csv") and written.read_bytes() == b"rows\n"
    assert (root / BASE).read_bytes() == b"first\n"
    assert (root / BASE.replace(".csv", "-2.csv")).read_bytes() == b"racer 2\n"
    assert (root / BASE.replace(".csv", "-3.csv")).read_bytes() == b"racer 3\n"
    assert registry.records() == ()


def test_ordinals_held_by_the_write_set_or_a_protected_read_are_skipped_untried(
        root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = take(root)
    (root / BASE).write_bytes(b"first\n")
    config = root / BASE.replace(".csv", "-3.csv")
    config.write_bytes(b"cfg")
    checked, reads = check(root, [automatic(root)], items=[item], extra=(protect(config, "config"),))
    tried = racer(monkeypatch, {})
    reserved = (comparison_key(root / BASE), comparison_key(root / BASE.replace(".csv", "-2.csv").upper()))
    written = publish(checked, reads, reserved=reserved)
    assert tried == [BASE, BASE.replace(".csv", "-4.csv")]
    assert written.name == BASE.replace(".csv", "-4.csv") and config.read_bytes() == b"cfg"
    assert not (root / BASE.replace(".csv", "-2.csv")).exists()


def test_the_base_name_vanishing_after_preflight_is_simply_used(root: Path) -> None:
    (root / BASE).write_bytes(b"earlier\n")
    checked, reads = check(root, [automatic(root)])
    (root / BASE).unlink()
    assert publish(checked, reads) == root / BASE


def test_no_free_ordinal_is_a_run_error_and_leaves_no_candidate(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "_MAX_ORDINAL", 3)
    for name in (BASE, BASE.replace(".csv", "-2.csv"), BASE.replace(".csv", "-3.csv")):
        (root / name).write_bytes(name.encode())
    checked, reads = check(root, [automatic(root)], overwrite=True)
    before = tree_state(root)
    registry = RunRegistry("run1")
    with pytest.raises(safety.RunOutputError) as caught:
        publish(checked, reads, overwrite=True, registry=registry)
    assert caught.value.exit_code == 1 and "no free report name" in str(caught.value)
    assert tree_state(root) == before and registry.records() == ()


def test_the_hard_link_route_also_takes_an_ordinal(root: Path, monkeypatch: pytest.MonkeyPatch,
                                                    allow_linked_names: object) -> None:
    monkeypatch.setattr(safety, "_RENAMEX", None)
    (root / BASE).write_bytes(b"earlier\n")
    checked, reads = check(root, [automatic(root)], overwrite=True)
    registry = RunRegistry("run1")
    written = publish(checked, reads, overwrite=True, registry=registry)
    assert written.name == BASE.replace(".csv", "-2.csv") and written.stat().st_nlink == 1
    assert (root / BASE).read_bytes() == b"earlier\n" and registry.records() == ()


def test_an_automatic_report_in_a_new_folder_with_a_taken_name(root: Path) -> None:
    folder = root / "Normalised"
    folder.mkdir()
    (folder / BASE).write_bytes(b"earlier\n")
    checked, reads = check(root, [automatic(folder)])
    written = publish(checked, reads)
    assert written == folder / BASE.replace(".csv", "-2.csv") and (folder / BASE).read_bytes() == b"earlier\n"

