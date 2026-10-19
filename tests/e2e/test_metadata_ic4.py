"""IC-4 N-5: a metadata sidecar must never replace a manifest read by the run."""
from __future__ import annotations

import json

from mutagen.id3 import PRIV
from mutagen.wave import WAVE

from tests.e2e.test_metadata_support import document, normalise


def test_m02_sidecar_cannot_overwrite_its_manifest(norma, tools, tmp_path):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    tags = WAVE(source)
    tags.add_tags()
    tags.tags.add(PRIV(owner="ic4-manifest-protection", data=b"precious native evidence"))
    tags.save(v2_version=4)
    source_before = tools.snapshot(tmp_path)[source.name]

    output_folder = tmp_path / "out"
    output_folder.mkdir()
    audio = output_folder / "take.flac"
    manifest = output_folder / "take.flac.tags.json"
    manifest_bytes = (json.dumps({"version": 1, "defaults": {"title": "Manifest title"},
                                 "files": {"../take.wav": {}}}, indent=2) + "\n").encode()
    manifest.write_bytes(manifest_bytes)
    options = ("-f", "flac", "--on-metadata-loss", "sidecar", "--manifest", manifest,
               "--overwrite", "--no-mark-processed")

    # Prove that this exact input/options combination really needs a sidecar;
    # a loss-free or invalid-manifest refusal would not exercise the barrier.
    _, control, control_audio = normalise(norma, source, tmp_path / "control", *options)
    assert control["dropped"] and control["sidecar"]
    assert control_audio.is_file()
    sidecar = document(control_audio.with_name(control_audio.name + ".tags.json"))
    assert sidecar["manifest"]["files"][control_audio.name]["title"] == "Manifest title"
    assert manifest.read_bytes() == manifest_bytes

    report = tmp_path / "collision.json"
    run = norma("normalise", source, "--config", "/dev/null", "-o", output_folder,
                "--report", report, *options)
    assert run.code == 1, run.stdout + run.stderr
    outcome, = document(report)["outcomes"]
    assert outcome["reason"] == "unsafe_output", outcome
    assert outcome["published_paths"] == []
    assert manifest.read_bytes() == manifest_bytes
    assert not audio.exists()
    assert sorted(path.name for path in output_folder.iterdir()) == [manifest.name]
    assert tools.snapshot(tmp_path)[source.name] == source_before
