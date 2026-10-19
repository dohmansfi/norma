"""TST-S-01 / TST-S-05 (discovery slice) — signature, structure and probe classification (safety § 2.5, § 8).

Containers are hand-written bytes in tmp_path; the probe is either a scripted fake ffprobe or,
where installed, the real one on a hand-written WAV.
"""

from __future__ import annotations

import errno
import hashlib
import os
import struct
from pathlib import Path

import pytest

from norma import discovery
from norma.adapters.ffprobe import locate_ffprobe
from norma.discovery import check_signature, classify, discover
from norma.errors import FileFailure, SkipFile

REAL_FFPROBE = Path("/usr/local/bin/ffprobe")


def item_for(path: Path):
    (item,) = discover([path]).items
    return item


def outcome(path: Path):
    """The C-13 (status, reason) of the signature step for ``path``, or the identified kind."""
    try:
        return check_signature(item_for(path)).kind
    except SkipFile as exc:
        return ("skipped", exc.reason)
    except FileFailure as exc:
        return ("failed", exc.reason)


# --- Recognised signatures -----------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("wav", "wav"), ("aiff", "aiff"), ("aifc", "aiff"), ("flac", "flac"), ("mp4", "mp4"), ("caf", "caf"),
    ("mp3", "mpeg"), ("adts", "adts"),
])
def test_every_recognised_signature_is_identified(tmp_path: Path, build, name: str, expected: str) -> None:
    data = {"wav": build.wav(), "aiff": build.aiff(), "aifc": build.aiff(aifc="NONE"), "flac": build.flac,
            "mp4": build.mp4, "caf": build.caf, "mp3": build.mp3, "adts": build.adts}[name]
    path = tmp_path / "x.bin"  # the extension never decides
    path.write_bytes(data)
    assert outcome(path) == expected


@pytest.mark.parametrize("magic", [b"RIFF", b"RF64", b"BW64"])
def test_wave_signature_accepts_all_riff_family_magics(tmp_path: Path, build, magic: bytes) -> None:
    data = bytearray(build.wav())
    data[:4] = magic
    path = tmp_path / "a.wav"
    path.write_bytes(bytes(data))
    # RF64/BW64 need a ds64 chunk: without one the size semantics are unsupported, never a guess.
    assert outcome(path) == ("wav" if magic == b"RIFF" else ("failed", "probe_error"))


def test_signature_and_extension_need_not_agree(tmp_path: Path, build) -> None:
    path = tmp_path / "really-flac.wav"
    path.write_bytes(build.flac)
    assert outcome(path) == "flac"


def test_id3_tag_then_flac_or_frame_sync(tmp_path: Path, build) -> None:
    for payload, kind in ((b"fLaC" + bytes(32), "flac"), (build.mp3, "mpeg"), (build.adts, "adts")):
        path = tmp_path / f"{kind}.mp3"
        path.write_bytes(build.id3(payload))
        assert outcome(path) == kind
    footer = tmp_path / "footer.mp3"
    footer.write_bytes(build.id3(build.mp3, footer=True))
    assert outcome(footer) == "mpeg"


def test_id3_tag_followed_by_neither_is_not_audio(tmp_path: Path, build) -> None:
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(build.id3(b"plain text, not audio" * 4))
    cut = tmp_path / "cut.mp3"
    cut.write_bytes(build.id3(b"", tag_size=2000)[:100])  # tag runs past the end of the file
    bad_size = tmp_path / "bad.mp3"
    bad_size.write_bytes(b"ID3\x04\x00\x00\xff\xff\xff\xff" + bytes(64))
    assert [outcome(p) for p in (junk, cut, bad_size)] == [("skipped", "not_audio")] * 3


def test_implausible_frame_headers_are_not_audio(tmp_path: Path) -> None:
    for number, head in enumerate((b"\xff\xe1\x90\x00", b"\xff\xfb\xf0\x00", b"\xff\xfb\x9c\x00", b"\xff\xf1\x3c\x00")):
        path = tmp_path / f"{number}.mp3"
        path.write_bytes(head + bytes(32))
        assert outcome(path) == ("skipped", "not_audio")


# --- Not audio and unreadable --------------------------------------------------------


@pytest.mark.parametrize("content", [b"", b"just some text, named like audio\n", bytes(range(256)), b"RIFF", b"RIFF\0\0\0\0AVI "])
def test_non_audio_content_is_skipped_not_audio(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "fake.wav"
    path.write_bytes(content)
    assert outcome(path) == ("skipped", "not_audio")


def test_denied_read_is_permission_denied_never_not_audio(tmp_path: Path, build) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can read everything")
    path = tmp_path / "locked.wav"
    path.write_bytes(build.wav())
    item = item_for(path)
    path.chmod(0)
    try:
        with pytest.raises(FileFailure) as caught:
            check_signature(item)
    finally:
        path.chmod(0o600)
    assert caught.value.reason == "permission_denied" and caught.value.stage == "signature"


def test_vanished_item_is_source_changed(tmp_path: Path, build) -> None:
    path = tmp_path / "gone.wav"
    path.write_bytes(build.wav())
    item = item_for(path)
    path.unlink()
    with pytest.raises(FileFailure) as caught:
        check_signature(item)
    assert caught.value.reason == "source_changed"


def test_replaced_item_is_source_changed(tmp_path: Path, build) -> None:
    path = tmp_path / "swap.wav"
    path.write_bytes(build.wav())
    item = item_for(path)
    other = tmp_path / "other.wav"
    other.write_bytes(build.wav())
    os.replace(other, path)  # same name, different inode
    with pytest.raises(FileFailure) as caught:
        check_signature(item)
    assert caught.value.reason == "source_changed"


def test_item_replaced_by_symlink_or_fifo_is_source_changed_without_hanging(tmp_path: Path, build) -> None:
    for kind in ("symlink", "fifo"):
        path = tmp_path / f"{kind}.wav"
        path.write_bytes(build.wav())
        item = item_for(path)
        path.unlink()
        if kind == "symlink":
            real = tmp_path / "real.wav"
            real.write_bytes(build.wav())
            path.symlink_to(real)
        else:
            os.mkfifo(path)
        with pytest.raises(FileFailure) as caught:
            check_signature(item)
        assert caught.value.reason == "source_changed", kind


def test_other_read_errors_are_probe_error(tmp_path: Path, build, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "eio.wav"
    path.write_bytes(build.wav())
    item = item_for(path)

    def broken(*args: object) -> bytes:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(discovery.os, "pread", broken)
    with pytest.raises(FileFailure) as caught:
        check_signature(item)
    assert caught.value.reason == "probe_error"


def test_read_denied_after_open_is_permission_denied(tmp_path: Path, build, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "late.wav"
    path.write_bytes(build.wav())
    item = item_for(path)

    def denied(*args: object) -> bytes:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(discovery.os, "pread", denied)
    with pytest.raises(FileFailure) as caught:
        check_signature(item)
    assert caught.value.reason == "permission_denied"


def test_the_signature_step_changes_nothing(tmp_path: Path, build) -> None:
    folder = tmp_path / "only"
    folder.mkdir()
    path = folder / "keep.wav"
    path.write_bytes(build.wav())
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns, path.stat().st_atime_ns)
    check_signature(item_for(path))
    after = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns, path.stat().st_atime_ns)
    assert after[:2] == before[:2]  # content and mtime; access times are not compared (safety § 1)
    assert sorted(p.name for p in folder.iterdir()) == ["keep.wav"]


# --- WAVE structure (§ 8) ------------------------------------------------------------


def riff(chunks: bytes, *, declared: int | None = None, magic: bytes = b"RIFF") -> bytes:
    body = b"WAVE" + chunks
    return magic + struct.pack("<I", len(body) if declared is None else declared) + body


def fmt_chunk(*, tag: int = 1, channels: int = 2, align: int = 4, bits: int = 16, size: int | None = None) -> bytes:
    body = struct.pack("<HHIIHH", tag, channels, 44100, 44100 * align, align, bits)
    return b"fmt " + struct.pack("<I", len(body) if size is None else size) + body


def extensible_fmt() -> bytes:
    body = struct.pack("<HHIIHH", 0xFFFE, 2, 44100, 176400, 4, 16) + struct.pack("<HHI", 22, 16, 3)
    body += struct.pack("<H", 1) + bytes(14)  # sub-format GUID starts with the real tag (1 = PCM)
    return b"fmt " + struct.pack("<I", len(body)) + body


def data_chunk(payload: bytes, *, declared: int | None = None) -> bytes:
    return b"data" + struct.pack("<I", len(payload) if declared is None else declared) + payload


def test_valid_wave_passes_including_trailing_chunks_and_odd_padding(tmp_path: Path) -> None:
    list_chunk = b"LIST" + struct.pack("<I", 5) + b"abcde" + b"\0"
    for name, chunks in {
        "plain": fmt_chunk() + data_chunk(bytes(400)),
        "trailing": fmt_chunk() + data_chunk(bytes(400)) + list_chunk,
        "mono8": fmt_chunk(channels=1, align=1, bits=8) + data_chunk(bytes(101)) + b"\0",
        "extensible": extensible_fmt() + data_chunk(bytes(400)),
    }.items():
        path = tmp_path / f"{name}.wav"
        path.write_bytes(riff(chunks))
        assert outcome(path) == "wav", name


def test_odd_final_chunk_without_its_pad_byte_is_accepted(tmp_path: Path) -> None:
    chunks = fmt_chunk(channels=1, align=1, bits=8) + data_chunk(bytes(101))
    path = tmp_path / "nopad.wav"
    path.write_bytes(riff(chunks, declared=4 + len(chunks) + 1))  # the form counts a pad byte that is absent
    assert outcome(path) == "wav"


def test_declared_data_longer_than_the_file_is_truncated(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    path.write_bytes(riff(fmt_chunk() + data_chunk(bytes(400), declared=4000), declared=4 + 24 + 8 + 4000))
    assert outcome(path) == ("failed", "truncated")


def test_form_longer_than_the_file_is_truncated_even_if_chunks_are_whole(tmp_path: Path) -> None:
    chunks = fmt_chunk() + data_chunk(bytes(400))
    path = tmp_path / "form.wav"
    path.write_bytes(riff(chunks, declared=4 + len(chunks) + 1000))
    assert outcome(path) == ("failed", "truncated")


def test_file_cut_inside_or_before_the_chunks_is_truncated(tmp_path: Path) -> None:
    whole = riff(fmt_chunk() + data_chunk(bytes(400)))
    for cut in (13, 20, 30, 36, 60, len(whole) - 1):
        path = tmp_path / f"cut{cut}.wav"
        path.write_bytes(whole[:cut])
        assert outcome(path) == ("failed", "truncated"), cut


def test_malformed_wave_without_a_proved_shortfall_is_probe_error(tmp_path: Path) -> None:
    cases = {
        "no-data": riff(fmt_chunk()),
        "no-fmt": riff(data_chunk(bytes(400))),
        "misaligned": riff(fmt_chunk() + data_chunk(bytes(401))),
        "zero-align": riff(fmt_chunk(align=0) + data_chunk(bytes(400))),
        "zero-channels": riff(fmt_chunk(channels=0) + data_chunk(bytes(400))),
        "short-fmt": riff(b"fmt " + struct.pack("<I", 8) + bytes(8) + data_chunk(bytes(400))),
        "streamed-riff": riff(fmt_chunk() + data_chunk(bytes(400)), declared=0xFFFFFFFF),
        "zero-riff": riff(fmt_chunk() + data_chunk(bytes(400)), declared=0),
        "streamed-data": riff(fmt_chunk() + data_chunk(bytes(400), declared=0xFFFFFFFF)),
    }
    for name, data in cases.items():
        path = tmp_path / f"{name}.wav"
        path.write_bytes(data)
        assert outcome(path) == ("failed", "probe_error"), name


def test_compressed_wave_skips_the_block_alignment_check(tmp_path: Path) -> None:
    path = tmp_path / "adpcm.wav"
    path.write_bytes(riff(fmt_chunk(tag=2, align=3, bits=4) + data_chunk(bytes(100))))
    assert outcome(path) == "wav"


def test_rf64_uses_ds64_sizes(tmp_path: Path) -> None:
    payload = bytes(400)
    body_tail = fmt_chunk() + b"data" + struct.pack("<I", 0xFFFFFFFF) + payload

    def ds64(riff_size: int, data_size: int) -> bytes:
        return b"ds64" + struct.pack("<I", 28) + struct.pack("<QQQI", riff_size, data_size, 100, 0)

    total = 4 + 36 + len(body_tail)
    good = tmp_path / "good.wav"
    good.write_bytes(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64(total, 400) + body_tail)
    assert outcome(good) == "wav"

    short = tmp_path / "short.wav"
    short.write_bytes(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64(total + 5000, 5400) + body_tail)
    assert outcome(short) == ("failed", "truncated")

    missing = tmp_path / "missing.wav"
    missing.write_bytes(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + body_tail)
    assert outcome(missing) == ("failed", "probe_error")  # unsupported size semantics: never a guessed truncation


# --- AIFF structure (§ 8) ------------------------------------------------------------


def test_valid_aiff_and_aifc_pass(tmp_path: Path, build) -> None:
    cases = {
        "16": build.aiff(), "24": build.aiff(bits=24, channels=1), "odd-bits": build.aiff(bits=12, frames=7),
        "odd-length": build.aiff(frames=3, channels=1, bits=8), "none": build.aiff(aifc="NONE"),
        "sowt": build.aiff(aifc="sowt"), "float": build.aiff(aifc="fl32", bits=32),
    }
    for name, data in cases.items():
        path = tmp_path / f"{name}.aif"
        path.write_bytes(data)
        assert outcome(path) == "aiff", name


def test_aiff_cut_anywhere_is_truncated(tmp_path: Path, build) -> None:
    whole = build.aiff(frames=100)
    for cut in (13, 20, 40, 60, len(whole) - 1):
        path = tmp_path / f"cut{cut}.aif"
        path.write_bytes(whole[:cut])
        assert outcome(path) == ("failed", "truncated"), cut


def test_aiff_comm_frames_beyond_the_payload_is_truncated(tmp_path: Path, build) -> None:
    data = bytearray(build.aiff(frames=100))
    comm = data.index(b"COMM") + 8
    data[comm + 2:comm + 6] = struct.pack(">I", 5000)  # COMM claims more frames than the file holds
    path = tmp_path / "frames.aif"
    path.write_bytes(bytes(data))
    assert outcome(path) == ("failed", "truncated")


def test_aiff_ssnd_declaring_less_than_comm_needs_is_probe_error(tmp_path: Path, build) -> None:
    data = bytearray(build.aiff(frames=100))
    ssnd = data.index(b"SSND")
    data[ssnd + 4:ssnd + 8] = struct.pack(">I", 8 + 40)  # chunk ends early, bytes remain in the file
    path = tmp_path / "ssnd.aif"
    path.write_bytes(bytes(data))
    assert outcome(path) == ("failed", "probe_error")


def test_aiff_malformed_structure_is_probe_error(tmp_path: Path, build) -> None:
    whole = bytearray(build.aiff())
    no_comm = bytes(whole).replace(b"COMM", b"JUNK")
    bad_offset = bytearray(whole)
    ssnd = bad_offset.index(b"SSND") + 8
    bad_offset[ssnd:ssnd + 4] = struct.pack(">I", 100000)
    zero_channels = bytearray(whole)
    comm = zero_channels.index(b"COMM") + 8
    zero_channels[comm:comm + 2] = b"\0\0"
    no_ssnd = bytes(whole).replace(b"SSND", b"JUNK")
    short_comm = bytearray(whole)
    short_comm[comm - 4:comm] = struct.pack(">I", 6)
    for name, data in {"no-comm": no_comm, "offset": bytes(bad_offset), "channels": bytes(zero_channels),
                       "no-ssnd": no_ssnd, "short-comm": bytes(short_comm)}.items():
        path = tmp_path / f"{name}.aif"
        path.write_bytes(data)
        assert outcome(path) == ("failed", "probe_error"), name


def test_compressed_aifc_checks_extents_only(tmp_path: Path, build) -> None:
    path = tmp_path / "ima4.aifc"
    data = bytearray(build.aiff(aifc="ima4", frames=100))
    comm = data.index(b"COMM") + 8
    data[comm + 2:comm + 6] = struct.pack(">I", 99999)  # frame arithmetic does not apply to compressed audio
    path.write_bytes(bytes(data))
    assert outcome(path) == "aiff"


# --- classify: signature, then the adapter probe -------------------------------------


def test_classify_returns_the_probe_for_a_good_file(tmp_path: Path, build, fake_probe) -> None:
    path = tmp_path / "good.wav"
    path.write_bytes(build.wav())
    fake_probe.answer("good.wav", build.document(build.stream()))
    result = classify(item_for(path), fake_probe.prober)
    assert result.codec == "pcm_s16le" and result.codec_class == "lossless" and result.container == "wav"


def test_no_signature_skips_before_the_probe_runs(tmp_path: Path, fake_probe) -> None:
    path = tmp_path / "notes.wav"
    path.write_text("not audio", encoding="utf-8")  # the fake has no answer: running it would be probe_error
    with pytest.raises(SkipFile) as caught:
        classify(item_for(path), fake_probe.prober)
    assert caught.value.reason == "not_audio" and caught.value.stage == "signature"


def test_structural_failures_stop_before_the_probe(tmp_path: Path, build, fake_probe) -> None:
    path = tmp_path / "short.wav"
    path.write_bytes(build.wav()[:-50])
    fake_probe.answer("short.wav", build.document(build.stream()))  # a probe would happily succeed
    with pytest.raises(FileFailure) as caught:
        classify(item_for(path), fake_probe.prober)
    assert caught.value.reason == "truncated"


def test_probe_without_audio_stream_is_not_audio(tmp_path: Path, build, fake_probe) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(build.mp4)
    fake_probe.answer("video.mp4", build.document({"index": 0, "codec_type": "video", "codec_name": "h264"},
                                                  format_name="mov,mp4,m4a,3gp,3g2,mj2"))
    with pytest.raises(SkipFile) as caught:
        classify(item_for(path), fake_probe.prober)
    assert caught.value.reason == "not_audio"


def test_unsupported_codec_is_skipped(tmp_path: Path, build, fake_probe) -> None:
    path = tmp_path / "drm.m4a"
    path.write_bytes(build.mp4)
    fake_probe.answer("drm.m4a", build.document(build.stream(codec_name="pcm_alaw", sample_fmt="s16"),
                                                format_name="mov,mp4,m4a,3gp,3g2,mj2"))
    with pytest.raises(SkipFile) as caught:
        classify(item_for(path), fake_probe.prober)
    assert caught.value.reason == "unsupported_codec"


def test_probe_failure_on_an_identified_file_is_probe_error(tmp_path: Path, build, fake_probe) -> None:
    path = tmp_path / "bad.flac"
    path.write_bytes(build.flac)
    fake_probe.answer("bad.flac", rc=1, stderr="Invalid data found when processing input")
    with pytest.raises(FileFailure) as caught:
        classify(item_for(path), fake_probe.prober)
    assert caught.value.reason == "probe_error"


def test_processing_mark_skips_unless_forced_and_never_hides_corruption(tmp_path: Path, build, fake_probe) -> None:
    good, bad = tmp_path / "marked.wav", tmp_path / "cut.wav"
    good.write_bytes(build.wav())
    bad.write_bytes(build.wav()[:-50])
    for name in ("marked.wav", "cut.wav"):
        fake_probe.answer(name, build.document(build.stream()))
    with pytest.raises(SkipFile) as caught:
        classify(item_for(good), fake_probe.prober, is_processed=lambda item: True)
    assert caught.value.reason == "already_processed"
    assert classify(item_for(good), fake_probe.prober, is_processed=lambda item: True, force=True).codec
    with pytest.raises(FileFailure) as corrupt:
        classify(item_for(bad), fake_probe.prober, is_processed=lambda item: True, force=True)
    assert corrupt.value.reason == "truncated"


def test_mark_test_is_not_consulted_for_unusable_files(tmp_path: Path, fake_probe) -> None:
    path = tmp_path / "text.wav"
    path.write_text("nothing", encoding="utf-8")
    calls: list[object] = []
    with pytest.raises(SkipFile) as caught:
        classify(item_for(path), fake_probe.prober, is_processed=lambda item: calls.append(item) or True)
    assert caught.value.reason == "not_audio" and not calls


@pytest.mark.skipif(not REAL_FFPROBE.is_file(), reason="installed ffprobe not found")
def test_real_ffprobe_on_a_hand_written_wave(tmp_path: Path, build) -> None:
    path = tmp_path / "real.wav"
    path.write_bytes(build.wav(frames=4410))
    result = classify(item_for(path), locate_ffprobe(REAL_FFPROBE))
    assert result.codec == "pcm_s16le" and result.channels == 2 and result.sample_rate_hz == 44100
    text = tmp_path / "text.wav"
    text.write_text("plain text", encoding="utf-8")
    with pytest.raises(SkipFile):
        classify(item_for(text), locate_ffprobe(REAL_FFPROBE))
