"""Deterministic C-12 recipes; no audio is stored in the source tree.

Use ``generate(id, folder, variant=...)`` with a caller-owned temporary folder.
``FixtureSession`` is a lazy session cache: first use writes into the current
function sandbox (or an explicit caller folder), subsequent uses only read.
Later suites can import the session-scoped ``fixture_path`` pytest fixture in
their local conftest. Copy cached inputs into tmp_path before modifying tags.

The locked S1 toolchain is recorded separately from TESTING's historical probe.
A recipe hash selects the cache, a toolchain hash selects its subdirectory.
Neither a cache hit nor an expected value claims application acceptance.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import struct
import sys
import tomllib
import unicodedata
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from norma.adapters import ffmpeg, ffprobe

CATALOGUE = Path(__file__).with_name("catalogue.toml")
RATE = 48000
COMMENT = 'Line one\nLine two; comma, quote "'
TITLE = "Perché 東京 🎵"
MARK = (
    "v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=limit;gain=+5.80;"
    "limited=2.60;out_i=-16.02;out_tp=-1.04;origin=pcm_s24le;"
    "ffmpeg=N-125892;date=2026-10-05"
)
ENCODINGS = {
    "wav16": ("wav", ["-af", "aresample=dither_method=none", "-c:a", "pcm_s16le"]),
    "wav24": ("wav", ["-c:a", "pcm_s24le"]),
    "wav32f": ("wav", ["-c:a", "pcm_f32le"]),
    "aiff24": ("aif", ["-c:a", "pcm_s24be"]),
    "flac24": ("flac", ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", "-compression_level", "8"]),
    "alac24": ("m4a", ["-c:a", "alac", "-sample_fmt", "s32p", "-bits_per_raw_sample", "24"]),
    "mp3-v0": ("mp3", ["-c:a", "libmp3lame", "-q:a", "0", "-write_xing", "1"]),
    "aac256": ("m4a", ["-c:a", "aac_at", "-b:a", "256k"]),
}
SIGNATURES = {
    "rf64": ("wav", ["-c:a", "pcm_s24le", "-rf64", "always"]),
    "aifc": ("aif", ["-c:a", "pcm_f32be"]),
    "adts": ("aac", ["-c:a", "aac_at", "-b:a", "256k", "-f", "adts"]),
    "id3-flac": ENCODINGS["flac24"],
    "id3-adts": ("aac", ["-c:a", "aac_at", "-b:a", "256k", "-f", "adts"]),
    "caf-pcm": ("caf", ["-c:a", "pcm_s24le"]),
    "caf-alac": ("caf", ["-c:a", "alac", "-sample_fmt", "s32p", "-bits_per_raw_sample", "24"]),
    "mp4-isom": ("m4a", ["-c:a", "aac_at", "-b:a", "256k", "-f", "mp4"]),
    "mp3-no-id3": ("mp3", ["-c:a", "libmp3lame", "-q:a", "0", "-id3v2_version", "0"]),
}


def catalogue() -> tuple[dict, ...]:
    """Read the static specification; absent optional TOML fields mean None."""
    with CATALOGUE.open("rb") as stream:
        return tuple(tomllib.load(stream)["fixtures"])


def fixture_spec(id: str, variant: str | None = None) -> dict:
    for spec in catalogue():
        if (spec["id"], spec.get("variant")) == (id, variant):
            return spec
    raise KeyError((id, variant))


def recipe_hash(spec: dict) -> str:
    """Includes implementation identity so an internal recipe edit invalidates it."""
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload + Path(__file__).read_bytes()).hexdigest()


def toolchain() -> dict[str, str]:
    """Actual identities, never the historical probe versions by assumption."""
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "soundfile", "pyloudnorm", "mutagen")}
    versions.update(python=platform.python_version(), machine=platform.machine(),
                    macos=platform.mac_ver()[0], libsndfile=sf.__libsndfile_version__)
    executable = ffmpeg.locate_ffmpeg()
    versions["ffmpeg"] = executable.version
    versions["ffmpeg_sha256"] = hashlib.sha256(executable.path.read_bytes()).hexdigest()
    prober = ffprobe.locate_ffprobe()
    versions["ffprobe"] = prober.version
    versions["ffprobe_sha256"] = hashlib.sha256(prober.path.read_bytes()).hexdigest()
    if sys.platform == "darwin":
        result = ffmpeg.check_result(ffmpeg.run_process(("/usr/bin/sips", "--version"), deadline_s=60))
        versions["sips"] = result.stdout.decode().strip()
        versions["sips_sha256"] = hashlib.sha256(Path("/usr/bin/sips").read_bytes()).hexdigest()
        profile = Path("/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc")
        versions["icc_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
    return versions


def tone(level: float, seconds: float, rate: int = RATE, frequency: float = 1000) -> np.ndarray:
    """Per-channel sine peak in dBFS; not RMS or a calibrated loudness value."""
    return 10 ** (level / 20) * np.sin(2 * np.pi * frequency * np.arange(round(seconds * rate), dtype=np.float64) / rate)


def stereo(x: np.ndarray) -> np.ndarray:
    return np.column_stack((x, x))


def pink(seconds: int, seed: int) -> np.ndarray:
    n = seconds * RATE
    z = np.random.Generator(np.random.PCG64(seed)).standard_normal((n, 2))
    f = np.fft.rfftfreq(n, 1 / RATE)
    w = np.zeros_like(f)
    selected = (20 <= f) & (f <= 20000)
    w[selected] = 1 / np.sqrt(f[selected])
    y = np.fft.irfft(np.fft.rfft(z, axis=0) * w[:, None], n=n, axis=0)
    return y / np.sqrt(np.mean(y * y, axis=0))


def signal(id: str, variant: str | None = None) -> tuple[np.ndarray, int, str]:
    """Pure sample constructors for the signal fixtures in TESTING § 4.3."""
    rate, subtype = RATE, "PCM_24"
    if id == "F-01":
        x = stereo(tone(-23, 20))
    elif id == "F-02":
        levels = {"lra1": [-20, -30], "lra2": [-20, -15], "lra3": [-40, -20],
                  "lra4": [-50, -35, -20, -35, -50]}
        x = stereo(np.concatenate([tone(level, 20) for level in levels[variant]])) if variant else stereo(tone(-33, 20))
    elif id == "F-03":
        if variant in {"endpoint", "tail4801", "tail5279"}:
            n = {"endpoint": 48000, "tail4801": 4801, "tail5279": 5279}[variant]
            x = np.zeros((n, 2), dtype=np.float64)
            if variant == "endpoint":
                x[[0, 1, 1000, n - 2, n - 1], :] = np.array([.02, .01, .03, .01, .02])[:, None]
            else:
                x[-1, :] = .5
            subtype = "FLOAT"
        else:
            n = np.arange(960000, dtype=np.float64)
            y = np.sqrt(2) * 10 ** (-6 / 20) * np.sin(np.pi * n / 2 + np.pi / 4)
            w = .5 - .5 * np.cos(np.pi * np.arange(480) / 479)
            y[:480] *= w
            y[-480:] *= w[::-1]
            x = stereo(y)
    elif id == "F-04":
        long = variant == "240s"
        x = pink(240 if long else 30, 705 if long else 704) * (0.06530769503157757 if long else 0.06537798862588726)
    elif id == "F-05":
        if variant == "c6":
            n = np.arange(30 * RATE)
            y = 0.027583193778991696 * np.sin(2 * np.pi * 997 * n / RATE)
            burst = 0.8636675061689802 * np.sin(np.pi * np.arange(480) / 2 + np.pi / 4)
            gain = 1
        else:
            y = tone(-24, 30)
            k = np.arange(480, dtype=np.float64)
            burst = (.27 if variant == "le" else .85) * np.sin(2 * np.pi * 1000 * k / RATE) * (.5 - .5 * np.cos(2 * np.pi * k / 479))
            gain = 1 if variant == "le" else 0.9806385461208077
        for start in [2, 7, 12, 17, 22, 27]:
            y[start * RATE:start * RATE + 480] += burst
        x = stereo(y * gain)
    elif id == "F-06":
        x = np.clip(pink(30, 706), -3.63, 3.63) * 0.26020672671220224
        subtype = "FLOAT"
    elif id == "F-07":
        x = tone(-17, 20)[:, None]
    elif id == "F-08":
        x = np.zeros((480000, 2), dtype=np.float64)
    elif id == "F-09":
        seconds = {None: 1, "100ms": .1, "300ms": .3, "450ms": .45, "quiet": 1}[variant]
        x = stereo(tone(-100 if variant == "quiet" else (-23 if variant is None else -20), seconds))
    elif id == "F-14":
        rate = 96000
        n = np.arange(960000, dtype=np.float64)
        x = stereo(.1 * np.sin(2 * np.pi * 1000 * n / rate) + .1 * np.sin(2 * np.pi * 30000 * n / rate))
    else:
        raise KeyError((id, variant))
    return x, rate, subtype


def _encode(source: Path, destination: Path, options: list[str]) -> None:
    executable = ffmpeg.locate_ffmpeg()
    ffmpeg.check_result(ffmpeg.run_process(
        ffmpeg.ffmpeg_argv(executable, "-y", *ffmpeg.input_args(source), "-map", "0:a:0",
                           "-map_metadata", "-1", "-ar", "48000", "-ac", "2", *options, str(destination)),
        executable=executable, deadline_s=ffmpeg.tool_deadline(sf.info(source).duration),
    ))


def decode(path: Path) -> np.ndarray:
    """Interleaved little-endian float64 decoded samples, including lossy audio."""
    executable = ffmpeg.locate_ffmpeg()
    result = ffmpeg.check_result(ffmpeg.run_process(
        ffmpeg.ffmpeg_argv(executable, *ffmpeg.input_args(path.absolute()), "-map", "0:a:0",
                           "-c:a", "pcm_f64le", "-f", "f64le", "pipe:1"),
        executable=executable, deadline_s=3600, stdout_limit=256 * 1024 * 1024,
    ))
    if result.stdout_truncated:
        raise ValueError("decoded fixture exceeds capture limit")
    return np.frombuffer(result.stdout, dtype="<f8")


def decoded_hash(path: Path) -> str:
    return hashlib.sha256(decode(path).tobytes()).hexdigest()


def riff_chunks(data: bytes):
    """Yield RIFF offsets and declared extents, retaining truncated evidence."""
    offset = 12
    while offset + 8 <= len(data):
        size = struct.unpack_from("<I", data, offset + 4)[0]
        yield data[offset:offset + 4], offset + 8, size
        offset += 8 + size + (size & 1)


def _riff_chunk(kind: bytes, payload: bytes) -> bytes:
    return kind + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")


def cover_png() -> bytes:
    """Exact seedless RGB PNG construction from TESTING § 4.7."""
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    raw = (b"\0" + bytes([40, 120, 200]) * 64) * 64
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _tag(path: Path, *, title: str = TITLE, cover: bytes | None = None, stale: bool = False, mark: str | None = None) -> None:
    # Test-data construction deliberately does not call the application writer.
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC, Picture
    from mutagen.id3 import ID3, APIC, COMM, Frames, TXXX
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm
    from mutagen.wave import WAVE

    suffix = path.suffix.lower()
    if suffix in {".wav", ".aif", ".mp3"}:
        audio = {".wav": WAVE, ".aif": AIFF, ".mp3": MP3}[suffix](path)
        if audio.tags is None:
            audio.add_tags()
        tags: ID3 = audio.tags
        values = {"TIT2": title}
        if cover is not None:
            values.update(TPE1="Test Artist", TPE2="Norma fixtures", TALB="Demo set", TYER="2026", TDAT="0510",
                          TRCK="3/12", TPOS="1/2", TCON="Demo", TCOM="Composer", TCMP="1")
        for frame, text in values.items():
            tags.add(Frames[frame](encoding=1, text=[text]))
        if cover is not None:
            tags.add(COMM(encoding=1, lang="eng", desc="", text=[COMMENT]))
            tags.add(APIC(encoding=1, mime="image/png", type=3, desc="", data=cover))
            tags.add(TXXX(encoding=1, desc="DEMO_NOTE", text=["seedless"]))
        if stale:
            for key, value in {"REPLAYGAIN_TRACK_GAIN": "-3.00 dB", "REPLAYGAIN_TRACK_PEAK": "0.900000"}.items():
                tags.add(TXXX(encoding=1, desc=key, text=[value]))
            tags.add(Frames["TSSE"](encoding=1, text=["fixture-old"]))
            tags.add(COMM(encoding=1, lang="eng", desc="iTunNORM", text=[" ".join(["000003E8"] * 10)]))
        if mark is not None:
            tags.add(TXXX(encoding=1, desc="NORMA", text=[mark]))
        audio.save(v2_version=3)
    elif suffix == ".flac":
        audio = FLAC(path)
        audio["TITLE"] = title
        if cover is not None:
            for key, value in {"ARTIST": "Test Artist", "ALBUMARTIST": "Norma fixtures", "ALBUM": "Demo set", "DATE": "2026-10-05",
                               "TRACKNUMBER": "3", "TRACKTOTAL": "12", "DISCNUMBER": "1", "DISCTOTAL": "2", "GENRE": "Demo",
                               "COMPOSER": "Composer", "COMMENT": COMMENT, "COMPILATION": "1", "DEMO_NOTE": "seedless"}.items():
                audio[key] = value
            picture = Picture()
            picture.type, picture.mime, picture.width, picture.height, picture.depth, picture.data = 3, "image/png", 64, 64, 24, cover
            audio.add_picture(picture)
        if stale:
            for key, value in {"REPLAYGAIN_TRACK_GAIN": "-3.00 dB", "REPLAYGAIN_TRACK_PEAK": "0.900000", "ENCODER": "fixture-old", "iTunNORM": " ".join(["000003E8"] * 10)}.items():
                audio[key] = value
        if mark is not None:
            audio["NORMA"] = mark
        audio.save()
    else:
        audio = MP4(path)
        audio["©nam"] = [title]
        if cover is not None:
            for key, value in {"©ART": "Test Artist", "aART": "Norma fixtures", "©alb": "Demo set", "©day": "2026-10-05", "©gen": "Demo", "©wrt": "Composer", "©cmt": COMMENT}.items():
                audio[key] = [value]
            audio["trkn"], audio["disk"], audio["cpil"] = [(3, 12)], [(1, 2)], True
            audio["covr"] = [MP4Cover(cover, imageformat=MP4Cover.FORMAT_PNG)]
            audio["----:com.apple.iTunes:DEMO_NOTE"] = [MP4FreeForm(b"seedless")]
        if stale:
            audio["©too"] = ["fixture-old"]
            for key, value in {"REPLAYGAIN_TRACK_GAIN": "-3.00 dB", "REPLAYGAIN_TRACK_PEAK": "0.900000", "iTunNORM": " ".join(["000003E8"] * 10)}.items():
                audio["----:com.apple.iTunes:" + key] = [MP4FreeForm(value.encode())]
        if mark is not None:
            audio["----:com.apple.iTunes:NORMA"] = [MP4FreeForm(mark.encode())]
        audio.save()


def _manifests(folder: Path, variant: str) -> Path:
    """One isolated mutation per manifest variant, with exact CSV/JSON ordering."""
    json_form = variant.startswith("json")
    mutation = variant.removeprefix("json-").removeprefix("csv-")
    defaults = {"artist": "Test Artist", "date": "2026-10-05"}
    fields = {"title": TITLE, "track": 3, "track_total": 12, "comment": COMMENT}
    file = "Song.wav" if mutation == "ambiguous" else "space name.wav"
    if mutation == "bad-date":
        defaults["date"] = "2026-02-30"
    if mutation == "bad-track":
        fields["track"] = 13
    if mutation == "clear":
        fields["title"] = None
    if mutation == "blank":
        fields.pop("title")
    if mutation in {"extra", "typo", "bad-extra"}:
        fields["extras"] = {{"extra": "MOOD", "typo": "titel", "bad-extra": "bad key"}[mutation]: "alternate" if mutation == "typo" else "hopeful"}
    if json_form:
        obj = {"version": 2 if mutation == "version" else 1, "defaults": defaults, "files": {file: fields}}
        text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
        if mutation == "duplicate":
            # Keep duplicate lexical JSON object keys, not a dict that lost them.
            member = json.dumps(file, ensure_ascii=False) + ": " + json.dumps(fields, ensure_ascii=False)
            text = '{"version":1,"defaults":' + json.dumps(defaults) + ',"files":{' + member + ',' + member + '}}\n'
        path = folder / "manifest.json"
        path.write_text(text, encoding="utf-8")
    else:
        header = ["file", "title", "artist", "date", "track", "comment"]
        rows = [["*", "", "Test Artist", defaults["date"], "", ""],
                [file, "[clear]" if mutation == "clear" else fields.get("title", ""), "", "", "13/12" if mutation == "bad-track" else "3/12", COMMENT]]
        if mutation == "duplicate":
            rows.append(rows[-1].copy())
        if mutation == "missing-file":
            header = header[1:]
            rows = [row[1:] for row in rows]
        if "extras" in fields:
            key, value = next(iter(fields["extras"].items()))
            header.append(key)
            rows[0].append("")
            rows[1].append(value)
        stream = io.StringIO(newline="")
        csv.writer(stream, delimiter=";" if mutation == "semicolon" else ",", lineterminator="\r\n").writerows([header, *rows])
        path = folder / "manifest.csv"
        path.write_bytes((b"\xef\xbb\xbf" if mutation == "semicolon" else b"") + stream.getvalue().encode())
    return path


def _construct(spec: dict, folder: Path) -> Path:
    id, variant = spec["id"], spec.get("variant")
    if id in {"F-01", "F-02", "F-03", "F-04", "F-05", "F-06", "F-07", "F-08", "F-09", "F-14"}:
        x, rate, subtype = signal(id, variant)
        path = folder / (id + ("-" + variant if variant else "") + ".wav")
        sf.write(path, x, rate, subtype=subtype, format="WAV")
        return path
    if id == "F-11":
        path = folder / "not-audio.wav"
        path.write_bytes(b"This is not audio.\n")
        return path
    if id == "F-10":
        parent = _construct(fixture_spec("F-01"), folder)
        path = folder / ("truncated.aif" if variant == "aiff" else "payload.flac" if variant == "payload" else "truncated.wav")
        if variant == "aiff":
            _encode(parent, path, ENCODINGS["aiff24"][1])
            data = path.read_bytes()
            offset = 12
            while offset + 8 <= len(data):
                size = struct.unpack_from(">I", data, offset + 4)[0]
                if data[offset:offset + 4] == b"SSND":
                    displacement = struct.unpack_from(">I", data, offset + 8)[0]
                    path.write_bytes(data[:offset + 16 + displacement + 5 * RATE * 6])
                    return path
                offset += 8 + size + (size & 1)
            raise ValueError("no AIFF SSND chunk")
        if variant == "payload":
            _encode(parent, path, ENCODINGS["flac24"][1])
            data = path.read_bytes()
            offset = 4
            while True:
                last = bool(data[offset] & 128)
                size = int.from_bytes(data[offset + 1:offset + 4], "big")
                offset += 4 + size
                if last:
                    break
            if data[offset] != 255 or data[offset + 1] & 252 != 248:
                raise ValueError("FLAC audio frame sync not located")
            # Keep frame header intact; a payload mutation must fail decoding.
            mutated = bytearray(data)
            mutated[offset + 96] ^= 255
            path.write_bytes(mutated)
            executable = ffmpeg.locate_ffmpeg()
            result = ffmpeg.run_process(ffmpeg.ffmpeg_argv(executable, "-xerror", *ffmpeg.input_args(path), "-f", "null", "-"), executable=executable, deadline_s=60)
            if result.returncode == 0:
                raise ValueError("FLAC payload mutation did not produce fatal decoder evidence")
            (folder / "corruption.json").write_text(json.dumps({
                "frame_start": offset, "payload_byte_offset": offset + 96,
                "argv": [arg.replace(str(folder), "<fixture>") for arg in result.argv],
                "returncode": result.returncode,
                "stderr_tail": [line.replace(str(folder), "<fixture>") for line in result.stderr_tail],
                "file_length_unchanged": len(mutated) == len(data),
            }, indent=2) + "\n", encoding="utf-8")
            return path
        data = parent.read_bytes()
        start = next(start for kind, start, size in riff_chunks(data) if kind == b"data")
        path.write_bytes(data[:start + 5 * RATE * 6])
        return path
    if id == "F-12":
        assert variant is not None
        parent = _construct(fixture_spec("F-04"), folder)
        extension, options = (ENCODINGS | SIGNATURES)[variant]
        path = folder / (variant + "." + extension)
        _encode(parent, path, options)
        if variant in {"id3-flac", "id3-adts"}:
            from mutagen.id3 import ID3, TIT2
            tags = ID3()
            tags.add(TIT2(encoding=1, text=["Signature fixture"]))
            tags.save(path, v2_version=3)
        return path
    if id == "F-13":
        if variant == "conflict":
            parent = _construct(fixture_spec("F-01"), folder)
            path = folder / "conflict.wav"
            _encode(parent, path, ["-c:a", "copy", "-metadata", "title=INFO title"])
        else:
            parent = _construct(fixture_spec("F-12", variant), folder)
            path = folder / ("tagged-" + parent.name)
            shutil.copyfile(parent, path)
        _tag(path, cover=cover_png())
        return path
    if id == "F-15":
        parent = _construct(fixture_spec("F-01"), folder)
        tree = folder / "paths"
        tree.mkdir()
        names = ["space name.wav", "Perché.wav", "unicode/" + unicodedata.normalize("NFD", "Perché") + ".wav",
                 "Test Artist's demo.wav", "東京 🎵.wav", "a/Song.wav", "b/song.WAV", "Artist/Album/03 - Perché.wav", "Artist/Album/004_Not_a_match.wav",
                 ".hidden.wav", "._sidecar.wav", "Normalised/output.wav", ".norma-originals/backup.wav"]
        for name in names:
            destination = tree / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent, destination)
            if name in {"a/Song.wav", "b/song.WAV"}:
                _tag(destination, title="shared")
        (tree / "alias.wav").symlink_to(tree / "space name.wav")
        os.link(tree / "space name.wav", tree / "hardlink.wav")
        return tree
    if id == "F-16":
        path = folder / {None: "cover.png", "heic": "cover.heic", "large": "large.jpg", "cmyk": "cmyk.jpg", "corrupt": "corrupt.jpg"}[variant]
        if variant == "corrupt":
            path.write_bytes(b"not an image\n")
        else:
            png = folder / "cover.png"
            png.write_bytes(cover_png())
            if variant:
                options = {"heic": ["-s", "format", "heic", "-s", "formatOptions", "90"],
                           "large": ["-s", "format", "jpeg", "-s", "formatOptions", "90", "-z", "6000", "6000"],
                           "cmyk": ["-m", "/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc", "-s", "format", "jpeg", "-s", "formatOptions", "90"]}[variant]
                ffmpeg.check_result(ffmpeg.run_process(("/usr/bin/sips", *options, str(png), "--out", str(path)), deadline_s=60))
        return path
    if id == "F-17":
        assert variant is not None
        return _manifests(folder, variant)
    if id == "F-18":
        if variant == "chunks":
            parent = _construct(fixture_spec("F-01"), folder)
            data = parent.read_bytes()
            bext = (b"Norma fixture".ljust(256, b"\0") + b"WS-07".ljust(32, b"\0") + b"F-18".ljust(32, b"\0") + b"2026-10-05" + b"00:00:00" + struct.pack("<QH", 0, 1) + bytes(64 + 190))
            data += _riff_chunk(b"cue ", struct.pack("<III4sIII", 1, 1, 48000, b"data", 0, 0, 48000))
            data += _riff_chunk(b"bext", bext) + _riff_chunk(b"iXML", b"<BWFXML><PROJECT>Norma fixture</PROJECT></BWFXML>")
            data = data[:4] + struct.pack("<I", len(data) - 8) + data[8:]
            path = folder / "chunks.wav"
            path.write_bytes(data)
            return path
        encoding = variant if variant in ENCODINGS else "flac24"
        parent = _construct(fixture_spec("F-13", encoding), folder)
        path = folder / ("stale-" + parent.name)
        shutil.copyfile(parent, path)
        mark = MARK if variant == "mark-valid" else "v=1;target=-14;tp=-1;gain=0;mode=linear" if variant == "mark-invalid" else None
        _tag(path, stale=True, mark=mark)
        return path
    raise KeyError((id, variant))


def _safe_folder(folder: Path) -> Path:
    folder = folder.absolute()
    resolved = folder.resolve()
    repository = Path(__file__).resolve().parents[2]
    if resolved.is_relative_to(repository):
        raise ValueError("fixture generation inside the repository is forbidden")
    if any(part.casefold() == "music" or part.casefold().endswith((".musiclibrary", ".itl")) for part in resolved.parts):
        raise ValueError("Music paths are forbidden")
    return resolved


def generate(id: str, folder: Path, *, variant: str | None = None) -> Path:
    """Build an immutable recipe cache in a caller-owned temporary folder.

    Files are created once, never overwritten. The receipt validates every
    member before a cache hit; changed data is an error, not silently repaired.
    Failed construction leaves no accepted receipt and refuses reuse.
    """
    spec = fixture_spec(id, variant)
    root = _safe_folder(Path(folder))
    versions = toolchain()
    tool_id = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    key = recipe_hash(spec)
    target = root / key / tool_id
    if not target.resolve().is_relative_to(root):
        raise ValueError("fixture cache escapes caller folder")
    receipt = target / "receipt.json"
    if receipt.exists():
        evidence = json.loads(receipt.read_text())
        if evidence["recipe_sha256"] != key or evidence["toolchain_id"] != tool_id:
            raise ValueError("cached fixture identity changed")
        result = target / evidence["result"]
        if not result.resolve().is_relative_to(target.resolve()):
            raise ValueError("cached fixture result escapes caller folder")
        for member in evidence["members"]:
            path = target / member["path"]
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError("cached fixture member escapes caller folder")
            if path.is_symlink():
                valid = os.readlink(path) == member["symlink"]
            else:
                valid = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == member["sha256"]
            if not valid:
                raise ValueError(f"cached fixture changed: {path}")
        return result
    target.mkdir(parents=True, exist_ok=False)
    path = _construct(spec, target)
    members = []
    for member in sorted(target.rglob("*")):
        if member.is_file() or member.is_symlink():
            row = {"path": str(member.relative_to(target))}
            if member.is_symlink():
                row["symlink"] = os.readlink(member)
            else:
                row["sha256"] = hashlib.sha256(member.read_bytes()).hexdigest()
            members.append(row)
    evidence = {"id": id, "variant": variant, "recipe_sha256": key, "toolchain_id": tool_id,
                "toolchain": versions, "date": datetime.now().astimezone().isoformat(),
                "result": str(path.relative_to(target)), "members": members,
                "spec": spec, "decoded_samples": {}, "audio_structure": {}}
    if id == "F-15":
        evidence["filesystem"] = {
            "case_sensitive": not (path / "SPACE NAME.WAV").exists(),
            "stored_spellings": [row["path"] for row in members],
            "hardlink_same_inode": os.stat(path / "space name.wav").st_ino == os.stat(path / "hardlink.wav").st_ino,
        }
    if id == "F-16" and variant != "corrupt":
        result = ffmpeg.check_result(ffmpeg.run_process((
            "/usr/bin/sips", "-g", "format", "-g", "pixelWidth", "-g", "pixelHeight", "-g", "space", str(path),
        ), deadline_s=60))
        evidence["image_structure"] = result.stdout.decode().replace(str(target), "<fixture>")
    # Malformed/non-audio artefacts have structural/byte oracles, not a fake decode.
    if spec["channels"] > 0 and not (id == "F-10" and variant == "payload"):
        prober = ffprobe.locate_ffprobe()
        for member in [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() == ".wav"):
            relative = str(member.relative_to(target))
            probe = ffprobe.probe(prober, member)
            samples = decode(member)
            evidence["decoded_samples"][relative] = hashlib.sha256(samples.tobytes()).hexdigest()
            evidence["audio_structure"][relative] = {
                "frames": len(samples) // probe.channels,
                "channels": probe.channels, "sample_rate_hz": probe.sample_rate_hz,
                "bit_depth": probe.bit_depth, "sample_float": probe.sample_float,
                "codec": probe.codec, "container": probe.container,
                "probed_duration_s": probe.duration_s,
                "decoded_duration_s": len(samples) / (probe.channels * probe.sample_rate_hz),
                "channel_layout": probe.channel_layout,
            }
    receipt.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class FixtureSession:
    """Session lifetime, lazy first use; writes only in the active sandbox.

    Without ``folder``, a new recipe uses cwd/fixtures (pytest's redirected
    tmp_path). Previously built recipes are returned read-only across tests;
    no writes into a sibling function's temporary directory are attempted.
    """

    def __init__(self) -> None:
        self._paths: dict[tuple[str, str | None, str | None], Path] = {}

    def __call__(self, id: str, variant: str | None = None, *, folder: Path | None = None) -> Path:
        key = (id, variant, str(Path(folder).resolve()) if folder is not None else None)
        if key not in self._paths:
            self._paths[key] = generate(id, folder if folder is not None else Path.cwd() / "fixtures", variant=variant)
        return self._paths[key]


@pytest.fixture(scope="session")
def fixture_path() -> FixtureSession:
    """Import into a consuming local conftest; shared conftest stays unchanged."""
    return FixtureSession()
