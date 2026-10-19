"""Independent S4 fixture/read-back helpers (TK-0717); no product imports.

The public CLI is the only driver. Mutagen and small raw container walks are
oracles, not Norma's reader, mapper, writer or report projection.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

import mutagen
from mutagen.aiff import AIFF
from mutagen.flac import FLAC
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

TARGETS = ("wav", "aiff", "flac", "alac", "mp3", "aac")
F13 = ("wav16", "wav24", "wav32f", "aiff24", "flac24", "alac24", "mp3-v0", "aac256")
ID3_FIELDS = dict(title="TIT2", artist="TPE1", album_artist="TPE2", album="TALB",
                  genre="TCON", composer="TCOM", compilation="TCMP")
VORBIS_FIELDS = dict(title="TITLE", artist="ARTIST", album_artist="ALBUMARTIST", album="ALBUM",
                     genre="GENRE", composer="COMPOSER", comment="COMMENT", date="DATE",
                     compilation="COMPILATION", track="TRACKNUMBER", track_total="TRACKTOTAL",
                     disc="DISCNUMBER", disc_total="DISCTOTAL")
MP4_FIELDS = dict(title="©nam", artist="©ART", album_artist="aART", album="©alb", genre="©gen",
                  composer="©wrt", comment="©cmt", date="©day", compilation="cpil")
INFO_FIELDS = dict(title="INAM", artist="IART", album="IPRD", date="ICRD", track="IPRT",
                   genre="IGNR", comment="ICMT")
NUMBERS = {"track", "track_total", "disc", "disc_total"}
CATALOGUE = tomllib.loads((Path(__file__).parents[1] / "fixtures/catalogue.toml").read_text())["fixtures"]
NO_STORE_VARIANTS = {"adts", "id3-adts", "caf-pcm", "caf-alac"}
CATALOGUE_AUDIO = [(row["id"], row.get("variant")) for row in CATALOGUE
                   if row["sample_rate_hz"] > 0 and row["id"] not in {"F-10", "F-15"}
                   and row.get("variant") not in NO_STORE_VARIANTS]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def chunks(data, *, big=False):
    position = 12
    while position + 8 <= len(data):
        key = data[position:position + 4]
        size = int.from_bytes(data[position + 4:position + 8], "big" if big else "little")
        yield key, data[position + 8:position + 8 + size]
        position += 8 + size + (size & 1)


def info_items(path):
    result = []
    for kind, body in chunks(path.read_bytes()):
        if kind == b"LIST" and body[:4] == b"INFO":
            offset = 4
            while offset + 8 <= len(body):
                size = int.from_bytes(body[offset + 4:offset + 8], "little")
                result.append((body[offset:offset + 4].decode("latin-1"), body[offset + 8:offset + 8 + size]))
                offset += 8 + size + (size & 1)
    return result


def atoms(data, start=0, end=None):
    end = len(data) if end is None else end
    while start + 8 <= end:
        size, name = struct.unpack_from(">I4s", data, start)
        header = 8
        if size == 1:
            size, header = struct.unpack_from(">Q", data, start + 8)[0], 16
        elif size == 0:
            size = end - start
        assert size >= header and start + size <= end
        yield name, start + header, start + size
        start += size


def ilst_items(path):
    data = path.read_bytes()
    start, end = 0, len(data)
    for name in (b"moov", b"udta", b"meta", b"ilst"):
        hits = [(a, b) for key, a, b in atoms(data, start, end) if key == name]
        if not hits:
            return []
        (start, end), = hits
        if name == b"meta":
            start += 4
    result = []
    for name, a, b in atoms(data, start, end):
        if name == b"----":
            parts = {key: data[x + 4:y].decode("utf-8") for key, x, y in atoms(data, a, b)
                     if key in (b"mean", b"name")}
            result.append("----:" + parts[b"mean"] + ":" + parts[b"name"])
        else:
            result.append(name.decode("latin-1"))
    return result


def container(path):
    return {".wav": WAVE, ".aif": AIFF, ".aiff": AIFF, ".flac": FLAC,
            ".mp3": MP3, ".m4a": MP4}[path.suffix.lower()](path)


def native_ids(path):
    """Occurrence IDs from independent frames/comments/atoms, retaining duplicates."""
    pairs = []
    suffix = path.suffix.lower()
    if suffix in (".wav", ".aif", ".aiff", ".mp3"):
        try:
            tags = ID3(path, translate=False, load_v1=False) if suffix == ".mp3" else None
            # Container constructors translate dates; reload the stored ID3 chunk verbatim.
            if suffix != ".mp3":
                import io
                bodies = [body for key, body in chunks(path.read_bytes(), big=suffix != ".wav")
                          if key.lower() == b"id3 "]
                tags = ID3(io.BytesIO(bodies[0]), translate=False) if bodies else None
            if tags is not None:
                pairs += [("id3v2", "PRIV:" + frame.owner if frame.FrameID == "PRIV" else frame.HashKey)
                          for frame in tags.values()]
        except mutagen.id3.ID3NoHeaderError:
            pass
        if suffix == ".wav":
            pairs += [("riff-info", key) for key, _ in info_items(path)]
        elif suffix in (".aif", ".aiff"):
            pairs += [("aiff-text", key.decode("latin-1")) for key, _ in chunks(path.read_bytes(), big=True)
                      if key in (b"NAME", b"AUTH", b"(c) ", b"ANNO")]
        else:
            from mutagen.apev2 import APEv2, APENoHeaderError
            try:
                pairs += [("apev2", key) for key in APEv2(path)]
            except APENoHeaderError:
                pass
            tail = path.read_bytes()[-128:]
            if tail[:3] == b"TAG":
                layout = (("title", 3, 33), ("artist", 33, 63), ("album", 63, 93), ("year", 93, 97),
                          ("comment", 97, 125 if tail[125] == 0 and tail[126] else 127))
                pairs += [("id3v1", key) for key, a, b in layout if tail[a:b].rstrip(b"\0 ")]
                if tail[125] == 0 and tail[126]:
                    pairs.append(("id3v1", "track"))
                if tail[127] != 255:
                    pairs.append(("id3v1", "genre"))
    elif suffix == ".flac":
        data = path.read_bytes()
        if data.startswith(b"ID3"):
            pairs += [("id3v2", frame.HashKey) for frame in ID3(path, translate=False).values()]
            tag_size = sum((data[6 + i] & 127) << (21 - i * 7) for i in range(4))
            offset = 10 + tag_size + 4
        else:
            offset = 4
        assert data[offset - 4:offset] == b"fLaC"
        while offset + 4 <= len(data):
            flag = data[offset]
            size = int.from_bytes(data[offset + 1:offset + 4], "big")
            body = data[offset + 4:offset + 4 + size]
            if flag & 127 == 4:
                vendor_size = int.from_bytes(body[:4], "little")
                pos = 4 + vendor_size
                count = int.from_bytes(body[pos:pos + 4], "little")
                pos += 4
                for _ in range(count):
                    n = int.from_bytes(body[pos:pos + 4], "little")
                    value = body[pos + 4:pos + 4 + n].decode("utf-8")
                    pairs.append(("vorbis", value.split("=", 1)[0].upper()))
                    pos += 4 + n
            elif flag & 127 == 6:
                pairs.append(("vorbis", "PICTURE"))
            offset += 4 + size
            if flag & 128:
                break
    elif suffix == ".m4a":
        pairs += [("mp4", key) for key in ilst_items(path)]
    counts = collections.Counter(pairs)
    seen = collections.Counter()
    result = []
    for pair in pairs:
        seen[pair] += 1
        result.append("store:" + ":".join(pair) + (f"#{seen[pair]}" if counts[pair] > 1 else ""))
    return set(result)


def readback(path, *, info=False):
    """Canonical values, extras and exact picture bytes, without the application reader."""
    audio = container(path)
    fields, extras, pictures = {}, {}, []
    if info:
        items = {k: v.rstrip(b"\0").decode("utf-8") for k, v in info_items(path)}
        fields = {field: items[key] for field, key in INFO_FIELDS.items() if key in items}
        if "track" in fields:
            fields["track"] = int(fields["track"])
        return fields, extras, pictures
    tags = audio.tags
    if tags is None:
        return fields, extras, pictures
    if isinstance(audio, FLAC):
        for field, key in VORBIS_FIELDS.items():
            if key in tags:
                value = " / ".join(tags[key])
                fields[field] = int(value) if field in NUMBERS else value == "1" if field == "compilation" else value
        extras = {k.upper(): " / ".join(v) for k, v in tags.items() if k.upper() not in set(VORBIS_FIELDS.values())}
        pictures = [p.data for p in audio.pictures]
    elif isinstance(audio, MP4):
        for field, key in MP4_FIELDS.items():
            if key in tags:
                fields[field] = bool(tags[key]) if field == "compilation" else " / ".join(tags[key])
        for key, number, total in (("trkn", "track", "track_total"), ("disk", "disc", "disc_total")):
            if key in tags:
                n, t = tags[key][0]
                if n:
                    fields[number] = n
                if t:
                    fields[total] = t
        extras = {k.split(":", 2)[2]: " / ".join(bytes(v).decode() for v in values)
                  for k, values in tags.items() if k.startswith("----:com.apple.iTunes:")}
        pictures = [bytes(p) for p in tags.get("covr", [])]
    else:
        for field, key in ID3_FIELDS.items():
            if key in tags:
                value = " / ".join(map(str, tags[key].text))
                fields[field] = value == "1" if field == "compilation" else value
        comments = [f for f in tags.getall("COMM") if f.desc == ""]
        if comments:
            fields["comment"] = " / ".join(comments[0].text)
        if "TDRC" in tags:
            fields["date"] = str(tags["TDRC"].text[0])
        for key, number, total in (("TRCK", "track", "track_total"), ("TPOS", "disc", "disc_total")):
            if key in tags:
                parts = str(tags[key].text[0]).split("/")
                if parts[0] not in ("", "0"):
                    fields[number] = int(parts[0])
                if len(parts) == 2:
                    fields[total] = int(parts[1])
        extras = {f.desc: " / ".join(f.text) for f in tags.getall("TXXX")}
        pictures = [f.data for f in tags.getall("APIC")]
    return fields, extras, pictures


def tag_wav(path, **fields):
    """Build source opinions independently, keeping absent fields genuinely absent."""
    from mutagen.id3 import Frames, TXXX
    audio = WAVE(path)
    if audio.tags is None:
        audio.add_tags()
    mapping = dict(ID3_FIELDS, comment="COMM", date="TDRC", track="TRCK", disc="TPOS")
    for key, value in fields.items():
        if key in mapping:
            kwargs = dict(encoding=3, text=[str(value)])
            if key == "comment":
                kwargs.update(lang="eng", desc="")
            audio.tags.add(Frames[mapping[key]](**kwargs))
        else:
            audio.tags.add(TXXX(encoding=3, desc=key, text=[str(value)]))
    audio.save(v2_version=4)
    return path


def document(path):
    return json.loads(path.read_text(encoding="utf-8"))


def details(run, kind):
    return [(cells[1], cells[2], json.loads(cells[3])) for cells in run.lines if cells[0] == kind]


def plan(norma, source, *options):
    run = norma("tags", "plan", source, "--config", "/dev/null", "-v", *options)
    assert run.code == 0, run.stdout + run.stderr
    return run


def branch_rows(run, kind, branch):
    return [row for _, label, row in details(run, kind) if label in (branch, "same in both alternatives")]


def assert_accounted(run, source):
    """Every independent source occurrence once per branch; no roll-up duplicates."""
    expected = native_ids(source)
    plans = details(run, "PLAN")
    assert plans, run.stdout
    for _, branch, _ in plans:
        native = branch_rows(run, "NATIVE_FATE", branch)
        source_rows = [r for r in native if r["source_ids"]]
        actual = [identity for r in source_rows for identity in r["source_ids"]]
        assert collections.Counter(actual) == collections.Counter(expected), (source, branch, actual, expected)
        assert len({r["key"] for r in native}) == len(native)
        rollups = branch_rows(run, "TAG", branch)
        assert len({r["key"] for r in rollups}) == len(rollups)
        for row in rollups:
            assert set(row["source_native_ids"]) <= expected
        carried = {i for row in source_rows for i in row["expected_ids"]}
        added = [r for r in native if not r["source_ids"]]
        assert all(r["status"] == "added" and not (set(r["expected_ids"]) & carried) for r in added)


def normalise(norma, source, root, *options, code=0):
    root.mkdir()
    report = root / "run.json"
    run = norma("normalise", source, "--config", "/dev/null", "-o", root / "out",
                "--report", report, *options)
    assert run.code == code, run.stdout + run.stderr
    row, = document(report)["outcomes"]
    return run, row, Path(row["out_path"]) if row["out_path"] else None


def ffmpeg_trap(tmp_path, monkeypatch, *, delegate=False):
    """Actual executable interception, also covering FFmpeg launched by the child CLI."""
    real = shutil.which("ffmpeg")
    assert real
    folder = tmp_path / "trap-bin"
    folder.mkdir()
    prober = shutil.which("ffprobe")
    assert prober
    (folder / "ffprobe").symlink_to(prober)
    calls = tmp_path / "ffmpeg-calls.jsonl"
    calls.write_text("")
    script = folder / "ffmpeg"
    script.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                      f"with open({str(calls)!r},'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                      + (f"os.execv({real!r},[{real!r},*sys.argv[1:]])\n" if delegate else "sys.exit(99)\n"))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(folder) + os.pathsep + os.environ["PATH"])
    return calls


def assert_no_render(calls, source):
    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
    source_calls = [args for args in invocations if str(source) in args and "-i" in args]
    assert all("null" in args and args[-1] in ("-", "pipe:1") for args in source_calls), source_calls
    assert not any("loudnorm" in arg for args in source_calls for arg in args), source_calls
    return invocations
