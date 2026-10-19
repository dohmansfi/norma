"""adapters/sips.py: argv, parsing, real sips and fakes for failure classes (TK-0609, S4; TST-M-06, TST-S-05)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from norma.adapters import sips
from norma.errors import Cancelled, EnvironmentError_, ToolError, ToolFailed, ToolMissing, ToolTimeout

_spec = importlib.util.spec_from_file_location("artwork_images", Path(__file__).with_name("artwork_images.py"))
assert _spec is not None and _spec.loader is not None
images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(images)

real_sips = pytest.mark.skipif(not images.HAVE_SIPS, reason="macOS sips not available")

GOOD = "/x/a.png\n  format: png\n  pixelWidth: 64\n  pixelHeight: 32\n  space: RGB\n"


def tool_for(path: Path) -> sips.Sips:
    return sips.locate_sips(path)


# --- parsing ------------------------------------------------------------------------------


def test_parse_info_reads_the_four_properties() -> None:
    assert sips.parse_info(GOOD) == sips.ImageInfo("png", 64, 32, "RGB")


def test_parse_info_lowercases_the_format_and_keeps_the_space() -> None:
    text = GOOD.replace("png", "JPEG").replace("RGB", "Gray")
    assert sips.parse_info(text) == sips.ImageInfo("jpeg", 64, 32, "Gray")


@pytest.mark.parametrize("text", [
    "",
    "Warning: x.png not a valid file - skipping\n",
    "/x\n  format: jpg\n  pixelWidth: <nil>\n  pixelHeight: <nil>\n  space: <nil>\n",
    GOOD.replace("64", "0"),
    GOOD.replace("64", "wide"),
    GOOD.replace("  space: RGB\n", ""),
    GOOD.replace("format: png", "format: <nil>"),
    GOOD.replace("RGB", "<nil>"),
], ids=["empty", "warning", "nil", "zero", "word", "no-space", "nil-format", "nil-space"])
def test_parse_info_refuses_anything_incomplete(text: str) -> None:
    assert sips.parse_info(text) is None


def test_the_path_line_is_not_a_property() -> None:
    assert sips.parse_properties("/a: b/c.png\n  format: png\n") == {"format": "png"}


# --- argv -----------------------------------------------------------------------------------


def fake_tool_object(tmp_path: Path) -> sips.Sips:
    return sips.Sips(tmp_path / "sips", (1, 2), "fake")


def test_inspect_argv_is_a_list_of_the_documented_properties(tmp_path: Path) -> None:
    argv = sips.inspect_argv(fake_tool_object(tmp_path), tmp_path / "a.png")
    assert argv == (str(tmp_path / "sips"), "-g", "format", "-g", "pixelWidth", "-g", "pixelHeight", "-g", "space",
                    str(tmp_path / "a.png"))


def test_convert_argv_minimal_jpeg(tmp_path: Path) -> None:
    argv = sips.convert_argv(fake_tool_object(tmp_path), tmp_path / "a", tmp_path / "b", image_format="jpeg", quality=90)
    assert argv[1:] == ("-s", "format", "jpeg", "-s", "formatOptions", "90", str(tmp_path / "a"), "--out",
                        str(tmp_path / "b"))


def test_convert_argv_scale_and_profile_come_first(tmp_path: Path) -> None:
    argv = sips.convert_argv(fake_tool_object(tmp_path), tmp_path / "a", tmp_path / "b", image_format="png",
                             max_side=3000, profile=sips.SRGB_PROFILE)
    assert argv[1:7] == ("-m", str(sips.SRGB_PROFILE), "-Z", "3000", "-s", "format")
    assert "formatOptions" not in argv


@pytest.mark.parametrize("bad", [Path("relative.png"), "text.png"])
def test_paths_must_be_absolute_local_paths(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ValueError, match="absolute"):
        sips.inspect_argv(fake_tool_object(tmp_path), bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absolute"):
        sips.convert_argv(fake_tool_object(tmp_path), tmp_path / "a", bad, image_format="png")  # type: ignore[arg-type]


def test_a_non_positive_scale_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sips.convert_argv(fake_tool_object(tmp_path), tmp_path / "a", tmp_path / "b", image_format="png", max_side=0)


# --- locate -------------------------------------------------------------------------------------


def test_locate_reads_the_version_of_a_fake(tmp_path: Path) -> None:
    tool = tool_for(images.fake_with(tmp_path, "exit 1\n"))
    assert tool.version == "sips-fake-1"
    assert tool.identity == (os.stat(tool.path).st_dev, os.stat(tool.path).st_ino)


def test_locate_missing_name_is_an_environment_error(tmp_path: Path) -> None:
    with pytest.raises(EnvironmentError_, match="not found on PATH"):
        sips.locate_sips("sips", search_path=str(tmp_path))


@pytest.mark.parametrize("name", ["relative/sips", "/nonexistent/sips"])
def test_locate_bad_explicit_path(name: str) -> None:
    with pytest.raises(EnvironmentError_, match="not found or not executable"):
        sips.locate_sips(name)


def test_locate_not_executable(tmp_path: Path) -> None:
    path = tmp_path / "sips"
    path.write_text("#!/bin/sh\n")
    with pytest.raises(EnvironmentError_):
        sips.locate_sips(path)


@pytest.mark.parametrize("body", ["exit 3\n", "echo\nexit 0\n", "sleep 0\nexit 0\n"], ids=["fails", "no-version", "silent"])
def test_locate_unusable_tool(tmp_path: Path, body: str) -> None:
    path = images.fake_tool(tmp_path, body)
    with pytest.raises(EnvironmentError_):
        sips.locate_sips(path)


def test_locate_timeout_is_an_environment_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sips, "INSPECT_DEADLINE_S", 0.3)
    path = images.fake_tool(tmp_path, "sleep 30\n")
    with pytest.raises(EnvironmentError_, match="unusable"):
        sips.locate_sips(path)


@real_sips
def test_locate_finds_the_system_sips() -> None:
    tool = sips.locate_sips("/usr/bin/sips")
    assert tool.version.startswith("sips-")


# --- real sips ----------------------------------------------------------------------------------


@real_sips
def test_real_inspect_png_jpeg_and_grey(tmp_path: Path) -> None:
    tool = sips.locate_sips("/usr/bin/sips")
    rgb = tmp_path / "rgb.png"
    rgb.write_bytes(images.png_bytes(64, 32))
    grey = tmp_path / "grey.png"
    grey.write_bytes(images.png_bytes(16, 16, 0, (100,)))
    jpeg = images.convert(rgb, tmp_path / "a.jpg", "-s", "format", "jpeg", "-s", "formatOptions", 90)
    assert sips.inspect(tool, rgb) == sips.ImageInfo("png", 64, 32, "RGB")
    assert sips.inspect(tool, grey).space == "Gray"
    assert sips.inspect(tool, jpeg) == sips.ImageInfo("jpeg", 64, 32, "RGB")


@real_sips
@pytest.mark.parametrize("name, data", [("text.jpg", b"not an image\n"), ("empty.png", b""), ("text", b"plain\n")])
def test_real_inspect_refuses_non_images(tmp_path: Path, name: str, data: bytes) -> None:
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(ToolError):  # a warning with <nil> fields, or sips itself failing on an untyped file
        sips.inspect(sips.locate_sips("/usr/bin/sips"), path)


@real_sips
def test_real_inspect_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(sips.NotAnImage):
        sips.inspect(sips.locate_sips("/usr/bin/sips"), tmp_path / "missing.png")


@real_sips
def test_real_convert_scales_down_and_never_up(tmp_path: Path) -> None:
    tool = sips.locate_sips("/usr/bin/sips")
    source = tmp_path / "a.png"
    source.write_bytes(images.png_bytes(64, 32))
    sips.convert(tool, source, tmp_path / "small.png", image_format="png", max_side=16)
    assert sips.inspect(tool, tmp_path / "small.png") == sips.ImageInfo("png", 16, 8, "RGB")
    sips.convert(tool, source, tmp_path / "same.png", image_format="png")
    assert sips.inspect(tool, tmp_path / "same.png").width_px == 64


@real_sips
def test_real_convert_refuses_an_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "a.png"
    source.write_bytes(images.png_bytes(8, 8))
    (tmp_path / "out.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="must not exist"):
        sips.convert(sips.locate_sips("/usr/bin/sips"), source, tmp_path / "out.jpg", image_format="jpeg")


@real_sips
def test_real_convert_of_a_non_image_fails(tmp_path: Path) -> None:
    source = tmp_path / "bad.jpg"
    source.write_bytes(b"not an image\n")
    with pytest.raises(ToolFailed):
        sips.convert(sips.locate_sips("/usr/bin/sips"), source, tmp_path / "out.jpg", image_format="jpeg")


# --- fakes: failure classes ---------------------------------------------------------------------


def test_fake_success_parses_stdout(tmp_path: Path) -> None:
    tool = tool_for(images.fake_with(tmp_path, f"cat <<'EOF'\n{GOOD}EOF\n"))
    assert sips.inspect(tool, tmp_path / "a.png") == sips.ImageInfo("png", 64, 32, "RGB")


def test_fake_exit_zero_with_nil_fields_is_not_an_image(tmp_path: Path) -> None:
    tool = tool_for(images.fake_with(tmp_path, "echo '  format: jpg'; echo '  pixelWidth: <nil>'\n"))
    with pytest.raises(sips.NotAnImage) as caught:
        sips.inspect(tool, tmp_path / "a.png")
    assert caught.value.argv[0] == str(tool.path)


def test_fake_nonzero_exit_keeps_a_bounded_stderr_tail(tmp_path: Path) -> None:
    tool = tool_for(images.fake_with(tmp_path, "i=0; while [ $i -lt 100 ]; do echo \"line $i\" >&2; i=$((i+1)); done; exit 13\n"))
    with pytest.raises(ToolFailed) as caught:
        sips.inspect(tool, tmp_path / "a.png")
    assert caught.value.returncode == 13
    assert len(caught.value.stderr_tail) == 20 and caught.value.stderr_tail[-1] == "line 99"


def test_fake_large_stderr_does_not_block(tmp_path: Path) -> None:
    body = f"head -c 2000000 /dev/zero | tr '\\0' 'e' >&2; echo >&2; cat <<'EOF'\n{GOOD}EOF\n"
    tool = tool_for(images.fake_with(tmp_path, body))
    started = time.monotonic()
    assert sips.inspect(tool, tmp_path / "a.png", deadline_s=20).width_px == 64
    assert time.monotonic() - started < 15


def test_fake_timeout_terminates_the_process_group(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    tool = tool_for(images.fake_with(tmp_path, f"sleep 60 &\necho $! > '{pidfile}'\nwait\n"))
    started = time.monotonic()
    with pytest.raises(ToolTimeout) as caught:
        sips.inspect(tool, tmp_path / "a.png", deadline_s=0.5)
    assert caught.value.deadline_s == 0.5
    assert time.monotonic() - started < 10
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the grandchild of the timed-out call survived")


def test_fake_timeout_escalates_to_kill_after_the_grace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sips, "TERMINATE_GRACE_S", 0.3)
    tool = tool_for(images.fake_with(tmp_path, "trap '' TERM\nwhile :; do sleep 0.1; done\n"))
    started = time.monotonic()
    with pytest.raises(ToolTimeout):
        sips.inspect(tool, tmp_path / "a.png", deadline_s=0.3)
    assert time.monotonic() - started < 10


def test_executable_gone_after_preflight_is_tool_missing(tmp_path: Path) -> None:
    path = images.fake_with(tmp_path, "exit 0\n")
    tool = tool_for(path)
    path.unlink()
    with pytest.raises(ToolMissing):
        sips.inspect(tool, tmp_path / "a.png")


def test_executable_no_longer_runnable_is_tool_missing(tmp_path: Path) -> None:
    path = images.fake_with(tmp_path, "exit 0\n")
    tool = tool_for(path)
    path.chmod(0o600)
    with pytest.raises(ToolMissing):
        sips.inspect(tool, tmp_path / "a.png")


def test_cancellation_before_and_during_a_call(tmp_path: Path) -> None:
    tool = tool_for(images.fake_with(tmp_path, "sleep 60\n"))
    event = threading.Event()
    event.set()
    with pytest.raises(Cancelled):
        sips.inspect(tool, tmp_path / "a.png", cancel_event=event)
    event.clear()
    threading.Timer(0.3, event.set).start()
    started = time.monotonic()
    with pytest.raises(Cancelled):
        sips.inspect(tool, tmp_path / "a.png", deadline_s=30, cancel_event=event)
    assert time.monotonic() - started < 10


def test_convert_that_leaves_no_output_is_not_an_image(tmp_path: Path) -> None:
    tool = tool_for(images.fake_with(tmp_path, "exit 0\n"))
    with pytest.raises(sips.NotAnImage):
        sips.convert(tool, tmp_path / "a.png", tmp_path / "b.jpg", image_format="jpeg")


def test_the_call_never_goes_through_a_shell(tmp_path: Path) -> None:
    hostile = tmp_path / "a; touch pwned.png"
    tool = tool_for(images.fake_with(tmp_path, f"printf '%s' \"$@\" > '{tmp_path / 'args'}'\n"))
    with pytest.raises(ToolError):
        sips.inspect(tool, hostile)
    assert hostile.name in (tmp_path / "args").read_text()
    assert not (tmp_path / "pwned.png").exists()


# --- layering -------------------------------------------------------------------------------------


def test_sips_is_not_imported_at_package_start() -> None:
    code = ("import sys\nimport norma, norma.adapters, norma.metadata, norma.models, norma.errors\n"
            "print('norma.adapters.sips' in sys.modules, 'norma.metadata.artwork' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False"]
