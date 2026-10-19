<h1>
  <picture>
    <source media="(prefers-reduced-motion: reduce)" srcset="assets/norma-wordmark.png">
    <img src="assets/norma-wordmark.gif" alt="Norma" width="814" height="150">
  </picture>
</h1>

A macOS command-line tool for measuring and normalising audio loudness to a chosen LUFS target and true-peak ceiling. Norma writes new copies with preserved or edited metadata, leaving the originals untouched.

**Version:** 0.1.0 · macOS · [GPL-3.0-or-later](#licence)

## Features

- Measure integrated loudness, true peak and loudness range; normalise with limiting, gain reduction or dynamic processing. Defaults: −16 LUFS and −1 dBTP.
- Process files or folders, including subfolders, and report failures without stopping the batch.
- Write WAV, FLAC, ALAC, AIFF, MP3 or AAC. Use the `apple` preset for ALAC listening copies.
- Preserve or edit metadata and artwork using command-line options, CSV/JSON manifests or filename patterns. Report fields that cannot be carried over.
- Preview changes, verify outputs and export CSV or JSON reports. Audio processing is entirely local.

Supports mono and stereo audio. See `norma formats` and the [example configurations and manifests](specs/examples/).

## Safety

- Originals are never modified. Copies go into `Normalised/` beside the inputs unless you choose another location.
- Existing outputs and reports require `--overwrite` to replace them; inputs cannot be overwritten.
- Processing marks prevent accidental reprocessing; `--force` bypasses that guard, not other safety or no-op checks.
- Norma never opens or imports into your Music library. Use `--dry-run` to preview normalisation.

## Commands

Use `norma COMMAND --help` for options and defaults.

| Command | What it does |
|---|---|
| `norma analyse` | Measure loudness and predict gain. |
| `norma normalise` | Write normalised copies; `--dry-run` previews without writing audio. |
| `norma verify` | Check audio; add `--check-tags` to check marks and explicit tag references. Never repairs. |
| `norma tags show` / `plan` / `template` | Inspect tags, preview edits or export an editable manifest. |
| `norma formats` | Show format capabilities. |
| `norma doctor` | Check required tools. |

### Example: prepare an EP for a listening review

With mixes in `After Hours/Mixes/`, prepare tagged ALAC listening copies. The −16 LUFS / −1 dBTP settings are an example, not a delivery standard or a substitute for creative mastering.

```bash
# Check tools and measure.
norma doctor
norma analyse "After Hours/Mixes" -t -16 -p -1

# Preview.
norma normalise "After Hours/Mixes" -f apple -t -16 -p -1 \
  --artist "dohmansfi" --album "After Hours" \
  -o "After Hours/Listening Copies" --dry-run

# Write copies after reviewing the preview.
norma normalise "After Hours/Mixes" -f apple -t -16 -p -1 \
  --artist "dohmansfi" --album "After Hours" \
  -o "After Hours/Listening Copies"

# Verify audio and tags.
norma verify "After Hours/Listening Copies" -t -16 -p -1 --check-tags \
  --artist "dohmansfi" --album "After Hours"
```

Review warnings before importing the copies into Apple Music yourself.

Optional preferences: `~/.config/norma/config.toml`. Command-line options override the file; built-in defaults apply otherwise.

## Dependencies

- macOS, Python 3.11+ and the built-in `sips` tool.
- FFmpeg with a matching `ffprobe` beside it. Norma does not bundle them; `norma doctor` checks the required filters and encoders. Use `--ffmpeg PATH` to select a specific pair.
- Standalone `uv` for the installation below. The Python dependency, `mutagen`, is installed automatically.

## Install from a local checkout

Install from the repository; Norma is not on a package index.

1. Install `uv` with its standalone installer.
2. Run `git clone https://github.com/dohmansfi/norma.git`, then `uv tool install ./norma`. If prompted, run `uv tool update-shell` and open a new Terminal window.
3. Check with `norma --version` and `norma doctor`.

## Known limitations

- **WAV tags in Apple Music:** Music 1.5.6.11 on macOS 15.6.1 imported both tested variants (INFO+ID3 and INFO-only) but ignored their title, artist, album and track tags. The cause is unknown and unfixed; other files or versions may behave differently. Norma verifies file tags, not Music's display.
- **MP3:** verification relies on FFmpeg's self-consistent statistics. V0 at very low sample rates is refused when its header cannot be distinguished from CBR.
- **Dynamic processing:** heavy ultrasonic content can cause a loudness-target miss; Norma reports failed verification rather than accepting the output.
- **Metadata:** unsupported fields are reported as losses. Bare numeric ID3 genres such as `17` are refused; literal genre text starting with `(` is escaped.
- No Finder Quick Actions are included.

## Licence

Copyright (C) 2026 dohmansfi. Licensed under [GPL-3.0-or-later](LICENSE). See [NOTICE](NOTICE) for application and third-party notices.
