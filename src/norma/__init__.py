"""Norma: analyse and normalise the loudness of music demo files.

The package initialiser is inert: it probes nothing, starts no process and
writes nothing (ARCHITECTURE § 3). The version comes from the installed
distribution metadata, so pyproject.toml stays its only home.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("norma")
except PackageNotFoundError:  # running from a source tree without installed metadata
    __version__ = "0+unknown"

__all__ = ["__version__"]
