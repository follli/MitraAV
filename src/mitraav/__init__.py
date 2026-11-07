"""MitraAV: a local, first-step OSINT lead finder."""
from importlib import metadata

try:
    __version__ = metadata.version("mitraav")
except metadata.PackageNotFoundError:            # running from a source tree that is not installed
    __version__ = "0.0.0"
