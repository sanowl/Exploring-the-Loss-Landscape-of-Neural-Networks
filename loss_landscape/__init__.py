"""Minimal package for loss-landscape exploration."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package not installed in editable / site-packages mode
    __version__ = "0.0.0" 