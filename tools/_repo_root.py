# -*- coding: utf-8 -*-
"""Shared repository-root resolution for tools.

Resolves the ByteTrack repository root from any script location by walking
upward until a directory containing both `.git` and `yolox` (or either marker,
depending on the checkout) is found.  This removes fragile `os.getcwd()` and
`Path("datasets")` assumptions.
"""
from pathlib import Path

_MARKERS = (".git", "yolox", "YOLOX_outputs")


def get_repo_root(start=None):
    """Return the absolute Path of the repository root.

    ``start`` defaults to the directory containing this helper file.  The
    search walks upward from ``start``; the first directory that contains at
    least one repository marker is returned.  If no marker is found, the
    filesystem root is returned (which will usually make the caller fail
    loudly with a clear path error).
    """
    if start is None:
        current = Path(__file__).resolve().parent
    else:
        current = Path(start).resolve()
        if current.is_file():
            current = current.parent

    while True:
        if any((current / marker).exists() for marker in _MARKERS):
            return current
        parent = current.parent
        if parent == current:
            return current
        current = parent


REPO_ROOT = get_repo_root()
