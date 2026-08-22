"""Where this installation keeps its caches.

Model weights are large and slow to fetch, so where they land is a real
decision rather than an implementation detail. The default is a ``.cache/``
directory inside the checkout, which keeps a clone self-contained: the code,
the evaluation data and the several gigabytes of weights it depends on all
move together.

Resolution order:

1. ``STT_CACHE_DIR`` if set — an absolute path, ``~`` expanded.
2. ``<checkout>/.cache`` when this package was imported from a source tree.
3. ``~/.cache`` otherwise, which is what an installed wheel gets.

Set ``STT_CACHE_DIR=~/.cache`` to restore the conventional per-user location.

One runtime cannot be redirected: fairseq2 hardcodes ``~/.cache/fairseq2`` and
exposes no environment variable for it, so making its checkpoints repo-local
means turning that directory into a symlink. :func:`fairseq2_status` reports
whether that has been done; nothing here does it silently.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Override for the cache root. Absolute, ``~`` accepted.
ENV_VAR = "STT_CACHE_DIR"

#: fairseq2 reads this path directly and offers no way to change it.
FAIRSEQ2_HOME = Path.home() / ".cache" / "fairseq2"

#: Subdirectory per runtime, so one store can be cleared without the others.
RUNTIME_DIRS = ("huggingface", "dolphin", "crispasr", "fairseq2", "stt")


def checkout_root() -> Path | None:
    """The source tree this package lives in, or None if it is installed."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else None


def cache_root() -> Path:
    """Directory holding every cache this project controls."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    root = checkout_root()
    return root / ".cache" if root else Path.home() / ".cache"


def cache_dir(name: str, create: bool = True) -> Path:
    """One runtime's cache directory beneath :func:`cache_root`."""
    path = cache_root() / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configure_environment() -> None:
    """Point third-party runtimes at our cache root.

    Called from ``stt/__init__.py`` because ``huggingface_hub`` reads ``HF_HOME``
    once, at import time — setting it later has no effect. Existing values are
    left alone, so an explicit ``HF_HOME`` still wins.
    """
    os.environ.setdefault("HF_HOME", str(cache_root() / "huggingface"))


def fairseq2_status() -> tuple[str, Path | None]:
    """How ``~/.cache/fairseq2`` is currently arranged.

    Returns one of ``absent``, ``local`` (a real directory in the home cache),
    ``linked`` (a symlink into our cache root) or ``elsewhere`` (a symlink
    somewhere else), with the resolved target where there is one.
    """
    if not FAIRSEQ2_HOME.exists() and not FAIRSEQ2_HOME.is_symlink():
        return "absent", None
    if not FAIRSEQ2_HOME.is_symlink():
        return "local", FAIRSEQ2_HOME
    target = FAIRSEQ2_HOME.resolve()
    wanted = (cache_root() / "fairseq2").resolve()
    return ("linked" if target == wanted else "elsewhere"), target


def directory_size_mb(path: Path) -> float | None:
    """Total size of ``path`` in MB, or None if it does not exist."""
    if not path.exists():
        return None
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:  # noqa: PERF203 - a vanishing temp file is not an error
            continue
    return total / (1024 * 1024)
