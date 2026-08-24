"""Detect hardware capabilities without hardcoding Apple chip generations.

Core layout and RAM are read at runtime for reporting and memory decisions.
Only the dtype benchmark is cached, keyed by chip, device, and torch version.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE = Path.home() / ".cache" / "stt" / "hardware.json"

#: macOS's name for the slow cores. Anything not called this is worth computing
#: on, which is what keeps the logic correct on chips that have no such level.
EFFICIENCY = "efficiency"


def _sysctl(key: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        ).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001 - absent keys are normal on other hardware
        return None


@dataclass(frozen=True)
class CoreLayout:
    """Physical core counts, keyed by what macOS calls each performance level."""

    #: ``(name, physical_cores)`` from fastest level to slowest.
    levels: tuple[tuple[str, int], ...]

    @property
    def compute(self) -> int:
        """Cores not named efficiency cores, reported but never imposed."""
        n = sum(count for name, count in self.levels if EFFICIENCY not in name.lower())
        return n or self.total

    @property
    def efficiency(self) -> int:
        return sum(count for name, count in self.levels if EFFICIENCY in name.lower())

    @property
    def total(self) -> int:
        return sum(count for _, count in self.levels) or (os.cpu_count() or 1)


def core_layout() -> CoreLayout:
    """Read the CPU's performance levels from macOS."""
    levels: list[tuple[str, int]] = []
    try:
        count = int(_sysctl("hw.nperflevels") or 0)
    except ValueError:
        count = 0

    for i in range(count):
        name = _sysctl(f"hw.perflevel{i}.name") or f"level{i}"
        try:
            physical = int(_sysctl(f"hw.perflevel{i}.physicalcpu") or 0)
        except ValueError:
            physical = 0
        if physical:
            levels.append((name, physical))

    if not levels:  # non-Apple, or sysctl unavailable
        levels = [("unknown", os.cpu_count() or 1)]
    return CoreLayout(tuple(levels))


def chip_name() -> str:
    """e.g. ``Apple M2 Max``. Used only as a cache key and for reporting."""
    return _sysctl("machdep.cpu.brand_string") or "unknown"


# ------------------------------------------------------------- dtype probing

#: Half formats considered for memory-bounded GPU inference. float32 is timed as
#: a control and used as a fallback when neither half format works.
_HALF_DTYPES = ("float16", "bfloat16")
_DTYPES = (*_HALF_DTYPES, "float32")


def _probe_dtypes(device: str, size: int = 4096, iterations: int = 6) -> dict[str, float]:
    """Time a matmul per dtype on ``device``. Returns GFLOP/s, higher is better."""
    import torch

    results: dict[str, float] = {}
    for name in _DTYPES:
        try:
            dtype = getattr(torch, name)
            a = torch.randn(size, size, device=device, dtype=dtype)
            b = torch.randn(size, size, device=device, dtype=dtype)
            for _ in range(2):
                a @ b
            if device == "mps":
                torch.mps.synchronize()

            import time

            started = time.perf_counter()
            for _ in range(iterations):
                a @ b
            if device == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started

            if elapsed > 0:
                results[name] = iterations * 2 * size**3 / elapsed / 1e9
        except Exception:  # noqa: BLE001 - an unsupported dtype is a valid answer
            continue
    return results


def _load_cache() -> dict[str, Any]:
    try:
        return json.loads(CACHE.read_text())
    except Exception:  # noqa: BLE001 - a missing or corrupt cache just means re-probe
        return {}


def _save_cache(data: dict[str, Any]) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, indent=2))
    except Exception:  # noqa: BLE001 - caching is an optimisation, never a requirement
        pass


def fastest_dtype(device: str, refresh: bool = False) -> str:
    """Name of the fastest working dtype on ``device``, measured once and cached.

    Prefers the fastest supported half format and falls back to float32. The
    cache key includes the torch version because kernel performance can change
    with the runtime.
    """
    import torch

    if device == "cpu":
        # CPU inference uses float32; half precision is handled as a separate
        # memory decision by the backend.
        return "float32"

    key = f"{chip_name()}|{device}|torch{torch.__version__}"
    cache = _load_cache()
    if not refresh and key in cache:
        return str(cache[key]["dtype"])

    scores = _probe_dtypes(device)
    halves = {k: v for k, v in scores.items() if k in _HALF_DTYPES}
    if not halves:
        return "float32"
    best = max(halves, key=lambda k: halves[k])
    cache[key] = {"dtype": best, "gflops": {k: round(v) for k, v in scores.items()}}
    _save_cache(cache)
    return best


def describe() -> dict[str, Any]:
    """Everything worth recording alongside a benchmark result.

    All of it is read from the machine. A timing without this is not
    reproducible, and a chip name alone does not distinguish a 32 GB machine
    from a 64 GB one running the same model at different precisions.
    """
    import platform

    layout = core_layout()
    return {
        "chip": chip_name(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cores": dict(layout.levels),
        "compute_threads": layout.compute,
        "cpu_count": layout.total,
        "ram_mb": total_ram_mb(),
    }


def total_ram_mb() -> int | None:
    """Physical memory in MB, or ``None`` where it cannot be determined."""
    raw = _sysctl("hw.memsize")
    try:
        return int(raw) // (1024 * 1024) if raw else None
    except ValueError:
        return None
