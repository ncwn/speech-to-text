"""What this Apple Silicon machine can actually do quickly.

Two choices matter enough to get right, and both are hardware-dependent in ways
that are easy to get wrong by assumption:

**Which 16-bit format.** Apple's GPUs are built around float16. bfloat16 is
accepted everywhere but is not equally accelerated: on an M2 Max a 4096²
matmul runs at 11,391 GFLOP/s in float16 and 5,348 in bfloat16 — bfloat16 is
2.1× *slower* than float16, and slower than float32. Whether any given
generation accelerates bfloat16 is disputed and changes between chips, so this
module measures the machine in front of it instead of consulting a table that
will be wrong for the next one.

**Which cores.** Apple splits the CPU into performance and efficiency cores,
and scheduling compute onto efficiency cores costs throughput. The split is not
a constant: an M2 Max is 8 performance + 4 efficiency, while an M5 Pro has no
efficiency cores at all — it pairs super cores with performance cores. macOS
names each level (``hw.perflevel0.name``), so the names decide, not the counts
and not the chip's marketing name.

Everything here is cached: the probe runs once per machine, not once per run.
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
        """Cores worth running compute on — everything but the efficiency ones.

        On an M2 Max this is 8 of 12. On an M5 Pro, whose levels are super and
        performance with no efficiency tier, it is all of them.
        """
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


def compute_threads() -> int:
    """Default thread count for CPU work: performance cores only.

    Handing work to efficiency cores as well tends to cost more in stragglers
    than the extra cores contribute, because a parallel step finishes with its
    slowest thread.
    """
    return core_layout().compute


# ------------------------------------------------------------- dtype probing

#: The choice that actually matters. Weights are held in 16 bits on the GPU for
#: memory reasons regardless — a 7B card at float32 would want ~28 GB — so the
#: open question is only *which* 16-bit format this chip runs faster. float32 is
#: timed too, as a control: a 16-bit format losing to it means it is emulated.
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

    Chooses between float16 and bfloat16 only. float32 is timed as a control but
    never selected: 16-bit is a memory decision already made, and on this class
    of model float32 weights do not fit comfortably on the GPU.

    Measured rather than looked up because the answer genuinely differs across
    Apple generations and the public accounts of which chips accelerate bfloat16
    disagree with each other. The cache key includes the torch version, since a
    kernel that is slow today may not be after an upgrade.
    """
    import torch

    if device == "cpu":
        # PyTorch has no native half-precision CPU kernels and emulates them:
        # measured at 4.1x slower than float32 on the 7B, for identical text.
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
    """Everything worth recording alongside a benchmark result."""
    layout = core_layout()
    return {
        "chip": chip_name(),
        "cores": {name: n for name, n in layout.levels},
        "compute_threads": layout.compute,
        "cpu_count": layout.total,
    }


def tune_torch_threads() -> int | None:
    """Point torch's CPU thread pool at the performance cores.

    Torch's own default is usually right on an M2 Max, but it is derived from
    the fastest performance level alone. That undercounts a chip whose second
    level is also fast — an M5 Pro pairs super cores with performance cores and
    has no efficiency tier, so taking only the top level would leave two thirds
    of the CPU idle.

    Returns the thread count set, or ``None`` if torch is not installed.
    """
    try:
        import torch
    except ImportError:
        return None

    threads = compute_threads()
    if threads and threads != torch.get_num_threads():
        torch.set_num_threads(threads)
    return threads
