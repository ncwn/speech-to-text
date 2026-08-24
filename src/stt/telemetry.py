"""What a transcription actually cost, beyond wall-clock time.

The real-time factor tells you a run was slow. It does not tell you *why*, and
the difference matters when deciding what to optimise:

* a backend pinned at one core is serialised, and threading will help;
* one using eight cores is compute-bound, and only a smaller model or a
  different runtime will help;
* one that is slow while barely using the CPU is waiting — on memory
  bandwidth, on the GPU, or on I/O — and none of the above will help.

Everything here comes from the standard library, so measurement costs no new
dependency and cannot itself fail a run.
"""

from __future__ import annotations

import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

#: ``ru_maxrss`` is bytes on macOS/BSD and kilobytes on Linux.
_MAXRSS_TO_MB = 1 / (1024 * 1024) if sys.platform == "darwin" else 1 / 1024


@dataclass
class ResourceUsage:
    """Resources consumed while transcribing one file."""

    wall_s: float
    #: User + system CPU time. Sums across threads, so it exceeds ``wall_s``
    #: whenever more than one core was working.
    cpu_s: float
    #: Process high-water RSS. This is monotonic for the life of the process,
    #: so on the second file it reports the peak of *everything so far* — it
    #: answers "how much memory does this need" rather than "how much did this
    #: file use". Model weights dominate it.
    peak_rss_mb: float
    #: Metal/CUDA memory the driver had allocated when the file finished, if
    #: the backend runs on a GPU we can query. ggml's Metal backend allocates
    #: outside torch and is invisible here.
    gpu_mb: float | None = None

    @property
    def cpu_utilization(self) -> float | None:
        """Cores kept busy on average — 1.0 is one core saturated."""
        if self.wall_s <= 0:
            return None
        return self.cpu_s / self.wall_s

    def to_dict(self) -> dict[str, Any]:
        d = {
            "wall_s": round(self.wall_s, 4),
            "cpu_s": round(self.cpu_s, 4),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
        }
        if self.cpu_utilization is not None:
            d["cpu_utilization"] = round(self.cpu_utilization, 2)
        if self.gpu_mb is not None:
            d["gpu_mb"] = round(self.gpu_mb, 1)
        return d


def _cpu_seconds() -> float:
    """User + system CPU time for this process and any children it waited on."""
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


def peak_rss_mb() -> float:
    """Process high-water resident set size, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MAXRSS_TO_MB


def gpu_allocated_mb() -> float | None:
    """GPU memory currently allocated through torch, if there is any.

    Returns ``None`` rather than 0.0 when there is no GPU or torch is not
    loaded, so "not measured" stays distinguishable from "measured as zero".
    Deliberately does not import torch: a backend that never loaded it should
    not pay a multi-second import just to be told the answer is None.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if torch.cuda.is_available() and torch.cuda.memory_allocated():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:  # noqa: BLE001 - telemetry must never break a run
        return None
    return None


@contextmanager
def measure():
    """Measure a block of work.

    Yields a one-element list that receives the :class:`ResourceUsage` on exit,
    so the caller can read it after the block::

        with measure() as usage:
            results = backend.transcribe([path])
        print(usage[0].cpu_utilization)

    Populated even when the block raises, so a failed decode still reports what
    it consumed before giving up.
    """
    box: list[ResourceUsage] = []
    started, cpu_started = time.perf_counter(), _cpu_seconds()
    try:
        yield box
    finally:
        box.append(
            ResourceUsage(
                wall_s=time.perf_counter() - started,
                cpu_s=max(0.0, _cpu_seconds() - cpu_started),
                peak_rss_mb=peak_rss_mb(),
                gpu_mb=gpu_allocated_mb(),
            )
        )


def describe_host() -> dict[str, Any]:
    """Static facts about the machine, recorded so runs stay comparable.

    A number measured on a busy laptop is not the same number measured on an
    idle one, and comparing across machines without this is meaningless.
    Detection lives in :mod:`stt.hardware`; this is the recording end of it.
    """
    from stt.hardware import describe

    return describe()
