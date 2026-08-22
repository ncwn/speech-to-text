"""Resource measurement for transcription runs.

The counters in this module deliberately separate exact totals from sampled
series. ``ru_maxrss`` and elapsed CPU time are useful totals, while a GPU mean
without its sample count can make a one-reading observation look authoritative.
The profiler therefore keeps raw samples and derives all summaries from them.
"""

from __future__ import annotations

import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# ``ru_maxrss`` is bytes on macOS/BSD and kilobytes on Linux.
_MAXRSS_TO_MB = 1 / (1024 * 1024) if sys.platform == "darwin" else 1 / 1024
MIN_SAMPLES = 5


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile with stable small-sample behaviour."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _time_weights(offsets_ns: list[int], window_ns: int) -> list[float]:
    """Give irregular samples midpoint-bounded coverage over a work window."""
    if len(offsets_ns) == 1:
        return [float(max(1, window_ns))]
    boundaries = [0.0]
    boundaries.extend(
        (left + right) / 2 for left, right in zip(offsets_ns, offsets_ns[1:], strict=False)
    )
    boundaries.append(float(window_ns))
    return [max(0.0, right - left) for left, right in zip(boundaries, boundaries[1:], strict=False)]


def _weighted_percentile(values: list[float], weights: list[float], fraction: float) -> float:
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return _percentile(values, fraction)
    target = total * fraction
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


@dataclass
class Series:
    """A sampled resource series and recomputable summary statistics."""

    n: int
    mean: float
    p50: float
    p95: float
    max: float
    idle_pct: float
    samples: list[float]
    offsets_ns: list[int] = field(default_factory=list)
    window_ns: int | None = None
    idle_threshold: float = 0.0
    source: str | None = None
    scope: str | None = None
    phase: str | None = None

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[float],
        *,
        idle_threshold: float = 0.0,
        offsets_ns: Iterable[int] | None = None,
        window_ns: int | None = None,
        source: str | None = None,
        scope: str | None = None,
        phase: str | None = None,
    ) -> Series:
        values = [float(value) for value in samples]
        if not values:
            raise ValueError("a Series needs at least one sample")
        offsets = [int(offset) for offset in offsets_ns] if offsets_ns is not None else []
        if offsets and len(offsets) != len(values):
            raise ValueError("offsets_ns must have one entry per sample")
        if offsets and (offsets != sorted(offsets) or offsets[0] < 0):
            raise ValueError("offsets_ns must be sorted and non-negative")
        if offsets and (window_ns is None or window_ns < offsets[-1]):
            raise ValueError("window_ns must include every sample offset")

        if offsets and window_ns is not None:
            weights = _time_weights(offsets, window_ns)
            total_weight = sum(weights)
            mean = sum(value * weight for value, weight in zip(values, weights, strict=True)) / (
                total_weight or 1.0
            )
            p50 = _weighted_percentile(values, weights, 0.50)
            p95 = _weighted_percentile(values, weights, 0.95)
            idle_pct = (
                100.0
                * sum(
                    weight
                    for value, weight in zip(values, weights, strict=True)
                    if value <= idle_threshold
                )
                / (total_weight or 1.0)
            )
        else:
            mean = sum(values) / len(values)
            p50 = _percentile(values, 0.50)
            p95 = _percentile(values, 0.95)
            idle_pct = 100.0 * sum(value <= idle_threshold for value in values) / len(values)
        return cls(
            n=len(values),
            mean=mean,
            p50=p50,
            p95=p95,
            max=max(values),
            idle_pct=idle_pct,
            samples=values,
            offsets_ns=offsets,
            window_ns=window_ns,
            idle_threshold=idle_threshold,
            source=source,
            scope=scope,
            phase=phase,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Series:
        """Rebuild a series, trusting raw samples when they are available."""
        raw = data.get("samples")
        if raw:
            series = cls.from_samples(
                raw,
                idle_threshold=float(data.get("idle_threshold", 0.0)),
                offsets_ns=data.get("offsets_ns"),
                window_ns=data.get("window_ns"),
                source=data.get("source"),
                scope=data.get("scope"),
                phase=data.get("phase"),
            )
            if "idle_pct" in data:
                series.idle_pct = float(data["idle_pct"])
            return series
        return cls(
            n=int(data.get("n", 0)),
            mean=float(data.get("mean", 0.0)),
            p50=float(data.get("p50", data.get("mean", 0.0))),
            p95=float(data.get("p95", data.get("max", data.get("mean", 0.0)))),
            max=float(data.get("max", data.get("mean", 0.0))),
            idle_pct=float(data.get("idle_pct", 0.0)),
            samples=[],
            offsets_ns=[],
            window_ns=data.get("window_ns"),
            idle_threshold=float(data.get("idle_threshold", 0.0)),
            source=data.get("source"),
            scope=data.get("scope"),
            phase=data.get("phase"),
        )

    def to_dict(self, *, include_series: bool = True) -> dict[str, Any]:
        data = {
            "n": self.n,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "max": self.max,
            "idle_pct": self.idle_pct,
            "samples": list(self.samples),
        }
        if self.offsets_ns:
            data["offsets_ns"] = list(self.offsets_ns)
        if self.window_ns is not None:
            data["window_ns"] = self.window_ns
        if self.idle_threshold:
            data["idle_threshold"] = self.idle_threshold
        for name in ("source", "scope", "phase"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data


def pool_series(
    series: Iterable[Series | None],
    *,
    idle_threshold: float = 0.0,
) -> Series | None:
    """Concatenate resource windows for a descriptive aggregate."""
    present = [item for item in series if item is not None and item.samples]
    if not present:
        return None
    if all(item.offsets_ns and item.window_ns is not None for item in present):
        samples: list[float] = []
        offsets: list[int] = []
        cursor = 0
        for item in present:
            samples.extend(item.samples)
            offsets.extend(cursor + offset for offset in item.offsets_ns)
            cursor += item.window_ns or 0
        return Series.from_samples(
            samples,
            idle_threshold=idle_threshold,
            offsets_ns=offsets,
            window_ns=cursor,
            source=present[0].source if len({item.source for item in present}) == 1 else "mixed",
            scope="pooled-windows",
            phase=present[0].phase if len({item.phase for item in present}) == 1 else "mixed",
        )
    samples = [sample for item in present for sample in item.samples]
    return Series.from_samples(samples, idle_threshold=idle_threshold)


@dataclass
class ResourceUsage:
    """Resources consumed while transcribing one file or one batch."""

    wall_s: float
    cpu_s: float
    peak_rss_mb: float
    started_ns: int | None = None
    ended_ns: int | None = None
    rss_peak_mb: float | None = None
    process_peak_rss_mb: float | None = None
    phase: str | None = None
    gpu_mb: float | None = None
    # Legacy summaries remain in the schema so old evidence files load. New
    # writers also persist the raw ``gpu_util`` series below.
    gpu_util_mean: float | None = None
    gpu_util_peak: float | None = None
    sampler_cpu_s: float = 0.0
    profiler_cpu_s: float = 0.0
    cpu: Series | None = None
    rss: Series | None = None
    #: Unique set size, collected only when explicitly asked for. See
    #: :func:`uss_now_mb` for why it is not on by default.
    uss: Series | None = None
    gpu_util: Series | None = None
    gpu_mem: Series | None = None

    def __post_init__(self) -> None:
        if self.gpu_util is not None:
            if self.gpu_util_mean is None:
                self.gpu_util_mean = self.gpu_util.mean
            if self.gpu_util_peak is None:
                self.gpu_util_peak = self.gpu_util.max

    @property
    def cpu_utilization(self) -> float | None:
        """Cores kept busy on average -- 1.0 is one core saturated."""
        if self.wall_s <= 0:
            return None
        return self.cpu_s / self.wall_s

    def to_dict(self, *, include_series: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "wall_s": round(self.wall_s, 4),
            "cpu_s": round(self.cpu_s, 4),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
        }
        if self.started_ns is not None:
            d["started_ns"] = self.started_ns
        if self.ended_ns is not None:
            d["ended_ns"] = self.ended_ns
        if self.rss_peak_mb is not None:
            d["rss_peak_mb"] = round(self.rss_peak_mb, 1)
        if self.process_peak_rss_mb is not None:
            d["process_peak_rss_mb"] = round(self.process_peak_rss_mb, 1)
        if self.phase is not None:
            d["phase"] = self.phase
        if self.cpu_utilization is not None:
            d["cpu_utilization"] = round(self.cpu_utilization, 2)
        if self.gpu_mb is not None:
            d["gpu_mb"] = round(self.gpu_mb, 1)
        if self.gpu_util_mean is not None:
            d["gpu_util_mean"] = round(self.gpu_util_mean, 1)
            d["gpu_util_peak"] = round(self.gpu_util_peak or 0.0, 1)
        if self.sampler_cpu_s:
            d["sampler_cpu_s"] = round(self.sampler_cpu_s, 4)
        if self.profiler_cpu_s:
            d["profiler_cpu_s"] = round(self.profiler_cpu_s, 4)
        if include_series:
            for name in ("cpu", "rss", "uss", "gpu_util", "gpu_mem"):
                value = getattr(self, name)
                if value is not None:
                    d[name] = value.to_dict()
        return d


def _self_cpu_seconds() -> float:
    """User + system CPU time for this process and all of its threads."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _cpu_seconds(*, include_children: bool = True) -> float:
    """Process CPU time, optionally including waited-for child processes."""
    total = _self_cpu_seconds()
    if not include_children:
        return total
    return total + _child_cpu_seconds()


def _child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MAXRSS_TO_MB


_PROCESS: Any = None


def rss_now_mb() -> float | None:
    """Current RSS, falling back to the process high-water mark if needed."""
    global _PROCESS
    try:
        if _PROCESS is None:
            import psutil

            _PROCESS = psutil.Process()
        return _PROCESS.memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 - telemetry must never break a run
        return peak_rss_mb()


def uss_now_mb() -> float | None:
    """Unique set size — memory that would be freed if this process exited.

    RSS counts shared pages, so a worker that memory-maps a 31 GB checkpoint
    reports memory it does not exclusively own. USS is the honest figure for
    "what did this model cost", but it is measurably dearer to collect: about
    0.1 ms per sample against 0.005 ms for RSS on an M2 Max, because it walks
    the process's memory map rather than reading a counter.

    That 20x is why this is opt-in and diagnostic rather than the default
    memory series, and why the observer calibration measures it rather than
    assuming it is free. ``None`` means the platform refused the read, which is
    common enough — some systems require elevated privileges — that it must
    never take the sampler down with it.
    """
    global _PROCESS
    try:
        if _PROCESS is None:
            import psutil

            _PROCESS = psutil.Process()
        return _PROCESS.memory_full_info().uss / (1024 * 1024)
    except Exception:  # noqa: BLE001 - telemetry must never break a run
        return None


def gpu_allocated_mb() -> float | None:
    """GPU memory currently allocated through torch, if there is any."""
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


# ioreg is expensive and observer-sensitive. Keep the GPU cadence at 250 ms;
# CPU and RSS are collected at the cheaper 50 ms profiler cadence.
SAMPLE_INTERVAL_S = 0.05
GPU_SAMPLE_INTERVAL_S = 0.25

_UTILISATION = re.compile(r'"Device Utilization %"=(\d+)')


def gpu_utilization_now() -> float | None:
    """Whole-GPU busy percentage, or ``None`` when ioreg is unavailable."""
    if shutil.which("ioreg") is None:
        return None
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:  # noqa: BLE001 - telemetry must never break a run
        return None
    found = _UTILISATION.search(out)
    return float(found.group(1)) if found else None


class Profiler:
    """Sample CPU/RSS frequently and GPU at a non-perturbing cadence."""

    def __init__(
        self,
        interval_s: float = SAMPLE_INTERVAL_S,
        gpu_interval_s: float = GPU_SAMPLE_INTERVAL_S,
        settle_s: float = SAMPLE_INTERVAL_S,
        sample_gpu: bool = True,
        sample_uss: bool = False,
    ) -> None:
        self.interval_s = max(0.001, interval_s)
        self.gpu_interval_s = max(self.interval_s, gpu_interval_s)
        self.gpu_every = max(1, round(self.gpu_interval_s / self.interval_s))
        self.settle_s = max(0.0, settle_s)
        self.sample_gpu = sample_gpu
        self.sample_uss = sample_uss
        self.samples: list[float] = []  # compatibility alias for GPU samples
        self.cpu_samples: list[float] = []
        self.rss_samples: list[float] = []
        self.uss_samples: list[float] = []
        self.gpu_mem_samples: list[float] = []
        self.gpu_offsets_ns: list[int] = []
        self.cpu_offsets_ns: list[int] = []
        self.rss_offsets_ns: list[int] = []
        self.uss_offsets_ns: list[int] = []
        self.gpu_mem_offsets_ns: list[int] = []
        self.cpu_cost_s = 0.0
        self.thread_cpu_s = 0.0
        self._stop = threading.Event()
        self._begin = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_ns: int | None = None
        self._ended_ns: int | None = None
        self.phase: str | None = None

    def _offset(self, timestamp_ns: int) -> int | None:
        if self._started_ns is None:
            return None
        if timestamp_ns < self._started_ns:
            return None
        if self._ended_ns is not None and timestamp_ns > self._ended_ns:
            return None
        return timestamp_ns - self._started_ns

    def _sample_gpu(self) -> None:
        before = _child_cpu_seconds()
        value = gpu_utilization_now()
        self.cpu_cost_s += max(0.0, _child_cpu_seconds() - before)
        timestamp_ns = time.perf_counter_ns()
        offset = self._offset(timestamp_ns)
        if value is not None and offset is not None:
            self.samples.append(value)
            self.gpu_offsets_ns.append(offset)
        memory = gpu_allocated_mb()
        memory_timestamp_ns = time.perf_counter_ns()
        memory_offset = self._offset(memory_timestamp_ns)
        if memory is not None and memory_offset is not None:
            self.gpu_mem_samples.append(memory)
            self.gpu_mem_offsets_ns.append(memory_offset)

    def _sample_cpu_rss(
        self,
        previous: tuple[float, float, float] | None,
    ) -> tuple[float, float, float]:
        now_ns = time.perf_counter_ns()
        now = now_ns / 1_000_000_000
        cpu_now = _self_cpu_seconds()
        child_now = _child_cpu_seconds()
        offset = self._offset(now_ns)
        if previous is not None:
            previous_time, previous_cpu, previous_child = previous
            elapsed = now - previous_time
            if elapsed > 0 and offset is not None:
                self.cpu_samples.append(max(0.0, (cpu_now - previous_cpu) / elapsed))
                self.cpu_offsets_ns.append(offset)
        rss = rss_now_mb()
        rss_timestamp_ns = time.perf_counter_ns()
        rss_offset = self._offset(rss_timestamp_ns)
        if rss is not None and rss_offset is not None:
            self.rss_samples.append(rss)
            self.rss_offsets_ns.append(rss_offset)
        if self.sample_uss:
            uss = uss_now_mb()
            uss_offset = self._offset(time.perf_counter_ns())
            if uss is not None and uss_offset is not None:
                self.uss_samples.append(uss)
                self.uss_offsets_ns.append(uss_offset)
        return now, cpu_now, child_now

    def _run(self) -> None:
        thread_started = time.thread_time()
        try:
            self._begin.wait()
            if self._stop.is_set():
                return
            previous: tuple[float, float, float] | None = (
                time.perf_counter(),
                _self_cpu_seconds(),
                _child_cpu_seconds(),
            )
            deadline = time.perf_counter() + self.settle_s
            tick = 0
            while not self._stop.wait(max(0.0, deadline - time.perf_counter())):
                tick += 1
                previous = self._sample_cpu_rss(previous)
                if self.sample_gpu and tick % self.gpu_every == 0:
                    self._sample_gpu()
                deadline += self.interval_s
        finally:
            self.thread_cpu_s = max(0.0, time.thread_time() - thread_started)

    def start(self) -> Profiler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def begin(self, started_ns: int, *, phase: str | None = None) -> None:
        self._started_ns = started_ns
        self.phase = phase
        self._begin.set()

    def record_rss(self, value: float | None, timestamp_ns: int) -> None:
        offset = self._offset(timestamp_ns)
        if value is not None and offset is not None:
            self.rss_samples.append(value)
            self.rss_offsets_ns.append(offset)

    def finish(self, ended_ns: int) -> None:
        self._ended_ns = ended_ns

    def stop(self) -> None:
        if self._ended_ns is None:
            self._ended_ns = time.perf_counter_ns()
        self._stop.set()
        self._begin.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.gpu_interval_s + 5)

    @property
    def window_ns(self) -> int | None:
        if self._started_ns is None or self._ended_ns is None:
            return None
        return max(0, self._ended_ns - self._started_ns)

    def _series(
        self,
        samples: list[float],
        offsets_ns: list[int],
        *,
        idle_threshold: float,
        source: str,
        scope: str,
    ) -> Series | None:
        if not samples:
            return None
        if self.window_ns is None or len(offsets_ns) != len(samples):
            return Series.from_samples(samples, idle_threshold=idle_threshold)
        paired = sorted(zip(offsets_ns, samples, strict=True))
        return Series.from_samples(
            [value for _, value in paired],
            idle_threshold=idle_threshold,
            offsets_ns=[offset for offset, _ in paired],
            window_ns=self.window_ns,
            source=source,
            scope=scope,
            phase=self.phase,
        )

    @property
    def mean(self) -> float | None:
        return sum(self.samples) / len(self.samples) if self.samples else None

    @property
    def peak(self) -> float | None:
        return max(self.samples) if self.samples else None

    @property
    def cpu(self) -> Series | None:
        return self._series(
            self.cpu_samples,
            self.cpu_offsets_ns,
            idle_threshold=0.05,
            source="process-cpu-time",
            scope="worker-process",
        )

    @property
    def rss(self) -> Series | None:
        return self._series(
            self.rss_samples,
            self.rss_offsets_ns,
            idle_threshold=0.0,
            source="psutil-rss",
            scope="worker-process",
        )

    @property
    def uss(self) -> Series | None:
        return self._series(
            self.uss_samples,
            self.uss_offsets_ns,
            idle_threshold=0.0,
            source="psutil-uss",
            scope="worker-process",
        )

    @property
    def gpu_util(self) -> Series | None:
        return self._series(
            self.samples,
            self.gpu_offsets_ns,
            idle_threshold=1.0,
            source="ioreg-device-utilization",
            scope="whole-gpu",
        )

    @property
    def gpu_mem(self) -> Series | None:
        return self._series(
            self.gpu_mem_samples,
            self.gpu_mem_offsets_ns,
            idle_threshold=0.0,
            source="torch-driver-allocation",
            scope="worker-process",
        )


# Kept as a public compatibility name for callers and old tests. New code can
# use ``Profiler`` when it needs all resource series.
GpuSampler = Profiler


def synchronize_device(device: str | None) -> None:
    """Wait for supported accelerator work without importing Torch eagerly."""
    if device not in {"mps", "cuda"}:
        return
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if device == "mps":
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def measure(
    sample_gpu: bool = False,
    *,
    profile: bool = True,
    device: str | None = None,
    phase: str | None = None,
    sample_uss: bool = False,
):
    """Measure a block and yield a one-element list containing its usage."""
    box: list[ResourceUsage] = []
    sampler = GpuSampler(sample_gpu=sample_gpu, sample_uss=sample_uss).start() if profile else None
    synchronize_device(device)
    cpu_started = _cpu_seconds(include_children=sample_gpu)
    started_ns = time.perf_counter_ns()
    rss_start = rss_now_mb()
    rss_start_ns = time.perf_counter_ns()
    if sampler is not None:
        sampler.begin(started_ns, phase=phase)
        sampler.record_rss(rss_start, rss_start_ns)
    try:
        yield box
    finally:
        synchronize_device(device)
        rss_end = rss_now_mb()
        rss_end_ns = time.perf_counter_ns()
        ended_ns = time.perf_counter_ns()
        if sampler is not None:
            sampler.record_rss(rss_end, rss_end_ns)
            sampler.finish(ended_ns)
            sampler.stop()
        # Take the CPU snapshot after the sampler joins so an in-flight ioreg
        # child is included and can be subtracted deterministically.
        cpu_total = max(0.0, _cpu_seconds(include_children=sample_gpu) - cpu_started)
        sampler_cost = getattr(sampler, "cpu_cost_s", 0.0) if sampler else 0.0
        profiler_cost = getattr(sampler, "thread_cpu_s", 0.0) if sampler else 0.0
        gpu_series = getattr(sampler, "gpu_util", None) if sampler else None
        rss_series = getattr(sampler, "rss", None) if sampler else None
        uss_series = getattr(sampler, "uss", None) if sampler else None
        current_rss = [value for value in (rss_start, rss_end) if value is not None]
        if rss_series:
            current_rss.extend(rss_series.samples)
        process_peak = peak_rss_mb()
        box.append(
            ResourceUsage(
                wall_s=max(0.0, (ended_ns - started_ns) / 1_000_000_000),
                cpu_s=max(0.0, cpu_total - sampler_cost - profiler_cost),
                peak_rss_mb=process_peak,
                started_ns=started_ns,
                ended_ns=ended_ns,
                rss_peak_mb=max(current_rss) if current_rss else None,
                process_peak_rss_mb=process_peak,
                phase=phase,
                gpu_mb=gpu_allocated_mb(),
                gpu_util_mean=(
                    gpu_series.mean if gpu_series else (sampler.mean if sampler else None)
                ),
                gpu_util_peak=(
                    gpu_series.max if gpu_series else (sampler.peak if sampler else None)
                ),
                sampler_cpu_s=sampler_cost,
                profiler_cpu_s=profiler_cost,
                cpu=getattr(sampler, "cpu", None) if sampler else None,
                rss=rss_series,
                uss=uss_series,
                gpu_util=gpu_series,
                gpu_mem=getattr(sampler, "gpu_mem", None) if sampler else None,
            )
        )


def saturation(usage: ResourceUsage, compute_cores: int) -> str:
    """Classify the busiest measured resource using robust GPU statistics."""
    gpu_series = usage.gpu_util
    gpu = gpu_series.p50 if gpu_series else usage.gpu_util_mean
    gpu_idle = gpu_series.idle_pct if gpu_series else 0.0
    cores = usage.cpu_utilization or 0.0
    if gpu is not None and gpu >= 60.0 and gpu_idle < 50.0:
        return "GPU-bound"
    if compute_cores > 0 and cores >= 0.7 * compute_cores:
        return "CPU-bound"
    if gpu is not None and gpu >= 25.0 and gpu_idle < 75.0:
        return "GPU-light"
    return "underutilised"


def describe_host() -> dict[str, Any]:
    """Static facts about the machine, recorded so runs stay comparable."""
    from stt.hardware import describe

    return describe()
