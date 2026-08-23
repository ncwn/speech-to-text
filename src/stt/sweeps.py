"""Pure stopping rules for bounded falsification sweeps.

The measured rows are supplied by an experiment archive.  This module does not
load a backend, so the stopping decision can be recomputed during verification.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from stt.measurement import MeasurementError

SWEEP_KIND = "batch-sweep-v1"
SWEEP_SCHEMA_VERSION = 1


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise MeasurementError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _simple_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MeasurementError(f"{name} must be a non-empty trimmed string")
    if any(character.isspace() for character in value) or "/" in value or "\\" in value:
        raise MeasurementError(f"{name} contains an unsafe identifier")


@dataclass(frozen=True)
class BatchSweepSpec:
    """Pre-registered batch sizes and a reproducible plateau rule."""

    sweep_id: str
    subject: str
    input_set_id: str
    batch_sizes: tuple[int, ...]
    max_batch: int
    plateau_fraction: float = 0.05
    plateau_consecutive: int = 2
    sessions: int = 5
    warmups: int = 3
    repeats: int = 3
    timeout_s: float = 7_200.0

    def validate(self) -> None:
        _simple_id(self.sweep_id, "sweep_id")
        _simple_id(self.subject, "subject")
        _simple_id(self.input_set_id, "input_set_id")
        if not self.batch_sizes or self.batch_sizes[0] != 1:
            raise MeasurementError("batch sweep must start at batch size 1")
        if tuple(sorted(set(self.batch_sizes))) != self.batch_sizes:
            raise MeasurementError("batch sizes must be unique and increasing")
        if any(size < 1 or size & (size - 1) for size in self.batch_sizes):
            raise MeasurementError("batch sizes must be positive powers of two")
        if self.max_batch != self.batch_sizes[-1] or self.max_batch < 1:
            raise MeasurementError("max_batch must equal the final declared batch size")
        _finite(self.plateau_fraction, "plateau_fraction")
        if self.plateau_fraction < 0:
            raise MeasurementError("plateau_fraction must be non-negative")
        if (
            self.plateau_consecutive < 1
            or self.sessions < 1
            or self.warmups < 0
            or self.repeats < 1
        ):
            raise MeasurementError("invalid batch sweep protocol")
        _finite(self.timeout_s, "timeout_s", positive=True)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BatchSweepSpec:
        try:
            result = cls(
                sweep_id=str(value["sweep_id"]),
                subject=str(value["subject"]),
                input_set_id=str(value["input_set_id"]),
                batch_sizes=tuple(int(item) for item in value["batch_sizes"]),
                max_batch=int(value["max_batch"]),
                plateau_fraction=float(value.get("plateau_fraction", 0.05)),
                plateau_consecutive=int(value.get("plateau_consecutive", 2)),
                sessions=int(value.get("sessions", 5)),
                warmups=int(value.get("warmups", 3)),
                repeats=int(value.get("repeats", 3)),
                timeout_s=float(value.get("timeout_s", 7_200.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid batch sweep specification: {exc}") from exc
        result.validate()
        return result


@dataclass(frozen=True)
class BatchObservation:
    """One verified condition summary from a batch-size arm."""

    batch_size: int
    median_rtf: float
    interval_low: float
    interval_high: float
    transcript_hashes: dict[str, str]
    complete: bool = True
    error: str | None = None

    def validate(self) -> None:
        if self.batch_size < 1:
            raise MeasurementError("batch observation size must be positive")
        _finite(self.median_rtf, "median_rtf", positive=True)
        _finite(self.interval_low, "interval_low", positive=True)
        _finite(self.interval_high, "interval_high", positive=True)
        if self.interval_low > self.interval_high:
            raise MeasurementError("batch observation interval is reversed")
        if not isinstance(self.complete, bool):
            raise MeasurementError("batch observation complete must be boolean")
        if not self.complete and not str(self.error or "").strip():
            raise MeasurementError("incomplete batch observation needs an error")
        if not self.transcript_hashes:
            raise MeasurementError("batch observation needs transcript hashes")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.transcript_hashes.items()
        ):
            raise MeasurementError("batch transcript hashes must be strings")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BatchObservation:
        try:
            result = cls(
                batch_size=int(value["batch_size"]),
                median_rtf=float(value["median_rtf"]),
                interval_low=float(value["interval_low"]),
                interval_high=float(value["interval_high"]),
                transcript_hashes={str(k): str(v) for k, v in value["transcript_hashes"].items()},
                complete=bool(value.get("complete", True)),
                error=value.get("error"),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise MeasurementError(f"invalid batch observation: {exc}") from exc
        result.validate()
        return result


def summarize_batch_sweep(
    spec: BatchSweepSpec,
    observations: Sequence[BatchObservation],
) -> dict[str, Any]:
    """Compute the first declared plateau and return a stable result object."""
    spec.validate()
    by_size: dict[int, BatchObservation] = {}
    for observation in observations:
        observation.validate()
        if observation.batch_size in by_size:
            raise MeasurementError(f"duplicate batch observation: {observation.batch_size}")
        by_size[observation.batch_size] = observation
    if set(by_size) != set(spec.batch_sizes):
        raise MeasurementError("batch observations do not cover the declared sweep")
    ordered = [by_size[size] for size in spec.batch_sizes]
    complete = [observation for observation in ordered if observation.complete]
    if len(complete) != len(ordered):
        failed = next(observation for observation in ordered if not observation.complete)
        return {
            "artifact_kind": SWEEP_KIND,
            "schema_version": SWEEP_SCHEMA_VERSION,
            "spec": spec.to_dict(),
            "observations": [observation.to_dict() for observation in ordered],
            "rows": [observation.to_dict() for observation in ordered],
            "status": "failed",
            "stop_at": failed.batch_size,
            "stop_reason": failed.error or "batch condition failed",
        }
    reference_hashes = complete[0].transcript_hashes
    if any(observation.transcript_hashes != reference_hashes for observation in complete[1:]):
        raise MeasurementError("batch sweep transcript hashes changed across conditions")
    rows: list[dict[str, Any]] = [{"batch_size": ordered[0].batch_size, "plateau": False}]
    consecutive = 0
    stop_at: int | None = None
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        speedup = previous.median_rtf / current.median_rtf
        speedup_low = previous.interval_low / current.interval_high
        speedup_high = previous.interval_high / current.interval_low
        improvement = speedup - 1.0
        plateau = improvement <= spec.plateau_fraction and speedup_low <= (
            1.0 + spec.plateau_fraction
        )
        consecutive = consecutive + 1 if plateau else 0
        if stop_at is None and consecutive >= spec.plateau_consecutive:
            stop_at = current.batch_size
        rows.append(
            {
                "batch_size": current.batch_size,
                "speedup": speedup,
                "speedup_low": speedup_low,
                "speedup_high": speedup_high,
                "improvement": improvement,
                "plateau": plateau,
            }
        )
    return {
        "artifact_kind": SWEEP_KIND,
        "schema_version": SWEEP_SCHEMA_VERSION,
        "spec": spec.to_dict(),
        "observations": [observation.to_dict() for observation in ordered],
        "rows": rows,
        "transcript_hashes": dict(reference_hashes),
        "status": "stopped" if stop_at is not None else "max-batch",
        "stop_at": stop_at or spec.max_batch,
        "stop_reason": (
            f"{spec.plateau_consecutive} consecutive plateau steps"
            if stop_at is not None
            else "declared maximum batch reached"
        ),
    }


def verify_batch_sweep(value: Mapping[str, Any]) -> list[str]:
    """Recompute a stored batch sweep and return descriptive issues."""
    try:
        if value.get("artifact_kind") != SWEEP_KIND:
            return ["batch sweep artifact kind is unsupported"]
        if value.get("schema_version") != SWEEP_SCHEMA_VERSION:
            return ["batch sweep schema version is unsupported"]
        spec = BatchSweepSpec.from_dict(value["spec"])
        # The first result row is a display row, not an observation. Rebuild
        # observations from the explicit archive rows when present.
        raw = value.get("observations")
        if not isinstance(raw, list):
            return ["batch sweep observations are missing"]
        expected = summarize_batch_sweep(
            spec, tuple(BatchObservation.from_dict(item) for item in raw)
        )
        if json.dumps(expected, sort_keys=True, separators=(",", ":")) != json.dumps(
            dict(value), sort_keys=True, separators=(",", ":")
        ):
            return ["batch sweep derived result differs from observations"]
    except (MeasurementError, KeyError, TypeError, ValueError) as exc:
        return [f"cannot verify batch sweep: {exc}"]
    return []


__all__ = [
    "BatchObservation",
    "BatchSweepSpec",
    "SWEEP_KIND",
    "SWEEP_SCHEMA_VERSION",
    "summarize_batch_sweep",
    "verify_batch_sweep",
]
