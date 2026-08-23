"""Runner and coverage accounting for long-audio sentinel probes."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stt.measurement import MeasurementError
from stt.results import Segment, TranscriptionResult
from stt.sentinel import BOUNDARIES_S, DURATION_S, SENTINEL_KIND

LONG_AUDIO_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LongAudioRunnerSpec:
    """A declared entry point and immutable sentinel identity."""

    runner_id: str
    backend: str
    model: str
    entrypoint: str
    audio_sha256: str
    annotation_sha256: str
    boundaries_s: tuple[float, ...] = BOUNDARIES_S

    def validate(self) -> None:
        if not self.runner_id or not self.backend or not self.model or not self.entrypoint:
            raise MeasurementError("long-audio runner identity is incomplete")
        for name, value in (
            ("audio_sha256", self.audio_sha256),
            ("annotation_sha256", self.annotation_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise MeasurementError(f"{name} must be a lowercase SHA-256")
        if tuple(sorted(self.boundaries_s)) != self.boundaries_s:
            raise MeasurementError("long-audio boundaries must be increasing")
        if any(
            not math.isfinite(value) or value <= 0 or value >= DURATION_S
            for value in self.boundaries_s
        ):
            raise MeasurementError("long-audio boundaries must be inside the sentinel")


@dataclass(frozen=True)
class LongAudioObservation:
    """Coverage facts recomputed from one runner result."""

    runner_id: str
    transcript: str
    elapsed_s: float
    n_segments: int
    matched_spans: int
    missed_spans: int
    duplicate_spans: int
    ordered: bool
    boundary_hits: tuple[float, ...]
    error: str | None = None

    def validate(self) -> None:
        if not self.runner_id or self.elapsed_s < 0:
            raise MeasurementError("invalid long-audio observation identity or timing")
        if (
            min(
                self.n_segments,
                self.matched_spans,
                self.missed_spans,
                self.duplicate_spans,
            )
            < 0
        ):
            raise MeasurementError("long-audio counts cannot be negative")
        if self.matched_spans + self.missed_spans < 0:
            raise MeasurementError("invalid long-audio span accounting")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def sentinel_identity(annotation_path: Path, audio_path: Path) -> tuple[str, str]:
    """Return the content identities bound to a runner declaration."""
    annotation_digest = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    audio_digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    return audio_digest, annotation_digest


def _overlap(left: Segment, right: tuple[float, float]) -> float:
    return max(0.0, min(left.end, right[1]) - max(left.start, right[0]))


def observe_segments(
    runner_id: str,
    result: TranscriptionResult,
    expected_spans: Sequence[tuple[float, float]],
    *,
    elapsed_s: float,
) -> LongAudioObservation:
    """Account for expected spans without assuming full timeline tiling."""
    segments = list(result.segments or ())
    ordered = all(
        current.start >= previous.end
        for previous, current in zip(segments, segments[1:], strict=False)
    )
    matched = 0
    duplicate = 0
    used: set[int] = set()
    for expected in expected_spans:
        candidates = [
            index for index, segment in enumerate(segments) if _overlap(segment, expected) > 0
        ]
        if candidates:
            matched += 1
            used.add(candidates[0])
            duplicate += max(0, len(candidates) - 1)
    boundaries = tuple(
        boundary
        for boundary in BOUNDARIES_S
        if any(segment.start <= boundary <= segment.end for segment in segments)
    )
    return LongAudioObservation(
        runner_id=runner_id,
        transcript=result.text,
        elapsed_s=elapsed_s,
        n_segments=len(segments),
        matched_spans=matched,
        missed_spans=len(expected_spans) - matched,
        duplicate_spans=duplicate,
        ordered=ordered,
        boundary_hits=boundaries,
        error=result.error,
    )


def run_runner(
    spec: LongAudioRunnerSpec,
    transcribe: Callable[[], TranscriptionResult],
    expected_spans: Sequence[tuple[float, float]],
) -> LongAudioObservation:
    """Run one declared entry point and account for its observed coverage."""
    spec.validate()
    started = time.perf_counter()
    try:
        result = transcribe()
    except Exception as exc:  # noqa: BLE001 - preserve runner failure as data
        result = TranscriptionResult(
            audio_path="",
            text="",
            backend=spec.backend,
            model=spec.model,
            error=f"{type(exc).__name__}: {exc}",
        )
    return observe_segments(
        spec.runner_id,
        result,
        expected_spans,
        elapsed_s=time.perf_counter() - started,
    )


def read_sentinel_spans(annotation_path: Path) -> tuple[tuple[float, float], ...]:
    """Read expected spans from a checked-in sentinel annotation."""
    value = json.loads(annotation_path.read_text(encoding="utf-8"))
    if value.get("artifact_kind") != SENTINEL_KIND:
        raise MeasurementError("unsupported long-audio sentinel annotation")
    spans = tuple((float(item["start_s"]), float(item["end_s"])) for item in value["spans"])
    if any(end <= start for start, end in spans):
        raise MeasurementError("sentinel annotation contains an invalid span")
    return spans


__all__ = [
    "LONG_AUDIO_SCHEMA_VERSION",
    "LongAudioObservation",
    "LongAudioRunnerSpec",
    "observe_segments",
    "read_sentinel_spans",
    "run_runner",
    "sentinel_identity",
]
