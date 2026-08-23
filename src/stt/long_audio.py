"""Runner, coverage accounting, and tamper-evident long-audio reports."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stt.measurement import MeasurementError
from stt.results import Segment, TranscriptionResult
from stt.sentinel import BOUNDARIES_S, DURATION_S, SENTINEL_KIND, verify_annotation

LONG_AUDIO_SCHEMA_VERSION = 2
LONG_AUDIO_ARTIFACT_KIND = "long-audio-observation-v2"
_LEGACY_LONG_AUDIO_ARTIFACT_KIND = "long-audio-observation-v1"

_SPEC_KEYS = frozenset(
    {
        "runner_id",
        "backend",
        "model",
        "entrypoint",
        "audio_sha256",
        "annotation_sha256",
        "boundaries_s",
    }
)
_OBSERVATION_KEYS = frozenset(
    {
        "runner_id",
        "transcript",
        "elapsed_s",
        "n_segments",
        "matched_spans",
        "missed_spans",
        "duplicate_spans",
        "ordered",
        "boundary_hits",
        "error",
    }
)
_SEGMENT_KEYS = frozenset({"text", "start", "end", "confidence", "speaker", "source"})
_RESULT_KEYS = frozenset(
    {
        "audio_path",
        "text",
        "backend",
        "model",
        "language",
        "elapsed_s",
        "audio_duration_s",
        "error",
        "metadata",
        "segments",
        "resources",
        "audio_id",
        "source_path",
        "source_sha256",
        "reference_id",
        "trusted",
        "trust_issues",
        "model_provenance",
        "rtf",
    }
)
_REPORT_KEYS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "spec",
        "result",
        "elapsed_s",
        "observation",
        "spec_sha256",
        "result_sha256",
        "observation_sha256",
    }
)


def _canonical_json(value: object) -> str:
    """Serialize a report component deterministically for its content digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_mapping_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise MeasurementError(f"{label} fields differ: missing={missing}, extra={extra}")


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
        if not all(
            isinstance(value, str) and value
            for value in (self.runner_id, self.backend, self.model, self.entrypoint)
        ):
            raise MeasurementError("long-audio runner identity is incomplete")
        for name, value in (
            ("audio_sha256", self.audio_sha256),
            ("annotation_sha256", self.annotation_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise MeasurementError(f"{name} must be a lowercase SHA-256")
        if not isinstance(self.boundaries_s, tuple) or not self.boundaries_s:
            raise MeasurementError("long-audio boundaries must be a tuple")
        if tuple(sorted(self.boundaries_s)) != self.boundaries_s:
            raise MeasurementError("long-audio boundaries must be increasing")
        if any(
            not _is_number(value)
            or not math.isfinite(float(value))
            or value <= 0
            or value >= DURATION_S
            for value in self.boundaries_s
        ):
            raise MeasurementError("long-audio boundaries must be inside the sentinel")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "runner_id": self.runner_id,
            "backend": self.backend,
            "model": self.model,
            "entrypoint": self.entrypoint,
            "audio_sha256": self.audio_sha256,
            "annotation_sha256": self.annotation_sha256,
            "boundaries_s": list(self.boundaries_s),
        }


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
        if not isinstance(self.runner_id, str) or not self.runner_id:
            raise MeasurementError("long-audio observation runner identity is incomplete")
        if not _is_number(self.elapsed_s) or not math.isfinite(float(self.elapsed_s)):
            raise MeasurementError("long-audio observation timing is invalid")
        if self.elapsed_s < 0:
            raise MeasurementError("long-audio observation timing is invalid")
        if not isinstance(self.transcript, str) or not isinstance(self.ordered, bool):
            raise MeasurementError("long-audio observation text or ordering is invalid")
        for name in ("n_segments", "matched_spans", "missed_spans", "duplicate_spans"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:  # bool is not a valid count.
                raise MeasurementError("long-audio counts must be non-negative integers")
        if not isinstance(self.boundary_hits, tuple):
            raise MeasurementError("long-audio boundary hits must be a tuple")
        if tuple(sorted(self.boundary_hits)) != self.boundary_hits or len(
            set(self.boundary_hits)
        ) != len(self.boundary_hits):
            raise MeasurementError("long-audio boundary hits must be sorted and unique")
        if any(
            not _is_number(value) or not math.isfinite(float(value)) for value in self.boundary_hits
        ):
            raise MeasurementError("long-audio boundary hits must be finite")
        if self.error is not None and not isinstance(self.error, str):
            raise MeasurementError("long-audio observation error must be text or null")

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


def _validate_expected_spans(expected_spans: Sequence[tuple[float, float]]) -> None:
    for span in expected_spans:
        if len(span) != 2:
            raise MeasurementError("long-audio expected spans must have two endpoints")
        start, end = span
        if (
            not _is_number(start)
            or not _is_number(end)
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or start < 0
            or end <= start
            or end > DURATION_S
        ):
            raise MeasurementError("long-audio expected span is invalid")


def observe_segments(
    runner_id: str,
    result: TranscriptionResult,
    expected_spans: Sequence[tuple[float, float]],
    *,
    elapsed_s: float,
    boundaries_s: Sequence[float] = BOUNDARIES_S,
) -> LongAudioObservation:
    """Account for expected spans without assuming full timeline tiling."""
    _validate_expected_spans(expected_spans)
    boundaries = tuple(boundaries_s)
    if not boundaries:
        raise MeasurementError("long-audio boundaries cannot be empty")
    if tuple(sorted(boundaries)) != boundaries or any(
        not _is_number(value)
        or not math.isfinite(float(value))
        or value <= 0
        or value >= DURATION_S
        for value in boundaries
    ):
        raise MeasurementError("long-audio boundaries are invalid")
    segments = list(result.segments or ())
    ordered = all(
        current.start >= previous.end
        for previous, current in zip(segments, segments[1:], strict=False)
    )
    matched = 0
    duplicate = 0
    for expected in expected_spans:
        candidates = [segment for segment in segments if _overlap(segment, expected) > 0]
        if candidates:
            matched += 1
            duplicate += max(0, len(candidates) - 1)
    boundary_hits = tuple(
        boundary
        for boundary in boundaries
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
        boundary_hits=boundary_hits,
        error=result.error,
    )


def _validate_raw_segment(value: Mapping[str, Any]) -> Segment:
    _validate_mapping_keys(value, _SEGMENT_KEYS, "long-audio segment")
    if not isinstance(value["text"], str):
        raise MeasurementError("long-audio segment text must be text")
    if not _is_number(value["start"]) or not _is_number(value["end"]):
        raise MeasurementError("long-audio segment timing must be numeric")
    start = float(value["start"])
    end = float(value["end"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise MeasurementError("long-audio segment timing is invalid")
    if end > DURATION_S:
        raise MeasurementError("long-audio segment exceeds the sentinel")
    confidence = value["confidence"]
    if confidence is not None and (
        not _is_number(confidence) or not math.isfinite(float(confidence))
    ):
        raise MeasurementError("long-audio segment confidence is invalid")
    if value["speaker"] is not None and not isinstance(value["speaker"], str):
        raise MeasurementError("long-audio segment speaker must be text or null")
    if not isinstance(value["source"], str):
        raise MeasurementError("long-audio segment source must be text")
    return Segment(**dict(value))


def _validate_raw_result(value: Mapping[str, Any]) -> None:
    _validate_mapping_keys(value, _RESULT_KEYS, "long-audio result")
    for name in ("audio_path", "text", "backend", "model"):
        if not isinstance(value[name], str):
            raise MeasurementError(f"long-audio result {name} must be text")
    for name in (
        "language",
        "error",
        "audio_id",
        "source_path",
        "source_sha256",
        "reference_id",
    ):
        if value[name] is not None and not isinstance(value[name], str):
            raise MeasurementError(f"long-audio result {name} must be text or null")
    if not isinstance(value["metadata"], dict):
        raise MeasurementError("long-audio result metadata must be an object")
    if value["resources"] is not None and not isinstance(value["resources"], dict):
        raise MeasurementError("long-audio result resources must be an object or null")
    if not isinstance(value["trusted"], bool):
        raise MeasurementError("long-audio result trusted must be boolean")
    if not isinstance(value["trust_issues"], list) or not all(
        isinstance(item, str) for item in value["trust_issues"]
    ):
        raise MeasurementError("long-audio result trust_issues must be a list of text")
    if value["model_provenance"] is not None and not isinstance(value["model_provenance"], dict):
        raise MeasurementError("long-audio result model_provenance must be an object or null")
    for name in ("elapsed_s", "audio_duration_s"):
        number = value[name]
        if number is not None and (
            not _is_number(number) or not math.isfinite(float(number)) or number < 0
        ):
            raise MeasurementError(f"long-audio result {name} is invalid")
    segments = value["segments"]
    if segments is not None:
        if not isinstance(segments, list):
            raise MeasurementError("long-audio result segments must be a list or null")
        for segment in segments:
            if not isinstance(segment, dict):
                raise MeasurementError("long-audio result segment must be an object")
            _validate_raw_segment(segment)
    expected_rtf = None
    if value["elapsed_s"] is not None and value["audio_duration_s"] not in (None, 0):
        expected_rtf = value["elapsed_s"] / value["audio_duration_s"]
    actual_rtf = value["rtf"]
    if actual_rtf is not None and (
        not _is_number(actual_rtf) or not math.isfinite(float(actual_rtf))
    ):
        raise MeasurementError("long-audio result rtf is invalid")
    if expected_rtf is None and actual_rtf is not None:
        raise MeasurementError("long-audio result rtf is not recomputable")
    if expected_rtf is not None and (
        actual_rtf is None or not math.isclose(float(actual_rtf), expected_rtf, rel_tol=1e-12)
    ):
        raise MeasurementError("long-audio result rtf differs from its elapsed and duration")


def _result_to_dict(result: TranscriptionResult) -> dict[str, Any]:
    value = result.to_dict()
    _validate_raw_result(value)
    return value


def _result_from_dict(value: Mapping[str, Any]) -> TranscriptionResult:
    _validate_raw_result(value)
    data = dict(value)
    data.pop("rtf")
    data.pop("resources")
    raw_segments = data["segments"]
    data["segments"] = (
        None
        if raw_segments is None
        else [_validate_raw_segment(segment) for segment in raw_segments]
    )
    return TranscriptionResult(**data)


@dataclass(frozen=True)
class LongAudioReport:
    """A v2 archive containing raw decoder output and its derived observation."""

    spec: LongAudioRunnerSpec
    result: TranscriptionResult
    elapsed_s: float
    observation: LongAudioObservation
    expected_spans: tuple[tuple[float, float], ...]

    def validate(self) -> None:
        self.spec.validate()
        if not _is_number(self.elapsed_s) or not math.isfinite(float(self.elapsed_s)):
            raise MeasurementError("long-audio report timing is invalid")
        if self.elapsed_s < 0:
            raise MeasurementError("long-audio report timing is invalid")
        if self.result.backend != self.spec.backend or self.result.model != self.spec.model:
            raise MeasurementError(
                "long-audio result identity differs from its runner specification"
            )
        _validate_raw_result(_result_to_dict(self.result))
        self.observation.validate()
        if self.observation.runner_id != self.spec.runner_id:
            raise MeasurementError("long-audio report runner identity differs")
        if self.observation.elapsed_s != self.elapsed_s:
            raise MeasurementError("long-audio report timing differs from its observation")
        recomputed = observe_segments(
            self.spec.runner_id,
            self.result,
            self.expected_spans,
            elapsed_s=self.elapsed_s,
            boundaries_s=self.spec.boundaries_s,
        )
        if recomputed != self.observation:
            raise MeasurementError("long-audio observation is not derived from its raw result")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        spec = self.spec.to_dict()
        result = _result_to_dict(self.result)
        observation = self.observation.to_dict()
        return {
            "artifact_kind": LONG_AUDIO_ARTIFACT_KIND,
            "schema_version": LONG_AUDIO_SCHEMA_VERSION,
            "spec": spec,
            "result": result,
            "elapsed_s": self.elapsed_s,
            "observation": observation,
            "spec_sha256": _digest(spec),
            "result_sha256": _digest(result),
            "observation_sha256": _digest(observation),
        }


def run_runner_report(
    spec: LongAudioRunnerSpec,
    transcribe: Callable[[], TranscriptionResult],
    expected_spans: Sequence[tuple[float, float]],
) -> LongAudioReport:
    """Run one entry point and retain raw output for a tamper-evident archive."""
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
    elapsed_s = time.perf_counter() - started
    observation = observe_segments(
        spec.runner_id,
        result,
        expected_spans,
        elapsed_s=elapsed_s,
        boundaries_s=spec.boundaries_s,
    )
    return LongAudioReport(
        spec=spec,
        result=result,
        elapsed_s=elapsed_s,
        observation=observation,
        expected_spans=tuple(expected_spans),
    )


def run_runner(
    spec: LongAudioRunnerSpec,
    transcribe: Callable[[], TranscriptionResult],
    expected_spans: Sequence[tuple[float, float]],
) -> LongAudioObservation:
    """Run one declared entry point and return its derived observation."""
    return run_runner_report(spec, transcribe, expected_spans).observation


def write_report(report: LongAudioReport, path: Path) -> None:
    """Write a validated v2 long-audio report as portable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_sentinel_spans(annotation_path: Path) -> tuple[tuple[float, float], ...]:
    """Read expected spans from a checked-in sentinel annotation."""
    value = json.loads(annotation_path.read_text(encoding="utf-8"))
    if value.get("artifact_kind") != SENTINEL_KIND:
        raise MeasurementError("unsupported long-audio sentinel annotation")
    spans = tuple((float(item["start_s"]), float(item["end_s"])) for item in value["spans"])
    _validate_expected_spans(spans)
    return spans


def _parse_spec(value: Any) -> LongAudioRunnerSpec:
    if not isinstance(value, dict):
        raise MeasurementError("long-audio report spec must be an object")
    _validate_mapping_keys(value, _SPEC_KEYS, "long-audio report spec")
    boundaries = value["boundaries_s"]
    if not isinstance(boundaries, list):
        raise MeasurementError("long-audio report boundaries must be a list")
    spec = LongAudioRunnerSpec(**{**value, "boundaries_s": tuple(boundaries)})
    spec.validate()
    return spec


def _parse_observation(value: Any) -> LongAudioObservation:
    if not isinstance(value, dict):
        raise MeasurementError("long-audio report observation must be an object")
    _validate_mapping_keys(value, _OBSERVATION_KEYS, "long-audio report observation")
    boundary_hits = value["boundary_hits"]
    if not isinstance(boundary_hits, list):
        raise MeasurementError("long-audio report boundary hits must be a list")
    observation = LongAudioObservation(**{**value, "boundary_hits": tuple(boundary_hits)})
    observation.validate()
    return observation


def _parse_report(value: Any, expected_spans: Sequence[tuple[float, float]]) -> LongAudioReport:
    if not isinstance(value, dict):
        raise MeasurementError("long-audio report must be an object")
    _validate_mapping_keys(value, _REPORT_KEYS, "long-audio report")
    if value["artifact_kind"] != LONG_AUDIO_ARTIFACT_KIND:
        raise MeasurementError("unsupported long-audio report artifact kind")
    if value["schema_version"] != LONG_AUDIO_SCHEMA_VERSION:
        raise MeasurementError("unsupported long-audio report schema")
    spec = _parse_spec(value["spec"])
    if not _is_number(value["elapsed_s"]):
        raise MeasurementError("long-audio report elapsed_s must be numeric")
    result_raw = value["result"]
    if not isinstance(result_raw, dict):
        raise MeasurementError("long-audio report result must be an object")
    result = _result_from_dict(result_raw)
    observation = _parse_observation(value["observation"])
    for name in ("spec_sha256", "result_sha256", "observation_sha256"):
        if not isinstance(value[name], str):
            raise MeasurementError("long-audio report content digests must be text")
    if value["spec_sha256"] != _digest(value["spec"]):
        raise MeasurementError("long-audio report spec digest differs")
    if value["result_sha256"] != _digest(result_raw):
        raise MeasurementError("long-audio report result digest differs")
    if value["observation_sha256"] != _digest(value["observation"]):
        raise MeasurementError("long-audio report observation digest differs")
    report = LongAudioReport(
        spec=spec,
        result=result,
        elapsed_s=float(value["elapsed_s"]),
        observation=observation,
        expected_spans=tuple(expected_spans),
    )
    report.validate()
    return report


def verify_observation(
    report_path: Path,
    *,
    audio_path: Path,
    annotation_path: Path,
) -> list[str]:
    """Verify a v2 report against immutable sentinel files and raw output."""
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot read long-audio observation: {exc}"]
    if isinstance(value, dict) and value.get("artifact_kind") == _LEGACY_LONG_AUDIO_ARTIFACT_KIND:
        return ["legacy long-audio observation-v1 reports are unsupported"]
    try:
        annotation_issues = verify_annotation(annotation_path)
        expected_spans = read_sentinel_spans(annotation_path)
        report = _parse_report(value, expected_spans)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        MeasurementError,
    ) as exc:
        return [f"cannot verify long-audio observation: {exc}"]
    audio_sha, annotation_sha = sentinel_identity(annotation_path, audio_path)
    issues: list[str] = list(annotation_issues)
    if report.spec.audio_sha256 != audio_sha:
        issues.append("long-audio observation audio identity differs")
    if report.spec.annotation_sha256 != annotation_sha:
        issues.append("long-audio observation annotation identity differs")
    if tuple(report.spec.boundaries_s) != BOUNDARIES_S:
        issues.append("long-audio observation boundaries differ from the sentinel")
    if report.spec.runner_id != report.observation.runner_id:
        issues.append("long-audio observation runner identity differs")
    if report.result.backend != report.spec.backend or report.result.model != report.spec.model:
        issues.append("long-audio observation result identity differs")
    if report.result.audio_duration_s is not None and not math.isclose(
        report.result.audio_duration_s, DURATION_S, rel_tol=0.0, abs_tol=1e-6
    ):
        issues.append("long-audio observation duration differs from the sentinel")
    return issues


__all__ = [
    "LONG_AUDIO_ARTIFACT_KIND",
    "LONG_AUDIO_SCHEMA_VERSION",
    "LongAudioObservation",
    "LongAudioReport",
    "LongAudioRunnerSpec",
    "observe_segments",
    "read_sentinel_spans",
    "run_runner",
    "run_runner_report",
    "sentinel_identity",
    "verify_observation",
    "write_report",
]
