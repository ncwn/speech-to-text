"""Pure, versioned evidence derivers.

Derivers receive already-loaded trusted records and return typed tables.  They
never format Markdown and they cannot silently skip identity or settings checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stt.measurement import MeasurementError
from stt.results import TranscriptionResult

Numeric = int | float
DERIVER_SCHEMA_VERSION = 1


def _percentile(values: Sequence[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _finite(value: object, name: str) -> Numeric:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{name} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise MeasurementError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class DerivedColumn:
    name: str
    label: str
    digits: int = 4

    def validate(self) -> None:
        if not self.name or not self.label or any(char.isspace() for char in self.name):
            raise MeasurementError("derived column names must be non-empty identifiers")
        if not isinstance(self.digits, int) or not 0 <= self.digits <= 9:
            raise MeasurementError("derived column digits must be between 0 and 9")


@dataclass(frozen=True)
class DerivedTable:
    """Typed table with stable row keys and numeric cells."""

    deriver: str
    columns: tuple[DerivedColumn, ...]
    rows: tuple[dict[str, Any], ...]
    row_key: str
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.deriver or ":" not in self.deriver:
            raise MeasurementError("derived table needs a versioned deriver name")
        if not self.columns:
            raise MeasurementError("derived table needs columns")
        columns = {column.name for column in self.columns}
        if len(columns) != len(self.columns):
            raise MeasurementError("derived table columns must be unique")
        for column in self.columns:
            column.validate()
        if self.row_key not in columns:
            raise MeasurementError("derived table row key must be a declared column")
        seen: set[str] = set()
        for index, row in enumerate(self.rows):
            if set(row) != columns:
                raise MeasurementError(f"derived row {index} does not match declared columns")
            key = row[self.row_key]
            if not isinstance(key, str) or not key or key in seen:
                raise MeasurementError(f"derived row {index} has an unstable or duplicate key")
            seen.add(key)
            for name, value in row.items():
                if name != self.row_key:
                    _finite(value, f"derived row {index}.{name}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": DERIVER_SCHEMA_VERSION,
            "deriver": self.deriver,
            "columns": [column.__dict__ for column in self.columns],
            "rows": [dict(row) for row in self.rows],
            "row_key": self.row_key,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class Requirements:
    """Central identity requirements applied before a deriver runs."""

    same_waveform: bool = True
    reference_sha256: str | None = None
    expected_reference_count: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    aligned_segments: bool = False

    def validate(self) -> None:
        if self.reference_sha256 is not None and (
            len(self.reference_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.reference_sha256)
        ):
            raise MeasurementError("reference_sha256 must be a lowercase SHA-256")
        if self.expected_reference_count is not None and self.expected_reference_count < 1:
            raise MeasurementError("expected_reference_count must be positive")
        if not isinstance(self.settings, dict):
            raise MeasurementError("deriver settings must be an object")


@dataclass(frozen=True)
class DeriveContext:
    runs: tuple[tuple[str, tuple[TranscriptionResult, ...]], ...]
    references: dict[str, str]
    requirements: Requirements = Requirements()
    normalization: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.requirements.validate()
        if not self.runs:
            raise MeasurementError("deriver context needs at least one run")
        if not self.references:
            raise MeasurementError("deriver context needs references")
        for label, results in self.runs:
            if not label or not results:
                raise MeasurementError("deriver runs need labels and records")
        if (
            self.requirements.expected_reference_count is not None
            and len(self.references) != self.requirements.expected_reference_count
        ):
            raise MeasurementError("reference count differs from the declared requirement")


Deriver = Callable[[DeriveContext], DerivedTable]
_DERIVERS: dict[str, Deriver] = {}


def register(name: str) -> Callable[[Deriver], Deriver]:
    if not name or ":" not in name:
        raise ValueError("deriver names must include a version suffix")

    def decorator(function: Deriver) -> Deriver:
        if name in _DERIVERS:
            raise ValueError(f"deriver is already registered: {name}")
        _DERIVERS[name] = function
        return function

    return decorator


def get(name: str) -> Deriver:
    try:
        return _DERIVERS[name]
    except KeyError as exc:
        raise MeasurementError(f"unknown evidence deriver: {name}") from exc


def reference_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference_identity(
    references: Mapping[str, str],
    *,
    path: Path | None,
    requirements: Requirements,
) -> None:
    requirements.validate()
    if (
        requirements.expected_reference_count is not None
        and len(references) != requirements.expected_reference_count
    ):
        raise MeasurementError("reference count differs from the declared requirement")
    if requirements.reference_sha256 is not None:
        if path is None or reference_sha256(path) != requirements.reference_sha256:
            raise MeasurementError("reference corpus checksum differs from the declared identity")


def validate_same_waveform(runs: Sequence[Sequence[TranscriptionResult]]) -> None:
    if not runs:
        raise MeasurementError("same-waveform validation needs runs")
    expected: dict[str, str] | None = None
    for results in runs:
        found = {result.reference_id: result.audio_id for result in results}
        if any(not reference or not audio for reference, audio in found.items()):
            raise MeasurementError("same-waveform validation needs reference/audio identities")
        if len(found) != len(results):
            raise MeasurementError("a run contains duplicate reference identities")
        if expected is None:
            expected = found
        elif found != expected:
            raise MeasurementError("runs do not contain the same reference-to-waveform map")


def validate_settings(
    runs: Sequence[Sequence[TranscriptionResult]], expected: Mapping[str, Any]
) -> None:
    for results in runs:
        for result in results:
            provenance = result.model_provenance
            resolved = provenance.get("resolved_settings") if isinstance(provenance, dict) else None
            if not isinstance(resolved, dict):
                raise MeasurementError("deriver run is missing resolved settings")
            for name, value in expected.items():
                if value is None or resolved.get(name) != value:
                    raise MeasurementError(
                        f"declared setting {name!r} differs from the resolved run"
                    )


def validate_context(context: DeriveContext, *, reference_path: Path | None = None) -> None:
    context.validate()
    requirements = context.requirements
    validate_reference_identity(context.references, path=reference_path, requirements=requirements)
    runs = [results for _, results in context.runs]
    if requirements.same_waveform:
        validate_same_waveform(runs)
    if requirements.settings:
        validate_settings(runs, requirements.settings)
    if requirements.aligned_segments:
        validate_aligned_segments(runs)


def validate_aligned_segments(runs: Sequence[Sequence[TranscriptionResult]]) -> None:
    """Require ordered, bounded aligned speech spans; silence gaps are valid."""
    for results in runs:
        for result in results:
            duration = result.audio_duration_s
            if duration is None or not math.isfinite(duration) or duration <= 0:
                raise MeasurementError("aligned segment validation needs audio duration")
            segments = result.segments
            if not segments:
                raise MeasurementError("aligned segment validation needs segments")
            previous_end = 0.0
            for segment in segments:
                if segment.source != "aligned":
                    raise MeasurementError("confidence derivers require source='aligned' segments")
                if (
                    not math.isfinite(segment.start)
                    or not math.isfinite(segment.end)
                    or segment.start < 0
                    or segment.end <= segment.start
                    or segment.end > duration
                    or segment.start < previous_end
                ):
                    raise MeasurementError("aligned segments must be ordered and bounded")
                if (
                    segment.confidence is None
                    or not math.isfinite(segment.confidence)
                    or not 0 <= segment.confidence <= 1
                ):
                    raise MeasurementError("aligned segment confidence must be in [0, 1]")
                previous_end = segment.end


def derive(
    context: DeriveContext,
    name: str,
    *,
    reference_path: Path | None = None,
) -> DerivedTable:
    validate_context(context, reference_path=reference_path)
    table = get(name)(context)
    table.validate()
    return table


@register("transcripts:v1")
def transcript_table(context: DeriveContext) -> DerivedTable:
    """A minimal structured transcript table useful for smoke verification."""
    rows = tuple(
        {
            "run": label,
            "n_records": len(results),
        }
        for label, results in context.runs
    )
    return DerivedTable(
        deriver="transcripts:v1",
        columns=(DerivedColumn("run", "Run"), DerivedColumn("n_records", "Records", 0)),
        rows=rows,
        row_key="run",
    )


@register("tails:v1")
def tails_table(context: DeriveContext) -> DerivedTable:
    """Derive corpus and per-clip tail CER under declared normalization."""
    from stt.burmese import NormalizeOptions
    from stt.evaluate import score_results

    try:
        options = NormalizeOptions(**context.normalization)
    except TypeError as exc:
        raise MeasurementError(f"invalid tail normalization: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for label, results in context.runs:
        score = score_results(
            list(results), context.references, options=options, check_encoding=True
        )
        if score.n_failed or score.partial_cer is None or len(score.scored) != len(results):
            raise MeasurementError(f"tail scoring is incomplete for {label}")
        clip_cers = sorted(float(item.cer) for item in score.scored)
        over = sum(value > 0.3 for value in clip_cers)
        rows.append(
            {
                "run": label,
                "corpus_cer": score.partial_cer,
                "p90_clip_cer": _percentile(clip_cers, 0.9),
                "over_0_3_pct": over / len(clip_cers) * 100.0,
                "over_0_3_n": over,
                "n": len(clip_cers),
            }
        )
    return DerivedTable(
        deriver="tails:v1",
        columns=(
            DerivedColumn("run", "Run"),
            DerivedColumn("corpus_cer", "Corpus CER", 4),
            DerivedColumn("p90_clip_cer", "Clip CER p90", 4),
            DerivedColumn("over_0_3_pct", "CER > 0.3 %", 1),
            DerivedColumn("over_0_3_n", "CER > 0.3 N", 0),
            DerivedColumn("n", "N", 0),
        ),
        rows=tuple(rows),
        row_key="run",
    )


@register("confidence:v1")
def confidence_table(context: DeriveContext) -> DerivedTable:
    values: list[dict[str, Any]] = []
    for label, results in context.runs:
        confidences = sorted(
            float(segment.confidence)
            for result in results
            for segment in result.segments or ()
            if segment.confidence is not None
        )
        if not confidences:
            raise MeasurementError("confidence deriver found no confidence values")

        values.append(
            {
                "run": label,
                "p25": _percentile(confidences, 0.25),
                "p50": _percentile(confidences, 0.50),
                "p75": _percentile(confidences, 0.75),
                "n_segments": len(confidences),
            }
        )
    return DerivedTable(
        deriver="confidence:v1",
        columns=(
            DerivedColumn("run", "Run"),
            DerivedColumn("p25", "P25"),
            DerivedColumn("p50", "P50"),
            DerivedColumn("p75", "P75"),
            DerivedColumn("n_segments", "Segments", 0),
        ),
        rows=tuple(values),
        row_key="run",
    )


@register("vote:v1")
def vote_table(context: DeriveContext) -> DerivedTable:
    from stt.vote import prepare_vote_groups

    runs = {label: list(results) for label, results in context.runs}
    labels = tuple(runs)
    groups = prepare_vote_groups(runs, labels[0], allow_partial=False)
    return DerivedTable(
        deriver="vote:v1",
        columns=(
            DerivedColumn("audio_id", "Audio"),
            DerivedColumn("n_voters", "Voters", 0),
        ),
        rows=tuple(
            {
                "audio_id": group.audio_id,
                "n_voters": len(group.usable_results),
            }
            for group in groups
        ),
        row_key="audio_id",
    )


@register("route:v1")
def route_table(context: DeriveContext) -> DerivedTable:
    from stt.cascade import prepare_route_pairs

    if len(context.runs) != 2:
        raise MeasurementError("route deriver needs exactly base and strong runs")
    pairs = prepare_route_pairs(
        list(context.runs[0][1]),
        list(context.runs[1][1]),
        allow_partial=False,
    )
    return DerivedTable(
        deriver="route:v1",
        columns=(DerivedColumn("audio_id", "Audio"), DerivedColumn("duration_s", "Duration")),
        rows=tuple(
            {"audio_id": pair.audio_id, "duration_s": float(pair.base.audio_duration_s or 0.0)}
            for pair in pairs.pairs
        ),
        row_key="audio_id",
    )


def baseline_table(path: Path) -> DerivedTable:
    """Derive trusted common-wall timing from a verified baseline-v2 artifact."""
    from stt.bench import verify_baseline

    issues = verify_baseline(path)
    if issues:
        raise MeasurementError("baseline-v2 verification failed: " + "; ".join(issues))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read baseline-v2: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(value.get("runs", [])):
        interval = run.get("rtf_ci95")
        if (
            not run.get("baseline_eligible")
            or run.get("error")
            or not isinstance(interval, list)
            or len(interval) != 2
        ):
            raise MeasurementError(f"baseline-v2 run {index} is not publishable")
        rows.append(
            {
                "subject": str(run["subject"]),
                "rtf": _finite(run.get("rtf"), f"baseline run {index}.rtf"),
                "ci_low": _finite(interval[0], f"baseline run {index}.ci_low"),
                "ci_high": _finite(interval[1], f"baseline run {index}.ci_high"),
                "peak_rss_mb": _finite(run.get("peak_rss_mb"), f"baseline run {index}.peak_rss_mb"),
                "sessions": _finite(run.get("workers"), f"baseline run {index}.workers"),
                "repeats": _finite(run.get("repeats"), f"baseline run {index}.repeats"),
            }
        )
    table = DerivedTable(
        deriver="baseline:v1",
        columns=(
            DerivedColumn("subject", "Subject"),
            DerivedColumn("rtf", "RTF", 4),
            DerivedColumn("ci_low", "CI low", 4),
            DerivedColumn("ci_high", "CI high", 4),
            DerivedColumn("peak_rss_mb", "Peak RSS MB", 0),
            DerivedColumn("sessions", "Sessions", 0),
            DerivedColumn("repeats", "Repeats", 0),
        ),
        rows=tuple(rows),
        row_key="subject",
        metadata={"baseline_id": value.get("baseline_id")},
    )
    table.validate()
    return table


def experiment_table(path: Path) -> DerivedTable:
    """Derive descriptive condition timing from a verified experiment-v1 archive."""
    from stt.experiment_archive import verify_experiment

    issues = verify_experiment(path)
    if issues:
        raise MeasurementError("experiment-v1 verification failed: " + "; ".join(issues))
    descriptor = path / "experiment.json" if path.is_dir() else path
    try:
        value = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read experiment-v1: {exc}") from exc
    summaries = list(value.get("condition_summaries", []))
    include_gpu = bool(summaries) and all(
        summary.get(name) is not None
        for summary in summaries
        for name in ("gpu_util_mean", "gpu_util_p50", "gpu_idle_pct")
    )
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries):
        if not summary.get("baseline_eligible") or summary.get("error"):
            raise MeasurementError(f"experiment-v1 condition {index} is not publishable")
        row = {
            "condition": str(summary["condition_id"]),
            "rtf": _finite(summary.get("rtf"), f"experiment condition {index}.rtf"),
            "peak_rss_mb": _finite(
                summary.get("peak_rss_mb"), f"experiment condition {index}.peak_rss_mb"
            ),
            "sessions": _finite(summary.get("workers"), f"experiment condition {index}.workers"),
            "repeats": _finite(summary.get("repeats"), f"experiment condition {index}.repeats"),
        }
        if include_gpu:
            row.update(
                {
                    "gpu_mean": _finite(
                        summary.get("gpu_util_mean"), f"experiment condition {index}.gpu_mean"
                    ),
                    "gpu_p50": _finite(
                        summary.get("gpu_util_p50"), f"experiment condition {index}.gpu_p50"
                    ),
                    "gpu_idle": _finite(
                        summary.get("gpu_idle_pct"), f"experiment condition {index}.gpu_idle"
                    ),
                }
            )
        rows.append(row)
    columns = [
        DerivedColumn("condition", "Condition"),
        DerivedColumn("rtf", "RTF", 4),
    ]
    if include_gpu:
        columns.extend(
            (
                DerivedColumn("gpu_mean", "GPU mean %", 1),
                DerivedColumn("gpu_p50", "GPU p50 %", 1),
                DerivedColumn("gpu_idle", "GPU idle %", 1),
            )
        )
    columns.extend(
        (
            DerivedColumn("peak_rss_mb", "Peak RSS MB", 0),
            DerivedColumn("sessions", "Sessions", 0),
            DerivedColumn("repeats", "Repeats", 0),
        )
    )
    table = DerivedTable(
        deriver="experiment:v1",
        columns=tuple(columns),
        rows=tuple(rows),
        row_key="condition",
        metadata={"experiment_id": value.get("experiment_id")},
    )
    table.validate()
    return table


def observer_table(path: Path) -> DerivedTable:
    """Derive observer accounting from a verified calibration experiment."""
    from stt.experiment_archive import verify_experiment

    issues = verify_experiment(path)
    if issues:
        raise MeasurementError("observer experiment verification failed: " + "; ".join(issues))
    descriptor = path / "experiment.json" if path.is_dir() else path
    value = json.loads(descriptor.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(value.get("condition_summaries", [])):
        observer = summary.get("observer")
        if (
            not summary.get("baseline_eligible")
            or summary.get("error")
            or not isinstance(observer, dict)
        ):
            raise MeasurementError(f"observer condition {index} is not publishable")
        rows.append(
            {
                "condition": str(summary["condition_id"]),
                "rtf": _finite(summary.get("rtf"), f"observer condition {index}.rtf"),
                "observer_cpu_s": _finite(
                    observer.get("observer_cpu_s"), f"observer condition {index}.cpu"
                ),
                "repeat_wall_s": _finite(
                    observer.get("repeat_wall_s"), f"observer condition {index}.wall"
                ),
                "uss_samples": _finite(
                    observer.get("uss_sample_count"), f"observer condition {index}.uss"
                ),
            }
        )
    table = DerivedTable(
        deriver="observer:v1",
        columns=(
            DerivedColumn("condition", "Condition"),
            DerivedColumn("rtf", "RTF", 4),
            DerivedColumn("observer_cpu_s", "Observer CPU s", 4),
            DerivedColumn("repeat_wall_s", "Repeat wall s", 3),
            DerivedColumn("uss_samples", "USS samples", 0),
        ),
        rows=tuple(rows),
        row_key="condition",
        metadata={"experiment_id": value.get("experiment_id")},
    )
    table.validate()
    return table


def input_control_table(path: Path) -> DerivedTable:
    """Derive preparation and inference facts from a verified input control."""
    from stt.experiment_archive import verify_experiment

    issues = verify_experiment(path)
    if issues:
        raise MeasurementError("input-control verification failed: " + "; ".join(issues))
    descriptor = path / "experiment.json" if path.is_dir() else path
    value = json.loads(descriptor.read_text(encoding="utf-8"))
    input_sets = {item["input_set_id"]: item for item in value["spec"]["input_sets"]}
    conditions = {item["condition_id"]: item for item in value["spec"]["conditions"]}
    contrasts = value.get("contrasts", [])
    changed = len(contrasts[0].get("transcript_changes", [])) if len(contrasts) == 1 else 0
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(value.get("condition_summaries", [])):
        condition = conditions[summary["condition_id"]]
        input_set = input_sets[condition["input_set_id"]]
        interval = summary.get("rtf_ci95")
        if (
            not summary.get("baseline_eligible")
            or summary.get("error")
            or not isinstance(interval, list)
            or len(interval) != 2
        ):
            raise MeasurementError(f"input-control condition {index} is not publishable")
        rows.append(
            {
                "condition": str(summary["condition_id"]),
                "preparation_wall_s": _finite(
                    input_set.get("preparation_wall_s"), f"input condition {index}.preparation"
                ),
                "rtf": _finite(summary.get("rtf"), f"input condition {index}.rtf"),
                "ci_low": _finite(interval[0], f"input condition {index}.ci_low"),
                "ci_high": _finite(interval[1], f"input condition {index}.ci_high"),
                "text_changes": changed,
            }
        )
    table = DerivedTable(
        deriver="input-control:v1",
        columns=(
            DerivedColumn("condition", "Condition"),
            DerivedColumn("preparation_wall_s", "Preparation wall s", 4),
            DerivedColumn("rtf", "RTF", 4),
            DerivedColumn("ci_low", "CI low", 4),
            DerivedColumn("ci_high", "CI high", 4),
            DerivedColumn("text_changes", "Text changes", 0),
        ),
        rows=tuple(rows),
        row_key="condition",
        metadata={"experiment_id": value.get("experiment_id")},
    )
    table.validate()
    return table


__all__ = [
    "DERIVER_SCHEMA_VERSION",
    "DeriveContext",
    "DerivedColumn",
    "DerivedTable",
    "Requirements",
    "derive",
    "get",
    "reference_sha256",
    "register",
    "transcript_table",
    "tails_table",
    "validate_context",
    "validate_aligned_segments",
    "validate_reference_identity",
    "validate_same_waveform",
    "validate_settings",
    "baseline_table",
    "experiment_table",
    "observer_table",
    "input_control_table",
]
