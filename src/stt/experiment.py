"""Pure experiment specifications, summaries, and paired contrasts.

Worker execution remains in :mod:`stt.bench`. This module declares which
conditions run, which input population each condition owns, and exactly which
directed contrasts may become gates. It contains no model loading or file I/O,
so the same functions can recompute an archived experiment offline.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from stt import bench
from stt.measurement import AudioInput, MeasurementError, SubjectSpec, WorkerResponse

EXPERIMENT_KIND = "experiment-v1"
EXPERIMENT_SCHEMA_VERSION = 1
STATISTICS_ESTIMATOR = "paired-session-median-log-ratio"
STATISTICS_VERSION = 1
QUANTILE_METHOD = "linear"
TRUSTED_MIN_SESSIONS = 5
TRUSTED_MIN_WARMUPS = 3
TRUSTED_MIN_REPEATS = 3
JoinIdentity = Literal["audio_id", "source_sha256"]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _simple_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MeasurementError(f"{name} must be a non-empty trimmed string")
    if any(character.isspace() for character in value):
        raise MeasurementError(f"{name} cannot contain whitespace")
    if value == "." or any(character in value for character in ("/", "\\")) or ".." in value:
        raise MeasurementError(f"{name} contains an unsafe path component")


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise MeasurementError(f"{name} must be a boolean")
    return value


def _subject_label(subject: SubjectSpec) -> str:
    return f"{subject.backend}/{subject.model}" if subject.model else subject.backend


def _subject_from_dict(value: Mapping[str, Any]) -> SubjectSpec:
    try:
        subject = SubjectSpec(
            backend=str(value["backend"]),
            model=value.get("model"),
            language=value.get("language"),
            batch_size=int(value.get("batch_size", 1)),
            options=dict(value.get("options", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MeasurementError(f"invalid subject specification: {exc}") from exc
    subject.validate()
    return subject


def _audio_from_dict(value: Mapping[str, Any]) -> AudioInput:
    try:
        audio = AudioInput(
            source_path=str(value["source_path"]),
            prepared_path=str(value["prepared_path"]),
            reference_id=str(value["reference_id"]),
            source_sha256=str(value["source_sha256"]),
            audio_id=str(value["audio_id"]),
            duration_s=float(value["duration_s"]),
            sample_rate=int(value["sample_rate"]),
            channels=int(value["channels"]),
            frames=int(value["frames"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MeasurementError(f"invalid experiment audio input: {exc}") from exc
    audio.validate()
    return audio


@dataclass(frozen=True)
class InputSetSpec:
    """One exact input population available to experiment conditions."""

    input_set_id: str
    inputs: tuple[AudioInput, ...]

    def validate(self) -> None:
        _simple_id(self.input_set_id, "input_set_id")
        if not self.inputs:
            raise MeasurementError("an experiment input set cannot be empty")
        for item in self.inputs:
            if not isinstance(item, AudioInput):
                raise MeasurementError("input sets must contain AudioInput values")
            item.validate()
        audio_ids = [item.audio_id for item in self.inputs]
        reference_ids = [item.reference_id for item in self.inputs]
        if len(set(audio_ids)) != len(audio_ids):
            raise MeasurementError("experiment input set contains duplicate audio_id values")
        if len(set(reference_ids)) != len(reference_ids):
            raise MeasurementError("experiment input set contains duplicate reference_id values")

    def identity_map(self, join_on: JoinIdentity) -> dict[str, str]:
        """Map canonical audio ids to the identity used by a contrast."""
        self.validate()
        if join_on not in {"audio_id", "source_sha256"}:
            raise MeasurementError(f"unsupported contrast join identity: {join_on}")
        mapping = {
            item.audio_id: item.audio_id if join_on == "audio_id" else item.source_sha256
            for item in self.inputs
        }
        if len(set(mapping.values())) != len(mapping):
            raise MeasurementError(f"{self.input_set_id} is not unique by {join_on}")
        return mapping

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "input_set_id": self.input_set_id,
            "inputs": [asdict(item) for item in self.inputs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InputSetSpec:
        try:
            result = cls(
                input_set_id=str(value["input_set_id"]),
                inputs=tuple(_audio_from_dict(item) for item in value["inputs"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid input set specification: {exc}") from exc
        result.validate()
        return result


@dataclass(frozen=True)
class ConditionSpec:
    """One execution and observer configuration in an experiment."""

    condition_id: str
    subject: SubjectSpec
    input_set_id: str
    profile: bool = False
    sample_uss: bool = False
    runner_id: str = "adapter"

    def validate(self) -> None:
        _simple_id(self.condition_id, "condition_id")
        _simple_id(self.input_set_id, "input_set_id")
        _simple_id(self.runner_id, "runner_id")
        if not isinstance(self.subject, SubjectSpec):
            raise MeasurementError("condition subject must be a SubjectSpec")
        self.subject.validate()
        if self.subject.options.get("language") != self.subject.language:
            raise MeasurementError("condition subject options must bind its language")
        if self.subject.options.get("batch_size") != self.subject.batch_size:
            raise MeasurementError("condition subject options must bind its batch size")
        if self.sample_uss and not self.profile:
            raise MeasurementError("USS sampling requires profiling")
        if self.runner_id != "adapter":
            raise MeasurementError(f"unsupported experiment runner: {self.runner_id}")

    @property
    def request_key(self) -> str:
        self.validate()
        return self.subject.request_key

    @property
    def subject_label(self) -> str:
        self.validate()
        return _subject_label(self.subject)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "condition_id": self.condition_id,
            "subject": asdict(self.subject),
            "input_set_id": self.input_set_id,
            "profile": self.profile,
            "sample_uss": self.sample_uss,
            "runner_id": self.runner_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConditionSpec:
        try:
            result = cls(
                condition_id=str(value["condition_id"]),
                subject=_subject_from_dict(value["subject"]),
                input_set_id=str(value["input_set_id"]),
                profile=_boolean(value.get("profile", False), "condition profile"),
                sample_uss=_boolean(value.get("sample_uss", False), "condition sample_uss"),
                runner_id=str(value.get("runner_id", "adapter")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid condition specification: {exc}") from exc
        result.validate()
        return result


@dataclass(frozen=True)
class ContrastSpec:
    """One directed treatment/control comparison."""

    contrast_id: str
    control_condition_id: str
    treatment_condition_id: str
    threshold: float = 0.02
    gating: bool = True
    join_on: JoinIdentity = "audio_id"
    require_same_execution: bool = True

    def validate(self) -> None:
        _simple_id(self.contrast_id, "contrast_id")
        _simple_id(self.control_condition_id, "control_condition_id")
        _simple_id(self.treatment_condition_id, "treatment_condition_id")
        if self.control_condition_id == self.treatment_condition_id:
            raise MeasurementError("a contrast needs distinct control and treatment conditions")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise MeasurementError("contrast threshold must be finite and non-negative")
        if self.join_on not in {"audio_id", "source_sha256"}:
            raise MeasurementError(f"unsupported contrast join identity: {self.join_on}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContrastSpec:
        try:
            result = cls(
                contrast_id=str(value["contrast_id"]),
                control_condition_id=str(value["control_condition_id"]),
                treatment_condition_id=str(value["treatment_condition_id"]),
                threshold=float(value.get("threshold", 0.02)),
                gating=_boolean(value.get("gating", True), "contrast gating"),
                join_on=str(value.get("join_on", "audio_id")),  # type: ignore[arg-type]
                require_same_execution=_boolean(
                    value.get("require_same_execution", True),
                    "contrast require_same_execution",
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid contrast specification: {exc}") from exc
        result.validate()
        return result


@dataclass(frozen=True)
class StatisticsSpec:
    """Frozen statistical semantics carried by every experiment artifact."""

    estimator: str = STATISTICS_ESTIMATOR
    version: int = STATISTICS_VERSION
    draws: int = bench.BOOTSTRAP_DRAWS
    quantile_method: str = QUANTILE_METHOD
    min_sessions: int = TRUSTED_MIN_SESSIONS
    min_warmups: int = TRUSTED_MIN_WARMUPS
    min_repeats: int = TRUSTED_MIN_REPEATS

    def validate(self) -> None:
        if self.estimator != STATISTICS_ESTIMATOR or self.version != STATISTICS_VERSION:
            raise MeasurementError("unsupported experiment statistics contract")
        if self.draws != bench.BOOTSTRAP_DRAWS or self.quantile_method != QUANTILE_METHOD:
            raise MeasurementError("experiment bootstrap implementation differs")
        if (
            self.min_sessions < TRUSTED_MIN_SESSIONS
            or self.min_warmups < TRUSTED_MIN_WARMUPS
            or self.min_repeats < TRUSTED_MIN_REPEATS
        ):
            raise MeasurementError("experiment statistical minimums cannot weaken the trust gate")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StatisticsSpec:
        try:
            result = cls(
                estimator=str(value.get("estimator", STATISTICS_ESTIMATOR)),
                version=int(value.get("version", STATISTICS_VERSION)),
                draws=int(value.get("draws", bench.BOOTSTRAP_DRAWS)),
                quantile_method=str(value.get("quantile_method", QUANTILE_METHOD)),
                min_sessions=int(value.get("min_sessions", 5)),
                min_warmups=int(value.get("min_warmups", 3)),
                min_repeats=int(value.get("min_repeats", 3)),
            )
        except (TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid statistics specification: {exc}") from exc
        result.validate()
        return result


@dataclass(frozen=True)
class ExperimentSpec:
    """Complete, serializable declaration of an experiment."""

    experiment_id: str
    input_sets: tuple[InputSetSpec, ...]
    conditions: tuple[ConditionSpec, ...]
    contrasts: tuple[ContrastSpec, ...]
    sessions: int
    warmups: int
    repeats: int
    schedule_seed: int = 0
    statistics: StatisticsSpec = field(default_factory=StatisticsSpec)

    def validate(self) -> None:
        _simple_id(self.experiment_id, "experiment_id")
        self.statistics.validate()
        if not self.input_sets or not self.conditions:
            raise MeasurementError("experiment needs input sets and conditions")
        if self.sessions < 1 or self.warmups < 0 or self.repeats < 1:
            raise MeasurementError("invalid experiment session protocol")
        for values, label, identity in (
            (self.input_sets, "input sets", lambda item: item.input_set_id),
            (self.conditions, "conditions", lambda item: item.condition_id),
            (self.contrasts, "contrasts", lambda item: item.contrast_id),
        ):
            for item in values:
                item.validate()
            identifiers = [identity(item) for item in values]
            if len(set(identifiers)) != len(identifiers):
                raise MeasurementError(f"experiment contains duplicate {label}")
        input_ids = {item.input_set_id for item in self.input_sets}
        condition_ids = {item.condition_id for item in self.conditions}
        for condition in self.conditions:
            if condition.input_set_id not in input_ids:
                raise MeasurementError(
                    f"condition {condition.condition_id} names an unknown input set"
                )
        for contrast in self.contrasts:
            if {
                contrast.control_condition_id,
                contrast.treatment_condition_id,
            } - condition_ids:
                raise MeasurementError(
                    f"contrast {contrast.contrast_id} names an unknown condition"
                )
            if contrast.gating and (
                self.sessions < self.statistics.min_sessions
                or self.warmups < self.statistics.min_warmups
                or self.repeats < self.statistics.min_repeats
            ):
                raise MeasurementError(
                    f"gating contrast {contrast.contrast_id} does not satisfy "
                    "the trusted experiment protocol"
                )

    @property
    def condition_map(self) -> dict[str, ConditionSpec]:
        return {item.condition_id: item for item in self.conditions}

    @property
    def input_map(self) -> dict[str, InputSetSpec]:
        return {item.input_set_id: item for item in self.input_sets}

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "experiment_id": self.experiment_id,
            "input_sets": [item.to_dict() for item in self.input_sets],
            "conditions": [item.to_dict() for item in self.conditions],
            "contrasts": [item.to_dict() for item in self.contrasts],
            "sessions": self.sessions,
            "warmups": self.warmups,
            "repeats": self.repeats,
            "schedule_seed": self.schedule_seed,
            "statistics": self.statistics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExperimentSpec:
        try:
            result = cls(
                experiment_id=str(value["experiment_id"]),
                input_sets=tuple(InputSetSpec.from_dict(item) for item in value["input_sets"]),
                conditions=tuple(ConditionSpec.from_dict(item) for item in value["conditions"]),
                contrasts=tuple(ContrastSpec.from_dict(item) for item in value["contrasts"]),
                sessions=int(value["sessions"]),
                warmups=int(value["warmups"]),
                repeats=int(value["repeats"]),
                schedule_seed=int(value.get("schedule_seed", 0)),
                statistics=StatisticsSpec.from_dict(value.get("statistics", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid experiment specification: {exc}") from exc
        result.validate()
        return result


@dataclass(frozen=True)
class ScheduleEntry:
    condition_id: str
    request_key: str
    launch_position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_schedule(spec: ExperimentSpec) -> list[list[ScheduleEntry]]:
    """Return the deterministic session-major condition schedule."""
    spec.validate()
    simple = bench.counterbalanced_schedule(
        tuple(condition.condition_id for condition in spec.conditions),
        spec.sessions,
        spec.schedule_seed,
    )
    conditions = spec.condition_map
    return [
        [
            ScheduleEntry(condition_id, conditions[condition_id].request_key, position)
            for condition_id, position in session
        ]
        for session in simple
    ]


def _stable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: environment.get(key)
        for key in (
            "python",
            "platform",
            "machine",
            "packages",
            "git_commit",
            "git_dirty",
            "uv_lock_sha256",
        )
    }


def _identity(
    values: Sequence[Mapping[str, Any]],
    required_fields: Sequence[str],
) -> str | None:
    if not values:
        return None
    for value in values:
        for field_name in required_fields:
            field_value = value.get(field_name)
            if field_value is None or field_value == "" or field_value == {}:
                return None
    identities = {_canonical_json(dict(value)) for value in values}
    if len(identities) != 1:
        return None
    return hashlib.sha256(next(iter(identities)).encode("ascii")).hexdigest()


def observer_accounting(
    condition: ConditionSpec,
    responses: Sequence[WorkerResponse],
) -> dict[str, Any]:
    """Aggregate complete-repeat observer cost without rounded wall inference."""
    sampler_cpu_s = 0.0
    profiler_cpu_s = 0.0
    repeat_wall_s = 0.0
    load_wall_s = 0.0
    uss_samples = 0
    for response in responses:
        load = response.load_resources or {}
        load_wall_s += float(load.get("wall_s", 0.0) or 0.0)
        for repeat in response.repeats:
            if not repeat.complete:
                continue
            repeat_wall_s += repeat.phase.duration_s
            sampler_cpu_s += float(repeat.resources.get("sampler_cpu_s", 0.0) or 0.0)
            profiler_cpu_s += float(repeat.resources.get("profiler_cpu_s", 0.0) or 0.0)
            raw_uss = repeat.resources.get("uss")
            if isinstance(raw_uss, Mapping):
                uss_samples += int(raw_uss.get("n", 0) or 0)
    if not condition.sample_uss:
        uss_status = "disabled"
    elif uss_samples:
        uss_status = "collected"
    else:
        uss_status = "requested-unavailable"
    return {
        "profile": condition.profile,
        "sample_uss": condition.sample_uss,
        "uss_status": uss_status,
        "uss_sample_count": uss_samples,
        "sampler_cpu_s": sampler_cpu_s,
        "profiler_cpu_s": profiler_cpu_s,
        "observer_cpu_s": sampler_cpu_s + profiler_cpu_s,
        "repeat_wall_s": repeat_wall_s,
        "load_wall_s": load_wall_s,
    }


def _condition_summary(
    spec: ExperimentSpec,
    condition: ConditionSpec,
    responses: Sequence[WorkerResponse],
    schedule: Sequence[Sequence[ScheduleEntry]],
) -> dict[str, Any]:
    inputs = spec.input_map[condition.input_set_id]
    positions: list[int] = []
    coordinates: list[tuple[int, int, str]] = []
    for session_index, session in enumerate(schedule):
        entry = next(item for item in session if item.condition_id == condition.condition_id)
        positions.append(entry.launch_position)
        coordinates.append(
            (
                session_index,
                entry.launch_position,
                f"{spec.experiment_id}:session-{session_index}",
            )
        )
    summary = bench.summarize_workers(
        condition.subject_label,
        list(responses),
        list(inputs.inputs),
        expected_workers=spec.sessions,
        expected_warmups=spec.warmups,
        expected_repeats=spec.repeats,
        expected_launch_positions=positions,
        expected_session_coordinates=coordinates,
        experiment_id=spec.experiment_id,
    )
    content_ids = {
        str(response.model_provenance.get("content_sha256"))
        for response in responses
        if isinstance(response.model_provenance, dict)
        and response.model_provenance.get("content_sha256")
    }
    summary.update(
        {
            "condition_id": condition.condition_id,
            "request_key": condition.request_key,
            "input_set_id": condition.input_set_id,
            "runner_id": condition.runner_id,
            "profile": condition.profile,
            "sample_uss": condition.sample_uss,
            "content_sha256": next(iter(content_ids), None) if len(content_ids) == 1 else None,
            "host_identity": _identity(
                [response.host for response in responses],
                ("chip", "platform", "machine"),
            ),
            "environment_identity": _identity(
                [_stable_environment(response.environment) for response in responses],
                (
                    "python",
                    "platform",
                    "machine",
                    "packages",
                    "git_commit",
                    "git_dirty",
                    "uv_lock_sha256",
                ),
            ),
            "protocol": {
                "sessions": spec.sessions,
                "warmups": spec.warmups,
                "repeats": spec.repeats,
                "schedule_seed": spec.schedule_seed,
            },
            "observer": observer_accounting(condition, responses),
        }
    )
    return summary


def _repeat_map(summary: Mapping[str, Any]) -> dict[str, dict[int, float]] | None:
    result: dict[str, dict[int, float]] = {}
    blocks = summary.get("worker_blocks")
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if not isinstance(block, Mapping) or not block.get("session_id"):
            return None
        session_id = str(block["session_id"])
        if session_id in result:
            return None
        values = block.get("repeat_rtfs")
        if not isinstance(values, list) or not values:
            return None
        repeats: dict[int, float] = {}
        for index, value in enumerate(values):
            if value is None:
                return None
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                return None
            repeats[index] = numeric
        result[session_id] = repeats
    return result or None


def _joined_hashes(
    summary: Mapping[str, Any],
    input_set: InputSetSpec,
    join_on: JoinIdentity,
) -> dict[str, str] | None:
    raw = summary.get("hashes")
    if not isinstance(raw, Mapping):
        return None
    identity = input_set.identity_map(join_on)
    if set(map(str, raw)) != set(identity):
        return None
    return {identity[str(audio_id)]: str(digest) for audio_id, digest in raw.items()}


def _contrast_summary(
    spec: ExperimentSpec,
    contrast: ContrastSpec,
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    conditions = spec.condition_map
    control_condition = conditions[contrast.control_condition_id]
    treatment_condition = conditions[contrast.treatment_condition_id]
    control = summaries[contrast.control_condition_id]
    treatment = summaries[contrast.treatment_condition_id]
    reasons: list[str] = []

    for role, summary in (("control", control), ("treatment", treatment)):
        if summary.get("error"):
            reasons.append(f"{role} summary is incomplete: {summary['error']}")
        if not summary.get("baseline_eligible", False):
            reasons.append(f"{role} summary is not provenance/protocol eligible")
        if summary.get("provenance_issues"):
            reasons.append(f"{role} summary has provenance issues")
    for field_name in ("host_identity", "environment_identity", "protocol"):
        if not control.get(field_name) or control.get(field_name) != treatment.get(field_name):
            reasons.append(f"cross-arm {field_name} differs")
    if not control.get("content_sha256") or (
        control.get("content_sha256") != treatment.get("content_sha256")
    ):
        reasons.append("cross-arm model content identity differs")
    if contrast.require_same_execution and (
        not control.get("execution_sha256")
        or control.get("execution_sha256") != treatment.get("execution_sha256")
    ):
        reasons.append("cross-arm execution identity differs")

    control_hashes = _joined_hashes(
        control,
        spec.input_map[control_condition.input_set_id],
        contrast.join_on,
    )
    treatment_hashes = _joined_hashes(
        treatment,
        spec.input_map[treatment_condition.input_set_id],
        contrast.join_on,
    )
    transcript_changes: list[str] = []
    if control_hashes is None or treatment_hashes is None:
        reasons.append("cross-arm transcript identity coverage is incomplete")
    else:
        transcript_changes = [
            identity
            for identity in sorted(set(control_hashes) | set(treatment_hashes))
            if control_hashes.get(identity) != treatment_hashes.get(identity)
        ]
        if transcript_changes:
            reasons.append("cross-arm transcripts changed")

    control_repeats = _repeat_map(control)
    treatment_repeats = _repeat_map(treatment)
    seed = _stable_seed(spec.experiment_id, contrast.contrast_id, spec.schedule_seed)
    ratio = (
        bench.bootstrap_paired_ratio(control_repeats, treatment_repeats, seed)
        if control_repeats is not None and treatment_repeats is not None
        else None
    )
    if ratio is None or not all(math.isfinite(value) for value in ratio):
        reasons.append("paired session/repeat timing is incomplete or invalid")
    point = ratio[0] if ratio else None
    low = ratio[1] if ratio else None
    high = ratio[2] if ratio else None
    limit = 1.0 + contrast.threshold
    if point is not None and point > limit:
        reasons.append("point estimate exceeds the material regression limit")
    if high is not None and high > limit:
        reasons.append("95% interval permits a material regression")

    gating_eligible = contrast.gating and not reasons
    return {
        "contrast_id": contrast.contrast_id,
        "control_condition_id": contrast.control_condition_id,
        "treatment_condition_id": contrast.treatment_condition_id,
        "direction": "treatment/control",
        "gating": contrast.gating,
        "gating_eligible": gating_eligible,
        "classification": "gating" if gating_eligible else "descriptive-only",
        "classification_reasons": reasons,
        "join_on": contrast.join_on,
        "require_same_execution": contrast.require_same_execution,
        "threshold": contrast.threshold,
        "material_regression_limit": limit,
        "bootstrap_seed": seed,
        "paired_ratio": ({"point": point, "low": low, "high": high} if ratio is not None else None),
        "transcript_changes": transcript_changes,
    }


def summarize_experiment(
    spec: ExperimentSpec,
    responses: Mapping[str, Sequence[WorkerResponse]] | Sequence[WorkerResponse],
) -> dict[str, Any]:
    """Derive all condition summaries and directed contrasts from raw workers."""
    spec.validate()
    grouped: dict[str, list[WorkerResponse]] = defaultdict(list)
    if isinstance(responses, Mapping):
        values = [response for group in responses.values() for response in group]
    else:
        values = list(responses)
    for response in values:
        if not isinstance(response, WorkerResponse):
            raise MeasurementError("experiment responses must be WorkerResponse values")
        if response.condition_id not in spec.condition_map:
            raise MeasurementError(f"response has undeclared condition {response.condition_id}")
        grouped[response.condition_id].append(response)

    schedule = build_schedule(spec)
    summaries = {
        condition.condition_id: _condition_summary(
            spec,
            condition,
            grouped.get(condition.condition_id, ()),
            schedule,
        )
        for condition in spec.conditions
    }
    contrasts = [_contrast_summary(spec, contrast, summaries) for contrast in spec.contrasts]
    gating = [item for item in contrasts if item["gating"]]
    return {
        "artifact_kind": EXPERIMENT_KIND,
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": spec.experiment_id,
        "spec": spec.to_dict(),
        "session_schedule": [[entry.to_dict() for entry in session] for session in schedule],
        "condition_summaries": [summaries[condition.condition_id] for condition in spec.conditions],
        "contrasts": contrasts,
        "gate_eligible": bool(gating) and all(item["gating_eligible"] for item in gating),
    }


build_experiment = summarize_experiment


__all__ = [
    "ConditionSpec",
    "ContrastSpec",
    "EXPERIMENT_KIND",
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentSpec",
    "InputSetSpec",
    "QUANTILE_METHOD",
    "STATISTICS_ESTIMATOR",
    "STATISTICS_VERSION",
    "TRUSTED_MIN_REPEATS",
    "TRUSTED_MIN_SESSIONS",
    "TRUSTED_MIN_WARMUPS",
    "ScheduleEntry",
    "StatisticsSpec",
    "build_experiment",
    "build_schedule",
    "observer_accounting",
    "summarize_experiment",
]
