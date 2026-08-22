"""Stage-by-stage parity checks for ASR adapter/reference implementations.

The parity report is deliberately independent of the benchmark and evidence
schemas.  It answers a narrower question: given one loaded model instance and
one canonical waveform, where do an adapter and its upstream entry point first
stop agreeing?  Optional runtime hooks return :class:`ParityTrace` objects;
the comparison layer never treats a missing value as an agreement.

The public API is intentionally small:

``canonical_pcm_facts``
    Read the exact PCM16 waveform identity used by the report.
``ParityTrace`` / ``ParityReport``
    JSON-safe-ish value objects for one trace and a report keyed by backend and
    model.
``compare_traces``
    Compare two traces with explicit reachability and float tolerances.
``run_parity``
    Load one backend once, collect adapter/reference traces from that instance,
    and return a ``parity-v1`` report.
``write_report`` / ``read_report``
    Atomically publish and validate a report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from stt.audio import canonical_audio_id

SCHEMA = "parity-v1"
SCHEMA_VERSION = 1
STAGE_NAMES = (
    "decoded_pcm",
    "features",
    "logits_or_encoder",
    "token_ids",
    "raw_transcript",
    "final_transcript",
)
FLOAT_STAGES = frozenset({"features", "logits_or_encoder"})
STATUSES = frozenset({"match", "differ", "not-reachable"})
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4
_HEX = frozenset("0123456789abcdef")
EntryPoint = Literal["adapter", "reference"]


class ParityError(ValueError):
    """A parity input or report violates the parity-v1 contract."""


@dataclass(frozen=True)
class StageTolerance:
    """Numeric tolerance for one reachable float stage."""

    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL

    def validate(self) -> None:
        if (
            not math.isfinite(self.atol)
            or not math.isfinite(self.rtol)
            or self.atol < 0
            or self.rtol < 0
        ):
            raise ParityError("parity tolerances must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {"atol": self.atol, "rtol": self.rtol}


@dataclass(frozen=True)
class ParityContract:
    """Expected stages and tolerances for one backend/model/revision."""

    backend: str
    model: str
    expected_reachability: Mapping[str, bool]
    tolerances: Mapping[str, StageTolerance] = field(default_factory=dict)

    def validate(self) -> None:
        subject_key(self.backend, self.model)
        if set(self.expected_reachability) != set(STAGE_NAMES):
            raise ParityError("parity reachability must declare every stage exactly once")
        if any(not isinstance(value, bool) for value in self.expected_reachability.values()):
            raise ParityError("parity reachability values must be booleans")
        if set(self.tolerances) - FLOAT_STAGES:
            raise ParityError("parity tolerances may only describe float stages")
        for stage in FLOAT_STAGES:
            tolerance = self.tolerances.get(stage, StageTolerance())
            if not isinstance(tolerance, StageTolerance):
                raise ParityError(f"invalid tolerance for {stage}")
            tolerance.validate()

    def tolerance(self, stage: str) -> StageTolerance:
        self.validate()
        return self.tolerances.get(stage, StageTolerance())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "backend": self.backend,
            "model": self.model,
            "expected_reachability": dict(self.expected_reachability),
            "tolerances": {
                stage: self.tolerance(stage).to_dict() for stage in sorted(FLOAT_STAGES)
            },
        }


def contract_for(backend: str, model: str) -> ParityContract:
    """Return the reviewed reachability contract for one implemented lane."""
    if (backend, model) == ("hf", "mms-1b-all"):
        return ParityContract(
            backend,
            model,
            {stage: True for stage in STAGE_NAMES},
            {
                "features": StageTolerance(atol=1e-6, rtol=1e-6),
                "logits_or_encoder": StageTolerance(atol=1e-5, rtol=1e-4),
            },
        )
    raise ParityError(f"no reviewed parity contract for {backend}/{model}")


def subject_key(backend: str, model: str) -> str:
    """Return the stable report key for one backend/model pair."""
    if not isinstance(backend, str) or not backend:
        raise ParityError("backend must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise ParityError("model must be a non-empty string")
    return f"{backend}+{model}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - diagnostics must remain serializable
            pass
    return str(value)


def _materialize(value: Any) -> Any:
    """Detach tensor-like values without importing an optional tensor runtime."""
    if value is None:
        return None
    detached = value
    detach = getattr(detached, "detach", None)
    if callable(detach):
        detached = detach()
    cpu = getattr(detached, "cpu", None)
    if callable(cpu):
        detached = cpu()
    numpy = getattr(detached, "numpy", None)
    if callable(numpy):
        try:
            detached = numpy()
        except Exception:  # noqa: BLE001 - leave custom values to JSON fallback
            pass
    return detached


def _array(value: Any) -> np.ndarray | None:
    value = _materialize(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            candidate = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if candidate.dtype.kind in "biufc":
            return candidate
    return None


def _canonical_bytes(value: Any) -> bytes:
    """Encode a trace value deterministically for exact digest comparisons."""
    value = _materialize(value)
    array = _array(value)
    if array is not None:
        # Include shape and dtype so ``[1, 2]`` and ``[[1, 2]]`` cannot hash the
        # same way, and normalise byte order for cross-machine reports.
        normalized = np.ascontiguousarray(array)
        header = json.dumps(
            {
                "dtype": str(normalized.dtype),
                "shape": list(normalized.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return header + b"\0" + normalized.tobytes(order="C")
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_json_default,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return repr(value).encode("utf-8")


def digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a parity value."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _representation(value: Any, supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe a value without embedding its full tensor in the report."""
    result = dict(supplied or {})
    materialized = _materialize(value)
    array = _array(materialized)
    if array is not None:
        result.setdefault("kind", "array")
        result.setdefault("dtype", str(array.dtype))
        result.setdefault("shape", list(array.shape))
    elif isinstance(materialized, str):
        result.setdefault("kind", "text")
        result.setdefault("encoding", "utf-8")
    elif isinstance(materialized, (bytes, bytearray)):
        result.setdefault("kind", "bytes")
    elif materialized is None:
        result.setdefault("kind", "unavailable")
    else:
        result.setdefault("kind", type(materialized).__name__)
    return result


def _max_differences(left: Any, right: Any) -> tuple[float | None, float | None, str | None]:
    left_array = _array(left)
    right_array = _array(right)
    if left_array is None or right_array is None:
        return None, None, None
    if left_array.shape != right_array.shape:
        return None, None, "shape mismatch"
    try:
        left_float = left_array.astype(np.float64, copy=False)
        right_float = right_array.astype(np.float64, copy=False)
        difference = np.abs(left_float - right_float)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.divide(
                difference,
                np.maximum(np.abs(right_float), np.finfo(np.float64).tiny),
            )
        finite_difference = difference[np.isfinite(difference)]
        finite_relative = relative[np.isfinite(relative)]
        max_abs = float(np.max(finite_difference)) if finite_difference.size else math.inf
        max_rel = float(np.max(finite_relative)) if finite_relative.size else math.inf
        if not np.all(np.isfinite(left_float)) or not np.all(np.isfinite(right_float)):
            return max_abs, max_rel, "non-finite value"
        return max_abs, max_rel, None
    except (TypeError, ValueError, OverflowError):
        return None, None, "values are not numeric arrays"


def _same_exact(left: Any, right: Any) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


@dataclass(frozen=True)
class ParityTrace:
    """Values observed through one entry point for one waveform.

    ``None`` means that the entry point cannot expose a stage.  Empty strings
    and empty arrays remain reachable and are compared normally.
    """

    decoded_pcm: Any = None
    features: Any = None
    logits_or_encoder: Any = None
    token_ids: Any = None
    raw_transcript: str | None = None
    final_transcript: str | None = None
    representations: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def value(self, stage: str) -> Any:
        if stage not in STAGE_NAMES:
            raise ParityError(f"unknown parity stage {stage!r}")
        return getattr(self, stage)

    def reachable(self, stage: str) -> bool:
        return self.value(stage) is not None

    def representation(self, stage: str, value: Any | None = None) -> dict[str, Any]:
        supplied = self.representations.get(stage)
        return _representation(self.value(stage) if value is None else value, supplied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": {
                stage: {
                    "reachable": self.reachable(stage),
                    "digest": digest(self.value(stage)) if self.reachable(stage) else None,
                    "representation": self.representation(stage),
                }
                for stage in STAGE_NAMES
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParityTrace:
        stages = data.get("stages", {})
        if not isinstance(stages, Mapping):
            raise ParityError("trace stages must be an object")
        values: dict[str, Any] = {}
        representations: dict[str, Mapping[str, Any]] = {}
        for stage in STAGE_NAMES:
            item = stages.get(stage)
            if not isinstance(item, Mapping) or not item.get("reachable", False):
                values[stage] = None
                continue
            # Reports intentionally store digests rather than potentially huge
            # tensors.  A read-back trace is therefore diagnostic metadata, not
            # a substitute for the original runtime trace.
            values[stage] = item.get("value")
            representations[stage] = dict(item.get("representation") or {})
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ParityError("trace metadata must be an object")
        return cls(**values, representations=representations, metadata=dict(metadata))


@dataclass(frozen=True)
class StageComparison:
    """Comparison of one independent parity stage."""

    name: str
    status: str
    adapter_digest: str | None
    reference_digest: str | None
    adapter_reachable: bool
    reference_reachable: bool
    adapter_representation: Mapping[str, Any]
    reference_representation: Mapping[str, Any]
    atol: float | None = None
    rtol: float | None = None
    max_abs: float | None = None
    max_rel: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.name not in STAGE_NAMES:
            raise ParityError(f"unknown parity stage {self.name!r}")
        if self.status not in STATUSES:
            raise ParityError(f"invalid parity status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "adapter": {
                "reachable": self.adapter_reachable,
                "digest": self.adapter_digest,
                "representation": dict(self.adapter_representation),
            },
            "reference": {
                "reachable": self.reference_reachable,
                "digest": self.reference_digest,
                "representation": dict(self.reference_representation),
            },
            "tolerance": (
                {"atol": self.atol, "rtol": self.rtol} if self.name in FLOAT_STAGES else None
            ),
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageComparison:
        adapter = data.get("adapter") or {}
        reference = data.get("reference") or {}
        tolerance = data.get("tolerance") or {}
        return cls(
            name=str(data.get("name", "")),
            status=str(data.get("status", "")),
            adapter_digest=adapter.get("digest"),
            reference_digest=reference.get("digest"),
            adapter_reachable=bool(adapter.get("reachable", False)),
            reference_reachable=bool(reference.get("reachable", False)),
            adapter_representation=dict(adapter.get("representation") or {}),
            reference_representation=dict(reference.get("representation") or {}),
            atol=tolerance.get("atol"),
            rtol=tolerance.get("rtol"),
            max_abs=data.get("max_abs"),
            max_rel=data.get("max_rel"),
            reason=data.get("reason"),
        )


def compare_stage(
    stage: str,
    adapter: Any,
    reference: Any,
    *,
    adapter_representation: Mapping[str, Any] | None = None,
    reference_representation: Mapping[str, Any] | None = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> StageComparison:
    """Compare one stage, preserving explicit reachability on both sides."""
    if stage not in STAGE_NAMES:
        raise ParityError(f"unknown parity stage {stage!r}")
    adapter_reachable = adapter is not None
    reference_reachable = reference is not None
    adapter_digest = digest(adapter) if adapter_reachable else None
    reference_digest = digest(reference) if reference_reachable else None
    adapter_repr = _representation(adapter, adapter_representation)
    reference_repr = _representation(reference, reference_representation)

    if not adapter_reachable and not reference_reachable:
        return StageComparison(
            stage,
            "not-reachable",
            None,
            None,
            False,
            False,
            adapter_repr,
            reference_repr,
            atol if stage in FLOAT_STAGES else None,
            rtol if stage in FLOAT_STAGES else None,
            reason="neither entry point exposes this stage",
        )
    if adapter_reachable != reference_reachable:
        return StageComparison(
            stage,
            "differ",
            adapter_digest,
            reference_digest,
            adapter_reachable,
            reference_reachable,
            adapter_repr,
            reference_repr,
            atol if stage in FLOAT_STAGES else None,
            rtol if stage in FLOAT_STAGES else None,
            reason="stage is reachable through only one entry point",
        )

    max_abs: float | None = None
    max_rel: float | None = None
    reason: str | None = None
    if stage in FLOAT_STAGES:
        max_abs, max_rel, reason = _max_differences(adapter, reference)
        if reason == "shape mismatch":
            status = "differ"
        elif max_abs is None or max_rel is None:
            status = "differ"
            reason = reason or "float stage did not contain comparable numeric arrays"
        else:
            status = (
                "match"
                if np.allclose(
                    _array(adapter),
                    _array(reference),
                    atol=atol,
                    rtol=rtol,
                    equal_nan=False,
                )
                else "differ"
            )
    else:
        status = "match" if _same_exact(adapter, reference) else "differ"
    return StageComparison(
        stage,
        status,
        adapter_digest,
        reference_digest,
        True,
        True,
        adapter_repr,
        reference_repr,
        atol if stage in FLOAT_STAGES else None,
        rtol if stage in FLOAT_STAGES else None,
        max_abs,
        max_rel,
        reason,
    )


def compare_traces(
    adapter: ParityTrace,
    reference: ParityTrace,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    tolerances: Mapping[str, StageTolerance] | None = None,
) -> tuple[StageComparison, ...]:
    """Compare all six stages independently and in their declared order."""
    if not isinstance(adapter, ParityTrace) or not isinstance(reference, ParityTrace):
        raise ParityError("compare_traces expects two ParityTrace instances")
    comparisons: list[StageComparison] = []
    for stage in STAGE_NAMES:
        tolerance = (tolerances or {}).get(stage, StageTolerance(atol, rtol))
        tolerance.validate()
        comparisons.append(
            compare_stage(
                stage,
                adapter.value(stage),
                reference.value(stage),
                adapter_representation=adapter.representations.get(stage),
                reference_representation=reference.representations.get(stage),
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
        )
    return tuple(comparisons)


def first_divergent_stage(stages: Iterable[StageComparison]) -> str | None:
    """Return the earliest stage that actually differs, if any."""
    return next((item.name for item in stages if item.status == "differ"), None)


def _pcm_digest(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    digest_value = hashlib.sha256()
    frames = 0
    sample_rate: int | None = None
    channels: int | None = None
    blocks: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as audio:
        sample_rate = int(audio.samplerate)
        channels = int(audio.channels)
        frames = int(audio.frames)
        for block in audio.blocks(blocksize=65_536, dtype="int16", always_2d=True):
            little_endian = np.asarray(block, dtype="<i2", order="C")
            digest_value.update(little_endian.tobytes(order="C"))
            blocks.append(little_endian)
    samples = (
        np.concatenate(blocks, axis=0) if blocks else np.empty((0, channels or 1), dtype="<i2")
    )
    if len(samples) != frames:
        raise ParityError(f"could not read all PCM frames from {path}")
    duration = frames / sample_rate if sample_rate else 0.0
    return (
        {
            "pcm_sha256": digest_value.hexdigest(),
            # ``sha256`` is a convenient generic alias used by callers that
            # do not need to know the stage-specific field name.
            "sha256": digest_value.hexdigest(),
            "audio_id": canonical_audio_id(path),
            "sample_rate": sample_rate,
            "channels": channels,
            "frames": frames,
            "duration_s": duration,
            "representation": {
                "kind": "pcm16",
                "dtype": "int16",
                "shape": [frames, channels],
                "byte_order": "little",
            },
        },
        samples,
    )


def canonical_pcm_facts(path: str | Path) -> dict[str, Any]:
    """Return path-independent PCM facts for the exact waveform under test."""
    facts, _ = _pcm_digest(Path(path))
    return facts


@dataclass(frozen=True)
class ParityCase:
    """One waveform comparison within a backend/model subject."""

    audio: Mapping[str, Any]
    stages: tuple[StageComparison, ...]
    first_divergent_stage: str | None
    contract_issues: tuple[str, ...] = ()
    adapter_metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio": dict(self.audio),
            "stages": [stage.to_dict() for stage in self.stages],
            "first_divergent_stage": self.first_divergent_stage,
            "contract_issues": list(self.contract_issues),
            "parity_eligible": not self.first_divergent_stage and not self.contract_issues,
            "adapter_metadata": dict(self.adapter_metadata),
            "reference_metadata": dict(self.reference_metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParityCase:
        stages = tuple(StageComparison.from_dict(item) for item in data.get("stages", []))
        if tuple(stage.name for stage in stages) != STAGE_NAMES:
            raise ParityError("parity case must contain each stage exactly once in order")
        return cls(
            audio=dict(data.get("audio") or {}),
            stages=stages,
            first_divergent_stage=data.get("first_divergent_stage"),
            contract_issues=tuple(str(item) for item in data.get("contract_issues", [])),
            adapter_metadata=dict(data.get("adapter_metadata") or {}),
            reference_metadata=dict(data.get("reference_metadata") or {}),
        )


@dataclass(frozen=True)
class ParitySubject:
    """All cases for one backend/model pair."""

    backend: str
    model: str
    cases: tuple[ParityCase, ...]
    language: str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    binding_stable: bool = True
    loaded_instance_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return subject_key(self.backend, self.model)

    @property
    def parity_eligible(self) -> bool:
        return (
            self.binding_stable
            and bool(self.cases)
            and all(
                not case.first_divergent_stage and not case.contract_issues for case in self.cases
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "key": self.key,
            "language": self.language,
            "settings": dict(self.settings),
            "binding_stable": self.binding_stable,
            "parity_eligible": self.parity_eligible,
            "loaded_instance_id": self.loaded_instance_id,
            "cases": [case.to_dict() for case in self.cases],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParitySubject:
        value = cls(
            backend=str(data.get("backend", "")),
            model=str(data.get("model", "")),
            cases=tuple(ParityCase.from_dict(item) for item in data.get("cases", [])),
            language=data.get("language"),
            settings=dict(data.get("settings") or {}),
            binding_stable=bool(data.get("binding_stable", False)),
            loaded_instance_id=data.get("loaded_instance_id"),
            metadata=dict(data.get("metadata") or {}),
        )
        if data.get("key") not in {None, value.key}:
            raise ParityError("parity subject key does not match backend and model")
        return value


@dataclass(frozen=True)
class ParityReport:
    """A parity-v1 report keyed by ``backend+model``."""

    subjects: Mapping[str, ParitySubject]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ParityError("unsupported parity report schema")
        for key, subject in self.subjects.items():
            if key != subject.key:
                raise ParityError(f"parity subject is stored under the wrong key: {key!r}")

    @property
    def keyed(self) -> Mapping[str, ParitySubject]:
        """Alias emphasizing that subjects are indexed by backend and model."""
        return self.subjects

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "subjects": {key: value.to_dict() for key, value in sorted(self.subjects.items())},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParityReport:
        if data.get("schema") != SCHEMA or data.get("schema_version") != SCHEMA_VERSION:
            raise ParityError("unsupported parity report schema")
        raw_subjects = data.get("subjects")
        if not isinstance(raw_subjects, Mapping):
            raise ParityError("parity report subjects must be an object")
        subjects = {str(key): ParitySubject.from_dict(value) for key, value in raw_subjects.items()}
        return cls(subjects, dict(data.get("metadata") or {}))


def _binding_fingerprint(backend: Any) -> str | None:
    binding = getattr(backend, "model_binding", None)
    if binding is None:
        return None
    try:
        payload = binding.provenance.to_dict()
        payload["paths"] = [item.identity_dict() for item in binding.paths]
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=_json_default
            ).encode("utf-8")
        ).hexdigest()
    except Exception as exc:  # noqa: BLE001 - report stability check should be explicit
        raise ParityError(f"cannot snapshot model binding: {exc}") from exc


def _trace_from_backend(
    backend: Any,
    path: Path,
    language: str | None,
    entrypoint: EntryPoint,
) -> ParityTrace:
    method_name = f"parity_{entrypoint}_trace"
    method = getattr(backend, method_name, None)
    if not callable(method):
        raise ParityError(
            f"{getattr(backend, 'name', type(backend).__name__)}/"
            f"{getattr(backend, 'model', 'unknown')} does not implement {method_name}"
        )
    trace = method(path, language=language)
    if not isinstance(trace, ParityTrace):
        raise ParityError(f"{method_name} must return ParityTrace")
    return trace


def _runtime_provenance(backend: Any) -> Any:
    collector = getattr(backend, "model_provenance", None)
    if not callable(collector):
        raise ParityError("parity backend cannot report model provenance")
    provenance = collector().finalized()
    if not provenance.complete:
        raise ParityError("parity backend has incomplete model provenance")
    return provenance


def _reachability_issues(
    contract: ParityContract,
    adapter: ParityTrace,
    reference: ParityTrace,
) -> list[str]:
    issues: list[str] = []
    for stage in STAGE_NAMES:
        expected = contract.expected_reachability[stage]
        actual = (adapter.reachable(stage), reference.reachable(stage))
        if expected and actual != (True, True):
            issues.append(f"required stage {stage} is not reachable through both entry points")
        elif not expected and actual != (False, False):
            issues.append(f"stage {stage} was declared unreachable but emitted a value")
    return issues


def run_parity(
    adapter_backend: Any,
    audio_paths: str | Path | Iterable[str | Path],
    language: str | None = None,
    *,
    reference_backend: Any,
    contract: ParityContract,
    entrypoint_order: tuple[EntryPoint, EntryPoint] = ("adapter", "reference"),
    metadata: Mapping[str, Any] | None = None,
) -> ParityReport:
    """Run two independent bound entry points under one explicit contract."""
    contract.validate()
    if adapter_backend is reference_backend:
        raise ParityError("adapter and reference parity require distinct backend instances")
    if entrypoint_order not in {("adapter", "reference"), ("reference", "adapter")}:
        raise ParityError("entrypoint_order must contain adapter and reference exactly once")
    for backend in (adapter_backend, reference_backend):
        if getattr(backend, "name", None) != contract.backend:
            raise ParityError("parity backend differs from its contract")
        if getattr(backend, "model", None) != contract.model:
            raise ParityError("parity model differs from its contract")
    adapter_binding = _binding_fingerprint(adapter_backend)
    reference_binding = _binding_fingerprint(reference_backend)
    if not adapter_binding or adapter_binding != reference_binding:
        raise ParityError("parity entry points do not share one immutable model binding")

    if isinstance(audio_paths, (str, Path)):
        paths = [Path(audio_paths)]
    else:
        paths = [Path(item) for item in audio_paths]
    if not paths:
        raise ParityError("at least one audio path is required")
    for path in paths:
        if not path.is_file():
            raise ParityError(f"audio path is not a file: {path}")

    traces: dict[EntryPoint, list[ParityTrace]] = {}
    provenances: dict[EntryPoint, Any] = {}
    backends = {"adapter": adapter_backend, "reference": reference_backend}
    for entrypoint in entrypoint_order:
        backend = backends[entrypoint]
        fingerprint = _binding_fingerprint(backend)
        try:
            backend.load()
            if not getattr(backend, "_loaded", False):
                raise ParityError(f"{entrypoint} backend did not enter its loaded state")
            provenances[entrypoint] = _runtime_provenance(backend)
            traces[entrypoint] = [
                _trace_from_backend(backend, path, language, entrypoint) for path in paths
            ]
            if _binding_fingerprint(backend) != fingerprint:
                raise ParityError(f"{entrypoint} model binding changed during parity")
        finally:
            if getattr(backend, "_loaded", False):
                backend.unload()

    adapter_provenance = provenances["adapter"]
    reference_provenance = provenances["reference"]
    if (
        adapter_provenance.content_sha256 != reference_provenance.content_sha256
        or adapter_provenance.execution_sha256 != reference_provenance.execution_sha256
    ):
        raise ParityError("parity entry points resolved different model executions")

    cases: list[ParityCase] = []
    for index, path in enumerate(paths):
        facts, _ = _pcm_digest(path)
        adapter = traces["adapter"][index]
        reference = traces["reference"][index]
        stages = compare_traces(adapter, reference, tolerances=contract.tolerances)
        contract_issues = _reachability_issues(contract, adapter, reference)
        cases.append(
            ParityCase(
                audio=facts,
                stages=stages,
                first_divergent_stage=first_divergent_stage(stages),
                contract_issues=tuple(contract_issues),
                adapter_metadata=adapter.metadata,
                reference_metadata=reference.metadata,
            )
        )
    settings = {
        key: value
        for key, value in {
            "device": adapter_provenance.resolved_settings.get("device"),
            "dtype": adapter_provenance.resolved_settings.get("dtype"),
        }.items()
        if value is not None
    }
    subject = ParitySubject(
        backend=contract.backend,
        model=contract.model,
        cases=tuple(cases),
        language=language,
        settings=settings,
        binding_stable=True,
        loaded_instance_id=None,
        metadata={
            **dict(metadata or {}),
            "contract": contract.to_dict(),
            "entrypoint_order": list(entrypoint_order),
            "content_sha256": adapter_provenance.content_sha256,
            "execution_sha256": adapter_provenance.execution_sha256,
        },
    )
    return ParityReport(
        subjects={subject.key: subject},
        metadata={"entrypoint_order": list(entrypoint_order)},
    )


def write_report(report: ParityReport | Mapping[str, Any], path: str | Path) -> None:
    """Atomically publish a parity report and fsync the replacement."""
    if isinstance(report, ParityReport):
        payload = report.to_dict()
    else:
        payload = dict(report)
        # Validate before writing so a malformed object cannot become a report.
        ParityReport.from_dict(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(
                payload, output, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_report(path: str | Path) -> ParityReport:
    """Read and validate an on-disk parity-v1 report."""
    with Path(path).open(encoding="utf-8") as source:
        data = json.load(source)
    return ParityReport.from_dict(data)


# Descriptive aliases used by callers that prefer the long form.
write_parity_report = write_report
read_parity_report = read_report
compare = compare_traces


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "FLOAT_STAGES",
    "ParityCase",
    "ParityContract",
    "ParityError",
    "ParityReport",
    "ParityStage",
    "ParitySubject",
    "ParityTrace",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGE_NAMES",
    "StageComparison",
    "StageTolerance",
    "canonical_pcm_facts",
    "compare",
    "compare_stage",
    "compare_traces",
    "digest",
    "first_divergent_stage",
    "read_parity_report",
    "read_report",
    "run_parity",
    "subject_key",
    "write_parity_report",
    "write_report",
]


# ``ParityStage`` was used in an early draft of the parity API.  Keep the name
# as a type alias so downstream callers do not need a migration for this tiny
# vocabulary object.
ParityStage = StageComparison
