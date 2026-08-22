"""Versioned raw measurement artifacts for isolated benchmark workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf

from stt.audio import PreparedAudio, is_valid_audio_id
from stt.provenance import ModelBinding

PROTOCOL_VERSION = 3
RUNTIME_PACKAGES = (
    "torch",
    "torchaudio",
    "transformers",
    "huggingface-hub",
    "tokenizers",
    "accelerate",
    "sentencepiece",
    "omnilingual-asr",
    "fairseq2",
    "crispasr",
    "dataoceanai-dolphin",
    "modelscope",
    "funasr",
    "soundfile",
    "psutil",
)


class MeasurementError(ValueError):
    """A raw measurement artifact violates the protocol contract."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _bool_field(data: dict[str, Any], name: str, default: bool = False) -> bool:
    value = data.get(name, default)
    if not isinstance(value, bool):
        raise MeasurementError(f"{name} must be a boolean")
    return value


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _command_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output or None


@dataclass(frozen=True)
class AudioInput:
    """Canonical waveform identity and the path presented to a backend."""

    source_path: str
    prepared_path: str
    reference_id: str
    source_sha256: str
    audio_id: str
    duration_s: float
    sample_rate: int
    channels: int
    frames: int

    @classmethod
    def from_prepared(cls, prepared: PreparedAudio) -> AudioInput:
        info = sf.info(prepared.prepared_path)
        return cls(
            source_path=str(prepared.source_path.resolve(strict=False)),
            prepared_path=str(prepared.prepared_path.resolve(strict=False)),
            reference_id=prepared.reference_id,
            source_sha256=prepared.source_sha256,
            audio_id=prepared.audio_id,
            duration_s=info.duration,
            sample_rate=info.samplerate,
            channels=info.channels,
            frames=info.frames,
        )

    def validate(self) -> None:
        if not self.reference_id:
            raise MeasurementError("reference_id is required")
        if not is_valid_audio_id(self.audio_id):
            raise MeasurementError(f"invalid audio_id for {self.reference_id}")
        if len(self.source_sha256) != 64:
            raise MeasurementError(f"invalid source_sha256 for {self.reference_id}")
        if self.duration_s <= 0 or self.sample_rate <= 0 or self.channels <= 0 or self.frames <= 0:
            raise MeasurementError(f"invalid prepared audio facts for {self.reference_id}")

    def prepared(self) -> PreparedAudio:
        self.validate()
        return PreparedAudio(
            source_path=Path(self.source_path),
            prepared_path=Path(self.prepared_path),
            reference_id=self.reference_id,
            source_sha256=self.source_sha256,
            audio_id=self.audio_id,
        )


@dataclass(frozen=True)
class SubjectSpec:
    """Requested execution tuple before runtime resolution."""

    backend: str
    model: str | None
    language: str | None
    batch_size: int
    options: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.backend:
            raise MeasurementError("backend is required")
        if self.batch_size < 1:
            raise MeasurementError("batch_size must be at least 1")
        try:
            _canonical_json(self.options)
        except (TypeError, ValueError) as exc:
            raise MeasurementError("subject options must be JSON serializable") from exc

    @property
    def request_key(self) -> str:
        self.validate()
        digest = hashlib.sha256(_canonical_json(asdict(self)).encode("ascii")).hexdigest()[:16]
        label = f"{self.backend}/{self.model}" if self.model else self.backend
        return f"{label}:{digest}"


@dataclass(frozen=True)
class WorkerRequest:
    """Everything an isolated process needs to execute one subject."""

    run_id: str
    subject: SubjectSpec
    inputs: tuple[AudioInput, ...]
    warmups: int = 1
    repeats: int = 1
    profile: bool = False
    sample_uss: bool = False
    worker_index: int = 0
    experiment_id: str | None = None
    session_id: str | None = None
    session_index: int | None = None
    launch_position: int | None = None
    condition_id: str = "default"
    schedule_seed: int | None = None
    model_binding: ModelBinding | None = None
    protocol_version: int = PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise MeasurementError(
                f"unsupported measurement protocol {self.protocol_version}; "
                f"expected {PROTOCOL_VERSION}"
            )
        if not self.run_id:
            raise MeasurementError("run_id is required")
        self.subject.validate()
        if not self.inputs:
            raise MeasurementError("at least one input is required")
        if self.warmups < 0 or self.repeats < 1 or self.worker_index < 0:
            raise MeasurementError("invalid warmup, repeat, or worker count")
        if self.sample_uss and not self.profile:
            raise MeasurementError("USS sampling requires profiling")
        if self.session_index is not None and self.session_index < 0:
            raise MeasurementError("session_index must be non-negative")
        if self.launch_position is not None and self.launch_position < 0:
            raise MeasurementError("launch_position must be non-negative")
        if not self.condition_id:
            raise MeasurementError("condition_id is required")
        if self.model_binding is not None:
            self.model_binding.validate()
        for item in self.inputs:
            item.validate()
        audio_ids = [item.audio_id for item in self.inputs]
        reference_ids = [item.reference_id for item in self.inputs]
        if len(set(audio_ids)) != len(audio_ids):
            raise MeasurementError("worker inputs contain duplicate audio_id values")
        if len(set(reference_ids)) != len(reference_ids):
            raise MeasurementError("worker inputs contain duplicate reference_id values")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def identity_sha256(self) -> str:
        """Bind every execution-affecting request field to one canonical identity."""
        self.validate()
        return hashlib.sha256(_canonical_json(asdict(self)).encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerRequest:
        try:
            subject = SubjectSpec(**data["subject"])
            inputs = tuple(AudioInput(**item) for item in data["inputs"])
            request = cls(
                run_id=data["run_id"],
                subject=subject,
                inputs=inputs,
                warmups=int(data.get("warmups", 1)),
                repeats=int(data.get("repeats", 1)),
                profile=_bool_field(data, "profile"),
                sample_uss=_bool_field(data, "sample_uss"),
                worker_index=int(data.get("worker_index", 0)),
                experiment_id=data.get("experiment_id"),
                session_id=data.get("session_id"),
                session_index=(
                    int(data["session_index"]) if data.get("session_index") is not None else None
                ),
                launch_position=(
                    int(data["launch_position"])
                    if data.get("launch_position") is not None
                    else None
                ),
                condition_id=str(data.get("condition_id", "default")),
                schedule_seed=(
                    int(data["schedule_seed"]) if data.get("schedule_seed") is not None else None
                ),
                model_binding=(
                    ModelBinding.from_dict(data["model_binding"])
                    if data.get("model_binding") is not None
                    else None
                ),
                protocol_version=int(data.get("protocol_version", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid worker request: {exc}") from exc
        request.validate()
        return request


@dataclass(frozen=True)
class PhaseEvent:
    """A monotonic, recomputable worker phase interval."""

    name: str
    started_ns: int
    ended_ns: int
    status: str = "ok"
    error: str | None = None
    repeat_index: int | None = None

    def validate(self) -> None:
        if not self.name:
            raise MeasurementError("phase name is required")
        if self.started_ns < 0 or self.ended_ns < self.started_ns:
            raise MeasurementError(f"invalid monotonic interval for {self.name}")
        if self.status not in {"ok", "error"}:
            raise MeasurementError(f"invalid phase status {self.status!r}")
        if self.status == "error" and not self.error:
            raise MeasurementError(f"error phase {self.name} needs an error detail")

    @property
    def duration_s(self) -> float:
        return (self.ended_ns - self.started_ns) / 1_000_000_000


@dataclass(frozen=True)
class ActivePhase:
    """A phase which was active at the last durable journal checkpoint."""

    name: str
    started_ns: int
    repeat_index: int | None = None

    def validate(self) -> None:
        if not self.name or self.started_ns < 0:
            raise MeasurementError("invalid active phase")


@dataclass(frozen=True)
class WorkerJournal:
    """Atomic checkpoint containing all work completed before interruption."""

    run_id: str
    request_key: str
    request_sha256: str
    worker_index: int
    subject: SubjectSpec
    host: dict[str, Any]
    environment: dict[str, Any]
    runtime: dict[str, Any]
    phases: tuple[PhaseEvent, ...] = ()
    repeats: tuple[RepeatRecord, ...] = ()
    load_resources: dict[str, Any] | None = None
    model_provenance: dict[str, Any] | None = None
    provenance_issues: tuple[str, ...] = ()
    active_phase: ActivePhase | None = None
    updated_ns: int = 0
    complete: bool = False
    error: str | None = None
    experiment_id: str | None = None
    session_id: str | None = None
    session_index: int | None = None
    launch_position: int | None = None
    condition_id: str = "default"
    schedule_seed: int | None = None
    protocol_version: int = PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise MeasurementError(f"unsupported journal protocol {self.protocol_version}")
        if not self.run_id or not self.request_key:
            raise MeasurementError("journal identity is required")
        self.subject.validate()
        if self.request_key != self.subject.request_key:
            raise MeasurementError("journal request key does not match its subject")
        if not _is_sha256(self.request_sha256):
            raise MeasurementError("journal request SHA-256 is invalid")
        previous_end = -1
        phase_names: set[str] = set()
        for phase in self.phases:
            phase.validate()
            if phase.name in phase_names:
                raise MeasurementError(f"duplicate journal phase {phase.name}")
            phase_names.add(phase.name)
            if phase.started_ns < previous_end:
                raise MeasurementError("journal phase intervals overlap or are out of order")
            previous_end = phase.ended_ns
        repeat_indices: set[int] = set()
        for repeat in self.repeats:
            repeat.phase.validate()
            if repeat.index in repeat_indices:
                raise MeasurementError(f"duplicate repeat index {repeat.index}")
            repeat_indices.add(repeat.index)
            if repeat.phase.repeat_index != repeat.index:
                raise MeasurementError(f"repeat phase index mismatch for {repeat.index}")
            if repeat.complete and repeat.phase.status != "ok":
                raise MeasurementError(f"complete repeat {repeat.index} has an error phase")
            if repeat.phase not in self.phases:
                raise MeasurementError(f"repeat {repeat.index} phase is absent from the journal")
        if self.active_phase is not None:
            self.active_phase.validate()
            if self.active_phase.started_ns < previous_end:
                raise MeasurementError("active journal phase overlaps completed work")
        if self.updated_ns < 0:
            raise MeasurementError("journal updated_ns must be non-negative")
        latest_event = max(
            previous_end,
            self.active_phase.started_ns if self.active_phase is not None else -1,
        )
        if self.updated_ns < latest_event:
            raise MeasurementError("journal checkpoint predates recorded work")
        if repeat_indices and repeat_indices != set(range(len(repeat_indices))):
            raise MeasurementError("journal repeat indices are not contiguous")
        if self.complete and self.active_phase is not None:
            raise MeasurementError("complete journal cannot retain an active phase")
        if self.complete and (
            self.error or not self.repeats or any(not item.complete for item in self.repeats)
        ):
            raise MeasurementError("complete journal contains incomplete work")
        if self.complete and any(phase.status != "ok" for phase in self.phases):
            raise MeasurementError("complete journal contains an error phase")
        if self.complete and not {"process-start", "load", "unload"}.issubset(phase_names):
            raise MeasurementError("complete journal is missing a lifecycle phase")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerJournal:
        try:
            value = cls(
                run_id=str(data["run_id"]),
                request_key=str(data["request_key"]),
                request_sha256=str(data["request_sha256"]),
                worker_index=int(data["worker_index"]),
                subject=SubjectSpec(**data["subject"]),
                host=dict(data.get("host", {})),
                environment=dict(data.get("environment", {})),
                runtime=dict(data.get("runtime", {})),
                phases=tuple(PhaseEvent(**item) for item in data.get("phases", [])),
                repeats=tuple(
                    RepeatRecord(
                        index=int(item["index"]),
                        phase=PhaseEvent(**item["phase"]),
                        expected_audio_s=float(item["expected_audio_s"]),
                        results=tuple(item.get("results", [])),
                        resources=dict(item.get("resources", {})),
                        complete=_bool_field(item, "complete"),
                        trust_issues=tuple(item.get("trust_issues", [])),
                    )
                    for item in data.get("repeats", [])
                ),
                load_resources=data.get("load_resources"),
                model_provenance=data.get("model_provenance"),
                provenance_issues=tuple(data.get("provenance_issues", [])),
                active_phase=(
                    ActivePhase(**data["active_phase"])
                    if data.get("active_phase") is not None
                    else None
                ),
                updated_ns=int(data.get("updated_ns", 0)),
                complete=_bool_field(data, "complete"),
                error=data.get("error"),
                experiment_id=data.get("experiment_id"),
                session_id=data.get("session_id"),
                session_index=(
                    int(data["session_index"]) if data.get("session_index") is not None else None
                ),
                launch_position=(
                    int(data["launch_position"])
                    if data.get("launch_position") is not None
                    else None
                ),
                condition_id=str(data.get("condition_id", "default")),
                schedule_seed=(
                    int(data["schedule_seed"]) if data.get("schedule_seed") is not None else None
                ),
                protocol_version=int(data.get("protocol_version", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid worker journal: {exc}") from exc
        value.validate()
        return value

    def failure_response(
        self,
        error: str,
        *,
        ended_ns: int,
        termination: str | None = None,
    ) -> WorkerResponse:
        """Convert recovered state into an explicitly incomplete response."""
        phases = list(self.phases)
        if self.active_phase is not None:
            active = self.active_phase
            phases.append(
                PhaseEvent(
                    active.name,
                    active.started_ns,
                    max(active.started_ns, ended_ns),
                    status="error",
                    error=error,
                    repeat_index=active.repeat_index,
                )
            )
        return WorkerResponse(
            run_id=self.run_id,
            request_key=self.request_key,
            request_sha256=self.request_sha256,
            worker_index=self.worker_index,
            subject=self.subject,
            host=self.host,
            environment=self.environment,
            runtime=self.runtime,
            phases=tuple(phases),
            repeats=self.repeats,
            load_resources=self.load_resources,
            model_provenance=self.model_provenance,
            resolved_model=self.runtime.get("model"),
            resolved_device=self.runtime.get("device"),
            resolved_dtype=self.runtime.get("dtype"),
            complete=False,
            error=error,
            provenance_issues=tuple(dict.fromkeys((*self.provenance_issues, "journal recovery"))),
            experiment_id=self.experiment_id,
            session_id=self.session_id,
            session_index=self.session_index,
            launch_position=self.launch_position,
            condition_id=self.condition_id,
            schedule_seed=self.schedule_seed,
            journal_recovered=True,
            termination=termination,
        )


@dataclass(frozen=True)
class RepeatRecord:
    """One complete-corpus measured repeat and its raw results."""

    index: int
    phase: PhaseEvent
    expected_audio_s: float
    results: tuple[dict[str, Any], ...]
    resources: dict[str, Any]
    complete: bool
    trust_issues: tuple[str, ...] = ()

    @property
    def corpus_rtf(self) -> float | None:
        if not self.complete or self.expected_audio_s <= 0:
            return None
        return self.phase.duration_s / self.expected_audio_s


@dataclass(frozen=True)
class WorkerResponse:
    """Crash-safe output from one isolated subject worker."""

    run_id: str
    request_key: str
    request_sha256: str
    worker_index: int
    subject: SubjectSpec
    host: dict[str, Any]
    environment: dict[str, Any]
    runtime: dict[str, Any]
    phases: tuple[PhaseEvent, ...]
    repeats: tuple[RepeatRecord, ...]
    load_resources: dict[str, Any] | None
    resolved_model: str | None = None
    resolved_device: str | None = None
    resolved_dtype: str | None = None
    model_provenance: dict[str, Any] | None = None
    complete: bool = False
    error: str | None = None
    provenance_issues: tuple[str, ...] = ()
    experiment_id: str | None = None
    session_id: str | None = None
    session_index: int | None = None
    launch_position: int | None = None
    condition_id: str = "default"
    schedule_seed: int | None = None
    journal_recovered: bool = False
    termination: str | None = None
    protocol_version: int = PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise MeasurementError(f"unsupported response protocol {self.protocol_version}")
        if not self.run_id or not self.request_key:
            raise MeasurementError("response identity is required")
        self.subject.validate()
        if self.request_key != self.subject.request_key:
            raise MeasurementError("response request key does not match its subject")
        if not _is_sha256(self.request_sha256):
            raise MeasurementError("response request SHA-256 is invalid")
        previous_end = -1
        phase_names: set[str] = set()
        for phase in self.phases:
            phase.validate()
            if phase.name in phase_names:
                raise MeasurementError(f"duplicate response phase {phase.name}")
            phase_names.add(phase.name)
            if phase.started_ns < previous_end:
                raise MeasurementError("response phase intervals overlap or are out of order")
            previous_end = phase.ended_ns
        repeat_indices: set[int] = set()
        for repeat in self.repeats:
            repeat.phase.validate()
            if repeat.index in repeat_indices:
                raise MeasurementError(f"duplicate repeat index {repeat.index}")
            repeat_indices.add(repeat.index)
            if repeat.phase.repeat_index != repeat.index:
                raise MeasurementError(f"repeat phase index mismatch for {repeat.index}")
            if repeat.complete and repeat.phase.status != "ok":
                raise MeasurementError(f"complete repeat {repeat.index} has an error phase")
            if repeat.phase not in self.phases:
                raise MeasurementError(f"repeat {repeat.index} phase is absent from the response")
        if self.complete and (self.error or any(not repeat.complete for repeat in self.repeats)):
            raise MeasurementError("complete response contains an error or incomplete repeat")
        if self.complete and not self.repeats:
            raise MeasurementError("complete response contains no measured repeats")
        if self.complete and any(phase.status != "ok" for phase in self.phases):
            raise MeasurementError("complete response contains an error phase")
        if self.complete and not {"process-start", "load", "unload"}.issubset(phase_names):
            raise MeasurementError("complete response is missing a lifecycle phase")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerResponse:
        try:
            subject = SubjectSpec(**data["subject"])
            phases = tuple(PhaseEvent(**item) for item in data.get("phases", []))
            repeats = tuple(
                RepeatRecord(
                    index=int(item["index"]),
                    phase=PhaseEvent(**item["phase"]),
                    expected_audio_s=float(item["expected_audio_s"]),
                    results=tuple(item.get("results", [])),
                    resources=dict(item.get("resources", {})),
                    complete=_bool_field(item, "complete"),
                    trust_issues=tuple(item.get("trust_issues", [])),
                )
                for item in data.get("repeats", [])
            )
            response = cls(
                run_id=data["run_id"],
                request_key=data["request_key"],
                request_sha256=str(data["request_sha256"]),
                worker_index=int(data["worker_index"]),
                subject=subject,
                host=dict(data.get("host", {})),
                environment=dict(data.get("environment", {})),
                runtime=dict(data.get("runtime", {})),
                phases=phases,
                repeats=repeats,
                load_resources=data.get("load_resources"),
                resolved_model=data.get("resolved_model"),
                resolved_device=data.get("resolved_device"),
                resolved_dtype=data.get("resolved_dtype"),
                model_provenance=data.get("model_provenance"),
                complete=_bool_field(data, "complete"),
                error=data.get("error"),
                provenance_issues=tuple(data.get("provenance_issues", [])),
                experiment_id=data.get("experiment_id"),
                session_id=data.get("session_id"),
                session_index=(
                    int(data["session_index"]) if data.get("session_index") is not None else None
                ),
                launch_position=(
                    int(data["launch_position"])
                    if data.get("launch_position") is not None
                    else None
                ),
                condition_id=str(data.get("condition_id", "default")),
                schedule_seed=(
                    int(data["schedule_seed"]) if data.get("schedule_seed") is not None else None
                ),
                journal_recovered=_bool_field(data, "journal_recovered"),
                termination=data.get("termination"),
                protocol_version=int(data.get("protocol_version", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"invalid worker response: {exc}") from exc
        response.validate()
        return response


def new_run_id() -> str:
    return uuid.uuid4().hex


def capture_environment(root: Path | None = None) -> dict[str, Any]:
    """Record runtime facts that can make otherwise identical timings differ."""
    if root is None:
        from stt.paths import checkout_root

        root = checkout_root() or Path.cwd()
    root = root.resolve()
    packages: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "absent"

    commit = _command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = _command_output(["git", "-C", str(root), "status", "--porcelain"])
    lock_path = root / "uv.lock"
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "git_commit": commit,
        "git_dirty": bool(dirty),
        # Recorded so process isolation is checkable rather than asserted: two
        # subjects sharing a pid would mean the second inherited the first's
        # RSS high-water and whatever runtime state it left behind.
        "pid": os.getpid(),
        "uv_lock_sha256": _sha256_file(lock_path),
        "power": _command_output(["pmset", "-g", "batt"]) if sys.platform == "darwin" else None,
        "thermal": (
            _command_output(["pmset", "-g", "therm"]) if sys.platform == "darwin" else None
        ),
    }


def write_json_atomic(data: dict[str, Any], path: Path) -> None:
    """Publish a complete JSON artifact without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_request(request: WorkerRequest, path: Path) -> None:
    write_json_atomic(request.to_dict(), path)


def read_request(path: Path) -> WorkerRequest:
    return WorkerRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_response(response: WorkerResponse, path: Path) -> None:
    response.validate()
    write_json_atomic(response.to_dict(), path)


def read_response(path: Path) -> WorkerResponse:
    return WorkerResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_journal(journal: WorkerJournal, path: Path) -> None:
    write_json_atomic(journal.to_dict(), path)


def read_journal(path: Path) -> WorkerJournal:
    return WorkerJournal.from_dict(json.loads(path.read_text(encoding="utf-8")))
