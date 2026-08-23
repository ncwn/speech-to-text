"""Transactional publication and offline verification for experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from stt.experiment import (
    EXPERIMENT_KIND,
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
    build_schedule,
    summarize_experiment,
)
from stt.measurement import (
    MeasurementError,
    WorkerJournal,
    WorkerRequest,
    WorkerResponse,
    read_journal,
    read_request,
    read_response,
    write_journal,
    write_json_atomic,
    write_request,
    write_response,
)
from stt.paths import checkout_root

DEFAULT_EXPERIMENT_ROOT = Path("evidence/experiments")
MANIFEST_VERSION = "experiment-manifest-v1"
REQUIRED_WORKER_KINDS = frozenset({"request", "response", "journal", "stdout", "stderr"})
_ARTIFACT_RE = re.compile(
    r"^(?P<stem>.+)\.(?P<kind>request|response|journal|stdout|stderr)\.(?:json|txt)$"
)
_PATH_FIELDS = {
    "audio_path",
    "source_path",
    "prepared_path",
    "path",
    "python_executable",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_bytes(value: object) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_hash(entries: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "files": [dict(item) for item in entries],
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _path_safe(relative: str | Path, root: Path) -> Path:
    value = Path(relative)
    if value.is_absolute() or not str(value) or ".." in value.parts:
        raise ValueError(f"invalid experiment artifact path: {relative}")
    target = (root / value).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"experiment artifact path escapes archive: {relative}") from exc
    return target


def _summary_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "artifact_manifest",
        "manifest_sha256",
        "manifest_version",
        "artifact_root",
        "summary_artifact",
    }
    return {key: deepcopy(item) for key, item in value.items() if key not in excluded}


def _source_groups(
    raw_artifacts: Mapping[str, Sequence[str | Path]],
) -> dict[tuple[str, str], dict[str, Path]]:
    groups: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for condition_id, paths in raw_artifacts.items():
        for raw_path in paths:
            path = Path(raw_path).resolve()
            match = _ARTIFACT_RE.match(path.name)
            if not match:
                raise ValueError(f"unrecognized raw experiment artifact: {path}")
            key = (str(condition_id), match.group("stem"))
            kind = match.group("kind")
            if kind in groups[key]:
                raise ValueError(f"duplicate {kind} artifact for {condition_id}/{key[1]}")
            groups[key][kind] = path
    return dict(groups)


def _read_worker_group(
    condition_id: str,
    files: Mapping[str, Path],
) -> tuple[WorkerRequest, WorkerResponse, WorkerJournal]:
    if set(files) != REQUIRED_WORKER_KINDS:
        missing = sorted(REQUIRED_WORKER_KINDS - set(files))
        extra = sorted(set(files) - REQUIRED_WORKER_KINDS)
        raise ValueError(
            f"worker artifact coverage differs for {condition_id}: missing={missing}, extra={extra}"
        )
    for path in files.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"raw experiment artifact is unavailable: {path}")
    request = read_request(files["request"])
    response = read_response(files["response"])
    journal = read_journal(files["journal"])
    if request.condition_id != condition_id:
        raise ValueError(
            f"raw artifact condition differs: expected {condition_id}, found {request.condition_id}"
        )
    return request, response, journal


def _portable_path(value: str) -> str:
    """Return a checkout-relative path or reject an external local path."""
    path = Path(value)
    if not path.is_absolute():
        return value
    root = checkout_root()
    if root is None:
        raise ValueError(f"cannot publish an absolute local path outside a checkout: {value}")
    try:
        relative = path.relative_to(root.resolve())
        if ".." in relative.parts:
            raise ValueError
        return relative.as_posix()
    except ValueError as exc:
        raise ValueError(f"experiment artifact exposes an external local path: {value}") from exc


def _portable_json(value: Any, *, field_name: str | None = None) -> Any:
    """Canonicalize structured path fields without touching transcript text."""
    if isinstance(value, dict):
        return {str(key): _portable_json(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_json(item, field_name=field_name) for item in value]
    if isinstance(value, str) and (
        field_name in _PATH_FIELDS or bool(field_name and field_name.endswith(("_path", "_dir")))
    ):
        return _portable_path(value)
    return value


def _reject_local_absolute_strings(value: Any, *, location: str = "root") -> None:
    """Prove no user-local absolute path escaped structured canonicalization."""
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_local_absolute_strings(item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_local_absolute_strings(item, location=f"{location}[{index}]")
        return
    if not isinstance(value, str):
        return
    root = checkout_root()
    local_prefixes = tuple(
        prefix
        for prefix in (
            str(root.resolve()) if root is not None else None,
            "/Users/",
            "/Volumes/",
        )
        if prefix
    )
    if value.startswith(local_prefixes):
        raise ValueError(f"experiment artifact retains a local path at {location}")


def _portable_worker_group(
    request: WorkerRequest,
    response: WorkerResponse,
    journal: WorkerJournal,
) -> tuple[WorkerRequest, WorkerResponse, WorkerJournal]:
    """Canonicalize archived copies and consistently rebind their request hash."""
    request_raw = _portable_json(request.to_dict())
    _reject_local_absolute_strings(request_raw, location="request")
    portable_request = WorkerRequest.from_dict(request_raw)
    request_sha256 = portable_request.identity_sha256

    response_raw = _portable_json(response.to_dict())
    response_raw["request_sha256"] = request_sha256
    _reject_local_absolute_strings(response_raw, location="response")
    portable_response = WorkerResponse.from_dict(response_raw)

    journal_raw = _portable_json(journal.to_dict())
    journal_raw["request_sha256"] = request_sha256
    _reject_local_absolute_strings(journal_raw, location="journal")
    portable_journal = WorkerJournal.from_dict(journal_raw)
    return portable_request, portable_response, portable_journal


def _portable_log_text(value: str) -> str:
    root = checkout_root()
    if root is not None:
        value = value.replace(str(root.resolve()), ".")
    if "/Users/" in value or "/Volumes/" in value:
        raise ValueError("experiment worker log retains an external local path")
    return value


def publish_experiment(
    summary: Mapping[str, Any],
    raw_artifacts: Mapping[str, Sequence[str | Path]],
    *,
    root: Path = DEFAULT_EXPERIMENT_ROOT,
) -> Path:
    """Verify a staged archive, then atomically publish it."""
    candidate = deepcopy(dict(summary))
    if candidate.get("artifact_kind") != EXPERIMENT_KIND:
        raise ValueError("only experiment-v1 summaries can be published")
    if candidate.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("experiment schema version is unsupported")
    try:
        spec = ExperimentSpec.from_dict(candidate["spec"])
    except (KeyError, MeasurementError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid experiment summary specification: {exc}") from exc
    if candidate.get("experiment_id") != spec.experiment_id:
        raise ValueError("experiment summary identity differs from its specification")

    groups = _source_groups(raw_artifacts)
    expected_group_count = len(spec.conditions) * spec.sessions
    if len(groups) != expected_group_count:
        raise ValueError(
            f"raw experiment worker count differs: expected {expected_group_count}, "
            f"found {len(groups)}"
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / spec.experiment_id
    try:
        # The empty directory is an atomic reservation. A competing publisher
        # loses here before it can inspect or clean up this transaction.
        archive.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"experiment archive already exists: {archive}") from exc
    staging: Path | None = None
    transaction = root / f".{spec.experiment_id}.transaction.json"
    manifest: list[dict[str, Any]] = []
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{spec.experiment_id}.", dir=root))
        write_json_atomic(
            {
                "experiment_id": spec.experiment_id,
                "staging": str(staging),
                "archive": str(archive),
            },
            transaction,
        )
        seen_request_sha256: set[str] = set()
        for (condition_id, _), files in sorted(groups.items()):
            request, response, journal = _read_worker_group(condition_id, files)
            request, response, journal = _portable_worker_group(request, response, journal)
            request_sha256 = request.identity_sha256
            if request_sha256 in seen_request_sha256:
                raise ValueError(f"duplicate worker request identity: {request_sha256}")
            seen_request_sha256.add(request_sha256)
            worker_name = f"{int(request.session_index or 0):04d}-{request_sha256[:16]}"
            worker_dir = Path("workers") / condition_id / worker_name
            for kind in sorted(REQUIRED_WORKER_KINDS):
                extension = "json" if kind in {"request", "response", "journal"} else "txt"
                relative = worker_dir / f"{kind}.{extension}"
                destination = _path_safe(relative, staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if kind == "request":
                    write_request(request, destination)
                elif kind == "response":
                    write_response(response, destination)
                elif kind == "journal":
                    write_journal(journal, destination)
                else:
                    value = files[kind].read_text(encoding="utf-8")
                    destination.write_text(_portable_log_text(value), encoding="utf-8")
                manifest.append(
                    {
                        "path": relative.as_posix(),
                        "kind": kind,
                        "condition_id": condition_id,
                        "request_sha256": request_sha256,
                        "session_id": request.session_id,
                        "session_index": request.session_index,
                        "sha256": _sha256(destination),
                        "size_bytes": destination.stat().st_size,
                    }
                )
            if response.request_sha256 != request_sha256:
                raise ValueError("raw response request identity differs")
            if journal.request_sha256 != request_sha256:
                raise ValueError("raw journal request identity differs")

        summary_projection = _summary_projection(candidate)
        summary_path = staging / "summary.json"
        write_json_atomic(summary_projection, summary_path)
        manifest.append(
            {
                "path": "summary.json",
                "kind": "summary",
                "condition_id": None,
                "request_sha256": None,
                "session_id": None,
                "session_index": None,
                "sha256": _sha256(summary_path),
                "size_bytes": summary_path.stat().st_size,
            }
        )
        manifest.sort(key=lambda item: str(item["path"]))
        candidate.update(
            {
                "manifest_version": MANIFEST_VERSION,
                "artifact_manifest": manifest,
                "manifest_sha256": _manifest_hash(manifest),
                "summary_artifact": "summary.json",
                "artifact_root": ".",
            }
        )
        write_json_atomic(candidate, staging / "experiment.json")
        issues = verify_experiment(staging)
        if issues:
            raise ValueError("staged experiment verification failed: " + "; ".join(issues))
        os.replace(staging, archive)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        # Only the process that created the empty reservation can reach here.
        # Never remove a descriptor-bearing archive during error cleanup.
        if archive.is_dir() and not (archive / "experiment.json").exists():
            shutil.rmtree(archive, ignore_errors=True)
        transaction.unlink(missing_ok=True)
        raise
    transaction.unlink(missing_ok=True)
    return archive / "experiment.json"


save_experiment = publish_experiment


def _descriptor_path(path: Path) -> Path:
    return path / "experiment.json" if path.is_dir() else path


def _read_descriptor(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    descriptor = _descriptor_path(path)
    if not descriptor.is_file() or descriptor.is_symlink():
        return None, [f"experiment descriptor is unavailable: {descriptor}"]
    try:
        value = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"cannot read experiment descriptor: {exc}"]
    if not isinstance(value, dict):
        return None, ["experiment descriptor must be an object"]
    return value, []


def _manifest_files(
    descriptor: Mapping[str, Any],
    root: Path,
    issues: list[str],
) -> dict[str, dict[str, Any]]:
    raw_entries = descriptor.get("artifact_manifest")
    if not isinstance(raw_entries, list) or not raw_entries:
        issues.append("experiment has no artifact manifest")
        return {}
    found: dict[str, dict[str, Any]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or not raw_entry.get("path"):
            issues.append("experiment manifest contains an invalid entry")
            continue
        relative = str(raw_entry["path"])
        if relative in found:
            issues.append(f"duplicate experiment manifest path: {relative}")
            continue
        try:
            target = _path_safe(relative, root)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if not target.is_file() or target.is_symlink():
            issues.append(f"experiment artifact is unavailable: {relative}")
            continue
        try:
            if _sha256(target) != raw_entry.get("sha256"):
                issues.append(f"experiment artifact checksum mismatch: {relative}")
            if target.stat().st_size != raw_entry.get("size_bytes"):
                issues.append(f"experiment artifact size mismatch: {relative}")
        except OSError as exc:
            issues.append(f"cannot inspect experiment artifact {relative}: {exc}")
        found[relative] = raw_entry
    canonical_entries = [found[path] for path in sorted(found)]
    if descriptor.get("manifest_version") != MANIFEST_VERSION:
        issues.append("experiment manifest version is unsupported")
    if descriptor.get("manifest_sha256") != _manifest_hash(canonical_entries):
        issues.append("experiment whole-manifest checksum mismatch")
    return found


def _request_issues(
    request: WorkerRequest,
    *,
    spec: ExperimentSpec,
    condition_id: str,
    request_sha256: str,
    schedule: Sequence[Sequence[Any]],
) -> list[str]:
    issues: list[str] = []
    condition = spec.condition_map.get(condition_id)
    if condition is None:
        return [f"archived request has undeclared condition {condition_id}"]
    if request.identity_sha256 != request_sha256:
        issues.append(f"archived request digest differs for {condition_id}")
    if request.run_id != spec.experiment_id or request.experiment_id != spec.experiment_id:
        issues.append(f"archived request experiment identity differs for {condition_id}")
    if request.condition_id != condition_id:
        issues.append(f"archived request condition identity differs for {condition_id}")
    if request.subject != condition.subject:
        issues.append(f"archived request subject differs for {condition_id}")
    expected_input_mode = spec.input_map[condition.input_set_id].input_mode
    if request.runner_id != condition.runner_id:
        issues.append(f"archived request runner differs for {condition_id}")
    if request.input_mode != expected_input_mode:
        issues.append(f"archived request input mode differs for {condition_id}")
    if request.profile != condition.profile or request.sample_uss != condition.sample_uss:
        issues.append(f"archived request observer flags differ for {condition_id}")
    expected_inputs = spec.input_map[condition.input_set_id].inputs
    if request.inputs != expected_inputs:
        issues.append(f"archived request input set differs for {condition_id}")
    if request.warmups != spec.warmups or request.repeats != spec.repeats:
        issues.append(f"archived request repeat protocol differs for {condition_id}")
    gating_conditions = {
        condition
        for contrast in spec.contrasts
        if contrast.gating
        for condition in (
            contrast.control_condition_id,
            contrast.treatment_condition_id,
        )
    }
    if condition_id in gating_conditions and request.model_binding is None:
        issues.append(f"archived gating request lacks a model binding for {condition_id}")
    if request.schedule_seed != spec.schedule_seed:
        issues.append(f"archived request schedule seed differs for {condition_id}")
    session_index = request.session_index
    if session_index is None or not 0 <= session_index < spec.sessions:
        issues.append(f"archived request session index differs for {condition_id}")
        return issues
    expected_entry = next(
        item for item in schedule[session_index] if item.condition_id == condition_id
    )
    expected_session_id = f"{spec.experiment_id}:session-{session_index}"
    if (
        request.worker_index != session_index
        or request.session_id != expected_session_id
        or request.launch_position != expected_entry.launch_position
    ):
        issues.append(f"archived request session coordinates differ for {condition_id}")
    return issues


def _response_matches_request(
    response: WorkerResponse | WorkerJournal,
    request: WorkerRequest,
) -> bool:
    return all(
        (
            response.run_id == request.run_id,
            response.request_key == request.subject.request_key,
            response.request_sha256 == request.identity_sha256,
            response.worker_index == request.worker_index,
            response.subject == request.subject,
            response.experiment_id == request.experiment_id,
            response.session_id == request.session_id,
            response.session_index == request.session_index,
            response.launch_position == request.launch_position,
            response.condition_id == request.condition_id,
            response.schedule_seed == request.schedule_seed,
        )
    )


def _journal_matches_response(journal: WorkerJournal, response: WorkerResponse) -> bool:
    fields = (
        "run_id",
        "request_key",
        "request_sha256",
        "worker_index",
        "subject",
        "host",
        "environment",
        "runtime",
        "phases",
        "repeats",
        "load_resources",
        "model_provenance",
        "provenance_issues",
        "complete",
        "error",
        "experiment_id",
        "session_id",
        "session_index",
        "launch_position",
        "condition_id",
        "schedule_seed",
        "protocol_version",
    )
    return (
        journal.active_phase is None
        and all(getattr(journal, field) == getattr(response, field) for field in fields)
        and response.resolved_model == journal.runtime.get("model")
        and response.resolved_device == journal.runtime.get("device")
        and response.resolved_dtype == journal.runtime.get("dtype")
        and not response.journal_recovered
        and response.termination is None
    )


def verify_experiment(path: Path) -> list[str]:
    """Return every issue found while recomputing an archived experiment."""
    descriptor, issues = _read_descriptor(Path(path))
    if descriptor is None:
        return issues
    if descriptor.get("artifact_kind") != EXPERIMENT_KIND:
        issues.append("experiment artifact kind is unsupported")
    if descriptor.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        issues.append("experiment schema version is unsupported")
    try:
        spec = ExperimentSpec.from_dict(descriptor.get("spec", {}))
    except (MeasurementError, TypeError, ValueError, KeyError) as exc:
        issues.append(f"invalid experiment specification: {exc}")
        return issues
    if descriptor.get("experiment_id") != spec.experiment_id:
        issues.append("experiment descriptor identity differs from its specification")
    archive_root = _descriptor_path(Path(path)).parent
    manifest = _manifest_files(descriptor, archive_root, issues)
    schedule = build_schedule(spec)
    expected_schedule = [[item.to_dict() for item in session] for session in schedule]
    if _canonical_json(descriptor.get("session_schedule")) != _canonical_json(expected_schedule):
        issues.append("experiment session schedule differs from its specification")

    grouped_entries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    summary_entry: dict[str, Any] | None = None
    for entry in manifest.values():
        kind = str(entry.get("kind", ""))
        if kind == "summary":
            if summary_entry is not None:
                issues.append("experiment contains multiple summary artifacts")
            summary_entry = entry
            continue
        request_sha256 = str(entry.get("request_sha256", ""))
        if kind not in REQUIRED_WORKER_KINDS or not request_sha256:
            issues.append(f"unsupported experiment artifact declaration: {entry.get('path')}")
            continue
        if kind in grouped_entries[request_sha256]:
            issues.append(f"duplicate archived {kind} for request {request_sha256}")
        grouped_entries[request_sha256][kind] = entry

    response_groups: dict[str, list[WorkerResponse]] = defaultdict(list)
    observed_coordinates: set[tuple[str, int | None]] = set()
    for request_sha256, entries in grouped_entries.items():
        if set(entries) != REQUIRED_WORKER_KINDS:
            issues.append(f"archived worker coverage differs for request {request_sha256}")
            continue
        condition_ids = {str(entry.get("condition_id")) for entry in entries.values()}
        if len(condition_ids) != 1:
            issues.append(f"archived worker condition declarations differ: {request_sha256}")
            continue
        condition_id = next(iter(condition_ids))
        declared_session_ids = {entry.get("session_id") for entry in entries.values()}
        declared_session_indices = {entry.get("session_index") for entry in entries.values()}
        if len(declared_session_ids) != 1 or len(declared_session_indices) != 1:
            issues.append(f"archived worker session declarations differ: {request_sha256}")
            continue
        paths = {
            kind: _path_safe(str(entry["path"]), archive_root) for kind, entry in entries.items()
        }
        try:
            request = read_request(paths["request"])
            response = read_response(paths["response"])
            journal = read_journal(paths["journal"])
        except (OSError, ValueError, MeasurementError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read archived worker {request_sha256}: {exc}")
            continue
        issues.extend(
            _request_issues(
                request,
                spec=spec,
                condition_id=condition_id,
                request_sha256=request_sha256,
                schedule=schedule,
            )
        )
        if declared_session_ids != {request.session_id} or declared_session_indices != {
            request.session_index
        }:
            issues.append(f"archived manifest session differs for request {request_sha256}")
        if not _response_matches_request(response, request):
            issues.append(f"archived response differs from request {request_sha256}")
        if not _response_matches_request(journal, request):
            issues.append(f"archived journal differs from request {request_sha256}")
        if not _journal_matches_response(journal, response):
            issues.append(f"archived journal/response state differs for {request_sha256}")
        if request.model_binding is not None:
            bound = request.model_binding.provenance
            recorded = response.model_provenance
            if not isinstance(recorded, dict) or (
                recorded.get("content_sha256") != bound.content_sha256
                or recorded.get("execution_sha256") != bound.execution_sha256
            ):
                issues.append(f"archived response provenance differs from binding {request_sha256}")
        coordinate = (condition_id, request.session_index)
        if coordinate in observed_coordinates:
            issues.append(f"duplicate experiment condition/session coordinate: {coordinate}")
        observed_coordinates.add(coordinate)
        response_groups[condition_id].append(response)

    expected_coordinates = {
        (condition.condition_id, session_index)
        for condition in spec.conditions
        for session_index in range(spec.sessions)
    }
    if observed_coordinates != expected_coordinates:
        missing = sorted(expected_coordinates - observed_coordinates)
        extra = sorted(observed_coordinates - expected_coordinates)
        issues.append(
            f"experiment worker coordinate coverage differs: missing={missing}, extra={extra}"
        )

    recomputed: dict[str, Any] | None = None
    try:
        recomputed = summarize_experiment(spec, response_groups)
    except (MeasurementError, TypeError, ValueError, KeyError) as exc:
        issues.append(f"cannot recompute experiment summary: {exc}")
    if recomputed is not None and _canonical_json(
        _summary_projection(descriptor)
    ) != _canonical_json(recomputed):
        issues.append("derived experiment summary differs from archived workers")

    if summary_entry is None:
        issues.append("experiment summary artifact is missing")
    else:
        try:
            summary_path = _path_safe(str(summary_entry["path"]), archive_root)
            stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if _canonical_json(stored_summary) != _canonical_json(_summary_projection(descriptor)):
                issues.append("experiment summary artifact differs from descriptor")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read experiment summary artifact: {exc}")
    return issues


__all__ = [
    "DEFAULT_EXPERIMENT_ROOT",
    "MANIFEST_VERSION",
    "REQUIRED_WORKER_KINDS",
    "publish_experiment",
    "save_experiment",
    "verify_experiment",
]
