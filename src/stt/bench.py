"""Reproducible benchmark orchestration and baseline-v2 statistics.

Raw worker responses remain the source of truth. Independent fresh workers are
the statistical blocks; repeats inside one worker are retained but never treated
as independent observations. Trusted timing gates require complete provenance,
counterbalanced sessions, and deterministic block bootstrap summaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stt.measurement import (
    AudioInput,
    MeasurementError,
    WorkerJournal,
    WorkerRequest,
    WorkerResponse,
    capture_environment,
    read_journal,
    read_request,
    read_response,
    write_json_atomic,
    write_request,
    write_response,
)
from stt.provenance import ModelProvenance, ProvenanceError
from stt.telemetry import Series, pool_series

#: Where the committed baseline lives.
BASELINE = Path("baselines/bench.json")
ARTIFACT_DIR = Path("outputs/bench")
MEASUREMENT_SCHEMA_VERSION = 3
BASELINE_SCHEMA_VERSION = 2
STATISTICS_VERSION = "block-bootstrap-v1"
BOOTSTRAP_DRAWS = 20_000

#: Clips to measure over. The FLEURS dev split is fetched in TSV order, so the
#: first N are stable across machines — and the stems are recorded anyway, so a
#: changed set is caught rather than silently compared.
CLIP_COUNT = 5
CLIP_DIR = Path("data/fleurs/audio")

#: (backend, model). The 7B is here because it is where the headline numbers
#: come from; it is also why this is a `make bench` target and not a git hook.
SUBJECTS: tuple[tuple[str, str | None], ...] = (
    ("hf", "mms-1b-all"),
    ("hf", "seamless-m4t-v2"),
    ("omniasr-gguf", None),
    ("dolphin", "small"),
    ("omniasr-torch", "omniASR_LLM_Unlimited_7B_v2"),
)


def text_hash(text: str) -> str:
    """Full SHA-256 transcript identity used by baseline-v2."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed_for(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def counterbalanced_schedule(
    subjects: list[str] | tuple[str, ...], sessions: int, seed: int
) -> list[list[tuple[str, int]]]:
    """Return a seeded cyclic schedule balanced across launch positions."""
    if not subjects or sessions < 1:
        raise ValueError("subjects and sessions are required")
    rng = np.random.default_rng(seed)
    base = list(subjects)
    rng.shuffle(base)
    return [
        [
            (subject, position)
            for position, subject in enumerate(
                base[index % len(base) :] + base[: index % len(base)]
            )
        ]
        for index in range(sessions)
    ]


def _bootstrap_median(values: list[float], seed: int) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        value = float(values[0])
        return value, value
    ordered = np.asarray(sorted(values), dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(ordered, size=(BOOTSTRAP_DRAWS, len(ordered)))
    medians = np.median(samples, axis=1)
    return (
        float(np.quantile(medians, 0.025, method="linear")),
        float(np.quantile(medians, 0.975, method="linear")),
    )


def _natural_variation_ratio(values: list[float], seed: int) -> float | None:
    """99th percentile ratio of two independently resampled block medians."""
    if not values:
        return None
    if len(values) == 1:
        return 1.0
    logs = np.log(np.asarray(sorted(values), dtype=float))
    rng = np.random.default_rng(seed)
    left = rng.choice(logs, size=(BOOTSTRAP_DRAWS, len(logs)))
    right = rng.choice(logs, size=(BOOTSTRAP_DRAWS, len(logs)))
    delta = np.median(right, axis=1) - np.median(left, axis=1)
    return float(math.exp(max(0.0, np.quantile(delta, 0.99, method="linear"))))


@dataclass
class Finding:
    """One regression, or one thing that merely deserves a mention."""

    subject: str
    signal: str
    detail: str
    fatal: bool = True


@dataclass
class Comparison:
    """The verdict for one baseline against one fresh measurement."""

    findings: list[Finding] = field(default_factory=list)
    compared_timings: bool = True
    host_note: str | None = None

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.fatal]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if not f.fatal]

    @property
    def ok(self) -> bool:
        return not self.failures


def clips(directory: Path = CLIP_DIR, count: int = CLIP_COUNT) -> list[Path]:
    """The fixed measurement set, in a stable order."""
    found = sorted(directory.glob("*.wav"))[:count]
    if len(found) < count:
        raise FileNotFoundError(
            f"Need {count} clips in {directory}, found {len(found)}. "
            "Run: uv run stt fetch-fleurs --limit 20"
        )
    return found


def host_signature() -> dict[str, Any]:
    """Capture the complete host/runtime context used by a v2 comparison."""
    from stt.telemetry import describe_host

    try:
        import torch

        version = torch.__version__
    except ImportError:  # pragma: no cover - torch is present in every extra
        version = "absent"
    environment = capture_environment()
    return {
        **describe_host(),
        "torch": version,
        "platform": environment.get("platform"),
        "machine": environment.get("machine"),
        "python": environment.get("python"),
        "packages": environment.get("packages", {}),
        "uv_lock_sha256": environment.get("uv_lock_sha256"),
        "git_commit": environment.get("git_commit"),
        "git_dirty": environment.get("git_dirty"),
        "power": environment.get("power"),
        "thermal": environment.get("thermal"),
    }


def _timing_host_identity(host: dict[str, Any]) -> dict[str, Any]:
    power = host.get("power")
    power_source = str(power).splitlines()[0] if power else None
    return {
        "chip": host.get("chip"),
        "platform": host.get("platform"),
        "machine": host.get("machine"),
        "cores": host.get("cores"),
        "compute_threads": host.get("compute_threads"),
        "cpu_count": host.get("cpu_count"),
        "ram_mb": host.get("ram_mb"),
        "python": host.get("python"),
        "packages": host.get("packages"),
        "uv_lock_sha256": host.get("uv_lock_sha256"),
        "power_source": power_source,
        "thermal": host.get("thermal"),
    }


def _artifact_stem(request: WorkerRequest) -> str:
    label = request.subject.backend
    if request.subject.model:
        label += f"-{request.subject.model}"
    safe = "".join(
        character if character.isalnum() or character in "._-" else "-" for character in label
    )
    digest = request.identity_sha256[:16]
    return f"worker-{request.worker_index:02d}-{safe}-{digest}"


def _matches_request(value: WorkerJournal | WorkerResponse, request: WorkerRequest) -> bool:
    return (
        value.run_id == request.run_id
        and value.request_key == request.subject.request_key
        and value.request_sha256 == request.identity_sha256
        and value.worker_index == request.worker_index
        and value.subject == request.subject
        and value.experiment_id == request.experiment_id
        and value.session_id == request.session_id
        and value.session_index == request.session_index
        and value.launch_position == request.launch_position
        and value.condition_id == request.condition_id
        and value.schedule_seed == request.schedule_seed
    )


def _journal_matches_response(journal: WorkerJournal, response: WorkerResponse) -> bool:
    if journal.active_phase is not None:
        return False
    shared_fields = (
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
        all(getattr(journal, field) == getattr(response, field) for field in shared_fields)
        and response.resolved_model == journal.runtime.get("model")
        and response.resolved_device == journal.runtime.get("device")
        and response.resolved_dtype == journal.runtime.get("dtype")
        and not response.journal_recovered
        and response.termination is None
    )


def _worker_group_exists(process: subprocess.Popen[str]) -> bool:
    if os.name != "posix":
        return process.poll() is None
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_worker_process(process: subprocess.Popen[str]) -> None:
    """Terminate the entire worker group and reap its leader before returning."""

    def signal_group(value: signal.Signals) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, value)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        try:
            process.send_signal(value)
        except OSError:
            pass

    signal_group(signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while _worker_group_exists(process) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _worker_group_exists(process):
        signal_group(signal.SIGKILL)
    if process.poll() is None:
        process.wait()
    kill_deadline = time.monotonic() + 5.0
    while _worker_group_exists(process) and time.monotonic() < kill_deadline:
        time.sleep(0.05)
    if _worker_group_exists(process):
        raise RuntimeError("worker process group survived SIGKILL")


def run_subject_worker(
    request: WorkerRequest,
    artifact_dir: Path = ARTIFACT_DIR,
    *,
    timeout_s: float = 7_200,
) -> tuple[WorkerResponse, Path]:
    """Run one subject in a fresh process and retain its raw request/response."""
    run_dir = artifact_dir / request.run_id
    stem = _artifact_stem(request)
    request_path = run_dir / f"{stem}.request.json"
    response_path = run_dir / f"{stem}.response.json"
    journal_path = run_dir / f"{stem}.journal.json"
    stdout_path = run_dir / f"{stem}.stdout.txt"
    stderr_path = run_dir / f"{stem}.stderr.txt"
    write_request(request, request_path)

    # A journal timestamp is a child-process monotonic timestamp.  Recording
    # the launch boundary lets recovery reject a checkpoint left by an earlier
    # attempt that happens to reuse the same artifact directory.
    launch_started_ns = time.perf_counter_ns()
    timed_out = False
    timeout_observed_ns: int | None = None
    launch_error: str | None = None
    cleanup_error: str | None = None
    return_code: int | None = None
    process: subprocess.Popen[str] | None = None

    def force_reap_worker() -> bool:
        """Make one final kill/reap attempt when cleanup is interrupted or fails."""
        if process is None:
            return False
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except BaseException:  # noqa: BLE001 - preserve the original interrupt
                pass
        try:
            process.kill()
        except BaseException:  # noqa: BLE001 - preserve the original interrupt
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            return False
        except BaseException:  # noqa: BLE001 - preserve the original interrupt
            return False
        try:
            if process.poll() is None or _worker_group_exists(process):
                return False
        except BaseException:  # noqa: BLE001 - termination cannot be proven
            return False
        return True

    def cleanup_worker(*, preserve_interrupt: bool = False) -> None:
        nonlocal cleanup_error
        if process is None:
            return
        try:
            _terminate_worker_process(process)
        except Exception as exc:  # noqa: BLE001 - preserve recovery after cleanup failure
            cleanup_error = f"worker process cleanup failed: {type(exc).__name__}: {exc}"
            if not force_reap_worker():
                cleanup_error += "; force-reap could not prove worker termination"
        except BaseException:  # noqa: BLE001 - preserve the original interrupt
            force_reap_worker()
            if not preserve_interrupt:
                raise

    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "stt.bench_worker",
                    str(request_path),
                    str(response_path),
                    str(journal_path),
                ],
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=(os.name == "posix"),
            )
            try:
                return_code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                timeout_observed_ns = time.perf_counter_ns()
                cleanup_worker()
                return_code = process.returncode
            except BaseException:
                cleanup_worker(preserve_interrupt=True)
                raise
    except OSError as exc:
        launch_error = f"worker launch failed: {exc}"

    if process is not None and not timed_out and return_code not in {None, 0, 1}:
        cleanup_worker()

    termination_parts: list[str] = []
    if timed_out:
        termination_parts.append(f"TimeoutError: subject worker timed out after {timeout_s:g}s")
    elif launch_error:
        termination_parts.append(launch_error)
    elif return_code not in {None, 0, 1}:
        termination_parts.append(f"worker exited with status {return_code}")
    if cleanup_error:
        termination_parts.append(cleanup_error)
    termination = "; ".join(termination_parts) or None

    response: WorkerResponse | None = None
    invalid_response = False
    if response_path.is_file():
        try:
            response = read_response(response_path)
        except (MeasurementError, OSError, ValueError, json.JSONDecodeError):
            response = None
            invalid_response = True
    if response is not None and not _matches_request(response, request):
        response = None
        invalid_response = True
        termination = "worker response identity mismatch"

    if response is not None and (
        (return_code == 0 and not response.complete)
        or (return_code == 1 and response.complete)
        or return_code not in {0, 1}
    ):
        response = None
        invalid_response = True
        termination = f"worker exit status {return_code} contradicts its response"

    if response is not None and termination is None:
        try:
            journal = read_journal(journal_path)
            if (
                not _matches_request(journal, request)
                or journal.updated_ns <= 0
                or journal.updated_ns < launch_started_ns
                or not _journal_matches_response(journal, response)
            ):
                raise MeasurementError("journal identity, freshness, or response check failed")
        except (MeasurementError, OSError, ValueError, json.JSONDecodeError) as exc:
            response = None
            invalid_response = True
            termination = f"journal validation failed: {exc}"

    if termination:
        if response is not None:
            invalid_response = True
        response = None

    if response is None:
        if invalid_response and response_path.is_file():
            invalid_path = response_path.with_name(f"{response_path.name}.invalid-{time.time_ns()}")
            os.replace(response_path, invalid_path)
        recovery_issue = None
        try:
            journal = read_journal(journal_path)
            if not _matches_request(journal, request) or journal.updated_ns < launch_started_ns:
                raise MeasurementError("journal identity or freshness mismatch")
            if journal.updated_ns <= 0:
                raise MeasurementError("journal is stale or has no checkpoint timestamp")
            detail = termination or "worker published no response"
            if journal.error:
                detail = f"{detail}; journal error: {journal.error}"
            response = journal.failure_response(
                detail,
                ended_ns=timeout_observed_ns or time.perf_counter_ns(),
                termination=termination,
            )
        except (MeasurementError, OSError, ValueError, json.JSONDecodeError) as exc:
            recovery_issue = f"journal recovery unavailable: {exc}"
            detail = termination or "worker published no response"
            response = WorkerResponse(
                run_id=request.run_id,
                request_key=request.subject.request_key,
                request_sha256=request.identity_sha256,
                worker_index=request.worker_index,
                subject=request.subject,
                host={},
                environment={},
                runtime={},
                phases=(),
                repeats=(),
                load_resources=None,
                complete=False,
                error=detail,
                provenance_issues=(recovery_issue,),
                experiment_id=request.experiment_id,
                session_id=request.session_id,
                session_index=request.session_index,
                launch_position=request.launch_position,
                condition_id=request.condition_id,
                schedule_seed=request.schedule_seed,
                journal_recovered=False,
                termination=termination,
            )
        response_output_path = response_path
        if cleanup_error:
            response_output_path = response_path.with_name(
                f"{response_path.name}.recovered-{uuid.uuid4().hex}"
            )
        if response_path.parent.exists():
            write_response(response, response_output_path)
        if cleanup_error:
            raise RuntimeError(
                f"{cleanup_error}; recovered response captured at {response_output_path}"
            )
        return response, response_output_path
    return response, response_path


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize_workers(
    subject: str,
    responses: list[WorkerResponse],
    inputs: list[AudioInput],
    *,
    expected_workers: int | None = None,
    expected_warmups: int | None = None,
    expected_repeats: int | None = None,
    expected_launch_positions: list[int] | tuple[int, ...] | None = None,
    expected_session_coordinates: list[tuple[int, int, str]] | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Summarize independent workers without treating repeats as replicates."""
    responses = sorted(
        responses,
        key=lambda response: (
            response.session_index if response.session_index is not None else sys.maxsize,
            response.worker_index,
            response.session_id or "",
            response.condition_id,
        ),
    )
    expected_ids = {item.audio_id for item in inputs}
    expected_refs = {item.audio_id: item.reference_id for item in inputs}
    errors = [
        response.error or "worker response is incomplete"
        for response in responses
        if not response.complete
    ]
    provenance_issues = list(
        dict.fromkeys(issue for response in responses for issue in response.provenance_issues)
    )
    if not responses:
        errors.append("no worker response")

    worker_blocks: list[dict[str, Any]] = []
    seen_workers: set[tuple[int, str | None]] = set()
    seen_sessions: set[tuple[int | None, int | None]] = set()
    hashes_by_repeat: list[dict[str, str]] = []
    request_keys: set[str] = set()
    condition_ids: set[str] = set()
    schedule_seeds: set[int | None] = set()
    host_identities: set[str] = set()
    environment_identities: set[str] = set()
    expected_audio_s = sum(item.duration_s for item in inputs)
    for response in responses:
        key = (response.worker_index, response.session_id)
        try:
            response.validate()
        except MeasurementError as exc:
            errors.append(f"worker {key} violates the response contract: {exc}")
        request_keys.add(response.request_key)
        condition_ids.add(response.condition_id)
        schedule_seeds.add(response.schedule_seed)
        host_identities.add(json.dumps(response.host, sort_keys=True, separators=(",", ":")))
        stable_environment = {
            field: response.environment.get(field)
            for field in (
                "python",
                "platform",
                "machine",
                "packages",
                "git_commit",
                "git_dirty",
                "uv_lock_sha256",
            )
        }
        environment_identities.add(
            json.dumps(stable_environment, sort_keys=True, separators=(",", ":"))
        )
        if key in seen_workers:
            errors.append(f"duplicate worker block {key}")
        seen_workers.add(key)
        response_subject = (
            f"{response.subject.backend}/{response.subject.model}"
            if response.subject.model
            else response.subject.backend
        )
        if response_subject != subject:
            errors.append(
                f"worker {key} subject mismatch: expected {subject}, got {response_subject}"
            )
        session_key = (response.session_index, response.launch_position)
        if session_key in seen_sessions:
            errors.append(f"duplicate session/launch coordinate {session_key}")
        seen_sessions.add(session_key)
        if experiment_id is not None and response.experiment_id != experiment_id:
            errors.append(f"worker {key} belongs to a different experiment")
        if experiment_id is not None and (
            response.session_id is None or response.session_index is None
        ):
            errors.append(f"worker {key} lacks independent session coordinates")
        provenance: ModelProvenance | None = None
        if response.model_provenance is None and experiment_id is not None:
            provenance_issues.append(f"worker {key} lacks model provenance")
        elif response.model_provenance is not None:
            try:
                provenance = ModelProvenance.from_dict(response.model_provenance)
                if experiment_id is not None and not provenance.complete:
                    provenance_issues.append(f"worker {key} has incomplete model provenance")
            except (KeyError, TypeError, ValueError, ProvenanceError) as exc:
                if experiment_id is not None:
                    provenance_issues.append(f"worker {key} has invalid model provenance: {exc}")
                provenance = None
        representative_values: list[float] = []
        repeat_values: list[float | None] = []
        worker_hashes: list[dict[str, str]] = []
        worker_rss: list[float] = []
        repeat_indices = [repeat.index for repeat in response.repeats]
        if expected_repeats is not None and len(response.repeats) != expected_repeats:
            errors.append(
                f"worker {key} has {len(response.repeats)} repeats; expected {expected_repeats}"
            )
        if expected_warmups is not None:
            warmup_count = sum(phase.name.startswith("warmup-") for phase in response.phases)
            if warmup_count != expected_warmups:
                errors.append(
                    f"worker {key} has {warmup_count} warmup phases; expected {expected_warmups}"
                )
        if len(set(repeat_indices)) != len(repeat_indices):
            errors.append(f"worker {key} contains duplicate repeat indices")
        if repeat_indices and set(repeat_indices) != set(range(len(repeat_indices))):
            errors.append(f"worker {key} repeat indices are not contiguous")
        for repeat in sorted(response.repeats, key=lambda item: item.index):
            if not repeat.complete:
                errors.extend(repeat.trust_issues or (f"repeat {repeat.index} is incomplete",))
            if not math.isclose(
                repeat.expected_audio_s,
                expected_audio_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(f"repeat {repeat.index} expected audio duration differs")
            actual = [str(result.get("audio_id")) for result in repeat.results]
            if len(actual) != len(expected_ids) or set(actual) != expected_ids:
                errors.append(f"repeat {repeat.index} audio identity coverage differs")
                continue
            mapping = {str(result.get("audio_id")): result for result in repeat.results}
            if set(mapping) != expected_ids or len(mapping) != len(actual):
                errors.append(f"repeat {repeat.index} contains duplicate audio identities")
                continue
            if any(
                result.get("reference_id") != expected_refs.get(audio_id)
                for audio_id, result in mapping.items()
            ):
                errors.append(f"repeat {repeat.index} reference identity differs")
            for result in mapping.values():
                if not result.get("trusted") or result.get("error"):
                    errors.append(f"repeat {repeat.index} contains an untrusted result")
                result_provenance = result.get("model_provenance")
                if result_provenance is None and experiment_id is not None:
                    provenance_issues.append(
                        f"repeat {repeat.index} contains a result without model provenance"
                    )
                elif result_provenance is not None and not isinstance(result_provenance, dict):
                    provenance_issues.append(
                        f"repeat {repeat.index} contains invalid model provenance"
                    )
                elif provenance is not None and result_provenance != response.model_provenance:
                    provenance_issues.append(
                        f"repeat {repeat.index} result provenance differs from worker"
                    )
            hashes = {
                audio_id: text_hash(str(result.get("text", "")))
                for audio_id, result in mapping.items()
            }
            worker_hashes.append(hashes)
            hashes_by_repeat.append(hashes)
            if repeat.corpus_rtf is not None and repeat.corpus_rtf > 0:
                repeat_values.append(repeat.corpus_rtf)
                representative_values.append(math.log(repeat.corpus_rtf))
            else:
                repeat_values.append(None)
            process_peak = repeat.resources.get("process_peak_rss_mb")
            if process_peak is None:
                process_peak = repeat.resources.get("peak_rss_mb")
            if process_peak is not None:
                worker_rss.append(float(process_peak))
        if worker_hashes and any(candidate != worker_hashes[0] for candidate in worker_hashes[1:]):
            errors.append(f"transcript hashes changed across repeats in worker {key}")
        representative = (
            math.exp(statistics.median(representative_values)) if representative_values else None
        )
        worker_blocks.append(
            {
                "worker_index": response.worker_index,
                "session_id": response.session_id,
                "session_index": response.session_index,
                "launch_position": response.launch_position,
                "repeat_rtfs": repeat_values,
                "representative_rtf": representative,
                "hashes": worker_hashes[0] if worker_hashes else {},
                "process_peak_rss_mb": max(worker_rss) if worker_rss else None,
            }
        )

    if expected_workers is not None and len(responses) != expected_workers:
        errors.append(f"expected {expected_workers} worker blocks, found {len(responses)}")
    if len(request_keys) > 1:
        errors.append("worker requests differ within the subject condition")
    if len(condition_ids) > 1:
        errors.append("condition identity changed across workers")
    if len(schedule_seeds) > 1:
        errors.append("schedule seed changed across workers")
    if len(host_identities) > 1:
        errors.append("worker host identity changed across sessions")
    if len(environment_identities) > 1:
        errors.append("worker runtime environment changed across sessions")
    if experiment_id is not None and expected_workers is not None:
        observed_sessions = {
            response.session_index for response in responses if response.session_index is not None
        }
        observed_positions = [
            response.launch_position
            for response in responses
            if response.launch_position is not None
        ]
        expected_sessions = set(range(expected_workers))
        if observed_sessions != expected_sessions:
            errors.append(
                "session matrix differs "
                f"(missing={sorted(expected_sessions - observed_sessions)}, "
                f"extra={sorted(observed_sessions - expected_sessions)})"
            )
        if expected_launch_positions is not None and sorted(observed_positions) != sorted(
            expected_launch_positions
        ):
            errors.append(
                "launch-position matrix differs "
                f"(expected={sorted(expected_launch_positions)}, "
                f"observed={sorted(observed_positions)})"
            )
        if expected_session_coordinates is not None:
            observed_coordinates = sorted(
                (
                    int(response.session_index),
                    int(response.launch_position),
                    str(response.session_id),
                )
                for response in responses
                if response.session_index is not None
                and response.launch_position is not None
                and response.session_id is not None
            )
            if observed_coordinates != sorted(expected_session_coordinates):
                errors.append("worker session coordinates differ from the declared schedule")

    hashes = hashes_by_repeat[0] if hashes_by_repeat else {}
    if any(candidate != hashes for candidate in hashes_by_repeat[1:]):
        errors.append("transcript hashes changed across measured repeats")

    block_rtfs = [
        float(block["representative_rtf"])
        for block in worker_blocks
        if block.get("representative_rtf") is not None
    ]
    endpoint_rss = [
        float(value)
        for response in responses
        for repeat in response.repeats
        if (value := repeat.resources.get("rss_peak_mb")) is not None
    ]
    lifetime_rss = [
        float(block["process_peak_rss_mb"])
        for block in worker_blocks
        if block.get("process_peak_rss_mb") is not None
    ]
    cpu_utilizations = [
        float(repeat.resources["cpu_s"]) / float(repeat.resources["wall_s"])
        for response in responses
        for repeat in response.repeats
        if float(repeat.resources.get("wall_s", 0)) > 0
    ]
    gpu_series: list[Series] = []
    for response in responses:
        for repeat in response.repeats:
            raw = repeat.resources.get("gpu_util")
            if raw:
                gpu_series.append(Series.from_dict(raw))
    gpu = pool_series(gpu_series, idle_threshold=1.0)
    complete = (
        not errors
        and bool(worker_blocks)
        and all(response.complete for response in responses)
        and all(block.get("representative_rtf") is not None for block in worker_blocks)
    )
    execution_ids = {
        str(response.model_provenance.get("execution_sha256"))
        for response in responses
        if isinstance(response.model_provenance, dict)
        and response.model_provenance.get("execution_sha256")
    }
    if len(execution_ids) > 1:
        errors.append("model execution provenance changed across workers")
        complete = False
    if not execution_ids and experiment_id is not None:
        provenance_issues.append("no execution provenance identity")
    if provenance_issues and experiment_id is not None:
        complete = False
    execution_identity = next(iter(execution_ids), "")
    stats_seed = _seed_for(
        STATISTICS_VERSION,
        experiment_id or "",
        execution_identity,
        subject,
        "rtf",
        "ci",
    )
    rtf_ci = _bootstrap_median(block_rtfs, stats_seed) if complete else None
    natural_variation = (
        _natural_variation_ratio(
            block_rtfs,
            _seed_for(
                STATISTICS_VERSION,
                experiment_id or "",
                execution_identity,
                subject,
                "rtf",
                "natural-variation",
            ),
        )
        if complete
        else None
    )

    return {
        "subject": subject,
        "model": responses[0].resolved_model if responses else None,
        "hashes": hashes,
        "transcript_hashes": hashes,
        "text_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if hashes
        else None,
        "rtf": statistics.median(block_rtfs) if complete and block_rtfs else None,
        "rtf_p25": _percentile(block_rtfs, 0.25) if complete else None,
        "rtf_p75": _percentile(block_rtfs, 0.75) if complete else None,
        "rtf_ci95": list(rtf_ci) if rtf_ci else None,
        "natural_variation_ratio_99": natural_variation,
        "worker_blocks": worker_blocks,
        "workers": len(worker_blocks),
        "repeats": sum(len(response.repeats) for response in responses),
        "endpoint_rss_mb": max(endpoint_rss) if endpoint_rss else None,
        "worker_lifetime_peak_rss_mb": max(lifetime_rss) if lifetime_rss else None,
        "peak_rss_mb": max(lifetime_rss) if lifetime_rss else None,
        "memory_metric": "worker-lifetime-ru-maxrss-diagnostic",
        "cpu_utilization": (statistics.median(cpu_utilizations) if cpu_utilizations else None),
        "gpu_util_mean": round(gpu.mean, 1) if gpu else None,
        "gpu_util_p50": round(gpu.p50, 1) if gpu else None,
        "gpu_util_n": gpu.n if gpu else None,
        "gpu_idle_pct": round(gpu.idle_pct, 1) if gpu else None,
        "gpu_scope": gpu.scope if gpu else None,
        "provenance_issues": provenance_issues,
        "execution_sha256": next(iter(execution_ids), None),
        "baseline_eligible": (
            complete
            and experiment_id is not None
            and not provenance_issues
            and len(execution_ids) == 1
        ),
        "error": errors[0] if errors else None,
    }


def compare(baseline: dict[str, Any], fresh: dict[str, Any]) -> Comparison:
    """Judge a fresh measurement against a stored v2 baseline.

    Legacy schema-v1 data is retained for transcript diagnostics only. v2 timing
    gates require identical execution provenance and independent worker blocks.
    """
    if (
        baseline.get("artifact_kind") == "baseline-v2"
        and fresh.get("artifact_kind") == "measurement-v3"
    ):
        return _compare_v2(baseline, fresh)
    return _compare_legacy(baseline, fresh)


def transcript_changes(
    baseline: dict[str, Any], fresh: dict[str, Any]
) -> list[dict[str, str | None]]:
    """Return the exact canonical-audio transcript differences between artifacts."""
    old_runs = {str(item.get("subject")): item for item in baseline.get("runs", [])}
    new_runs = {str(item.get("subject")): item for item in fresh.get("runs", [])}
    changes: list[dict[str, str | None]] = []
    for subject in sorted(set(old_runs) | set(new_runs)):
        before = old_runs.get(subject, {}).get("hashes", {})
        after = new_runs.get(subject, {}).get("hashes", {})
        for audio_id in sorted(set(before) | set(after)):
            if before.get(audio_id) != after.get(audio_id):
                changes.append(
                    {
                        "subject": subject,
                        "audio_id": audio_id,
                        "baseline_sha256": before.get(audio_id),
                        "fresh_sha256": after.get(audio_id),
                    }
                )
    return changes


def _compare_legacy(baseline: dict[str, Any], fresh: dict[str, Any]) -> Comparison:
    """Compatibility comparator for the old lossy baseline format."""
    result = Comparison()
    result.compared_timings = False
    result.host_note = (
        "schema-v1 baseline is transcript-diagnostic only; timing gates are disabled."
    )

    old_host, new_host = baseline.get("host", {}), fresh.get("host", {})
    if old_host != new_host:
        result.host_note = (
            f"baseline recorded on {old_host.get('chip', '?')} / "
            f"torch {old_host.get('torch', '?')}; this is "
            f"{new_host.get('chip', '?')} / torch {new_host.get('torch', '?')}. "
            "Comparing transcripts only."
        )

    old_schema = baseline.get("schema_version", 1)
    new_schema = fresh.get("schema_version", 1)
    if old_schema != new_schema:
        schema_note = (
            f"measurement schema changed from {old_schema} to {new_schema}; "
            "comparing transcripts only."
        )
        result.host_note = f"{result.host_note} {schema_note}" if result.host_note else schema_note

    old_batch = baseline.get("batch_size", 1)
    new_batch = fresh.get("batch_size", 1)
    if old_batch != new_batch:
        batch_note = (
            f"batch size changed from {old_batch} to {new_batch}; comparing transcripts only."
        )
        result.host_note = f"{result.host_note} {batch_note}" if result.host_note else batch_note

    if baseline.get("clips") != fresh.get("clips"):
        result.findings.append(
            Finding("clips", "set", "the measured clips differ from the baseline")
        )
        return result

    old_runs = {r["subject"]: r for r in baseline.get("runs", [])}
    new_runs = {r["subject"]: r for r in fresh.get("runs", [])}

    for subject in sorted(set(old_runs) | set(new_runs)):
        before, after = old_runs.get(subject), new_runs.get(subject)
        if before is None:
            result.findings.append(Finding(subject, "new", "not in the baseline", fatal=False))
            continue
        if after is None:
            result.findings.append(Finding(subject, "missing", "in the baseline but not measured"))
            continue
        if after.get("error"):
            result.findings.append(Finding(subject, "error", after["error"]))
            continue

        old_hashes = before.get("hashes", {})
        new_hashes = after.get("hashes", {})
        changed = [
            stem
            for stem in sorted(set(old_hashes) | set(new_hashes))
            if old_hashes.get(stem) != new_hashes.get(stem)
        ]
        if changed:
            result.findings.append(
                Finding(
                    subject,
                    "text",
                    f"{len(changed)}/{len(set(old_hashes) | set(new_hashes))} transcripts changed "
                    f"({', '.join(changed[:2])}{'…' if len(changed) > 2 else ''})",
                )
            )

    return result


def _bootstrap_ratio(
    before: list[float], after: list[float], seed: int
) -> tuple[float, float, float] | None:
    if not before or not after or any(value <= 0 for value in (*before, *after)):
        return None
    rng = np.random.default_rng(seed)
    old = np.log(np.asarray(sorted(before), dtype=float))
    new = np.log(np.asarray(sorted(after), dtype=float))
    old_samples = rng.choice(old, size=(BOOTSTRAP_DRAWS, len(old)))
    new_samples = rng.choice(new, size=(BOOTSTRAP_DRAWS, len(new)))
    deltas = np.median(new_samples, axis=1) - np.median(old_samples, axis=1)
    point = math.exp(float(np.median(new) - np.median(old)))
    low = math.exp(float(np.quantile(deltas, 0.025, method="linear")))
    high = math.exp(float(np.quantile(deltas, 0.975, method="linear")))
    return point, low, high


def bootstrap_paired_ratio(
    before: Mapping[str, Mapping[int, float]],
    after: Mapping[str, Mapping[int, float]],
    seed: int,
) -> tuple[float, float, float] | None:
    """Return a deterministic paired block-bootstrap ratio interval.

    Historical and fresh workers must use :func:`_bootstrap_ratio`; this helper
    is reserved for conditions measured within one counterbalanced experiment.
    """
    if not before or set(before) != set(after):
        return None
    session_deltas: list[float] = []
    for session_id in sorted(before):
        old_repeats = before[session_id]
        new_repeats = after[session_id]
        if not old_repeats or set(old_repeats) != set(new_repeats):
            return None
        old = np.asarray([old_repeats[index] for index in sorted(old_repeats)], dtype=float)
        new = np.asarray([new_repeats[index] for index in sorted(new_repeats)], dtype=float)
        if (
            np.any(~np.isfinite(old))
            or np.any(~np.isfinite(new))
            or np.any(old <= 0)
            or np.any(new <= 0)
        ):
            return None
        session_deltas.append(float(np.median(np.log(new) - np.log(old))))
    rng = np.random.default_rng(seed)
    deltas = np.asarray(sorted(session_deltas), dtype=float)
    indices = rng.integers(0, len(deltas), size=(BOOTSTRAP_DRAWS, len(deltas)))
    medians = np.median(deltas[indices], axis=1)
    point = math.exp(float(np.median(deltas)))
    low = math.exp(float(np.quantile(medians, 0.025, method="linear")))
    high = math.exp(float(np.quantile(medians, 0.975, method="linear")))
    if not all(math.isfinite(value) for value in (point, low, high)):
        return None
    return point, low, high


def _compare_v2(baseline: dict[str, Any], fresh: dict[str, Any]) -> Comparison:
    result = Comparison()
    if baseline.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
        result.findings.append(Finding("baseline", "schema", "baseline-v2 schema is unsupported"))
        return result
    if fresh.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        result.findings.append(Finding("fresh", "schema", "measurement-v3 schema is unsupported"))
        return result
    if baseline.get("clips") != fresh.get("clips"):
        result.findings.append(Finding("clips", "set", "the measured corpus differs"))
        return result
    if baseline.get("corpus") != fresh.get("corpus"):
        result.findings.append(Finding("corpus", "identity", "audio/reference identity differs"))
        return result

    if baseline.get("protocol") != fresh.get("protocol"):
        result.compared_timings = False
        result.host_note = "measurement protocol changed; comparing transcripts only."
    if _timing_host_identity(baseline.get("host", {})) != _timing_host_identity(
        fresh.get("host", {})
    ):
        result.compared_timings = False
        result.host_note = "host/runtime identity changed; comparing transcripts only."
    expected_statistics = {"version": STATISTICS_VERSION, "draws": BOOTSTRAP_DRAWS}
    if baseline.get("statistics") != expected_statistics:
        result.findings.append(
            Finding("baseline", "statistics", "baseline statistics contract is unsupported")
        )
        result.compared_timings = False

    old_runs = {item.get("subject"): item for item in baseline.get("runs", [])}
    new_runs = {item.get("subject"): item for item in fresh.get("runs", [])}
    for subject in sorted(set(old_runs) | set(new_runs)):
        before, after = old_runs.get(subject), new_runs.get(subject)
        if before is None:
            result.findings.append(Finding(subject, "new", "not in the baseline", fatal=False))
            continue
        if after is None:
            result.findings.append(Finding(subject, "missing", "not measured in the fresh run"))
            continue
        old_hashes = before.get("hashes", {})
        new_hashes = after.get("hashes", {})
        changed = [
            key
            for key in sorted(set(old_hashes) | set(new_hashes))
            if old_hashes.get(key) != new_hashes.get(key)
        ]
        if changed:
            result.findings.append(
                Finding(subject, "text", f"{len(changed)} transcript hash(es) changed")
            )
            continue
        if after.get("error"):
            result.findings.append(Finding(subject, "error", after["error"]))
            continue
        if not before.get("baseline_eligible", False):
            result.findings.append(
                Finding(subject, "baseline", "stored subject is not baseline-eligible")
            )
            continue
        if not before.get("execution_sha256") or not after.get("execution_sha256"):
            result.findings.append(
                Finding(subject, "provenance", "model execution identity is missing")
            )
            continue
        if before.get("execution_sha256") != after.get("execution_sha256"):
            result.findings.append(
                Finding(subject, "provenance", "model/runtime execution identity changed")
            )
            continue
        if not after.get("baseline_eligible", False):
            result.findings.append(
                Finding(
                    subject,
                    "protocol",
                    "fresh run is transcript-only; trusted timing comparison is suppressed",
                    fatal=False,
                )
            )
            continue
        if not result.compared_timings:
            continue
        old_blocks = [
            float(item["representative_rtf"])
            for item in before.get("worker_blocks", [])
            if item.get("representative_rtf") is not None
        ]
        new_blocks = [
            float(item["representative_rtf"])
            for item in after.get("worker_blocks", [])
            if item.get("representative_rtf") is not None
        ]
        ratio = _bootstrap_ratio(
            old_blocks,
            new_blocks,
            _seed_for(
                STATISTICS_VERSION,
                baseline.get("baseline_id", ""),
                fresh.get("experiment_id", fresh.get("run_id", "")),
                subject,
                before.get("execution_sha256", ""),
                after.get("execution_sha256", ""),
                "rtf",
                "compare",
            ),
        )
        threshold = float(before.get("natural_variation_ratio_99") or 1.0)
        if before.get("natural_variation_ratio_99") is None:
            result.findings.append(
                Finding(subject, "statistics", "baseline has no stored natural-variation limit")
            )
            continue
        if ratio is None:
            result.findings.append(Finding(subject, "rtf", "worker blocks are missing"))
            continue
        point, low, high = ratio
        detail = f"{point:.3f}x ratio, 95% CI [{low:.3f}, {high:.3f}], noise limit {threshold:.3f}x"
        if low > threshold:
            result.findings.append(Finding(subject, "rtf", detail))
        elif point > threshold:
            result.findings.append(Finding(subject, "rtf", detail, fatal=False))

    return result


def load_baseline(path: Path = BASELINE) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _canonical_json_bytes(data: dict[str, Any]) -> bytes:
    """Return the exact JSON byte representation used by artifact writers."""
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _baseline_summary_projection(baseline: dict[str, Any]) -> dict[str, Any]:
    """Recover the measurement summary written inside an accepted baseline."""
    summary = dict(baseline)
    summary["artifact_kind"] = "measurement-v3"
    for field_name in (
        "baseline_id",
        "baseline_schema_version",
        "statistics",
        "artifact_manifest",
        "artifact_root",
    ):
        summary.pop(field_name, None)
    return summary


def _recover_baseline_transactions(path: Path, archive_parent: Path) -> None:
    """Remove archives left by a crashed publication, preserving committed ones."""
    try:
        committed = load_baseline(path)
    except (OSError, ValueError, json.JSONDecodeError):
        committed = None
    committed_id = committed.get("baseline_id") if isinstance(committed, dict) else None
    for marker in archive_parent.glob(".*.transaction.json"):
        try:
            transaction = json.loads(marker.read_text(encoding="utf-8"))
            baseline_id = str(transaction["baseline_id"])
            if not baseline_id or Path(baseline_id).name != baseline_id:
                continue
            archive_root = archive_parent / baseline_id
            staging = Path(str(transaction.get("staging", "")))
            candidate = Path(str(transaction.get("candidate", "")))
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if committed_id == baseline_id:
            marker.unlink(missing_ok=True)
            continue
        shutil.rmtree(archive_root, ignore_errors=True)
        if staging != archive_parent and staging.parent.resolve() == archive_parent.resolve():
            shutil.rmtree(staging, ignore_errors=True)
        if candidate.is_file() and candidate.parent.resolve() == path.parent.resolve():
            candidate.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)


def save_baseline(data: dict[str, Any], path: Path = BASELINE) -> None:
    """Archive raw artifacts and publish a v2 baseline atomically."""
    data = deepcopy(data)
    if data.get("artifact_kind") != "measurement-v3":
        raise ValueError("only measurement-v3 artifacts can become a baseline")
    if data.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError("measurement-v3 schema is unsupported")
    if data.get("host", {}).get("git_dirty") is not False:
        raise ValueError("cannot archive a baseline from a dirty source worktree")
    protocol = data.get("protocol", {})
    if (
        int(protocol.get("workers", 0)) < 5
        or int(protocol.get("warmups", 0)) < 3
        or int(protocol.get("repeats", 0)) < 3
        or bool(protocol.get("profile", False))
    ):
        raise ValueError("measurement does not satisfy the trusted performance protocol")
    expected_subjects = {f"{backend}/{model}" if model else backend for backend, model in SUBJECTS}
    actual_subjects = {str(run.get("subject")) for run in data.get("runs", [])}
    if actual_subjects != expected_subjects or len(data.get("runs", [])) != len(expected_subjects):
        raise ValueError("measurement does not contain the complete benchmark subject set")
    run_subject_order = [str(run.get("subject", "")) for run in data.get("runs", [])]
    declared_subject_order = data.get("subject_order", run_subject_order)
    if (
        not isinstance(declared_subject_order, list)
        or any(not isinstance(subject, str) or not subject for subject in declared_subject_order)
        or len(set(declared_subject_order)) != len(declared_subject_order)
        or declared_subject_order != run_subject_order
    ):
        raise ValueError("measurement subject order does not match its benchmark runs")
    data["subject_order"] = list(declared_subject_order)
    existing = load_baseline(path)
    if existing is not None:
        if existing.get("artifact_kind") != "baseline-v2":
            raise ValueError("cannot replace an unverifiable legacy baseline")
        existing_issues = verify_baseline(path)
        if existing_issues:
            raise ValueError("cannot replace an invalid baseline: " + "; ".join(existing_issues))
        changes = transcript_changes(existing, data)
        if changes:
            approval = data.get("transcript_approval")
            if not isinstance(approval, dict) or not str(approval.get("note", "")).strip():
                raise ValueError("transcript changes require a nonempty approval note")
            summary_entries = [
                item
                for item in existing.get("artifact_manifest", [])
                if isinstance(item, dict) and item.get("kind") == "summary"
            ]
            if len(summary_entries) != 1:
                raise ValueError("superseded baseline has no unique summary artifact")
            summary_entry = summary_entries[0]
            expected_link = {
                "path": summary_entry.get("path"),
                "sha256": summary_entry.get("sha256"),
            }
            if (
                approval.get("changed_transcripts") != changes
                or approval.get("subjects") != sorted({str(item["subject"]) for item in changes})
                or approval.get("supersedes_baseline_id") != existing.get("baseline_id")
                or approval.get("superseded_summary") != expected_link
            ):
                raise ValueError("transcript approval does not match the superseded baseline")
    ineligible = [
        str(run.get("subject", "?"))
        for run in data.get("runs", [])
        if not run.get("baseline_eligible", False)
    ]
    if ineligible:
        raise ValueError("cannot archive an ineligible measurement: " + ", ".join(ineligible))
    baseline_id = str(data.get("run_id") or _seed_for(time.time_ns()))
    if not baseline_id or Path(baseline_id).name != baseline_id:
        raise ValueError("baseline run_id must be a simple artifact directory name")
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_parent = path.parent / "artifacts"
    archive_parent.mkdir(parents=True, exist_ok=True)
    _recover_baseline_transactions(path, archive_parent)
    archive_root = archive_parent / baseline_id
    if archive_root.exists():
        raise FileExistsError(f"baseline artifact directory already exists: {archive_root}")
    staging = Path(tempfile.mkdtemp(prefix=f".{baseline_id}.", dir=archive_parent))
    candidate_path = path.with_name(f".{path.name}.{baseline_id}.candidate")
    transaction_path = archive_parent / f".{baseline_id}.transaction.json"
    transaction = {
        "baseline_id": baseline_id,
        "archive_root": str(archive_root),
        "staging": str(staging),
        "candidate": str(candidate_path),
    }
    manifest: list[dict[str, Any]] = []
    manifest_prefix = Path(path.parent.name) if path.parent.name else Path()
    try:
        write_json_atomic(transaction, transaction_path)
        sources: list[tuple[Path, str | None]] = []
        run_sources: dict[int, list[Path]] = {}
        summary_source = Path(str(data.get("raw_artifact_dir", ""))) / "summary.json"
        if not summary_source.is_file():
            raise FileNotFoundError(f"derived summary is missing: {summary_source}")
        sources.append((summary_source, None))
        for run_index, run in enumerate(data.get("runs", [])):
            run_sources[run_index] = []
            for raw in run.get("raw_artifacts", []):
                source = Path(raw)
                if not source.is_file():
                    raise FileNotFoundError(f"raw artifact is missing: {source}")
                sources.append((source, str(run.get("subject"))))
                run_sources[run_index].append(source)
        seen_sources: set[Path] = set()
        archived_by_source: dict[Path, str] = {}
        context_by_stem: dict[str, dict[str, Any]] = {}
        for source, subject in sources:
            if not source.name.endswith(".request.json"):
                continue
            try:
                context = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            stem = source.name.removesuffix(".request.json")
            context_by_stem[stem] = {
                "subject": subject,
                "session_id": context.get("session_id"),
                "session_index": context.get("session_index"),
            }
        for source, subject in sources:
            source = source.resolve()
            if source in seen_sources:
                continue
            seen_sources.add(source)
            destination = staging / source.name
            if destination.exists():
                raise FileExistsError(f"duplicate raw artifact name: {source.name}")
            shutil.copy2(source, destination)
            relative = str(manifest_prefix / "artifacts" / baseline_id / source.name)
            archived_by_source[source] = relative
            source_name = source.name
            if source_name == "summary.json":
                kind = "summary"
            else:
                kind = next(
                    (
                        label
                        for label in ("request", "journal", "response", "stdout", "stderr")
                        if f".{label}." in source_name
                    ),
                    source.suffix.removeprefix("."),
                )
                if kind == "json" and source_name.endswith("response.json"):
                    kind = "response"
                elif kind == "json" and source_name.endswith("request.json"):
                    kind = "request"
                elif kind == "json" and source_name.endswith("journal.json"):
                    kind = "journal"
            context_subject = subject
            session_id = None
            session_index = None
            artifact_stem = next(
                (
                    source_name.split(f".{label}.", 1)[0]
                    for label in ("request", "journal", "response", "stdout", "stderr")
                    if f".{label}." in source_name
                ),
                "",
            )
            if artifact_stem in context_by_stem:
                context = context_by_stem[artifact_stem]
                context_subject = context_subject or context.get("subject")
                session_id = context.get("session_id")
                session_index = context.get("session_index")
            if kind in {"request", "journal", "response"}:
                try:
                    raw_context = json.loads(source.read_text(encoding="utf-8"))
                    context_subject = context_subject or raw_context.get("subject", {}).get(
                        "backend"
                    )
                    session_id = raw_context.get("session_id")
                    session_index = raw_context.get("session_index")
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            manifest.append(
                {
                    "path": relative,
                    "kind": kind,
                    "subject": context_subject,
                    "session_id": session_id,
                    "session_index": session_index,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "size_bytes": destination.stat().st_size,
                }
            )
        for run_index, run in enumerate(data.get("runs", [])):
            run["raw_artifacts"] = [
                archived_by_source[source.resolve()] for source in run_sources[run_index]
            ]
        if summary_source.resolve() in archived_by_source:
            data["summary_artifact"] = archived_by_source[summary_source.resolve()]
            summary_destination = staging / summary_source.name
            write_json_atomic(data, summary_destination)
            for item in manifest:
                if item.get("kind") == "summary":
                    item["sha256"] = hashlib.sha256(summary_destination.read_bytes()).hexdigest()
                    item["size_bytes"] = summary_destination.stat().st_size
                    break
        os.replace(staging, archive_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(archive_root, ignore_errors=True)
        transaction_path.unlink(missing_ok=True)
        raise

    baseline = {
        **data,
        "artifact_kind": "baseline-v2",
        "baseline_id": baseline_id,
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "statistics": {
            "version": STATISTICS_VERSION,
            "draws": BOOTSTRAP_DRAWS,
        },
        "artifact_manifest": manifest,
        "artifact_root": str(manifest_prefix / "artifacts" / baseline_id),
    }
    published = False
    try:
        write_json_atomic(baseline, candidate_path)
        verification_issues = verify_baseline(candidate_path)
        if verification_issues:
            raise ValueError(
                "candidate baseline failed raw-artifact verification: "
                + "; ".join(verification_issues)
            )
        write_json_atomic(baseline, path)
        published = True
    finally:
        candidate_path.unlink(missing_ok=True)
        if published:
            transaction_path.unlink(missing_ok=True)
        else:
            shutil.rmtree(archive_root, ignore_errors=True)
            transaction_path.unlink(missing_ok=True)


def verify_baseline(path: Path = BASELINE) -> list[str]:
    """Verify raw artifact checksums and recompute every stored subject summary."""
    if not path.is_file():
        return [f"baseline does not exist: {path}"]
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"cannot read baseline: {exc}"]
    if baseline.get("artifact_kind") != "baseline-v2":
        return ["baseline is legacy schema-v1 and has no verifiable raw artifact manifest"]
    if baseline.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
        return ["baseline-v2 schema version is unsupported"]
    issues: list[str] = []
    expected_statistics = {"version": STATISTICS_VERSION, "draws": BOOTSTRAP_DRAWS}
    if baseline.get("statistics") != expected_statistics:
        issues.append("baseline statistics contract is unsupported")
    manifest_items = baseline.get("artifact_manifest")
    if not isinstance(manifest_items, list) or not manifest_items:
        return ["baseline has no raw artifact manifest"]
    # Manifest paths are rooted at the directory containing the baseline's
    # parent directory (for example, repository root for baselines/bench.json).
    # This remains correct for a temporary custom path such as tmp/bench.json.
    root = path.parent.parent
    for item in manifest_items:
        if not isinstance(item, dict) or not item.get("path"):
            issues.append("invalid raw artifact manifest entry")
            continue
        relative_path = Path(str(item.get("path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            issues.append(f"invalid artifact path: {relative_path}")
            continue
        artifact = (root / relative_path).resolve()
        try:
            artifact.relative_to(root.resolve())
        except ValueError:
            issues.append(f"artifact path escapes repository root: {relative_path}")
            continue
        if not artifact.is_file():
            issues.append(f"missing raw artifact: {artifact}")
            continue
        digest_builder = hashlib.sha256()
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest_builder.update(block)
        digest = digest_builder.hexdigest()
        if digest != item.get("sha256") or artifact.stat().st_size != item.get("size_bytes"):
            issues.append(f"raw artifact checksum mismatch: {artifact}")
    summary_entries = [
        item for item in manifest_items if isinstance(item, dict) and item.get("kind") == "summary"
    ]
    if not summary_entries:
        issues.append("baseline manifest has no derived summary artifact")
    elif len(summary_entries) != 1:
        issues.append("baseline manifest must contain exactly one derived summary artifact")
    approval = baseline.get("transcript_approval")
    if approval is not None:
        if not isinstance(approval, dict) or not str(approval.get("note", "")).strip():
            issues.append("transcript approval is missing its rationale")
        else:
            prior = approval.get("superseded_summary")
            prior_id = str(approval.get("supersedes_baseline_id", ""))
            if not isinstance(prior, dict) or not prior_id:
                issues.append("transcript approval is missing its superseded baseline link")
            else:
                prior_relative = Path(str(prior.get("path", "")))
                if (
                    prior_relative.is_absolute()
                    or ".." in prior_relative.parts
                    or prior_id not in prior_relative.parts
                ):
                    issues.append("transcript approval has an invalid superseded summary path")
                else:
                    prior_path = (root / prior_relative).resolve()
                    try:
                        prior_path.relative_to(root.resolve())
                        prior_bytes = prior_path.read_bytes()
                        if hashlib.sha256(prior_bytes).hexdigest() != prior.get("sha256"):
                            issues.append("superseded summary checksum mismatch")
                        else:
                            prior_summary = json.loads(prior_bytes)
                            changes = transcript_changes(prior_summary, baseline)
                            if approval.get("changed_transcripts") != changes:
                                issues.append(
                                    "transcript approval differs from the superseded summary"
                                )
                            if approval.get("subjects") != sorted(
                                {str(item["subject"]) for item in changes}
                            ):
                                issues.append("transcript approval subject list is inconsistent")
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        issues.append(f"cannot verify superseded summary: {exc}")
    if len(summary_entries) == 1:
        summary_item = summary_entries[0]
        summary_relative = Path(str(summary_item.get("path", "")))
        if summary_relative.is_absolute() or ".." in summary_relative.parts:
            issues.append(f"invalid derived summary path: {summary_relative}")
        else:
            summary_path = (root / summary_relative).resolve()
            try:
                summary_path.relative_to(root.resolve())
                summary_bytes = summary_path.read_bytes()
                summary = json.loads(summary_bytes)
                if summary.get("artifact_kind") != "measurement-v3":
                    issues.append(f"derived summary is not measurement-v3: {summary_path}")
                expected_summary = _baseline_summary_projection(baseline)
                expected_bytes = _canonical_json_bytes(expected_summary)
                if summary_bytes != expected_bytes:
                    issues.append(
                        f"derived summary is not the canonical committed summary: {summary_path}"
                    )
                elif summary != expected_summary:
                    issues.append(f"derived summary does not match baseline data: {summary_path}")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read derived summary {summary_path}: {exc}")
    corpus = baseline.get("corpus", [])
    inputs = [
        AudioInput(
            source_path="",
            prepared_path="",
            reference_id=str(item["reference_id"]),
            source_sha256="0" * 64,
            audio_id=str(item["audio_id"]),
            duration_s=float(item["duration_s"]),
            sample_rate=16_000,
            channels=1,
            frames=max(1, int(round(float(item["duration_s"]) * 16_000))),
        )
        for item in corpus
    ]
    if not inputs:
        issues.append("baseline has no canonical corpus identity")
        return issues
    manifest_root = root
    manifest_kinds = {
        str(item.get("path")): str(item.get("kind"))
        for item in manifest_items
        if isinstance(item, dict)
    }
    manifest_paths = set(manifest_kinds)
    protocol = baseline.get("protocol", {})
    expected_workers = int(protocol.get("workers", 0)) or None
    expected_warmups = int(protocol.get("warmups", 0))
    expected_repeats = int(protocol.get("repeats", 0)) or None
    expected_profile = bool(protocol.get("profile", False))
    expected_experiment = str(baseline.get("experiment_id", baseline.get("run_id", "")))
    schedule = baseline.get("session_schedule", [])
    run_subjects = [str(run.get("subject", "")) for run in baseline.get("runs", [])]
    declared_order = baseline.get("subject_order", run_subjects)
    if (
        not isinstance(declared_order, list)
        or any(not isinstance(subject, str) or not subject for subject in declared_order)
        or len(set(declared_order)) != len(declared_order)
        or set(declared_order) != set(run_subjects)
        or len(run_subjects) != len(set(run_subjects))
    ):
        issues.append("baseline subject order does not match its declared runs")
        subject_labels = run_subjects
    else:
        subject_labels = list(declared_order)
    try:
        normalized_schedule = [
            [(str(subject), int(position)) for subject, position in session] for session in schedule
        ]
        expected_schedule = counterbalanced_schedule(
            subject_labels,
            int(expected_workers or 0),
            int(baseline.get("schedule_seed")),
        )
        if normalized_schedule != expected_schedule:
            issues.append("declared session schedule does not match its seed and subject set")
    except (TypeError, ValueError):
        normalized_schedule = []
        issues.append("declared session schedule is invalid")
    top_level_host = baseline.get("host", {})
    for run in baseline.get("runs", []):
        subject = str(run.get("subject"))
        requests: dict[str, WorkerRequest] = {}
        responses: dict[str, WorkerResponse] = {}
        journals: dict[str, WorkerJournal] = {}
        raw_paths = [str(raw) for raw in run.get("raw_artifacts", [])]
        for raw in raw_paths:
            if raw not in manifest_paths:
                issues.append(f"run {run.get('subject')} references unmanifested artifact: {raw}")
        run_kinds = {manifest_kinds.get(raw) for raw in raw_paths}
        missing_kinds = {"request", "journal", "response", "stdout", "stderr"} - run_kinds
        if missing_kinds:
            issues.append(
                f"run {run.get('subject')} is missing raw artifact kind(s): "
                + ", ".join(sorted(missing_kinds))
            )
        if expected_workers is not None:
            for expected_kind in ("request", "journal", "response", "stdout", "stderr"):
                found_count = sum(manifest_kinds.get(raw) == expected_kind for raw in raw_paths)
                if found_count != expected_workers:
                    issues.append(
                        f"run {subject} has {found_count} {expected_kind} artifact(s); "
                        f"expected {expected_workers}"
                    )
        for raw in run.get("raw_artifacts", []):
            kind = manifest_kinds.get(str(raw))
            artifact_path = manifest_root / str(raw)
            try:
                if kind == "request":
                    value = read_request(artifact_path)
                    digest = value.identity_sha256
                    target: dict[str, Any] = requests
                elif kind == "response":
                    value = read_response(artifact_path)
                    digest = value.request_sha256
                    target = responses
                elif kind == "journal":
                    value = read_journal(artifact_path)
                    digest = value.request_sha256
                    target = journals
                else:
                    continue
            except (OSError, ValueError, MeasurementError) as exc:
                issues.append(f"cannot read {kind} {artifact_path}: {exc}")
                continue
            if digest in target:
                issues.append(f"duplicate {kind} identity for {subject}: {digest}")
            target[digest] = value
        if not responses:
            issues.append(f"subject {subject} has no response artifacts")
            continue
        identity_sets = (set(requests), set(responses), set(journals))
        if not (identity_sets[0] == identity_sets[1] == identity_sets[2]):
            issues.append(f"request/journal/response identity coverage differs for {subject}")
        for request_sha256 in sorted(set.union(*identity_sets)):
            request = requests.get(request_sha256)
            response = responses.get(request_sha256)
            journal = journals.get(request_sha256)
            if request is None or response is None or journal is None:
                continue
            if not _matches_request(response, request):
                issues.append(f"request/response identity mismatch for {subject}")
            if not _matches_request(journal, request):
                issues.append(f"request/journal identity mismatch for {subject}")
            if not _journal_matches_response(journal, response):
                issues.append(f"journal/response state mismatch for {subject}")
            if any(top_level_host.get(key) != value for key, value in response.host.items()):
                issues.append(f"worker host facts differ from the top-level host for {subject}")
            for key in (
                "python",
                "platform",
                "machine",
                "packages",
                "git_commit",
                "git_dirty",
                "uv_lock_sha256",
            ):
                if top_level_host.get(key) != response.environment.get(key):
                    issues.append(
                        f"worker environment differs from the top-level host for {subject}: {key}"
                    )
            request_subject = (
                f"{request.subject.backend}/{request.subject.model}"
                if request.subject.model
                else request.subject.backend
            )
            if request_subject != subject:
                issues.append(f"archived request subject mismatch for {subject}")
            requested_corpus = [
                {
                    "audio_id": item.audio_id,
                    "reference_id": item.reference_id,
                    "duration_s": item.duration_s,
                }
                for item in request.inputs
            ]
            if requested_corpus != corpus:
                issues.append(f"archived request corpus mismatch for {subject}")
            if (
                request.warmups != expected_warmups
                or request.repeats != expected_repeats
                or request.profile != expected_profile
                or request.experiment_id != expected_experiment
                or request.schedule_seed != baseline.get("schedule_seed")
            ):
                issues.append(f"archived request protocol mismatch for {subject}")
            expected_session_id = (
                f"{expected_experiment}:session-{request.session_index}"
                if request.session_index is not None
                else None
            )
            expected_position = (
                next(
                    (
                        position
                        for scheduled_subject, position in normalized_schedule[
                            request.session_index or 0
                        ]
                        if scheduled_subject == subject
                    ),
                    None,
                )
                if normalized_schedule
                and request.session_index is not None
                and 0 <= request.session_index < len(normalized_schedule)
                else None
            )
            if (
                request.session_id != expected_session_id
                or request.worker_index != request.session_index
                or request.launch_position != expected_position
            ):
                issues.append(f"archived request session coordinate mismatch for {subject}")
        expected_positions = [
            int(position)
            for session in schedule
            for scheduled_subject, position in session
            if scheduled_subject == subject
        ]
        expected_coordinates = [
            (session_index, int(position), f"{expected_experiment}:session-{session_index}")
            for session_index, session in enumerate(normalized_schedule)
            for scheduled_subject, position in session
            if scheduled_subject == subject
        ]
        recomputed = summarize_workers(
            subject,
            list(responses.values()),
            inputs,
            expected_workers=expected_workers,
            expected_warmups=expected_warmups,
            expected_repeats=expected_repeats,
            expected_launch_positions=expected_positions,
            expected_session_coordinates=expected_coordinates,
            experiment_id=expected_experiment,
        )
        stored = {key: value for key, value in run.items() if key != "raw_artifacts"}
        if json.dumps(stored, sort_keys=True, separators=(",", ":")) != json.dumps(
            recomputed, sort_keys=True, separators=(",", ":")
        ):
            issues.append(f"derived summary mismatch: {subject}")
    return issues
