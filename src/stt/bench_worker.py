"""Isolated benchmark subject worker.

Run as ``python -m stt.bench_worker REQUEST.json RESPONSE.json``. The response
is atomically published even when model loading or transcription fails.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import soundfile as sf

from stt.audio import PreparedAudio, canonical_audio_id, file_sha256
from stt.execution import create_backend, transcribe_corpus
from stt.measurement import (
    ActivePhase,
    AudioInput,
    PhaseEvent,
    RepeatRecord,
    WorkerJournal,
    WorkerRequest,
    WorkerResponse,
    capture_environment,
    read_request,
    write_journal,
    write_response,
)
from stt.provenance import validate_binding
from stt.telemetry import ResourceUsage, describe_host, measure

BackendFactory = Callable[[str, str | None, dict[str, Any]], Any]


def _phase(name: str, usage: ResourceUsage, *, repeat_index: int | None = None) -> PhaseEvent:
    if usage.started_ns is None or usage.ended_ns is None:
        raise RuntimeError(f"measurement for {name} has no monotonic boundaries")
    return PhaseEvent(name, usage.started_ns, usage.ended_ns, repeat_index=repeat_index)


def _validate_input(item: AudioInput, *, input_mode: str) -> None:
    item.validate()
    resolved = item.prepared()
    source = resolved.source_path
    prepared = resolved.prepared_path
    if file_sha256(source) != item.source_sha256:
        raise RuntimeError(f"source bytes changed before worker start: {item.reference_id}")
    if input_mode == "prepared" and canonical_audio_id(prepared) != item.audio_id:
        raise RuntimeError(f"prepared waveform changed before worker start: {item.reference_id}")
    presented = prepared if input_mode == "prepared" else source
    info = sf.info(presented)
    if input_mode == "prepared":
        actual = (info.duration, info.samplerate, info.channels, info.frames)
        expected = (item.duration_s, item.sample_rate, item.channels, item.frames)
        if actual != expected:
            raise RuntimeError(
                f"prepared audio facts changed before worker start: {item.reference_id}"
            )
    elif info.duration <= 0 or info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
        raise RuntimeError(f"source audio facts are invalid: {item.reference_id}")


def _presented_input(item: AudioInput, *, input_mode: str) -> PreparedAudio:
    prepared = item.prepared()
    if input_mode == "prepared":
        return prepared
    return prepared.__class__(
        source_path=prepared.source_path,
        prepared_path=prepared.source_path,
        reference_id=item.reference_id,
        source_sha256=item.source_sha256,
        audio_id=item.audio_id,
    )


def _runtime_facts(instance: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    device = getattr(instance, "resolved_device", None)
    dtype = getattr(instance, "dtype", None) or getattr(instance, "resolved_dtype", None)
    if dtype is not None:
        dtype = str(dtype)
    revision = getattr(instance, "revision", None)
    pipeline = getattr(instance, "pipeline", None)
    model = getattr(pipeline, "model", None)
    config = getattr(model, "config", None)
    revision = revision or getattr(config, "_commit_hash", None)
    facts = {
        "model": getattr(instance, "model", None),
        "device": device,
        "dtype": dtype,
        "revision": revision,
        "n_threads": getattr(instance, "n_threads", None),
    }
    issues: list[str] = []
    if device is None:
        issues.append("runtime did not report the selected device")
    return facts, tuple(issues)


def execute_request(
    request: WorkerRequest,
    *,
    backend_factory: BackendFactory = create_backend,
    root: Path | None = None,
    checkpoint: Callable[[WorkerJournal], None] | None = None,
) -> WorkerResponse:
    """Execute one subject without allowing model state to cross the process boundary."""
    process_started_ns = time.perf_counter_ns()
    request.validate()
    request_sha256 = request.identity_sha256
    host: dict[str, Any] = {}
    environment: dict[str, Any] = {}
    phases: list[PhaseEvent] = []
    repeats: list[RepeatRecord] = []
    load_resources: dict[str, Any] | None = None
    instance: Any = None
    runtime: dict[str, Any] = {}
    provenance_issues: tuple[str, ...] = ()
    error: str | None = None
    model_provenance: dict[str, Any] | None = (
        request.model_binding.provenance.to_dict() if request.model_binding else None
    )
    active_phase: ActivePhase | None = ActivePhase("process-start", process_started_ns)

    def save_checkpoint(*, complete: bool = False, journal_error: str | None = None) -> None:
        if checkpoint is None:
            return
        checkpoint(
            WorkerJournal(
                run_id=request.run_id,
                request_key=request.subject.request_key,
                request_sha256=request_sha256,
                worker_index=request.worker_index,
                subject=request.subject,
                host=host,
                environment=environment,
                runtime=runtime,
                phases=tuple(phases),
                repeats=tuple(repeats),
                load_resources=load_resources,
                model_provenance=model_provenance,
                provenance_issues=provenance_issues,
                active_phase=active_phase,
                updated_ns=time.perf_counter_ns(),
                complete=complete,
                error=journal_error,
                experiment_id=request.experiment_id,
                session_id=request.session_id,
                session_index=request.session_index,
                launch_position=request.launch_position,
                condition_id=request.condition_id,
                schedule_seed=request.schedule_seed,
            )
        )

    save_checkpoint()
    host = describe_host()
    environment = capture_environment(root)
    save_checkpoint()

    try:
        if request.model_binding is not None:
            binding_issues = validate_binding(request.model_binding)
            if binding_issues:
                raise RuntimeError("model binding validation failed: " + "; ".join(binding_issues))
        for item in request.inputs:
            _validate_input(item, input_mode=request.input_mode)
        prepared = [
            _presented_input(item, input_mode=request.input_mode) for item in request.inputs
        ]
        instance = backend_factory(
            request.subject.backend,
            request.subject.model,
            request.subject.options,
        )
        if request.model_binding is not None:
            bind_model = getattr(instance, "bind_model", None)
            if not callable(bind_model):
                raise RuntimeError("backend cannot accept an immutable model binding")
            bind_model(request.model_binding)
        instance._provenance_environment = environment

        load_started_ns = time.perf_counter_ns()
        phases.append(PhaseEvent("process-start", process_started_ns, load_started_ns))
        active_phase = ActivePhase("load", load_started_ns)
        save_checkpoint()
        load_device = getattr(instance, "resolved_device", None)
        with measure(
            profile=request.profile,
            device=load_device,
            phase="load",
            sample_uss=request.sample_uss,
        ) as measured_load:
            instance.load()
        load_usage = measured_load[0]
        phases.append(_phase("load", load_usage))
        load_resources = load_usage.to_dict()
        active_phase = None
        if request.model_binding is not None:
            validation_started_ns = time.perf_counter_ns()
            active_phase = ActivePhase("load-validation", validation_started_ns)
            save_checkpoint()
            binding_issues = validate_binding(request.model_binding)
            if binding_issues:
                raise RuntimeError(
                    "model binding changed during load: " + "; ".join(binding_issues)
                )
            phases.append(
                PhaseEvent("load-validation", validation_started_ns, time.perf_counter_ns())
            )
            active_phase = None
        runtime, provenance_issues = _runtime_facts(instance)
        runtime["runner_id"] = request.runner_id
        runtime["input_mode"] = request.input_mode
        try:
            model_provenance = instance.model_provenance().finalized().to_dict()
            provenance_issues = (*provenance_issues, *model_provenance.get("issues", []))
        except Exception as exc:  # noqa: BLE001 - provenance remains diagnostic
            provenance_issues = (
                *provenance_issues,
                "model provenance unavailable: backend did not provide an immutable "
                f"model revision or artifact collector ({exc})",
            )
        if environment.get("git_dirty"):
            provenance_issues = (*provenance_issues, "adapter worktree is dirty")
        next_name = "warmup-0" if request.warmups else "repeat-0"
        active_phase = ActivePhase(next_name, time.perf_counter_ns())
        save_checkpoint()

        for index in range(request.warmups):
            active_phase = ActivePhase(f"warmup-{index}", time.perf_counter_ns())
            save_checkpoint()
            warmup_results, warmup_usage = transcribe_corpus(
                instance,
                request.subject.backend,
                prepared,
                request.subject.language,
                request.subject.batch_size,
                profile=False,
                host=host,
                phase=f"warmup-{index}",
                require_provenance=request.experiment_id is not None,
            )
            try:
                model_provenance = instance.model_provenance().finalized().to_dict()
                provenance_issues = tuple(
                    dict.fromkeys((*provenance_issues, *model_provenance.get("issues", [])))
                )
            except Exception as exc:  # noqa: BLE001 - keep the completed phase diagnostic
                provenance_issues = tuple(
                    dict.fromkeys(
                        (*provenance_issues, f"post-warmup provenance unavailable: {exc}")
                    )
                )
            if request.model_binding is not None:
                binding_issues = validate_binding(request.model_binding)
                if binding_issues:
                    raise RuntimeError(
                        f"model binding changed during warmup {index}: " + "; ".join(binding_issues)
                    )
            warmup_error = next((result.error for result in warmup_results if result.error), None)
            event = _phase(f"warmup-{index}", warmup_usage)
            if warmup_error:
                event = PhaseEvent(
                    event.name,
                    event.started_ns,
                    event.ended_ns,
                    status="error",
                    error=warmup_error,
                )
                phases.append(event)
                active_phase = None
                raise RuntimeError(f"warmup {index} failed: {warmup_error}")
            phases.append(event)
            next_name = f"warmup-{index + 1}" if index + 1 < request.warmups else "repeat-0"
            active_phase = ActivePhase(next_name, time.perf_counter_ns())
            save_checkpoint()

        expected_audio_s = sum(item.duration_s for item in request.inputs)
        for index in range(request.repeats):
            active_phase = ActivePhase(f"repeat-{index}", time.perf_counter_ns(), index)
            save_checkpoint()
            results, usage = transcribe_corpus(
                instance,
                request.subject.backend,
                prepared,
                request.subject.language,
                request.subject.batch_size,
                profile=request.profile,
                sample_uss=request.sample_uss,
                host=host,
                phase=f"repeat-{index}",
                require_provenance=request.experiment_id is not None,
            )
            try:
                model_provenance = instance.model_provenance().finalized().to_dict()
                provenance_issues = tuple(
                    dict.fromkeys((*provenance_issues, *model_provenance.get("issues", [])))
                )
            except Exception as exc:  # noqa: BLE001 - preserve the repeat artifact
                provenance_issues = tuple(
                    dict.fromkeys(
                        (*provenance_issues, f"post-repeat provenance unavailable: {exc}")
                    )
                )
            binding_issues: tuple[str, ...] = ()
            if request.model_binding is not None:
                binding_issues = validate_binding(request.model_binding)
                if binding_issues:
                    provenance_issues = tuple(dict.fromkeys((*provenance_issues, *binding_issues)))
                    for result in results:
                        result.trusted = False
                        result.trust_issues = list(
                            dict.fromkeys(
                                (*result.trust_issues, "model binding changed during execution")
                            )
                        )
            issues = tuple(
                dict.fromkeys(issue for result in results for issue in result.trust_issues)
            )
            complete = len(results) == len(request.inputs) and all(
                result.trusted and not result.error for result in results
            )
            event = _phase(f"repeat-{index}", usage, repeat_index=index)
            if not complete:
                detail = issues[0] if issues else "repeat is incomplete"
                event = PhaseEvent(
                    event.name,
                    event.started_ns,
                    event.ended_ns,
                    status="error",
                    error=detail,
                    repeat_index=index,
                )
            phases.append(event)
            repeats.append(
                RepeatRecord(
                    index=index,
                    phase=event,
                    expected_audio_s=expected_audio_s,
                    results=tuple(result.to_dict() for result in results),
                    resources=usage.to_dict(),
                    complete=complete,
                    trust_issues=issues,
                )
            )
            next_name = f"repeat-{index + 1}" if index + 1 < request.repeats else "unload"
            active_phase = ActivePhase(next_name, time.perf_counter_ns())
            save_checkpoint()
    except Exception as exc:  # noqa: BLE001 - worker failures belong in the artifact
        error = f"{type(exc).__name__}: {exc}"
        if active_phase is not None:
            phases.append(
                PhaseEvent(
                    active_phase.name,
                    active_phase.started_ns,
                    max(active_phase.started_ns, time.perf_counter_ns()),
                    status="error",
                    error=error,
                    repeat_index=active_phase.repeat_index,
                )
            )
        active_phase = None
        save_checkpoint(journal_error=error)
    finally:
        if instance is not None:
            try:
                active_phase = ActivePhase("unload", time.perf_counter_ns())
                save_checkpoint()
                with measure(profile=False, phase="unload") as measured_unload:
                    instance.unload()
                phases.append(_phase("unload", measured_unload[0]))
                active_phase = None
                save_checkpoint()
            except Exception as exc:  # noqa: BLE001 - preserve unload failures too
                unload_error = f"{type(exc).__name__}: {exc}"
                error = f"{error}; unload failed: {unload_error}" if error else unload_error
                if active_phase is not None:
                    phases.append(
                        PhaseEvent(
                            active_phase.name,
                            active_phase.started_ns,
                            max(active_phase.started_ns, time.perf_counter_ns()),
                            status="error",
                            error=unload_error,
                        )
                    )
                active_phase = None
                save_checkpoint(journal_error=error)

    complete = (
        error is None
        and len(repeats) == request.repeats
        and all(repeat.complete for repeat in repeats)
    )
    if request.model_binding is not None:
        binding_issues = validate_binding(request.model_binding)
        if binding_issues:
            provenance_issues = tuple(dict.fromkeys((*provenance_issues, *binding_issues)))
            complete = False
    if not phases:
        phases.append(
            PhaseEvent(
                "process-start",
                process_started_ns,
                time.perf_counter_ns(),
                status="error",
                error=error or "worker stopped before loading",
            )
        )
    active_phase = None
    save_checkpoint(complete=complete, journal_error=error)
    return WorkerResponse(
        run_id=request.run_id,
        request_key=request.subject.request_key,
        request_sha256=request_sha256,
        worker_index=request.worker_index,
        subject=request.subject,
        host=host,
        environment=environment,
        runtime=runtime,
        phases=tuple(phases),
        repeats=tuple(repeats),
        load_resources=load_resources,
        model_provenance=model_provenance,
        resolved_model=runtime.get("model"),
        resolved_device=runtime.get("device"),
        resolved_dtype=runtime.get("dtype"),
        complete=complete,
        error=error,
        provenance_issues=provenance_issues,
        experiment_id=request.experiment_id,
        session_id=request.session_id,
        session_index=request.session_index,
        launch_position=request.launch_position,
        condition_id=request.condition_id,
        schedule_seed=request.schedule_seed,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) not in {2, 3}:
        print(
            "usage: python -m stt.bench_worker REQUEST.json RESPONSE.json JOURNAL.json",
            file=sys.stderr,
        )
        return 2
    request_path, response_path = map(Path, arguments[:2])
    journal_path = (
        Path(arguments[2]) if len(arguments) == 3 else response_path.with_suffix(".journal.json")
    )
    try:
        request = read_request(request_path)
        response = execute_request(
            request,
            checkpoint=lambda journal: write_journal(journal, journal_path),
        )
        write_journal(
            WorkerJournal.from_dict(
                {
                    **response.to_dict(),
                    "active_phase": None,
                    "updated_ns": time.perf_counter_ns(),
                    "complete": response.complete,
                }
            ),
            journal_path,
        )
        write_response(response, response_path)
    except Exception as exc:  # noqa: BLE001 - malformed requests may prevent a response
        print(f"benchmark worker failed before response publication: {exc}", file=sys.stderr)
        return 2
    return 0 if response.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
