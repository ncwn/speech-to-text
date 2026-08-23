"""Explicit experiment specifications and fail-closed paired contrasts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import stt.experiment_archive as archive_mod
from stt.experiment import (
    ConditionSpec,
    ContrastSpec,
    ExperimentSpec,
    InputSetSpec,
    StatisticsSpec,
    build_schedule,
    summarize_experiment,
)
from stt.experiment_archive import publish_experiment, verify_experiment
from stt.measurement import (
    AudioInput,
    MeasurementError,
    PhaseEvent,
    RepeatRecord,
    SubjectSpec,
    WorkerJournal,
    WorkerRequest,
    WorkerResponse,
    write_journal,
    write_request,
    write_response,
)
from stt.provenance import ArtifactDigest, ModelBinding, ModelProvenance


def _audio(*, suffix: str = "1", source: str = "2") -> AudioInput:
    return AudioInput(
        source_path=f"data/source/{suffix}.wav",
        prepared_path=f"data/prepared/{suffix}.wav",
        reference_id=f"clip-{suffix}",
        source_sha256=source * 64,
        audio_id=f"pcm16:16000:1:{suffix * 64}",
        duration_s=1.0,
        sample_rate=16_000,
        channels=1,
        frames=16_000,
    )


def _options(device: str) -> dict[str, object]:
    return {"device": device, "language": "mya_Mymr", "batch_size": 1}


def _provenance(subject: SubjectSpec, *, device: str = "cpu") -> ModelProvenance:
    artifact = ArtifactDigest(
        "weights",
        "model.bin",
        7,
        "a" * 64,
        path=str(Path.cwd() / ".cache/test-model.bin"),
    )
    return ModelProvenance(
        backend=subject.backend,
        requested_model=str(subject.model),
        source_kind="test",
        source_locator="test/model",
        upstream_revision="b" * 40,
        revision_status="pinned",
        artifacts=(artifact,),
        runtime_packages={"runtime": "1"},
        requested_settings=dict(subject.options),
        resolved_settings={"device": device, "dtype": "float32"},
        adapter_git_commit="c" * 40,
        adapter_git_dirty=False,
        uv_lock_sha256="d" * 64,
    ).finalized()


def _spec(
    *,
    conditions: tuple[ConditionSpec, ...] | None = None,
    contrasts: tuple[ContrastSpec, ...] | None = None,
    input_sets: tuple[InputSetSpec, ...] | None = None,
    sessions: int = 5,
    warmups: int = 3,
    repeats: int = 3,
) -> ExperimentSpec:
    subject = SubjectSpec("fake", "model", "mya_Mymr", 1, _options("cpu"))
    conditions = conditions or (
        ConditionSpec("off", subject, "canonical"),
        ConditionSpec("profile", subject, "canonical", profile=True),
        ConditionSpec(
            "profile-uss",
            subject,
            "canonical",
            profile=True,
            sample_uss=True,
        ),
    )
    contrasts = contrasts or (
        ContrastSpec("profile-overhead", "off", "profile"),
        ContrastSpec("uss-overhead", "off", "profile-uss"),
    )
    return ExperimentSpec(
        experiment_id="observer",
        input_sets=input_sets or (InputSetSpec("canonical", (_audio(),)),),
        conditions=conditions,
        contrasts=contrasts,
        sessions=sessions,
        warmups=warmups,
        repeats=repeats,
        schedule_seed=17,
    )


def _responses(
    spec: ExperimentSpec,
    ratios: dict[str, float],
    *,
    text: dict[str, str] | None = None,
) -> dict[str, list[WorkerResponse]]:
    schedule = build_schedule(spec)
    found: dict[str, list[WorkerResponse]] = {
        condition.condition_id: [] for condition in spec.conditions
    }
    text = text or {}
    for condition in spec.conditions:
        inputs = spec.input_map[condition.input_set_id].inputs
        device = str(condition.subject.options.get("device", "cpu"))
        provenance = _provenance(condition.subject, device=device)
        for session_index, session in enumerate(schedule):
            entry = next(item for item in session if item.condition_id == condition.condition_id)
            phases = [PhaseEvent("process-start", 0, 1), PhaseEvent("load", 1, 2)]
            cursor = 2
            for warmup_index in range(spec.warmups):
                phase = PhaseEvent(f"warmup-{warmup_index}", cursor, cursor + 1)
                phases.append(phase)
                cursor += 1
            repeats = []
            duration_ns = max(1, round(ratios[condition.condition_id] * 1_000_000_000))
            for repeat_index in range(spec.repeats):
                phase = PhaseEvent(
                    f"repeat-{repeat_index}",
                    cursor,
                    cursor + duration_ns,
                    repeat_index=repeat_index,
                )
                cursor += duration_ns
                phases.append(phase)
                results = tuple(
                    {
                        "audio_id": item.audio_id,
                        "reference_id": item.reference_id,
                        "audio_path": str(Path.cwd() / item.prepared_path),
                        "source_path": str(Path.cwd() / item.source_path),
                        "text": text.get(condition.condition_id, "stable"),
                        "trusted": True,
                        "error": None,
                        "model_provenance": provenance.to_dict(),
                    }
                    for item in inputs
                )
                resources: dict[str, object] = {
                    "wall_s": ratios[condition.condition_id],
                    "cpu_s": 0.5,
                    "rss_peak_mb": 10.0,
                    "process_peak_rss_mb": 12.0,
                }
                if condition.profile:
                    resources.update({"profiler_cpu_s": 0.01, "sampler_cpu_s": 0.005})
                if condition.sample_uss:
                    resources["uss"] = {"n": 2, "samples": [7.0, 8.0]}
                repeats.append(
                    RepeatRecord(
                        index=repeat_index,
                        phase=phase,
                        expected_audio_s=sum(item.duration_s for item in inputs),
                        results=results,
                        resources=resources,
                        complete=True,
                    )
                )
            phases.append(PhaseEvent("unload", cursor, cursor + 1))
            request = WorkerRequest(
                run_id=spec.experiment_id,
                subject=condition.subject,
                inputs=inputs,
                warmups=spec.warmups,
                repeats=spec.repeats,
                profile=condition.profile,
                sample_uss=condition.sample_uss,
                worker_index=session_index,
                experiment_id=spec.experiment_id,
                session_id=f"{spec.experiment_id}:session-{session_index}",
                session_index=session_index,
                launch_position=entry.launch_position,
                condition_id=condition.condition_id,
                schedule_seed=spec.schedule_seed,
                runner_id=condition.runner_id,
                input_mode=spec.input_map[condition.input_set_id].input_mode,
                model_binding=ModelBinding(provenance, provenance.artifacts),
            )
            found[condition.condition_id].append(
                WorkerResponse(
                    run_id=spec.experiment_id,
                    request_key=condition.request_key,
                    request_sha256=request.identity_sha256,
                    worker_index=session_index,
                    subject=condition.subject,
                    host={"chip": "test", "platform": "test-os", "machine": "arm64"},
                    environment={
                        "python": "3.12",
                        "python_executable": str(Path.cwd() / ".venv/bin/python3"),
                        "platform": "test-os",
                        "machine": "arm64",
                        "packages": {"runtime": "1"},
                        "git_commit": "c" * 40,
                        "git_dirty": False,
                        "uv_lock_sha256": "d" * 64,
                        "pid": 1_000 + session_index,
                    },
                    runtime={
                        "model": condition.subject.model,
                        "device": device,
                        "dtype": "float32",
                    },
                    phases=tuple(phases),
                    repeats=tuple(repeats),
                    load_resources={"wall_s": 0.25},
                    resolved_model=condition.subject.model,
                    resolved_device=device,
                    resolved_dtype="float32",
                    model_provenance=provenance.to_dict(),
                    complete=True,
                    experiment_id=spec.experiment_id,
                    session_id=f"{spec.experiment_id}:session-{session_index}",
                    session_index=session_index,
                    launch_position=entry.launch_position,
                    condition_id=condition.condition_id,
                    schedule_seed=spec.schedule_seed,
                )
            )
    return found


def _raw_artifacts(
    root: Path,
    spec: ExperimentSpec,
    responses: dict[str, list[WorkerResponse]],
) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {condition.condition_id: [] for condition in spec.conditions}
    conditions = spec.condition_map
    inputs = spec.input_map
    for condition_id, condition_responses in responses.items():
        condition = conditions[condition_id]
        for response in condition_responses:
            assert response.model_provenance is not None
            provenance = ModelProvenance.from_dict(response.model_provenance)
            request = WorkerRequest(
                run_id=spec.experiment_id,
                subject=condition.subject,
                inputs=inputs[condition.input_set_id].inputs,
                warmups=spec.warmups,
                repeats=spec.repeats,
                profile=condition.profile,
                sample_uss=condition.sample_uss,
                worker_index=response.worker_index,
                experiment_id=spec.experiment_id,
                session_id=response.session_id,
                session_index=response.session_index,
                launch_position=response.launch_position,
                condition_id=condition_id,
                schedule_seed=spec.schedule_seed,
                runner_id=condition.runner_id,
                input_mode=inputs[condition.input_set_id].input_mode,
                model_binding=ModelBinding(provenance, provenance.artifacts),
            )
            assert request.identity_sha256 == response.request_sha256
            stem = f"worker-{response.session_index:02d}-{condition_id}"
            request_path = root / f"{stem}.request.json"
            response_path = root / f"{stem}.response.json"
            journal_path = root / f"{stem}.journal.json"
            stdout_path = root / f"{stem}.stdout.txt"
            stderr_path = root / f"{stem}.stderr.txt"
            write_request(request, request_path)
            write_response(response, response_path)
            write_journal(
                WorkerJournal.from_dict(
                    {
                        **response.to_dict(),
                        "active_phase": None,
                        "updated_ns": response.phases[-1].ended_ns,
                    }
                ),
                journal_path,
            )
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            paths[condition_id].extend(
                [request_path, response_path, journal_path, stdout_path, stderr_path]
            )
    return paths


def test_spec_round_trip_keeps_explicit_observer_topology():
    spec = _spec(sessions=6)

    restored = ExperimentSpec.from_dict(spec.to_dict())
    schedule = build_schedule(restored)

    assert restored == spec
    assert [
        (item.control_condition_id, item.treatment_condition_id) for item in spec.contrasts
    ] == [
        ("off", "profile"),
        ("off", "profile-uss"),
    ]
    assert all(len(session) == 3 for session in schedule)
    positions = {
        condition.condition_id: [
            entry.launch_position
            for session in schedule
            for entry in session
            if entry.condition_id == condition.condition_id
        ]
        for condition in spec.conditions
    }
    assert all(sorted(values) == [0, 0, 1, 1, 2, 2] for values in positions.values())


def test_explicit_contrast_direction_survives_reversed_condition_declaration():
    control = ConditionSpec(
        "control",
        SubjectSpec("fake", "model", "mya_Mymr", 1, _options("cpu")),
        "canonical",
    )
    treatment = ConditionSpec(
        "treatment",
        SubjectSpec("fake", "model", "mya_Mymr", 1, _options("mps")),
        "canonical",
    )
    contrast = ContrastSpec(
        "device",
        "control",
        "treatment",
        require_same_execution=False,
    )
    spec = _spec(conditions=(treatment, control), contrasts=(contrast,))

    result = summarize_experiment(spec, _responses(spec, {"control": 1.0, "treatment": 1.01}))
    comparison = result["contrasts"][0]

    assert comparison["direction"] == "treatment/control"
    assert comparison["paired_ratio"]["point"] == pytest.approx(1.01)
    assert comparison["gating_eligible"]
    assert result["gate_eligible"]


def test_incomplete_provenance_and_transcript_change_cannot_gate():
    spec = _spec()
    responses = _responses(
        spec,
        {"off": 1.0, "profile": 1.0, "profile-uss": 1.0},
        text={"profile": "changed"},
    )
    responses["profile-uss"][0] = replace(
        responses["profile-uss"][0],
        provenance_issues=("forced provenance failure",),
    )

    result = summarize_experiment(spec, responses)
    by_id = {item["contrast_id"]: item for item in result["contrasts"]}

    assert not by_id["profile-overhead"]["gating_eligible"]
    assert "cross-arm transcripts changed" in by_id["profile-overhead"]["classification_reasons"]
    assert not by_id["uss-overhead"]["gating_eligible"]
    assert any("provenance" in reason for reason in by_id["uss-overhead"]["classification_reasons"])
    assert not result["gate_eligible"]


def test_point_and_interval_limit_are_fail_closed():
    spec = _spec()
    result = summarize_experiment(
        spec,
        _responses(spec, {"off": 1.0, "profile": 1.03, "profile-uss": 1.0}),
    )
    comparison = result["contrasts"][0]

    assert comparison["paired_ratio"]["point"] == pytest.approx(1.03)
    assert not comparison["gating_eligible"]
    assert "point estimate exceeds" in " ".join(comparison["classification_reasons"])


def test_source_identity_can_join_different_prepared_waveforms():
    source = "9"
    canonical = InputSetSpec("canonical", (_audio(suffix="1", source=source),))
    native = InputSetSpec("native", (_audio(suffix="8", source=source),), input_mode="source")
    subject = SubjectSpec("fake", "model", "mya_Mymr", 1, _options("cpu"))
    conditions = (
        ConditionSpec("prepared", subject, "canonical"),
        ConditionSpec("native", subject, "native", runner_id="source"),
    )
    contrast = ContrastSpec("input", "prepared", "native", join_on="source_sha256")
    spec = _spec(
        conditions=conditions,
        contrasts=(contrast,),
        input_sets=(canonical, native),
    )

    result = summarize_experiment(spec, _responses(spec, {"prepared": 1.0, "native": 1.0}))

    assert result["contrasts"][0]["gating_eligible"]


def test_source_runner_requires_source_input_set():
    subject = SubjectSpec("fake", "model", "mya_Mymr", 1, _options("cpu"))
    with pytest.raises(MeasurementError, match="requires a source input set"):
        _spec(
            conditions=(ConditionSpec("native", subject, "canonical", runner_id="source"),),
            contrasts=(),
        ).validate()


def test_gating_protocol_and_nonfinite_threshold_are_rejected():
    with pytest.raises(MeasurementError, match="trusted experiment protocol"):
        _spec(sessions=1).validate()
    with pytest.raises(MeasurementError, match="finite and non-negative"):
        replace(_spec().contrasts[0], threshold=float("nan")).validate()
    malformed = _spec().to_dict()
    malformed["conditions"][0]["profile"] = "false"
    with pytest.raises(MeasurementError, match="must be a boolean"):
        ExperimentSpec.from_dict(malformed)
    with pytest.raises(MeasurementError, match="unsupported experiment runner"):
        replace(_spec().conditions[0], runner_id="upstream").validate()
    weakened = replace(
        _spec(),
        statistics=StatisticsSpec(min_sessions=1, min_warmups=0, min_repeats=1),
    )
    with pytest.raises(MeasurementError, match="cannot weaken the trust gate"):
        weakened.validate()


def test_observer_accounting_names_uss_collection_state():
    spec = _spec()
    result = summarize_experiment(
        spec,
        _responses(spec, {"off": 1.0, "profile": 1.0, "profile-uss": 1.0}),
    )
    summaries = {item["condition_id"]: item for item in result["condition_summaries"]}

    assert summaries["off"]["observer"]["uss_status"] == "disabled"
    assert summaries["profile"]["observer"]["uss_status"] == "disabled"
    assert summaries["profile-uss"]["observer"]["uss_status"] == "collected"
    assert summaries["profile-uss"]["observer"]["uss_sample_count"] == 30


def test_missing_host_and_environment_identity_cannot_gate():
    spec = _spec()
    responses = _responses(spec, {"off": 1.0, "profile": 1.0, "profile-uss": 1.0})
    for condition_id, condition_responses in responses.items():
        responses[condition_id] = [
            replace(response, host={}, environment={}) for response in condition_responses
        ]

    result = summarize_experiment(spec, responses)

    assert not result["gate_eligible"]
    assert all(
        any("identity differs" in reason for reason in contrast["classification_reasons"])
        for contrast in result["contrasts"]
    )


def test_experiment_archive_round_trip_and_raw_tamper_detection(tmp_path):
    spec = _spec()
    responses = _responses(spec, {"off": 1.0, "profile": 1.0, "profile-uss": 1.0})
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = _raw_artifacts(raw_dir, spec, responses)
    summary = summarize_experiment(spec, responses)
    raw_request = next(path for paths in raw.values() for path in paths if ".request." in path.name)
    raw_request_before = raw_request.read_text(encoding="utf-8")
    assert str(Path.cwd()) in raw_request_before

    descriptor = publish_experiment(summary, raw, root=tmp_path / "experiments")

    assert verify_experiment(descriptor) == []
    assert raw_request.read_text(encoding="utf-8") == raw_request_before
    for path in descriptor.parent.rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            assert str(Path.cwd()) not in content
            assert "/Users/" not in content
            assert "/Volumes/" not in content
    archived_response = next(descriptor.parent.glob("workers/**/response.json"))
    archived_response.write_text("tampered\n", encoding="utf-8")
    assert any("checksum mismatch" in issue for issue in verify_experiment(descriptor))


def test_archive_verifier_binds_observer_flags_after_manifest_rehash(tmp_path):
    spec = _spec()
    responses = _responses(spec, {"off": 1.0, "profile": 1.0, "profile-uss": 1.0})
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    descriptor_path = publish_experiment(
        summarize_experiment(spec, responses),
        _raw_artifacts(raw_dir, spec, responses),
        root=tmp_path / "experiments",
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in descriptor["artifact_manifest"]
        if item["condition_id"] == "profile" and item["kind"] == "request"
    )
    request_path = descriptor_path.parent / entry["path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["profile"] = False
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    entry["sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    entry["size_bytes"] = request_path.stat().st_size
    descriptor["manifest_sha256"] = archive_mod._manifest_hash(  # noqa: SLF001
        descriptor["artifact_manifest"]
    )
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True), encoding="utf-8")

    issues = verify_experiment(descriptor_path)

    assert any("observer flags differ" in issue for issue in issues)
    assert not any("request.json" in issue and "checksum mismatch" in issue for issue in issues)


def test_archive_verifier_requires_a_binding_for_gating_requests(tmp_path):
    spec = _spec()
    responses = _responses(spec, {"off": 1.0, "profile": 1.0, "profile-uss": 1.0})
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    descriptor_path = publish_experiment(
        summarize_experiment(spec, responses),
        _raw_artifacts(raw_dir, spec, responses),
        root=tmp_path / "experiments",
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in descriptor["artifact_manifest"]
        if item["condition_id"] == "off" and item["kind"] == "request"
    )
    request_path = descriptor_path.parent / entry["path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["model_binding"] = None
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    entry["sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    entry["size_bytes"] = request_path.stat().st_size
    descriptor["manifest_sha256"] = archive_mod._manifest_hash(  # noqa: SLF001
        descriptor["artifact_manifest"]
    )
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True), encoding="utf-8")

    issues = verify_experiment(descriptor_path)

    assert any("gating request lacks a model binding" in issue for issue in issues)


def test_existing_archive_is_never_removed_by_a_competing_publication(tmp_path):
    spec = _spec()
    responses = _responses(spec, {"off": 1.0, "profile": 1.0, "profile-uss": 1.0})
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    archive = tmp_path / "experiments" / spec.experiment_id
    archive.mkdir(parents=True)
    marker = archive / "experiment.json"
    marker.write_text("already published\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        publish_experiment(
            summarize_experiment(spec, responses),
            _raw_artifacts(raw_dir, spec, responses),
            root=tmp_path / "experiments",
        )

    assert marker.read_text(encoding="utf-8") == "already published\n"


def test_archive_verifier_returns_issues_for_malformed_descriptor(tmp_path):
    descriptor = tmp_path / "experiment.json"
    descriptor.write_text('{"artifact_kind":"experiment-v1","spec":null}\n', encoding="utf-8")

    issues = verify_experiment(descriptor)

    assert issues
    assert any("specification" in issue for issue in issues)
