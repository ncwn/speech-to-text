"""Adversarial coverage for the reproducible baseline-v2 trust boundary."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import stt.bench as bench_mod
from stt.bench import (
    BOOTSTRAP_DRAWS,
    STATISTICS_VERSION,
    bootstrap_paired_ratio,
    compare,
    counterbalanced_schedule,
    legacy_transcript_changes,
    save_baseline,
    summarize_workers,
    transcript_changes,
    verify_baseline,
)
from stt.measurement import (
    AudioInput,
    PhaseEvent,
    RepeatRecord,
    SubjectSpec,
    WorkerJournal,
    WorkerRequest,
    WorkerResponse,
    write_journal,
    write_json_atomic,
    write_request,
    write_response,
)
from stt.provenance import ArtifactDigest, ModelBinding, ModelProvenance


def _host(chip: str = "test-chip") -> dict[str, object]:
    return {
        "chip": chip,
        "platform": "test-os",
        "machine": "arm64",
        "python": "3.12",
        "packages": {"runtime": "1"},
        "uv_lock_sha256": "2" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "power": "Now drawing from 'AC Power'",
        "thermal": "No thermal warning level has been recorded",
    }


def _comparison_artifact(*, transcript: str = "a", execution: str = "e" * 64, host=None):
    run = {
        "subject": "fake/model",
        "hashes": {"pcm16:16000:1:" + "1" * 64: hashlib.sha256(transcript.encode()).hexdigest()},
        "baseline_eligible": True,
        "execution_sha256": execution,
        "worker_blocks": [{"representative_rtf": 1.0}] * 5,
        "natural_variation_ratio_99": 1.0,
        "error": None,
    }
    common = {
        "host": host or _host(),
        "clips": ["clip"],
        "corpus": [
            {
                "audio_id": "pcm16:16000:1:" + "1" * 64,
                "reference_id": "clip",
                "duration_s": 1.0,
            }
        ],
        "protocol": {"workers": 5, "warmups": 3, "repeats": 3, "profile": False},
        "runs": [run],
    }
    baseline = {
        **common,
        "artifact_kind": "baseline-v2",
        "baseline_schema_version": 2,
        "baseline_id": "baseline",
        "statistics": {"version": STATISTICS_VERSION, "draws": BOOTSTRAP_DRAWS},
    }
    fresh = {
        **deepcopy(common),
        "artifact_kind": "measurement-v3",
        "schema_version": 3,
        "experiment_id": "fresh",
    }
    return baseline, fresh


def test_schedule_cycles_for_more_sessions_than_subjects():
    schedule = counterbalanced_schedule(("a", "b", "c"), 7, seed=4)

    assert schedule[0] == schedule[3] == schedule[6]
    assert schedule[1] == schedule[4]
    assert schedule[2] == schedule[5]


def test_paired_bootstrap_requires_matching_sessions_and_repeat_indices():
    before = {"s0": {0: 1.0, 1: 2.0}, "s1": {0: 2.0, 1: 4.0}}
    after = {"s0": {0: 2.0, 1: 4.0}, "s1": {0: 4.0, 1: 8.0}}

    point, low, high = bootstrap_paired_ratio(before, after, seed=7) or (0.0, 0.0, 0.0)

    assert point == pytest.approx(2.0)
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(2.0)
    assert bootstrap_paired_ratio(before, {"s0": after["s0"]}, seed=7) is None
    assert (
        bootstrap_paired_ratio(
            before,
            {**after, "s1": {0: 4.0}},
            seed=7,
        )
        is None
    )
    for invalid in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
        assert (
            bootstrap_paired_ratio(
                before,
                {**after, "s1": {0: invalid, 1: 8.0}},
                seed=7,
            )
            is None
        )


def test_reordered_launch_positions_do_not_move_current_rss_between_subjects(tmp_path):
    """Current RSS follows the isolated subject, not its position in a session."""
    weight = tmp_path / "model.bin"
    weight.write_bytes(b"weights")
    base_provenance, _ = _provenance(weight)
    audio = AudioInput(
        source_path=str(tmp_path / "source.wav"),
        prepared_path=str(tmp_path / "prepared.wav"),
        reference_id="clip",
        source_sha256="3" * 64,
        audio_id="pcm16:16000:1:" + "4" * 64,
        duration_s=1.0,
        sample_rate=16_000,
        channels=1,
        frames=16_000,
    )
    schedule = [
        {"model-a": 0, "model-b": 1},
        {"model-a": 1, "model-b": 0},
        {"model-a": 0, "model-b": 1},
    ]

    def response(model: str, session_index: int, rss_mb: float) -> WorkerResponse:
        provenance = replace(
            base_provenance,
            requested_model=model,
            source_locator=f"test/{model}",
            content_sha256=None,
            execution_sha256=None,
        ).finalized()
        subject = SubjectSpec("fake", model, "mya_Mymr", 1)
        repeat_phase = PhaseEvent("repeat-0", 2, 3, repeat_index=0)
        return WorkerResponse(
            run_id="rss-reorder",
            request_key=subject.request_key,
            request_sha256=hashlib.sha256(f"{model}:{session_index}".encode("ascii")).hexdigest(),
            worker_index=session_index,
            subject=subject,
            host={"chip": "test-chip", "platform": "test-os", "machine": "arm64"},
            environment={
                "python": "3.12",
                "platform": "test-os",
                "machine": "arm64",
                "packages": {"runtime": "1"},
                "git_commit": "b" * 40,
                "git_dirty": False,
                "uv_lock_sha256": "2" * 64,
                "pid": 10_000 + session_index + (100 if model == "model-b" else 0),
            },
            runtime={"model": model, "device": "cpu", "dtype": "float32"},
            phases=(
                PhaseEvent("process-start", 0, 1),
                PhaseEvent("load", 1, 2),
                repeat_phase,
                PhaseEvent("unload", 3, 4),
            ),
            repeats=(
                RepeatRecord(
                    index=0,
                    phase=repeat_phase,
                    expected_audio_s=1.0,
                    results=(
                        {
                            "audio_id": audio.audio_id,
                            "reference_id": audio.reference_id,
                            "text": "stable",
                            "trusted": True,
                            "error": None,
                            "model_provenance": provenance.to_dict(),
                        },
                    ),
                    resources={
                        "wall_s": 1.0,
                        "cpu_s": 0.5,
                        "rss_peak_mb": rss_mb,
                        "process_peak_rss_mb": rss_mb + 5.0,
                    },
                    complete=True,
                ),
            ),
            load_resources={"wall_s": 1.0},
            resolved_model=model,
            resolved_device="cpu",
            resolved_dtype="float32",
            model_provenance=provenance.to_dict(),
            complete=True,
            experiment_id="rss-reorder",
            session_id=f"rss-reorder:session-{session_index}",
            session_index=session_index,
            launch_position=schedule[session_index][model],
            schedule_seed=17,
        )

    responses = {
        "model-a": [response("model-a", index, 40.0 + index) for index in range(3)],
        "model-b": [response("model-b", index, 140.0 + index) for index in range(3)],
    }
    summaries = {}
    for model, model_responses in responses.items():
        coordinates = [
            (index, schedule[index][model], f"rss-reorder:session-{index}") for index in range(3)
        ]
        summaries[model] = summarize_workers(
            f"fake/{model}",
            list(reversed(model_responses)),
            [audio],
            expected_workers=3,
            expected_warmups=0,
            expected_repeats=1,
            expected_launch_positions=[item[1] for item in coordinates],
            expected_session_coordinates=coordinates,
            experiment_id="rss-reorder",
        )

    assert summaries["model-a"]["endpoint_rss_mb"] == 42.0
    assert summaries["model-b"]["endpoint_rss_mb"] == 142.0
    assert summaries["model-a"]["worker_lifetime_peak_rss_mb"] == 47.0
    assert summaries["model-b"]["worker_lifetime_peak_rss_mb"] == 147.0
    assert summaries["model-a"]["baseline_eligible"]
    assert summaries["model-b"]["baseline_eligible"]


def test_transcript_change_is_not_hidden_by_provenance_change():
    baseline, fresh = _comparison_artifact()
    fresh["runs"][0]["hashes"] = {
        "pcm16:16000:1:" + "1" * 64: hashlib.sha256(b"changed").hexdigest()
    }
    fresh["runs"][0]["execution_sha256"] = "f" * 64

    verdict = compare(baseline, fresh)

    assert [finding.signal for finding in verdict.failures] == ["text"]
    assert transcript_changes(baseline, fresh)[0]["audio_id"].startswith("pcm16:")


def test_legacy_transcript_changes_maps_reference_ids_and_short_hashes():
    audio_id = "pcm16:16000:1:" + "1" * 64
    full_hash = hashlib.sha256(b"stable").hexdigest()
    legacy = {"runs": [{"subject": "fake/model", "hashes": {"clip": full_hash[:16]}}]}
    fresh = {
        "corpus": [{"reference_id": "clip", "audio_id": audio_id}],
        "runs": [{"subject": "fake/model", "hashes": {audio_id: full_hash}}],
    }

    assert legacy_transcript_changes(legacy, fresh) == []
    fresh["runs"][0]["hashes"][audio_id] = hashlib.sha256(b"changed").hexdigest()
    assert legacy_transcript_changes(legacy, fresh) == [
        {
            "subject": "fake/model",
            "audio_id": audio_id,
            "reference_id": "clip",
            "baseline_sha256": full_hash[:16],
            "fresh_sha256": hashlib.sha256(b"changed").hexdigest(),
        }
    ]


def test_host_change_suppresses_timing_but_not_transcript_checks():
    baseline, fresh = _comparison_artifact()
    fresh["host"] = _host("different-chip")

    verdict = compare(baseline, fresh)

    assert verdict.ok
    assert not verdict.compared_timings
    assert "host/runtime identity changed" in (verdict.host_note or "")


def _provenance(weight: Path) -> tuple[ModelProvenance, ModelBinding]:
    payload = weight.read_bytes()
    artifact = ArtifactDigest(
        role="weights",
        name=weight.name,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        path=str(weight),
    )
    provenance = ModelProvenance(
        backend="fake",
        requested_model="model",
        source_kind="test",
        source_locator="test/model",
        upstream_revision="a" * 40,
        revision_status="pinned",
        artifacts=(artifact,),
        runtime_packages={"runtime": "1"},
        requested_settings={"batch_size": 1, "language": "mya_Mymr"},
        resolved_settings={"device": "cpu", "dtype": "float32"},
        adapter_git_commit="b" * 40,
        uv_lock_sha256="2" * 64,
    ).finalized()
    return provenance, ModelBinding(provenance, (artifact,))


def _worker_artifacts(
    root: Path,
    session_index: int,
    audio: AudioInput,
    provenance: ModelProvenance,
    binding: ModelBinding,
) -> tuple[WorkerResponse, list[str]]:
    subject = SubjectSpec("fake", "model", "mya_Mymr", 1)
    request = WorkerRequest(
        run_id="experiment",
        subject=subject,
        inputs=(audio,),
        warmups=3,
        repeats=3,
        worker_index=session_index,
        experiment_id="experiment",
        session_id=f"experiment:session-{session_index}",
        session_index=session_index,
        launch_position=0,
        schedule_seed=9,
        model_binding=binding,
    )
    result = {
        "audio_id": audio.audio_id,
        "reference_id": audio.reference_id,
        "text": "stable",
        "trusted": True,
        "error": None,
        "model_provenance": provenance.to_dict(),
    }
    phases = [
        PhaseEvent("process-start", 0, 1),
        PhaseEvent("load", 1, 2),
        *(PhaseEvent(f"warmup-{index}", 2 + index, 3 + index) for index in range(3)),
    ]
    repeats: list[RepeatRecord] = []
    for index in range(3):
        phase = PhaseEvent(f"repeat-{index}", 5 + index, 6 + index, repeat_index=index)
        phases.append(phase)
        repeats.append(
            RepeatRecord(
                index=index,
                phase=phase,
                expected_audio_s=1.0,
                results=(result,),
                resources={"wall_s": 1.0, "cpu_s": 0.5, "process_peak_rss_mb": 10.0},
                complete=True,
            )
        )
    phases.append(PhaseEvent("unload", 8, 9))
    response = WorkerResponse(
        run_id=request.run_id,
        request_key=request.subject.request_key,
        request_sha256=request.identity_sha256,
        worker_index=session_index,
        subject=subject,
        host={"chip": "test-chip", "platform": "test-os", "machine": "arm64"},
        environment={
            "python": "3.12",
            "platform": "test-os",
            "machine": "arm64",
            "packages": {"runtime": "1"},
            "git_commit": "b" * 40,
            "git_dirty": False,
            "uv_lock_sha256": "2" * 64,
        },
        runtime={"model": "model", "device": "cpu", "dtype": "float32"},
        phases=tuple(phases),
        repeats=tuple(repeats),
        load_resources={"wall_s": 1.0},
        resolved_model="model",
        resolved_device="cpu",
        resolved_dtype="float32",
        model_provenance=provenance.to_dict(),
        complete=True,
        experiment_id=request.experiment_id,
        session_id=request.session_id,
        session_index=session_index,
        launch_position=0,
        schedule_seed=9,
    )
    stem = f"worker-{session_index:02d}-{request.identity_sha256[:16]}"
    request_path = root / f"{stem}.request.json"
    response_path = root / f"{stem}.response.json"
    journal_path = root / f"{stem}.journal.json"
    stdout_path = root / f"{stem}.stdout.txt"
    stderr_path = root / f"{stem}.stderr.txt"
    write_request(request, request_path)
    write_response(response, response_path)
    write_journal(
        WorkerJournal.from_dict({**response.to_dict(), "active_phase": None, "updated_ns": 10}),
        journal_path,
    )
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return response, [
        str(request_path),
        str(response_path),
        str(journal_path),
        str(stdout_path),
        str(stderr_path),
    ]


def _accepted_measurement(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(bench_mod, "SUBJECTS", (("fake", "model"),))
    raw = tmp_path / "raw"
    raw.mkdir()
    weight = tmp_path / "model.bin"
    weight.write_bytes(b"weights")
    provenance, binding = _provenance(weight)
    audio = AudioInput(
        source_path=str(tmp_path / "source.wav"),
        prepared_path=str(tmp_path / "prepared.wav"),
        reference_id="clip",
        source_sha256="3" * 64,
        audio_id="pcm16:16000:1:" + "4" * 64,
        duration_s=1.0,
        sample_rate=16_000,
        channels=1,
        frames=16_000,
    )
    responses: list[WorkerResponse] = []
    artifacts: list[str] = []
    for session_index in range(5):
        response, paths = _worker_artifacts(raw, session_index, audio, provenance, binding)
        responses.append(response)
        artifacts.extend(paths)
    run = summarize_workers(
        "fake/model",
        responses,
        [audio],
        expected_workers=5,
        expected_warmups=3,
        expected_repeats=3,
        expected_launch_positions=[0] * 5,
        experiment_id="experiment",
    )
    assert run["baseline_eligible"]
    run["raw_artifacts"] = artifacts
    measurement = {
        "artifact_kind": "measurement-v3",
        "schema_version": 3,
        "experiment_id": "experiment",
        "run_id": "experiment",
        "host": _host(),
        "clips": ["clip"],
        "corpus": [
            {
                "audio_id": audio.audio_id,
                "reference_id": audio.reference_id,
                "duration_s": audio.duration_s,
            }
        ],
        "schedule_seed": 9,
        "session_schedule": [[("fake/model", 0)] for _ in range(5)],
        "protocol": {"workers": 5, "warmups": 3, "repeats": 3, "profile": False},
        "raw_artifact_dir": str(raw),
        "runs": [run],
    }
    write_json_atomic(measurement, raw / "summary.json")
    return measurement, tmp_path / "baselines" / "bench.json"


def test_five_session_archive_verifies_one_to_one(monkeypatch, tmp_path):
    measurement, baseline_path = _accepted_measurement(monkeypatch, tmp_path)

    duplicate = deepcopy(measurement)
    duplicate["runs"].append(deepcopy(duplicate["runs"][0]))
    with pytest.raises(ValueError, match="complete benchmark subject set"):
        save_baseline(duplicate, tmp_path / "duplicate" / "bench.json")

    save_baseline(measurement, baseline_path)

    assert verify_baseline(baseline_path) == []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["session_schedule"][0][0][1] = 1
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert any("session schedule" in issue for issue in verify_baseline(baseline_path))
    baseline["session_schedule"][0][0][1] = 0
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert verify_baseline(baseline_path) == []
    response_item = next(
        item for item in baseline["artifact_manifest"] if item["kind"] == "response"
    )
    archived_response = tmp_path / response_item["path"]
    archived_response.write_text("tampered\n", encoding="utf-8")
    assert any("checksum mismatch" in issue for issue in verify_baseline(baseline_path))


def test_save_baseline_refuses_a_legacy_file_at_the_target(monkeypatch, tmp_path):
    measurement, baseline_path = _accepted_measurement(monkeypatch, tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps({"runs": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot replace an unverifiable legacy baseline"):
        save_baseline(measurement, baseline_path)


def test_committed_migration_record_matches_published_baseline():
    migration = json.loads(Path("baselines/migration-v1-to-v2.json").read_text(encoding="utf-8"))
    baseline = json.loads(Path("baselines/bench.json").read_text(encoding="utf-8"))

    assert migration["baseline_id"] == baseline["baseline_id"]
    assert migration["transcript_changes"] == baseline["legacy_migration"]["transcript_changes"]
    assert migration["approval_note"] == baseline["legacy_migration"]["approval_note"]


def test_verify_rejects_unsupported_baseline_schema(monkeypatch, tmp_path):
    measurement, baseline_path = _accepted_measurement(monkeypatch, tmp_path)
    save_baseline(measurement, baseline_path)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["baseline_schema_version"] = 99
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    assert verify_baseline(baseline_path) == ["baseline-v2 schema version is unsupported"]


def test_verify_rejects_noncanonical_archived_summary_even_when_rehashed(monkeypatch, tmp_path):
    measurement, baseline_path = _accepted_measurement(monkeypatch, tmp_path)
    save_baseline(measurement, baseline_path)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    summary_entry = next(
        item for item in baseline["artifact_manifest"] if item["kind"] == "summary"
    )
    summary_path = tmp_path / summary_entry["path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["host"]["tampered"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    summary_entry["sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    summary_entry["size_bytes"] = summary_path.stat().st_size
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    assert any("canonical committed summary" in issue for issue in verify_baseline(baseline_path))


def test_verify_uses_archived_subject_order_after_subject_registry_changes(monkeypatch, tmp_path):
    measurement, baseline_path = _accepted_measurement(monkeypatch, tmp_path)
    save_baseline(measurement, baseline_path)
    monkeypatch.setattr(bench_mod, "SUBJECTS", (("different", "model"),))

    assert verify_baseline(baseline_path) == []


def test_failed_baseline_publication_removes_archive(monkeypatch, tmp_path):
    measurement, baseline_path = _accepted_measurement(monkeypatch, tmp_path)
    original_measurement = deepcopy(measurement)
    original_write = bench_mod.write_json_atomic

    def fail_final_write(data, path):
        if Path(path) == baseline_path:
            raise OSError("simulated baseline publication failure")
        original_write(data, path)

    monkeypatch.setattr(bench_mod, "write_json_atomic", fail_final_write)
    with pytest.raises(OSError, match="publication failure"):
        save_baseline(measurement, baseline_path)

    assert not (baseline_path.parent / "artifacts" / "experiment").exists()
    assert not list((baseline_path.parent / "artifacts").glob(".*.transaction.json"))
    assert measurement == original_measurement

    monkeypatch.setattr(bench_mod, "write_json_atomic", original_write)
    save_baseline(measurement, baseline_path)
    assert verify_baseline(baseline_path) == []


def test_request_identity_covers_the_full_execution_request():
    audio = AudioInput(
        source_path="source.wav",
        prepared_path="prepared.wav",
        reference_id="clip",
        source_sha256="3" * 64,
        audio_id="pcm16:16000:1:" + "4" * 64,
        duration_s=1.0,
        sample_rate=16_000,
        channels=1,
        frames=16_000,
    )
    request = WorkerRequest("run", SubjectSpec("fake", "model", None, 1), (audio,))

    assert request.identity_sha256 != replace(request, warmups=2).identity_sha256
    assert request.identity_sha256 != replace(request, profile=True).identity_sha256
