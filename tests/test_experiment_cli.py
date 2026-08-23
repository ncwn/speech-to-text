"""CLI boundary for explicit experiment specifications and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from stt.cli import app
from stt.experiment import ExperimentSpec, build_schedule
from stt.measurement import WorkerResponse
from stt.provenance import ArtifactDigest, ModelBinding, ModelProvenance

runner = CliRunner()


def test_observer_spec_pins_subjects_arms_and_balanced_schedule(monkeypatch, tmp_path):
    monkeypatch.setattr("stt.cli.checkout_root", lambda: tmp_path)
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16_000), 16_000, subtype="PCM_16")
    output = tmp_path / "observer.json"

    result = runner.invoke(
        app,
        ["experiment", "observer-spec", str(audio), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    spec = ExperimentSpec.from_dict(json.loads(output.read_text(encoding="utf-8")))
    assert len(spec.conditions) == 6
    assert len(spec.contrasts) == 4
    assert all(
        not Path(item.source_path).is_absolute() and not Path(item.prepared_path).is_absolute()
        for item in spec.input_sets[0].inputs
    )
    by_id = {condition.condition_id: condition for condition in spec.conditions}
    assert by_id["fast-off"].subject.backend == "hf"
    assert by_id["fast-off"].subject.options["device"] == "mps"
    assert by_id["slow-off"].subject.backend == "omniasr-torch"
    assert by_id["slow-off"].subject.options["device"] == "cpu"
    assert by_id["fast-profile"].profile and not by_id["fast-profile"].sample_uss
    assert by_id["fast-profile-uss"].profile and by_id["fast-profile-uss"].sample_uss
    assert {
        (contrast.control_condition_id, contrast.treatment_condition_id)
        for contrast in spec.contrasts
    } == {
        ("fast-off", "fast-profile"),
        ("fast-off", "fast-profile-uss"),
        ("slow-off", "slow-profile"),
        ("slow-off", "slow-profile-uss"),
    }
    schedule = build_schedule(spec)
    for condition in spec.conditions:
        assert sorted(
            entry.launch_position
            for session in schedule
            for entry in session
            if entry.condition_id == condition.condition_id
        ) == list(range(6))


def test_candidate_spec_declares_cartesian_device_dtype_matrix(tmp_path):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16_000), 16_000, subtype="PCM_16")
    output = tmp_path / "candidates.json"

    result = runner.invoke(
        app,
        [
            "experiment",
            "candidate-spec",
            str(audio),
            "--output",
            str(output),
            "--backend",
            "fake",
            "--model",
            "model",
            "--devices",
            "cpu,mps",
            "--dtypes",
            "float32,float16",
        ],
    )

    assert result.exit_code == 0, result.output
    spec = ExperimentSpec.from_dict(json.loads(output.read_text(encoding="utf-8")))
    candidates = {
        (item.subject.options["device"], item.subject.options["dtype"]) for item in spec.conditions
    }
    assert candidates == {
        ("cpu", "float32"),
        ("cpu", "float16"),
        ("mps", "float32"),
        ("mps", "float16"),
    }
    assert not spec.contrasts


def test_input_spec_records_preparation_outside_paired_inference(monkeypatch, tmp_path):
    monkeypatch.setattr("stt.cli.checkout_root", lambda: tmp_path)
    monkeypatch.setattr("stt.cli.DEFAULT_CACHE", tmp_path / "cache")
    audio = tmp_path / "float.wav"
    sf.write(audio, np.linspace(-0.5, 0.5, 16_000), 16_000, subtype="FLOAT")
    output = tmp_path / "input.json"

    result = runner.invoke(
        app,
        [
            "experiment",
            "input-spec",
            str(audio),
            "--output",
            str(output),
            "--id",
            "input-fixture",
        ],
    )

    assert result.exit_code == 0, result.output
    spec = ExperimentSpec.from_dict(json.loads(output.read_text(encoding="utf-8")))
    by_id = spec.input_map
    assert by_id["canonical"].input_mode == "prepared"
    assert by_id["native"].input_mode == "source"
    assert by_id["canonical"].preparation_kind == "canonical-decode-resample"
    assert by_id["native"].preparation_kind == "native-identity-probe"
    assert by_id["canonical"].inputs[0].source_sha256 == by_id["native"].inputs[0].source_sha256
    assert by_id["canonical"].inputs[0].audio_id != by_id["native"].inputs[0].audio_id
    assert spec.contrasts[0].join_on == "source_sha256"


def test_experiment_verify_reports_a_malformed_archive(tmp_path):
    descriptor = tmp_path / "experiment.json"
    descriptor.write_text('{"artifact_kind":"experiment-v1","spec":null}\n', encoding="utf-8")

    result = runner.invoke(app, ["experiment", "verify", str(descriptor)])

    assert result.exit_code == 1
    assert "specification" in result.output


def test_experiment_run_refuses_a_stale_raw_directory_before_loading_models(tmp_path):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16_000), 16_000, subtype="PCM_16")
    spec_path = tmp_path / "observer.json"
    generated = runner.invoke(
        app,
        ["experiment", "observer-spec", str(audio), "--output", str(spec_path)],
    )
    assert generated.exit_code == 0, generated.output
    raw_root = tmp_path / "raw"
    (raw_root / "observer").mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            str(spec_path),
            "--artifacts",
            str(raw_root),
            "--archive-root",
            str(tmp_path / "archive"),
        ],
    )

    assert result.exit_code != 0
    assert "raw experiment artifact directory already exists" in result.output


def test_observer_spec_refuses_an_unbalanced_five_session_gate(tmp_path):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16_000), 16_000, subtype="PCM_16")

    result = runner.invoke(
        app,
        [
            "experiment",
            "observer-spec",
            str(audio),
            "--output",
            str(tmp_path / "observer.json"),
            "--sessions",
            "5",
        ],
    )

    assert result.exit_code != 0
    assert "requires at least 6 sessions" in result.output


def test_gating_run_stops_after_the_first_incomplete_worker(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16_000), 16_000, subtype="PCM_16")
    spec_path = tmp_path / "observer.json"
    generated = runner.invoke(
        app,
        ["experiment", "observer-spec", str(audio), "--output", str(spec_path)],
    )
    assert generated.exit_code == 0, generated.output

    artifact_path = tmp_path / "model.bin"
    artifact_path.write_bytes(b"weights")
    artifact = ArtifactDigest(
        "weights",
        "model.bin",
        artifact_path.stat().st_size,
        hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        path=str(artifact_path),
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
        resolved_settings={"device": "cpu", "dtype": "float32"},
        adapter_git_commit="b" * 40,
        uv_lock_sha256="c" * 64,
    ).finalized()
    binding = ModelBinding(provenance, (artifact,))
    monkeypatch.setattr("stt.cli.preflight_model_binding", lambda *args, **kwargs: binding)
    monkeypatch.setattr("stt.cli.bench_mod.capture_environment", lambda: {"git_dirty": False})
    calls = []

    def fail_worker(request, artifact_dir, timeout_s):
        calls.append(request.condition_id)
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
            error="forced worker failure",
            experiment_id=request.experiment_id,
            session_id=request.session_id,
            session_index=request.session_index,
            launch_position=request.launch_position,
            condition_id=request.condition_id,
            schedule_seed=request.schedule_seed,
        )
        return response, artifact_dir / request.run_id / "worker.response.json"

    monkeypatch.setattr("stt.cli.bench_mod.run_subject_worker", fail_worker)

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            str(spec_path),
            "--artifacts",
            str(tmp_path / "raw"),
            "--archive-root",
            str(tmp_path / "archive"),
        ],
    )

    assert result.exit_code == 1
    assert len(calls) == 1
    assert f"stopped after {calls[0]} failed" in result.output
