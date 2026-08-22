"""CLI boundary for explicit experiment specifications and verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from stt.cli import app
from stt.experiment import ExperimentSpec, build_schedule

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
