"""Versioned raw worker artifacts stay recomputable and fail closed."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from stt.audio import prepare_audio
from stt.measurement import (
    AudioInput,
    MeasurementError,
    PhaseEvent,
    RepeatRecord,
    SubjectSpec,
    WorkerRequest,
    WorkerResponse,
    capture_environment,
    read_request,
    read_response,
    write_request,
    write_response,
)


def _request(tmp_path) -> WorkerRequest:
    source = tmp_path / "clip.wav"
    sf.write(source, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    prepared = prepare_audio(source, tmp_path / "cache")
    return WorkerRequest(
        run_id="run-1",
        subject=SubjectSpec("fake", "model", "mya_Mymr", 2, {"device": "cpu"}),
        inputs=(AudioInput.from_prepared(prepared),),
        warmups=2,
        repeats=3,
    )


def test_worker_request_round_trip_keeps_canonical_audio_facts(tmp_path):
    request = _request(tmp_path)
    path = tmp_path / "request.json"

    write_request(request, path)
    restored = read_request(path)

    assert restored == request
    assert restored.inputs[0].duration_s == pytest.approx(0.1)
    assert restored.inputs[0].prepared() == request.inputs[0].prepared()
    assert restored.subject.request_key.startswith("fake/model:")


def test_worker_request_round_trip_keeps_uss_observer_choice(tmp_path):
    request = replace(_request(tmp_path), profile=True, sample_uss=True)
    path = tmp_path / "request.json"

    write_request(request, path)
    restored = read_request(path)

    assert restored.sample_uss is True
    assert restored.identity_sha256 == request.identity_sha256


def test_worker_request_rejects_uss_without_profiler(tmp_path):
    request = replace(_request(tmp_path), sample_uss=True)

    with pytest.raises(MeasurementError, match="USS sampling requires profiling"):
        request.validate()


@pytest.mark.parametrize("field", ["profile", "sample_uss"])
def test_worker_request_rejects_string_booleans(tmp_path, field):
    raw = _request(tmp_path).to_dict()
    raw[field] = "false"

    with pytest.raises(MeasurementError, match=f"{field} must be a boolean"):
        WorkerRequest.from_dict(raw)


def test_request_rejects_duplicate_waveform_identity(tmp_path):
    request = _request(tmp_path)
    duplicate = WorkerRequest(
        run_id=request.run_id,
        subject=request.subject,
        inputs=(request.inputs[0], request.inputs[0]),
    )

    with pytest.raises(MeasurementError, match="duplicate audio_id"):
        duplicate.validate()


def test_request_rejects_unknown_protocol(tmp_path):
    request = _request(tmp_path).to_dict()
    request["protocol_version"] = 99

    with pytest.raises(MeasurementError, match="unsupported measurement protocol"):
        WorkerRequest.from_dict(request)


def test_phase_duration_is_recomputed_from_monotonic_events():
    phase = PhaseEvent("corpus", 1_000_000_000, 3_500_000_000)
    assert phase.duration_s == 2.5

    with pytest.raises(MeasurementError, match="invalid monotonic interval"):
        PhaseEvent("corpus", 2, 1).validate()


def test_worker_response_round_trip_recomputes_corpus_rtf(tmp_path):
    request = _request(tmp_path)
    phase = PhaseEvent("corpus", 10, 100_000_010, repeat_index=0)
    phases = (
        PhaseEvent("process-start", 0, 1),
        PhaseEvent("load", 1, 10),
        phase,
        PhaseEvent("unload", 100_000_010, 100_000_011),
    )
    repeat = RepeatRecord(
        index=0,
        phase=phase,
        expected_audio_s=0.1,
        results=({"audio_id": request.inputs[0].audio_id, "trusted": True},),
        resources={"wall_s": 0.1},
        complete=True,
    )
    response = WorkerResponse(
        run_id=request.run_id,
        request_key=request.subject.request_key,
        request_sha256=request.identity_sha256,
        worker_index=0,
        subject=request.subject,
        host={"chip": "test"},
        environment={"python": "test"},
        runtime={"device": "cpu"},
        phases=phases,
        repeats=(repeat,),
        load_resources=None,
        complete=True,
    )
    path = tmp_path / "response.json"

    write_response(response, path)
    restored = read_response(path)

    assert restored == response
    assert restored.repeats[0].corpus_rtf == pytest.approx(1.0)


def test_atomic_writer_does_not_leave_temporary_file(tmp_path):
    request = _request(tmp_path)
    path = tmp_path / "request.json"
    write_request(request, path)

    assert json.loads(path.read_text())["run_id"] == "run-1"
    assert not list(tmp_path.glob(".request.json.*.tmp"))


def test_environment_capture_uses_checkout_when_launched_elsewhere(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    environment = capture_environment()

    assert environment["git_commit"]
    assert environment["uv_lock_sha256"]
