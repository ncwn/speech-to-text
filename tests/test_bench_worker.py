"""An isolated worker retains phases, repeats, failures, and input identity."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import soundfile as sf

from stt.audio import prepare_audio
from stt.bench import run_subject_worker, summarize_workers
from stt.bench_worker import execute_request
from stt.measurement import (
    AudioInput,
    SubjectSpec,
    WorkerRequest,
    read_response,
    write_request,
)
from stt.results import TranscriptionResult


class FakeBackend:
    model = "fake-model"
    resolved_device = "cpu"

    def __init__(self) -> None:
        self.loaded = False
        self.calls = 0

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def transcribe(self, paths, language=None, batch_size=1):
        assert self.loaded
        self.calls += 1
        return [TranscriptionResult(str(path), "text", "fake", self.model) for path in paths]


def _request(tmp_path, *, backend="fake", warmups=1, repeats=2):
    source = tmp_path / "clip.wav"
    sf.write(source, np.zeros(1600), 8_000, subtype="PCM_16")
    prepared = prepare_audio(source, tmp_path / "cache")
    return WorkerRequest(
        run_id="run-1",
        subject=SubjectSpec(backend, "fake-model", "mya_Mymr", 2),
        inputs=(AudioInput.from_prepared(prepared),),
        warmups=warmups,
        repeats=repeats,
    )


def test_worker_loads_once_and_records_warmups_and_repeats(tmp_path):
    backend = FakeBackend()
    response = execute_request(
        _request(tmp_path),
        backend_factory=lambda backend_name, model, options: backend,
        root=tmp_path,
    )

    assert response.complete
    assert backend.calls == 3
    assert [phase.name for phase in response.phases] == [
        "process-start",
        "load",
        "warmup-0",
        "repeat-0",
        "repeat-1",
        "unload",
    ]
    assert len(response.repeats) == 2
    assert all(repeat.complete for repeat in response.repeats)
    assert all(repeat.corpus_rtf is not None for repeat in response.repeats)
    assert response.runtime["device"] == "cpu"
    assert "immutable model revision" in response.provenance_issues[0]

    summary = summarize_workers("fake/fake-model", [response], list(_request(tmp_path).inputs))
    assert summary["rtf"] is not None
    assert summary["peak_rss_mb"] is not None
    assert summary["memory_metric"] == "worker-lifetime-ru-maxrss-diagnostic"
    assert not summary["baseline_eligible"]

    incomplete = replace(response, complete=False, error=None)
    incomplete_summary = summarize_workers(
        "fake/fake-model", [incomplete], list(_request(tmp_path).inputs)
    )
    assert incomplete_summary["error"] == "worker response is incomplete"


def test_worker_rejects_prepared_audio_changed_after_request(tmp_path):
    request = _request(tmp_path)
    sf.write(request.inputs[0].prepared_path, np.ones(3200), 16_000, subtype="PCM_16")

    response = execute_request(
        request,
        backend_factory=lambda backend_name, model, options: FakeBackend(),
        root=tmp_path,
    )

    assert not response.complete
    assert "prepared waveform changed" in (response.error or "")
    assert not response.repeats


def test_worker_subprocess_publishes_failure_response(tmp_path):
    request = _request(tmp_path, backend="definitely-not-a-backend", warmups=0, repeats=1)
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    write_request(request, request_path)

    completed = subprocess.run(
        [sys.executable, "-m", "stt.bench_worker", str(request_path), str(response_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    response = read_response(response_path)
    assert not response.complete
    assert response.error


def test_parent_accepts_a_valid_incomplete_worker_response(tmp_path):
    request = _request(tmp_path, backend="definitely-not-a-backend", warmups=0, repeats=1)

    response, response_path = run_subject_worker(
        request,
        artifact_dir=tmp_path / "artifacts",
        timeout_s=30,
    )

    assert response_path.is_file()
    assert not response.complete
    assert response.error
    assert not response.journal_recovered
    assert response.termination is None
    stem = response_path.name.removesuffix(".response.json")
    assert (response_path.parent / f"{stem}.journal.json").is_file()
