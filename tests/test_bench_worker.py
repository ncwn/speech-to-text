"""An isolated worker retains phases, repeats, failures, and input identity."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import soundfile as sf

import stt.bench_worker as bench_worker_mod
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


def test_worker_passes_archived_uss_choice_to_corpus_measurement(monkeypatch, tmp_path):
    seen: list[bool] = []
    original = bench_worker_mod.transcribe_corpus

    def recording_transcribe_corpus(*args, **kwargs):
        seen.append(bool(kwargs.get("sample_uss")))
        return original(*args, **kwargs)

    monkeypatch.setattr(bench_worker_mod, "transcribe_corpus", recording_transcribe_corpus)
    request = replace(
        _request(tmp_path, warmups=0, repeats=1),
        profile=True,
        sample_uss=True,
    )

    response = execute_request(
        request,
        backend_factory=lambda backend_name, model, options: FakeBackend(),
        root=tmp_path,
    )

    assert response.repeats
    assert seen == [True]


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


def test_source_runner_presents_source_path_without_rewriting_prepared_identity(tmp_path):
    request = replace(
        _request(tmp_path, warmups=0, repeats=1),
        runner_id="source",
        input_mode="source",
    )
    seen: list[str] = []

    class SourceBackend(FakeBackend):
        def transcribe(self, paths, language=None, batch_size=1):
            seen.extend(str(path) for path in paths)
            return super().transcribe(paths, language, batch_size)

    response = execute_request(
        request,
        backend_factory=lambda backend_name, model, options: SourceBackend(),
        root=tmp_path,
    )

    assert response.complete
    assert seen == [request.inputs[0].source_path]
    assert response.runtime["input_mode"] == "source"
    assert response.runtime["runner_id"] == "source"


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


def test_every_subject_gets_its_own_process(tmp_path):
    """One subject per process, checked rather than assumed.

    A worker that loaded a second model would inherit the first one's RSS
    high-water and whatever runtime state it left behind, which is the whole
    reason the benchmark spawns a process per subject. Distinct pids are the
    observable form of that claim; the reported lifetime RSS comes from each
    worker's own getrusage, so it cannot carry across.
    """
    pids: list[int] = []
    for index in range(3):
        request = replace(
            _request(tmp_path, backend="definitely-not-a-backend", warmups=0, repeats=1),
            worker_index=index,
            session_index=index,
            # Deliberately varied: if isolation depended on launch order, this
            # is where it would show.
            launch_position=(index + 1) % 3,
        )
        response, _ = run_subject_worker(request, artifact_dir=tmp_path / "artifacts", timeout_s=60)
        pid = response.environment.get("pid")
        assert isinstance(pid, int)
        pids.append(pid)

    assert len(set(pids)) == 3, f"workers shared a process: {pids}"
    assert os.getpid() not in pids, "a subject ran inside the orchestrator"
