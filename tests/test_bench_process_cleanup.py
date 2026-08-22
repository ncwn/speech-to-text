"""Parent cleanup keeps abnormal workers fail-closed and journal-recoverable."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import stt.bench as bench_mod
from stt.audio import prepare_audio
from stt.measurement import (
    ActivePhase,
    AudioInput,
    SubjectSpec,
    WorkerJournal,
    WorkerRequest,
    read_response,
    write_journal,
)


def _request(tmp_path: Path) -> WorkerRequest:
    source = tmp_path / "clip.wav"
    sf.write(source, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    prepared = prepare_audio(source, tmp_path / "cache")
    return WorkerRequest(
        run_id="run-1",
        subject=SubjectSpec("fake", "model", "mya_Mymr", 1),
        inputs=(AudioInput.from_prepared(prepared),),
        warmups=0,
        repeats=1,
    )


class _AbnormalProcess:
    pid = 12345
    returncode = 2

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        raise AssertionError(f"unexpected direct signal: {value}")


class _InterruptProcess(_AbnormalProcess):
    def __init__(self):
        self.wait_calls = 0
        self.killed = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise KeyboardInterrupt
        return self.returncode

    def kill(self):
        self.killed = True


class _AliveProcess(_AbnormalProcess):
    returncode = None

    def __init__(self):
        self.alive = True
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        if self.alive:
            raise AssertionError("force reap waited before killing the worker")
        self.returncode = -signal.SIGKILL
        return self.returncode

    def poll(self):
        return None if self.alive else self.returncode

    def kill(self):
        self.alive = False


def _popen_with_journal(request: WorkerRequest, process: _AbnormalProcess, calls: list[dict]):
    def fake_popen(arguments, **kwargs):
        calls.append(kwargs)
        journal_path = Path(arguments[5])
        checkpoint_ns = time.perf_counter_ns()
        write_journal(
            WorkerJournal(
                run_id=request.run_id,
                request_key=request.subject.request_key,
                request_sha256=request.identity_sha256,
                worker_index=request.worker_index,
                subject=request.subject,
                host={},
                environment={},
                runtime={},
                active_phase=ActivePhase("process-start", checkpoint_ns),
                updated_ns=checkpoint_ns,
            ),
            journal_path,
        )
        return process

    return fake_popen


def test_abnormal_exit_terminates_worker_group_and_recovers_journal(monkeypatch, tmp_path):
    request = _request(tmp_path)
    process = _AbnormalProcess()
    popen_calls: list[dict] = []
    cleanup_calls: list[object] = []
    monkeypatch.setattr(
        bench_mod.subprocess,
        "Popen",
        _popen_with_journal(request, process, popen_calls),
    )
    monkeypatch.setattr(
        bench_mod,
        "_terminate_worker_process",
        lambda value: cleanup_calls.append(value),
    )

    response, _ = bench_mod.run_subject_worker(
        request,
        artifact_dir=tmp_path / "artifacts",
        timeout_s=30,
    )

    assert cleanup_calls == [process]
    assert len(popen_calls) == 1
    assert popen_calls[0]["text"] is True
    assert popen_calls[0]["start_new_session"] is (os.name == "posix")
    assert response.journal_recovered
    assert not response.complete
    assert response.termination == "worker exited with status 2"
    assert "worker exited with status 2" in (response.error or "")


def test_cleanup_failure_is_recorded_in_recovered_response(monkeypatch, tmp_path):
    request = _request(tmp_path)
    process = _AbnormalProcess()
    monkeypatch.setattr(
        bench_mod.subprocess,
        "Popen",
        _popen_with_journal(request, process, []),
    )

    def fail_cleanup(value):
        assert value is process
        raise RuntimeError("group survived SIGKILL")

    monkeypatch.setattr(bench_mod, "_terminate_worker_process", fail_cleanup)

    def missing_group(*_args):
        raise ProcessLookupError

    monkeypatch.setattr(bench_mod.os, "killpg", missing_group)

    with pytest.raises(RuntimeError, match="group survived SIGKILL") as raised:
        bench_mod.run_subject_worker(
            request,
            artifact_dir=tmp_path / "artifacts",
            timeout_s=30,
        )

    assert "recovered response captured at" in str(raised.value)
    run_dir = tmp_path / "artifacts" / request.run_id
    assert not list(run_dir.glob("*.response.json"))
    recovered = list(run_dir.glob("*.response.json.recovered-*"))
    assert len(recovered) == 1
    response = read_response(recovered[0])
    assert response.journal_recovered
    assert not response.complete
    assert "group survived SIGKILL" in (response.error or "")


def test_wait_interrupt_terminates_and_reaps_before_reraising(monkeypatch, tmp_path):
    request = _request(tmp_path)
    process = _InterruptProcess()
    monkeypatch.setattr(
        bench_mod.subprocess,
        "Popen",
        _popen_with_journal(request, process, []),
    )
    cleanup_calls: list[object] = []
    monkeypatch.setattr(
        bench_mod,
        "_terminate_worker_process",
        lambda value: cleanup_calls.append(value),
    )

    with pytest.raises(KeyboardInterrupt):
        bench_mod.run_subject_worker(
            request,
            artifact_dir=tmp_path / "artifacts",
            timeout_s=30,
        )

    assert cleanup_calls == [process]
    assert process.wait_calls == 1


def test_wait_interrupt_uses_force_reap_when_cleanup_raises(monkeypatch, tmp_path):
    request = _request(tmp_path)
    process = _InterruptProcess()
    monkeypatch.setattr(
        bench_mod.subprocess,
        "Popen",
        _popen_with_journal(request, process, []),
    )

    def missing_group(*_args):
        raise ProcessLookupError

    monkeypatch.setattr(bench_mod.os, "killpg", missing_group)

    def fail_cleanup(value):
        assert value is process
        raise RuntimeError("group survived SIGKILL")

    monkeypatch.setattr(bench_mod, "_terminate_worker_process", fail_cleanup)

    with pytest.raises(KeyboardInterrupt):
        bench_mod.run_subject_worker(
            request,
            artifact_dir=tmp_path / "artifacts",
            timeout_s=30,
        )

    assert process.killed
    assert process.wait_calls == 2


def test_timeout_cleanup_failure_force_reaps_live_worker(monkeypatch, tmp_path):
    request = _request(tmp_path)
    process = _AliveProcess()
    monkeypatch.setattr(
        bench_mod.subprocess,
        "Popen",
        _popen_with_journal(request, process, []),
    )
    killpg_calls: list[object] = []

    def fake_killpg(_pid, value):
        killpg_calls.append(value)
        if value == signal.SIGKILL:
            process.alive = False
        elif value == 0 and process.alive:
            return
        elif value == 0:
            raise ProcessLookupError

    monkeypatch.setattr(bench_mod.os, "killpg", fake_killpg)

    def fail_cleanup(value):
        assert value is process
        raise RuntimeError("group survived SIGKILL")

    monkeypatch.setattr(bench_mod, "_terminate_worker_process", fail_cleanup)

    with pytest.raises(RuntimeError, match="recovered response captured at"):
        bench_mod.run_subject_worker(
            request,
            artifact_dir=tmp_path / "artifacts",
            timeout_s=0.01,
        )

    assert not process.alive
    assert signal.SIGKILL in killpg_calls
    assert 0 in killpg_calls
    assert process.wait_calls == 2
