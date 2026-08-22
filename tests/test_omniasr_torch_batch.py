from __future__ import annotations

from pathlib import Path

import pytest

from stt.backends.omniasr_torch import OmniASRTorchBackend


class RecordingPipeline:
    def __init__(self, transcripts: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[list[str], list[str] | None, int]] = []
        self.transcripts = transcripts or {}

    def transcribe(self, paths, *, lang, batch_size):
        self.calls.append((list(paths), lang, batch_size))
        return [self.transcripts.get(Path(path).name, Path(path).stem) for path in paths]


def _backend(pipeline) -> OmniASRTorchBackend:
    backend = OmniASRTorchBackend("omniASR_LLM_Unlimited_300M_v2")
    backend.pipeline = pipeline
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"
    # Keep these tests independent of the optional omnilingual_asr package.
    backend._check_language = lambda language: None
    return backend


def test_transcribe_passes_all_paths_and_matching_language_list(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("first.wav", "second.wav", "third.wav")]
    durations = {path: float(index + 1) for index, path in enumerate(paths)}
    monkeypatch.setattr("stt.audio.duration_of", durations.__getitem__)
    pipeline = RecordingPipeline(
        {"first.wav": " one ", "second.wav": " two ", "third.wav": " three "}
    )

    results = _backend(pipeline).transcribe(paths, language="mya_Mymr", batch_size=4)

    assert pipeline.calls == [([str(path) for path in paths], ["mya_Mymr"] * 3, 4)]
    assert [result.text for result in results] == ["one", "two", "three"]
    assert [result.audio_path for result in results] == [str(path) for path in paths]
    assert all(result.metadata["elapsed_s_source"] == "batch_proportional" for result in results)
    assert sum(result.elapsed_s or 0.0 for result in results) == pytest.approx(
        results[0].metadata["batch_elapsed_s"]
    )


def test_length_validation_keeps_invalid_file_out_of_batch(monkeypatch, tmp_path):
    long_path = tmp_path / "too-long.wav"
    short_path = tmp_path / "short.wav"
    durations = {long_path: 41.0, short_path: 2.0}
    monkeypatch.setattr("stt.audio.duration_of", durations.__getitem__)
    pipeline = RecordingPipeline()
    backend = OmniASRTorchBackend("omniASR_LLM_7B_v2")
    backend.pipeline = pipeline
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"
    backend._check_language = lambda language: None

    results = backend.transcribe([long_path, short_path], batch_size=2)

    assert pipeline.calls == [([str(short_path)], None, 2)]
    assert "accepts at most 40s" in (results[0].error or "")
    assert results[1].error is None
    assert [result.audio_path for result in results] == [str(long_path), str(short_path)]


def test_batch_exception_retries_each_file_and_preserves_errors(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("ok.wav", "bad.wav", "also-ok.wav")]
    monkeypatch.setattr("stt.audio.duration_of", lambda path: 1.0)

    class BatchFails:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def transcribe(self, paths, *, lang, batch_size):
            paths = list(paths)
            self.calls.append(paths)
            if len(paths) > 1:
                raise RuntimeError("batch unsupported")
            if Path(paths[0]).name == "bad.wav":
                raise ValueError("bad audio")
            return [Path(paths[0]).stem]

    pipeline = BatchFails()
    results = _backend(pipeline).transcribe(paths, batch_size=3)

    assert pipeline.calls[0] == [str(path) for path in paths]
    assert pipeline.calls[1:] == [[str(path)] for path in paths]
    assert [result.text for result in results] == ["ok", "", "also-ok"]
    assert results[1].error == "ValueError: bad audio"
    assert all(result.metadata.get("batch_fallback") for result in results)
    assert all(
        result.metadata.get("elapsed_s_source") == "measured"
        for result in results
        if not result.error
    )


def test_batch_size_must_be_positive(monkeypatch, tmp_path):
    path = tmp_path / "clip.wav"
    monkeypatch.setattr("stt.audio.duration_of", lambda _: 1.0)
    backend = _backend(RecordingPipeline())
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        backend.transcribe([path], batch_size=0)
