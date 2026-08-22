"""Canonical audio identity and conversion-cache integrity."""

from __future__ import annotations

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from stt import audio


def _pcm(seed: int, *, frames: int = 800) -> np.ndarray:
    values = np.arange(frames, dtype=np.int32) * (seed * 17 + 1)
    return ((values % 50_000) - 25_000).astype(np.int16)


def _write(path: Path, pcm: np.ndarray, *, channels: int = 1, format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = pcm if channels == 1 else np.column_stack([pcm, -pcm])
    sf.write(path, samples, audio.TARGET_SAMPLE_RATE, subtype="PCM_16", format=format)


def _fake_ffmpeg(monkeypatch, *, barrier: threading.Barrier | None = None) -> list[Path]:
    outputs: list[Path] = []
    lock = threading.Lock()

    def run(command, *, check):
        assert check is True
        source = Path(command[command.index("-i") + 1])
        destination = Path(command[-1])
        if barrier is not None:
            barrier.wait(timeout=5)
        samples, _ = sf.read(source, dtype="int16", always_2d=True)
        sf.write(
            destination,
            samples[:, 0],
            audio.TARGET_SAMPLE_RATE,
            subtype="PCM_16",
            format="WAV",
        )
        with lock:
            outputs.append(destination)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio.shutil, "which", lambda command: f"/fake/{command}")
    monkeypatch.setattr(audio.subprocess, "run", run)
    return outputs


def test_cache_key_does_not_collide_for_same_legacy_parent_and_stem(monkeypatch, tmp_path):
    _fake_ffmpeg(monkeypatch)
    first = tmp_path / "one" / "a" / "clip.wav"
    second = tmp_path / "two" / "a" / "clip.flac"
    _write(first, _pcm(1), channels=2)
    _write(second, _pcm(2), channels=2)

    first_prepared = audio.prepare_audio(first, tmp_path / "cache")
    second_prepared = audio.prepare_audio(second, tmp_path / "cache")

    assert first_prepared.prepared_path != second_prepared.prepared_path
    assert first_prepared.source_sha256 != second_prepared.source_sha256
    assert first_prepared.audio_id != second_prepared.audio_id


def test_editing_a_source_never_reuses_its_previous_conversion(monkeypatch, tmp_path):
    _fake_ffmpeg(monkeypatch)
    source = tmp_path / "recording.wav"
    _write(source, _pcm(1), channels=2)
    first = audio.prepare_audio(source, tmp_path / "cache")

    _write(source, _pcm(9), channels=2)
    second = audio.prepare_audio(source, tmp_path / "cache")

    assert first.source_sha256 != second.source_sha256
    assert first.prepared_path != second.prepared_path
    assert first.audio_id != second.audio_id
    assert first.prepared_path.exists()


def test_same_pcm_in_different_containers_has_the_same_audio_id(tmp_path):
    pcm = _pcm(4)
    wav = tmp_path / "one.wav"
    flac = tmp_path / "elsewhere" / "two.flac"
    _write(wav, pcm)
    _write(flac, pcm, format="FLAC")

    wav_prepared = audio.prepare_audio(wav, tmp_path / "cache")
    flac_prepared = audio.prepare_audio(flac, tmp_path / "cache")

    assert wav_prepared.source_sha256 != flac_prepared.source_sha256
    assert wav_prepared.audio_id == flac_prepared.audio_id
    assert audio.is_valid_audio_id(wav_prepared.audio_id)


def test_changed_pcm_changes_audio_id_even_when_stream_dimensions_match(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write(first, _pcm(3))
    _write(second, _pcm(5))

    assert audio.canonical_audio_id(first) != audio.canonical_audio_id(second)


def test_non_pcm16_source_is_converted_to_the_canonical_format(monkeypatch, tmp_path):
    _fake_ffmpeg(monkeypatch)
    source = tmp_path / "float.wav"
    sf.write(source, _pcm(2).astype(np.float32) / 32768, audio.TARGET_SAMPLE_RATE, subtype="FLOAT")

    prepared = audio.prepare_audio(source, tmp_path / "cache")

    assert prepared.prepared_path != source
    assert sf.info(prepared.prepared_path).subtype == "PCM_16"


def test_invalid_and_incomplete_cache_entries_are_rebuilt(monkeypatch, tmp_path):
    conversions = _fake_ffmpeg(monkeypatch)
    source = tmp_path / "stereo.wav"
    cache = tmp_path / "cache"
    _write(source, _pcm(6), channels=2)
    expected = cache / (f"{audio.CONVERSION_RECIPE_VERSION}__{audio.file_sha256(source)}.wav")
    cache.mkdir()
    expected.write_bytes(b"not a wave file")

    assert audio.to_16k_mono(source, cache) == expected
    assert len(conversions) == 1
    assert audio.probe(expected)[:2] == (audio.TARGET_SAMPLE_RATE, 1)

    complete = expected.read_bytes()
    expected.write_bytes(complete[:30])
    assert audio.to_16k_mono(source, cache) == expected
    assert len(conversions) == 2
    assert audio.canonical_audio_id(expected).startswith("pcm16:16000:1:")


def test_interrupted_conversion_is_never_published(monkeypatch, tmp_path):
    source = tmp_path / "stereo.wav"
    cache = tmp_path / "cache"
    _write(source, _pcm(7), channels=2)
    monkeypatch.setattr(audio.shutil, "which", lambda command: f"/fake/{command}")

    def interrupt(command, *, check):
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(audio.subprocess, "run", interrupt)
    try:
        audio.to_16k_mono(source, cache)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("interrupted ffmpeg should propagate its failure")

    expected = cache / (f"{audio.CONVERSION_RECIPE_VERSION}__{audio.file_sha256(source)}.wav")
    assert not expected.exists()
    assert not list(cache.glob("*.tmp.wav"))


def test_concurrent_conversion_publishes_one_complete_cache_entry(monkeypatch, tmp_path):
    source = tmp_path / "stereo.wav"
    cache = tmp_path / "cache"
    _write(source, _pcm(8), channels=2)
    temporary_outputs = _fake_ffmpeg(monkeypatch, barrier=threading.Barrier(2))

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: audio.to_16k_mono(source, cache), range(2)))

    assert outputs[0] == outputs[1]
    assert len(temporary_outputs) == 2
    assert audio.canonical_audio_id(outputs[0]).startswith("pcm16:16000:1:")
    assert not list(cache.glob("*.tmp.wav"))


def test_prepared_audio_preserves_reference_and_source_paths(tmp_path):
    source = tmp_path / "clip.wav"
    _write(source, _pcm(10))

    prepared = audio.prepare_audio(
        source,
        tmp_path / "cache",
        reference_id="dataset-row-42",
    )

    assert prepared.source_path == source
    assert prepared.prepared_path == source
    assert prepared.reference_id == "dataset-row-42"
