"""The long-audio sentinel is deterministic and checked in by hash."""

from pathlib import Path

import soundfile as sf

from stt.sentinel import annotation, pcm16_sha256, verify_annotation, waveform


def test_checked_in_sentinel_annotation_matches_generated_fixture():
    path = Path(__file__).parents[1] / "data" / "sentinels" / "long-audio-boundary-v1.json"

    assert verify_annotation(path) == []
    assert annotation()["pcm16_sha256"]
    assert waveform().shape == (720_000,)


def test_checked_in_wav_has_the_declared_pcm_identity():
    path = Path(__file__).parents[1] / "data" / "sentinels" / "long-audio-boundary-v1.wav"
    samples, sample_rate = sf.read(path, dtype="float32")

    assert sample_rate == 16_000
    assert pcm16_sha256(samples) == annotation()["pcm16_sha256"]
