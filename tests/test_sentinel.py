"""The long-audio sentinel is deterministic and checked in by hash."""

from pathlib import Path

from stt.sentinel import annotation, verify_annotation, waveform


def test_checked_in_sentinel_annotation_matches_generated_fixture():
    path = Path(__file__).parents[1] / "data" / "sentinels" / "long-audio-boundary-v1.json"

    assert verify_annotation(path) == []
    assert annotation()["pcm16_sha256"]
    assert waveform().shape == (720_000,)
