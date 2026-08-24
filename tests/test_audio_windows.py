"""Windowed decoding — the shared loop behind Seamless and Dolphin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stt import audio
from stt.audio import join_segments, windowed

RATE = 16_000


def _speech(seconds: float) -> np.ndarray:
    """A waveform loud enough that split_on_quiet has to look for a boundary."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.4 * np.sin(2 * np.pi * 220 * t)).astype("float32")


def test_short_audio_is_a_single_window():
    segments = windowed(_speech(3.0), RATE, 10.0, lambda _c: "x")
    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == pytest.approx(3.0)


def test_windows_tile_the_input_without_gaps_or_overlap():
    segments = windowed(_speech(25.0), RATE, 10.0, lambda _c: "x")
    assert len(segments) > 1
    assert segments[0].start == 0.0
    assert segments[-1].end == pytest.approx(25.0)
    for a, b in zip(segments, segments[1:], strict=False):
        assert a.end == pytest.approx(b.start)


def test_each_window_is_timed_by_where_it_came_from():
    """Without this the offsets are lost and subtitles are impossible."""
    segments = windowed(_speech(20.0), RATE, 5.0, lambda c: str(len(c)))
    for s in segments:
        assert s.end > s.start
        assert int(s.text) == pytest.approx(round((s.end - s.start) * RATE), abs=1)


def test_windows_are_marked_as_chunk_timings():
    """A window boundary is not a word boundary, and must not claim to be."""
    segments = windowed(_speech(5.0), RATE, 10.0, lambda _c: "x")
    assert segments[0].source == "chunk"


def test_a_sub_200ms_tail_is_not_decoded():
    """Models hallucinate a token for a sliver of audio."""
    calls: list[int] = []

    def decode(chunk):
        calls.append(len(chunk))
        return "x"

    windowed(_speech(10.05), RATE, 10.0, decode)
    assert all(n >= RATE * 0.2 for n in calls)


def test_windows_that_decode_to_nothing_are_dropped():
    assert windowed(_speech(20.0), RATE, 5.0, lambda _c: "   ") == []


def test_joining_reproduces_the_flat_transcript():
    segments = windowed(_speech(20.0), RATE, 5.0, lambda _c: "ကမ္ဘာ")
    assert join_segments(segments) == " ".join(["ကမ္ဘာ"] * len(segments))


def test_conversion_cache_separates_same_named_dataset_files(monkeypatch, tmp_path):
    sources = [tmp_path / name / "audio" / "clip.mp3" for name in ("one", "two")]
    for source in sources:
        source.parent.mkdir(parents=True)
        source.write_bytes(source.parent.parent.name.encode())

    calls: list[Path] = []

    def convert(command, *, check):
        assert check
        output = Path(command[-1])
        output.write_bytes(b"wav")
        calls.append(output)

    monkeypatch.setattr(audio, "needs_conversion", lambda _path: True)
    monkeypatch.setattr(audio.shutil, "which", lambda _name: "/opt/homebrew/bin/ffmpeg")
    monkeypatch.setattr(audio.subprocess, "run", convert)

    converted = [audio.to_16k_mono(source, tmp_path / "cache") for source in sources]

    assert converted[0] != converted[1]
    assert [path.name for path in converted] == ["clip.16k.wav", "clip.16k.wav"]
    assert audio.to_16k_mono(sources[0], tmp_path / "cache") == converted[0]
    assert calls == converted
