"""Windowed decoding — the shared loop behind Seamless and Dolphin."""

from __future__ import annotations

import numpy as np
import pytest

from stt.audio import find_audio, join_segments, windowed

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


# --- directory scanning ------------------------------------------------------


def test_find_audio_reports_what_it_skipped(tmp_path):
    """A short file count used to be the only clue that files were dropped."""
    for name in ("a.wav", "b.mov", "c.aiff", "notes.txt", ".DS_Store"):
        (tmp_path / name).touch()

    files, skipped = find_audio([tmp_path])

    assert sorted(p.name for p in files) == ["a.wav", "b.mov", "c.aiff"]
    assert skipped == []


def test_find_audio_flags_genuinely_unknown_extensions(tmp_path):
    (tmp_path / "a.wav").touch()
    (tmp_path / "recording.xyz").touch()

    files, skipped = find_audio([tmp_path])

    assert [p.name for p in files] == ["a.wav"]
    assert [p.name for p in skipped] == ["recording.xyz"]


def test_an_explicit_file_is_taken_whatever_its_extension(tmp_path):
    odd = tmp_path / "recording.xyz"
    odd.touch()

    files, skipped = find_audio([odd])

    assert files == [odd]
    assert skipped == []


def test_find_audio_still_raises_on_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_audio([tmp_path / "nope"])
