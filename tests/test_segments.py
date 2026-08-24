"""Segment schema, JSONL round-tripping, and subtitle output."""

from __future__ import annotations

import json

from stt.results import (
    Segment,
    TranscriptionResult,
    read_jsonl,
    write_jsonl,
    write_srt,
    write_vtt,
)


def _result(**kw) -> TranscriptionResult:
    base = dict(audio_path="a.wav", text="ကမ္ဘာ", backend="b", model="m")
    return TranscriptionResult(**{**base, **kw})


def test_segments_survive_a_jsonl_round_trip(tmp_path):
    result = _result(
        segments=[
            Segment("ကမ္ဘာ", 0.0, 1.5, confidence=0.9, source="native"),
            Segment("မြန်မာ", 1.5, 3.0, speaker="S1"),
        ]
    )
    path = tmp_path / "r.jsonl"
    write_jsonl([result], path)

    back = read_jsonl(path)[0]
    assert back.segments is not None
    assert [s.text for s in back.segments] == ["ကမ္ဘာ", "မြန်မာ"]
    assert back.segments[0].confidence == 0.9
    assert back.segments[0].source == "native"
    assert back.segments[1].speaker == "S1"
    assert back.segments[1].source == "chunk"  # default


def test_results_without_optional_segments_load(tmp_path):
    path = tmp_path / "result.jsonl"
    path.write_text(
        json.dumps(
            {
                "audio_path": "a.wav",
                "text": "ကမ္ဘာ",
                "backend": "b",
                "model": "m",
                "language": None,
                "elapsed_s": 1.0,
                "audio_duration_s": 2.0,
                "error": None,
                "metadata": {},
                "rtf": 0.5,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    back = read_jsonl(path)[0]
    assert back.segments is None
    assert back.text == "ကမ္ဘာ"
    assert back.rtf == 0.5


def test_absent_and_empty_segments_are_both_falsy(tmp_path):
    path = tmp_path / "r.jsonl"
    write_jsonl([_result(segments=None), _result(segments=[])], path)
    back = read_jsonl(path)
    assert back[0].segments is None
    assert not back[1].segments


def test_srt_numbers_cues_and_formats_timestamps(tmp_path):
    result = _result(
        segments=[
            Segment("တစ်", 0.0, 1.25),
            Segment("နှစ်", 1.25, 3661.5),  # crosses an hour
        ]
    )
    path = tmp_path / "s.srt"
    assert write_srt(result, path) == 2

    body = path.read_text(encoding="utf-8")
    assert body.startswith("1\n00:00:00,000 --> 00:00:01,250\nတစ်\n")
    assert "2\n00:00:01,250 --> 01:01:01,500\nနှစ်\n" in body


def test_vtt_has_a_header_and_dotted_timestamps(tmp_path):
    result = _result(segments=[Segment("တစ်", 0.0, 1.25)])
    path = tmp_path / "s.vtt"
    assert write_vtt(result, path) == 1

    body = path.read_text(encoding="utf-8")
    assert body.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.250" in body


def test_cues_are_sorted_and_blank_segments_dropped(tmp_path):
    result = _result(
        segments=[
            Segment("ဒုတိယ", 5.0, 6.0),
            Segment("   ", 2.0, 3.0),  # whitespace only
            Segment("ပထမ", 0.0, 1.0),
        ]
    )
    path = tmp_path / "s.srt"
    assert write_srt(result, path) == 2

    body = path.read_text(encoding="utf-8")
    assert body.index("ပထမ") < body.index("ဒုတိယ")
    assert "   \n" not in body


def test_subtitles_from_an_untimed_result_are_empty_not_an_error(tmp_path):
    path = tmp_path / "s.srt"
    assert write_srt(_result(segments=None), path) == 0
    assert path.read_text(encoding="utf-8") == ""


# --------------------------------------------- backend confidence conversion


def test_an_uncomputed_no_speech_probability_becomes_unknown():
    """CrispASR's omniasr-llm backend reports -1.0 when it did not measure.

    Passing that straight through as `1 - (-1)` yields a confidence of 2.0,
    which would quietly corrupt any routing decision made from it.
    """
    from stt.backends.omniasr_gguf import _speech_confidence

    assert _speech_confidence(-1.0) is None
    assert _speech_confidence(1.5) is None


def test_a_real_no_speech_probability_inverts_to_confidence():
    from stt.backends.omniasr_gguf import _speech_confidence

    assert _speech_confidence(0.0) == 1.0
    assert _speech_confidence(0.25) == 0.75
    assert _speech_confidence(1.0) == 0.0
