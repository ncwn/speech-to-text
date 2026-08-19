"""Common result types shared by every backend, plus output writers.

Every backend returns ``TranscriptionResult`` objects so that the CLI, the
evaluation harness, and any future backend (Dolphin, ElevenLabs Scribe v2,
Google Chirp 3) all speak the same shape.

A result carries the whole transcript as ``text`` and, where the backend can
supply it, a list of :class:`Segment` locating each piece of that transcript in
time. ``text`` remains the source of truth for scoring; segments are additive.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: How a segment's timings were obtained. Recorded so that a run mixing
#: backends stays honest about which timings were measured and which inferred.
#:
#: ``native``   the decoder emitted the timing itself (CrispASR, Whisper)
#: ``chunk``    the timing is the window we fed the model, not a word boundary
#: ``aligned``  recovered after the fact by forced alignment (see :mod:`stt.align`)
SegmentSource = str


@dataclass
class Segment:
    """One timed span of a transcript."""

    text: str
    #: Seconds from the start of the file.
    start: float
    end: float
    #: 0..1, higher is better. ``None`` when the backend offers no signal.
    confidence: float | None = None
    #: Reserved for diarization; nothing populates this yet.
    speaker: str | None = None
    source: SegmentSource = "chunk"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptionResult:
    """One audio file transcribed by one backend."""

    audio_path: str
    text: str
    backend: str
    model: str
    language: str | None = None
    # Wall-clock seconds spent inside the backend for this file.
    elapsed_s: float | None = None
    # Duration of the source audio, used to compute the real-time factor.
    audio_duration_s: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Timed breakdown of ``text``, when the backend can provide one. ``None``
    #: means "not available", which is different from "the audio was silent".
    segments: list[Segment] | None = None

    @property
    def rtf(self) -> float | None:
        """Real-time factor: seconds of compute per second of audio.

        Lower is faster; 1.0 means transcription takes as long as the clip.
        """
        if not self.elapsed_s or not self.audio_duration_s:
            return None
        return self.elapsed_s / self.audio_duration_s

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)  # nested Segment dataclasses flatten to dicts here
        d["rtf"] = self.rtf
        return d


def write_jsonl(results: list[TranscriptionResult], path: Path) -> None:
    """Write results as JSON Lines — one record per audio file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def write_text(results: list[TranscriptionResult], path: Path) -> None:
    """Write plain transcripts, one per line, in input order.

    This is the human-facing artifact, so model spacing artifacts are tidied
    here. The JSONL keeps the raw decoder output, which is what scoring reads.
    """
    from stt.burmese import tidy_spacing

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(tidy_spacing(r.text) + "\n")


def read_jsonl(path: Path) -> list[TranscriptionResult]:
    """Read back a JSONL file written by :func:`write_jsonl`.

    Tolerates files written before a field existed, so older result files stay
    loadable as the schema grows.
    """
    results: list[TranscriptionResult] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.pop("rtf", None)  # derived property, not a constructor field
            segments = d.pop("segments", None)
            results.append(
                TranscriptionResult(
                    **d,
                    segments=[Segment(**s) for s in segments] if segments else None,
                )
            )
    return results


# ------------------------------------------------------------------ subtitles


def _timestamp(seconds: float, decimal: str = ",") -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (WebVTT)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{ms:03d}"


def _cues(result: TranscriptionResult) -> list[Segment]:
    """Subtitle-ready segments: non-empty text, in time order."""
    from stt.burmese import tidy_spacing

    if not result.segments:
        return []
    cues = [
        Segment(
            text=tidy_spacing(s.text).strip(),
            start=s.start,
            end=s.end,
            confidence=s.confidence,
            speaker=s.speaker,
            source=s.source,
        )
        for s in result.segments
    ]
    return sorted((c for c in cues if c.text), key=lambda c: (c.start, c.end))


def write_srt(result: TranscriptionResult, path: Path) -> int:
    """Write one result as SubRip. Returns the number of cues written."""
    cues = _cues(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for index, cue in enumerate(cues, start=1):
            f.write(f"{index}\n")
            f.write(f"{_timestamp(cue.start)} --> {_timestamp(cue.end)}\n")
            f.write(f"{cue.text}\n\n")
    return len(cues)


def write_vtt(result: TranscriptionResult, path: Path) -> int:
    """Write one result as WebVTT. Returns the number of cues written."""
    cues = _cues(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for cue in cues:
            f.write(f"{_timestamp(cue.start, '.')} --> {_timestamp(cue.end, '.')}\n")
            f.write(f"{cue.text}\n\n")
    return len(cues)
