"""Common result types shared by every backend, plus output writers.

Every backend returns ``TranscriptionResult`` objects so that the CLI, the
evaluation harness, and any future backend (Dolphin, ElevenLabs Scribe v2,
Google Chirp 3) all speak the same shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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

    @property
    def rtf(self) -> float | None:
        """Real-time factor: seconds of compute per second of audio.

        Lower is faster; 1.0 means transcription takes as long as the clip.
        """
        if not self.elapsed_s or not self.audio_duration_s:
            return None
        return self.elapsed_s / self.audio_duration_s

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
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
    """Read back a JSONL file written by :func:`write_jsonl`."""
    results: list[TranscriptionResult] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.pop("rtf", None)  # derived property, not a constructor field
            results.append(TranscriptionResult(**d))
    return results
