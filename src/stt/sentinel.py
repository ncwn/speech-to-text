"""Deterministic long-audio boundary sentinel and annotation verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stt.measurement import MeasurementError

SENTINEL_KIND = "long-audio-sentinel-v1"
SAMPLE_RATE = 16_000
DURATION_S = 45.0
BOUNDARIES_S = (20.0, 40.0)


@dataclass(frozen=True)
class SentinelSpan:
    label: str
    start_s: float
    end_s: float

    def validate(self) -> None:
        if not self.label or self.start_s < 0 or self.end_s <= self.start_s:
            raise MeasurementError("invalid sentinel span")
        if self.end_s > DURATION_S:
            raise MeasurementError("sentinel span exceeds fixture duration")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "label": self.label,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


SPANS = (
    SentinelSpan("before-boundary", 18.75, 19.75),
    SentinelSpan("across-boundary", 19.75, 20.75),
    SentinelSpan("after-boundary", 20.75, 21.75),
    SentinelSpan("before-second-boundary", 38.75, 39.75),
    SentinelSpan("across-second-boundary", 39.75, 40.75),
    SentinelSpan("after-second-boundary", 40.75, 41.75),
)


def waveform() -> np.ndarray:
    """Return the fixed mono float32 waveform used by the sentinel."""
    samples = np.zeros(round(DURATION_S * SAMPLE_RATE), dtype=np.float32)
    for index, span in enumerate(SPANS):
        start = round(span.start_s * SAMPLE_RATE)
        end = round(span.end_s * SAMPLE_RATE)
        timeline = np.arange(end - start, dtype=np.float32) / SAMPLE_RATE
        frequency = 220.0 + index * 37.0
        tone = 0.22 * np.sin(2.0 * np.pi * frequency * timeline)
        envelope = np.minimum(1.0, timeline * 20.0) * np.minimum(
            1.0, (span.end_s - span.start_s - timeline) * 20.0
        )
        samples[start:end] += tone * envelope
    return np.clip(samples, -1.0, 1.0)


def pcm16_sha256(samples: np.ndarray | None = None) -> str:
    values = waveform() if samples is None else np.asarray(samples, dtype=np.float32)
    pcm16 = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    return hashlib.sha256(pcm16.tobytes()).hexdigest()


def annotation() -> dict[str, Any]:
    for span in SPANS:
        span.validate()
    return {
        "artifact_kind": SENTINEL_KIND,
        "sample_rate": SAMPLE_RATE,
        "duration_s": DURATION_S,
        "boundaries_s": list(BOUNDARIES_S),
        "pcm16_sha256": pcm16_sha256(),
        "spans": [span.to_dict() for span in SPANS],
    }


def verify_annotation(path: Path) -> list[str]:
    """Verify a checked-in annotation against the generated fixture."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = annotation()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot read sentinel annotation: {exc}"]
    if value != expected:
        return ["sentinel annotation differs from the deterministic fixture"]
    return []


__all__ = [
    "BOUNDARIES_S",
    "DURATION_S",
    "SAMPLE_RATE",
    "SENTINEL_KIND",
    "SPANS",
    "annotation",
    "pcm16_sha256",
    "verify_annotation",
    "waveform",
]
