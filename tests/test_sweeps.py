"""Recomputable stopping rules for bounded batch sweeps."""

from __future__ import annotations

import pytest

from stt.measurement import MeasurementError
from stt.sweeps import BatchObservation, BatchSweepSpec, summarize_batch_sweep, verify_batch_sweep


def _spec() -> BatchSweepSpec:
    return BatchSweepSpec("batch", "fake-model", "fixed", (1, 2, 4, 8), 8)


def _observation(size: int, rtf: float, *, spread: float = 0.01) -> BatchObservation:
    return BatchObservation(size, rtf, rtf - spread, rtf + spread, {"clip": "stable"})


def test_batch_sweep_stops_after_two_recomputable_plateaus():
    result = summarize_batch_sweep(
        _spec(),
        (
            _observation(1, 1.0),
            _observation(2, 0.80),
            _observation(4, 0.79),
            _observation(8, 0.78),
        ),
    )

    assert result["status"] == "stopped"
    assert result["stop_at"] == 8
    assert result["rows"][2]["plateau"]
    assert verify_batch_sweep(result) == []


def test_batch_sweep_rejects_transcript_changes():
    observations = (
        _observation(1, 1.0),
        BatchObservation(2, 0.5, 0.49, 0.51, {"clip": "changed"}),
        _observation(4, 0.4),
        _observation(8, 0.3),
    )
    with pytest.raises(MeasurementError, match="transcript hashes changed"):
        summarize_batch_sweep(_spec(), observations)


def test_batch_sweep_records_failed_condition_instead_of_raising():
    failed = BatchObservation(4, 1.0, 1.0, 1.0, {"clip": "stable"}, False, "out of memory")
    result = summarize_batch_sweep(
        _spec(), (_observation(1, 1.0), _observation(2, 0.8), failed, _observation(8, 0.7))
    )

    assert result["status"] == "failed"
    assert result["stop_at"] == 4
    assert verify_batch_sweep(result) == []


def test_batch_sweep_requires_declared_powers_of_two():
    with pytest.raises(MeasurementError, match="powers of two"):
        BatchSweepSpec("bad", "fake-model", "fixed", (1, 3), 3).validate()
