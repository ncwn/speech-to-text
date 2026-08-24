"""Verify the committed Metal System Trace corroboration."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "evidence/instruments/observer-fast-profile-v3.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intervals(path: Path) -> tuple[int, list[tuple[int, int]]]:
    root = ET.parse(path).getroot()
    values: dict[tuple[str, str], int] = {}
    spans: list[tuple[int, int]] = []
    for row in root.iter("row"):
        found: list[int] = []
        for tag in ("start-time", "duration"):
            node = row.find(tag)
            if node is None:
                break
            if node.get("id") and node.text:
                value = int(node.text)
                values[(tag, str(node.get("id")))] = value
                found.append(value)
            elif node.get("ref"):
                found.append(values[(tag, str(node.get("ref")))])
        if len(found) == 2:
            spans.append((found[0], found[1]))
    return len(spans), sorted(set(spans))


def test_metal_trace_export_matches_the_corpus_phase():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    trace = summary["metal_system_trace"]
    time_path = ROOT / trace["time_info_export"]["path"]
    mps_path = ROOT / trace["mps_hw_export"]["path"]
    assert _sha256(time_path) == trace["time_info_export"]["sha256"]
    assert _sha256(mps_path) == trace["mps_hw_export"]["sha256"]

    time_root = ET.parse(time_path).getroot()
    mabs = int(time_root.find(".//mach-absolute-time").text)  # type: ignore[union-attr]
    timebase = time_root.findall(".//mach-timebase-info-field")
    numer, denom = (int(item.text) for item in timebase)
    epoch_ns = Fraction(mabs * numer, denom)
    phase = summary["corpus_phase"]
    phase_start = float((Fraction(phase["perf_counter_started_ns"]) - epoch_ns) / 1_000_000_000)
    phase_end = float((Fraction(phase["perf_counter_ended_ns"]) - epoch_ns) / 1_000_000_000)
    assert phase_start == pytest.approx(phase["trace_relative_start_s"])
    assert phase_end == pytest.approx(phase["trace_relative_end_s"])

    row_count, intervals = _intervals(mps_path)
    mps = trace["mps_hw_export"]
    assert row_count == mps["row_count"]
    assert len(intervals) == mps["distinct_intervals"]
    first = min(start for start, _ in intervals) / 1_000_000_000
    last = max(start + duration for start, duration in intervals) / 1_000_000_000
    merged: list[tuple[int, int]] = []
    for start, duration in intervals:
        end = start + duration
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    active_union = sum(end - start for start, end in merged) / 1_000_000_000
    assert first == pytest.approx(mps["first_start_s"])
    assert last == pytest.approx(mps["last_end_s"])
    assert active_union == pytest.approx(mps["active_union_s"])
    assert phase_start <= first < last <= phase_end

    for path in (SUMMARY, time_path, mps_path):
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/Volumes/" not in text
