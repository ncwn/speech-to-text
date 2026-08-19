"""Resource measurement around a transcription."""

from __future__ import annotations

import json

from stt.results import TranscriptionResult, read_jsonl, write_jsonl
from stt.telemetry import ResourceUsage, describe_host, measure, peak_rss_mb


def test_measure_reports_wall_and_cpu_time():
    with measure() as usage:
        sum(i * i for i in range(200_000))
    u = usage[0]
    assert u.wall_s > 0
    assert u.cpu_s > 0
    assert u.peak_rss_mb > 0


def test_busy_work_on_one_thread_reports_about_one_core():
    with measure() as usage:
        sum(i * i for i in range(2_000_000))
    # Pure Python holds the GIL, so this cannot exceed a core by much.
    assert 0.5 < usage[0].cpu_utilization < 1.5


def test_a_block_that_raises_still_reports_what_it_used():
    """A failed decode consumed resources too, and we want to know."""
    try:
        with measure() as usage:
            sum(i * i for i in range(200_000))
            raise ValueError("boom")
    except ValueError:
        pass
    assert usage and usage[0].cpu_s > 0


def test_utilization_is_unknown_rather_than_infinite_for_zero_time():
    assert ResourceUsage(wall_s=0.0, cpu_s=0.0, peak_rss_mb=1.0).cpu_utilization is None


def test_peak_rss_is_reported_in_plausible_megabytes():
    """Guards the macOS-bytes vs Linux-kilobytes unit difference."""
    assert 1 < peak_rss_mb() < 1_000_000


def test_unmeasured_gpu_is_none_not_zero():
    """`None` means "no GPU here"; 0.0 would claim a measurement."""
    usage = ResourceUsage(wall_s=1.0, cpu_s=1.0, peak_rss_mb=1.0)
    assert usage.gpu_mb is None
    assert "gpu_mb" not in usage.to_dict()


def test_derived_utilization_is_written_but_is_not_a_field():
    d = ResourceUsage(wall_s=2.0, cpu_s=4.0, peak_rss_mb=1.0).to_dict()
    assert d["cpu_utilization"] == 2.0
    assert json.dumps(d)  # must stay JSON-serialisable


def test_resources_survive_a_jsonl_round_trip(tmp_path):
    result = TranscriptionResult(
        audio_path="a.wav",
        text="ကမ္ဘာ",
        backend="b",
        model="m",
        resources=ResourceUsage(wall_s=2.0, cpu_s=8.0, peak_rss_mb=512.0, gpu_mb=64.0),
    )
    path = tmp_path / "r.jsonl"
    write_jsonl([result], path)

    back = read_jsonl(path)[0].resources
    assert back is not None
    assert back.cpu_s == 8.0
    assert back.gpu_mb == 64.0
    assert back.cpu_utilization == 4.0  # recomputed, not read back as a field


def test_results_without_resources_still_load(tmp_path):
    path = tmp_path / "r.jsonl"
    write_jsonl([TranscriptionResult("a.wav", "x", "b", "m")], path)
    assert read_jsonl(path)[0].resources is None


def test_host_description_identifies_the_machine():
    host = describe_host()
    assert host["cpu_count"] and host["cpu_count"] > 0
    assert host["machine"]
