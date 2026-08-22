"""Resource measurement around a transcription."""

from __future__ import annotations

import json

import pytest

from stt.results import TranscriptionResult, read_jsonl, write_jsonl
from stt.telemetry import Profiler, ResourceUsage, Series, describe_host, measure, peak_rss_mb


def test_measure_reports_wall_and_cpu_time():
    with measure() as usage:
        sum(i * i for i in range(200_000))
    u = usage[0]
    assert u.wall_s > 0
    assert u.cpu_s > 0
    assert u.peak_rss_mb > 0


def test_busy_work_reports_at_least_one_half_core():
    with measure() as usage:
        sum(i * i for i in range(2_000_000))
    # The exact total includes any native worker threads owned by the process.
    assert usage[0].cpu_utilization > 0.5


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


def test_resource_series_can_be_omitted_from_load_metadata():
    usage = ResourceUsage(
        wall_s=1.0,
        cpu_s=1.0,
        peak_rss_mb=1.0,
        cpu=Series.from_samples([1.0]),
    )
    compact = usage.to_dict(include_series=False)
    assert "cpu" not in compact
    assert compact["cpu_utilization"] == 1.0


def test_resources_survive_a_jsonl_round_trip(tmp_path):
    result = TranscriptionResult(
        audio_path="a.wav",
        text="ကမ္ဘာ",
        backend="b",
        model="m",
        resources=ResourceUsage(
            wall_s=2.0,
            cpu_s=8.0,
            peak_rss_mb=512.0,
            gpu_mb=64.0,
            gpu_util=Series.from_samples([0, 50, 100], idle_threshold=1.0),
        ),
    )
    path = tmp_path / "r.jsonl"
    write_jsonl([result], path)

    back = read_jsonl(path)[0].resources
    assert back is not None
    assert back.cpu_s == 8.0
    assert back.gpu_mb == 64.0
    assert back.cpu_utilization == 4.0  # recomputed, not read back as a field
    assert back.gpu_util and back.gpu_util.samples == [0.0, 50.0, 100.0]


def test_results_without_resources_still_load(tmp_path):
    path = tmp_path / "r.jsonl"
    write_jsonl([TranscriptionResult("a.wav", "x", "b", "m")], path)
    assert read_jsonl(path)[0].resources is None


def test_host_description_identifies_the_machine():
    host = describe_host()
    assert host["cpu_count"] and host["cpu_count"] > 0
    assert host["machine"]


def test_series_statistics_keep_count_shape_and_raw_samples():
    series = Series.from_samples([0, 10, 20, 30, 100], idle_threshold=0)
    assert series.n == 5
    assert series.mean == 32.0
    assert series.p50 == 20.0
    assert series.p95 == pytest.approx(86.0)
    assert series.max == 100.0
    assert series.idle_pct == 20.0
    assert series.samples == [0.0, 10.0, 20.0, 30.0, 100.0]


def test_timestamped_series_uses_time_weighted_summaries():
    series = Series.from_samples(
        [0, 100],
        offsets_ns=[10, 90],
        window_ns=100,
        idle_threshold=1,
        source="observer",
        scope="whole-device",
        phase="corpus",
    )

    assert series.mean == pytest.approx(50.0)
    assert series.idle_pct == pytest.approx(50.0)
    assert series.offsets_ns == [10, 90]
    assert series.window_ns == 100
    assert series.to_dict()["source"] == "observer"


def test_timestamped_series_round_trip_recomputes_with_recorded_threshold():
    original = Series.from_samples(
        [0, 2, 100], offsets_ns=[10, 20, 90], window_ns=100, idle_threshold=1
    )
    restored = Series.from_dict(original.to_dict())

    assert restored.mean == original.mean
    assert restored.idle_pct == original.idle_pct
    assert restored.offsets_ns == original.offsets_ns


def test_unknown_result_resource_series_and_segment_fields_warn_and_load(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text(
        json.dumps(
            {
                "audio_path": "a.wav",
                "text": "x",
                "backend": "b",
                "model": "m",
                "future_result": True,
                "segments": [
                    {
                        "text": "x",
                        "start": 0,
                        "end": 1,
                        "future_segment": True,
                    }
                ],
                "resources": {
                    "wall_s": 1,
                    "cpu_s": 1,
                    "peak_rss_mb": 1,
                    "future_resource": 1,
                    "cpu": {
                        "n": 1,
                        "mean": 1,
                        "p50": 1,
                        "p95": 1,
                        "max": 1,
                        "idle_pct": 0,
                        "samples": [1],
                        "future_series": 1,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning) as caught:
        loaded = read_jsonl(path)[0]
    assert len(caught) == 4
    assert loaded.text == "x"
    assert loaded.segments and loaded.segments[0].text == "x"
    assert loaded.resources and loaded.resources.cpu and loaded.resources.cpu.n == 1


# --- GPU sampling ------------------------------------------------------------


def test_sampler_cost_is_subtracted_from_cpu_seconds(monkeypatch):
    """Unsubtracted, ioreg's own CPU lands on the workload's bill."""
    from stt import telemetry

    class FakeSampler(telemetry.GpuSampler):
        def start(self):
            self.samples = [40.0, 80.0]
            self.cpu_cost_s = 0.25
            return self

        def stop(self):
            return None

    monkeypatch.setattr(telemetry, "GpuSampler", FakeSampler)
    with telemetry.measure(sample_gpu=True) as box:
        pass
    usage = box[0]

    assert usage.sampler_cpu_s == 0.25
    # cpu_s clamps at zero rather than going negative on a trivial block.
    assert usage.cpu_s >= 0.0
    assert usage.gpu_util_mean == 60.0
    assert usage.gpu_util_peak == 80.0


def test_no_gpu_fields_when_sampling_is_off():
    from stt.telemetry import measure

    with measure(sample_gpu=False) as box:
        pass
    usage = box[0]

    assert usage.gpu_util_mean is None
    assert usage.gpu_util_peak is None
    assert usage.sampler_cpu_s == 0.0
    assert "gpu_util_mean" not in usage.to_dict()


def test_unprofiled_measurement_still_reports_current_rss_window():
    with measure(profile=False, phase="corpus") as box:
        values = bytearray(1024)

    usage = box[0]
    assert values
    assert usage.started_ns is not None
    assert usage.ended_ns is not None
    assert usage.ended_ns >= usage.started_ns
    assert usage.rss_peak_mb is not None
    assert usage.process_peak_rss_mb == usage.peak_rss_mb
    assert usage.phase == "corpus"


def test_measure_synchronizes_before_and_after_work(monkeypatch):
    from stt import telemetry

    events: list[str] = []
    monkeypatch.setattr(
        telemetry, "synchronize_device", lambda device: events.append(f"sync:{device}")
    )

    with telemetry.measure(profile=False, device="mps"):
        events.append("work")

    assert events == ["sync:mps", "work", "sync:mps"]


def test_profiler_rejects_samples_that_finish_after_work_window(monkeypatch):
    profiler = Profiler(sample_gpu=True)
    profiler._started_ns = 100
    profiler._ended_ns = 200
    profiler.samples = [20.0]
    profiler.gpu_offsets_ns = [50]

    monkeypatch.setattr("stt.telemetry.time.perf_counter_ns", lambda: 250)
    monkeypatch.setattr("stt.telemetry.gpu_utilization_now", lambda: 99.0)
    profiler._sample_gpu()

    assert profiler.samples == [20.0]
    assert profiler.gpu_util and profiler.gpu_util.samples == [20.0]


def test_gpu_utilisation_returns_none_without_ioreg(monkeypatch):
    from stt import telemetry

    monkeypatch.setattr(telemetry.shutil, "which", lambda _: None)
    assert telemetry.gpu_utilization_now() is None


def test_saturation_names_the_busy_resource():
    from stt.telemetry import ResourceUsage, saturation

    def usage(cpu_s, wall_s=1.0, gpu=None):
        return ResourceUsage(wall_s=wall_s, cpu_s=cpu_s, peak_rss_mb=1.0, gpu_util_mean=gpu)

    assert saturation(usage(0.3, gpu=85.0), 8) == "GPU-bound"
    assert saturation(usage(7.0), 8) == "CPU-bound"
    assert saturation(usage(0.3, gpu=40.0), 8) == "GPU-light"
    assert saturation(usage(0.3), 8) == "underutilised"
    # A GPU-bound run is called that even when the CPU is nearly idle: that is
    # the GGUF case, and calling it "underutilised" was the old blind spot.
    assert saturation(usage(0.05, gpu=78.0), 12) == "GPU-bound"
