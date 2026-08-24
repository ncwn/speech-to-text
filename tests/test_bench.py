"""The regression gate's judgement, exercised without loading a model.

The thresholds encode a measured fact: transcripts are byte-deterministic, so
text is compared with no tolerance, while RTF varies ~2% run to run and is not.
"""

from __future__ import annotations

from stt.bench import compare, text_hash

HOST = {"chip": "Apple M2 Max", "torch": "2.8.0"}


def measurement(*, rtf=0.16, rss=1000.0, gpu=80.0, texts=("a", "b"), host=None, error=None):
    return {
        "host": host or HOST,
        "clips": ["clip1", "clip2"],
        "runs": [
            {
                "subject": "hf/seamless-m4t-v2",
                "hashes": {f"clip{i + 1}": text_hash(t) for i, t in enumerate(texts)},
                "rtf": rtf,
                "peak_rss_mb": rss,
                "gpu_util_mean": gpu,
                "gpu_util_p50": gpu,
                "gpu_util_n": 5,
                "error": error,
            }
        ],
    }


def test_an_unchanged_run_passes():
    assert compare(measurement(), measurement()).ok


def test_one_changed_transcript_fails_with_no_tolerance():
    verdict = compare(measurement(), measurement(texts=("a", "CHANGED")))
    assert not verdict.ok
    (failure,) = verdict.failures
    assert failure.signal == "text"
    assert "1/2" in failure.detail


def test_speed_drift_within_tolerance_passes():
    assert compare(measurement(rtf=0.16), measurement(rtf=0.20)).ok


def test_legacy_speed_regression_is_not_a_trusted_gate():
    verdict = compare(measurement(rtf=0.16), measurement(rtf=0.25))
    assert verdict.ok
    assert not verdict.compared_timings


def test_legacy_memory_is_not_a_trusted_gate():
    verdict = compare(measurement(rss=1000), measurement(rss=1800))
    assert verdict.ok
    assert not verdict.compared_timings


def test_legacy_gpu_is_descriptive_only():
    verdict = compare(measurement(gpu=80), measurement(gpu=20))
    assert verdict.ok
    assert not verdict.compared_timings


def test_gpu_comparison_is_skipped_when_samples_are_thin():
    baseline = measurement(gpu=80)
    fresh = measurement(gpu=20)
    baseline["runs"][0]["gpu_util_n"] = 2
    fresh["runs"][0]["gpu_util_n"] = 2
    verdict = compare(baseline, fresh)
    assert not [f for f in verdict.findings if f.signal == "gpu"]


def test_a_crashed_subject_is_a_failure():
    verdict = compare(measurement(), measurement(error="RuntimeError: boom"))
    assert [f.signal for f in verdict.failures] == ["error"]


def test_timings_are_ignored_on_a_different_host():
    """An RTF from another machine is a different number, not a worse one."""
    other = {"chip": "Apple M5 Pro", "torch": "2.8.0"}
    verdict = compare(measurement(rtf=0.16), measurement(rtf=0.90, host=other))
    assert verdict.ok
    assert not verdict.compared_timings
    assert "Comparing transcripts only" in (verdict.host_note or "")


def test_text_is_still_compared_on_a_different_host():
    other = {"chip": "Apple M5 Pro", "torch": "2.8.0"}
    verdict = compare(measurement(), measurement(texts=("a", "X"), host=other))
    assert [f.signal for f in verdict.failures] == ["text"]


def test_a_changed_clip_set_is_caught_rather_than_compared():
    fresh = measurement()
    fresh["clips"] = ["clip1", "clip9"]
    verdict = compare(measurement(), fresh)
    assert [f.signal for f in verdict.failures] == ["set"]


def test_a_new_subject_is_reported_without_failing():
    stored = measurement()
    stored["runs"] = []
    verdict = compare(stored, measurement())
    assert verdict.ok
    assert [f.signal for f in verdict.warnings] == ["new"]


def test_a_vanished_subject_fails_the_full_comparison():
    fresh = measurement()
    fresh["runs"] = []
    verdict = compare(measurement(), fresh)
    assert not verdict.ok
    assert [f.signal for f in verdict.failures] == ["missing"]


def test_changed_measurement_schema_disables_timing_comparison():
    baseline = measurement(rtf=0.1)
    fresh = measurement(rtf=9.0)
    fresh["schema_version"] = 2

    verdict = compare(baseline, fresh)

    assert verdict.ok
    assert not verdict.compared_timings
    assert "measurement schema changed" in (verdict.host_note or "")


def test_batch_size_change_skips_timing_comparison():
    baseline = measurement()
    fresh = measurement(rtf=1.0)
    fresh["batch_size"] = 8
    verdict = compare(baseline, fresh)
    assert verdict.ok
    assert not verdict.compared_timings
    assert "batch size changed" in (verdict.host_note or "")
