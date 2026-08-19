"""Apple Silicon capability detection."""

from __future__ import annotations

import sys

import pytest

from stt.hardware import CoreLayout, chip_name, compute_threads, core_layout, describe


def _layout(*levels: tuple[str, int]) -> CoreLayout:
    return CoreLayout(levels)


def test_efficiency_cores_are_excluded_from_compute():
    """An M2 Max is 8 performance + 4 efficiency; only the 8 are worth using."""
    layout = _layout(("Performance", 8), ("Efficiency", 4))
    assert layout.compute == 8
    assert layout.efficiency == 4
    assert layout.total == 12


def test_a_chip_without_efficiency_cores_uses_all_of_them():
    """An M5 Pro pairs super cores with performance cores and has no efficiency
    tier. Taking only the fastest level would idle two thirds of the CPU."""
    layout = _layout(("Super", 6), ("Performance", 12))
    assert layout.compute == 18
    assert layout.efficiency == 0


def test_level_names_decide_not_level_order():
    """Position is not meaningful — only whether a level is the efficiency one."""
    layout = _layout(("Efficiency", 4), ("Performance", 8))
    assert layout.compute == 8


def test_name_matching_is_case_insensitive():
    assert _layout(("performance", 4), ("efficiency", 4)).compute == 4


def test_a_single_level_machine_uses_every_core():
    assert _layout(("unknown", 16)).compute == 16


def test_an_all_efficiency_machine_still_reports_usable_cores():
    """Guards against returning zero threads, which would deadlock a pool."""
    assert _layout(("Efficiency", 4)).compute == 4


def test_compute_threads_is_positive_on_this_machine():
    assert compute_threads() >= 1


def test_real_layout_is_consistent():
    layout = core_layout()
    assert layout.total >= layout.compute >= 1
    assert layout.compute + layout.efficiency == layout.total


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Silicon only")
def test_chip_name_is_reported_on_macos():
    assert "Apple" in chip_name() or chip_name() == "unknown"


def test_describe_reports_what_a_benchmark_needs():
    d = describe()
    assert d["compute_threads"] >= 1
    assert d["cpu_count"] >= 1
    assert isinstance(d["cores"], dict)


# ------------------------------------------------------------------- dtype


def test_cpu_never_gets_half_precision():
    """PyTorch emulates half precision on CPU: 4.1x slower for identical text."""
    pytest.importorskip("torch")
    from stt.hardware import fastest_dtype

    assert fastest_dtype("cpu") == "float32"


def test_gpu_dtype_choice_is_between_the_two_half_formats():
    """float32 is timed as a control but must never be selected: holding a 7B
    card's weights at float32 would want ~28 GB on the GPU."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no Metal device")
    from stt.hardware import fastest_dtype

    assert fastest_dtype("mps") in {"float16", "bfloat16"}


def test_the_dtype_probe_result_is_cached(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no Metal device")
    import stt.hardware as hw

    monkeypatch.setattr(hw, "CACHE", tmp_path / "hw.json")
    calls: list[str] = []
    real = hw._probe_dtypes
    monkeypatch.setattr(hw, "_probe_dtypes", lambda *a, **k: (calls.append("x"), real(*a, **k))[1])

    first = hw.fastest_dtype("mps")
    second = hw.fastest_dtype("mps")
    assert first == second
    assert len(calls) == 1  # the second call came from cache
