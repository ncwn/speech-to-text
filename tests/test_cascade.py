"""Routing between a cheap model and an expensive one."""

from __future__ import annotations

from stt.cascade import Interval, escalated_share, plan, stitch
from stt.results import Segment


def _segs(confidences: list[float | None], length: float = 1.0) -> list[Segment]:
    return [
        Segment(f"s{i}", i * length, (i + 1) * length, confidence=c)
        for i, c in enumerate(confidences)
    ]


def test_no_escalation_budget_means_no_work():
    assert plan(_segs([0.1] * 8), escalate=0.0) == []
    assert plan([], escalate=0.5) == []


def test_the_least_confident_block_is_chosen_first():
    segments = _segs([0.9, 0.9, 0.1, 0.1])
    assert plan(segments, escalate=0.5, block_size=2) == [Interval(2.0, 4.0)]


def test_a_budget_is_never_overshot_by_a_whole_block():
    """Checking the budget after adding would turn a 30% request into 100%."""
    segments = _segs([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    intervals = plan(segments, escalate=0.5, block_size=2)
    assert escalated_share(segments, intervals) <= 0.5


def test_the_most_uncertain_block_is_taken_even_if_it_exceeds_the_budget():
    """Otherwise a coarse block size makes a small request silently a no-op."""
    segments = _segs([0.1, 0.1, 0.9, 0.9])
    intervals = plan(segments, escalate=0.01, block_size=2)
    assert intervals == [Interval(0.0, 2.0)]


def test_escalation_is_bounded_by_audio_duration_not_segment_count():
    """A budget in seconds is what makes the compute cost predictable."""
    segments = [
        Segment("long", 0.0, 90.0, confidence=0.1),
        Segment("a", 90.0, 91.0, confidence=0.2),
        Segment("b", 91.0, 92.0, confidence=0.3),
    ]
    # The first block alone already blows a 25% budget, so nothing follows it.
    intervals = plan(segments, escalate=0.25, block_size=1)
    assert intervals == [Interval(0.0, 90.0)]


def test_adjacent_escalated_blocks_merge_into_one_interval():
    """Every interval boundary is a seam, and seams cost characters."""
    segments = _segs([0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
    intervals = plan(segments, escalate=0.7, block_size=2)
    assert intervals == [Interval(0.0, 4.0)]


def test_separated_regions_stay_separate():
    segments = _segs([0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
    intervals = plan(segments, escalate=0.7, block_size=2)
    assert intervals == [Interval(0.0, 2.0), Interval(4.0, 6.0)]


def test_missing_confidence_is_treated_as_maximally_uncertain():
    """A backend that cannot say how sure it is should not be trusted."""
    segments = _segs([None, None, 0.5, 0.5])
    assert plan(segments, escalate=0.5, block_size=2) == [Interval(0.0, 2.0)]


def test_a_block_size_below_one_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="at least 1"):
        plan(_segs([0.1, 0.2]), escalate=0.5, block_size=0)


# --------------------------------------------------------------------- stitch


def test_stitching_nothing_returns_the_base_unchanged():
    base = _segs([0.5, 0.5])
    assert stitch(base, _segs([0.9, 0.9]), []) == base


def test_escalated_spans_take_the_strong_model_and_the_rest_keeps_the_base():
    base = [Segment("base0", 0.0, 1.0), Segment("base1", 1.0, 2.0)]
    strong = [Segment("strong0", 0.0, 1.0), Segment("strong1", 1.0, 2.0)]
    out = stitch(base, strong, [Interval(1.0, 2.0)])
    assert [s.text for s in out] == ["base0", "strong1"]


def test_every_segment_lands_exactly_once():
    """Midpoint membership is what stops text being dropped or duplicated."""
    base = _segs([0.5] * 6)
    strong = [Segment(f"S{i}", i * 1.0, (i + 1) * 1.0) for i in range(6)]
    out = stitch(base, strong, [Interval(2.0, 4.0)])
    assert len(out) == 6
    assert [s.text for s in out] == ["s0", "s1", "S2", "S3", "s4", "s5"]


def test_stitched_output_stays_in_time_order():
    base = _segs([0.5] * 4)
    strong = [Segment(f"S{i}", i * 1.0, (i + 1) * 1.0) for i in range(4)]
    out = stitch(base, strong, [Interval(0.0, 1.0), Interval(3.0, 4.0)])
    assert [s.start for s in out] == [0.0, 1.0, 2.0, 3.0]


def test_escalated_share_reports_the_compute_bill():
    segments = _segs([0.5] * 10)
    assert escalated_share(segments, [Interval(0.0, 3.0)]) == 0.3
    assert escalated_share(segments, []) == 0.0
