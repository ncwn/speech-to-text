"""Routing between a cheap model and an expensive one."""

from __future__ import annotations

import hashlib
import json

from typer.testing import CliRunner

from stt.cascade import (
    Interval,
    RouteInputError,
    corpus_escalated_share,
    escalated_share,
    merge_intervals,
    plan,
    prepare_route_pairs,
    stitch,
)
from stt.cli import app
from stt.results import Segment, TranscriptionResult, read_jsonl, write_jsonl


def _segs(confidences: list[float | None], length: float = 1.0) -> list[Segment]:
    return [
        Segment(f"s{i}", i * length, (i + 1) * length, confidence=c)
        for i, c in enumerate(confidences)
    ]


def test_no_escalation_budget_means_no_work():
    assert plan(_segs([0.1] * 8), 8.0, escalate=0.0) == []
    assert plan([], 8.0, escalate=0.5) == []


def test_the_least_confident_block_is_chosen_first():
    segments = _segs([0.9, 0.9, 0.1, 0.1])
    assert plan(segments, 4.0, escalate=0.5, block_size=2) == [Interval(2.0, 4.0)]


def test_a_budget_is_never_overshot_by_a_whole_block():
    """Checking the budget after adding would turn a 30% request into 100%."""
    segments = _segs([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    intervals = plan(segments, 6.0, escalate=0.5, block_size=2)
    assert escalated_share(6.0, intervals) <= 0.5


def test_the_most_uncertain_block_is_taken_even_if_it_exceeds_the_budget():
    """Otherwise a coarse block size makes a small request silently a no-op."""
    segments = _segs([0.1, 0.1, 0.9, 0.9])
    intervals = plan(segments, 4.0, escalate=0.01, block_size=2)
    assert intervals == [Interval(0.0, 2.0)]


def test_escalation_is_bounded_by_audio_duration_not_segment_count():
    """A budget in seconds is what makes the compute cost predictable."""
    segments = [
        Segment("long", 0.0, 90.0, confidence=0.1),
        Segment("a", 90.0, 91.0, confidence=0.2),
        Segment("b", 91.0, 92.0, confidence=0.3),
    ]
    # The first block alone already blows a 25% budget, so nothing follows it.
    intervals = plan(segments, 92.0, escalate=0.25, block_size=1)
    assert intervals == [Interval(0.0, 90.0)]


def test_adjacent_escalated_blocks_merge_into_one_interval():
    """Every interval boundary is a seam, and seams cost characters."""
    segments = _segs([0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
    intervals = plan(segments, 6.0, escalate=0.7, block_size=2)
    assert intervals == [Interval(0.0, 4.0)]


def test_separated_regions_stay_separate():
    segments = _segs([0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
    intervals = plan(segments, 6.0, escalate=0.7, block_size=2)
    assert intervals == [Interval(0.0, 2.0), Interval(4.0, 6.0)]


def test_missing_confidence_is_treated_as_maximally_uncertain():
    """A backend that cannot say how sure it is should not be trusted."""
    segments = _segs([None, None, 0.5, 0.5])
    assert plan(segments, 4.0, escalate=0.5, block_size=2) == [Interval(0.0, 2.0)]


def test_a_block_size_below_one_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="at least 1"):
        plan(_segs([0.1, 0.2]), 2.0, escalate=0.5, block_size=0)


def test_gap_in_a_merged_block_counts_against_the_budget():
    segments = [
        Segment("a", 0.0, 1.0, confidence=0.1),
        Segment("b", 10.0, 11.0, confidence=0.1),
        Segment("c", 12.0, 13.0, confidence=0.2),
    ]

    intervals = plan(segments, 20.0, escalate=0.6, block_size=2)

    # The first block costs eleven seconds, not two. Adding c would cost
    # thirteen seconds and exceed the twelve-second budget.
    assert intervals == [Interval(0.0, 11.0)]
    assert escalated_share(20.0, intervals) == 0.55


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
    assert escalated_share(10.0, [Interval(0.0, 3.0)]) == 0.3
    assert escalated_share(10.0, []) == 0.0


def test_interval_cost_is_clipped_and_deduplicated():
    intervals = [Interval(-2.0, 4.0), Interval(3.0, 7.0), Interval(9.0, 12.0)]
    assert merge_intervals(intervals, 10.0) == [Interval(0.0, 7.0), Interval(9.0, 10.0)]
    assert escalated_share(10.0, intervals) == 0.8


def test_corpus_share_is_weighted_by_audio_seconds():
    routed = [
        (1.0, [Interval(0.0, 1.0)]),
        (9.0, []),
    ]
    assert corpus_escalated_share(routed) == 0.1


# --- the `stt route` CLI command ---------------------------------------------


def _run(audio, model, segments):
    digest = hashlib.sha256(str(audio).encode()).hexdigest()
    return TranscriptionResult(
        audio_path=audio,
        source_path=audio,
        source_sha256=digest,
        reference_id=str(audio),
        audio_id=f"pcm16:16000:1:{digest}",
        text=" ".join(s.text for s in segments) if segments else "",
        backend="test",
        model=model,
        audio_duration_s=float(segments[-1].end) if segments else 0.0,
        segments=segments or None,
        trusted=True,
    )


def _seg(text, start, end, confidence=None):
    return Segment(text=text, start=start, end=end, confidence=confidence)


def test_route_pairs_join_by_audio_id_even_when_paths_differ():
    segments = [_seg("x", 0, 1, confidence=0.5)]
    base = _run("base.wav", "cheap", segments)
    strong = _run("strong.wav", "strong", segments)
    strong.audio_id = base.audio_id

    prepared = prepare_route_pairs([base], [strong])

    assert len(prepared.pairs) == 1
    assert prepared.pairs[0].base.audio_path == "base.wav"
    assert prepared.pairs[0].strong.audio_path == "strong.wav"
    assert prepared.complete


def test_route_pairs_do_not_fall_back_to_identical_paths():
    segments = [_seg("x", 0, 1, confidence=0.5)]
    base = _run("same.wav", "cheap", segments)
    strong = _run("same.wav", "strong", segments)
    strong.audio_id = f"pcm16:16000:1:{'f' * 64}"

    import pytest

    with pytest.raises(RouteInputError, match="coverage differs"):
        prepare_route_pairs([base], [strong])


def test_route_strict_preflight_rejects_failures_and_untrusted_inputs():
    segments = [_seg("x", 0, 1, confidence=0.5)]
    base = _run("same.wav", "cheap", segments)
    strong = _run("same.wav", "strong", segments)
    base.trusted = False
    strong.error = "decoder failed"

    import pytest

    with pytest.raises(RouteInputError) as exc:
        prepare_route_pairs([base], [strong])

    assert "untrusted" in str(exc.value)
    assert "decoder failed" in str(exc.value)


def test_route_strict_preflight_rejects_trusted_inputs_with_trust_issues():
    segments = [_seg("x", 0, 1, confidence=0.5)]
    base = _run("same.wav", "cheap", segments)
    strong = _run("same.wav", "strong", segments)
    base.trust_issues = ["stale provenance"]

    import pytest

    with pytest.raises(RouteInputError, match="stale provenance"):
        prepare_route_pairs([base], [strong])


def test_malformed_trusted_value_is_rejected_after_jsonl_load(tmp_path):
    segments = [_seg("x", 0, 1, confidence=0.5)]
    record = _run("same.wav", "cheap", segments).to_dict()
    record["trusted"] = "false"
    path = tmp_path / "malformed-trusted.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    (base,) = read_jsonl(path)
    strong = _run("strong.wav", "strong", segments)
    strong.audio_id = base.audio_id

    import pytest

    with pytest.raises(RouteInputError, match="trusted field must be a JSON boolean"):
        prepare_route_pairs([base], [strong])


def test_partial_route_propagates_constituent_trust_issues():
    segments = [_seg("x", 0, 1, confidence=0.5)]
    base = _run("same.wav", "cheap", segments)
    strong = _run("same.wav", "strong", segments)
    base.trust_issues = ["stale provenance"]

    prepared = prepare_route_pairs([base], [strong], allow_partial=True)

    assert len(prepared.pairs) == 1
    assert not prepared.pairs[0].complete
    assert "base input trust issue: stale provenance" in prepared.pairs[0].trust_issues


def test_partial_route_returns_only_covered_routeable_pairs_with_issues():
    segments = [_seg("x", 0, 1, confidence=0.5)]
    base_a = _run("a.wav", "cheap", segments)
    base_b = _run("b.wav", "cheap", segments)
    base_c = _run("c.wav", "cheap", segments)
    strong_a = _run("a.wav", "strong", segments)
    strong_b = _run("b.wav", "strong", segments)
    strong_b.error = "decoder failed"

    prepared = prepare_route_pairs(
        [base_a, base_b, base_c],
        [strong_a, strong_b],
        allow_partial=True,
    )

    assert [pair.audio_id for pair in prepared.pairs] == [base_a.audio_id]
    assert set(prepared.skipped_audio_ids) == {base_b.audio_id, base_c.audio_id}
    assert "partial mode enabled" in prepared.pairs[0].trust_issues
    assert not prepared.complete


def test_route_pairs_require_real_matching_audio_duration():
    segments = [_seg("x", 0, 1, confidence=0.5)]
    base = _run("same.wav", "cheap", segments)
    strong = _run("same.wav", "strong", segments)
    strong.audio_duration_s = 2.0

    import pytest

    with pytest.raises(RouteInputError, match="disagree on audio duration"):
        prepare_route_pairs([base], [strong])


def _invoke(*args):
    return CliRunner().invoke(app, ["route", *[str(a) for a in args]])


def test_route_splices_the_strong_model_into_unconfident_spans(tmp_path):
    base = [_seg(f"b{i}", i, i + 1, confidence=0.1 if i < 2 else 0.9) for i in range(4)]
    strong = [_seg(f"s{i}", i, i + 1) for i in range(4)]
    base_p, strong_p, out = tmp_path / "b.jsonl", tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    write_jsonl([_run("a.wav", "cheap", base)], base_p)
    write_jsonl([_run("a.wav", "strong", strong)], strong_p)

    result = _invoke(base_p, strong_p, "-o", out, "--escalate", "0.5", "--block", "2")

    assert result.exit_code == 0, result.output
    (routed,) = read_jsonl(out)
    # The two least-confident segments come from the strong run, the rest do not.
    assert routed.text == "s0 s1 b2 b3"
    assert routed.metadata["seams"] == 1
    assert not routed.trusted
    assert routed.model_provenance is None
    assert "derived route lacks complete model provenance" in routed.trust_issues


def test_route_refuses_a_base_run_without_segments(tmp_path):
    base_p, strong_p, out = tmp_path / "b.jsonl", tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    write_jsonl([_run("a.wav", "cheap", None)], base_p)
    write_jsonl([_run("a.wav", "strong", [_seg("s0", 0, 1)])], strong_p)

    result = _invoke(base_p, strong_p, "-o", out)

    assert result.exit_code == 1
    assert "no segments" in result.output
    assert not out.exists()


def test_route_refuses_when_no_segment_carries_confidence(tmp_path):
    """Missing confidence reads as 0.0 to `plan`, which would escalate everything."""
    base_p, strong_p, out = tmp_path / "b.jsonl", tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    write_jsonl([_run("a.wav", "cheap", [_seg("b0", 0, 1), _seg("b1", 1, 2)])], base_p)
    write_jsonl([_run("a.wav", "strong", [_seg("s0", 0, 1), _seg("s1", 1, 2)])], strong_p)

    result = _invoke(base_p, strong_p, "-o", out)

    assert result.exit_code == 1
    assert "no confidence" in result.output


def test_route_requires_complete_coverage_unless_partial_is_explicit(tmp_path):
    base_p, strong_p, out = tmp_path / "b.jsonl", tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    covered = [_seg("b0", 0, 1, confidence=0.2), _seg("b1", 1, 2, confidence=0.9)]
    write_jsonl(
        [_run("a.wav", "cheap", covered), _run("b.wav", "cheap", covered)],
        base_p,
    )
    write_jsonl([_run("a.wav", "strong", [_seg("s0", 0, 1), _seg("s1", 1, 2)])], strong_p)

    result = _invoke(base_p, strong_p, "-o", out)

    assert result.exit_code == 1
    assert "coverage differs" in result.output
    assert not out.exists()

    partial = _invoke(base_p, strong_p, "-o", out, "--allow-partial")

    assert partial.exit_code == 0, partial.output
    assert "partial" in partial.output
    (routed,) = read_jsonl(out)
    assert routed.audio_path == "a.wav"
    assert not routed.trusted
