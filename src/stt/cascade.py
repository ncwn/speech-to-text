"""Spend the expensive model on coarse, low-confidence spans.

Coarse blocks limit transcript seams, and adjacent selected blocks are merged.
Historical measurements behind the defaults live in ``docs/findings.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from stt.results import Segment

#: Coarse blocks reduce joins between independently segmented transcripts.
DEFAULT_BLOCK = 16

#: Historical default share of audio handed to the expensive model.
DEFAULT_ESCALATE = 0.3


@dataclass(frozen=True)
class Interval:
    """A span of audio to re-transcribe with the expensive model."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _blocks(count: int, block_size: int) -> list[range]:
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    return [range(i, min(i + block_size, count)) for i in range(0, count, block_size)]


def plan(
    segments: list[Segment],
    escalate: float = DEFAULT_ESCALATE,
    block_size: int = DEFAULT_BLOCK,
) -> list[Interval]:
    """Choose which spans to re-transcribe, least confident first.

    ``escalate`` is a share of total audio *duration*, not of segment count, so
    the compute it authorises is predictable regardless of how long individual
    segments happen to be.

    Segments with no confidence are treated as maximally uncertain: a backend
    that cannot tell us how sure it is should not be trusted by default.
    """
    if not segments or escalate <= 0:
        return []

    blocks = _blocks(len(segments), block_size)
    total = sum(s.duration for s in segments)
    if total <= 0:
        return []

    def uncertainty(block: range) -> float:
        scores = [
            segments[i].confidence if segments[i].confidence is not None else 0.0 for i in block
        ]
        return sum(scores) / len(scores)

    chosen: set[int] = set()
    spent = 0.0
    for block in sorted(blocks, key=uncertainty):
        duration = sum(segments[i].duration for i in block)
        # Stop before exceeding the budget rather than after. Testing on the way
        # out lets one final block overshoot by its whole width, which on coarse
        # blocks can turn a request for 30% of the compute into 100% of it.
        # The most uncertain block is always taken, so a non-zero request never
        # silently does nothing.
        if chosen and (spent + duration) / total > escalate:
            break
        chosen.update(block)
        spent += duration

    return _merge_adjacent(segments, chosen)


def _merge_adjacent(segments: list[Segment], chosen: set[int]) -> list[Interval]:
    """Collapse runs of chosen segments into single intervals.

    Each interval boundary is a seam, and seams cost characters, so adjacent
    escalations must not be handed over as separate spans.
    """
    intervals: list[Interval] = []
    run_start: int | None = None
    for i in range(len(segments)):
        if i in chosen and run_start is None:
            run_start = i
        elif i not in chosen and run_start is not None:
            intervals.append(Interval(segments[run_start].start, segments[i - 1].end))
            run_start = None
    if run_start is not None:
        intervals.append(Interval(segments[run_start].start, segments[-1].end))
    return intervals


def stitch(
    base: list[Segment],
    strong: list[Segment],
    intervals: list[Interval],
) -> list[Segment]:
    """Replace the escalated spans of ``base`` with ``strong``'s take on them.

    A segment belongs to an interval when its midpoint does — using the midpoint
    rather than either edge means every segment lands in exactly one place, so
    nothing is dropped at a boundary or emitted twice.
    """
    if not intervals:
        return list(base)

    def inside(segment: Segment) -> bool:
        middle = (segment.start + segment.end) / 2
        return any(i.start <= middle < i.end for i in intervals)

    kept = [s for s in base if not inside(s)]
    replaced = [s for s in strong if inside(s)]
    return sorted(kept + replaced, key=lambda s: (s.start, s.end))


def escalated_share(segments: list[Segment], intervals: list[Interval]) -> float:
    """Fraction of total audio duration the intervals cover — the compute bill."""
    total = sum(s.duration for s in segments)
    if total <= 0:
        return 0.0
    return sum(i.duration for i in intervals) / total
