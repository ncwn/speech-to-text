"""Spend the expensive model only where the cheap one is unsure.

Error is concentrated in the least-confident spans, so escalating a minority of
the audio captures most of the strong model's advantage. Two rules fall out of
the measurements and both are enforced here:

**Switch in coarse blocks.** Each splice seam costs characters, because the two
models' segment boundaries do not coincide. Fragmentation, not the escalation
itself, is what ruins a naive cascade.

**Merge what is adjacent.** Neighbouring escalated blocks become one interval,
so consecutive escalations cost one seam rather than two.

See ``docs/findings.md#routing`` and ``#seam-tax``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from stt.audio import is_valid_audio_id
from stt.results import Segment, TranscriptionResult

#: Blocks are this many base segments wide. Coarse on purpose: finer routing
#: loses more to the seam tax than it gains. docs/findings.md#seam-tax
DEFAULT_BLOCK = 16

#: Share of audio duration handed to the expensive model, at the knee of the
#: measured curve. docs/findings.md#routing
DEFAULT_ESCALATE = 0.3


@dataclass(frozen=True)
class Interval:
    """A span of audio to re-transcribe with the expensive model."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class RouteInputError(ValueError):
    """Raised when two runs cannot support trustworthy routing."""


def _trust_issues(result: TranscriptionResult) -> tuple[str, ...]:
    """Normalize a record's trust issues without trusting its JSON shape."""
    raw = getattr(result, "trust_issues", ())
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        return (raw,)
    try:
        return tuple(str(issue) for issue in raw)
    except TypeError:
        return (str(raw),)


@dataclass(frozen=True)
class RoutePair:
    """A base and strong result proven to describe the same waveform."""

    audio_id: str
    base: TranscriptionResult
    strong: TranscriptionResult
    trust_issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.trust_issues


@dataclass(frozen=True)
class RoutePreparation:
    """Validated route pairs plus diagnostics for inputs that were omitted."""

    pairs: tuple[RoutePair, ...]
    skipped_audio_ids: tuple[str, ...] = ()
    trust_issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.skipped_audio_ids and not self.trust_issues


def _index_route_run(
    results: Sequence[TranscriptionResult], label: str
) -> tuple[dict[str, TranscriptionResult], list[str]]:
    indexed: dict[str, TranscriptionResult] = {}
    issues: list[str] = []
    for result in results:
        audio_id = result.audio_id
        if not is_valid_audio_id(audio_id):
            issues.append(f"{label}: result lacks a verified audio_id")
            continue
        assert audio_id is not None  # narrowed by is_valid_audio_id
        if audio_id in indexed:
            raise RouteInputError(f"{label}: duplicate audio_id {audio_id}")
        indexed[audio_id] = result
    return indexed, issues


def _pair_problem(base: TranscriptionResult, strong: TranscriptionResult) -> str | None:
    """Return the first reason a matched pair cannot be routed."""
    if base.error:
        return f"base failed: {base.error}"
    if strong.error:
        return f"strong failed: {strong.error}"
    if not base.segments:
        return "base run has no segments"
    if not strong.segments:
        return "strong run has no segments"
    if all(segment.confidence is None for segment in base.segments):
        return "base run carries no confidence"
    if base.audio_duration_s is None or base.audio_duration_s <= 0:
        return "base run has no valid audio duration"
    if strong.audio_duration_s is None or strong.audio_duration_s <= 0:
        return "strong run has no valid audio duration"
    if not math.isclose(base.audio_duration_s, strong.audio_duration_s, abs_tol=1e-6):
        return "paired runs disagree on audio duration"
    return None


def prepare_route_pairs(
    base: Sequence[TranscriptionResult],
    strong: Sequence[TranscriptionResult],
    *,
    allow_partial: bool = False,
) -> RoutePreparation:
    """Preflight two runs and join them exclusively by verified ``audio_id``.

    Strict mode rejects the operation before callers write output unless both
    runs have identical, trusted, error-free, routeable coverage and no
    constituent trust issues. Partial mode returns only covered routeable pairs
    and records why other base IDs were skipped; every returned pair is
    explicitly untrusted.
    """
    base_by_id, base_issues = _index_route_run(base, "base")
    strong_by_id, strong_issues = _index_route_run(strong, "strong")
    issues = [*base_issues, *strong_issues]
    if not base_by_id:
        issues.append("base run contains no verified audio IDs")
    if not strong_by_id:
        issues.append("strong run contains no verified audio IDs")

    base_ids = set(base_by_id)
    strong_ids = set(strong_by_id)
    missing = base_ids - strong_ids
    extra = strong_ids - base_ids
    if missing or extra:
        issues.append(
            f"coverage differs ({len(missing)} missing from strong, {len(extra)} extra in strong)"
        )

    pair_problems: dict[str, str] = {}
    for audio_id in base_ids & strong_ids:
        problem = _pair_problem(base_by_id[audio_id], strong_by_id[audio_id])
        if problem:
            pair_problems[audio_id] = problem
        if not base_by_id[audio_id].trusted:
            issues.append(f"base: {audio_id} is untrusted")
        if not strong_by_id[audio_id].trusted:
            issues.append(f"strong: {audio_id} is untrusted")
        for label, result in (
            ("base", base_by_id[audio_id]),
            ("strong", strong_by_id[audio_id]),
        ):
            result_issues = _trust_issues(result)
            if result_issues:
                issues.append(f"{label}: {audio_id} has trust issues: {'; '.join(result_issues)}")

    if not allow_partial and (issues or pair_problems):
        details = [
            *issues,
            *(f"{audio_id}: {problem}" for audio_id, problem in pair_problems.items()),
        ]
        raise RouteInputError("; ".join(details))

    global_issues = ["partial mode enabled", *issues] if allow_partial else []
    pairs: list[RoutePair] = []
    skipped = set(missing) | set(pair_problems)
    for result in base:
        audio_id = result.audio_id
        if (
            not is_valid_audio_id(audio_id)
            or audio_id not in strong_by_id
            or audio_id in pair_problems
        ):
            continue
        assert audio_id is not None
        pair_issues = list(global_issues)
        if not result.trusted:
            pair_issues.append("base input is untrusted")
        if not strong_by_id[audio_id].trusted:
            pair_issues.append("strong input is untrusted")
        pair_issues.extend(
            f"{label} input trust issue: {issue}"
            for label, candidate in (
                ("base", result),
                ("strong", strong_by_id[audio_id]),
            )
            for issue in _trust_issues(candidate)
        )
        pairs.append(
            RoutePair(
                audio_id=audio_id,
                base=result,
                strong=strong_by_id[audio_id],
                trust_issues=tuple(dict.fromkeys(pair_issues)),
            )
        )

    return RoutePreparation(
        pairs=tuple(pairs),
        skipped_audio_ids=tuple(audio_id for audio_id in base_by_id if audio_id in skipped),
        trust_issues=tuple(dict.fromkeys(global_issues)),
    )


def _blocks(count: int, block_size: int) -> list[range]:
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    return [range(i, min(i + block_size, count)) for i in range(0, count, block_size)]


def plan(
    segments: list[Segment],
    audio_duration_s: float,
    *,
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
    if not segments or escalate <= 0 or audio_duration_s <= 0:
        return []

    blocks = _blocks(len(segments), block_size)

    def uncertainty(block: range) -> float:
        scores = [
            segments[i].confidence if segments[i].confidence is not None else 0.0 for i in block
        ]
        return sum(scores) / len(scores)

    chosen: set[int] = set()
    for block in sorted(blocks, key=uncertainty):
        candidate = chosen | set(block)
        candidate_intervals = _merge_adjacent(segments, candidate, audio_duration_s)
        candidate_cost = sum(interval.duration for interval in candidate_intervals)
        # The merged spans are what the strong model actually receives. Their
        # duration includes gaps between adjacent segments and is therefore the
        # only honest budget cost. The first block is always allowed so a small
        # non-zero request does not silently become a no-op.
        if chosen and candidate_cost > escalate * audio_duration_s:
            continue
        chosen = candidate

    return _merge_adjacent(segments, chosen, audio_duration_s)


def _merge_adjacent(
    segments: list[Segment], chosen: set[int], audio_duration_s: float
) -> list[Interval]:
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
    return merge_intervals(intervals, audio_duration_s)


def merge_intervals(intervals: Iterable[Interval], audio_duration_s: float) -> list[Interval]:
    """Clip intervals to the audio bounds and return their non-overlapping union."""
    if audio_duration_s <= 0:
        return []
    clipped = sorted(
        (
            Interval(max(0.0, interval.start), min(audio_duration_s, interval.end))
            for interval in intervals
        ),
        key=lambda interval: (interval.start, interval.end),
    )
    merged: list[Interval] = []
    for interval in clipped:
        if interval.duration <= 0:
            continue
        if not merged or interval.start > merged[-1].end:
            merged.append(interval)
        else:
            previous = merged[-1]
            merged[-1] = Interval(previous.start, max(previous.end, interval.end))
    return merged


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


def escalated_seconds(audio_duration_s: float, intervals: Sequence[Interval]) -> float:
    """Seconds of unique, in-bounds audio handed to the expensive model."""
    return sum(interval.duration for interval in merge_intervals(intervals, audio_duration_s))


def escalated_share(audio_duration_s: float, intervals: Sequence[Interval]) -> float:
    """Fraction of actual audio duration covered by the interval union."""
    if audio_duration_s <= 0:
        return 0.0
    return escalated_seconds(audio_duration_s, intervals) / audio_duration_s


def corpus_escalated_share(
    routed: Iterable[tuple[float, Sequence[Interval]]],
) -> float:
    """Duration-weighted escalation share across a corpus of audio files."""
    total_audio = 0.0
    total_escalated = 0.0
    for audio_duration_s, intervals in routed:
        if audio_duration_s <= 0:
            continue
        total_audio += audio_duration_s
        total_escalated += escalated_seconds(audio_duration_s, intervals)
    if total_audio <= 0:
        return 0.0
    return total_escalated / total_audio
