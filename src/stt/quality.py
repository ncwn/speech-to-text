"""Detect repeated spans that CER alone does not describe.

:func:`find_loops` finds long character n-grams repeated more often than the
reference, or more than once when no reference is available.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from stt.burmese import normalize

#: Long enough to avoid reporting common short phrases as decoder loops.
DEFAULT_K = 30


@dataclass(frozen=True)
class LoopSite:
    """One repeated span, with how often it appears here and in the reference."""

    text: str
    count: int
    reference_count: int

    @property
    def excess(self) -> int:
        """Repetitions beyond what the reference justifies."""
        return self.count - max(self.reference_count, 1)


def find_loops(
    hypothesis: str,
    reference: str | None = None,
    k: int = DEFAULT_K,
    min_count: int = 2,
) -> list[LoopSite]:
    """Find spans the model repeated more than the reference does.

    Overlapping n-grams from one loop are collapsed to a single site, so four
    copies of a clause report as one finding rather than a dozen.
    """
    hyp = normalize(hypothesis)
    if len(hyp) < k:
        return []
    counts = Counter(hyp[i : i + k] for i in range(len(hyp) - k + 1))

    ref_counts: Counter[str] = Counter()
    if reference is not None:
        ref = normalize(reference)
        ref_counts = Counter(ref[i : i + k] for i in range(len(ref) - k + 1))

    candidates = [
        (gram, n, ref_counts.get(gram, 0))
        for gram, n in counts.items()
        if n >= min_count and n > ref_counts.get(gram, 0)
    ]
    candidates.sort(key=lambda c: (-(c[1] - c[2]), -c[1]))

    # One loop yields many overlapping n-grams — a 40-character clause repeated
    # four times matches at eleven different offsets. Reporting each as its own
    # defect would tell a user they have eleven problems when they have one, so
    # candidates are collapsed by *where* they occur rather than by how similar
    # their text looks: a comparison on text alone misses pairs that overlap by
    # less than they differ.
    sites: list[LoopSite] = []
    covered: list[tuple[int, int]] = []
    for gram, n, ref_n in candidates:
        spans = _occurrences(hyp, gram)
        if any(
            any(start < c_end and c_start < end for c_start, c_end in covered)
            for start, end in spans
        ):
            continue
        sites.append(LoopSite(gram, n, ref_n))
        covered.extend(spans)
    return sites


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    """Every ``(start, end)`` at which ``needle`` appears, including overlaps."""
    spans: list[tuple[int, int]] = []
    start = text.find(needle)
    while start != -1:
        spans.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    return spans


def loop_summary(sites: list[LoopSite], k: int = DEFAULT_K) -> str:
    """One-line human summary, or an empty string when the output is clean."""
    if not sites:
        return ""
    chars = sum(s.excess * k for s in sites)
    return f"{len(sites)} repeated span(s), ~{chars} excess characters"
