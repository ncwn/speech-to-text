"""Reference-free defect detection for ASR output.

CER answers "how many characters are wrong". It does not answer "did the
decoder get stuck", and the two come apart badly on long audio: a sentence
emitted four times costs only a few percent of CER but tells you the model
lost the plot, and everything downstream of it is suspect.

Two failure modes matter for the models in this repo:

* **Tight loops** — ``ဖြစ်တဲ့ဖြစ်တဲ့ဖြစ်တဲ့``, a handful of syllables cycling.
  4-bit quantised decoders do this.
* **Sentence loops** — a whole clause repeated verbatim at a distance. Full
  precision models do this instead, and a short sliding window will not see it.

:func:`find_loops` catches both by looking for long character n-grams that
appear more often than they do in the reference (or, with no reference, more
than once at all).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from stt.burmese import normalize

#: Long enough that a repeat is a decoder loop rather than a common turn of
#: phrase. Burmese averages roughly four characters per syllable, so 30
#: characters is on the order of a clause.
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

    sites: list[LoopSite] = []
    for gram, n, ref_n in candidates:
        # Two n-grams from the same loop overlap heavily; keep the first only.
        if any(gram[: k // 2] in seen.text or seen.text[: k // 2] in gram for seen in sites):
            continue
        sites.append(LoopSite(gram, n, ref_n))
    return sites


def loop_summary(sites: list[LoopSite], k: int = DEFAULT_K) -> str:
    """One-line human summary, or an empty string when the output is clean."""
    if not sites:
        return ""
    chars = sum(s.excess * k for s in sites)
    return f"{len(sites)} repeated span(s), ~{chars} excess characters"
