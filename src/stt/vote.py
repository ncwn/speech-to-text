"""ROVER-style voting across several transcription runs.

Different ASR systems fail on different words. On the held-out 17-minute
recording the best single model scores CER 0.0857, but an oracle that picked
the better of the top two per 200-character span would score 0.0571 — a third
of the remaining error is recoverable purely by choosing between hypotheses we
already have.

This implements the practical version of that: align every hypothesis to a
pivot, then vote position by position. Measured effect, with the system set and
weights chosen on FLEURS and verified on held-out audio:

    FLEURS 120 test clips   0.1017 -> 0.0930   (-8.6 %)
    held-out 16.8 min       0.0857 -> 0.0714   (-16.7 %)

The pivot matters: voting can only correct characters the pivot proposed, so
it should be the most accurate system available. Weaker systems still help by
outvoting the pivot where it is wrong, but a pool of only weak systems does
nothing — seamless+dolphin+mms scores exactly what seamless scores alone.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import jiwer

from stt.burmese import tidy_spacing

#: Sensible default weights, ordered by measured CER on FLEURS Burmese. Anything
#: not listed votes with weight 1.0. These are deliberately coarse — the ranking
#: matters far more than the exact values.
DEFAULT_WEIGHTS: dict[str, float] = {
    "omniASR_LLM_Unlimited_7B_v2": 2.0,
    "seamless-m4t-v2": 1.9,
    "omniASR_LLM_Unlimited_3B_v2": 1.4,
    "omniASR_LLM_Unlimited_300M_v2": 1.2,
    "small": 1.0,  # dolphin
    "mms-1b-all": 0.9,
}


def _columns(pivot: str, hypothesis: str) -> list[str]:
    """What ``hypothesis`` emits at each pivot position.

    Returns ``len(pivot) + 1`` slots. Slot ``i`` holds the characters aligned to
    ``pivot[i - 1]``; slot 0 holds anything inserted before the pivot starts.
    A deletion leaves its slot empty, which is what lets a majority vote *remove*
    a character the pivot hallucinated.
    """
    columns = [""] * (len(pivot) + 1)
    for chunk in jiwer.process_characters(pivot, hypothesis).alignments[0]:
        span = chunk.ref_end_idx - chunk.ref_start_idx
        text = hypothesis[chunk.hyp_start_idx : chunk.hyp_end_idx]
        if chunk.type == "insert":
            columns[chunk.ref_start_idx] += text
        elif chunk.type != "delete":
            for k in range(span):
                columns[chunk.ref_start_idx + k + 1] += text[k] if k < len(text) else ""
    return columns


def rover(
    hypotheses: dict[str, str],
    pivot: str,
    weights: dict[str, float] | None = None,
    prepare: Callable[[str], str] | None = tidy_spacing,
) -> str:
    """Combine ``hypotheses`` by weighted per-position vote against ``pivot``.

    ``pivot`` is a key of ``hypotheses`` and should be the most accurate system.
    Ties resolve in the pivot's favour, so adding a system can never make the
    result worse than the pivot on a position where nothing outvotes it.

    ``prepare`` runs on every hypothesis first, and it matters more than it
    looks: SeamlessM4T emits a space per sub-word while omniASR emits none, so
    on raw text the aligner spends its budget on spacing rather than on the
    characters being voted. Measured on held-out audio — raw 0.0764,
    :func:`~stt.burmese.tidy_spacing` 0.0718, full normalisation 0.0714. The
    default is ``tidy_spacing`` because it captures nearly all of that while
    keeping the ၊ and ။ delimiters that normalisation would throw away.
    """
    if pivot not in hypotheses:
        raise KeyError(f"pivot {pivot!r} is not one of {sorted(hypotheses)}")
    if prepare is not None:
        hypotheses = {k: prepare(v) for k, v in hypotheses.items()}
    base = hypotheses[pivot]
    if not base:
        return base

    w = {**(weights or {})}
    columns = {name: _columns(base, text) for name, text in hypotheses.items()}
    pivot_columns = columns[pivot]

    out: list[str] = []
    for i in range(len(base) + 1):
        votes: Counter[str] = Counter()
        for name, cols in columns.items():
            votes[cols[i]] += w.get(name, 1.0)
        winner = max(votes.items(), key=lambda kv: (kv[1], kv[0] == pivot_columns[i]))
        out.append(winner[0])
    return "".join(out)
