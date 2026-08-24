"""ROVER-style voting across transcription runs.

Hypotheses are aligned to a pivot and voted position by position. The pivot
should be the strongest input because voting can only correct characters that
it proposed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import jiwer

from stt.burmese import tidy_spacing


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

    ``prepare`` normalizes model-specific spacing before alignment. The default
    keeps Burmese delimiters that full scoring normalization would remove.
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
