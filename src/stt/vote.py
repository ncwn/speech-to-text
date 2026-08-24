"""ROVER-style voting across several transcription runs.

Different ASR systems fail on different words, so choosing between hypotheses
already in hand recovers accuracy no single model reaches. This aligns every
hypothesis to a pivot, then votes position by position.

The pivot matters: voting can only correct characters the pivot proposed, so it
should be the most accurate system available. A pool of only weak systems does
nothing. See ``docs/findings.md#voting``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jiwer

from stt.audio import is_valid_audio_id
from stt.burmese import tidy_spacing
from stt.results import TranscriptionResult

#: Ordered by measured CER on FLEURS Burmese; anything unlisted votes at 1.0.
#: Deliberately coarse — the ranking matters far more than the exact values.
DEFAULT_WEIGHTS: dict[str, float] = {
    "omniASR_LLM_Unlimited_7B_v2": 2.0,
    "seamless-m4t-v2": 1.9,
    "omniASR_LLM_Unlimited_3B_v2": 1.4,
    "omniASR_LLM_Unlimited_300M_v2": 1.2,
    "small": 1.0,  # dolphin
    "mms-1b-all": 0.9,
}


class VoteInputError(ValueError):
    """Raised when a set of runs cannot support a trustworthy vote."""


def verified_audio_id(result: TranscriptionResult) -> str | None:
    """Return a structurally valid canonical waveform ID, if one is present.

    This deliberately does not derive identity from ``audio_path``. Older JSONL
    records remain readable, but a filename or stem is not proof that two model
    runs consumed the same waveform.
    """
    audio_id = getattr(result, "audio_id", None)
    if is_valid_audio_id(audio_id):
        return audio_id
    return None


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
class VoteGroup:
    """All model results associated with one verified input waveform."""

    audio_id: str
    pivot_label: str
    expected_voters: tuple[str, ...]
    results: Mapping[str, TranscriptionResult]
    missing_voters: tuple[str, ...] = ()
    failed_voters: tuple[str, ...] = ()
    trust_issues: tuple[str, ...] = ()

    @property
    def pivot(self) -> TranscriptionResult:
        return self.results[self.pivot_label]

    @property
    def usable_results(self) -> dict[str, TranscriptionResult]:
        """Successful hypotheses only; failures must never vote as deletions."""
        return {label: result for label, result in self.results.items() if not result.error}

    @property
    def complete(self) -> bool:
        return not self.missing_voters and not self.failed_voters and not self.trust_issues

    @property
    def elapsed_s(self) -> float | None:
        """Total cost of all expected voters, when every timing is known."""
        if self.missing_voters or self.failed_voters:
            return None
        values = [self.results[label].elapsed_s for label in self.expected_voters]
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    def provenance(self) -> dict[str, object]:
        """Stable metadata for a derived vote result."""
        return {
            "pivot": self.pivot_label,
            "expected_voters": list(self.expected_voters),
            "used_voters": sorted(self.usable_results),
            "missing_voters": list(self.missing_voters),
            "failed_voters": list(self.failed_voters),
        }


def prepare_vote_groups(
    runs: Mapping[str, Sequence[TranscriptionResult]],
    pivot: str,
    *,
    allow_partial: bool = False,
) -> list[VoteGroup]:
    """Validate and join transcription runs by canonical waveform identity.

    Strict mode rejects the entire operation before callers write any output if
    coverage differs, a required result failed, or an input is untrusted or
    carries trust issues.
    Partial mode follows the pivot's ordered ID set, omits failed non-pivot
    hypotheses, and records every deficiency on the returned group. A failed
    pivot remains present so the caller can preserve it as an error result.
    """
    if pivot not in runs:
        raise VoteInputError(f"pivot {pivot!r} is not one of {sorted(runs)}")
    if len(runs) < 2:
        raise VoteInputError("need at least two runs to vote between")

    labels = tuple(runs)
    indexed: dict[str, dict[str, TranscriptionResult]] = {}
    run_issues: dict[str, list[str]] = {label: [] for label in labels}
    errors: list[str] = []

    for label, results in runs.items():
        by_id: dict[str, TranscriptionResult] = {}
        unidentified = 0
        for result in results:
            audio_id = verified_audio_id(result)
            if audio_id is None:
                unidentified += 1
                continue
            if audio_id in by_id:
                errors.append(f"{label}: duplicate audio_id {audio_id}")
                continue
            by_id[audio_id] = result
        if unidentified:
            issue = f"{label}: {unidentified} result(s) lack a verified audio_id"
            run_issues[label].append(issue)
            if not allow_partial or label == pivot:
                errors.append(issue)
        indexed[label] = by_id

    if errors:
        raise VoteInputError("; ".join(errors))

    pivot_ids = tuple(indexed[pivot])
    pivot_set = set(pivot_ids)
    if not pivot_ids:
        raise VoteInputError("pivot run contains no verified audio IDs")

    if not allow_partial:
        for label in labels:
            ids = set(indexed[label])
            missing = pivot_set - ids
            extra = ids - pivot_set
            if missing or extra:
                errors.append(
                    f"{label}: coverage differs from pivot "
                    f"({len(missing)} missing, {len(extra)} extra)"
                )
            for audio_id, result in indexed[label].items():
                if result.error:
                    errors.append(f"{label}: {audio_id} failed: {result.error}")
                if not getattr(result, "trusted", False):
                    errors.append(f"{label}: {audio_id} is untrusted")
                issues_for_result = _trust_issues(result)
                if issues_for_result:
                    errors.append(
                        f"{label}: {audio_id} has trust issues: {'; '.join(issues_for_result)}"
                    )
        if errors:
            raise VoteInputError("; ".join(errors))

    groups: list[VoteGroup] = []
    for audio_id in pivot_ids:
        present = {
            label: indexed[label][audio_id] for label in labels if audio_id in indexed[label]
        }
        missing = tuple(label for label in labels if label not in present)
        failed = tuple(label for label, result in present.items() if result.error)
        issues = ["partial mode enabled"] if allow_partial else []
        issues.extend(issue for label in labels for issue in run_issues[label])
        issues.extend(f"missing voter: {label}" for label in missing)
        issues.extend(f"failed voter: {label}" for label in failed)
        issues.extend(
            f"untrusted voter: {label}"
            for label, result in present.items()
            if not getattr(result, "trusted", False)
        )
        issues.extend(
            f"trust issue from voter {label}: {issue}"
            for label, result in present.items()
            for issue in _trust_issues(result)
        )
        groups.append(
            VoteGroup(
                audio_id=audio_id,
                pivot_label=pivot,
                expected_voters=labels,
                results=present,
                missing_voters=missing,
                failed_voters=failed,
                trust_issues=tuple(dict.fromkeys(issues)),
            )
        )
    return groups


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
    characters being voted. ``tidy_spacing`` is the default because it fixes
    that while keeping the ၊ and ။ delimiters that full normalisation would
    throw away. See ``docs/findings.md#voting``.
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
