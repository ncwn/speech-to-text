"""Accuracy scoring for Burmese ASR output.

CER is the headline metric — see :mod:`stt.burmese` for why WER does not apply.
WER is computed alongside it for languages that do use spaces; for Burmese read
it as "roughly meaningless" rather than as a second opinion.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import jiwer

from stt.audio import is_valid_audio_id
from stt.burmese import NormalizeOptions, describe_encoding, is_zawgyi, normalize
from stt.results import TranscriptionResult


@dataclass
class ScoredItem:
    """One hypothesis scored against its reference."""

    audio_path: str
    reference: str
    hypothesis: str
    cer: float
    wer: float
    ref_chars: int
    error: str | None = None
    audio_id: str | None = None
    reference_id: str | None = None
    ref_words: int = 0
    char_errors: int = 0
    word_errors: int = 0
    elapsed_s: float | None = None
    audio_duration_s: float | None = None
    trusted: bool = False
    trust_issues: list[str] | None = None


@dataclass
class Score:
    """Corpus-level result for one backend/model run."""

    backend: str
    model: str
    items: list[ScoredItem]

    @property
    def scored(self) -> list[ScoredItem]:
        return [i for i in self.items if i.error is None]

    @property
    def total(self) -> int:
        """Number of expected results and unexpected records considered."""
        return len(self.items)

    @property
    def n_scored(self) -> int:
        return len(self.scored)

    @property
    def n_failed(self) -> int:
        return self.total - self.n_scored

    @property
    def coverage(self) -> float:
        return self.n_scored / self.total if self.total else 0.0

    @property
    def identity_complete(self) -> bool:
        """Whether every record has explicit, trusted join provenance."""
        return bool(self.items) and all(item.trusted for item in self.items)

    @property
    def timing_complete(self) -> bool:
        """Whether every scored item has a valid timing denominator."""
        return bool(self.scored) and all(_has_valid_timing(item) for item in self.scored)

    @property
    def complete(self) -> bool:
        """Whether headline metrics cover one complete, trusted population."""
        return (
            self.total > 0
            and self.n_failed == 0
            and self.identity_complete
            and self.timing_complete
        )

    @property
    def trusted(self) -> bool:
        """Compatibility-friendly name for evidence eligibility."""
        return self.complete

    @property
    def trust_issues(self) -> list[str]:
        issues = {issue for item in self.items for issue in (item.trust_issues or [])}
        if not self.items:
            issues.add("empty evaluation population")
        if self.n_failed:
            issues.add(f"{self.n_failed} result(s) failed scoring")
        if self.scored and not self.timing_complete:
            issues.add("one or more scored results lack valid timing")
        return sorted(issues)

    @property
    def partial_cer(self) -> float | None:
        """Diagnostic corpus CER over only successfully scored records.

        Weighted by length rather than averaged per utterance, so one short
        clip transcribed badly does not dominate the number.
        """
        items = self.scored
        if not items:
            return None
        total_chars = sum(i.ref_chars for i in items)
        if total_chars == 0:
            return None
        return sum(i.char_errors for i in items) / total_chars

    @property
    def partial_wer(self) -> float | None:
        """Diagnostic standard corpus WER over successfully scored records."""
        items = self.scored
        if not items:
            return None
        total_words = sum(i.ref_words for i in items)
        if total_words == 0:
            return None
        return sum(i.word_errors for i in items) / total_words

    @property
    def partial_rtf(self) -> float | None:
        """Diagnostic RTF over exactly the population used by partial CER.

        If even one scored result lacks timing, no RTF is returned. Silently
        dropping it would give CER and RTF different denominators again.
        """
        items = self.scored
        if not items or not all(_has_valid_timing(item) for item in items):
            return None
        elapsed = sum(item.elapsed_s or 0.0 for item in items)
        duration = sum(item.audio_duration_s or 0.0 for item in items)
        return elapsed / duration

    @property
    def cer(self) -> float | None:
        """Publishable corpus CER, suppressed for incomplete/untrusted runs."""
        return self.partial_cer if self.complete else None

    @property
    def wer(self) -> float | None:
        """Publishable standard corpus WER, suppressed when incomplete."""
        return self.partial_wer if self.complete else None

    @property
    def rtf(self) -> float | None:
        """Publishable corpus RTF over the same population as :attr:`cer`."""
        return self.partial_rtf if self.complete else None


def _has_valid_timing(item: ScoredItem) -> bool:
    return (
        item.elapsed_s is not None
        and item.elapsed_s >= 0
        and item.audio_duration_s is not None
        and item.audio_duration_s > 0
    )


def _legacy_reference_id(audio_path: str) -> str:
    stem = Path(audio_path).stem
    return stem.removesuffix(".16k")


def _identity_issues(result: TranscriptionResult) -> list[str]:
    issues = list(result.trust_issues)
    if not result.trusted:
        issues.append("result is not trusted")
    if not is_valid_audio_id(result.audio_id):
        issues.append("missing or invalid audio_id")
    if not result.reference_id:
        issues.append("missing reference_id")
    return list(dict.fromkeys(issues))


def load_references(path: Path) -> dict[str, str]:
    """Load a reference file keyed by audio stem.

    Accepts a two-column TSV (``audio_id<TAB>transcript``), with or without a
    header row. The key is matched against the audio file's stem, so
    ``foo.wav``, ``foo``, and ``data/x/foo.wav`` all resolve to the same entry.
    """
    refs: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            key, text = row[0].strip(), row[1].strip()
            if key.lower() in {"id", "audio", "audio_id", "file", "file_name"}:
                continue  # header
            refs[Path(key).stem] = text
    return refs


def score_results(
    results: list[TranscriptionResult],
    references: dict[str, str],
    options: NormalizeOptions | None = None,
    check_encoding: bool = True,
) -> Score:
    """Score a run against references, normalising both sides identically.

    Legacy path-based lookup remains available for diagnostic ``partial_*``
    metrics. Headline metrics require explicit trusted identities and complete
    coverage of the supplied reference population.
    """
    opts = options or NormalizeOptions()
    items: list[ScoredItem] = []
    matched_references: set[str] = set()
    seen_result_ids: set[str] = set()

    for r in results:
        result_reference_id = r.reference_id or _legacy_reference_id(r.audio_path)
        lookup = references.get(result_reference_id)
        identity_issues = _identity_issues(r)
        item_trusted = not identity_issues

        if result_reference_id in seen_result_ids:
            items.append(
                ScoredItem(
                    r.audio_path,
                    lookup or "",
                    r.text,
                    1.0,
                    1.0,
                    0,
                    error=f"duplicate result for reference {result_reference_id!r}",
                    audio_id=r.audio_id,
                    reference_id=result_reference_id,
                    elapsed_s=r.elapsed_s,
                    audio_duration_s=r.audio_duration_s,
                    trusted=False,
                    trust_issues=[*identity_issues, "duplicate reference_id"],
                )
            )
            continue
        seen_result_ids.add(result_reference_id)
        if lookup is not None:
            matched_references.add(result_reference_id)

        if r.error:
            items.append(
                ScoredItem(
                    r.audio_path,
                    lookup or "",
                    "",
                    1.0,
                    1.0,
                    0,
                    error=r.error,
                    audio_id=r.audio_id,
                    reference_id=result_reference_id,
                    elapsed_s=r.elapsed_s,
                    audio_duration_s=r.audio_duration_s,
                    trusted=False,
                    trust_issues=identity_issues,
                )
            )
            continue
        if lookup is None:
            items.append(
                ScoredItem(
                    r.audio_path,
                    "",
                    r.text,
                    1.0,
                    1.0,
                    0,
                    error=f"no reference for {result_reference_id!r}",
                    audio_id=r.audio_id,
                    reference_id=result_reference_id,
                    elapsed_s=r.elapsed_s,
                    audio_duration_s=r.audio_duration_s,
                    trusted=False,
                    trust_issues=[*identity_issues, "reference_id not present in reference set"],
                )
            )
            continue

        if check_encoding and r.text.strip():
            ref_zawgyi = is_zawgyi(lookup)
            hyp_zawgyi = is_zawgyi(r.text)
            if ref_zawgyi != hyp_zawgyi:
                # Scoring across encodings yields a meaningless ~1.0 CER.
                items.append(
                    ScoredItem(
                        r.audio_path,
                        lookup,
                        r.text,
                        float("nan"),
                        float("nan"),
                        0,
                        error=(
                            "encoding mismatch: reference is "
                            f"{describe_encoding(lookup)}, hypothesis is "
                            f"{describe_encoding(r.text)}. Convert one side before scoring."
                        ),
                        audio_id=r.audio_id,
                        reference_id=result_reference_id,
                        elapsed_s=r.elapsed_s,
                        audio_duration_s=r.audio_duration_s,
                        trusted=False,
                        trust_issues=identity_issues,
                    )
                )
                continue

        ref = normalize(lookup, opts)
        hyp = normalize(r.text, opts)

        if not ref:
            items.append(
                ScoredItem(
                    r.audio_path,
                    lookup,
                    r.text,
                    1.0,
                    1.0,
                    0,
                    error="empty reference after normalisation",
                    audio_id=r.audio_id,
                    reference_id=result_reference_id,
                    elapsed_s=r.elapsed_s,
                    audio_duration_s=r.audio_duration_s,
                    trusted=False,
                    trust_issues=identity_issues,
                )
            )
            continue

        character_score = jiwer.process_characters(ref, hyp)
        char_errors = (
            character_score.substitutions + character_score.deletions + character_score.insertions
        )
        # WER needs whitespace to tokenise on; recompute without stripping it.
        spaced = NormalizeOptions(
            nfc=opts.nfc,
            strip_whitespace=False,
            strip_punctuation=opts.strip_punctuation,
            normalize_digits=opts.normalize_digits,
            lowercase=opts.lowercase,
        )
        ref_spaced = normalize(lookup, spaced)
        hyp_spaced = normalize(r.text, spaced)
        word_score = jiwer.process_words(ref_spaced, hyp_spaced)
        word_errors = word_score.substitutions + word_score.deletions + word_score.insertions
        ref_words = word_score.hits + word_score.substitutions + word_score.deletions

        items.append(
            ScoredItem(
                audio_path=r.audio_path,
                reference=lookup,
                hypothesis=r.text,
                cer=float(character_score.cer),
                wer=float(word_score.wer),
                ref_chars=len(ref),
                audio_id=r.audio_id,
                reference_id=result_reference_id,
                ref_words=ref_words,
                char_errors=char_errors,
                word_errors=word_errors,
                elapsed_s=r.elapsed_s,
                audio_duration_s=r.audio_duration_s,
                trusted=item_trusted,
                trust_issues=identity_issues,
            )
        )

    for reference_id in references:
        if reference_id in matched_references:
            continue
        items.append(
            ScoredItem(
                audio_path="",
                reference=references[reference_id],
                hypothesis="",
                cer=1.0,
                wer=1.0,
                ref_chars=0,
                error=f"no result for reference {reference_id!r}",
                reference_id=reference_id,
                trusted=False,
                trust_issues=["missing result"],
            )
        )

    backend = results[0].backend if results else "?"
    model = results[0].model if results else "?"
    return Score(backend=backend, model=model, items=items)


def corpus_rtf(results: list[TranscriptionResult]) -> float | None:
    """Return the corpus real-time factor for successful transcriptions.

    A mean of per-file RTFs gives every clip the same weight, even when their
    durations differ.  Corpus RTF weights each file by its audio duration:
    total backend time divided by total audio time.
    """
    elapsed = 0.0
    duration = 0.0
    for result in results:
        if result.error:
            continue
        if result.elapsed_s is None or result.audio_duration_s is None:
            continue
        if result.audio_duration_s <= 0 or result.elapsed_s < 0:
            continue
        elapsed += result.elapsed_s
        duration += result.audio_duration_s
    return elapsed / duration if duration > 0 else None


def mean_rtf(results: list[TranscriptionResult]) -> float | None:
    """Compatibility alias for :func:`corpus_rtf`.

    The name is retained because the CLI and existing consumers import it;
    the calculation now follows the corpus-weighted definition.
    """
    return corpus_rtf(results)
