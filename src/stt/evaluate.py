"""Accuracy scoring for Burmese ASR output.

CER is the headline metric — see :mod:`stt.burmese` for why WER does not apply
to a script written without word delimiters. WER is still computed and reported
alongside it for languages that do use spaces, but for Burmese it should be
read as "roughly meaningless" rather than as a second opinion.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import jiwer

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
    def n_failed(self) -> int:
        return len(self.items) - len(self.scored)

    @property
    def cer(self) -> float:
        """Corpus CER — total edits over total reference characters.

        Weighted by length rather than averaged per utterance, so one short
        clip transcribed badly does not dominate the number.
        """
        items = self.scored
        if not items:
            return float("nan")
        total_chars = sum(i.ref_chars for i in items)
        if total_chars == 0:
            return float("nan")
        return sum(i.cer * i.ref_chars for i in items) / total_chars

    @property
    def wer(self) -> float:
        items = self.scored
        if not items:
            return float("nan")
        return sum(i.wer for i in items) / len(items)


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
    """Score a run against references, normalising both sides identically."""
    opts = options or NormalizeOptions()
    items: list[ScoredItem] = []

    for r in results:
        stem = Path(r.audio_path).stem
        # Audio normalised by stt.audio picks up a ".16k" suffix; strip it so
        # converted files still match their reference entry.
        lookup = references.get(stem) or references.get(stem.removesuffix(".16k"))

        if r.error:
            items.append(ScoredItem(r.audio_path, lookup or "", "", 1.0, 1.0, 0, error=r.error))
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
                    error=f"no reference for {stem!r}",
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
                )
            )
            continue

        cer = jiwer.cer(ref, hyp)
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
        wer = jiwer.wer(ref_spaced, hyp_spaced) if ref_spaced else 1.0

        items.append(
            ScoredItem(
                audio_path=r.audio_path,
                reference=lookup,
                hypothesis=r.text,
                cer=float(cer),
                wer=float(wer),
                ref_chars=len(ref),
            )
        )

    backend = results[0].backend if results else "?"
    model = results[0].model if results else "?"
    return Score(backend=backend, model=model, items=items)


def mean_rtf(results: list[TranscriptionResult]) -> float | None:
    """Average real-time factor across successfully transcribed files."""
    rtfs = [r.rtf for r in results if r.rtf is not None and not r.error]
    return sum(rtfs) / len(rtfs) if rtfs else None
