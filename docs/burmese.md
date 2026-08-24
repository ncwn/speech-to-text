# Evaluating Burmese ASR

Three properties of written Burmese will silently corrupt an accuracy number if
the harness ignores them. Text normalisation and encoding detection live in
`src/stt/burmese.py`; `score_results()` applies the encoding guard.

## 1. No word delimiters — use CER, not WER

Burmese is written without spaces between words. Whatever spaces appear are
phrase-level, and both human annotators and ASR models place them
inconsistently. Word Error Rate therefore measures segmentation habits rather
than recognition quality; it routinely exceeds 1.0 on output that is nearly
perfect.

This repo reports **CER** as the headline metric and strips all whitespace
before comparing. WER is printed too, but labelled as not meaningful — treat it
as a diagnostic for whether a model is emitting spaces at all, nothing more.

Corpus CER is length-weighted (total edits ÷ total reference characters) rather
than an average of per-clip rates, so one badly-transcribed short clip cannot
dominate the score.

## 2. Zawgyi vs Unicode

Two incompatible encodings occupy the same Myanmar code block. Zawgyi assigns
different code points to the same visual glyphs and does not follow Unicode's
storage order. Text that renders identically can be entirely different bytes.

Scoring a Unicode reference against a Zawgyi hypothesis yields a meaningless CER
near 1.0. Rather than report that number, `score_results()` detects the
encoding of both sides (using Google's `myanmartools` Markov detector) and marks
the item as an error when they disagree.

Check any file or string:

```bash
uv run stt check-encoding data/fleurs/references.tsv
uv run stt check-encoding "မြန်မာစာ"
```

FLEURS `my_mm` references are Unicode. The guard applies to every scored
hypothesis and refuses Unicode/Zawgyi mismatches, including external data from
legacy corpora, older websites, or systems still using Zawgyi fonts.

Converting Zawgyi to Unicode needs PyICU, which is optional because it requires
a native ICU build:

```bash
brew install icu4c pkg-config
PATH="$(brew --prefix icu4c)/bin:$PATH" \
PKG_CONFIG_PATH="$(brew --prefix icu4c)/lib/pkgconfig" \
uv pip install PyICU
```

## 3. Combining-mark order and digits

The same syllable can be encoded with its combining marks in different orders.
NFC normalisation collapses the common cases, and is applied to both sides
before comparison.

Myanmar digits (`၀`–`၉`) are mapped to ASCII, so a model writing `2025` is not
penalised against a reference writing `၂၀၂၅`.

## Normalisation applied before scoring

By default, both reference and hypothesis go through:

1. Unicode NFC
2. Myanmar digits → ASCII
3. Punctuation removed, including `၊` (U+104A) and `။` (U+104B)
4. Lowercasing (only affects Latin text mixed in)
5. All whitespace removed

Override with `--keep-whitespace` and `--keep-punctuation` on `stt eval` when
you want to measure those dimensions rather than normalise them away.

## Reference files

The authoritative TSV layout, `audio_id` contract, converted-file naming, and
held-out audio workflow are in
[`data/reference/README.md`](../data/reference/README.md). Keep that contract
in sync with the transcription JSONL rather than duplicating it here.

## Getting reference data

`uv run stt fetch-fleurs` pulls the Burmese split of
[FLEURS](https://huggingface.co/datasets/google/fleurs) — read speech with human
transcripts, already in Unicode. The dev split is used by default because it is
suited to quick iteration.

FLEURS is read speech from a narrow domain. A model that scores well there can
still struggle with conversational Burmese, regional accents, or code-switching
with English. For custom audio, pass the file path explicitly to `stt transcribe`
and add its recorded stem and transcript to the reference contract above.
