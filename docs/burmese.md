# Evaluating Burmese ASR

The harness normalizes Burmese text before scoring and guards against likely
Unicode/Zawgyi mismatches. The implementation is in `src/stt/burmese.py`; the
evaluation flow is in `src/stt/evaluate.py`.

## Character error rate, not whitespace-token WER

Spaces in Burmese do not provide stable word boundaries. They may mark phrases,
and references and models can place them differently. The whitespace-token WER
computed by generic ASR tooling therefore mixes recognition errors with
segmentation choices and is unsuitable for this harness.

The CLI prints WER because the shared evaluator supports other languages, but
it must not be used to rank Burmese models. CER is the headline metric. By
default, the harness removes whitespace before calculating it.

Corpus CER is length-weighted: total character edits divided by total reference
characters. It is not the average of per-file CER values.

## Unicode and Zawgyi

Unicode and Zawgyi can render similar Burmese text from different code-point
sequences. Scoring a reference in one encoding against a hypothesis in the
other does not measure recognition quality.

The harness uses Google Myanmar Tools' statistical detector and a threshold of
`0.5`. This is a heuristic, not an encoding proof. The upstream detector
documentation says short strings can score near the middle and strings that
are valid in both encodings may return any value. Other Myanmar-script
languages can also affect classification. Treat `unicode (p=...)` and `zawgyi
(p=...)` as detector results, inspect ambiguous or high-impact items manually,
and tune a threshold only against representative labelled data.

If the reference and hypothesis fall on opposite sides of the threshold, the
item is marked as an error instead of receiving CER or WER. The harness does
not convert between encodings.

Inspect a file or string:

```bash
uv run stt check-encoding data/fleurs/references.tsv
uv run stt check-encoding "မြန်မာစာ"
```

The detector behavior and threshold guidance come from [Google Myanmar
Tools](https://github.com/google/myanmar-tools#using-the-zawgyi-detector).

## Normalization

By default, reference and hypothesis both receive:

1. Unicode NFC normalization.
2. Myanmar digits `၀` through `၉` mapped to ASCII `0` through `9`.
3. Configured Myanmar and ASCII punctuation removed, including `၊` and `။`.
4. Lowercasing, which affects Latin text mixed into Burmese.
5. All whitespace removed.

NFC handles canonical Unicode equivalence; it is not a general repair for
invalid or legacy text. The [Unicode Myanmar
chapter](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-16/)
is the primary reference for encoded character behavior.

Use `--keep-whitespace` or `--keep-punctuation` with `stt eval` only when the
experiment is explicitly measuring those dimensions. Record the flags with the
result because they change the scoring contract.

## References and corpus scope

Reference files are two-column TSVs keyed by audio stem. A result is not
reproducible unless the audio identity, reference text, split, normalization
options, and provenance are fixed together.

`uv run stt fetch-fleurs` fetches the Burmese `my_mm` configuration of
[FLEURS](https://huggingface.co/datasets/google/fleurs). The dev split is the
CLI default. FLEURS is read speech from a limited domain; a score on it does not
establish performance on conversational Burmese, regional accents, noisy
recordings, or English code-switching.
