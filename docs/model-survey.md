# Burmese ASR survey

This is a dated, non-exhaustive research snapshot, checked on 2026-08-24. Model
and vendor catalogs change, so this page does not claim to enumerate every
Burmese-capable checkpoint. Current repo support is in
[models.md](models.md); local measurements and their evidence status are in
[findings.md](findings.md).

## Selected primary sources

| Subject | Primary source | Snapshot conclusion |
|---|---|---|
| Meta Omnilingual ASR | [repository](https://github.com/facebookresearch/omnilingual-asr), [Burmese results](https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/main/per_language_results_table_7B_llm_asr.csv) | Open multilingual ASR with Burmese coverage; the published corpus is not comparable to this repo's historical observations. |
| Burmese Whisper fine-tune | [`chuuhtetnaing/whisper-large-v3-myanmar`](https://huggingface.co/chuuhtetnaing/whisper-large-v3-myanmar) | Weights are released; the card's WER is on its own evaluation set and should not be ranked against unrelated corpora. |
| myMediWhisper | [paper](https://arxiv.org/abs/2608.11036) | A clinical-speech paper, not a released general-purpose checkpoint in this snapshot. |
| DataoceanAI Dolphin | [repository and language table](https://github.com/DataoceanAI/Dolphin/blob/main/languages.md) | Burmese is listed; this repo implements the public `base` and `small` checkpoints. |
| MMS-1B | [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all) | Burmese adapter is available; the weights are CC-BY-NC-4.0. |
| SeamlessM4T v2 | [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large) | Burmese speech recognition is available; the weights are CC-BY-NC-4.0. |
| Burmese w2v-BERT fine-tune | [`YonaKhine/finetuned-w2v2-bert-burmese-asr`](https://huggingface.co/YonaKhine/finetuned-w2v2-bert-burmese-asr) | A released Burmese CTC fine-tune with no published score recorded here. |

The implemented rows above are cataloged by the repo in [models.md](models.md).
Their runtime, device, precision, and benchmark details must not be copied into
this survey.

## Important scope checks

- **Published error rates are not interchangeable.** Model cards, papers, and
  this repo use different corpora and normalization rules. Use the common local
  evidence in [findings.md](findings.md) when comparing implemented backends.
- **BURMESE-SAN is a text benchmark, not an ASR system.** See the
  [paper](https://arxiv.org/abs/2602.18788) before using it as speech evidence.
- **Audio-capable language models are not automatically transcribers.** An
  audio-understanding model needs verbatim ASR evidence before it belongs in an
  ASR comparison.
- **Absence is not a durable finding.** A model absent from this snapshot may
  appear later on Hugging Face or in a vendor catalog; recheck the primary source
  before adopting or ruling out a candidate.

## Licensing

The external cards above carry their own terms. In particular, MMS-1B and
SeamlessM4T are CC-BY-NC-4.0, while omniASR is Apache-2.0. Licensing is included
here only as a research caveat; the implementation catalog remains
[models.md](models.md).
