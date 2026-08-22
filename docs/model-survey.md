# Burmese ASR model survey

Verification of the claims in a ChatGPT deep-research report (`deep-research-report.md`),
checked against the Hugging Face API, arXiv, and vendor documentation on 2026-08-18.

The report's citations are `citeturnNsearchM` tokens rather than URLs, so nothing in it
is verifiable as delivered. Everything below was re-checked from primary sources.

## What holds up

| Claim | Verdict | Evidence |
|---|---|---|
| BURMESE-SAN, 3,920 samples, 7 tasks, native-speaker-built | **Real** | [arXiv:2602.18788](https://arxiv.org/abs/2602.18788) |
| myMediWhisper, ~28 h corpus, 23.44 % WER (Medium, full FT) | **Real** | [arXiv:2608.11036](https://arxiv.org/abs/2608.11036) |
| `whisper-large-v3-myanmar` exists | **Real** | `chuuhtetnaing/whisper-large-v3-myanmar`, Apache-2.0 |
| SEA-LION v4.5 27B and E2B | **Real** | `aisingapore/Qwen-SEA-LION-v4.5-27B-IT`, `aisingapore/Gemma-SEA-LION-v4.5-E2B-IT`, both MIT |
| MMS-1B and `mms-tts-mya` are CC-BY-NC-4.0 | **Real** | `facebook/mms-1b-all`, `facebook/mms-tts-mya` |
| SeamlessM4T Large | **Real** | `facebook/seamless-m4t-v2-large`, CC-BY-NC-4.0 |
| Google Chirp supports `my-MM` | **Real, with a caveat** | `chirp`/`chirp_2` in `asia-southeast1` + `europe-west4`; **`chirp_3` only in `eu`** |
| Azure `my-MM` STT + NilarNeural / ThihaNeural | **Real** | Both voices listed, Standard tier, no styles/roles |

## What does not survive contact

**myMediWhisper has no released weights.** It is the report's headline ASR
recommendation ("P1 medical ASR", 5.0 rating) but no `myMediWhisper` repo exists on
Hugging Face. It is a paper, not a downloadable model — and it is trained on clinical
dialogue, so it would be the wrong domain for general audio regardless.

**The `whisper-large-v3-myanmar` number is domain-transplanted.** The report cites
32.03 % WER, which is myMediWhisper's measurement of it *on medical test data*. The
model's own card reports **54.9 % WER on its own eval set**. It was fine-tuned on
`myanmar-speech-dataset-openslr-80` — roughly 2,500 utterances of **read speech** — at
learning rate 3e-4 for 30 epochs. That is a small, narrow corpus and an aggressive
schedule for a 1.55B model. Treat it as an experiment, not a general recogniser.

**BURMESE-SAN is a text benchmark, not a speech one.** It measures NLU/reasoning/
generation. The report's executive summary leads with it and lets its scores frame the
whole document, including the ASR sections, where they have no bearing.

**SEA-LION is not an ASR system.** The 27B is text + vision. The E2B has an audio
encoder, but audio *understanding* is not verbatim transcription — the report says this
itself, then rates E2B 2.0 for "Transcription" anyway.

## What the report leaves out

**Meta Omnilingual ASR is absent entirely.** Released November 2025, 1,672 languages,
Apache-2.0, with a **published Burmese CER of 4.4** for `omniASR_LLM_Unlimited_7B_v2` and
321 hours of Burmese training data. It is the strongest open Burmese ASR result on
record and it is the model this repo already implements. Its omission is the single
largest gap in the report's speech coverage.

**ElevenLabs Scribe is absent.** Its documentation places Burmese (`mya`) in the
">10 % to ≤20 % WER" band — the best Burmese figure any vendor publishes, better than
every open checkpoint the report surveys.

**Dolphin is absent.** (DataoceanAI Dolphin, 40 Eastern languages.)

## Full sweep of locally-runnable Burmese ASR (2026-08-18)

Searched the Hugging Face model index for every ASR model tagged or named for Burmese,
plus the current general-purpose ASR releases, and checked each one's actual language
coverage rather than trusting the tags.

### Confirmed to support Burmese

Measured here where a row says "benchmarked" — those numbers live in
[Baseline](findings.md#baseline), not in this table.

| Model | Params | Licence | Runtime | Burmese evidence |
|---|---:|---|---|---|
| `omniASR_LLM_Unlimited_7B_v2` | 7B | Apache-2.0 | fairseq2, Metal | Meta claims CER 4.4; benchmarked |
| `omniASR_LLM_*_3B/1B/300M_v2` | 3B–300M | Apache-2.0 | fairseq2 Metal; 300M/1B also GGUF | benchmarked |
| `DataoceanAI/dolphin-small` | 372M | Apache-2.0 | funasr | benchmarked; no published Burmese score |
| `DataoceanAI/dolphin-base` | 140M | Apache-2.0 | funasr | same |
| `facebook/mms-1b-all` | 1B | **CC-BY-NC** | transformers, MPS | benchmarked |
| `facebook/seamless-m4t-v2-large` | 2.3B | **CC-BY-NC** | transformers, MPS | benchmarked; ASR-only for Burmese |
| `chuuhtetnaing/whisper-large-v3-myanmar` | 1.55B | Apache-2.0 | transformers/MLX | **54.9 % WER on its own eval** |
| `chuuhtetnaing/whisper-{medium,small,tiny}-myanmar` | 769M–39M | Apache-2.0 | transformers/MLX | same corpus, smaller |
| `YonaKhine/finetuned-w2v2-bert-burmese-asr` | ~600M | MIT | transformers | no published score |
| `thantzinphyo/whisper-*-myanmar-*` | tiny/small | — | transformers | domain-specific ("shopvoice") |

Dolphin's language table is at
[`DataoceanAI/Dolphin/languages.md`](https://raw.githubusercontent.com/DataoceanAI/Dolphin/main/languages.md);
only `base` and `small` are public — `medium` (910M) and `large` (1.68B) are not released.

### Checked and ruled out

| Model | Why not |
|---|---|
| `Qwen/Qwen3-ASR-1.7B` / `0.6B` | 30 languages + 22 Chinese dialects. **No Burmese.** |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | 13 languages. No Burmese. |
| `nvidia/parakeet-*`, `nemotron-asr-*` | English / European only |
| `openai/whisper-*` (stock) | Burmese below OpenAI's own published quality threshold |
| `myMediWhisper` | **No weights released** — paper only |

### Notes on the omniASR checkpoints

`facebook/omniASR-LLM-{300M,1B,3B,7B}` on Hugging Face are raw fairseq2 `.pt` files, not
`transformers` ports. That does not block Metal: fairseq2's own MPS path runs every card,
including the 7B ([Device defaults](findings.md#device-defaults)). What the GGUF ladder
still caps is *quantised* inference — CrispASR converts up to 1B and no further
(0.8.29 is current).

Burmese CER by size is not published — Meta's
[`per_language_results_table_7B_llm_asr.csv`](https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/main/per_language_results_table_7B_llm_asr.csv)
covers the 7B only. Measured here, the scaling curve is real but shallow, about
one CER point per 10× parameters — see [Methodology](findings.md#methodology).

## Practical ranking for general-domain Burmese audio

Superseded by measurement — see [Baseline](findings.md#baseline), which scores
these on 120 FLEURS Burmese **test** clips rather than reasoning from vendor
claims.

Two things this survey work got wrong are recorded in
[Methodology](findings.md#methodology): published Burmese numbers are not
comparable across papers, and a 12-clip screen inverted the ranking rather than
merely adding noise to it.

## Licensing

`facebook/*` speech weights — MMS ASR, `mms-tts-mya`, SeamlessM4T — are **CC-BY-NC-4.0**.
Fine for evaluation here; not for a commercial product. omniASR is Apache-2.0.
