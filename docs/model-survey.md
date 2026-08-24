# Burmese ASR survey

This is a dated, non-exhaustive snapshot checked on 2026-08-24. It records only
claims supported by primary upstream model cards, repositories, or papers.
Published scores are not comparable across different corpora or normalization
rules, and this repository has no tracked local benchmark evidence.

| Subject | Primary source | Verified scope |
|---|---|---|
| Meta Omnilingual ASR | [repository and model table](https://github.com/facebookresearch/omnilingual-asr#models), [v2 asset cards](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v2.yaml), [inference guide](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/inference/README.md#44-punctuation-and-capitalization) | Burmese is in the multilingual system. v2 CTC and LLM cards are limited to audio under 40 seconds; v2 Unlimited LLM cards cover longer audio. Output is spoken form without punctuation or capitalization. Apache-2.0. |
| Burmese Whisper fine-tune | [`chuuhtetnaing/whisper-large-v3-myanmar`](https://huggingface.co/chuuhtetnaing/whisper-large-v3-myanmar) | Apache-2.0 weights trained on the listed OpenSLR-80-derived dataset. Its reported WER belongs to that card's evaluation setup and is not a cross-model result. |
| DataoceanAI Dolphin | [repository](https://github.com/DataoceanAI/Dolphin), [language table](https://github.com/DataoceanAI/Dolphin/blob/main/languages.md), [transcription code](https://github.com/DataoceanAI/Dolphin/blob/main/dolphin/transcribe.py), [constants](https://github.com/DataoceanAI/Dolphin/blob/main/dolphin/constants.py) | Burmese is listed as `my`/`MM`. Upstream uses `SPEECH_LENGTH = 30` to choose long-form handling and cap VAD segments; this does not make its single-file extractor a fixed Whisper window. Apache-2.0. |
| MMS-1B | [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all) | The multilingual CTC card includes the `mya` adapter. CC-BY-NC-4.0. |
| SeamlessM4T v2 | [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large#supported-languages) | Burmese (`mya`, `Mymr`) is source speech and source text, and target text. It is not a Burmese target-speech language. CC-BY-NC-4.0. |
| Burmese w2v-BERT fine-tune | [`YonaKhine/finetuned-w2v2-bert-burmese-asr`](https://huggingface.co/YonaKhine/finetuned-w2v2-bert-burmese-asr) | MIT-licensed CTC fine-tune of `facebook/w2v-bert-2.0` on the listed OpenSLR-80-derived dataset. The card reports evaluation WER `0.4256`, but leaves its evaluation-data section unspecified; whitespace-token WER is unsuitable for this harness. |
| FLEURS | [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) | Public multilingual read-speech dataset with a Burmese `my_mm` configuration. A FLEURS score requires an identified split, immutable inputs, and this harness's stated normalization before it can be reproduced. |

Do not infer Apple Silicon support, memory use, download size, accuracy, or
throughput from a model's parameter count or multilingual language table. Those
properties require either an upstream statement for the exact runtime or local
evidence that records the resolved device and artifacts.
