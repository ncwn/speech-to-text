# Models and runtimes

This is the implementation catalog for the current checkout. `uv run stt
models` is authoritative for accepted model IDs, defaults, and download
estimates because those values can change with the code.

## List, download, and select

Listing models does not download weights:

```bash
uv run stt models
uv run stt models --backend hf
```

Download one model through its backend's normal cache, then select that same ID
for transcription:

```bash
uv run stt models --backend hf --download whisper-my-small
uv run stt transcribe recording.mp3 \
  --backend hf --model whisper-my-small
```

Use `--backend BACKEND --download MODEL` for each additional model needed. The
command does not prefetch the other cards in that backend.

## Implemented backends

| Backend | Runtime | `auto` device behavior | Implemented scope |
|---|---|---|---|
| `omniasr-torch` | Meta PyTorch/fairseq2 | CUDA, then MPS, then CPU; a failed MPS decode is retried on CPU | Upstream omniASR v2 CTC, LLM, and Unlimited cards |
| `omniasr-gguf` | CrispASR/ggml | CrispASR chooses the native backend; Metal is available on Apple Silicon | Selected downloadable GGUF conversions up to 1B |
| `hf` | Hugging Face Transformers | CUDA, then MPS, then CPU; no automatic CPU retry | Burmese Whisper fine-tunes, MMS-1B, SeamlessM4T v2, and w2v-BERT |
| `dolphin` | DataoceanAI Dolphin | CUDA when available, otherwise CPU; MPS must be requested explicitly | Public `base` and `small` checkpoints |

The `omniasr-torch` MPS path is a repository policy: the adapter passes
`device="mps"` to the PyTorch pipeline and catches decode failures so it can
reload on CPU. It is not a claim that fairseq2 officially supports Metal or that
every fairseq2 operation is implemented by MPS. A result recorded after retry
must identify CPU as the resolved device.

## Model behavior

The fairseq2 adapter accepts only the v2 cards listed by the CLI. CTC and
non-Unlimited LLM cards reject audio longer than 40 seconds. Unlimited cards
are the upstream long-audio variants.

The omniASR models are trained to emit spoken-form transcripts without
punctuation or capitalization. The adapter does not add either. This behavior
is documented in Meta's [inference
guide](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/inference/README.md#44-punctuation-and-capitalization).

For SeamlessM4T v2, the upstream language table lists Burmese (`mya`, `Mymr`)
as source speech and source text, with target text only. This repository uses
its speech-to-text path; Burmese target speech is not supported by that card.

Dolphin's upstream `SPEECH_LENGTH = 30` controls when its CLI chooses long-form
transcription and caps VAD segments at 30 seconds. It is not evidence that the
single-file feature extractor is a fixed Whisper 30-second window. This adapter
windows input at 30 seconds as its own integration policy.

## Defaults and precision

The current defaults are:

| Backend | Default model |
|---|---|
| `omniasr-torch` | `omniASR_LLM_Unlimited_7B_v2` |
| `omniasr-gguf` | `llm-unlimited-300m-v2` |
| `hf` | `seamless-m4t-v2` |
| `dolphin` | `small` |

Defaults do not cause setup to download weights. A transcription or explicit
model-download command fetches only the selected model.

Dolphin, GGUF, and Transformers downloads are pinned to immutable upstream
revisions. A plain `stt compare` and `--all-cached-models` run Hugging Face in
cached-files-only mode, so a missing file cannot trigger a download after its
cache check. An explicit `--only hf` comparison or transcription retains normal
first-use download behavior.

Dolphin downloads are restricted to the five files its loader consumes at a
pinned Hugging Face revision. The adapter verifies every file, including
`train.yaml`, against its SHA-256 digest before passing the cache to Dolphin.
GGUF URLs are likewise pinned to immutable Hugging Face revisions, and the
adapter verifies the selected model's SHA-256 digest before native loading.
The fairseq2 adapter verifies each returned checkpoint's exact byte length and
the shared tokenizer's byte length and SHA-256 digest. Upstream does not publish
checkpoint digests, so its checkpoint check detects incomplete or wrong-sized
files but cannot detect a same-sized modification. A failed check leaves the
cache untouched for inspection or quarantine.

For `omniasr-torch`, `--dtype auto` measures float16 and bfloat16 on an
accelerator. On CPU it uses float32 when the machine has the adapter's estimated
memory headroom and bfloat16 otherwise. The Transformers backend currently uses
float32 by default. GGUF precision is part of the selected file.

## Upstream identity and licences

- [Meta Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
  publishes the model table and Apache-2.0 code and weights. The exact v2 asset
  definitions are in the upstream [asset
  cards](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v2.yaml).
- The GGUF files come from the exact `cstr/*-GGUF` URLs encoded in the adapter.
  Check both the conversion card and the original omniASR terms before
  redistribution.
- The implemented [Burmese Whisper large
  card](https://huggingface.co/chuuhtetnaing/whisper-large-v3-myanmar) is
  Apache-2.0; the medium, small, and stock Whisper cards retain their own card
  metadata.
- [MMS-1B](https://huggingface.co/facebook/mms-1b-all) and [SeamlessM4T
  v2](https://huggingface.co/facebook/seamless-m4t-v2-large) are
  CC-BY-NC-4.0.
- The implemented [Burmese w2v-BERT
  fine-tune](https://huggingface.co/YonaKhine/finetuned-w2v2-bert-burmese-asr)
  is MIT.
- [Dolphin](https://github.com/DataoceanAI/Dolphin) publishes its code and
  checkpoints under Apache-2.0.

Model licences are separate from this harness's MIT licence. Recheck the exact
upstream card and revision before redistribution or product use.
