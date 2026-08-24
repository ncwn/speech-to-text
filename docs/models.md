# Models and runtimes

This page is the current catalog for the backends implemented in this repo.
Run `uv run stt models` for the accepted model IDs and download estimates; that
command is the source of truth for the mutable lists.

- [Setup and cache troubleshooting](setup-macos.md)
- [Historical measured findings](findings.md)
- [External Burmese ASR survey](model-survey.md)

## Implemented backends

| Backend | Runtime | Apple Silicon default | Scope |
|---|---|---|---|
| `omniasr-torch` | Meta PyTorch + fairseq2 | MPS when available; CPU fallback | Upstream omniASR cards, including the 7B LLM |
| `omniasr-gguf` | CrispASR/ggml | Metal when available | GGUF omniASR cards up to 1B; 300M Unlimited is the default |
| `hf` | Hugging Face `transformers` | MPS when available; CPU fallback | Burmese Whisper fine-tunes, MMS-1B, SeamlessM4T v2, and w2v-BERT |
| `dolphin` | DataoceanAI Dolphin + funasr | CPU by default; MPS is explicit | Public `base` and `small` checkpoints |

The PyTorch backend resolves `--device auto` as CUDA, then MPS, then CPU. If a
Metal decode fails, it reloads on CPU for the rest of that run. The other
backends keep their own runtime defaults because the fastest device is model
dependent.

## Model families

**CTC** uses an encoder and CTC head in one forward pass. It has no autoregressive
decoder, so output behavior depends on the selected checkpoint rather than a
language prompt.

**LLM** uses an autoregressive decoder and accepts an omniASR language code.
Punctuation is checkpoint output; the adapter does not synthesize it. See
[findings](findings.md) for historical Burmese observations.

**Unlimited** cards are the long-audio option in the omniASR adapter. The adapter
rejects non-Unlimited cards above 40 seconds; the exact upstream segmentation
protocol is not part of this repo's public contract.

## Defaults and device policy

The defaults are `omniASR_LLM_Unlimited_7B_v2` for `omniasr-torch`,
`llm-unlimited-300m-v2` for `omniasr-gguf`, `seamless-m4t-v2` for `hf`, and
`small` for `dolphin`. Use `uv run stt models` instead of copying a full model
table into documentation.

For `omniasr-torch`, `--dtype auto` chooses the fastest measured 16-bit format
on a GPU. On CPU it chooses float32 when the actual machine has enough RAM for
the card and bfloat16 only when that headroom is unavailable. `hf` currently
defaults to float32; GGUF precision is part of the selected model file.

`fairseq2n` requires the exact `torch==2.8.0` ABI pin. Do not upgrade torch
independently; see [setup-macos.md](setup-macos.md) for installation details.

## Weights and licensing

Weights are downloaded outside the repository. Cache locations and recovery
steps belong in [setup-macos.md](setup-macos.md), not in this catalog.

The omniASR weights are Apache-2.0. MMS-1B and SeamlessM4T weights are
CC-BY-NC-4.0. Every other checkpoint keeps the license stated by its upstream
card; check that card before redistribution.

## Future backends

ElevenLabs Scribe v2 and Google Chirp 3 are not implemented. A future remote
backend must set `is_local = False`. `stt compare` currently does not turn that
field into an upload warning, so it is metadata rather than a consent prompt.
