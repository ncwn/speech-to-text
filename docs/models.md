# The omniASR landscape on Apple Silicon

Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
covers 1,672 languages. Burmese (`mya_Mymr`) is in the supported set with
**321.1 hours** of training data and a published **CER of 4.4** on the 7B LLM
model. Two neighbouring languages in the same script are also covered: Mon
(`mnw_Mymr`) and Shan (`shn_Mymr`).

## Running the 7B on Apple Silicon

**fairseq2's PyTorch MPS path works for the 7B LLM card**, which this repo
previously stated was impossible. Measured on five FLEURS clips, same model,
same audio:

| config | RTF | peak RSS | GPU | text |
|---|---:|---:|---:|---|
| CPU, bfloat16 | 8.85 | 22.3 GB | — | reference |
| CPU, float32 | 2.14 | — | — | identical |
| **Metal (MPS), bfloat16** | **0.70** | **13.9 GB** | 17.1 GB | identical |

Corpus CER was 0.0280 in all three. Metal is **12.6× faster than CPU at the same
dtype**, 3.0× faster than CPU's best dtype, and uses less memory — so
`--device auto` now selects it, with an automatic fall back to CPU if a Metal
kernel fails mid-run.

Two things this table also settles:

* **bfloat16 on CPU is a trap.** PyTorch has no native half-precision CPU
  kernels and emulates them, costing 4.1× for identical output. On CPU, float32
  is the fast path; bfloat16 is only for machines that cannot hold float32.
* The published RTF 1.79 baseline for the 7B was a float32 CPU run.

The other two runtimes still have the gaps described below, so they remain the
reason to reach for `omniasr-gguf` when iterating:

| Runtime | GPU | Variants ported | API |
|---|---|---|---|
| Meta `omnilingual-asr` (PyTorch + fairseq2) | CUDA, **Metal (measured here)** | all | Python |
| [soniqo/speech-swift](https://github.com/soniqo/speech-swift) (MLX + CoreML) | Metal / ANE | **CTC only**, 300M–7B | Swift |
| [CrispASR](https://github.com/CrispStrobe/CrispASR) (ggml) | Metal | **LLM 300M/1B**, CTC 300M/1B | C++, Python, others |

- The MLX port's docs state the LLM decoder variant is "a separate follow-up
  module" — it has not been built. Its CTC models also ignore the language
  hint entirely, which matters when you specifically want Burmese decoding.
- The GGUF ladder stops at 1B. No 3B or 7B LLM conversion exists.

## Model families

**CTC** — wav2vec2 encoder plus a linear CTC head. One forward pass, no decoder
loop, so it is fast. It ignores the language hint and emits no punctuation.

**LLM** — the same encoder feeding a LLaMA decoder. Autoregressive, slower,
noticeably more accurate, accepts a language code, and produces punctuation.
This is the family worth testing for Burmese.

**Unlimited** — LLM variants that decode arbitrarily long audio via a 15-second
sliding-segment protocol. Non-Unlimited cards **reject audio over 40 seconds**.
Accuracy is comparable, so prefer Unlimited unless you are reproducing a
specific published number. Fine-tuning recipes do not support them.

## Available checkpoints

### `omniasr-torch` (PyTorch, CPU on this machine)

Downloads to `~/.cache/fairseq2/assets/`, fp32.

| Card | Download |
|---|---|
| `omniASR_LLM_Unlimited_300M_v2` | 6.5 GB |
| `omniASR_LLM_Unlimited_1B_v2` | 9.1 GB |
| `omniASR_LLM_Unlimited_3B_v2` | 17.5 GB |
| `omniASR_LLM_Unlimited_7B_v2` | **31.2 GB** |

Any card name from the upstream repo works — `omniASR_CTC_*`, the length-limited
`omniASR_LLM_*_v2`, and the zero-shot `omniASR_LLM_7B_ZS` included.

### `omniasr-gguf` (ggml, Metal GPU)

Downloads to `~/.cache/crispasr/`. See `stt models` for the live list.

| Name | Size | Notes |
|---|---|---|
| `llm-unlimited-300m-v2` | 1.0 GB | default; Q4_K, unlimited length |
| `llm-unlimited-300m-v2-f16` | 3.1 GB | unquantised, for measuring quantisation loss |
| `llm-1b` | 1.4 GB | Q4_K, chunked |
| `ctc-1b-v2` | 658 MB | no language hint, no punctuation |
| `ctc-300m-v2` | 194 MB | smallest |

## Memory and dtype

fairseq2n pins `torch==2.8.0` exactly because it links against libtorch's C++
ABI. Upgrading torch alone produces symbol errors.

On CPU, **float32 is the fast path** — PyTorch has no native half-precision CPU
kernels and emulates them, which upstream users measured as 2–3× slower than
fp32. But fp32 for the 7B needs roughly 34 GB resident on top of reading a
31 GB checkpoint, so `--dtype auto` selects bfloat16 for the 3B and 7B cards
and float32 below that. Override with `--dtype float32` if you have the memory
headroom and want the speed.

## Where the weights live

Neither cache is inside this repo, and both are gitignored regardless:

```
~/.cache/fairseq2/assets/     PyTorch checkpoints (up to 31 GB each)
~/.cache/crispasr/            GGUF checkpoints (~1 GB each)
```

## Planned backends

The `ASRBackend` interface is what the remaining engines you mentioned will
implement:

- **Dolphin** — local, Asian-language focused
- **ElevenLabs Scribe v2** — cloud; needs an API key and sends audio off-machine
- **Google Chirp 3** — cloud; same caveat

Cloud backends should set `is_local = False` so `stt compare` can warn before
uploading audio.
