# The omniASR landscape on Apple Silicon

Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
covers 1,672 languages. Burmese (`mya_Mymr`) is in the supported set with
**321 hours** of training data and a published **CER of 4.4** on the 7B LLM
card — a figure that does not reproduce here, see
[Methodology](findings.md#methodology). Two neighbouring languages in the same
script are also covered: Mon (`mnw_Mymr`) and Shan (`shn_Mymr`).

## Runtimes

| Runtime | GPU | Variants ported | API |
|---|---|---|---|
| Meta `omnilingual-asr` (PyTorch + fairseq2) | CUDA, **Metal** | all | Python |
| [soniqo/speech-swift](https://github.com/soniqo/speech-swift) (MLX + CoreML) | Metal / ANE | **CTC only**, 300M–7B | Swift |
| [CrispASR](https://github.com/CrispStrobe/CrispASR) (ggml) | Metal | **LLM 300M/1B**, CTC 300M/1B | C++, Python, others |

fairseq2's PyTorch MPS path works for every card, including the 7B — measured in
[Device defaults](findings.md#device-defaults). The other two runtimes still
have gaps, which is what keeps `omniasr-gguf` useful for fast iteration:

- The MLX port's docs state the LLM decoder variant is "a separate follow-up
  module" — it has not been built. Its CTC models also ignore the language hint
  entirely, which matters when you specifically want Burmese decoding.
- The GGUF ladder stops at 1B. No 3B or 7B LLM conversion exists.

## Model families

**CTC** — wav2vec2 encoder plus a linear CTC head. One forward pass, no decoder
loop, so it is fast. It ignores the language hint.

**LLM** — the same encoder feeding a LLaMA decoder. Autoregressive, slower,
noticeably more accurate, and accepts a language code. This is the family worth
testing for Burmese.

**Unlimited** — LLM variants that decode arbitrarily long audio via a 15-second
sliding-segment protocol. Non-Unlimited cards **reject audio over 40 seconds**.
Accuracy is comparable, so prefer Unlimited unless you are reproducing a
specific published number. Fine-tuning recipes do not support them.

Neither family emits punctuation for Burmese. On a 17-minute narration the 7B
returned 17,017 characters with zero `၊` and zero `။` — see
[Held-out](findings.md#held-out).

## Available checkpoints

### `omniasr-torch`

Downloads to `.cache/fairseq2/assets/` in the checkout, fp32.

| Card | Download |
|---|---|
| `omniASR_LLM_Unlimited_300M_v2` | 6.5 GB |
| `omniASR_LLM_Unlimited_1B_v2` | 9.1 GB |
| `omniASR_LLM_Unlimited_3B_v2` | 17.5 GB |
| `omniASR_LLM_Unlimited_7B_v2` | **31.2 GB** |

Any card name from the upstream repo works — `omniASR_CTC_*`, the length-limited
`omniASR_LLM_*_v2`, and the zero-shot `omniASR_LLM_7B_ZS` included. `stt models`
lists the common ones.

### `omniasr-gguf`

Downloads to `.cache/crispasr/` in the checkout.

| Name | Size | Notes |
|---|---|---|
| `llm-unlimited-300m-v2` | 1.0 GB | default; Q4_K, unlimited length |
| `llm-unlimited-300m-v2-f16` | 3.1 GB | unquantised, for measuring quantisation loss |
| `llm-1b` | 1.4 GB | Q4_K, chunked |
| `ctc-1b-v2` | 658 MB | no language hint |
| `ctc-300m-v2` | 194 MB | smallest |

Q4_K quantisation is not a mild degradation on this task — it is bimodal and
occasionally catastrophic. See [Quantisation](findings.md#quantisation) before
using a GGUF card for anything but iteration.

## Memory and dtype

`fairseq2n` pins `torch==2.8.0` exactly because it links against libtorch's C++
ABI; see [setup-macos.md](setup-macos.md#things-that-will-bite-you).

Device and precision are chosen by measurement at runtime, not by a table in
this repo — `--device auto` and `--dtype auto` are the defaults, and
`stt hardware` shows what they resolve to. The measurements behind them are in
[Precision](findings.md#precision).

## Planned backends

The `ASRBackend` interface is what the remaining engines will implement:

- **ElevenLabs Scribe v2** — cloud; needs an API key and sends audio off-machine
- **Google Chirp 3** — cloud; same caveat

Cloud backends should set `is_local = False` so `stt compare` can warn before
uploading audio.
