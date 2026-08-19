# speech-to-text

A Burmese-focused (မြန်မာ) harness for testing and comparing speech-to-text
models — local and cloud — on the same audio, with the same metric.

Built for Apple Silicon. Implements Meta's **Omnilingual ASR** (through two
runtimes), Hugging Face `transformers` (SeamlessM4T, MMS, Whisper fine-tunes,
w2v-BERT) and **DataoceanAI Dolphin**. ElevenLabs Scribe v2 and Google Chirp 3
drop in as additional backends without restructuring anything.

On 120 FLEURS Burmese test clips the winner is `omniASR_LLM_Unlimited_7B_v2` at
**CER 0.1017**, Apache-2.0, CPU-only at RTF 1.79. See
[Measured baseline](#measured-baseline).

## Why this exists

Burmese is genuinely awkward to evaluate:

- **No spaces between words**, so Word Error Rate is meaningless. This repo
  scores **CER** and strips whitespace before comparing.
- **Zawgyi vs Unicode** — two incompatible encodings sharing the same code
  block. Text that looks identical on screen can be completely different bytes.
  A Unicode reference scored against a Zawgyi hypothesis gives a garbage number
  near 1.0, so the harness detects the encoding and refuses rather than lying.
- **Combining-mark ordering** varies; NFC normalisation is applied first.

## Backends

| Backend | Runtime | Compute on M2 Max | Models |
|---|---|---|---|
| `omniasr-torch` | Meta's official PyTorch + fairseq2 | **CPU only** | Every omniASR card, incl. `omniASR_LLM_Unlimited_7B_v2` |
| `omniasr-gguf` | [CrispASR](https://github.com/CrispStrobe/CrispASR) (ggml) | **Metal GPU** | omniASR LLM 300M / 1B and Unlimited 300M v2 |
| `hf` | Hugging Face `transformers` | **Metal GPU** | Burmese Whisper fine-tunes, MMS-1B, SeamlessM4T v2, w2v-BERT |
| `dolphin` | [DataoceanAI Dolphin](https://github.com/DataoceanAI/Dolphin) | CPU | Dolphin base (140M) / small (372M) |

There is no GPU path for the 7B LLM variant on a Mac. Meta's runtime has no
validated Metal support; the [MLX port](https://github.com/soniqo/speech-swift)
covers only the CTC variants and is Swift-only; the GGUF ladder stops at 1B.
So the intended workflow is: iterate on **SeamlessM4T** via the `hf` backend
(Metal, RTF 0.16), then produce final output with `omniasr-torch` on the 7B.
Do not iterate on `omniasr-gguf` — 4-bit quantisation makes that model
unreliable rather than merely worse, as the benchmark below shows.

Burmese (`mya_Mymr`) is supported by omniASR with 321 hours of training data.
Meta publishes CER 4.4 for the 7B; this harness measures 0.1017 for the same
checkpoint — see [the note on why](#metas-published-cer-44-does-not-reproduce).

## Setup

Requires macOS 14+ on Apple Silicon, [uv](https://docs.astral.sh/uv/), and ffmpeg.

```bash
brew install ffmpeg libsndfile
./scripts/bootstrap.sh
```

That creates a Python 3.12 virtualenv and installs both runtimes. To install
selectively:

```bash
uv sync                     # core CLI + evaluation only
uv sync --extra gguf        # + Metal/GGUF runtime      (small)
uv sync --extra omniasr     # + PyTorch/fairseq2        (~3 GB of wheels)
uv sync --extra hf          # + transformers on Metal
uv sync --extra dolphin     # + Dolphin (pulls funasr + modelscope)
```

`fairseq2n` links against libtorch's C++ ABI, so `torch==2.8.0` is pinned
exactly. Do not upgrade torch independently.

## Usage

```bash
# What is installed and working
uv run stt backends

# Grab Burmese audio with reference transcripts (FLEURS my_mm dev split)
uv run stt fetch-fleurs --limit 20

# Transcribe on the GPU with the small model
uv run stt transcribe data/fleurs/audio -b omniasr-gguf --limit 5

# Transcribe with the 7B reference model (slow, CPU)
uv run stt transcribe data/fleurs/audio -b omniasr-torch \
    -m omniASR_LLM_Unlimited_7B_v2 --limit 5

# Score any run against the references
uv run stt eval outputs/omniasr-gguf.jsonl --reference data/fleurs/references.tsv

# Run every installed backend over the same clips and compare
uv run stt compare data/fleurs/audio --reference data/fleurs/references.tsv --limit 10

# Combine several runs by per-character vote — best model FIRST
uv run stt vote out-7b.jsonl out-seamless.jsonl out-dolphin.jsonl out-mms.jsonl \
    -o voted.jsonl

# Transcribe and write subtitles, recovering timings if the backend has none
uv run stt transcribe recording.mp3 -b omniasr-torch --align --srt

# Time a transcript you already have, against its audio
uv run stt align recording.mp3 --results outputs/omniasr-torch.jsonl
```

## Measured baseline

Every backend over the **same 120 FLEURS Burmese `test` clips** (29.6 minutes)
on an M2 Max, scored as CER with whitespace stripped — Burmese has no word
delimiters, so WER is meaningless here. Lower CER is better; lower RTF is
faster.

| Model | Backend | Compute | CER | RTF | Licence |
|---|---|---|---:|---:|---|
| **omniASR 7B Unlimited v2 fp32** | `omniasr-torch` | CPU | **0.1017** | 1.79 | **Apache-2.0** |
| omniASR 3B Unlimited v2 fp32 | `omniasr-torch` | CPU | 0.1196 | 1.54 | Apache-2.0 |
| omniASR 300M Unlimited v2 fp32 | `omniasr-torch` | CPU | 0.1298 | 1.29 | Apache-2.0 |
| SeamlessM4T v2 large | `hf` | Metal | 0.1301 | **0.16** | CC-BY-NC |
| Dolphin small (372M) | `dolphin` | CPU | 0.1722 | 0.18 | Apache-2.0 |
| MMS-1B | `hf` | Metal | 0.1796 | **0.04** | CC-BY-NC |

### Which one to use

**omniASR 7B** tops this table and is Apache-2.0. It needs 31.2 GB on disk,
~40 GB of RAM at fp32, and CPU-only inference — RTF 1.79 on short clips but
**3.60 on a single long file**, so budget an hour for a 17-minute recording. It
also emits **no punctuation at all**. Use it when licence and raw accuracy
matter more than turnaround, and when you can post-process sentence breaks.

**SeamlessM4T v2** is the practical default. Eleven times faster on Metal
(RTF 0.16), it segments output into real sentences, and on held-out audio it is
*more* accurate than the 7B once a single proper noun is set aside — see
[the held-out check](#held-out-check-a-17-minute-recording-with-a-human-reference).
The catch is the licence: **CC-BY-NC**, so evaluation only, never a product.

**Dolphin small** is the Apache-2.0 fallback if you need permissive licensing
and Metal-class speed and can accept ~30 % more error.

Note how little separates omniASR 300M from Seamless (0.1298 vs 0.1301) despite
a 10x parameter difference and completely different architectures. Above about
300M, the returns on this task are small.

### Do not trust a 12-clip screen

An earlier version of this table used 12 FLEURS **dev** clips. It was wrong in
both directions:

| Model | 12 dev clips | 120 test clips |
|---|---:|---:|
| SeamlessM4T v2 | 0.0594 | 0.1301 |
| omniASR 3B fp32 | 0.0594 | 0.1196 |
| omniASR 300M fp32 | 0.0959 | 0.1298 |
| Dolphin small | 0.0939 | 0.1722 |
| MMS-1B | 0.1448 | 0.1796 |

Every model scored roughly twice as well on the small sample, and the Seamless
vs omniASR gap — 38 % on 12 clips — vanished entirely on 120. Small samples here
did not merely add noise, they inverted the ranking. Twelve clips is a smoke
test for "does this backend work at all", nothing more.

### Meta's published CER 4.4 does not reproduce

Meta reports **CER 4.4** for `omniASR_LLM_Unlimited_7B_v2` on Burmese. Measured
here on 120 FLEURS test clips, that same checkpoint scores **0.1017** — 2.3x
higher. The scaling curve behind it is smooth and well behaved:

| Card | CER | RTF |
|---|---:|---:|
| 300M | 0.1298 | 1.29 |
| 3B | 0.1196 | 1.54 |
| 7B | 0.1017 | 1.79 |

so this is not a broken checkpoint or a bad decode — the model scales exactly as
it should. The gap is almost certainly text normalisation: Burmese CER moves a
long way depending on how you treat combining-mark order, the ၊ and ။
delimiters, and whitespace, and Meta does not publish its rule.

**Vendor CER numbers for Burmese are not comparable across papers — including
Meta's, and including this table.** What is comparable is the ordering *within*
one table, which is why every number here comes from the same harness on the
same clips.

### Every model has a bad tail

For long audio the tail matters more than the average, because one runaway
segment contaminates everything after it. Over the same 120 clips:

| Model | p50 | p90 | worst | clips over 0.3 |
|---|---:|---:|---:|---:|
| **omniASR 7B fp32** | **0.071** | **0.195** | **0.503** | **3 / 120** |
| omniASR 3B fp32 | 0.084 | 0.252 | 0.512 | 8 / 120 |
| SeamlessM4T v2 | 0.088 | 0.247 | 0.509 | 6 / 120 |
| omniASR 300M fp32 | 0.104 | 0.212 | 0.525 | 5 / 120 |
| Dolphin small | 0.146 | 0.291 | 0.525 | 10 / 120 |
| MMS-1B | 0.150 | 0.287 | 0.583 | 9 / 120 |

On 12 clips Dolphin and Seamless appeared never to fail. At 120 clips every
model puts 2.5–8 % of clips over CER 0.3. The 7B has the best tail by a clear
margin — roughly half the failure rate of anything else — and that, more than
the average, is what matters for a long recording, where one runaway segment
contaminates everything after it.

### Voting across models beats every single model

The systems fail on different words. An oracle picking the better of the top
two per 200-character span would score **0.0571** where the best single model
scores 0.0857 — a third of the remaining error is recoverable without a better
model, just by choosing between hypotheses already in hand.

`stt vote` implements the practical version: align every run to a pivot, then
vote position by position, weighted by measured accuracy. Chosen on FLEURS and
verified on held-out audio, changing nothing between the two:

| | best single | voted | |
|---|---:|---:|---:|
| FLEURS 120 test clips | 0.1017 | **0.0945** | −7.1 % |
| held-out 16.8 min recording | 0.0857 | **0.0718** | −16.2 % |

Three things that turned out to matter:

**The pivot must be your best model.** Voting can only correct characters the
pivot proposed, so the first run given anchors the result. Ties break toward
the pivot, which is why adding a system can never do worse than a wash.

**A pool of weak systems achieves nothing.** SeamlessM4T + Dolphin + MMS scores
0.1301 — exactly what SeamlessM4T scores alone. The 7B is load-bearing.

**Spacing convention dominates the alignment.** SeamlessM4T emits a space per
sub-word, omniASR emits none, so on raw text the aligner spends its budget on
whitespace instead of on the characters being voted: raw 0.0764,
`tidy_spacing` 0.0718, full normalisation 0.0714. The default is
`tidy_spacing`, which captures nearly all of it while keeping the ၊ and ။
delimiters that full normalisation discards.

Cost is modest. Adding SeamlessM4T, Dolphin and MMS to a 7B run takes the total
real-time factor from 3.60 to 4.03 — 12 % more compute for 8.6 % less error on
FLEURS. Adding the 3B as well reaches 0.0900 but costs RTF 6.63, which is 84 %
more compute for a further 3 %.

### Held-out check: a 17-minute recording with a human reference

FLEURS is public and may be in these models' training data, so the ranking was
re-checked against a hand-corrected transcript of a 17-minute Burmese film
recap — audio none of them has seen. The FLEURS ordering held:

| Model | CER | sub | del | ins | Licence |
|---|---:|---:|---:|---:|---|
| omniASR 7B fp32 | **0.0857** | 561 | 460 | 439 | Apache-2.0 |
| SeamlessM4T v2 | 0.0887 | 810 | 248 | 453 | CC-BY-NC |
| Dolphin small | 0.1366 | 809 | 801 | 717 | Apache-2.0 |
| omniASR 300M Q4_K | 0.2245 | 1204 | **2092** | 528 | Apache-2.0 |

Both leaders score *better* here than on FLEURS, so the contamination worry did
not materialise into an inflated ranking.

**But the top two are separated entirely by one proper noun.** The protagonist's
name occurs 99 times — 2.9 % of the transcript. The reference spells it `ဂျုံး`
(Joon). omniASR 7B produces `ဂျုံး` 78 times; SeamlessM4T produces `ဂျွန်`
(John) 92 times and `ဂျုံး` never. Those two spellings differ in three
characters, so every mention costs three substitutions — which is where 277 of
Seamless's 810 substitutions come from, and why its top three "character
confusions" are really one word.

Correct that single name and the ranking inverts:

| Model | as measured | with the name corrected |
|---|---:|---:|
| SeamlessM4T v2 | 0.0887 | **0.0725** |
| omniASR 7B fp32 | 0.0857 | 0.0831 |

So: **SeamlessM4T is about 13 % more accurate on running text, and omniASR 7B
wins overall only because it gets an out-of-vocabulary Korean name right.** That
is a real advantage — a decoder that defaults to the frequent, familiar `ဂျွန်`
is doing plausible-but-wrong normalisation, and on a film recap the character's
name is exactly what you cannot afford to lose. It is just not the general
accuracy advantage the headline number implies.

The practical consequence is that a corpus CER cannot be read as a single
verdict on a specific recording. Check what the errors *are*.

### CER does not measure punctuation, and for long audio it should

Scoring normalises punctuation away on both sides, which is correct for
comparing recognition accuracy and misleading for choosing a transcriber. On a
17-minute Burmese narration:

| Model | characters | ၊ | ။ | long-form RTF |
|---|---:|---:|---:|---:|
| omniASR 7B | 17,017 | 0 | 0 | 3.60 |
| SeamlessM4T v2 | 17,684 | 26 | 191 | 0.26 |
| Dolphin small | 17,224 | 29 | 108 | 0.12 |
| omniASR 300M Q4_K | 15,498 | 0 | 0 | 0.27 |

**omniASR emits no punctuation at all** — the 7B returns seventeen thousand
characters with no sentence boundary anywhere. SeamlessM4T segments the same
audio into 191 sentences. Neither behaviour shows up in CER.

Note also that omniASR's real-time factor doubles on a single long file (1.79 on
short clips, 3.60 here), so the 7B took just under an hour for 17 minutes of
audio against four minutes for SeamlessM4T.

### 4-bit quantisation is the one clear loser

The 300M GGUF at Q4_K scored 0.6647 on the dev clips against 0.0959 for the
same model at fp32, and it is bimodal rather than uniformly bad: most clips are
fine, then a few collapse into decoder repetition loops emitting three to four
times the reference length. On the 17-minute file it produced `၁၁၀၁၁၁သက်၁၁`
where every fp32 model produced the sentence. Use it for iteration, never for
output.

Reproduce with:

```bash
uv run stt fetch-fleurs --split test --limit 120 --dest data/fleurs-test
uv run stt compare data/fleurs-test/audio -r data/fleurs-test/references.tsv
```

### Confidence predicts error, which is what makes routing possible

Every backend now returns `Segment`s carrying `start`, `end` and — where it can
be had — a confidence. Three of them already knew all of this and were throwing
it away: CrispASR times every segment, and Seamless and Dolphin compute window
offsets they then discarded.

The most accurate model was the problem. omniASR's fairseq2 pipeline returns a
bare `List[str]`: no timings, no scores, nothing to point at. `stt align`
recovers all of it after the fact by forced alignment against MMS-1B, whose
Burmese adapter is character-level — the right granularity for a script written
without word delimiters. Out-of-vocabulary characters on a real transcript came
to **0.15%**, all uppercase Latin, which lowercasing removes.

The signal is real, measured on the 7B's transcript of the held-out recording
against the human reference:

| | mean confidence | mean CER |
|---|---:|---:|
| least-confident quartile | 0.690 | 0.1739 |
| most-confident quartile | 0.867 | 0.0288 |

Pearson **r = −0.75**, and a **6.0×** error ratio between the quartiles. That
matters more than it sounds: error is concentrated, so it can be bought cheaply.

| escalate this share of audio | errors it covers | leverage |
|---:|---:|---:|
| 10% | 28.6% | 2.77× |
| 25% | 47.9% | 1.87× |
| 50% | 78.4% | 1.56× |

Voting currently pays 4× compute on every second of audio. Spending it only
where confidence is low should buy most of the accuracy for a fraction of that.

A caveat on the other confidence source: CrispASR's `no_speech_prob` measures
how likely a span is to be *silence*, not how likely the transcript is to be
right, and the omniasr-llm backend does not compute it at all — it returns
`-1.0`. That is now reported as unknown rather than converted into a confidence
of 2.0.

### Subtitles have to respect Burmese syllable clusters

omniASR emits no `၊` or `။` whatsoever, so cue boundaries fall back to a length
limit. Cutting blindly there splits grapheme clusters, and a cue that opens with
a bare `ာ` is not readable:

```
...လူတိုင်းသိချင်ကြပါတယ်ကောင်းတ
ာလုပ်ရင်နတ်ပြည်ရောက်မယ်...
```

Burmese stacks vowel signs, medials and tone marks onto a base consonant, and
the virama binds the consonant after it. Forced breaks now retreat to the
longest pause in the tail of the cue that also keeps clusters intact. On the
17-minute recording that took cues starting with a combining mark from 1 to 0,
and moved the breaks onto real phrase boundaries.

## Model weights

Downloaded on first use, cached outside this repo:

- `omniasr-torch` → `~/.cache/fairseq2/assets/`
- `omniasr-gguf` → `~/.cache/crispasr/`

## Layout

```
src/stt/
  cli.py              Typer CLI: transcribe, align, eval, compare, vote, backends
  audio.py            Discovery + 16 kHz mono normalisation (cached)
  burmese.py          NFC, Zawgyi detection, CER-safe normalisation
  datasets.py         FLEURS Burmese fetcher
  results.py          TranscriptionResult + Segment; JSONL/text/SRT/VTT writers
  evaluate.py         CER/WER scoring
  registry.py         Backend registry
  vote.py             ROVER-style voting across runs
  align.py            CTC forced alignment: timestamps + confidence for any text
  quality.py          Reference-free defect detection (decoder loops)
  native.py           fd-level silencing of chatty native runtimes
  backends/
    base.py           The ASRBackend interface every engine implements
    omniasr_torch.py  Meta PyTorch/fairseq2 runtime
    omniasr_gguf.py   CrispASR ggml/Metal runtime
    transformers_asr.py  Whisper / MMS / Seamless / w2v-BERT on Metal
    dolphin.py        DataoceanAI Dolphin
scripts/bootstrap.sh  Prerequisite checks + environment setup
docs/                 setup notes, model landscape, Burmese specifics
data/, outputs/       gitignored working directories
```

## Adding a backend

Write one module in `src/stt/backends/`, subclass `ASRBackend`, decorate with
`@register`, and import it in `backends/__init__.py`. See `base.py` for the
contract — notably: import heavy dependencies inside `load()`, not at module
scope, and report per-file failures as results with `error` set rather than
raising.

## Documentation

- [`docs/setup-macos.md`](docs/setup-macos.md) — install details and troubleshooting
- [`docs/models.md`](docs/models.md) — the omniASR model/runtime landscape on Apple Silicon
- [`docs/model-survey.md`](docs/model-survey.md) — every Burmese-capable ASR model, verified
- [`docs/burmese.md`](docs/burmese.md) — encoding, normalisation, and why CER

## Licence

MIT for this harness. Model weights carry their own licences — omniASR is
Apache-2.0.
