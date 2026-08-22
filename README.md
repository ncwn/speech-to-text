# speech-to-text

A Burmese-focused (မြန်မာ) harness for testing and comparing speech-to-text
models — local and cloud — on the same audio, with the same metric.

Built for Apple Silicon. Implements Meta's **Omnilingual ASR** (through two
runtimes), Hugging Face `transformers` (SeamlessM4T, MMS, Whisper fine-tunes,
w2v-BERT) and **DataoceanAI Dolphin**. ElevenLabs Scribe v2 and Google Chirp 3
drop in as additional backends without restructuring anything.

Historical runs ranked `omniASR_LLM_Unlimited_7B_v2` first, but those artifacts
predate trusted waveform identity and the isolated measurement protocol. They are
preserved as unverified context, not current claims. Publication remains blocked
until the runs are regenerated through the evidence gate; see
[docs/findings.md](docs/findings.md).

## Why this exists

Burmese is genuinely awkward to evaluate:

- **No spaces between words**, so Word Error Rate is meaningless. This repo
  scores **CER** and strips whitespace before comparing.
- **Zawgyi vs Unicode** — two incompatible encodings sharing the same code
  block. Text that looks identical on screen can be completely different bytes,
  so the harness detects the encoding and refuses rather than reporting a
  garbage number near 1.0.
- **Combining-mark ordering** varies; NFC normalisation is applied first.

[docs/burmese.md](docs/burmese.md) has the details.

## Backends

| Backend | Runtime | Models |
|---|---|---|
| `omniasr-torch` | Meta's PyTorch + fairseq2 | Every omniASR card, incl. `omniASR_LLM_Unlimited_7B_v2` |
| `omniasr-gguf` | [CrispASR](https://github.com/CrispStrobe/CrispASR) (ggml) | omniASR LLM 300M / 1B and Unlimited 300M v2 |
| `hf` | Hugging Face `transformers` | Burmese Whisper fine-tunes, MMS-1B, SeamlessM4T v2, w2v-BERT |
| `dolphin` | [DataoceanAI Dolphin](https://github.com/DataoceanAI/Dolphin) | Dolphin base (140M) / small (372M) |

All four can run on Metal. Three default to it; Dolphin defaults to CPU, where
it is measurably faster. See
[Device defaults](docs/findings.md#device-defaults).

Burmese (`mya_Mymr`) is supported by omniASR. Vendor scores and this harness use
different corpora and normalisation, so their numbers are not interchangeable;
see [Methodology](docs/findings.md#methodology).

## Setup

Requires macOS 14+ on Apple Silicon, [uv](https://docs.astral.sh/uv/), and ffmpeg.

```bash
brew install ffmpeg libsndfile
./scripts/bootstrap.sh
```

That creates a Python 3.12 virtualenv and installs the runtimes. To install
selectively:

```bash
uv sync                     # core CLI + evaluation only
uv sync --extra gguf        # + Metal/GGUF runtime      (small)
uv sync --extra omniasr     # + PyTorch/fairseq2        (~3 GB of wheels)
uv sync --extra hf          # + transformers on Metal
uv sync --extra dolphin     # + Dolphin (pulls funasr + modelscope)
```

Install details, troubleshooting and the download workarounds for the large
checkpoints are in [docs/setup-macos.md](docs/setup-macos.md).

## Usage

```bash
# What is installed and working, and what this machine is
uv run stt doctor
uv run stt backends
uv run stt models
uv run stt hardware

# Grab Burmese audio with reference transcripts (FLEURS my_mm dev split)
uv run stt fetch-fleurs --limit 20

# Transcribe on the GPU with the small model
uv run stt transcribe data/fleurs/audio -b omniasr-gguf --limit 5

# Transcribe with the 7B reference model
uv run stt transcribe data/fleurs/audio -b omniasr-torch \
    -m omniASR_LLM_Unlimited_7B_v2 --limit 5

# Score any run against the references
uv run stt eval outputs/omniasr-gguf-llm-unlimited-300m-v2.jsonl \
    --reference data/fleurs/references.tsv

# Run every installed backend over the same clips and compare
uv run stt compare data/fleurs/audio --reference data/fleurs/references.tsv --limit 10

# Combine several runs by per-character vote — best model FIRST
uv run stt vote out-7b.jsonl out-seamless.jsonl out-dolphin.jsonl out-mms.jsonl \
    -o voted.jsonl

# Re-transcribe only the least-confident spans with a stronger model
uv run stt route base.jsonl strong.jsonl -o routed.jsonl

# Transcribe and write subtitles, recovering timings if the backend has none
uv run stt transcribe recording.mp3 -b omniasr-torch --align --srt

# Time a transcript you already have, against its audio
uv run stt align recording.mp3 --results outputs/omniasr-torch-omniASR_LLM_Unlimited_7B_v2.jsonl

# Is this text Unicode or Zawgyi?
uv run stt check-encoding data/fleurs/references.tsv

# Verify an accepted baseline without loading model weights
uv run stt bench --verify
```

## Results

No accuracy or performance result is currently publication-eligible. The
historical runs remain useful for forming hypotheses, but their JSONL lacks the
trusted identity and complete provenance now required by `stt evidence`.
[docs/findings.md](docs/findings.md) records that migration state and retains the
old tables explicitly as unverified history.

Trusted performance uses baseline-v2 artifacts: five counterbalanced isolated
sessions, three warmups, three measured repeats, immutable model/runtime
provenance, and an archived raw-artifact manifest. A one-worker `stt bench` run
is a transcript-only smoke diagnostic; it is never a performance baseline. A
backend that cannot report its resolved device or another execution-defining
setting remains diagnostic and cannot enter a trusted baseline.

## Model weights

Downloaded on first use into `.cache/` inside the checkout, which is gitignored.
Set `STT_CACHE_DIR` to move them elsewhere, and run `uv run stt doctor` to see
where they are and whether every runtime is installed. Details, including the
one runtime that needs a symlink, are in
[docs/setup-macos.md](docs/setup-macos.md#disk).

## Layout

```
src/stt/
  cli.py              Typer CLI: transcribe, align, eval, compare, vote, route, …
  audio.py            Discovery + 16 kHz mono normalisation (cached)
  burmese.py          NFC, Zawgyi detection, CER-safe normalisation
  datasets.py         FLEURS Burmese fetcher
  results.py          TranscriptionResult + Segment; JSONL/text/SRT/VTT writers
  execution.py        One synchronized complete-corpus backend boundary
  measurement.py      Versioned raw worker/repeat/provenance artifacts
  bench_worker.py     One isolated benchmark subject process
  evaluate.py         CER/WER scoring
  registry.py         Backend registry
  vote.py             ROVER-style voting across runs
  cascade.py          Confidence-driven routing between a cheap and a strong model
  align.py            CTC forced alignment: timestamps + confidence for any text
  quality.py          Reference-free defect detection (decoder loops)
  hardware.py         Chip detection and the precision probe
  paths.py            Cache-root resolution (repo-local by default)
  telemetry.py        CPU/RAM/GPU measurement around each transcription
  native.py           fd-level silencing of chatty native runtimes
  backends/
    base.py           The ASRBackend interface every engine implements
    omniasr_torch.py  Meta PyTorch/fairseq2 runtime
    omniasr_gguf.py   CrispASR ggml/Metal runtime
    transformers_asr.py  Whisper / MMS / Seamless / w2v-BERT
    dolphin.py        DataoceanAI Dolphin
scripts/bootstrap.sh  Prerequisite checks + environment setup
docs/                 findings, setup notes, model landscape, Burmese specifics
data/, outputs/       gitignored working directories
```

## Adding a backend

Write one module in `src/stt/backends/`, subclass `ASRBackend`, decorate with
`@register`, and import it in `backends/__init__.py`. See `base.py` for the
contract — notably: import heavy dependencies inside `load()`, not at module
scope, and report per-file failures as results with `error` set rather than
raising.

## Documentation

- [`docs/findings.md`](docs/findings.md) — evidence status and historical measurements
- [`evidence/README.md`](evidence/README.md) — publishable-evidence contract
- [`docs/setup-macos.md`](docs/setup-macos.md) — install details and troubleshooting
- [`docs/models.md`](docs/models.md) — the omniASR model/runtime landscape
- [`docs/model-survey.md`](docs/model-survey.md) — every Burmese-capable ASR model, verified
- [`docs/burmese.md`](docs/burmese.md) — encoding, normalisation, and why CER

## Licence

MIT for this harness. Model weights carry their own licences — omniASR is
Apache-2.0.
