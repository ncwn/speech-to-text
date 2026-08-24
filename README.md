# speech-to-text

A Burmese-focused speech-to-text harness for running local ASR models over the
same audio and scoring them with the same normalization.

The repository supports four local backends on Apple Silicon: Meta
Omnilingual ASR through fairseq2 or CrispASR, Hugging Face Transformers, and
DataoceanAI Dolphin. Cloud backends are not implemented.

## Why Burmese needs its own harness

- Burmese spaces do not provide reliable word boundaries, so whitespace-token
  WER is unsuitable for this harness. Character Error Rate (CER) is primary.
- Zawgyi and Unicode can render similarly while using incompatible encodings;
  evaluation rejects detector-identified mismatches.
- Unicode normalization and consistent treatment of digits, punctuation, and
  whitespace are required for comparable scores.

The exact scoring behavior and detector limitations are documented in
[Evaluating Burmese ASR](docs/burmese.md).

## Backends

| Backend | Runtime | Models |
|---|---|---|
| `omniasr-torch` | Meta fairseq2 | Upstream omniASR v2 card names |
| `omniasr-gguf` | CrispASR/ggml | Downloadable 300M and 1B GGUF variants |
| `hf` | Hugging Face Transformers | Whisper fine-tunes, MMS, SeamlessM4T, w2v-BERT |
| `dolphin` | DataoceanAI Dolphin | Dolphin base and small |

Run `uv run stt backends` for installed runtimes. Runtime constraints, model
licences, and device behavior are in [Models and runtimes](docs/models.md).

## Setup

The default bootstrap targets Apple Silicon and installs the GGUF and fairseq2
runtimes. Its fairseq2 path requires macOS 14 or newer, Xcode Command Line
Tools, and Homebrew `libsndfile`. All setups require
[uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
xcode-select -p || xcode-select --install
# If the installer opened, finish it before continuing.
xcode-select -p
brew install ffmpeg libsndfile
./scripts/bootstrap.sh
```

Bootstrap installs the project environment and runtime dependencies only. It
does not download model weights, remove partial downloads, or modify existing
model caches. A GGUF-only setup has fewer prerequisites:

```bash
brew install ffmpeg
./scripts/bootstrap.sh --gguf-only
```

See [Setup on macOS](docs/setup-macos.md) for the verified steps, complete extra
sets, cache locations, and recovery without deleting existing weights.

## Select a model

List the available backends and models before downloading anything:

```bash
uv run stt backends
uv run stt models
uv run stt models --backend hf
```

Download one selected model through its normal upstream cache, then pass the
same backend and model ID to transcription:

```bash
uv run stt models --backend hf --download whisper-my-small
uv run stt transcribe recording.mp3 \
    --backend hf --model whisper-my-small
```

Listing models does not download weights. Transcription still downloads the
selected model on first use when it is not cached.

## Usage

```bash
# Inspect this checkout and machine
uv run stt backends
uv run stt models
uv run stt hardware

# Fetch 20 Burmese FLEURS dev clips and their references
uv run stt fetch-fleurs --limit 20

# Transcribe with the default GGUF model
uv run stt transcribe data/fleurs/audio -b omniasr-gguf --limit 5

# Transcribe with one explicitly selected omniASR card
uv run stt transcribe data/fleurs/audio -b omniasr-torch \
    -m omniASR_LLM_Unlimited_7B_v2 --limit 5

# Score a run against the fetched references
uv run stt eval outputs/omniasr-gguf.jsonl \
    --reference data/fleurs/references.tsv

# Compare every model whose existing cache can be verified; no downloads
uv run stt compare data/fleurs/audio \
    --reference data/fleurs/references.tsv --limit 10 \
    --all-cached-models

# Compare named backend defaults; each large first-use download is confirmed
uv run stt compare data/fleurs/audio \
    --reference data/fleurs/references.tsv --limit 10 \
    --only omniasr-gguf --only hf

# Combine runs by per-character vote; put the pivot run first
uv run stt vote out-7b.jsonl out-seamless.jsonl out-dolphin.jsonl \
    -o voted.jsonl

# Check whether text is detector-identified as Unicode or Zawgyi
uv run stt check-encoding data/fleurs/references.tsv
```

Keep the large-download confirmations enabled for comparisons; they apply to
every selected backend.

Forced alignment requires the `hf` extra in the same environment as the
transcription backend. For fairseq2 transcription plus alignment:

```bash
uv sync --frozen --extra omniasr --extra hf
uv run stt transcribe recording.mp3 -b omniasr-torch --align --srt
uv run stt align recording.mp3 --results outputs/omniasr-torch.jsonl
```

Use `uv run stt --help` and `uv run stt <command> --help` as the authoritative
CLI reference.

## Results status

This repository does not currently track the inputs and artifacts required to
reproduce a model accuracy or performance comparison. [Findings
status](docs/findings.md) defines the evidence required before publishing one.

## Layout

```text
src/stt/            CLI, scoring, alignment, voting, telemetry, and backends
tests/              fast offline tests
docs/               workflow, setup, model, scoring, and evidence documentation
data/<dataset>/     audio, references, scripts, and provenance for one corpus
data/fleurs*/       fetched, gitignored public datasets
outputs/            generated, gitignored transcription runs
```

## Adding a backend

Implement `ASRBackend` in `src/stt/backends/`, register the class, and import its
module from `src/stt/backends/__init__.py`. Heavy optional dependencies belong
inside `load()` or another method that needs them. Return one
`TranscriptionResult` per input, including failures, so a bad file does not
discard the rest of a run.

## Documentation

README is the repository documentation index:

- [Git and pull request workflow](docs/GIT_WORKFLOW.md)
- [Setup on macOS](docs/setup-macos.md)
- [Models and runtimes](docs/models.md)
- [Dated external model survey](docs/model-survey.md)
- [Benchmarking and CER](docs/benchmarking.md)
- [Evaluating Burmese ASR](docs/burmese.md)
- [Findings status and evidence requirements](docs/findings.md)
- [Eternity held-out dataset](data/eternity-2026/README.md)
- [Contributing](CONTRIBUTING.md)

## Licence

The harness is MIT licensed. Model weights retain their own licences; consult
the upstream sources linked from [Models and runtimes](docs/models.md) before
redistribution or product use.
