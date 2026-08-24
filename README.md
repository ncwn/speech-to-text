# speech-to-text

A Burmese-focused speech-to-text harness for running local ASR models over the
same audio and scoring them with the same normalization.

The repository supports four backend families on Apple Silicon: Meta
Omnilingual ASR through fairseq2 or CrispASR, Hugging Face Transformers, and
DataoceanAI Dolphin. Cloud backends are not implemented.

## Why Burmese needs its own harness

- Burmese has no word delimiters, so Character Error Rate (CER), not Word Error
  Rate, is the primary metric.
- Zawgyi and Unicode can render similarly while using incompatible encodings;
  evaluation rejects mismatched pairs.
- Unicode normalization and consistent treatment of digits, punctuation, and
  whitespace are required for comparable scores.

See [Evaluating Burmese ASR](docs/burmese.md) for the scoring contract.

## Backends

| Backend | Runtime | Models |
|---|---|---|
| `omniasr-torch` | Meta fairseq2 | Upstream omniASR card names |
| `omniasr-gguf` | CrispASR/ggml | Bundled 300M and 1B GGUF cards |
| `hf` | Hugging Face Transformers | Whisper fine-tunes, MMS, SeamlessM4T, w2v-BERT |
| `dolphin` | DataoceanAI Dolphin | Dolphin base and small |

Run `uv run stt backends` for installed runtimes and `uv run stt models` for
the model names accepted by the current checkout. Device defaults, model
constraints, and licences live in [Models and runtimes](docs/models.md).

## Setup

Requires macOS 14 or newer on Apple Silicon, Xcode Command Line Tools,
[uv](https://docs.astral.sh/uv/), ffmpeg, and Homebrew libsndfile.

```bash
xcode-select --install
brew install ffmpeg libsndfile
./scripts/bootstrap.sh
```

The bootstrap script installs the two omniASR runtimes. Install another backend
only when needed:

```bash
uv sync --extra hf
uv sync --extra dolphin
```

See [Setup on macOS](docs/setup-macos.md) for selective installs, cache paths,
large-download recovery, and troubleshooting.

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

# Transcribe with omniASR 7B; auto selects an available accelerator
uv run stt transcribe data/fleurs/audio -b omniasr-torch \
    -m omniASR_LLM_Unlimited_7B_v2 --limit 5

# Score a run against the fetched references
uv run stt eval outputs/omniasr-gguf.jsonl \
    --reference data/fleurs/references.tsv

# Compare installed backends; --yes accepts any large-download prompts
uv run stt compare data/fleurs/audio \
    --reference data/fleurs/references.tsv --limit 10 --yes

# Combine runs by per-character vote; put the pivot/best run first
uv run stt vote out-7b.jsonl out-seamless.jsonl out-dolphin.jsonl \
    -o voted.jsonl

# Check whether text is Unicode or Zawgyi
uv run stt check-encoding data/fleurs/references.tsv
```

Forced alignment requires the Hugging Face extra:

```bash
uv sync --extra hf
uv run stt transcribe recording.mp3 -b omniasr-torch --align --srt
uv run stt align recording.mp3 --results outputs/omniasr-torch.jsonl
```

Use `uv run stt --help` and `uv run stt <command> --help` as the authoritative
CLI reference.

## Results status

Current `main` does not track the result files or provenance needed to reproduce
its earlier CER and performance tables from a clean clone. Those observations
are kept in one place, explicitly marked historical and unverified:
[Measured findings](docs/findings.md). Do not promote them to current baselines
without committing the supporting evidence and a reproducible validation path.

## Layout

```text
src/stt/            CLI, scoring, alignment, voting, telemetry, and backends
tests/              fast offline tests
docs/               durable workflow, setup, model, scoring, and findings docs
data/reference/     tracked hand-corrected reference transcripts
data/fleurs*/       fetched, gitignored public datasets
outputs/            generated, gitignored transcription runs
```

## Adding a backend

Implement `ASRBackend` in `src/stt/backends/`, register the class, and import its
module from `src/stt/backends/__init__.py`. Heavy optional dependencies belong
inside `load()`. Return one `TranscriptionResult` per input, including failures,
so a bad file does not discard the rest of a run.

## Documentation

- [Git and pull request workflow](docs/GIT_WORKFLOW.md)
- [Setup on macOS](docs/setup-macos.md)
- [Models and runtimes](docs/models.md)
- [Dated external model survey](docs/model-survey.md)
- [Evaluating Burmese ASR](docs/burmese.md)
- [Measured findings](docs/findings.md)
- [Held-out reference data](data/reference/README.md)
- [Contributing](CONTRIBUTING.md)

## Licence

The harness is MIT licensed. Model weights retain their own licences; check
[Models and runtimes](docs/models.md) before product use.
