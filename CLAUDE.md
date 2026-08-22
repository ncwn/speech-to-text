# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Burmese-focused ASR **evaluation harness**, not a product. Its job is to run several
speech-to-text engines over the same audio and produce numbers comparable to each other.
Most design decisions here exist to stop a measurement from lying.

**`docs/findings.md` is the single home for every measured number.** Code comments and the
README state the rule a measurement produced and link to the anchor that has the figures.
When a measurement changes, it changes there — do not restate figures elsewhere.

## Commands

```bash
./scripts/bootstrap.sh          # prerequisite checks + uv sync
uv sync --extra hf              # transformers (Seamless, MMS, Whisper) on Metal
uv sync --extra omniasr         # PyTorch/fairseq2 (~3 GB of wheels)
uv sync --extra gguf            # CrispASR ggml/Metal
uv sync --extra dolphin         # Dolphin (pulls funasr + modelscope)

make check                      # ruff + offline suite, ~6 s — what the hook runs
make bench                      # isolated smoke gate + weights tests, minutes
make bench-update               # intentionally blocked until provenance gates pass
make hooks                      # enable the tracked pre-commit hook

uv run stt doctor               # runtimes, prerequisites, and where weights live
uv run stt experiment --help    # explicit paired experiments and offline verification
uv run pytest tests/test_vote.py::test_name -x
```

**Guards.** `make check` runs on every commit via `.githooks/pre-commit`. The
bench is deliberately *not* in the hook — it loads five models. Tests needing
real weights carry `@pytest.mark.weights` and are deselected by default.

CLI: `doctor`, `bench` (including `--verify`), `experiment`, `parity`, `backends`, `models`,
`hardware`, `fetch-fleurs`, `transcribe`, `align`, `eval`, `compare`, `vote`, `route`,
`check-encoding`.
`evidence` checks or updates the artifact-backed measured blocks in `docs/findings.md`.

`tests/test_docs_consistency.py` fails if a command is missing from that list, an
artifact-backed findings block differs from the evidence renderer, or a relative/deep
documentation link goes stale.

## Environment constraints

- **Python 3.12 exactly.** `omnilingual-asr` caps there; `fairseq2n` ships cp310–cp312.
- **`torch==2.8.0` is pinned and must not be bumped independently** — `fairseq2n` links
  against libtorch's C++ ABI. The `hf` extra pins the same version so both compose in one venv.
- macOS 14+ / Apple Silicon. Homebrew `libsndfile` is required separately: `fairseq2n`
  `dlopen`s the system library, and the bundled `soundfile` copy does not satisfy it.
- **Weights cache to `<checkout>/.cache/`** by default, resolved in `paths.py`
  (`STT_CACHE_DIR` overrides; installed wheels fall back to `~/.cache`). `paths.py` must
  set `HF_HOME` before anything imports `huggingface_hub`, which is why
  `stt/__init__.py` calls `configure_environment()` at import time.
- **fairseq2 cannot be redirected** — it reads `~/.cache/fairseq2` directly and exposes no
  env var, so repo-local means a symlink. `stt doctor` reports which arrangement is live.
- **The Hugging Face cache is shared with other projects.** Anything that moves it must
  move only the repos in `transformers_asr.MODELS` plus `align.MMS_REPO`, never the
  directory wholesale.

## Architecture

Everything flows through one shape: `TranscriptionResult` (`results.py`), written as JSONL.
Every command either produces or consumes that file.

```
audio.py  →  execution.py  →  backends/*  →  outputs/*.jsonl  →  evaluate / vote / align / route
(identity)   (common wall)    (per engine)     (diagnostic/source records)

bench → measurement request → fresh `bench_worker` process → raw response/repeats
```

- **`registry.py` + `backends/__init__.py`** — backends self-register via `@register` at
  import time. Adding an engine means one new module plus one line in `backends/__init__.py`.
- **`backends/base.py`** — the contract. Two rules are load-bearing: heavy/optional imports
  go **inside** `load()`/`is_available()` (so `stt backends` and the whole test suite work
  with no runtime installed), and a per-file failure is returned as a result with `error`
  set rather than raised (so one bad clip does not abort a sweep).
- **`execution.transcribe_corpus`** is the common synchronized wall boundary. It passes
  the complete corpus once so the backend owns batching/prefetch, stamps trusted identity,
  and retains failed work in one outer resource window.
- **`measurement.py` + `bench_worker.py`** define versioned raw artifacts and isolate one
  subject per process. A worker loads once, warms up, records repeats, and cannot inherit a
  previous model's RSS high-water or runtime state.
- **`results.Segment.source`** records how a timing was obtained (`native` / `chunk` /
  `aligned`), so a mixed run stays honest about which timings were measured and which
  inferred. `text` is the scoring source of truth; segments are additive.
- **`audio.windowed` / `split_on_quiet`** — fixed-window models (Seamless, Dolphin) share
  one chunking loop, so window offsets become real `Segment` timings.
- **`align.py`** recovers timings and per-character confidence for any transcript by CTC
  forced alignment against MMS-1B. This is what gives omniASR — which returns a bare
  `List[str]` — subtitles and per-region confidence.
- **`cascade.py`** + `stt route` — confidence-driven routing, spliced in coarse blocks.

Each backend owns a `MODELS` dict keyed by short name; `stt models` renders all four from
one loop. Keep new backends to that shape.

Default transcription output is `outputs/<backend>-<model>.jsonl`. It includes the model
because keying on the backend alone silently overwrote one run with another. JSONL is
diagnostic by default; only artifacts declared and accepted by `evidence/manifest.json`
can generate published numbers.

## Burmese invariants — do not "simplify" these

`burmese.py` exists because three properties of the script silently corrupt accuracy numbers:

1. **CER is the metric; WER is meaningless.** No word delimiters, so whitespace is stripped
   before comparison. Corpus CER is length-weighted (total edits ÷ total reference chars),
   never an average of per-clip rates.
2. **Zawgyi vs Unicode.** `score_results` detects both sides and marks the item an **error**
   on mismatch rather than reporting the garbage ~1.0 CER it would otherwise produce.
3. **NFC first**, because combining-mark order varies.

`tidy_spacing` is applied to human-facing `.txt`, subtitles and vote inputs — but the JSONL
keeps raw decoder output, which is what scoring reads. Subtitle breaks must not split a
grapheme cluster; `align.splits_a_cluster` guards that.

Full detail in `docs/burmese.md`.

## Measurement conventions

The repo's history is largely "a claim was measured and the measurement changed the code".
Keep that standard:

- **Detect the machine, do not prescribe to it.** `hardware.py` probes dtypes with a real
  matmul and caches per chip + torch version. Thread counts are reported, never imposed.
  No chip constants in the code.
- **Device and dtype defaults are per-model, not global** — and each one is measured. If you
  change a default, measure it and update `docs/findings.md`.
- **Sample size.** 12 clips inverted the model ranking versus 120. FLEURS may be in training
  data, so held-out audio (`data/reference/`, tracked in git precisely because it cannot be
  re-fetched) is the check.
- **Stitching transcripts costs characters per seam**, so anything that splices model outputs
  switches in coarse blocks and must be measured with that tax included.
- CER hides punctuation, decoder loops (`quality.find_loops`), and where time goes.
  `telemetry` keeps exact CPU/wall totals plus optional timestamped CPU, current-RSS,
  Torch-allocation, and whole-GPU observations.
- **Common-wall RTF is the performance boundary.** Backend item `elapsed_s` values have
  different scopes and remain diagnostic only. Torch MPS/CUDA work is synchronized before
  and after the complete-corpus call; failed attempts remain inside that wall.
- **RSS labels are deliberately conservative.** v2 records endpoint RSS and the isolated
  worker's process-lifetime `ru_maxrss` as diagnostics. Neither is an in-window peak or a
  regression gate until observer calibration exists.
- **`ioreg` is a whole-GPU observer, not model occupancy.** Samples carry monotonic offsets,
  source/scope/phase, are time-weighted, and anything completing after the work boundary is
  rejected. Profiling is off for gating runs until observer overhead is calibrated.
- **Transcript hashes are an exact change detector inside one fully pinned subject.** A
  mismatch is a failure requiring investigation, but is not automatically blamed on code
  when model/runtime/input provenance differs.
- **Trusted performance requires v2 artifacts.** Five counterbalanced sessions, three
  warmups, three measured repeats, no profiling, complete audio/reference identity, clean
  source state, immutable model provenance, and archived raw worker artifacts are mandatory.
  A one-worker run is a transcript-only smoke diagnostic. `stt bench --verify` checks the
  archived manifest and recomputes summaries without loading a model.

## Voting

`vote.rover` aligns hypotheses to a **pivot** and votes per character position; ties break
toward the pivot, so adding a system is never worse than a wash. The pivot must be the most
accurate model — voting can only correct characters the pivot proposed — and a pool of only
weak systems achieves nothing. `DEFAULT_WEIGHTS` is a coarse ranking; order matters far more
than the values.

## Docs

`docs/findings.md` (numbers) · `setup-macos.md` (install, troubleshooting) · `models.md`
(omniASR landscape) · `model-survey.md` (Burmese-capable models, verified) · `burmese.md`
(encoding, normalisation). The README is an overview and points at these; keep results out
of it beyond the three-row summary.

Commit messages here are prose: what was measured, what it said, what changed as a result.
