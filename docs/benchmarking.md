# Benchmarking and CER

Keep each evaluation corpus self-contained:

```text
data/<dataset>/
  README.md                 source, retrieval date, licence, split, corrections
  audio/<audio-id>.<ext>    local input; ignored by Git
  transcripts/<audio-id>.txt  optional human-readable scripts
  references.tsv           audio_id<TAB>transcript
```

FLEURS already uses the core `audio/` plus `references.tsv` shape. Use a unique
filename stem for every clip within a dataset. The evaluator matches `audio_id`
to that stem; directories and extensions do not participate. Audio that needs
conversion is cached under `data/.converted/` without changing this ID. Do not
commit audio unless its licence and repository policy explicitly allow
redistribution.

`references.tsv` may include this header:

```tsv
audio_id	transcript
clip-001	The complete reference transcript for clip-001.
```

The transcript must occupy one TSV field. Keep tabs out of it and flatten
newlines to spaces. The sidecar transcript files are for review only; the CLI
does not read them.

## Prepare a public test set

Fetch the Burmese FLEURS test split, or substitute a prepared external dataset:

```bash
uv run stt fetch-fleurs --split test --limit 120 --dest data/fleurs-test
uv run stt check-encoding data/fleurs-test/references.tsv
```

Record the source URL, retrieval date, rights, selection rule, and any manual
transcript corrections in the dataset README before treating results as
reproducible evidence.

## Benchmark one model

List and selectively download the model, then give every run a dataset- and
model-specific output path:

```bash
uv run stt models --backend hf
uv run stt models --backend hf --download whisper-my-small

uv run stt transcribe data/fleurs-test/audio \
  --backend hf --model whisper-my-small \
  --output outputs/fleurs-test/single/hf--whisper-my-small.jsonl

uv run stt eval outputs/fleurs-test/single/hf--whisper-my-small.jsonl \
  --reference data/fleurs-test/references.tsv --per-file
```

`transcribe` can download the selected model on first use. Running the explicit
`models --download` step first makes that network and disk change deliberate.

## Benchmark cached models in batch

Run every model whose complete cache the adapter can verify, without downloading
new weights:

```bash
uv run stt compare data/fleurs-test/audio \
  --reference data/fleurs-test/references.tsv \
  --all-cached-models \
  --output-dir outputs/fleurs-test/cached-sweep
```

Each run is written as `<backend>--<model>.jsonl`. A plain `stt compare` runs
only the verifiably cached default model for each backend; `--only` selects
backend defaults and may prompt for a first-use download.

Output writers replace an existing path. Use a new `--output-dir` for every
measurement run, and keep single-model outputs in a different directory from a
cached-model sweep.

The batch table suppresses CER when any result failed or lacks a matching
reference. Inspect an individual JSONL with `stt eval --per-file` before comparing
models, and require zero failed rows for a complete corpus comparison.

## Migrate the split layout

Moving `data/audio/<name>` and its transcript into one dataset directory does
not change command parsing; `transcribe`, `compare`, and `eval` all take explicit
paths. Set the manifest `audio_id` to the moved audio filename stem.

Historical JSONL records retain the converted path used by that run. If a move
also renames the source, those results will not match the new manifest ID. Keep
the old manifest with the old outputs or regenerate the transcription; do not
compare a partial score across the two identities.

## Read CER

CER is the total character edits divided by total normalized reference
characters, so lower is better and `0.10` means about 10 percent. It is
length-weighted across the corpus, not an average of per-file CER values.

`--keep-whitespace` and `--keep-punctuation` change the scoring contract and
must be recorded with the result. Whitespace-token WER is not a valid Burmese
ranking metric. The complete normalization and encoding contract lives in
[Evaluating Burmese ASR](burmese.md).

For a publishable comparison, retain the exact dataset manifest, model revision,
command, dependency lock, device and dtype, JSONL outputs, failures, and host
details required by [Findings status](findings.md).
