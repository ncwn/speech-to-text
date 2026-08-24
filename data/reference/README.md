# Held-out reference transcripts

Hand-corrected references for audio kept outside the repository. FLEURS is public
and may overlap model training data, so this clip provides a separate held-out
check. The source audio is not stored here or fetched by the harness; only the
reference text is tracked.

## `eternity-2026`

A 16.8-minute Burmese narration (film recap, single speaker).
The reference contains 18,767 Unicode characters and was hand-corrected by a
native speaker.

The audio is third-party copyrighted material. Keep it outside the repository
and pass its path explicitly to `stt transcribe`.

## Reference ID contract

`stt eval` reads the two-column TSV below. The human-readable `.txt` copy is
optional; only `references.tsv` is consumed by the evaluator.

```
data/reference/<name>.txt          the transcript, human-readable
data/reference/references.tsv      audio_id <TAB> transcript, whitespace flattened
```

`audio_id` must match the audio stem recorded in the transcription JSONL. A
source that is not already 16 kHz mono is converted to
`<parent>__<stem>.16k.wav` under `data/.converted/`; the parent directory name
is therefore part of the stem. For example, a source under `Downloads/` gets a
`Downloads__...` result stem. Evaluation first tries that exact stem and then
tries the same stem without the `.16k` suffix. It does not remove the parent
prefix.

The included row is keyed for the original `Downloads__...` result. Putting the
same file under `data/audio/` would produce an `audio__...` stem and will not
match that row; keep the source parent name or update the row to the stem in the
new JSONL.
