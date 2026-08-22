# Held-out reference transcripts

Ground truth for audio the models cannot have trained on. FLEURS is public and
appears in several of these models' training sets, so a benchmark that only uses
FLEURS cannot tell accuracy from memorisation. Everything here is hand-corrected
and unavailable anywhere else — unlike `data/fleurs*/`, it cannot be re-fetched,
which is why this directory is tracked in git while the rest of `data/` is not.

## `eternity-2026`

A 16.8-minute Burmese narration (film recap, single speaker, studio-clean).
18,767 characters, Unicode, hand-corrected by a native speaker.

The audio itself is **not** in the repo — it is third-party copyrighted
material. `references.tsv` keys on the audio stem, so drop the source file into
`data/audio/` and the harness will find it.

What every model scores on this clip, and why the top two are separated
entirely by one proper noun, is in
[`docs/findings.md`](../../docs/findings.md#held-out).

## Adding another

Two files, then it works with `stt eval` unchanged:

```
data/reference/<name>.txt          the transcript, human-readable
data/reference/references.tsv      audio_id <TAB> transcript, whitespace flattened
```

The `audio_id` must match the audio file's stem as the transcribe run saw it,
including the `.16k` suffix if the harness resampled it.
