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

Measured on this clip:

| Model | CER |
|---|---:|
| omniASR 7B fp32 | 0.0857 |
| SeamlessM4T v2 | 0.0887 |
| Dolphin small | 0.1366 |
| omniASR 300M Q4_K | 0.2245 |

**Read those with the proper-noun caveat.** The protagonist's name occurs 99
times (2.9 % of the text). omniASR 7B spells it `ဂျုံး` as the reference does;
SeamlessM4T spells it `ဂျွန်` every time. Correct that one name and Seamless
(0.0725) overtakes omniASR 7B (0.0831). See the README's held-out section.

## Adding another

Two files, then it works with `stt eval` unchanged:

```
data/reference/<name>.txt          the transcript, human-readable
data/reference/references.tsv      audio_id <TAB> transcript, whitespace flattened
```

The `audio_id` must match the audio file's stem as the transcribe run saw it,
including the `.16k` suffix if the harness resampled it.
