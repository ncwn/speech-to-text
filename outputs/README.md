# Tracked runs

Working directory for transcription output. Almost everything here is scratch
and gitignored — **except the runs below**, which are the evidence for the
numbers in [`docs/findings.md`](../docs/findings.md).

They are tracked for the same reason `data/reference/` is: not because they are
large or precious, but because losing them costs hours of GPU time and, until
they are regenerated, every published figure becomes unverifiable. At 1.1 MB
that is a bad trade to leave to chance.

This is not hypothetical. The voting result was stated three times with two
different values — `README.md` said CER 0.0945, `vote.py` and `cli.py` said
0.0930. Re-scoring `t120-voted.jsonl` settled it in seconds. Without the file
the tie-breaker would have been a fresh four-model vote over 120 clips.

## The 120-clip baseline

FLEURS Burmese `test` split, 29.6 minutes. Backs
[Baseline](../docs/findings.md#baseline), [Tails](../docs/findings.md#tails) and
[Routing](../docs/findings.md#routing).

| File | Model |
|---|---|
| `t120-omni7b.jsonl` | omniASR 7B Unlimited v2 |
| `t120-omni3b.jsonl` | omniASR 3B Unlimited v2 |
| `t120-omni300m.jsonl` | omniASR 300M Unlimited v2 |
| `t120-seamless-1.jsonl` | SeamlessM4T v2 large |
| `t120-dolphin-small.jsonl` | Dolphin small |
| `t120-mms.jsonl` | MMS-1B |
| `t120-voted.jsonl` | four-model vote, 7B pivot |

`t120-seamless-2.jsonl` is a **second, independent run of the same model on the
same clips**. It exists to show that transcripts are byte-deterministic —
120/120 identical — which is what lets `stt bench` compare by hash with no
tolerance instead of arguing about a threshold.

## The held-out recording

A 16.8-minute narration with a hand-corrected reference, described in
[`data/reference/`](../data/reference/README.md). Backs
[Held-out](../docs/findings.md#held-out), [Voting](../docs/findings.md#voting)
and [Seam tax](../docs/findings.md#seam-tax).

| File | Model |
|---|---|
| `eternity-7b.jsonl` | omniASR 7B — the only run carrying segments and confidence |
| `eternity-seamless.jsonl` | SeamlessM4T v2 |
| `eternity-dolphin.jsonl` | Dolphin small |
| `eternity-gguf.jsonl` | omniASR 300M Q4_K — the quantisation failure case |
| `eternity-voted.jsonl` | four-model vote |

The audio itself is **not** in the repo; it is third-party copyrighted material.

## Re-checking a published number

```bash
uv run stt eval outputs/t120-voted.jsonl -r data/fleurs-test/references.tsv
uv run stt eval outputs/eternity-voted.jsonl -r data/reference/references.tsv
```

Both must reproduce the figures in
[Voting](../docs/findings.md#voting). The 120-clip runs need the test split:
`uv run stt fetch-fleurs --split test --limit 120 --dest data/fleurs-test`.

## Adding a run here

Track a run only when a document cites it. Give it a name that says which model
and which clip set, add a row above, and leave everything else gitignored —
`outputs/` is a working directory first and an archive second.
