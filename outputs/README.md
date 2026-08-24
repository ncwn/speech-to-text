# Tracked runs

`outputs/` is a scratch directory by default. The only current evidence files
tracked here are the trusted FLEURS accuracy runs declared by
[`evidence/manifest.json`](../evidence/manifest.json):

- `evidence-fleurs-test-{mms,seamless,dolphin,omni7b}.jsonl`
- `evidence-fleurs-dev-{mms,seamless,dolphin,omni7b}.jsonl`

They carry canonical waveform/source identity, complete model provenance, and
trusted coverage. `docs/findings.md` derives accuracy and tail tables from these
files. Common-wall performance comes from `baselines/bench.json`, while observer,
batching, dtype, utilization, and long-audio results live under `evidence/`.

The older `t120-*.jsonl` files are retained as historical diagnostics only. They
are not current manifest sources and must not be used to publish a new number.
No `eternity-*.jsonl`, vote, or route artifact is tracked: the held-out material
is copyrighted, and derived runs do not yet carry complete lineage provenance.

Track a new output only after adding a fail-closed manifest declaration and a
recomputing test. Leave every other transcription under the normal gitignore.
