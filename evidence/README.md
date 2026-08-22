# Evidence manifest

`manifest.json` declares the result artifacts allowed to generate measured
blocks in `docs/findings.md`. Paths are relative to the repository root.

Each block names one reference corpus, its JSONL runs, the exact backend and
model expected in each run, and ordered metrics with explicit decimal digits.
The document must contain exactly one matching marker pair:

```markdown
<!-- stt-evidence:block-id:start -->
generated content
<!-- stt-evidence:block-id:end -->
```

Numeric rendering is fail closed. Every record must be trusted, error-free,
carry valid source and canonical PCM identities, carry a complete immutable
model provenance manifest (artifact digests, pinned/content-addressed revision,
runtime identity, and execution hash), cover the complete reference set, and
map each reference to the same waveform across all runs. A mismatch renders the
fixed **Unverified legacy evidence** status instead of partial numbers.

`status_only` records the intentional migration state for current legacy
artifacts. Remove it and declare `metrics` only after trusted runs have been
regenerated. Supported metrics are `cer`, `wer`, `rtf`, `n_total`, and
`n_scored`.
