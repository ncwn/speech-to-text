# Evidence manifest

`manifest.json` declares the result artifacts allowed to generate measured
blocks in `docs/findings.md`. Paths are relative to the repository root.

Each block names one reference corpus and one typed source:

- trusted transcription JSONL for accuracy/transcript-derived tables;
- `baseline-v2` for common-wall performance; or
- `experiment-v1` for observer, input, dtype, batching, and utilization tables.

Typed sources bind the tracked reference checksum/population and declared input
count before rendering. JSONL blocks additionally name the exact backend/model,
resolved settings, and ordered metrics with explicit decimal digits.
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

Measured-looking prose outside generated blocks is rejected. A current figure
must be derived inside its manifest-owned marker pair; historical mechanisms are
described qualitatively or marked status-only.

`status_only` records an explicit policy or upstream blocker, such as
non-redistributable held-out data, missing aligned artifacts, or unresolved
native device identity. Remove it only when a trusted reproducible source exists.
