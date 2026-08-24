# Findings status

No reproducible measured findings are tracked in this repository.

Earlier drafts contained CER, WER, real-time factor, held-out, routing, device,
precision, and ensemble conclusions without the complete inputs and artifacts
needed to audit them from a clean clone. Those numbers are not current
baselines and are not retained as historical evidence.

## Evidence required for a measured finding

A future comparison must track or durably identify:

- The evaluation manifest, corpus split, audio identities and checksums,
  reference transcripts, and their provenance and licence.
- Every model repository, immutable revision, weight checksum, backend version,
  dependency lock, normalization option, and language code.
- The complete command, hardware, macOS version, resolved device and dtype, and
  whether any device retry occurred.
- Per-file transcription JSONL, failures, raw timing and resource telemetry,
  aggregate output, and enough logs to diagnose exclusions.
- A clean-clone procedure that reproduces the aggregate from the tracked or
  immutable inputs without relying on an operator's existing caches.

Small smoke runs may validate that a backend executes, but they are not model
selection evidence. Published comparisons must use the same committed manifest
and scoring contract for every model. Bot-generated summaries and review
comments are advisory until a person verifies the underlying artifacts.
