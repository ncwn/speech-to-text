# Eternity 2026

A 16.8-minute Burmese film-recap narration with one speaker. The tracked
reference contains 18,767 Unicode characters and was hand-corrected by a native
speaker. FLEURS may overlap model training data, so this supplies a separate
held-out check.

```text
audio/eternity.mp3  third-party source; local and ignored by Git
transcripts/eternity.txt  human-readable corrected transcript
references.tsv      evaluator input keyed by audio ID `eternity`
```

The source audio is copyrighted and is not distributed by this repository.
Place an authorized local copy at the path above before running the commands in
the [benchmarking guide](../../../../docs/benchmarking.md). Only `references.tsv` is
consumed by `stt eval`; the transcript copy is retained for human review. Its
source URL and retrieval date are not recorded, so this dataset is not
publishable benchmark evidence until that provenance is supplied.
