# CTC Product Gate

This is the one bounded experiment required before choosing the first PYAW
live-dictation backend. It measures the official PyTorch/fairseq2 Omnilingual
CTC cards on the checked-in Burmese FLEURS test set.

The gate deliberately does not use `data/audio/eternity.mp3`: non-Unlimited
Omnilingual cards reject audio longer than 40 seconds, and CTC is being tested
as a short-window/live candidate rather than a long-form decoder.

## Run

```bash
scripts/bench_ctc_product_gate.sh
```

The default gate runs:

| Model | Device | Dtype | Batch sizes | Default clips |
| --- | --- | --- | --- | ---: |
| `omniASR_CTC_300M_v2` | `mps` | `float16` | 1, 8 | all 120 |
| `omniASR_CTC_1B_v2` | `mps` | `float16` | 1, 8 | all 120 |

Override the bounded run without editing the script:

```bash
CTC_GATE_LIMIT=16 CTC_GATE_DEVICE=mps CTC_GATE_DTYPE=float16 \
  scripts/bench_ctc_product_gate.sh
```

Artifacts land under `outputs/ctc-product-gate/` and are ignored. Each arm
records CER, RTF, model provenance, GPU utilization, CPU time, and RSS.

## Decision Rule

Choose CTC for the first PYAW live backend only if it is materially faster than
Seamless while staying within the accepted Burmese quality range on both batch
sizes. Batch 1 is the latency proxy; batch 8 is the throughput/utilization
proxy.

If CTC fails the quality gate, move to PYAW with a backend-neutral API and keep
Seamless as a research fallback. Do not start full model fine-tuning from this
result alone; collect real dictation failures first.
