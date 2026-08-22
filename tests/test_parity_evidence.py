"""Verify committed official-reference parity reports."""

from __future__ import annotations

import json
from pathlib import Path

from stt.parity import STAGE_NAMES, ParityReport

ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    ROOT / "evidence/parity/hf-mms-1b-all-cpu-v1.json",
    ROOT / "evidence/parity/hf-mms-1b-all-cpu-v1-reference-first.json",
)


def test_mms_parity_passes_in_both_entrypoint_orders():
    raw = [json.loads(path.read_text(encoding="utf-8")) for path in REPORTS]
    reports = [ParityReport.from_dict(item) for item in raw]
    assert [item.metadata["entrypoint_order"] for item in reports] == [
        ["adapter", "reference"],
        ["reference", "adapter"],
    ]

    cases = []
    for report in reports:
        subject = report.subjects["hf+mms-1b-all"]
        assert subject.parity_eligible
        assert subject.settings == {"device": "cpu", "dtype": "float32"}
        assert len(subject.cases) == 1
        case = subject.cases[0]
        assert not case.contract_issues
        assert case.first_divergent_stage is None
        assert tuple(stage.name for stage in case.stages) == STAGE_NAMES
        assert all(stage.status == "match" for stage in case.stages)
        cases.append(case)

    first_digests = [(stage.adapter_digest, stage.reference_digest) for stage in cases[0].stages]
    second_digests = [(stage.adapter_digest, stage.reference_digest) for stage in cases[1].stages]
    assert first_digests == second_digests
    for path in REPORTS:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/Volumes/" not in text
