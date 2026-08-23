"""Artifact-backed documentation evidence tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from stt.evidence import UNVERIFIED_MESSAGE, EvidenceError, check_evidence, update_evidence
from stt.provenance import ArtifactDigest, ModelProvenance
from stt.results import TranscriptionResult, read_jsonl, write_jsonl


def _audio_id(number: int) -> str:
    return f"pcm16:16000:1:{number:064x}"


def _result(reference_id: str, *, model: str = "model-a", trusted: bool = True):
    provenance = ModelProvenance(
        backend="test-backend",
        requested_model=model,
        source_kind="test",
        source_locator="test/model",
        upstream_revision="a" * 40,
        revision_status="pinned",
        artifacts=(ArtifactDigest("weights", "weights.bin", 1, "1" * 64),),
        runtime_packages={"test": "1"},
        # A complete manifest has to name the device it actually ran on:
        # without one, a CPU run and a Metal run of the same checkpoint hash
        # to the same execution identity.
        resolved_settings={"device": "cpu", "dtype": "float32"},
        adapter_git_commit="b" * 40,
        uv_lock_sha256="2" * 64,
    ).finalized()
    return TranscriptionResult(
        audio_path=f"prepared/{reference_id}.wav",
        source_path=f"source/{reference_id}.flac",
        source_sha256=f"{7:064x}",
        audio_id=_audio_id(int(reference_id[-1]) + 1),
        reference_id=reference_id,
        text="hello world",
        backend="test-backend",
        model=model,
        elapsed_s=1.0,
        audio_duration_s=2.0,
        trusted=trusted,
        model_provenance=provenance.to_dict(),
    )


def _fixture(
    tmp_path: Path,
    *,
    model: str = "model-a",
    reference_ids: tuple[str, ...] = ("clip1", "clip2"),
    trusted: bool = True,
) -> Path:
    (tmp_path / "evidence").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "docs" / "findings.md").write_text(
        "before\n<!-- stt-evidence:baseline:start -->\nstale 0.27\n"
        "<!-- stt-evidence:baseline:end -->\nafter\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "refs.tsv").write_text(
        "audio_id\ttranscript\nclip1\thello world\nclip2\thello world\n",
        encoding="utf-8",
    )
    write_jsonl(
        [_result(reference_id, model=model, trusted=trusted) for reference_id in reference_ids],
        tmp_path / "outputs" / "run.jsonl",
    )
    manifest = {
        "version": 1,
        "document": "docs/findings.md",
        "blocks": [
            {
                "id": "baseline",
                "corpus": "fixture-corpus",
                "reference": "data/refs.tsv",
                "runs": [
                    {
                        "artifact": "outputs/run.jsonl",
                        "label": "Fixture model",
                        "backend": "test-backend",
                        "model": "model-a",
                    }
                ],
                "metrics": [
                    {"name": "cer", "label": "CER", "digits": 4},
                    {"name": "rtf", "label": "RTF", "digits": 2},
                ],
            }
        ],
    }
    path = tmp_path / "evidence" / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_update_generates_metric_keyed_table_and_preserves_surrounding_prose(tmp_path):
    manifest = _fixture(tmp_path)

    check = update_evidence(manifest)

    document = check.document.read_text(encoding="utf-8")
    assert check.matches
    assert check.publishable
    assert document.startswith("before\n") and document.endswith("\nafter\n")
    assert "| Fixture model | `test-backend` | 0.0000 | 0.50 |" in document
    assert check_evidence(manifest).ok


def test_check_detects_wrong_documented_value(tmp_path):
    manifest = _fixture(tmp_path)
    check = check_evidence(manifest)

    assert not check.matches
    assert check.publishable
    assert "0.0000" in check.expected_text
    assert "stale 0.27" not in check.expected_text


def test_wrong_model_refuses_numeric_output(tmp_path):
    manifest = _fixture(tmp_path, model="other-model")
    check = check_evidence(manifest)

    assert not check.publishable
    assert "model mismatch" in "\n".join(check.issues)
    assert UNVERIFIED_MESSAGE in check.expected_text
    assert "| Fixture model |" not in check.expected_text


def test_wrong_corpus_refuses_numeric_output(tmp_path):
    manifest = _fixture(tmp_path, reference_ids=("clip1",))
    check = check_evidence(manifest)

    assert not check.publishable
    assert "corpus mismatch" in "\n".join(check.issues)
    assert UNVERIFIED_MESSAGE in check.expected_text


def test_untrusted_artifact_refuses_numeric_output(tmp_path):
    manifest = _fixture(tmp_path, trusted=False)
    check = check_evidence(manifest)

    assert not check.publishable
    assert "untrusted" in "\n".join(check.issues)
    assert UNVERIFIED_MESSAGE in check.expected_text


def test_mixed_model_execution_identity_refuses_numeric_output(tmp_path):
    manifest = _fixture(tmp_path)
    artifact = tmp_path / "outputs" / "run.jsonl"
    results = read_jsonl(artifact)
    provenance = ModelProvenance.from_dict(results[1].model_provenance or {})
    results[1].model_provenance = (
        replace(
            provenance,
            requested_settings={"batch_size": 8},
            content_sha256="",
            execution_sha256="",
        )
        .finalized()
        .to_dict()
    )
    write_jsonl(results, artifact)

    check = check_evidence(manifest)

    assert not check.publishable
    assert "one model execution identity" in "\n".join(check.issues)


def test_identity_mismatch_between_runs_refuses_numeric_output(tmp_path):
    manifest = _fixture(tmp_path)
    second_artifact = tmp_path / "outputs" / "second.jsonl"
    second = [_result("clip1"), _result("clip2")]
    second[0].audio_id = _audio_id(999)
    write_jsonl(second, second_artifact)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["blocks"][0]["runs"].append(
        {
            "artifact": "outputs/second.jsonl",
            "label": "Second model run",
            "backend": "test-backend",
            "model": "model-a",
        }
    )
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    check = check_evidence(manifest)

    assert not check.publishable
    assert "audio identity mismatch" in "\n".join(check.issues)
    assert UNVERIFIED_MESSAGE in check.expected_text


def test_legacy_jsonl_defaults_to_untrusted(tmp_path):
    manifest = _fixture(tmp_path)
    artifact = tmp_path / "outputs" / "run.jsonl"
    records = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    for record in records:
        for field in ("audio_id", "source_path", "source_sha256", "reference_id", "trusted"):
            record.pop(field, None)
    artifact.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    check = check_evidence(manifest)

    assert not check.publishable
    assert "untrusted" in "\n".join(check.issues)
    assert UNVERIFIED_MESSAGE in check.expected_text


def test_malformed_trusted_value_refuses_numeric_output(tmp_path):
    manifest = _fixture(tmp_path)
    artifact = tmp_path / "outputs" / "run.jsonl"
    records = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    records[0]["trusted"] = "false"
    artifact.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    check = check_evidence(manifest)

    assert not check.publishable
    assert "trusted field must be a JSON boolean" in "\n".join(check.issues)
    assert UNVERIFIED_MESSAGE in check.expected_text


def test_status_only_block_does_not_inspect_legacy_artifacts(tmp_path):
    manifest = _fixture(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    block = raw["blocks"][0]
    block["status_only"] = True
    block["reference"] = "data/missing-reference.tsv"
    block["runs"][0]["artifact"] = "outputs/missing-run.jsonl"
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    updated = update_evidence(manifest)

    assert not updated.publishable
    assert updated.issues == [f"baseline: status_only: {UNVERIFIED_MESSAGE}"]
    assert check_evidence(manifest).matches


def test_versioned_deriver_renders_a_structured_table(tmp_path):
    manifest = _fixture(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["blocks"][0]["deriver"] = "transcripts:v1"
    raw["blocks"][0]["metrics"] = []
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    check = update_evidence(manifest)

    assert check.publishable
    assert "| Run | Records |" in check.document.read_text(encoding="utf-8")


def test_status_reason_is_rendered_without_opening_legacy_files(tmp_path):
    manifest = _fixture(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["blocks"][0]["status_only"] = True
    raw["blocks"][0]["status_reason"] = "specific upstream limitation"
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    check = update_evidence(manifest)

    assert "specific upstream limitation" in check.document.read_text(encoding="utf-8")


def test_baseline_v2_source_renders_only_after_offline_verification(tmp_path):
    manifest = _fixture(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    block = raw["blocks"][0]
    block["source_kind"] = "baseline-v2"
    block["source"] = str((Path.cwd() / "baselines" / "bench.json").resolve())
    block["deriver"] = "baseline:v1"
    block["runs"] = []
    block["metrics"] = []
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    check = update_evidence(manifest)

    assert check.publishable
    assert "| Subject | RTF | CI low | CI high |" in check.document.read_text(encoding="utf-8")


def test_experiment_v1_source_renders_only_after_offline_verification(tmp_path):
    manifest = _fixture(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    block = raw["blocks"][0]
    block["source_kind"] = "experiment-v1"
    block["source"] = str(
        (Path.cwd() / "evidence" / "experiments" / "mms-batch-smoke" / "experiment.json").resolve()
    )
    block["deriver"] = "experiment:v1"
    block["runs"] = []
    block["metrics"] = []
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    check = update_evidence(manifest)

    assert check.publishable
    assert "| Condition | RTF | Peak RSS MB |" in check.document.read_text(encoding="utf-8")


def test_check_rejects_unowned_markdown_table(tmp_path):
    manifest = _fixture(tmp_path)
    document = manifest.parent.parent / "docs" / "findings.md"
    document.write_text(
        "before\n| stale | value |\n|---|---|\n"
        "<!-- stt-evidence:baseline:start -->\nstale 0.27\n"
        "<!-- stt-evidence:baseline:end -->\nafter\n",
        encoding="utf-8",
    )

    with pytest.raises(EvidenceError, match="unowned Markdown table row"):
        check_evidence(manifest)
