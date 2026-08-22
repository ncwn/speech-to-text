"""Fail-closed contracts for official-reference parity."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from stt.parity import (
    STAGE_NAMES,
    ParityContract,
    ParityError,
    ParityTrace,
    StageTolerance,
    contract_for,
    run_parity,
)
from stt.provenance import ArtifactDigest, ModelBinding, ModelProvenance


def _binding(tmp_path, *, execution_device: str = "cpu") -> ModelBinding:
    path = tmp_path / "model.bin"
    path.write_bytes(b"weights")
    artifact = ArtifactDigest(
        "weights",
        "model.bin",
        path.stat().st_size,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path=str(path),
    )
    provenance = ModelProvenance(
        backend="fake",
        requested_model="model",
        source_kind="test",
        source_locator="test/model",
        upstream_revision="a" * 40,
        revision_status="pinned",
        artifacts=(artifact,),
        runtime_packages={"runtime": "1"},
        resolved_settings={"device": execution_device, "dtype": "float32"},
        adapter_git_commit="b" * 40,
        adapter_git_dirty=False,
        uv_lock_sha256="c" * 64,
    ).finalized()
    assert provenance.complete
    return ModelBinding(provenance, (artifact,))


def _trace(*, decoded_pcm="pcm", feature_delta: float = 0.0) -> ParityTrace:
    return ParityTrace(
        decoded_pcm=decoded_pcm,
        features=np.asarray([[1.0 + feature_delta, 2.0]], dtype=np.float32),
        logits_or_encoder=np.asarray([[0.1, 0.9]], dtype=np.float32),
        token_ids=[1],
        raw_transcript="text",
        final_transcript="text",
    )


class FakeBackend:
    name = "fake"
    model = "model"

    def __init__(self, binding, trace):
        self.model_binding = binding
        self.trace = trace
        self._loaded = False
        self.calls = []

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def model_provenance(self):
        return self.model_binding.provenance

    def parity_adapter_trace(self, path, *, language=None):
        self.calls.append(("adapter", path.name, language))
        return self.trace

    def parity_reference_trace(self, path, *, language=None):
        self.calls.append(("reference", path.name, language))
        return self.trace


def _contract(**reachability) -> ParityContract:
    expected = {stage: True for stage in STAGE_NAMES}
    expected.update(reachability)
    return ParityContract(
        "fake",
        "model",
        expected,
        {stage: StageTolerance(1e-5, 1e-4) for stage in ("features", "logits_or_encoder")},
    )


def test_parity_uses_distinct_bound_instances_and_explicit_hooks(tmp_path):
    audio = tmp_path / "clip.wav"
    import soundfile as sf

    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    binding = _binding(tmp_path)
    adapter = FakeBackend(binding, _trace())
    reference = FakeBackend(binding, _trace())

    report = run_parity(
        adapter,
        audio,
        "mya_Mymr",
        reference_backend=reference,
        contract=_contract(),
        entrypoint_order=("reference", "adapter"),
    )

    subject = report.subjects["fake+model"]
    assert subject.parity_eligible
    assert subject.metadata["entrypoint_order"] == ["reference", "adapter"]
    assert adapter.calls == [("adapter", "clip.wav", "mya_Mymr")]
    assert reference.calls == [("reference", "clip.wav", "mya_Mymr")]
    assert not adapter._loaded and not reference._loaded


def test_required_missing_stage_and_real_pcm_divergence_fail_parity(tmp_path):
    audio = tmp_path / "clip.wav"
    import soundfile as sf

    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    binding = _binding(tmp_path)
    missing = replace(_trace(decoded_pcm="adapter"), features=None)
    report = run_parity(
        FakeBackend(binding, missing),
        audio,
        reference_backend=FakeBackend(binding, _trace(decoded_pcm="reference")),
        contract=_contract(),
    )
    case = report.subjects["fake+model"].cases[0]

    assert case.first_divergent_stage == "decoded_pcm"
    assert "required stage features" in " ".join(case.contract_issues)
    assert not report.subjects["fake+model"].parity_eligible


def test_declared_unreachable_stage_must_be_missing_on_both_sides(tmp_path):
    audio = tmp_path / "clip.wav"
    import soundfile as sf

    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    binding = _binding(tmp_path)
    without_logits = replace(_trace(), logits_or_encoder=None)
    report = run_parity(
        FakeBackend(binding, without_logits),
        audio,
        reference_backend=FakeBackend(binding, without_logits),
        contract=_contract(logits_or_encoder=False),
    )

    assert report.subjects["fake+model"].parity_eligible


def test_parity_rejects_missing_hooks_and_execution_drift(tmp_path):
    audio = tmp_path / "clip.wav"
    import soundfile as sf

    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    binding = _binding(tmp_path)
    adapter = FakeBackend(binding, _trace())
    reference = FakeBackend(binding, _trace())
    reference.parity_reference_trace = None
    with pytest.raises(ParityError, match="does not implement parity_reference_trace"):
        run_parity(
            adapter,
            audio,
            reference_backend=reference,
            contract=_contract(),
        )

    drifted = ModelBinding(
        replace(
            binding.provenance,
            resolved_settings={"device": "mps", "dtype": "float32"},
            content_sha256=None,
            execution_sha256=None,
        ).finalized(),
        binding.paths,
    )
    with pytest.raises(ParityError, match="do not share one immutable model binding"):
        run_parity(
            FakeBackend(binding, _trace()),
            audio,
            reference_backend=FakeBackend(drifted, _trace()),
            contract=_contract(),
        )


def test_dolphin_contract_requires_all_observed_stages():
    contract = contract_for("dolphin", "small")
    assert contract.expected_reachability == {stage: True for stage in STAGE_NAMES}
    assert set(contract.tolerances) == {"features", "logits_or_encoder"}


def test_omniasr_contract_requires_all_observed_stages():
    contract = contract_for("omniasr-torch", "omniASR_LLM_Unlimited_7B_v2")
    assert contract.expected_reachability == {stage: True for stage in STAGE_NAMES}
    assert set(contract.tolerances) == {"features", "logits_or_encoder"}
