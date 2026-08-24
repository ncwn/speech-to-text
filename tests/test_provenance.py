"""Canonical model provenance rejects forged identity while remaining path-independent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import stt.provenance as provenance
from stt.backends.base import ASRBackend
from stt.provenance import (
    ArtifactDigest,
    ModelBinding,
    ModelProvenance,
    ProvenanceError,
    digest_file,
    unresolved_execution_issues,
)


def _provenance() -> ModelProvenance:
    return ModelProvenance(
        backend="test",
        requested_model="model",
        source_kind="huggingface",
        source_locator="org/model",
        upstream_revision="a" * 40,
        revision_status="pinned",
        artifacts=(ArtifactDigest("weights", "model.bin", 1, "1" * 64, path="/cache/a/model.bin"),),
        runtime_packages={"runtime": "1"},
        requested_settings={"batch_size": 1},
        resolved_settings={"device": "cpu", "dtype": "float32"},
        adapter_git_commit="b" * 40,
        uv_lock_sha256="2" * 64,
    ).finalized()


class _ConsumerBackend(ASRBackend):
    name = "consumer"

    def __init__(self, model: str = "model", **options):
        super().__init__(model, **options)
        self.resolved_device = "cpu"
        self.resolved_dtype = "float32"

    def load(self):
        self._loaded = True

    def transcribe(self, audio_paths: list[Path], language=None, batch_size=1):
        del language, batch_size
        return []


class _AliasedConsumerBackend(_ConsumerBackend):
    provenance_backend_aliases = ("legacy-consumer",)
    provenance_model_aliases = ("legacy-model",)


def test_forged_canonical_hash_is_rejected():
    raw = _provenance().to_dict()
    raw["execution_sha256"] = "f" * 64

    with pytest.raises(ProvenanceError, match="canonical payload"):
        ModelProvenance.from_dict(raw)


def test_execution_identity_includes_adapter_commit_and_runtime_settings():
    original = _provenance()
    other_commit = replace(
        original,
        adapter_git_commit="c" * 40,
        content_sha256="",
        execution_sha256="",
    ).finalized()
    other_batch = replace(
        original,
        requested_settings={"batch_size": 8},
        content_sha256="",
        execution_sha256="",
    ).finalized()

    assert other_commit.execution_sha256 != original.execution_sha256
    assert other_batch.execution_sha256 != original.execution_sha256


def test_schema_one_execution_identity_remains_readable_with_legacy_hash_semantics():
    original = replace(
        _provenance(),
        schema_version=1,
        content_sha256="",
        execution_sha256="",
    ).finalized()
    other_commit = replace(
        original,
        adapter_git_commit="c" * 40,
        content_sha256="",
        execution_sha256="",
    ).finalized()

    assert other_commit.execution_sha256 == original.execution_sha256
    assert ModelProvenance.from_dict(original.to_dict()) == original
    assert original.complete


def test_local_artifact_path_does_not_change_content_identity():
    original = _provenance()
    relocated_artifact = replace(original.artifacts[0], path="/other/cache/model.bin")
    relocated = replace(
        original,
        artifacts=(relocated_artifact,),
        content_sha256="",
        execution_sha256="",
    ).finalized()

    assert relocated.content_sha256 == original.content_sha256
    assert relocated.execution_sha256 == original.execution_sha256


def test_binding_rejects_backend_and_model_identity_mismatch():
    value = _provenance()
    backend = _ConsumerBackend()
    binding = ModelBinding(
        replace(
            value,
            backend="other-backend",
            requested_model="other-model",
            content_sha256="",
            execution_sha256="",
        ).finalized(),
        value.artifacts,
    )

    with pytest.raises(ProvenanceError, match="model binding backend does not match consumer"):
        backend.bind_model(binding)


def test_binding_accepts_explicit_compatibility_aliases():
    value = _provenance()
    binding = ModelBinding(
        replace(
            value,
            backend="legacy-consumer",
            requested_model="legacy-model",
            content_sha256="",
            execution_sha256="",
        ).finalized(),
        value.artifacts,
    )

    backend = _AliasedConsumerBackend()
    backend.bind_model(binding)

    assert backend.model_binding is binding


def test_bound_settings_require_exact_runtime_key_sets():
    value = _provenance()
    value = replace(
        value,
        requested_settings={"batch_size": 1, "language": "mya_Mymr"},
        resolved_settings={
            "device": "cpu",
            "dtype": "float32",
            "n_threads": None,
            "chunk_seconds": None,
        },
        content_sha256="",
        execution_sha256="",
    ).finalized()
    value = replace(value, backend="consumer").finalized()
    backend = _ConsumerBackend(batch_size=1)
    backend.bind_model(ModelBinding(value, value.artifacts))
    backend.set_execution_settings(batch_size=1, unexpected=True)

    current = backend.model_provenance()

    assert any(
        "requested setting keys changed from preflight" in issue
        and "language" in issue
        and "unexpected" in issue
        for issue in current.issues
    )
    assert not current.complete


def test_bound_resolved_settings_do_not_inherit_unknown_preflight_keys():
    value = replace(
        _provenance(),
        backend="consumer",
        resolved_settings={
            "device": "cpu",
            "dtype": "float32",
            "precision_probe": "float32",
        },
        content_sha256="",
        execution_sha256="",
    ).finalized()
    backend = _ConsumerBackend()
    backend.bind_model(ModelBinding(value, value.artifacts))

    current = backend.model_provenance()

    assert any(
        "resolved setting keys changed from preflight" in issue and "precision_probe" in issue
        for issue in current.issues
    )
    assert "precision_probe" not in current.resolved_settings
    assert not current.complete


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        (
            "upstream_revision",
            "A" * 40,
            "pinned model provenance requires a 40-character",
        ),
        ("adapter_git_commit", "b" * 39, "invalid adapter Git commit"),
        ("uv_lock_sha256", "c" * 63, "invalid uv.lock SHA-256"),
        ("content_sha256", "D" * 64, "invalid provenance content SHA-256"),
        ("execution_sha256", "e" * 63, "invalid provenance execution SHA-256"),
    ],
)
def test_identity_fields_require_canonical_hex_formats(field_name, value, message):
    candidate = replace(_provenance(), **{field_name: value})
    if field_name not in {"content_sha256", "execution_sha256"}:
        candidate = replace(candidate, content_sha256="", execution_sha256="")

    with pytest.raises(ProvenanceError, match=message):
        candidate.validate()


def test_incomplete_diagnostic_provenance_remains_serializable_and_ineligible():
    diagnostic = replace(
        _provenance(),
        upstream_revision="main",
        revision_status="unknown",
        artifacts=(),
        runtime_packages={},
        adapter_git_commit=None,
        uv_lock_sha256=None,
        issues=("Hugging Face snapshot path is unavailable",),
        content_sha256="",
        execution_sha256="",
    )

    diagnostic.validate()
    restored = ModelProvenance.from_dict(diagnostic.to_dict())

    assert restored.upstream_revision == "main"
    assert restored.issues == diagnostic.issues
    assert not restored.complete


def test_model_binding_rejects_duplicate_path_names():
    provenance_value = _provenance()
    duplicate = replace(provenance_value.artifacts[0], path="/other/cache/model.bin")

    with pytest.raises(ProvenanceError, match="duplicate path names"):
        ModelBinding(provenance_value, (provenance_value.artifacts[0], duplicate)).validate()


def test_model_binding_rejects_unexpected_path_entries():
    provenance_value = _provenance()
    extra = ArtifactDigest("weights", "extra.bin", 1, "1" * 64, path="/cache/extra.bin")

    with pytest.raises(ProvenanceError, match="unexpected path"):
        ModelBinding(provenance_value, (*provenance_value.artifacts, extra)).validate()


def test_model_binding_rejects_path_digest_mismatch():
    provenance_value = _provenance()
    mismatched = replace(provenance_value.artifacts[0], sha256="2" * 64)

    with pytest.raises(ProvenanceError, match="does not match its provenance digest"):
        ModelBinding(provenance_value, (mismatched,)).validate()


def test_model_binding_rejects_duplicate_provenance_artifact_names():
    provenance_value = _provenance()
    duplicate = replace(provenance_value.artifacts[0], path="/other/cache/model.bin")
    duplicate_provenance = replace(
        provenance_value,
        artifacts=(provenance_value.artifacts[0], duplicate),
        content_sha256="",
        execution_sha256="",
    )

    with pytest.raises(ProvenanceError, match="duplicate artifact names"):
        ModelBinding(duplicate_provenance, duplicate_provenance.artifacts).validate()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda stat: stat.pop("ctime_ns"),
        lambda stat: stat.__setitem__("extra", 1),
        lambda stat: stat.__setitem__("size", str(stat["size"])),
        lambda stat: stat.__setitem__("inode", True),
    ],
)
def test_malformed_cached_stat_forces_a_fresh_hash(tmp_path, monkeypatch, mutation):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"weights")
    cache_root = tmp_path / "cache"
    digest_file(artifact, cache_root=cache_root)
    cache_path = next((cache_root / "provenance").glob("*.json"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    mutation(cached["stat"])
    cache_path.write_text(json.dumps(cached), encoding="utf-8")

    original_sha256 = provenance.hashlib.sha256
    calls: list[bytes] = []

    def tracked_sha256(value: bytes = b""):
        calls.append(value)
        return original_sha256(value)

    monkeypatch.setattr(provenance.hashlib, "sha256", tracked_sha256)
    result = digest_file(artifact, cache_root=cache_root)

    assert result.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert any(value == b"" for value in calls)
    repaired = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(repaired["stat"]) == {"device", "inode", "size", "mtime_ns", "ctime_ns"}


def test_cached_record_with_extra_field_forces_a_fresh_hash(tmp_path, monkeypatch):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"weights")
    cache_root = tmp_path / "cache"
    digest_file(artifact, cache_root=cache_root)
    cache_path = next((cache_root / "provenance").glob("*.json"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    cached["extra"] = "reject"
    cache_path.write_text(json.dumps(cached), encoding="utf-8")

    original_sha256 = provenance.hashlib.sha256
    calls: list[bytes] = []

    def tracked_sha256(value: bytes = b""):
        calls.append(value)
        return original_sha256(value)

    monkeypatch.setattr(provenance.hashlib, "sha256", tracked_sha256)
    digest_file(artifact, cache_root=cache_root)

    assert any(value == b"" for value in calls)


def _pinned_provenance(**overrides: object) -> ModelProvenance:
    """A manifest that is complete apart from whatever the caller overrides."""
    fields: dict[str, object] = {
        "backend": "omniasr-gguf",
        "requested_model": "llm-unlimited-300m-v2",
        "source_kind": "huggingface-gguf",
        "source_locator": f"https://huggingface.co/x/y/resolve/{'a' * 40}/m.gguf",
        "upstream_revision": "a" * 40,
        "revision_status": "pinned",
        "artifacts": (ArtifactDigest("weights", "m.gguf", 7, "1" * 64),),
        "runtime_packages": {"crispasr": "0.8.29"},
        "requested_settings": {},
        "resolved_settings": {"device": "cpu", "dtype": None, "n_threads": 4},
        "adapter_git_commit": "b" * 40,
        "adapter_git_dirty": False,
        "uv_lock_sha256": "c" * 64,
    }
    fields.update(overrides)
    return ModelProvenance(**fields).finalized()  # type: ignore[arg-type]


@pytest.mark.parametrize("device", [None, "", "   "])
def test_an_unresolved_device_cannot_be_complete(device):
    """CrispASR picks CUDA/Metal/Vulkan/CPU internally and never says which.

    Without a device the execution hash is the same whichever one it chose, so
    two runs that are not comparable become indistinguishable in provenance.
    That is why this is refused outright rather than recorded as a blank label.
    """
    provenance = _pinned_provenance(
        resolved_settings={"device": device, "dtype": None, "n_threads": 4}
    )

    assert provenance.complete is False
    assert unresolved_execution_issues(provenance.resolved_settings)


def test_a_resolved_device_stays_complete():
    """The check must not make every backend ineligible along the way."""
    assert _pinned_provenance().complete is True
    assert unresolved_execution_issues({"device": "mps"}) == ()


def test_completeness_is_recomputed_rather_than_read_from_stored_issues():
    """Provenance written before the check existed must not be trusted now.

    `complete` is consulted on manifests loaded back out of JSONL, where the
    recorded issue list is whatever the writer happened to know at the time.
    """
    stored = _pinned_provenance(
        resolved_settings={"device": None, "dtype": None, "n_threads": 4}
    ).to_dict()
    stored["issues"] = []

    assert ModelProvenance.from_dict(stored).complete is False


def test_bound_dolphin_cpu_provenance_keeps_resolved_dtype_after_load(tmp_path, monkeypatch):
    """Dolphin's bound worker state must match its preflight dtype."""
    import sys
    from types import SimpleNamespace

    from stt.backends.dolphin import DolphinBackend

    calls: list[tuple[str, str, str]] = []

    def load_model(size: str, directory: str, device: str) -> object:
        calls.append((size, directory, device))
        return object()

    monkeypatch.setitem(sys.modules, "dolphin", SimpleNamespace(load_model=load_model))

    preflight_backend = DolphinBackend("small", device="cpu")
    assert preflight_backend._resolve_dtype("cpu") == "float32"

    artifact_path = tmp_path / "small.pt"
    artifact_path.write_bytes(b"weights")
    binding_provenance = _pinned_provenance(
        backend="dolphin",
        requested_model="small",
        source_kind="modelscope",
        source_locator="DataoceanAI/dolphin-small",
        artifacts=(
            ArtifactDigest(
                "weights",
                "small.pt",
                artifact_path.stat().st_size,
                hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                path=str(artifact_path),
            ),
        ),
        requested_settings={"device": "cpu"},
        resolved_settings={
            "device": "cpu",
            "dtype": "float32",
            "n_threads": None,
            "chunk_seconds": None,
            "size": "small",
        },
    )
    backend = DolphinBackend("small", device="cpu")
    backend.bind_model(ModelBinding(binding_provenance, binding_provenance.artifacts))

    backend.load()
    current = backend.model_provenance()

    assert calls == [("small", str(tmp_path), "cpu")]
    assert backend.resolved_dtype == "float32"
    assert current.resolved_settings["dtype"] == "float32"
    assert not any("resolved setting 'dtype' changed" in issue for issue in current.issues)
    assert current.complete is True
