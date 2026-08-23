"""Immutable model provenance and content-addressed artifact digests.

The benchmark must distinguish a model name from the bytes and runtime that
were actually executed.  This module keeps that identity independent of local
cache paths while retaining paths as diagnostic bindings for workers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from stt.paths import cache_dir

PROVENANCE_SCHEMA_VERSION = 2
SUPPORTED_PROVENANCE_SCHEMA_VERSIONS = frozenset({1, PROVENANCE_SCHEMA_VERSION})
DIGEST_CACHE_VERSION = 1
_STAT_FINGERPRINT_KEYS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})
_DIGEST_CACHE_KEYS = frozenset({"version", "stat", "size_bytes", "sha256"})
_HEX_DIGITS = frozenset("0123456789abcdef")


class ProvenanceError(ValueError):
    """A provenance contract or artifact binding is invalid."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in _HEX_DIGITS for character in value)
    )


def _validate_optional_hex(value: object, *, field_name: str, length: int) -> None:
    # Empty values intentionally represent unavailable diagnostic identity;
    # any supplied identity must still be canonical.
    if value is None or value == "":
        return
    if not _is_lower_hex(value, length):
        raise ProvenanceError(
            f"invalid {field_name}: expected {length} lowercase hexadecimal characters"
        )


def digest_bytes(value: bytes, *, role: str, name: str) -> ArtifactDigest:
    """Create a path-independent digest for a small selected metadata blob."""
    return ArtifactDigest(
        role=role,
        name=name,
        size_bytes=len(value),
        sha256=_sha256_bytes(value),
    )


def _stat_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _valid_stat_fingerprint(value: object) -> bool:
    """Return whether a cache stat is the complete, JSON-safe fingerprint."""
    if not isinstance(value, dict) or set(value) != _STAT_FINGERPRINT_KEYS:
        return False
    return all(type(item) is int and item >= 0 for item in value.values())


def _same_stat(left: object, right: object) -> bool:
    """Compare only complete fingerprints with exact integer field types."""
    return _valid_stat_fingerprint(left) and _valid_stat_fingerprint(right) and left == right


def _valid_digest_cache_record(value: object, expected_stat: dict[str, int]) -> bool:
    """Check every field before trusting a cached digest."""
    if not isinstance(value, dict) or set(value) != _DIGEST_CACHE_KEYS:
        return False
    if type(value["version"]) is not int or value["version"] != DIGEST_CACHE_VERSION:
        return False
    cached_stat = value["stat"]
    if not _same_stat(cached_stat, expected_stat):
        return False
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 0:
        return False
    if value["size_bytes"] != expected_stat["size"]:
        return False
    digest = value["sha256"]
    return (
        type(digest) is str
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


@dataclass(frozen=True)
class ArtifactDigest:
    """A path-independent digest of one loader-consumed artifact."""

    role: str
    name: str
    size_bytes: int
    sha256: str
    path: str | None = None
    stat: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.role, str) or not isinstance(self.name, str):
            raise ProvenanceError("artifact role and name must be strings")
        if not self.role or not self.name:
            raise ProvenanceError("artifact role and name are required")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ProvenanceError(f"negative artifact size for {self.name}")
        if not _is_lower_hex(self.sha256, 64):
            raise ProvenanceError(f"invalid artifact SHA-256 for {self.name}")
        if self.stat:
            if not _valid_stat_fingerprint(self.stat):
                raise ProvenanceError(f"invalid artifact stat for {self.name}")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "role": self.role,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ModelProvenance:
    """Model/runtime identity attached to a worker and every result."""

    backend: str
    requested_model: str
    source_kind: str
    source_locator: str
    upstream_revision: str | None
    revision_status: str
    artifacts: tuple[ArtifactDigest, ...] = ()
    runtime_packages: dict[str, str] = field(default_factory=dict)
    requested_settings: dict[str, Any] = field(default_factory=dict)
    resolved_settings: dict[str, Any] = field(default_factory=dict)
    fallback_history: tuple[dict[str, Any], ...] = ()
    adapter_git_commit: str | None = None
    adapter_git_dirty: bool = False
    uv_lock_sha256: str | None = None
    issues: tuple[str, ...] = ()
    schema_version: int = PROVENANCE_SCHEMA_VERSION
    content_sha256: str = ""
    execution_sha256: str = ""

    def _content_payload(self) -> dict[str, Any]:
        artifacts = sorted(
            (item.identity_dict() for item in self.artifacts),
            key=lambda item: item["name"],
        )
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "requested_model": self.requested_model,
            "source_kind": self.source_kind,
            "source_locator": self.source_locator,
            "upstream_revision": self.upstream_revision,
            "revision_status": self.revision_status,
            "artifacts": artifacts,
        }

    def _execution_payload(self) -> dict[str, Any]:
        payload = {
            **self._content_payload(),
            "runtime_packages": dict(sorted(self.runtime_packages.items())),
            "requested_settings": self.requested_settings,
            "resolved_settings": self.resolved_settings,
            "fallback_history": list(self.fallback_history),
            "uv_lock_sha256": self.uv_lock_sha256,
        }
        if self.schema_version >= 2:
            payload["adapter_git_commit"] = self.adapter_git_commit
        return payload

    def finalized(self) -> ModelProvenance:
        """Return a copy with canonical content and execution hashes populated."""
        content = _sha256_bytes(_canonical_json(self._content_payload()).encode("ascii"))
        execution = _sha256_bytes(_canonical_json(self._execution_payload()).encode("ascii"))
        return ModelProvenance(
            **{
                **asdict(self),
                "artifacts": tuple(self.artifacts),
                "content_sha256": content,
                "execution_sha256": execution,
            }
        )

    def _canonical_hashes_match(self) -> bool:
        try:
            expected = self.finalized()
        except ProvenanceError:
            return False
        return (
            self.content_sha256 == expected.content_sha256
            and self.execution_sha256 == expected.execution_sha256
        )

    @property
    def complete(self) -> bool:
        try:
            self.validate()
        except ProvenanceError:
            return False
        return (
            not self.issues
            and not self.adapter_git_dirty
            and not self.fallback_history
            # Checked here rather than trusting a recorded issue, because this
            # property is also consulted on provenance read back from JSONL,
            # which may have been written before the check existed.
            and not unresolved_execution_issues(self.resolved_settings)
            and self.revision_status in {"pinned", "unavailable-content-addressed"}
            and bool(self.artifacts)
            and bool(self.runtime_packages)
            and bool(self.adapter_git_commit)
            and bool(self.uv_lock_sha256)
            and bool(self.content_sha256)
            and bool(self.execution_sha256)
            and self._canonical_hashes_match()
        )

    @property
    def execution_identity(self) -> str:
        """Canonical runtime identity used for comparing independent workers."""
        return self.execution_sha256

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version not in SUPPORTED_PROVENANCE_SCHEMA_VERSIONS
        ):
            raise ProvenanceError(f"unsupported provenance schema {self.schema_version}")
        if not isinstance(self.backend, str) or not isinstance(self.requested_model, str):
            raise ProvenanceError("backend and requested model must be strings")
        if not self.backend or not self.requested_model:
            raise ProvenanceError("backend and requested model are required")
        if not isinstance(self.source_kind, str):
            raise ProvenanceError("model source kind must be a string")
        if not isinstance(self.source_locator, str):
            raise ProvenanceError("model source locator must be a string")
        if not isinstance(self.revision_status, str) or self.revision_status not in {
            "pinned",
            "unavailable-content-addressed",
            "unknown",
        }:
            raise ProvenanceError(f"invalid revision status {self.revision_status}")
        if not self.source_locator:
            raise ProvenanceError("model source locator is required")
        if self.upstream_revision is not None and not isinstance(self.upstream_revision, str):
            raise ProvenanceError("upstream revision must be a string or null")
        if self.revision_status == "pinned" and not _is_lower_hex(self.upstream_revision, 40):
            raise ProvenanceError(
                "pinned model provenance requires a 40-character lowercase hexadecimal "
                "upstream revision"
            )
        _validate_optional_hex(self.adapter_git_commit, field_name="adapter Git commit", length=40)
        _validate_optional_hex(self.uv_lock_sha256, field_name="uv.lock SHA-256", length=64)
        for artifact in self.artifacts:
            artifact.validate()
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ProvenanceError("model provenance contains duplicate artifact names")
        _validate_optional_hex(
            self.content_sha256,
            field_name="provenance content SHA-256",
            length=64,
        )
        _validate_optional_hex(
            self.execution_sha256,
            field_name="provenance execution SHA-256",
            length=64,
        )
        if (self.content_sha256 or self.execution_sha256) and not self._canonical_hashes_match():
            raise ProvenanceError("stored provenance hashes do not match the canonical payload")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelProvenance:
        def optional_hash(name: str) -> str:
            value = data.get(name, "")
            return "" if value is None else str(value)

        artifacts = tuple(ArtifactDigest(**item) for item in data.get("artifacts", []))
        value = cls(
            backend=str(data.get("backend", "")),
            requested_model=str(data.get("requested_model", "")),
            source_kind=str(data.get("source_kind", "unknown")),
            source_locator=str(data.get("source_locator", "")),
            upstream_revision=data.get("upstream_revision"),
            revision_status=str(data.get("revision_status", "unknown")),
            artifacts=artifacts,
            runtime_packages=dict(data.get("runtime_packages", {})),
            requested_settings=dict(data.get("requested_settings", {})),
            resolved_settings=dict(data.get("resolved_settings", {})),
            fallback_history=tuple(data.get("fallback_history", [])),
            adapter_git_commit=data.get("adapter_git_commit"),
            adapter_git_dirty=bool(data.get("adapter_git_dirty", False)),
            uv_lock_sha256=data.get("uv_lock_sha256"),
            issues=tuple(data.get("issues", [])),
            schema_version=int(data.get("schema_version", 0)),
            content_sha256=optional_hash("content_sha256"),
            execution_sha256=optional_hash("execution_sha256"),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class ModelBinding:
    """Worker-local paths bound to an immutable model provenance record."""

    provenance: ModelProvenance
    paths: tuple[ArtifactDigest, ...] = ()

    def validate(self) -> None:
        self.provenance.validate()
        for item in self.paths:
            item.validate()
        artifact_names = [item.name for item in self.provenance.artifacts]
        path_names = [item.name for item in self.paths]
        if len(path_names) != len(set(path_names)):
            raise ProvenanceError("model binding contains duplicate path names")
        unexpected = sorted(set(path_names) - set(artifact_names))
        if unexpected:
            raise ProvenanceError(
                "model binding contains unexpected path binding(s): " + ", ".join(unexpected)
            )
        by_name = {item.name: item for item in self.paths}
        for artifact in self.provenance.artifacts:
            bound = by_name.get(artifact.name)
            if bound is None:
                raise ProvenanceError(f"artifact {artifact.name} has no worker path binding")
            if bound.identity_dict() != artifact.identity_dict():
                raise ProvenanceError(
                    f"path binding for {artifact.name} does not match its provenance digest"
                )
            # Small selected metadata (for example a fairseq2 card manifest)
            # may be represented directly by its content digest.  File-backed
            # loader inputs still need an explicit path binding so workers can
            # re-hash the bytes before and after execution.
            if not bound.path and artifact.role != "card-metadata":
                raise ProvenanceError(f"artifact {artifact.name} has no worker path binding")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "provenance": self.provenance.to_dict(),
            "paths": [item.__dict__ for item in self.paths],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelBinding:
        value = cls(
            provenance=ModelProvenance.from_dict(dict(data["provenance"])),
            paths=tuple(ArtifactDigest(**item) for item in data.get("paths", [])),
        )
        value.validate()
        return value


def _cache_key(path: Path, fingerprint: dict[str, int]) -> str:
    return _sha256_bytes(
        _canonical_json({"path": str(path.resolve()), "stat": fingerprint}).encode("ascii")
    )


@contextmanager
def _cache_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        handle.close()


def digest_file(
    path: Path,
    *,
    cache_root: Path | None = None,
    role: str = "unknown",
    name: str | None = None,
) -> ArtifactDigest:
    """Hash a file once, reusing only a complete unchanged stat fingerprint."""
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ProvenanceError(f"artifact is not a regular file: {path}")
    root = cache_root or cache_dir("stt")
    directory = root / "provenance"
    directory.mkdir(parents=True, exist_ok=True)
    before = _stat_fingerprint(path)
    cache_path = directory / f"{_cache_key(path, before)}.json"
    lock_path = directory / ".lock"
    with _cache_lock(lock_path):
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if _valid_digest_cache_record(cached, before):
                    return ArtifactDigest(
                        role=role,
                        name=name or path.name,
                        size_bytes=cached["size_bytes"],
                        sha256=cached["sha256"],
                        path=str(path),
                        stat=before,
                    )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = _stat_fingerprint(path)
        if not _same_stat(before, after):
            raise ProvenanceError(f"artifact changed while hashing: {path}")
        record = {
            "version": DIGEST_CACHE_VERSION,
            "stat": after,
            "size_bytes": after["size"],
            "sha256": digest.hexdigest(),
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{cache_path.name}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, cache_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return ArtifactDigest(
            role=role,
            name=name or path.name,
            size_bytes=after["size"],
            sha256=digest.hexdigest(),
            path=str(path),
            stat=after,
        )


def validate_binding(
    binding: ModelBinding,
    *,
    cache_root: Path | None = None,
    verify_hash: bool = True,
) -> tuple[str, ...]:
    """Validate worker paths against declared stat and content digests.

    Hashes are served by the stat-keyed digest cache, so validating a binding
    does not repeatedly scan an unchanged multi-gigabyte checkpoint.
    """
    issues: list[str] = []
    try:
        binding.validate()
    except ProvenanceError as exc:
        return (str(exc),)
    declared = {item.name: item for item in binding.provenance.artifacts}
    bound_by_name = {item.name: item for item in binding.paths}
    for artifact in binding.provenance.artifacts:
        bound = bound_by_name.get(artifact.name)
        if bound is None:
            issues.append(f"missing path binding for {artifact.name}")
            continue
        if not bound.path:
            if artifact.role != "card-metadata":
                issues.append(f"missing path binding for {artifact.name}")
            continue
        path = Path(bound.path)
        try:
            actual = digest_file(path, cache_root=cache_root)
        except (OSError, ProvenanceError) as exc:
            issues.append(f"artifact unavailable {artifact.name}: {exc}")
            continue
        if artifact.stat and not _same_stat(actual.stat, artifact.stat):
            issues.append(f"artifact stat changed: {artifact.name}")
        if verify_hash and (
            actual.size_bytes != artifact.size_bytes or actual.sha256 != artifact.sha256
        ):
            issues.append(f"artifact digest changed: {artifact.name}")
        if bound.stat and not _same_stat(actual.stat, bound.stat):
            issues.append(f"bound artifact stat changed: {artifact.name}")
    for name in set(declared) - set(bound_by_name):
        issues.append(f"missing path binding for {name}")
    return tuple(issues)


def artifact_manifest_sha256(artifacts: tuple[ArtifactDigest, ...]) -> str:
    payload = sorted(
        (artifact.identity_dict() for artifact in artifacts),
        key=lambda item: item["name"],
    )
    return _sha256_bytes(_canonical_json(payload).encode("ascii"))


def preflight_model_binding(
    backend: str,
    model: str | None,
    options: dict[str, Any] | None = None,
    *,
    environment: dict[str, Any] | None = None,
) -> ModelBinding:
    """Resolve model bytes and runtime identity before an isolated worker starts.

    Construction is intentionally used instead of ``load``: backend constructors
    describe the selected model and cache locations without allocating weights.
    The worker receives the resulting immutable manifest and validates it before
    and after execution.
    """
    from stt.registry import get_backend

    cls = get_backend(backend)
    kwargs = {key: value for key, value in (options or {}).items() if value is not None}
    instance = cls(model, **kwargs) if model else cls(**kwargs)
    provenance = collect_runtime_provenance(instance, environment).finalized()
    paths = tuple(provenance.artifacts)
    binding = ModelBinding(provenance=provenance, paths=paths)
    binding.validate()
    return binding


def unresolved_execution_issues(resolved_settings: Mapping[str, Any]) -> tuple[str, ...]:
    """Return issues for execution settings a run could not actually observe.

    A device is required because it is the one resolved setting that changes
    the numbers without changing anything the manifest would otherwise record.
    Two runs of the same checkpoint on CPU and on Metal produce the same
    execution hash when the device is unknown, so an unresolved device does not
    merely leave a label blank — it makes the execution identity non-unique,
    which is the property every downstream join relies on.

    Only the device is checked. Requiring a dtype here would sweep in runtimes
    whose precision is a property of the checkpoint rather than a runtime
    choice, and reporting those as incomplete would say something false.
    """
    device = resolved_settings.get("device")
    if not isinstance(device, str) or not device.strip():
        return (
            "resolved compute device is unavailable, so this run cannot carry a "
            "unique execution identity",
        )
    return ()


def _digest_with_role(path: Path, role: str, name: str) -> ArtifactDigest:
    return digest_file(path, role=role, name=name)


def collect_runtime_provenance(
    instance: Any,
    environment: dict[str, Any] | None = None,
) -> ModelProvenance:
    """Collect a best-effort manifest for the installed backend instance.

    Backends may override this through ``model_binding``.  The generic path is
    intentionally conservative: absent or mutable upstream identity becomes an
    explicit issue instead of a trusted guess.
    """
    environment = environment or {}
    backend = str(getattr(instance, "name", instance.__class__.__name__))
    model = str(getattr(instance, "model", ""))
    options = dict(getattr(instance, "options", {}))
    # Constructor arguments that are intentionally kept as backend attributes
    # are part of the execution identity too.  Keep them in a stable, JSON
    # serializable map even when the backend's generic options dict is empty.
    for key, attribute in (
        ("device", "device_arg"),
        ("dtype", "dtype_arg"),
        ("chunk_seconds", "chunk_seconds"),
        ("n_threads", "n_threads"),
    ):
        value = getattr(instance, attribute, None)
        if value is not None:
            options.setdefault(key, value)
    artifacts: list[ArtifactDigest] = []
    issues: list[str] = []
    source_kind = "unknown"
    source_locator = model
    revision: str | None = None
    revision_status = "unknown"

    def add(path: Path, role: str, name: str) -> None:
        try:
            artifacts.append(_digest_with_role(path, role, name))
        except (OSError, ProvenanceError) as exc:
            issues.append(f"artifact {name} unavailable: {exc}")

    spec = getattr(instance, "spec", None)
    repo = getattr(spec, "repo", None)
    if repo:
        source_kind = "huggingface"
        source_locator = str(repo)
        snapshot = getattr(instance, "snapshot_path", None)
        allow_patterns = (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "feature_extractor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "added_tokens.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "spiece.model",
            "sentencepiece.bpe.model",
            "tokenizer.model",
            "*.py",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "pytorch_model-*.bin",
        )
        if getattr(spec, "family", None) == "mms" and getattr(spec, "lang", None):
            language = str(spec.lang)
            allow_patterns = (
                *allow_patterns,
                f"adapter.{language}.safetensors",
                f"adapter.{language}.bin",
                f"vocabs/{language}.txt",
            )
        if snapshot is None:
            try:
                from huggingface_hub import HfApi, snapshot_download

                # Resolve the requested mutable alias once. Cache inventory is
                # never used to choose a revision: every machine receives the
                # same commit-addressed snapshot or preflight fails closed.
                model_info = HfApi().model_info(str(repo), revision="main")
                revision = str(model_info.sha)
                available_files = {
                    str(sibling.rfilename) for sibling in (model_info.siblings or ())
                }
                selected_files = {
                    filename
                    for filename in available_files
                    if any(fnmatch(filename, pattern) for pattern in allow_patterns)
                }
                if any(
                    filename == "model.safetensors" or fnmatch(filename, "model-*.safetensors")
                    for filename in selected_files
                ):
                    selected_files = {
                        filename
                        for filename in selected_files
                        if filename != "pytorch_model.bin"
                        and not fnmatch(filename, "pytorch_model-*.bin")
                    }
                if getattr(spec, "family", None) == "mms" and getattr(spec, "lang", None):
                    adapter_safe = f"adapter.{spec.lang}.safetensors"
                    if adapter_safe in selected_files:
                        selected_files.discard(f"adapter.{spec.lang}.bin")
                allow_patterns = tuple(sorted(selected_files))
                snapshot = Path(
                    snapshot_download(
                        repo_id=str(repo),
                        revision=revision,
                        allow_patterns=allow_patterns,
                    )
                )
            except (ImportError, OSError):
                snapshot = None
            except Exception as exc:  # noqa: BLE001 - preserve a fail-closed issue
                issues.append(f"Hugging Face snapshot resolution failed: {exc}")
                snapshot = None
        if snapshot is None:
            issues.append("Hugging Face snapshot path is unavailable")
        else:
            snapshot = Path(snapshot)
            revision = snapshot.name
            revision_status = "pinned" if len(revision) == 40 else "unknown"
            if revision_status != "pinned":
                issues.append("Hugging Face model revision is not a full commit")
            for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
                relative = path.relative_to(snapshot).as_posix()
                if any(fnmatch(relative, pattern) for pattern in allow_patterns):
                    add(path, "model-or-processor", relative)
    elif backend == "omniasr-gguf":
        source_kind = "huggingface-gguf"
        source_locator = str(getattr(spec, "url", model))
        model_path = getattr(instance, "model_path", None)
        if not model_path:
            try:
                from huggingface_hub import HfApi, hf_hub_download

                parsed = urlparse(source_locator)
                parts = parsed.path.strip("/").split("/")
                resolve_index = parts.index("resolve")
                repo = "/".join(parts[:resolve_index])
                filename = "/".join(parts[resolve_index + 2 :])
                revision = str(HfApi().model_info(repo, revision="main").sha)
                model_path = Path(
                    hf_hub_download(repo_id=repo, filename=filename, revision=revision)
                )
                instance.model_path = str(model_path)
                instance.resolved_revision = revision
            except Exception as exc:  # noqa: BLE001 - preflight must fail closed
                issues.append(f"GGUF commit-addressed download failed: {exc}")
        if model_path:
            add(Path(model_path), "weights", Path(model_path).name)
        revision = revision or getattr(instance, "resolved_revision", None)
        marker = "/resolve/"
        if not revision and marker in source_locator:
            revision = source_locator.split(marker, 1)[1].split("/", 1)[0]
        if revision and "/resolve/main/" in source_locator:
            source_locator = source_locator.replace("/resolve/main/", f"/resolve/{revision}/")
        revision_status = "pinned" if revision and revision != "main" else "unknown"
        if revision_status != "pinned":
            issues.append("GGUF source is not pinned to an immutable upstream revision")
        revision_error = getattr(instance, "_revision_error", None)
        if revision_error:
            issues.append(str(revision_error))
    elif backend == "omniasr-torch":
        source_kind = "fairseq2-card"
        source_locator = model
        try:
            from fairseq2.assets import AssetDownloadManager, get_asset_store
            from fairseq2.models import hub as model_hub

            store = get_asset_store()
            card = store.retrieve_card(model)
            tokenizer_name = card.field("tokenizer_ref").as_(str)
            tokenizer_card = store.retrieve_card(tokenizer_name)
            manager = model_hub.get_dependency_resolver().resolve(AssetDownloadManager)
            checkpoint_path = manager.download_model(
                card.field("checkpoint").as_uri(), model, progress=False
            )
            tokenizer_path = manager.download_tokenizer(
                tokenizer_card.field("tokenizer").as_uri(),
                tokenizer_name,
                progress=False,
            )
            add(Path(checkpoint_path), "weights", Path(checkpoint_path).name)
            add(Path(tokenizer_path), "tokenizer", Path(tokenizer_path).name)
            card_metadata = _canonical_json(
                {
                    "name": card.name,
                    "model_family": card.field("model_family").as_(str),
                    "model_arch": card.field("model_arch").as_(str),
                    "checkpoint": str(card.field("checkpoint").as_uri()),
                    "tokenizer_ref": tokenizer_name,
                    "tokenizer_family": tokenizer_card.field("tokenizer_family").as_(str),
                    "tokenizer": str(tokenizer_card.field("tokenizer").as_uri()),
                }
            ).encode("ascii")
            artifacts.append(
                digest_bytes(
                    card_metadata,
                    role="card-metadata",
                    name=f"{model}.card.json",
                )
            )
        except Exception as exc:  # noqa: BLE001 - content coverage must be complete
            issues.append(f"fairseq2 card resolution failed: {exc}")
        revision_status = "unavailable-content-addressed"
    elif backend == "dolphin":
        source_kind = "modelscope"
        source_locator = f"DataoceanAI/dolphin-{model}"
        try:
            from stt.backends.dolphin import cache_dir

            directory = cache_dir(model)
            for name, role in (
                (f"{model}.pt", "weights"),
                ("train.yaml", "config"),
                ("feats_stats.npz", "cmvn"),
                ("units.txt", "tokenizer"),
                ("bpe.model", "tokenizer"),
            ):
                path = directory / name
                if path.is_file():
                    add(path, role, name)
                else:
                    issues.append(f"Dolphin consumed artifact is unavailable: {name}")
        except (ImportError, OSError) as exc:
            issues.append(f"Dolphin artifact directory unavailable: {exc}")
        revision_status = "unavailable-content-addressed"
    else:
        issues.append("backend did not provide an immutable model revision or artifact collector")

    resolved_device = getattr(instance, "resolved_device", None)
    if resolved_device is None:
        resolver = getattr(instance, "_resolve_device", None)
        if callable(resolver):
            try:
                resolved_device = resolver()
                instance.resolved_device = resolved_device
            except Exception as exc:  # noqa: BLE001 - leave an explicit unknown
                issues.append(f"device resolution unavailable: {exc}")
    resolved_dtype = getattr(instance, "resolved_dtype", None)
    if resolved_dtype is None:
        resolver = getattr(instance, "_resolve_dtype", None)
        if callable(resolver) and resolved_device is not None:
            try:
                resolved_dtype = str(resolver(resolved_device)).replace("torch.", "")
                instance.resolved_dtype = resolved_dtype
            except Exception as exc:  # noqa: BLE001 - leave an explicit unknown
                issues.append(f"dtype resolution unavailable: {exc}")
    resolved = {
        "device": resolved_device,
        "dtype": resolved_dtype,
        "n_threads": (
            getattr(instance, "n_threads", None)
            if backend != "omniasr-gguf" or getattr(instance, "n_threads", None) is not None
            else 4
        ),
        "chunk_seconds": getattr(instance, "chunk_seconds", None),
    }
    if spec is not None:
        for field_name in ("family", "lang", "crisp_backend", "size", "unlimited"):
            if hasattr(spec, field_name):
                resolved[field_name] = getattr(spec, field_name)
    fallback_history = tuple(getattr(instance, "_fallback_history", ()))
    if fallback_history:
        issues.append("runtime fallback occurred")
    issues.extend(unresolved_execution_issues(resolved))
    provenance = ModelProvenance(
        backend=backend,
        requested_model=model,
        source_kind=source_kind,
        source_locator=source_locator,
        upstream_revision=revision,
        revision_status=revision_status,
        artifacts=tuple(artifacts),
        runtime_packages=dict(environment.get("packages", {})),
        requested_settings=options,
        resolved_settings=resolved,
        fallback_history=fallback_history,
        adapter_git_commit=environment.get("git_commit"),
        adapter_git_dirty=bool(environment.get("git_dirty")),
        uv_lock_sha256=environment.get("uv_lock_sha256"),
        issues=tuple(dict.fromkeys(issues)),
    ).finalized()
    return provenance
