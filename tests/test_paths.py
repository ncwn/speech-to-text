"""Cache-root resolution.

The default is repo-local so a checkout is self-contained; these pin the
override order and the fairseq2 symlink reporting, which is the one runtime
that cannot be redirected by configuration.
"""

from __future__ import annotations

from pathlib import Path

from stt import paths


def test_env_var_overrides_everything(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "elsewhere"))
    assert paths.cache_root() == (tmp_path / "elsewhere").resolve()


def test_env_var_expands_a_tilde(monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, "~/somewhere")
    assert paths.cache_root() == (Path.home() / "somewhere").resolve()


def test_defaults_to_the_checkout(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    root = paths.checkout_root()
    assert root is not None, "tests run from a source tree"
    assert paths.cache_root() == root / ".cache"


def test_falls_back_to_home_when_installed(monkeypatch):
    """An installed wheel has no pyproject.toml above it, so it uses ~/.cache."""
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "checkout_root", lambda: None)
    assert paths.cache_root() == Path.home() / ".cache"


def test_cache_dir_creates_on_demand(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert not (tmp_path / "dolphin").exists()
    assert paths.cache_dir("dolphin").is_dir()
    assert not paths.cache_dir("crispasr", create=False).exists()


def test_configure_environment_does_not_clobber_an_explicit_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/somewhere/deliberate")
    paths.configure_environment()
    import os

    assert os.environ["HF_HOME"] == "/somewhere/deliberate"


def test_configure_environment_points_hf_at_the_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    paths.configure_environment()
    import os

    assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")


def test_fairseq2_status_reports_a_plain_directory(monkeypatch, tmp_path):
    home_cache = tmp_path / "home" / "fairseq2"
    home_cache.mkdir(parents=True)
    monkeypatch.setattr(paths, "FAIRSEQ2_HOME", home_cache)
    assert paths.fairseq2_status()[0] == "local"


def test_fairseq2_status_recognises_a_link_into_the_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "root"))
    target = tmp_path / "root" / "fairseq2"
    target.mkdir(parents=True)
    link = tmp_path / "home-fairseq2"
    link.symlink_to(target)
    monkeypatch.setattr(paths, "FAIRSEQ2_HOME", link)
    assert paths.fairseq2_status()[0] == "linked"


def test_fairseq2_status_flags_a_link_somewhere_else(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "root"))
    other = tmp_path / "unrelated"
    other.mkdir()
    link = tmp_path / "home-fairseq2"
    link.symlink_to(other)
    monkeypatch.setattr(paths, "FAIRSEQ2_HOME", link)
    assert paths.fairseq2_status()[0] == "elsewhere"


def test_fairseq2_status_when_nothing_is_there(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "FAIRSEQ2_HOME", tmp_path / "nope")
    assert paths.fairseq2_status() == ("absent", None)


def test_directory_size_counts_files_not_symlinks(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 2048)
    (tmp_path / "link.bin").symlink_to(tmp_path / "a.bin")
    assert paths.directory_size_mb(tmp_path) == 2048 / (1024 * 1024)
    assert paths.directory_size_mb(tmp_path / "missing") is None
