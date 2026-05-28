"""Security regression tests for OmniVoice hardening.

These tests verify that no HuggingFace Hub downloads, Gradio dependencies,
or remote model identifiers remain in the production codebase.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "omnivoice"

# Patterns that indicate remote model access attempts
FORBIDDEN_IMPORTS = [
    "gradio",
    "huggingface_hub",
]

FORBIDDEN_CALLS = [
    "snapshot_download",
    "hf_hub_download",
    "huggingface-cli",
]

# Remote model identifiers that should never appear as defaults
FORBIDDEN_REMOTE_IDS = [
    "k2-fsa/",
    "openai/whisper",
    "eustlb/",
    "Qwen/",
    "facebook/",
]


def _python_files():
    """Yield all .py files under omnivoice/."""
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        yield path


class TestNoForbiddenImports:
    """Ensure no forbidden packages are imported anywhere."""

    @pytest.mark.parametrize("filepath", list(_python_files()), ids=lambda p: str(p.relative_to(ROOT)))
    def test_no_gradio_or_hf_hub_import(self, filepath: Path):
        source = filepath.read_text()
        for pkg in FORBIDDEN_IMPORTS:
            # Match: import gradio, from gradio ..., import huggingface_hub, etc.
            pattern = rf"^\s*(import|from)\s+{re.escape(pkg)}"
            matches = re.findall(pattern, source, re.MULTILINE)
            assert not matches, (
                f"Forbidden import '{pkg}' found in {filepath.relative_to(ROOT)}"
            )


class TestNoForbiddenCalls:
    """Ensure no functions that download from remote are called."""

    @pytest.mark.parametrize("filepath", list(_python_files()), ids=lambda p: str(p.relative_to(ROOT)))
    def test_no_remote_download_calls(self, filepath: Path):
        source = filepath.read_text()
        for call in FORBIDDEN_CALLS:
            assert call not in source, (
                f"Forbidden call '{call}' found in {filepath.relative_to(ROOT)}"
            )


class TestNoRemoteModelDefaults:
    """Ensure no remote HuggingFace model identifiers are used as defaults."""

    @pytest.mark.parametrize("filepath", list(_python_files()), ids=lambda p: str(p.relative_to(ROOT)))
    def test_no_remote_model_ids_as_defaults(self, filepath: Path):
        source = filepath.read_text()
        for remote_id in FORBIDDEN_REMOTE_IDS:
            # Skip comments and docstrings (only flag actual code usage)
            for i, line in enumerate(source.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Allow in docstrings (triple-quoted) and pure comment lines
                if remote_id in line and not stripped.startswith("#"):
                    # Check it's in a string assignment / default, not just docs
                    if "default" in line.lower() or "=" in line:
                        # Allow if it's inside a docstring block or comment
                        if '"""' not in line and "'''" not in line and "#" not in line.split(remote_id)[0]:
                            assert False, (
                                f"Remote model ID '{remote_id}' found as potential default "
                                f"in {filepath.relative_to(ROOT)}:{i}: {line.strip()}"
                            )


class TestOfflineEnvironment:
    """Verify that environment variables enforce offline mode."""

    def test_hf_hub_offline_set(self):
        import omnivoice  # noqa: F401
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_transformers_offline_set(self):
        import omnivoice  # noqa: F401
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_hf_datasets_offline_set(self):
        import omnivoice  # noqa: F401
        assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


class TestModelPathsCentralized:
    """Verify centralized model_paths module is valid."""

    def test_model_paths_importable(self):
        from omnivoice import model_paths
        assert hasattr(model_paths, "OMNIVOICE_MODEL_DIR")
        assert hasattr(model_paths, "ASR_MODEL_DIR")
        assert hasattr(model_paths, "LLM_BASE_DIR")
        assert hasattr(model_paths, "AUDIO_TOKENIZER_DIR")

    def test_model_paths_are_path_objects(self):
        from omnivoice import model_paths
        assert isinstance(model_paths.OMNIVOICE_MODEL_DIR, Path)
        assert isinstance(model_paths.ASR_MODEL_DIR, Path)
        assert isinstance(model_paths.LLM_BASE_DIR, Path)

    def test_model_paths_no_remote_ids(self):
        """Model paths should never contain forward slashes typical of HF repo IDs."""
        from omnivoice import model_paths
        for attr in dir(model_paths):
            if attr.startswith("_") or attr == "resolve":
                continue
            val = getattr(model_paths, attr)
            if isinstance(val, Path):
                path_str = str(val)
                # Should not look like a HF repo ID (org/model)
                assert not re.match(
                    r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", path_str
                ), f"model_paths.{attr} looks like a remote repo ID: {path_str}"


class TestRequireLocalPath:
    """Verify the path gatekeeper function rejects remote identifiers."""

    def test_rejects_hf_repo_id(self):
        from omnivoice.utils.common import require_local_path
        with pytest.raises(ValueError):
            require_local_path("k2-fsa/OmniVoice", arg_name="model")

    def test_rejects_empty_string(self):
        from omnivoice.utils.common import require_local_path
        with pytest.raises(ValueError):
            require_local_path("", arg_name="model")

    def test_rejects_nonexistent_path(self):
        from omnivoice.utils.common import require_local_path
        with pytest.raises(ValueError):
            require_local_path("/nonexistent/path/xyz123", arg_name="model")

    def test_accepts_existing_path(self, tmp_path):
        from omnivoice.utils.common import require_local_path
        d = tmp_path / "model"
        d.mkdir()
        result = require_local_path(str(d), arg_name="model", expect_dir=True)
        assert result == str(d.resolve())


class TestNoPyprojectGradio:
    """Verify gradio is not in project dependencies."""

    def test_no_gradio_dependency(self):
        pyproject = ROOT / "pyproject.toml"
        content = pyproject.read_text()
        # Check it's not listed as a dependency
        assert "gradio" not in content.lower(), "gradio found in pyproject.toml"

