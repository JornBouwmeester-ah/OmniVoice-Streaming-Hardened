import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# ---------------------------------------------------------------------------
# Enforce offline-only mode: prevent any Hugging Face Hub or transformers
# library from making network requests.  All models must be local.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Security: detect and block the Gradio frpc reverse-proxy binary.
# If this file exists it means something attempted (or is attempting) to
# tunnel the service to the public internet — a critical security violation.
# ---------------------------------------------------------------------------
_FRPC_SEARCH_PATHS = [
    Path.home() / ".cache" / "huggingface" / "gradio" / "frpc",
    Path.home() / ".cache" / "huggingface" / "gradio",
    Path("/root/.cache/huggingface/gradio/frpc"),
    Path("/root/.cache/huggingface/gradio"),
]


def _check_frpc_not_present() -> None:
    """Refuse to start if any frpc binary or gradio tunnel directory exists."""
    for search_path in _FRPC_SEARCH_PATHS:
        if not search_path.exists():
            continue
        # Check for the frpc binary itself or any file containing 'frpc'
        if search_path.is_file():
            raise RuntimeError(
                f"\n{'='*72}\n"
                f"SECURITY VIOLATION: Gradio frpc tunnel binary detected!\n"
                f"  Path: {search_path}\n\n"
                f"This binary opens a reverse proxy tunnel to the public internet.\n"
                f"It must NOT exist on a hardened deployment.\n\n"
                f"Action required:\n"
                f"  rm -rf {search_path.parent}\n"
                f"  # Then investigate how it appeared (check process history, cron, etc.)\n"
                f"{'='*72}"
            )
        if search_path.is_dir():
            frpc_files = list(search_path.glob("**/frpc*"))
            if frpc_files:
                file_list = "\n  ".join(str(f) for f in frpc_files[:10])
                raise RuntimeError(
                    f"\n{'='*72}\n"
                    f"SECURITY VIOLATION: Gradio frpc tunnel binaries detected!\n"
                    f"  Directory: {search_path}\n"
                    f"  Files found:\n  {file_list}\n\n"
                    f"These binaries open a reverse proxy tunnel to the public internet.\n"
                    f"They must NOT exist on a hardened deployment.\n\n"
                    f"Action required:\n"
                    f"  rm -rf {search_path}\n"
                    f"  # Then investigate how they appeared\n"
                    f"{'='*72}"
                )


_check_frpc_not_present()

warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message="invalid escape sequence",
    module="pydub.utils",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.distributed.algorithms.ddp_comm_hooks",
)

try:
    __version__ = version("omnivoice")
except PackageNotFoundError:
    __version__ = "0.0.0"

from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceConfig,
    OmniVoiceGenerationConfig,
)
from omnivoice import model_paths  # noqa: F401 – centralized path registry

__all__ = ["OmniVoice", "OmniVoiceConfig", "OmniVoiceGenerationConfig", "model_paths"]
