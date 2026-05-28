import os
import warnings
from importlib.metadata import PackageNotFoundError, version

# ---------------------------------------------------------------------------
# Enforce offline-only mode: prevent any Hugging Face Hub or transformers
# library from making network requests.  All models must be local.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

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
