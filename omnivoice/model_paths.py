"""Centralized model path definitions for OmniVoice.

All model paths are resolved from local filesystem locations only.

Users should set appropriate environment variables or edit the defaults
below to match their local model directory structure.

Typical layout::

    models/
    ├── omnivoice/              # Main TTS model (OmniVoice weights + tokenizer)
    ├── asr-model/              # Whisper ASR model (local checkout)
    ├── audio-tokenizer/        # HiggsAudio v2 tokenizer (within omnivoice/ or standalone)
    └── eval/
        ├── mos/
        │   └── utmos22_strong_step7459_v1.pt
        ├── wer/
        │   ├── whisper-large-v3/       # Whisper for WER evaluation
        │   ├── paraformer-zh/          # Paraformer for Chinese WER
        │   ├── hubert-large-ls960-ft/  # HuBERT for English WER
        │   └── SenseVoiceSmall/        # SenseVoice for Cantonese CER
        └── speaker_similarity/
            ├── wavlm_large/            # WavLM SSL model (local s3prl hub)
            └── wavlm_large_finetune.pth  # Fine-tuned ECAPA-TDNN
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directory for all models.  Override with OMNIVOICE_MODELS_ROOT env var.
# Defaults to ./models relative to the current working directory.
# ---------------------------------------------------------------------------
MODELS_ROOT = Path(
    os.getenv("OMNIVOICE_MODELS_ROOT", "./models")
).expanduser().resolve()

# ---------------------------------------------------------------------------
# Core inference models
# ---------------------------------------------------------------------------

# Main OmniVoice TTS model directory (contains config.json, model weights,
# tokenizer files, and audio_tokenizer/ subdirectory).
OMNIVOICE_MODEL_DIR = Path(
    os.getenv("OMNIVOICE_MODEL_DIR", str(MODELS_ROOT / "omnivoice"))
)

# Whisper-based ASR model for reference audio transcription (auto-voice mode).
ASR_MODEL_DIR = Path(
    os.getenv("OMNIVOICE_ASR_MODEL_DIR", str(MODELS_ROOT / "asr-model"))
)

# ---------------------------------------------------------------------------
# Training models
# ---------------------------------------------------------------------------

# Base LLM for initializing a new OmniVoice model (used by training/builder).
LLM_BASE_DIR = Path(
    os.getenv("OMNIVOICE_LLM_BASE_DIR", str(MODELS_ROOT / "llm-base"))
)

# Audio tokenizer path used during data preprocessing / token extraction.
AUDIO_TOKENIZER_DIR = Path(
    os.getenv("OMNIVOICE_AUDIO_TOKENIZER_DIR", str(MODELS_ROOT / "audio-tokenizer"))
)

# ---------------------------------------------------------------------------
# Evaluation models (used by omnivoice.eval)
# ---------------------------------------------------------------------------

EVAL_MODELS_ROOT = Path(
    os.getenv("OMNIVOICE_EVAL_MODELS_ROOT", str(MODELS_ROOT / "eval"))
)

# Whisper large-v3 for English WER evaluation
EVAL_WHISPER_DIR = Path(
    os.getenv("OMNIVOICE_EVAL_WHISPER_DIR", str(EVAL_MODELS_ROOT / "wer" / "whisper-large-v3"))
)

# Paraformer for Chinese WER evaluation
EVAL_PARAFORMER_DIR = Path(
    os.getenv("OMNIVOICE_EVAL_PARAFORMER_DIR", str(EVAL_MODELS_ROOT / "wer" / "paraformer-zh"))
)

# HuBERT large for English WER evaluation
EVAL_HUBERT_DIR = Path(
    os.getenv("OMNIVOICE_EVAL_HUBERT_DIR", str(EVAL_MODELS_ROOT / "wer" / "hubert-large-ls960-ft"))
)

# UTMOS model weights (.pt file)
EVAL_UTMOS_WEIGHTS = Path(
    os.getenv("OMNIVOICE_EVAL_UTMOS_WEIGHTS", str(EVAL_MODELS_ROOT / "mos" / "utmos22_strong_step7459_v1.pt"))
)

# SenseVoiceSmall for Cantonese CER evaluation
EVAL_SENSEVOICE_DIR = Path(
    os.getenv("OMNIVOICE_EVAL_SENSEVOICE_DIR", str(EVAL_MODELS_ROOT / "wer" / "SenseVoiceSmall"))
)

# Speaker similarity: WavLM SSL model directory (local s3prl hub format)
EVAL_WAVLM_DIR = Path(
    os.getenv(
        "OMNIVOICE_EVAL_WAVLM_DIR",
        str(EVAL_MODELS_ROOT / "speaker_similarity" / "wavlm_large"),
    )
)

# Speaker similarity: fine-tuned ECAPA-TDNN weights
EVAL_ECAPA_TDNN_WEIGHTS = Path(
    os.getenv(
        "OMNIVOICE_EVAL_ECAPA_TDNN_WEIGHTS",
        str(EVAL_MODELS_ROOT / "speaker_similarity" / "wavlm_large_finetune.pth"),
    )
)


def resolve(path: Path) -> str:
    """Return the resolved string path, raising if it doesn't exist."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Required local model path does not exist: {resolved}\n"
            f"Set the appropriate OMNIVOICE_* environment variable or place "
            f"the model at the expected location."
        )
    return str(resolved)

