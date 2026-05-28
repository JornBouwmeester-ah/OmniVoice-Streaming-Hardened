#!/usr/bin/env bash
# ===========================================================================
# download_models.sh — One-time model download script (run on your Mac)
#
# Downloads all required models from Hugging Face to a local directory.
# After downloading, transfer the entire directory to your VM:
#
#   rsync -avP ./models/ user@vm:/opt/omnivoice/models/
#
# Then on the VM, set:
#   export OMNIVOICE_MODELS_ROOT=/opt/omnivoice/models
#
# Requirements:
#   pip install huggingface-hub   (provides the 'hf' CLI)
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh [--dest /path/to/models]
#
# ===========================================================================
set -euo pipefail

# Default destination matches model_paths.py MODELS_ROOT default
DEST="${1:-./models}"

# Strip --dest flag if used
if [[ "${1:-}" == "--dest" ]]; then
    DEST="${2:?Usage: $0 --dest /path/to/models}"
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  OmniVoice Model Downloader (one-time, local Mac only)     ║"
echo "║  Target: ${DEST}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check huggingface-cli is available
if ! command -v hf &> /dev/null; then
    echo "ERROR: 'hf' CLI not found."
    echo "Install with: pip install huggingface-hub"
    exit 1
fi

mkdir -p "${DEST}"

# ===========================================================================
# Core inference models
# ===========================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 [1/9] Downloading OmniVoice TTS model..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
hf download k2-fsa/OmniVoice --local-dir "${DEST}/omnivoice"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 [2/9] Downloading Whisper ASR model (for auto-voice transcription)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
hf download openai/whisper-large-v3-turbo --local-dir "${DEST}/asr-model"

## ===========================================================================
## Training models (optional — skip if you only need inference)
## ===========================================================================
#
#echo ""
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#echo "📦 [3/9] Downloading base LLM for training (Qwen3-0.6B)..."
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#hf download Qwen/Qwen3-0.6B --local-dir "${DEST}/llm-base"
#
#echo ""
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#echo "📦 [4/9] Downloading HiggsAudio v2 tokenizer..."
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#hf download eustlb/higgs-audio-v2-tokenizer --local-dir "${DEST}/audio-tokenizer"
#
## ===========================================================================
## Evaluation models (optional — skip if you don't run evaluations)
## ===========================================================================
#
#echo ""
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#echo "📦 [5/9] Downloading Whisper large-v3 (WER evaluation)..."
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#hf download openai/whisper-large-v3 --local-dir "${DEST}/eval/wer/whisper-large-v3"
#
#echo ""
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#echo "📦 [6/9] Downloading Paraformer-zh (Chinese WER evaluation)..."
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#hf download funasr/paraformer-zh --local-dir "${DEST}/eval/wer/paraformer-zh"
#
#echo ""
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#echo "📦 [7/9] Downloading HuBERT large (English WER evaluation)..."
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#hf download facebook/hubert-large-ls960-ft --local-dir "${DEST}/eval/wer/hubert-large-ls960-ft"
#
#echo ""
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#echo "📦 [8/9] Downloading TTS evaluation models (UTMOS, speaker similarity)..."
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#hf download k2-fsa/TTS_eval_models --local-dir "${DEST}/eval"
#
#echo ""
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#echo "📦 [9/9] Downloading SenseVoiceSmall (Cantonese CER evaluation)..."
#echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#hf download FunAudioLLM/SenseVoiceSmall --local-dir "${DEST}/eval/wer/SenseVoiceSmall"

# ===========================================================================
# Done
# ===========================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ All models downloaded to: ${DEST}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Transfer to your VM:                                      ║"
echo "║    rsync -avP ${DEST}/ user@vm:/opt/omnivoice/models/      ║"
echo "║                                                            ║"
echo "║  Then on the VM, set env var:                              ║"
echo "║    export OMNIVOICE_MODELS_ROOT=/opt/omnivoice/models      ║"
echo "║                                                            ║"
echo "║  Start the server:                                         ║"
echo "║    omnivoice-openai-tts-server --port 6655                 ║"
echo "║                                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Print directory tree summary
echo "Directory layout:"
if command -v tree &> /dev/null; then
    tree -L 2 "${DEST}"
else
    find "${DEST}" -maxdepth 2 -type d | sort
fi
