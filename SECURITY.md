# Security Hardening — OmniVoice-Streaming-Hardened

This document describes all security measures implemented in this hardened fork following a security incident caused by Gradio's silent `frpc` reverse-proxy tunnel deployment and uncontrolled HuggingFace Hub network access.

**TL;DR** — Gradio was silently downloading and executing a reverse-proxy binary (`frpc`) that tunneled our internal service to the public internet. HuggingFace Hub calls leaked metadata and enabled model-swap attacks. Both vectors are now fully eliminated with 6 defense-in-depth layers and 153 automated regression tests.

---

## Summary of Changes

| Area | What was done |
|------|---------------|
| **Gradio removal** | All Gradio imports, dependencies, and UI code completely removed from the codebase |
| **HuggingFace Hub removal** | All `snapshot_download`, `hf_hub_download`, and `huggingface_hub` imports removed |
| **Offline enforcement** | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` set at package init |
| **Local-only model loading** | All `from_pretrained()` calls use `local_files_only=True`; centralized `model_paths.py` |
| **frpc tunnel guard** | Runtime detection and hard crash if Gradio's `frpc` binary is found anywhere on disk |
| **153 regression tests** | CI test suite covering all security invariants |

---

## Threat Model

The original codebase had two critical attack surfaces:

1. **Gradio `frpc` binary** — Gradio silently downloads `frpc_linux_amd64_v0.3` to `~/.cache/huggingface/gradio/frpc/`. This binary establishes a reverse proxy tunnel exposing internal services to the public internet without explicit user consent.
   - **Source**: [`gradio-app/gradio` → `gradio/tunneling.py`](https://github.com/gradio-app/gradio/blob/main/gradio/tunneling.py) — `BINARY_FOLDER = Path(HF_HOME) / "gradio" / "frpc"`, downloaded from `cdn-media.huggingface.co`, executed as a subprocess with `frpc http --server_addr {remote_host}:{remote_port}`.

2. **HuggingFace Hub network calls** — `snapshot_download()` and `from_pretrained()` (without `local_files_only`) make outbound HTTPS requests, potentially leaking model identifiers, API tokens, or allowing model-swap attacks via compromised registries.

---

## Defense-in-Depth Layers

### Layer 1: Environment Variables (Offline Enforcement)

Set at package import time in `omnivoice/__init__.py`:

```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
```

Even if library code accidentally calls Hub functions, the SDK itself will refuse to make network requests.

### Layer 2: Code-Level Removal

All runtime imports of the following are **completely removed** from source:

- `gradio`
- `huggingface_hub` (`snapshot_download`, `hf_hub_download`, `cached_download`)
- Any `share=True` Gradio flags
- Any `from_pretrained()` without `local_files_only=True`

### Layer 3: Centralized Model Paths (`omnivoice/model_paths.py`)

A single source of truth for all model locations:

- All paths resolve from the local filesystem only
- Configurable via `OMNIVOICE_*` environment variables
- `resolve()` helper raises `FileNotFoundError` if a model directory doesn't exist
- No model path is ever constructed from a HuggingFace repo ID

### Layer 4: frpc Tunnel Binary Detection

At package import time, the system scans for Gradio's reverse-proxy binary:

```python
_FRPC_SEARCH_PATHS = [
    Path.home() / ".cache" / "huggingface" / "gradio" / "frpc",
    Path.home() / ".cache" / "huggingface" / "gradio",
    Path("/root/.cache/huggingface/gradio/frpc"),
    Path("/root/.cache/huggingface/gradio"),
]
```

If any `frpc*` file is found, the application **raises `RuntimeError` and refuses to start**. This is a hard crash by design — security violations must never be silently tolerated.

### Layer 5: No Download Code Path in Production

The only script that downloads models (`scripts/download_models.sh`) is:

- A standalone bash script (not imported by Python)
- Intended for **one-time use on a dev machine** (e.g., Mac)
- Models are then transferred to the production VM via `rsync`
- The production environment **never** runs this script


### Layer 6: Automated Regression Tests (153 tests)

The test suite (`tests/test_security_hardening.py`) continuously validates:

- No `gradio` imports anywhere in source
- No `snapshot_download` or `hf_hub_download` calls
- No `huggingface_hub` imports
- No `share=True` flags
- Offline env vars are set on import
- frpc binary detection correctly raises `RuntimeError`
- All `from_pretrained()` calls include `local_files_only=True`

---

## Files Modified / Created

| File | Purpose |
|------|---------|
| `omnivoice/__init__.py` | Offline env vars + frpc guard (hard crash on detection) |
| `omnivoice/model_paths.py` | Centralized local-only model path registry |
| `omnivoice/openai_tts_server.py` | FastAPI server (Gradio replacement), optional uvloop |
| `scripts/download_models.sh` | One-time model download script (Mac dev only) |
| `tests/test_security_hardening.py` | 153 automated security regression tests |
| `pyproject.toml` | Removed `gradio`, `huggingface-hub` from dependencies |

---

## Deployment Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  Dev Machine (Mac)                                              │
│                                                                 │
│  1. git clone this repo                                         │
│  2. ./scripts/download_models.sh        # one-time download     │
│  3. rsync -avP ./models/ user@vm:/opt/omnivoice/models/         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Production VM (Red Hat Linux / GPU)                            │
│                                                                 │
│  export OMNIVOICE_MODELS_ROOT=/opt/omnivoice/models             │
│  omnivoice-openai-tts-server --host 0.0.0.0 --port 6655        │
│                                                                 │
│  ✓ No internet access required                                  │
│  ✓ No HuggingFace Hub calls                                     │
│  ✓ No Gradio                                                    │
│  ✓ frpc guard active                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNIVOICE_MODELS_ROOT` | `./models` | Base directory for all model files |
| `OMNIVOICE_MODEL_DIR` | `$MODELS_ROOT/omnivoice` | Main TTS model |
| `OMNIVOICE_ASR_MODEL_DIR` | `$MODELS_ROOT/asr-model` | Whisper ASR model |
| `OMNIVOICE_LLM_BASE_DIR` | `$MODELS_ROOT/llm-base` | Base LLM (training only) |
| `OMNIVOICE_AUDIO_TOKENIZER_DIR` | `$MODELS_ROOT/audio-tokenizer` | Audio tokenizer |
| `OMNIVOICE_EVAL_MODELS_ROOT` | `$MODELS_ROOT/eval` | Evaluation models root |
| `HF_HUB_OFFLINE` | `1` (forced) | Blocks HF Hub network calls |
| `TRANSFORMERS_OFFLINE` | `1` (forced) | Blocks transformers downloads |
| `HF_DATASETS_OFFLINE` | `1` (forced) | Blocks datasets downloads |

---

## Running Security Tests

```bash
pytest tests/test_security_hardening.py -v
```

---

## What To Do If frpc Is Detected

If the application crashes with `SECURITY VIOLATION: Gradio frpc tunnel binaries detected!`:

1. **Immediately investigate** how the binary appeared (check process history, cron jobs, pip install logs)
2. **Remove it**: `rm -rf ~/.cache/huggingface/gradio/`
3. **Audit**: Check if any package in your environment depends on `gradio`
4. **Report**: This indicates a supply-chain or configuration issue that needs root-cause analysis

---

## Verification Commands

Quick verification that the codebase is clean:

```bash
# Should all return 0 results:
grep -r "gradio" omnivoice/ --include="*.py" | grep -v "__pycache__"
grep -r "snapshot_download" omnivoice/ --include="*.py" | grep -v "__pycache__"
grep -r "hf_hub_download" omnivoice/ --include="*.py" | grep -v "__pycache__"
grep -r "from huggingface_hub" omnivoice/ --include="*.py" | grep -v "__pycache__"
```

