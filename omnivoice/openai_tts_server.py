#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from html import escape
import json
import logging
import math
import os
import re
import subprocess
import io
import tempfile
import time
import unicodedata
import urllib.request
import wave
from pathlib import Path
import sys
from typing import Literal, Optional
from uuid import uuid4

import inflect
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydub import AudioSegment

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.model_paths import OMNIVOICE_MODEL_DIR, ASR_MODEL_DIR
from omnivoice.utils.audio import cross_fade_chunks
from omnivoice.utils.common import (
    get_best_device,
    require_local_path,
    resolve_inference_dtype,
)
from omnivoice.utils.text import ABBREVIATIONS, CLOSING_MARKS, add_punctuation


LOG = logging.getLogger("omnivoice.openai_tts_server")
BACKEND_MODEL_ID = os.getenv("OMNIVOICE_MODEL_ID", str(OMNIVOICE_MODEL_DIR))
API_MODEL_ID = os.getenv("OMNIVOICE_API_MODEL_ID", "omnivoice")
DEFAULT_HOST = os.getenv("OMNIVOICE_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("OMNIVOICE_PORT", "6655"))
DEFAULT_VOICE = os.getenv("OMNIVOICE_DEFAULT_VOICE", "alloy")
DEFAULT_AUDIO_CHUNK_DURATION = float(
    os.getenv("OMNIVOICE_AUDIO_CHUNK_DURATION", "15.0")
)
DEFAULT_AUDIO_CHUNK_THRESHOLD = float(
    os.getenv("OMNIVOICE_AUDIO_CHUNK_THRESHOLD", "30.0")
)
DEFAULT_SENTENCE_CHUNKING_MIN_CHARS = int(
    os.getenv("OMNIVOICE_SENTENCE_CHUNKING_MIN_CHARS", "280")
)
PREWARM_LOCAL_VOICE_PROMPTS = (
    os.getenv("OMNIVOICE_PREWARM_LOCAL_VOICE_PROMPTS", "1") != "0"
)
CHUNKED_PIPELINE_ENABLED = os.getenv("OMNIVOICE_CHUNKED_PIPELINE", "1") != "0"
CHUNK_MAX_WORKERS = max(1, min(2, int(os.getenv("OMNIVOICE_CHUNK_MAX_WORKERS", "2"))))
CHUNK_MAX_QUEUE_DEPTH = max(1, int(os.getenv("OMNIVOICE_CHUNK_MAX_QUEUE_DEPTH", "64")))
CHUNK_MIN_PARALLEL_CHUNKS = max(
    2, int(os.getenv("OMNIVOICE_CHUNK_MIN_PARALLEL_CHUNKS", "8"))
)
CHUNK_MIN_PARALLEL_CHARS = max(
    1, int(os.getenv("OMNIVOICE_CHUNK_MIN_PARALLEL_CHARS", "1600"))
)
CHUNK_TARGET_CHARS = max(
    64, int(os.getenv("OMNIVOICE_CHUNK_TARGET_CHARS", str(DEFAULT_SENTENCE_CHUNKING_MIN_CHARS)))
)
CHUNK_MIN_CHARS = max(1, int(os.getenv("OMNIVOICE_CHUNK_MIN_CHARS", "80")))
CHUNK_MAX_CHARS = max(
    CHUNK_TARGET_CHARS,
    int(os.getenv("OMNIVOICE_CHUNK_MAX_CHARS", str(max(640, CHUNK_TARGET_CHARS * 3)))),
)
CHUNK_RETRY_COUNT = max(0, min(3, int(os.getenv("OMNIVOICE_CHUNK_RETRY_COUNT", "1"))))
CHUNK_TIMEOUT_SECONDS = max(1.0, float(os.getenv("OMNIVOICE_CHUNK_TIMEOUT", "300")))
CHUNK_SECONDARY_PORT_OFFSET = int(os.getenv("OMNIVOICE_CHUNK_SECONDARY_PORT_OFFSET", "1"))
CHUNK_SECONDARY_STARTUP_TIMEOUT = max(
    1.0, float(os.getenv("OMNIVOICE_CHUNK_SECONDARY_STARTUP_TIMEOUT", "180"))
)
CHUNK_MIN_FREE_GPU_MB = max(0, int(os.getenv("OMNIVOICE_CHUNK_MIN_FREE_GPU_MB", "6000")))
CUDA_MEMORY_FRACTION = float(os.getenv("OMNIVOICE_CUDA_MEMORY_FRACTION", "0.50"))
WORKER_MODE = os.getenv("OMNIVOICE_WORKER_MODE", "0") == "1"
DEFAULT_ASR_MODEL_ID = os.getenv("OMNIVOICE_ASR_MODEL_ID", str(ASR_MODEL_DIR))
MAX_SANITIZED_INPUT_CHARS = int(os.getenv("OMNIVOICE_MAX_INPUT_CHARS", "12000"))
MAX_RAW_INPUT_CHARS = int(
    os.getenv("OMNIVOICE_MAX_RAW_INPUT_CHARS", str(MAX_SANITIZED_INPUT_CHARS * 2))
)
DEBUG_PREVIEW_CHARS = int(os.getenv("OMNIVOICE_DEBUG_PREVIEW_CHARS", "160"))
LOCAL_VOICE_REFERENCE_ROOT = Path(
    os.getenv("OMNIVOICE_LOCAL_VOICE_ROOT", "/home/op/Libro-Gregoria-Variacion/audio")
)

VALID_TLDS = [
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "mil",
    "int",
    "biz",
    "info",
    "name",
    "pro",
    "coop",
    "museum",
    "travel",
    "jobs",
    "mobi",
    "tel",
    "asia",
    "cat",
    "xxx",
    "aero",
    "arpa",
    "bg",
    "br",
    "ca",
    "cn",
    "de",
    "es",
    "eu",
    "fr",
    "in",
    "it",
    "jp",
    "mx",
    "nl",
    "ru",
    "uk",
    "us",
    "io",
    "co",
]
VALID_UNITS = {
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "in": "inch",
    "ft": "foot",
    "yd": "yard",
    "mi": "mile",
    "g": "gram",
    "kg": "kilogram",
    "mg": "milligram",
    "s": "second",
    "ms": "millisecond",
    "min": "minute",
    "h": "hour",
    "l": "liter",
    "ml": "milliliter",
    "cl": "centiliter",
    "dl": "deciliter",
    "kph": "kilometer per hour",
    "mph": "mile per hour",
    "mi/h": "mile per hour",
    "m/s": "meter per second",
    "km/h": "kilometer per hour",
    "mm/s": "millimeter per second",
    "cm/s": "centimeter per second",
    "ft/s": "feet per second",
    "cm/h": "centimeter per hour",
    "degc": "degree celsius",
    "degf": "degree fahrenheit",
    "°c": "degree celsius",
    "°f": "degree fahrenheit",
    "pa": "pascal",
    "kpa": "kilopascal",
    "mpa": "megapascal",
    "atm": "atmosphere",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
    "v": "volt",
    "kv": "kilovolt",
    "mv": "megavolt",
    "a": "amp",
    "ma": "milliamp",
    "ka": "kiloamp",
    "w": "watt",
    "kw": "kilowatt",
    "mw": "megawatt",
    "j": "joule",
    "kj": "kilojoule",
    "mj": "megajoule",
    "ω": "ohm",
    "kω": "kiloohm",
    "mω": "megaohm",
    "f": "farad",
    "µf": "microfarad",
    "uf": "microfarad",
    "nf": "nanofarad",
    "pf": "picofarad",
    "b": "bit",
    "kb": "kilobit",
    "mb": "megabit",
    "gb": "gigabit",
    "tb": "terabit",
    "pb": "petabit",
    "kbps": "kilobit per second",
    "mbps": "megabit per second",
    "gbps": "gigabit per second",
    "tbps": "terabit per second",
    "px": "pixel",
}
MONEY_UNITS = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}
INFLECT_ENGINE = inflect.engine()

SMART_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
)
CJK_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "、": ",",
        "。": ".",
        "，": ",",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
    }
)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MODEL_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^<>|\r\n]{1,128}\|>")
HTML_TAG_PATTERN = re.compile(r"</?[^>\r\n]{1,256}>")

# LLM reasoning/thinking block patterns — strip tag + content entirely
_THINK_RE = re.compile(
    r"<(think|thinking|reasoning|reflection)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Markdown patterns for stripping
_MD_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_MD_HORIZ_RULE_RE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
_MD_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_MD_BLOCKQUOTE_RE = re.compile(r"^[ \t]*>[ \t]?(.*)$", re.MULTILINE)
_MD_UNORDERED_LIST_RE = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
_MD_ORDERED_LIST_RE = re.compile(r"^[ \t]*\d+\.[ \t]+", re.MULTILINE)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_TABLE_SEP_RE = re.compile(r"^\|?[ \t]*[-:]+[-| :\t]*\|?\s*$", re.MULTILINE)
_MD_TABLE_PIPE_RE = re.compile(r"\|")

URL_PATTERN = re.compile(
    r"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+(\.(?:"
    + "|".join(VALID_TLDS)
    + r"))+|[0-9]{1,3}(?:\.[0-9]{1,3}){3})(:[0-9]+)?([/?][^\s]*)?",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
)
PHONE_GROUP_PATTERN = re.compile(
    r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})"
)
GENERIC_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d(). \-]{7,}\d)")
UNIT_PATTERN = re.compile(
    r"((?<!\w)([+-]?)(\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\s*("
    + "|".join(sorted(VALID_UNITS, key=len, reverse=True))
    + r")(?=[^\w\d]|\b)",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"([0-9]{1,2}\s*:\s*[0-9]{2}(?:\s*:\s*[0-9]{2})?)(\s*(?:pm|am)\b)?",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(
    r"(-?)(["
    + "".join(MONEY_UNITS.keys())
    + r"])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
    re.IGNORECASE,
)
PROTECTED_TAG_PATTERN = re.compile(r"\[[^\[\]\r\n]{1,128}\]")
WHITESPACE_PATTERN = re.compile(r"\s+")
OPTIONAL_PLURALIZATION_PATTERN = re.compile(r"\(s\)")
THOUSANDS_SEPARATOR_PATTERN = re.compile(r"(?<=\d),(?=\d)")
RANGE_DASH_PATTERN = re.compile(r"(?<=\d)-(?=\d)")
TRAILING_URL_PUNCTUATION = ".,!?;:)"
SYMBOL_REPLACEMENTS = {
    "~": " ",
    "@": " at ",
    "#": " number ",
    "$": " dollar ",
    "&": " and ",
    "%": " percent ",
    "^": " ",
    "*": " ",
    "_": " ",
    "|": " ",
    "\\": " ",
    "/": " slash ",
    "=": " equals ",
    "+": " plus ",
    "<": " less than ",
    ">": " greater than ",
}


@dataclass(frozen=True, slots=True)
class VoiceOptionDefinition:
    id: str
    fallback_name: str
    fallback_instruct: Optional[str]
    sample_file: Optional[str] = None
    sample_ref_text: Optional[str] = None
    sample_label: Optional[str] = None
    sample_duration_seconds: Optional[float] = None
    default_language: Optional[str] = None

    @property
    def sample_path(self) -> Optional[Path]:
        if self.sample_file is None:
            return None
        return LOCAL_VOICE_REFERENCE_ROOT / self.sample_file

    def has_local_sample(self) -> bool:
        sample_path = self.sample_path
        return sample_path is not None and sample_path.is_file()

    def display_name(self) -> str:
        if self.has_local_sample() and self.sample_label is not None:
            duration = (
                f" ({int(round(self.sample_duration_seconds))}s local reference)"
                if self.sample_duration_seconds is not None
                else " (local reference)"
            )
            return f"{self.id} -> {self.sample_label}{duration}"
        return self.fallback_name


@dataclass(frozen=True, slots=True)
class ResolvedVoice:
    voice_id: str
    display_name: str
    instruct: Optional[str]
    ref_audio_path: Optional[Path]
    ref_text: Optional[str]
    default_language: Optional[str]


VOICE_OPTIONS: list[VoiceOptionDefinition] = [
    VoiceOptionDefinition(
        id="alloy",
        fallback_name="alloy (male, moderate pitch, american accent)",
        fallback_instruct="male, moderate pitch, american accent",
        sample_file="omnivoice_refs/britishMan_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Man",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="echo",
        fallback_name="echo (male, low pitch, british accent)",
        fallback_instruct="male, low pitch, british accent",
        sample_file="omnivoice_refs/CordobesMan_10s.wav",
        sample_ref_text=(
            "Che, te quiero presentar los ultimos cambios en la Feli que nace "
            "de adentro tuyo. Vos sos el que maneja tu propio destino, no hay otro."
        ),
        sample_label="Cordobes Man",
        sample_duration_seconds=10.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="fable",
        fallback_name="fable (male, high pitch, australian accent)",
        fallback_instruct="male, high pitch, australian accent",
        sample_file="omnivoice_refs/britishMan_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Man",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="onyx",
        fallback_name="onyx (male, very low pitch, american accent)",
        fallback_instruct="male, very low pitch, american accent",
        sample_file="omnivoice_refs/CordobesMan_10s.wav",
        sample_ref_text=(
            "Che, te quiero presentar los ultimos cambios en la Feli que nace "
            "de adentro tuyo. Vos sos el que maneja tu propio destino, no hay otro."
        ),
        sample_label="Cordobes Man",
        sample_duration_seconds=10.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="nova",
        fallback_name="nova (female, moderate pitch, american accent)",
        fallback_instruct="female, moderate pitch, american accent",
        sample_file="omnivoice_refs/britishWoman_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Woman",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="shimmer",
        fallback_name="shimmer (female, high pitch, british accent)",
        fallback_instruct="female, high pitch, british accent",
        sample_file="omnivoice_refs/testargentina_8s.wav",
        sample_ref_text=(
            "Definitivamente el departamento tiene que tener terraza o balcon. "
            "Hace mucho tiempo que tenia ganas de usar esta aplicacion para aprender a cocinar."
        ),
        sample_label="Argentina Voice",
        sample_duration_seconds=8.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="british_man",
        fallback_name="british_man (male, moderate pitch, british accent)",
        fallback_instruct="male, moderate pitch, british accent",
        sample_file="omnivoice_refs/britishMan_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Man",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="british_woman",
        fallback_name="british_woman (female, moderate pitch, british accent)",
        fallback_instruct="female, moderate pitch, british accent",
        sample_file="omnivoice_refs/britishWoman_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Woman",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="cordobes_man",
        fallback_name="cordobes_man (male, moderate pitch, argentinian accent)",
        fallback_instruct="male, moderate pitch, argentinian accent",
        sample_file="omnivoice_refs/CordobesMan_10s.wav",
        sample_ref_text=(
            "Che, te quiero presentar los ultimos cambios en la Feli que nace "
            "de adentro tuyo. Vos sos el que maneja tu propio destino, no hay otro."
        ),
        sample_label="Cordobes Man",
        sample_duration_seconds=10.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="argentina_voice",
        fallback_name="argentina_voice (female, warm pitch, argentinian accent)",
        fallback_instruct="female, warm pitch, argentinian accent",
        sample_file="speaker_5679x1_8886x1_endstrimmed_13s.wav",
        sample_ref_text=(
            "El otro dia vi un video en Facebook sobre un aparatito que te ponías en la "
            "oreja y directamente te traducía, ¿eso existe? Casablanca es la película "
            "favorita de personas nacidas entre mil novecientos ochenta y mil novecientos "
            "ochenta y cinco."
        ),
        sample_label="Speakers 5679+8886 (ends-trimmed, internal silence preserved)",
        sample_duration_seconds=12.9,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="mergy",
        fallback_name="mergy (female, Welsh base + British instruct bias)",
        fallback_instruct="female, young adult, moderate pitch, british accent",
        sample_file="downloads_audio_clean.wav",
        sample_ref_text="Julian drew on the Jewish equation of divinity and law.",
        sample_label="Welsh female base (downloads/audio.wav cleaned)",
        sample_duration_seconds=4.1,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="auto",
        fallback_name="auto (voice design)",
        fallback_instruct=None,
    ),
]
VOICE_LOOKUP = {item.id: item for item in VOICE_OPTIONS}
SUPPORTED_MODEL_OPTIONS = [
    {"id": API_MODEL_ID, "name": "OmniVoice"},
    {"id": "tts-1", "name": "OmniVoice (OpenAI alias)"},
    {"id": "tts-1-hd", "name": "OmniVoice HD (OpenAI alias)"},
    {"id": "gpt-4o-mini-tts", "name": "OmniVoice mini TTS (OpenAI alias)"},
]
SUPPORTED_MODEL_ALIASES = {
    API_MODEL_ID,
    BACKEND_MODEL_ID,
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
}
SUPPORTED_RESPONSE_FORMATS = {
    "mp3": ("audio/mpeg", "mp3"),
    "wav": ("audio/wav", "wav"),
    "flac": ("audio/flac", "flac"),
    "ogg": ("audio/ogg", "ogg"),
    "opus": ("audio/opus", "opus"),
}
SUPPORTED_TRANSCRIPTION_RESPONSE_FORMATS = {
    "json",
    "text",
    "srt",
    "verbose_json",
    "vtt",
}
SUPPORTED_TRANSCRIPTION_MODEL_ALIASES = {
    DEFAULT_ASR_MODEL_ID,
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
}


def _build_frontend_page() -> str:
    voice_cards = []
    local_voice_count = 0

    for option in VOICE_OPTIONS:
        has_local_sample = option.has_local_sample()
        if has_local_sample:
            local_voice_count += 1

        source_label = (
            "Local reference clip" if has_local_sample else "Voice design prompt"
        )
        duration_label = (
            f"{int(round(option.sample_duration_seconds))}s clip"
            if option.sample_duration_seconds is not None
            else "No reference clip"
        )
        language_label = option.default_language or "auto"
        detail_text = (
            option.sample_ref_text
            or option.fallback_instruct
            or "Automatic voice selection"
        )

        voice_cards.append(
            f"""
            <article class="voice-card">
              <div class="voice-card__top">
                <span class="badge">{escape(option.id)}</span>
                <span class="status">{escape(source_label)}</span>
              </div>
              <h3>{escape(option.display_name())}</h3>
              <p class="meta">Language: {escape(language_label)} · {escape(duration_label)}</p>
              <p class="body">{escape(detail_text)}</p>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OmniVoice | OpenAI-compatible TTS</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #07111f;
        --panel: rgba(10, 20, 36, 0.92);
        --panel-strong: rgba(13, 28, 49, 0.98);
        --text: #eaf2ff;
        --muted: #a8b8d8;
        --accent: #6ea8ff;
        --accent-strong: #8d6bff;
        --border: rgba(122, 154, 219, 0.18);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(110, 168, 255, 0.25), transparent 28%),
          radial-gradient(circle at top right, rgba(141, 107, 255, 0.2), transparent 24%),
          linear-gradient(180deg, #05101d 0%, #07111f 52%, #09131f 100%);
        color: var(--text);
      }}

      a {{
        color: inherit;
      }}

      .shell {{
        width: min(1180px, calc(100% - 32px));
        margin: 0 auto;
        padding: 32px 0 48px;
      }}

      .hero,
      .panel {{
        background: linear-gradient(180deg, var(--panel), var(--panel-strong));
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: 0 28px 64px rgba(0, 0, 0, 0.28);
      }}

      .hero {{
        padding: 36px;
        display: grid;
        gap: 18px;
      }}

      .eyebrow {{
        margin: 0;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.78rem;
        color: var(--accent);
      }}

      h1 {{
        margin: 0;
        font-size: clamp(2.3rem, 5vw, 4.2rem);
        line-height: 1.02;
      }}

      .lede {{
        margin: 0;
        max-width: 68ch;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
      }}

      .actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }}

      .button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 11px 16px;
        border-radius: 999px;
        border: 1px solid rgba(146, 175, 235, 0.28);
        background: rgba(255, 255, 255, 0.04);
        text-decoration: none;
        font-weight: 600;
      }}

      .button.primary {{
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        color: #08101d;
        border-color: transparent;
      }}

      .stats {{
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
      }}

      .stat {{
        padding: 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(146, 175, 235, 0.14);
      }}

      .stat span {{
        display: block;
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 6px;
      }}

      .stat strong {{
        font-size: 1.05rem;
      }}

      .panel {{
        margin-top: 22px;
        padding: 24px;
      }}

      .panel h2 {{
        margin: 0 0 8px;
        font-size: 1.4rem;
      }}

      .panel p.subhead {{
        margin: 0 0 18px;
        color: var(--muted);
        line-height: 1.6;
      }}

      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 14px;
      }}

      .voice-card {{
        padding: 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(146, 175, 235, 0.14);
      }}

      .voice-card__top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 12px;
      }}

      .badge,
      .status {{
        display: inline-flex;
        align-items: center;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        line-height: 1;
      }}

      .badge {{
        background: rgba(110, 168, 255, 0.16);
        color: #d6e5ff;
      }}

      .status {{
        background: rgba(141, 107, 255, 0.16);
        color: #e6dcff;
      }}

      .voice-card h3 {{
        margin: 0 0 8px;
        font-size: 1.08rem;
      }}

      .meta {{
        margin: 0 0 10px;
        color: var(--muted);
        font-size: 0.92rem;
      }}

      .body {{
        margin: 0;
        color: #d6e1f6;
        line-height: 1.6;
      }}

      .credits {{
        display: grid;
        gap: 10px;
      }}

      .credits strong {{
        color: #f5f8ff;
      }}

      .footer {{
        margin-top: 18px;
        color: var(--muted);
        font-size: 0.92rem;
      }}

      @media (max-width: 640px) {{
        .shell {{
          width: min(100% - 20px, 1180px);
          padding-top: 10px;
        }}

        .hero,
        .panel {{
          border-radius: 20px;
        }}

        .hero {{
          padding: 24px;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">OpenAI-compatible TTS</p>
        <h1>OmniVoice</h1>
        <p class="lede">
          FastAPI text-to-speech on <strong>0.0.0.0:6655</strong> with request sanitization,
          sentence-aware chunking, and local voice presets that work cleanly with Open WebUI.
        </p>
        <div class="actions">
          <a class="button primary" href="/v1/audio/speech">Speech endpoint</a>
          <a class="button" href="/v1/audio/voices">Voice catalogue</a>
          <a class="button" href="/health">Health</a>
        </div>
        <div class="stats">
          <div class="stat"><span>Local voice presets</span><strong>{len(VOICE_OPTIONS)}</strong></div>
          <div class="stat"><span>Local reference clips</span><strong>{local_voice_count}</strong></div>
          <div class="stat"><span>API models</span><strong>{len(SUPPORTED_MODEL_OPTIONS)}</strong></div>
          <div class="stat"><span>Sentence chunking</span><strong>Enabled</strong></div>
        </div>
      </section>

      <section class="panel">
        <h2>Voice presets</h2>
        <p class="subhead">
          Each preset maps to either a compact local reference clip or a direct voice-design
          prompt, so Open WebUI can pick a stable voice without guessing.
        </p>
        <div class="grid">
          {"".join(voice_cards)}
        </div>
      </section>

      <section class="panel credits">
        <h2>Credits</h2>
        <p>
          Thanks to the original OmniVoice creators and contributors for the model, research,
          and open-source implementation this server is built on.
        </p>
        <p class="footer">
          Browser page: <a href="/ui">/ui</a> · JSON endpoints: <a href="/v1/audio/models">/v1/audio/models</a>,
          <a href="/v1/audio/voices">/v1/audio/voices</a>
        </p>
      </section>
    </main>
  </body>
</html>"""


def _render_frontend_page() -> HTMLResponse:
    return HTMLResponse(_build_frontend_page())


def _configure_logging() -> None:
    if getattr(_configure_logging, "_configured", False):
        return
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=False,
    )
    _configure_logging._configured = True


def _cuda_device_index(device: str) -> int:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0
    if ":" not in device:
        return torch.cuda.current_device()
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return torch.cuda.current_device()


def _configure_cuda_memory_fraction(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    if CUDA_MEMORY_FRACTION <= 0.0 or CUDA_MEMORY_FRACTION > 1.0:
        return
    try:
        device_index = _cuda_device_index(device)
        torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION, device_index)
        LOG.info(
            "Configured CUDA memory fraction %.2f for device %s",
            CUDA_MEMORY_FRACTION,
            device,
        )
    except Exception:
        LOG.exception("Failed to configure CUDA memory fraction for %s", device)


DEFAULT_DEVICE = get_best_device()
DEFAULT_ASR_DEVICE = os.getenv("OMNIVOICE_ASR_DEVICE", "cpu")
DEFAULT_ASR_IDLE_TIMEOUT = float(os.getenv("OMNIVOICE_ASR_IDLE_TIMEOUT", "300"))


class TextSanitizationOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    normalize: bool = True
    unit_normalization: bool = False
    strip_html: bool = True
    strip_model_control_tokens: bool = True
    quote_normalization: bool = True
    url_normalization: bool = True
    email_normalization: bool = True
    optional_pluralization_normalization: bool = True
    phone_normalization: bool = True
    number_normalization: bool = True
    money_normalization: bool = True
    time_normalization: bool = True
    replace_remaining_symbols: bool = True


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: Optional[str] = Field(default=None, max_length=MAX_RAW_INPUT_CHARS)
    text: Optional[str] = Field(default=None, max_length=MAX_RAW_INPUT_CHARS)
    model: Optional[str] = None
    voice: Optional[str] = Field(default=None, max_length=512)
    response_format: Literal["mp3", "wav", "flac", "ogg", "opus"] = "mp3"
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    language: Optional[str] = Field(default=None, max_length=64)
    ref_text: Optional[str] = Field(default=None, max_length=MAX_RAW_INPUT_CHARS)
    instruct: Optional[str] = Field(default=None, max_length=1024)
    duration: Optional[float] = Field(default=None, gt=0.0, le=3600.0)
    num_step: Optional[int] = Field(default=None, ge=1, le=128)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    t_shift: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    denoise: Optional[bool] = None
    postprocess_output: Optional[bool] = None
    layer_penalty_factor: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    position_temperature: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    class_temperature: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    normalization_options: TextSanitizationOptions = Field(
        default_factory=TextSanitizationOptions
    )
    sentence_chunking: bool = True
    sentence_chunking_min_chars: int = Field(
        default=DEFAULT_SENTENCE_CHUNKING_MIN_CHARS,
        ge=64,
        le=MAX_SANITIZED_INPUT_CHARS,
    )
    audio_chunk_duration: Optional[float] = Field(default=None, gt=0.5, le=60.0)
    audio_chunk_threshold: Optional[float] = Field(default=None, ge=0.0, le=600.0)

    @model_validator(mode="after")
    def _validate_text_fields(self) -> SpeechRequest:
        values = [value for value in (self.input, self.text) if value is not None]
        if not values:
            raise ValueError("One of 'input' or 'text' must be provided")
        if len(values) == 2 and values[0].strip() != values[1].strip():
            raise ValueError(
                "When both 'input' and 'text' are provided they must contain the same text"
            )
        return self

    def raw_text(self) -> str:
        return self.input if self.input is not None else self.text or ""


@dataclass(slots=True)
class PreparedSpeechRequest:
    text: str
    instruct: Optional[str]
    ref_text: Optional[str]
    language: Optional[str]
    response_format: str
    voice_id: str
    voice_display_name: str
    voice_ref_audio_path: Optional[Path]
    voice_ref_text: Optional[str]
    chunk_plan: list[str]
    force_sentence_chunking: bool
    chunk_detector: str
    parallel_workers: int
    generation_config: OmniVoiceGenerationConfig


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: int
    text: str
    worker_id: int


@dataclass(slots=True)
class ChunkSynthesisResult:
    chunk_id: int
    worker_id: int
    input_length: int
    waveform: torch.Tensor
    sample_rate: int
    retry_count: int
    started_at: float
    ended_at: float
    latency_s: float


class ChunkSynthesisError(RuntimeError):
    def __init__(self, chunk_id: int, failure_reason: str) -> None:
        self.chunk_id = chunk_id
        self.failure_reason = failure_reason
        super().__init__(
            f"Chunk {chunk_id} failed after {CHUNK_RETRY_COUNT + 1} attempt(s): "
            f"{failure_reason}"
        )


class _VoicePromptLRUCache:
    """Bounded LRU cache for ``VoiceClonePrompt`` objects.

    Each cached entry may hold GPU tensors, so unconstrained growth would
    exhaust device memory when serving many distinct voices.  The default cap
    of 128 entries is large enough for typical deployments while preventing
    unbounded accumulation.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = max(1, maxsize)
        self._store: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str) -> object:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def __setitem__(self, key: str, value: object) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class OmniVoiceService:
    def __init__(self, model_id: str, device: str, idle_timeout: float = 300.0) -> None:
        self.model_id = model_id
        self.device = device
        self.model: OmniVoice | None = None
        self.load_error: str | None = None
        self.load_lock: asyncio.Lock | None = None
        self.generation_lock: asyncio.Semaphore | None = None
        self._idle_timeout = idle_timeout
        self._idle_task: asyncio.Task | None = None
        self._last_used: float = 0.0
        self._voice_prompt_cache: _VoicePromptLRUCache = _VoicePromptLRUCache(
            maxsize=128
        )
        self._voice_prewarm_task: asyncio.Task | None = None

    def set_lock(self, lock: asyncio.Lock) -> None:
        self.load_lock = lock

    def set_generation_lock(self, lock: asyncio.Semaphore) -> None:
        self.generation_lock = lock

    def _touch(self) -> None:
        self._last_used = time.monotonic()
        self._schedule_idle_offload()

    def _schedule_idle_offload(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
        if self._idle_timeout > 0 and self.model is not None:
            self._idle_task = asyncio.get_event_loop().create_task(
                self._idle_offload_loop()
            )

    async def _idle_offload_loop(self) -> None:
        try:
            while True:
                elapsed = time.monotonic() - self._last_used
                remaining = self._idle_timeout - elapsed
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            return

        if self.model is not None:
            LOG.info(
                "Idle timeout (%.0fs) reached, offloading model from %s",
                self._idle_timeout,
                self.device,
            )
            try:
                if self.device.startswith("cuda"):
                    await asyncio.to_thread(self._offload_model_sync)
                else:
                    self.model = None
                LOG.info("Model offloaded successfully")
            except Exception:
                LOG.exception("Failed to offload model")

    def _offload_model_sync(self) -> None:
        if self.model is None:
            return
        import gc

        self.model.cpu()
        del self.model
        self.model = None
        self._voice_prompt_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model_sync(self) -> OmniVoice:
        LOG.info("Loading OmniVoice model %s on %s", self.model_id, self.device)
        _configure_cuda_memory_fraction(self.device)
        if self.device.startswith("cuda"):
            from transformers import modeling_utils

            original_allocator_warmup = modeling_utils.caching_allocator_warmup

            # The default Transformers warmup can briefly allocate several extra
            # GiB on cold load. Skipping it keeps OmniVoice lazy-loading viable
            # on the RTX 3060 when other workloads already share that card.
            modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
            try:
                model = OmniVoice.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                )
            finally:
                modeling_utils.caching_allocator_warmup = original_allocator_warmup
        else:
            model = OmniVoice.from_pretrained(
                self.model_id,
                device_map=self.device,
            )
        model.eval()
        LOG.info(
            "Model loaded on %s with sampling rate %s",
            model.device,
            getattr(model, "sampling_rate", "unknown"),
        )
        return model

    async def get_model(self) -> OmniVoice:
        if self.model is not None:
            self._touch()
            return self.model
        if self.load_lock is None:
            raise RuntimeError("Load lock is not initialized")

        async with self.load_lock:
            if self.model is not None:
                self._touch()
                return self.model

            try:
                self.load_error = None
                self.model = await asyncio.to_thread(self._load_model_sync)
                self._touch()
            except Exception as exc:
                self.load_error = str(exc)
                LOG.exception("Failed to load OmniVoice")
                raise

            return self.model

    def health(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "api_model_id": API_MODEL_ID,
            "device": self.device,
            "model_loaded": self.model is not None,
            "load_error": self.load_error,
            "default_voice": DEFAULT_VOICE,
            "idle_timeout": self._idle_timeout,
            "voice_prompt_cache_size": len(self._voice_prompt_cache),
            "voice_prompt_prewarm_enabled": PREWARM_LOCAL_VOICE_PROMPTS,
            "voice_prompt_prewarm_running": (
                self._voice_prewarm_task is not None and not self._voice_prewarm_task.done()
            ),
        }

    def get_or_create_voice_clone_prompt(
        self,
        model: OmniVoice,
        *,
        cache_key: str,
        ref_audio_path: Path,
        ref_text: Optional[str] = None,
    ):
        cached = self._voice_prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
        )
        self._voice_prompt_cache[cache_key] = prompt
        return prompt

    def _prewarm_local_voice_prompts_sync(self, model: OmniVoice) -> int:
        warmed = 0
        for cache_key, ref_audio_path, ref_text in _iter_unique_local_voice_prompt_specs():
            if self._voice_prompt_cache.get(cache_key) is not None:
                continue
            self.get_or_create_voice_clone_prompt(
                model,
                cache_key=cache_key,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
            )
            warmed += 1
        return warmed

    async def prewarm_local_voice_prompts(self) -> int:
        model = await self.get_model()
        if self.generation_lock is not None:
            async with self.generation_lock:
                return await asyncio.to_thread(
                    self._prewarm_local_voice_prompts_sync, model
                )
        return await asyncio.to_thread(self._prewarm_local_voice_prompts_sync, model)

    def schedule_voice_prompt_prewarm(self) -> None:
        if not PREWARM_LOCAL_VOICE_PROMPTS or WORKER_MODE:
            return
        if self._voice_prewarm_task is not None and not self._voice_prewarm_task.done():
            return

        async def _run() -> None:
            try:
                warmed = await self.prewarm_local_voice_prompts()
                LOG.info("Local voice prompt prewarm completed (%d prompt(s) warmed)", warmed)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("Failed to prewarm local voice prompts")

        self._voice_prewarm_task = asyncio.get_event_loop().create_task(_run())


class ASRService:
    def __init__(self, model_id: str, device: str, idle_timeout: float = 300.0) -> None:
        self.model_id = model_id
        self.device = device
        self.pipeline = None
        self.load_error: str | None = None
        self.load_lock: asyncio.Lock | None = None
        self._idle_timeout = idle_timeout
        self._idle_task: asyncio.Task | None = None
        self._last_used: float = 0.0

    def set_lock(self, lock: asyncio.Lock) -> None:
        self.load_lock = lock

    def _touch(self) -> None:
        self._last_used = time.monotonic()
        self._schedule_idle_offload()

    def _schedule_idle_offload(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
        if self._idle_timeout > 0 and self.pipeline is not None:
            self._idle_task = asyncio.get_event_loop().create_task(
                self._idle_offload_loop()
            )

    async def _idle_offload_loop(self) -> None:
        try:
            while True:
                elapsed = time.monotonic() - self._last_used
                remaining = self._idle_timeout - elapsed
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            return

        if self.pipeline is not None:
            LOG.info(
                "Idle timeout (%.0fs) reached, offloading ASR model from %s",
                self._idle_timeout,
                self.device,
            )
            try:
                await asyncio.to_thread(self._offload_pipeline_sync)
                LOG.info("ASR model offloaded successfully")
            except Exception:
                LOG.exception("Failed to offload ASR model")

    def _offload_pipeline_sync(self) -> None:
        if self.pipeline is None:
            return

        import gc

        del self.pipeline
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_pipeline_sync(self):
        from transformers import pipeline as hf_pipeline

        local_model_id = require_local_path(
            self.model_id,
            arg_name="OMNIVOICE_ASR_MODEL_ID",
            expect_dir=True,
        )
        LOG.info("Loading ASR model %s on %s", local_model_id, self.device)
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=local_model_id,
            dtype=resolve_inference_dtype(self.device),
            device_map=self.device,
            local_files_only=True,
        )
        LOG.info("ASR model loaded on %s", self.device)
        return pipe

    async def get_pipeline(self):
        if self.pipeline is not None:
            self._touch()
            return self.pipeline
        if self.load_lock is None:
            raise RuntimeError("ASR load lock is not initialized")

        async with self.load_lock:
            if self.pipeline is not None:
                self._touch()
                return self.pipeline

            try:
                self.load_error = None
                self.pipeline = await asyncio.to_thread(self._load_pipeline_sync)
                self._touch()
            except Exception as exc:
                self.load_error = str(exc)
                LOG.exception("Failed to load ASR model")
                raise

            return self.pipeline

    async def transcribe_file(
        self,
        file_path: str,
        *,
        task: Literal["transcribe", "translate"],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        return_timestamps: bool | str = False,
    ) -> dict[str, object]:
        pipe = await self.get_pipeline()
        return await asyncio.to_thread(
            self._transcribe_file_sync,
            pipe,
            file_path,
            task,
            language,
            prompt,
            temperature,
            return_timestamps,
        )

    def _transcribe_file_sync(
        self,
        pipe,
        file_path: str,
        task: Literal["transcribe", "translate"],
        language: Optional[str],
        prompt: Optional[str],
        temperature: Optional[float],
        return_timestamps: bool | str,
    ) -> dict[str, object]:
        call_kwargs: dict[str, object] = {}
        generate_kwargs: dict[str, object] = {"task": task}

        if language:
            generate_kwargs["language"] = language
        if prompt:
            generate_kwargs["prompt"] = prompt
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = generate_kwargs
        if return_timestamps:
            call_kwargs["return_timestamps"] = return_timestamps

        result = pipe(file_path, **call_kwargs)
        if isinstance(result, dict):
            return result
        return {"text": str(result).strip()}

    def health(self) -> dict[str, object]:
        return {
            "asr_model_id": self.model_id,
            "asr_device": self.device,
            "asr_model_loaded": self.pipeline is not None,
            "asr_load_error": self.load_error,
            "asr_idle_timeout": self._idle_timeout,
        }


def _json_get(url: str, timeout: float = 2.0) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        decoded = json.loads(payload.decode("utf-8"))
        return decoded if isinstance(decoded, dict) else None
    except Exception:
        return None


def _gpu_free_total_mb(device: str) -> tuple[int, int] | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        index = _cuda_device_index(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        return free_bytes // (1024 * 1024), total_bytes // (1024 * 1024)
    except Exception:
        LOG.debug("Unable to read GPU memory info for %s", device, exc_info=True)
        return None


def _has_secondary_worker_gpu_headroom(device: str) -> tuple[bool, str]:
    if not device.startswith("cuda"):
        return False, "device is not CUDA"
    memory = _gpu_free_total_mb(device)
    if memory is None:
        return False, "GPU memory info unavailable"
    free_mb, total_mb = memory
    half_total_mb = total_mb // 2
    required_mb = max(CHUNK_MIN_FREE_GPU_MB, half_total_mb // 2)
    if free_mb < required_mb:
        return (
            False,
            f"insufficient free GPU memory: {free_mb} MiB free, {required_mb} MiB required",
        )
    return True, f"{free_mb} MiB free of {total_mb} MiB"


class SecondaryWorkerManager:
    def __init__(self, host: str = "127.0.0.1", port: int | None = None) -> None:
        self.host = host
        self.port = port if port is not None else DEFAULT_PORT + CHUNK_SECONDARY_PORT_OFFSET
        self.process: subprocess.Popen | None = None
        self.launch_count = 0
        self.last_failure: str | None = None
        self._lock: asyncio.Lock | None = None

    def set_lock(self, lock: asyncio.Lock) -> None:
        self._lock = lock

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def health(self) -> dict[str, object]:
        ready = self.is_healthy()
        return {
            "chunked_pipeline_enabled": CHUNKED_PIPELINE_ENABLED,
            "chunk_worker_mode": WORKER_MODE,
            "chunk_max_workers": CHUNK_MAX_WORKERS,
            "chunk_max_queue_depth": CHUNK_MAX_QUEUE_DEPTH,
            "chunk_min_parallel_chunks": CHUNK_MIN_PARALLEL_CHUNKS,
            "chunk_min_parallel_chars": CHUNK_MIN_PARALLEL_CHARS,
            "chunk_max_chars": CHUNK_MAX_CHARS,
            "chunk_retry_count": CHUNK_RETRY_COUNT,
            "secondary_worker_url": self.base_url,
            "secondary_worker_ready": ready,
            "secondary_worker_pid": self.process.pid if self.process is not None else None,
            "secondary_worker_launch_count": self.launch_count,
            "secondary_worker_last_failure": self.last_failure,
        }

    def is_healthy(self) -> bool:
        if self.process is not None and self.process.poll() is not None:
            return False
        payload = _json_get(f"{self.base_url}/health", timeout=1.0)
        return bool(payload and payload.get("status") == "ok")

    async def ensure_started(self) -> bool:
        if WORKER_MODE or not CHUNKED_PIPELINE_ENABLED or CHUNK_MAX_WORKERS < 2:
            return False

        async with self._get_lock():
            if self.is_healthy():
                return True

            if self.process is not None and self.process.poll() is not None:
                LOG.warning(
                    "Secondary chunk worker exited with code %s", self.process.returncode
                )
                self.process = None
            elif self.process is not None:
                LOG.warning(
                    "Secondary chunk worker is unhealthy but still running; restarting pid=%s",
                    self.process.pid,
                )
                self.process.terminate()
                try:
                    await asyncio.to_thread(self.process.wait, 5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait)
                self.process = None

            has_headroom, reason = _has_secondary_worker_gpu_headroom(service.device)
            if not has_headroom:
                self.last_failure = reason
                LOG.warning("Chunked pipeline falling back to single worker: %s", reason)
                return False

            try:
                self.process = await asyncio.to_thread(self._start_process_sync)
                await self._wait_until_healthy()
                self.launch_count += 1
                self.last_failure = None
                LOG.info(
                    "Secondary chunk worker ready at %s (pid=%s)",
                    self.base_url,
                    self.process.pid if self.process is not None else "unknown",
                )
                return True
            except Exception as exc:
                self.last_failure = str(exc)
                LOG.exception("Failed to start secondary chunk worker")
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                self.process = None
                return False

    def _start_process_sync(self) -> subprocess.Popen:
        log_dir = Path(os.getenv("OMNIVOICE_WORKER_LOG_DIR", "/home/op/.cache/omnivoice-pool/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "chunk-worker-2.log"
        env = os.environ.copy()
        env.update(
            {
                "OMNIVOICE_WORKER_MODE": "1",
                "OMNIVOICE_CHUNKED_PIPELINE": "0",
                "OMNIVOICE_PORT": str(self.port),
                "OMNIVOICE_CUDA_MEMORY_FRACTION": "0.50",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50",
            }
        )
        cmd = [
            sys.executable,
            "-m",
            "omnivoice.openai_tts_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--device",
            service.device,
            "--model-id",
            service.model_id,
            "--idle-timeout",
            str(service._idle_timeout),
            "--worker-mode",
        ]
        LOG.info("Launching secondary OmniVoice chunk worker: %s", " ".join(cmd))
        log_file = log_path.open("ab")
        return subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + CHUNK_SECONDARY_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"secondary worker exited early with code {self.process.returncode}"
                )
            if self.is_healthy():
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(
            f"secondary worker did not become healthy within {CHUNK_SECONDARY_STARTUP_TIMEOUT:.1f}s"
        )

    async def shutdown(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        LOG.info("Stopping secondary chunk worker pid=%s", self.process.pid)
        self.process.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=10)
        except asyncio.TimeoutError:
            self.process.kill()
            await asyncio.to_thread(self.process.wait)
        self.process = None


def _protect_bracket_tags(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        key = f"OMNIVOICETAG{uuid4().hex.upper()}X{len(protected)}TAG"
        protected[key] = match.group(0)
        return key

    return PROTECTED_TAG_PATTERN.sub(_replace, text), protected


def _restore_bracket_tags(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def _normalize_email(match: re.Match[str]) -> str:
    user, domain = match.group(0).split("@", 1)
    user = (
        user.replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
    )
    domain = domain.replace(".", " dot ").replace("-", " dash ")
    return f"{user} at {domain}"


def _normalize_url(match: re.Match[str]) -> str:
    original = match.group(0)
    url = original.rstrip(TRAILING_URL_PUNCTUATION)
    trailing = original[len(url) :]
    spoken = url
    spoken = re.sub(
        r"^https?://",
        lambda item: "https " if item.group(0).lower().startswith("https") else "http ",
        spoken,
        flags=re.IGNORECASE,
    )
    spoken = re.sub(r"^www\.", "www ", spoken, flags=re.IGNORECASE)
    spoken = spoken.replace(".", " dot ")
    spoken = spoken.replace("/", " slash ")
    spoken = spoken.replace("?", " question mark ")
    spoken = spoken.replace("=", " equals ")
    spoken = spoken.replace("&", " and ")
    spoken = spoken.replace("-", " dash ")
    spoken = spoken.replace("_", " underscore ")
    spoken = spoken.replace(":", " colon ")
    spoken = WHITESPACE_PATTERN.sub(" ", spoken).strip()
    return f"{spoken}{trailing}"


def _normalize_phone(match: re.Match[str]) -> str:
    value = match.group(0)
    digits = [char for char in value if char.isdigit()]
    if not digits:
        return value
    prefix = "plus " if value.strip().startswith("+") else ""
    return prefix + " ".join(digits)


def _should_apply_english_expansions(language: Optional[str]) -> bool:
    if language is None:
        return True
    language_key = language.lower().replace("_", "-").split("-", 1)[0]
    return language_key in {"en", "eng"}


def _conditional_int(number: float, threshold: float = 0.00001):
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def _translate_multiplier(multiplier: str) -> str:
    mapping = {
        "k": "thousand",
        "m": "million",
        "b": "billion",
        "t": "trillion",
    }
    key = multiplier.strip().lower()
    return mapping.get(key, multiplier.strip())


def _split_four_digit_year(number: float) -> str:
    number_str = str(_conditional_int(number))
    return (
        f"{INFLECT_ENGINE.number_to_words(number_str[:2])} "
        f"{INFLECT_ENGINE.number_to_words(number_str[2:])}"
    )


def _normalize_unit(match: re.Match[str]) -> str:
    unit_string = match.group(6).strip()
    unit_name = VALID_UNITS.get(unit_string.lower())
    if unit_name is None:
        return match.group(0)

    parts = unit_name.split(" ")
    number = match.group(1).strip().replace(",", "")
    if parts[0].endswith("bit") and unit_string[-1:] == "B":
        parts[0] = parts[0][:-3] + "byte"
    parts[0] = INFLECT_ENGINE.no(parts[0], number)
    return " ".join(parts)


def _normalize_grouped_phone(match: re.Match[str]) -> str:
    country_code, _, area_code, telephone_prefix, line_number = match.groups()
    parts: list[str] = []
    if country_code:
        parts.append(
            INFLECT_ENGINE.number_to_words(
                country_code.replace("+", ""),
                group=1,
                comma="",
            )
        )
    parts.append(
        INFLECT_ENGINE.number_to_words(
            area_code.replace("(", "").replace(")", ""),
            group=1,
            comma="",
        )
    )
    parts.append(INFLECT_ENGINE.number_to_words(telephone_prefix, group=1, comma=""))
    parts.append(INFLECT_ENGINE.number_to_words(line_number, group=1, comma=""))
    return ", ".join(filter(None, parts))


def _normalize_time(match: re.Match[str]) -> str:
    time_value = match.group(1)
    meridiem = (match.group(2) or "").strip().lower()
    time_parts = [part.strip() for part in time_value.split(":")]

    result = [INFLECT_ENGINE.number_to_words(int(time_parts[0]))]
    minutes = int(time_parts[1])
    if minutes == 0:
        result.append("o'clock")
    elif minutes < 10:
        result.append(f"oh {INFLECT_ENGINE.number_to_words(minutes)}")
    else:
        result.append(INFLECT_ENGINE.number_to_words(minutes))

    if len(time_parts) > 2:
        seconds = int(time_parts[2])
        result.append(
            f"and {INFLECT_ENGINE.number_to_words(seconds)} "
            f"{INFLECT_ENGINE.plural('second', seconds)}"
        )

    if meridiem:
        result.append(meridiem)
    return " ".join(result)


def _normalize_money(match: re.Match[str]) -> str:
    bill, coin = MONEY_UNITS[match.group(2)]
    amount_text = match.group(3)
    try:
        amount = float(amount_text)
    except ValueError:
        return match.group(0)

    if match.group(1) == "-":
        amount *= -1

    multiplier = _translate_multiplier(match.group(4))
    multiplier_suffix = f" {multiplier}" if multiplier else ""
    abs_amount = abs(amount)

    if abs_amount % 1 == 0 or multiplier:
        spoken_amount = INFLECT_ENGINE.number_to_words(_conditional_int(abs_amount))
        spoken_bill = INFLECT_ENGINE.plural(
            bill,
            count=max(int(abs_amount), 1),
        )
        prefix = "minus " if amount < 0 else ""
        return f"{prefix}{spoken_amount}{multiplier_suffix} {spoken_bill}"

    whole_amount = int(math.floor(abs_amount))
    cents = int(amount_text.split(".")[-1].ljust(2, "0"))
    spoken_bill = INFLECT_ENGINE.plural(bill, count=max(whole_amount, 1))
    prefix = "minus " if amount < 0 else ""
    return (
        f"{prefix}{INFLECT_ENGINE.number_to_words(whole_amount)} {spoken_bill} and "
        f"{INFLECT_ENGINE.number_to_words(cents)} "
        f"{INFLECT_ENGINE.plural(coin, count=cents)}"
    )


def _normalize_number(match: re.Match[str]) -> str:
    try:
        number = float(match.group(2))
    except ValueError:
        return match.group(0)

    if match.group(1) == "-":
        number *= -1

    multiplier = _translate_multiplier(match.group(3))
    if not multiplier:
        abs_number = abs(number)
        if (
            abs_number % 1 == 0
            and len(str(int(abs_number))) == 4
            and abs_number > 1500
            and int(abs_number) % 1000 > 9
        ):
            prefix = "minus " if number < 0 else ""
            return f"{prefix}{_split_four_digit_year(abs_number)}"

    spoken_number = INFLECT_ENGINE.number_to_words(_conditional_int(abs(number)))
    prefix = "minus " if number < 0 else ""
    multiplier_suffix = f" {multiplier}" if multiplier else ""
    return f"{prefix}{spoken_number}{multiplier_suffix}"


def _normalize_titles_and_abbreviations(text: str) -> str:
    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
    text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
    return text


def _normalize_english_like_text(text: str, options: TextSanitizationOptions) -> str:
    normalized = text
    if options.unit_normalization:
        normalized = UNIT_PATTERN.sub(_normalize_unit, normalized)
    if options.optional_pluralization_normalization:
        normalized = OPTIONAL_PLURALIZATION_PATTERN.sub("s", normalized)
        normalized = re.sub(r"(?<=\d)s\b", " s", normalized, flags=re.IGNORECASE)
    if options.phone_normalization:
        normalized = PHONE_GROUP_PATTERN.sub(_normalize_grouped_phone, normalized)
    if options.time_normalization:
        normalized = TIME_PATTERN.sub(_normalize_time, normalized)

    normalized = _normalize_titles_and_abbreviations(normalized)
    normalized = THOUSANDS_SEPARATOR_PATTERN.sub("", normalized)

    if options.money_normalization:
        normalized = MONEY_PATTERN.sub(_normalize_money, normalized)
    if options.number_normalization:
        normalized = NUMBER_PATTERN.sub(_normalize_number, normalized)

    return normalized


def _strip_llm_artifacts(text: str) -> str:
    """Remove LLM reasoning/thinking blocks (tag + content) from text."""
    return _THINK_RE.sub(" ", text)


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting, keeping readable text content."""
    # Remove fenced code blocks entirely
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    # Remove inline code
    text = _MD_INLINE_CODE_RE.sub(" ", text)
    # Convert headings to plain text
    text = _MD_HEADING_RE.sub(r"\1", text)
    # Remove horizontal rules
    text = _MD_HORIZ_RULE_RE.sub(" ", text)
    # Unwrap bold/italic/strikethrough, keeping the inner text
    text = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _MD_STRIKE_RE.sub(r"\1", text)
    # Unwrap blockquotes
    text = _MD_BLOCKQUOTE_RE.sub(r"\1", text)
    # Remove list markers
    text = _MD_UNORDERED_LIST_RE.sub("", text)
    text = _MD_ORDERED_LIST_RE.sub("", text)
    # Unwrap images (keep alt text), then links (keep link text)
    text = _MD_IMAGE_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    # Remove table separator rows, then pipe characters
    text = _MD_TABLE_SEP_RE.sub(" ", text)
    text = _MD_TABLE_PIPE_RE.sub(" ", text)
    return text


def _basic_cleanup(
    text: str,
    *,
    strip_html: bool,
    strip_model_control_tokens: bool,
    normalize_quotes: bool,
) -> str:
    if normalize_quotes:
        text = text.translate(SMART_PUNCTUATION_TRANSLATION)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(CJK_PUNCTUATION_TRANSLATION)
    if strip_model_control_tokens:
        text = MODEL_CONTROL_TOKEN_PATTERN.sub(" ", text)
    if strip_html:
        text = HTML_TAG_PATTERN.sub(" ", text)
    text = CONTROL_CHAR_PATTERN.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def sanitize_speech_text(
    text: str,
    *,
    language: Optional[str],
    options: TextSanitizationOptions,
) -> str:
    if not text:
        return ""
    # Strip LLM reasoning blocks and markdown before any other processing
    text = _strip_llm_artifacts(text)
    text = _strip_markdown(text)
    if not options.normalize:
        return add_punctuation(
            _basic_cleanup(
                text,
                strip_html=options.strip_html,
                strip_model_control_tokens=options.strip_model_control_tokens,
                normalize_quotes=options.quote_normalization,
            )
        )

    protected_text, protected = _protect_bracket_tags(text)
    sanitized = _basic_cleanup(
        protected_text,
        strip_html=options.strip_html,
        strip_model_control_tokens=options.strip_model_control_tokens,
        normalize_quotes=options.quote_normalization,
    )

    if options.email_normalization:
        sanitized = EMAIL_PATTERN.sub(_normalize_email, sanitized)
    if options.url_normalization:
        sanitized = URL_PATTERN.sub(_normalize_url, sanitized)
    if _should_apply_english_expansions(language):
        sanitized = _normalize_english_like_text(sanitized, options)
    elif options.optional_pluralization_normalization:
        sanitized = OPTIONAL_PLURALIZATION_PATTERN.sub("s", sanitized)
    if options.phone_normalization:
        sanitized = GENERIC_PHONE_PATTERN.sub(_normalize_phone, sanitized)
    if options.replace_remaining_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            sanitized = sanitized.replace(symbol, replacement)

    sanitized = RANGE_DASH_PATTERN.sub(" to ", sanitized)
    sanitized = re.sub(
        r"(?:[A-Za-z]\.){2,} [a-z]",
        lambda match: match.group(0).replace(".", "-"),
        sanitized,
    )
    sanitized = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", sanitized)
    sanitized = WHITESPACE_PATTERN.sub(" ", sanitized).strip()
    sanitized = _restore_bracket_tags(sanitized, protected)
    if language is not None:
        LOG.debug("Sanitized text for language=%s -> %s", language, sanitized[:120])
    return add_punctuation(sanitized)


def sanitize_prompt_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = _basic_cleanup(
        text,
        strip_html=True,
        strip_model_control_tokens=True,
        normalize_quotes=True,
    )
    return cleaned or None


def _split_sentences_pysbd(text: str) -> list[str] | None:
    try:
        import pysbd  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        segmenter = pysbd.Segmenter(language="en", clean=False)
        sentences = [item.strip() for item in segmenter.segment(text) if item.strip()]
        return sentences or None
    except Exception:
        LOG.debug("pysbd sentence detection failed", exc_info=True)
        return None


def _split_sentences_nltk(text: str) -> list[str] | None:
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        sentences = [item.strip() for item in sent_tokenize(text) if item.strip()]
        return sentences or None
    except LookupError:
        return None
    except Exception:
        LOG.debug("nltk sentence detection failed", exc_info=True)
        return None


def _looks_like_decimal_period(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _looks_like_abbreviation_period(text: str, index: int) -> bool:
    prefix = text[: index + 1].rstrip()
    if not prefix:
        return False
    last_word = prefix.split()[-1]
    if last_word in ABBREVIATIONS:
        return True
    # Initialisms such as "U.S." or "D.C." should stay in the current sentence.
    return bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", last_word))


def _split_sentences_regex(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char not in ".!?。！？…":
            index += 1
            continue

        if char == "." and (
            _looks_like_decimal_period(text, index)
            or _looks_like_abbreviation_period(text, index)
        ):
            index += 1
            continue

        # Treat a run of dots or unicode ellipsis as one boundary.
        end = index + 1
        while end < len(text) and text[end] in ".…":
            end += 1
        while end < len(text) and text[end] in CLOSING_MARKS:
            end += 1

        piece = text[start:end].strip()
        if piece:
            sentences.append(piece)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
        index = start

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or ([text.strip()] if text.strip() else [])


def _detect_sentences(text: str) -> tuple[list[str], str]:
    if not text.strip():
        return [], "empty"
    for source, splitter in (
        ("pysbd", _split_sentences_pysbd),
        ("nltk", _split_sentences_nltk),
    ):
        sentences = splitter(text)
        if sentences:
            return sentences, source
    return _split_sentences_regex(text), "regex"


def _split_long_chunk_by_budget(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in text.split(" "):
        word_len = len(word)
        next_len = current_len + (1 if current else 0) + word_len
        if current and next_len > max_chars:
            parts.append(" ".join(current).strip())
            current = [word]
            current_len = word_len
        elif word_len > max_chars:
            if current:
                parts.append(" ".join(current).strip())
                current = []
                current_len = 0
            for start in range(0, word_len, max_chars):
                parts.append(word[start : start + max_chars])
        else:
            current.append(word)
            current_len = next_len
    if current:
        parts.append(" ".join(current).strip())
    return [part for part in parts if part]


def _merge_short_sentence_chunks(
    sentences: list[str],
    *,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        for part in _split_long_chunk_by_budget(sentence, max_chars):
            if not current:
                current = part
                continue

            candidate = f"{current} {part}".strip()
            if len(current) < min_chars or len(candidate) <= target_chars:
                current = candidate
            else:
                chunks.append(current)
                current = part

    if current:
        chunks.append(current)

    # If the final chunk is tiny, fold it back into the previous chunk when
    # doing so stays within the hard budget. This avoids one-word tail chunks.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        candidate = f"{chunks[-2]} {chunks[-1]}".strip()
        if len(candidate) <= max_chars:
            chunks[-2] = candidate
            chunks.pop()

    return chunks


def _rebalance_interleaved_chunks(chunks: list[str], max_chars: int) -> list[str]:
    """Reduce odd/even worker skew without changing text order."""
    if len(chunks) < 4:
        return chunks

    def loads(values: list[str]) -> tuple[int, int]:
        return (
            sum(len(value) for index, value in enumerate(values) if index % 2 == 0),
            sum(len(value) for index, value in enumerate(values) if index % 2 == 1),
        )

    balanced = list(chunks)
    for _ in range(4):
        worker_1_load, worker_2_load = loads(balanced)
        skew = abs(worker_1_load - worker_2_load)
        if skew <= max_chars:
            break
        heavy_parity = 0 if worker_1_load > worker_2_load else 1
        candidate_indices = [
            index
            for index, value in enumerate(balanced)
            if index % 2 == heavy_parity and len(value) > max_chars // 2
        ]
        if not candidate_indices:
            break
        split_index = max(candidate_indices, key=lambda index: len(balanced[index]))
        parts = _split_long_chunk_by_budget(balanced[split_index], max(1, len(balanced[split_index]) // 2))
        if len(parts) < 2:
            break
        balanced = balanced[:split_index] + parts + balanced[split_index + 1 :]

    return balanced


def _plan_sentence_chunks_with_source(text: str, min_chars: int) -> tuple[list[str], str]:
    if len(text) <= min_chars:
        return [text], "none"

    sentences, detector = _detect_sentences(text)
    if not sentences:
        return [text], detector

    target_chars = max(min_chars, CHUNK_TARGET_CHARS)
    planned = _merge_short_sentence_chunks(
        sentences,
        min_chars=max(CHUNK_MIN_CHARS, min_chars // 3),
        target_chars=target_chars,
        max_chars=CHUNK_MAX_CHARS,
    )
    planned = _rebalance_interleaved_chunks(planned, CHUNK_MAX_CHARS)
    if len(planned) > CHUNK_MAX_QUEUE_DEPTH:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Input produced {len(planned)} chunks, exceeding queue cap "
                f"{CHUNK_MAX_QUEUE_DEPTH}."
            ),
        )
    return planned or [text], detector


def _plan_sentence_chunks(text: str, min_chars: int) -> list[str]:
    chunks, _ = _plan_sentence_chunks_with_source(text, min_chars)
    return chunks


def _build_text_chunks(chunks: list[str]) -> list[TextChunk]:
    non_empty_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return [
        TextChunk(
            chunk_id=index + 1,
            text=chunk,
            worker_id=1 if index % 2 == 0 else 2,
        )
        for index, chunk in enumerate(non_empty_chunks)
    ]


def _supported_models() -> list[dict[str, str]]:
    return [dict(item) for item in SUPPORTED_MODEL_OPTIONS]


def _supported_voices() -> list[dict[str, str]]:
    return [{"id": item.id, "name": item.display_name()} for item in VOICE_OPTIONS]


def _iter_unique_local_voice_prompt_specs() -> list[tuple[str, Path, Optional[str]]]:
    seen: set[tuple[str, Optional[str]]] = set()
    specs: list[tuple[str, Path, Optional[str]]] = []
    for option in VOICE_OPTIONS:
        sample_path = option.sample_path
        if sample_path is None or not sample_path.is_file():
            continue
        cache_key = f"{sample_path.resolve()}::{option.sample_ref_text!r}"
        dedupe_key = (str(sample_path.resolve()), option.sample_ref_text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        specs.append((cache_key, sample_path, option.sample_ref_text))
    return specs


def _truncate_preview(text: Optional[str], limit: int = DEBUG_PREVIEW_CHARS) -> str:
    if not text:
        return ""
    preview = WHITESPACE_PATTERN.sub(" ", text).strip()
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit].rstrip()}..."


def _normalization_summary(options: TextSanitizationOptions) -> str:
    enabled = [key for key, value in options.model_dump().items() if value]
    return ",".join(enabled) if enabled else "none"


def _guess_request_source(request: Request) -> str:
    user_agent = (request.headers.get("user-agent") or "").strip()
    user_agent_lower = user_agent.lower()
    if "python-requests" in user_agent_lower:
        return "python-requests/open-webui-like"
    if "mozilla" in user_agent_lower:
        return "browser"
    if user_agent:
        return user_agent[:80]
    return "unknown"


def _resolve_transcription_model(requested: Optional[str]) -> str:
    if not requested:
        return asr_service.model_id
    if (
        requested in SUPPORTED_TRANSCRIPTION_MODEL_ALIASES
        or requested == asr_service.model_id
    ):
        return asr_service.model_id
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported transcription model '{requested}'. Supported aliases: "
            f"{', '.join(sorted(SUPPORTED_TRANSCRIPTION_MODEL_ALIASES))}"
        ),
    )


def _coalesce_timestamp_granularities(
    timestamp_granularities: Optional[list[str]],
    bracketed_timestamp_granularities: Optional[list[str]],
) -> list[str]:
    merged: list[str] = []
    for value in (timestamp_granularities or []) + (
        bracketed_timestamp_granularities or []
    ):
        normalized = value.strip().lower()
        if not normalized:
            continue
        if normalized not in {"segment", "word"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "timestamp_granularities must only contain 'segment' or 'word'"
                ),
            )
        if normalized not in merged:
            merged.append(normalized)
    return merged


def _resolve_asr_return_timestamps(
    response_format: str,
    timestamp_granularities: list[str],
) -> bool | str:
    if "word" in timestamp_granularities:
        return "word"
    if (
        response_format in {"verbose_json", "srt", "vtt"}
        or "segment" in timestamp_granularities
    ):
        return True
    return False


def _audio_duration_seconds(file_path: str) -> float:
    try:
        return float(AudioSegment.from_file(file_path).duration_seconds)
    except Exception:
        return 0.0


def _normalize_transcript_chunks(
    raw_chunks: object,
    duration_seconds: float,
    text: str,
) -> list[dict[str, object]]:
    chunks = raw_chunks if isinstance(raw_chunks, list) else []
    normalized: list[dict[str, object]] = []

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        timestamp = chunk.get("timestamp")
        start = end = None
        if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
            start = float(timestamp[0]) if timestamp[0] is not None else None
            end = float(timestamp[1]) if timestamp[1] is not None else None
        chunk_text = str(chunk.get("text") or "").strip()
        normalized.append(
            {
                "id": idx,
                "text": chunk_text,
                "start": start,
                "end": end,
            }
        )

    if normalized or not text:
        return normalized

    return [
        {
            "id": 0,
            "text": text,
            "start": 0.0,
            "end": duration_seconds or 0.0,
        }
    ]


def _format_transcription_timestamp(seconds: Optional[float], *, vtt: bool) -> str:
    total_ms = max(0, int(round((seconds or 0.0) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _render_subtitle_transcript(
    chunks: list[dict[str, object]],
    *,
    vtt: bool,
) -> str:
    lines: list[str] = ["WEBVTT", ""] if vtt else []
    for idx, chunk in enumerate(chunks, start=1):
        start = _format_transcription_timestamp(chunk.get("start"), vtt=vtt)
        end = _format_transcription_timestamp(chunk.get("end"), vtt=vtt)
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        if not vtt:
            lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _format_transcription_response(
    *,
    task: Literal["transcribe", "translate"],
    response_format: str,
    raw_result: dict[str, object],
    language: Optional[str],
    duration_seconds: float,
    timestamp_granularities: list[str],
    temperature: Optional[float],
) -> dict[str, object] | Response:
    text = str(raw_result.get("text") or "").strip()
    chunks = _normalize_transcript_chunks(
        raw_result.get("chunks"), duration_seconds, text
    )

    if response_format == "text":
        return Response(content=text, media_type="text/plain; charset=utf-8")

    if response_format == "srt":
        return Response(
            content=_render_subtitle_transcript(chunks, vtt=False),
            media_type="text/plain; charset=utf-8",
        )

    if response_format == "vtt":
        return Response(
            content=_render_subtitle_transcript(chunks, vtt=True),
            media_type="text/vtt; charset=utf-8",
        )

    if response_format == "verbose_json":
        words = []
        if "word" in timestamp_granularities:
            words = [
                {
                    "word": str(chunk.get("text") or "").strip(),
                    "start": chunk.get("start"),
                    "end": chunk.get("end"),
                }
                for chunk in chunks
                if str(chunk.get("text") or "").strip()
            ]
        segments = [
            {
                "id": chunk["id"],
                "seek": 0,
                "start": chunk.get("start") if chunk.get("start") is not None else 0.0,
                "end": chunk.get("end")
                if chunk.get("end") is not None
                else duration_seconds,
                "text": str(chunk.get("text") or "").strip(),
                "tokens": [],
                "temperature": temperature if temperature is not None else 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
            for chunk in chunks
            if str(chunk.get("text") or "").strip()
        ]
        payload: dict[str, object] = {
            "task": task,
            "language": str(raw_result.get("language") or language or ""),
            "duration": duration_seconds,
            "text": text,
            "segments": segments,
        }
        if words:
            payload["words"] = words
        return payload

    return {"text": text}


async def _handle_audio_transcription(
    *,
    task: Literal["transcribe", "translate"],
    file: UploadFile,
    model: Optional[str],
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    temperature: Optional[float],
    timestamp_granularities: Optional[list[str]],
    bracketed_timestamp_granularities: Optional[list[str]],
) -> dict[str, object] | Response:
    _resolve_transcription_model(model)
    timestamp_values = _coalesce_timestamp_granularities(
        timestamp_granularities,
        bracketed_timestamp_granularities,
    )
    if response_format not in SUPPORTED_TRANSCRIPTION_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported response_format '{response_format}'. Supported values: "
                f"{', '.join(sorted(SUPPORTED_TRANSCRIPTION_RESPONSE_FORMATS))}"
            ),
        )

    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    prompt_text = sanitize_prompt_text(prompt)
    uploaded = await file.read()
    if not uploaded:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded)
            temp_path = tmp.name

        raw_result = await asr_service.transcribe_file(
            temp_path,
            task=task,
            language=language,
            prompt=prompt_text,
            temperature=temperature,
            return_timestamps=_resolve_asr_return_timestamps(
                response_format,
                timestamp_values,
            ),
        )
        return _format_transcription_response(
            task=task,
            response_format=response_format,
            raw_result=raw_result,
            language=language,
            duration_seconds=_audio_duration_seconds(temp_path),
            timestamp_granularities=timestamp_values,
            temperature=temperature,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("Audio %s failed", task)
        raise HTTPException(status_code=500, detail=f"Audio {task} failed") from exc
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _resolve_model(requested: Optional[str]) -> str:
    if not requested:
        return API_MODEL_ID
    if requested in SUPPORTED_MODEL_ALIASES or requested == service.model_id:
        return API_MODEL_ID
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported model '{requested}'. Supported aliases: "
            f"{', '.join(sorted(SUPPORTED_MODEL_ALIASES))}"
        ),
    )


def _resolve_voice(requested: Optional[str]) -> ResolvedVoice:
    voice = (requested or DEFAULT_VOICE).strip()
    if not voice:
        raise HTTPException(status_code=400, detail="voice must not be empty")

    voice_key = voice.lower()
    preset = VOICE_LOOKUP.get(voice_key)
    if preset is not None:
        use_local_sample = preset.has_local_sample()
        return ResolvedVoice(
            voice_id=preset.id,
            display_name=preset.display_name(),
            instruct=None if use_local_sample else preset.fallback_instruct,
            ref_audio_path=preset.sample_path if use_local_sample else None,
            ref_text=preset.sample_ref_text if use_local_sample else None,
            default_language=preset.default_language,
        )

    return ResolvedVoice(
        voice_id=voice,
        display_name=voice,
        instruct=voice,
        ref_audio_path=None,
        ref_text=None,
        default_language=None,
    )


def _prepare_request(payload: SpeechRequest) -> PreparedSpeechRequest:
    _resolve_model(payload.model)
    response_format = payload.response_format
    resolved_voice = _resolve_voice(payload.voice)
    effective_language = payload.language or resolved_voice.default_language

    text = sanitize_speech_text(
        payload.raw_text(),
        language=effective_language,
        options=payload.normalization_options,
    )
    if not text:
        raise HTTPException(
            status_code=400, detail="Input text is empty after sanitization"
        )
    if len(text) > MAX_SANITIZED_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input text is too long after sanitization ({len(text)} chars). "
                f"Maximum allowed is {MAX_SANITIZED_INPUT_CHARS}."
            ),
        )

    instruct = sanitize_prompt_text(payload.instruct)
    if instruct is None and resolved_voice.ref_audio_path is None:
        instruct = sanitize_prompt_text(resolved_voice.instruct)

    resolved_ref_text = (
        sanitize_prompt_text(payload.ref_text)
        if payload.ref_text is not None
        else sanitize_prompt_text(resolved_voice.ref_text)
    )
    ref_text = resolved_ref_text if resolved_voice.ref_audio_path is None else None
    chunk_plan, chunk_detector = _plan_sentence_chunks_with_source(
        text, payload.sentence_chunking_min_chars
    )
    force_sentence_chunking = payload.sentence_chunking and len(chunk_plan) > 1

    generation_config_kwargs: dict[str, object] = {}
    for field_name in (
        "num_step",
        "guidance_scale",
        "t_shift",
        "denoise",
        "postprocess_output",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
    ):
        value = getattr(payload, field_name)
        if value is not None:
            generation_config_kwargs[field_name] = value

    generation_config_kwargs["audio_chunk_duration"] = (
        payload.audio_chunk_duration
        if payload.audio_chunk_duration is not None
        else DEFAULT_AUDIO_CHUNK_DURATION
    )
    generation_config_kwargs["audio_chunk_threshold"] = (
        0.0
        if force_sentence_chunking
        else (
            payload.audio_chunk_threshold
            if payload.audio_chunk_threshold is not None
            else DEFAULT_AUDIO_CHUNK_THRESHOLD
        )
    )

    return PreparedSpeechRequest(
        text=text,
        instruct=instruct,
        ref_text=ref_text,
        language=effective_language,
        response_format=response_format,
        voice_id=resolved_voice.voice_id,
        voice_display_name=resolved_voice.display_name,
        voice_ref_audio_path=resolved_voice.ref_audio_path,
        voice_ref_text=resolved_ref_text
        if resolved_voice.ref_audio_path is not None
        else None,
        chunk_plan=chunk_plan,
        force_sentence_chunking=force_sentence_chunking,
        chunk_detector=chunk_detector,
        parallel_workers=1,
        generation_config=OmniVoiceGenerationConfig.from_dict(generation_config_kwargs),
    )


def _waveform_to_bytes(
    waveform: torch.Tensor,
    sample_rate: int,
    response_format: str,
) -> tuple[bytes, str]:
    media_type, suffix = SUPPORTED_RESPONSE_FORMATS[response_format]

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.detach().cpu().float().clamp(-1.0, 1.0).contiguous()

    # torchaudio 2.9 routes file-like writes through torchcodec, which fails for
    # in-memory BytesIO WAV output in this environment. Write canonical PCM WAV
    # bytes ourselves, then feed those bytes to ffmpeg for compressed formats.
    pcm_waveform = (waveform * 32767.0).round().to(torch.int16).contiguous()
    if pcm_waveform.shape[0] == 1:
        pcm_bytes = pcm_waveform.squeeze(0).numpy().tobytes()
    else:
        pcm_bytes = pcm_waveform.transpose(0, 1).contiguous().numpy().tobytes()

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(int(pcm_waveform.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    wav_bytes = wav_buffer.getvalue()

    if response_format == "wav":
        return wav_bytes, media_type

    # For compressed formats, pipe WAV bytes to ffmpeg via stdin and read
    # the encoded output from stdout.  This avoids writing any temp files.
    _container_format = {
        "mp3": "mp3",
        "flac": "flac",
        "ogg": "ogg",
        "opus": "ogg",  # libopus in an Ogg container (same as .opus extension)
    }
    container = _container_format.get(response_format)
    if container is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'",
        )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",  # read WAV from stdin
    ]

    if response_format == "mp3":
        ffmpeg_cmd += ["-codec:a", "libmp3lame", "-b:a", "192k"]
    elif response_format == "flac":
        ffmpeg_cmd += ["-codec:a", "flac"]
    elif response_format in {"ogg", "opus"}:
        ffmpeg_cmd += ["-codec:a", "libopus"]

    ffmpeg_cmd += ["-f", container, "pipe:1"]  # write to stdout

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            input=wav_bytes,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is required to encode the requested audio format",
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode(errors="replace").strip()
        raise HTTPException(
            status_code=500,
            detail=f"Audio encoding failed: {stderr or exc}",
        ) from exc
    return result.stdout, media_type


def _wav_bytes_to_waveform(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        audio_buffer = io.BytesIO(wav_bytes)
        waveform, decoded_sample_rate = torchaudio.load(audio_buffer)
        return waveform, int(decoded_sample_rate)

    samples = torch.frombuffer(frames, dtype=torch.int16).clone().float() / 32767.0
    if channels > 1:
        samples = samples.reshape(-1, channels).transpose(0, 1).contiguous()
    else:
        samples = samples.unsqueeze(0)
    return samples, int(sample_rate)


def _speech_payload_for_chunk(payload: SpeechRequest, chunk: TextChunk) -> dict[str, object]:
    body = payload.model_dump(mode="json", exclude_none=True)
    body["input"] = chunk.text
    body.pop("text", None)
    body["response_format"] = "wav"
    body["sentence_chunking"] = False
    # A total-request duration cannot be applied independently to each chunk.
    body.pop("duration", None)
    return body


async def _synthesize_prepared_waveform(
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
    *,
    request_id: Optional[str] = None,
    worker_id: int = 1,
) -> tuple[torch.Tensor, int]:
    model = await service.get_model()

    generation_kwargs: dict[str, object] = {
        "text": prepared.text,
        "language": prepared.language,
        "speed": payload.speed,
        "generation_config": prepared.generation_config,
    }
    if prepared.ref_text is not None:
        generation_kwargs["ref_text"] = prepared.ref_text
    if prepared.instruct is not None:
        generation_kwargs["instruct"] = prepared.instruct
    if payload.duration is not None:
        generation_kwargs["duration"] = payload.duration

    LOG.info(
        "Synthesizing speech (voice=%s, response_format=%s, text_chars=%d, text_chunks=%d, forced_sentence_chunking=%s, voice_source=%s)",
        prepared.voice_id,
        prepared.response_format,
        len(prepared.text),
        len(prepared.chunk_plan),
        prepared.force_sentence_chunking,
        "local-reference"
        if prepared.voice_ref_audio_path is not None
        else "style-prompt",
    )
    LOG.debug("Sanitized input preview: %s", prepared.text[:200])

    def _run_generation() -> tuple[torch.Tensor, int]:
        generation_args = dict(generation_kwargs)
        if prepared.voice_ref_audio_path is not None:
            generation_args["voice_clone_prompt"] = (
                service.get_or_create_voice_clone_prompt(
                    model,
                    cache_key=(
                        f"{prepared.voice_ref_audio_path.resolve()}::{prepared.voice_ref_text!r}"
                    ),
                    ref_audio_path=prepared.voice_ref_audio_path,
                    ref_text=prepared.voice_ref_text,
                )
            )
        with torch.inference_mode():
            audios = model.generate(**generation_args)
        if not audios:
            raise RuntimeError("OmniVoice returned no audio")
        return audios[0], int(model.sampling_rate)

    try:
        if service.generation_lock is not None:
            async with service.generation_lock:
                return await asyncio.to_thread(_run_generation)
        return await asyncio.to_thread(_run_generation)
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception(
            "Speech synthesis failed (request_id=%s, worker_id=%s, voice=%s, voice_source=%s, language=%s)",
            request_id or "n/a",
            worker_id,
            prepared.voice_id,
            "local-reference"
            if prepared.voice_ref_audio_path is not None
            else "style-prompt",
            prepared.language,
        )
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from exc


async def _synthesize_chunk_local(
    chunk: TextChunk,
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
    *,
    request_id: str,
    retry_count: int,
) -> ChunkSynthesisResult:
    chunk_prepared = replace(
        prepared,
        text=chunk.text,
        response_format="wav",
        chunk_plan=[chunk.text],
        force_sentence_chunking=False,
        parallel_workers=prepared.parallel_workers,
        generation_config=replace(
            prepared.generation_config,
            audio_chunk_threshold=(
                payload.audio_chunk_threshold
                if payload.audio_chunk_threshold is not None
                else DEFAULT_AUDIO_CHUNK_THRESHOLD
            ),
        ),
    )
    started_at = time.time()
    start = time.perf_counter()
    waveform, sample_rate = await _synthesize_prepared_waveform(
        chunk_prepared,
        payload,
        request_id=request_id,
        worker_id=1,
    )
    ended_at = time.time()
    latency_s = time.perf_counter() - start
    return ChunkSynthesisResult(
        chunk_id=chunk.chunk_id,
        worker_id=1,
        input_length=len(chunk.text),
        waveform=waveform,
        sample_rate=sample_rate,
        retry_count=retry_count,
        started_at=started_at,
        ended_at=ended_at,
        latency_s=latency_s,
    )


async def _synthesize_chunk_remote(
    chunk: TextChunk,
    payload: SpeechRequest,
    *,
    request_id: str,
    retry_count: int,
) -> ChunkSynthesisResult:
    body = _speech_payload_for_chunk(payload, chunk)
    url = f"{secondary_worker_manager.base_url}/v1/audio/speech"
    started_at = time.time()
    start = time.perf_counter()

    def _post() -> bytes:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-OmniVoice-Parent-Request-Id": request_id,
                "X-OmniVoice-Chunk-Id": str(chunk.chunk_id),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=CHUNK_TIMEOUT_SECONDS) as response:
            return response.read()

    wav_bytes = await asyncio.to_thread(_post)
    waveform, sample_rate = _wav_bytes_to_waveform(wav_bytes)
    ended_at = time.time()
    latency_s = time.perf_counter() - start
    return ChunkSynthesisResult(
        chunk_id=chunk.chunk_id,
        worker_id=2,
        input_length=len(chunk.text),
        waveform=waveform,
        sample_rate=sample_rate,
        retry_count=retry_count,
        started_at=started_at,
        ended_at=ended_at,
        latency_s=latency_s,
    )


async def _run_chunk_on_worker(
    worker_id: int,
    chunk: TextChunk,
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
    *,
    request_id: str,
    retry_count: int,
) -> ChunkSynthesisResult:
    if worker_id == 2:
        return await _synthesize_chunk_remote(
            chunk,
            payload,
            request_id=request_id,
            retry_count=retry_count,
        )
    return await _synthesize_chunk_local(
        chunk,
        prepared,
        payload,
        request_id=request_id,
        retry_count=retry_count,
    )


async def _process_chunk_with_retry(
    chunk: TextChunk,
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
    *,
    request_id: str,
    secondary_available: bool,
) -> ChunkSynthesisResult:
    preferred = chunk.worker_id if secondary_available else 1
    alternate = 1 if preferred == 2 else 2
    worker_order = [preferred]
    if secondary_available and alternate not in worker_order:
        worker_order.append(alternate)
    if 1 not in worker_order:
        worker_order.append(1)

    failure_reason = "unknown"
    for attempt in range(CHUNK_RETRY_COUNT + 1):
        worker_id = worker_order[min(attempt, len(worker_order) - 1)]
        try:
            result = await asyncio.wait_for(
                _run_chunk_on_worker(
                    worker_id,
                    chunk,
                    prepared,
                    payload,
                    request_id=request_id,
                    retry_count=attempt,
                ),
                timeout=CHUNK_TIMEOUT_SECONDS,
            )
            LOG.info(
                "chunk_id=%s worker_id=%s input_length=%s start_time=%.6f end_time=%.6f latency=%.3fs retry_count=%s failure_reason=-",
                result.chunk_id,
                result.worker_id,
                result.input_length,
                result.started_at,
                result.ended_at,
                result.latency_s,
                result.retry_count,
            )
            return result
        except asyncio.CancelledError:
            LOG.info(
                "chunk_id=%s worker_id=%s input_length=%s retry_count=%s cancelled",
                chunk.chunk_id,
                worker_id,
                len(chunk.text),
                attempt,
            )
            raise
        except Exception as exc:
            failure_reason = str(exc) or exc.__class__.__name__
            LOG.warning(
                "chunk_id=%s worker_id=%s input_length=%s retry_count=%s failure_reason=%s",
                chunk.chunk_id,
                worker_id,
                len(chunk.text),
                attempt,
                failure_reason,
            )
            if attempt >= CHUNK_RETRY_COUNT:
                break
    raise ChunkSynthesisError(chunk.chunk_id, failure_reason)


async def _iter_ordered_chunk_results(
    chunks: list[TextChunk],
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
    *,
    request_id: str,
    secondary_available: bool,
):
    if len(chunks) > CHUNK_MAX_QUEUE_DEPTH:
        raise HTTPException(
            status_code=413,
            detail=f"Chunk queue depth {len(chunks)} exceeds cap {CHUNK_MAX_QUEUE_DEPTH}",
        )

    semaphore = asyncio.Semaphore(CHUNK_MAX_WORKERS if secondary_available else 1)

    async def _run(chunk: TextChunk) -> ChunkSynthesisResult:
        async with semaphore:
            return await _process_chunk_with_retry(
                chunk,
                prepared,
                payload,
                request_id=request_id,
                secondary_available=secondary_available,
            )

    tasks = [asyncio.create_task(_run(chunk)) for chunk in chunks]
    reorder_buffer: dict[int, ChunkSynthesisResult] = {}
    next_chunk_id = 1

    try:
        for completed in asyncio.as_completed(tasks):
            result = await completed
            reorder_buffer[result.chunk_id] = result
            while next_chunk_id in reorder_buffer:
                yield reorder_buffer.pop(next_chunk_id)
                next_chunk_id += 1
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        raise
    except Exception:
        for task in tasks:
            task.cancel()
        raise
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _should_use_parallel_chunking(
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
) -> bool:
    if WORKER_MODE or not CHUNKED_PIPELINE_ENABLED:
        return False
    if CHUNK_MAX_WORKERS < 2:
        return False
    if payload.duration is not None:
        return False
    if not payload.sentence_chunking or not prepared.force_sentence_chunking:
        return False
    if len(prepared.text) < CHUNK_MIN_PARALLEL_CHARS:
        return False
    return len(prepared.chunk_plan) >= CHUNK_MIN_PARALLEL_CHUNKS


async def _try_synthesize_chunked(
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
    *,
    request_id: str,
) -> tuple[bytes, str] | None:
    if not _should_use_parallel_chunking(prepared, payload):
        return None

    chunks = _build_text_chunks(prepared.chunk_plan)
    if len(chunks) < CHUNK_MIN_PARALLEL_CHUNKS:
        return None

    secondary_available = await secondary_worker_manager.ensure_started()
    if not secondary_available:
        return None

    prepared.parallel_workers = 2
    LOG.info(
        "Dispatching %d text chunks across two workers (request_id=%s, detector=%s)",
        len(chunks),
        request_id,
        prepared.chunk_detector,
    )

    waveforms: list[torch.Tensor] = []
    sample_rate: int | None = None
    try:
        async for result in _iter_ordered_chunk_results(
            chunks,
            prepared,
            payload,
            request_id=request_id,
            secondary_available=secondary_available,
        ):
            waveforms.append(result.waveform)
            sample_rate = result.sample_rate if sample_rate is None else sample_rate
    except ChunkSynthesisError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not waveforms or sample_rate is None:
        raise HTTPException(status_code=500, detail="Chunked synthesis returned no audio")

    merged = cross_fade_chunks(waveforms, sample_rate)
    return await asyncio.to_thread(
        _waveform_to_bytes,
        merged,
        sample_rate,
        prepared.response_format,
    )


async def _synthesize_prepared(
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
    *,
    request_id: Optional[str] = None,
) -> tuple[PreparedSpeechRequest, bytes, str]:
    effective_request_id = request_id or uuid4().hex[:12]
    chunked = await _try_synthesize_chunked(
        prepared,
        payload,
        request_id=effective_request_id,
    )
    if chunked is not None:
        audio_bytes, media_type = chunked
        return prepared, audio_bytes, media_type

    waveform, sample_rate = await _synthesize_prepared_waveform(
        prepared,
        payload,
        request_id=effective_request_id,
        worker_id=1,
    )
    audio_bytes, media_type = await asyncio.to_thread(
        _waveform_to_bytes,
        waveform,
        sample_rate,
        prepared.response_format,
    )
    return prepared, audio_bytes, media_type


@asynccontextmanager
async def lifespan(_: FastAPI):
    _configure_logging()
    service.set_lock(asyncio.Lock())
    service.set_generation_lock(asyncio.Semaphore(1))
    asr_service.set_lock(asyncio.Lock())
    secondary_worker_manager.set_lock(asyncio.Lock())
    service.schedule_voice_prompt_prewarm()
    LOG.info(
        "Starting OmniVoice TTS server (api_model=%s, backend_model=%s, device=%s, idle_timeout=%.0fs, asr_model=%s, asr_device=%s, asr_idle_timeout=%.0fs, worker_mode=%s, chunked_pipeline=%s, chunk_max_workers=%s)",
        API_MODEL_ID,
        service.model_id,
        service.device,
        service._idle_timeout,
        asr_service.model_id,
        asr_service.device,
        asr_service._idle_timeout,
        WORKER_MODE,
        CHUNKED_PIPELINE_ENABLED,
        CHUNK_MAX_WORKERS,
    )
    yield
    if service._voice_prewarm_task is not None:
        service._voice_prewarm_task.cancel()
    await secondary_worker_manager.shutdown()
    if service._idle_task is not None:
        service._idle_task.cancel()
    if asr_service._idle_task is not None:
        asr_service._idle_task.cancel()
    if service.model is not None:
        LOG.info("Shutting down, offloading model...")
        try:
            if service.device.startswith("cuda"):
                await asyncio.to_thread(service._offload_model_sync)
            else:
                service.model = None
        except Exception:
            LOG.exception("Error offloading model during shutdown")
    if asr_service.pipeline is not None:
        LOG.info("Shutting down, offloading ASR model...")
        try:
            await asyncio.to_thread(asr_service._offload_pipeline_sync)
        except Exception:
            LOG.exception("Error offloading ASR model during shutdown")


service = OmniVoiceService(BACKEND_MODEL_ID, DEFAULT_DEVICE)
asr_service = ASRService(
    DEFAULT_ASR_MODEL_ID,
    DEFAULT_ASR_DEVICE,
    idle_timeout=DEFAULT_ASR_IDLE_TIMEOUT,
)
secondary_worker_manager = SecondaryWorkerManager()
app = FastAPI(title="OmniVoice OpenAI-Compatible TTS", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    request_id = uuid4().hex[:12]
    request.state.request_id = request_id
    request.state.request_source = _guess_request_source(request)
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        LOG.exception(
            "Request failed (request_id=%s, source=%s, client=%s, %s %s, %.1fms)",
            request_id,
            request.state.request_source,
            getattr(request.client, "host", "unknown"),
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-OmniVoice-Request-Id"] = request_id
    if request.url.path not in {"/", "/health"}:
        LOG.info(
            "%s %s -> %s in %.1fms (request_id=%s, source=%s, client=%s)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
            request.state.request_source,
            getattr(request.client, "host", "unknown"),
        )
    return response


@app.get("/")
@app.get("/health")
@app.get("/v1/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "supported_response_formats": sorted(SUPPORTED_RESPONSE_FORMATS),
        "supported_transcription_response_formats": sorted(
            SUPPORTED_TRANSCRIPTION_RESPONSE_FORMATS
        ),
        **service.health(),
        **asr_service.health(),
        "chunked_pipeline": secondary_worker_manager.health(),
    }


@app.get("/ui", response_class=HTMLResponse)
@app.get("/credits", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    return _render_frontend_page()


@app.get("/v1/models")
@app.get("/models")
async def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": model["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
            for model in _supported_models()
        ],
    }


@app.get("/audio/models")
@app.get("/v1/audio/models")
async def list_audio_models() -> dict[str, list[dict[str, str]]]:
    return {"models": _supported_models()}


@app.get("/audio/voices")
@app.get("/v1/audio/voices")
async def list_audio_voices() -> dict[str, list[dict[str, str]]]:
    return {"voices": _supported_voices()}


@app.post("/audio/speech")
@app.post("/v1/audio/speech")
async def audio_speech(payload: SpeechRequest, request: Request) -> Response:
    _ = request.headers.get("authorization")
    request_id = getattr(request.state, "request_id", uuid4().hex[:12])
    request_source = getattr(
        request.state, "request_source", _guess_request_source(request)
    )
    prepared = _prepare_request(payload)
    voice_mode = (
        "local-reference"
        if prepared.voice_ref_audio_path is not None
        else "style-prompt"
    )
    ref_text_source = (
        "request"
        if payload.ref_text is not None
        else "preset"
        if prepared.voice_ref_text is not None
        else "none"
    )
    LOG.info(
        "Prepared speech request (request_id=%s, source=%s, voice=%s, voice_mode=%s, language=%s, raw_chars=%d, sanitized_chars=%d, normalization=%s, ref_audio=%s, ref_text_source=%s, ref_text_chars=%d, preview_raw=%r, preview_sanitized=%r)",
        request_id,
        request_source,
        prepared.voice_id,
        voice_mode,
        prepared.language,
        len(payload.raw_text()),
        len(prepared.text),
        _normalization_summary(payload.normalization_options),
        prepared.voice_ref_audio_path.name
        if prepared.voice_ref_audio_path is not None
        else "-",
        ref_text_source,
        len(prepared.voice_ref_text or prepared.ref_text or ""),
        _truncate_preview(payload.raw_text()),
        _truncate_preview(prepared.text),
    )
    prepared, audio_bytes, media_type = await _synthesize_prepared(
        prepared,
        payload,
        request_id=request_id,
    )
    _, suffix = SUPPORTED_RESPONSE_FORMATS[prepared.response_format]
    headers = {
        "Content-Disposition": f'attachment; filename="speech.{suffix}"',
        "X-OmniVoice-Text-Chunks": str(len(prepared.chunk_plan)),
        "X-OmniVoice-Forced-Chunking": str(prepared.force_sentence_chunking).lower(),
        "X-OmniVoice-Sentence-Detector": prepared.chunk_detector,
        "X-OmniVoice-Parallel-Workers": str(prepared.parallel_workers),
        "X-OmniVoice-Voice-Mode": voice_mode,
        "X-OmniVoice-Sanitized-Chars": str(len(prepared.text)),
        "X-OmniVoice-Language": prepared.language or "",
        "X-OmniVoice-Voice-Source": (
            "local-reference"
            if prepared.voice_ref_audio_path is not None
            else "style-prompt"
        ),
    }
    return Response(content=audio_bytes, media_type=media_type, headers=headers)


@app.post("/audio/transcriptions")
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Form(
        default="json"
    ),
    temperature: Optional[float] = Form(default=None),
    timestamp_granularities: Optional[list[str]] = Form(default=None),
    bracketed_timestamp_granularities: Optional[list[str]] = Form(
        default=None,
        alias="timestamp_granularities[]",
    ),
):
    return await _handle_audio_transcription(
        task="transcribe",
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        bracketed_timestamp_granularities=bracketed_timestamp_granularities,
    )


@app.post("/audio/translations")
@app.post("/v1/audio/translations")
async def audio_translations(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Form(
        default="json"
    ),
    temperature: Optional[float] = Form(default=None),
    timestamp_granularities: Optional[list[str]] = Form(default=None),
    bracketed_timestamp_granularities: Optional[list[str]] = Form(
        default=None,
        alias="timestamp_granularities[]",
    ),
):
    return await _handle_audio_transcription(
        task="translate",
        file=file,
        model=model,
        language=None,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        bracketed_timestamp_granularities=bracketed_timestamp_granularities,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice OpenAI-compatible TTS server"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--model-id",
        default=BACKEND_MODEL_ID,
        help="Local OmniVoice checkpoint directory.",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=float(os.getenv("OMNIVOICE_IDLE_TIMEOUT", "300")),
        help="Seconds of inactivity before offloading model from GPU (0 to disable)",
    )
    parser.add_argument(
        "--worker-mode",
        action="store_true",
        default=WORKER_MODE,
        help="Run as an internal chunk worker and never launch another worker",
    )
    return parser


def main() -> None:
    _configure_logging()
    args = build_parser().parse_args()

    model_id = require_local_path(args.model_id, arg_name="--model-id", expect_dir=True)

    global WORKER_MODE
    WORKER_MODE = bool(args.worker_mode)
    if WORKER_MODE:
        os.environ["OMNIVOICE_WORKER_MODE"] = "1"

    global service
    service = OmniVoiceService(
        model_id, args.device, idle_timeout=args.idle_timeout
    )
    global asr_service
    asr_service = ASRService(
        DEFAULT_ASR_MODEL_ID,
        DEFAULT_ASR_DEVICE,
        idle_timeout=DEFAULT_ASR_IDLE_TIMEOUT,
    )
    global secondary_worker_manager
    secondary_worker_manager = SecondaryWorkerManager(port=args.port + CHUNK_SECONDARY_PORT_OFFSET)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        workers=1,
        loop="uvloop",
        http="httptools",
    )


if __name__ == "__main__":
    main()
