from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load video_analyzer/.env first (works when cwd is repo root), then cwd .env
_PKG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PKG_ROOT / ".env")
load_dotenv()


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration from environment variables.

    Transcription is **local Hugging Face ASR only** (never calls OpenAI ``whisper-1``).
    ``openai_base_url``: optional gateway URL (e.g. Pubmatic); vision/summary only.
    """

    openai_api_key: str
    openai_base_url: str | None = None
    hf_transcription_model: str = "distil-whisper/distil-small.en"
    hf_chunk_length_sec: float = 30.0
    hf_asr_batch_size: int = 8
    vision_model: str = "(paid) gpt-4o-mini"
    summary_model: str = "(paid) gpt-4o-mini"
    frame_interval_sec: float = 15.0
    max_frames: int = 24
    vision_batch_size: int = 8
    max_audio_bytes: int = 24 * 1024 * 1024

    @staticmethod
    def from_env() -> "Settings":
        key = "sk-4e7h5BLXEa-N5FX430dmHw" #os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        legacy = _env("VIDEO_ANALYZER_TRANSCRIPTION_BACKEND", "").strip().lower()
        if legacy == "openai":
            raise ValueError(
                "OpenAI Whisper API (whisper-1) is disabled: your key cannot access it. "
                "Remove VIDEO_ANALYZER_TRANSCRIPTION_BACKEND=openai from your environment. "
                "Transcription uses local Hugging Face ASR only; install: pip install 'video-analyzer[hf]'"
            )
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        return Settings(
            openai_api_key=key,
            openai_base_url=base_url,
            hf_transcription_model=_env(
                "VIDEO_ANALYZER_HF_TRANSCRIPTION_MODEL",
                "distil-whisper/distil-small.en",
            ),
            hf_chunk_length_sec=float(_env("VIDEO_ANALYZER_HF_CHUNK_LENGTH_SEC", "30")),
            hf_asr_batch_size=int(_env("VIDEO_ANALYZER_HF_ASR_BATCH_SIZE", "8")),
            vision_model=_env("VIDEO_ANALYZER_VISION_MODEL", "(paid) gpt-4o-mini"),
            summary_model=_env("VIDEO_ANALYZER_SUMMARY_MODEL", "(paid) gpt-4o-mini"),
            frame_interval_sec=float(_env("VIDEO_ANALYZER_FRAME_INTERVAL_SEC", "15")),
            max_frames=int(_env("VIDEO_ANALYZER_MAX_FRAMES", "24")),
            vision_batch_size=int(_env("VIDEO_ANALYZER_VISION_BATCH_SIZE", "8")),
        )
