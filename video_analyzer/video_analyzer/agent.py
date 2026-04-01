from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from video_analyzer.config import Settings
from video_analyzer.media import (
    MediaError,
    ensure_ffmpeg,
    extract_audio_mp3,
    extract_frames_even_interval,
    get_duration_sec,
    has_audio_stream,
)


@dataclass
class AnalysisResult:
    video_path: Path
    duration_sec: float | None
    transcript: str | None
    visual_notes: str
    summary: str


def _data_url_for_image(path: Path) -> str:
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _pick_torch_device_and_dtype() -> tuple[str | int, str]:
    """Return (device, dtype_str) for transformers ASR."""
    try:
        import torch
    except ImportError as e:
        raise MediaError(
            "Local transcription needs PyTorch. Install: pip install torch"
        ) from e
    if torch.cuda.is_available():
        return 0, "float16"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "float16"
    return "cpu", "float32"


class VideoAnalyzerAgent:
    """
    Pipeline agent: extract audio → local HF transcript (never OpenAI whisper-1),
    sample frames → vision captions in batches → final LLM summary.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.client = OpenAI(
            api_key="abc", #TODO: update #api_key=self.settings.openai_api_key,
            base_url="https://llm.pubmatic.com" #base_url=self.settings.openai_base_url,
        )
        self._hf_asr_pipe: object | None = None

    def _get_hf_asr_pipeline(self) -> object:
        if self._hf_asr_pipe is not None:
            return self._hf_asr_pipe
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise MediaError(
                "Local ASR requires transformers (and PyTorch). "
                "Install: pip install 'video-analyzer[hf]' or "
                "pip install transformers torch soundfile accelerate"
            ) from e
        device, dtype_str = _pick_torch_device_and_dtype()
        dtype = getattr(torch, dtype_str)
        model_id = self.settings.hf_transcription_model
        self._hf_asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dtype,
            device=device,
        )
        return self._hf_asr_pipe

    def _transcribe_hf(self, audio_path: Path) -> str:
        pipe = self._get_hf_asr_pipeline()
        out = pipe(
            str(audio_path),
            chunk_length_s=self.settings.hf_chunk_length_sec,
            batch_size=self.settings.hf_asr_batch_size,
            return_timestamps=False,
        )
        if isinstance(out, dict):
            text = (out.get("text") or "").strip()
        else:
            text = str(out).strip()
        return text

    def transcribe(self, audio_path: Path) -> str:
        size = audio_path.stat().st_size
        if size > self.settings.max_audio_bytes:
            raise MediaError(
                f"Compressed audio is still {size} bytes (limit {self.settings.max_audio_bytes}). "
                "Use a shorter clip or lower bitrate; limit is set for API upload / memory."
            )
        return self._transcribe_hf(audio_path)

    def describe_frame_batch(
        self,
        frames: list[tuple[Path, float]],
    ) -> str:
        if not frames:
            return ""
        user_parts: list[dict] = [
            {
                "type": "text",
                "text": (
                    "You are helping summarize a video. For each image in order, the timestamp "
                    "in seconds is given in the label. Briefly describe what is visible "
                    "(scene, actions, on-screen text, slides, UI). "
                    "Use bullet points, one per image, format: `- [t=12.0s] ...`"
                ),
            }
        ]
        for path, ts in frames:
            user_parts.append({"type": "text", "text": f"Frame at t={ts}s:"})
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url_for_image(path)},
                }
            )

        resp = self.client.chat.completions.create(
            model=self.settings.vision_model,
            messages=[
                {
                    "role": "system",
                    "content": "Be concise and factual; skip speculation.",
                },
                {"role": "user", "content": user_parts},
            ],
            max_tokens=1200,
        )
        choice = resp.choices[0].message.content
        return (choice or "").strip()

    def summarize(
        self,
        transcript: str | None,
        visual_notes: str,
        duration_sec: float | None,
    ) -> str:
        lines = [
            "You are a creatve designer assistant for an advertising agency. Produce a clear summary of the video advertising campaign.",
            "",
        ]
        if duration_sec is not None:
            lines.append(f"Approximate duration: {duration_sec:.1f} seconds.")
            lines.append("")
        if transcript:
            lines.append("## Transcript (from speech)")
            lines.append(transcript)
            lines.append("")
        else:
            lines.append("## Transcript")
            lines.append("(No usable audio transcript was produced.)")
            lines.append("")
        lines.append("## Visual notes (from sampled frames)")
        lines.append(visual_notes or "(No frames analyzed.)")
        lines.append("")
        lines.append(
            "Write the final answer with these sections:\n"
            "# Title\n"
            "One line.\n\n"
            "# Overview\n"
            "2-4 sentences.\n\n"
            "# Key points\n"
            "Bullet list.\n\n"
            "# Notable visuals\n"
            "Short bullets, or say if none stood out.\n"
        )
        prompt = "\n".join(lines)

        resp = self.client.chat.completions.create(
            model=self.settings.summary_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return (resp.choices[0].message.content or "").strip()

    def analyze(
        self,
        video_path: Path,
        *,
        frame_interval_sec: float | None = None,
        max_frames: int | None = None,
        include_frames: bool = True,
    ) -> AnalysisResult:
        path = video_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        ensure_ffmpeg()
        duration = get_duration_sec(path)

        interval = frame_interval_sec if frame_interval_sec is not None else self.settings.frame_interval_sec
        cap = max_frames if max_frames is not None else self.settings.max_frames

        transcript: str | None = None
        if has_audio_stream(path):
            with tempfile.TemporaryDirectory(prefix="video_analyzer_audio_") as td:
                mp3 = Path(td) / "audio.mp3"
                extract_audio_mp3(path, mp3)
                transcript = self.transcribe(mp3)
                if not transcript:
                    transcript = None

        visual_notes = ""
        if include_frames:
            frames = extract_frames_even_interval(path, interval, cap)
            batches: list[str] = []
            bs = max(1, self.settings.vision_batch_size)
            for i in range(0, len(frames), bs):
                chunk = frames[i : i + bs]
                batches.append(self.describe_frame_batch(chunk))
            visual_notes = "\n\n".join(b for b in batches if b)

        summary = self.summarize(transcript, visual_notes, duration)

        return AnalysisResult(
            video_path=path,
            duration_sec=duration,
            transcript=transcript,
            visual_notes=visual_notes,
            summary=summary,
        )
