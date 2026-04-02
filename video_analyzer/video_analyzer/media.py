from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class MediaError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise MediaError(
            "ffmpeg and ffprobe must be on your PATH. Install ffmpeg (e.g. brew install ffmpeg)."
        )


def get_duration_sec(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"ffprobe failed: {e.stderr or e}") from e
    try:
        return float(out.stdout.strip())
    except ValueError as e:
        raise MediaError(f"Could not parse duration from ffprobe output: {out.stdout!r}") from e


def extract_audio_mp3(video_path: Path, out_path: Path) -> None:
    """Speech-optimized mono MP3 to stay under Whisper upload limits when possible."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "48k",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"ffmpeg audio extract failed: {e.stderr or e}") from e


def has_audio_stream(video_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"ffprobe failed: {e.stderr or e}") from e
    return "audio" in out.stdout.lower() or bool(out.stdout.strip())


def extract_frames_even_interval(
    video_path: Path,
    interval_sec: float,
    max_frames: int,
) -> list[tuple[Path, float]]:
    """
    Extract PNG frames every ``interval_sec``, capped at ``max_frames``.
    Uses PNG (not MJPEG) so odd dimensions (e.g. 1440×780) and full-range YUV do not break encoding.
    Returns list of (image_path, timestamp_sec).
    """
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    duration = get_duration_sec(video_path)
    if duration <= 0:
        return []

    tmp = Path(tempfile.mkdtemp(prefix="video_analyzer_frames_"))
    frames: list[tuple[Path, float]] = []
    t = 0.0
    idx = 0
    while t < duration and len(frames) < max_frames:
        out_file = tmp / f"frame_{idx:05d}.png"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(t),
            "-i",
            str(video_path),
            "-an",
            "-sn",
            "-frames:v",
            "1",
            "-vcodec",
            "png",
            str(out_file),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise MediaError(f"ffmpeg frame extract failed at t={t}: {e.stderr or e}") from e
        if out_file.exists() and out_file.stat().st_size > 0:
            frames.append((out_file, round(t, 2)))
        t += interval_sec
        idx += 1

    return frames
