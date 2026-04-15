# Video analyzer (sample agent)

This sample project analyzes a local or remote video file (e.g. `.mp4`, `.mov`, and other formats **ffmpeg** supports):

1. **Speech** — extracts audio with `ffmpeg`, transcribes locally with **Hugging Face** ASR (no OpenAI `whisper-1` call).
2. **Visuals** — samples JPEG frames at a fixed interval and asks a **vision** model to describe them.
3. **Summary** — merges transcript + visual notes into a structured markdown summary.

## Prerequisites

- **Python 3.9+**
- **`ffmpeg` and `ffprobe`** on your `PATH` (e.g. `brew install ffmpeg` on macOS).
- **`pip install -e ".[hf]"`** (or equivalent) for local transcription (`transformers`, PyTorch, etc.).
- An **API key** for **chat/vision** models only (e.g. `(paid) gpt-4o-mini` on a gateway); transcription does not use it.

## Setup

From this directory:

```bash
cd video_analyzer
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
uv sync --extra hf
```

Alternatively: `pip install -e ".[hf]"` in a virtual environment.

## Usage

```bash
uv run video-analyze /path/to/video.mp4
uv run video-analyze /path/to/promo.mov
```

Or:

```bash
uv run python -m video_analyzer /path/to/video.mp4
```

Options:

- **`--context-file PATH`** and **`--webpage-url URL`** — pass **both** to append **Ad metadata (JSON)** (`title`, `icon_url`, `button_txt`, `description`) using the video report plus local file text and fetched webpage text. Example: `samples/ad_context_example.txt`, `samples/example_run.sh`.
- `--frame-interval SEC` — seconds between sampled frames (default: 15, or `VIDEO_ANALYZER_FRAME_INTERVAL_SEC`).
- `--max-frames N` — cap frames for cost control (default: 24).
- `--no-frames` — transcript only (no vision calls).
- `-o out.md` — write the full markdown report to a file.

## Configuration

Environment variables (see `.env.example`):

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required |
| `VIDEO_ANALYZER_VISION_MODEL` | Vision + frame batch captions (default `gpt-4o-mini`) |
| `VIDEO_ANALYZER_SUMMARY_MODEL` | Final summary (default `gpt-4o-mini`) |
| `VIDEO_ANALYZER_ENRICHMENT_MODEL` | Ad metadata JSON step (default `gpt-4o-mini`) |
| `VIDEO_ANALYZER_FRAME_INTERVAL_SEC` | Frame sampling interval |
| `VIDEO_ANALYZER_MAX_FRAMES` | Maximum frames extracted |
| `VIDEO_ANALYZER_VISION_BATCH_SIZE` | Frames per vision API call |
| `VIDEO_ANALYZER_CURL_IMPERSONATE` | Browser TLS profile for `--webpage-url` (default `chrome120`; see [curl_cffi](https://github.com/yifeikong/curl_cffi)) |

`--webpage-url` uses **curl_cffi** (declared dependency) to mimic a real browser TLS stack; plain `urllib` is only a fallback if `curl_cffi` cannot be imported.

## Limits

- The Whisper API accepts uploads up to **25 MB**. This project compresses audio to mono MP3; very long videos may still exceed the limit — use a shorter clip or lower bitrate by editing `media.extract_audio_mp3`.
- Vision calls are billed per image; reduce `--max-frames` or increase `--frame-interval` to save cost.

## Programmatic use

```python
from pathlib import Path
from video_analyzer import VideoAnalyzerAgent

agent = VideoAnalyzerAgent()
result = agent.analyze(Path("sample.mp4"))
print(result.summary)
```

## License

MIT (same spirit as the parent course repo; adjust if your org requires otherwise).
