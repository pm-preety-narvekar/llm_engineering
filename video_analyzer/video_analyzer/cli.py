from __future__ import annotations

import argparse
import sys
from pathlib import Path

from video_analyzer.agent import AnalysisResult, VideoAnalyzerAgent
from video_analyzer.config import Settings
from video_analyzer.media import MediaError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze an MP4: transcribe speech, caption sampled frames, print a summary.",
    )
    p.add_argument(
        "video",
        type=str,
        help="Local path to a video file, or http(s) URL (downloaded to a temp file first)",
    )
    p.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help="Seconds between sampled frames (default: from env or 15)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: from env or 24)",
    )
    p.add_argument(
        "--no-frames",
        action="store_true",
        help="Skip frame extraction and vision; transcript-only summary",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write full result (markdown) to this file",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    agent = VideoAnalyzerAgent(settings)
    try:
        result = agent.analyze(
            args.video,
            frame_interval_sec=args.frame_interval,
            max_frames=args.max_frames,
            include_frames=not args.no_frames,
        )
    except (MediaError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1

    text = _format_output(result)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"\nWrote: {args.output}", file=sys.stderr)

    return 0


def _format_output(result: AnalysisResult) -> str:
    parts = []
    parts.append(f"**Source:** {result.source}\n\n")
    if result.duration_sec is not None:
        parts.append(f"**Duration:** {result.duration_sec:.1f}s\n")
    parts.append("## Summary\n\n")
    parts.append(result.summary)
    parts.append("\n\n---\n\n## Transcript\n\n")
    if result.transcript:
        parts.append(result.transcript)
    else:
        parts.append("_No transcript (silent or no audio)._")
    if result.visual_notes:
        parts.append("\n\n## Visual notes (frames)\n\n")
        parts.append(result.visual_notes)
    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
