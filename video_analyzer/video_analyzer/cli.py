from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_analyzer.agent import AnalysisResult, VideoAnalyzerAgent
from video_analyzer.config import Settings
from video_analyzer.media import MediaError
from video_analyzer.webtext import fetch_webpage_plain_text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze a video (e.g. MP4/MOV): transcribe speech, caption frames, print a summary.",
    )
    p.add_argument(
        "video",
        type=str,
        help="Local path (.mp4, .mov, …) or http(s) URL to a video file",
    )
    p.add_argument(
        "--context-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Local text/markdown file with campaign notes (use with --webpage-url for ad JSON)",
    )
    p.add_argument(
        "--webpage-url",
        type=str,
        default=None,
        metavar="URL",
        help="Public http(s) page whose text is fetched (use with --context-file for ad JSON)",
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
        help="Write full result (markdown + optional JSON) to this file",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if (args.context_file is None) != (args.webpage_url is None):
        print(
            "Use both --context-file and --webpage-url together for ad metadata JSON, or omit both.",
            file=sys.stderr,
        )
        return 2

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

    metadata: dict[str, str] | None = None
    if args.context_file is not None and args.webpage_url is not None:
        try:
            ctx_path = args.context_file.expanduser().resolve()
            context_text = ctx_path.read_text(encoding="utf-8")
        except OSError as e:
            print(e, file=sys.stderr)
            return 1
        try:
            webpage_text = fetch_webpage_plain_text(args.webpage_url)
        except MediaError as e:
            print(e, file=sys.stderr)
            return 1
        try:
            metadata = agent.infer_ad_metadata_json(result, context_text, webpage_text)
        except MediaError as e:
            print(e, file=sys.stderr)
            return 1

    text = _format_output(result, metadata)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"\nWrote: {args.output}", file=sys.stderr)

    return 0


def _format_output(result: AnalysisResult, metadata: dict[str, str] | None) -> str:
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
    if metadata is not None:
        parts.append("\n\n---\n\n## Ad metadata (JSON)\n\n")
        parts.append(json.dumps(metadata, indent=2, ensure_ascii=False))
        parts.append("\n")
    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
