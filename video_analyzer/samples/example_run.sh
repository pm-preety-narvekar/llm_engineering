#!/usr/bin/env bash
# Example: analyze a remote video + local context file + public webpage → markdown report + JSON ad metadata.
# Run from the `video_analyzer` directory with your venv activated and `.env` configured.

set -euo pipefail

VIDEO_URL="${1:-https://mathworksheets.dreamhosters.com/wp-content/uploads/2025/07/v09044g40000c2bceem39r9m517i44og.mov}"
CONTEXT_FILE="${2:-samples/ad_context_example.txt}"
WEB_URL="${3:-https://mathworksheets.dreamhosters.com/}"

python -m video_analyzer "$VIDEO_URL" \
  --context-file "$CONTEXT_FILE" \
  --webpage-url "$WEB_URL"
