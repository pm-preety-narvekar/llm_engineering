"""Download HTTP(S) videos to a temp file for local ffmpeg processing."""

from __future__ import annotations

import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from video_analyzer.media import MediaError

DEFAULT_USER_AGENT = "video-analyzer/0.1"


def is_http_url(s: str) -> bool:
    """True for remote video URLs. Do not pass these through :class:`pathlib.Path`.resolve()."""
    t = s.strip()
    if len(t) < 8:
        return False
    low = t.lower()
    if low.startswith("https://") or low.startswith("http://"):
        return True
    p = urlparse(t)
    return p.scheme in ("http", "https") and bool(p.netloc)


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    if name and "." in name:
        return name
    return "video_download.mp4"


def download_video(url: str, dest_dir: Path, *, max_bytes: int) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / _filename_from_url(url)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = 0
            with out.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        out.unlink(missing_ok=True)
                        raise MediaError(
                            f"Download exceeded {max_bytes} bytes "
                            "(VIDEO_ANALYZER_MAX_DOWNLOAD_BYTES)."
                        )
                    f.write(chunk)
    except urllib.error.HTTPError as e:
        raise MediaError(f"HTTP {e.code} while downloading video: {url}") from e
    except urllib.error.URLError as e:
        raise MediaError(f"Failed to download video: {e.reason!r}") from e
    if not out.exists() or out.stat().st_size == 0:
        raise MediaError(f"Downloaded file is empty: {url}")
    return out


@contextmanager
def local_video_path(
    path_or_url: str | Path,
    *,
    max_download_bytes: int,
) -> Iterator[tuple[Path, str]]:
    """
    Yield ``(local_path, source_label)``.
    For URLs, downloads into a temp directory that is removed after the block.
    """
    s = str(path_or_url).strip()
    if not s:
        raise ValueError("Empty video path or URL")

    if is_http_url(s):
        td = tempfile.TemporaryDirectory(prefix="video_analyzer_url_")
        try:
            local = download_video(s, Path(td.name), max_bytes=max_download_bytes)
            yield local, s
        finally:
            td.cleanup()
    else:
        p = Path(s).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        yield p, s
