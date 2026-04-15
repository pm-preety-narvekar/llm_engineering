"""Download HTTP(S) videos to a temp file for local ffmpeg processing."""

from __future__ import annotations

import re
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

from video_analyzer.media import MediaError

DEFAULT_USER_AGENT = "video-analyzer/0.1"


def normalize_video_input(s: str) -> str:
    """
    Fix copy-pasted URLs that use JSON-style escaped slashes (``https:\\/\\/host\\/path``).
    Those are not valid ``https://`` strings and would be mistaken for local paths.
    """
    t = s.strip()
    if "\\/" in t:
        t = t.replace("\\/", "/")
    return t


def is_http_url(s: str) -> bool:
    """True for remote video URLs. Do not pass these through :class:`pathlib.Path`.resolve()."""
    t = normalize_video_input(s)
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
    if name:
        return name
    return "video_download.mp4"


def _filename_from_content_disposition(value: str) -> str | None:
    """Parse ``filename`` / ``filename*`` from a Content-Disposition header."""
    m = re.search(r"filename\*=(?:UTF-8''|)([^;\s]+)", value, re.I)
    if m:
        return unquote(m.group(1).strip().strip('"'))
    m = re.search(r'filename="([^"]+)"', value)
    if m:
        return m.group(1)
    m = re.search(r"filename=([^;\s]+)", value)
    if m:
        return m.group(1).strip().strip('"')
    return None


def download_video(url: str, dest_dir: Path, *, max_bytes: int) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    out: Path | None = None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            name = _filename_from_url(url)
            cd = resp.headers.get("Content-Disposition")
            if cd:
                parsed = _filename_from_content_disposition(cd)
                if parsed:
                    safe = Path(parsed).name
                    if safe:
                        name = safe
            out = dest_dir / name
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
    if out is None or not out.exists() or out.stat().st_size == 0:
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
    s = normalize_video_input(str(path_or_url))
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
