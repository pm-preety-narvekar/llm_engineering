"""Fetch a public webpage and extract plain text for LLM context."""

from __future__ import annotations

import gzip
import os
import re
import urllib.error
import urllib.request
from html import unescape
from urllib.parse import urlparse

from video_analyzer.media import MediaError
from video_analyzer.remote import normalize_video_input

_MAX_HTML_BYTES = 2 * 1024 * 1024
_MAX_TEXT_CHARS = 14_000
# Only the leading fraction of HTML is parsed (default 30%): full pages are not passed to the LLM.
_ENV_FRACTION = "VIDEO_ANALYZER_WEBPAGE_CONTENT_FRACTION"


def _html_content_fraction() -> float:
    raw = os.environ.get(_ENV_FRACTION, "").strip()
    if not raw:
        return 0.3
    try:
        f = float(raw)
    except ValueError:
        return 0.3
    return max(0.05, min(1.0, f))


def _leading_fraction_of_html(html: str) -> str:
    """Keep only the first N% of HTML (by Unicode length) for extraction."""
    if not html:
        return html
    frac = _html_content_fraction()
    n = max(1, int(len(html) * frac))
    return html[:n]


def _truncate_utf8_bytes(s: str, max_bytes: int) -> str:
    """Keep the start of ``s`` so UTF-8 encoded length is at most ``max_bytes``."""
    b = s.encode("utf-8", errors="replace")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="replace")

_DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _browser_headers(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    ua = os.environ.get("VIDEO_ANALYZER_WEB_USER_AGENT", "").strip() or _DEFAULT_BROWSER_UA
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Referer": origin + "/",
    }


def _html_to_plain(html: str) -> str:
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", html)
    html = re.sub(r"<[^>]+>", " ", html)
    text = unescape(html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_TEXT_CHARS]


def _fetch_html_with_urllib(u: str) -> str:
    """Fallback when curl_cffi is unavailable; many CDNs return 403."""
    req = urllib.request.Request(u, headers=_browser_headers(u), method="GET")
    raw: bytes = b""
    ct = ""
    enc = ""
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read(_MAX_HTML_BYTES)
            ct = resp.headers.get("Content-Type", "")
            enc = resp.headers.get("Content-Encoding", "")
    except urllib.error.HTTPError as e:
        hint = ""
        if e.code in (403, 401):
            hint = (
                " Install dependencies so TLS/browser impersonation is used: pip install curl-cffi>=0.7"
            )
        raise MediaError(f"HTTP {e.code} fetching webpage: {u}.{hint}") from e
    except urllib.error.URLError as e:
        raise MediaError(f"Failed to fetch webpage: {e.reason!r}") from e
    if raw[:2] == b"\x1f\x8b" or (enc and "gzip" in enc.lower()):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass
    if len(raw) > _MAX_HTML_BYTES:
        raw = raw[:_MAX_HTML_BYTES]
    charset_m = re.search(r"charset=([\w.-]+)", ct, re.I)
    charset = charset_m.group(1) if charset_m else "utf-8"
    html = raw.decode(charset, errors="replace")
    return _truncate_utf8_bytes(html, _MAX_HTML_BYTES)


def _fetch_html_with_curl_cffi(u: str) -> str:
    """TLS + HTTP/2 fingerprint impersonation; bypasses many bot blocks that urllib hits."""
    from curl_cffi import requests as curl_requests

    imp = os.environ.get("VIDEO_ANALYZER_CURL_IMPERSONATE", "chrome120").strip() or "chrome120"
    try:
        r = curl_requests.get(
            u,
            impersonate=imp,
            timeout=60,
            allow_redirects=True,
            stream=True,
        )
    except Exception as e:
        raise MediaError(f"curl_cffi request failed for {u}: {e}") from e
    if r.status_code >= 400:
        raise MediaError(
            f"HTTP {r.status_code} fetching webpage: {u}. "
            "Try another browser profile: set VIDEO_ANALYZER_CURL_IMPERSONATE to e.g. chrome110, "
            "edge101, or safari17_0 (see curl_cffi impersonate options)."
        )
    raw = b""
    try:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            raw += chunk
            if len(raw) >= _MAX_HTML_BYTES:
                raw = raw[:_MAX_HTML_BYTES]
                break
    finally:
        r.close()
    ct = r.headers.get("Content-Type", "")
    charset_m = re.search(r"charset=([\w.-]+)", ct, re.I)
    charset = charset_m.group(1) if charset_m else "utf-8"
    html = raw.decode(charset, errors="replace")
    return _truncate_utf8_bytes(html, _MAX_HTML_BYTES)


def fetch_webpage_plain_text(url: str) -> str:
    u = normalize_video_input(url.strip())
    if not u.lower().startswith(("http://", "https://")):
        raise MediaError(f"Not an http(s) URL: {url!r}")

    try:
        html = _fetch_html_with_curl_cffi(u)
    except ImportError:
        html = _fetch_html_with_urllib(u)
    html = _leading_fraction_of_html(html)
    return _html_to_plain(html)
