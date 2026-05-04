#!/usr/bin/env python3
"""
WanGP API — model file size resolver with persistent cache.

Resolves the ``Content-Length`` of every URL in ``defaults/*.json`` by
issuing HTTP HEAD requests against HuggingFace (or wherever the URL
points), then caches the result on disk.

Cache file: ``~/.wan2gp/model_sizes.json`` (override via
``WAN2GP_SIZE_CACHE``).

Cache entry shape::

    {
      "<url>": {
        "bytes": 12345678,
        "etag": "\"deadbeef\"",
        "last_modified": "Wed, 04 May 2026 12:34:56 GMT",
        "fetched_at": "2026-05-04T12:35:00Z",
        "error": null   # populated only on failure (e.g. "404", "timeout")
      }
    }

Resolution policy:

  - Cached + recent (< ``CACHE_TTL_SECONDS``)         → return immediately.
  - Cached + stale, server returns matching ETag      → bump ``fetched_at``.
  - Cached + stale, ETag differs / no-cache returned  → record new size.
  - Uncached                                          → fetch, record.
  - HEAD fails                                        → record error, do
                                                        not block the caller;
                                                        retry on next access.

Concurrency: a small thread pool fetches in parallel; calls are
non-blocking from the API's perspective — the resolver returns "what we
know now" and dispatches a background refresh for stale or missing
entries.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CACHE_TTL_SECONDS = 7 * 24 * 3600  # 1 week
HEAD_TIMEOUT = 15  # seconds
MAX_PARALLEL_FETCHES = 6


def _cache_path() -> Path:
    env = os.environ.get("WAN2GP_SIZE_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".wan2gp" / "model_sizes.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SizeCache:
    """Thread-safe URL → size cache, persisted to JSON."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self._path = cache_path or _cache_path()
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_PARALLEL_FETCHES,
            thread_name_prefix="wan2gp-size",
        )
        self._inflight: dict[str, Future] = {}
        self._load()

    # ----- persistence -----

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                self._data = {k: v for k, v in payload.items() if isinstance(v, dict)}
        except Exception:
            self._data = {}

    def _persist_locked(self) -> None:
        if not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp, self._path)
            self._dirty = False
        except Exception:
            # Best-effort: a failure here doesn't break the API.
            pass

    def flush(self) -> None:
        with self._lock:
            self._persist_locked()

    # ----- public access -----

    def get(self, url: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data.get(url)
            return dict(entry) if entry else None

    def get_many(self, urls: list[str]) -> dict[str, dict[str, Any] | None]:
        with self._lock:
            return {u: (dict(self._data[u]) if u in self._data else None) for u in urls}

    def is_stale(self, entry: dict[str, Any] | None) -> bool:
        if not entry:
            return True
        if entry.get("error"):
            # Retry failed entries after an hour, not a week.
            try:
                fetched = datetime.strptime(
                    entry.get("fetched_at", ""), "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                return True
            return (datetime.now(timezone.utc) - fetched).total_seconds() > 3600
        try:
            fetched = datetime.strptime(
                entry.get("fetched_at", ""), "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except Exception:
            return True
        return (datetime.now(timezone.utc) - fetched).total_seconds() > CACHE_TTL_SECONDS

    def request_refresh(self, urls: list[str]) -> None:
        """Submit background HEAD requests for any stale or missing URLs."""
        with self._lock:
            for url in urls:
                if not url or not url.startswith("http"):
                    continue
                if url in self._inflight and not self._inflight[url].done():
                    continue
                if not self.is_stale(self._data.get(url)):
                    continue
                self._inflight[url] = self._executor.submit(self._fetch_and_store, url)

    def block_until_known(self, urls: list[str], *, timeout: float = 10.0) -> None:
        """Best-effort: wait briefly for inflight HEADs to land. Used by the
        first call to ``/api/models`` so the response includes sizes when
        they're cheap to obtain."""
        self.request_refresh(urls)
        deadline = time.time() + timeout
        with self._lock:
            futures = [f for u, f in self._inflight.items() if u in urls and not f.done()]
        for fut in futures:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                fut.result(timeout=remaining)
            except Exception:
                pass

    # ----- fetch -----

    def _fetch_and_store(self, url: str) -> dict[str, Any]:
        existing = self.get(url) or {}
        result = self._head(url, existing)
        with self._lock:
            self._data[url] = result
            self._dirty = True
            self._inflight.pop(url, None)
            self._persist_locked()
        return result

    def _head(self, url: str, existing: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "wan2gp-api/1.0")
        if existing.get("etag"):
            req.add_header("If-None-Match", existing["etag"])
        try:
            with urllib.request.urlopen(req, timeout=HEAD_TIMEOUT) as resp:
                length = resp.headers.get("Content-Length")
                etag = resp.headers.get("ETag")
                last_modified = resp.headers.get("Last-Modified")
                # HF's CDN issues 302 → resolved file. urlopen follows by default.
                bytes_val = int(length) if length and length.isdigit() else existing.get("bytes")
                return {
                    "bytes": bytes_val,
                    "etag": etag or existing.get("etag"),
                    "last_modified": last_modified or existing.get("last_modified"),
                    "fetched_at": _now_iso(),
                    "error": None,
                }
        except urllib.error.HTTPError as exc:
            if exc.code == 304 and existing.get("bytes"):
                # Not modified — keep the size, refresh the timestamp.
                return {
                    **existing,
                    "fetched_at": _now_iso(),
                    "error": None,
                }
            return {
                **existing,
                "fetched_at": _now_iso(),
                "error": f"http {exc.code}",
            }
        except Exception as exc:
            return {
                **existing,
                "fetched_at": _now_iso(),
                "error": str(exc)[:200],
            }


_GLOBAL_CACHE: SizeCache | None = None
_GLOBAL_LOCK = threading.Lock()


def get_cache() -> SizeCache:
    global _GLOBAL_CACHE
    with _GLOBAL_LOCK:
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = SizeCache()
        return _GLOBAL_CACHE


def resolve_sizes(urls: list[str], *, wait_seconds: float = 0.0) -> dict[str, dict[str, Any] | None]:
    """Look up sizes for a list of URLs. Triggers a background refresh for
    any stale entries. If ``wait_seconds`` > 0, blocks briefly so first-call
    responses include freshly-fetched data instead of nulls.
    """
    cache = get_cache()
    cache.request_refresh(urls)
    if wait_seconds > 0:
        cache.block_until_known(urls, timeout=wait_seconds)
    return cache.get_many(urls)


def total_bytes(urls: list[str], lookup: dict[str, dict[str, Any] | None]) -> int | None:
    """Sum of known sizes for ``urls`` (only the *first* URL — variants are
    alternatives, not additive). Returns ``None`` if size unknown."""
    if not urls:
        return None
    primary = urls[0]
    entry = lookup.get(primary)
    if entry and entry.get("bytes"):
        return int(entry["bytes"])
    return None


if __name__ == "__main__":
    # CLI: python agent_api_sizes.py <url> [<url> ...]
    import sys
    cache = get_cache()
    urls = sys.argv[1:]
    if not urls:
        print(json.dumps(cache._data, indent=2, default=str))
        sys.exit(0)
    cache.block_until_known(urls, timeout=20.0)
    out = {u: cache.get(u) for u in urls}
    print(json.dumps(out, indent=2, default=str))
