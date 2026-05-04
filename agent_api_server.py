#!/usr/bin/env python3
"""
WanGP Agent API Server (hardened)
=================================

Async-job HTTP layer in front of the in-process WanGP runtime. Provides:

  - POST   /api/jobs              — create job (returns immediately)
  - GET    /api/jobs              — list recent jobs
  - GET    /api/jobs/:id          — status + result
  - DELETE /api/jobs/:id          — cancel
  - GET    /api/jobs/:id/events   — SSE stream of progress updates
  - GET    /api/health            — rich health (gpu, queue, version)
  - GET    /api/models            — model_type entries with capability hints
  - GET    /api/loras             — loras for a model_type
  - GET    /api/settings          — default settings template
  - POST   /api/release           — release VRAM
  - GET    /api/file              — download (constrained to outputs root)
  - POST   /api/generate          — DEPRECATED back-compat (sync wrap of /api/jobs)
  - POST   /api/batch             — DEPRECATED back-compat (sync, single batch task)

Configuration (env vars):

  WAN2GP_TOKEN          — bearer token. Unset = no auth (LAN trust mode).
  WAN2GP_OUTPUTS_ROOT   — outputs root for /api/file (default: <repo>/outputs).
  WAN2GP_JOB_DB         — SQLite job log path (default: ~/.wan2gp/jobs.sqlite).
  WAN2GP_LOG_PROMPTS    — "1" to include prompt text in logs (default: off).
  WAN2GP_JOB_HISTORY    — number of jobs to retain in DB (default: 200).
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any


WANGP_ROOT = Path(__file__).resolve().parent
SERVER_VERSION = "0.4.0"

# Auto-derived schema + size cache live in their own modules so this file
# stays focused on HTTP plumbing.
import agent_api_introspect
import agent_api_sizes


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_LOCK = threading.Lock()
_LOG_PROMPTS = os.environ.get("WAN2GP_LOG_PROMPTS") == "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    """Emit a single JSON log line to stdout."""
    record = {"ts": _now_iso(), "level": level, "event": event}
    for key, value in fields.items():
        if value is None:
            continue
        record[key] = value
    line = json.dumps(record, default=str)
    with _LOG_LOCK:
        print(line, flush=True)


def _redact_request(req: dict[str, Any]) -> dict[str, Any]:
    """Remove prompt/text fields from a logged request unless WAN2GP_LOG_PROMPTS is set."""
    if _LOG_PROMPTS:
        return req
    redacted = {}
    for key, value in req.items():
        if key in ("prompt", "negative_prompt") and isinstance(value, str) and value:
            redacted[key] = f"<{len(value)} chars redacted>"
        else:
            redacted[key] = value
    return redacted


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_LOCK = threading.Lock()
_ULID_LAST_MS = 0
_ULID_LAST_RAND = 0


def make_job_id() -> str:
    """Generate a sortable ULID-style id with a `j_` prefix.

    Format: j_<10 chars timestamp ms><16 chars randomness> in Crockford base32.
    """
    global _ULID_LAST_MS, _ULID_LAST_RAND
    with _ULID_LOCK:
        ms = int(time.time() * 1000)
        if ms <= _ULID_LAST_MS:
            ms = _ULID_LAST_MS
            _ULID_LAST_RAND += 1
            rand_int = _ULID_LAST_RAND
        else:
            _ULID_LAST_MS = ms
            rand_int = int.from_bytes(secrets.token_bytes(10), "big")
            _ULID_LAST_RAND = rand_int
    ts_part = ""
    n = ms
    for _ in range(10):
        ts_part = _ULID_ALPHABET[n & 0x1F] + ts_part
        n >>= 5
    rand_part = ""
    n = rand_int & ((1 << 80) - 1)
    for _ in range(16):
        rand_part = _ULID_ALPHABET[n & 0x1F] + rand_part
        n >>= 5
    return f"j_{ts_part}{rand_part}"


def make_request_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Capability metadata (auto-derived from family handler feature flags)
# ---------------------------------------------------------------------------

def capability_for_model_type(model_type: str) -> str:
    """Best-effort capability label for a model_type string.

    Pulls from the introspect index (handler-derived feature flags). Falls
    back to a name-based heuristic if the index has no entry for the model.
    """
    try:
        entry = agent_api_introspect.get_model_entry(model_type)
        if entry is not None:
            return entry["capability"]
    except Exception:
        pass
    mt = model_type.lower()
    if any(tag in mt for tag in ("t2v", "i2v", "v2v", "video", "hunyuan", "ltx", "longcat", "magi")):
        return "video-generation"
    if any(tag in mt for tag in ("tts", "audio", "ace_step", "heartmula", "kugel", "chatterbox")):
        return "audio-generation"
    return "image-generation"


# ---------------------------------------------------------------------------
# Job model + store
# ---------------------------------------------------------------------------

JOB_STATUSES = ("queued", "running", "completed", "failed", "cancelled", "cancelling")
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


@dataclass
class JobRecord:
    job_id: str
    capability: str
    model_type: str
    status: str = "queued"
    progress: float = 0.0
    step: int = 0
    total_steps: int = 0
    queue_position: int | None = None
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    request: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    error: str | None = None
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    """In-memory job records with SQLite-backed persistence of the last N jobs."""

    def __init__(self, db_path: Path, history_limit: int = 200):
        self._db_path = db_path
        self._history_limit = history_limit
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []  # creation order
        self._lock = threading.RLock()
        self._listeners: dict[str, list[queue.Queue]] = {}
        self._init_db()
        self._load_existing()

    # --- persistence ---

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs(created_at)")
            conn.commit()

    def _load_existing(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute("SELECT job_id, record_json FROM jobs ORDER BY created_at ASC")
            for job_id, record_json in cur.fetchall():
                try:
                    payload = json.loads(record_json)
                    rec = JobRecord(**payload)
                except Exception:
                    continue
                # Any non-terminal job left in the DB is from a prior server
                # process that died. Mark it failed so callers don't poll forever.
                if rec.status not in TERMINAL_STATUSES:
                    rec.status = "failed"
                    rec.error = "server restarted"
                    rec.completed_at = rec.completed_at or _now_iso()
                self._jobs[job_id] = rec
                self._order.append(job_id)
        self._trim_locked()

    def _persist_locked(self, rec: JobRecord) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO jobs(job_id, record_json, created_at) VALUES (?, ?, ?)",
                    (rec.job_id, json.dumps(rec.to_dict()), rec.created_at),
                )
                conn.commit()
        except Exception as exc:
            log_event("job_persist_failed", level="warn", job_id=rec.job_id, error=str(exc))

    def _trim_locked(self) -> None:
        if len(self._order) <= self._history_limit:
            return
        excess = len(self._order) - self._history_limit
        # Only evict terminal jobs (never an in-flight one).
        evicted: list[str] = []
        for job_id in list(self._order):
            if len(evicted) >= excess:
                break
            rec = self._jobs.get(job_id)
            if rec and rec.status in TERMINAL_STATUSES:
                evicted.append(job_id)
        for job_id in evicted:
            self._order.remove(job_id)
            self._jobs.pop(job_id, None)
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                    conn.commit()
            except Exception:
                pass

    # --- public api ---

    def create(self, *, request: dict[str, Any], request_id: str) -> JobRecord:
        model_type = str(request.get("model_type") or "")
        rec = JobRecord(
            job_id=make_job_id(),
            capability=capability_for_model_type(model_type),
            model_type=model_type,
            status="queued",
            created_at=_now_iso(),
            request=request,
            request_id=request_id,
        )
        with self._lock:
            self._jobs[rec.job_id] = rec
            self._order.append(rec.job_id)
            self._persist_locked(rec)
            self._trim_locked()
        self._broadcast(rec.job_id, "status", rec.to_dict())
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[JobRecord]:
        with self._lock:
            ordered = list(reversed(self._order))[:limit]
            return [self._jobs[j] for j in ordered if j in self._jobs]

    def update(self, job_id: str, **fields: Any) -> JobRecord | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            for key, value in fields.items():
                if hasattr(rec, key):
                    setattr(rec, key, value)
            self._persist_locked(rec)
        self._broadcast(job_id, "status", rec.to_dict())
        return rec

    def queued_jobs(self) -> list[JobRecord]:
        with self._lock:
            return [self._jobs[j] for j in self._order if self._jobs[j].status == "queued"]

    def running_job(self) -> JobRecord | None:
        with self._lock:
            for j in self._order:
                rec = self._jobs[j]
                if rec.status in ("running", "cancelling"):
                    return rec
        return None

    def assign_queue_positions(self) -> None:
        """Recompute queue_position for all queued jobs (1 = next)."""
        with self._lock:
            queued = [j for j in self._order if self._jobs[j].status == "queued"]
            for idx, job_id in enumerate(queued, start=1):
                rec = self._jobs[job_id]
                if rec.queue_position != idx:
                    rec.queue_position = idx
                    self._persist_locked(rec)

    # --- listeners (SSE) ---

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._listeners.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            listeners = self._listeners.get(job_id) or []
            if q in listeners:
                listeners.remove(q)
            if not listeners and job_id in self._listeners:
                del self._listeners[job_id]

    def _broadcast(self, job_id: str, kind: str, data: Any) -> None:
        with self._lock:
            listeners = list(self._listeners.get(job_id, ()))
        for q in listeners:
            try:
                q.put_nowait({"kind": kind, "data": data})
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class JobWorker:
    """Single background thread that runs queued jobs serially via the WanGP session."""

    def __init__(self, agent: Any, store: JobStore):
        self._agent = agent
        self._store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._current_job_id: str | None = None
        self._current_session_job: Any = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="wan2gp-job-worker")
        self._thread.start()

    @property
    def current_job_id(self) -> str | None:
        with self._lock:
            return self._current_job_id

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)
        self._store.assign_queue_positions()

    def request_cancel(self, job_id: str) -> str:
        """Request cancellation. Returns the resulting status hint."""
        rec = self._store.get(job_id)
        if rec is None:
            return "not_found"
        if rec.status in TERMINAL_STATUSES:
            return rec.status
        if rec.status == "queued":
            self._store.update(
                job_id,
                status="cancelled",
                completed_at=_now_iso(),
                queue_position=None,
            )
            self._store.assign_queue_positions()
            log_event("job_cancelled", job_id=job_id, request_id=rec.request_id, reason="queued")
            return "cancelled"
        # running or cancelling: signal worker
        with self._lock:
            session_job = self._current_session_job if self._current_job_id == job_id else None
        self._store.update(job_id, status="cancelling")
        if session_job is not None:
            try:
                session_job.cancel()
            except Exception as exc:
                log_event("job_cancel_signal_failed", level="warn", job_id=job_id, error=str(exc))
        return "cancelling"

    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            rec = self._store.get(job_id)
            if rec is None or rec.status != "queued":
                continue
            self._run_one(rec)

    def _run_one(self, rec: JobRecord) -> None:
        started = time.time()
        with self._lock:
            self._current_job_id = rec.job_id
            self._current_session_job = None
        self._store.update(
            rec.job_id,
            status="running",
            started_at=_now_iso(),
            queue_position=None,
        )
        self._store.assign_queue_positions()
        log_event(
            "job_started",
            job_id=rec.job_id,
            request_id=rec.request_id,
            capability=rec.capability,
            model_type=rec.model_type,
        )

        try:
            self._agent._ensure_session()
            session = self._agent._session
            session_job = session.submit_task(rec.request)
            with self._lock:
                self._current_session_job = session_job

            last_persist = 0.0
            for event in session_job.events.iter(timeout=0.5):
                # Detect external cancel request
                rec_now = self._store.get(rec.job_id)
                if rec_now and rec_now.status == "cancelling":
                    try:
                        session_job.cancel()
                    except Exception:
                        pass

                if event.kind == "progress":
                    p = event.data
                    progress_int = getattr(p, "progress", 0) or 0
                    progress = progress_int / 100.0 if progress_int > 1 else float(progress_int)
                    step = getattr(p, "current_step", None) or 0
                    total = getattr(p, "total_steps", None) or 0
                    now = time.time()
                    # Throttle DB writes; broadcast every event
                    if now - last_persist >= 0.5 or step == total:
                        self._store.update(
                            rec.job_id,
                            progress=round(progress, 4),
                            step=step,
                            total_steps=total,
                        )
                        last_persist = now
                    else:
                        # local mutate without broadcast/persist storm
                        live = self._store.get(rec.job_id)
                        if live is not None:
                            live.progress = round(progress, 4)
                            live.step = step
                            live.total_steps = total

            result = session_job.result()
            duration = round(time.time() - started, 2)
            files = list(result.generated_files or [])

            if result.cancelled:
                self._store.update(
                    rec.job_id,
                    status="cancelled",
                    completed_at=_now_iso(),
                    duration_seconds=duration,
                    files=files,
                )
                log_event("job_cancelled", job_id=rec.job_id, request_id=rec.request_id, duration_ms=int(duration * 1000))
                self._release_after_cancel()
            elif result.success:
                self._store.update(
                    rec.job_id,
                    status="completed",
                    completed_at=_now_iso(),
                    duration_seconds=duration,
                    progress=1.0,
                    files=files,
                )
                log_event(
                    "job_completed",
                    job_id=rec.job_id,
                    request_id=rec.request_id,
                    duration_ms=int(duration * 1000),
                    files=len(files),
                )
            else:
                err = "; ".join(str(e.message) for e in (result.errors or [])) or "generation failed"
                self._store.update(
                    rec.job_id,
                    status="failed",
                    completed_at=_now_iso(),
                    duration_seconds=duration,
                    files=files,
                    error=err,
                )
                log_event(
                    "job_failed",
                    level="error",
                    job_id=rec.job_id,
                    request_id=rec.request_id,
                    error=err,
                )
        except Exception as exc:
            duration = round(time.time() - started, 2)
            tb = traceback.format_exc(limit=4)
            self._store.update(
                rec.job_id,
                status="failed",
                completed_at=_now_iso(),
                duration_seconds=duration,
                error=f"{exc}",
            )
            log_event(
                "job_failed",
                level="error",
                job_id=rec.job_id,
                request_id=rec.request_id,
                error=str(exc),
                traceback=tb,
            )
        finally:
            with self._lock:
                self._current_job_id = None
                self._current_session_job = None

    def _release_after_cancel(self) -> None:
        try:
            self._agent.release_model()
            log_event("vram_released", reason="post_cancel")
        except Exception as exc:
            log_event("vram_release_failed", level="warn", error=str(exc))


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def _gpu_info() -> dict[str, Any] | None:
    """Return GPU info using torch (already imported by WanGP) or nvidia-smi fallback."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            idx = 0
            name = torch.cuda.get_device_name(idx)
            free, total = torch.cuda.mem_get_info(idx)
            used = total - free
            return {
                "name": name,
                "vram_total_mb": int(total // (1024 * 1024)),
                "vram_used_mb": int(used // (1024 * 1024)),
            }
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        if out:
            first = out.splitlines()[0]
            name, total, used = [s.strip() for s in first.split(",")]
            return {
                "name": name,
                "vram_total_mb": int(total),
                "vram_used_mb": int(used),
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# /api/file path constraint
# ---------------------------------------------------------------------------

def _outputs_root() -> Path:
    env = os.environ.get("WAN2GP_OUTPUTS_ROOT")
    if env:
        return Path(env).resolve()
    return (WANGP_ROOT / "outputs").resolve()


def _path_inside(child_path: Path, parent: Path) -> bool:
    try:
        child_real = Path(os.path.realpath(child_path))
        parent_real = Path(os.path.realpath(parent))
    except Exception:
        return False
    try:
        child_real.relative_to(parent_real)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

CORS_EXPOSE_HEADERS = "Deprecation, Link, X-Request-Id"
CORS_ALLOW_HEADERS = "Authorization, Content-Type, X-Request-Id"
CORS_ALLOW_METHODS = "GET, POST, DELETE, OPTIONS"
CORS_MAX_AGE = "600"


def _parse_cors_origins(raw: str | None) -> tuple[set[str], bool]:
    """Return (exact_origins, wildcard). ``raw`` is comma-separated."""
    if not raw:
        return set(), False
    origins = {o.strip() for o in raw.split(",") if o.strip()}
    wildcard = "*" in origins
    origins.discard("*")
    return origins, wildcard


def _cors_origin_for(request_origin: str, allowed: set[str], wildcard: bool) -> str | None:
    """Return the value to echo in Access-Control-Allow-Origin, or None to skip CORS."""
    if not request_origin:
        return None
    if request_origin in allowed:
        return request_origin
    if wildcard:
        # We never use credentialed CORS, so echoing back "*" is fine and lets
        # any origin call us when the operator opted in via WAN2GP_CORS_ORIGINS=*.
        return "*"
    return None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _build_handler(*, agent: Any, store: JobStore, worker: JobWorker, token: str | None,
                   started_at: float, cors_allowed: set[str], cors_wildcard: bool,
                   ) -> type[BaseHTTPRequestHandler]:

    outputs_root = _outputs_root()
    cors_enabled = bool(cors_allowed) or cors_wildcard
    # First /api/models call blocks briefly to populate size cache; later
    # calls just read from the warm cache.
    _api_models_warmed = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # Suppress BaseHTTPServer's default access log; we emit JSON ourselves.
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        # ---------- middleware ----------

        def _cors_headers(self) -> dict[str, str]:
            """Headers to add to every response when CORS is enabled and the origin matches."""
            if not cors_enabled:
                return {}
            origin = self.headers.get("Origin", "")
            allow = _cors_origin_for(origin, cors_allowed, cors_wildcard)
            if allow is None:
                # Still send Vary so caches don't conflate origins.
                return {"Vary": "Origin"}
            return {
                "Access-Control-Allow-Origin": allow,
                "Vary": "Origin",
                "Access-Control-Expose-Headers": CORS_EXPOSE_HEADERS,
            }

        def _send_cors_preflight(self) -> int:
            origin = self.headers.get("Origin", "")
            allow = _cors_origin_for(origin, cors_allowed, cors_wildcard) if cors_enabled else None
            if allow is None:
                # No CORS or origin not allowed — return 403 so the browser surfaces it.
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return 403
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", CORS_ALLOW_METHODS)
            self.send_header("Access-Control-Allow-Headers", CORS_ALLOW_HEADERS)
            self.send_header("Access-Control-Expose-Headers", CORS_EXPOSE_HEADERS)
            self.send_header("Access-Control-Max-Age", CORS_MAX_AGE)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return 204

        def _check_auth(self) -> bool:
            if not token:
                return True
            header = self.headers.get("Authorization", "")
            if header == f"Bearer {token}":
                return True
            self._json(401, {"error": "unauthorized"})
            return False

        def _request_id(self) -> str:
            rid = self.headers.get("X-Request-Id", "").strip()
            if not rid or len(rid) > 128:
                rid = make_request_id()
            return rid

        def _path_parts(self) -> tuple[str, dict[str, str]]:
            parsed = urllib.parse.urlsplit(self.path)
            return parsed.path, dict(urllib.parse.parse_qsl(parsed.query))

        def _read_body(self) -> Any:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(n))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON body: {exc}") from exc

        def _json(self, code: int, payload: Any, *, extra_headers: dict[str, str] | None = None) -> None:
            raw = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for key, value in self._cors_headers().items():
                self.send_header(key, value)
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ---------- dispatch ----------

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch("DELETE")

        def do_OPTIONS(self) -> None:  # noqa: N802
            request_id = self._request_id()
            self._current_request_id = request_id
            t_start = time.time()
            status_code = self._send_cors_preflight()
            log_event(
                "request",
                request_id=request_id,
                method="OPTIONS",
                path=self._path_parts()[0],
                status=status_code,
                duration_ms=int((time.time() - t_start) * 1000),
            )

        def _dispatch(self, method: str) -> None:
            path, qs = self._path_parts()
            request_id = self._request_id()
            self._current_request_id = request_id
            t_start = time.time()
            status_code = 200
            try:
                # Health is always allowed without auth.
                if path == "/api/health" and method == "GET":
                    self._handle_health()
                    return
                if not self._check_auth():
                    status_code = 401
                    return

                if method == "GET":
                    status_code = self._route_get(path, qs)
                elif method == "POST":
                    status_code = self._route_post(path, qs)
                elif method == "DELETE":
                    status_code = self._route_delete(path)
                else:
                    self._json(405, {"error": "method not allowed"})
                    status_code = 405
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                status_code = 400
            except Exception as exc:
                tb = traceback.format_exc(limit=4)
                log_event("request_error", level="error", request_id=request_id,
                          path=path, method=method, error=str(exc), traceback=tb)
                self._json(500, {"error": "internal server error"})
                status_code = 500
            finally:
                duration_ms = int((time.time() - t_start) * 1000)
                # Skip noisy logs for SSE streams that we already logged on entry.
                if path != "/api/health" or status_code >= 400:
                    log_event(
                        "request",
                        request_id=request_id,
                        method=method,
                        path=path,
                        status=status_code,
                        duration_ms=duration_ms,
                    )

        # ---------- GET routes ----------

        def _route_get(self, path: str, qs: dict[str, str]) -> int:
            if path == "/api/models":
                # First call to /api/models block briefly so size_bytes is
                # populated on the cold-cache path. After that the size cache
                # is warm and resolves instantly.
                wait = 4.0 if not _api_models_warmed.is_set() else 0.0
                self._json(200, _models_payload(wait_for_sizes=wait))
                _api_models_warmed.set()
                return 200
            if path.startswith("/api/models/"):
                model_type = path[len("/api/models/"):]
                if not model_type:
                    self._json(404, {"error": "not found"})
                    return 404
                detail = _model_detail_payload(model_type)
                if detail is None:
                    self._json(404, {"error": f"unknown model_type: {model_type}"})
                    return 404
                self._json(200, detail)
                return 200
            if path == "/api/loras":
                self._json(200, agent.list_loras(qs.get("model_type", "z_image")))
                return 200
            if path == "/api/settings":
                self._json(200, agent.get_default_settings())
                return 200
            if path == "/api/settings/schema":
                self._json(200, _settings_schema_payload())
                return 200
            if path == "/api/file":
                return self._serve_file(qs)
            if path == "/api/jobs":
                limit = int(qs.get("limit", "50"))
                jobs = [r.to_dict() for r in store.list(limit=limit)]
                self._json(200, {"jobs": jobs})
                return 200
            if path.startswith("/api/jobs/"):
                tail = path[len("/api/jobs/"):]
                if tail.endswith("/events"):
                    return self._stream_events(tail[:-len("/events")])
                rec = store.get(tail)
                if rec is None:
                    self._json(404, {"error": "job not found"})
                    return 404
                self._json(200, rec.to_dict())
                return 200
            self._json(404, {"error": "not found"})
            return 404

        # ---------- POST routes ----------

        def _route_post(self, path: str, qs: dict[str, str]) -> int:
            if path == "/api/jobs":
                body = self._read_body()
                if not isinstance(body, dict) or not body.get("model_type"):
                    self._json(400, {"error": "request must be a JSON object with model_type"})
                    return 400
                # Validate against the auto-derived schema before queueing.
                err = agent_api_introspect.validate_request(body["model_type"], body)
                if err:
                    self._json(400, {"error": err, "model_type": body["model_type"]})
                    return 400
                rec = store.create(request=body, request_id=self._current_request_id)
                worker.submit(rec.job_id)
                log_event(
                    "job_created",
                    job_id=rec.job_id,
                    request_id=rec.request_id,
                    capability=rec.capability,
                    model_type=rec.model_type,
                    request=_redact_request(body),
                )
                store.assign_queue_positions()
                self._json(202, store.get(rec.job_id).to_dict())
                return 202
            if path == "/api/settings/validate":
                body = self._read_body()
                if not isinstance(body, dict) or not body.get("model_type"):
                    self._json(400, {"error": "request must be a JSON object with model_type"})
                    return 400
                err = agent_api_introspect.validate_request(body["model_type"], body)
                if err:
                    self._json(200, {"valid": False, "error": err})
                    return 200
                self._json(200, {"valid": True})
                return 200
            if path == "/api/release":
                agent.release_model()
                self._json(200, {"ok": True})
                return 200
            if path == "/api/generate":
                return self._legacy_generate()
            if path == "/api/batch":
                return self._legacy_batch()
            self._json(404, {"error": "not found"})
            return 404

        def _route_delete(self, path: str) -> int:
            if path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/"):]
                if not job_id:
                    self._json(404, {"error": "not found"})
                    return 404
                rec = store.get(job_id)
                if rec is None:
                    self._json(404, {"error": "job not found"})
                    return 404
                if rec.status in TERMINAL_STATUSES:
                    self._json(409, {"error": f"job is {rec.status}", "status": rec.status})
                    return 409
                outcome = worker.request_cancel(job_id)
                code = 200 if outcome == "cancelled" else 202
                self._json(code, store.get(job_id).to_dict())
                return code
            self._json(404, {"error": "not found"})
            return 404

        # ---------- specific handlers ----------

        def _handle_health(self) -> None:
            running = store.running_job()
            queued = len(store.queued_jobs())
            gpu = _gpu_info()
            payload: dict[str, Any] = {
                "status": "ok",
                "version": SERVER_VERSION,
                "uptime_seconds": int(time.time() - started_at),
                "queue": {
                    "running": 1 if running is not None else 0,
                    "queued": queued,
                },
                "current_job_id": worker.current_job_id,
            }
            if gpu is not None:
                payload["gpu"] = gpu
            else:
                payload["status"] = "degraded"
                payload["reason"] = "gpu unreachable"
                self._json(503, payload)
                return
            self._json(200, payload)

        def _serve_file(self, qs: dict[str, str]) -> int:
            fp = qs.get("path", "")
            if not fp:
                self._json(400, {"error": "missing path"})
                return 400
            try:
                requested = Path(fp)
                real = Path(os.path.realpath(requested))
            except Exception:
                self._json(403, {"error": "access denied"})
                return 403
            if not _path_inside(real, outputs_root):
                # Do NOT echo path back — avoids leaking layout.
                self._json(403, {"error": "access denied"})
                return 403
            if not real.is_file():
                self._json(404, {"error": "file not found"})
                return 404
            try:
                size = real.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{real.name}"')
                for key, value in self._cors_headers().items():
                    self.send_header(key, value)
                self.end_headers()
                with open(real, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return 200

        def _stream_events(self, job_id: str) -> int:
            rec = store.get(job_id)
            if rec is None:
                self._json(404, {"error": "job not found"})
                return 404
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            for key, value in self._cors_headers().items():
                self.send_header(key, value)
            self.end_headers()
            sub = store.subscribe(job_id)

            def write_event(kind: str, data: Any) -> bool:
                try:
                    self.wfile.write(f"event: {kind}\n".encode())
                    self.wfile.write(b"data: " + json.dumps(data, default=str).encode() + b"\n\n")
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    return False

            try:
                # Send initial snapshot
                if not write_event("status", store.get(job_id).to_dict()):
                    return 200
                if rec.status in TERMINAL_STATUSES:
                    return 200
                while True:
                    try:
                        evt = sub.get(timeout=15.0)
                    except queue.Empty:
                        # heartbeat
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return 200
                        continue
                    if not write_event(evt["kind"], evt["data"]):
                        return 200
                    data = evt.get("data") or {}
                    status = data.get("status") if isinstance(data, dict) else None
                    if status in TERMINAL_STATUSES:
                        return 200
            finally:
                store.unsubscribe(job_id, sub)

        # ---------- back-compat ----------

        def _legacy_headers(self) -> dict[str, str]:
            return {
                "Deprecation": "true",
                "Link": '</api/jobs>; rel="successor-version"',
            }

        def _legacy_generate(self) -> int:
            body = self._read_body()
            if not isinstance(body, dict) or not body.get("model_type"):
                self._json(400, {"error": "request must include model_type"},
                           extra_headers=self._legacy_headers())
                return 400
            err = agent_api_introspect.validate_request(body["model_type"], body)
            if err:
                self._json(400, {"error": err, "model_type": body["model_type"]},
                           extra_headers=self._legacy_headers())
                return 400
            rec = store.create(request=body, request_id=self._current_request_id)
            worker.submit(rec.job_id)
            log_event(
                "job_created",
                job_id=rec.job_id,
                request_id=rec.request_id,
                capability=rec.capability,
                model_type=rec.model_type,
                via="legacy_generate",
                request=_redact_request(body),
            )
            self._wait_for_terminal(rec.job_id)
            final = store.get(rec.job_id)
            payload = _legacy_payload(final)
            self._json(200, payload, extra_headers=self._legacy_headers())
            return 200

        def _legacy_batch(self) -> int:
            body = self._read_body()
            if not isinstance(body, list) or not body:
                self._json(400, {"error": "request must be a non-empty JSON array"},
                           extra_headers=self._legacy_headers())
                return 400
            # Submit each entry as its own job, run sequentially via the same queue.
            job_ids: list[str] = []
            for item in body:
                if not isinstance(item, dict) or not item.get("model_type"):
                    self._json(400, {"error": "each batch item must include model_type"},
                               extra_headers=self._legacy_headers())
                    return 400
                err = agent_api_introspect.validate_request(item["model_type"], item)
                if err:
                    self._json(400, {"error": err, "model_type": item["model_type"]},
                               extra_headers=self._legacy_headers())
                    return 400
                rec = store.create(request=item, request_id=self._current_request_id)
                worker.submit(rec.job_id)
                job_ids.append(rec.job_id)
            for jid in job_ids:
                self._wait_for_terminal(jid)
            files: list[str] = []
            errors: list[str] = []
            successful = 0
            duration = 0.0
            for jid in job_ids:
                rec = store.get(jid)
                if rec is None:
                    continue
                files.extend(rec.files or [])
                if rec.status == "completed":
                    successful += 1
                elif rec.error:
                    errors.append(rec.error)
                if rec.duration_seconds:
                    duration += rec.duration_seconds
            payload = {
                "success": successful == len(job_ids),
                "files": files,
                "errors": errors,
                "total_tasks": len(job_ids),
                "successful_tasks": successful,
                "failed_tasks": len(job_ids) - successful,
                "duration_seconds": round(duration, 2),
            }
            self._json(200, payload, extra_headers=self._legacy_headers())
            return 200

        def _wait_for_terminal(self, job_id: str, timeout: float | None = None) -> None:
            deadline = None if timeout is None else time.time() + timeout
            while True:
                rec = store.get(job_id)
                if rec and rec.status in TERMINAL_STATUSES:
                    return
                if deadline is not None and time.time() >= deadline:
                    return
                time.sleep(0.5)

    return _Handler


def _legacy_payload(rec: JobRecord | None) -> dict[str, Any]:
    if rec is None:
        return {
            "success": False,
            "files": [],
            "errors": ["job not found"],
            "total_tasks": 1,
            "successful_tasks": 0,
            "failed_tasks": 1,
            "duration_seconds": 0,
        }
    success = rec.status == "completed"
    errors = [rec.error] if rec.error else []
    return {
        "success": success,
        "files": rec.files,
        "errors": errors,
        "total_tasks": 1,
        "successful_tasks": 1 if success else 0,
        "failed_tasks": 0 if success else 1,
        "duration_seconds": rec.duration_seconds or 0,
        "job_id": rec.job_id,
        "status": rec.status,
    }


# ---------------------------------------------------------------------------
# Model listing — auto-derived from defaults/*.json + family handler flags
# ---------------------------------------------------------------------------

def _summary_entry(entry: dict[str, Any], size_lookup: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Compact per-model entry for the /api/models list response."""
    defaults = entry.get("defaults") or {}
    primary_url = entry["urls"][0] if entry["urls"] else None
    size_info = size_lookup.get(primary_url) if primary_url else None
    return {
        "model_type": entry["model_type"],
        "architecture": entry["architecture"],
        "family": entry["family"],
        "capability": entry["capability"],
        "name": entry["name"],
        "description": entry["description"],
        "param_count_b": entry["param_count_b"],
        "default_resolution": defaults.get("resolution"),
        "default_steps": defaults.get("num_inference_steps"),
        "default_video_length": defaults.get("video_length"),
        "size_bytes": (size_info or {}).get("bytes") if size_info else None,
        "size_status": (size_info or {}).get("error") if size_info else "pending",
        "quant_variants": entry["quant_variants"],
        "applicable_settings_count": len(entry.get("applicable_settings") or []),
        "url_count": len(entry["urls"]),
    }


def _families_legacy(index: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Backwards-compat ``families`` map: family_name → [model_type, ...]."""
    families: dict[str, list[str]] = {}
    for mt, entry in index.items():
        families.setdefault(entry["family"], []).append(mt)
    return families


def _models_payload(*, wait_for_sizes: float = 0.0) -> dict[str, Any]:
    """Build the /api/models response, with a brief HEAD-cache warm-up on
    first call so size_bytes is populated when cheap to obtain."""
    index_data = agent_api_introspect.build_index()
    index = index_data["models"]
    all_primary_urls = [e["urls"][0] for e in index.values() if e["urls"]]
    size_lookup = agent_api_sizes.resolve_sizes(all_primary_urls, wait_seconds=wait_for_sizes)
    models_list = [_summary_entry(entry, size_lookup) for entry in index.values()]
    models_list.sort(key=lambda m: (m["family"], m["model_type"]))
    return {
        "models": models_list,
        "families": _families_legacy(index),
        "errors": index_data.get("errors") or [],
    }


def _model_detail_payload(model_type: str) -> dict[str, Any] | None:
    """Build the /api/models/{model_type} response: full enriched entry + sizes."""
    entry = agent_api_introspect.get_model_entry(model_type)
    if entry is None:
        return None
    public = agent_api_introspect.public_entry(entry)
    size_lookup = agent_api_sizes.resolve_sizes(entry["urls"], wait_seconds=2.0)
    public["sizes"] = [
        {"url": url, **(size_lookup.get(url) or {"bytes": None, "error": "pending"})}
        for url in entry["urls"]
    ]
    public["primary_size_bytes"] = agent_api_sizes.total_bytes(entry["urls"], size_lookup)
    return public


def _settings_schema_payload() -> dict[str, Any]:
    """Build the /api/settings/schema response."""
    schema = agent_api_introspect.get_settings_schema()
    template_path = WANGP_ROOT / "models" / "_settings.json"
    raw_keys: list[dict[str, Any]] = []
    if template_path.is_file():
        try:
            template = json.loads(template_path.read_text())
            registered = {entry["key"] for entry in schema}
            for k, v in template.items():
                if k in registered:
                    continue
                raw_keys.append({
                    "key": k,
                    "type": _infer_type(v),
                    "default": v,
                })
        except Exception:
            pass
    return {
        "registered": schema,
        "freeform": raw_keys,
        "note": (
            "registered: typed/bounded settings discovered from "
            "shared/extra_settings.py (label/min/max/step). "
            "freeform: every other key from models/_settings.json with a "
            "type inferred from its default value. Both are accepted by "
            "POST /api/jobs."
        ),
    }


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return "string"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(*, host: str = "0.0.0.0", port: int = 8100, profile: int = 4,
          attention: str = "sage2", outputs_root: str | None = None,
          token: str | None = None, history_limit: int | None = None,
          cors_origins: str | None = None) -> None:
    """Start the hardened Wan2GP API server (blocking)."""

    if outputs_root:
        os.environ["WAN2GP_OUTPUTS_ROOT"] = str(Path(outputs_root).resolve())
    if token:
        os.environ["WAN2GP_TOKEN"] = token
    if cors_origins is not None:
        os.environ["WAN2GP_CORS_ORIGINS"] = cors_origins

    auth_token = os.environ.get("WAN2GP_TOKEN") or None
    history = history_limit or int(os.environ.get("WAN2GP_JOB_HISTORY", "200"))
    db_env = os.environ.get("WAN2GP_JOB_DB")
    db_path = Path(db_env).expanduser() if db_env else (Path.home() / ".wan2gp" / "jobs.sqlite")
    cors_allowed, cors_wildcard = _parse_cors_origins(os.environ.get("WAN2GP_CORS_ORIGINS"))

    # Bring up the WanGP agent — but lazily ensure session, so HTTP can
    # respond before model load completes. We import here so the file remains
    # importable for tests/tools without dragging in the runtime.
    from agent_api import WanGPAgent
    agent = WanGPAgent(
        profile=profile,
        attention=attention,
        verbose=True,
    )

    store = JobStore(db_path=db_path, history_limit=history)
    worker = JobWorker(agent=agent, store=store)
    started_at = time.time()

    handler_cls = _build_handler(
        agent=agent,
        store=store,
        worker=worker,
        token=auth_token,
        started_at=started_at,
        cors_allowed=cors_allowed,
        cors_wildcard=cors_wildcard,
    )

    srv = _Server((host, port), handler_cls)

    log_event(
        "server_started",
        host=host,
        port=port,
        version=SERVER_VERSION,
        outputs_root=str(_outputs_root()),
        job_db=str(db_path),
        history_limit=history,
        auth=bool(auth_token),
        cors=("*" if cors_wildcard else sorted(cors_allowed)) if (cors_allowed or cors_wildcard) else None,
    )
    if not auth_token:
        log_event(
            "auth_disabled",
            level="warn",
            message="WAN2GP_TOKEN unset — server is unauthenticated. "
                    "Acceptable on a trusted LAN only.",
        )

    print(f"WanGP Agent API server (v{SERVER_VERSION}) on http://{host}:{port}", flush=True)
    print(f"  outputs: {_outputs_root()}", flush=True)
    print(f"  job db:  {db_path}", flush=True)
    print(f"  auth:    {'bearer token required' if auth_token else 'OFF (no token set)'}", flush=True)
    if cors_wildcard:
        print(f"  cors:    * (any origin)", flush=True)
    elif cors_allowed:
        print(f"  cors:    {', '.join(sorted(cors_allowed))}", flush=True)
    else:
        print(f"  cors:    OFF", flush=True)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        log_event("server_shutdown", reason="sigint")
    finally:
        with contextlib.suppress(Exception):
            agent.close()
        with contextlib.suppress(Exception):
            srv.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WanGP Agent API server (hardened)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--profile", type=int, default=4)
    parser.add_argument("--attention", default="sage2")
    parser.add_argument("--outputs-root", default=None,
                        help="Override outputs root (default: <repo>/outputs)")
    parser.add_argument("--token", default=None,
                        help="Bearer token; if omitted falls back to WAN2GP_TOKEN env var")
    parser.add_argument("--history-limit", type=int, default=None,
                        help="Number of jobs to retain in SQLite (default: 200)")
    parser.add_argument("--cors-origins", default=None,
                        help="Comma-separated origin allow-list, or '*'. Falls back to WAN2GP_CORS_ORIGINS env. Empty = CORS disabled.")
    args = parser.parse_args()

    serve(
        host=args.host,
        port=args.port,
        profile=args.profile,
        attention=args.attention,
        outputs_root=args.outputs_root,
        token=args.token,
        history_limit=args.history_limit,
        cors_origins=args.cors_origins,
    )
