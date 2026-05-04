# Wan2GP `agent_api.py` Hardening Spec

This is a hand-off spec for maturing the existing Wan2GP wrapper (`agent_api.py`, port `8100` on the 3060 host) before the shared capability layer starts depending on it.

Companion to `Wan2GP-NETWORK-API.md`, which describes the current state.

The goal is to make this server trustworthy enough that every app in the hub can call it through a shared adapter without footguns or timeouts.

---

## Scope

In scope:
- security hardening of the existing endpoints
- async job model (so video generations do not hang HTTP clients)
- cancellation and lifecycle
- auth
- observability (job ids, structured logs)
- minor API surface additions that make the platform adapter easier to write
- operational: keep the server running on its own

Out of scope (handled later by the platform layer, not this server):
- multi-GPU routing
- cross-app job history UI
- provider fallback
- cost / quota accounting

---

## Findings from the current API

Things the current design gets right:
- separated from the Gradio UI on a stable port
- clean endpoint set: `health`, `generate`, `batch`, `models`, `loras`, `settings`, `release`
- bound to `0.0.0.0` for LAN access
- model and lora discovery already exposed
- `release` endpoint exists for VRAM lifecycle

Things to fix before apps integrate:

1. **`GET /api/file?path=...` is a read-any-file primitive.** No path constraint. On a trusted LAN it is *fine in practice* but free to lock down.
2. **`POST /api/generate` is synchronous.** A wan21 video at 30 steps takes minutes. Browsers, fetch defaults, and reverse proxies will time out long before the response.
3. **No job id even on the sync path.** Server-side and client-side logs cannot be correlated.
4. **One global generation lock with no cancel.** A wedged job stalls the whole 3060 until the process is killed.
5. **No auth.** LAN trust is the current posture but adding a shared token is trivial and survives wifi changes / guest devices.
6. **Capability is implicit in `model_type`** (`z_image` → image, `wan21_t2v_14B` → video). Adapter has to hardcode this knowledge. Easy to surface from the server instead.
7. **Magic values leak through the wire format** (`seed: -1`, `image_mode: 1`, `loras_multipliers` as a string, `force_fps: ""`). Acceptable for a thin wrapper; the platform adapter will hide these from apps.
8. **Server is started by hand.** Companion doc currently says "nothing is listening on port 8100." Anything that depends on it needs the server to be always-up.

---

## Required changes

### 1. Constrain `/api/file` to the outputs directory

Currently:
```
GET /api/file?path=<any absolute path>
```

Required behavior:
- Resolve `path` to an absolute, canonical path (`os.path.realpath`).
- Reject if it does not start with the configured outputs root (default `/media/peter/AI/Wan2GP/outputs`).
- Reject symlinks that escape the outputs root.
- Return `403` with a JSON error body on rejection. Do not echo the requested path back in the error (avoids leaking filesystem layout).

Outputs root must be configurable via env var `WAN2GP_OUTPUTS_ROOT`.

Acceptance:
- `curl '.../api/file?path=/etc/passwd'` returns 403.
- `curl '.../api/file?path=/media/peter/AI/Wan2GP/outputs/../../etc/passwd'` returns 403.
- A normal generation result download still works.

---

### 2. Add a real async job model

Replace the synchronous `POST /api/generate` contract with an async one. Keep the existing endpoint working in a back-compat mode for now (see §9).

New endpoints:

```
POST   /api/jobs              # create a job, returns { job_id, status: "queued" }
GET    /api/jobs              # list recent jobs (last N, configurable)
GET    /api/jobs/:id          # status + result
DELETE /api/jobs/:id          # cancel
GET    /api/jobs/:id/events   # SSE stream of progress events (optional but nice)
```

Job record:
```json
{
  "job_id": "j_01HXXXXXXXXXXXXXXXXXXXXXXX",
  "capability": "image-generation",
  "model_type": "z_image",
  "status": "queued | running | completed | failed | cancelled",
  "progress": 0.0,
  "step": 0,
  "total_steps": 8,
  "created_at": "2026-05-03T14:00:00Z",
  "started_at": null,
  "completed_at": null,
  "duration_seconds": null,
  "request": { ... echo of normalized request ... },
  "files": [],
  "error": null
}
```

Job ids:
- ULID or `j_` prefix + ULID. Sortable by creation time. No UUIDv4.

Storage:
- In-memory dict is fine for v1. Persist last ~200 jobs to a SQLite file at `WAN2GP_JOB_DB` (default `~/.wan2gp/jobs.sqlite`) so a restart does not lose recent history.

Concurrency:
- Keep the existing single-generation lock. Jobs simply queue.
- Surface queue position in the job record: `"queue_position": 2`.

Acceptance:
- Creating a video job returns immediately (<200 ms) with `status: "queued"`.
- Polling `/api/jobs/:id` reports `running` then `completed` with `files` populated.
- Restarting the server preserves the last 200 job records (status of in-flight jobs becomes `failed` with `error: "server restarted"`).

---

### 3. Cancellation

`DELETE /api/jobs/:id` semantics:
- If `queued`: remove from queue, mark `cancelled`. Return 200.
- If `running`: signal the worker to abort at the next safe step boundary, mark `cancelling` then `cancelled`. Return 202.
- If terminal: return 409 with current status.

A `cancelled` job must release any held VRAM (call the same path `POST /api/release` does).

Acceptance:
- A long video job can be cancelled mid-run and the next queued job starts within a few seconds.
- Cancelling does not crash the server or leave the generation lock stuck.

---

### 4. Progress events

While a job is running, write progress to the job record at every diffusion step (or at most every 500 ms — whichever is less frequent). Required fields updated: `progress`, `step`, `total_steps`.

Optional but recommended: `GET /api/jobs/:id/events` as Server-Sent Events emitting one JSON line per progress update plus a final `completed` / `failed` / `cancelled` event. This avoids clients polling at high frequency.

Acceptance:
- During a 30-step video, `progress` advances from 0.0 to ~1.0 visibly across polls.
- SSE stream (if implemented) terminates cleanly on terminal state.

---

### 5. Auth

Add a single shared bearer token, off by default for back-compat:

- Env var: `WAN2GP_TOKEN`.
- If unset: server runs unauthenticated and logs a warning on startup.
- If set: every endpoint except `GET /api/health` requires `Authorization: Bearer <token>`. Return 401 on missing/invalid.

No rotation, no per-user, no scopes. This is a LAN gate, not an identity system.

Acceptance:
- With token set, calls without the header return 401.
- `GET /api/health` always works (used for liveness probes).

---

### 6. Request ids and structured logs

- Every request gets an `X-Request-Id` (generate one if the client did not send it).
- Every job log line includes `job_id` and `request_id`.
- Logs are JSON lines to stdout (one event per line). Fields: `ts`, `level`, `event`, `request_id`, `job_id?`, `capability?`, `duration_ms?`, plus event-specific fields.
- No prompt text in logs by default. Add `WAN2GP_LOG_PROMPTS=1` to opt in for debugging.

Acceptance:
- Tailing the server log during a generation shows `job_created` → `job_started` → repeated `job_progress` → `job_completed`, all with the same `job_id`.

---

### 7. Capability hints in `/api/models`

Currently `/api/models` returns model types but the caller has to know which are images and which are videos.

Required addition: each entry returns its capability and a coarse cost hint.

```json
{
  "models": [
    {
      "model_type": "z_image",
      "capability": "image-generation",
      "default_resolution": "1024x1024",
      "default_steps": 8,
      "speed_hint": "fast"
    },
    {
      "model_type": "wan21_t2v_14B",
      "capability": "video-generation",
      "default_resolution": "832x480",
      "default_steps": 30,
      "speed_hint": "slow"
    }
  ]
}
```

`speed_hint` is one of `"fast" | "medium" | "slow"`. It is a hand-maintained label, not measured.

Acceptance:
- Platform adapter can build its capability → model list purely from `/api/models` without any hardcoded mapping.

---

### 8. Health endpoint should mean something

`GET /api/health` currently returns `{"status":"ok"}` if the process is up. Make it report what the platform actually needs:

```json
{
  "status": "ok",
  "version": "0.3.1",
  "uptime_seconds": 1234,
  "gpu": {
    "name": "NVIDIA GeForce RTX 3060",
    "vram_total_mb": 12288,
    "vram_used_mb": 4321
  },
  "queue": { "running": 1, "queued": 2 },
  "current_job_id": "j_01HXXXX..."
}
```

If the GPU is unreachable or in a bad state, return 503 with `status: "degraded"` and a short reason.

Acceptance:
- Platform `provider-health.js` can poll this every ~5s and decide whether to route work here.

---

### 9. Back-compat for the current `POST /api/generate` and `POST /api/batch`

Both synchronous endpoints stay for one release cycle. Implement each as: create one or more jobs, block until terminal, return the legacy response shape. Add deprecation headers:

```
Deprecation: true
Sunset: <date>
Link: </api/jobs>; rel="successor-version"
```

Both are deprecated together: `/api/batch` is just a multi-task variant of `/api/generate` and there is no reason to keep one without the other once the async API is in.

This means existing scripts and the Python `WanGPAgent` class keep working while the platform adapter is being written against `/api/jobs`.

Acceptance:
- Old `curl` examples in `Wan2GP-NETWORK-API.md` still produce a working response.
- Both `/api/generate` and `/api/batch` responses include `Deprecation: true` and the successor-version link header.

---

### 10. Run it as a service

Stop relying on a hand-typed launch command. Pick one:

**Option A — systemd user unit** (preferred):
- Unit file at `~/.config/systemd/user/wan2gp-api.service`.
- `ExecStart=/media/peter/AI/Wan2GP/venv/bin/python /media/peter/AI/Wan2GP/agent_api.py serve --host 0.0.0.0 --port 8100`
- `Restart=on-failure`, `RestartSec=5`.
- `Environment=WAN2GP_TOKEN=...` (and other env vars).
- Enable with `systemctl --user enable --now wan2gp-api.service`.
- `loginctl enable-linger peter` so it survives logout.

**Option B — a tiny `launch.sh` + tmux session**, only if systemd is not workable on this host.

Acceptance:
- Reboot the 3060 host. Without any manual action, `curl http://192.168.1.199:8100/api/health` returns 200 within a minute of login.

Update `Wan2GP-NETWORK-API.md` to reflect the new "always on" reality once this lands.

---

### 11. CORS for browser clients

Configurable per-request allow-list driven by `WAN2GP_CORS_ORIGINS`
(comma-separated origins, or `*`, default empty → CORS disabled).

Behavior:
- `Origin` matches the allow-list → response carries
  `Access-Control-Allow-Origin: <echoed origin>`, `Vary: Origin`, and
  `Access-Control-Expose-Headers: Deprecation, Link, X-Request-Id`.
- `Origin` set to `*` in `WAN2GP_CORS_ORIGINS` → server echoes back `*`
  (we never use credentialed CORS, so wildcard is safe in this mode).
- Origin not in allow-list → response carries `Vary: Origin` only, no
  `Access-Control-Allow-Origin`. The browser will block the response.
- `OPTIONS` preflight from an allowed origin → `204 No Content` with
  `Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS`,
  `Access-Control-Allow-Headers: Authorization, Content-Type, X-Request-Id`,
  `Access-Control-Expose-Headers: Deprecation, Link, X-Request-Id`,
  `Access-Control-Max-Age: 600`.
- `OPTIONS` preflight from a disallowed origin → `403 Forbidden`, no
  CORS headers (preflight failure is the right way to surface this in
  browsers).
- `OPTIONS` always bypasses the bearer-token check (preflight cannot send
  `Authorization`).

Tokens for browser use: callers should pass the bearer token in the
`Authorization` header on each request. Do **not** put the token in
`document.cookie`; we do not set `Access-Control-Allow-Credentials`. The
exposure risk of putting a long-lived token into browser code is real
— see `Wan2GP-CLIENT-SETUP.md` § Browser apps for guidance.

Acceptance:
- With `WAN2GP_CORS_ORIGINS=http://localhost:5173`, an `OPTIONS` from
  `Origin: http://localhost:5173` returns 204 with the headers above; a
  GET from the same origin carries `Access-Control-Allow-Origin`.
- With the same setting, an `OPTIONS` from `Origin: http://evil.example`
  returns 403 with no `Access-Control-*` headers.
- With env var unset, no `Access-Control-Allow-Origin` header is ever
  emitted.

Status: **implemented**. Configurable via env var
`WAN2GP_CORS_ORIGINS`, CLI flag `--cors-origins`, or the systemd unit's
`Environment=WAN2GP_CORS_ORIGINS=...` line.

---

## Non-goals / explicitly do not do

- Do not add user accounts, OAuth, or per-app keys. One shared token is enough.
- Do not add HTTPS termination here — if it is ever needed, put it behind a reverse proxy.
- Do not introduce a queue broker (Redis, RabbitMQ). In-process queue + SQLite log is correct for one box and one GPU.
- Do not change the underlying Wan2GP request fields. Keep `image_mode`, `flow_shift`, `loras_multipliers`, `seed: -1`, etc. as they are. The platform adapter hides them from apps.
- Do not try to support concurrent generations. The hardware does not.

---

## Suggested implementation order

1. Job model + new `/api/jobs` endpoints (queued/running/completed/failed). Most of the value is here.
2. Cancellation + VRAM release on cancel.
3. Progress updates (polling first, SSE later if useful).
4. `/api/file` path constraint.
5. Auth token.
6. Request ids + structured JSON logs.
7. `/api/models` capability + speed hints.
8. Richer `/api/health`.
9. Back-compat shim on `/api/generate`.
10. systemd unit.

Each of 1–9 is independently shippable and testable. 10 should land last, after the rest is stable.

---

## Test checklist

- [ ] Image job end-to-end via `/api/jobs` returns files.
- [ ] Video job end-to-end via `/api/jobs` returns files.
- [ ] Cancel during queued: job marked cancelled, never runs.
- [ ] Cancel during running: worker stops within seconds, VRAM released, next job starts.
- [ ] `/api/file` rejects paths outside outputs root including `..` traversal and symlink escape.
- [ ] With `WAN2GP_TOKEN` set, unauthenticated calls (except health) return 401.
- [ ] Health endpoint reports GPU info and queue state.
- [ ] `/api/models` includes `capability` and `speed_hint` for every entry.
- [ ] Logs are JSON lines and correlate by `job_id` / `request_id`.
- [ ] Server survives a host reboot via systemd and is reachable on the LAN within a minute.
- [ ] Existing `POST /api/generate` synchronous calls still work and return `Deprecation: true`.

---

## Bottom line

Once 1–6 land, the platform layer can wrap this server as `providers/local-3060-wan2gp.js` and treat it like any other capability provider. The remaining items (7–10) make the integration nice rather than just possible.
