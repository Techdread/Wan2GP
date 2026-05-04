# Wan2GP Network API Guide

This is the LAN-facing programmable API around WanGP. Apps on the network
should integrate against this server, **not** against the Gradio web UI on
port 7860.

The server lives in `agent_api_server.py` (delegated to from
`agent_api.py serve`). Hardening details — async jobs, auth, observability —
are tracked in the companion file `Wan2GP-API-HARDENING.md`.

---

## Endpoints

```
GET    /api/health                        — liveness + GPU + queue + version
GET    /api/models                        — list of model_type entries with sizes + capability hints
GET    /api/models/{model_type}           — full enriched entry: defaults, applicable settings, model_def, sizes
GET    /api/loras?model_type=...          — lora files for a model
GET    /api/settings                      — global default settings template
GET    /api/settings/schema               — typed settings schema (registered + freeform)
POST   /api/settings/validate             — dry-run validate a request without queueing it
GET    /api/file?path=...                 — download a file (constrained to outputs root)

POST   /api/jobs                          — create a job (returns immediately, validated)
GET    /api/jobs?limit=N                  — list recent jobs
GET    /api/jobs/{id}                     — status + result
DELETE /api/jobs/{id}                     — cancel
GET    /api/jobs/{id}/events              — Server-Sent Events stream

POST   /api/release                       — release VRAM

# Deprecated (kept for one release):
POST   /api/generate                      — sync wrapper around /api/jobs
POST   /api/batch                         — sync wrapper around /api/jobs
```

Every response except `/api/health` is gated by an optional bearer token
(`WAN2GP_TOKEN`). If the env var is unset, the server runs unauthenticated
and logs a warning on boot.

Every request is tagged with `X-Request-Id` (echoed in JSON logs). Clients
may set their own; otherwise the server assigns one.

---

## Quick start

### Run the server

For one-off testing:

```bash
cd /media/peter/AI/Wan2GP
WAN2GP_TOKEN=mysecret ./venv/bin/python agent_api.py serve \
    --host 0.0.0.0 --port 8100
```

For "always on" via systemd user unit:

```bash
cd /media/peter/AI/Wan2GP
WAN2GP_TOKEN=mysecret ./deploy/install-systemd-unit.sh
```

The installer enables `wan2gp-api.service`, calls `loginctl enable-linger`
so it survives logout, and writes the unit to
`~/.config/systemd/user/wan2gp-api.service`.

### Health check

```bash
curl http://192.168.1.199:8100/api/health
```

```json
{
  "status": "ok",
  "version": "0.3.1",
  "uptime_seconds": 1234,
  "queue": {"running": 0, "queued": 0},
  "current_job_id": null,
  "gpu": {
    "name": "NVIDIA GeForce RTX 3060",
    "vram_total_mb": 12288,
    "vram_used_mb": 4321
  }
}
```

If the GPU is unreachable, `/api/health` returns `503` with
`status: "degraded"` and a short `reason` field. Use this for liveness probes.

---

## Auth

```bash
# All non-health endpoints require:
curl -H 'Authorization: Bearer mysecret' http://192.168.1.199:8100/api/jobs
```

Missing or wrong token → `401`. Health is always public.

There is no per-user identity, no rotation, no scopes. This is a LAN gate.

---

## Async job lifecycle (recommended)

### 1. Create a job

```bash
curl -X POST http://192.168.1.199:8100/api/jobs \
  -H 'Authorization: Bearer mysecret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "z_image",
    "prompt": "A cinematic fox in a rainy cyberpunk street",
    "resolution": "1024x1024",
    "num_inference_steps": 8,
    "seed": -1
  }'
```

Returns immediately (`202 Accepted`):

```json
{
  "job_id": "j_01HXXXXXXXXXXXXXXXXXXXX",
  "capability": "image-generation",
  "model_type": "z_image",
  "status": "queued",
  "queue_position": 1,
  "progress": 0.0,
  "step": 0,
  "total_steps": 0,
  "created_at": "2026-05-03T19:03:14Z",
  "started_at": null,
  "completed_at": null,
  "request": { ... },
  "files": [],
  "error": null,
  "request_id": "..."
}
```

### 2. Poll status

```bash
curl -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/jobs/j_01HXXXXXXXXXXXXXXXXXXXX
```

`status` transitions: `queued` → `running` → `completed | failed | cancelled`.
While running, `progress` (0.0-1.0), `step`, and `total_steps` advance.

### 3. Stream progress (SSE)

```bash
curl -N -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/jobs/j_01HXXXX/events
```

The connection emits `event: status` lines with the full job record on each
update, plus heartbeat comments every 15s, and closes on terminal status.

### 4. Cancel

```bash
curl -X DELETE -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/jobs/j_01HXXXX
```

- Queued → 200, `status: cancelled`.
- Running → 202, `status: cancelling` (worker stops at next safe step, VRAM released).
- Terminal → 409.

### 5. Download the result

```bash
# files[] in the job record are server-side absolute paths.
curl -L -H 'Authorization: Bearer mysecret' -o fox.jpg \
  "http://192.168.1.199:8100/api/file?path=/media/peter/AI/Wan2GP/outputs/fox.jpg"
```

`/api/file` rejects anything outside `WAN2GP_OUTPUTS_ROOT`
(default `/media/peter/AI/Wan2GP/outputs`). Symlink escapes and `..`
traversal are checked via `realpath` and return `403`.

---

## Video example

```bash
curl -X POST http://192.168.1.199:8100/api/jobs \
  -H 'Authorization: Bearer mysecret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "wan21_t2v_14B",
    "prompt": "A cat walking through a moonlit garden",
    "resolution": "832x480",
    "num_inference_steps": 30,
    "video_length": 81,
    "guidance_scale": 5.0,
    "flow_shift": 3.0
  }'
```

Then poll `/api/jobs/{id}` until `status: "completed"` and download via
`/api/file`.

---

## Capability discovery

### List all models (`GET /api/models`)

```bash
curl -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/models
```

Every field below is **auto-derived**: from `defaults/<model_type>.json`,
the family handler's `query_model_def()` feature flags, and a HEAD-cached
HuggingFace lookup for `size_bytes`. Adding a new model is zero-work —
drop a new `defaults/<x>.json` and it appears here on next call.

```json
{
  "models": [
    {
      "model_type": "z_image",
      "architecture": "z_image",
      "family": "z_image",
      "capability": "image-generation",
      "name": "Z-Image Turbo 6B",
      "description": "Z-Image is a powerful and highly efficient image generation model with 6B parameters...",
      "param_count_b": 6.0,
      "default_resolution": "1024x1024",
      "default_steps": 8,
      "default_video_length": null,
      "size_bytes": 12309879106,
      "size_status": null,
      "quant_variants": ["bf16", "int8"],
      "applicable_settings_count": 3,
      "url_count": 2
    },
    {
      "model_type": "ltx2_22B_distilled",
      "architecture": "ltx2_22B",
      "family": "ltx2",
      "capability": "video-generation",
      "name": "LTX-2 2.3 Distilled 1.0 22B",
      "param_count_b": 22.0,
      "default_resolution": "1280x720",
      "default_steps": 8,
      "default_video_length": 241,
      "size_bytes": 37987776440,
      "quant_variants": ["int8"],
      "applicable_settings_count": 18,
      "url_count": 2
    }
  ],
  "families": { "z_image": ["z_image", "z_image_base", ...], "ltx2": [...], ... },
  "errors": []
}
```

- `capability`: one of `image-generation | video-generation | audio-generation` — derived from the model's feature flags (`image_outputs`, `i2v_class`, `t2v_class`, `vace_class`, ...).
- `param_count_b`: parsed from the model name (e.g. `"22B"` in `"LTX-2 2.3 Distilled 1.0 22B"`).
- `size_bytes`: HEAD-resolved size of the primary safetensors file. The first call to `/api/models` blocks briefly (≤4 s) to populate the cache; later calls are instant. Persisted to `~/.wan2gp/model_sizes.json` (override via `WAN2GP_SIZE_CACHE`).
- `size_status`: `null` if cached cleanly, `"pending"` if still resolving, or an HTTP error string.
- `quant_variants`: detected from filenames (`bf16`, `fp16`, `int8`, `fp4`, `q4_k_m`, etc.).
- `applicable_settings_count`: how many bounded/typed settings apply to this model. Use `GET /api/models/{model_type}` for the full list.

### Single model detail (`GET /api/models/{model_type}`)

```bash
curl -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/models/z_image
```

Returns the full enriched entry — useful for building UIs or for an agent
that wants the precise schema of valid inputs for a specific model:

```json
{
  "model_type": "z_image",
  "architecture": "z_image",
  "family": "z_image",
  "capability": "image-generation",
  "name": "Z-Image Turbo 6B",
  "description": "...",
  "param_count_b": 6.0,
  "urls": ["https://huggingface.co/.../ZImageTurbo_bf16.safetensors", "..."],
  "preload_urls": [],
  "quant_variants": ["bf16", "int8"],
  "resolution_choices": null,
  "primary_size_bytes": 12309879106,
  "sizes": [
    {"url": "...", "bytes": 12309879106, "etag": "...", "fetched_at": "..."},
    {"url": "...", "bytes": 6154900000, "etag": "..."}
  ],
  "applicable_settings": [
    {"key": "NAG_scale", "label": "NAG Scale", "type": "number", "min": 1.0, "max": 20.0, "step": 0.01, "custom": false},
    {"key": "NAG_tau",   "label": "NAG Tau",   "type": "number", "min": 1.0, "max": 5.0,  "step": 0.01, "custom": false},
    {"key": "NAG_alpha", "label": "NAG Alpha", "type": "number", "min": 0.0, "max": 2.0,  "step": 0.01, "custom": false}
  ],
  "defaults": { "resolution": "1024x1024", "num_inference_steps": 8, "guidance_scale": 0, ... },
  "model_def": { "image_outputs": true, "guidance_max_phases": 0, "NAG": true, ... },
  "handler_loaded": true
}
```

`applicable_settings` is the set of typed/bounded settings whose visibility
resolver returns true for this model (e.g. NAG only matters for `z_image`,
not `z_image_base`; sliding-window settings only apply when the model
supports it). It's the smallest "extra dials" surface you need to build a
fully-functional UI for any model.

### Settings schema (`GET /api/settings/schema`)

```bash
curl -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/settings/schema
```

```json
{
  "registered": [
    {"key": "guidance_scale", "label": "Guidance (CFG)", "type": "number", "min": 1.0, "max": 20.0, "step": 0.1, "custom": false},
    {"key": "flow_shift",     "label": "Shift Scale",    "type": "number", "min": 1.0, "max": 25.0, "step": 0.1, "custom": false},
    "..."
  ],
  "freeform": [
    {"key": "prompt",            "type": "string",  "default": ""},
    {"key": "seed",              "type": "integer", "default": -1},
    {"key": "resolution",        "type": "string",  "default": "832x480"},
    {"key": "image_start",       "type": "null",    "default": null},
    "..."
  ],
  "note": "registered: typed/bounded settings discovered from shared/extra_settings.py..."
}
```

- `registered`: settings with explicit type/range/step from
  `shared/extra_settings.py`. Adding a new bounded setting in WanGP
  automatically surfaces here.
- `freeform`: every other key in `models/_settings.json`, with a type
  inferred from its default value. Both lists are accepted by `POST /api/jobs`.

### Validate without queueing (`POST /api/settings/validate`)

```bash
curl -X POST -H 'Authorization: Bearer mysecret' \
  -H 'Content-Type: application/json' \
  http://192.168.1.199:8100/api/settings/validate \
  -d '{"model_type": "z_image", "num_inference_steps": 8, "NAG_scale": 999}'
```

```json
{"valid": false, "error": "NAG Scale must be at most 20."}
```

Useful for form validation before submitting a job. `POST /api/jobs` runs
the same validation and returns `400` if it fails.

### Capability summary

A platform adapter can build its capability → model list purely from
`/api/models` without hardcoding any mapping. For finer control (per-model
dials), pull `/api/models/{model_type}` lazily.

---

## Python client

```python
from agent_api import WanGPAgent

agent = WanGPAgent(url="http://192.168.1.199:8100", token="mysecret")

# Async job API
job = agent.submit_job({
    "model_type": "z_image",
    "prompt": "A cinematic fox in a rainy cyberpunk street",
    "resolution": "1024x1024",
    "num_inference_steps": 8,
})
final = agent.wait_for_job(job["job_id"])
if final["status"] == "completed":
    agent.download_file(final["files"][0], "fox.jpg")

# Convenience wrappers (use legacy sync /api/generate under the hood,
# now annotated with Deprecation: true)
result = agent.generate_image(
    prompt="A sunset",
    model="z_image",
)
```

---

## Observability

Logs are JSON lines on stdout. Fields: `ts`, `level`, `event`,
`request_id`, `job_id?`, `capability?`, `model_type?`, `duration_ms?`,
plus event-specific fields. Examples:

```
{"ts":"...","event":"server_started","host":"0.0.0.0","port":8100,"auth":true}
{"ts":"...","event":"job_created","job_id":"j_...","request_id":"...","capability":"image-generation","request":{...}}
{"ts":"...","event":"job_started","job_id":"j_...","request_id":"..."}
{"ts":"...","event":"job_completed","job_id":"j_...","duration_ms":12340,"files":1}
{"ts":"...","event":"vram_released","reason":"post_cancel"}
{"ts":"...","event":"request","request_id":"...","method":"POST","path":"/api/jobs","status":202,"duration_ms":405}
```

Prompts are redacted by default. Set `WAN2GP_LOG_PROMPTS=1` to include them
when debugging.

The last 200 jobs are persisted to SQLite at `~/.wan2gp/jobs.sqlite` (override
via `WAN2GP_JOB_DB`). On restart, jobs that were in `running` are marked
`failed` with `error: "server restarted"` so callers don't poll forever.

---

## Configuration reference

| Env var               | Default                              | Purpose                                                      |
| --------------------- | ------------------------------------ | ------------------------------------------------------------ |
| `WAN2GP_TOKEN`        | _(unset)_                            | Bearer token. Unset = unauth (LAN trust mode).               |
| `WAN2GP_OUTPUTS_ROOT` | `<repo>/outputs`                     | Root for `/api/file` realpath check.                         |
| `WAN2GP_JOB_DB`       | `~/.wan2gp/jobs.sqlite`              | Persistent job log path.                                     |
| `WAN2GP_JOB_HISTORY`  | `200`                                | Number of jobs to retain.                                    |
| `WAN2GP_LOG_PROMPTS`  | _(unset)_                            | "1" to include prompt text in JSON logs.                     |
| `WAN2GP_CORS_ORIGINS` | _(unset)_                            | Comma-separated origin allow-list, or `*`. Empty = CORS off. |
| `WAN2GP_SIZE_CACHE`   | `~/.wan2gp/model_sizes.json`         | Persisted HEAD cache of model file sizes.                    |

CLI flags on `agent_api.py serve` mirror the env vars: `--host`, `--port`,
`--profile`, `--attention`, `--token`, `--outputs-root`, `--history-limit`,
`--cors-origins`.

---

## Constraints to be aware of

- **One generation at a time** — the GPU is single-tenant; jobs queue rather
  than run in parallel.
- **No HTTPS** — terminate TLS in a reverse proxy if you ever expose this off
  LAN.
- **Local in-process mode and the Gradio web UI cannot coexist** — either
  use the dedicated API server or the web UI on `7860`, not both at once.
- **Back-compat shims (`POST /api/generate`, `POST /api/batch`) are deprecated**
  — they still work and return `Deprecation: true` plus
  `Link: </api/jobs>; rel="successor-version"`. Migrate to `/api/jobs`.

---

## Adding a new model

The whole capability + schema layer is auto-derived. Cost of adding a model:

| What you add                               | Work in this API |
| ------------------------------------------ | ---------------- |
| New `defaults/<model_type>.json`           | **Zero** — it appears in `/api/models` on next call. Capability, params, defaults, applicable settings, and size are all derived from the file + the family handler's feature flags. |
| New architecture (e.g. a new `*_handler`)  | **One line** — add it to `wgp.py:family_handlers`. This API re-reads that list via AST on each rebuild. |
| New typed setting                          | **One entry** in `shared/extra_settings.py` (`_add_setting(...)`) — surfaces in `/api/settings/schema` and per-model `applicable_settings` automatically. |

To force a rebuild without restarting (e.g. after dropping a new
`defaults/*.json`), call `agent_api_introspect.invalidate_cache()` from a
Python REPL — the next `/api/models` call rebuilds.

---

## Web UI

The Gradio UI is still on `0.0.0.0:7860` (e.g. `http://192.168.1.199:7860`)
for human use. Don't build other programs against it — it's noisy and not
stable-looking. Use `/api/jobs` instead.
